//! Receives device-independent gaze samples from host-side eye trackers.
//!
//! The UDP wire format is fixed-size and little-endian:
//!
//! | Offset | Size | Field |
//! | --- | --- | --- |
//! | 0 | 4 | Magic bytes `AGAZ` |
//! | 4 | 1 | Protocol version (`1`) |
//! | 5 | 1 | Mode: `0` combined, `1` per-eye |
//! | 6 | 1 | Validity mask: bit 0 first/combined, bit 1 right |
//! | 7 | 1 | Reserved (`0`) |
//! | 8 | 4 | Wrapping packet sequence number |
//! | 12 | 4 | Sample age at send time, in microseconds |
//! | 16 | 12 | First/combined gaze direction: `x`, `y`, `z` |
//! | 28 | 12 | Right gaze direction: `x`, `y`, `z` |
//!
//! All multi-byte values use little-endian encoding. Directions must already use ALVR's canonical
//! right-handed, head-local space (`+X` right, `+Y` up, `-Z` forward) and must have unit length.
//! The receiver only validates the protocol and maps its timing into the ALVR tracking timeline;
//! it does not apply device-specific projection, angle limits or sensitivity scaling.

use alvr_common::{glam::Vec3, info, parking_lot::Mutex, warn};
use alvr_packets::{GazeDirection, GazeSample};
use std::{
    net::{Ipv4Addr, UdpSocket},
    sync::atomic::{AtomicBool, Ordering},
    thread,
    time::{Duration, Instant},
};

const PACKET_MAGIC: &[u8; 4] = b"AGAZ";
const PROTOCOL_VERSION: u8 = 1;
const COMBINED_MODE: u8 = 0;
const PER_EYE_MODE: u8 = 1;
const PACKET_SIZE: usize = 40;
const DIRECTION_SIZE: usize = 12;
const FIRST_GAZE_OFFSET: usize = 16;
const VALID_GAZE_MASK: u8 = 0b11;
const DIRECTION_LENGTH_SQUARED_TOLERANCE: f32 = 0.02;
const READ_TIMEOUT: Duration = Duration::from_millis(100);
const SAMPLE_STALE_TIMEOUT: Duration = Duration::from_millis(100);
const MALFORMED_PACKET_LOG_INTERVAL: Duration = Duration::from_secs(1);

static LATEST_SAMPLE: Mutex<Option<ReceivedSample>> = Mutex::new(None);
static RECEIVER_RUNNING: AtomicBool = AtomicBool::new(false);
static RECEIVER_HANDLE: Mutex<Option<thread::JoinHandle<()>>> = Mutex::new(None);

#[derive(Clone, Copy)]
struct ReceivedSample {
    directions: [GazeDirection; 2],
    source_age: Duration,
    received_at: Instant,
}

struct DecodedSample {
    sequence: u32,
    source_age: Duration,
    directions: [GazeDirection; 2],
}

fn read_u32(packet: &[u8], offset: usize) -> Option<u32> {
    Some(u32::from_le_bytes(
        packet.get(offset..offset + 4)?.try_into().ok()?,
    ))
}

fn read_f32(packet: &[u8], offset: usize) -> Option<f32> {
    Some(f32::from_le_bytes(
        packet.get(offset..offset + 4)?.try_into().ok()?,
    ))
}

fn decode_direction(packet: &[u8], index: usize) -> Option<GazeDirection> {
    let offset = FIRST_GAZE_OFFSET + index * DIRECTION_SIZE;
    let direction = Vec3::new(
        read_f32(packet, offset)?,
        read_f32(packet, offset + 4)?,
        read_f32(packet, offset + 8)?,
    );
    let length_squared = direction.length_squared();

    if !length_squared.is_finite()
        || (length_squared - 1.0).abs() > DIRECTION_LENGTH_SQUARED_TOLERANCE
    {
        return None;
    }

    GazeDirection::new(direction)
}

fn decode_packet(packet: &[u8]) -> Option<DecodedSample> {
    if packet.len() != PACKET_SIZE || packet.get(..PACKET_MAGIC.len())? != PACKET_MAGIC {
        return None;
    }

    let [version, mode, validity_mask, reserved] =
        packet.get(PACKET_MAGIC.len()..8)?.try_into().ok()?;
    if version != PROTOCOL_VERSION || reserved != 0 {
        return None;
    }

    if validity_mask & !VALID_GAZE_MASK != 0 || validity_mask == 0 {
        return None;
    }

    let sequence = read_u32(packet, 8)?;
    let source_age = Duration::from_micros(read_u32(packet, 12)? as u64);
    if source_age > SAMPLE_STALE_TIMEOUT {
        return None;
    }

    let first = || decode_direction(packet, 0);
    let right = || decode_direction(packet, 1);
    let directions = match (mode, validity_mask) {
        (COMBINED_MODE, 0b01) | (PER_EYE_MODE, 0b01) => [first()?; 2],
        (PER_EYE_MODE, 0b10) => [right()?; 2],
        (PER_EYE_MODE, 0b11) => [first()?, right()?],
        _ => return None,
    };

    Some(DecodedSample {
        sequence,
        source_age,
        directions,
    })
}

fn sequence_is_newer(sequence: u32, previous: u32) -> bool {
    let distance = sequence.wrapping_sub(previous);

    distance != 0 && distance < (1 << 31)
}

fn receiver_loop(port: u16) {
    let socket = match UdpSocket::bind((Ipv4Addr::LOCALHOST, port)) {
        Ok(socket) => socket,
        Err(error) => {
            warn!("Failed to bind external gaze receiver on 127.0.0.1:{port}: {error}");
            RECEIVER_RUNNING.store(false, Ordering::SeqCst);
            return;
        }
    };

    if let Err(error) = socket.set_read_timeout(Some(READ_TIMEOUT)) {
        warn!("Failed to configure external gaze receiver: {error}");
        RECEIVER_RUNNING.store(false, Ordering::SeqCst);
        return;
    }

    info!("External gaze receiver listening on 127.0.0.1:{port} (AGAZ protocol v1)");

    let mut buffer = [0_u8; 512];
    let mut last_sequence = None;
    let mut last_accepted_at: Option<Instant> = None;
    let mut last_malformed_log: Option<Instant> = None;

    while RECEIVER_RUNNING.load(Ordering::SeqCst) {
        match socket.recv_from(&mut buffer) {
            Ok((size, _)) => {
                let now = Instant::now();
                let Some(sample) = decode_packet(&buffer[..size]) else {
                    if last_malformed_log.is_none_or(|timestamp| {
                        now.saturating_duration_since(timestamp) >= MALFORMED_PACKET_LOG_INTERVAL
                    }) {
                        warn!("Ignoring malformed or unsupported external gaze packet");
                        last_malformed_log = Some(now);
                    }
                    continue;
                };

                let sequence_restarted = last_accepted_at.is_none_or(|timestamp| {
                    now.saturating_duration_since(timestamp) > SAMPLE_STALE_TIMEOUT
                });
                if !sequence_restarted
                    && last_sequence
                        .is_some_and(|previous| !sequence_is_newer(sample.sequence, previous))
                {
                    continue;
                }

                last_sequence = Some(sample.sequence);
                last_accepted_at = Some(now);
                *LATEST_SAMPLE.lock() = Some(ReceivedSample {
                    directions: sample.directions,
                    source_age: sample.source_age,
                    received_at: now,
                });
            }
            Err(error)
                if matches!(
                    error.kind(),
                    std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
                ) => {}
            Err(error) => {
                warn!("External gaze receiver stopped: {error}");
                break;
            }
        }
    }

    RECEIVER_RUNNING.store(false, Ordering::SeqCst);
}

pub fn start_receiver(port: u16) {
    if RECEIVER_RUNNING.swap(true, Ordering::SeqCst) {
        return;
    }

    clear_sample();
    let handle = match thread::Builder::new()
        .name("external-gaze".into())
        .spawn(move || receiver_loop(port))
    {
        Ok(handle) => handle,
        Err(error) => {
            warn!("Failed to start external gaze receiver: {error}");
            RECEIVER_RUNNING.store(false, Ordering::SeqCst);
            return;
        }
    };

    *RECEIVER_HANDLE.lock() = Some(handle);
}

pub fn stop_receiver() {
    RECEIVER_RUNNING.store(false, Ordering::SeqCst);
    if let Some(handle) = RECEIVER_HANDLE.lock().take() {
        handle.join().ok();
    }
    clear_sample();
}

pub fn clear_sample() {
    *LATEST_SAMPLE.lock() = None;
}

pub fn latest_sample(reference_timestamp: Duration) -> Option<GazeSample> {
    let now = Instant::now();
    let sample = (*LATEST_SAMPLE.lock())?;
    let total_age = sample
        .source_age
        .saturating_add(now.saturating_duration_since(sample.received_at));

    (total_age <= SAMPLE_STALE_TIMEOUT).then_some(GazeSample {
        sample_timestamp: reference_timestamp.saturating_sub(total_age),
        directions: sample.directions,
    })
}
