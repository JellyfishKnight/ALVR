//! Projects canonical head-local gaze samples into ALVR foveated-encoding center shifts and
//! keeps the encoder and decoder transforms synchronized for each video frame.

use crate::bindings::FfiFoveationCenters;
use alvr_common::{
    ViewParams,
    glam::{Vec2, Vec3},
    parking_lot::Mutex,
};
use alvr_packets::{GazeDirection, GazeSample};
use alvr_session::{FoveatedEncodingConfig, GazeInputSource};
use std::{collections::VecDeque, time::Duration};

const GAZE_FILTER_TIME_CONSTANT: Duration = Duration::from_millis(30);
const INVALID_GAZE_HOLD_DURATION: Duration = Duration::from_millis(100);
const CENTER_HISTORY_CAPACITY: usize = 360;
#[cfg(feature = "foveation-diagnostics")]
const GAZE_DIAGNOSTIC_LOG_INTERVAL: Duration = Duration::from_millis(500);

static FOVEATION_STATE: Mutex<FilteredGazeState> = Mutex::new(FilteredGazeState::new());
static RENDERED_CENTERS: Mutex<CenterHistory> = Mutex::new(CenterHistory::new());
#[cfg(feature = "foveation-diagnostics")]
static LAST_GAZE_DIAGNOSTIC_LOG_TIMESTAMP: Mutex<Option<Duration>> = Mutex::new(None);

struct CenterHistory {
    samples: VecDeque<(Duration, [Vec2; 2])>,
}

impl CenterHistory {
    const fn new() -> Self {
        Self {
            samples: VecDeque::new(),
        }
    }

    fn clear(&mut self) {
        self.samples.clear();
    }

    fn insert(&mut self, timestamp: Duration, centers: [Vec2; 2]) {
        self.samples.push_back((timestamp, centers));

        while self.samples.len() > CENTER_HISTORY_CAPACITY {
            self.samples.pop_front();
        }
    }

    fn get(&self, timestamp: Duration) -> Option<[Vec2; 2]> {
        self.samples
            .iter()
            .rev()
            .find_map(|(sample_timestamp, centers)| {
                (*sample_timestamp == timestamp).then_some(*centers)
            })
    }
}

struct FilteredGazeState {
    center_history: CenterHistory,
    live_source: Option<GazeSource>,
    filtered_directions: Option<[Vec3; 2]>,
    last_filter_timestamp: Option<Duration>,
    last_valid_sample_timestamp: Option<Duration>,
}

impl FilteredGazeState {
    const fn new() -> Self {
        Self {
            center_history: CenterHistory::new(),
            live_source: None,
            filtered_directions: None,
            last_filter_timestamp: None,
            last_valid_sample_timestamp: None,
        }
    }

    fn clear(&mut self) {
        self.center_history.clear();
        self.live_source = None;
        self.filtered_directions = None;
        self.last_filter_timestamp = None;
        self.last_valid_sample_timestamp = None;
    }

    fn update(
        &mut self,
        frame_timestamp: Duration,
        sample: Option<(GazeSource, Duration, [Vec3; 2])>,
        view_params: [ViewParams; 2],
        config: &FoveatedEncodingConfig,
    ) -> bool {
        let directions = if let Some((source, sample_timestamp, raw_directions)) = sample {
            if self.live_source != Some(source) {
                self.filtered_directions = None;
                self.last_filter_timestamp = None;
            }

            let filtered_directions = if let (Some(previous), Some(previous_timestamp)) =
                (self.filtered_directions, self.last_filter_timestamp)
            {
                let delta_s = sample_timestamp
                    .saturating_sub(previous_timestamp)
                    .as_secs_f32();
                let alpha = 1.0 - (-delta_s / GAZE_FILTER_TIME_CONSTANT.as_secs_f32()).exp();
                let [previous_left, previous_right] = previous;
                let [left, right] = raw_directions;

                [
                    lerp_direction(previous_left, left, alpha),
                    lerp_direction(previous_right, right, alpha),
                ]
            } else {
                raw_directions
            };

            self.live_source = Some(source);
            self.filtered_directions = Some(filtered_directions);
            self.last_filter_timestamp = Some(sample_timestamp);
            self.last_valid_sample_timestamp = Some(sample_timestamp);

            Some(filtered_directions)
        } else if self
            .last_valid_sample_timestamp
            .is_some_and(|sample_timestamp| {
                frame_timestamp.saturating_sub(sample_timestamp) <= INVALID_GAZE_HOLD_DURATION
            })
        {
            self.filtered_directions
        } else {
            self.live_source = None;
            self.filtered_directions = None;
            self.last_filter_timestamp = None;
            self.last_valid_sample_timestamp = None;

            None
        };

        let centers = directions.and_then(|directions| {
            head_local_directions_to_centers(directions, view_params, config)
        });
        if let Some(centers) = centers {
            self.center_history.insert(frame_timestamp, centers);
        }

        centers.is_some()
    }

    fn get(&self, timestamp: Duration) -> Option<[Vec2; 2]> {
        self.center_history.get(timestamp)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GazeSource {
    ExternalUdp,
    Headset,
    Held,
    Static,
}

impl GazeSource {
    pub fn description(self) -> &'static str {
        match self {
            Self::ExternalUdp => "external UDP gaze",
            Self::Headset => "headset gaze",
            Self::Held => "last valid gaze (short hold)",
            Self::Static => "configured static center",
        }
    }
}

fn lerp_direction(previous: Vec3, current: Vec3, alpha: f32) -> Vec3 {
    GazeDirection::new(previous.lerp(current, alpha))
        .map(GazeDirection::direction)
        .unwrap_or(current)
}

fn project_direction_to_view_uv(direction: Vec3, view_params: ViewParams) -> Option<Vec2> {
    if !direction.is_finite() || direction.z >= -f32::EPSILON {
        return None;
    }

    let tan_left = view_params.fov.left.tan();
    let tan_right = view_params.fov.right.tan();
    let tan_up = view_params.fov.up.tan();
    let tan_down = view_params.fov.down.tan();
    let horizontal_span = tan_right - tan_left;
    let vertical_span = tan_up - tan_down;

    if horizontal_span <= f32::EPSILON || vertical_span <= f32::EPSILON {
        return None;
    }

    let tan_x = direction.x / -direction.z;
    let tan_y = direction.y / -direction.z;
    let uv = Vec2::new(
        (tan_x - tan_left) / horizontal_span,
        (tan_up - tan_y) / vertical_span,
    );

    uv.is_finite().then_some(uv)
}

fn uv_delta_to_center_shift(delta: Vec2, config: &FoveatedEncodingConfig) -> Vec2 {
    let movable_fraction = Vec2::new(1.0 - config.center_size_x, 1.0 - config.center_size_y);

    Vec2::new(
        if movable_fraction.x > f32::EPSILON {
            delta.x * 2.0 / movable_fraction.x
        } else {
            0.0
        },
        if movable_fraction.y > f32::EPSILON {
            delta.y * 2.0 / movable_fraction.y
        } else {
            0.0
        },
    )
}

fn head_local_directions_to_centers(
    directions: [Vec3; 2],
    view_params: [ViewParams; 2],
    config: &FoveatedEncodingConfig,
) -> Option<[Vec2; 2]> {
    let [left_direction, right_direction] = directions;
    let [left_view, right_view] = view_params;

    Some([
        head_local_direction_to_center(left_direction, left_view, false, config)?,
        head_local_direction_to_center(right_direction, right_view, true, config)?,
    ])
}

fn head_local_direction_to_center(
    direction: Vec3,
    view: ViewParams,
    mirror_x: bool,
    config: &FoveatedEncodingConfig,
) -> Option<Vec2> {
    // Canonical gaze is head-local; each asymmetric eye projection operates in eye-local space.
    let eye_local_direction = view.pose.orientation.inverse() * direction;
    let gaze_uv = project_direction_to_view_uv(eye_local_direction, view)?;

    // Dynamic foveation must use the absolute per-eye projection. Treating gaze as an offset from
    // the configured static center gives the two eyes unrelated projection origins, so their
    // high-density centers do not fuse at the same visual direction. The server's packed texture
    // mirrors only the right-eye X coordinate.
    let packed_gaze_uv = if mirror_x {
        Vec2::new(1.0 - gaze_uv.x, gaze_uv.y)
    } else {
        gaze_uv
    };
    let center_shift = uv_delta_to_center_shift(packed_gaze_uv - Vec2::splat(0.5), config);

    Some(center_shift.clamp(Vec2::splat(-1.0), Vec2::splat(1.0)))
}

#[unsafe(export_name = "GetEyeTrackedFoveationCenters")]
extern "C" fn get_eye_tracked_foveation_centers(timestamp_ns: u64) -> FfiFoveationCenters {
    if let Some([left, right]) = FOVEATION_STATE
        .lock()
        .get(Duration::from_nanos(timestamp_ns))
    {
        FfiFoveationCenters {
            valid: true,
            leftX: left.x,
            leftY: left.y,
            rightX: right.x,
            rightY: right.y,
        }
    } else {
        FfiFoveationCenters {
            valid: false,
            leftX: 0.0,
            leftY: 0.0,
            rightX: 0.0,
            rightY: 0.0,
        }
    }
}

#[unsafe(export_name = "SetEncoderFoveationCenters")]
extern "C" fn set_encoder_foveation_centers(
    timestamp_ns: u64,
    left_x: f32,
    left_y: f32,
    right_x: f32,
    right_y: f32,
) {
    RENDERED_CENTERS.lock().insert(
        Duration::from_nanos(timestamp_ns),
        [Vec2::new(left_x, left_y), Vec2::new(right_x, right_y)],
    );
}

pub fn update(
    timestamp: Duration,
    external_gaze: Option<GazeSample>,
    headset_gaze: Option<GazeSample>,
    view_params: [ViewParams; 2],
    config: &FoveatedEncodingConfig,
) -> GazeSource {
    let (selected_source, sample) = match config.gaze_input_source {
        GazeInputSource::None => (GazeSource::Static, None),
        GazeInputSource::Headset => (GazeSource::Headset, headset_gaze),
        GazeInputSource::ExternalUdp { .. } => (GazeSource::ExternalUdp, external_gaze),
    };
    let sample = sample.map(|sample| {
        (
            selected_source,
            sample.sample_timestamp,
            sample.directions.map(GazeDirection::direction),
        )
    });

    let has_live_sample = sample.is_some();
    #[cfg(feature = "foveation-diagnostics")]
    let diagnostic_sample = sample;
    #[cfg(feature = "foveation-diagnostics")]
    let (has_centers, centers) = {
        let mut state = FOVEATION_STATE.lock();
        let has_centers = state.update(timestamp, sample, view_params, config);

        (has_centers, state.get(timestamp))
    };
    #[cfg(not(feature = "foveation-diagnostics"))]
    let has_centers = {
        let mut state = FOVEATION_STATE.lock();
        state.update(timestamp, sample, view_params, config)
    };

    #[cfg(feature = "foveation-diagnostics")]
    if let (Some((source, sample_timestamp, directions)), Some(centers)) =
        (diagnostic_sample, centers)
    {
        let mut last_log_timestamp = LAST_GAZE_DIAGNOSTIC_LOG_TIMESTAMP.lock();
        let should_log = match *last_log_timestamp {
            Some(last_timestamp) => {
                timestamp < last_timestamp
                    || timestamp.saturating_sub(last_timestamp) >= GAZE_DIAGNOSTIC_LOG_INTERVAL
            }
            None => true,
        };

        if should_log {
            let [left_direction, right_direction] = directions;
            let [left_center, right_center] = centers;

            alvr_common::info!(
                "[Eye-tracked foveation diagnostic] source={}, age_ms={:.2}, \
                 left_dir=({:.4},{:.4},{:.4}), right_dir=({:.4},{:.4},{:.4}), \
                 left_center=({:.4},{:.4}), right_center=({:.4},{:.4})",
                source.description(),
                timestamp.saturating_sub(sample_timestamp).as_secs_f64() * 1000.0,
                left_direction.x,
                left_direction.y,
                left_direction.z,
                right_direction.x,
                right_direction.y,
                right_direction.z,
                left_center.x,
                left_center.y,
                right_center.x,
                right_center.y,
            );
            *last_log_timestamp = Some(timestamp);
        }
    }

    if has_live_sample && has_centers {
        selected_source
    } else if has_centers {
        GazeSource::Held
    } else {
        GazeSource::Static
    }
}

pub fn reset() {
    FOVEATION_STATE.lock().clear();
    RENDERED_CENTERS.lock().clear();
    #[cfg(feature = "foveation-diagnostics")]
    {
        *LAST_GAZE_DIAGNOSTIC_LOG_TIMESTAMP.lock() = None;
    }
}

pub fn rendered_centers(timestamp: Duration) -> Option<[Vec2; 2]> {
    RENDERED_CENTERS.lock().get(timestamp)
}
