use alvr_common::{glam::Vec2, parking_lot::Mutex};
use std::{collections::VecDeque, time::Duration};

const CENTER_HISTORY_CAPACITY: usize = 360;

static RENDERED_CENTERS: Mutex<VecDeque<(Duration, [Vec2; 2])>> = Mutex::new(VecDeque::new());

#[unsafe(export_name = "SetEncoderFoveationCenters")]
extern "C" fn set_encoder_foveation_centers(
    timestamp_ns: u64,
    left_x: f32,
    left_y: f32,
    right_x: f32,
    right_y: f32,
) {
    let mut history = RENDERED_CENTERS.lock();
    history.push_back((
        Duration::from_nanos(timestamp_ns),
        [Vec2::new(left_x, left_y), Vec2::new(right_x, right_y)],
    ));

    while history.len() > CENTER_HISTORY_CAPACITY {
        history.pop_front();
    }
}

pub fn reset() {
    RENDERED_CENTERS.lock().clear();
}

pub fn rendered_centers(timestamp: Duration) -> Option<[Vec2; 2]> {
    RENDERED_CENTERS
        .lock()
        .iter()
        .rev()
        .find_map(|(sample_timestamp, centers)| {
            (*sample_timestamp == timestamp).then_some(*centers)
        })
}
