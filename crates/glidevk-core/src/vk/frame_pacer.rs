use std::time::{Duration, Instant};

const DEFAULT_FPS: u32 = 60;
const MAX_FPS: u32 = 1000;

/// Limit presentation attempts without accumulating debt after a slow frame.
pub(super) struct FramePacer {
    interval: Option<Duration>,
    last_present: Option<Instant>,
}

impl FramePacer {
    pub(super) fn from_env() -> Self {
        Self::from_value(std::env::var("GLIDEVK_FPS").ok().as_deref())
    }

    fn from_value(value: Option<&str>) -> Self {
        let fps = value
            .and_then(|s| s.parse::<u32>().ok())
            .filter(|&fps| fps <= MAX_FPS)
            .unwrap_or(DEFAULT_FPS);
        Self {
            interval: (fps > 0)
                .then(|| Duration::from_nanos(1_000_000_000u64.div_ceil(fps as u64))),
            last_present: None,
        }
    }

    fn remaining(&self, now: Instant) -> Duration {
        match (self.interval, self.last_present) {
            (Some(interval), Some(last)) => {
                interval.saturating_sub(now.saturating_duration_since(last))
            }
            _ => Duration::ZERO,
        }
    }

    pub(super) fn wait(&mut self) {
        if self.interval.is_none() {
            return;
        }
        loop {
            let delay = self.remaining(Instant::now());
            if delay.is_zero() {
                break;
            }
            std::thread::sleep(delay);
        }
        self.last_present = Some(Instant::now());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_default_custom_and_uncapped_values() {
        assert_eq!(
            FramePacer::from_value(None).interval,
            Some(Duration::from_nanos(16_666_667))
        );
        assert_eq!(
            FramePacer::from_value(Some("30")).interval,
            Some(Duration::from_nanos(33_333_334))
        );
        assert_eq!(FramePacer::from_value(Some("0")).interval, None);
        assert_eq!(
            FramePacer::from_value(Some("bad")).interval,
            FramePacer::from_value(None).interval
        );
        assert_eq!(
            FramePacer::from_value(Some("1001")).interval,
            FramePacer::from_value(None).interval
        );
    }

    #[test]
    fn sleeps_only_for_the_remaining_frame_interval() {
        let mut pacer = FramePacer::from_value(Some("50"));
        let start = Instant::now();
        assert_eq!(pacer.remaining(start), Duration::ZERO);
        pacer.last_present = Some(start);
        assert_eq!(
            pacer.remaining(start + Duration::from_millis(7)),
            Duration::from_millis(13)
        );
        assert_eq!(
            pacer.remaining(start + Duration::from_millis(50)),
            Duration::ZERO
        );
        pacer.interval = None;
        assert_eq!(pacer.remaining(start), Duration::ZERO);
    }
}
