//! How a model's decoding requests split into calls. Rows of one call share
//! the GPU's work per layer, so one call needs the least GPU time per token;
//! but while a call's rows wait for their experts to be read, the GPU idles,
//! and a second call's GPU work can fill the wait. So the worker runs either
//! as many full calls as the requests fill, the rest waiting (the oldest
//! first): best when the GPU is busy anyway (most experts in memory); or, while
//! the GPU idles waiting for reads, the requests split evenly over at least two
//! calls, every request in one.
//!
//! Measured on DeepSeek V4.1 Flash with UD-IQ1_M experts at 50 GiB: one call
//! left the GPU idle 24-26% of the time while its rows waited for reads, two
//! calls 5-9%, three 1-2%; two calls of about N/2 rows were the fastest for 2 to
//! 8 requests (8 requests: 2x4 47-50 tok/s in all, one 6-row call with two
//! waiting 41-43).

/// Splitting starts once the GPU was idle more than this share of a window
/// while the engine thread waited for expert reads, and ends below
/// MERGE_BELOW: between the two the arrangement stays.
pub const SPLIT_ABOVE: f64 = 0.12;
pub const MERGE_BELOW: f64 = 0.03;
/// Seconds of decoding that one decision looks at.
pub const WINDOW: f64 = 2.0;
/// Seconds after a change of the arrangement, of the decoding requests or
/// after anything else ran, before a window starts: the calls of the old
/// arrangement finish first.
pub const SETTLE: f64 = 1.0;

/// The engine's process-wide counters that a decision reads (engine_times):
/// the engine thread's seconds in all its states, and the GPU's idle seconds
/// while the engine thread waited for reads.
#[derive(Clone, Copy, Debug)]
pub struct Times {
    pub thread: f64,
    pub idle_reading: f64,
}

#[derive(Debug)]
pub struct CallChooser {
    /// The requests split evenly over at least two calls.
    split: bool,
    /// The decoding requests of the steady looks so far.
    decoding: Option<usize>,
    /// The window's start: its time and the counters then.
    start: Option<(f64, Times)>,
    /// No window starts before this time.
    quiet_until: f64,
}

impl Default for CallChooser {
    fn default() -> Self {
        Self::new()
    }
}

impl CallChooser {
    pub fn new() -> Self {
        CallChooser { split: false, decoding: None, start: None, quiet_until: 0.0 }
    }

    /// The calls that `decoding` requests run in at once and the rows of each,
    /// at most `most`. Not split: as many full calls as the requests fill (at
    /// least one), the rest waiting. Split: at least two calls and as few as
    /// `most` allows, every request in one, the rows as even as the calls
    /// allow (never more calls than requests).
    pub fn calls(&self, decoding: usize, most: usize) -> (usize, usize) {
        let most = most.max(1);
        if !self.split {
            return ((decoding / most).max(1), most);
        }
        let calls = decoding.div_ceil(most).max(2).min(decoding).max(1);
        (calls, decoding.div_ceil(calls).clamp(1, most))
    }

    /// Whether the requests split evenly over at least two calls.
    pub fn split(&self) -> bool {
        self.split
    }

    /// Looks at the counters at `now` (seconds). `steady`: two or more of
    /// this model's requests decode and nothing else runs (no prefill, no
    /// other model), so the counters show only these calls. A window starts
    /// SETTLE seconds after the last look that was not steady or had other
    /// decoding requests, and after a change of the arrangement; after WINDOW
    /// seconds it decides from the GPU's idle share while reads were awaited.
    pub fn observe(&mut self, now: f64, times: Times, decoding: usize, steady: bool) {
        if !steady || self.decoding != Some(decoding) {
            self.decoding = steady.then_some(decoding);
            self.start = None;
            self.quiet_until = now + SETTLE;
            return;
        }
        if now < self.quiet_until {
            return;
        }
        let Some((t0, base)) = self.start else {
            self.start = Some((now, times));
            return;
        };
        if now - t0 < WINDOW {
            return;
        }
        let share = (times.idle_reading - base.idle_reading) / (times.thread - base.thread).max(1e-9);
        if (self.split && share < MERGE_BELOW) || (!self.split && share > SPLIT_ABOVE) {
            self.split = !self.split;
            self.start = None;
            self.quiet_until = now + SETTLE;
        } else {
            self.start = Some((now, times));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Feeds `seconds` of decoding in quarter-second looks, the GPU idle
    /// `share` of the time while reads were awaited; returns the time and the
    /// counters after.
    fn run(c: &mut CallChooser, mut now: f64, mut t: Times, seconds: f64, share: f64, decoding: usize) -> (f64, Times) {
        let end = now + seconds;
        while now < end {
            now += 0.25;
            t.thread += 0.25;
            t.idle_reading += 0.25 * share;
            c.observe(now, t, decoding, true);
        }
        (now, t)
    }

    const START: Times = Times { thread: 100.0, idle_reading: 10.0 };

    #[test]
    fn one_call_that_waits_for_reads_becomes_two() {
        let mut c = CallChooser::new();
        assert_eq!(c.calls(4, 6), (1, 6));
        let (now, t) = run(&mut c, 0.0, START, SETTLE + WINDOW - 0.5, 0.25, 4);
        assert_eq!(c.calls(4, 6), (1, 6), "no decision before a whole window");
        run(&mut c, now, t, 1.0, 0.25, 4);
        assert_eq!(c.calls(4, 6), (2, 2));
        assert_eq!(c.calls(8, 6), (2, 4));
    }

    #[test]
    fn two_calls_that_keep_the_gpu_busy_become_one() {
        let mut c = CallChooser::new();
        let (now, t) = run(&mut c, 0.0, START, 4.0, 0.25, 4);
        assert!(c.split());
        let (now, t) = run(&mut c, now, t, 10.0, 0.07, 4);
        assert!(c.split(), "between the thresholds the arrangement stays");
        run(&mut c, now, t, 4.0, 0.01, 4);
        assert_eq!(c.calls(4, 6), (1, 6));
    }

    #[test]
    fn a_busy_gpu_keeps_one_call_and_the_middle_band_does_not_flap() {
        let mut c = CallChooser::new();
        let (now, t) = run(&mut c, 0.0, START, 20.0, 0.01, 4);
        assert!(!c.split());
        let (now, t) = run(&mut c, now, t, 20.0, 0.10, 4);
        assert!(!c.split(), "10% is below the split threshold");
        let (now, t) = run(&mut c, now, t, 4.0, 0.20, 4);
        assert!(c.split());
        run(&mut c, now, t, 20.0, 0.05, 4);
        assert!(c.split(), "5% is above the merge threshold");
    }

    #[test]
    fn windows_that_are_not_steady_or_change_the_requests_decide_nothing() {
        let mut c = CallChooser::new();
        let mut now = 0.0;
        let mut t = START;
        for k in 0..80 {
            now += 0.25;
            t.thread += 0.25;
            t.idle_reading += 0.25 * 0.3;
            c.observe(now, t, 4, k % 8 != 0); // something else runs every two seconds
        }
        assert!(!c.split());
        for k in 0..80 {
            now += 0.25;
            t.thread += 0.25;
            t.idle_reading += 0.25 * 0.3;
            c.observe(now, t, 3 + k % 2, true); // the decoding requests change
        }
        assert!(!c.split());
    }

    #[test]
    fn full_calls_leave_the_rest_waiting_and_split_calls_take_everyone() {
        let mut c = CallChooser::new();
        // as today: full calls of 6, the rest waiting
        assert_eq!(c.calls(1, 6), (1, 6));
        assert_eq!(c.calls(4, 6), (1, 6));
        assert_eq!(c.calls(7, 6), (1, 6), "one 6-row call, the seventh waits");
        assert_eq!(c.calls(12, 6), (2, 6));
        assert_eq!(c.calls(13, 6), (2, 6));
        run(&mut c, 0.0, START, 4.0, 0.3, 4);
        assert!(c.split());
        // split: every request in a call, the rows as even as the calls allow
        assert_eq!(c.calls(1, 6), (1, 1));
        assert_eq!(c.calls(2, 6), (2, 1));
        assert_eq!(c.calls(3, 6), (2, 2));
        assert_eq!(c.calls(6, 6), (2, 3));
        assert_eq!(c.calls(7, 6), (2, 4), "4 + 3");
        assert_eq!(c.calls(12, 6), (2, 6));
        assert_eq!(c.calls(13, 6), (3, 5), "5 + 5 + 3");
        assert_eq!(c.calls(4, 1), (4, 1), "one row per call");
    }
}
