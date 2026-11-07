//! What the view keeps of the last minute: one sample per snapshot, twice a
//! second, with the figures its tabs draw and each decoding request's routed
//! row of its latest decode step. The samples live in the view's memory only:
//! nothing is written anywhere, and they are gone when the view ends.

use std::collections::VecDeque;

use super::{Decode, Phase, Snapshot};

/// Samples kept: 60 s at two a second.
pub const SAMPLES: usize = 120;

/// A decoding request's routed row: its experts, layer by layer, and where
/// each one was (an index into the theme's sources).
#[derive(Clone, Debug, PartialEq)]
pub struct Route {
    pub request: u64,
    pub experts: Vec<u16>,
    pub sources: Vec<u8>,
}

#[derive(Clone, Debug, Default)]
pub struct Sample {
    /// When the snapshot was taken: seconds since the view started.
    pub at: f64,
    pub phase: Phase,
    /// The decoded rows' counters since the start.
    pub decode: Decode,
    /// Every expert byte read since the start, as the reads complete: the
    /// decoded rows' and the prefill's counters take theirs only at the end
    /// of a step or of a prefill's chunk.
    pub read_bytes: u64,
    /// The prompt rows computed since the start: the engine counts a chunk's
    /// rows when it ends; the chunk running counts by the layers it went
    /// through.
    pub prompt_rows: f64,
    /// The decoding requests' tokens/s.
    pub speeds: Vec<(u64, f64)>,
    /// The requests in their prefill.
    pub prefilling: Vec<u64>,
    /// The requests that came since the sample before.
    pub arrivals: Vec<u64>,
    /// While a prefill runs: the prompt rows computed per second since the
    /// sample before.
    pub prefill_rate: f64,
    /// Experts read from the SSD, GB/s.
    pub ssd: f64,
    /// Of the decoded rows' experts, the share in memory, read ahead or late.
    pub hit: Option<f64>,
    /// The GPU's busy share of the decode steps' time.
    pub gpu: Option<f64>,
    /// The wait for expert reads, ms per decoded row.
    pub wait_ms: Option<f64>,
    pub routes: Vec<Route>,
}

#[derive(Default)]
pub struct History {
    pub samples: VecDeque<Sample>,
    /// The highest request number seen.
    newest: u64,
    /// Each decoding request's (time, tokens generated) in its latest ten
    /// samples.
    tracks: Vec<(u64, VecDeque<(f64, u64)>)>,
    /// The Mac's swap-outs in the first sample.
    pub swapouts: Option<u64>,
}

impl History {
    /// A sample of the snapshot taken `now` seconds after the view started.
    pub fn observe(&mut self, s: &Snapshot, now: f64) {
        let phase = s.phase;
        let numbers: Vec<u64> = s.requests.iter().map(|r| r.number).collect();
        let arrivals: Vec<u64> = numbers.iter().copied().filter(|&n| n > self.newest).collect();
        self.newest = numbers.iter().copied().fold(self.newest, u64::max);
        let speeds: Vec<(u64, f64)> =
            s.requests.iter().filter(|r| r.generated > 0).map(|r| (r.number, self.speed(r.number, r.generated, now))).collect();
        self.tracks.retain(|t| s.requests.iter().any(|r| r.number == t.0));
        self.swapouts.get_or_insert(s.memory.swapouts);
        let prompt_rows = s.prompts.rows as f64 + s.prefill.rows as f64 * s.prefill.layer as f64 / s.layers.max(1) as f64;
        let (mut ssd, mut prefill_rate, mut hit, mut gpu, mut wait_ms) = (0.0, 0.0, None, None, None);
        if let Some(p) = self.samples.back() {
            let dt = now - p.at;
            if dt > 0.0 {
                ssd = s.cache.read_bytes.saturating_sub(p.read_bytes) as f64 / dt / 1e9;
                if phase == Phase::Prefill {
                    prefill_rate = (prompt_rows - p.prompt_rows).max(0.0) / dt;
                }
            }
            let d = s.decode.since(&p.decode);
            (hit, gpu, wait_ms) = (d.hit(), d.gpu_busy(), d.wait_ms());
        }
        let routes = routes(s);
        let prefilling = if phase == Phase::Prefill { numbers } else { Vec::new() };
        let (at, decode, read_bytes) = (now, s.decode, s.cache.read_bytes);
        self.push(Sample { at, phase, decode, read_bytes, prompt_rows, speeds, prefilling, arrivals, prefill_rate, ssd, hit, gpu, wait_ms, routes });
    }

    /// A request's tokens/s over its latest ten samples (five seconds).
    pub(super) fn speed(&mut self, number: u64, generated: u64, now: f64) -> f64 {
        let k = match self.tracks.iter().position(|t| t.0 == number) {
            Some(k) => k,
            None => {
                self.tracks.push((number, VecDeque::new()));
                self.tracks.len() - 1
            }
        };
        let samples = &mut self.tracks[k].1;
        if samples.len() == 10 {
            samples.pop_front();
        }
        samples.push_back((now, generated));
        let (&(t0, g0), &(t1, g1)) = (samples.front().unwrap(), samples.back().unwrap());
        if t1 > t0 { g1.saturating_sub(g0) as f64 / (t1 - t0) } else { 0.0 }
    }

    /// The decoding requests' tokens/s in the latest sample.
    pub fn speeds(&self) -> &[(u64, f64)] {
        self.samples.back().map_or(&[], |x| &x.speeds)
    }

    pub fn push(&mut self, s: Sample) {
        if self.samples.len() == SAMPLES {
            self.samples.pop_front();
        }
        self.samples.push_back(s);
    }

    /// The request's latest n routed rows, oldest first.
    pub fn routes(&self, request: u64, n: usize) -> Vec<&Route> {
        let mut r: Vec<&Route> =
            self.samples.iter().rev().filter_map(|s| s.routes.iter().find(|r| r.request == request)).take(n).collect();
        r.reverse();
        r
    }

    /// The mean of a figure over the latest n samples that have it.
    pub fn mean(&self, n: usize, f: impl Fn(&Sample) -> Option<f64>) -> Option<f64> {
        let v: Vec<f64> = self.samples.iter().rev().take(n).filter_map(f).collect();
        if v.is_empty() { None } else { Some(v.iter().sum::<f64>() / v.len() as f64) }
    }
}

/// The snapshot's routed rows: of the decoding requests that have a whole one.
pub fn routes(s: &Snapshot) -> Vec<Route> {
    let whole = |n: usize| n > 0 && n == s.experts_per_row as usize;
    s.requests
        .iter()
        .filter_map(|r| {
            let x = r.route.as_ref().filter(|x| whole(x.experts.len()) && whole(x.sources.len()))?;
            Some(Route { request: r.number, experts: x.experts.clone(), sources: x.sources.clone() })
        })
        .collect()
}
