//! The Qwen worker: one thread owns the Qwen engine (engine.rs). Requests
//! arrive as prompt tokens over a queue (at most [`WAITING`] wait) and get
//! their events back on a channel, as the DeepSeek worker's do.
//!
//! Each pass of the loop runs one prompt pass of the oldest request still
//! prefilling ([`PREFILL`] positions, or [`SHARED_PREFILL`] while others
//! decode), then one decoding pass over every request that decodes (up to
//! [`ROWS`] rows), so that a long prompt does not stall the replies in
//! progress. A request that decodes alone verifies the MTP head's draft in
//! its pass (two rows); every decoded token gives the head its keys, so that
//! its drafts see the whole context. The prompt reuses the
//! states and snapshots that kv.rs's cache keeps, with no rewind room (a
//! DeltaNet state cannot go back) and no replay: in place, from a snapshot,
//! or cold. Every prompt end gets a snapshot with its logits, and a prefill
//! captures the point where its prompt parts from every state.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock};
use std::thread;
use std::time::{Duration, Instant};

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use tokio::sync::mpsc;

use super::engine::{self, PREFILL, ROWS, Row, Snap, State};
use crate::kv::{self, Kind, Plan};

/// Requests that may wait for their turn.
pub const WAITING: usize = 16;
/// A prompt pass's positions while other requests decode.
const SHARED_PREFILL: usize = 128;
/// A prefill snapshots its state at every multiple of this many positions
/// (within the KV cache's budget): a later prompt that shares a long prefix
/// with it but parts before its end (another conversation with the same
/// system prompt and tools) resumes from the last one before the point
/// where it parts.
const SNAP_GRID: usize = 2048;
/// Idle this long, the model leaves the GPU's residency set (engine::reside),
/// so that another model's process can keep more of its own resident.
const RELEASE_AFTER: Duration = Duration::from_secs(60);
const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sampling {
    pub temperature: f64,
    pub top_p: f64,
    /// 0: every token.
    pub top_k: usize,
    pub seed: Option<u64>,
}

pub struct Request {
    pub prompt: Vec<u32>,
    pub max_tokens: usize,
    pub sampling: Sampling,
    /// Logged when the request is queued.
    pub note: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Finish {
    /// An end token.
    Stop,
    /// max_tokens or the context's end.
    Length,
    Shutdown,
    Cancelled,
}

#[derive(Debug)]
pub enum Event {
    /// Its prefill ran: prompt tokens and those it reused.
    Started { prompt: usize, reused: usize },
    /// One decoded token; never the end token.
    Token(u32),
    /// Its end; the last event.
    Done { finish: Finish },
    /// Refused before its prefill; the last event.
    Refused(String),
}

pub enum SubmitError {
    Busy(String),
    Invalid(String),
}

#[derive(Default)]
struct Flags {
    cancelled: AtomicBool,
    ended: AtomicBool,
}

/// Cancels its request when dropped (the response ended), unless it ended.
pub struct Cancel {
    id: u64,
    flags: Arc<Flags>,
}

impl Drop for Cancel {
    fn drop(&mut self) {
        if !self.flags.ended.load(Ordering::SeqCst) && !self.flags.cancelled.swap(true, Ordering::SeqCst) {
            crate::engine::log(&format!("req {}: cancellation requested", self.id));
        }
    }
}

struct Job {
    id: u64,
    request: Request,
    tx: mpsc::UnboundedSender<Event>,
    flags: Arc<Flags>,
    queued: Instant,
}

impl Job {
    fn last(&self, e: Event) {
        let _ = self.tx.send(e);
        self.flags.ended.store(true, Ordering::SeqCst);
    }
}

#[derive(Default)]
struct Queue {
    jobs: VecDeque<Job>,
    closed: bool,
    numbers: u64,
}

struct Shared {
    queue: Mutex<Queue>,
    wake: Condvar,
    ctx: usize,
}

static SHARED: OnceLock<Arc<Shared>> = OnceLock::new();

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

pub struct Config {
    pub dir: String,
    pub batch: usize,
    pub kv_bytes: u64,
    pub ctx: usize,
}

/// Loads the model on the worker's thread and starts it; false if the load
/// failed (the engine logs why).
pub fn start(c: &Config) -> bool {
    let shared = Arc::new(Shared { queue: Mutex::new(Queue::default()), wake: Condvar::new(), ctx: c.ctx });
    if SHARED.set(shared.clone()).is_err() {
        return false;
    }
    let (ready_tx, ready_rx) = std::sync::mpsc::channel();
    let (dir, batch, kv_bytes) = (c.dir.clone(), c.batch.clamp(1, ROWS), c.kv_bytes);
    thread::Builder::new()
        .name("qwen".into())
        .spawn(move || {
            engine::load(&dir);
            let _ = ready_tx.send(());
            Worker::new(shared, batch, kv_bytes).run();
        })
        .expect("the worker thread starts");
    ready_rx.recv().is_ok()
}

/// Closes the queue and ends the requests in progress (at the next pass).
pub fn stop() {
    if let Some(s) = SHARED.get() {
        lock(&s.queue).closed = true;
        s.wake.notify_all();
    }
}

pub fn submit(r: Request) -> Result<(Cancel, mpsc::UnboundedReceiver<Event>), SubmitError> {
    let s = SHARED.get().ok_or(SubmitError::Busy("the model is not loaded".into()))?;
    if r.prompt.is_empty() {
        return Err(SubmitError::Invalid("the prompt is empty".into()));
    }
    let t = r.sampling;
    if !(t.temperature.is_finite() && t.temperature >= 0.0 && (0.0..=1.0).contains(&t.top_p)) {
        return Err(SubmitError::Invalid("temperature must be finite and nonnegative, top_p in [0, 1]".into()));
    }
    if r.prompt.len() >= s.ctx {
        return Err(SubmitError::Invalid(format!("the prompt has {} tokens; the context holds {}", r.prompt.len(), s.ctx)));
    }
    let (tx, rx) = mpsc::unbounded_channel();
    let flags = Arc::new(Flags::default());
    let mut q = lock(&s.queue);
    if q.closed || q.jobs.len() >= WAITING {
        return Err(SubmitError::Busy(if q.closed { "the server is stopping".into() } else { format!("{WAITING} requests wait already") }));
    }
    q.numbers += 1;
    let id = q.numbers;
    crate::engine::log(&format!("req {id}: {}", r.note));
    q.jobs.push_back(Job { id, request: r, tx, flags: flags.clone(), queued: Instant::now() });
    drop(q);
    s.wake.notify_all();
    Ok((Cancel { id, flags }, rx))
}

// ---- sampling ----

fn argmax(l: &[f32]) -> u32 {
    let mut best = 0;
    for (i, &v) in l.iter().enumerate() {
        if v > l[best] {
            best = i;
        }
    }
    best as u32
}

/// Whether token a ranks before token b: the higher logit, then the lower id.
fn before(l: &[f32], a: u32, b: u32) -> bool {
    l[a as usize].total_cmp(&l[b as usize]).then(b.cmp(&a)) == std::cmp::Ordering::Greater
}

/// The k highest-ranked tokens, in rank order, in one pass over the logits.
fn top_k(l: &[f32], k: usize) -> Vec<u32> {
    let mut best: Vec<u32> = Vec::with_capacity(k + 1);
    for i in 0..l.len() as u32 {
        if best.len() == k && !before(l, i, best[k - 1]) {
            continue;
        }
        let at = best.partition_point(|&b| before(l, b, i));
        best.insert(at, i);
        best.truncate(k);
    }
    best
}

/// A token from logits: greedy at temperature 0, else from the top_k
/// highest-ranked tokens (all with 0), cut to the top_p nucleus in rank
/// order, at the temperature. With all tokens and top_p 1 the draw goes
/// over the tokens in id order, unsorted.
fn sample(l: &[f32], t: &Sampling, rng: &mut StdRng) -> u32 {
    if t.temperature == 0.0 {
        return argmax(l);
    }
    let k = if t.top_k == 0 || t.top_k > l.len() { l.len() } else { t.top_k };
    let order: Vec<u32> = if k == l.len() && t.top_p >= 1.0 {
        (0..l.len() as u32).collect()
    } else if k <= 256 {
        top_k(l, k)
    } else {
        let mut order: Vec<u32> = (0..l.len() as u32).collect();
        let by = |a: &u32, b: &u32| if before(l, *a, *b) { std::cmp::Ordering::Less } else { std::cmp::Ordering::Greater };
        if k < l.len() {
            order.select_nth_unstable_by(k - 1, by);
            order.truncate(k);
        }
        order.sort_unstable_by(by);
        order
    };
    let top = order.iter().map(|&i| l[i as usize]).fold(f32::NEG_INFINITY, f32::max) as f64;
    let mut p: Vec<f64> = order.iter().map(|&i| ((l[i as usize] as f64 - top) / t.temperature).exp()).collect();
    let sum: f64 = p.iter().sum();
    let mut kept = 0.0;
    let mut n = 0;
    while n < p.len() && (n == 0 || kept < t.top_p * sum) {
        kept += p[n];
        n += 1;
    }
    p.truncate(n);
    let mut r = rng.random::<f64>() * kept;
    for (i, &w) in p.iter().enumerate() {
        if r < w {
            return order[i];
        }
        r -= w;
    }
    order[n - 1]
}

// ---- the worker ----

enum Phase {
    /// Prefilling: positions done of the prompt, and the branch capture.
    Prefill { done: usize, capture: Option<usize> },
    /// Decoding: the next token to run (sampled, already sent), and the MTP
    /// head's draft of the token after it.
    Decode { next: u32, draft: Option<u32> },
    Ended,
}

struct Run {
    job: Job,
    state: kv::Id,
    n: usize,
    reused: usize,
    phase: Phase,
    made: usize,
    max_new: usize,
    rng: StdRng,
    t0: Instant,
    t_decode: Option<Instant>,
    /// MTP drafts verified, and accepted.
    drafts: (usize, usize),
}

struct Worker {
    s: Arc<Shared>,
    batch: usize,
    cache: kv::Cache,
    states: HashMap<kv::Id, State>,
    snaps: HashMap<(kv::Id, usize), (Snap, Option<Vec<f32>>)>,
    runs: Vec<Run>,
    /// Admitted requests not planned yet: a request is planned when its
    /// prefill is about to start, after the prefills before it, so that it
    /// resumes from the states and snapshots they left (requests that arrive
    /// together with the same system prompt and tools share it).
    waiting: VecDeque<Job>,
    limits: kv::Limits,
    mtp: bool,
    /// What the cache gave up during a decoding pass, freed after it: the
    /// pass's rows point to its states.
    deferred: Option<Vec<kv::Freed>>,
}

fn no_replay(_: usize) -> usize {
    0
}

/// What a state of `positions` positions takes: the engine doubles its keys'
/// room from 1,024 positions.
fn state_bytes(positions: usize) -> u64 {
    let mut cap = 1024;
    while cap < positions {
        cap *= 2;
    }
    engine::state_bytes(cap)
}

impl Worker {
    fn new(s: Arc<Shared>, batch: usize, kv_bytes: u64) -> Worker {
        Worker {
            s,
            batch,
            cache: kv::Cache::new(kv_bytes, 256),
            states: HashMap::new(),
            snaps: HashMap::new(),
            runs: Vec::new(),
            waiting: VecDeque::new(),
            limits: kv::Limits { rewind: 0, replay: false, replay_cost: no_replay },
            mtp: engine::mtp_available(),
            deferred: None,
        }
    }

    fn run(mut self) {
        let mut busy = Instant::now(); // the last work
        let mut released = false;
        loop {
            let closed = {
                let mut q = lock(&self.s.queue);
                while q.jobs.is_empty() && self.runs.is_empty() && self.waiting.is_empty() && !q.closed && !crate::engine::stopping() {
                    q = self.s.wake.wait_timeout(q, Duration::from_millis(500)).map(|(g, _)| g).unwrap_or_else(|e| e.into_inner().0);
                    if !released && busy.elapsed() >= RELEASE_AFTER {
                        drop(q);
                        engine::reside(false);
                        released = true;
                        crate::engine::log(&format!(
                            "idle {} s: the model's buffers left the GPU's residency set",
                            RELEASE_AFTER.as_secs()
                        ));
                        q = lock(&self.s.queue);
                    }
                }
                q.closed || crate::engine::stopping()
            };
            if released && !closed {
                let t = Instant::now();
                engine::reside(true);
                released = false;
                crate::engine::log(&format!(
                    "the model's buffers are back in the GPU's residency set ({:.0} ms)",
                    t.elapsed().as_secs_f64() * 1e3
                ));
            }
            busy = Instant::now();
            if closed {
                for r in std::mem::take(&mut self.runs) {
                    r.job.last(Event::Done { finish: Finish::Shutdown });
                }
                for j in std::mem::take(&mut self.waiting) {
                    j.last(Event::Done { finish: Finish::Shutdown });
                }
                let mut q = lock(&self.s.queue);
                q.closed = true;
                for j in q.jobs.drain(..) {
                    j.last(Event::Done { finish: Finish::Shutdown });
                }
                return;
            }
            while self.runs.len() + self.waiting.len() < self.batch {
                let Some(job) = lock(&self.s.queue).jobs.pop_front() else { break };
                self.waiting.push_back(job);
            }
            self.cancelled();
            // the next prefill starts only after the one before it ended
            while !self.runs.iter().any(|r| matches!(r.phase, Phase::Prefill { .. }))
                && let Some(job) = self.waiting.pop_front()
            {
                if let Some(run) = self.begin(job) {
                    self.runs.push(run);
                }
            }
            self.prefill_chunk();
            self.decode_pass();
        }
    }

    fn free_snaps(&mut self, freed: kv::Freed) {
        if let Some(later) = &mut self.deferred {
            later.push(freed);
            return;
        }
        let (snaps, states) = freed;
        for (id, s) in snaps {
            self.snaps.remove(&(id, s.pos));
        }
        for (id, ss) in states {
            for s in ss {
                self.snaps.remove(&(id, s.pos));
            }
            self.states.remove(&id);
        }
    }

    /// A request's start: its plan, its state and its first prefill chunk to come.
    fn begin(&mut self, job: Job) -> Option<Run> {
        let prompt: Vec<i32> = job.request.prompt.iter().map(|&t| t as i32).collect();
        let n = prompt.len();
        let plan = self.cache.plan(&prompt, self.limits);
        let capture = plan.capture(n);
        let state_need = match plan {
            Plan::InPlace { state, .. } => {
                state_bytes(n + 1).saturating_sub(self.cache.entries.get(&state).map_or(0, |e| e.bytes))
            }
            _ => state_bytes(n + 1),
        };
        let snap = engine::snap_bytes();
        let need = state_need + plan.snapshot_need(n, snap, snap + 4 * engine::vocab() as u64);
        let keep: Vec<kv::Id> = plan.source().into_iter().collect();
        let Some(ev) = self.cache.fit(need, !matches!(plan, Plan::InPlace { .. }), &keep) else {
            let m = format!(
                "the conversation's state needs {:.1} GiB; the KV cache allows {:.1} GiB",
                state_bytes(n + 1) as f64 / GIB,
                self.cache.limit as f64 / GIB
            );
            crate::engine::log(&format!("req {}: refused: {m}", job.id));
            job.last(Event::Refused(m));
            return None;
        };
        let freed = self.cache.evict(&ev);
        self.free_snaps(freed);
        let from = plan.reused();
        let id = match plan {
            Plan::Cold | Plan::Replay { .. } => {
                let id = self.cache.add(0);
                self.states.insert(id, State::new());
                id
            }
            Plan::InPlace { state, from } => {
                self.cache.touch(state, from);
                if from == n {
                    let (snap, _) = &self.snaps[&(state, n)];
                    self.states.get_mut(&state).expect("a live state").restore(snap);
                }
                state
            }
            Plan::Snapshot { source, snap, at, .. } => {
                let id = self.cache.add(0);
                let mut st = State::new();
                st.reserve(n + 1);
                st.copy_from(&self.states[&source], snap);
                st.restore(&self.snaps[&(source, snap)].0);
                self.states.insert(id, st);
                self.cache.touch(source, at);
                id
            }
        };
        let st = self.states.get_mut(&id).expect("a live state");
        st.reserve(n + 1);
        let bytes = state_bytes(n + 1);
        for s in self.cache.start(id, &prompt, from, bytes, 0) {
            self.snaps.remove(&(id, s.pos));
        }
        let max_new = job.request.max_tokens.max(1).min(self.s.ctx - n);
        let seed = job.request.sampling.seed.unwrap_or_else(|| rand::rng().random());
        let number = self.cache.entries[&id].number;
        crate::engine::log(&format!(
            "req {}: {n} prompt tokens, session {number} ({}), {from} reused; at most {max_new} new tokens, {}; waited {:.1} s",
            job.id,
            plan.kind(n),
            if job.request.sampling.temperature > 0.0 { "sampled" } else { "greedy" },
            job.queued.elapsed().as_secs_f64()
        ));
        let mut run = Run {
            job,
            state: id,
            n,
            reused: from,
            phase: Phase::Prefill { done: from, capture },
            made: 0,
            max_new,
            rng: StdRng::seed_from_u64(seed),
            t0: Instant::now(),
            t_decode: None,
            drafts: (0, 0),
        };
        if from == n {
            // the same prompt again: its prompt-end snapshot holds the logits
            let source = plan.source().expect("a saved prompt");
            let logits = self.snaps[&(source, n)].1.clone().expect("a prompt end's logits");
            self.started(&mut run, &logits);
        }
        Some(run)
    }

    /// The prefill ended with these logits: the first token, or the end.
    fn started(&mut self, run: &mut Run, logits: &[f32]) {
        let _ = run.job.tx.send(Event::Started { prompt: run.n, reused: run.reused });
        let dt = run.t0.elapsed().as_secs_f64();
        if run.n > run.reused {
            crate::engine::log(&format!(
                "req {}: prefill of {} tokens in {dt:.2} s ({:.0} tokens/s)",
                run.job.id,
                run.n - run.reused,
                (run.n - run.reused) as f64 / dt.max(1e-9)
            ));
        }
        run.t_decode = Some(Instant::now());
        let next = sample(logits, &run.job.request.sampling, &mut run.rng);
        run.phase = Phase::Decode { next, draft: None };
        self.emit(run, next);
        if self.mtp && !Self::ended(run) {
            let st: *mut State = self.states.get_mut(&run.state).expect("a live state");
            let l = engine::mtp(&[Row { state: st, token: next, save: false }], 1);
            run.phase = Phase::Decode { next, draft: Some(argmax(&l)) };
        }
    }

    /// A sampled token: the end, the token limit, or one more token.
    fn emit(&mut self, run: &mut Run, token: u32) {
        if engine::is_stop(token) {
            run.made += 1;
            self.end(run, Finish::Stop);
            return;
        }
        let _ = run.job.tx.send(Event::Token(token));
        run.made += 1;
        if run.made >= run.max_new {
            self.end(run, Finish::Length);
        }
    }

    /// Marks the run ended (removed after the pass).
    fn end(&mut self, run: &mut Run, finish: Finish) {
        let st = &self.states[&run.state];
        let pos = st.pos();
        // the tokens whose rows the state holds: a prefill that ended early ran part of its prompt
        if let Some(e) = self.cache.entries.get_mut(&run.state) {
            e.tokens.truncate(pos);
        }
        let decode = run.t_decode.map_or(0.0, |t| t.elapsed().as_secs_f64());
        let drafts = if run.drafts.0 > 0 {
            format!(", {} of {} MTP drafts accepted", run.drafts.1, run.drafts.0)
        } else {
            String::new()
        };
        crate::engine::log(&format!(
            "req {}: {} after {} tokens ({:.2} tokens/s{drafts}), {:.2} s in all",
            run.job.id,
            match finish {
                Finish::Stop => "end token",
                Finish::Length => "token limit",
                Finish::Shutdown => "server stop",
                Finish::Cancelled => "cancelled",
            },
            run.made,
            run.made.saturating_sub(1) as f64 / decode.max(1e-9),
            run.t0.elapsed().as_secs_f64()
        ));
        let bytes = state_bytes(pos + 1);
        let have = self.cache.snap(run.state, pos).is_some();
        let end = (!have).then(|| kv::Snap { pos, kind: Kind::GenerationEnd, bytes: engine::snap_bytes(), stale: 0 });
        let (take, freed) = self.cache.end(run.state, bytes, end);
        if take {
            let snap = self.states[&run.state].snap();
            self.snaps.insert((run.state, pos), (snap, None));
        }
        self.free_snaps(freed);
        run.job.last(Event::Done { finish });
        run.phase = Phase::Ended;
    }

    fn ended(r: &Run) -> bool {
        matches!(r.phase, Phase::Ended)
    }

    fn cancelled(&mut self) {
        let (gone, kept): (VecDeque<Job>, VecDeque<Job>) =
            std::mem::take(&mut self.waiting).into_iter().partition(|j| j.flags.cancelled.load(Ordering::SeqCst));
        for j in gone {
            crate::engine::log(&format!("req {}: cancelled before its prefill", j.id));
            j.last(Event::Done { finish: Finish::Cancelled });
        }
        self.waiting = kept;
        let mut runs = std::mem::take(&mut self.runs);
        for r in runs.iter_mut() {
            if !Self::ended(r) && r.job.flags.cancelled.load(Ordering::SeqCst) {
                self.end(r, Finish::Cancelled);
            }
        }
        runs.retain(|r| !Self::ended(r));
        self.runs = runs;
    }

    /// One chunk of the oldest prefill.
    fn prefill_chunk(&mut self) {
        let Some(i) = self.runs.iter().position(|r| matches!(r.phase, Phase::Prefill { .. })) else { return };
        let mut run = self.runs.remove(i);
        let Phase::Prefill { done, capture } = run.phase else { unreachable!() };
        let shared = self.runs.iter().any(|r| matches!(r.phase, Phase::Decode { .. }));
        let mut end = (done + if shared { SHARED_PREFILL } else { PREFILL }).min(run.n).min((done / SNAP_GRID + 1) * SNAP_GRID);
        if let Some(at) = capture
            && done < at
            && at < end
        {
            end = at;
        }
        let tokens = &run.job.request.prompt[done..end];
        let last = end == run.n;
        let st: *mut State = self.states.get_mut(&run.state).expect("a live state");
        let rows: Vec<Row> = tokens.iter().map(|&token| Row { state: st, token, save: false }).collect();
        let logits = engine::prefill(&rows, usize::from(last));
        let grid = end % SNAP_GRID == 0 && self.cache.snap(run.state, end).is_none();
        if !last && (capture == Some(end) || grid) {
            let bytes = engine::snap_bytes();
            let fits = capture == Some(end) || self.cache.fit(bytes, false, &[run.state]).is_some_and(|ev| {
                let freed = self.cache.evict(&ev);
                self.free_snaps(freed);
                true
            });
            if fits {
                let snap = self.states[&run.state].snap();
                self.snaps.insert((run.state, end), (snap, None));
                self.cache.add_snap(run.state, kv::Snap { pos: end, kind: Kind::Branch, bytes, stale: 0 });
            }
        }
        if last {
            let snap = self.states[&run.state].snap();
            self.snaps.insert((run.state, end), (snap, Some(logits.clone())));
            let bytes = engine::snap_bytes() + 4 * logits.len() as u64;
            self.cache.add_snap(run.state, kv::Snap { pos: end, kind: Kind::PromptEnd, bytes, stale: 0 });
            self.started(&mut run, &logits);
        } else {
            run.phase = Phase::Prefill { done: end, capture };
        }
        if !Self::ended(&run) {
            let at = i.min(self.runs.len());
            self.runs.insert(at, run);
        }
    }

    /// One row for each decoding request; a request decoding alone also
    /// verifies its draft. Then the MTP head's keys for the tokens decoded,
    /// and its next draft.
    fn decode_pass(&mut self) {
        let idx: Vec<usize> = (0..self.runs.len().min(ROWS))
            .filter(|&i| matches!(self.runs[i].phase, Phase::Decode { .. }))
            .collect();
        if idx.is_empty() {
            return;
        }
        let alone = idx.len() == 1;
        let mut rows: Vec<Row> = Vec::new();
        let mut first: Vec<(usize, bool)> = Vec::new(); // each run's first row, and whether it verifies a draft
        for &i in &idx {
            let r = &self.runs[i];
            let Phase::Decode { next, draft } = r.phase else { unreachable!() };
            let st: *mut State = self.states.get_mut(&r.state).expect("a live state");
            let room = unsafe { (*st).pos() } + 3 < self.s.ctx;
            let verify = alone && room && draft.is_some();
            first.push((rows.len(), verify));
            rows.push(Row { state: st, token: next, save: verify });
            if verify {
                rows.push(Row { state: st, token: draft.expect("a draft"), save: false });
            }
        }
        let logits = engine::forward(&rows, rows.len());
        let vocab = engine::vocab();
        let at = |row: usize| &logits[row * vocab..(row + 1) * vocab];
        self.deferred = Some(Vec::new());
        let mut fill: Vec<Row> = Vec::new(); // MTP rows that only give keys
        let mut drafting: Vec<(usize, Row)> = Vec::new(); // a run's MTP row with a draft
        let mut runs = std::mem::take(&mut self.runs);
        for (j, &i) in idx.iter().enumerate() {
            let (o, verify) = first[j];
            let r = &mut runs[i];
            let Phase::Decode { next, draft } = r.phase else { unreachable!() };
            let st = rows[o].state;
            self.cache.push(r.state, next as i32);
            if r.job.flags.cancelled.load(Ordering::SeqCst) {
                if verify {
                    unsafe { (*st).back() };
                }
                self.end(r, Finish::Cancelled);
                continue;
            }
            let y = sample(at(o), &r.job.request.sampling, &mut r.rng);
            let kept = verify && Some(y) == draft;
            if verify {
                r.drafts.0 += 1;
                if kept {
                    r.drafts.1 += 1;
                    self.cache.push(r.state, y as i32);
                    if self.mtp {
                        fill.push(Row { state: st, token: y, save: true }); // the kept row's key, ended or not
                    }
                } else {
                    unsafe { (*st).back() };
                }
            }
            // the last token sampled holds position pos; another needs pos + 1 below the end
            let pos = unsafe { (*st).pos() };
            let freed = self.cache.grow(r.state, state_bytes(pos + 3));
            self.free_snaps(freed);
            self.emit(r, y);
            if Self::ended(r) {
                continue;
            }
            let mut last = y;
            if kept {
                // the draft was the token sampled: its row's logits give the next
                let y2 = sample(at(o + 1), &r.job.request.sampling, &mut r.rng);
                self.emit(r, y2);
                if Self::ended(r) {
                    continue;
                }
                last = y2;
            }
            if pos + 1 >= self.s.ctx {
                self.end(r, Finish::Length);
                continue;
            }
            r.phase = Phase::Decode { next: last, draft: None };
            let row = Row { state: st, token: last, save: false };
            if self.mtp {
                if alone { drafting.push((i, row)) } else { fill.push(row) }
            }
        }
        if !fill.is_empty() || !drafting.is_empty() {
            let n = drafting.len();
            let mut m = fill;
            m.extend(drafting.iter().map(|&(_, row)| row));
            let l = engine::mtp(&m, n);
            for (k, &(i, _)) in drafting.iter().enumerate() {
                if let Phase::Decode { next, .. } = runs[i].phase {
                    runs[i].phase = Phase::Decode { next, draft: Some(argmax(&l[k * vocab..(k + 1) * vocab])) };
                }
            }
        }
        runs.retain(|r| !Self::ended(r));
        self.runs = runs;
        for freed in self.deferred.take().unwrap_or_default() {
            self.free_snaps(freed);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The one-pass top-k gives the tokens and the order a full sort gives,
    /// so a seeded draw takes the same token.
    #[test]
    fn top_k_ranks_as_a_full_sort() {
        let mut rng = StdRng::seed_from_u64(3);
        for case in 0..200 {
            let n = 1 + case * 37 % 3000;
            // few distinct values, so that ties are common
            let l: Vec<f32> = (0..n).map(|_| (rng.random::<u32>() % 50) as f32 * 0.25).collect();
            let mut all: Vec<u32> = (0..n as u32).collect();
            all.sort_by(|&a, &b| l[b as usize].total_cmp(&l[a as usize]).then(a.cmp(&b)));
            for k in [1, 2, 7, 20, 64, 256] {
                let k = k.min(n);
                assert_eq!(top_k(&l, k), all[..k].to_vec(), "n {n} k {k}");
            }
        }
    }

    #[test]
    fn sampling_keeps_greedy_top_k_and_top_p() {
        let l = [0.0f32, 3.0, 1.0, 2.9];
        let mut rng = StdRng::seed_from_u64(1);
        let greedy = Sampling { temperature: 0.0, top_p: 1.0, top_k: 0, seed: None };
        assert_eq!(sample(&l, &greedy, &mut rng), 1);
        let top1 = Sampling { temperature: 1.0, top_p: 1.0, top_k: 1, seed: None };
        for _ in 0..20 {
            assert_eq!(sample(&l, &top1, &mut rng), 1);
        }
        let top2 = Sampling { temperature: 1.0, top_p: 1.0, top_k: 2, seed: None };
        for _ in 0..50 {
            assert!(matches!(sample(&l, &top2, &mut rng), 1 | 3));
        }
        let nucleus = Sampling { temperature: 1.0, top_p: 0.01, top_k: 0, seed: None };
        for _ in 0..20 {
            assert_eq!(sample(&l, &nucleus, &mut rng), 1);
        }
    }
}
