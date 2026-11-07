//! The server's worker (docs/server.md): one thread that loads the model and
//! makes every engine call. HTTP handlers submit requests to a bounded queue
//! (at most [`WAITING`] wait) and get their events back on a channel. The
//! worker admits waiting requests while fewer than --batch decode, prefills
//! each while the others wait (resuming from the KV prefix cache, kv.rs),
//! then advances every decoding request by a token: alone with its drafts
//! (--draft, --lookup), or together with the others in one step of
//! myds4_rows. It also builds GET /v1/status and the live view's snapshot.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use serde_json::{Value, json};
use tokio::sync::mpsc;

use crate::engine::{self, Images, Logits, Lookup, PrefillInfo, RawSnapshot, RawState, Sampler};
use crate::kv::{self, Cache, Kind, Limits, Plan, Snap};

/// Requests waiting for the model at most; more are answered 429.
pub const WAITING: usize = 16;
/// Attention states kept at most, whatever --kv-cache allows.
pub const STATES: usize = 256;
const IMAGE_TOKEN: i32 = 129264;
const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

// ---- requests: the handlers' side ----

/// How a request ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Finish {
    /// The end token.
    Stop,
    /// `max_tokens`, or the context's end.
    Length,
    Cancelled,
    /// The server's stop.
    Shutdown,
}

/// What the worker reports about a request.
#[derive(Debug)]
pub enum Event {
    /// Its prefill ran: prompt tokens (images expanded), and those it reused.
    Started { prompt: usize, reused: usize },
    /// The tokens one decoding step kept; never the end token.
    Tokens(Vec<u32>),
    /// Its end; the last event. (The worker logs its counts.)
    Done { finish: Finish },
    /// Refused before its prefill; the last event.
    Refused(String),
}

/// A request: the rendered prompt (the byte 0xFF at each image's place), its
/// images' file bytes, sampling and its token limit.
pub struct Request<'a> {
    pub prompt: &'a [u8],
    pub images: &'a [Vec<u8>],
    pub temperature: f64,
    pub top_p: f64,
    pub seed: Option<u64>,
    pub max_tokens: i64,
    /// Logged as "req ID NOTE" when the request is queued.
    pub note: &'a str,
}

pub enum SubmitError {
    /// Too many requests wait, or the server stops.
    Busy(String),
    Invalid(String),
}

#[derive(Default)]
struct Flags {
    cancelled: AtomicBool,
    ended: AtomicBool,
}

/// Cancels its request when dropped: when the response ends, whether it was
/// finished, cut by its client before or after it started, or ended early by
/// a stop sequence. A request that ended already is left alone.
pub struct Cancel {
    id: u64,
    flags: Arc<Flags>,
}

impl Cancel {
    /// The request's id, its number in the log.
    pub fn id(&self) -> u64 {
        self.id
    }
}

impl Drop for Cancel {
    fn drop(&mut self) {
        if !self.flags.ended.load(Ordering::SeqCst) && !self.flags.cancelled.swap(true, Ordering::SeqCst) {
            engine::log(&format!("req {}: cancellation requested", self.id));
        }
    }
}

struct Job {
    id: u64,
    prompt: Vec<i32>,
    images: Vec<Vec<u8>>,
    temperature: f64,
    top_p: f64,
    seed: Option<u64>,
    max_tokens: i64,
    tx: mpsc::UnboundedSender<Event>,
    flags: Arc<Flags>,
    queued: Instant,
}

impl Job {
    /// The request's last event.
    fn last(&self, e: Event) {
        let _ = self.tx.send(e);
        self.flags.ended.store(true, Ordering::SeqCst);
    }
}

// ---- what the threads share ----

#[derive(Default)]
struct Queue {
    jobs: VecDeque<Job>,
    closed: bool,
    numbers: u64,
    /// The live view's question for a node's text: a state's number and
    /// positions start..end ([`ask_text`]).
    text: Option<(u64, usize, usize)>,
}

/// The tokens of a state's positions start..end, for the live view's text of
/// a node: at most [`TEXT_TOKENS`] of them from start, none once the state
/// is gone. The view turns them into text (engine::text, from any thread).
pub struct NodeText {
    pub session: u64,
    pub start: usize,
    pub end: usize,
    pub ids: Vec<i32>,
}

/// The tokens a node's text holds at most.
const TEXT_TOKENS: usize = 65_536;

/// A decoding request's routed row of one decode step: the experts its token
/// chose in each layer and where each one was (myds4_routes).
struct Route {
    request: u64,
    experts: Vec<u16>,
    sources: Vec<u8>,
}

/// Whether the live view asked for the decoding requests' routed rows: it
/// asks with each snapshot, the worker copies them after its next decode
/// step, and the next snapshot takes them, so the view gets each row once.
static ROUTES: AtomicBool = AtomicBool::new(false);

/// The live view asks for the routed rows of the next decode step.
pub fn ask_routes() {
    ROUTES.store(true, Ordering::Relaxed);
}

#[derive(Clone, Copy, Default)]
struct LiveReq {
    number: u64,
    session: u64,
    prompt: usize,
    reused: usize,
    generated: usize,
}

#[derive(Clone, Copy, Default)]
struct Prefilled {
    rows: u64,
    bytes: u64,
    seconds: f64,
}

/// What the worker is doing and has done, for /v1/status and the live view.
struct Status {
    phase: &'static str,
    since: Instant,
    active: usize,
    request: u64,
    prompt: usize,
    reused: usize,
    generated: usize,
    live: Vec<LiveReq>,
    requests: u64,
    prompt_tokens: u64,
    reused_tokens: u64,
    generated_tokens: u64,
    kinds: [u64; 4], // new, continued, saved, forked
    prefill: PrefillInfo,
    /// The ended prefills: their time, and the engine's prompt rows and their
    /// expert bytes when the latest ended. The engine counts those at each
    /// chunk's end; taken together, their ratios hold while a long
    /// prefill's chunks go on.
    prefilled: Prefilled,
    /// The tokens generated since the start: by the requests that ended and
    /// by those decoding.
    decoded: u64,
    /// The worker's answer to the live view's question for a node's text.
    text: Option<NodeText>,
    /// The decoding requests' latest routed rows, for the live view.
    routes: Vec<Route>,
    /// The cache's states (number, tokens, last use), its tree, bytes and limit.
    states: Vec<(u64, usize, Instant)>,
    tree: Vec<(kv::Node, Instant)>,
    kv_bytes: u64,
}

struct Shared {
    queue: Mutex<Queue>,
    wake: Condvar,
    status: Mutex<Status>,
    started: Instant,
    ced: i32,
    lookup: i32,
    expert_batch: bool,
    prefill_prune: bool,
    runahead: bool,
    kv_limit: f64,
}

static SHARED: Mutex<Option<Arc<Shared>>> = Mutex::new(None);
static THREAD: Mutex<Option<JoinHandle<()>>> = Mutex::new(None);

fn shared() -> Arc<Shared> {
    SHARED.lock().unwrap().clone().expect("the worker started")
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// Starts the worker's thread, which loads the model; returns once it takes
/// requests, or false for invalid settings (the engine says why on stderr).
pub fn start(c: &engine::Config) -> bool {
    let s = Arc::new(Shared {
        queue: Mutex::new(Queue::default()),
        wake: Condvar::new(),
        status: Mutex::new(Status {
            phase: "idle",
            since: Instant::now(),
            active: 0,
            request: 0,
            prompt: 0,
            reused: 0,
            generated: 0,
            live: Vec::new(),
            requests: 0,
            prompt_tokens: 0,
            reused_tokens: 0,
            generated_tokens: 0,
            kinds: [0; 4],
            prefill: PrefillInfo::default(),
            prefilled: Prefilled::default(),
            decoded: 0,
            text: None,
            routes: Vec::new(),
            states: Vec::new(),
            tree: Vec::new(),
            kv_bytes: 0,
        }),
        started: Instant::now(),
        ced: c.ced,
        lookup: c.lookup,
        expert_batch: c.expert_batch,
        prefill_prune: c.prefill_prune,
        runahead: c.runahead,
        kv_limit: c.kv_gb,
    });
    *SHARED.lock().unwrap() = Some(s.clone());
    let (tx, rx) = std::sync::mpsc::channel();
    let c = c.clone();
    let handle = thread::Builder::new()
        .name("myds4-worker".into())
        .stack_size(16 << 20)
        .spawn(move || {
            if !engine::load(&c) {
                let _ = tx.send(false);
                return;
            }
            // A failure here ends the process, as the engine's own do: the
            // requests could not end otherwise. (The panic's message is out.)
            let run = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let mut w = Worker::new(c, s);
                let _ = tx.send(true);
                w.run();
            }));
            if run.is_err() {
                crate::view::leave();
                std::process::exit(1);
            }
            engine::finish();
        })
        .expect("the worker's thread");
    let ok = rx.recv().unwrap_or(false);
    if ok {
        *THREAD.lock().unwrap() = Some(handle);
    } else {
        let _ = handle.join();
    }
    ok
}

/// Requests the stop as SIGINT does and wakes the worker (the live view's q).
pub fn request_stop() {
    engine::request_stop();
    if let Some(s) = SHARED.lock().unwrap().clone() {
        let _q = lock(&s.queue);
        s.wake.notify_all();
    }
}

/// The live view asks for the text of state `session`'s positions start..end;
/// the worker answers between its steps ([`text_answer`]). A new question
/// replaces one not answered yet.
pub fn ask_text(session: u64, start: usize, end: usize) {
    if let Some(s) = SHARED.lock().unwrap().clone() {
        lock(&s.queue).text = Some((session, start, end));
        s.wake.notify_all();
    }
}

/// The worker's answer to the latest [`ask_text`], once.
pub fn text_answer() -> Option<NodeText> {
    let s = SHARED.lock().unwrap().clone()?;
    lock(&s.status).text.take()
}

/// Requests the stop, waits for the worker to end every request, save the
/// predictor state and return.
pub fn stop() {
    request_stop();
    if let Some(h) = THREAD.lock().unwrap().take() {
        let _ = h.join();
    }
}

/// Queues a request: the guard that cancels it when dropped (and holds its
/// id), and its events. It is tokenized here, on the caller's thread.
pub fn submit(r: &Request) -> Result<(Cancel, mpsc::UnboundedReceiver<Event>), SubmitError> {
    let bad = if r.prompt.is_empty() {
        Some("the prompt is empty")
    } else if !(r.temperature.is_finite() && r.temperature >= 0.0 && (0.0..=1.0).contains(&r.top_p)) {
        Some("temperature must be a finite nonnegative number and top_p must be in [0, 1]")
    } else if r.max_tokens < 1 {
        Some("max_tokens must be at least 1")
    } else {
        None
    };
    if let Some(m) = bad {
        return Err(SubmitError::Invalid(m.into()));
    }
    let prompt = engine::tokenize_prompt(r.prompt);
    let places = prompt.iter().filter(|&&t| t == IMAGE_TOKEN).count();
    if prompt.is_empty() {
        return Err(SubmitError::Invalid("the prompt is empty".into()));
    }
    if places != r.images.len() {
        return Err(SubmitError::Invalid(format!("the prompt has {places} image places and {} images", r.images.len())));
    }
    let s = shared();
    let (tx, rx) = mpsc::unbounded_channel();
    let flags = Arc::new(Flags::default());
    let mut q = lock(&s.queue);
    if q.closed || q.jobs.len() >= WAITING {
        let m = if q.closed { "the server is stopping" } else { "the server is busy: too many requests are waiting" };
        return Err(SubmitError::Busy(m.into()));
    }
    q.numbers += 1;
    let id = q.numbers;
    if !r.note.is_empty() {
        engine::log(&format!("req {id} {}", r.note)); // before the worker can take it
    }
    let n = prompt.len();
    q.jobs.push_back(Job {
        id,
        prompt,
        images: r.images.to_vec(),
        temperature: r.temperature,
        top_p: r.top_p,
        seed: r.seed,
        max_tokens: r.max_tokens,
        tx,
        flags: flags.clone(),
        queued: Instant::now(),
    });
    engine::log(&format!("req {id}: queued, waiting={}, prompt_tokens={n}, max_new={}", q.jobs.len(), r.max_tokens));
    s.wake.notify_all();
    Ok((Cancel { id, flags }, rx))
}

// ---- GET /v1/status and the live view's snapshot ----

fn round(x: f64, digits: i32) -> f64 {
    let f = 10f64.powi(digits);
    (x * f).round() / f
}

/// GET /v1/status's JSON.
pub fn status() -> String {
    let s = shared();
    let waiting = lock(&s.queue).jobs.len();
    let c = engine::stats();
    let st = lock(&s.status);
    let now = Instant::now();
    let sessions: Vec<Value> = st
        .states
        .iter()
        .map(|&(number, tokens, used)| json!({"session": number, "tokens": tokens, "idle_seconds": round((now - used).as_secs_f64(), 0)}))
        .collect();
    let p = &st.prefill;
    json!({
        "phase": st.phase, "phase_seconds": round((now - st.since).as_secs_f64(), 1), "decoding": st.active,
        "request": st.request, "prompt_tokens": st.prompt, "reused_tokens": st.reused, "generated_tokens": st.generated,
        "waiting": waiting, "sessions": sessions,
        "kv_cache_gib": round(st.kv_bytes as f64 / GIB, 3), "kv_cache_limit_gib": round(s.kv_limit, 3),
        "totals": {"requests": st.requests, "prompt_tokens": st.prompt_tokens, "generated_tokens": st.generated_tokens},
        "ced_depth": s.ced, "lookup_depth": s.lookup, "expert_batch": s.expert_batch, "prefill_prune": s.prefill_prune,
        "runahead": s.runahead,
        "model_decode": {"computed_rows": c.decode_rows, "seconds": c.decode_seconds,
                         "main_queue_gpu_seconds": c.decode_gpu_seconds, "requested_expert_gb": c.decode.bytes as f64 / 1e9,
                         "read_wait_seconds": c.decode.read_s},
        "last_prefill": {"chunks": p.chunks, "tokens": p.tokens, "decoder_tokens": p.decoder_tokens,
                         "ced_applied": p.ced_applied != 0, "prune_applied": p.prune_applied != 0,
                         "prefix_tokens": p.prefix_tokens, "prefix_encoder_only": false},
        "expert_cache": {"slots": c.slots, "gib": round(c.cache_bytes as f64 / GIB, 1),
                         "hits": c.cache.hits + c.cache.prefetch_hits, "misses": c.cache.misses,
                         "read_gb": round(c.cache.bytes as f64 / 1e9, 1), "slots_released": c.cache.given_back},
        "memory": {"available_gib": round(c.available as f64 / GIB, 1), "footprint_gib": round(c.footprint as f64 / GIB, 1)},
    })
    .to_string()
}

/// The live view's snapshot (JSON): phase, requests, statistics, memory and
/// the states' prefix tree.
pub fn view() -> String {
    let s = shared();
    let waiting = lock(&s.queue).jobs.len();
    let c = engine::stats();
    let mut st = lock(&s.status);
    let now = Instant::now();
    // the routed rows copied since the snapshot before, each in one snapshot
    let routes = std::mem::take(&mut st.routes);
    let requests: Vec<Value> = st
        .live
        .iter()
        .map(|r| {
            let mut v = json!({"number": r.number, "session": r.session, "prompt": r.prompt, "reused": r.reused, "generated": r.generated});
            if let Some(x) = routes.iter().find(|x| x.request == r.number) {
                v["route"] = json!({"experts": x.experts, "sources": x.sources});
            }
            v
        })
        .collect();
    let tree: Vec<Value> = st
        .tree
        .iter()
        .map(|(x, used)| {
            json!({"lines": x.lines, "start": x.start, "end": x.end, "sessions": x.states, "hits": x.hits,
                   "session": x.number, "via": x.via, "busy": x.busy, "saved": x.saved,
                   "idle_seconds": if x.number != 0 { round((now - *used).as_secs_f64(), 1) } else { 0.0 },
                   "gib": round(x.bytes as f64 / GIB, 3)})
        })
        .collect();
    let (d, p) = (&c.decode, &c.prefill);
    let pinned = c.cache_bytes.saturating_sub(c.slots.max(0) as u64 * c.slot_bytes);
    let v = json!({
        "uptime_seconds": round(s.started.elapsed().as_secs_f64(), 0), "phase": st.phase,
        "phase_seconds": round((now - st.since).as_secs_f64(), 1), "waiting": waiting, "requests": requests,
        "layers": c.layers, "experts_per_row": c.experts_per_row,
        "prefill": {"layer": c.prefill_layer, "rows": c.prefill_rows_now},
        "totals": {"requests": st.requests, "prompt_tokens": st.prompt_tokens, "reused_tokens": st.reused_tokens,
                   "generated_tokens": st.generated_tokens},
        "kinds": {"new": st.kinds[0], "continued": st.kinds[1], "saved": st.kinds[2], "forked": st.kinds[3]},
        "sessions": st.states.len(), "kv_gib": round(st.kv_bytes as f64 / GIB, 3), "kv_limit_gib": round(s.kv_limit, 3),
        "tree": tree,
        "decode": {"rows": c.decode_rows, "tokens": st.decoded, "seconds": round(c.decode_seconds, 3),
                   "gpu_seconds": round(c.decode_gpu_seconds, 3),
                   "in_memory": d.row_own, "read_ahead": d.row_ahead, "late": d.row_late, "substituted": d.row_sub,
                   "from_ssd": d.row_ssd, "bytes": d.bytes, "read_wait_seconds": round(d.read_s, 3),
                   "prefetches": d.prefetches, "cancelled": d.cancelled, "remaps": d.remaps},
        "prompts": {"rows": c.prefill_rows, "hits": p.hits, "prefetch_hits": p.prefetch_hits, "misses": p.misses,
                    "substituted": p.substituted, "bytes": p.bytes},
        "prefilled": {"rows": st.prefilled.rows, "bytes": st.prefilled.bytes, "seconds": round(st.prefilled.seconds, 3)},
        "cache": {"slots": c.slots, "mapped": c.mapped, "cap": c.cap, "max_slots": c.max_slots, "slot_bytes": c.slot_bytes,
                  "experts": c.experts, "uncached": c.uncached, "uncached_bytes": c.uncached_bytes,
                  "evictions": c.cache.evictions, "given_back": c.cache.given_back,
                  // every expert byte read, counted as each read completes (the
                  // prefill's own counter takes a chunk's reads when it ends)
                  "read_bytes": c.cache.bytes},
        "memory": {"footprint": c.footprint, "dense": c.dense_bytes, "pinned": pinned, "gpu_in_use": c.gpu_in_use,
                   "gpu_limit": if c.gpu_limit == i64::MAX { Value::Null } else { json!(c.gpu_limit) },
                   "total": c.total, "free": c.free, "file": c.file, "file_kept": c.file_kept, "wired": c.wired,
                   "compressor": c.compressor, "compressed": c.compressed, "swap_used": c.swap_used,
                   "swapouts": c.swapouts, "pressure": c.pressure},
    });
    // the decode steps take the status's lock; the text is written without it
    drop(st);
    v.to_string()
}

// ---- the worker ----

/// A request from its prefill to its end.
struct Gen {
    job: Job,
    sampler: Sampler,
    state: kv::Id,
    number: u64,
    n: usize,
    from: usize,
    pos: usize,
    generated: usize,
    reasoning: usize,
    thinking: bool,
    ended: bool,
    cancelled: bool,
    t: i32,
    out: Vec<u32>,
    draft_passes: usize,
    draft_accepted: usize,
    lookup: Option<Lookup>,
    lookup_indexed: usize,
    lookup_passes: usize,
    lookup_offered: usize,
    lookup_accepted: usize,
    max_new: usize,
    t1: Instant,
    tick: Instant,
    c1: engine::CacheCounts,
}

struct Worker {
    cfg: engine::Config,
    s: Arc<Shared>,
    ctx: usize,
    batch: usize,
    cache: Cache,
    limits: Limits,
    states: HashMap<kv::Id, *mut RawState>,
    snaps: HashMap<(kv::Id, usize), *mut RawSnapshot>,
    used: HashMap<kv::Id, Instant>,
    /// The state in use, if known (after myds4_rows it is not).
    active: Option<kv::Id>,
    gens: Vec<Gen>,
    eos: i32,
    think_start: i32,
    think_end: i32,
    shown: usize,
    published: Instant,
    /// The experts a decoded row holds: the live view's routed rows.
    per_row: usize,
}

impl Worker {
    fn new(cfg: engine::Config, s: Arc<Shared>) -> Worker {
        let special = |t: &str| engine::special(t).unwrap_or_else(|| panic!("no special token {t}"));
        let limit = (cfg.kv_gb * GIB) as u64;
        engine::log(&format!(
            "kv cache: up to {:.1} GiB (--kv-cache); a state takes {:.0} MiB at first, {:.2} GiB at {} tokens; a snapshot {:.1} MiB",
            cfg.kv_gb,
            engine::state_bytes(std::ptr::null(), 0) as f64 / 1048576.0,
            engine::state_bytes(std::ptr::null(), cfg.ctx as usize) as f64 / GIB,
            cfg.ctx,
            engine::snapshot_bytes(true) as f64 / 1048576.0
        ));
        Worker {
            ctx: cfg.ctx as usize,
            batch: (cfg.batch.max(1) as usize).min(engine::max_rows()),
            cache: Cache::new(limit, STATES),
            // exact with --ced off, bounded with --ced
            limits: Limits { rewind: engine::rewind(), replay: true, replay_cost: engine::replay_rows },
            states: HashMap::new(),
            snaps: HashMap::new(),
            used: HashMap::new(),
            active: None,
            gens: Vec::new(),
            eos: special("<｜end▁of▁sentence｜>"),
            think_start: special("<think>"),
            think_end: special("</think>"),
            shown: 0,
            published: Instant::now(),
            per_row: engine::stats().experts_per_row.max(0) as usize,
            cfg,
            s,
        }
    }

    fn use_state(&mut self, id: kv::Id) {
        engine::state_use(self.states[&id]);
        self.active = Some(id);
    }

    /// Frees a snapshot's engine copy.
    fn free_snap(&mut self, id: kv::Id, pos: usize) {
        if let Some(x) = self.snaps.remove(&(id, pos)) {
            engine::snapshot_free(x);
        }
    }

    /// The tokens of state `session`'s positions start..end, for the live
    /// view: at most TEXT_TOKENS from start, none if the state is gone.
    fn node_text(&self, session: u64, start: usize, end: usize) -> NodeText {
        let tokens = self.cache.entries.values().find(|e| e.number == session).map_or(&[][..], |e| &e.tokens[..]);
        let to = end.min(tokens.len());
        let from = start.min(to);
        NodeText { session, start, end, ids: tokens[from..to.min(from + TEXT_TOKENS)].to_vec() }
    }

    /// Before a pass that makes the busy state `id` hold `positions`: the
    /// cache charges its growth before the engine grows it (kv::Cache::grow),
    /// and what the cache gives up for it is freed.
    fn make_room(&mut self, id: kv::Id, positions: usize) {
        let before = self.cache.entries[&id].bytes;
        let (snaps, states) = self.cache.grow(id, engine::state_bytes(self.states[&id], positions));
        self.free(id, snaps, states);
        let after = self.cache.entries[&id].bytes;
        if after > before && self.cache.over() > 0 {
            engine::log(&format!(
                "kv cache: a decoding state grows to {:.2} GiB at {positions} positions; with only decoding states left, the cache takes {:.2} GiB of the {:.1} GiB --kv-cache allows",
                after as f64 / GIB,
                self.cache.bytes() as f64 / GIB,
                self.cfg.kv_gb
            ));
        }
    }

    /// Frees what an eviction or a truncation dropped, with `working` in use.
    fn free(&mut self, working: kv::Id, snaps: Vec<(kv::Id, Snap)>, states: Vec<(kv::Id, Vec<Snap>)>) {
        for (id, s) in snaps {
            self.free_snap(id, s.pos);
        }
        if !states.is_empty() && self.active != Some(working) {
            self.use_state(working);
        }
        for (id, held) in states {
            for s in held {
                self.free_snap(id, s.pos);
            }
            if let Some(st) = self.states.remove(&id) {
                engine::state_free(st);
            }
            self.used.remove(&id);
        }
    }

    /// The cache's states and tree, for the status and the view.
    fn publish(&mut self) {
        let now = Instant::now();
        let states = self
            .cache
            .entries
            .iter()
            .map(|(id, e)| (e.number, e.tokens.len(), *self.used.get(id).unwrap_or(&now)))
            .collect();
        let by_number: HashMap<u64, Instant> =
            self.cache.entries.iter().map(|(id, e)| (e.number, *self.used.get(id).unwrap_or(&now))).collect();
        let tree = self
            .cache
            .tree(24, 40)
            .into_iter()
            .map(|x| {
                let t = *by_number.get(&x.number).unwrap_or(&now);
                (x, t)
            })
            .collect();
        let mut st = lock(&self.s.status);
        st.states = states;
        st.tree = tree;
        st.kv_bytes = self.cache.bytes();
        self.published = now;
    }

    fn set_status(&mut self, phase: &'static str, prefilling: Option<(u64, u64, usize, usize)>) {
        let mut st = lock(&self.s.status);
        if st.phase != phase {
            st.since = Instant::now();
        }
        st.phase = phase;
        st.active = self.gens.len();
        st.decoded = st.generated_tokens + self.gens.iter().map(|g| g.generated as u64).sum::<u64>();
        let first = self.gens.first();
        (st.request, st.prompt, st.reused, st.generated) = match (prefilling, first) {
            (Some((id, _, n, from)), _) => (id, n, from, 0),
            (None, Some(g)) => (g.job.id, g.n, g.from, g.generated),
            _ => (0, 0, 0, 0),
        };
        st.live = match prefilling {
            Some((id, number, n, from)) => vec![LiveReq { number: id, session: number, prompt: n, reused: from, generated: 0 }],
            None if phase == "decode" => self
                .gens
                .iter()
                .map(|g| LiveReq { number: g.job.id, session: g.number, prompt: g.n, reused: g.from, generated: g.generated })
                .collect(),
            None => Vec::new(),
        };
    }

    fn refuse(&mut self, job: Job, message: &str) {
        engine::log(&format!("req {}: refused: {message}", job.id));
        job.last(Event::Refused(message.into()));
    }

    fn end_waiting(&self, job: Job, finish: Finish) {
        engine::log(&format!(
            "req {}: ended before prefill ({}), waited {:.3} s",
            job.id,
            if finish == Finish::Cancelled { "cancelled" } else { "shutdown" },
            job.queued.elapsed().as_secs_f64()
        ));
        job.last(Event::Done { finish });
    }

    /// Prefills a request: its generation, or None if it was refused.
    fn begin(&mut self, job: Job) -> Option<Gen> {
        let sampler = Sampler::new(job.temperature, job.top_p, job.seed).expect("checked when submitted");
        let mut prompt = job.prompt.clone();
        let mut images = None;
        if !job.images.is_empty() {
            match Images::open(&job.images) {
                Err(m) => {
                    self.refuse(job, &m);
                    return None;
                }
                Ok(mut x) => match x.expand(&prompt, self.ctx + 1) {
                    Some(p) => {
                        prompt = p;
                        images = Some(x);
                    }
                    None => {
                        let m = format!("the prompt's image places and its {} images differ", job.images.len());
                        self.refuse(job, &m);
                        return None;
                    }
                },
            }
        }
        let n = prompt.len();
        if n >= self.ctx {
            let m = format!("the prompt has {n} tokens; the context holds {}", self.ctx);
            self.refuse(job, &m);
            return None;
        }
        let max_new = (job.max_tokens.max(1) as usize).min(self.ctx - n);
        let plan = self.cache.plan(&prompt, self.limits);
        let capture = plan.capture(n);
        // The state at the prompt's end and its first token, and the
        // snapshots the plan takes.
        let state_need = match plan {
            Plan::InPlace { state, .. } => {
                engine::state_bytes(self.states[&state], n + 1).saturating_sub(self.cache.entries[&state].bytes)
            }
            _ => engine::state_bytes(std::ptr::null(), n + 1),
        };
        let need = state_need + plan.snapshot_need(n, engine::snapshot_bytes(false), engine::snapshot_bytes(true));
        let new_state = !matches!(plan, Plan::InPlace { .. });
        let keep: Vec<kv::Id> = plan.source().into_iter().collect();
        let Some(ev) = self.cache.fit(need, new_state, &keep) else {
            let m = format!(
                "the conversation's attention state needs {:.1} GiB; --kv-cache allows {:.1} GiB",
                engine::state_bytes(std::ptr::null(), n + 1) as f64 / GIB,
                self.cfg.kv_gb
            );
            self.refuse(job, &m);
            return None;
        };
        let t0 = Instant::now();
        let wait = (t0 - job.queued).as_secs_f64();
        // The state the plan works from is in use while others are freed.
        let fresh = new_state.then(engine::state_new);
        let working = match plan.source() {
            Some(id) => id,
            None => {
                let id = self.cache.add(0);
                self.states.insert(id, fresh.expect("a new state"));
                id
            }
        };
        self.use_state(working);
        let (snaps, states) = self.cache.evict(&ev);
        self.free(working, snaps, states);
        let from = plan.reused();
        let id = match plan {
            Plan::Cold => {
                engine::reset();
                working
            }
            Plan::InPlace { state, from } => {
                self.cache.touch(state, from);
                state
            }
            Plan::Snapshot { source, at, .. } | Plan::Replay { source, at } => {
                let id = self.cache.add(0);
                let st = fresh.expect("a new state");
                self.states.insert(id, st);
                engine::state_copy(st, self.states[&source], at);
                self.cache.touch(source, at);
                self.used.insert(source, Instant::now());
                self.use_state(id);
                if let Plan::Snapshot { .. } = plan {
                    engine::snapshot_restore(st, self.snaps[&(source, at)]);
                }
                id
            }
        };
        let bytes = engine::state_bytes(self.states[&id], n + 1);
        let dropped = self.cache.start(id, &prompt, from, bytes);
        for s in dropped {
            self.free_snap(id, s.pos);
        }
        self.used.insert(id, Instant::now());
        let number = self.cache.entries[&id].number;
        let kind = plan.kind(n);
        let thinking = prompt[n - 1] == self.think_start;
        engine::log(&format!(
            "req {}: {n} prompt tokens, session {number} ({kind}), {from} reused; thinking {}; at most {max_new} new tokens, {}; waited {wait:.1} s",
            job.id,
            if thinking { "on" } else { "off" },
            if job.temperature > 0.0 { "sampled" } else { "greedy" }
        ));
        self.set_status("prefill", Some((job.id, number, n, from)));
        self.publish();
        let queued = lock(&self.s.queue).jobs.len();
        engine::log(&format!(
            "req {}: prefill start, waiting={queued}, new_tokens={}, ced_depth={}",
            job.id,
            n - from,
            self.cfg.ced
        ));
        if let Some(x) = images.as_mut() {
            // The images whose spans the prefill reaches: a replay computes again
            // from its first position, before the branch point.
            let first = if matches!(plan, Plan::Replay { .. }) { engine::replay_start(from) } else { from };
            let encoded = x.encode(first);
            if encoded > 0 {
                engine::log(&format!(
                    "req {}: {encoded} of {} images encoded in {:.2} s",
                    job.id,
                    job.images.len(),
                    t0.elapsed().as_secs_f64()
                ));
            }
        }
        let c0 = engine::stats().cache;
        let (lg, info): (Logits, PrefillInfo) = if from == n {
            // The same prompt again: its prompt-end snapshot holds its logits.
            let source = plan.source().expect("a saved prompt");
            let x = self.snaps[&(source, n)];
            if source == id {
                engine::snapshot_restore(self.states[&id], x);
            }
            engine::seek(&prompt, n);
            (engine::snapshot_logits(x), PrefillInfo::default())
        } else {
            let replay = matches!(plan, Plan::Replay { .. });
            let Some((lg, captured, info)) = engine::prefill(&prompt, from, replay, capture) else {
                panic!("the engine refused a prefill of positions {from} to {n}");
            };
            if let Some(at) = capture
                && !captured.is_null()
            {
                self.snaps.insert((id, at), captured);
                self.cache.add_snap(id, Snap { pos: at, kind: Kind::Branch, bytes: engine::snapshot_bytes(false) });
            }
            let x = engine::snapshot_take(self.states[&id], n, true);
            self.snaps.insert((id, n), x);
            self.cache.add_snap(id, Snap { pos: n, kind: Kind::PromptEnd, bytes: engine::snapshot_bytes(true) });
            (lg, info)
        };
        drop(images);
        let t1 = Instant::now();
        let dt = (t1 - t0).as_secs_f64();
        let c1 = engine::stats();
        let read = (c1.cache.bytes - c0.bytes) as f64 / 1e9;
        let replayed = if matches!(plan, Plan::Replay { .. }) {
            format!(
                "; {} replay at {from}: up to {} earlier rows computed again",
                if self.cfg.ced > 0 { "a bounded" } else { "an exact" },
                (info.tokens as usize).saturating_sub(n - from)
            )
        } else {
            String::new()
        };
        engine::log(&format!(
            "req {}: prefill of {} tokens in {dt:.2} s ({:.0} tokens/s), {read:.2} GB of experts read{replayed}",
            job.id,
            n - from,
            (n - from) as f64 / dt.max(1e-9)
        ));
        engine::log(&format!(
            "req {}: prefill plan: {} chunks, {} encoder tokens, {} decoder tokens, ced_applied={}, prune_applied={}, prefix={} (full)",
            job.id, info.chunks, info.tokens, info.decoder_tokens, info.ced_applied, info.prune_applied, info.prefix_tokens
        ));
        let _ = job.tx.send(Event::Started { prompt: n, reused: from });
        {
            let mut st = lock(&self.s.status);
            st.prefill = info;
            if from < n {
                st.prefilled.seconds += dt;
            }
            (st.prefilled.rows, st.prefilled.bytes) = (c1.prefill_rows, c1.prefill.bytes);
            st.kinds[match kind {
                "new" => 0,
                "kept" => 1,
                "saved" => 2,
                _ => 3,
            }] += 1;
        }
        let mut sampler = sampler;
        let t = sampler.sample(lg);
        let lookup = if self.cfg.lookup > 0 {
            let mut l = Lookup::new(self.ctx).unwrap_or_else(|| panic!("cannot initialize context lookup"));
            if !l.append(&prompt) {
                panic!("cannot initialize context lookup");
            }
            Some(l)
        } else {
            None
        };
        Some(Gen {
            job,
            sampler,
            state: id,
            number,
            n,
            from,
            pos: n,
            generated: 0,
            reasoning: 0,
            thinking,
            ended: false,
            cancelled: false,
            t,
            out: Vec::new(),
            draft_passes: 0,
            draft_accepted: 0,
            lookup,
            lookup_indexed: n,
            lookup_passes: 0,
            lookup_offered: 0,
            lookup_accepted: 0,
            max_new,
            t1,
            tick: t1,
            c1: engine::stats().cache,
        })
    }

    /// The token t ran at g's position: into its state's tokens and the step's output.
    fn emit(cache: &mut Cache, g: &mut Gen, t: i32, eos: i32, think_end: i32) {
        if let Some(l) = g.lookup.as_mut()
            && g.lookup_indexed == g.pos
        {
            if !l.append(&[t]) {
                panic!("context lookup exceeds the configured context");
            }
            g.lookup_indexed += 1;
        }
        if g.thinking {
            g.reasoning += 1;
            g.thinking = t != think_end;
        }
        cache.push(g.state, t);
        g.pos += 1;
        g.generated += 1;
        g.ended = t == eos; // a kept draft ends the text
        if !g.ended {
            g.out.push(t as u32);
        }
    }

    /// Whether the generation ends before its next token: at the end token,
    /// its budget, the context's end, or when it was cancelled.
    fn stop(g: &mut Gen, eos: i32, ctx: usize) -> bool {
        if g.job.flags.cancelled.load(Ordering::SeqCst) {
            g.cancelled = true;
        }
        if g.ended || g.cancelled {
            return true;
        }
        if g.t == eos {
            g.generated += 1;
            g.ended = true;
        }
        g.ended || g.generated >= g.max_new || g.pos >= ctx
    }

    fn finish(&mut self, mut g: Gen, shutdown: bool) {
        if !g.out.is_empty() {
            let _ = g.job.tx.send(Event::Tokens(std::mem::take(&mut g.out)));
        }
        let why = if g.cancelled {
            Finish::Cancelled
        } else if shutdown {
            Finish::Shutdown
        } else if g.ended {
            Finish::Stop
        } else {
            Finish::Length
        };
        let name = match why {
            Finish::Stop => "stop",
            Finish::Length => "length",
            Finish::Cancelled => "client gone",
            Finish::Shutdown => "stopped by the server's stop",
        };
        g.job.last(Event::Done { finish: why });
        // The generation's end, for forks there, when the cache keeps it.
        let end = self.cache.snap(g.state, g.pos).is_none().then(|| Snap {
            pos: g.pos,
            kind: Kind::GenerationEnd,
            bytes: engine::snapshot_bytes(false),
        });
        let (take, (snaps, states)) = self.cache.end(g.state, engine::state_bytes(self.states[&g.state], g.pos), end);
        self.free(g.state, snaps, states);
        if take {
            let x = engine::snapshot_take(self.states[&g.state], g.pos, false);
            self.snaps.insert((g.state, g.pos), x);
        }
        self.used.insert(g.state, Instant::now());
        let s = engine::stats();
        let (c2, dt) = (s.cache, g.t1.elapsed().as_secs_f64());
        let hits = c2.hits - g.c1.hits + c2.prefetch_hits - g.c1.prefetch_hits;
        let misses = c2.misses - g.c1.misses;
        engine::log(&format!(
            "req {}: {name}: {} tokens ({} reasoning) in {dt:.1} s, {:.1} tokens/s; experts {:.1}% hit, {:.2} GB read; kv {} sessions, {:.2} GiB; available {:.1} GiB, footprint {:.1} GiB",
            g.job.id,
            g.generated,
            g.reasoning,
            g.generated as f64 / dt.max(1e-9),
            100.0 * hits as f64 / ((hits + misses).max(1)) as f64,
            (c2.bytes - g.c1.bytes) as f64 / 1e9,
            self.cache.entries.len(),
            self.cache.bytes() as f64 / GIB,
            s.available as f64 / GIB,
            s.footprint as f64 / GIB
        ));
        engine::log(&format!(
            "req {}: sampling temperature {}, top_p {}; {} draft passes, {} drafts accepted",
            g.job.id, g.job.temperature, g.job.top_p, g.draft_passes, g.draft_accepted
        ));
        if g.lookup.is_some() {
            engine::log(&format!(
                "req {}: context lookup {} passes, {} offered, {} accepted",
                g.job.id, g.lookup_passes, g.lookup_offered, g.lookup_accepted
            ));
        }
        let mut st = lock(&self.s.status);
        st.requests += 1;
        st.prompt_tokens += g.n as u64;
        st.generated_tokens += g.generated as u64;
        st.reused_tokens += g.from as u64;
        drop(st);
        self.publish();
    }

    /// One token for each generation: alone with its drafts, or together,
    /// each row as it would run alone.
    fn advance(&mut self) {
        let (eos, think_end, ctx) = (self.eos, self.think_end, self.ctx);
        if self.gens.len() == 1 {
            let state = self.gens[0].state;
            self.use_state(state);
            let lookup = self.cfg.lookup as usize;
            let (mut rows, proposal, k, pos) = {
                let g = &mut self.gens[0];
                let mut rows = vec![g.t];
                let room = (g.max_new as isize - g.generated as isize - 1).min(ctx as isize - g.pos as isize - 1);
                let mut proposal = 0;
                if let Some(l) = g.lookup.as_mut() {
                    if g.lookup_indexed != g.pos || !l.append(&[g.t]) {
                        panic!("context lookup history is not aligned with the generation");
                    }
                    g.lookup_indexed += 1;
                    if room > 0 {
                        let mut buf = [0i32; 16];
                        proposal = l.propose(lookup.min(room as usize), &mut buf);
                        rows.extend_from_slice(&buf[..proposal]);
                    }
                }
                let k = if proposal > 0 {
                    g.lookup_passes += 1;
                    g.lookup_offered += proposal;
                    proposal
                } else {
                    engine::draft_block().min(room.max(0) as usize)
                };
                (rows, proposal, k, g.pos)
            };
            // The rows the pass writes: the token and its drafts or proposals.
            self.make_room(state, pos + 1 + k);
            let g = &mut self.gens[0];
            let (next, a) = if k > 0 {
                if proposal == 0 {
                    let mut d = vec![0i32; k];
                    engine::draft(g.t, g.pos, &mut d);
                    rows.extend_from_slice(&d);
                }
                let lg = engine::verify(&rows, g.pos).expect("a verify pass in the limits");
                let (next, a) = g.sampler.verify(lg, &rows[1..], eos);
                g.draft_passes += 1;
                g.draft_accepted += a;
                engine::accept(&rows[..a + 1], g.pos);
                (next, a)
            } else {
                let lg = engine::step(g.t, g.pos).expect("a step in the context");
                (g.sampler.sample(lg), 0)
            };
            if proposal > 0
                && let Some(l) = g.lookup.as_mut()
            {
                l.feedback(k, a);
                g.lookup_accepted += a;
            }
            for &t in &rows[..=a] {
                if g.ended {
                    break;
                }
                Self::emit(&mut self.cache, g, t, eos, think_end);
            }
            g.t = next;
            return;
        }
        let rooms: Vec<(kv::Id, usize)> = self.gens.iter().map(|g| (g.state, g.pos + 1)).collect();
        for (id, positions) in rooms {
            self.make_room(id, positions);
        }
        let mut st: Vec<*mut RawState> = self.gens.iter().map(|g| self.states[&g.state]).collect();
        let tokens: Vec<i32> = self.gens.iter().map(|g| g.t).collect();
        let pos: Vec<i32> = self.gens.iter().map(|g| g.pos as i32).collect();
        let lg = engine::rows(&mut st, &tokens, &pos).expect("rows in the limits");
        self.active = None;
        for (r, g) in self.gens.iter_mut().enumerate() {
            Self::emit(&mut self.cache, g, tokens[r], eos, think_end);
            // Safety: myds4_rows returns m rows of the vocabulary.
            g.t = g.sampler.sample(unsafe { lg.add(r * engine::vocab()) });
        }
    }

    /// The decoding requests' routed rows of the latest step, for the live
    /// view: row r is gens[r]'s (myds4_rows runs them in that order; alone, a
    /// request's token is row 0). A few kilobytes, twice a second.
    fn copy_routes(&self) {
        let routes: Vec<Route> = self
            .gens
            .iter()
            .enumerate()
            .filter_map(|(r, g)| engine::routes(r, self.per_row).map(|(experts, sources)| Route { request: g.job.id, experts, sources }))
            .collect();
        lock(&self.s.status).routes = routes;
    }

    /// Admits waiting requests while the batch has room (prefilling each while
    /// the others wait), ends the generations that are done, and advances the
    /// others by a token. While no request runs, the expert cache gives slots
    /// back every second if other programs need the memory. On the stop, every
    /// request ends with Finish::Shutdown.
    fn run(&mut self) {
        let s = self.s.clone();
        while !engine::stopping() {
            let mut q = lock(&s.queue);
            while q.jobs.is_empty() && q.text.is_none() && self.gens.is_empty() && !engine::stopping() {
                let (guard, timeout) = s.wake.wait_timeout(q, Duration::from_secs(1)).unwrap_or_else(|e| e.into_inner());
                q = guard;
                if timeout.timed_out() && q.jobs.is_empty() {
                    drop(q);
                    engine::idle();
                    q = lock(&s.queue);
                }
            }
            let question = q.text.take();
            let mut admit = Vec::new();
            while self.gens.len() + admit.len() < self.batch && !engine::stopping() {
                let Some(job) = q.jobs.pop_front() else { break };
                admit.push(job);
            }
            if !admit.is_empty() {
                engine::log(&format!(
                    "queue: admitted={} for prefill, waiting={}, decoding={}",
                    admit.len(),
                    q.jobs.len(),
                    self.gens.len()
                ));
            }
            drop(q);
            if let Some((session, start, end)) = question {
                let answer = self.node_text(session, start, end);
                lock(&s.status).text = Some(answer);
            }
            for job in admit {
                if engine::stopping() {
                    self.end_waiting(job, Finish::Shutdown);
                } else if job.flags.cancelled.load(Ordering::SeqCst) {
                    engine::log(&format!("req {}: client gone while waiting", job.id));
                    self.end_waiting(job, Finish::Cancelled);
                } else if let Some(g) = self.begin(job) {
                    self.gens.push(g);
                }
            }
            if engine::stopping() {
                break;
            }
            let (eos, ctx) = (self.eos, self.ctx);
            let mut i = 0;
            while i < self.gens.len() {
                if Self::stop(&mut self.gens[i], eos, ctx) {
                    let g = self.gens.swap_remove(i);
                    self.finish(g, false);
                } else {
                    i += 1;
                }
            }
            if self.gens.len() > 1 && self.gens.len() != self.shown {
                engine::log(&format!("batch: {} requests decoding together", self.gens.len()));
            }
            self.shown = self.gens.len();
            if self.gens.is_empty() {
                self.set_status("idle", None);
                continue;
            }
            if engine::stopping() {
                break;
            }
            self.advance();
            if ROUTES.swap(false, Ordering::Relaxed) {
                self.copy_routes();
            }
            for g in &mut self.gens {
                if !g.out.is_empty() {
                    let _ = g.job.tx.send(Event::Tokens(std::mem::take(&mut g.out)));
                }
                if g.tick.elapsed() >= Duration::from_secs(10) {
                    g.tick = Instant::now();
                    engine::log(&format!(
                        "req {}: {} tokens, {:.1} tokens/s",
                        g.job.id,
                        g.generated,
                        g.generated as f64 / (g.tick - g.t1).as_secs_f64()
                    ));
                }
            }
            self.set_status("decode", None);
            if self.published.elapsed() >= Duration::from_secs(1) {
                self.publish();
            }
        }
        for g in std::mem::take(&mut self.gens) {
            self.finish(g, true);
        }
        let waiting: Vec<Job> = {
            let mut q = lock(&s.queue);
            q.closed = true;
            q.jobs.drain(..).collect()
        };
        for job in waiting {
            self.end_waiting(job, Finish::Shutdown);
        }
        self.set_status("idle", None);
    }
}
