//! The server's worker (docs/server.md): one engine thread that loads every
//! model and makes every engine call, so that all models' passes go through
//! one GPU command queue. HTTP handlers submit requests, each to its model,
//! to a bounded queue (at most [`WAITING`] wait per model) and get their
//! events back on a channel.
//!
//! Each turn of the loop gives every model one step, in turn: it admits the
//! model's waiting requests while fewer than its batch run, runs one prefill
//! call of its oldest prefilling request (the whole prompt for a model that
//! plans its prefill over the whole prompt; otherwise up to its prefill rows,
//! [`SHARED_PREFILL`] while others decode, so that a long prompt does not
//! stall the replies in progress), then advances every decoding request by a
//! token: alone with the model's drafts (draft, lookup), or together with the
//! others in one decoding pass. Prompts resume from the model's KV prefix
//! cache (kv.rs). The worker also builds each model's status and the live
//! view's snapshot.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use serde_json::{Value, json};
use tokio::sync::mpsc;

use crate::engine::{self, CacheCounts, Images, Logits, Lookup, Model, PrefillInfo, RawSnapshot, RawState, Sampler};
use crate::kv::{self, Cache, Kind, Limits, Plan, Snap};

/// Requests waiting for a model at most; more are answered 429.
pub const WAITING: usize = 16;
/// Attention states the shared cache keeps at most, across all models.
pub const STATES: usize = 256;
/// A prefill call's rows while other requests of the model decode, for a
/// model whose prefill runs in calls of its prefill rows.
const SHARED_PREFILL: usize = 128;
/// A prefill in calls snapshots its state at every multiple of this many
/// positions (within the KV cache's budget), when the model cannot replay: a
/// later prompt that shares a long prefix with it but parts before its end
/// (another conversation with the same system prompt and tools) resumes from
/// the last one before the point where it parts.
const SNAP_GRID: usize = 2048;
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

/// A request's sampling: greedy at temperature 0, else from the top_k
/// highest-ranked tokens (0: all) cut to the top_p nucleus.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sampling {
    pub temperature: f64,
    pub top_p: f64,
    pub top_k: u32,
    pub seed: Option<u64>,
}

/// A request: its model (an index of [`start`]'s models), its prompt's token
/// ids (each image's place the model's image token), its images' file bytes,
/// sampling and its token limit.
pub struct Request {
    pub model: usize,
    pub prompt: Vec<i32>,
    pub images: Vec<Vec<u8>>,
    pub sampling: Sampling,
    pub max_tokens: i64,
    /// Logged as "req ID NOTE" when the request is queued.
    pub note: String,
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
    request: Request,
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

/// A served model's settings: the name requests give, the engine's options,
/// and the worker's.
#[derive(Clone, Debug, PartialEq)]
pub struct Served {
    pub id: String,
    pub engine: engine::Options,
    /// Requests running together at most (at most the model's rows per pass).
    pub batch: usize,
}

#[derive(Default)]
struct Queue {
    /// Each model's waiting requests.
    jobs: Vec<VecDeque<Job>>,
    closed: bool,
    numbers: u64,
    /// The live view's question for a node's text: a model, a state's number
    /// and positions start..end ([`ask_text`]).
    text: Option<(usize, u64, usize, usize)>,
}

/// The tokens of a state's positions start..end, for the live view's text of
/// a node: at most [`TEXT_TOKENS`] of them from start, none once the state
/// is gone. The view turns them into text ([`text`], from any thread).
pub struct NodeText {
    pub model: usize,
    pub session: u64,
    pub start: usize,
    pub end: usize,
    pub ids: Vec<i32>,
}

/// The tokens a node's text holds at most.
const TEXT_TOKENS: usize = 65_536;

/// A decoding request's routed row of one decoding pass: the experts its
/// token chose in each layer and where each one was (myds4_routes).
struct Route {
    request: u64,
    experts: Vec<u16>,
    sources: Vec<u8>,
}

/// Whether the live view asked for the decoding requests' routed rows: it
/// asks with each snapshot, the worker copies them after its next decoding
/// pass, and the next snapshot takes them, so the view gets each row once.
static ROUTES: AtomicBool = AtomicBool::new(false);

/// The live view asks for the routed rows of the next decoding pass.
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

/// What a model's worker is doing and has done, for its status and the
/// live view.
#[derive(Default)]
struct Status {
    phase: &'static str,
    since: Option<Instant>,
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
    kv_allocated: u64,
}

/// A model as the threads share it: its settings, its engine handle (its
/// tokenizer and statistics take any thread) and its status.
pub struct ModelShared {
    pub served: Served,
    pub model: Arc<Model>,
    status: Mutex<Status>,
}

struct Shared {
    queue: Mutex<Queue>,
    wake: Condvar,
    models: Vec<ModelShared>,
    budgets: crate::config::Budgets,
    kv_used: AtomicU64,
    started: Instant,
}

static SHARED: Mutex<Option<Arc<Shared>>> = Mutex::new(None);
static THREAD: Mutex<Option<JoinHandle<()>>> = Mutex::new(None);

fn shared() -> Arc<Shared> {
    SHARED.lock().unwrap().clone().expect("the worker started")
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// The models the worker serves, in their order: an index of this list is a
/// request's model.
pub fn models() -> Vec<(String, Arc<Model>, Served)> {
    shared().models.iter().map(|m| (m.served.id.clone(), m.model.clone(), m.served.clone())).collect()
}

/// Starts the worker's engine thread, which initializes the engine and loads
/// every model; returns once it takes requests, or false when a model's
/// settings are invalid (the engine says why on stderr).
pub fn start(served: Vec<Served>, budgets: crate::config::Budgets, on_log: Option<extern "C" fn(*const std::ffi::c_char)>, on_die: Option<extern "C" fn()>) -> bool {
    let (tx, rx) = std::sync::mpsc::channel();
    let handle = thread::Builder::new()
        .name("myds4-engine".into())
        .stack_size(16 << 20)
        .spawn(move || {
            if !engine::init(budgets.reserve_gb, on_log, on_die) {
                let _ = tx.send(false);
                return;
            }
            let mut models = Vec::new();
            for s in &served {
                let Some(m) = Model::load(&s.engine) else {
                    let _ = tx.send(false);
                    return;
                };
                models.push(ModelShared { served: s.clone(), model: Arc::new(m), status: Mutex::new(Status::default()) });
            }
            let n = models.len();
            let s = Arc::new(Shared {
                queue: Mutex::new(Queue { jobs: (0..n).map(|_| VecDeque::new()).collect(), ..Queue::default() }),
                wake: Condvar::new(),
                models,
                budgets,
                kv_used: AtomicU64::new(0),
                started: Instant::now(),
            });
            *SHARED.lock().unwrap() = Some(s.clone());
            // A failure here ends the process, as the engine's own do: the
            // requests could not end otherwise. (The panic's message is out.)
            let run = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let mut w = Worker::new(s.clone());
                let _ = tx.send(true);
                w.run();
            }));
            if run.is_err() {
                crate::view::leave();
                std::process::exit(1);
            }
            for m in &s.models {
                m.model.finish();
            }
        })
        .expect("the engine's thread");
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

/// The live view asks for the text of model `model`'s state `session`'s
/// positions start..end; the worker answers between its steps
/// ([`text_answer`]). A new question replaces one not answered yet.
pub fn ask_text(model: usize, session: u64, start: usize, end: usize) {
    if let Some(s) = SHARED.lock().unwrap().clone() {
        lock(&s.queue).text = Some((model, session, start, end));
        s.wake.notify_all();
    }
}

/// The worker's answer to the latest [`ask_text`] about model `model`, once.
pub fn text_answer(model: usize) -> Option<NodeText> {
    let s = SHARED.lock().unwrap().clone()?;
    lock(&s.models.get(model)?.status).text.take()
}

/// The text of a model's token ids, for the live view (engine::Tokenizer::text).
pub fn text(model: usize, ids: &[i32]) -> Vec<(String, bool)> {
    let s = SHARED.lock().unwrap().clone(); // none before the worker starts
    s.and_then(|s| s.models.get(model).map(|m| m.model.tokenizer().text(ids))).unwrap_or_default()
}

/// Requests the stop, waits for the worker to end every request, save the
/// models' states and return.
pub fn stop() {
    request_stop();
    if let Some(h) = THREAD.lock().unwrap().take() {
        let _ = h.join();
    }
}

/// Queues a request: the guard that cancels it when dropped (and holds its
/// id), and its events.
pub fn submit(r: Request) -> Result<(Cancel, mpsc::UnboundedReceiver<Event>), SubmitError> {
    let s = shared();
    let bad = if r.model >= s.models.len() {
        Some("no such model".to_string())
    } else if r.prompt.is_empty() {
        Some("the prompt is empty".into())
    } else if !(r.sampling.temperature.is_finite() && r.sampling.temperature >= 0.0 && (0.0..=1.0).contains(&r.sampling.top_p)) {
        Some("temperature must be a finite nonnegative number and top_p must be in [0, 1]".into())
    } else if r.max_tokens < 1 {
        Some("max_tokens must be at least 1".into())
    } else {
        None
    };
    if let Some(m) = bad {
        return Err(SubmitError::Invalid(m));
    }
    let (tx, rx) = mpsc::unbounded_channel();
    let flags = Arc::new(Flags::default());
    let mut q = lock(&s.queue);
    if q.closed || q.jobs[r.model].len() >= WAITING {
        let m = if q.closed { "the server is stopping" } else { "the server is busy: too many requests are waiting" };
        return Err(SubmitError::Busy(m.into()));
    }
    q.numbers += 1;
    let id = q.numbers;
    if !r.note.is_empty() {
        engine::log(&format!("req {id} {}", r.note)); // before the worker can take it
    }
    let (n, max_new, model) = (r.prompt.len(), r.max_tokens, r.model);
    q.jobs[model].push_back(Job { id, request: r, tx, flags: flags.clone(), queued: Instant::now() });
    engine::log(&format!(
        "req {id}: queued for {}, waiting={}, prompt_tokens={n}, max_new={max_new}",
        s.models[model].served.id,
        q.jobs[model].len()
    ));
    s.wake.notify_all();
    Ok((Cancel { id, flags }, rx))
}

// ---- a model's status and the live view's snapshot ----

fn round(x: f64, digits: i32) -> f64 {
    let f = 10f64.powi(digits);
    (x * f).round() / f
}

fn memory_models(s: &Shared) -> Vec<Value> {
    s.models.iter().map(|m| {
        let c = m.model.stats();
        let st = lock(&m.status);
        json!({"model": m.served.id, "weights": c.dense_bytes, "experts": c.cache_bytes,
               "kv": st.kv_allocated, "kv_committed": st.kv_bytes,
               "tracked": c.dense_bytes + c.cache_bytes + st.kv_allocated})
    }).collect()
}

/// Model `model`'s status (JSON): what it is doing, its requests, KV prefix
/// cache, counters, expert cache and the process's memory.
pub fn status(model: usize) -> Value {
    let s = shared();
    let Some(m) = s.models.get(model) else { return Value::Null };
    let waiting = lock(&s.queue).jobs[model].len();
    let c = m.model.stats();
    let allocations = memory_models(&s);
    let st = lock(&m.status);
    let now = Instant::now();
    let sessions: Vec<Value> = st
        .states
        .iter()
        .map(|&(number, tokens, used)| json!({"session": number, "tokens": tokens, "idle_seconds": round((now - used).as_secs_f64(), 0)}))
        .collect();
    let p = &st.prefill;
    let o = &m.served.engine;
    json!({
        "phase": if st.phase.is_empty() { "idle" } else { st.phase },
        "phase_seconds": round(st.since.map_or(0.0, |t| (now - t).as_secs_f64()), 1), "decoding": st.active,
        "request": st.request, "prompt_tokens": st.prompt, "reused_tokens": st.reused, "generated_tokens": st.generated,
        "waiting": waiting, "sessions": sessions,
        "kv_cache_gib": round(st.kv_bytes as f64 / GIB, 3), "kv_cache_limit_gib": round(s.budgets.kv_gb, 3),
        "kv_cache_scope": "server", "kv_cache_total_gib": round(s.kv_used.load(Ordering::Relaxed) as f64 / GIB, 3),
        "reserve_gib": s.budgets.reserve_gb, "model_memory": allocations,
        "totals": {"requests": st.requests, "prompt_tokens": st.prompt_tokens, "generated_tokens": st.generated_tokens},
        "ced_depth": o.ced, "lookup_depth": o.lookup, "prefill_prune": o.prefill_prune, "runahead": o.runahead, "pipeline": o.pipeline,
        "drafts": m.model.draft_block(),
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
}

/// The live view's snapshot of model `model` (JSON): phase, requests,
/// statistics, memory and the states' prefix tree.
pub fn view(model: usize) -> String {
    let s = shared();
    let Some(m) = s.models.get(model) else { return "{}".into() };
    let waiting = lock(&s.queue).jobs[model].len();
    let c = m.model.stats();
    let allocations = memory_models(&s);
    let mut st = lock(&m.status);
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
        "model": m.served.id, "models": s.models.len(),
        "uptime_seconds": round(s.started.elapsed().as_secs_f64(), 0),
        "phase": if st.phase.is_empty() { "idle" } else { st.phase },
        "phase_seconds": round(st.since.map_or(0.0, |t| (now - t).as_secs_f64()), 1), "waiting": waiting, "requests": requests,
        "layers": c.layers, "experts_per_row": c.experts_per_row,
        "prefill": {"layer": c.prefill_layer, "rows": c.prefill_rows_now},
        "totals": {"requests": st.requests, "prompt_tokens": st.prompt_tokens, "reused_tokens": st.reused_tokens,
                   "generated_tokens": st.generated_tokens},
        "kinds": {"new": st.kinds[0], "continued": st.kinds[1], "saved": st.kinds[2], "forked": st.kinds[3]},
        "sessions": st.states.len(), "kv_gib": round(st.kv_bytes as f64 / GIB, 3), "kv_limit_gib": round(s.budgets.kv_gb, 3),
        "kv_total_gib": round(s.kv_used.load(Ordering::Relaxed) as f64 / GIB, 3),
        "reserve_gib": s.budgets.reserve_gb, "model_memory": allocations,
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
    // the decoding passes take the status's lock; the text is written without it
    drop(st);
    v.to_string()
}

// ---- the worker ----

/// A prefill in progress: its prompt and how far it has run.
struct Prefill {
    prompt: Vec<i32>,
    done: usize,
    capture: Option<usize>,
    origin: (usize, usize),
    replay: bool,
    kind: &'static str,
    images: Option<Images>,
    t0: Instant,
    c0: CacheCounts,
}

/// A request from its prefill to its end.
struct Gen {
    job: Job,
    sampler: Sampler,
    state: kv::Id,
    number: u64,
    n: usize,
    from: usize,
    /// Some while its prompt prefills.
    prefill: Option<Prefill>,
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
    c1: CacheCounts,
}

impl Gen {
    fn decoding(&self) -> bool {
        self.prefill.is_none()
    }
}

/// One model's engine handle and requests. State IDs belong to Memory's
/// shared budget; prefix matching remains scoped to this model.
struct Slot {
    index: usize,
    model: Arc<Model>,
    served: Served,
    ctx: usize,
    batch: usize,
    gens: Vec<Gen>,
    think_start: Option<i32>,
    think_end: Option<i32>,
    shown: usize,
    /// The experts a decoded row holds: the live view's routed rows.
    per_row: usize,
}

// One KV budget and one LRU clock across models. Prefix matching is
// model-scoped; engine handles retain their owning model for eviction.
struct Memory {
    cache: Cache,
    states: HashMap<kv::Id, *mut RawState>,
    snaps: HashMap<(kv::Id, usize), *mut RawSnapshot>,
    owners: HashMap<kv::Id, Arc<Model>>,
    used: HashMap<kv::Id, Instant>,
    published: Instant,
}

impl Memory {
    fn new(limit: u64) -> Self {
        Self { cache: Cache::new(limit, STATES), states: HashMap::new(), snaps: HashMap::new(),
               owners: HashMap::new(), used: HashMap::new(), published: Instant::now() }
    }

    fn add(&mut self, model: usize, owner: Arc<Model>) -> kv::Id {
        let id = self.cache.add_for(model, 0);
        self.owners.insert(id, owner);
        id
    }

    /// Frees a snapshot's engine copy.
    fn free_snap(&mut self, id: kv::Id, pos: usize) {
        if let Some(x) = self.snaps.remove(&(id, pos)) {
            self.owners[&id].snapshot_free(x);
        }
    }

    /// Frees what an eviction or a truncation dropped.
    fn free(&mut self, snaps: Vec<(kv::Id, Snap)>, states: Vec<(kv::Id, Vec<Snap>)>) {
        for (id, s) in snaps {
            self.free_snap(id, s.pos);
        }
        for (id, held) in states {
            for s in held {
                self.free_snap(id, s.pos);
            }
            if let Some(st) = self.states.remove(&id) {
                self.owners[&id].state_free(st);
            }
            self.used.remove(&id);
            self.owners.remove(&id);
        }
    }

    fn publish(&mut self, s: &Shared) {
        let now = Instant::now();
        s.kv_used.store(self.cache.bytes(), Ordering::Relaxed);
        for (model, m) in s.models.iter().enumerate() {
            let entries: Vec<_> = self.cache.entries.iter().filter(|(_, e)| e.model == model).collect();
            let by_number: HashMap<_, _> = entries.iter().map(|(id, e)|
                (e.number, *self.used.get(id).unwrap_or(&now))).collect();
            let mut st = lock(&m.status);
            st.states = entries.iter().map(|(id, e)|
                (e.number, e.tokens.len(), *self.used.get(id).unwrap_or(&now))).collect();
            st.tree = self.cache.tree_for(model, 24, 40).into_iter().map(|x| {
                let used = *by_number.get(&x.number).unwrap_or(&now);
                (x, used)
            }).collect();
            st.kv_bytes = self.cache.bytes_for(model);
            st.kv_allocated = entries.iter().map(|(id, e)|
                self.owners[id].state_bytes(self.states[id], 0) + e.snaps.iter().map(|x| x.bytes).sum::<u64>()).sum();
        }
        self.published = now;
    }
}

struct Worker {
    s: Arc<Shared>,
    slots: Vec<Slot>,
    memory: Memory,
    maintained: Instant,
}

impl Worker {
    fn new(s: Arc<Shared>) -> Worker {
        let slots = s
            .models
            .iter()
            .enumerate()
            .map(|(index, m)| {
                let model = m.model.clone();
                let served = m.served.clone();
                engine::log(&format!(
                    "{}: shares the {:.1} GiB KV budget; a state takes {:.0} MiB at first, {:.2} GiB at {} tokens; a snapshot {:.1} MiB",
                    served.id,
                    s.budgets.kv_gb,
                    model.state_bytes(std::ptr::null(), 0) as f64 / 1048576.0,
                    model.state_bytes(std::ptr::null(), served.engine.ctx as usize) as f64 / GIB,
                    served.engine.ctx,
                    model.snapshot_bytes(true) as f64 / 1048576.0
                ));
                Slot {
                    index,
                    ctx: served.engine.ctx as usize,
                    batch: served.batch.max(1).min(model.max_rows()),
                    gens: Vec::new(),
                    think_start: model.tokenizer().special("<think>"),
                    think_end: model.tokenizer().special("</think>"),
                    shown: 0,
                    per_row: model.stats().experts_per_row.max(0) as usize,
                    model,
                    served,
                }
            })
            .collect();
        let memory = Memory::new((s.budgets.kv_gb * GIB) as u64);
        Worker { s, slots, memory, maintained: Instant::now() }
    }

    /// Apply one RAM reserve between engine calls, including while only a
    /// dense model is active. Release expert slots first, then idle KV.
    fn maintain(&mut self) {
        let mut released = 0;
        for slot in &self.slots {
            if slot.per_row > 0 {
                let before = slot.model.stats().cache_bytes;
                slot.model.idle();
                released += before.saturating_sub(slot.model.stats().cache_bytes);
            } else if slot.gens.is_empty() {
                slot.model.idle();
            }
        }
        if let Some(slot) = self.slots.first() {
            let available = slot.model.stats().available;
            let reserve = (self.s.budgets.reserve_gb * GIB) as u64;
            // Released expert pages can take a moment to reach the VM
            // counters. Do not free the same deficit twice in this turn.
            let deficit = reserve.saturating_sub(available).saturating_sub(released);
            if deficit > 0 {
                let (snaps, states) = self.memory.cache.reclaim(deficit);
                self.memory.free(snaps, states);
                self.memory.publish(&self.s);
            }
        }
        self.maintained = Instant::now();
    }

    /// Every model takes one step a turn: admitting, one prefill call, one
    /// decoding step. Shared memory maintenance runs between model calls
    /// about once a second. On the stop, every request ends with
    /// Finish::Shutdown.
    fn run(&mut self) {
        let s = self.s.clone();
        while !engine::stopping() {
            if self.maintained.elapsed() >= Duration::from_secs(1) { self.maintain(); }
            let mut q = lock(&s.queue);
            let idle = |q: &Queue, slots: &[Slot]| {
                q.jobs.iter().all(|j| j.is_empty()) && q.text.is_none() && slots.iter().all(|x| x.gens.is_empty())
            };
            while idle(&q, &self.slots) && !engine::stopping() {
                let (guard, timeout) = s.wake.wait_timeout(q, Duration::from_secs(1)).unwrap_or_else(|e| e.into_inner());
                q = guard;
                if timeout.timed_out() && q.jobs.iter().all(|j| j.is_empty()) {
                    drop(q);
                    self.maintain();
                    q = lock(&s.queue);
                }
            }
            let question = q.text.take();
            let mut admitted: Vec<Vec<Job>> = Vec::new();
            for x in &self.slots {
                let mut admit = Vec::new();
                while x.gens.len() + admit.len() < x.batch && !engine::stopping() {
                    let Some(job) = q.jobs[x.index].pop_front() else { break };
                    admit.push(job);
                }
                if !admit.is_empty() {
                    engine::log(&format!(
                        "{}: admitted={}, waiting={}, running={}",
                        x.served.id,
                        admit.len(),
                        q.jobs[x.index].len(),
                        x.gens.len()
                    ));
                }
                admitted.push(admit);
            }
            drop(q);
            if let Some((model, session, start, end)) = question
                && let Some(x) = self.slots.get(model)
            {
                let answer = x.node_text(session, start, end, &self.memory);
                lock(&s.models[model].status).text = Some(answer);
            }
            for (x, admit) in self.slots.iter_mut().zip(admitted) {
                x.step(&s, admit, &mut self.memory);
            }
        }
        for x in &mut self.slots {
            for g in std::mem::take(&mut x.gens) {
                x.finish(&s, g, true, &mut self.memory);
            }
        }
        let waiting: Vec<Job> = {
            let mut q = lock(&s.queue);
            q.closed = true;
            q.jobs.iter_mut().flat_map(|j| j.drain(..)).collect()
        };
        for job in waiting {
            end_waiting(job, Finish::Shutdown);
        }
        for x in &mut self.slots {
            x.set_status(&s, "idle");
        }
    }
}

fn refuse(job: Job, message: &str) {
    engine::log(&format!("req {}: refused: {message}", job.id));
    job.last(Event::Refused(message.into()));
}

fn end_waiting(job: Job, finish: Finish) {
    engine::log(&format!(
        "req {}: ended before prefill ({}), waited {:.3} s",
        job.id,
        if finish == Finish::Cancelled { "cancelled" } else { "shutdown" },
        job.queued.elapsed().as_secs_f64()
    ));
    job.last(Event::Done { finish });
}

impl Slot {
    /// The model's limits for its cache's plans.
    fn plan(&self, prompt: &[i32], memory: &Memory) -> Plan {
        let model = self.model.clone();
        let cost = move |k: usize| model.replay_rows(k).unwrap_or(0);
        let limits = Limits { rewind: self.model.rewind(), replay: self.model.replay_rows(1).is_some(), replay_cost: &cost };
        memory.cache.plan_for(self.index, prompt, limits)
    }

    /// One step of the model: its admitted requests start, one prefill call
    /// runs, the generations that are done end, the others advance a token.
    fn step(&mut self, s: &Shared, admit: Vec<Job>, memory: &mut Memory) {
        for job in admit {
            if engine::stopping() {
                end_waiting(job, Finish::Shutdown);
            } else if job.flags.cancelled.load(Ordering::SeqCst) {
                engine::log(&format!("req {}: client gone while waiting", job.id));
                end_waiting(job, Finish::Cancelled);
            } else {
                self.waiting_start(s, job, memory);
            }
        }
        if engine::stopping() {
            return;
        }
        self.prefill_call(s, memory);
        let ctx = self.ctx;
        let mut i = 0;
        while i < self.gens.len() {
            if self.gens[i].decoding() && Self::over(&self.model, &mut self.gens[i], ctx) {
                let g = self.gens.swap_remove(i);
                self.finish(s, g, false, memory);
            } else if !self.gens[i].decoding() && self.gens[i].job.flags.cancelled.load(Ordering::SeqCst) {
                let mut g = self.gens.swap_remove(i);
                g.cancelled = true;
                self.prefill_cut(&mut g, memory);
                self.finish(s, g, false, memory);
            } else {
                i += 1;
            }
        }
        let decoding = self.gens.iter().filter(|g| g.decoding()).count();
        if decoding > 1 && decoding != self.shown {
            engine::log(&format!("{}: {decoding} requests decoding together", self.served.id));
        }
        self.shown = decoding;
        if self.gens.is_empty() {
            self.set_status(s, "idle");
            return;
        }
        if decoding > 0 {
            self.advance(memory);
            if self.per_row > 0 && ROUTES.swap(false, Ordering::Relaxed) {
                self.copy_routes(s);
            }
        }
        for g in &mut self.gens {
            if !g.out.is_empty() {
                let _ = g.job.tx.send(Event::Tokens(std::mem::take(&mut g.out)));
            }
            if g.decoding() && g.tick.elapsed() >= Duration::from_secs(10) {
                g.tick = Instant::now();
                engine::log(&format!(
                    "req {}: {} tokens, {:.1} tokens/s",
                    g.job.id,
                    g.generated,
                    g.generated as f64 / (g.tick - g.t1).as_secs_f64()
                ));
            }
        }
        let phase = if self.gens.iter().any(|g| !g.decoding()) { "prefill" } else { "decode" };
        self.set_status(s, phase);
        if memory.published.elapsed() >= Duration::from_secs(1) {
            memory.publish(s);
        }
    }

    /// The tokens of state `session`'s positions start..end, for the live
    /// view: at most TEXT_TOKENS from start, none if the state is gone.
    fn node_text(&self, session: u64, start: usize, end: usize, memory: &Memory) -> NodeText {
        let tokens = memory.cache.entries.values().find(|e| e.model == self.index && e.number == session).map_or(&[][..], |e| &e.tokens[..]);
        let to = end.min(tokens.len());
        let from = start.min(to);
        NodeText { model: self.index, session, start, end, ids: tokens[from..to.min(from + TEXT_TOKENS)].to_vec() }
    }

    /// Before a pass that makes the busy state `id` hold `positions`: the
    /// cache charges its growth before the engine grows it (kv::Cache::grow),
    /// and what the cache gives up for it is freed.
    fn make_room(&mut self, id: kv::Id, positions: usize, memory: &mut Memory) {
        let before = memory.cache.entries[&id].bytes;
        let (snaps, states) = memory.cache.grow(id, self.model.state_bytes(memory.states[&id], positions));
        memory.free(snaps, states);
        let after = memory.cache.entries[&id].bytes;
        if after > before && memory.cache.over() > 0 {
            engine::log(&format!(
                "{}: a decoding state grows to {:.2} GiB at {positions} positions; with only decoding states left, the kv cache takes {:.2} GiB of the {:.1} GiB it allows",
                self.served.id,
                after as f64 / GIB,
                memory.cache.bytes() as f64 / GIB,
                memory.cache.limit as f64 / GIB
            ));
        }
        if after > before { memory.publish(&shared()); }
    }

    fn set_status(&mut self, s: &Shared, phase: &'static str) {
        let mut st = lock(&s.models[self.index].status);
        if st.phase != phase || st.since.is_none() {
            st.since = Some(Instant::now());
        }
        st.phase = phase;
        st.active = self.gens.iter().filter(|g| g.decoding()).count();
        st.decoded = st.generated_tokens + self.gens.iter().map(|g| g.generated as u64).sum::<u64>();
        let first = self.gens.iter().find(|g| !g.decoding()).or(self.gens.first());
        (st.request, st.prompt, st.reused, st.generated) =
            first.map_or((0, 0, 0, 0), |g| (g.job.id, g.n, g.from, g.generated));
        st.live = self
            .gens
            .iter()
            .map(|g| LiveReq { number: g.job.id, session: g.number, prompt: g.n, reused: g.from, generated: g.generated })
            .collect();
    }

    /// A request's start: its images and prompt checked, the KV prefix cache's
    /// plan carried out, its prefill to come (or, for the same prompt again,
    /// its first token from the prompt-end snapshot). Its plan is made here,
    /// when it is admitted, after the prefills before it.
    fn waiting_start(&mut self, s: &Shared, job: Job, memory: &mut Memory) {
        let mut prompt = job.request.prompt.clone();
        let mut images = None;
        if !job.request.images.is_empty() {
            match self.model.images(&job.request.images) {
                Err(m) => return refuse(job, &m),
                Ok(mut x) => match x.expand(&prompt, self.ctx + 1) {
                    Some(p) => {
                        prompt = p;
                        images = Some(x);
                    }
                    None => {
                        let m = format!("the prompt's image places and its {} images differ", job.request.images.len());
                        return refuse(job, &m);
                    }
                },
            }
        }
        let n = prompt.len();
        if n >= self.ctx {
            let m = format!("the prompt has {n} tokens; the context holds {}", self.ctx);
            return refuse(job, &m);
        }
        let sm = job.request.sampling;
        let Some(sampler) = Sampler::new(sm.temperature, sm.top_p, sm.top_k, sm.seed) else {
            return refuse(job, "invalid sampling settings");
        };
        let max_new = (job.request.max_tokens.max(1) as usize).min(self.ctx - n);
        let plan = self.plan(&prompt, memory);
        let origin = memory.cache.origin(&plan, n);
        let capture = plan.capture(n);
        // The state at the prompt's end and its first token, and the
        // snapshots the plan takes.
        let state_need = match plan {
            Plan::InPlace { state, .. } => {
                self.model.state_bytes(memory.states[&state], n + 1).saturating_sub(memory.cache.entries[&state].bytes)
            }
            _ => self.model.state_bytes(std::ptr::null(), n + 1),
        };
        let need = state_need + plan.snapshot_need(n, self.model.snapshot_bytes(false), self.model.snapshot_bytes(true));
        let new_state = !matches!(plan, Plan::InPlace { .. });
        let keep: Vec<kv::Id> = plan.source().into_iter().collect();
        let Some(ev) = memory.cache.fit(need, new_state, &keep) else {
            let busy: u64 = memory.cache.entries.values().filter(|e| e.busy).map(|e| e.bytes).sum();
            let m = format!(
                "the conversation's attention state needs {:.1} GiB ({:.1} GiB with its snapshots); the kv cache allows {:.1} GiB, and the requests running take {:.1} GiB of it",
                self.model.state_bytes(std::ptr::null(), n + 1) as f64 / GIB,
                need as f64 / GIB,
                memory.cache.limit as f64 / GIB,
                busy as f64 / GIB
            );
            return refuse(job, &m);
        };
        let wait = job.queued.elapsed().as_secs_f64();
        let (snaps, states) = memory.cache.evict(&ev);
        memory.free(snaps, states);
        let from = plan.reused();
        let id = match plan {
            Plan::Cold => {
                let id = memory.add(self.index, self.model.clone());
                memory.states.insert(id, self.model.state_new());
                id
            }
            Plan::InPlace { state, from } => {
                memory.cache.touch(state, from);
                state
            }
            Plan::Snapshot { source, at, .. } | Plan::Replay { source, at } => {
                let id = memory.add(self.index, self.model.clone());
                let st = self.model.state_new();
                memory.states.insert(id, st);
                self.model.state_copy(st, memory.states[&source], at);
                memory.cache.touch(source, at);
                memory.used.insert(source, Instant::now());
                if let Plan::Snapshot { snap, .. } = plan {
                    self.model.snapshot_restore(st, memory.snaps[&(source, snap)]);
                }
                id
            }
        };
        let bytes = self.model.state_bytes(memory.states[&id], n + 1);
        let dropped = memory.cache.start(id, &prompt, from, bytes, kv::stale_at(origin, from));
        let mut promised = Vec::new();
        if from < n {
            promised.push(Snap { pos: n, kind: Kind::PromptEnd, bytes: self.model.snapshot_bytes(true), stale: kv::stale_at(origin, n) });
        }
        if let Some(pos) = capture {
            promised.push(Snap { pos, kind: Kind::Branch, bytes: self.model.snapshot_bytes(false), stale: kv::stale_at(origin, pos) });
        }
        memory.cache.reserve_snapshots(id, promised);
        for x in dropped {
            memory.free_snap(id, x.pos);
        }
        memory.used.insert(id, Instant::now());
        memory.publish(s);
        let number = memory.cache.entries[&id].number;
        let kind = plan.kind(n);
        let thinking = self.think_start == Some(prompt[n - 1]);
        engine::log(&format!(
            "req {}: {} {n} prompt tokens, session {number} ({kind}), {from} reused; thinking {}; at most {max_new} new tokens, {}; waited {wait:.1} s",
            job.id,
            self.served.id,
            if thinking { "on" } else { "off" },
            if sm.temperature > 0.0 { "sampled" } else { "greedy" }
        ));
        let replay = matches!(plan, Plan::Replay { .. });
        if let Some(x) = images.as_mut() {
            // The images whose spans the prefill reaches: a replay computes again
            // from its first position, before the branch point.
            let first = if replay { self.model.replay_start(from) } else { from };
            let t = Instant::now();
            let encoded = x.encode(first);
            if encoded > 0 {
                engine::log(&format!(
                    "req {}: {encoded} of {} images encoded in {:.2} s",
                    job.id,
                    job.request.images.len(),
                    t.elapsed().as_secs_f64()
                ));
            }
        }
        let now = Instant::now();
        let mut g = Gen {
            job,
            sampler,
            state: id,
            number,
            n,
            from,
            prefill: Some(Prefill {
                prompt,
                done: from,
                capture,
                origin,
                replay,
                kind,
                images,
                t0: now,
                c0: self.model.stats().cache,
            }),
            pos: n,
            generated: 0,
            reasoning: 0,
            thinking,
            ended: false,
            cancelled: false,
            t: 0,
            out: Vec::new(),
            draft_passes: 0,
            draft_accepted: 0,
            lookup: None,
            lookup_indexed: n,
            lookup_passes: 0,
            lookup_offered: 0,
            lookup_accepted: 0,
            max_new,
            t1: now,
            tick: now,
            c1: CacheCounts::default(),
        };
        if from == n {
            // The same prompt again: its prompt-end snapshot holds its logits.
            let source = plan.source().expect("a saved prompt");
            let x = memory.snaps[&(source, n)];
            if source == id {
                self.model.snapshot_restore(memory.states[&id], x);
            }
            let prompt = g.prefill.as_ref().map(|p| p.prompt.clone()).unwrap_or_default();
            self.model.seek(memory.states[&id], &prompt, n);
            let lg = self.model.snapshot_logits(x);
            self.started(s, &mut g, lg, PrefillInfo::default(), memory);
        }
        self.gens.push(g);
    }

    /// One prefill call of the oldest request still prefilling: the rest of
    /// its prompt for a model that takes it in one call, otherwise up to its
    /// prefill rows (fewer while others decode), cut at the capture point and,
    /// for a model that cannot replay, at the snapshot grid.
    fn prefill_call(&mut self, s: &Shared, memory: &mut Memory) {
        let Some(i) = self.gens.iter().position(|g| !g.decoding()) else { return };
        let others = self.gens.iter().any(|g| g.decoding());
        let chunked = self.model.prefill_rows();
        let can_replay = self.model.replay_rows(1).is_some();
        let (state, rows) = (self.gens[i].state, self.gens[i].n);
        let st = memory.states[&state];
        let p = self.gens[i].prefill.as_ref().expect("a prefill");
        let (done, capture) = (p.done, p.capture);
        let end = match chunked {
            None => rows,
            Some(most) => {
                let mut end = (done + if others { SHARED_PREFILL.min(most) } else { most }).min(rows);
                if !can_replay {
                    end = end.min((done / SNAP_GRID + 1) * SNAP_GRID);
                }
                if let Some(at) = capture
                    && done < at
                    && at < end
                {
                    end = at;
                }
                end
            }
        };
        if self.gens[i].job.flags.cancelled.load(Ordering::SeqCst) {
            return; // ended by step
        }
        let replay = p.replay && done == self.gens[i].from;
        // in one call the capture is the prefill's own; in calls, a snapshot where a call ends
        let inner = if chunked.is_none() { capture } else { None };
        let prompt = &p.prompt[..end];
        let Some((lg, captured, info)) = self.model.prefill(st, prompt, done, replay, inner) else {
            panic!("the engine refused a prefill of positions {done} to {end}");
        };
        let origin = p.origin;
        memory.cache.prefilled(state, prompt, kv::stale_at(origin, end));
        if let Some(at) = inner
            && !captured.is_null()
        {
            memory.snaps.insert((state, at), captured);
            let stale = kv::stale_at(origin, at);
            memory.cache.add_snap(state, Snap { pos: at, kind: Kind::Branch, bytes: self.model.snapshot_bytes(false), stale });
        }
        let last = end == rows;
        if chunked.is_some() && !last {
            let grid = !can_replay && end % SNAP_GRID == 0 && memory.cache.snap(state, end).is_none();
            if capture == Some(end) || grid {
                let bytes = self.model.snapshot_bytes(false);
                let fits = capture == Some(end)
                    || memory.cache.fit(bytes, false, &[state]).is_some_and(|ev| {
                        let (snaps, states) = memory.cache.evict(&ev);
                        memory.free(snaps, states);
                        true
                    });
                if fits {
                    let x = self.model.snapshot_take(st, end, false);
                    memory.snaps.insert((state, end), x);
                    let stale = kv::stale_at(origin, end);
                    memory.cache.add_snap(state, Snap { pos: end, kind: Kind::Branch, bytes, stale });
                }
            }
        }
        if !last {
            self.gens[i].prefill.as_mut().expect("a prefill").done = end;
            return;
        }
        let x = self.model.snapshot_take(st, rows, true);
        memory.snaps.insert((state, rows), x);
        let stale = kv::stale_at(origin, rows);
        memory.cache.add_snap(state, Snap { pos: rows, kind: Kind::PromptEnd, bytes: self.model.snapshot_bytes(true), stale });
        let mut g = self.gens.remove(i);
        self.started(s, &mut g, lg, info, memory);
        self.gens.insert(i, g);
    }

    /// A cancelled request whose prefill ran part of its prompt: its state's
    /// tokens are those its rows hold.
    fn prefill_cut(&mut self, g: &mut Gen, memory: &mut Memory) {
        if let Some(p) = g.prefill.take()
            && let Some(e) = memory.cache.entries.get_mut(&g.state)
        {
            e.tokens.truncate(p.done);
            g.pos = p.done;
        }
    }

    /// The prefill ended with these logits: the request's first token.
    fn started(&mut self, s: &Shared, g: &mut Gen, lg: Logits, info: PrefillInfo, memory: &mut Memory) {
        memory.cache.clear_pending(g.state);
        let p = g.prefill.take().expect("a prefill");
        let n = g.n;
        let from = g.from;
        let t1 = Instant::now();
        let dt = (t1 - p.t0).as_secs_f64();
        let c1 = self.model.stats();
        let read = (c1.cache.bytes - p.c0.bytes) as f64 / 1e9;
        if from < n {
            let replayed = if p.replay {
                format!(
                    "; {} replay at {from}: up to {} earlier rows computed again",
                    if self.served.engine.ced > 0 { "a bounded" } else { "an exact" },
                    (info.tokens as usize).saturating_sub(n - from)
                )
            } else {
                String::new()
            };
            engine::log(&format!(
                "req {}: prefill of {} tokens in {dt:.2} s ({:.0} tokens/s), {read:.2} GB of experts read{replayed}",
                g.job.id,
                n - from,
                (n - from) as f64 / dt.max(1e-9)
            ));
            engine::log(&format!(
                "req {}: prefill plan: {} chunks, {} encoder tokens, {} decoder tokens, ced_applied={}, prune_applied={}, prefix={} (full)",
                g.job.id, info.chunks, info.tokens, info.decoder_tokens, info.ced_applied, info.prune_applied, info.prefix_tokens
            ));
        }
        let _ = g.job.tx.send(Event::Started { prompt: n, reused: from });
        {
            let mut st = lock(&s.models[self.index].status);
            st.prefill = info;
            if from < n {
                st.prefilled.seconds += dt;
            }
            (st.prefilled.rows, st.prefilled.bytes) = (c1.prefill_rows, c1.prefill.bytes);
            st.kinds[match p.kind {
                "new" => 0,
                "kept" => 1,
                "saved" => 2,
                _ => 3,
            }] += 1;
        }
        drop(p.images);
        g.t = self.model.sample(lg, &mut g.sampler);
        if self.served.engine.lookup > 0 {
            let mut l = Lookup::new(self.ctx).unwrap_or_else(|| panic!("cannot initialize context lookup"));
            if !l.append(&p.prompt) {
                panic!("cannot initialize context lookup");
            }
            g.lookup = Some(l);
        }
        g.t1 = t1;
        g.tick = t1;
        g.c1 = c1.cache;
        memory.publish(s);
    }

    /// The token t ran at g's position: into its state's tokens and the step's output.
    fn emit(model: &Model, cache: &mut Cache, g: &mut Gen, t: i32, think_end: Option<i32>) {
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
            g.thinking = Some(t) != think_end;
        }
        cache.push(g.state, t);
        g.pos += 1;
        g.generated += 1;
        g.ended = model.stop(t); // a kept draft ends the text
        if !g.ended {
            g.out.push(t as u32);
        }
    }

    /// Whether the generation ends before its next token: at an end token,
    /// its budget, the context's end, or when it was cancelled.
    fn over(model: &Model, g: &mut Gen, ctx: usize) -> bool {
        if g.job.flags.cancelled.load(Ordering::SeqCst) {
            g.cancelled = true;
        }
        if g.ended || g.cancelled {
            return true;
        }
        if model.stop(g.t) {
            g.generated += 1;
            g.ended = true;
            return true;
        }
        if g.generated + 1 >= g.max_new || g.pos + 1 >= ctx {
            // The token waiting is the reply's last: it goes out without a
            // pass, so its state ends at the reply's last token run, which a
            // model that cannot rewind needs for the next turn to continue it.
            if g.thinking {
                g.reasoning += 1;
            }
            g.out.push(g.t as u32);
            g.generated += 1;
            return true;
        }
        false
    }

    fn finish(&mut self, s: &Shared, mut g: Gen, shutdown: bool, memory: &mut Memory) {
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
        if g.prefill.is_some() {
            self.prefill_cut(&mut g, memory);
        }
        // The generation's end, for forks there, when the cache keeps it.
        let stale = memory.cache.entries[&g.state].stale;
        let end = memory.cache.snap(g.state, g.pos).is_none().then(|| Snap {
            pos: g.pos,
            kind: Kind::GenerationEnd,
            bytes: self.model.snapshot_bytes(false),
            stale,
        });
        let (take, (snaps, states)) = memory.cache.end(g.state, self.model.state_bytes(memory.states[&g.state], g.pos), end);
        if take {
            let x = self.model.snapshot_take(memory.states[&g.state], g.pos, false);
            memory.snaps.insert((g.state, g.pos), x);
        }
        memory.free(snaps, states);
        memory.used.insert(g.state, Instant::now());
        let st = self.model.stats();
        let (c2, dt) = (st.cache, g.t1.elapsed().as_secs_f64());
        let hits = c2.hits - g.c1.hits + c2.prefetch_hits - g.c1.prefetch_hits;
        let misses = c2.misses - g.c1.misses;
        let experts = if self.per_row > 0 {
            format!(
                "; experts {:.1}% hit, {:.2} GB read",
                100.0 * hits as f64 / ((hits + misses).max(1)) as f64,
                (c2.bytes - g.c1.bytes) as f64 / 1e9
            )
        } else {
            String::new()
        };
        engine::log(&format!(
            "req {}: {name}: {} tokens ({} reasoning) in {dt:.1} s, {:.1} tokens/s{experts}; kv {} sessions, {:.2} GiB; available {:.1} GiB, footprint {:.1} GiB",
            g.job.id,
            g.generated,
            g.reasoning,
            g.generated as f64 / dt.max(1e-9),
            memory.cache.entries.len(),
            memory.cache.bytes() as f64 / GIB,
            st.available as f64 / GIB,
            st.footprint as f64 / GIB
        ));
        let sm = g.job.request.sampling;
        engine::log(&format!(
            "req {}: sampling temperature {}, top_p {}, top_k {}; {} draft passes, {} drafts accepted",
            g.job.id, sm.temperature, sm.top_p, sm.top_k, g.draft_passes, g.draft_accepted
        ));
        if g.lookup.is_some() {
            engine::log(&format!(
                "req {}: context lookup {} passes, {} offered, {} accepted",
                g.job.id, g.lookup_passes, g.lookup_offered, g.lookup_accepted
            ));
        }
        let mut status = lock(&s.models[self.index].status);
        status.requests += 1;
        status.prompt_tokens += g.n as u64;
        status.generated_tokens += g.generated as u64;
        status.reused_tokens += g.from as u64;
        drop(status);
        memory.publish(s);
    }

    /// One token for each decoding generation: alone with its drafts, or
    /// together, each row as it would run alone.
    fn advance(&mut self, memory: &mut Memory) {
        let (think_end, ctx) = (self.think_end, self.ctx);
        let decoding: Vec<usize> = (0..self.gens.len()).filter(|&i| self.gens[i].decoding()).collect();
        if decoding.len() == 1 {
            let gi = decoding[0];
            let state = self.gens[gi].state;
            let st = memory.states[&state];
            let lookup = self.served.engine.lookup as usize;
            let verify_room = self.model.max_verify().saturating_sub(1);
            let (mut rows, proposal, k, pos) = {
                let g = &mut self.gens[gi];
                let mut rows = vec![g.t];
                // Drafts or proposals the pass may run: after the token and
                // every one of them is kept, the token sampled after them
                // still fits the budget and a position before the context's
                // end (it may go out without a pass, as over() gives it).
                let room = (g.max_new as isize - g.generated as isize - 2)
                    .min(ctx as isize - g.pos as isize - 2)
                    .min(verify_room as isize);
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
                    self.model.draft_block().min(room.max(0) as usize)
                };
                (rows, proposal, k, g.pos)
            };
            // The rows the pass writes: the token and its drafts or proposals.
            self.make_room(state, pos + 1 + k, memory);
            let model = self.model.clone();
            let g = &mut self.gens[gi];
            let (next, a) = if k > 0 {
                if proposal == 0 {
                    let mut d = vec![0i32; k];
                    model.draft(st, g.t, g.pos, &mut d);
                    rows.extend_from_slice(&d);
                }
                let lg = model.verify(st, &rows, g.pos).expect("a verify pass in the limits");
                let (next, a) = model.sample_verify(lg, &rows[1..], &mut g.sampler);
                g.draft_passes += 1;
                g.draft_accepted += a;
                model.accept(st, &rows[..a + 1], g.pos);
                (next, a)
            } else {
                let lg = model.rows(&mut [st], &[g.t], &[g.pos as i32]).expect("a decoding pass in the context");
                (model.sample(lg, &mut g.sampler), 0)
            };
            if proposal > 0
                && let Some(l) = g.lookup.as_mut()
            {
                l.feedback(k, a);
                g.lookup_accepted += a;
            }
            let mut kept = 0;
            for &t in &rows[..=a] {
                if g.ended {
                    break;
                }
                Self::emit(&model, &mut memory.cache, g, t, think_end);
                kept += 1;
            }
            // the pass's rows after the kept tokens took places in the rings
            memory.cache.rejected(g.state, rows.len() - kept);
            g.t = next;
            return;
        }
        let rooms: Vec<(kv::Id, usize)> = decoding.iter().map(|&i| (self.gens[i].state, self.gens[i].pos + 1)).collect();
        for (id, positions) in rooms {
            self.make_room(id, positions, memory);
        }
        let mut st: Vec<*mut RawState> = decoding.iter().map(|&i| memory.states[&self.gens[i].state]).collect();
        let tokens: Vec<i32> = decoding.iter().map(|&i| self.gens[i].t).collect();
        let pos: Vec<i32> = decoding.iter().map(|&i| self.gens[i].pos as i32).collect();
        let model = self.model.clone();
        let lg = model.rows(&mut st, &tokens, &pos).expect("rows in the limits");
        let vocab = model.vocab();
        for (r, &i) in decoding.iter().enumerate() {
            let g = &mut self.gens[i];
            Self::emit(&model, &mut memory.cache, g, tokens[r], think_end);
            // Safety: myds4_rows returns one row of the vocabulary for each state.
            g.t = model.sample(unsafe { lg.add(r * vocab) }, &mut g.sampler);
        }
    }

    /// The decoding requests' routed rows of the latest pass, for the live
    /// view: row r is the r-th decoding request's (alone, its token is row 0).
    fn copy_routes(&self, s: &Shared) {
        let routes: Vec<Route> = self
            .gens
            .iter()
            .filter(|g| g.decoding())
            .enumerate()
            .filter_map(|(r, g)| {
                self.model.routes(r, self.per_row).map(|(experts, sources)| Route { request: g.job.id, experts, sources })
            })
            .collect();
        lock(&s.models[self.index].status).routes = routes;
    }
}
