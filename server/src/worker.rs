//! One engine thread registers the configured models and loads each model
//! on its first inference request. It makes every engine call, so passes use
//! one GPU command queue. HTTP handlers submit requests, each to its model,
//! to a bounded queue (at most [`WAITING`] wait per model) and get their
//! events back on a channel.
//!
//! Each turn of the loop gives every model one step, in turn: it admits the
//! model's waiting requests while fewer than its batch run, starts one prefill
//! call of its oldest prefilling request (the whole prompt for a model that
//! plans its prefill over the whole prompt; otherwise up to its prefill rows)
//! if none runs, and starts a decoding call for every decoding request with no
//! call running: alone with the model's drafts (draft, lookup), or together
//! with the others, at most the model's rows per call. Calls run on streams
//! (engine.rs), of any model at the same time: after the models' steps the
//! loop advances the streams for a short time slice and finishes the call
//! that ended. Prompts resume from the model's KV prefix cache (kv.rs). The
//! worker also builds each model's status and the live view's snapshot.

use std::collections::{HashMap, HashSet, VecDeque};
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use serde_json::{Value, json};
use tokio::sync::mpsc;

use crate::control::{self, Action, Control, Phase, Target};
use crate::engine::{
    self, CacheCounts, DraftChoice, Logits, Lookup, Media, Model, PrefillInfo, RawSnapshot, RawState, Sampler,
    Stream,
};
use crate::disk::{Engine, LoadError, Store};
use crate::kv::{self, Cache, Entry, Freed, Kind, Limits, Plan, Snap};

/// Requests waiting for a model at most; more are answered 429.
pub const WAITING: usize = 16;
/// Attention states the shared cache keeps at most, across all models.
pub const STATES: usize = 256;
/// A prefill in calls snapshots its state at every multiple of this many
/// positions (within the KV cache's budget), when the model cannot replay: a
/// later prompt that shares a long prefix with it but parts before its end
/// (another conversation with the same system prompt and tools) resumes from
/// the last one before the point where it parts.
const SNAP_GRID: usize = 2048;
/// How long the streams run before the worker looks at its queue again.
const RUN_SLICE_SECONDS: f64 = 0.005;
/// A request that has just finished its prefill waits for the prefills still
/// to come while they are at most this many tokens in all, and for at most
/// HOLD_TIME: its first calls then start full.
const HOLD_PREFILL_ROWS: usize = 4096;
const HOLD_TIME: Duration = Duration::from_secs(20);
const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

/// How many decoding calls of a model run at once: as many as its decoding
/// requests fill (a call takes at most `most` rows), and one for a model whose
/// decoder keeps its jobs in model-wide state.
fn calls_at_once(decoding: usize, most: usize, serial: bool) -> usize {
    if serial { 1 } else { (decoding / most).max(1) }
}

/// Whether a request ready to decode waits for the `prefilling` prompt tokens
/// still to come: only a request that has not decoded yet, for short prefills,
/// and not for long.
fn held_for_prefills(generated: usize, waited: Duration, prefilling: usize) -> bool {
    prefilling > 0 && prefilling <= HOLD_PREFILL_ROWS && generated == 0 && waited <= HOLD_TIME
}

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
    /// Its prefill ran: prompt tokens (media expanded), and those it reused.
    Started { prompt: usize, reused: usize },
    /// The tokens one decoding step kept; never the end token.
    Tokens(Vec<u32>),
    /// Its end; the last event. (The worker logs its counts.)
    Done { finish: Finish },
    /// Refused before its prefill; the last event.
    Refused(String),
    /// Model preparation failed; other registered models remain available.
    Failed(String),
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
/// ids (each image's place the model's image token), its media' file bytes,
/// sampling and its token limit.
pub struct Request {
    pub model: usize,
    pub prompt: Vec<i32>,
    pub media: Vec<Vec<u8>>,
    pub sampling: Sampling,
    pub max_tokens: i64,
    /// Logged as "req ID NOTE" when the request is queued.
    pub note: String,
}

pub enum SubmitError {
    /// Too many requests wait, or the server stops.
    Busy(String),
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
        if !self.flags.ended.load(Ordering::SeqCst)
            && !self.flags.cancelled.swap(true, Ordering::SeqCst)
        {
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
    /// Seconds without requests before its weights and experts unload; 0 never.
    pub idle_unload: f64,
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
    control: Control,
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
    load_error: Option<String>,
    engine: engine::Stats,
    loaded: bool,
    drafts: usize,
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
    kinds: [u64; 5], // new, continued, saved, forked, loaded
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
    pub tokenizer: engine::Tokenizer,
    status: Mutex<Status>,
}

struct Shared {
    queue: Mutex<Queue>,
    wake: Condvar,
    models: Vec<ModelShared>,
    budgets: crate::config::Budgets,
    kv_used: AtomicU64,
    /// The states saved on the SSD, for the status and the live view.
    disk: Mutex<DiskStatus>,
    started: Instant,
}

/// The saved states: whether the store is open (its folder), how many there
/// are and their bytes against the budget, and this run's loads, saves and
/// bytes written.
#[derive(Clone, Debug, Default)]
pub struct DiskStatus {
    pub dir: Option<String>,
    pub states: usize,
    pub bytes: u64,
    pub limit: u64,
    pub loads: u64,
    pub saves: u64,
    pub written: u64,
}

static SHARED: Mutex<Option<Arc<Shared>>> = Mutex::new(None);
static THREAD: Mutex<Option<JoinHandle<()>>> = Mutex::new(None);

fn shared() -> Arc<Shared> {
    SHARED.lock().unwrap().clone().expect("the worker started")
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

pub fn control_status() -> Value {
    let s = shared();
    lock(&s.queue).control.snapshot()
}

pub fn yield_request(request: control::YieldRequest) -> Result<(u16, Value), control::Error> {
    let s = shared();
    let mut q = lock(&s.queue);
    if q.closed || engine::stopping() {
        return Err(control::Error {
            status: 503,
            code: "server_stopping",
            message: "the server is stopping".into(),
        });
    }
    let status = q.control.yield_request(request)?;
    let value = q.control.snapshot();
    s.wake.notify_all();
    Ok((status, value))
}

pub fn resume_request(request: control::ResumeRequest) -> Result<(u16, Value), control::Error> {
    let s = shared();
    let mut q = lock(&s.queue);
    if q.closed || engine::stopping() {
        return Err(control::Error {
            status: 503,
            code: "server_stopping",
            message: "the server is stopping".into(),
        });
    }
    let ended = q.control.resume_request(&request.operation_id)?;
    let mut value = q.control.snapshot();
    if ended {
        value["resumed_operation_id"] = json!(request.operation_id);
    }
    s.wake.notify_all();
    Ok((if ended { 200 } else { 202 }, value))
}

/// The models the worker serves, in their order: an index of this list is a
/// request's model.
pub fn models() -> Vec<(String, engine::Tokenizer, Served)> {
    shared()
        .models
        .iter()
        .map(|m| (m.served.id.clone(), m.tokenizer, m.served.clone()))
        .collect()
}

/// Registers model metadata on the engine thread without loading weights.
/// Returns once it takes requests, or false when a model's
/// settings are invalid (the engine says why on stderr).
pub fn start(
    served: Vec<Served>,
    budgets: crate::config::Budgets,
    kv_dir: Option<PathBuf>,
    on_log: Option<extern "C" fn(*const std::ffi::c_char)>,
    on_die: Option<extern "C" fn()>,
) -> bool {
    let (tx, rx) = std::sync::mpsc::channel();
    let handle = thread::Builder::new()
        .name("myds4-engine".into())
        .stack_size(16 << 20)
        .spawn(move || {
            if !engine::init(
                budgets.memory_gb,
                budgets.reserve_gb,
                on_log,
                on_die,
            ) {
                let _ = tx.send(false);
                return;
            }
            engine::refresh_stats();
            let mut models = Vec::new();
            let mut engines = Vec::new();
            for s in &served {
                let Some(m) = Model::open(&s.engine) else {
                    let _ = tx.send(false);
                    return;
                };
                models.push(ModelShared {
                    served: s.clone(),
                    tokenizer: m.tokenizer(),
                    status: Mutex::new(Status {
                        engine: m.stats(),
                        drafts: m.draft_block(),
                        ..Status::default()
                    }),
                });
                engines.push(Rc::new(m));
            }
            let n = models.len();
            let s = Arc::new(Shared {
                queue: Mutex::new(Queue {
                    jobs: (0..n).map(|_| VecDeque::new()).collect(),
                    control: Control::new(control::Limits {
                        model_bytes: engine::memory().limit,
                        kv_bytes: (budgets.kv_gb * GIB) as u64,
                    }),
                    ..Queue::default()
                }),
                wake: Condvar::new(),
                models,
                budgets,
                kv_used: AtomicU64::new(0),
                disk: Mutex::new(DiskStatus::default()),
                started: Instant::now(),
            });
            *SHARED.lock().unwrap() = Some(s.clone());
            // A failure here ends the process, as the engine's own do: the
            // requests could not end otherwise. (The panic's message is out.)
            let run = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let mut w = Worker::new(s.clone(), engines, kv_dir);
                let _ = tx.send(true);
                w.run();
                w.save_at_quit();
                for slot in &w.slots {
                    slot.model.finish();
                }
            }));
            if run.is_err() {
                crate::ui::leave();
                std::process::exit(1);
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
/// A second request while the states are saved at quit ends that saving.
pub fn request_stop() {
    engine::request_stop();
    wake();
}

fn wake() {
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
    s.and_then(|s| s.models.get(model).map(|m| m.tokenizer.text(ids)))
        .unwrap_or_default()
}

/// Requests the stop (unless one was: that request is the user's, and a
/// second one would end saving the states), waits for the worker to end every
/// request, save the states and the models' own and return.
pub fn stop() {
    if engine::stopping() {
        wake();
    } else {
        request_stop();
    }
    if let Some(h) = THREAD.lock().unwrap().take() {
        let _ = h.join();
    }
}

/// Queues a request: the guard that cancels it when dropped (and holds its
/// id), and its events.
pub fn submit(r: Request) -> Result<(Cancel, mpsc::UnboundedReceiver<Event>), SubmitError> {
    let s = shared();
    let (tx, rx) = mpsc::unbounded_channel();
    let flags = Arc::new(Flags::default());
    let mut q = lock(&s.queue);
    if q.closed || q.jobs[r.model].len() >= WAITING {
        let m = if q.closed {
            "the server is stopping"
        } else {
            "the server is busy: too many requests are waiting"
        };
        return Err(SubmitError::Busy(m.into()));
    }
    q.numbers += 1;
    let id = q.numbers;
    if !r.note.is_empty() {
        engine::log(&format!("req {id} {}", r.note)); // before the worker can take it
    }
    let (n, max_new, model) = (r.prompt.len(), r.max_tokens, r.model);
    q.jobs[model].push_back(Job {
        id,
        request: r,
        tx,
        flags: flags.clone(),
        queued: Instant::now(),
    });
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
    s.models
        .iter()
        .map(|m| {
            let st = lock(&m.status);
            let c = st.engine;
            json!({"model": m.served.id, "weights": c.dense_bytes, "experts": c.cache_bytes,
               "workspace": c.model_workspace, "kv": st.kv_allocated, "kv_committed": st.kv_bytes,
               "sessions": st.states.len(),
               "tracked": c.dense_bytes + c.cache_bytes + c.model_workspace + st.kv_allocated})
        })
        .collect()
}

/// Model `model`'s status (JSON): what it is doing, its requests, KV prefix
/// cache, counters, expert cache and the process's memory.
pub fn status(model: usize) -> Value {
    let s = shared();
    let Some(m) = s.models.get(model) else {
        return Value::Null;
    };
    let (waiting, control_phase, kv_limit) = {
        let q = lock(&s.queue);
        (
            q.jobs[model].len(),
            q.control.phase,
            q.control.effective.kv_bytes,
        )
    };
    let c = lock(&m.status).engine;
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
        "loaded": st.loaded, "load_error": st.load_error,
        "phase": if control_phase != Phase::Running { control_phase.label() } else if st.phase.is_empty() { "idle" } else { st.phase },
        "phase_seconds": round(st.since.map_or(0.0, |t| (now - t).as_secs_f64()), 1), "decoding": st.active,
        "request": st.request, "prompt_tokens": st.prompt, "reused_tokens": st.reused, "generated_tokens": st.generated,
        "waiting": waiting, "sessions": sessions,
        "kv_cache_gib": round(st.kv_bytes as f64 / GIB, 3), "kv_cache_limit_gib": round(kv_limit as f64 / GIB, 3),
        "kv_cache_scope": "server", "kv_cache_total_gib": round(s.kv_used.load(Ordering::Relaxed) as f64 / GIB, 3),
        "kv_disk": disk_json(&s),
        "reserve_gib": s.budgets.reserve_gb, "memory_gib": s.budgets.memory_gb, "model_memory": allocations,
        "model_memory_budget": {"limit": c.model_memory_limit, "allocated": c.model_memory_used,
                                "committed": c.model_memory_committed, "peak": c.model_memory_peak},
        "totals": {"requests": st.requests, "prompt_tokens": st.prompt_tokens, "generated_tokens": st.generated_tokens},
        "ced_depth": o.deepseek().map_or(0, |d| d.ced), "lookup_depth": o.lookup,
        "prefill_prune": o.deepseek().is_some_and(|d| d.prefill_prune),
        "runahead": o.deepseek().is_some_and(|d| d.runahead), "pipeline": o.deepseek().is_some_and(|d| d.pipeline),
        "drafts": st.drafts,
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
        "engine_times": engine_times(&c),
    })
}

/// Where the process's time went since start (the same for every model): the
/// engine thread's seconds by its state, the GPU main queue's busy seconds
/// and its idle seconds between command buffers by the engine thread's state
/// then, and the SSD's seconds with an expert read in flight.
fn engine_times(c: &engine::Stats) -> Value {
    let states = |t: &[f64; 5]| {
        json!({"outside": t[0], "pass": t[1], "wait_gpu": t[2], "wait_read": t[3], "wait_other": t[4]})
    };
    json!({"thread_seconds": states(&c.thread_s), "gpu_busy_seconds": c.gpu_busy_s,
           "gpu_idle_seconds": states(&c.gpu_idle_s), "ssd_busy_seconds": c.ssd_busy_s,
           "demand_read_seconds": c.demand_read_s, "background_read_seconds": c.background_read_s})
}

/// The states saved on the SSD (null without them): their folder, count and
/// GiB against the budget, and this run's loads, saves and GiB written.
fn disk_json(s: &Shared) -> Value {
    let d = lock(&s.disk).clone();
    match d.dir {
        None => Value::Null,
        Some(dir) => json!({"dir": dir, "states": d.states, "gib": round(d.bytes as f64 / GIB, 3),
                            "limit_gib": round(d.limit as f64 / GIB, 3), "loads": d.loads, "saves": d.saves,
                            "written_gib": round(d.written as f64 / GIB, 3)}),
    }
}

/// The live view's snapshot of model `model` (JSON): phase, requests,
/// statistics, memory and the states' prefix tree.
pub fn view(model: usize) -> String {
    let s = shared();
    let Some(m) = s.models.get(model) else {
        return "{}".into();
    };
    let (waiting, control_phase, kv_limit) = {
        let q = lock(&s.queue);
        (
            q.jobs[model].len(),
            q.control.phase,
            q.control.effective.kv_bytes,
        )
    };
    let c = lock(&m.status).engine;
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
    let pinned = c
        .cache_bytes
        .saturating_sub(c.slots.max(0) as u64 * c.slot_bytes);
    let mut v = json!({
        "model": m.served.id, "models": s.models.len(),
        "uptime_seconds": round(s.started.elapsed().as_secs_f64(), 0),
        "phase": if control_phase != Phase::Running { control_phase.label() } else if st.phase.is_empty() { "idle" } else { st.phase },
        "phase_seconds": round(st.since.map_or(0.0, |t| (now - t).as_secs_f64()), 1), "waiting": waiting, "requests": requests,
        "layers": c.layers, "experts_per_row": c.experts_per_row,
        "prefill": {"layer": c.prefill_layer, "rows": c.prefill_rows_now},
        "totals": {"requests": st.requests, "prompt_tokens": st.prompt_tokens, "reused_tokens": st.reused_tokens,
                   "generated_tokens": st.generated_tokens},
        "kinds": {"new": st.kinds[0], "continued": st.kinds[1], "saved": st.kinds[2], "forked": st.kinds[3],
                  "loaded": st.kinds[4]},
        "sessions": st.states.len(), "kv_gib": round(st.kv_bytes as f64 / GIB, 3), "kv_limit_gib": round(kv_limit as f64 / GIB, 3),
        "kv_total_gib": round(s.kv_used.load(Ordering::Relaxed) as f64 / GIB, 3),
        "reserve_gib": s.budgets.reserve_gb, "memory_gib": s.budgets.memory_gb, "model_memory": allocations,
        "model_memory_budget": {"limit": c.model_memory_limit, "allocated": c.model_memory_used,
                                "committed": c.model_memory_committed, "peak": c.model_memory_peak},
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
    // the routed experts' geometry and the saved states (past the json!
    // macro's recursion limit)
    v["routed_layers"] = json!(c.routed_layers);
    v["experts_per_layer"] = json!(c.experts_per_layer);
    v["kv_disk"] = disk_json(&s);
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
    media: Option<Media>,
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
    /// How many of the draft model's drafts its passes run, when the model drafts.
    choice: Option<DraftChoice>,
    /// The start and the draft count of its pass in flight while no other call
    /// of the model runs beside it: the chooser learns from such passes only.
    pass: Option<(Instant, usize)>,
    max_new: usize,
    t1: Instant,
    tick: Instant,
    c1: CacheCounts,
    /// Whether a call of this request is running on a stream.
    flying: bool,
    /// Since when it can decode: the request that has waited longest goes first.
    ready: Instant,
}

impl Gen {
    fn decoding(&self) -> bool {
        self.prefill.is_none()
    }
}

/// An engine call running on a stream, and what to do when it ends.
enum Call {
    /// One prefill call of the request: its prompt up to `end`, and the
    /// snapshot position the call itself takes, if any.
    Prefill { id: u64, end: usize, inner: Option<usize> },
    /// One token for each of these requests, row by row.
    Rows { ids: Vec<u64>, tokens: Vec<i32> },
    /// The request's token and its drafts or proposals (`proposal` of them
    /// proposed) in one pass.
    Verify { id: u64, rows: Vec<i32>, proposal: usize },
}

/// A call and the stream it runs on.
struct Flight {
    stream: Stream,
    call: Call,
}

/// One model's engine handle and requests. State IDs belong to Memory's
/// shared budget; prefix matching remains scoped to this model.
struct Slot {
    index: usize,
    model: Rc<Model>,
    served: Served,
    ctx: usize,
    batch: usize,
    gens: Vec<Gen>,
    /// Admitted requests that wait for their turn to plan their prefill.
    pending: VecDeque<Job>,
    think_start: Option<i32>,
    think_end: Option<i32>,
    shown: usize,
    /// The experts a decoded row holds: the live view's routed rows.
    per_row: usize,
    /// When the model last had a request, for its idle unload.
    used: Instant,
    /// The calls running now, and the streams that ended a call and wait for
    /// another. A prefill's stream holds its large working buffers only while
    /// a prompt fills.
    flights: Vec<Flight>,
    prefill_stream: Option<Stream>,
    decode_streams: Vec<Stream>,
}

// One KV budget and one LRU clock across models. Prefix matching is
// model-scoped; engine handles retain their owning model for eviction.
struct StoredState {
    owner: Rc<Model>,
    raw: *mut RawState,
    snapshots: HashMap<usize, *mut RawSnapshot>,
    used: Instant,
}

impl Drop for StoredState {
    fn drop(&mut self) {
        for (_, snapshot) in self.snapshots.drain() {
            self.owner.snapshot_free(snapshot);
        }
        self.owner.state_free(self.raw);
    }
}

struct Memory {
    cache: Cache,
    entries: HashMap<kv::Id, StoredState>,
    published: Instant,
    /// The states saved on the SSD (server/src/disk.rs); none without them.
    disk: Option<Store>,
    /// States whose prefill or decoding pass the stop cut short: their rows
    /// are not saved.
    cut: HashSet<kv::Id>,
}

impl Memory {
    fn new(limit: u64) -> Self {
        Self {
            cache: Cache::new(limit, STATES),
            entries: HashMap::new(),
            published: Instant::now(),
            disk: None,
            cut: HashSet::new(),
        }
    }

    /// What eviction gave up: its snapshots freed, and its states saved to
    /// the SSD (as the cache decides; the most recently used first, so that
    /// an older state that a newer one holds is not written) and freed.
    fn evicted(&mut self, (snaps, mut states): Freed) {
        for (id, snapshot) in snaps {
            self.free_snap(id, snapshot.pos);
        }
        states.sort_by_key(|(_, e)| std::cmp::Reverse(e.used));
        for (id, e) in states {
            self.save(id, &e, true);
            self.entries.remove(&id);
        }
    }

    /// Writes state `id` (its entry `e`, out of the cache) to the SSD as the
    /// cache decides; `kept`: the states still in the cache stay in memory.
    fn save(&mut self, id: kv::Id, e: &Entry, kept: bool) {
        let Memory {
            disk: Some(store),
            cache,
            entries,
            cut,
            ..
        } = self
        else {
            return;
        };
        let Some(st) = entries.get(&id) else {
            return;
        };
        if cut.contains(&id) {
            return;
        }
        let snapshot = |pos: usize| st.snapshots.get(&pos).copied().unwrap_or(std::ptr::null_mut());
        let used = std::time::SystemTime::now() - st.used.elapsed();
        if let Some(line) = store.save(cache, &Engine { model: &st.owner, state: st.raw }, &snapshot, e, used, kept) {
            engine::log(&format!("kv: {line}"));
        }
    }

    fn add(&mut self, model: usize, owner: Rc<Model>) -> kv::Id {
        let id = self.cache.add_for(model, 0);
        let raw = owner.state_new();
        self.entries.insert(
            id,
            StoredState {
                owner,
                raw,
                snapshots: HashMap::new(),
                used: Instant::now(),
            },
        );
        id
    }

    fn snapshot(&self, id: kv::Id, pos: usize) -> *mut RawSnapshot {
        self.entries[&id].snapshots[&pos]
    }

    fn save_snapshot(&mut self, id: kv::Id, pos: usize, snapshot: *mut RawSnapshot) {
        let entry = self
            .entries
            .get_mut(&id)
            .expect("a state owns its snapshots");
        if let Some(previous) = entry.snapshots.insert(pos, snapshot) {
            entry.owner.snapshot_free(previous);
        }
    }

    fn free_snap(&mut self, id: kv::Id, pos: usize) {
        let entry = self
            .entries
            .get_mut(&id)
            .expect("a state owns its snapshots");
        if let Some(snapshot) = entry.snapshots.remove(&pos) {
            entry.owner.snapshot_free(snapshot);
        }
    }

    fn free(&mut self, snaps: Vec<(kv::Id, Snap)>, states: Vec<(kv::Id, Vec<Snap>)>) {
        for (id, snapshot) in snaps {
            self.free_snap(id, snapshot.pos);
        }
        for (id, _) in states {
            self.entries.remove(&id);
        }
    }

    fn publish(&mut self, s: &Shared) {
        let now = Instant::now();
        s.kv_used.store(self.cache.bytes(), Ordering::Relaxed);
        for (model, m) in s.models.iter().enumerate() {
            let entries: Vec<_> = self
                .cache
                .entries
                .iter()
                .filter(|(_, e)| e.model == model)
                .collect();
            let by_number: HashMap<_, _> = entries
                .iter()
                .map(|(id, e)| (e.number, self.entries[id].used))
                .collect();
            let mut st = lock(&m.status);
            st.states = entries
                .iter()
                .map(|(id, e)| (e.number, e.tokens.len(), self.entries[id].used))
                .collect();
            st.tree = self
                .cache
                .tree_for(model, 24, 40)
                .into_iter()
                .map(|x| {
                    let used = *by_number.get(&x.number).unwrap_or(&now);
                    (x, used)
                })
                .collect();
            st.kv_bytes = self.cache.bytes_for(model);
            st.kv_allocated = entries
                .iter()
                .map(|(id, e)| {
                    self.entries[id].owner.state_bytes(self.entries[id].raw, 0)
                        + e.snaps.iter().map(|x| x.bytes).sum::<u64>()
                })
                .sum();
        }
        *lock(&s.disk) = match &self.disk {
            Some(store) => DiskStatus {
                dir: Some(store.dir().display().to_string()),
                states: self.cache.saved.len(),
                bytes: self.cache.saved_bytes(),
                limit: self.cache.disk_limit,
                loads: store.loads,
                saves: store.saves,
                written: store.written,
            },
            None => DiskStatus::default(),
        };
        self.published = now;
    }
}

struct Worker {
    s: Arc<Shared>,
    slots: Vec<Slot>,
    memory: Memory,
    maintained: Instant,
    published: Instant,
}

impl Worker {
    fn new(s: Arc<Shared>, engines: Vec<Rc<Model>>, kv_dir: Option<PathBuf>) -> Worker {
        let slots: Vec<Slot> = s
            .models
            .iter()
            .zip(engines)
            .enumerate()
            .map(|(index, (m, model))| {
                let served = m.served.clone();
                Slot {
                    index,
                    ctx: served.engine.ctx as usize,
                    batch: served.batch.max(1),
                    gens: Vec::new(),
                    pending: VecDeque::new(),
                    think_start: model.tokenizer().special("<think>"),
                    think_end: model.tokenizer().special("</think>"),
                    shown: 0,
                    per_row: model.stats().experts_per_row.max(0) as usize,
                    used: Instant::now(),
                    flights: Vec::new(),
                    prefill_stream: None,
                    decode_streams: Vec::new(),
                    model,
                    served,
                }
            })
            .collect();
        let mut memory = Memory::new((s.budgets.kv_gb * GIB) as u64);
        if let Some(dir) = kv_dir {
            Self::open_store(&mut memory, &slots, &s.budgets, &dir);
        }
        let mut w = Worker {
            s,
            slots,
            memory,
            maintained: Instant::now(),
            published: Instant::now(),
        };
        w.memory.publish(&w.s);
        w
    }

    /// The saved states in `dir`: those whose identity is a model's here
    /// can be read for its prompts; the others stay until room is needed.
    fn open_store(memory: &mut Memory, slots: &[Slot], budgets: &crate::config::Budgets, dir: &std::path::Path) {
        let (store, found) = match Store::open(dir) {
            Ok(x) => x,
            Err(m) => {
                engine::log(&format!("kv: no saved states: {m}"));
                return;
            }
        };
        let identities: Vec<[u8; 32]> = slots.iter().map(|x| x.model.state_identity()).collect();
        memory.cache.disk_limit = (budgets.kv_disk_gb * GIB) as u64;
        memory.cache.disk_tokens = budgets.kv_disk_tokens;
        let (mut usable, mut tokens) = (0, 0);
        for mut f in found {
            f.saved.model = identities.iter().position(|i| *i == f.identity);
            if f.saved.model.is_some() {
                usable += 1;
                tokens += f.saved.tokens.len();
            }
            memory.cache.insert_saved(f.id, f.saved);
        }
        engine::log(&format!(
            "kv: saved states in {}: {} ({:.1} GiB of {:.1} GiB); {usable} of them, {tokens} tokens, for the models here",
            dir.display(),
            memory.cache.saved.len(),
            memory.cache.saved_bytes() as f64 / GIB,
            budgets.kv_disk_gb
        ));
        memory.disk = Some(store);
        // a smaller budget than the states take: the oldest go now
        if let (Some(room), Some(store)) = (memory.cache.disk_room(0, None, None), memory.disk.as_mut()) {
            for id in room {
                store.forget(&mut memory.cache, id);
            }
        }
    }

    /// At quit, after the requests ended: every state not cut short is
    /// saved as the cache decides, the most recently used first. A second
    /// stop request ends the saving.
    fn save_at_quit(&mut self) {
        if self.memory.disk.is_none() {
            return;
        }
        let mut order: Vec<(u64, kv::Id)> = self.memory.cache.entries.iter().map(|(&id, e)| (e.used, id)).collect();
        order.sort_unstable_by(|a, b| b.cmp(a));
        let (count, t) = (order.len(), Instant::now());
        for (done, (_, id)) in order.into_iter().enumerate() {
            if engine::stop_requests() >= 2 {
                engine::log(&format!("kv: a second stop: {} states not saved", count - done));
                break;
            }
            // out of the cache: the states not saved yet do not hold its tokens for it
            let Some(e) = self.memory.cache.entries.remove(&id) else {
                continue;
            };
            self.memory.save(id, &e, false);
        }
        if count > 0 {
            engine::log(&format!("kv: {count} states considered at quit in {:.1} s", t.elapsed().as_secs_f64()));
        }
    }

    /// A model that has gone its idle_unload seconds without requests, with
    /// none waiting, gives back its expert cache and weights; its KV stays,
    /// and its next request loads it again. Not while a yield holds memory.
    fn unload_idle(&mut self) {
        let waiting: Vec<bool> = {
            let q = lock(&self.s.queue);
            if q.control.phase != Phase::Running {
                return;
            }
            q.jobs.iter().map(|j| !j.is_empty()).collect()
        };
        let mut drained = false;
        for slot in &self.slots {
            let after = slot.served.idle_unload;
            if after <= 0.0
                || !slot.gens.is_empty()
                || !slot.pending.is_empty()
                || waiting.get(slot.index).copied().unwrap_or(false)
                || slot.used.elapsed().as_secs_f64() < after
                || !slot.model.loaded()
            {
                continue;
            }
            if !drained {
                engine::drain(); // nothing in flight still reads what goes
                drained = true;
            }
            let freed = slot.model.unload();
            if freed > 0 {
                engine::log(&format!(
                    "{}: unloaded after {after:.0} s without requests; {:.1} GiB back",
                    slot.served.id,
                    freed as f64 / GIB
                ));
            }
        }
    }

    /// Apply one RAM reserve between engine calls, including while only a
    /// dense model is active. Release expert slots first, then idle KV.
    fn maintain(&mut self) {
        self.unload_idle();
        engine::refresh_stats();
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
            let deficit = deficit.saturating_sub(engine::reclaim(deficit));
            if deficit > 0 {
                let (snaps, states) = self.memory.cache.reclaim(deficit);
                self.memory.evicted((snaps, states));
                self.memory.publish(&self.s);
            }
        }
        self.publish_counters();
        self.maintained = Instant::now();
    }

    fn publish_counters(&self) {
        for slot in &self.slots {
            let mut status = lock(&self.s.models[slot.index].status);
            status.engine = slot.model.stats();
            status.loaded = slot.model.loaded();
            status.drafts = slot.model.draft_block();
        }
        let used = engine::memory();
        lock(&self.s.queue).control.retained = control::Limits {
            model_bytes: used.model_bytes,
            kv_bytes: used.kv_bytes,
        };
    }

    fn resume_models(&mut self) {
        for slot in &mut self.slots {
            if slot.gens.is_empty() {
                continue;
            }
            if !slot.model.loaded() {
                if let Err(error) = slot.model.load() {
                    if engine::stopping() {
                        continue; // quitting stopped the load: the stop ends its requests
                    }
                    lock(&self.s.models[slot.index].status).load_error = Some(error.clone());
                    for generation in std::mem::take(&mut slot.gens) {
                        let snapshots = self.memory.cache.remove(generation.state);
                        self.memory
                            .free(Vec::new(), vec![(generation.state, snapshots)]);
                        generation.job.last(Event::Failed(error.clone()));
                    }
                    continue;
                }
            }
            slot.model.active(true);
        }
    }

    /// Runs every call in flight, of every model, for a short while: until one
    /// finishes (its request then goes on) or the time slice ends, so that new
    /// requests and the control interface are served meanwhile.
    fn advance_flights(&mut self) {
        let mut streams = Vec::new();
        let mut owners = Vec::new();
        for (slot, x) in self.slots.iter().enumerate() {
            for (flight, f) in x.flights.iter().enumerate() {
                streams.push(f.stream.raw());
                owners.push((slot, flight));
            }
        }
        if streams.is_empty() {
            return;
        }
        if let Some(i) = engine::run(&streams, RUN_SLICE_SECONDS) {
            let (slot, flight) = owners[i];
            let s = self.s.clone();
            let x = &mut self.slots[slot];
            let done = x.flights.remove(flight);
            x.finish_flight(done, &s, &mut self.memory);
        }
    }

    /// Lets every call in flight end, of every model.
    fn drain_flights(&mut self) {
        while self.slots.iter().any(|x| !x.flights.is_empty()) {
            self.advance_flights();
        }
    }

    fn control_action(&mut self, action: Action) {
        self.drain_flights();
        let configured = lock(&self.s.queue).control.configured;
        engine::drain();
        match action {
            Action::Yield(targets) => {
                for slot in &self.slots {
                    slot.model.active(false);
                }
                let before = engine::memory();
                engine::reclaim(match targets.model {
                    Target::Minimum => u64::MAX,
                    Target::Bytes(cap) => before.model_bytes.saturating_sub(cap),
                });
                let (snaps, states) = self.memory.cache.reclaim(match targets.kv {
                    Target::Minimum => u64::MAX,
                    Target::Bytes(cap) => self.memory.cache.bytes().saturating_sub(cap),
                });
                self.memory.evicted((snaps, states));
                engine::drain();
                let used = engine::memory();
                let retained = control::Limits {
                    model_bytes: used.model_bytes,
                    kv_bytes: used.kv_bytes,
                };
                let protected_kv = used.kv_bytes.max(self.memory.cache.bytes());
                let limits = control::Limits {
                    model_bytes: match targets.model {
                        Target::Minimum => used.model_bytes,
                        Target::Bytes(cap) => cap,
                    },
                    kv_bytes: match targets.kv {
                        Target::Minimum => protected_kv,
                        Target::Bytes(cap) => cap,
                    },
                };
                if used.model_bytes > limits.model_bytes || protected_kv > limits.kv_bytes {
                    self.resume_models();
                    let used = engine::memory();
                    lock(&self.s.queue).control.failed(format!(
                        "limits cannot retain protected buffers: {} model bytes and {} KV bytes are required",
                        retained.model_bytes, protected_kv), control::Limits { model_bytes: used.model_bytes, kv_bytes: used.kv_bytes });
                } else {
                    engine::memory_limit(limits.model_bytes);
                    self.memory.cache.limit = limits.kv_bytes;
                    lock(&self.s.queue).control.yielded(limits, retained);
                    engine::log(&format!(
                        "yielded: {} model bytes and {} KV bytes retained",
                        retained.model_bytes, retained.kv_bytes
                    ));
                }
            }
            Action::Resume => {
                engine::memory_limit(configured.model_bytes);
                self.memory.cache.limit = configured.kv_bytes;
                lock(&self.s.queue).control.effective = configured;
                self.resume_models();
                let used = engine::memory();
                lock(&self.s.queue).control.resumed(control::Limits {
                    model_bytes: used.model_bytes,
                    kv_bytes: used.kv_bytes,
                });
                engine::log("resumed: configured model and KV limits restored");
            }
        }
        engine::refresh_stats();
        self.memory.publish(&self.s);
        self.publish_counters();
    }

    /// Every model takes one step a turn: admitting, one prefill call, one
    /// decoding step. Shared memory maintenance runs between model calls
    /// about once a second. On the stop, every request ends with
    /// Finish::Shutdown.
    fn run(&mut self) {
        let s = self.s.clone();
        while !engine::stopping() {
            if self.maintained.elapsed() >= Duration::from_secs(1) {
                self.maintain();
            }
            let mut q = lock(&s.queue);
            if let Some(action) = q.control.next_action() {
                drop(q);
                self.control_action(action);
                continue;
            }
            if q.control.phase != Phase::Running {
                let (guard, _) = s
                    .wake
                    .wait_timeout(q, Duration::from_secs(1))
                    .unwrap_or_else(|e| e.into_inner());
                drop(guard);
                continue;
            }
            let idle = |q: &Queue, slots: &[Slot]| {
                q.jobs.iter().all(|j| j.is_empty())
                    && q.text.is_none()
                    && q.control.phase == Phase::Running
                    && slots.iter().all(|x| x.gens.is_empty() && x.pending.is_empty())
            };
            while idle(&q, &self.slots) && !engine::stopping() {
                let (guard, timeout) = s
                    .wake
                    .wait_timeout(q, Duration::from_secs(1))
                    .unwrap_or_else(|e| e.into_inner());
                q = guard;
                if timeout.timed_out() && q.jobs.iter().all(|j| j.is_empty()) {
                    drop(q);
                    self.maintain();
                    q = lock(&s.queue);
                }
            }
            if q.control.phase != Phase::Running {
                continue;
            }
            let question = q.text.take();
            let mut admitted: Vec<Vec<Job>> = Vec::new();
            for x in &self.slots {
                let mut admit = Vec::new();
                while x.gens.len() + x.pending.len() + admit.len() < x.batch && !engine::stopping() {
                    let Some(job) = q.jobs[x.index].pop_front() else {
                        break;
                    };
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
                if !x.gens.is_empty() {
                    x.used = Instant::now();
                }
                x.model.active(!x.gens.is_empty());
            }
            self.advance_flights();
            if self.published.elapsed() >= Duration::from_millis(250) {
                engine::refresh_stats();
                self.publish_counters();
                self.published = Instant::now();
            }
        }
        self.drain_flights();
        for x in &mut self.slots {
            for g in std::mem::take(&mut x.gens) {
                x.finish(&s, g, true, &mut self.memory);
            }
        }
        let mut waiting: Vec<Job> = {
            let mut q = lock(&s.queue);
            q.closed = true;
            q.jobs.iter_mut().flat_map(|j| j.drain(..)).collect()
        };
        for x in &mut self.slots {
            waiting.extend(x.pending.drain(..));
        }
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
        if finish == Finish::Cancelled {
            "cancelled"
        } else {
            "shutdown"
        },
        job.queued.elapsed().as_secs_f64()
    ));
    job.last(Event::Done { finish });
}

impl Slot {
    /// The model's limits for its cache's plans.
    fn plans(&self, prompt: &[i32], memory: &Memory) -> Vec<Plan> {
        let model = self.model.clone();
        let cost = move |k: usize| model.replay_rows(k).unwrap_or(0);
        let limits = Limits {
            rewind: self.model.rewind(),
            replay: self.model.replay_rows(1).is_some(),
            replay_cost: &cost,
        };
        memory.cache.plans_for(self.index, prompt, limits)
    }

    /// What a prompt of `n` tokens started by `plan` adds to the KV cache: the
    /// state at the prompt's end and its first token, a saved state's snapshot
    /// read before it is restored (with its logits, the prompt-end snapshot of
    /// the same prompt again), and the snapshots the plan takes.
    fn plan_need(&self, plan: &Plan, n: usize, memory: &Memory) -> u64 {
        let state = match *plan {
            Plan::InPlace { state, .. } => self
                .model
                .state_bytes(memory.entries[&state].raw, n + 1)
                .saturating_sub(memory.cache.entries[&state].bytes),
            _ => self.model.state_bytes(std::ptr::null(), n + 1),
        };
        let loaded = match plan {
            Plan::Load { snap: Some(_), .. } => self.model.snapshot_bytes(true),
            _ => 0,
        };
        state
            + loaded
            + plan.snapshot_need(
                n,
                self.model.snapshot_bytes(false),
                self.model.snapshot_bytes(true),
            )
    }

    /// One step of the model: its admitted requests start, one prefill call
    /// runs, the generations that are done end, the others advance a token.
    fn step(&mut self, s: &Shared, admit: Vec<Job>, memory: &mut Memory) {
        self.pending.extend(admit);
        // A request plans which state its prefill resumes from when the
        // prefills before it have ended, so that it can resume from theirs:
        // one request at a time starts its prefill.
        while !self.gens.iter().any(|g| !g.decoding()) {
            let Some(job) = self.pending.pop_front() else {
                break;
            };
            if engine::stopping() {
                end_waiting(job, Finish::Shutdown);
            } else if job.flags.cancelled.load(Ordering::SeqCst) {
                engine::log(&format!("req {}: client gone while waiting", job.id));
                end_waiting(job, Finish::Cancelled);
            } else {
                {
                    if !self.model.loaded() {
                        self.set_status(s, "loading");
                    }
                    // A call in flight keeps the model's memory; its load must
                    // not run under the call, which it would resize buffers for.
                    if self.flights.is_empty()
                        && let Err(error) = self.model.load()
                    {
                        if engine::stopping() {
                            end_waiting(job, Finish::Shutdown); // quitting stopped the load
                            continue;
                        }
                        lock(&s.models[self.index].status).load_error = Some(error.clone());
                        engine::log(&format!("{}: {error}", self.served.id));
                        job.last(Event::Failed(error));
                        continue;
                    }
                    lock(&s.models[self.index].status).load_error = None;
                    self.used = Instant::now();
                    self.per_row = self.model.stats().experts_per_row.max(0) as usize;
                    engine::log(&format!(
                        "{}: loaded; shares the {:.1} GiB KV budget",
                        self.served.id, s.budgets.kv_gb
                    ));
                }
                self.waiting_start(s, job, memory);
            }
        }
        if engine::stopping() {
            return;
        }
        self.prefill_start(memory);
        let ctx = self.ctx;
        let mut i = 0;
        while i < self.gens.len() {
            if !self.gens[i].flying && self.gens[i].decoding() && Self::over(&self.model, &mut self.gens[i], ctx) {
                let g = self.gens.swap_remove(i);
                self.finish(s, g, false, memory);
            } else if !self.gens[i].flying
                && !self.gens[i].decoding()
                && self.gens[i].job.flags.cancelled.load(Ordering::SeqCst)
            {
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
            engine::log(&format!(
                "{}: {decoding} requests decoding together",
                self.served.id
            ));
        }
        self.shown = decoding;
        if self.gens.is_empty() {
            self.prefill_stream = None;
            self.decode_streams.clear();
            self.set_status(s, "idle");
            return;
        }
        if decoding > 0 {
            self.decode_start(memory);
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
        let phase = if self.gens.iter().any(|g| !g.decoding()) {
            "prefill"
        } else {
            "decode"
        };
        self.set_status(s, phase);
        if memory.published.elapsed() >= Duration::from_secs(1) {
            memory.publish(s);
        }
    }

    /// The tokens of state `session`'s positions start..end, for the live
    /// view: at most TEXT_TOKENS from start, none if the state is gone.
    fn node_text(&self, session: u64, start: usize, end: usize, memory: &Memory) -> NodeText {
        let tokens = memory
            .cache
            .entries
            .values()
            .find(|e| e.model == self.index && e.number == session)
            .map_or(&[][..], |e| &e.tokens[..]);
        let to = end.min(tokens.len());
        let from = start.min(to);
        NodeText {
            model: self.index,
            session,
            start,
            end,
            ids: tokens[from..to.min(from + TEXT_TOKENS)].to_vec(),
        }
    }

    /// Before a pass that makes the busy state `id` hold `positions`: the
    /// cache charges its growth before the engine grows it (kv::Cache::grow),
    /// and what the cache gives up for it is freed.
    /// A call starts beside the passes in flight: their times no longer
    /// measure them alone, so the draft-count chooser does not learn from them.
    fn crowd(&mut self) {
        for g in &mut self.gens {
            g.pass = None;
        }
    }

    fn make_room(&mut self, id: kv::Id, positions: usize, memory: &mut Memory) {
        let before = memory.cache.entries[&id].bytes;
        let (snaps, states) = memory.cache.grow(
            id,
            self.model.state_bytes(memory.entries[&id].raw, positions),
        );
        memory.evicted((snaps, states));
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
        if after > before {
            memory.publish(&shared());
        }
    }

    fn set_status(&mut self, s: &Shared, phase: &'static str) {
        let mut st = lock(&s.models[self.index].status);
        if st.phase != phase || st.since.is_none() {
            st.since = Some(Instant::now());
        }
        st.phase = phase;
        st.active = self.gens.iter().filter(|g| g.decoding()).count();
        st.decoded =
            st.generated_tokens + self.gens.iter().map(|g| g.generated as u64).sum::<u64>();
        let first = self
            .gens
            .iter()
            .find(|g| !g.decoding())
            .or(self.gens.first());
        (st.request, st.prompt, st.reused, st.generated) =
            first.map_or((0, 0, 0, 0), |g| (g.job.id, g.n, g.from, g.generated));
        st.live = self
            .gens
            .iter()
            .map(|g| LiveReq {
                number: g.job.id,
                session: g.number,
                prompt: g.n,
                reused: g.from,
                generated: g.generated,
            })
            .collect();
    }

    /// A request's start: its media and prompt checked, the KV prefix cache's
    /// plan carried out, its prefill to come (or, for the same prompt again,
    /// its first token from the prompt-end snapshot). Its plan is made here,
    /// when it is admitted, after the prefills before it.
    fn waiting_start(&mut self, s: &Shared, job: Job, memory: &mut Memory) {
        let mut prompt = job.request.prompt.clone();
        let mut media = None;
        if !job.request.media.is_empty() {
            match self.model.media(&job.request.media) {
                Err(m) => return refuse(job, &m),
                Ok(mut x) => match x.expand(&prompt, self.ctx + 1) {
                    Some(p) => {
                        prompt = p;
                        media = Some(x);
                    }
                    None => {
                        let m = format!(
                            "the prompt's image places and its {} media differ",
                            job.request.media.len()
                        );
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
        // The cheapest plan that fits. Each keeps what it reuses; the last, a
        // cold start, keeps nothing, so that a state that does not fit beside
        // the prompt's own can go.
        let mut need = 0;
        let mut admitted = None;
        for p in self.plans(&prompt, memory) {
            need = self.plan_need(&p, n, memory);
            let keep: Vec<kv::Id> = p.source().into_iter().collect();
            if let Some(ev) = memory.cache.fit(need, !matches!(p, Plan::InPlace { .. }), &keep) {
                admitted = Some((p, ev));
                break;
            }
        }
        let Some((plan, ev)) = admitted else {
            let busy: u64 = memory
                .cache
                .entries
                .values()
                .filter(|e| e.busy)
                .map(|e| e.bytes)
                .sum();
            let m = format!(
                "the conversation's attention state needs {:.1} GiB ({:.1} GiB with its snapshots); the kv cache allows {:.1} GiB, and the requests running take {:.1} GiB of it",
                self.model.state_bytes(std::ptr::null(), n + 1) as f64 / GIB,
                need as f64 / GIB,
                memory.cache.limit as f64 / GIB,
                busy as f64 / GIB
            );
            return refuse(job, &m);
        };
        let origin = memory.cache.origin(&plan, n);
        let capture = plan.capture(n);
        // the saved state the plan reads, as planned: what eviction saves now
        // leaves it as it is
        let planned = match plan {
            Plan::Load { saved, .. } => {
                memory.cache.pinned = Some(saved);
                memory.cache.saved.get(&saved).map(|d| d.version)
            }
            _ => None,
        };
        let wait = job.queued.elapsed().as_secs_f64();
        let (snaps, states) = memory.cache.evict(&ev);
        memory.evicted((snaps, states));
        if let Plan::Load { saved, .. } = plan
            && memory.cache.saved.get(&saved).map(|d| d.version) != planned
        {
            memory.cache.pinned = None;
            return self.waiting_start(s, job, memory); // it changed meanwhile: planned again
        }
        let from = plan.reused();
        let id = match plan {
            Plan::Cold => {
                let id = memory.add(self.index, self.model.clone());
                id
            }
            Plan::InPlace { state, from } => {
                memory.cache.touch(state, from);
                state
            }
            Plan::Snapshot { source, at, .. } | Plan::Replay { source, at } => {
                let id = memory.add(self.index, self.model.clone());
                let st = memory.entries[&id].raw;
                self.model.state_copy(st, memory.entries[&source].raw, at);
                memory.cache.touch(source, at);
                memory.cache.inherit(id, source, at);
                memory.entries.get_mut(&source).expect("state").used = Instant::now();
                if let Plan::Snapshot { snap, .. } = plan {
                    self.model
                        .snapshot_restore(st, memory.snapshot(source, snap));
                }
                id
            }
            Plan::Load { saved, snap, at, .. } => {
                let id = memory.add(self.index, self.model.clone());
                let st = memory.entries[&id].raw;
                let t = Instant::now();
                let store = memory.disk.as_mut().expect("saved states have their store");
                let read = store.load(&mut memory.cache, &Engine { model: &self.model, state: st }, saved, snap, at);
                memory.cache.pinned = None;
                match read {
                    Ok(x) => {
                        memory.cache.link_saved(id, saved, at);
                        if !x.is_null() && at == n {
                            memory.save_snapshot(id, n, x); // the same prompt again: restored below
                        } else if !x.is_null() {
                            self.model.snapshot_restore(st, x);
                            self.model.snapshot_free(x);
                        }
                        engine::log(&format!(
                            "req {}: {at} positions of saved state {saved} read in {:.2} s",
                            job.id,
                            t.elapsed().as_secs_f64()
                        ));
                        id
                    }
                    Err(error) => {
                        // planned again without it: deleted when damaged, or
                        // passed over for this request when it cannot be read now
                        match error {
                            LoadError::Damaged(m) => {
                                engine::log(&format!(
                                    "req {}: saved state {saved} cannot be read ({m}); it is deleted",
                                    job.id
                                ));
                                store.forget(&mut memory.cache, saved);
                            }
                            LoadError::Unavailable(m) => {
                                engine::log(&format!(
                                    "req {}: saved state {saved} cannot be read now ({m}); planned without it",
                                    job.id
                                ));
                                memory.cache.skip = Some(saved);
                            }
                        }
                        let snapshots = memory.cache.remove(id);
                        memory.free(Vec::new(), vec![(id, snapshots)]);
                        self.waiting_start(s, job, memory);
                        memory.cache.skip = None;
                        return;
                    }
                }
            }
        };
        let bytes = self.model.state_bytes(memory.entries[&id].raw, n + 1);
        let dropped = memory
            .cache
            .start(id, &prompt, from, bytes, kv::stale_at(origin, from));
        if let Plan::Load { at, snap: Some(_), .. } = plan
            && at == n
        {
            memory.cache.add_snap(
                id,
                Snap {
                    pos: n,
                    kind: Kind::PromptEnd,
                    bytes: self.model.snapshot_bytes(true),
                    stale: kv::stale_at(origin, n),
                },
            );
        }
        let mut promised = Vec::new();
        if from < n {
            promised.push(Snap {
                pos: n,
                kind: Kind::PromptEnd,
                bytes: self.model.snapshot_bytes(true),
                stale: kv::stale_at(origin, n),
            });
        }
        if let Some(pos) = capture {
            promised.push(Snap {
                pos,
                kind: Kind::Branch,
                bytes: self.model.snapshot_bytes(false),
                stale: kv::stale_at(origin, pos),
            });
        }
        memory.cache.reserve_snapshots(id, promised);
        for x in dropped {
            memory.free_snap(id, x.pos);
        }
        memory.entries.get_mut(&id).expect("state").used = Instant::now();
        memory.publish(s);
        let number = memory.cache.entries[&id].number;
        let kind = plan.kind(n);
        let thinking = self.think_start == Some(prompt[n - 1]);
        engine::log(&format!(
            "req {}: {} {n} prompt tokens, session {number} ({kind}), {from} reused; thinking {}; at most {max_new} new tokens, {}; waited {wait:.1} s",
            job.id,
            self.served.id,
            if thinking { "on" } else { "off" },
            if sm.temperature > 0.0 {
                "sampled"
            } else {
                "greedy"
            }
        ));
        let replay = plan.replays();
        if let Some(x) = media.as_mut() {
            // The media whose spans the prefill reaches: a replay computes again
            // from its first position, before the branch point.
            let first = if replay {
                self.model.replay_start(from)
            } else {
                from
            };
            let t = Instant::now();
            let encoded = match x.encode(first) {
                Ok(count) => count,
                Err(error) => {
                    let snapshots = memory.cache.remove(id);
                    memory.free(Vec::new(), vec![(id, snapshots)]);
                    memory.publish(s);
                    job.last(Event::Failed(error));
                    return;
                }
            };
            if encoded > 0 {
                engine::log(&format!(
                    "req {}: {encoded} of {} media encoded in {:.2} s",
                    job.id,
                    job.request.media.len(),
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
                media,
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
            choice: None,
            pass: None,
            max_new,
            t1: now,
            tick: now,
            c1: CacheCounts::default(),
            flying: false,
            ready: now,
        };
        if from == n {
            // The same prompt again: its prompt-end snapshot holds its logits.
            let source = plan.source().unwrap_or(id); // a saved state's prompt end is the new state's own
            let x = memory.snapshot(source, n);
            if source == id {
                self.model.snapshot_restore(memory.entries[&id].raw, x);
            }
            let prompt = g
                .prefill
                .as_ref()
                .map(|p| p.prompt.clone())
                .unwrap_or_default();
            self.model.seek(memory.entries[&id].raw, &prompt, n);
            let lg = self.model.snapshot_logits(x);
            self.started(s, &mut g, lg, PrefillInfo::default(), memory);
        }
        self.gens.push(g);
    }

    /// The index in `gens` of a request whose call just ended.
    fn index_of(&self, id: u64) -> usize {
        self.gens
            .iter()
            .position(|g| g.job.id == id)
            .expect("a running call's request")
    }

    /// Starts one prefill call of the oldest request still prefilling, if no
    /// prefill call runs: the rest of its prompt for a model that takes it in
    /// one call, otherwise up to its prefill rows, cut at the capture point
    /// and, for a model that cannot replay, at the snapshot grid.
    fn prefill_start(&mut self, memory: &mut Memory) {
        if self.flights.iter().any(|f| matches!(f.call, Call::Prefill { .. })) {
            return;
        }
        if self.one_call_at_a_time() && !self.flights.is_empty() {
            return;
        }
        let Some(i) = self.gens.iter().position(|g| !g.decoding()) else {
            return;
        };
        let chunked = self.model.prefill_rows();
        let can_replay = self.model.replay_rows(1).is_some();
        let (state, rows) = (self.gens[i].state, self.gens[i].n);
        let st = memory.entries[&state].raw;
        let p = self.gens[i].prefill.as_ref().expect("a prefill");
        let (done, capture) = (p.done, p.capture);
        let end = match chunked {
            None => rows,
            Some(most) => {
                let mut end = (done + most).min(rows);
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
        let stream = self
            .prefill_stream
            .take()
            .unwrap_or_else(|| Stream::new(&self.model));
        stream.prefill(st, &p.prompt[..end], done, replay, inner, p.media.as_ref());
        let id = self.gens[i].job.id;
        self.gens[i].flying = true;
        self.crowd();
        self.flights.push(Flight {
            stream,
            call: Call::Prefill { id, end, inner },
        });
    }

    /// A prefill call has ended: the request's snapshots are saved, and at its
    /// prompt's end its first token is sampled.
    fn prefill_finish(
        &mut self,
        stream: Stream,
        id: u64,
        end: usize,
        inner: Option<usize>,
        s: &Shared,
        memory: &mut Memory,
    ) {
        let i = self.index_of(id);
        self.gens[i].flying = false;
        let chunked = self.model.prefill_rows();
        let can_replay = self.model.replay_rows(1).is_some();
        let (state, rows) = (self.gens[i].state, self.gens[i].n);
        let st = memory.entries[&state].raw;
        let Some(lg) = stream.logits() else {
            if engine::stopping() {
                return; // quitting stopped the prefill: the stop ends the request
            }
            let generation = self.gens.remove(i);
            let snapshots = memory.cache.remove(generation.state);
            memory.free(Vec::new(), vec![(generation.state, snapshots)]);
            memory.publish(s);
            generation.job.last(Event::Failed(
                "the prefill's working memory does not fit: memory is short".into(),
            ));
            return;
        };
        let (captured, info) = (stream.captured(), stream.info());
        let p = self.gens[i].prefill.as_ref().expect("a prefill");
        let (origin, capture) = (p.origin, p.capture);
        memory
            .cache
            .prefilled(state, &p.prompt[..end], kv::stale_at(origin, end));
        if let Some(at) = inner
            && !captured.is_null()
        {
            memory.save_snapshot(state, at, captured);
            let stale = kv::stale_at(origin, at);
            memory.cache.add_snap(
                state,
                Snap {
                    pos: at,
                    kind: Kind::Branch,
                    bytes: self.model.snapshot_bytes(false),
                    stale,
                },
            );
        }
        let last = end == rows;
        if chunked.is_some() && !last {
            let grid =
                !can_replay && end % SNAP_GRID == 0 && memory.cache.snap(state, end).is_none();
            if capture == Some(end) || grid {
                let bytes = self.model.snapshot_bytes(false);
                let fits = capture == Some(end)
                    || memory.cache.fit(bytes, false, &[state]).is_some_and(|ev| {
                        let (snaps, states) = memory.cache.evict(&ev);
                        memory.evicted((snaps, states));
                        true
                    });
                if fits {
                    let x = self.model.snapshot_take(st, end, std::ptr::null());
                    memory.save_snapshot(state, end, x);
                    let stale = kv::stale_at(origin, end);
                    memory.cache.add_snap(
                        state,
                        Snap {
                            pos: end,
                            kind: Kind::Branch,
                            bytes,
                            stale,
                        },
                    );
                }
            }
        }
        if !last {
            self.gens[i].prefill.as_mut().expect("a prefill").done = end;
            self.prefill_stream = Some(stream);
            return;
        }
        let x = self.model.snapshot_take(st, rows, lg);
        memory.save_snapshot(state, rows, x);
        let stale = kv::stale_at(origin, rows);
        memory.cache.add_snap(
            state,
            Snap {
                pos: rows,
                kind: Kind::PromptEnd,
                bytes: self.model.snapshot_bytes(true),
                stale,
            },
        );
        let mut g = self.gens.remove(i);
        self.started(s, &mut g, lg, info, memory);
        self.gens.insert(i, g);
        // `lg` is the stream's until here; the prompt's large buffers go with it.
        drop(stream);
    }

    /// A call has ended: the requests it served go on.
    fn finish_flight(&mut self, flight: Flight, s: &Shared, memory: &mut Memory) {
        let Flight { stream, call } = flight;
        match call {
            Call::Prefill { id, end, inner } => self.prefill_finish(stream, id, end, inner, s, memory),
            Call::Rows { ids, tokens } => self.rows_finish(stream, ids, tokens, s, memory),
            Call::Verify { id, rows, proposal } => self.verify_finish(stream, id, rows, proposal, s, memory),
        }
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
    fn started(
        &mut self,
        s: &Shared,
        g: &mut Gen,
        lg: Logits,
        info: PrefillInfo,
        memory: &mut Memory,
    ) {
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
                    if self.served.engine.deepseek().is_some_and(|d| d.ced > 0) {
                        "a bounded"
                    } else {
                        "an exact"
                    },
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
                g.job.id,
                info.chunks,
                info.tokens,
                info.decoder_tokens,
                info.ced_applied,
                info.prune_applied,
                info.prefix_tokens
            ));
        }
        let _ = g.job.tx.send(Event::Started {
            prompt: n,
            reused: from,
        });
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
                "loaded" => 4,
                _ => 3,
            }] += 1;
        }
        drop(p.media);
        g.t = self.model.sample(lg, &mut g.sampler);
        if self.served.engine.lookup > 0 {
            let mut l =
                Lookup::new(self.ctx).unwrap_or_else(|| panic!("cannot initialize context lookup"));
            if !l.append(&p.prompt) {
                panic!("cannot initialize context lookup");
            }
            g.lookup = Some(l);
        }
        if self.model.draft_block() > 0 {
            g.choice = Some(DraftChoice::new());
        }
        g.t1 = t1;
        g.tick = t1;
        g.ready = t1;
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
        if let Some(p) = &g.prefill {
            // not saved: a prefill the stop cut short, and a replay that never
            // ran (its rolling rows were never rebuilt)
            if shutdown || (p.replay && p.done == g.from) {
                memory.cut.insert(g.state);
            }
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
        let (take, (snaps, states)) = memory.cache.end(
            g.state,
            self.model.state_bytes(memory.entries[&g.state].raw, g.pos),
            end,
        );
        if take {
            let x = self
                .model
                .snapshot_take(memory.entries[&g.state].raw, g.pos, std::ptr::null());
            memory.save_snapshot(g.state, g.pos, x);
        }
        memory.evicted((snaps, states));
        memory.entries.get_mut(&g.state).expect("state").used = Instant::now();
        let st = self.model.stats();
        {
            let mut status = lock(&s.models[self.index].status);
            status.engine = st;
            status.loaded = self.model.loaded();
            status.drafts = self.model.draft_block();
        }
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

    /// A decoding pass that failed because experts it needed could not load:
    /// quitting stopped their reads, or memory stayed short after idle memory
    /// went back. While quitting, its requests end as every request does at
    /// the stop. Otherwise they end with an error, and their states, which the
    /// pass may have changed in part, are discarded.
    fn pass_failed(&mut self, failed: &[usize], memory: &mut Memory) {
        if engine::stopping() {
            for &i in failed {
                memory.cut.insert(self.gens[i].state); // the pass may have changed it in part: not saved
            }
            return;
        }
        let mut failed = failed.to_vec();
        failed.sort_unstable_by(|a, b| b.cmp(a));
        for i in failed {
            let generation = self.gens.remove(i);
            let snapshots = memory.cache.remove(generation.state);
            memory.free(Vec::new(), vec![(generation.state, snapshots)]);
            engine::log(&format!("req {}: a pass could not load its experts: memory is short", generation.job.id));
            generation.job.last(Event::Failed(
                "a pass could not load the model's experts: memory is short".into(),
            ));
        }
    }

    /// A stream for a decoding call: one that ended a call, or a new one.
    fn decode_stream(&mut self) -> Stream {
        self.decode_streams
            .pop()
            .unwrap_or_else(|| Stream::new(&self.model))
    }

    /// The DeepSeek runahead and pipeline decoders keep their jobs in
    /// model-wide state: a model with one of them runs one call at a time.
    fn one_call_at_a_time(&self) -> bool {
        self.served.engine.deepseek().is_some_and(|d| d.runahead || d.pipeline)
    }

    /// Starts one token for each decoding request with no call running: alone
    /// with its drafts or proposals, or together, each row as it would run
    /// alone, at most the model's rows per call.
    fn decode_start(&mut self, memory: &mut Memory) {
        // Rows of one call share the GPU's work per layer, so calls stay as
        // full as the requests allow: at most as many run at once as the
        // decoding requests fill, and a request that finds no room waits for
        // the next call. A call starts as soon as requests are free and there
        // is room for it: groups of requests run independently of each other,
        // and each call's waits are filled by the others' GPU work.
        let most = self.model.max_rows().max(1);
        let total = self.gens.iter().filter(|g| g.decoding()).count();
        let serial = self.one_call_at_a_time();
        if serial && !self.flights.is_empty() {
            return;
        }
        let allowed = calls_at_once(total, most, serial);
        let mut running = self
            .flights
            .iter()
            .filter(|f| !matches!(f.call, Call::Prefill { .. }))
            .count();
        if running >= allowed {
            return;
        }
        let ctx = self.ctx;
        let mut decoding: Vec<usize> = (0..self.gens.len())
            .filter(|&i| self.gens[i].decoding() && !self.gens[i].flying)
            .collect();
        if decoding.is_empty() {
            return;
        }
        decoding.sort_by_key(|&i| self.gens[i].ready);
        // Requests that have just finished their prefill wait for the short
        // prefills still to come: their first calls then start full, and they
        // do not take SSD reads and cache room from the prefills. A long
        // prefill keeps no one waiting, and neither does any wait past HOLD_TIME.
        let prefilling: usize = self
            .gens
            .iter()
            .filter(|g| !g.decoding())
            .map(|g| g.n - g.prefill.as_ref().map_or(g.n, |p| p.done))
            .sum::<usize>()
            + self.pending.iter().map(|j| j.request.prompt.len()).sum::<usize>();
        let now = Instant::now();
        decoding.retain(|&i| !held_for_prefills(self.gens[i].generated, now.duration_since(self.gens[i].ready), prefilling));
        if decoding.is_empty() {
            return;
        }
        if total == 1 {
            let gi = decoding[0];
            let state = self.gens[gi].state;
            let st = memory.entries[&state].raw;
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
                    let most = self.model.draft_block().min(room.max(0) as usize);
                    match g.choice.as_mut() {
                        Some(c) if most > 0 => c.count(most),
                        _ => most,
                    }
                };
                (rows, proposal, k, g.pos)
            };
            // The rows the pass writes: the token and its drafts or proposals.
            self.make_room(state, pos + 1 + k, memory);
            let stream = self.decode_stream();
            self.crowd();
            let alone = self.flights.is_empty();
            let g = &mut self.gens[gi];
            g.flying = true;
            if alone && proposal == 0 && g.choice.is_some() {
                g.pass = Some((Instant::now(), k));
            }
            let id = g.job.id;
            let call = if k > 0 {
                if proposal == 0 {
                    rows.resize(k + 1, 0);
                }
                stream.verify(st, &mut rows, proposal > 0, g.pos, &mut g.sampler);
                Call::Verify { id, rows, proposal }
            } else {
                stream.rows(&mut [st], &[g.t], &[g.pos as i32]);
                Call::Rows { ids: vec![id], tokens: vec![g.t] }
            };
            self.flights.push(Flight { stream, call });
            return;
        }
        for group in decoding.chunks(most) {
            if running >= allowed {
                break;
            }
            running += 1;
            for &i in group {
                let (id, positions) = (self.gens[i].state, self.gens[i].pos + 1);
                self.make_room(id, positions, memory);
            }
            let mut st: Vec<*mut RawState> = group
                .iter()
                .map(|&i| memory.entries[&self.gens[i].state].raw)
                .collect();
            let tokens: Vec<i32> = group.iter().map(|&i| self.gens[i].t).collect();
            let pos: Vec<i32> = group.iter().map(|&i| self.gens[i].pos as i32).collect();
            let ids: Vec<u64> = group.iter().map(|&i| self.gens[i].job.id).collect();
            let stream = self.decode_stream();
            stream.rows(&mut st, &tokens, &pos);
            for &i in group {
                self.gens[i].flying = true;
            }
            self.crowd();
            self.flights.push(Flight {
                stream,
                call: Call::Rows { ids, tokens },
            });
        }
    }

    /// A call of one token per request has ended: the requests emit their
    /// tokens and sample their next ones.
    fn rows_finish(&mut self, stream: Stream, ids: Vec<u64>, tokens: Vec<i32>, s: &Shared, memory: &mut Memory) {
        let indices: Vec<usize> = ids.iter().map(|&id| self.index_of(id)).collect();
        let ended = Instant::now();
        let mut passes = Vec::new();  // a request's pass without drafts, chosen while drafting
        for &i in &indices {
            self.gens[i].flying = false;
            self.gens[i].ready = ended;
            if let Some(pass) = self.gens[i].pass.take() {
                passes.push((i, pass));
            }
        }
        let Some(lg) = stream.logits() else {
            self.decode_streams.push(stream);
            self.pass_failed(&indices, memory);
            return;
        };
        for (i, (started, count)) in passes {
            if let Some(c) = self.gens[i].choice.as_mut() {
                c.feedback(count, 0, ended.duration_since(started).as_secs_f64());
            }
        }
        let vocab = self.model.vocab();
        let model = self.model.clone();
        let think_end = self.think_end;
        for (r, &i) in indices.iter().enumerate() {
            let g = &mut self.gens[i];
            Self::emit(&model, &mut memory.cache, g, tokens[r], think_end);
            // Safety: myds4_rows returns one row of the vocabulary for each state.
            g.t = model.sample(unsafe { lg.add(r * vocab) }, &mut g.sampler);
        }
        self.decode_streams.push(stream);
        // Calls that ran beside each other wrote the same routes: the live
        // view shows routes only while one decoding call runs.
        let one_call = self.gens.iter().filter(|g| g.decoding()).count() <= self.model.max_rows().max(1);
        if self.per_row > 0 && one_call && ROUTES.swap(false, Ordering::Relaxed) {
            self.copy_routes(s, &ids);
        }
    }

    /// A token and its drafts or proposals have run: the kept ones go out.
    fn verify_finish(&mut self, stream: Stream, id: u64, rows: Vec<i32>, proposal: usize, s: &Shared, memory: &mut Memory) {
        let gi = self.index_of(id);
        self.gens[gi].flying = false;
        self.gens[gi].ready = Instant::now();
        let pass = self.gens[gi].pass.take();
        if stream.logits().is_none() {
            self.decode_streams.push(stream);
            self.pass_failed(&[gi], memory);
            return;
        }
        let (next, accepted) = stream.next();
        self.decode_streams.push(stream);
        let model = self.model.clone();
        let think_end = self.think_end;
        let g = &mut self.gens[gi];
        g.draft_passes += 1;
        g.draft_accepted += accepted;
        if let (Some((started, count)), Some(c)) = (pass, g.choice.as_mut()) {
            c.feedback(count, accepted, started.elapsed().as_secs_f64());
        }
        if proposal > 0
            && let Some(l) = g.lookup.as_mut()
        {
            l.feedback(rows.len() - 1, accepted);
            g.lookup_accepted += accepted;
        }
        let mut kept = 0;
        for &t in &rows[..=accepted] {
            if g.ended {
                break;
            }
            Self::emit(&model, &mut memory.cache, g, t, think_end);
            kept += 1;
        }
        // the pass's rows after the kept tokens took places in the rings
        memory.cache.rejected(g.state, rows.len() - kept);
        g.t = next;
        if self.per_row > 0 && ROUTES.swap(false, Ordering::Relaxed) {
            self.copy_routes(s, &[id]);
        }
    }

    /// The routed rows of the call that just ended, for the live view: row r
    /// is the r-th request of the call (`ids`).
    fn copy_routes(&self, s: &Shared, ids: &[u64]) {
        let routes: Vec<Route> = ids
            .iter()
            .enumerate()
            .filter_map(|(r, &request)| {
                self.model
                    .routes(r, self.per_row)
                    .map(|(experts, sources)| Route {
                        request,
                        experts,
                        sources,
                    })
            })
            .collect();
        lock(&s.models[self.index].status).routes = routes;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decoding_calls_run_as_many_as_their_requests_fill() {
        assert_eq!(calls_at_once(12, 6, false), 2);
        assert_eq!(calls_at_once(13, 6, false), 2);
        assert_eq!(calls_at_once(5, 6, false), 1);
        // a decoder that keeps its jobs in model-wide state runs one call at a time
        assert_eq!(calls_at_once(12, 6, true), 1);
    }

    #[test]
    fn a_new_request_waits_only_for_short_prefills_and_not_for_long() {
        let moment = Duration::from_secs(1);
        assert!(held_for_prefills(0, moment, 100));
        assert!(!held_for_prefills(3, moment, 100)); // it has decoded before
        assert!(!held_for_prefills(0, moment, 0)); // no prefill to wait for
        assert!(!held_for_prefills(0, moment, HOLD_PREFILL_ROWS + 1)); // a long prefill keeps no one waiting
        assert!(!held_for_prefills(0, HOLD_TIME + moment, 100)); // the wait has an end
    }
}
