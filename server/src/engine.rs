//! The engine's C ABI (src/engine/engine.h) and a safe interface over it: a request
//! becomes a channel of events, and dropping a [`Cancel`] cancels it.

use std::ffi::{CStr, CString, c_char, c_int, c_long, c_void};
use std::ptr;

use deepseek_recipe::stream::StreamTokenizer;
use tokio::sync::mpsc;

const ABI: c_int = 5;
const STARTED: c_int = 0;
const TOKENS: c_int = 1;
const DONE: c_int = 2;
const QUEUED: c_int = 0;
const BUSY: c_int = 1;

#[repr(C)]
struct RawConfig {
    abi: c_int,
    model: *const c_char,
    experts: *const c_char,
    ctx: c_int,
    cache_gb: f64,
    reserve_gb: f64,
    cache_half_life: c_int,
    cache_read_cost: c_int,
    substitute: f32,
    gpu_route: c_int,
    draft: c_int,
    draft_cache: c_int,
    draft_trim: c_int,
    online_predictor: c_int,
    predictor_state: *const c_char,
    kv_gb: f64,
    batch: c_int,
    ced: c_int,
    lookup: c_int,
    expert_batch: c_int,
    prefill_prune: c_int,
    on_log: Option<extern "C" fn(*const c_char)>,
    on_die: Option<extern "C" fn()>,
}

#[repr(C)]
struct RawImage {
    data: *const u8,
    len: usize,
}

#[repr(C)]
struct RawEvent {
    kind: c_int,
    id: u64,
    tokens: *const i32,
    ntokens: c_int,
    prompt: c_int,
    reused: c_int,
    generated: c_int,
    reasoning: c_int,
    finish: c_int,
    message: *const c_char,
}

#[repr(C)]
struct RawRequest {
    tokens: *const i32,
    ntokens: c_int,
    text: *const c_char,
    len: usize,
    images: *const RawImage,
    nimages: c_int,
    temperature: f64,
    top_p: f64,
    seed: u64,
    seed_set: c_int,
    max_tokens: c_long,
    on_event: extern "C" fn(*mut c_void, *const RawEvent),
    user: *mut c_void,
    note: *const c_char,
}

unsafe extern "C" {
    fn myds4_start(c: *const RawConfig) -> c_int;
    fn myds4_stopping() -> c_int;
    fn myds4_request_stop();
    fn myds4_stop();
    fn myds4_submit(r: *const RawRequest, id: *mut u64, message: *mut c_char, cap: usize) -> c_int;
    fn myds4_cancel(id: u64);
    fn myds4_status() -> *mut c_char;
    fn myds4_view() -> *mut c_char;
    fn myds4_free(p: *mut c_void);
    fn myds4_log(line: *const c_char);
    #[cfg(test)]
    fn myds4_tokenizer(dir: *const c_char) -> c_int;
    fn myds4_tokenize(text: *const c_char, len: usize, chat: c_int, out: *mut i32, max: c_int) -> c_int;
    fn myds4_piece(id: i32, len: *mut usize) -> *const c_char;
    fn myds4_added(id: i32) -> c_int;
}

/// The engine's settings: the server's model and engine options.
#[derive(Default)]
pub struct Config {
    pub model: String,
    pub experts: Option<String>,
    pub ctx: i32,
    pub cache_gb: f64,
    pub reserve_gb: f64,
    pub cache_half_life: i32,
    pub cache_read_cost: bool,
    pub substitute: f32,
    pub gpu_route: bool,
    pub draft: i32,
    pub draft_cache: i32,
    pub draft_trim: bool,
    pub online_predictor: bool,
    pub predictor_state: Option<String>,
    pub kv_gb: f64,
    pub batch: i32,
    pub ced: i32,
    pub lookup: i32,
    pub expert_batch: bool,
    pub prefill_prune: bool,
    /// Receives the log's lines (with their time) instead of stderr; it runs
    /// under the log's lock and must not call [`log`].
    pub on_log: Option<extern "C" fn(*const c_char)>,
    /// Runs before a fatal engine error's message and the process's exit.
    pub on_die: Option<extern "C" fn()>,
}

fn c_string(s: &str) -> CString {
    CString::new(s.replace('\0', "")).expect("no NUL bytes left")
}

/// Waits for the model lock, loads the model and starts the engine's
/// thread; false for invalid settings (the engine says why on stderr).
pub fn start(c: &Config) -> bool {
    let model = c_string(&c.model);
    let experts = c.experts.as_deref().map(c_string);
    let state = c.predictor_state.as_deref().map(c_string);
    let raw = RawConfig {
        abi: ABI,
        model: model.as_ptr(),
        experts: experts.as_ref().map_or(ptr::null(), |s| s.as_ptr()),
        ctx: c.ctx,
        cache_gb: c.cache_gb,
        reserve_gb: c.reserve_gb,
        cache_half_life: c.cache_half_life,
        cache_read_cost: c.cache_read_cost.into(),
        substitute: c.substitute,
        gpu_route: c.gpu_route.into(),
        draft: c.draft,
        draft_cache: c.draft_cache,
        draft_trim: c.draft_trim.into(),
        online_predictor: c.online_predictor.into(),
        predictor_state: state.as_ref().map_or(ptr::null(), |s| s.as_ptr()),
        kv_gb: c.kv_gb,
        batch: c.batch,
        ced: c.ced,
        lookup: c.lookup,
        expert_batch: c.expert_batch.into(),
        prefill_prune: c.prefill_prune.into(),
        on_log: c.on_log,
        on_die: c.on_die,
    };
    unsafe { myds4_start(&raw) == 0 }
}

/// Whether a stop was requested: SIGINT, SIGTERM or the live view's q.
pub fn stopping() -> bool {
    unsafe { myds4_stopping() != 0 }
}

/// Requests the stop as SIGINT does and returns at once.
pub fn request_stop() {
    unsafe { myds4_request_stop() }
}

/// Ends every request, saves the predictor state and waits for the engine.
pub fn stop() {
    unsafe { myds4_stop() }
}

/// A line for the log, with the time: on stderr, or to the live view's pane.
pub fn log(line: &str) {
    let line = c_string(line);
    unsafe { myds4_log(line.as_ptr()) }
}

/// GET /v1/status's JSON.
pub fn status() -> String {
    unsafe {
        let p = myds4_status();
        let s = CStr::from_ptr(p).to_string_lossy().into_owned();
        myds4_free(p.cast());
        s
    }
}

/// The live view's snapshot (JSON): phase, requests, statistics, memory and
/// the sessions' prefix tree.
pub fn view() -> String {
    unsafe {
        let p = myds4_view();
        let s = CStr::from_ptr(p).to_string_lossy().into_owned();
        myds4_free(p.cast());
        s
    }
}

/// Loads the checkpoint's tokenizer without the model.
#[cfg(test)]
pub fn load_tokenizer(dir: &str) {
    let dir = c_string(dir);
    unsafe { myds4_tokenizer(dir.as_ptr()) };
}

/// The token ids of plain text, without special tokens.
pub fn tokenize(text: &str) -> Vec<i32> {
    let mut out = vec![0i32; text.len() + 2];
    let n = unsafe { myds4_tokenize(text.as_ptr().cast(), text.len(), 0, out.as_mut_ptr(), out.len() as c_int) };
    out.truncate(n.max(0) as usize);
    out
}

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

/// What the engine reports about a request.
#[derive(Debug)]
pub enum Event {
    /// Its prefill ran: prompt tokens (images expanded), and those a session held.
    Started { prompt: usize, reused: usize },
    /// The tokens one decoding step kept; never the end token.
    Tokens(Vec<u32>),
    /// Its end; the last event. (The engine logs its counts.)
    Done { finish: Finish },
    /// Refused before its prefill; the last event.
    Refused(String),
}

/// A request for the engine: the rendered prompt (the byte 0xFF at each
/// image's place), its images' file bytes, sampling and its token limit.
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

type Sender = mpsc::UnboundedSender<Event>;

extern "C" fn on_event(user: *mut c_void, e: *const RawEvent) {
    // Safety: user is the Sender that submit boxed; e is valid during the call.
    let (tx, e) = unsafe { (&*(user as *const Sender), &*e) };
    let event = match e.kind {
        STARTED => Event::Started { prompt: e.prompt.max(0) as usize, reused: e.reused.max(0) as usize },
        TOKENS => {
            let ids = unsafe { std::slice::from_raw_parts(e.tokens, e.ntokens.max(0) as usize) };
            Event::Tokens(ids.iter().map(|&t| t as u32).collect())
        }
        DONE => Event::Done {
            finish: match e.finish {
                0 => Finish::Stop,
                1 => Finish::Length,
                2 => Finish::Cancelled,
                _ => Finish::Shutdown,
            },
        },
        _ => Event::Refused(if e.message.is_null() {
            String::new()
        } else {
            unsafe { CStr::from_ptr(e.message) }.to_string_lossy().into_owned()
        }),
    };
    let last = matches!(event, Event::Done { .. } | Event::Refused(_));
    let _ = tx.send(event);
    if last {
        // The engine calls back no more for this request.
        drop(unsafe { Box::from_raw(user as *mut Sender) });
    }
}

/// Queues a request: the guard that cancels it when dropped (and holds its
/// id), and its events.
pub fn submit(r: &Request) -> Result<(Cancel, mpsc::UnboundedReceiver<Event>), SubmitError> {
    let (tx, rx) = mpsc::unbounded_channel();
    let user = Box::into_raw(Box::new(tx)) as *mut c_void;
    let images: Vec<RawImage> = r.images.iter().map(|i| RawImage { data: i.as_ptr(), len: i.len() }).collect();
    let note = c_string(r.note);
    let raw = RawRequest {
        tokens: ptr::null(),
        ntokens: 0,
        text: r.prompt.as_ptr().cast(),
        len: r.prompt.len(),
        images: images.as_ptr(),
        nimages: images.len() as c_int,
        temperature: r.temperature,
        top_p: r.top_p,
        seed: r.seed.unwrap_or(0),
        seed_set: r.seed.is_some().into(),
        max_tokens: r.max_tokens as c_long,
        on_event,
        user,
        note: note.as_ptr(),
    };
    let mut id = 0u64;
    let mut message = [0 as c_char; 256];
    let result = unsafe { myds4_submit(&raw, &mut id, message.as_mut_ptr(), message.len()) };
    if result == QUEUED {
        return Ok((Cancel(id), rx));
    }
    drop(unsafe { Box::from_raw(user as *mut Sender) });
    let message = unsafe { CStr::from_ptr(message.as_ptr()) }.to_string_lossy().into_owned();
    Err(if result == BUSY { SubmitError::Busy(message) } else { SubmitError::Invalid(message) })
}

/// Cancels its request when dropped: when the response ends, whether it was
/// finished, cut by its client before or after it started, or ended early by
/// a stop sequence. The engine ignores the ids of requests that ended.
pub struct Cancel(u64);

impl Cancel {
    /// The request's id, its number in the log.
    pub fn id(&self) -> u64 {
        self.0
    }
}

impl Drop for Cancel {
    fn drop(&mut self) {
        unsafe { myds4_cancel(self.0) }
    }
}

/// The engine's tokenizer for recipe's stream processor: a token's text is
/// its bytes, and the special tokens are the tokenizer's added tokens.
pub struct Tokenizer;

impl StreamTokenizer for Tokenizer {
    fn decode_ids(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String, String> {
        let mut bytes = Vec::new();
        for &id in ids {
            if skip_special_tokens && self.is_special_token(id) {
                continue;
            }
            let mut n = 0usize;
            let p = unsafe { myds4_piece(id as i32, &mut n) };
            bytes.extend_from_slice(unsafe { std::slice::from_raw_parts(p.cast::<u8>(), n) });
        }
        // An unfinished character at the end becomes U+FFFD, so the processor
        // waits for the token that finishes it.
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    fn count_tokens(&self, text: &str) -> Option<usize> {
        Some(tokenize(text).len())
    }

    fn is_special_token(&self, id: u32) -> bool {
        unsafe { myds4_added(id as i32) != 0 }
    }
}
