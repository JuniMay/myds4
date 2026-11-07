//! The engine's C ABI (src/engine/engine.h) and thin safe wrappers over it.
//! The model's functions (states, snapshots, prefill, decoding, sampling,
//! lookups, images, statistics) run on the worker's thread only
//! (worker.rs); the log, the stop and the tokenizer on any thread.

use std::ffi::{CString, c_char, c_int};
use std::ptr;

use deepseek_recipe::stream::StreamTokenizer;

const ABI: c_int = 6;

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
    ced: c_int,
    lookup: c_int,
    expert_batch: c_int,
    prefill_prune: c_int,
    on_log: Option<extern "C" fn(*const c_char)>,
    on_die: Option<extern "C" fn()>,
}

/// An engine attention state (myds4_state).
#[repr(C)]
pub struct RawState {
    _opaque: [u8; 0],
}

/// A snapshot of a state's rolling rows (myds4_snapshot).
#[repr(C)]
pub struct RawSnapshot {
    _opaque: [u8; 0],
}

#[repr(C)]
struct RawSampler {
    _opaque: [u8; 0],
}

#[repr(C)]
struct RawLookup {
    _opaque: [u8; 0],
}

#[repr(C)]
struct RawImages {
    _opaque: [u8; 0],
}

/// A prefill's plan (myds4_prefill_info).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct PrefillInfo {
    pub chunks: c_int,
    pub tokens: c_int,
    pub decoder_tokens: c_int,
    pub ced_applied: c_int,
    pub prune_applied: c_int,
    pub prefix_tokens: c_int,
}

/// The expert cache's counters (myds4_cache_counts).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct CacheCounts {
    pub hits: u64,
    pub misses: u64,
    pub prefetch_hits: u64,
    pub prefetches: u64,
    pub bytes: u64,
    pub evictions: u64,
    pub late: u64,
    pub cancelled: u64,
    pub substituted: u64,
    pub remaps: u64,
    pub given_back: u64,
    pub row_own: u64,
    pub row_ahead: u64,
    pub row_late: u64,
    pub row_sub: u64,
    pub row_ssd: u64,
    pub read_s: f64,
}

/// The engine's statistics (myds4_stats).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Stats {
    pub cache: CacheCounts,
    pub decode: CacheCounts,
    pub prefill: CacheCounts,
    pub decode_rows: u64,
    pub prefill_rows: u64,
    pub decode_seconds: f64,
    pub decode_gpu_seconds: f64,
    pub prefill_layer: c_int,
    pub prefill_rows_now: c_int,
    pub slots: c_int,
    pub mapped: c_int,
    pub cap: c_int,
    pub max_slots: c_int,
    pub experts: c_int,
    pub uncached: c_int,
    pub slot_bytes: u64,
    pub uncached_bytes: u64,
    pub cache_bytes: u64,
    pub dense_bytes: u64,
    pub footprint: u64,
    pub gpu_in_use: u64,
    pub available: u64,
    pub gpu_limit: i64,
    pub total: u64,
    pub free: u64,
    pub file: u64,
    pub file_kept: u64,
    pub wired: u64,
    pub compressor: u64,
    pub compressed: u64,
    pub swap_used: u64,
    pub swapouts: u64,
    pub pressure: c_int,
    pub layers: c_int,
    pub experts_per_row: c_int,
}

unsafe extern "C" {
    fn myds4_load(c: *const RawConfig) -> c_int;
    fn myds4_set_runahead(enabled: c_int) -> c_int;
    fn myds4_finish();
    fn myds4_stopping() -> c_int;
    fn myds4_request_stop();
    fn myds4_idle() -> c_int;
    fn myds4_log(line: *const c_char);
    fn myds4_max_rows() -> c_int;
    fn myds4_draft_block() -> c_int;
    fn myds4_rewind() -> c_int;
    fn myds4_vocab() -> c_int;
    fn myds4_state_new() -> *mut RawState;
    fn myds4_state_free(s: *mut RawState);
    fn myds4_state_bytes(s: *const RawState, positions: c_int) -> usize;
    fn myds4_state_copy(dst: *mut RawState, src: *const RawState, positions: c_int);
    fn myds4_state_use(s: *mut RawState);
    fn myds4_reset();
    fn myds4_snapshot_take(s: *mut RawState, pos: c_int, with_logits: c_int) -> *mut RawSnapshot;
    fn myds4_snapshot_restore(s: *mut RawState, x: *const RawSnapshot);
    fn myds4_snapshot_free(x: *mut RawSnapshot);
    fn myds4_snapshot_bytes(with_logits: c_int) -> usize;
    fn myds4_snapshot_logits(x: *const RawSnapshot) -> *const f32;
    fn myds4_prefill(
        tokens: *const i32,
        n: c_int,
        from: c_int,
        replay: c_int,
        capture: c_int,
        captured: *mut *mut RawSnapshot,
        info: *mut PrefillInfo,
    ) -> *const f32;
    fn myds4_replay_rows(k: c_int) -> c_int;
    fn myds4_replay_start(k: c_int) -> c_int;
    fn myds4_step(token: i32, pos: c_int) -> *const f32;
    fn myds4_rows(states: *mut *mut RawState, tokens: *const i32, pos: *const i32, m: c_int) -> *const f32;
    fn myds4_draft(token: i32, pos: c_int, drafts: *mut i32, n: c_int);
    fn myds4_verify(rows: *const i32, m: c_int, pos: c_int) -> *const f32;
    fn myds4_accept(rows: *const i32, n: c_int, pos: c_int);
    fn myds4_seek(tokens: *const i32, pos: c_int);
    fn myds4_sampler_new(temperature: f64, top_p: f64, seed: u64, seed_set: c_int) -> *mut RawSampler;
    fn myds4_sampler_free(s: *mut RawSampler);
    fn myds4_sample(logits: *const f32, s: *mut RawSampler) -> i32;
    fn myds4_sample_verify(logits: *const f32, drafts: *const i32, n: c_int, eos: i32, s: *mut RawSampler, accepted: *mut c_int) -> i32;
    fn myds4_lookup_new(capacity: c_int) -> *mut RawLookup;
    fn myds4_lookup_free(x: *mut RawLookup);
    fn myds4_lookup_append(x: *mut RawLookup, tokens: *const i32, n: c_int) -> c_int;
    fn myds4_lookup_propose(x: *const RawLookup, max: c_int, drafts: *mut i32) -> c_int;
    fn myds4_lookup_feedback(x: *mut RawLookup, offered: c_int, accepted: c_int);
    fn myds4_images_open(data: *const *const u8, len: *const usize, n: c_int, message: *mut c_char, cap: usize) -> *mut RawImages;
    fn myds4_images_expand(x: *mut RawImages, input: *const i32, n: c_int, out: *mut i32, max: c_int) -> c_int;
    fn myds4_images_encode(x: *mut RawImages, from: c_int) -> c_int;
    fn myds4_images_close(x: *mut RawImages);
    fn myds4_stats_get(s: *mut Stats);
    fn myds4_routes(row: c_int, experts: *mut u16, sources: *mut u8) -> c_int;
    #[cfg(test)]
    fn myds4_tokenizer(dir: *const c_char) -> c_int;
    fn myds4_tokenize(text: *const c_char, len: usize, chat: c_int, out: *mut i32, max: c_int) -> c_int;
    fn myds4_piece(id: i32, len: *mut usize) -> *const c_char;
    fn myds4_added(id: i32) -> c_int;
    fn myds4_special(text: *const c_char, len: usize) -> i32;
}

/// The engine's settings and the server's: the model, engine and cache options.
#[derive(Default, Clone)]
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
    /// --kv-cache: what the attention states and snapshots may take, GiB (the worker's).
    pub kv_gb: f64,
    /// --batch: requests decoding together at most (the worker's).
    pub batch: i32,
    pub ced: i32,
    pub lookup: i32,
    pub expert_batch: bool,
    pub prefill_prune: bool,
    pub runahead: bool,
    /// Receives the log's lines (with their time) instead of stderr; it runs
    /// under the log's lock and must not call [`log`].
    pub on_log: Option<extern "C" fn(*const c_char)>,
    /// Runs before a fatal engine error's message and the process's exit.
    pub on_die: Option<extern "C" fn()>,
}

fn c_string(s: &str) -> CString {
    CString::new(s.replace('\0', "")).expect("no NUL bytes left")
}

/// Loads the model on this thread, which from then on is the engine's (it
/// needs a 16 MiB stack); false for invalid settings (the engine says why).
pub fn load(c: &Config) -> bool {
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
        ced: c.ced,
        lookup: c.lookup,
        expert_batch: c.expert_batch.into(),
        prefill_prune: c.prefill_prune.into(),
        on_log: c.on_log,
        on_die: c.on_die,
    };
    unsafe { myds4_set_runahead(c.runahead.into()) == 0 && myds4_load(&raw) == 0 }
}

/// After the last request: saves --predictor-state and reports the predictor.
pub fn finish() {
    unsafe { myds4_finish() }
}

/// Whether a stop was requested: SIGINT, SIGTERM or [`request_stop`].
pub fn stopping() -> bool {
    unsafe { myds4_stopping() != 0 }
}

/// Requests the stop as SIGINT does (the worker notices it within a second,
/// or at once through worker::request_stop).
pub fn request_stop() {
    unsafe { myds4_request_stop() }
}

/// While no request runs: the expert cache gives slots back if other
/// programs need the memory; the slots it keeps.
pub fn idle() -> i32 {
    unsafe { myds4_idle() }
}

/// A line for the log, with the time: on stderr, or to the live view's pane.
pub fn log(line: &str) {
    let line = c_string(line);
    unsafe { myds4_log(line.as_ptr()) }
}

/// The model's limits: rows of different states one step takes, DSpark's
/// drafts per pass, the rows of one verify pass, and the positions a state
/// can continue before its end.
pub fn max_rows() -> usize {
    unsafe { myds4_max_rows() as usize }
}
pub fn draft_block() -> usize {
    unsafe { myds4_draft_block() as usize }
}
pub fn rewind() -> usize {
    unsafe { myds4_rewind() as usize }
}
/// The vocabulary: the floats in one row of logits.
pub fn vocab() -> usize {
    unsafe { myds4_vocab() as usize }
}

// ---- states and snapshots (the worker's thread) ----

pub fn state_new() -> *mut RawState {
    unsafe { myds4_state_new() }
}
/// Not the state in use.
pub fn state_free(s: *mut RawState) {
    unsafe { myds4_state_free(s) }
}
/// The bytes a state (null: a new one) takes holding that many positions.
pub fn state_bytes(s: *const RawState, positions: usize) -> u64 {
    unsafe { myds4_state_bytes(s, positions as c_int) as u64 }
}
/// src's global rows before `positions` into dst, which is not in use.
pub fn state_copy(dst: *mut RawState, src: *const RawState, positions: usize) {
    unsafe { myds4_state_copy(dst, src, positions as c_int) }
}
pub fn state_use(s: *mut RawState) {
    unsafe { myds4_state_use(s) }
}
pub fn reset() {
    unsafe { myds4_reset() }
}
pub fn snapshot_take(s: *mut RawState, pos: usize, with_logits: bool) -> *mut RawSnapshot {
    unsafe { myds4_snapshot_take(s, pos as c_int, with_logits.into()) }
}
pub fn snapshot_restore(s: *mut RawState, x: *const RawSnapshot) {
    unsafe { myds4_snapshot_restore(s, x) }
}
pub fn snapshot_free(x: *mut RawSnapshot) {
    unsafe { myds4_snapshot_free(x) }
}
pub fn snapshot_bytes(with_logits: bool) -> u64 {
    unsafe { myds4_snapshot_bytes(with_logits.into()) as u64 }
}
/// The logits a snapshot keeps (null without).
pub fn snapshot_logits(x: *const RawSnapshot) -> *const f32 {
    unsafe { myds4_snapshot_logits(x) }
}

// ---- running the model, in the state in use ----

/// Logits of one row or more, valid until the next call that runs the model.
pub type Logits = *const f32;

/// Prefills positions `from..` of `tokens` (all of the prompt); with `replay`,
/// the state holds only the global rows before `from` (exact with --ced off,
/// bounded and approximate with --ced). With `capture`, the rolling rows at
/// that position (a replay's `from`, or one after `from`) come back as a
/// snapshot, or null when it would hold only the encoder layers. None for
/// invalid arguments.
pub fn prefill(
    tokens: &[i32],
    from: usize,
    replay: bool,
    capture: Option<usize>,
) -> Option<(Logits, *mut RawSnapshot, PrefillInfo)> {
    let mut captured = ptr::null_mut();
    let mut info = PrefillInfo::default();
    let at = capture.map_or(-1, |p| p as c_int);
    let lg = unsafe {
        myds4_prefill(tokens.as_ptr(), tokens.len() as c_int, from as c_int, replay.into(), at, &mut captured, &mut info)
    };
    (!lg.is_null()).then_some((lg, captured, info))
}
/// What a replay at k computes again before k: rows per layer, on average.
pub fn replay_rows(k: usize) -> usize {
    unsafe { myds4_replay_rows(k as c_int).max(0) as usize }
}
/// The first position an exact replay at k computes again: its images from there are encoded.
pub fn replay_start(k: usize) -> usize {
    unsafe { myds4_replay_start(k as c_int).max(0) as usize }
}
pub fn step(token: i32, pos: usize) -> Option<Logits> {
    let lg = unsafe { myds4_step(token, pos as c_int) };
    (!lg.is_null()).then_some(lg)
}
pub fn rows(states: &mut [*mut RawState], tokens: &[i32], pos: &[i32]) -> Option<Logits> {
    let lg = unsafe { myds4_rows(states.as_mut_ptr(), tokens.as_ptr(), pos.as_ptr(), states.len() as c_int) };
    (!lg.is_null()).then_some(lg)
}
pub fn draft(token: i32, pos: usize, drafts: &mut [i32]) {
    unsafe { myds4_draft(token, pos as c_int, drafts.as_mut_ptr(), drafts.len() as c_int) }
}
pub fn verify(rows: &[i32], pos: usize) -> Option<Logits> {
    let lg = unsafe { myds4_verify(rows.as_ptr(), rows.len() as c_int, pos as c_int) };
    (!lg.is_null()).then_some(lg)
}
pub fn accept(rows: &[i32], pos: usize) {
    unsafe { myds4_accept(rows.as_ptr(), rows.len() as c_int, pos as c_int) }
}
pub fn seek(tokens: &[i32], pos: usize) {
    unsafe { myds4_seek(tokens.as_ptr(), pos as c_int) }
}

// ---- sampling, lookups and images ----

pub struct Sampler(*mut RawSampler);

impl Sampler {
    /// None for invalid settings.
    pub fn new(temperature: f64, top_p: f64, seed: Option<u64>) -> Option<Sampler> {
        let s = unsafe { myds4_sampler_new(temperature, top_p, seed.unwrap_or(0), seed.is_some().into()) };
        (!s.is_null()).then_some(Sampler(s))
    }
    pub fn sample(&mut self, lg: Logits) -> i32 {
        unsafe { myds4_sample(lg, self.0) }
    }
    /// Rows of logits for a token and its drafts: the next token, and how many
    /// drafts it kept; eos ends the kept ones.
    pub fn verify(&mut self, lg: Logits, drafts: &[i32], eos: i32) -> (i32, usize) {
        let mut accepted = 0;
        let next = unsafe { myds4_sample_verify(lg, drafts.as_ptr(), drafts.len() as c_int, eos, self.0, &mut accepted) };
        (next, accepted.max(0) as usize)
    }
}

impl Drop for Sampler {
    fn drop(&mut self) {
        unsafe { myds4_sampler_free(self.0) }
    }
}

pub struct Lookup(*mut RawLookup);

impl Lookup {
    pub fn new(capacity: usize) -> Option<Lookup> {
        let x = unsafe { myds4_lookup_new(capacity as c_int) };
        (!x.is_null()).then_some(Lookup(x))
    }
    /// False when full.
    pub fn append(&mut self, tokens: &[i32]) -> bool {
        unsafe { myds4_lookup_append(self.0, tokens.as_ptr(), tokens.len() as c_int) != 0 }
    }
    pub fn propose(&self, max: usize, drafts: &mut [i32]) -> usize {
        unsafe { myds4_lookup_propose(self.0, max.min(drafts.len()) as c_int, drafts.as_mut_ptr()).max(0) as usize }
    }
    pub fn feedback(&mut self, offered: usize, accepted: usize) {
        unsafe { myds4_lookup_feedback(self.0, offered as c_int, accepted as c_int) }
    }
}

impl Drop for Lookup {
    fn drop(&mut self) {
        unsafe { myds4_lookup_free(self.0) }
    }
}

/// A prompt's images, decoded; closed (and given back) when dropped.
pub struct Images(*mut RawImages);

impl Images {
    pub fn open(images: &[Vec<u8>]) -> Result<Images, String> {
        let data: Vec<*const u8> = images.iter().map(|i| i.as_ptr()).collect();
        let len: Vec<usize> = images.iter().map(|i| i.len()).collect();
        let mut message = [0 as c_char; 256];
        let x = unsafe { myds4_images_open(data.as_ptr(), len.as_ptr(), images.len() as c_int, message.as_mut_ptr(), message.len()) };
        if x.is_null() {
            return Err(unsafe { std::ffi::CStr::from_ptr(message.as_ptr()) }.to_string_lossy().into_owned());
        }
        Ok(Images(x))
    }
    /// The prompt with each image place replaced by its image's span, or None
    /// if the places and images differ or it would exceed `max` tokens.
    pub fn expand(&mut self, prompt: &[i32], max: usize) -> Option<Vec<i32>> {
        let mut out = vec![0i32; max + 2];
        let n = unsafe { myds4_images_expand(self.0, prompt.as_ptr(), prompt.len() as c_int, out.as_mut_ptr(), max as c_int) };
        (n >= 0).then(|| {
            out.truncate(n as usize);
            out
        })
    }
    /// Encodes the images that positions from `from` reach, for the next
    /// prefill; how many.
    pub fn encode(&mut self, from: usize) -> usize {
        unsafe { myds4_images_encode(self.0, from as c_int).max(0) as usize }
    }
}

impl Drop for Images {
    fn drop(&mut self) {
        unsafe { myds4_images_close(self.0) }
    }
}

pub fn stats() -> Stats {
    let mut s = Stats::default();
    unsafe { myds4_stats_get(&mut s) };
    s
}

/// The experts row `row` of the latest decode step chose, `per_row` of them
/// (6 per layer, layer by layer), and where each one was (myds4_routes); None
/// when the row did not run every layer in that step. From the worker's
/// thread.
pub fn routes(row: usize, per_row: usize) -> Option<(Vec<u16>, Vec<u8>)> {
    let (mut experts, mut sources) = (vec![0u16; per_row], vec![0u8; per_row]);
    // Safety: the engine writes layers * 6 entries, per_row of them (experts_per_row).
    (unsafe { myds4_routes(row as c_int, experts.as_mut_ptr(), sources.as_mut_ptr()) } != 0).then_some((experts, sources))
}

// ---- the tokenizer, from any thread ----

/// Loads the checkpoint's tokenizer without the model.
#[cfg(test)]
pub fn load_tokenizer(dir: &str) {
    let dir = c_string(dir);
    unsafe { myds4_tokenizer(dir.as_ptr()) };
}

fn tokenize_as(text: &[u8], chat: bool) -> Vec<i32> {
    let mut out = vec![0i32; text.len() + 2];
    let n = unsafe { myds4_tokenize(text.as_ptr().cast(), text.len(), chat.into(), out.as_mut_ptr(), out.len() as c_int) };
    out.truncate(n.max(0) as usize);
    out
}

/// The token ids of plain text, without special tokens.
pub fn tokenize(text: &str) -> Vec<i32> {
    tokenize_as(text.as_bytes(), false)
}

/// The token ids of a rendered prompt: a special token's spelling is that
/// token, the byte 0xFF an image's place.
pub fn tokenize_prompt(prompt: &[u8]) -> Vec<i32> {
    tokenize_as(prompt, true)
}

/// The id of a special token's spelling.
pub fn special(text: &str) -> Option<i32> {
    let id = unsafe { myds4_special(text.as_ptr().cast(), text.len()) };
    (id >= 0).then_some(id)
}

/// A token's bytes; none for an unknown id.
fn piece(id: i32) -> &'static [u8] {
    let mut n = 0usize;
    // Safety: myds4_piece gives a token's bytes, length 0 for an unknown id;
    // the tokenizer and its bytes stay loaded while the process runs.
    let p = unsafe { myds4_piece(id, &mut n) };
    if p.is_null() || n == 0 { &[] } else { unsafe { std::slice::from_raw_parts(p.cast::<u8>(), n) } }
}

/// The text of token ids, for the live view, in runs: the special tokens
/// (the tokenizer's added tokens) each alone and marked true, the text
/// between them, and one "[image]" for each run of an image's key (the
/// negative ids of [`Images::expand`]).
pub fn text(ids: &[i32]) -> Vec<(String, bool)> {
    let mut runs = Vec::new();
    let mut bytes = Vec::new();
    let flush = |bytes: &mut Vec<u8>, runs: &mut Vec<(String, bool)>| {
        if !bytes.is_empty() {
            runs.push((String::from_utf8_lossy(bytes).into_owned(), false));
            bytes.clear();
        }
    };
    let mut last = 0i32;
    for &id in ids {
        if id < 0 {
            if id != last {
                bytes.extend_from_slice(b"[image]");
            }
        } else if unsafe { myds4_added(id) } != 0 {
            flush(&mut bytes, &mut runs);
            runs.push((String::from_utf8_lossy(piece(id)).into_owned(), true));
        } else {
            bytes.extend_from_slice(piece(id));
        }
        last = id;
    }
    flush(&mut bytes, &mut runs);
    runs
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
            bytes.extend_from_slice(piece(id as i32));
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
