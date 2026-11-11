//! The engine's C ABI (src/engine/engine.h) and thin safe wrappers over it.
//! A [`Model`] is one registered checkpoint; its weights load on demand. Its
//! functions on states, snapshots, prefill, decoding, sampling, media and
//! routes run on the worker's engine thread only (worker.rs); the log, the
//! stop, the statistics and the tokenizer on any thread.

use std::ffi::{CString, c_char, c_int};
use std::ptr;

use deepseek_recipe::stream::StreamTokenizer;

const ABI: c_int = 19;

#[repr(C)]
struct RawProcess {
    abi: c_int,
    memory_gb: f64,
    reserve_gb: f64,
    on_log: Option<extern "C" fn(*const c_char)>,
    on_die: Option<extern "C" fn()>,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct RawDeepSeekConfig {
    experts: *const c_char,
    cache_half_life: c_int,
    cache_read_cost: c_int,
    substitute: f32,
    draft: c_int,
    draft_cache: c_int,
    online_predictor: c_int,
    predictor_state: *const c_char,
    ced: c_int,
    prefill_prune: c_int,
    runahead: c_int,
    pipeline: c_int,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct RawQwenConfig {
    draft: c_int,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct RawNemotronConfig {
    draft: c_int,
    draft_model: *const c_char,
}

#[repr(C)]
union RawModelOptions {
    deepseek: RawDeepSeekConfig,
    qwen: RawQwenConfig,
    nemotron: RawNemotronConfig,
}

#[repr(C)]
struct RawConfig {
    abi: c_int,
    model: *const c_char,
    architecture: c_int,
    ctx: c_int,
    lookup: c_int,
    options: RawModelOptions,
}

#[repr(C)]
struct RawModel {
    _opaque: [u8; 0],
}

/// A state of one model (myds4_state).
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

/// A stream of one model's calls (myds4_stream).
#[repr(C)]
pub struct RawStream {
    _opaque: [u8; 0],
}

#[repr(C)]
struct RawLookup {
    _opaque: [u8; 0],
}

#[repr(C)]
struct RawDraftChoice {
    _opaque: [u8; 0],
}

#[repr(C)]
struct RawMedia {
    _opaque: [u8; 0],
}

#[repr(C)]
struct RawTokenizer {
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

/// A model's counters and the process's memory (myds4_stats).
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
    pub routed_layers: c_int,
    pub experts_per_layer: c_int,
    pub model_memory_limit: u64,
    pub model_memory_used: u64,
    pub model_memory_committed: u64,
    pub model_memory_peak: u64,
    pub kv_allocated: u64,
    pub model_workspace: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct MemoryInfo {
    pub limit: u64,
    pub model_bytes: u64,
    pub kv_bytes: u64,
}

unsafe extern "C" {
    fn myds4_init(p: *const RawProcess) -> c_int;
    fn myds4_stopping() -> c_int;
    fn myds4_request_stop();
    fn myds4_log(line: *const c_char);
    fn myds4_open(c: *const RawConfig) -> *mut RawModel;
    fn myds4_load(m: *mut RawModel) -> c_int;
    fn myds4_loaded(m: *const RawModel) -> c_int;
    fn myds4_active(m: *mut RawModel, active: c_int);
    fn myds4_finish(m: *mut RawModel);
    fn myds4_idle(m: *mut RawModel) -> c_int;
    fn myds4_unload(m: *mut RawModel) -> u64;
    fn myds4_reclaim(bytes: u64) -> u64;
    fn myds4_memory() -> MemoryInfo;
    fn myds4_memory_limit(bytes: u64);
    fn myds4_drain();
    fn myds4_max_rows(m: *const RawModel) -> c_int;
    fn myds4_draft_block(m: *const RawModel) -> c_int;
    fn myds4_max_verify(m: *const RawModel) -> c_int;
    fn myds4_rewind(m: *const RawModel) -> c_int;
    fn myds4_vocab(m: *const RawModel) -> c_int;
    fn myds4_stop(m: *const RawModel, token: i32) -> c_int;
    fn myds4_prefill_rows(m: *const RawModel) -> c_int;
    fn myds4_state_new(m: *mut RawModel) -> *mut RawState;
    fn myds4_state_free(m: *mut RawModel, s: *mut RawState);
    fn myds4_state_bytes(m: *const RawModel, s: *const RawState, positions: c_int) -> usize;
    fn myds4_state_copy(
        m: *mut RawModel,
        dst: *mut RawState,
        src: *const RawState,
        positions: c_int,
    );
    fn myds4_snapshot_take(
        m: *mut RawModel,
        s: *mut RawState,
        pos: c_int,
        logits: *const f32,
    ) -> *mut RawSnapshot;
    fn myds4_snapshot_restore(m: *mut RawModel, s: *mut RawState, x: *const RawSnapshot);
    fn myds4_snapshot_free(m: *mut RawModel, x: *mut RawSnapshot);
    fn myds4_snapshot_bytes(m: *const RawModel, with_logits: c_int) -> usize;
    fn myds4_snapshot_logits(m: *const RawModel, x: *const RawSnapshot) -> *const f32;
    fn myds4_stop_requests() -> c_int;
    fn myds4_state_write(
        m: *mut RawModel,
        s: *mut RawState,
        from: c_int,
        to: c_int,
        fd: c_int,
    ) -> i64;
    fn myds4_state_read(
        m: *mut RawModel,
        s: *mut RawState,
        positions: c_int,
        fds: *const c_int,
        count: c_int,
    ) -> c_int;
    fn myds4_snapshot_write(
        m: *mut RawModel,
        x: *const RawSnapshot,
        s: *mut RawState,
        pos: c_int,
        fd: c_int,
    ) -> i64;
    fn myds4_snapshot_read(m: *mut RawModel, fd: c_int) -> *mut RawSnapshot;
    fn myds4_state_identity(m: *mut RawModel, out: *mut u8);
    fn myds4_replay_rows(m: *const RawModel, k: c_int) -> c_int;
    fn myds4_replay_start(m: *const RawModel, k: c_int) -> c_int;
    fn myds4_seek(m: *mut RawModel, s: *mut RawState, tokens: *const i32, pos: c_int);
    fn myds4_stream_new(m: *mut RawModel) -> *mut RawStream;
    fn myds4_stream_free(x: *mut RawStream);
    fn myds4_stream_prefill(
        x: *mut RawStream,
        s: *mut RawState,
        tokens: *const i32,
        n: c_int,
        from: c_int,
        replay: c_int,
        capture: c_int,
        images: *const RawMedia,
    );
    fn myds4_stream_rows(
        x: *mut RawStream,
        states: *mut *mut RawState,
        tokens: *const i32,
        pos: *const i32,
        count: c_int,
    );
    fn myds4_stream_verify(
        x: *mut RawStream,
        s: *mut RawState,
        rows: *mut i32,
        drafts: c_int,
        proposed: c_int,
        pos: c_int,
        sampler: *mut RawSampler,
    );
    fn myds4_run(streams: *const *mut RawStream, n: c_int, seconds: f64) -> c_int;
    fn myds4_stream_logits(x: *const RawStream) -> *const f32;
    fn myds4_stream_captured(x: *const RawStream) -> *mut RawSnapshot;
    fn myds4_stream_info(x: *const RawStream, info: *mut PrefillInfo);
    fn myds4_stream_next(x: *const RawStream, accepted: *mut c_int) -> i32;
    fn myds4_sampler_new(
        temperature: f64,
        top_p: f64,
        top_k: c_int,
        seed: u64,
        seed_set: c_int,
    ) -> *mut RawSampler;
    fn myds4_sampler_free(s: *mut RawSampler);
    fn myds4_sample(m: *const RawModel, logits: *const f32, s: *mut RawSampler) -> i32;
    fn myds4_lookup_new(capacity: c_int) -> *mut RawLookup;
    fn myds4_lookup_free(x: *mut RawLookup);
    fn myds4_lookup_append(x: *mut RawLookup, tokens: *const i32, n: c_int) -> c_int;
    fn myds4_lookup_propose(x: *const RawLookup, max: c_int, drafts: *mut i32) -> c_int;
    fn myds4_lookup_feedback(x: *mut RawLookup, offered: c_int, accepted: c_int);
    fn myds4_draft_choice_new() -> *mut RawDraftChoice;
    fn myds4_draft_choice_free(x: *mut RawDraftChoice);
    fn myds4_draft_choice_count(x: *mut RawDraftChoice, most: c_int) -> c_int;
    fn myds4_draft_choice_feedback(x: *mut RawDraftChoice, count: c_int, kept: c_int, seconds: f64);
    fn myds4_media_open(
        m: *mut RawModel,
        data: *const *const u8,
        len: *const usize,
        n: c_int,
        message: *mut c_char,
        cap: usize,
    ) -> *mut RawMedia;
    fn myds4_media_expand(
        x: *mut RawMedia,
        input: *const i32,
        n: c_int,
        out: *mut i32,
        max: c_int,
    ) -> c_int;
    fn myds4_media_encode(x: *mut RawMedia, from: c_int) -> c_int;
    fn myds4_media_close(x: *mut RawMedia);
    fn myds4_refresh_stats();
    fn myds4_stats_get(m: *const RawModel, s: *mut Stats);
    fn myds4_routes(m: *const RawModel, row: c_int, experts: *mut u16, sources: *mut u8) -> c_int;
    fn myds4_model_tokenizer(m: *const RawModel) -> *mut RawTokenizer;
    #[cfg(test)]
    fn myds4_tokenizer_open(dir: *const c_char, image: *const c_char) -> *mut RawTokenizer;
    fn myds4_tokenize(
        t: *mut RawTokenizer,
        text: *const c_char,
        len: usize,
        mode: c_int,
        out: *mut i32,
        max: c_int,
    ) -> c_int;
    fn myds4_piece(t: *const RawTokenizer, id: i32, len: *mut usize) -> *const c_char;
    fn myds4_added(t: *const RawTokenizer, id: i32) -> c_int;
    fn myds4_special(t: *const RawTokenizer, text: *const c_char, len: usize) -> i32;
}

fn c_string(s: &str) -> CString {
    CString::new(s.replace('\0', "")).expect("no NUL bytes left")
}

// ---- the process ----

/// Installs the log (on_log receives its lines, with their time, under the
/// log's lock, and must not call [`log`]), the stop's signal handlers and the
/// failure hook, on this thread, which from then on is the engine's (it
/// needs a 16 MiB stack). False when called twice.
pub fn init(
    memory_gb: f64,
    reserve_gb: f64,
    on_log: Option<extern "C" fn(*const c_char)>,
    on_die: Option<extern "C" fn()>,
) -> bool {
    let p = RawProcess {
        abi: ABI,
        memory_gb,
        reserve_gb,
        on_log,
        on_die,
    };
    unsafe { myds4_init(&p) == 0 }
}

/// Whether a stop was requested: SIGINT, SIGTERM or [`request_stop`].
pub fn stopping() -> bool {
    unsafe { myds4_stopping() != 0 }
}

/// How many stops were requested: saving states at quit ends at the second.
pub fn stop_requests() -> u32 {
    unsafe { myds4_stop_requests().max(0) as u32 }
}

pub fn refresh_stats() {
    unsafe { myds4_refresh_stats() };
}

pub fn reclaim(bytes: u64) -> u64 {
    unsafe { myds4_reclaim(bytes) }
}

pub fn memory() -> MemoryInfo {
    unsafe { myds4_memory() }
}
pub fn memory_limit(bytes: u64) {
    unsafe { myds4_memory_limit(bytes) };
}
pub fn drain() {
    unsafe { myds4_drain() };
}

/// Requests the stop as SIGINT does (the worker notices it within a second,
/// or at once through worker::request_stop).
pub fn request_stop() {
    unsafe { myds4_request_stop() }
}

/// A line for the log, with the time: on stderr, or to the live view's pane.
pub fn log(line: &str) {
    let line = c_string(line);
    unsafe { myds4_log(line.as_ptr()) }
}

/// Which format a checkpoint's requests take: its family, from config.json's
/// model_type.
#[repr(i32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Architecture {
    DeepSeek = 1,
    Qwen = 2,
    Nemotron = 3,
}

impl Architecture {
    /// The family of a checkpoint directory, or why none.
    pub fn of(dir: &str) -> Result<Architecture, String> {
        let path = std::path::Path::new(dir).join("config.json");
        let text =
            std::fs::read(&path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        let config: serde_json::Value =
            serde_json::from_slice(&text).map_err(|e| format!("{}: {e}", path.display()))?;
        match config.get("model_type").and_then(serde_json::Value::as_str) {
            Some("deepseek_v41") => Ok(Architecture::DeepSeek),
            Some("qwen3_5" | "qwen3_5_text" | "qwen4_exp" | "qwen4_exp_text") => {
                Ok(Architecture::Qwen)
            }
            Some("nemo_speechlm") => Ok(Architecture::Nemotron),
            Some(t) => Err(format!(
                "{}: model_type {t} is not one this server runs",
                path.display()
            )),
            None => Err(format!("{}: no model_type", path.display())),
        }
    }
}

// ---- models ----

/// Settings shared by every model, plus one architecture's options.
#[derive(Default, Clone, Debug, PartialEq)]
pub struct Options {
    pub model: String,
    pub ctx: i32,
    pub draft: i32,
    pub lookup: i32,
    pub backend: BackendOptions,
}

#[derive(Default, Clone, Debug, PartialEq)]
pub enum BackendOptions {
    #[default]
    Unspecified,
    DeepSeek(DeepSeekOptions),
    Qwen,
    Nemotron {
        draft_model: Option<String>,
    },
}

#[derive(Default, Clone, Debug, PartialEq)]
pub struct DeepSeekOptions {
    pub experts: Option<String>,
    pub cache_half_life: i32,
    pub cache_read_cost: bool,
    pub draft_cache: i32,
    pub online_predictor: bool,
    pub predictor_state: Option<String>,
    pub ced: i32,
    pub prefill_prune: bool,
    pub runahead: bool,
    pub pipeline: bool,
}

impl Options {
    pub fn configure(&mut self, architecture: Architecture) -> Result<(), String> {
        self.backend = match (&self.backend, architecture) {
            (BackendOptions::Unspecified, Architecture::DeepSeek) => {
                BackendOptions::DeepSeek(DeepSeekOptions::default())
            }
            (BackendOptions::Unspecified, Architecture::Qwen) => BackendOptions::Qwen,
            (BackendOptions::Unspecified, Architecture::Nemotron) => {
                BackendOptions::Nemotron { draft_model: None }
            }
            (BackendOptions::DeepSeek(d), Architecture::DeepSeek) => {
                BackendOptions::DeepSeek(d.clone())
            }
            (BackendOptions::Qwen, Architecture::Qwen) => BackendOptions::Qwen,
            (BackendOptions::Nemotron { draft_model }, Architecture::Nemotron) => {
                BackendOptions::Nemotron {
                    draft_model: draft_model.clone(),
                }
            }
            _ => return Err("model options do not match checkpoint architecture".into()),
        };
        Ok(())
    }
    pub fn deepseek(&self) -> Option<&DeepSeekOptions> {
        match &self.backend {
            BackendOptions::DeepSeek(d) => Some(d),
            _ => None,
        }
    }
    pub fn set_draft_model(&mut self, path: String) -> Result<(), String> {
        if matches!(self.backend, BackendOptions::Unspecified) {
            self.backend = BackendOptions::Nemotron { draft_model: None };
        }
        match &mut self.backend {
            BackendOptions::Nemotron { draft_model } => {
                *draft_model = Some(path);
                Ok(())
            }
            _ => Err("draft-model applies to Nemotron".into()),
        }
    }
    pub fn deepseek_mut(&mut self) -> Result<&mut DeepSeekOptions, String> {
        if matches!(self.backend, BackendOptions::Unspecified) {
            self.backend = BackendOptions::DeepSeek(DeepSeekOptions::default());
        }
        match &mut self.backend {
            BackendOptions::DeepSeek(d) => Ok(d),
            _ => Err("this option applies to DeepSeek".into()),
        }
    }
}

/// Logits of one row or more, valid until the next call that runs a model.
pub type Logits = *const f32;

/// A registered model whose weights may still be unloaded.
pub struct Model {
    raw: *mut RawModel,
    directory: String,
    draft_directory: Option<String>,
    validated: std::cell::Cell<bool>,
}

fn validate_weight_files(directory: &str) -> Result<(), String> {
    use std::io::Read;
    let root = std::path::Path::new(directory);
    let index = root.join("model.safetensors.index.json");
    let bytes = if index.exists() {
        std::fs::read(&index).map_err(|e| format!("model weight index is unavailable: {e}"))?
    } else {
        b"null".to_vec()
    };
    let json: serde_json::Value =
        serde_json::from_slice(&bytes).map_err(|e| format!("invalid model weight index: {e}"))?;
    let mut files = std::collections::BTreeMap::<&str, Vec<&str>>::new();
    if json.is_null() {
        files.insert("model.safetensors", Vec::new());
    } else {
        let weights = json
            .get("weight_map")
            .and_then(serde_json::Value::as_object)
            .filter(|m| !m.is_empty())
            .ok_or("model weight index has no weight_map")?;
        for (name, file) in weights {
            files
                .entry(file.as_str().ok_or("weight shard name is not text")?)
                .or_default()
                .push(name);
        }
    }
    for (name, tensors) in files {
        let mut file = std::fs::File::open(root.join(name))
            .map_err(|e| format!("model weight shard {name} is unavailable: {e}"))?;
        let size = file.metadata().map_err(|e| e.to_string())?.len();
        let mut prefix = [0u8; 8];
        file.read_exact(&mut prefix)
            .map_err(|e| format!("cannot read weight shard {name}: {e}"))?;
        let header = u64::from_le_bytes(prefix);
        if header > 64 * 1024 * 1024 || header > size.saturating_sub(8) {
            return Err(format!("invalid weight header in {name}"));
        }
        let mut data = vec![0; header as usize];
        file.read_exact(&mut data)
            .map_err(|e| format!("cannot read weight header {name}: {e}"))?;
        let values: serde_json::Value = serde_json::from_slice(&data)
            .map_err(|e| format!("invalid weight header {name}: {e}"))?;
        let tensors = if tensors.is_empty() {
            values
                .as_object()
                .ok_or("weight header is not an object")?
                .keys()
                .filter(|key| key.as_str() != "__metadata__")
                .map(String::as_str)
                .collect()
        } else {
            tensors
        };
        for tensor in tensors {
            let offsets = values
                .get(tensor)
                .and_then(|v| v.get("data_offsets"))
                .and_then(serde_json::Value::as_array)
                .filter(|v| v.len() == 2)
                .ok_or_else(|| format!("missing weight {tensor} in {name}"))?;
            let start = offsets[0].as_u64().ok_or("invalid weight offset")?;
            let end = offsets[1].as_u64().ok_or("invalid weight offset")?;
            if start > end || end > size - header - 8 {
                return Err(format!("weight {tensor} extends past {name}"));
            }
        }
    }
    Ok(())
}

impl Model {
    /// Registers metadata and tokenizer without loading weights.
    pub fn open(o: &Options) -> Option<Model> {
        let architecture = Architecture::of(&o.model).ok()?;
        let mut o = o.clone();
        if let Err(error) = o.configure(architecture) {
            log(&error);
            return None;
        }
        let model = c_string(&o.model);
        let draft_model = match &o.backend {
            BackendOptions::Nemotron { draft_model } => {
                draft_model.as_ref().map(|path| c_string(path))
            }
            _ => None,
        };
        let experts = o
            .deepseek()
            .and_then(|d| d.experts.as_deref())
            .map(c_string);
        let state = o
            .deepseek()
            .and_then(|d| d.predictor_state.as_deref())
            .map(c_string);
        let options = match &o.backend {
            BackendOptions::DeepSeek(d) => RawModelOptions {
                deepseek: RawDeepSeekConfig {
                    experts: experts.as_ref().map_or(ptr::null(), |s| s.as_ptr()),
                    cache_half_life: d.cache_half_life,
                    cache_read_cost: d.cache_read_cost.into(),
                    // the engine's approximate substitution stays off: a test of the C engine only
                    substitute: 0.0,
                    draft: o.draft,
                    draft_cache: d.draft_cache,
                    online_predictor: d.online_predictor.into(),
                    predictor_state: state.as_ref().map_or(ptr::null(), |s| s.as_ptr()),
                    ced: d.ced,
                    prefill_prune: d.prefill_prune.into(),
                    runahead: d.runahead.into(),
                    pipeline: d.pipeline.into(),
                },
            },
            BackendOptions::Qwen => RawModelOptions {
                qwen: RawQwenConfig { draft: o.draft },
            },
            BackendOptions::Nemotron { .. } => RawModelOptions {
                nemotron: RawNemotronConfig {
                    draft: o.draft,
                    draft_model: draft_model.as_ref().map_or(ptr::null(), |p| p.as_ptr()),
                },
            },
            BackendOptions::Unspecified => unreachable!("architecture resolved"),
        };
        let raw = RawConfig {
            abi: ABI,
            model: model.as_ptr(),
            architecture: architecture as c_int,
            ctx: o.ctx,
            lookup: o.lookup,
            options,
        };
        let m = unsafe { myds4_open(&raw) };
        (!m.is_null()).then_some(Model {
            raw: m,
            directory: o.model.clone(),
            draft_directory: match &o.backend {
                BackendOptions::Nemotron { draft_model } if o.draft > 0 => draft_model.clone(),
                _ => None,
            },
            validated: std::cell::Cell::new(false),
        })
    }

    pub fn load(&self) -> Result<(), String> {
        if !self.validated.get() {
            validate_weight_files(&self.directory)?;
            if let Some(directory) = &self.draft_directory {
                validate_weight_files(directory)?;
            }
            self.validated.set(true);
        }
        if unsafe { myds4_load(self.raw) != 0 } {
            Ok(())
        } else {
            Err("the model could not be loaded".into())
        }
    }
    pub fn active(&self, active: bool) {
        unsafe { myds4_active(self.raw, active.into()) };
    }

    pub fn loaded(&self) -> bool {
        unsafe { myds4_loaded(self.raw) != 0 }
    }

    /// After the last request: what the model saves or reports on exit.
    pub fn finish(&self) {
        unsafe { myds4_finish(self.raw) }
    }

    /// While the model runs no request: memory goes back to other programs if
    /// they need it; its expert cache's slots after it.
    pub fn idle(&self) -> i32 {
        unsafe { myds4_idle(self.raw) }
    }

    /// While the model runs no request: its expert cache and weights go back
    /// to the system, its KV states stay, and the next load reads them again.
    /// The bytes given back.
    pub fn unload(&self) -> u64 {
        unsafe { myds4_unload(self.raw) }
    }

    /// Rows of different states one decoding pass takes; the drafts per pass
    /// the model makes itself; the rows of one verify pass; the positions a
    /// state can continue before its end; the floats in a row of logits; the
    /// most rows one prefill call should take (None: the whole prompt).
    pub fn max_rows(&self) -> usize {
        unsafe { myds4_max_rows(self.raw) as usize }
    }
    pub fn draft_block(&self) -> usize {
        unsafe { myds4_draft_block(self.raw) as usize }
    }
    pub fn max_verify(&self) -> usize {
        unsafe { myds4_max_verify(self.raw) as usize }
    }
    pub fn rewind(&self) -> usize {
        unsafe { myds4_rewind(self.raw) as usize }
    }
    pub fn vocab(&self) -> usize {
        unsafe { myds4_vocab(self.raw) as usize }
    }
    pub fn prefill_rows(&self) -> Option<usize> {
        let n = unsafe { myds4_prefill_rows(self.raw) };
        (n > 0).then_some(n as usize)
    }
    /// Whether a token ends a reply.
    pub fn stop(&self, token: i32) -> bool {
        unsafe { myds4_stop(self.raw, token) != 0 }
    }

    // ---- states and snapshots (the engine thread) ----

    pub fn state_new(&self) -> *mut RawState {
        unsafe { myds4_state_new(self.raw) }
    }
    pub fn state_free(&self, s: *mut RawState) {
        unsafe { myds4_state_free(self.raw, s) }
    }
    /// The bytes a state (null: a new one) takes holding that many positions.
    pub fn state_bytes(&self, s: *const RawState, positions: usize) -> u64 {
        unsafe { myds4_state_bytes(self.raw, s, positions as c_int) as u64 }
    }
    /// src's rows before `positions` that later positions do not change, into dst.
    pub fn state_copy(&self, dst: *mut RawState, src: *const RawState, positions: usize) {
        unsafe { myds4_state_copy(self.raw, dst, src, positions as c_int) }
    }
    pub fn snapshot_take(
        &self,
        s: *mut RawState,
        pos: usize,
        logits: *const f32,
    ) -> *mut RawSnapshot {
        unsafe { myds4_snapshot_take(self.raw, s, pos as c_int, logits) }
    }
    pub fn snapshot_restore(&self, s: *mut RawState, x: *const RawSnapshot) {
        unsafe { myds4_snapshot_restore(self.raw, s, x) }
    }
    pub fn snapshot_free(&self, x: *mut RawSnapshot) {
        unsafe { myds4_snapshot_free(self.raw, x) }
    }
    pub fn snapshot_bytes(&self, with_logits: bool) -> u64 {
        unsafe { myds4_snapshot_bytes(self.raw, with_logits.into()) as u64 }
    }
    /// The logits a snapshot keeps (null without).
    pub fn snapshot_logits(&self, x: *const RawSnapshot) -> Logits {
        unsafe { myds4_snapshot_logits(self.raw, x) }
    }

    // ---- saved states (the engine thread; server/src/disk.rs) ----

    /// One part of a saved state: the rows of `s` that positions from `from`
    /// on wrote, up to `to`, into `fd`. The bytes written; None when a write
    /// failed or a second stop request ended it.
    pub fn state_write(&self, s: *mut RawState, from: usize, to: usize, fd: c_int) -> Option<u64> {
        let n = unsafe { myds4_state_write(self.raw, s, from as c_int, to as c_int, fd) };
        (n >= 0).then_some(n as u64)
    }
    /// Fills `s`, a new state, with the rows before `positions` from a saved
    /// state's parts, the first written first. False when a part is short,
    /// damaged or of another shape (`s` is then of no use).
    pub fn state_read(&self, s: *mut RawState, positions: usize, fds: &[c_int]) -> bool {
        unsafe { myds4_state_read(self.raw, s, positions as c_int, fds.as_ptr(), fds.len() as c_int) == 0 }
    }
    /// Snapshot `x`, or (null) the rolling rows of `s` at its end `pos`,
    /// without logits, into `fd`. The bytes written, or None.
    pub fn snapshot_write(
        &self,
        x: *const RawSnapshot,
        s: *mut RawState,
        pos: usize,
        fd: c_int,
    ) -> Option<u64> {
        let n = unsafe { myds4_snapshot_write(self.raw, x, s, pos as c_int, fd) };
        (n >= 0).then_some(n as u64)
    }
    /// A snapshot that snapshot_write wrote; null when the file is short,
    /// damaged or of another shape.
    pub fn snapshot_read(&self, fd: c_int) -> *mut RawSnapshot {
        unsafe { myds4_snapshot_read(self.raw, fd) }
    }
    /// What this model's saved states depend on (engine.h).
    pub fn state_identity(&self) -> [u8; 32] {
        let mut out = [0u8; 32];
        unsafe { myds4_state_identity(self.raw, out.as_mut_ptr()) };
        out
    }

    // ---- running the model (the engine thread) ----

    /// What a replay at k computes again before k, rows per layer on average;
    /// None when the model cannot replay.
    pub fn replay_rows(&self, k: usize) -> Option<usize> {
        let n = unsafe { myds4_replay_rows(self.raw, k as c_int) };
        (n >= 0).then_some(n as usize)
    }
    /// The first position a replay at k computes again: its media from there are encoded.
    pub fn replay_start(&self, k: usize) -> usize {
        unsafe { myds4_replay_start(self.raw, k as c_int).max(0) as usize }
    }
    pub fn seek(&self, s: *mut RawState, tokens: &[i32], pos: usize) {
        unsafe { myds4_seek(self.raw, s, tokens.as_ptr(), pos as c_int) }
    }

    // ---- sampling, media, statistics ----

    pub fn sample(&self, lg: Logits, s: &mut Sampler) -> i32 {
        unsafe { myds4_sample(self.raw, lg, s.0) }
    }

    /// A prompt's decoded media for this model's encoder.
    pub fn media(&self, media: &[Vec<u8>]) -> Result<Media, String> {
        let data: Vec<*const u8> = media.iter().map(|i| i.as_ptr()).collect();
        let len: Vec<usize> = media.iter().map(|i| i.len()).collect();
        let mut message = [0 as c_char; 256];
        let x = unsafe {
            myds4_media_open(
                self.raw,
                data.as_ptr(),
                len.as_ptr(),
                media.len() as c_int,
                message.as_mut_ptr(),
                message.len(),
            )
        };
        if x.is_null() {
            return Err(unsafe { std::ffi::CStr::from_ptr(message.as_ptr()) }
                .to_string_lossy()
                .into_owned());
        }
        Ok(Media(x))
    }

    /// Called only by the engine worker; the worker publishes copies for readers.
    pub fn stats(&self) -> Stats {
        let mut stats = Stats::default();
        unsafe { myds4_stats_get(self.raw, &mut stats) };
        stats
    }

    /// The experts row `row` of the latest decoding pass chose, `per_row` of
    /// them, and where each one was; None for a model without routed experts
    /// or a row that did not run every layer. The engine thread.
    pub fn routes(&self, row: usize, per_row: usize) -> Option<(Vec<u16>, Vec<u8>)> {
        let (mut experts, mut sources) = (vec![0u16; per_row], vec![0u8; per_row]);
        // Safety: the engine writes per_row (experts_per_row) entries of each.
        (unsafe {
            myds4_routes(
                self.raw,
                row as c_int,
                experts.as_mut_ptr(),
                sources.as_mut_ptr(),
            )
        } != 0)
            .then_some((experts, sources))
    }

    /// The model's tokenizer.
    pub fn tokenizer(&self) -> Tokenizer {
        Tokenizer(unsafe { myds4_model_tokenizer(self.raw) })
    }
}

/// A checkpoint's tokenizer: a model's own, or one opened from a checkpoint
/// directory without its model. It lives while the process does; any thread.
#[derive(Clone, Copy)]
pub struct Tokenizer(*mut RawTokenizer);

// Safety: each engine tokenizer has its own lock; it is never freed.
unsafe impl Send for Tokenizer {}
unsafe impl Sync for Tokenizer {}

impl Tokenizer {
    /// The tokenizer of a checkpoint directory, with `image` the spelling of
    /// its model's image token (or none); one the engine cannot run ends the
    /// process with a message. (The format tests read it without a model.)
    #[cfg(test)]
    pub fn open(dir: &str, image: Option<&str>) -> Tokenizer {
        let dir = c_string(dir);
        let image = image.map(c_string);
        Tokenizer(unsafe {
            myds4_tokenizer_open(
                dir.as_ptr(),
                image.as_ref().map_or(ptr::null(), |s| s.as_ptr()),
            )
        })
    }

    fn tokenize_as(&self, text: &[u8], mode: c_int) -> Vec<i32> {
        let mut out = vec![0i32; text.len() + 2];
        let n = unsafe {
            myds4_tokenize(
                self.0,
                text.as_ptr().cast(),
                text.len(),
                mode,
                out.as_mut_ptr(),
                out.len() as c_int,
            )
        };
        out.truncate(n.max(0) as usize);
        out
    }
    /// Text alone: the added tokens' spellings in it stay text.
    pub fn tokenize_text(&self, text: &str) -> Vec<i32> {
        self.tokenize_as(text.as_bytes(), 0)
    }
    /// Text in which the added tokens' spellings are those tokens.
    pub fn tokenize(&self, text: &str) -> Vec<i32> {
        self.tokenize_as(text.as_bytes(), 1)
    }
    /// A rendered prompt: the added tokens' spellings are those tokens, the
    /// byte 0xFF an image's place.
    pub fn tokenize_prompt(&self, prompt: &[u8]) -> Vec<i32> {
        self.tokenize_as(prompt, 2)
    }
    /// The id of an added token's spelling.
    pub fn special(&self, text: &str) -> Option<i32> {
        let id = unsafe { myds4_special(self.0, text.as_ptr().cast(), text.len()) };
        (id >= 0).then_some(id)
    }
    /// Whether a token is one of the tokenizer's added tokens.
    pub fn added(&self, id: i32) -> bool {
        unsafe { myds4_added(self.0, id) != 0 }
    }
    /// Whether a token is an added token that tokenizer.json marks special
    /// (one a reply's text leaves out, such as a vision marker).
    pub fn marked_special(&self, id: i32) -> bool {
        unsafe { myds4_added(self.0, id) == 2 }
    }
    /// A token's bytes; none for an unknown id.
    pub fn piece(&self, id: i32) -> &'static [u8] {
        let mut n = 0usize;
        // Safety: myds4_piece gives a token's bytes, length 0 for an unknown
        // id; the tokenizer and its bytes stay loaded while the process runs.
        let p = unsafe { myds4_piece(self.0, id, &mut n) };
        if p.is_null() || n == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(p.cast::<u8>(), n) }
        }
    }

    /// The text of token ids, for the live view, in runs: the added tokens
    /// each alone and marked true, the text between them, and one "[image]"
    /// for each run of an image's key (the negative ids of [`Media::expand`]).
    pub fn text(&self, ids: &[i32]) -> Vec<(String, bool)> {
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
            } else if self.added(id) {
                flush(&mut bytes, &mut runs);
                runs.push((String::from_utf8_lossy(self.piece(id)).into_owned(), true));
            } else {
                bytes.extend_from_slice(self.piece(id));
            }
            last = id;
        }
        flush(&mut bytes, &mut runs);
        runs
    }
}

/// A tokenizer for recipe's stream processor: a token's text is its bytes,
/// and the special tokens are the tokenizer's added tokens.
impl StreamTokenizer for Tokenizer {
    fn decode_ids(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String, String> {
        let mut bytes = Vec::new();
        for &id in ids {
            if skip_special_tokens && self.is_special_token(id) {
                continue;
            }
            bytes.extend_from_slice(self.piece(id as i32));
        }
        // An unfinished character at the end becomes U+FFFD, so the processor
        // waits for the token that finishes it.
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    fn count_tokens(&self, text: &str) -> Option<usize> {
        Some(self.tokenize(text).len())
    }

    fn is_special_token(&self, id: u32) -> bool {
        self.added(id as i32)
    }
}

// ---- samplers, lookups and media ----

pub struct Sampler(*mut RawSampler);

impl Sampler {
    /// None for invalid settings; top_k 0 takes every token.
    pub fn new(temperature: f64, top_p: f64, top_k: u32, seed: Option<u64>) -> Option<Sampler> {
        let s = unsafe {
            myds4_sampler_new(
                temperature,
                top_p,
                top_k as c_int,
                seed.unwrap_or(0),
                seed.is_some().into(),
            )
        };
        (!s.is_null()).then_some(Sampler(s))
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
        unsafe {
            myds4_lookup_propose(self.0, max.min(drafts.len()) as c_int, drafts.as_mut_ptr()).max(0)
                as usize
        }
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

/// How many of the draft model's drafts each pass of one request runs: the
/// count with the most expected tokens per second, from the drafts kept and
/// the times of earlier passes (src/model/draft.h). The outputs do not
/// depend on it.
pub struct DraftChoice(*mut RawDraftChoice);

impl DraftChoice {
    pub fn new() -> DraftChoice {
        DraftChoice(unsafe { myds4_draft_choice_new() })
    }
    /// 0 to `most`.
    pub fn count(&mut self, most: usize) -> usize {
        unsafe { myds4_draft_choice_count(self.0, most.min(c_int::MAX as usize) as c_int).max(0) as usize }
    }
    /// A pass with `count` drafts kept the first `kept` of them and took
    /// `seconds`, with no other call beside it.
    pub fn feedback(&mut self, count: usize, kept: usize, seconds: f64) {
        unsafe { myds4_draft_choice_feedback(self.0, count as c_int, kept as c_int, seconds) }
    }
}

impl Drop for DraftChoice {
    fn drop(&mut self) {
        unsafe { myds4_draft_choice_free(self.0) }
    }
}

/// A prompt's media, decoded; closed (and given back) when dropped.
pub struct Media(*mut RawMedia);

impl Media {
    /// The prompt with each image place replaced by its image's span, or None
    /// if the places and media differ or it would exceed `max` tokens.
    pub fn expand(&mut self, prompt: &[i32], max: usize) -> Option<Vec<i32>> {
        let mut out = vec![0i32; max + 2];
        let n = unsafe {
            myds4_media_expand(
                self.0,
                prompt.as_ptr(),
                prompt.len() as c_int,
                out.as_mut_ptr(),
                max as c_int,
            )
        };
        (n >= 0).then(|| {
            out.truncate(n as usize);
            out
        })
    }
    /// Encodes the media that positions from `from` reach, for the next
    /// prefill; how many.
    pub fn encode(&mut self, from: usize) -> Result<usize, String> {
        let count = unsafe { myds4_media_encode(self.0, from as c_int) };
        if count < 0 {
            Err(
                "the global model-memory budget cannot fit the image encoder and its workspace"
                    .into(),
            )
        } else {
            Ok(count as usize)
        }
    }
}

impl Drop for Media {
    fn drop(&mut self) {
        unsafe { myds4_media_close(self.0) }
    }
}

#[cfg(test)]
mod ownership_tests {
    use super::*;

    #[test]
    fn model_handles_cannot_cross_threads() {
        trait NotSend<Marker> {
            fn check() {}
        }
        impl<T: ?Sized> NotSend<()> for T {}
        struct SendType;
        impl<T: ?Sized + Send> NotSend<SendType> for T {}
        let _ = <Model as NotSend<_>>::check;

        trait NotSync<Marker> {
            fn check() {}
        }
        impl<T: ?Sized> NotSync<()> for T {}
        struct SyncType;
        impl<T: ?Sized + Sync> NotSync<SyncType> for T {}
        let _ = <Model as NotSync<_>>::check;
    }
}

/// One model's calls, one after another, on their own stack and working
/// buffers (myds4_stream): the calls of several streams run at the same time.
/// A stream's call runs when `run` advances it; what the call was given stays
/// valid until it finishes.
pub struct Stream(*mut RawStream);

impl Stream {
    pub fn new(model: &Model) -> Stream {
        Stream(unsafe { myds4_stream_new(model.raw) })
    }
    pub fn raw(&self) -> *mut RawStream {
        self.0
    }
    pub fn prefill(
        &self,
        s: *mut RawState,
        tokens: &[i32],
        from: usize,
        replay: bool,
        capture: Option<usize>,
        media: Option<&Media>,
    ) {
        unsafe {
            myds4_stream_prefill(
                self.0,
                s,
                tokens.as_ptr(),
                tokens.len() as c_int,
                from as c_int,
                replay.into(),
                capture.map_or(-1, |p| p as c_int),
                media.map_or(ptr::null(), |media| media.0),
            )
        }
    }
    pub fn rows(&self, states: &mut [*mut RawState], tokens: &[i32], pos: &[i32]) {
        unsafe {
            myds4_stream_rows(
                self.0,
                states.as_mut_ptr(),
                tokens.as_ptr(),
                pos.as_ptr(),
                states.len() as c_int,
            )
        }
    }
    /// The token rows[0] at pos and its drafts, verified together; the model's
    /// drafts fill rows[1..] unless the caller proposed them.
    pub fn verify(&self, s: *mut RawState, rows: &mut [i32], proposed: bool, pos: usize, sampler: &mut Sampler) {
        unsafe {
            myds4_stream_verify(
                self.0,
                s,
                rows.as_mut_ptr(),
                rows.len() as c_int - 1,
                proposed.into(),
                pos as c_int,
                sampler.0,
            )
        }
    }
    /// The finished call's logits; None: it did not fit in memory.
    pub fn logits(&self) -> Option<Logits> {
        let lg = unsafe { myds4_stream_logits(self.0) };
        (!lg.is_null()).then_some(lg)
    }
    pub fn captured(&self) -> *mut RawSnapshot {
        unsafe { myds4_stream_captured(self.0) }
    }
    pub fn info(&self) -> PrefillInfo {
        let mut info = PrefillInfo::default();
        unsafe { myds4_stream_info(self.0, &mut info) };
        info
    }
    /// A verify call's next token and the drafts it kept.
    pub fn next(&self) -> (i32, usize) {
        let mut accepted = 0;
        let token = unsafe { myds4_stream_next(self.0, &mut accepted) };
        (token, accepted.max(0) as usize)
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        unsafe { myds4_stream_free(self.0) }
    }
}

/// Advances the streams until a call finishes or `seconds` pass: that stream's
/// index, or None.
pub fn run(streams: &[*mut RawStream], seconds: f64) -> Option<usize> {
    let i = unsafe { myds4_run(streams.as_ptr(), streams.len() as c_int, seconds) };
    (i >= 0).then_some(i as usize)
}
