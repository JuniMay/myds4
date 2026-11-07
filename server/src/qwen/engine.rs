//! The Qwen3.8 engine's C ABI (src/qwen/qwen.h) and thin safe wrappers. The
//! model's functions run on the Qwen worker's thread only (worker.rs).

use std::ffi::{CString, c_char, c_int};

/// Rows one decoding pass takes at most (QWEN_ROWS), and one prompt pass
/// (QWEN_PREFILL).
pub const ROWS: usize = 8;
pub const PREFILL: usize = 512;

#[repr(C)]
pub struct RawState {
    _opaque: [u8; 0],
}

#[repr(C)]
pub struct RawSnap {
    _opaque: [u8; 0],
}

#[repr(C)]
struct RawRow {
    state: *mut RawState,
    token: i32,
    save: c_int,
}

unsafe extern "C" {
    fn qwen_load(dir: *const c_char);
    fn qwen_vocab() -> c_int;
    fn qwen_stop(token: c_int) -> c_int;
    fn qwen_state_new() -> *mut RawState;
    fn qwen_state_free(s: *mut RawState);
    fn qwen_state_pos(s: *const RawState) -> c_int;
    fn qwen_state_bytes(positions: c_int) -> u64;
    fn qwen_state_reserve(s: *mut RawState, positions: c_int);
    fn qwen_state_copy(dst: *mut RawState, src: *const RawState, positions: c_int);
    fn qwen_forward(rows: *const RawRow, n: c_int, logits: c_int) -> *const f32;
    fn qwen_prefill(rows: *const RawRow, n: c_int, logits: c_int) -> *const f32;
    fn qwen_mtp_available() -> c_int;
    fn qwen_mtp(rows: *const RawRow, n: c_int, logits: c_int) -> *const f32;
    fn qwen_state_back(s: *mut RawState);
    fn qwen_snap_take(s: *const RawState) -> *mut RawSnap;
    fn qwen_snap_restore(s: *mut RawState, snap: *const RawSnap);
    fn qwen_snap_free(snap: *mut RawSnap);
    fn qwen_snap_bytes() -> u64;
    fn qwen_reside(on: c_int);
    fn shutdown_watch();
}

/// Loads the checkpoint (dies on failure, as the C engine does), and has
/// SIGINT and SIGTERM request a stop.
pub fn load(dir: &str) {
    let dir = CString::new(dir).expect("a path without NUL");
    unsafe {
        shutdown_watch();
        qwen_load(dir.as_ptr());
    }
}

/// Takes the engine's buffers out of the GPU's residency set (`false`), or
/// puts them back; the engine puts them back itself before its next work.
pub fn reside(on: bool) {
    unsafe { qwen_reside(on as c_int) }
}

pub fn vocab() -> usize {
    unsafe { qwen_vocab() as usize }
}

pub fn is_stop(token: u32) -> bool {
    unsafe { qwen_stop(token as c_int) != 0 }
}

pub fn snap_bytes() -> u64 {
    unsafe { qwen_snap_bytes() }
}

pub fn state_bytes(positions: usize) -> u64 {
    unsafe { qwen_state_bytes(positions as c_int) }
}

/// A sequence's state (qwen_state_new), freed when dropped.
pub struct State(*mut RawState);

// The worker's thread alone uses states.
unsafe impl Send for State {}

impl State {
    pub fn new() -> State {
        State(unsafe { qwen_state_new() })
    }
    pub fn pos(&self) -> usize {
        unsafe { qwen_state_pos(self.0) as usize }
    }
    pub fn reserve(&mut self, positions: usize) {
        unsafe { qwen_state_reserve(self.0, positions as c_int) }
    }
    /// Takes `src`'s rows before `positions` (and its recurrent state, which
    /// a snapshot restored next replaces unless `positions` is src's end).
    pub fn copy_from(&mut self, src: &State, positions: usize) {
        unsafe { qwen_state_copy(self.0, src.0, positions as c_int) }
    }
    /// Returns to the position after the last saved row.
    pub fn back(&mut self) {
        unsafe { qwen_state_back(self.0) }
    }
    pub fn snap(&self) -> Snap {
        Snap(unsafe { qwen_snap_take(self.0) })
    }
    pub fn restore(&mut self, snap: &Snap) {
        unsafe { qwen_snap_restore(self.0, snap.0) }
    }
}

impl Drop for State {
    fn drop(&mut self) {
        unsafe { qwen_state_free(self.0) }
    }
}

/// A snapshot of a state's recurrent part (qwen_snap_take), freed when dropped.
pub struct Snap(*mut RawSnap);

unsafe impl Send for Snap {}

impl Drop for Snap {
    fn drop(&mut self) {
        unsafe { qwen_snap_free(self.0) }
    }
}

/// A row of a pass: the state (one of the worker's live states; a state may
/// stand for several rows, which a slice of `&mut` could not), its token,
/// and whether the state keeps a copy of itself after the row (for
/// [`State::back`]; in an MTP row: whether the row takes the saved row's
/// hidden).
#[derive(Clone, Copy)]
pub struct Row {
    pub state: *mut State,
    pub token: u32,
    pub save: bool,
}

fn raw(rows: &[Row]) -> Vec<RawRow> {
    rows.iter().map(|r| RawRow { state: unsafe { (*r.state).0 }, token: r.token as i32, save: r.save as c_int }).collect()
}

fn logits_of(p: *const f32, logits: usize) -> Vec<f32> {
    if logits == 0 || p.is_null() {
        return Vec::new();
    }
    unsafe { std::slice::from_raw_parts(p, logits * vocab()) }.to_vec()
}

/// A decoding pass (at most [`ROWS`] rows; a state's rows consecutive, at its
/// next positions): the logits of the last `logits` rows, `vocab()` floats
/// each. Each row's result is the same for any number of rows.
pub fn forward(rows: &[Row], logits: usize) -> Vec<f32> {
    assert!(!rows.is_empty() && rows.len() <= ROWS && logits <= rows.len());
    let r = raw(rows);
    logits_of(unsafe { qwen_forward(r.as_ptr(), r.len() as c_int, logits as c_int) }, logits)
}

/// A prompt pass (at most [`PREFILL`] rows; logits for at most [`ROWS`]):
/// each row's result is the same however a prompt is split into passes.
pub fn prefill(rows: &[Row], logits: usize) -> Vec<f32> {
    assert!(!rows.is_empty() && rows.len() <= PREFILL && logits <= rows.len().min(ROWS));
    let r = raw(rows);
    logits_of(unsafe { qwen_prefill(r.as_ptr(), r.len() as c_int, logits as c_int) }, logits)
}

pub fn mtp_available() -> bool {
    unsafe { qwen_mtp_available() != 0 }
}

/// The MTP head over rows (token, the state's last hidden or with `save` its
/// saved row's): their keys, and the draft logits of the last `logits` rows.
pub fn mtp(rows: &[Row], logits: usize) -> Vec<f32> {
    assert!(!rows.is_empty() && rows.len() <= ROWS && logits <= rows.len());
    let r = raw(rows);
    logits_of(unsafe { qwen_mtp(r.as_ptr(), r.len() as c_int, logits as c_int) }, logits)
}
