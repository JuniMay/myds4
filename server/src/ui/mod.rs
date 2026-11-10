//! The live view (--tui): the alternate screen of stderr's terminal, from the
//! worker's snapshot ([`crate::worker::view`]) twice a second, in six tabs:
//!
//! 1. Overview: the requests and their decoding speed, the throughput over
//!    the last 5 s, the last minute and since the start, the experts per
//!    decoded token (with what substitution adds), memory on the GPU, in RAM
//!    and on the SSD, and the latest log lines.
//! 2. KV cache: the KV prefix cache's tree of states along the token
//!    positions, and the text of the node chosen with the keys or a click;
//!    with several models, every model's sessions above it.
//! 3. Experts: a decoding request's experts in every layer for its last
//!    sampled tokens, and where each one was.
//! 4. Last minute: the phase, the tokens/s, the SSD's reads, the hit rate and
//!    each request, one column per snapshot.
//! 5. Log: the latest log lines.
//! 6. Memory: the Mac's memory as a treemap, each model with its weights,
//!    experts, workspace and KV beside the server's other memory, other
//!    programs, the file cache and free memory; the models' limit, the shared
//!    KV budget and the reserve.
//!
//! What the tabs draw of the last minute stays in the view's memory; nothing
//! is written to files. On exit the normal screen comes back with the last log
//! lines. The q key, or Ctrl-C, stops the server as SIGINT does.
//!
//! The engine's log reaches the view through [`on_log`]; lines logged while
//! the view is not on the screen go to stderr as well. While the view is up,
//! stderr is a pipe ([`capture`]): what anything else writes there, such as
//! the engine's lines about a model's load, reaches the log with its time, and
//! the view alone writes to the terminal. [`leave`] restores the terminal and
//! stderr on every way out: the normal end, the engine's fatal errors
//! ([`on_die`]), this program's own and a panic.

use std::collections::VecDeque;
use std::ffi::{CStr, c_char};
use std::fs::File;
use std::io::{self, BufRead, BufReader, Write};
use std::os::fd::{AsFd, AsRawFd, FromRawFd, OwnedFd};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, TryLockError, mpsc};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use ratatui::buffer::Buffer;
use ratatui::crossterm::event::{
    self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers, MouseButton, MouseEventKind,
};
use ratatui::crossterm::terminal::{self, EnterAlternateScreen, LeaveAlternateScreen};
use ratatui::crossterm::{cursor, execute};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Paragraph, Widget};
use serde::Deserialize;
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

use crate::engine;

mod experts;
mod frame;
mod history;
mod kv;
mod memory;
mod minute;
mod overview;
mod theme;

use frame::{Answer, Tab, Target, Text, Ui, label, strong, text};
use history::History;
use theme::{c, get};

const RING: usize = 256; // log lines kept
const TAIL: u64 = 30; // log lines printed on exit
const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

// ---- the worker's snapshot (worker::view) ----

/// What the worker does; a phase this view does not know reads as idle.
#[derive(Deserialize, Default, Clone, Copy, PartialEq, Eq, Debug)]
#[serde(rename_all = "lowercase")]
pub enum Phase {
    Prefill,
    Decode,
    #[default]
    #[serde(other)]
    Idle,
}

#[derive(Deserialize, Default)]
pub struct Snapshot {
    /// The model shown, and how many the server serves.
    #[serde(default)]
    model: String,
    #[serde(default)]
    models: usize,
    uptime_seconds: f64,
    phase: Phase,
    phase_seconds: f64,
    waiting: u64,
    requests: Vec<Req>,
    layers: u64,
    experts_per_row: u64,
    prefill: Prefill,
    totals: Totals,
    kinds: Kinds,
    sessions: u64,
    kv_gib: f64,
    kv_limit_gib: f64,
    #[serde(default)]
    kv_total_gib: f64,
    #[serde(default)]
    reserve_gib: f64,
    /// The memory setting, an upper limit; 0: none.
    #[serde(default)]
    memory_gib: f64,
    #[serde(default)]
    model_memory: Vec<ModelMemory>,
    #[serde(default)]
    model_memory_budget: ModelBudget,
    tree: Vec<Node>,
    decode: Decode,
    prompts: Prompts,
    #[serde(default)]
    prefilled: Prefilled,
    cache: Cache,
    memory: Memory,
}

#[derive(Deserialize, Default)]
struct ModelBudget {
    limit: u64,
    allocated: u64,
}

#[derive(Deserialize, Default)]
struct ModelMemory {
    model: String,
    weights: u64,
    experts: u64,
    #[serde(default)]
    workspace: u64,
    kv: u64,
    kv_committed: u64,
    /// The model's sessions in the shared KV cache.
    #[serde(default)]
    sessions: u64,
}

#[derive(Deserialize, Default)]
struct Req {
    number: u64,
    session: u64,
    prompt: u64,
    reused: u64,
    generated: u64,
    /// Its routed row of a decode step since the snapshot before, if one ran.
    #[serde(default)]
    route: Option<Routed>,
}

/// A decoded row's experts, layer by layer (experts_per_row of them), and
/// where each was: 0 in memory, 1 read ahead, 2 late, 3 substituted, 4 from
/// the SSD.
#[derive(Deserialize, Default, Clone)]
struct Routed {
    experts: Vec<u16>,
    sources: Vec<u8>,
}

#[derive(Deserialize, Default)]
struct Prefill {
    layer: u64,
    rows: u64,
}

#[derive(Deserialize, Default)]
struct Totals {
    requests: u64,
    prompt_tokens: u64,
    reused_tokens: u64,
    generated_tokens: u64,
}

#[derive(Deserialize, Default)]
struct Kinds {
    new: u64,
    continued: u64,
    saved: u64,
    forked: u64,
}

/// A run of positions that a set of sessions share: the sessions' prefix tree.
#[derive(Deserialize, Default)]
struct Node {
    lines: String, // the tree's lines before it (box drawing)
    start: u64,
    end: u64,
    sessions: u64, // sessions through it
    hits: u64,     // requests that reused some of it
    session: u64,  // the session ending here, 0 for a shared prefix
    #[serde(default)]
    via: u64, // a session through it, whose tokens hold its text
    busy: bool,
    saved: i64, // that session's saved position, -1 for none
    idle_seconds: f64,
    gib: f64,
}

/// The statistics of decoded rows since the start: each row's selections by
/// where the expert was.
#[derive(Deserialize, Default, Clone, Copy, Debug)]
struct Decode {
    /// Rows computed: a token each, and each draft a verification checks.
    rows: u64,
    /// The tokens generated: by the requests that ended and by those decoding.
    #[serde(default)]
    tokens: u64,
    seconds: f64,
    gpu_seconds: f64,
    in_memory: u64,
    read_ahead: u64,
    late: u64,
    substituted: u64,
    from_ssd: u64,
    bytes: u64,
    read_wait_seconds: f64,
    prefetches: u64,
    cancelled: u64,
    remaps: u64,
}

impl Decode {
    /// The counts since `b`, an earlier snapshot's.
    fn since(&self, b: &Decode) -> Decode {
        let d = |x: u64, y: u64| x.saturating_sub(y);
        Decode {
            rows: d(self.rows, b.rows),
            tokens: d(self.tokens, b.tokens),
            seconds: self.seconds - b.seconds,
            gpu_seconds: self.gpu_seconds - b.gpu_seconds,
            in_memory: d(self.in_memory, b.in_memory),
            read_ahead: d(self.read_ahead, b.read_ahead),
            late: d(self.late, b.late),
            substituted: d(self.substituted, b.substituted),
            from_ssd: d(self.from_ssd, b.from_ssd),
            bytes: d(self.bytes, b.bytes),
            read_wait_seconds: self.read_wait_seconds - b.read_wait_seconds,
            prefetches: d(self.prefetches, b.prefetches),
            cancelled: d(self.cancelled, b.cancelled),
            remaps: d(self.remaps, b.remaps),
        }
    }

    /// The rows' experts by where each one was, in the theme's order of
    /// sources (experts::SOURCE_NAMES).
    fn by_source(&self) -> [u64; 5] {
        [
            self.in_memory,
            self.read_ahead,
            self.late,
            self.substituted,
            self.from_ssd,
        ]
    }

    /// Of the rows' experts, the share ready when needed: in memory, read
    /// ahead or late.
    fn hit(&self) -> Option<f64> {
        let x = self.by_source();
        let all: u64 = x.iter().sum();
        (all > 0).then(|| (x[0] + x[1] + x[2]) as f64 / all as f64)
    }

    /// The GPU's busy share of the decode steps' time.
    fn gpu_busy(&self) -> Option<f64> {
        (self.seconds > 0.0).then(|| (self.gpu_seconds / self.seconds).clamp(0.0, 1.0))
    }

    /// The wait for expert reads, ms per row.
    fn wait_ms(&self) -> Option<f64> {
        (self.rows > 0).then(|| self.read_wait_seconds.max(0.0) / self.rows as f64 * 1e3)
    }
}

#[derive(Deserialize, Default)]
struct Prompts {
    rows: u64,
    hits: u64,
    prefetch_hits: u64,
    misses: u64,
    substituted: u64,
    bytes: u64,
}

/// The ended prefills: their time, and the prompt rows and expert bytes the
/// engine had counted when the latest ended, taken together.
#[derive(Deserialize, Default)]
struct Prefilled {
    rows: u64,
    bytes: u64,
    seconds: f64,
}

#[derive(Deserialize, Default)]
struct Cache {
    slots: u64,
    mapped: u64,
    cap: u64,
    max_slots: u64,
    slot_bytes: u64,
    experts: u64,
    uncached: u64,
    uncached_bytes: u64,
    evictions: u64,
    given_back: u64,
    /// Every expert byte read since the start, counted as each read completes.
    #[serde(default)]
    read_bytes: u64,
}

#[derive(Deserialize, Default)]
struct Memory {
    footprint: u64,
    dense: u64,
    gpu_in_use: u64,
    gpu_limit: Option<u64>,
    total: u64,
    free: u64,
    file: u64,
    file_kept: u64,
    wired: u64,
    compressor: u64,
    compressed: u64,
    swap_used: u64,
    swapouts: u64,
    pressure: i64,
}

// ---- the log ----

struct Ring {
    lines: VecDeque<String>,
    logged: u64,  // lines logged since the start
    printed: u64, // of them, the first ones already on stderr
}

static RING_LINES: Mutex<Ring> = Mutex::new(Ring {
    lines: VecDeque::new(),
    logged: 0,
    printed: 0,
});
static ON: AtomicBool = AtomicBool::new(false); // the alternate screen is up
static RAW: AtomicBool = AtomicBool::new(false); // the terminal is in raw mode for the keys
static MOUSE: AtomicBool = AtomicBool::new(false); // the terminal reports the mouse to the view
static DRAW: Mutex<()> = Mutex::new(()); // held while a frame is written
static CAPTURE: Mutex<Option<Capture>> = Mutex::new(None); // while stderr is the pipe

/// Stderr as a pipe into the log: the terminal it was, to give it back, and
/// the end of the thread that reads the pipe.
struct Capture {
    terminal: OwnedFd,
    done: mpsc::Receiver<()>,
}

fn ring() -> MutexGuard<'static, Ring> {
    RING_LINES.lock().unwrap_or_else(|e| e.into_inner())
}

/// A line into the log; the oldest goes when the ring is full.
fn keep(r: &mut Ring, line: String) {
    r.logged += 1;
    if r.lines.len() == RING {
        r.lines.pop_front();
    }
    r.lines.push_back(line);
}

/// The engine's log lines (with their time), from the thread that logs.
pub extern "C" fn on_log(line: *const c_char) {
    // Safety: the engine passes a NUL-terminated line, valid during the call.
    let line = unsafe { CStr::from_ptr(line) }
        .to_string_lossy()
        .into_owned();
    let mut r = ring();
    if !ON.load(Ordering::Acquire) {
        eprintln!("{line}");
        r.printed = r.logged + 1;
    }
    keep(&mut r, line);
}

fn capture_state() -> MutexGuard<'static, Option<Capture>> {
    CAPTURE.lock().unwrap_or_else(|e| e.into_inner())
}

/// Points stderr at a pipe whose lines [`collect`] puts in the log, and
/// returns the terminal stderr was, for the view to draw on. Without a pipe,
/// stderr stays as it is and the result is None.
fn capture() -> Option<File> {
    let terminal = io::stderr().as_fd().try_clone_to_owned().ok()?;
    let view = File::from(terminal.try_clone().ok()?);
    let mut ends = [0; 2];
    // Safety: pipe fills both descriptors when it succeeds.
    if unsafe { libc::pipe(ends.as_mut_ptr()) } != 0 {
        return None;
    }
    // Safety: both descriptors are new and owned here.
    let (read, write) = unsafe { (OwnedFd::from_raw_fd(ends[0]), OwnedFd::from_raw_fd(ends[1])) };
    for end in [&read, &write] {
        // Safety: a descriptor owned here; a program this one starts gets neither end.
        unsafe { libc::fcntl(end.as_raw_fd(), libc::F_SETFD, libc::FD_CLOEXEC) };
    }
    let (finished, done) = mpsc::channel();
    thread::Builder::new()
        .name("stderr".into())
        .spawn(move || {
            collect(File::from(read));
            let _ = finished.send(());
        })
        .ok()?;
    // Safety: descriptor 2 becomes the pipe's writing end, which then stays
    // open through it alone.
    if unsafe { libc::dup2(write.as_raw_fd(), 2) } < 0 {
        return None;
    }
    *capture_state() = Some(Capture { terminal, done });
    Some(view)
}

/// Gives stderr its terminal back, once what was written to the pipe is in
/// the log. Any thread may call it, any number of times.
fn release() {
    let Some(c) = capture_state().take() else {
        return;
    };
    // Safety: descriptor 2 becomes the terminal again. The pipe's writing end
    // closes with it, so the reader comes to the end of what was written.
    unsafe { libc::dup2(c.terminal.as_raw_fd(), 2) };
    let _ = c.done.recv_timeout(Duration::from_millis(500));
}

/// Each line from the pipe into the log, with its time. Of a line that
/// carriage returns rewrite, the last version, as a terminal shows it.
fn collect(pipe: File) {
    let mut lines = BufReader::new(pipe);
    let mut line = Vec::new();
    while matches!(lines.read_until(b'\n', &mut line), Ok(n) if n > 0) {
        let text = String::from_utf8_lossy(&line);
        let shown = text
            .trim_end_matches(['\n', '\r'])
            .rsplit('\r')
            .next()
            .unwrap_or("");
        if !shown.trim().is_empty() {
            engine::log(shown);
        }
        line.clear();
    }
}

/// Before the engine's fatal error message: the normal screen comes back.
pub extern "C" fn on_die() {
    leave();
}

/// Stderr on its terminal again, the normal screen, the terminal's settings
/// and the log lines it has not shown yet. Any thread may call it, any number
/// of times.
pub fn leave() {
    // A frame being written finishes first; a thread that fails while it
    // writes one (a panic) holds the lock, so the wait is bounded.
    let mut draw = None;
    for _ in 0..200 {
        match DRAW.try_lock() {
            Ok(g) => {
                draw = Some(g);
                break;
            }
            Err(TryLockError::Poisoned(e)) => {
                draw = Some(e.into_inner());
                break;
            }
            Err(TryLockError::WouldBlock) => thread::sleep(Duration::from_millis(5)),
        }
    }
    // before the log's lock: the pipe's last lines go through it
    release();
    let mut r = ring();
    if !ON.swap(false, Ordering::AcqRel) {
        return;
    }
    let mut err = io::stderr();
    if MOUSE.swap(false, Ordering::AcqRel) {
        let _ = mouse_reports(&mut err, false);
    }
    if RAW.swap(false, Ordering::AcqRel) {
        let _ = terminal::disable_raw_mode();
    }
    let _ = execute!(err, LeaveAlternateScreen, cursor::Show);
    let from = r.printed.max(r.logged.saturating_sub(TAIL));
    let first = r.logged - r.lines.len() as u64; // the index of the oldest line kept
    for (i, line) in r.lines.iter().enumerate() {
        if first + i as u64 >= from {
            let _ = writeln!(err, "{line}");
        }
    }
    r.printed = r.logged;
    drop(draw);
}

/// Asks the terminal to report the mouse's clicks and wheel (mode 1000) in
/// the SGR form (1006), or to stop. The view uses no moves, which crossterm's
/// EnableMouseCapture also asks for (1002, 1003).
fn mouse_reports(terminal: &mut impl Write, on: bool) -> io::Result<()> {
    terminal.write_all(if on {
        b"\x1b[?1000h\x1b[?1006h"
    } else {
        b"\x1b[?1006l\x1b[?1000l"
    })?;
    terminal.flush()
}

fn dim() -> Style {
    Style::new().add_modifier(Modifier::DIM)
}

fn bold() -> Style {
    Style::new().add_modifier(Modifier::BOLD)
}

fn color(x: theme::Rgb) -> Style {
    Style::new().fg(c(x))
}

/// An integer with thousands separators.
fn count(v: f64) -> String {
    let d = format!("{:.0}", v.max(0.0));
    let mut o = String::new();
    for (i, c) in d.chars().enumerate() {
        o.push(c);
        if (d.len() - i - 1) % 3 == 0 && i + 1 < d.len() {
            o.push(',');
        }
    }
    o
}

fn duration(s: f64) -> String {
    let t = s.max(0.0) as u64;
    if t >= 3600 {
        format!("{}h{:02}m", t / 3600, t / 60 % 60)
    } else if t >= 60 {
        format!("{}m{:02}s", t / 60, t % 60)
    } else {
        format!("{t}s")
    }
}

fn pct(part: f64, whole: f64) -> String {
    format!(
        "{:.1}%",
        if whole > 0.0 {
            100.0 * part / whole
        } else {
            0.0
        }
    )
}

fn width(spans: &[Span]) -> usize {
    spans.iter().map(|s| s.content.width()).sum()
}

/// The spans, then spaces to n columns (at least one).
fn pad(mut spans: Vec<Span<'static>>, n: usize) -> Vec<Span<'static>> {
    let k = n.saturating_sub(width(&spans)).max(1);
    spans.push(text(" ".repeat(k)));
    spans
}

/// A bar of w cells: each part's share in its style, the rest dim.
fn bar(w: usize, parts: &[(f64, Style)]) -> Vec<Span<'static>> {
    let (mut out, mut sum, mut done) = (Vec::new(), 0.0, 0);
    for &(share, style) in parts {
        sum += share.max(0.0);
        let to = ((sum * w as f64 + 0.5) as usize).min(w);
        if to > done {
            out.push(Span::styled("█".repeat(to - done), style));
            done = to;
        }
    }
    if done < w {
        out.push(Span::styled("░".repeat(w - done), color(get().rule)));
    }
    out
}

/// A cell filled by eighths from the bottom, and from the left: [k] is k
/// eighths full.
const LOWER: [&str; 9] = [" ", "▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"];
const LEFT: [&str; 9] = [" ", "▏", "▎", "▍", "▌", "▋", "▊", "▉", "█"];

/// Parts stacked along n cells, the first at (x, y) and each next one a
/// `step` on, filled by `eighths`: each part its length in cells and its
/// color, an eighth of a cell at the finest. A cell where a part ends shows
/// it in front of the next part.
fn stack(
    buf: &mut Buffer,
    (x, y): (u16, u16),
    step: (i32, i32),
    n: u16,
    eighths: &[&str; 9],
    parts: &[(f64, theme::Rgb)],
) {
    let mut ends = Vec::new(); // (end in eighths, color)
    let mut sum = 0.0;
    for &(len, col) in parts {
        if len > 0.0 {
            sum += len * 8.0;
            ends.push(((sum.round() as i64).min(n as i64 * 8), col));
        }
    }
    for k in 0..n as i64 {
        let Some(first) = ends.iter().position(|&(end, _)| end > k * 8) else {
            break;
        };
        let (end, col) = ends[first];
        let at = &mut buf[(
            (x as i64 + step.0 as i64 * k) as u16,
            (y as i64 + step.1 as i64 * k) as u16,
        )];
        at.set_symbol(eighths[(end - k * 8).min(8) as usize])
            .set_fg(c(col));
        if end < k * 8 + 8
            && let Some(&(_, next)) = ends.get(first + 1)
        {
            at.set_bg(c(next));
        }
    }
}

/// s in at most n columns, "…" at the cut.
fn clip(s: &str, n: usize) -> String {
    if s.width() <= n {
        return s.into();
    }
    let (mut o, mut col) = (String::new(), 0);
    for c in s.chars() {
        let cw = c.width().unwrap_or(0);
        if col + cw + 1 > n {
            break;
        }
        o.push(c);
        col += cw;
    }
    o.push('…');
    o
}

/// A line of at most w columns. A longer one ends after its last whole figure
/// (before two spaces or " ·"), else its last whole word, with " …".
fn fit(line: Line<'static>, w: usize) -> Line<'static> {
    if line.width() <= w {
        return line;
    }
    let cells: Vec<(char, Style)> = line
        .spans
        .iter()
        .flat_map(|s| s.content.chars().map(move |c| (c, s.style)))
        .collect();
    let limit = w.saturating_sub(2);
    let (mut col, mut word, mut figure, mut stop) = (0, 0, 0, cells.len());
    for (i, &(c, _)) in cells.iter().enumerate() {
        let cw = c.width().unwrap_or(0);
        if col + cw > limit {
            stop = i;
            break;
        }
        if c == ' ' && col > 0 && cells[i - 1].0 != ' ' {
            word = i;
            if matches!(cells.get(i + 1), Some((' ' | '·', _))) {
                figure = i;
            }
        }
        col += cw;
    }
    let cut = if figure > 0 {
        figure
    } else if word > 0 {
        word
    } else {
        stop
    };
    let mut spans: Vec<Span<'static>> = Vec::new();
    for &(c, style) in &cells[..cut] {
        match spans.last_mut() {
            Some(last) if last.style == style => last.content.to_mut().push(c),
            _ => spans.push(Span::styled(c.to_string(), style)),
        }
    }
    if w >= 2 {
        spans.push(Span::styled(" …", dim()));
    }
    Line::from(spans)
}

/// A section's rule.
fn head(w: usize, title: &str) -> Line<'static> {
    let t = get();
    let rule = w.saturating_sub(title.width() + 3);
    Line::from(vec![
        Span::styled("─ ", color(t.rule)),
        Span::styled(format!("{title} "), bold().fg(c(t.accent))),
        Span::styled("─".repeat(rule), color(t.rule)),
    ])
}

/// The server's state for the chrome's first row: the phase and the uptime.
fn state(s: &Snapshot, h: &History) -> Line<'static> {
    let t = get();
    let up = label(format!(" · up {} ", duration(s.uptime_seconds)));
    match s.phase {
        Phase::Decode => {
            let total: f64 = h.speeds().iter().map(|p| p.1).sum();
            Line::from(vec![
                Span::styled("● ", color(t.accent)),
                strong("decoding"),
                text(format!("  {total:.1} tok/s")),
                up,
            ])
        }
        Phase::Prefill => Line::from(vec![
            Span::styled("● ", color(t.prefill)),
            strong("prefill"),
            text(format!("  layer {}/{}", s.prefill.layer, s.layers)),
            up,
        ]),
        Phase::Idle => Line::from(vec![
            text("○ "),
            strong("idle"),
            text(format!(" {}", duration(s.phase_seconds))),
            up,
        ]),
    }
}

/// A whole screen: the chrome and the tab that `ui` shows. The clickable
/// places drawn go to `ui.hits`.
fn draw_screen(
    buf: &mut Buffer,
    s: &Snapshot,
    h: &History,
    ui: &mut Ui,
    title: &str,
    log: &[String],
) {
    // with several models, the model shown (the key bar has the next)
    let shown;
    let title = if s.models > 1 {
        shown = format!("{title} · showing {}", s.model);
        shown.as_str()
    } else {
        title
    };
    ui.hits.clear();
    let area = buf.area;
    let inner = frame::draw(buf, area, title, state(s, h), s.models, ui);
    if inner.height == 0 {
        // too small for the chrome: the title alone
        buf.set_line(
            area.x,
            area.y,
            &fit(Line::from(strong(format!(" {title}"))), area.width as usize),
            area.width,
        );
        return;
    }
    let (w, rows) = (inner.width as usize, inner.height as usize);
    match ui.tab {
        Tab::Overview => Paragraph::new(overview::draw(s, h, log, w, rows)).render(inner, buf),
        Tab::Kv => kv::draw(buf, inner, s, ui),
        Tab::Experts => experts::draw(buf, inner, s, h, ui),
        Tab::Minute => minute::draw(buf, inner, h),
        Tab::Memory => memory::draw(buf, inner, s),
        Tab::Log => {
            ui.page = rows.max(1);
            ui.log_back = ui.log_back.min(log.len().saturating_sub(rows));
            let end = log.len() - ui.log_back;
            let lines: Vec<Line> = log[end.saturating_sub(rows)..end]
                .iter()
                .map(|x| fit(Line::from(text(format!(" {x}"))), w))
                .collect();
            Paragraph::new(lines).render(inner, buf)
        }
    }
}

// ---- the keys and the mouse ----

/// Whether a key stops the server: q, or Ctrl-C (raw mode delivers it as a key).
fn quits(k: &KeyEvent) -> bool {
    k.kind == KeyEventKind::Press
        && (matches!(k.code, KeyCode::Char('q' | 'Q'))
            || (k.code == KeyCode::Char('c') && k.modifiers.contains(KeyModifiers::CONTROL)))
}

/// What an input asks of the view's loop.
#[derive(Debug, PartialEq, Eq)]
enum Asked {
    /// Draw again with the change.
    Redraw,
    /// Turn the mouse's reports on or off.
    Mouse(bool),
    /// Another model: its history starts again.
    Model,
    Quit,
    Nothing,
}

/// Chooses node i of the tree; with its panel open (or `open`), asks for its text.
fn choose(ui: &mut Ui, s: &Snapshot, i: usize, open: bool) {
    let n = &s.tree[i];
    let changed = ui.node.and_then(|k| kv::find(&s.tree, k)) != Some(i);
    ui.node = Some(kv::key(n));
    if (open || ui.text.is_some()) && (changed || ui.text.is_none()) {
        ui.scroll = 0;
        ui.text = Some(Text {
            session: n.via,
            start: n.start,
            end: n.end,
            answer: None,
        });
        crate::worker::ask_text(ui.model, n.via, n.start as usize, n.end as usize);
    }
}

/// The worker's answer to the view's question for a node's text: taken, as
/// text, if it answers the question asked last. Returns whether it did.
fn take_text(ui: &mut Ui, a: crate::worker::NodeText) -> bool {
    match &mut ui.text {
        Some(t)
            if t.answer.is_none()
                && a.model == ui.model
                && (t.session, t.start, t.end) == (a.session, a.start as u64, a.end as u64) =>
        {
            t.answer = Some(Answer {
                runs: crate::worker::text(a.model, &a.ids),
                tokens: a.ids.len(),
                ..Answer::default()
            });
            true
        }
        _ => false,
    }
}

/// Model `i` in the same tab: its own requests, tree and counters.
fn show_model(ui: &mut Ui, i: usize) {
    *ui = Ui {
        model: i,
        tab: ui.tab,
        mouse: ui.mouse,
        ..Ui::default()
    };
}

fn handle(ev: &Event, ui: &mut Ui, s: &Snapshot, h: &History) -> Asked {
    match ev {
        Event::Key(k) if quits(k) => Asked::Quit,
        Event::Key(k) if k.kind == KeyEventKind::Press => key(k.code, ui, s, h),
        Event::Mouse(m) if ui.mouse => match m.kind {
            MouseEventKind::Down(MouseButton::Left) => match ui.target(m.column, m.row) {
                Some(Target::Tab(tab)) => {
                    ui.tab = tab;
                    Asked::Redraw
                }
                Some(Target::Node(i)) if i < s.tree.len() => {
                    choose(ui, s, i, true);
                    Asked::Redraw
                }
                Some(Target::Request(n)) => {
                    ui.request = Some(n);
                    Asked::Redraw
                }
                Some(Target::Model(i)) if i < s.models && i != ui.model => {
                    show_model(ui, i);
                    Asked::Model
                }
                _ => Asked::Nothing,
            },
            MouseEventKind::ScrollUp => scroll(ui, -3),
            MouseEventKind::ScrollDown => scroll(ui, 3),
            _ => Asked::Nothing,
        },
        Event::Resize(..) => Asked::Redraw,
        _ => Asked::Nothing,
    }
}

/// Scrolls the text panel or the log by n lines (negative: back).
fn scroll(ui: &mut Ui, n: isize) -> Asked {
    match ui.tab {
        Tab::Kv if ui.text.is_some() => ui.scroll = ui.scroll.saturating_add_signed(n),
        Tab::Log => ui.log_back = ui.log_back.saturating_add_signed(-n),
        _ => return Asked::Nothing,
    }
    Asked::Redraw
}

fn key(code: KeyCode, ui: &mut Ui, s: &Snapshot, h: &History) -> Asked {
    let page = ui.page.max(1) as isize;
    match code {
        KeyCode::Char(d @ '1'..='6') => ui.tab = frame::TABS[d as usize - '1' as usize].0,
        KeyCode::Tab | KeyCode::BackTab => {
            let at = frame::TABS.iter().position(|t| t.0 == ui.tab).unwrap_or(0);
            let n = frame::TABS.len();
            ui.tab = frame::TABS[if code == KeyCode::Tab {
                (at + 1) % n
            } else {
                (at + n - 1) % n
            }]
            .0;
        }
        KeyCode::Char('m' | 'M') => {
            ui.mouse = !ui.mouse;
            return Asked::Mouse(ui.mouse);
        }
        KeyCode::Char('n' | 'N') if s.models > 1 => {
            show_model(ui, (ui.model + 1) % s.models);
            return Asked::Model;
        }
        KeyCode::Left | KeyCode::Right | KeyCode::Up | KeyCode::Down if ui.tab == Tab::Kv => {
            let m = match code {
                KeyCode::Left => kv::Move::Left,
                KeyCode::Right => kv::Move::Right,
                KeyCode::Up => kv::Move::Up,
                _ => kv::Move::Down,
            };
            let from = ui.node.and_then(|key| kv::find(&s.tree, key));
            match kv::step(&s.tree, from, m) {
                Some(i) => choose(ui, s, i, false),
                None => return Asked::Nothing,
            }
        }
        KeyCode::Enter if ui.tab == Tab::Kv => {
            let at = ui
                .node
                .and_then(|key| kv::find(&s.tree, key))
                .or((!s.tree.is_empty()).then_some(0));
            match at {
                Some(i) => choose(ui, s, i, true),
                None => return Asked::Nothing,
            }
        }
        KeyCode::Esc if ui.tab == Tab::Kv && ui.text.is_some() => ui.text = None,
        KeyCode::Esc if ui.tab == Tab::Kv => ui.node = None,
        KeyCode::PageUp => return scroll(ui, -page),
        KeyCode::PageDown => return scroll(ui, page),
        KeyCode::Up if ui.tab == Tab::Log => return scroll(ui, -1),
        KeyCode::Down if ui.tab == Tab::Log => return scroll(ui, 1),
        KeyCode::Home if ui.tab == Tab::Log => ui.log_back = usize::MAX,
        KeyCode::End if ui.tab == Tab::Log => ui.log_back = 0,
        KeyCode::Home if ui.tab == Tab::Kv => ui.scroll = 0,
        KeyCode::End if ui.tab == Tab::Kv => ui.scroll = usize::MAX,
        KeyCode::Char(b @ ('[' | ']')) if ui.tab == Tab::Experts => {
            let reqs = experts::requests(s, h);
            if reqs.is_empty() {
                return Asked::Nothing;
            }
            let at = ui
                .request
                .and_then(|n| reqs.iter().position(|&r| r == n))
                .unwrap_or(0);
            let next = if b == ']' {
                (at + 1) % reqs.len()
            } else {
                (at + reqs.len() - 1) % reqs.len()
            };
            ui.request = Some(reqs[next]);
        }
        _ => return Asked::Nothing,
    }
    Asked::Redraw
}

// ---- the view's thread ----

/// The colors the view uses (--tui-theme).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Colors {
    /// By the terminal's background, as it answers; light without an answer.
    Auto,
    Light,
    Dark,
}

/// The view on the screen; [`View::finish`] ends it.
pub struct View {
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

/// The alternate screen and the thread that draws it; None if the terminal
/// cannot show it. Keys and the mouse work where the terminal goes into raw
/// mode.
pub fn start(title: String, colors: Colors) -> Option<View> {
    let keys = terminal::enable_raw_mode().is_ok();
    RAW.store(keys, Ordering::Release);
    theme::set_true_color(theme::true_color_terminal());
    // the terminal answers on stdin: ask only when stdin is the terminal
    use std::io::IsTerminal;
    let ask = keys && io::stdin().is_terminal();
    theme::set(match colors {
        Colors::Light => theme::LIGHT,
        Colors::Dark => theme::DARK,
        Colors::Auto => ask
            .then(|| theme::detect(Duration::from_millis(200)))
            .flatten()
            .map_or(theme::LIGHT, theme::for_ground),
    });
    let mut err = io::stderr();
    if execute!(err, EnterAlternateScreen, cursor::Hide).is_err() {
        if RAW.swap(false, Ordering::AcqRel) {
            let _ = terminal::disable_raw_mode();
        }
        return None;
    }
    {
        let _r = ring();
        ON.store(true, Ordering::Release);
    }
    if keys && mouse_reports(&mut err, true).is_ok() {
        MOUSE.store(true, Ordering::Release);
    }
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        leave();
        hook(info)
    }));
    // from here the view alone writes to the terminal
    let Some(mut screen) = capture().and_then(|t| Terminal::new(CrosstermBackend::new(t)).ok())
    else {
        leave();
        return None;
    };
    let stop = Arc::new(AtomicBool::new(false));
    let done = stop.clone();
    let thread = thread::Builder::new()
        .name("view".into())
        .spawn(move || run(&mut screen, &title, keys, &done))
        .ok();
    Some(View { stop, thread })
}

impl View {
    /// Stops drawing; then the normal screen and the last log lines.
    pub fn finish(mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
        leave();
    }
}

fn run(screen: &mut Terminal<CrosstermBackend<File>>, title: &str, keys: bool, stop: &AtomicBool) {
    let started = Instant::now();
    let mut history = History::default();
    let mut ui = Ui {
        mouse: MOUSE.load(Ordering::Acquire),
        ..Ui::default()
    };
    let mut snapshot = Snapshot::default();
    let mut next = Instant::now();
    let mut keys = keys;
    let mut redraw = true;
    while !stop.load(Ordering::Acquire) {
        if Instant::now() >= next {
            // The snapshot before the screen's lock: an engine failing while
            // it holds what the snapshot needs can still restore the terminal.
            if let Ok(s) = serde_json::from_str::<Snapshot>(&crate::worker::view(ui.model)) {
                history.observe(&s, started.elapsed().as_secs_f64());
                snapshot = s;
                redraw = true;
            }
            crate::worker::ask_routes();
            next = Instant::now() + Duration::from_millis(500);
        }
        if ui.text.as_ref().is_some_and(|t| t.answer.is_none())
            && let Some(a) = crate::worker::text_answer(ui.model)
        {
            redraw |= take_text(&mut ui, a);
        }
        if redraw {
            let log: Vec<String> = ring().lines.iter().cloned().collect();
            let _draw = DRAW.lock().unwrap_or_else(|e| e.into_inner());
            if !ON.load(Ordering::Acquire) {
                return;
            }
            let _ = screen.draw(|f| {
                draw_screen(f.buffer_mut(), &snapshot, &history, &mut ui, title, &log);
            });
            redraw = false;
        }
        // the keys and the mouse until the next snapshot
        let Some(ev) = input(&mut keys, next) else {
            continue;
        };
        match handle(&ev, &mut ui, &snapshot, &history) {
            Asked::Quit => {
                if !engine::stopping() {
                    engine::log("q: stopping after the current step");
                }
                crate::worker::request_stop();
            }
            Asked::Mouse(on) => {
                if mouse_reports(screen.backend_mut(), on).is_ok() {
                    MOUSE.store(on, Ordering::Release);
                }
                redraw = true;
            }
            Asked::Model => {
                // the other model's snapshot at once, its history from now
                history = History::default();
                next = Instant::now();
                redraw = true;
            }
            Asked::Redraw => redraw = true,
            Asked::Nothing => {}
        }
    }
}

/// The next key, click, scroll or resize before `until`; None at `until`.
/// Once the keys cannot be read, the wait is a sleep and `keys` is false.
fn input(keys: &mut bool, until: Instant) -> Option<Event> {
    loop {
        let left = until.checked_duration_since(Instant::now())?;
        if !*keys {
            thread::sleep(left);
            return None;
        }
        match event::poll(left) {
            Ok(false) => return None,
            Ok(true) => match event::read() {
                Ok(ev) => return Some(ev),
                Err(_) => *keys = false,
            },
            Err(_) => *keys = false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::backend::TestBackend;
    use ratatui::layout::Rect;

    // A snapshot in the form worker::view writes, decoding two requests.
    const SNAPSHOT: &str = r#"{"uptime_seconds": 3725, "phase": "decode", "phase_seconds": 2.5, "waiting": 1,
        "requests": [{"number": 12, "session": 3, "prompt": 41234, "reused": 40000, "generated": 77},
                     {"number": 13, "session": 4, "prompt": 900, "reused": 0, "generated": 5}],
        "layers": 40, "experts_per_row": 240, "prefill": {"layer": 0, "rows": 0},
        "totals": {"requests": 13, "prompt_tokens": 90000, "reused_tokens": 60000, "generated_tokens": 4000},
        "kinds": {"new": 4, "continued": 6, "saved": 2, "forked": 1},
        "sessions": 4, "kv_gib": 1.234, "kv_limit_gib": 8.000, "tree": [
          {"lines": "", "start": 0, "end": 3000, "sessions": 3, "hits": 5, "session": 0, "via": 3, "busy": false, "saved": -1, "idle_seconds": 0.0, "gib": 0.000},
          {"lines": "├─ ", "start": 3000, "end": 41234, "sessions": 1, "hits": 2, "session": 3, "via": 3, "busy": true, "saved": 41000, "idle_seconds": 0.0, "gib": 0.600},
          {"lines": "└─ ", "start": 3000, "end": 9000, "sessions": 2, "hits": 0, "session": 0, "via": 1, "busy": false, "saved": -1, "idle_seconds": 0.0, "gib": 0.000},
          {"lines": "   ├─ ", "start": 9000, "end": 9500, "sessions": 1, "hits": 0, "session": 1, "via": 1, "busy": false, "saved": 9400, "idle_seconds": 75.0, "gib": 0.100},
          {"lines": "   └─ ", "start": 9000, "end": 9900, "sessions": 1, "hits": 0, "session": 2, "via": 2, "busy": false, "saved": -1, "idle_seconds": 4000.0, "gib": 0.120}],
        "decode": {"rows": 5000, "tokens": 5000, "seconds": 210.5, "gpu_seconds": 150.2, "in_memory": 1100000, "read_ahead": 90000,
                   "late": 2000, "substituted": 30000, "from_ssd": 68000, "bytes": 400000000000,
                   "read_wait_seconds": 30.1, "prefetches": 95000, "cancelled": 800, "remaps": 900},
        "prompts": {"rows": 120000, "hits": 900000, "prefetch_hits": 100, "misses": 200000, "substituted": 0, "bytes": 800000000000},
        "prefilled": {"rows": 120000, "bytes": 800000000000, "seconds": 300.0},
        "cache": {"slots": 3500, "mapped": 3260, "cap": 3600, "max_slots": 4000, "slot_bytes": 11534336, "experts": 15360,
                  "uncached": 11860, "uncached_bytes": 86000000000, "evictions": 123456, "given_back": 12},
        "memory": {"footprint": 52000000000, "dense": 8600000000, "pinned": 900000000, "gpu_in_use": 47000000000,
                   "gpu_limit": null, "total": 137438953472, "free": 9000000000, "file": 30000000000, "file_kept": 4000000000,
                   "wired": 20000000000, "compressor": 1000000000, "compressed": 3000000000, "swap_used": 0, "swapouts": 1000,
                   "pressure": 1}}"#;

    fn snapshot() -> Snapshot {
        serde_json::from_str(SNAPSHOT).expect("worker::view's form")
    }

    fn log(n: usize) -> Vec<String> {
        (0..n)
            .map(|i| {
                format!(
                    "12:00:{:02}.000 req {i}: 1,000 prompt tokens, session 3 (kept), 900 reused",
                    i % 60
                )
            })
            .collect()
    }

    fn row(b: &Buffer, y: u16) -> String {
        (0..b.area.width)
            .map(|x| b[(x, y)].symbol().to_string())
            .collect()
    }

    fn all(b: &Buffer) -> String {
        (0..b.area.height)
            .map(|y| row(b, y))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// A routed row for request n: experts e, e+1, … in every layer, from
    /// where `source` says.
    fn routed(n: u64, e: u16, source: impl Fn(usize) -> u8) -> Req {
        let experts: Vec<u16> = (0..240).map(|i| (e + i as u16 * 7) % 384).collect();
        let sources: Vec<u8> = (0..240).map(source).collect();
        Req {
            number: n,
            session: 3,
            prompt: 1000,
            reused: 0,
            generated: 50,
            route: Some(Routed { experts, sources }),
        }
    }

    /// A minute of samples: decoding with two requests and their routed rows.
    fn minute_of(s: &mut Snapshot) -> History {
        let mut h = History::default();
        for k in 0..130 {
            s.requests = vec![routed(12, k, |i| (i % 5) as u8), routed(13, 2 * k, |_| 0)];
            s.requests[0].generated = 10 + 8 * k as u64;
            s.requests[1].generated = 5 + 6 * k as u64;
            s.decode.rows += 14;
            s.decode.tokens += 14;
            s.decode.seconds += 0.5;
            s.decode.gpu_seconds += 0.4;
            s.decode.in_memory += 3000;
            s.decode.from_ssd += 60;
            s.decode.bytes += 900_000_000;
            s.cache.read_bytes += 900_000_000;
            h.observe(s, k as f64 * 0.5);
        }
        h
    }

    /// The screen of a tab at w x h cells.
    fn draw(tab: Tab, w: u16, ht: u16, setup: impl Fn(&mut Ui)) -> (Buffer, Ui) {
        let mut s = snapshot();
        let h = minute_of(&mut s);
        let mut ui = Ui {
            tab,
            ..Ui::default()
        };
        setup(&mut ui);
        let mut b = Buffer::empty(Rect::new(0, 0, w, ht));
        draw_screen(
            &mut b,
            &s,
            &h,
            &mut ui,
            "myds4 · 127.0.0.1:8008 · deepseek-v4.1-flash · GGUF experts",
            &log(80),
        );
        (b, ui)
    }

    #[test]
    fn every_tab_fits_the_terminal_with_the_title_and_the_tabs_first() {
        for tab in frame::TABS.map(|t| t.0) {
            for (w, h) in [(220, 60), (160, 46), (120, 40), (80, 24), (40, 12), (12, 4)] {
                let (b, _) = draw(tab, w, h, |_| {});
                assert!(
                    row(&b, 0).starts_with(" myds4"),
                    "{tab:?} {w}x{h}: {}",
                    row(&b, 0)
                );
                if h >= 5 && w >= 20 {
                    assert!(
                        row(&b, 1).contains("1 Overview"),
                        "{tab:?} {w}x{h}: {}",
                        row(&b, 1)
                    );
                    assert!(
                        row(&b, h - 1).contains("q stop the server") || w < 100,
                        "{tab:?} {w}x{h}: {}",
                        row(&b, h - 1)
                    );
                }
            }
        }
    }

    #[test]
    fn mac_memory_counts_file_cache_beyond_the_kept_share_as_available() {
        let mut s = snapshot();
        s.memory.total = 128 * 1024u64.pow(3);
        s.memory.free = 58 * 1024u64.pow(3);
        s.memory.file = 36 * 1024u64.pow(3);
        s.memory.file_kept = 20 * 1024u64.pow(3);
        let lines = overview::draw(&s, &History::default(), &[], 240, 60);
        let mac = lines
            .iter()
            .map(|l| {
                l.spans
                    .iter()
                    .map(|s| s.content.as_ref())
                    .collect::<String>()
            })
            .find(|l| l.starts_with(" Mac "))
            .expect("Mac memory row");
        assert!(mac.contains("54.0 / 128 GiB"), "{mac}");
        assert!(mac.contains("available 74.0"), "{mac}");
        assert!(mac.contains("file cache 36.0"), "{mac}");
    }

    /// Two models in the memory of a Mac of 128 GiB: DeepSeek's weights,
    /// experts and KV, Qwen's weights and KV, and the server's other memory.
    fn two_models() -> Snapshot {
        let mut s = snapshot();
        s.model = "DeepSeek".into();
        s.models = 2;
        s.kv_gib = 1.0;
        s.kv_total_gib = 3.0;
        s.kv_limit_gib = 12.0;
        s.reserve_gib = 10.0;
        let gib = 1024u64.pow(3);
        s.memory.total = 128 * gib;
        s.memory.footprint = 89 * gib;
        s.memory.free = 8 * gib;
        s.memory.file = 19 * gib;
        s.model_memory = vec![
            ModelMemory {
                workspace: 0,
                model: "DeepSeek".into(),
                weights: 8 * gib,
                experts: 56 * gib,
                kv: gib,
                kv_committed: gib,
                sessions: 2,
            },
            ModelMemory {
                workspace: 0,
                model: "Qwen".into(),
                weights: 20 * gib,
                experts: 0,
                kv: 2 * gib,
                kv_committed: 2 * gib,
                sessions: 1,
            },
        ];
        s
    }

    #[test]
    fn the_memory_map_shows_every_model_its_parts_and_the_rest() {
        let s = two_models();
        for (w, h) in [(160, 30), (80, 30), (40, 32), (12, 4)] {
            let mut b = Buffer::empty(ratatui::layout::Rect::new(0, 0, w, h));
            let mut ui = Ui {
                tab: Tab::Memory,
                ..Ui::default()
            };
            draw_screen(&mut b, &s, &History::default(), &mut ui, "myds4", &[]);
            if w >= 80 {
                let screen = all(&b);
                // each model under its name, with its parts inside it
                assert!(
                    screen.contains("DeepSeek 65.0 GiB") && screen.contains("Qwen 22.0 GiB"),
                    "{screen}"
                );
                assert!(screen.contains("experts 56.0 GiB"), "{screen}");
                // the rest of the Mac's memory beside them, by its sizes:
                // other programs and the file cache
                if w == 160 {
                    for size in [" 12.0 GiB", " 19.0 GiB"] {
                        assert!(screen.contains(size), "{size}: {screen}");
                    }
                }
                assert_eq!(screen.matches("3.00 / 12.00 GiB, shared").count(), 1);
                if std::env::var_os("MYDS4_PRINT_MEMORY_VIEW").is_some() {
                    eprintln!("{screen}");
                }
            }
        }
        let mut ui = Ui::default();
        key(KeyCode::Char('6'), &mut ui, &s, &History::default());
        assert_eq!(ui.tab, Tab::Memory);
    }

    /// The four models of this server's configuration, with their real ids
    /// (17 to 44 columns); the last is not loaded.
    fn four_models() -> Snapshot {
        let mut s = two_models();
        let gib = 1024u64.pow(3);
        s.models = 4;
        s.model = "deepseek-v4.1-flash".into();
        s.reserve_gib = 5.0;
        s.memory.footprint = 75 * gib;
        s.memory.free = 25 * gib;
        s.memory.file = 12 * gib;
        let m = |model: &str, weights, experts, kv, sessions| ModelMemory {
            model: model.into(),
            weights,
            experts,
            workspace: if weights > 0 { gib / 10 } else { 0 },
            kv,
            kv_committed: kv,
            sessions,
        };
        s.model_memory = vec![
            m(
                "deepseek-v4.1-flash",
                7 * gib + gib / 5,
                40 * gib,
                gib / 16,
                3,
            ),
            m("qwen3.8-27b-nvfp4", 15 * gib, 0, gib / 2, 1),
            m("qwen3.8-flash-next-fp4-local", 4 * gib, 6 * gib, gib / 5, 2),
            m("nvidia-nemotron-3.5-transcribe-30b-a4b-nvfp4", 0, 0, 0, 0),
        ];
        s
    }

    #[test]
    fn four_long_model_ids_keep_their_sizes_and_sessions() {
        let s = four_models();
        let ids: Vec<&str> = s.model_memory.iter().map(|m| m.model.as_str()).collect();
        for w in [160, 120] {
            let mut ui = Ui {
                tab: Tab::Memory,
                ..Ui::default()
            };
            let mut b = Buffer::empty(Rect::new(0, 0, w, 40));
            draw_screen(&mut b, &s, &History::default(), &mut ui, "myds4", &[]);
            let screen = all(&b);
            // A long name gives way to its model's size; the parts too small
            // for their sizes follow on the model's first row. The reserve's
            // share of free memory is marked.
            for size in ["47.4 GiB", "15.6 GiB", "10.3 GiB", "KV 0.06 GiB", "5.0 GiB"] {
                assert!(screen.contains(size), "{w} columns, {size}: {screen}");
            }
            assert!(screen.contains("3 of 4 in memory"), "{screen}");
            // every model's sessions in the KV tab's rows of models
            ui.tab = Tab::Kv;
            let mut b = Buffer::empty(Rect::new(0, 0, w, 40));
            draw_screen(&mut b, &s, &History::default(), &mut ui, "myds4", &[]);
            let screen = all(&b);
            let rows: String = screen.lines().skip(3).take(3).collect();
            for id in &ids {
                assert!(
                    rows.contains(&format!("{id}  ")),
                    "{w} columns, {id}: {screen}"
                );
            }
            assert!(rows.contains("3 sessions · 0.06 GiB"), "{screen}");
            if std::env::var_os("MYDS4_PRINT_MEMORY_VIEW").is_some() {
                eprintln!("{screen}");
            }
        }
    }

    #[test]
    fn the_kv_tab_lists_every_models_sessions_and_a_click_shows_another() {
        let s = two_models();
        let mut ui = Ui {
            tab: Tab::Kv,
            ..Ui::default()
        };
        let mut b = Buffer::empty(Rect::new(0, 0, 160, 46));
        let title = "myds4 · 127.0.0.1:8008 · 2 models";
        draw_screen(&mut b, &s, &History::default(), &mut ui, title, &[]);
        let screen = all(&b);
        let lines: Vec<&str> = screen.lines().collect();
        assert!(lines[0].contains("2 models · showing DeepSeek"), "{screen}");
        // the tab's first row: every model with its sessions and KV
        assert!(
            lines[3].contains("DeepSeek  2 sessions · 1.00 GiB")
                && lines[3].contains("Qwen  1 session · 2.00 GiB"),
            "{screen}"
        );
        assert!(screen.contains("n next model"), "{screen}");
        // a click on Qwen shows Qwen's sessions in the same tab
        let hit = *ui
            .hits
            .iter()
            .find(|h| h.target == Target::Model(1))
            .expect("Qwen's place in the row");
        let ev = Event::Mouse(event::MouseEvent {
            kind: MouseEventKind::Down(MouseButton::Left),
            column: hit.x0,
            row: hit.y,
            modifiers: KeyModifiers::NONE,
        });
        assert_eq!(handle(&ev, &mut ui, &s, &History::default()), Asked::Model);
        assert_eq!((ui.model, ui.tab), (1, Tab::Kv));
    }

    #[test]
    fn the_overview_shows_the_throughput_and_the_logs_tail() {
        let (b, _) = draw(Tab::Overview, 160, 46, |_| {});
        let screen = all(&b);
        for name in [
            "Throughput",
            "decoding",
            "prefill",
            "SSD read",
            "hit rate",
            "GPU busy",
            "read wait",
            "served",
        ] {
            assert!(screen.contains(name), "no {name}:\n{screen}");
        }
        // since the start: 5,000 decoded rows (plus the minute's) in their seconds; 120,000 prompt rows in 300 s
        assert!(
            screen.contains("400 tok/s"),
            "the prefill since the start:\n{screen}"
        );
        assert!(
            row(&b, 44).contains("req 79:"),
            "the log's newest line last: {}",
            row(&b, 44)
        );
        assert!(!screen.contains("prompt pass"));
    }

    #[test]
    fn the_kv_tab_draws_the_tree_and_a_click_or_enter_opens_a_nodes_text() {
        let (b, ui) = draw(Tab::Kv, 160, 46, |_| {});
        let screen = all(&b);
        assert!(
            screen.contains("s3") && screen.contains("#12 decoding"),
            "{screen}"
        );
        assert!(screen.contains("●"), "the sessions' ends");
        // every node can be clicked
        let nodes: Vec<usize> = ui
            .hits
            .iter()
            .filter_map(|h| {
                if let Target::Node(i) = h.target {
                    Some(i)
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(nodes.len(), 5, "{nodes:?}");
        // a click on the busy session's node chooses it and asks for its text
        let s = snapshot();
        let mut ui = ui;
        let hit = *ui
            .hits
            .iter()
            .find(|h| h.target == Target::Node(1))
            .unwrap();
        let ev = Event::Mouse(event::MouseEvent {
            kind: MouseEventKind::Down(MouseButton::Left),
            column: hit.x0,
            row: hit.y,
            modifiers: KeyModifiers::NONE,
        });
        assert_eq!(handle(&ev, &mut ui, &s, &History::default()), Asked::Redraw);
        assert_eq!(ui.node, Some(frame::NodeKey { via: 3, at: 3000 }));
        let asked = Text {
            session: 3,
            start: 3000,
            end: 41234,
            answer: None,
        };
        assert_eq!(ui.text, Some(asked.clone()));
        // the answer shows; another node's does not
        assert!(!take_text(
            &mut ui,
            crate::worker::NodeText {
                model: 0,
                session: 1,
                start: 9000,
                end: 9500,
                ids: Vec::new()
            }
        ));
        assert_eq!(ui.text, Some(asked));
        // the session decodes on before the answer comes: its node grows, and
        // stays chosen with the text asked for
        let mut grown = snapshot();
        grown.tree[1].end = 41300;
        assert_eq!(ui.node.and_then(|k| kv::find(&grown.tree, k)), Some(1));
        // (without a tokenizer here, the answer's tokens become no text: the
        // runs drawn are set after it)
        assert!(take_text(
            &mut ui,
            crate::worker::NodeText {
                model: 0,
                session: 3,
                start: 3000,
                end: 41234,
                ids: Vec::new()
            }
        ));
        let runs = vec![
            ("<｜User｜>".to_string(), true),
            ("Hi there".to_string(), false),
        ];
        ui.text.as_mut().expect("the text").answer = Some(Answer {
            runs,
            tokens: 38234,
            ..Answer::default()
        });
        let mut b = Buffer::empty(Rect::new(0, 0, 160, 46));
        draw_screen(&mut b, &grown, &History::default(), &mut ui, "myds4", &[]);
        let text = all(&b);
        assert!(
            text.contains("s3 · tokens 3,000–41,234") && text.contains("Hi there"),
            "{text}"
        );
        // esc closes it, and the arrows move between the nodes
        assert_eq!(
            key(KeyCode::Esc, &mut ui, &s, &History::default()),
            Asked::Redraw
        );
        assert_eq!(ui.text, None);
        key(KeyCode::Down, &mut ui, &s, &History::default());
        assert_eq!(
            ui.node.and_then(|k| kv::find(&s.tree, k)),
            Some(2),
            "the row below: 3,000–9,000"
        );
    }

    #[test]
    fn the_mouse_still_works_after_a_nodes_text_opens() {
        // the KV tab: a node chosen with the keys, and its text open and long
        let (s, h) = (snapshot(), History::default());
        let mut ui = Ui {
            tab: Tab::Kv,
            ..Ui::default()
        };
        key(KeyCode::Down, &mut ui, &s, &h);
        key(KeyCode::Enter, &mut ui, &s, &h);
        let runs = vec![(
            (1..=300).map(|i| format!("{i}\n")).collect::<String>(),
            false,
        )];
        ui.text.as_mut().expect("the text asked for").answer = Some(Answer {
            runs,
            tokens: 600,
            ..Answer::default()
        });
        let mut b = Buffer::empty(Rect::new(0, 0, 160, 46));
        draw_screen(&mut b, &s, &h, &mut ui, "myds4", &[]);
        let mouse = |kind, column, row| {
            Event::Mouse(event::MouseEvent {
                kind,
                column,
                row,
                modifiers: KeyModifiers::NONE,
            })
        };
        // the wheel scrolls the text
        assert_eq!(
            handle(&mouse(MouseEventKind::ScrollDown, 120, 20), &mut ui, &s, &h),
            Asked::Redraw
        );
        draw_screen(&mut b, &s, &h, &mut ui, "myds4", &[]);
        assert!(all(&b).contains("lines 4–"), "{}", all(&b));
        // a click on a tab chooses it
        let tab = *ui
            .hits
            .iter()
            .find(|x| x.target == Target::Tab(Tab::Minute))
            .expect("the tab's place");
        assert_eq!(
            handle(
                &mouse(MouseEventKind::Down(MouseButton::Left), tab.x0, tab.y),
                &mut ui,
                &s,
                &h
            ),
            Asked::Redraw
        );
        assert_eq!(ui.tab, Tab::Minute);
    }

    #[test]
    fn the_kv_tabs_panel_shows_the_text_with_the_special_tokens() {
        let (b, _) = draw(Tab::Kv, 160, 46, |ui| {
            ui.node = Some(frame::NodeKey { via: 3, at: 0 });
            let runs = vec![
                ("<｜begin▁of▁sentence｜>".to_string(), true),
                ("You review this project".to_string(), false),
            ];
            ui.text = Some(Text {
                session: 3,
                start: 0,
                end: 3000,
                answer: Some(Answer {
                    runs,
                    tokens: 3000,
                    ..Answer::default()
                }),
            });
        });
        let screen = all(&b);
        assert!(
            screen.contains("s3 · tokens 0–3,000"),
            "a shared node shows the text of a session through it:\n{screen}"
        );
        assert!(screen.contains("You review this project"), "{screen}");
        assert!(screen.contains("lines 1–1 of 1"), "{screen}");
    }

    #[test]
    fn the_expert_map_shows_each_layer_and_switches_requests() {
        let (b, ui) = draw(Tab::Experts, 220, 60, |_| {});
        let screen = all(&b);
        assert!(
            screen.contains("#12 · s3") && screen.contains("#13 · s3"),
            "the request tabs:\n{screen}"
        );
        assert!(screen.contains("12 tokens"), "{screen}");
        assert!(
            row(&b, 6).starts_with("  1 "),
            "layer 1's row: {}",
            row(&b, 6)
        );
        assert!(screen.contains(" 40 "), "layer 40's row");
        assert!(screen.contains("a cell: two experts"), "{screen}");
        // ] goes to the next request
        let (mut s, mut ui) = (snapshot(), ui);
        let h = minute_of(&mut s);
        assert_eq!(key(KeyCode::Char(']'), &mut ui, &s, &h), Asked::Redraw);
        assert_eq!(ui.request, Some(13));
        // narrower: four experts a cell
        let (b, _) = draw(Tab::Experts, 160, 60, |_| {});
        assert!(all(&b).contains("a cell: four experts"));
    }

    #[test]
    fn the_last_minute_draws_a_column_per_sample() {
        let (b, _) = draw(Tab::Minute, 160, 46, |_| {});
        let screen = all(&b);
        for name in [
            "phase", "tokens/s", "SSD read", "hit rate", "requests", "#12", "#13", "now",
        ] {
            assert!(screen.contains(name), "no {name}:\n{screen}");
        }
        assert!(screen.contains("decoding") && !screen.contains("prompt pass"));
    }

    #[test]
    fn the_keys_choose_tabs_and_scroll_the_log() {
        let (s, h) = (snapshot(), History::default());
        let mut ui = Ui::default();
        assert_eq!(key(KeyCode::Char('4'), &mut ui, &s, &h), Asked::Redraw);
        assert_eq!(ui.tab, Tab::Minute);
        key(KeyCode::Tab, &mut ui, &s, &h);
        assert_eq!(ui.tab, Tab::Log);
        key(KeyCode::Tab, &mut ui, &s, &h);
        assert_eq!(ui.tab, Tab::Memory);
        key(KeyCode::Tab, &mut ui, &s, &h);
        assert_eq!(ui.tab, Tab::Overview);
        key(KeyCode::BackTab, &mut ui, &s, &h);
        assert_eq!(ui.tab, Tab::Memory);
        key(KeyCode::BackTab, &mut ui, &s, &h);
        assert_eq!(ui.tab, Tab::Log);
        key(KeyCode::Up, &mut ui, &s, &h);
        assert_eq!(ui.log_back, 1);
        key(KeyCode::End, &mut ui, &s, &h);
        assert_eq!(ui.log_back, 0);
        assert_eq!(
            key(KeyCode::Char('m'), &mut ui, &s, &h),
            Asked::Mouse(false)
        );
        assert_eq!(key(KeyCode::Char('x'), &mut ui, &s, &h), Asked::Nothing);
        // a scrolled log shows older lines
        let (b, _) = draw(Tab::Log, 120, 30, |ui| ui.log_back = 10);
        assert!(row(&b, 28).contains("req 69:"), "{}", row(&b, 28));
    }

    #[test]
    fn speeds_come_from_the_samples_of_each_request() {
        let mut h = History::default();
        assert_eq!(h.speed(12, 10, 0.0), 0.0);
        assert!((h.speed(12, 30, 1.0) - 20.0).abs() < 1e-9);
        assert!((h.speed(12, 40, 2.0) - 15.0).abs() < 1e-9);
    }

    #[test]
    fn the_ssd_rate_follows_the_reads_not_the_end_of_a_prefill() {
        // A prefill reads 12 GB/s for 4 s. The engine's read counter grows
        // as the reads complete; its prefill counter takes all 48 GB only
        // when the prefill ends, in the last sample.
        let mut h = History::default();
        for k in 0..9u64 {
            let mut v: serde_json::Value = serde_json::from_str(SNAPSHOT).unwrap();
            v["phase"] = "prefill".into();
            v["cache"]["read_bytes"] = (k * 6_000_000_000).into();
            v["prompts"]["bytes"] =
                (800_000_000_000u64 + if k == 8 { 48_000_000_000 } else { 0 }).into();
            let s: Snapshot = serde_json::from_value(v).unwrap();
            h.observe(&s, k as f64 * 0.5);
        }
        let rates: Vec<f64> = h.samples.iter().skip(1).map(|x| x.ssd).collect();
        assert!(rates.iter().all(|&r| (r - 12.0).abs() < 1e-9), "{rates:?}");
    }

    #[test]
    fn the_since_start_figures_take_counters_counted_together() {
        // A second prefill's first chunk has ended. The engine counts a
        // chunk's rows and expert bytes when the chunk ends, the worker a
        // prefill's time when the prefill ends; since the start: the first
        // prefill's 16,384 rows and 24 GB in its 20 s. Decoding: 1,000 rows,
        // drafts included, gave 600 tokens in 10 s.
        let mut v: serde_json::Value = serde_json::from_str(SNAPSHOT).unwrap();
        v["phase"] = "prefill".into();
        v["prompts"]["rows"] = 32_768.into();
        v["prompts"]["bytes"] = 48_000_000_000u64.into();
        v["prompts"]["seconds"] = 20.0.into();
        v["prefilled"] =
            serde_json::json!({"rows": 16_384, "bytes": 24_000_000_000u64, "seconds": 20.0});
        for (k, x) in [("rows", 1000), ("tokens", 600), ("bytes", 0)] {
            v["decode"][k] = x.into();
        }
        v["decode"]["seconds"] = 10.0.into();
        let s: Snapshot = serde_json::from_value(v).unwrap();
        let lines = overview::throughput(&s, &History::default(), 160);
        let row = |name: &str| {
            let text = |l: &Line| {
                l.spans
                    .iter()
                    .map(|s| s.content.as_ref())
                    .collect::<String>()
            };
            lines
                .iter()
                .map(text)
                .find(|l| l.trim_start().starts_with(name))
                .unwrap_or_default()
        };
        // SSD read: (0 + 24 GB) / (10 + 20) s
        let got = [row("prefill"), row("SSD read"), row("decoding")];
        let want = ["819 tok/s", "0.8 GB/s", "60.0 tok/s"];
        assert!(got.iter().zip(want).all(|(g, w)| g.contains(w)), "{got:#?}");
    }

    #[test]
    fn the_ssd_reads_windows_count_the_samples_that_worked() {
        // two idle samples, then a decoding one that read 1 GB in 0.5 s
        let mut h = History::default();
        for (k, (phase, read)) in [("idle", 0u64), ("idle", 0), ("decode", 1_000_000_000)]
            .into_iter()
            .enumerate()
        {
            let mut v: serde_json::Value = serde_json::from_str(SNAPSHOT).unwrap();
            v["phase"] = phase.into();
            v["cache"]["read_bytes"] = read.into();
            h.observe(&serde_json::from_value(v).unwrap(), k as f64 * 0.5);
        }
        let lines = overview::throughput(&snapshot(), &h, 160);
        let text = |l: &Line| {
            l.spans
                .iter()
                .map(|s| s.content.as_ref())
                .collect::<String>()
        };
        let ssd = lines
            .iter()
            .map(text)
            .find(|l| l.trim_start().starts_with("SSD read"))
            .unwrap_or_default();
        // the last 5 s: the decoding sample's, as since the start counts the time worked
        assert!(ssd.contains("2.0 GB/s"), "{ssd}");
    }

    #[test]
    fn a_prefill_rate_needs_no_sample_between_two_prefills() {
        // Request A's prefill runs. Between two samples it ends, one decode
        // step runs, and request B's prefill starts, its phase's clock from 0.
        let at = |rows: u64, chunk: u64, layer: u64, phase_seconds: f64| {
            let mut v: serde_json::Value = serde_json::from_str(SNAPSHOT).unwrap();
            v["phase"] = "prefill".into();
            v["phase_seconds"] = phase_seconds.into();
            v["prompts"]["rows"] = rows.into();
            v["prefill"] = serde_json::json!({"rows": chunk, "layer": layer});
            serde_json::from_value::<Snapshot>(v).unwrap()
        };
        let mut h = History::default();
        h.observe(&at(0, 16_384, 36, 10.0), 0.0);
        h.observe(&at(16_384, 4_096, 4, 0.3), 0.5);
        // 16,384 x 36/40 rows, then 16,384 + 4,096 x 4/40: 2,048 rows in 0.5 s
        let rate = h.samples.back().unwrap().prefill_rate;
        assert!((rate - 4096.0).abs() < 1e-6, "{rate}");
    }

    #[test]
    fn a_chosen_node_stays_chosen_when_a_fork_splits_it() {
        // s1 and s2 share 0–9000; then s3 shares 0–6000 with them
        let node = |lines: &str, start: u64, end: u64, session: u64, via: u64| Node {
            lines: lines.into(),
            start,
            end,
            session,
            via,
            ..Node::default()
        };
        let before = vec![
            node("", 0, 9000, 0, 2),
            node("├─ ", 9000, 9500, 1, 1),
            node("└─ ", 9000, 9800, 2, 2),
        ];
        let after = vec![
            node("", 0, 6000, 0, 3),
            node("├─ ", 6000, 9000, 0, 2),
            node("│  ├─ ", 9000, 9500, 1, 1),
            node("│  └─ ", 9000, 9800, 2, 2),
            node("└─ ", 6000, 6500, 3, 3),
        ];
        let key = kv::key(&before[0]);
        assert_eq!(kv::find(&before, key), Some(0));
        // the node that now holds its first position
        assert_eq!(kv::find(&after, key), Some(0));
    }

    #[test]
    fn the_samples_take_the_differences_of_the_counters() {
        let mut s = snapshot();
        let h = minute_of(&mut s);
        assert_eq!(h.samples.len(), history::SAMPLES);
        let last = h.samples.back().unwrap();
        // 900 MB in 0.5 s; 3,000 of 3,060 experts in memory; 0.4 of 0.5 s on the GPU
        assert!((last.ssd - 1.8).abs() < 1e-9, "{}", last.ssd);
        assert!((last.hit.unwrap() - 3000.0 / 3060.0).abs() < 1e-9);
        assert!((last.gpu.unwrap() - 0.8).abs() < 1e-9);
        assert_eq!(last.routes.len(), 2);
        // a request seen first in a sample arrives there
        assert!(
            h.samples.iter().all(|x| x.arrivals.is_empty()),
            "both were there from the first sample on"
        );
        let mut s2 = snapshot();
        let mut h2 = History::default();
        h2.observe(&s2, 0.0);
        s2.requests.push(Req {
            number: 14,
            ..Req::default()
        });
        h2.observe(&s2, 0.5);
        assert_eq!(h2.samples.back().unwrap().arrivals, vec![14]);
        // a row of another size than the snapshot's experts per row is not kept
        let mut s3 = snapshot();
        s3.requests = vec![routed(12, 0, |_| 0)];
        s3.experts_per_row = 6 * 43;
        let mut h3 = History::default();
        h3.observe(&s3, 0.0);
        assert!(h3.routes(12, 1).is_empty());
    }

    #[test]
    fn a_long_line_ends_after_a_whole_figure() {
        let line = Line::from(vec![
            label("sessions"),
            text(" 4   1.23 / 8.0 GiB   "),
            label("reused"),
            text(" 66.7%   new 4"),
        ]);
        let cut = fit(line, 30);
        let s: String = cut.spans.iter().map(|s| s.content.as_ref()).collect();
        assert!(cut.width() <= 30, "{s}");
        assert_eq!(s, "sessions 4   1.23 / 8.0 GiB …");
    }

    #[test]
    fn figures_and_bars() {
        assert_eq!(count(1234567.0), "1,234,567");
        assert_eq!(count(999.0), "999");
        assert_eq!(duration(59.0), "59s");
        assert_eq!(duration(61.0), "1m01s");
        assert_eq!(duration(3725.0), "1h02m");
        assert_eq!(width(&bar(10, &[(0.25, Style::new()), (0.25, dim())])), 10);
        assert_eq!(
            overview::spark(&[Some(0.0), Some(1.0), Some(2.0)], 3, true),
            "▁▄█"
        );
        assert_eq!(
            overview::spark(&[Some(0.0), None, Some(2.0)], 3, true),
            "▁ █"
        );
        // all zero: at the base
        assert_eq!(overview::spark(&[Some(0.0); 3], 3, true), "▁▁▁");
    }

    #[test]
    fn q_and_ctrl_c_stop_the_server() {
        let k = |code, modifiers| KeyEvent::new(code, modifiers);
        assert!(quits(&k(KeyCode::Char('q'), KeyModifiers::NONE)));
        assert!(quits(&k(KeyCode::Char('Q'), KeyModifiers::SHIFT)));
        assert!(quits(&k(KeyCode::Char('c'), KeyModifiers::CONTROL)));
        assert!(!quits(&k(KeyCode::Char('c'), KeyModifiers::NONE)));
        assert!(!quits(&k(KeyCode::Enter, KeyModifiers::NONE)));
        let mut ui = Ui::default();
        let ev = Event::Key(k(KeyCode::Char('q'), KeyModifiers::NONE));
        assert_eq!(
            handle(&ev, &mut ui, &snapshot(), &History::default()),
            Asked::Quit
        );
    }

    #[test]
    fn a_screen_draws_on_a_terminal() {
        let mut term = Terminal::new(TestBackend::new(100, 24)).unwrap();
        let mut h = History::default();
        h.push(history::Sample {
            phase: Phase::Decode,
            speeds: vec![(12, 15.0)],
            ..history::Sample::default()
        });
        let mut ui = Ui::default();
        term.draw(|fr| {
            draw_screen(
                fr.buffer_mut(),
                &snapshot(),
                &h,
                &mut ui,
                "myds4 · test",
                &log(5),
            )
        })
        .unwrap();
        let b = term.backend().buffer();
        assert!(row(b, 0).starts_with(" myds4  test"), "{}", row(b, 0));
        assert!(
            row(b, 0).contains("decoding  15.0 tok/s · up 1h02m"),
            "{}",
            row(b, 0)
        );
        assert!(row(b, 3).starts_with("─ Now "), "{}", row(b, 3));
        assert!(
            row(b, 4).contains("decoding") && row(b, 4).contains("1 waiting"),
            "{}",
            row(b, 4)
        );
    }

    /// Prints every tab drawn from the sample data, as text, to look at:
    /// cargo test … ui::tests::show_every_tab -- --ignored --nocapture
    #[test]
    #[ignore]
    fn show_every_tab() {
        for (tab, w, h) in [
            (Tab::Overview, 160, 46),
            (Tab::Kv, 160, 46),
            (Tab::Experts, 220, 52),
            (Tab::Experts, 160, 52),
            (Tab::Minute, 160, 46),
        ] {
            let (b, _) = draw(tab, w, h, |ui| {
                if tab == Tab::Kv {
                    ui.node = Some(frame::NodeKey { via: 3, at: 3000 });
                    let runs = [
                        ("<｜User｜>", true),
                        ("Summarize the design.", false),
                        ("<｜Assistant｜>", true),
                        ("</think>", true),
                    ]
                    .into_iter()
                    .chain([("The design keeps …", false)])
                    .map(|(s, special)| (s.to_string(), special))
                    .collect();
                    let answer = Some(Answer {
                        runs,
                        tokens: 38234,
                        ..Answer::default()
                    });
                    ui.text = Some(Text {
                        session: 3,
                        start: 3000,
                        end: 41234,
                        answer,
                    });
                }
            });
            println!("{tab:?} at {w} x {h}:\n{}\n", all(&b));
        }
    }
}
