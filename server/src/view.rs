//! The live view (--tui): the terminal's alternate screen on stderr, redrawn
//! twice a second from the engine's snapshot ([`engine::view`]). It shows the
//! requests and their decoding speed, the experts per decoded row (with what
//! substitution adds), memory on the GPU, in RAM and on the SSD, the
//! sessions' prefix tree and the latest log lines. Nothing is written to
//! files: on exit the normal screen comes back with the last log lines. The
//! q key, or Ctrl-C, stops the server as SIGINT does.
//!
//! The engine's log reaches the view through [`on_log`]; lines logged while
//! the view is not on the screen go to stderr as well. [`leave`] restores the
//! terminal on every way out: the normal end, the engine's fatal errors
//! ([`on_die`]), this program's own and a panic.

use std::collections::VecDeque;
use std::ffi::{CStr, c_char};
use std::io::{self, Stderr, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, TryLockError};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use ratatui::crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use ratatui::crossterm::terminal::{self, EnterAlternateScreen, LeaveAlternateScreen};
use ratatui::crossterm::{cursor, execute};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
use serde::Deserialize;
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

use crate::engine;

const RING: usize = 256; // log lines kept
const TAIL: u64 = 30; // log lines printed on exit
const FRAMES: usize = 21; // statistics kept: 10 s at two frames a second
const SPEEDS: usize = 60; // the requests' tokens/s, the last 30 s
const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

// ---- the engine's snapshot (myds4_view) ----

#[derive(Deserialize, Default)]
pub struct Snapshot {
    uptime_seconds: f64,
    phase: String,
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
    tree: Vec<Node>,
    decode: Decode,
    prompts: Prompts,
    cache: Cache,
    memory: Memory,
}

#[derive(Deserialize, Default)]
struct Req {
    number: u64,
    session: u64,
    prompt: u64,
    reused: u64,
    generated: u64,
}

#[derive(Deserialize, Default)]
struct Prefill {
    layer: u64,
    rows: u64,
}

#[derive(Deserialize, Default)]
struct Totals {
    prompt_tokens: u64,
    reused_tokens: u64,
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
    busy: bool,
    saved: i64, // that session's saved position, -1 for none
    idle_seconds: f64,
    gib: f64,
}

/// The statistics of decoded rows since the start: each row's selections by
/// where the expert was.
#[derive(Deserialize, Default, Clone, Copy)]
struct Decode {
    rows: u64,
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

#[derive(Deserialize, Default)]
struct Prompts {
    rows: u64,
    hits: u64,
    prefetch_hits: u64,
    misses: u64,
    substituted: u64,
    bytes: u64,
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

static RING_LINES: Mutex<Ring> = Mutex::new(Ring { lines: VecDeque::new(), logged: 0, printed: 0 });
static ON: AtomicBool = AtomicBool::new(false); // the alternate screen is up
static RAW: AtomicBool = AtomicBool::new(false); // the terminal is in raw mode for the keys
static DRAW: Mutex<()> = Mutex::new(()); // held while a frame is written

fn ring() -> MutexGuard<'static, Ring> {
    RING_LINES.lock().unwrap_or_else(|e| e.into_inner())
}

/// The engine's log lines (with their time), from the thread that logs.
pub extern "C" fn on_log(line: *const c_char) {
    // Safety: the engine passes a NUL-terminated line, valid during the call.
    let line = unsafe { CStr::from_ptr(line) }.to_string_lossy().into_owned();
    let mut r = ring();
    if !ON.load(Ordering::Acquire) {
        eprintln!("{line}");
        r.printed = r.logged + 1;
    }
    r.logged += 1;
    if r.lines.len() == RING {
        r.lines.pop_front();
    }
    r.lines.push_back(line);
}

/// Before the engine's fatal error message: the normal screen comes back.
pub extern "C" fn on_die() {
    leave();
}

/// The normal screen, the terminal's settings and the log lines it has not
/// shown yet. Any thread may call it, any number of times.
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
    let mut r = ring();
    if !ON.swap(false, Ordering::AcqRel) {
        return;
    }
    if RAW.swap(false, Ordering::AcqRel) {
        let _ = terminal::disable_raw_mode();
    }
    let mut err = io::stderr();
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

// ---- the frames ----

/// What the view keeps across frames: the speeds and the statistics windows.
pub struct Frames {
    tracks: Vec<(u64, VecDeque<(f64, u64)>)>, // a decoding request's (time, tokens) samples
    speeds: VecDeque<f64>,
    snaps: VecDeque<(f64, Decode)>,
    swapouts: Option<u64>, // the Mac's swap-outs when the view started
}

impl Frames {
    pub fn new() -> Frames {
        Frames { tracks: Vec::new(), speeds: VecDeque::from(vec![0.0; SPEEDS]), snaps: VecDeque::new(), swapouts: None }
    }

    /// A request's tokens/s over its last ten samples (five seconds).
    fn speed(&mut self, number: u64, generated: u64, now: f64) -> f64 {
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
}

fn dim() -> Style {
    Style::new().add_modifier(Modifier::DIM)
}

fn bold() -> Style {
    Style::new().add_modifier(Modifier::BOLD)
}

fn color(c: Color) -> Style {
    Style::new().fg(c)
}

fn label(s: impl Into<String>) -> Span<'static> {
    Span::styled(s.into(), dim())
}

fn text(s: impl Into<String>) -> Span<'static> {
    Span::raw(s.into())
}

fn strong(s: impl Into<String>) -> Span<'static> {
    Span::styled(s.into(), bold())
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
    format!("{:.1}%", if whole > 0.0 { 100.0 * part / whole } else { 0.0 })
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
        out.push(Span::styled("░".repeat(w - done), dim()));
    }
    out
}

/// The values' heights in eighths.
fn spark(v: &[f64]) -> String {
    const H: [&str; 8] = ["▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"];
    let top = v.iter().cloned().fold(0.0, f64::max);
    v.iter().map(|&x| H[if top > 0.0 { (7.99 * x / top) as usize } else { 0 }]).collect()
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
    let cells: Vec<(char, Style)> =
        line.spans.iter().flat_map(|s| s.content.chars().map(move |c| (c, s.style))).collect();
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
    let cut = if figure > 0 { figure } else if word > 0 { word } else { stop };
    let mut spans: Vec<Span<'static>> = Vec::new();
    for &(c, style) in &cells[..cut] {
        match spans.last_mut() {
            Some(last) if last.style == style => last.content.to_mut().push(c),
            _ => spans.push(Span::styled(c.to_string(), style)),
        }
    }
    spans.push(Span::styled(" …", dim()));
    Line::from(spans)
}

/// A section's rule.
fn head(w: usize, title: &str) -> Line<'static> {
    let rule = w.saturating_sub(title.width() + 3);
    Line::from(vec![
        Span::styled(format!("─ {title} "), bold().fg(Color::Cyan)),
        Span::styled("─".repeat(rule), color(Color::Cyan)),
    ])
}

/// A frame of w by h cells: the lines, each fitted to w. now: seconds since
/// the view started; log: the latest log lines.
pub fn frame(s: &Snapshot, f: &mut Frames, now: f64, title: &str, log: &[String], w: usize, h: usize) -> Vec<Line<'static>> {
    let gw = if w >= 110 { 24 } else { 10 }; // a gauge's cells
    let mut l: Vec<Line<'static>> = Vec::new();
    let up = format!("q quits · up {} ", duration(s.uptime_seconds));
    let title = format!(" {}", clip(title, w.saturating_sub(up.width() + 2)));
    let gap = w.saturating_sub(title.width() + up.width()).max(1);
    l.push(Line::from(Span::styled(
        format!("{title}{}{up}", " ".repeat(gap)),
        Style::new().add_modifier(Modifier::REVERSED),
    )));

    // now: the phase, the requests and their speed
    let decoding = s.phase == "decode";
    let mut sp = vec![0.0; s.requests.len()];
    for (i, r) in s.requests.iter().enumerate() {
        if r.generated > 0 {
            sp[i] = f.speed(r.number, r.generated, now);
        }
    }
    f.tracks.retain(|t| s.requests.iter().any(|r| r.number == t.0));
    let sum: f64 = sp.iter().sum();
    f.speeds.pop_front();
    f.speeds.push_back(if decoding { sum } else { 0.0 });
    l.push(head(w, "Now"));
    let queue = if s.waiting > 0 {
        vec![text("   "), Span::styled(format!("{} waiting", s.waiting), color(Color::Yellow))]
    } else {
        vec![]
    };
    let mut line = vec![text(" ")];
    if decoding {
        let ns = if w > 100 { 40 } else { 20 };
        let speeds: Vec<f64> = f.speeds.iter().skip(SPEEDS - ns).cloned().collect();
        line.extend(pad(vec![Span::styled("decoding", bold().fg(Color::Green))], 12));
        line.push(text(format!("{} request{}   ", s.requests.len(), if s.requests.len() == 1 { "" } else { "s" })));
        line.push(strong(format!("{sum:.1}")));
        line.push(text(" tok/s   "));
        line.push(Span::styled(spark(&speeds), color(Color::Green)));
    } else if s.phase == "prefill" && s.prefill.rows > 0 {
        let layers = s.layers.max(1) as f64;
        line.extend(pad(vec![Span::styled("prefilling", bold().fg(Color::Cyan))], 12));
        line.extend(bar(gw, &[(s.prefill.layer as f64 / layers, color(Color::Cyan))]));
        line.extend([text("  "), label("layer"), text(" "), strong(s.prefill.layer.to_string())]);
        line.extend([text(format!("/{}   ", s.layers)), label("tokens"), text(format!(" {}", count(s.prefill.rows as f64)))]);
    } else if s.phase == "prefill" {
        line.extend(pad(vec![Span::styled("prefilling", bold().fg(Color::Cyan))], 12));
        line.push(label("starting; images are encoded first"));
    } else {
        line.extend(pad(vec![strong("idle")], 12));
        line.push(text(duration(s.phase_seconds)));
    }
    line.extend(queue);
    l.push(Line::from(line));
    for (i, r) in s.requests.iter().enumerate().take(8) {
        let mut line = vec![text("   ")];
        line.extend(pad(vec![strong(format!("#{}", r.number))], 7));
        line.extend([label("session"), text(format!(" {:<5}", r.session))]);
        line.extend([label("prompt"), text(format!(" {:<8}", count(r.prompt as f64)))]);
        line.extend([label("reused"), text(format!(" {:<8}", count(r.reused as f64)))]);
        line.extend([label("generated"), text(format!(" {:<8}", count(r.generated as f64)))]);
        if r.generated > 0 {
            line.push(Span::styled(format!("{:.1} tok/s", sp[i]), color(Color::Green)));
        }
        l.push(Line::from(line));
    }

    // experts per decoded token, the last 10 s (or since start)
    let c = s.decode;
    if f.snaps.len() == FRAMES {
        f.snaps.pop_front();
    }
    f.snaps.push_back((now, c));
    let old = f.snaps.iter().find(|(t, d)| d.rows < c.rows && now - t <= 11.0).map(|&(_, d)| d);
    let b = old.unwrap_or_default();
    let rows = c.rows.saturating_sub(b.rows) as f64;
    let part = [
        c.in_memory.saturating_sub(b.in_memory) as f64,
        c.read_ahead.saturating_sub(b.read_ahead) as f64,
        c.late.saturating_sub(b.late) as f64,
        c.substituted.saturating_sub(b.substituted) as f64,
        c.from_ssd.saturating_sub(b.from_ssd) as f64,
    ];
    let counted: f64 = part.iter().sum();
    let bytes = c.bytes.saturating_sub(b.bytes) as f64;
    let per_row = s.experts_per_row as f64;
    l.push(head(
        w,
        &if rows > 0.0 {
            format!("Experts per token · {} · {} tokens", if old.is_some() { "last 10 s" } else { "since start" }, count(rows))
        } else {
            "Experts per token".into()
        },
    ));
    if rows > 0.0 && counted > 0.0 {
        const NAME: [&str; 5] = ["in memory", "read ahead", "late", "substituted", "from SSD"];
        let colors = [Color::Green, Color::Cyan, Color::Yellow, Color::Magenta, Color::Red];
        let shares: Vec<(f64, Style)> = (0..5).map(|i| (part[i] / counted, color(colors[i]))).collect();
        let each = counted / rows;
        let mut line = vec![text(" ")];
        line.extend(bar(if w >= 110 { 40 } else { 20 }, &shares));
        line.extend([text("   "), label("hit rate"), text(" "), strong(pct(part[0] + part[1] + part[2], counted))]);
        if part[3] > 0.0 {
            line.extend([text("   "), label("with substitutes"), text(" "), strong(pct(counted - part[4], counted))]);
        }
        if each < per_row - 0.05 {
            line.extend([text("   "), label("counted"), text(format!(" {each:.1}/{}", s.experts_per_row))]);
        }
        l.push(Line::from(line));
        // a table: the sources' names, experts per token, shares
        let mut t = [vec![text(format!(" {:10}", ""))], vec![text(" "), label("experts   ")], vec![text(" "), label("share     ")]];
        for i in 0..5 {
            let cw = NAME[i].len() + 4;
            t[0].extend([text("  "), Span::styled("■", color(colors[i])), text(" "), label(NAME[i])]);
            t[1].push(text(format!("{:>cw$.1}", part[i] / rows)));
            t[2].push(text(format!("{:>cw$}", pct(part[i], counted))));
        }
        l.extend(t.into_iter().map(Line::from));
        let wall = c.seconds - b.seconds;
        let gpu = c.gpu_seconds - b.gpu_seconds;
        let remaps = c.remaps.saturating_sub(b.remaps) as f64 / rows; // slots mapped for the GPU again: its memory was short
        let mut line = vec![text(" "), label("per token "), text(" "), label("SSD")];
        let rate = if wall > 0.0 { bytes / wall / 1e9 } else { 0.0 };
        line.push(text(format!(" {:.2} GB · {rate:.1} GB/s   ", bytes / rows / 1e9)));
        line.extend([label("wait"), text(format!(" {:.1} ms   ", (c.read_wait_seconds - b.read_wait_seconds) / rows * 1e3))]);
        line.extend([label("GPU"), text(format!(" {:.1} ms · {} busy   ", gpu / rows * 1e3, pct(gpu, wall)))]);
        line.push(label("read ahead"));
        line.push(text(format!(
            " {:.1} · {:.1} cancelled",
            c.prefetches.saturating_sub(b.prefetches) as f64 / rows,
            c.cancelled.saturating_sub(b.cancelled) as f64 / rows
        )));
        if remaps >= 0.05 {
            line.extend([text("   "), Span::styled(format!("remapped {remaps:.1}"), color(Color::Yellow))]);
        }
        l.push(Line::from(line));
    } else if rows > 0.0 {
        // only --gpu-route layers, which the GPU looks up without counting
        let note = format!("{} tokens; the GPU looked up all their experts (--gpu-route)", count(rows));
        l.push(Line::from(vec![text(" "), label(note)]));
    } else {
        l.push(Line::from(vec![text(" "), label("no decoded tokens yet")]));
    }
    let p = &s.prompts;
    if p.rows > 0 {
        // a prompt batch reads each expert once for all its rows
        let used = (p.hits + p.prefetch_hits + p.misses) as f64;
        let sub = p.substituted as f64 / p.rows as f64;
        let mut line = vec![text(" "), label("prompts   "), text(format!(" {} tokens", count(p.rows as f64)))];
        if p.substituted > 0 {
            line.extend([text("   "), label("substituted")]);
            line.push(text(format!(" {sub:.1}/{} · {}", s.experts_per_row, pct(sub, per_row))));
        }
        line.extend([text("   "), label("in memory"), text(format!(" {}   ", pct((p.hits + p.prefetch_hits) as f64, used)))]);
        line.extend([label("read"), text(format!(" {:.1} GB", p.bytes as f64 / 1e9))]);
        l.push(Line::from(line));
    }

    // memory: a row each for the GPU, RAM, the SSD and the Mac: a gauge, the main figure, the parts
    l.push(head(w, "Memory"));
    let m = &s.memory;
    let k = &s.cache;
    let gb = k.slot_bytes as f64 / GIB;
    let mapped = k.mapped as f64 * gb;
    let weights = m.dense as f64 / GIB;
    let inuse = m.gpu_in_use as f64 / GIB;
    let mine = mapped + weights;
    let lim = m.gpu_limit.map_or(0.0, |b| b as f64 / GIB);
    let ram = m.total as f64 / GIB;
    let fw = 18; // the main figure's columns
    let share = |x: f64, whole: f64| if whole > 0.0 { x / whole } else { 0.0 };
    {
        // the driver's resident memory (all apps): myds4's mapped experts and weights while the GPU works; it releases them while it idles
        let all = inuse >= mine;
        let fig = if lim > 0.0 {
            vec![strong(format!("{inuse:.1}")), text(format!(" / {lim:.1} GiB"))]
        } else {
            vec![strong(format!("{inuse:.1}")), text(" GiB")]
        };
        let mut line = vec![text(" "), strong("GPU"), text("  ")];
        let slots = label(format!("({} slots)", count(k.mapped as f64)));
        if all {
            line.extend(bar(
                gw,
                &[(share(mapped, lim), color(Color::Blue)), (share(weights, lim), color(Color::Cyan)), (share(inuse - mine, lim), dim())],
            ));
            line.push(text("  "));
            line.extend(pad(fig, fw));
            line.extend([Span::styled("■", color(Color::Blue)), text(" "), label("experts"), text(format!(" {mapped:.1} ")), slots]);
            line.extend([text("   "), Span::styled("■", color(Color::Cyan)), text(" "), label("weights")]);
            line.extend([text(format!(" {weights:.1}   ")), label("other"), text(format!(" {:.1}", inuse - mine))]);
        } else {
            line.extend(bar(gw, &[(share(inuse, lim), Style::new())]));
            line.push(text("  "));
            line.extend(pad(fig, fw));
            line.extend([label("idle; resident when busy:"), text(" "), label("experts"), text(format!(" {mapped:.1} ")), slots]);
            line.extend([text("   "), label("weights"), text(format!(" {weights:.1}"))]);
        }
        l.push(Line::from(line));
    }
    {
        // myds4's footprint against the RAM: its experts and the rest
        let foot = m.footprint as f64 / GIB;
        let ex = k.slots as f64 * gb;
        let rest = (foot - ex).max(0.0);
        let mut line = vec![text(" "), strong("RAM"), text("  ")];
        line.extend(bar(gw, &[(share(ex, ram), color(Color::Blue)), (share(rest, ram), color(Color::Cyan))]));
        line.push(text("  "));
        line.extend(pad(vec![label("myds4"), text(" "), strong(format!("{foot:.1}")), text(" GiB")], fw));
        line.extend([Span::styled("■", color(Color::Blue)), text(" "), label("experts"), text(format!(" {ex:.1}   "))]);
        line.extend([Span::styled("■", color(Color::Cyan)), text(" "), label("rest"), text(format!(" {rest:.1}   ")), label("slots")]);
        line.push(text(format!(" {} / {} ", count(k.slots as f64), count(k.cap as f64))));
        line.push(label(format!("(max {})", count(k.max_slots as f64))));
        l.push(Line::from(line));
    }
    {
        // the experts that are only on the SSD
        let mut line = vec![text(" "), strong("SSD"), text("  ")];
        line.extend(bar(gw, &[(share(k.uncached as f64, k.experts as f64), Style::new())]));
        line.push(text("  "));
        line.extend(pad(vec![strong(format!("{:.1}", k.uncached_bytes as f64 / 1e9)), text(" GB")], fw));
        line.extend([label("experts"), text(format!(" {} / {}   ", count(k.uncached as f64), count(k.experts as f64)))]);
        line.extend([label("evicted"), text(format!(" {}   ", count(k.evictions as f64)))]);
        line.extend([label("given back"), text(format!(" {}", count(k.given_back as f64)))]);
        l.push(Line::from(line));
    }
    {
        // the Mac's memory in use: without the file cache, then the file cache
        let swap0 = *f.swapouts.get_or_insert(m.swapouts);
        let spare = (m.free.saturating_add(m.file).min(m.total)) as f64 / GIB;
        let file = m.file as f64 / GIB;
        let used = ram - spare;
        let level = if m.pressure >= 4 {
            Span::styled("critical", color(Color::Red))
        } else if m.pressure >= 2 {
            Span::styled("warning", color(Color::Yellow))
        } else {
            Span::styled("normal", color(Color::Green))
        };
        let mut line = vec![text(" "), strong("Mac"), text("  ")];
        line.extend(bar(gw, &[(share(used, ram), Style::new()), (share(file, ram), dim())]));
        line.push(text("  "));
        line.extend(pad(vec![strong(format!("{used:.1}")), text(format!(" / {ram:.0} GiB"))], fw));
        line.extend([label("available"), text(" "), strong(format!("{spare:.1}")), text("   "), label("pressure"), text(" "), level]);
        line.extend([text("   "), label("file cache"), text(format!(" {file:.1}   ")), label("wired")]);
        line.extend([text(format!(" {:.1}   ", m.wired as f64 / GIB)), label("swap"), text(format!(" {:.1}", m.swap_used as f64 / GIB))]);
        l.push(Line::from(line));
        l.push(Line::from(vec![
            text("      "),
            label(format!(
                "compressor {:.1} GiB holding {:.1} · kernel cache target {:.1} · {} swap-outs since start",
                m.compressor as f64 / GIB,
                m.compressed as f64 / GIB,
                m.file_kept as f64 / GIB,
                count(m.swapouts.saturating_sub(swap0) as f64)
            )),
        ]));
    }

    // the sessions' prefix tree
    l.push(head(w, "KV cache"));
    let t = &s.totals;
    let mut line = vec![text(" "), label("sessions"), text(format!(" {}   {:.2} / {:.1} GiB   ", s.sessions, s.kv_gib, s.kv_limit_gib))];
    line.extend([label("reused"), text(format!(" {}   ", pct(t.reused_tokens as f64, t.prompt_tokens as f64)))]);
    line.extend([label("new"), text(format!(" {}   ", s.kinds.new)), label("continued"), text(format!(" {}   ", s.kinds.continued))]);
    line.extend([label("saved"), text(format!(" {}   ", s.kinds.saved)), label("forked"), text(format!(" {}", s.kinds.forked))]);
    l.push(Line::from(line));
    let room = h as i64 - l.len() as i64 - 3;
    let nodes = (room - room / 3).clamp(0, s.tree.len() as i64) as usize;
    for x in &s.tree[..nodes] {
        let mut line = vec![text(" "), Span::styled(x.lines.clone(), dim()), text(if x.session > 0 { "●" } else { "○" })];
        line.push(text(format!(" {}–{} ", count(x.start as f64), count(x.end as f64))));
        line.push(label(format!("({} tok)", count(x.end.saturating_sub(x.start) as f64))));
        if x.sessions > 1 {
            line.extend([text(format!("   {} ", x.sessions)), label("sessions")]);
        }
        if x.hits > 0 {
            line.extend([text("   "), strong(x.hits.to_string()), text(" "), label(if x.hits == 1 { "hit" } else { "hits" })]);
        }
        if x.session > 0 {
            line.extend([text("   "), label("session"), text(format!(" {}   ", x.session))]);
            if x.busy {
                line.push(Span::styled("decoding", color(Color::Green)));
            } else {
                line.extend([label("idle"), text(format!(" {}", duration(x.idle_seconds)))]);
            }
            if x.saved >= 0 {
                line.extend([text("   "), label("saved at"), text(format!(" {}", count(x.saved as f64)))]);
            }
            line.push(text(format!("   {:.2} GiB", x.gib)));
        }
        l.push(Line::from(line));
    }
    if nodes < s.tree.len() {
        l.push(Line::from(Span::styled(format!("   … {} more", s.tree.len() - nodes), dim())));
    }

    // the log's last lines
    l.push(head(w, "Log"));
    let show = h.saturating_sub(l.len());
    for line in &log[log.len().saturating_sub(show)..] {
        l.push(Line::from(Span::styled(format!(" {line}"), dim())));
    }
    l.truncate(h);
    l.into_iter().map(|line| fit(line, w)).collect()
}

/// Whether a key stops the server: q, or Ctrl-C (raw mode delivers it as a key).
fn quits(k: &KeyEvent) -> bool {
    k.kind == KeyEventKind::Press
        && (matches!(k.code, KeyCode::Char('q' | 'Q'))
            || (k.code == KeyCode::Char('c') && k.modifiers.contains(KeyModifiers::CONTROL)))
}

// ---- the view's thread ----

/// The view on the screen; [`View::finish`] ends it.
pub struct View {
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

/// The alternate screen and the thread that draws it; None if the terminal
/// cannot show it. Keys work where the terminal goes into raw mode.
pub fn start(title: String) -> Option<View> {
    let keys = terminal::enable_raw_mode().is_ok();
    RAW.store(keys, Ordering::Release);
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
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        leave();
        hook(info)
    }));
    let mut screen = match Terminal::new(CrosstermBackend::new(io::stderr())) {
        Ok(t) => t,
        Err(_) => {
            leave();
            return None;
        }
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

fn run(screen: &mut Terminal<CrosstermBackend<Stderr>>, title: &str, keys: bool, stop: &AtomicBool) {
    let started = Instant::now();
    let mut frames = Frames::new();
    let mut keys = keys;
    while !stop.load(Ordering::Acquire) {
        // The snapshot before the screen's lock: an engine failing while it
        // holds what the snapshot needs can still restore the terminal.
        let snapshot = serde_json::from_str::<Snapshot>(&engine::view());
        let log: Vec<String> = ring().lines.iter().cloned().collect();
        if let Ok(s) = snapshot {
            let _draw = DRAW.lock().unwrap_or_else(|e| e.into_inner());
            if !ON.load(Ordering::Acquire) {
                return;
            }
            let now = started.elapsed().as_secs_f64();
            let _ = screen.draw(|f| {
                let area = f.area();
                let lines = frame(&s, &mut frames, now, title, &log, area.width as usize, area.height as usize);
                f.render_widget(Paragraph::new(lines), area);
            });
        }
        keys = wait(keys, Duration::from_millis(500));
    }
}

/// The wait for the next frame, reading the keys typed meanwhile; false once
/// the keys cannot be read (the wait is a sleep from then on).
fn wait(keys: bool, time: Duration) -> bool {
    if !keys {
        thread::sleep(time);
        return false;
    }
    let until = Instant::now() + time;
    while let Some(left) = until.checked_duration_since(Instant::now()) {
        match event::poll(left) {
            Ok(false) => break,
            Ok(true) => match event::read() {
                Ok(Event::Key(k)) if quits(&k) => {
                    if !engine::stopping() {
                        engine::log("q: stopping after the current step");
                    }
                    engine::request_stop();
                }
                Ok(_) => {}
                Err(_) => {
                    thread::sleep(left);
                    return false;
                }
            },
            Err(_) => {
                thread::sleep(left);
                return false;
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::backend::TestBackend;

    // A snapshot in the form engine.c's myds4_view writes, decoding two requests.
    const SNAPSHOT: &str = r#"{"uptime_seconds": 3725, "phase": "decode", "phase_seconds": 2.5, "waiting": 1,
        "requests": [{"number": 12, "session": 3, "prompt": 41234, "reused": 40000, "generated": 77},
                     {"number": 13, "session": 4, "prompt": 900, "reused": 0, "generated": 5}],
        "layers": 40, "experts_per_row": 240, "prefill": {"layer": 0, "rows": 0},
        "totals": {"requests": 13, "prompt_tokens": 90000, "reused_tokens": 60000, "generated_tokens": 4000},
        "kinds": {"new": 4, "continued": 6, "saved": 2, "forked": 1},
        "sessions": 4, "kv_gib": 1.234, "kv_limit_gib": 8.000, "tree": [
          {"lines": "", "start": 0, "end": 3000, "sessions": 3, "hits": 5, "session": 0, "busy": false, "saved": -1, "idle_seconds": 0.0, "gib": 0.000},
          {"lines": "├─ ", "start": 3000, "end": 41234, "sessions": 1, "hits": 2, "session": 3, "busy": true, "saved": 41000, "idle_seconds": 0.0, "gib": 0.600},
          {"lines": "└─ ", "start": 3000, "end": 9000, "sessions": 2, "hits": 0, "session": 0, "busy": false, "saved": -1, "idle_seconds": 0.0, "gib": 0.000},
          {"lines": "   ├─ ", "start": 9000, "end": 9500, "sessions": 1, "hits": 0, "session": 1, "busy": false, "saved": 9400, "idle_seconds": 75.0, "gib": 0.100},
          {"lines": "   └─ ", "start": 9000, "end": 9900, "sessions": 1, "hits": 0, "session": 2, "busy": false, "saved": -1, "idle_seconds": 4000.0, "gib": 0.120}],
        "decode": {"rows": 5000, "seconds": 210.5, "gpu_seconds": 150.2, "in_memory": 1100000, "read_ahead": 90000,
                   "late": 2000, "substituted": 30000, "from_ssd": 68000, "bytes": 400000000000,
                   "read_wait_seconds": 30.1, "prefetches": 95000, "cancelled": 800, "remaps": 900},
        "prompts": {"rows": 120000, "hits": 900000, "prefetch_hits": 100, "misses": 200000, "substituted": 0, "bytes": 800000000000},
        "cache": {"slots": 3500, "mapped": 3260, "cap": 3600, "max_slots": 4000, "slot_bytes": 11534336, "experts": 15360,
                  "uncached": 11860, "uncached_bytes": 86000000000, "evictions": 123456, "given_back": 12},
        "memory": {"footprint": 52000000000, "dense": 8600000000, "pinned": 900000000, "gpu_in_use": 47000000000,
                   "gpu_limit": null, "total": 137438953472, "free": 9000000000, "file": 30000000000, "file_kept": 4000000000,
                   "wired": 20000000000, "compressor": 1000000000, "compressed": 3000000000, "swap_used": 0, "swapouts": 1000,
                   "pressure": 1}}"#;

    fn snapshot() -> Snapshot {
        serde_json::from_str(SNAPSHOT).expect("myds4_view's form")
    }

    #[test]
    fn mac_memory_counts_file_cache_as_available() {
        let mut s = snapshot();
        s.memory.total = 128 * 1024u64.pow(3);
        s.memory.free = 58 * 1024u64.pow(3);
        s.memory.file = 36 * 1024u64.pow(3);
        let mut f = Frames::new();
        let lines = frame(&s, &mut f, 1.0, "myds4", &[], 240, 60);
        let mac = lines.iter().map(|l| l.spans.iter().map(|s| s.content.as_ref()).collect::<String>())
            .find(|l| l.starts_with(" Mac ")).expect("Mac memory row");
        assert!(mac.contains("34.0 / 128 GiB"), "{mac}");
        assert!(mac.contains("available 94.0"), "{mac}");
        assert!(mac.contains("file cache 36.0"), "{mac}");
    }

    fn log(n: usize) -> Vec<String> {
        (0..n).map(|i| format!("12:00:{:02}.000 req {i}: 1,000 prompt tokens, session 3 (kept), 900 reused", i % 60)).collect()
    }

    fn row(b: &ratatui::buffer::Buffer, y: u16) -> String {
        (0..b.area.width).map(|x| b[(x, y)].symbol().to_string()).collect()
    }

    #[test]
    fn a_frame_fits_the_terminal_with_the_title_first() {
        for (w, h) in [(100, 24), (160, 50), (40, 12), (12, 4)] {
            let mut f = Frames::new();
            let lines = frame(&snapshot(), &mut f, 1.0, "myds4 · 127.0.0.1:8008 · deepseek-v4.1-flash · GGUF experts", &log(80), w, h);
            assert!(lines.len() <= h, "{w}x{h}: {} lines", lines.len());
            assert!(lines.iter().all(|l| l.width() <= w), "{w}x{h}: a line is wider than the terminal");
            let first: String = lines[0].spans.iter().map(|s| s.content.as_ref()).collect();
            assert!(first.starts_with(" myds4") || w < 20, "{w}x{h}: {first}");
            if w >= 40 {
                assert!(first.ends_with("q quits · up 1h02m "), "{w}x{h}: {first}");
            }
        }
    }

    #[test]
    fn the_log_fills_the_rows_below_the_tree() {
        let mut f = Frames::new();
        let lines = frame(&snapshot(), &mut f, 1.0, "myds4", &log(80), 160, 50);
        assert_eq!(lines.len(), 50);
        let last: String = lines[49].spans.iter().map(|s| s.content.as_ref()).collect();
        assert!(last.contains("req 79:"), "{last}");
        let text: Vec<String> = lines.iter().map(|l| l.spans.iter().map(|s| s.content.as_ref()).collect()).collect();
        assert!(text.iter().any(|t| t.contains("●") && t.contains("decoding")), "the busy session's node");
    }

    #[test]
    fn speeds_come_from_the_samples_of_each_request() {
        let mut f = Frames::new();
        assert_eq!(f.speed(12, 10, 0.0), 0.0);
        assert!((f.speed(12, 30, 1.0) - 20.0).abs() < 1e-9);
        assert!((f.speed(12, 40, 2.0) - 15.0).abs() < 1e-9);
    }

    #[test]
    fn a_long_line_ends_after_a_whole_figure() {
        let line = Line::from(vec![label("sessions"), text(" 4   1.23 / 8.0 GiB   "), label("reused"), text(" 66.7%   new 4")]);
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
        assert_eq!(spark(&[0.0, 1.0, 2.0]), "▁▄█");
    }

    #[test]
    fn q_and_ctrl_c_stop_the_server() {
        let key = |code, modifiers| KeyEvent::new(code, modifiers);
        assert!(quits(&key(KeyCode::Char('q'), KeyModifiers::NONE)));
        assert!(quits(&key(KeyCode::Char('Q'), KeyModifiers::SHIFT)));
        assert!(quits(&key(KeyCode::Char('c'), KeyModifiers::CONTROL)));
        assert!(!quits(&key(KeyCode::Char('c'), KeyModifiers::NONE)));
        assert!(!quits(&key(KeyCode::Enter, KeyModifiers::NONE)));
    }

    #[test]
    fn a_frame_draws_on_a_terminal() {
        let mut screen = Terminal::new(TestBackend::new(100, 24)).unwrap();
        let mut f = Frames::new();
        screen
            .draw(|fr| {
                let area = fr.area();
                let lines = frame(&snapshot(), &mut f, 1.0, "myds4 · test", &log(5), area.width as usize, area.height as usize);
                fr.render_widget(Paragraph::new(lines), area);
            })
            .unwrap();
        let b = screen.backend().buffer();
        assert!(row(b, 0).starts_with(" myds4 · test"), "{}", row(b, 0));
        assert!(row(b, 1).starts_with("─ Now "), "{}", row(b, 1));
        assert!(row(b, 2).contains("decoding") && row(b, 2).contains("1 waiting"), "{}", row(b, 2));
    }
}
