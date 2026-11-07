//! The view's chrome: the title and the server's state, the tabs, and the
//! keys of the tab shown at the bottom; and what the person chose in it.

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};

use super::color;
use super::theme::{c, get};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Tab {
    Overview,
    Kv,
    Experts,
    Minute,
    Log,
}

pub const TABS: [(Tab, &str); 5] =
    [(Tab::Overview, "Overview"), (Tab::Kv, "KV cache"), (Tab::Experts, "Experts"), (Tab::Minute, "Last minute"), (Tab::Log, "Log")];

/// What a click on a place of the screen chooses.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Target {
    Tab(Tab),
    /// A node of the KV tree: its index in the snapshot's tree.
    Node(usize),
    /// A request of the expert map.
    Request(u64),
}

/// A place a click acts on: columns x0..x1 of row y.
#[derive(Clone, Copy, Debug)]
pub struct Hit {
    pub y: u16,
    pub x0: u16,
    pub x1: u16,
    pub target: Target,
}

/// A node of the KV tree as the person chose it: a state through it and a
/// position in it. The tree changes between snapshots: a node grows while its
/// state decodes, a fork splits it, an eviction merges it. The key finds the
/// node holding the position on the state's path ([`super::kv::find`]).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct NodeKey {
    pub via: u64,
    pub at: u64,
}

/// The selected node's text while its panel is open: the session and
/// positions asked for, and the worker's answer once it comes (it answers
/// between its steps).
#[derive(Clone, Debug, PartialEq)]
pub struct Text {
    pub session: u64,
    pub start: u64,
    pub end: u64,
    pub answer: Option<Answer>,
}

/// The text of a node's first `tokens` tokens in runs, each special token
/// alone and marked true (engine::text); and the runs in lines of `width`
/// columns, as the panel drew them last (width 0: not yet).
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Answer {
    pub runs: Vec<(String, bool)>,
    pub tokens: usize,
    pub lines: Vec<Line<'static>>,
    pub width: usize,
}

/// What the person chose in the view.
pub struct Ui {
    /// The model the view shows (an index of the worker's models).
    pub model: usize,
    pub tab: Tab,
    pub node: Option<NodeKey>,
    pub text: Option<Text>,
    /// The text's first line shown.
    pub scroll: usize,
    /// The expert map's request; none: the first with a routed row.
    pub request: Option<u64>,
    /// Whether the view takes the mouse (off: the terminal selects text).
    pub mouse: bool,
    /// The log's lines scrolled back from the newest.
    pub log_back: usize,
    /// The clickable places of the frame drawn last.
    pub hits: Vec<Hit>,
    /// The lines a page holds in the frame drawn last: the text panel's, the log's.
    pub page: usize,
}

impl Default for Ui {
    fn default() -> Ui {
        Ui {
            model: 0,
            tab: Tab::Overview,
            node: None,
            text: None,
            scroll: 0,
            request: None,
            mouse: true,
            log_back: 0,
            hits: Vec::new(),
            page: 10,
        }
    }
}

impl Ui {
    /// What a click at column x of row y chooses.
    pub fn target(&self, x: u16, y: u16) -> Option<Target> {
        self.hits.iter().find(|h| h.y == y && (h.x0..h.x1).contains(&x)).map(|h| h.target)
    }
}

pub fn label(s: impl Into<String>) -> Span<'static> {
    Span::styled(s.into(), Style::new().fg(c(get().label)))
}

pub fn text(s: impl Into<String>) -> Span<'static> {
    Span::raw(s.into())
}

pub fn strong(s: impl Into<String>) -> Span<'static> {
    Span::styled(s.into(), Style::new().add_modifier(Modifier::BOLD))
}

/// Draws the chrome; returns the area left for the tab.
pub fn draw(buf: &mut Buffer, area: Rect, title: &str, state: Line<'static>, ui: &mut Ui) -> Rect {
    if area.height < 5 || area.width < 20 {
        return Rect { height: 0, ..area };
    }
    let t = get();
    let w = area.width;
    // the title, and the state at the right
    let sw = state.width() as u16 + 1;
    let room = w.saturating_sub(sw + 1) as usize;
    let head = Line::from(vec![strong(" myds4  "), label(super::clip(title.trim_start_matches("myds4 · "), room.saturating_sub(8)))]);
    buf.set_line(area.x, area.y, &head, w);
    if sw + 12 <= w {
        buf.set_line(area.right() - sw, area.y, &state, sw);
    }
    // the tabs, then their rule
    let tabs = TABS
        .iter()
        .enumerate()
        .map(|(i, &(tab, name))| (Line::from(vec![label(format!(" {} ", i + 1)), text(format!("{name} "))]), Target::Tab(tab), tab == ui.tab))
        .collect();
    tab_row(buf, area.x + 1, area.y + 1, area.right(), tabs, &mut ui.hits);
    buf.set_string(area.x, area.y + 2, "─".repeat(w as usize), color(t.rule));
    // the keys
    let mut spans = vec![text(" ")];
    for (k, what) in keys(ui) {
        spans.extend([strong(k), text(" "), label(what), text("   ")]);
    }
    buf.set_line(area.x, area.bottom() - 1, &super::fit(Line::from(spans), w as usize), w);
    Rect { x: area.x, y: area.y + 3, width: w, height: area.height - 4 }
}

/// Items in a row, as tabs: from column x of row y up to `right`, two
/// columns apart, each its line and the target a click on it chooses; the
/// chosen one on the accent.
pub fn tab_row(buf: &mut Buffer, mut x: u16, y: u16, right: u16, items: Vec<(Line<'static>, Target, bool)>, hits: &mut Vec<Hit>) {
    let t = get();
    for (line, target, chosen) in items {
        let w = line.width() as u16;
        if x + w > right {
            break;
        }
        if chosen {
            let plain: String = line.spans.iter().map(|s| s.content.as_ref()).collect();
            buf.set_string(x, y, plain, color(t.on_accent).bg(c(t.accent)).add_modifier(Modifier::BOLD));
        } else {
            buf.set_line(x, y, &line, w);
        }
        hits.push(Hit { y, x0: x, x1: x + w, target });
        x += w + 2;
    }
}

fn keys(ui: &Ui) -> Vec<(&'static str, &'static str)> {
    let mut k = vec![("1–5", "tabs")];
    match ui.tab {
        Tab::Kv if ui.text.is_some() => {
            k.extend([("←→", "node"), ("↑↓", "row"), ("PgUp PgDn Home End", "scroll the text"), ("esc", "close it")])
        }
        Tab::Kv => k.extend([("←→", "node"), ("↑↓", "row"), ("⏎ or click", "its text")]),
        Tab::Experts => k.extend([("[ ] or click", "request")]),
        Tab::Log => k.extend([("↑↓ PgUp PgDn", "scroll"), ("Home End", "oldest, newest")]),
        _ => {}
    }
    k.push(("m", if ui.mouse { "mouse: clicks" } else { "mouse: selects text" }));
    k.push(("q", "stop the server"));
    k
}
