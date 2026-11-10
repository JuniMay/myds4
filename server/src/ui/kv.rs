//! The KV cache tab: the sessions' prefix tree along the token positions,
//! branching at any depth, and the text of the selected node.

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

use super::frame::{Hit, NodeKey, Target, Ui, label, strong, tab_row, text};
use super::theme::{Rgb, c, get, mix};
use super::{GIB, Node, Snapshot, color, count, duration, pct};

/// A node's place in the drawing.
pub struct Place {
    pub parent: Option<usize>,
    pub children: Vec<usize>,
    pub row: usize,
}

/// The tree's rows. The snapshot lists the nodes depth first, each after a
/// box-drawing prefix of three columns per level. A node's first child goes
/// on in its row; each other child takes the rows below the subtrees before it.
pub fn places(tree: &[Node]) -> (Vec<Place>, usize) {
    let mut p: Vec<Place> = Vec::with_capacity(tree.len());
    let mut last: Vec<usize> = Vec::new(); // the latest node at each depth
    for (i, n) in tree.iter().enumerate() {
        let depth = n.lines.chars().count() / 3;
        let parent = if depth > 0 && last.len() >= depth {
            Some(last[depth - 1])
        } else {
            None
        };
        last.truncate(depth);
        last.push(i);
        if let Some(q) = parent {
            p[q].children.push(i);
        }
        p.push(Place {
            parent,
            children: Vec::new(),
            row: 0,
        });
    }
    fn place(p: &mut [Place], i: usize, rows: &mut usize) -> usize {
        let kids = p[i].children.clone();
        let row = match kids.split_first() {
            None => {
                *rows += 1;
                *rows - 1
            }
            Some((&first, rest)) => {
                let r = place(p, first, rows);
                for &k in rest {
                    place(p, k, rows);
                }
                r
            }
        };
        p[i].row = row;
        row
    }
    let mut rows = 0;
    for i in 0..p.len() {
        if p[i].parent.is_none() {
            place(&mut p, i, &mut rows);
        }
    }
    (p, rows)
}

/// The node the key names: on the path of the key's state, the node holding
/// its position; none once the state has left the tree.
pub fn find(tree: &[Node], key: NodeKey) -> Option<usize> {
    let holds = |n: &Node| n.start <= key.at && key.at < n.end.max(n.start + 1);
    let (p, _) = places(tree);
    // from the state's own node up its path
    let mut at = tree.iter().position(|n| n.session == key.via);
    while let Some(i) = at {
        if holds(&tree[i]) {
            return Some(i);
        }
        at = p[i].parent;
    }
    // a tree cut short before the state's own node
    tree.iter().position(|n| n.via == key.via && holds(n))
}

/// The key of a node: the state through it used last, and its first position.
pub fn key(n: &Node) -> NodeKey {
    NodeKey {
        via: n.via,
        at: n.start,
    }
}

/// A move in the tree.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Move {
    Left,
    Right,
    Up,
    Down,
}

/// The node a move leads to from node i (from none: the first node): along
/// its row, or to the row above or below, the node there nearest its start.
pub fn step(tree: &[Node], from: Option<usize>, m: Move) -> Option<usize> {
    if tree.is_empty() {
        return None;
    }
    let Some(i) = from.filter(|&i| i < tree.len()) else {
        return Some(0);
    };
    let (p, rows) = places(tree);
    let r = p[i].row;
    let in_row = |row: usize| -> Vec<usize> {
        let mut v: Vec<usize> = (0..tree.len()).filter(|&k| p[k].row == row).collect();
        v.sort_by_key(|&k| tree[k].start);
        v
    };
    match m {
        Move::Left | Move::Right => {
            let row = in_row(r);
            let at = row.iter().position(|&k| k == i)?;
            let next = if m == Move::Left {
                at.checked_sub(1)
            } else {
                Some(at + 1)
            };
            next.and_then(|j| row.get(j).copied()).or(Some(i))
        }
        Move::Up | Move::Down => {
            let target = if m == Move::Up {
                r.checked_sub(1)
            } else {
                Some(r + 1).filter(|&x| x < rows)
            };
            let Some(target) = target else { return Some(i) };
            let row = in_row(target);
            let s = tree[i].start;
            row.iter()
                .copied()
                .min_by_key(|&k| {
                    if tree[k].start <= s && s < tree[k].end.max(tree[k].start + 1) {
                        0
                    } else {
                        tree[k].start.abs_diff(s) + 1
                    }
                })
                .or(Some(i))
        }
    }
}

/// Whether a decoding session goes through each node, and the shortest idle
/// time of the sessions through it.
fn liveness(tree: &[Node], p: &[Place]) -> (Vec<bool>, Vec<f64>) {
    let mut busy: Vec<bool> = tree.iter().map(|n| n.session > 0 && n.busy).collect();
    let mut idle: Vec<f64> = tree
        .iter()
        .map(|n| {
            if n.session > 0 {
                n.idle_seconds
            } else {
                f64::INFINITY
            }
        })
        .collect();
    for i in (0..tree.len()).rev() {
        if let Some(q) = p[i].parent {
            busy[q] |= busy[i];
            idle[q] = idle[q].min(idle[i]);
        }
    }
    (busy, idle)
}

/// A line's color: the accent while a session through it decodes, else gray,
/// paler with its sessions' idle time.
fn tone(busy: bool, idle: f64) -> Rgb {
    let t = get();
    if busy {
        t.accent
    } else if idle < 60.0 {
        t.idle[0]
    } else if idle < 3600.0 {
        t.idle[1]
    } else {
        t.idle[2]
    }
}

/// Token positions, short: 950, 12K, 1.2M.
fn short(n: u64) -> String {
    if n < 1000 {
        n.to_string()
    } else if n < 1_000_000 {
        format!("{}K", n / 1000)
    } else {
        format!("{:.1}M", n as f64 / 1e6)
    }
}

/// The session's request while it decodes: "#12".
fn request_of(s: &Snapshot, session: u64) -> Option<u64> {
    s.requests
        .iter()
        .find(|r| r.session == session)
        .map(|r| r.number)
}

/// With several models, every model's sessions in rows at the top, the model
/// shown on the accent; a click on another shows it. Returns the rows used.
fn models(buf: &mut Buffer, area: Rect, s: &Snapshot, ui: &mut Ui) -> u16 {
    if s.models < 2 {
        return 0;
    }
    let mut rows: Vec<Vec<(Line<'static>, Target, bool)>> = vec![Vec::new()];
    let mut x = area.x + 1;
    for (i, m) in s.model_memory.iter().enumerate() {
        let sessions = match m.sessions {
            0 => "no sessions".to_string(),
            1 => "1 session".to_string(),
            n => format!("{n} sessions"),
        };
        let line = Line::from(vec![
            strong(format!(" {}", m.model)),
            label(format!(
                "  {sessions} · {:.2} GiB ",
                m.kv_committed as f64 / GIB
            )),
        ]);
        let w = line.width() as u16 + 2;
        if x + w > area.right() && rows.last().is_some_and(|r| !r.is_empty()) {
            rows.push(Vec::new());
            x = area.x + 1;
        }
        x += w;
        if let Some(row) = rows.last_mut() {
            row.push((line, Target::Model(i), i == ui.model));
        }
    }
    let used = (rows.len() as u16).min(area.height / 2);
    for (r, row) in rows.into_iter().take(used as usize).enumerate() {
        tab_row(
            buf,
            area.x + 1,
            area.y + r as u16,
            area.right(),
            row,
            &mut ui.hits,
        );
    }
    used
}

pub fn draw(buf: &mut Buffer, area: Rect, s: &Snapshot, ui: &mut Ui) {
    let used = models(buf, area, s, ui);
    let area = Rect {
        y: area.y + used,
        height: area.height - used,
        ..area
    };
    let t = get();
    let k = &s.kinds;
    let tot = &s.totals;
    let head = Line::from(vec![
        text(" "),
        strong(format!(
            "{} session{}",
            s.sessions,
            if s.sessions == 1 { "" } else { "s" }
        )),
        text(format!(
            "   this model {:.2} GiB · shared {:.2} / {:.1} GiB",
            s.kv_gib,
            s.kv_total_gib.max(s.kv_gib),
            s.kv_limit_gib
        )),
        label("   reused "),
        text(pct(tot.reused_tokens as f64, tot.prompt_tokens as f64)),
        label(" of the prompt tokens   new "),
        text(k.new.to_string()),
        label(" · continued "),
        text(k.continued.to_string()),
        label(" · saved "),
        text(k.saved.to_string()),
        label(" · forked "),
        text(k.forked.to_string()),
        label(" · loaded "),
        text(k.loaded.to_string()),
    ]);
    buf.set_line(
        area.x,
        area.y,
        &super::fit(head, area.width as usize),
        area.width,
    );
    if let Some(d) = &s.kv_disk {
        let line = Line::from(vec![
            label(" on the SSD "),
            strong(format!("{} saved state{}", d.states, if d.states == 1 { "" } else { "s" })),
            text(format!(" {:.2} / {:.1} GiB", d.gib, d.limit_gib)),
            label("   this run: read "),
            text(d.loads.to_string()),
            label(" · saved "),
            text(format!("{} ({:.2} GiB written)", d.saves, d.written_gib)),
            label("   in "),
            text(d.dir.clone()),
        ]);
        buf.set_line(area.x, area.y + 1, &super::fit(line, area.width as usize), area.width);
    }
    if s.tree.is_empty() || area.height < 8 || area.width < 40 {
        buf.set_line(
            area.x,
            area.y + 2,
            &Line::from(label(" no sessions yet")),
            area.width,
        );
        return;
    }
    let selected = ui.node.and_then(|key| find(&s.tree, key));
    let panel = if ui.text.is_some() {
        (area.width as u32 * 2 / 5)
            .max(52)
            .min(area.width as u32 / 2) as u16
    } else {
        0
    };
    let tw = area.width - panel;
    let (pl, rows) = places(&s.tree);
    let (busy, idle) = liveness(&s.tree, &pl);
    let x0 = area.x + 2;
    let x1 = (area.x + tw).saturating_sub(43).max(x0 + 12);
    let span = s.tree.iter().map(|n| n.end).max().unwrap_or(1).max(1) as f64;
    let col = |pos: u64| x0 + ((pos as f64 / span).min(1.0) * (x1 - x0) as f64).round() as u16;

    // the positions' axis
    let ay = area.y + 2;
    let step = [
        1_000u64, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000, 200_000, 500_000, 1_000_000,
    ]
    .into_iter()
    .find(|&st| (st as f64 / span) * ((x1 - x0) as f64) >= 10.0)
    .unwrap_or(1_000_000);
    buf.set_string(
        x0,
        ay + 1,
        "─".repeat((x1 - x0 + 1) as usize),
        color(t.rule),
    );
    let mut tick = 0;
    while tick as f64 <= span {
        let x = col(tick);
        buf[(x, ay + 1)].set_symbol("┬");
        let name = short(tick);
        let nx = x.saturating_sub(name.width() as u16 / 2).max(area.x);
        buf.set_string(nx, ay, &name, color(t.label));
        tick += step;
    }
    buf.set_string(x1 + 2, ay, "tokens", color(t.label));

    // the rows: two lines each, the last line of the area for the selection
    let y0 = ay + 3;
    let fit = ((area.bottom().saturating_sub(y0 + 2)) / 2) as usize;
    let row_y = |r: usize| y0 + 2 * r as u16;
    let shown = |r: usize| r < fit;
    let heavy = |i: usize| busy[i];
    let style = |i: usize| Style::new().fg(c(tone(busy[i], idle[i])));
    let from_col = |i: usize| {
        if pl[i].parent.is_none() {
            col(s.tree[i].start)
        } else {
            col(s.tree[i].start) + 1
        }
    };

    // the lines: a root from its start, a child from the column after its fork
    for (i, n) in s.tree.iter().enumerate() {
        let r = pl[i].row;
        if !shown(r) {
            continue;
        }
        let (from, to) = (from_col(i), col(n.end).max(from_col(i)));
        let ch = if heavy(i) { "━" } else { "─" };
        for x in from..=to {
            buf[(x, row_y(r))].set_symbol(ch).set_style(style(i));
        }
        ui.hits.push(Hit {
            y: row_y(r),
            x0: from,
            x1: to + 1,
            target: Target::Node(i),
        });
    }
    // the forks: a junction on the parent's row, a corner on each later
    // child's row, and the vertical between
    for (q, n) in s.tree.iter().enumerate() {
        let kids = &pl[q].children;
        if kids.len() < 2 || !shown(pl[q].row) {
            continue;
        }
        let x = col(n.end);
        let down_heavy = kids[1..].iter().any(|&k| heavy(k));
        let junction = match (heavy(q), down_heavy) {
            (true, true) => "┳",
            (true, false) => "┯",
            (false, true) => "┰",
            (false, false) => "┬",
        };
        buf[(x, row_y(pl[q].row))]
            .set_symbol(junction)
            .set_style(style(q));
        let last = *kids.last().unwrap();
        for (j, &k) in kids.iter().enumerate().skip(1) {
            let rest_heavy = kids[j..].iter().any(|&m| heavy(m));
            let from_y = if j == 1 {
                row_y(pl[q].row) + 1
            } else {
                row_y(pl[kids[j - 1]].row) + 1
            };
            let to_y = row_y(pl[k].row);
            let vstyle = Style::new().fg(c(tone(
                rest_heavy,
                kids[j..]
                    .iter()
                    .map(|&m| idle[m])
                    .fold(f64::INFINITY, f64::min),
            )));
            for y in from_y..to_y.min(row_y(fit)) {
                let cell = &mut buf[(x, y)];
                let crossing = matches!(cell.symbol(), "─" | "━");
                cell.set_symbol(match (crossing, rest_heavy) {
                    (true, true) => "╂",
                    (true, false) => "┼",
                    (false, true) => "┃",
                    (false, false) => "│",
                })
                .set_style(vstyle);
            }
            if shown(pl[k].row) {
                let corner = match (k == last, heavy(k)) {
                    (true, true) => "┗",
                    (true, false) => "└",
                    (false, true) => "┣",
                    (false, false) => "├",
                };
                buf[(x, to_y)].set_symbol(corner).set_style(style(k));
            }
        }
    }
    // the sessions' ends and saved positions; a label after each row's end
    for (i, n) in s.tree.iter().enumerate() {
        if n.session == 0 || !shown(pl[i].row) {
            continue;
        }
        let y = row_y(pl[i].row);
        let end = col(n.end);
        buf[(end, y)].set_symbol("●").set_style(style(i));
        if n.saved >= 0 {
            let saved = n.saved as u64;
            let mut at = i; // the node on the session's path that holds the saved position
            while s.tree[at].start > saved {
                match pl[at].parent {
                    Some(q) => at = q,
                    None => break,
                }
            }
            if shown(pl[at].row) {
                buf[(col(saved), row_y(pl[at].row))]
                    .set_symbol("◆")
                    .set_style(style(at).add_modifier(Modifier::BOLD));
            }
        }
        if pl[i].children.is_empty() {
            let mut spans = vec![Span::styled(
                format!("s{}", n.session),
                Style::new().add_modifier(Modifier::BOLD).fg(c(if busy[i] {
                    t.accent
                } else {
                    t.ink
                })),
            )];
            if n.busy {
                let req = request_of(s, n.session)
                    .map(|r| format!("  #{r} decoding"))
                    .unwrap_or("  decoding".into());
                spans.push(Span::styled(req, color(t.accent)));
            } else {
                spans.push(text(format!("  idle {}", duration(n.idle_seconds))));
            }
            spans.push(label(format!(
                "  {} tok  {:.2} GiB",
                count(n.end as f64),
                n.gib
            )));
            let lx = end + 2;
            buf.set_line(
                lx,
                y,
                &Line::from(spans),
                (area.x + tw).saturating_sub(lx + 1),
            );
        }
    }
    if rows > fit {
        let more = format!("… {} more rows", rows - fit);
        buf.set_string(x0, row_y(fit).min(area.bottom() - 2), more, color(t.label));
    }

    // the selected node: its line on a tint, and what it holds
    if let Some(i) = selected.filter(|&i| shown(pl[i].row)) {
        let n = &s.tree[i];
        let y = row_y(pl[i].row);
        for x in from_col(i)..=col(n.end).max(from_col(i)) {
            buf[(x, y)]
                .set_bg(c(mix(t.ground, t.accent, 0.16)))
                .set_fg(c(t.accent))
                .modifier
                .insert(Modifier::BOLD);
        }
        let mut spans = vec![Span::styled(" ▸ ", color(t.accent))];
        let whose_name = if n.session > 0 {
            format!("s{}'s ", n.session)
        } else {
            String::new()
        };
        spans.push(strong(format!(
            "{whose_name}tokens {}–{}",
            count(n.start as f64),
            count(n.end as f64)
        )));
        spans.push(label(format!(
            "  {} tokens",
            count(n.end.saturating_sub(n.start) as f64)
        )));
        if n.session == 0 {
            spans.push(label(format!("  · shared by {} sessions", n.sessions)));
        }
        if n.session > 0 && n.saved >= 0 {
            spans.push(label(format!("  · saved at {}", count(n.saved as f64))));
        }
        if n.hits > 0 {
            spans.push(label(format!(
                "  · reused by {} request{}",
                n.hits,
                if n.hits == 1 { "" } else { "s" }
            )));
        }
        if ui.text.is_none() {
            spans.push(label(format!("  · ⏎ shows s{}'s text there", n.via)));
        }
        buf.set_line(
            area.x,
            area.bottom() - 1,
            &super::fit(Line::from(spans), tw as usize),
            tw,
        );
    }

    // the text panel: of the positions asked for (a decoding session's node
    // grows past them)
    if let (Some(asked), Some(_)) = (&mut ui.text, selected) {
        let (session, start, end) = (asked.session, asked.start, asked.end);
        let px = area.x + tw;
        for y in ay..area.bottom() {
            buf[(px, y)].set_symbol("│").set_style(color(t.rule));
        }
        let title = Line::from(vec![
            strong(format!(
                " s{session} · tokens {}–{}",
                count(start as f64),
                count(end as f64)
            )),
            label(format!(
                "  {} tokens",
                count(end.saturating_sub(start) as f64)
            )),
        ]);
        buf.set_line(
            px + 1,
            ay,
            &super::fit(title, (panel - 1) as usize),
            panel - 1,
        );
        let room = (area.bottom() - ay - 3) as usize;
        ui.page = room.max(1);
        match &mut asked.answer {
            None => {
                buf.set_string(px + 2, ay + 2, "reading the text …", color(t.label));
            }
            Some(a) => {
                let w = (panel - 4) as usize;
                if a.width != w {
                    (a.lines, a.width) = (wrap(&a.runs, w), w);
                }
                let lines = &a.lines;
                ui.scroll = ui.scroll.min(lines.len().saturating_sub(room));
                for (j, line) in lines.iter().skip(ui.scroll).take(room).enumerate() {
                    buf.set_line(px + 2, ay + 2 + j as u16, line, panel - 3);
                }
                let last = (ui.scroll + room).min(lines.len());
                let mut place = format!(
                    " lines {}–{} of {}",
                    (ui.scroll + 1).min(last),
                    last,
                    lines.len()
                );
                if a.tokens < end.saturating_sub(start) as usize {
                    place += &format!(" · the first {} tokens", count(a.tokens as f64));
                }
                buf.set_string(px + 1, area.bottom() - 1, place, color(t.label));
            }
        }
    }
}

/// The text's runs in lines of at most w columns, broken at its newlines and
/// at spaces where it can; the special tokens (the runs marked true) whole
/// where they fit, in the accent.
pub fn wrap(runs: &[(String, bool)], w: usize) -> Vec<Line<'static>> {
    let special = Style::new()
        .fg(c(get().accent))
        .add_modifier(Modifier::BOLD);
    let w = w.max(1);
    let (mut out, mut line, mut col) = (Vec::new(), Vec::<Span<'static>>::new(), 0);
    for (run, is_special) in runs {
        let style = if *is_special { special } else { Style::new() };
        let words: Vec<&str> = if *is_special {
            vec![run.as_str()]
        } else {
            run.split_inclusive([' ', '\n']).collect()
        };
        for word in words {
            let (word, newline) = word.strip_suffix('\n').map_or((word, false), |x| (x, true));
            if col + word.width() > w && col > 0 {
                out.push(Line::from(std::mem::take(&mut line)));
                col = 0;
            }
            let mut chunk = String::new();
            for ch in word.chars() {
                let cw = ch.width().unwrap_or(0);
                if col + cw > w {
                    line.push(Span::styled(std::mem::take(&mut chunk), style));
                    out.push(Line::from(std::mem::take(&mut line)));
                    col = 0;
                }
                chunk.push(ch);
                col += cw;
            }
            if !chunk.is_empty() {
                line.push(Span::styled(chunk, style));
            }
            if newline {
                out.push(Line::from(std::mem::take(&mut line)));
                col = 0;
            }
        }
    }
    out.push(Line::from(line));
    out
}
