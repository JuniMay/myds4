//! The Memory tab: the Mac's installed memory as a treemap, the way disk
//! cleaners draw a disk. Each rectangle's area follows its bytes. Every model
//! is a rectangle under its name, split into its weights, experts, workspace
//! and KV; beside the models are the server's other memory, other programs
//! (the system's memory among them), the file cache, and free memory, with
//! the reserve's share of it marked.
use super::frame::{label, strong, text};
use super::theme::{Rgb, c, get, mix};
use super::{GIB, ModelMemory, Snapshot, clip, color, fit};
use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use unicode_width::UnicodeWidthStr;

/// A rectangle of the map; a model's holds its parts below its name.
struct Block {
    name: String,
    bytes: u64,
    tone: Rgb,
    parts: Vec<Block>,
    /// The model the view shows.
    chosen: bool,
}

fn leaf(name: &str, bytes: u64, tone: Rgb) -> Block {
    Block {
        name: name.into(),
        bytes,
        tone,
        parts: Vec::new(),
        chosen: false,
    }
}

fn gib(bytes: u64) -> String {
    let g = bytes as f64 / GIB;
    if g < 1.0 {
        format!("{g:.2} GiB")
    } else {
        format!("{g:.1} GiB")
    }
}

/// Rectangles in `area`, one for each size in order, that tile it, each with
/// cells in proportion to its size: the squarified layout (Bruls, Huizing and
/// van Wijk), which fills a row along the shorter side of the room left while
/// another rectangle makes the row's worst aspect ratio no worse. A cell
/// counts as twice as tall as wide. A zero size, or one too small for a cell,
/// gets an empty rectangle.
pub fn squarify(sizes: &[u64], area: Rect) -> Vec<Rect> {
    let mut out = vec![Rect::new(area.x, area.y, 0, 0); sizes.len()];
    let total: f64 = sizes.iter().map(|&s| s as f64).sum();
    if total <= 0.0 || area.width == 0 || area.height == 0 {
        return out;
    }
    let mut order: Vec<usize> = (0..sizes.len()).filter(|&i| sizes[i] > 0).collect();
    order.sort_by(|&a, &b| sizes[b].cmp(&sizes[a]));
    // the room left, in units where a cell is 1 wide and 2 tall
    let (mut x, mut y) = (0.0f64, 0.0f64);
    let (mut w, mut h) = (area.width as f64, 2.0 * area.height as f64);
    let scale = w * h / total;
    // a row's worst ratio of length to thickness, along a side of the room
    let worst = |row: &[usize], sum: f64, side: f64| {
        let thick = sum / side;
        row.iter()
            .map(|&i| {
                let long = sizes[i] as f64 * scale / thick;
                (thick / long).max(long / thick)
            })
            .fold(0.0, f64::max)
    };
    let mut i = 0;
    while i < order.len() {
        let side = w.min(h);
        let (mut j, mut sum, mut best) = (i, 0.0, f64::INFINITY);
        while j < order.len() {
            let more = sum + sizes[order[j]] as f64 * scale;
            let ratio = worst(&order[i..=j], more, side);
            if ratio > best {
                break;
            }
            (best, sum, j) = (ratio, more, j + 1);
        }
        let thick = sum / side;
        let mut along = 0.0;
        for &k in &order[i..j] {
            let long = sizes[k] as f64 * scale / thick;
            out[k] = if w >= h {
                cells(area, x, y + along, thick, long)
            } else {
                cells(area, x + along, y, long, thick)
            };
            along += long;
        }
        if w >= h {
            x += thick;
            w -= thick;
        } else {
            y += thick;
            h -= thick;
        }
        i = j;
    }
    out
}

/// The cells of a rectangle given in the map's units (a cell 1 wide and 2
/// tall). Its edges round to the nearest cell edge, so rectangles that share
/// an edge share it in cells.
fn cells(area: Rect, x: f64, y: f64, w: f64, h: f64) -> Rect {
    let edge = |v: f64, max: u16| (v.round().max(0.0) as u16).min(max);
    let (x0, x1) = (edge(x, area.width), edge(x + w, area.width));
    let (y0, y1) = (edge(y / 2.0, area.height), edge((y + h) / 2.0, area.height));
    Rect::new(area.x + x0, area.y + y0, x1 - x0, y1 - y0)
}

fn fill(buf: &mut Buffer, r: Rect, bg: Rgb) {
    let style = Style::new().fg(c(get().ink)).bg(c(bg));
    for y in r.top()..r.bottom() {
        for x in r.left()..r.right() {
            buf[(x, y)].set_symbol(" ").set_style(style);
        }
    }
}

/// A rectangle's name and size inside it: on one line when they fit, else on
/// two, else the size alone; false when not even the size fits.
fn name_in(buf: &mut Buffer, r: Rect, name: &str, bytes: u64, bg: Rgb) -> bool {
    let style = Style::new().fg(c(get().ink)).bg(c(bg));
    let (w, size) = (r.width as usize, gib(bytes));
    let one = format!(" {name} {size}");
    if one.width() <= w {
        buf.set_string(r.x, r.y, one, style);
    } else if r.height >= 2 && size.width() < w {
        buf.set_string(r.x, r.y, format!(" {}", clip(name, w - 1)), style);
        buf.set_string(r.x, r.y + 1, format!(" {size}"), style);
    } else if size.width() < w {
        buf.set_string(r.x, r.y, format!(" {size}"), style);
    } else {
        return false;
    }
    true
}

/// A block in its rectangle. One with no room for its size goes to `small`,
/// with the model it belongs to ("" for none).
fn draw_block(
    buf: &mut Buffer,
    r: Rect,
    b: &Block,
    owner: &str,
    small: &mut Vec<(String, String)>,
) {
    let t = get();
    let named = || (owner.to_string(), format!("{} {}", b.name, gib(b.bytes)));
    if r.width == 0 || r.height == 0 {
        small.push(named());
        return;
    }
    if b.parts.is_empty() {
        let bg = mix(t.ground, b.tone, 0.5);
        fill(buf, r, bg);
        if !name_in(buf, r, &b.name, b.bytes, bg) {
            small.push(named());
        }
        return;
    }
    // A model: its name and size on its first row, its parts below. A column
    // of the background at its right keeps its parts apart from the next
    // model's. A long name gives way to the size; the parts too small for
    // their own size follow on the first row when they fit there.
    let r = Rect {
        width: if r.width > 2 { r.width - 1 } else { r.width },
        ..r
    };
    let below = Rect {
        y: r.y + 1,
        height: r.height - 1,
        ..r
    };
    let sizes: Vec<u64> = b.parts.iter().map(|p| p.bytes).collect();
    let mut unlabeled = Vec::new();
    for (p, pr) in b.parts.iter().zip(squarify(&sizes, below)) {
        draw_block(buf, pr, p, &b.name, &mut unlabeled);
    }
    let (fg, bg) = if b.chosen {
        (t.on_accent, t.accent)
    } else {
        (t.ink, mix(t.ground, t.ink, 0.18))
    };
    fill(buf, Rect { height: 1, ..r }, bg);
    let size = gib(b.bytes);
    let room = (r.width as usize).saturating_sub(size.width() + 2);
    let mut title = if room >= 4 {
        format!(" {} {size}", clip(&b.name, room))
    } else {
        clip(&format!(" {size}"), r.width as usize)
    };
    for (i, (owner, what)) in unlabeled.into_iter().enumerate() {
        let gap = if i == 0 { "   " } else { " · " };
        if title.width() + gap.width() + what.width() <= r.width as usize {
            title = format!("{title}{gap}{what}");
        } else {
            small.push((owner, what));
        }
    }
    buf.set_string(
        r.x,
        r.y,
        title,
        Style::new()
            .fg(c(fg))
            .bg(c(bg))
            .add_modifier(Modifier::BOLD),
    );
}

pub fn draw(buf: &mut Buffer, area: Rect, s: &Snapshot) {
    let t = get();
    let m = &s.memory;
    let width = area.width as usize;
    let budget = &s.model_memory_budget;
    // available as the engine counts it: free pages and the file cache beyond
    // the share the kernel keeps
    let beyond = if m.file_kept > 0 {
        m.file.saturating_sub(m.file_kept)
    } else {
        0
    };
    let held = |mm: &ModelMemory| mm.weights + mm.experts + mm.workspace + mm.kv;
    let present = s.model_memory.iter().filter(|mm| held(mm) > 0).count();
    let head = [
        Line::from(vec![
            strong(" Models "),
            text(format!(
                "{:.2} / {:.2} GiB now",
                budget.allocated as f64 / GIB,
                budget.limit as f64 / GIB
            )),
            label(if s.memory_gib > 0.0 {
                format!(" (setting {:.0} GiB)", s.memory_gib)
            } else {
                String::new()
            }),
            label(if s.model_memory.len() > 1 {
                format!(", {present} of {} in memory", s.model_memory.len())
            } else {
                String::new()
            }),
            label("   KV "),
            text(format!(
                "{:.2} / {:.2} GiB, shared",
                s.kv_total_gib.max(s.kv_gib),
                s.kv_limit_gib
            )),
            label("   reserve "),
            text(format!("{:.0} GiB", s.reserve_gib)),
            label(" kept available for other programs"),
        ]),
        Line::from(vec![
            strong(" Mac "),
            text(format!("{:.0} GiB installed", m.total as f64 / GIB)),
            label("   available "),
            text(gib(m.free.saturating_add(beyond).min(m.total))),
            label("   this server "),
            text(gib(m.footprint)),
        ]),
    ];
    for (i, line) in head.into_iter().enumerate().take(area.height as usize) {
        buf.set_line(area.x, area.y + i as u16, &fit(line, width), area.width);
    }
    if area.height < 8 || area.width < 20 {
        return;
    }

    // every part its own hue, so that parts side by side stay apart
    let kinds = [
        ("weights", t.sources[2]),
        ("experts", t.sources[0]),
        ("workspace", t.sources[3]),
        ("KV", t.prefill),
    ];
    let mut blocks = Vec::new();
    let mut tracked = 0u64;
    for mm in &s.model_memory {
        let parts: Vec<Block> = kinds
            .iter()
            .zip([mm.weights, mm.experts, mm.workspace, mm.kv])
            .filter(|&(_, bytes)| bytes > 0)
            .map(|(&(name, tone), bytes)| leaf(name, bytes, tone))
            .collect();
        let bytes = held(mm);
        if bytes == 0 {
            continue;
        }
        tracked += bytes;
        blocks.push(Block {
            name: mm.model.clone(),
            bytes,
            tone: t.ink,
            parts,
            chosen: mm.model == s.model,
        });
    }
    // In use without the file cache: this server, other programs and the
    // system. Of the free memory, the reserve's share is marked.
    let in_use = m.total.saturating_sub(m.free.saturating_add(m.file));
    let others = in_use.saturating_sub(m.footprint.max(tracked));
    let reserve = m.free.min((s.reserve_gib * GIB) as u64);
    let rest = [
        (
            "other server memory",
            m.footprint.saturating_sub(tracked),
            t.requests[4],
        ),
        ("other programs", others, t.label),
        ("file cache", m.file, t.rule),
        ("reserve", reserve, t.requests[2]),
        ("free", m.free - reserve, t.faint),
    ];
    for (name, bytes, tone) in rest {
        blocks.push(leaf(name, bytes, tone));
    }
    let map = Rect::new(area.x, area.y + 3, area.width, area.height - 5);
    let sizes: Vec<u64> = blocks.iter().map(|b| b.bytes).collect();
    let mut small = Vec::new();
    for (b, r) in blocks.iter().zip(squarify(&sizes, map)) {
        if b.bytes > 0 {
            draw_block(buf, r, b, "", &mut small);
        }
    }

    // What each color is; free memory, nearly the background, is a hollow
    // square. Then what has no room for its size.
    let mut key = vec![text(" ")];
    for (name, tone) in kinds
        .into_iter()
        .chain(rest.iter().map(|&(name, _, tone)| (name, tone)))
    {
        key.push(if name == "free" {
            Span::styled("□ ", color(t.label))
        } else {
            Span::styled("■ ", color(mix(t.ground, tone, 0.5)))
        });
        key.push(label(format!("{name}   ")));
    }
    buf.set_line(
        area.x,
        area.bottom() - 2,
        &fit(Line::from(key), width),
        area.width,
    );
    // what has no room for its size, after the model it belongs to
    let mut groups: Vec<(String, Vec<String>)> = Vec::new();
    for (owner, what) in small {
        match groups.last_mut() {
            Some((o, list)) if *o == owner => list.push(what),
            _ => groups.push((owner, vec![what])),
        }
    }
    if !groups.is_empty() {
        let parts: Vec<String> = groups
            .into_iter()
            .map(|(owner, list)| match owner.as_str() {
                "" => list.join(", "),
                _ => format!("{owner}: {}", list.join(", ")),
            })
            .collect();
        let note = format!(" too small to label: {}", parts.join("; "));
        buf.set_line(
            area.x,
            area.bottom() - 1,
            &fit(Line::from(label(note)), width),
            area.width,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// How many of the rectangles cover each cell of `area`.
    fn covers(rects: &[Rect], area: Rect) -> Vec<u32> {
        let mut n = vec![0u32; area.width as usize * area.height as usize];
        for r in rects {
            for y in r.top()..r.bottom() {
                for x in r.left()..r.right() {
                    assert!(area.contains((x, y).into()), "{r:?} in {area:?}");
                    n[(y - area.y) as usize * area.width as usize + (x - area.x) as usize] += 1;
                }
            }
        }
        n
    }

    #[test]
    fn rectangles_tile_the_area_once() {
        let mut seed = 0x5eed_u64;
        let mut next = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            seed >> 33
        };
        for _ in 0..500 {
            let area = Rect::new(3, 2, 1 + (next() % 160) as u16, 1 + (next() % 50) as u16);
            let sizes: Vec<u64> = (0..1 + next() % 14)
                .map(|_| if next() % 5 == 0 { 0 } else { next() % 100_000 })
                .collect();
            let rects = squarify(&sizes, area);
            assert_eq!(rects.len(), sizes.len());
            let n = covers(&rects, area);
            if sizes.iter().all(|&s| s == 0) {
                assert!(n.iter().all(|&k| k == 0));
            } else {
                assert!(
                    n.iter().all(|&k| k == 1),
                    "{sizes:?} in {area:?}: {rects:?}"
                );
            }
            for (s, r) in sizes.iter().zip(&rects) {
                assert!(*s > 0 || r.width * r.height == 0, "{sizes:?}");
            }
        }
    }

    #[test]
    fn areas_follow_the_sizes() {
        let area = Rect::new(0, 0, 120, 40);
        let sizes = [600, 250, 100, 50];
        let total: u64 = sizes.iter().sum();
        for (s, r) in sizes.iter().zip(squarify(&sizes, area)) {
            let want = *s as f64 / total as f64 * 4800.0;
            let got = (r.width * r.height) as f64;
            // each edge moves at most half a cell
            let slack = (r.width + r.height) as f64;
            assert!(
                (got - want).abs() <= slack,
                "{s}: {got} cells for {want:.0}, {r:?}"
            );
        }
    }

    #[test]
    fn equal_sizes_in_a_square_look_square() {
        // 40 columns by 20 rows look square: four equal sizes, a quarter each
        let rects = squarify(&[5, 5, 5, 5], Rect::new(0, 0, 40, 20));
        for r in rects {
            assert_eq!((r.width, r.height), (20, 10), "{r:?}");
        }
    }
}
