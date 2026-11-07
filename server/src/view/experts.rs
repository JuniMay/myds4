//! The Experts tab: one request's experts, layer by layer, of its last
//! sampled tokens, each in the color of where it was when last chosen, and
//! at the right each layer's experts of those tokens by where each one was.

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::text::{Line, Span};

use super::chrome::{Target, Ui, label, strong, tab_row, text};
use super::history::History;
use super::theme::{Rgb, c, get};
use super::{LEFT, Snapshot, color, stack};

const EXPERTS: usize = 384;
const TOPK: usize = 6;
/// Sampled tokens shown: 6 s of snapshots.
const KEEP: usize = 12;
/// The bar's cells.
const BAR: u16 = 12;
/// Where an expert was, from the most telling: from the SSD, late,
/// substituted, read ahead, in memory.
const TELLING: [u8; 5] = [4, 2, 3, 1, 0];
pub const SOURCE_NAMES: [&str; 5] = ["in memory", "read ahead", "late", "substituted", "from SSD"];
/// The columns the map needs at two experts a cell: the layers' numbers, the
/// cells, the bar and its note.
const WIDE: usize = 4 + EXPERTS / 2 + 2 + BAR as usize + 7;

/// Of a group of experts, the one chosen most recently, the most telling
/// source first among equals.
fn youngest(group: &[Option<(usize, u8)>]) -> Option<(usize, u8)> {
    group.iter().flatten().copied().min_by_key(|&(age, src)| (age, TELLING.iter().position(|&t| t == src).unwrap_or(5)))
}

/// The requests the map can show: those with a routed row in the samples.
pub fn requests(s: &Snapshot, h: &History) -> Vec<u64> {
    s.requests.iter().map(|r| r.number).filter(|&n| !h.routes(n, 1).is_empty()).collect()
}

pub fn draw(buf: &mut Buffer, area: Rect, s: &Snapshot, h: &History, ui: &mut Ui) {
    let t = get();
    let speeds = h.speeds();
    // the decoding requests with routed rows, as tabs
    let reqs = requests(s, h);
    let chosen = ui.request.filter(|n| reqs.contains(n)).or(reqs.first().copied());
    buf.set_string(area.x + 1, area.y, "Request", color(t.label));
    let items = reqs
        .iter()
        .map(|&n| {
            let session = s.requests.iter().find(|r| r.number == n).map_or(0, |r| r.session);
            let speed = speeds.iter().find(|p| p.0 == n).map_or(0.0, |p| p.1);
            (Line::from(label(format!(" #{n} · s{session} · {speed:.1} tok/s "))), Target::Request(n), Some(n) == chosen)
        })
        .collect();
    tab_row(buf, area.x + 10, area.y, area.right(), items, &mut ui.hits);
    let Some(req) = chosen else {
        let why = if s.requests.is_empty() {
            " no request is decoding"
        } else {
            " no decoding request has a routed row yet: the engine records one each decode step"
        };
        buf.set_line(area.x, area.y + 2, &Line::from(label(why)), area.width);
        return;
    };
    let routes = h.routes(req, KEEP);
    let kept = routes.len();
    let layers = routes.last().map_or(0, |r| r.experts.len() / TOPK);

    // two experts per cell where the terminal is wide enough, else four
    let per = if area.width as usize >= WIDE { 2 } else { 4 };
    let cells = (EXPERTS / per) as u16;
    if (area.width as usize) < 4 + cells as usize + 2 + BAR as usize {
        buf.set_line(area.x, area.y + 2, &Line::from(label(format!(" the map needs {} columns", 4 + cells + 2 + BAR))), area.width);
        return;
    }
    let gx = area.x + 4;
    let gy = area.y + 3;
    for e in [0usize, 64, 128, 192, 256, 320] {
        buf.set_string(gx + (e / per) as u16, gy - 1, e.to_string(), color(t.label));
    }
    buf.set_string(gx + cells - 3, gy - 1, "383", color(t.label));
    let bx = gx + cells + 2;
    buf.set_string(bx, gy - 1, format!("{kept} tokens"), color(t.label));

    let bottom = area.bottom().saturating_sub(3);
    for l in 0..layers {
        let y = gy + l as u16;
        if y >= bottom {
            break;
        }
        buf.set_string(area.x, y, format!("{:>3}", l + 1), color(t.label));
        // each expert's latest choice in the samples: (age, source)
        let mut seen: Vec<Option<(usize, u8)>> = vec![None; EXPERTS];
        let mut tally = [0u32; 5];
        for (k, r) in routes.iter().enumerate() {
            let age = kept - 1 - k;
            for j in 0..TOPK {
                let (e, src) = (r.experts[l * TOPK + j] as usize, r.sources[l * TOPK + j].min(4));
                if e < EXPERTS {
                    seen[e] = Some((age, src));
                }
                tally[src as usize] += 1;
            }
        }
        let half = per / 2;
        for cell in 0..cells as usize {
            let group = &seen[cell * per..(cell + 1) * per];
            let (left, right) = (youngest(&group[..half]), youngest(&group[half..]));
            let at = &mut buf[(gx + cell as u16, y)];
            match (left, right) {
                (Some(a), Some(b)) => {
                    at.set_symbol("▌").set_fg(c(t.sources[a.1 as usize])).set_bg(c(t.sources[b.1 as usize]));
                }
                (Some(a), None) => {
                    at.set_symbol("▌").set_fg(c(t.sources[a.1 as usize]));
                }
                (None, Some(b)) => {
                    at.set_symbol("▐").set_fg(c(t.sources[b.1 as usize]));
                }
                (None, None) => {}
            }
        }
        // the layer's experts of the kept tokens, by where each one was
        let total = tally.iter().sum::<u32>().max(1) as f64;
        let parts: Vec<(f64, Rgb)> = (0..5).map(|i| (tally[i] as f64 / total * BAR as f64, t.sources[i])).collect();
        stack(buf, (bx, y), (1, 0), BAR, &LEFT, &parts);
        if tally[4] > 0 {
            buf.set_string(bx + BAR + 1, y, format!("SSD {}", tally[4]), color(t.sources[4]));
        }
    }

    // the key
    let mut key = vec![text(" ")];
    for (i, name) in SOURCE_NAMES.iter().enumerate() {
        key.extend([Span::styled("■ ", color(t.sources[i])), text(*name), text("   ")]);
    }
    key.push(label(if per == 2 { "a cell: two experts, left and right" } else { "a cell: four experts, each half the latest of two" }));
    buf.set_line(area.x, area.bottom() - 2, &super::fit(Line::from(key), area.width as usize), area.width);
    let note = vec![
        text(" "),
        strong(format!("#{req}")),
        label(format!(
            " · the experts of its last {kept} sampled tokens, one every 0.5 s, each where it was when last chosen · bars: their {} experts per layer",
            kept * TOPK
        )),
    ];
    buf.set_line(area.x, area.bottom() - 1, &super::fit(Line::from(note), area.width as usize), area.width);
}
