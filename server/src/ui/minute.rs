//! The Last minute tab: one column per sample (0.5 s) of the last minute: the
//! phase and the requests' arrivals, the requests' tokens/s stacked, the SSD's
//! read rate, the hit rate, and each request's prefill and decoding.

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};

use super::frame::{label, strong, text};
use super::history::{History, SAMPLES, Sample};
use super::theme::{Rgb, c, get, mix};
use super::{LOWER, Phase, color, stack};

/// A column of stacked parts in `rows` cells from y_bottom up: each part a
/// (height in cells, color), the lowest first.
fn column(buf: &mut Buffer, x: u16, y_bottom: u16, rows: u16, parts: &[(f64, Rgb)]) {
    stack(buf, (x, y_bottom), (0, -1), rows, &LOWER, parts);
}

/// A chart's frame: its name above it, its top and base values at the left,
/// and a base line.
fn frame(
    buf: &mut Buffer,
    x: u16,
    y: u16,
    rows: u16,
    w: u16,
    name: &str,
    top: String,
    base: String,
) {
    let t = get();
    buf.set_string(x, y - 1, name, Style::new().add_modifier(Modifier::BOLD));
    buf.set_string(
        x + 13 - top.chars().count().min(13) as u16,
        y,
        &top,
        color(t.label),
    );
    buf.set_string(
        x + 13 - base.chars().count().min(13) as u16,
        y + rows - 1,
        &base,
        color(t.label),
    );
    for yy in y..y + rows {
        buf[(x + 14, yy)].set_symbol("┊").set_fg(c(t.rule));
    }
    buf.set_string(x + 15, y + rows, "─".repeat(w as usize), color(t.rule));
}

/// A rounded top for a chart, at least v.
fn nice(v: f64) -> f64 {
    let p = 10f64.powf(v.max(1e-9).log10().floor());
    [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
        .into_iter()
        .map(|m| m * p)
        .find(|&t| t >= v)
        .unwrap_or(10.0 * p)
}

/// The rows the tab needs: the phase, three charts, a request's row, the axis.
const ROWS: u16 = 5 + 9 + 2 + 6 + 2 + 6 + 2 + 1 + 1;

pub fn draw(buf: &mut Buffer, area: Rect, h: &History) {
    let t = get();
    if area.height < ROWS || area.width < 60 {
        let note = format!(" the last minute needs {ROWS} rows and 60 columns");
        buf.set_line(area.x, area.y, &Line::from(label(note)), area.width);
        return;
    }
    let cx = area.x + 16; // the charts' first column
    let w = (SAMPLES as u16).min(area.width.saturating_sub(16 + 24));
    let first = SAMPLES - w as usize; // the oldest sample shown, of the full minute
    let samples: Vec<&Sample> = h.samples.iter().collect();
    let at = |col: usize| -> Option<&Sample> {
        let k = (first + col) as i64 - (SAMPLES as i64 - samples.len() as i64);
        if k >= 0 {
            samples.get(k as usize).copied()
        } else {
            None
        }
    };
    let right = cx + w + 2; // the current values' column
    let latest = samples.last();
    let decoding = mix(t.ground, t.accent, 0.45);

    // the phase, with a mark where a request arrived
    let mut y = area.y;
    buf.set_string(
        area.x + 1,
        y + 1,
        "phase",
        Style::new().add_modifier(Modifier::BOLD),
    );
    for col in 0..w as usize {
        let x = cx + col as u16;
        if let Some(s) = at(col) {
            let tone = match s.phase {
                Phase::Prefill => t.prefill,
                Phase::Decode => decoding,
                Phase::Idle => t.faint,
            };
            buf[(x, y + 1)].set_symbol("█").set_fg(c(tone));
            if let Some(&r) = s.arrivals.first() {
                buf.set_string(x, y, format!("▼#{r}"), color(t.label));
            }
        }
    }
    let key = Line::from(vec![
        Span::styled("■", color(t.prefill)),
        label(" prefill    "),
        Span::styled("■", color(decoding)),
        label(" decoding    "),
        Span::styled("■", color(t.faint)),
        label(" idle    "),
        text("▼"),
        label(" a request arrived"),
    ]);
    buf.set_line(cx, y + 2, &key, w);

    // the requests' tokens/s, stacked in their colors
    y += 5;
    let rows = 9;
    let mut order: Vec<u64> = Vec::new();
    for s in &samples {
        for r in s.prefilling.iter().chain(s.speeds.iter().map(|p| &p.0)) {
            if !order.contains(r) {
                order.push(*r);
            }
        }
    }
    let hue =
        |r: u64| t.requests[order.iter().position(|&o| o == r).unwrap_or(0) % t.requests.len()];
    let top = nice(
        samples
            .iter()
            .map(|s| s.speeds.iter().map(|p| p.1).sum::<f64>())
            .fold(1.0, f64::max),
    );
    frame(
        buf,
        area.x + 1,
        y,
        rows,
        w,
        "tokens/s",
        format!("{top:.0}"),
        "0".into(),
    );
    for col in 0..w as usize {
        if let Some(s) = at(col) {
            let parts: Vec<(f64, Rgb)> = s
                .speeds
                .iter()
                .map(|&(r, v)| (v / top * rows as f64, hue(r)))
                .collect();
            column(buf, cx + col as u16, y + rows - 1, rows, &parts);
        }
    }
    if let Some(s) = latest {
        let total: f64 = s.speeds.iter().map(|p| p.1).sum();
        buf.set_line(
            right,
            y,
            &Line::from(vec![strong(format!("{total:.1}")), label(" tok/s now")]),
            24,
        );
        for (j, &(r, v)) in s.speeds.iter().enumerate().take(rows as usize - 1) {
            let line = Line::from(vec![
                Span::styled("■ ", color(hue(r))),
                text(format!("#{r} {v:.1}")),
            ]);
            buf.set_line(right, y + 1 + j as u16, &line, 24);
        }
    }

    // the SSD's read rate
    y += rows + 2;
    let rows = 6;
    let top = nice(samples.iter().map(|s| s.ssd).fold(0.5, f64::max));
    frame(
        buf,
        area.x + 1,
        y,
        rows,
        w,
        "SSD read",
        format!("{top} GB/s"),
        "0".into(),
    );
    for col in 0..w as usize {
        if let Some(s) = at(col) {
            column(
                buf,
                cx + col as u16,
                y + rows - 1,
                rows,
                &[(s.ssd / top * rows as f64, t.sources[4])],
            );
        }
    }
    if let Some(s) = latest {
        buf.set_line(
            right,
            y,
            &Line::from(vec![strong(format!("{:.1}", s.ssd)), label(" GB/s now")]),
            24,
        );
    }

    // the hit rate, from 80%
    y += rows + 2;
    let rows = 6;
    let floor = 0.8;
    frame(
        buf,
        area.x + 1,
        y,
        rows,
        w,
        "hit rate",
        "100%".into(),
        "80%".into(),
    );
    for col in 0..w as usize {
        if let Some(hit) = at(col).and_then(|s| s.hit) {
            let v = ((hit - floor) / (1.0 - floor)).clamp(0.0, 1.0) * rows as f64;
            column(
                buf,
                cx + col as u16,
                y + rows - 1,
                rows,
                &[(v.max(0.125), t.sources[0])],
            );
        }
    }
    if let Some(hit) = latest.and_then(|s| s.hit) {
        buf.set_line(
            right,
            y,
            &Line::from(vec![strong(format!("{:.1}%", hit * 100.0)), label(" now")]),
            24,
        );
    }
    buf.set_string(right, y + 1, "in memory, read ahead", color(t.label));
    buf.set_string(right, y + 2, "or late: decoded rows", color(t.label));

    // each request: its prefill, then its decoding
    y += rows + 2;
    buf.set_string(
        area.x + 1,
        y,
        "requests",
        Style::new().add_modifier(Modifier::BOLD),
    );
    let mut ty = y + 1;
    for (j, &r) in order.iter().enumerate() {
        let yy = y + j as u16;
        if yy + 1 >= area.bottom() {
            break;
        }
        ty = yy + 1;
        buf.set_string(
            area.x + 11,
            yy,
            format!("#{r}"),
            color(hue(r)).add_modifier(Modifier::BOLD),
        );
        for col in 0..w as usize {
            if let Some(s) = at(col) {
                let x = cx + col as u16;
                if s.prefilling.contains(&r) {
                    buf[(x, yy)].set_symbol("━").set_fg(c(t.prefill));
                } else if s.speeds.iter().any(|p| p.0 == r) {
                    buf[(x, yy)].set_symbol("━").set_fg(c(hue(r)));
                }
            }
        }
        if let Some(s) = latest {
            let now = if s.prefilling.contains(&r) {
                "prefill".to_string()
            } else if let Some(p) = s.speeds.iter().find(|p| p.0 == r) {
                format!("decoding {:.1} tok/s", p.1)
            } else {
                "done".to_string()
            };
            buf.set_string(right, yy, now, color(t.label));
        }
    }

    // the time axis, under the requests
    let ty = ty.min(area.bottom() - 1);
    let first = first as u16;
    for (at_sample, name) in [(0u16, "−60 s"), (30, "−45 s"), (60, "−30 s"), (90, "−15 s")]
    {
        if at_sample >= first && at_sample - first + 6 < w {
            buf.set_string(cx + at_sample - first, ty, name, color(t.label));
        }
    }
    buf.set_string(cx + w - 3, ty, "now", color(t.label));
}
