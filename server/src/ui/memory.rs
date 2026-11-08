//! Memory for every model together. Tile area follows tracked allocations;
//! the process footprint and the one shared KV budget are separate figures.
use super::frame::{label, strong, text};
use super::theme::{c, get};
use super::{GIB, ModelMemory, Snapshot, color, fit};
use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::text::Line;
use ratatui::widgets::{Block, Borders, Paragraph, Widget};

fn tile(buf: &mut Buffer, area: Rect, m: &ModelMemory, selected: bool) {
    if area.width < 2 || area.height < 2 {
        return;
    }
    let theme = get();
    let tone = if selected {
        theme.accent
    } else {
        theme.prefill
    };
    let mix = |a: u8, b: u8| ((a as u16 * 9 + b as u16) / 10) as u8;
    let background = (
        mix(theme.ground.0, tone.0),
        mix(theme.ground.1, tone.1),
        mix(theme.ground.2, tone.2),
    );
    let style = color(theme.ink).bg(c(background));
    let block = Block::default()
        .borders(Borders::ALL)
        .style(style)
        .border_style(color(tone).bg(c(background)))
        .title(fit(
            Line::from(strong(format!(" {} ", m.model))),
            area.width.saturating_sub(2) as usize,
        ));
    let inner = block.inner(area);
    block.render(area, buf);
    let lines = [
        Line::from(strong(format!("{:.2} GiB tracked", m.tracked as f64 / GIB))),
        Line::from(text(format!(
            "Weights       {:>8.2} GiB",
            m.weights as f64 / GIB
        ))),
        Line::from(text(format!(
            "Experts       {:>8.2} GiB",
            m.experts as f64 / GIB
        ))),
        Line::from(text(format!(
            "KV allocated  {:>8.2} GiB",
            m.kv as f64 / GIB
        ))),
        Line::from(text(format!(
            "KV committed  {:>8.2} GiB",
            m.kv_committed as f64 / GIB
        ))),
    ]
    .into_iter()
    .map(|l| fit(l, inner.width as usize))
    .collect::<Vec<_>>();
    Paragraph::new(lines).style(style).render(inner, buf);
}

pub fn draw(buf: &mut Buffer, area: Rect, s: &Snapshot) {
    let width = area.width as usize;
    let used = s.kv_total_gib.max(s.kv_gib);
    let lines = [
        Line::from(strong("All models · memory")),
        Line::from(text(format!(
            "Shared KV budget: {:.2} / {:.2} GiB",
            used, s.kv_limit_gib
        ))),
        Line::from(text(format!(
            "System reserve target: {:.1} GiB available",
            s.reserve_gib
        ))),
        Line::from(text(format!(
            "Process footprint: {:.2} GiB · available: {:.2} GiB",
            s.memory.footprint as f64 / GIB,
            (s.memory.free + s.memory.file) as f64 / GIB
        ))),
        Line::from(label(
            "Tiles: weights, experts and allocated KV. Runtime buffers are outside these totals.",
        )),
    ];
    for (i, line) in lines.into_iter().enumerate().take(area.height as usize) {
        buf.set_line(area.x, area.y + i as u16, &fit(line, width), area.width);
    }
    if area.height <= 6 || s.model_memory.is_empty() {
        return;
    }
    let remaining = Rect::new(area.x, area.y + 6, area.width, area.height - 6);
    let n = s.model_memory.len();
    if width >= n * 30 {
        let total: u64 = s.model_memory.iter().map(|m| m.tracked).sum();
        let mut left = remaining.width;
        let mut x = remaining.x;
        for (i, m) in s.model_memory.iter().enumerate() {
            let w = if i + 1 == n {
                left
            } else {
                let share = if total > 0 {
                    (remaining.width as u128 * m.tracked as u128 / total as u128) as u16
                } else {
                    remaining.width / n as u16
                };
                share
                    .max(28)
                    .min(left.saturating_sub(28 * (n - i - 1) as u16))
            };
            tile(
                buf,
                Rect::new(x, remaining.y, w, remaining.height.min(10)),
                m,
                m.model == s.model,
            );
            left -= w;
            x += w;
        }
    } else {
        let h = (remaining.height / n as u16).max(1);
        for (i, m) in s.model_memory.iter().enumerate() {
            let y = remaining.y + i as u16 * h;
            if y >= area.bottom() {
                break;
            }
            tile(
                buf,
                Rect::new(remaining.x, y, remaining.width, h.min(area.bottom() - y)),
                m,
                m.model == s.model,
            );
        }
    }
}
