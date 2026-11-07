//! The Overview tab: the requests now and their speed; the throughput
//! (decoding, prefill, the SSD's reads, the hit rate, the GPU and the reads'
//! wait) over the last 5 s, the last minute and since the start, with the
//! last minute as a sparkline; the experts per decoded token; memory on the
//! GPU, in RAM and on the SSD; and the log's latest lines.

use ratatui::style::Style;
use ratatui::text::{Line, Span};

use super::chrome::{label, strong, text};
use super::experts::SOURCE_NAMES;
use super::history::{History, Sample};
use super::theme::{Rgb, c, get};
use super::{GIB, LOWER, Phase, Snapshot, bar, bold, color, count, dim, duration, fit, head, pad, pct};

/// The values in at most n columns, each the mean of its samples, in eighths
/// of a cell from 0, or from the least of them for a share; a column without
/// a value is blank.
pub fn spark(v: &[Option<f64>], n: usize, from_zero: bool) -> String {
    if v.is_empty() || n == 0 {
        return String::new();
    }
    let per = v.len().div_ceil(n.min(v.len())).max(1);
    let cols: Vec<Option<f64>> = v
        .chunks(per)
        .map(|ch| {
            let xs: Vec<f64> = ch.iter().flatten().copied().collect();
            if xs.is_empty() { None } else { Some(xs.iter().sum::<f64>() / xs.len() as f64) }
        })
        .collect();
    let top = cols.iter().flatten().copied().fold(0.0, f64::max);
    let low = if from_zero { 0.0 } else { cols.iter().flatten().copied().fold(top, f64::min) };
    cols.iter()
        .map(|x| match x {
            Some(x) if top > low => LOWER[1 + ((7.99 * (x - low) / (top - low)) as usize).min(7)],
            // all the same: at the base from 0, else in the middle
            Some(_) => LOWER[if from_zero { 1 } else { 4 }],
            None => " ",
        })
        .collect()
}

fn cell(v: Option<String>, w: usize) -> Span<'static> {
    match v {
        Some(v) => text(format!("{v:<w$}")),
        None => label(format!("{:<w$}", "—")),
    }
}

/// A figure's row: its name, how to take it from a sample, its value since
/// the start, how to show it, its color, and whether its sparkline starts at 0.
struct Figure {
    name: &'static str,
    of: fn(&Sample) -> Option<f64>,
    since: Option<f64>,
    show: fn(f64) -> String,
    tone: Rgb,
    from_zero: bool,
}

pub fn throughput(s: &Snapshot, h: &History, w: usize) -> Vec<Line<'static>> {
    let t = get();
    let mut l = vec![head(w, "Throughput")];
    let cw = 16; // a figure's columns
    let mut top = vec![text(format!(" {:16}", ""))];
    for name in ["last 5 s", "last minute", "since start"] {
        top.push(label(format!("{name:<cw$}")));
    }
    top.push(label("the last minute"));
    l.push(Line::from(top));

    let d = &s.decode;
    let p = &s.prefilled;
    let work = d.seconds + p.seconds;
    let figures = [
        Figure {
            name: "decoding",
            of: |x: &Sample| (x.phase == Phase::Decode).then(|| x.speeds.iter().map(|p| p.1).sum::<f64>()),
            since: (d.seconds > 0.0).then(|| d.tokens as f64 / d.seconds),
            show: |v| format!("{v:.1} tok/s"),
            tone: t.accent,
            from_zero: true,
        },
        Figure {
            name: "prefill",
            of: |x: &Sample| (x.phase == Phase::Prefill && x.prefill_rate > 0.0).then_some(x.prefill_rate),
            since: (p.seconds > 0.0).then(|| p.rows as f64 / p.seconds),
            show: |v| format!("{} tok/s", count(v)),
            tone: t.prefill,
            from_zero: true,
        },
        Figure {
            name: "SSD read",
            of: |x: &Sample| (x.phase != Phase::Idle).then_some(x.ssd),
            since: (work > 0.0).then(|| (d.bytes + p.bytes) as f64 / 1e9 / work),
            show: |v| format!("{v:.1} GB/s"),
            tone: t.sources[4],
            from_zero: true,
        },
        Figure {
            name: "hit rate",
            of: |x: &Sample| x.hit,
            since: d.hit(),
            show: |v| pct(v, 1.0),
            tone: t.sources[0],
            from_zero: false,
        },
        Figure {
            name: "GPU busy",
            of: |x: &Sample| x.gpu,
            since: d.gpu_busy(),
            show: |v| pct(v, 1.0),
            tone: t.accent,
            from_zero: false,
        },
        Figure {
            name: "read wait",
            of: |x: &Sample| x.wait_ms,
            since: d.wait_ms(),
            show: |v| format!("{v:.1} ms/token"),
            tone: t.sources[4],
            from_zero: true,
        },
    ];
    let sw = w.saturating_sub(1 + 16 + 3 * cw).min(60);
    for f in &figures {
        let mut line = vec![text(" "), strong(format!("{:16}", f.name))];
        line.push(cell(h.mean(10, f.of).map(f.show), cw));
        line.push(cell(h.mean(h.samples.len(), f.of).map(f.show), cw));
        line.push(cell(f.since.map(f.show), cw));
        let v: Vec<Option<f64>> = h.samples.iter().map(f.of).collect();
        line.push(Span::styled(spark(&v, sw, f.from_zero), color(f.tone)));
        l.push(Line::from(line));
    }
    let tot = &s.totals;
    l.push(Line::from(vec![
        text(" "),
        strong(format!("{:16}", "served")),
        text(format!("{} requests", count(tot.requests as f64))),
        label("   prompts "),
        text(format!("{} tokens", count(tot.prompt_tokens as f64))),
        label(", reused "),
        text(pct(tot.reused_tokens as f64, tot.prompt_tokens as f64)),
        label("   generated "),
        text(format!("{} tokens", count(tot.generated_tokens as f64))),
    ]));
    l
}

/// The Overview tab's lines, each fitted to w: the requests now, the
/// throughput, the experts per decoded token, memory and the log's tail.
pub fn draw(s: &Snapshot, h: &History, log: &[String], w: usize, rows: usize) -> Vec<Line<'static>> {
    let t = get();
    let gw = if w >= 110 { 24 } else { 10 }; // a gauge's cells
    let mut l: Vec<Line<'static>> = Vec::new();

    // now: the phase, the requests and their speed
    let decoding = s.phase == Phase::Decode;
    let speeds = h.speeds();
    let speed = |n: u64| speeds.iter().find(|p| p.0 == n).map_or(0.0, |p| p.1);
    let sum: f64 = speeds.iter().map(|p| p.1).sum();
    l.push(head(w, "Now"));
    let queue = if s.waiting > 0 {
        vec![text("   "), Span::styled(format!("{} waiting", s.waiting), color(t.sources[2]))]
    } else {
        vec![]
    };
    let mut line = vec![text(" ")];
    if decoding {
        let ns: usize = if w > 100 { 40 } else { 20 };
        // the latest ns samples' decoding speed, 0 before the first
        let mut recent = vec![Some(0.0); ns.saturating_sub(h.samples.len())];
        let total = |x: &Sample| Some(if x.phase == Phase::Decode { x.speeds.iter().map(|p| p.1).sum() } else { 0.0 });
        recent.extend(h.samples.iter().skip(h.samples.len().saturating_sub(ns)).map(total));
        line.extend(pad(vec![Span::styled("decoding", bold().fg(c(t.accent)))], 12));
        line.push(text(format!("{} request{}   ", s.requests.len(), if s.requests.len() == 1 { "" } else { "s" })));
        line.push(strong(format!("{sum:.1}")));
        line.push(text(" tok/s   "));
        line.push(Span::styled(spark(&recent, ns, true), color(t.accent)));
    } else if s.phase == Phase::Prefill && s.prefill.rows > 0 {
        let layers = s.layers.max(1) as f64;
        line.extend(pad(vec![Span::styled("prefill", bold().fg(c(t.prefill)))], 12));
        line.extend(bar(gw, &[(s.prefill.layer as f64 / layers, color(t.prefill))]));
        line.extend([text("  "), label("layer"), text(" "), strong(s.prefill.layer.to_string())]);
        line.extend([text(format!("/{}   ", s.layers)), label("tokens"), text(format!(" {}", count(s.prefill.rows as f64)))]);
    } else if s.phase == Phase::Prefill {
        line.extend(pad(vec![Span::styled("prefill", bold().fg(c(t.prefill)))], 12));
        line.push(label("starting; images are encoded first"));
    } else {
        line.extend(pad(vec![strong("idle")], 12));
        line.push(text(duration(s.phase_seconds)));
    }
    line.extend(queue);
    l.push(Line::from(line));
    for r in s.requests.iter().take(8) {
        let mut line = vec![text("   ")];
        line.extend(pad(vec![strong(format!("#{}", r.number))], 7));
        line.extend([label("session"), text(format!(" {:<5}", r.session))]);
        line.extend([label("prompt"), text(format!(" {:<8}", count(r.prompt as f64)))]);
        line.extend([label("reused"), text(format!(" {:<8}", count(r.reused as f64)))]);
        line.extend([label("generated"), text(format!(" {:<8}", count(r.generated as f64)))]);
        if r.generated > 0 {
            line.push(Span::styled(format!("{:.1} tok/s", speed(r.number)), color(t.accent)));
        }
        l.push(Line::from(line));
    }

    l.extend(throughput(s, h, w));

    // experts per decoded token, the last 10 s (21 samples) or since start
    let now = h.samples.back().map_or(0.0, |x| x.at);
    let old = h.samples.iter().rev().take(21).filter(|x| x.decode.rows < s.decode.rows && now - x.at <= 11.0).last().map(|x| x.decode);
    let x = s.decode.since(&old.unwrap_or_default());
    let rows_d = x.rows as f64;
    let part = x.by_source().map(|n| n as f64);
    let counted: f64 = part.iter().sum();
    let bytes = x.bytes as f64;
    let per_row = s.experts_per_row as f64;
    l.push(head(
        w,
        &if rows_d > 0.0 {
            format!("Experts per token · {} · {} tokens", if old.is_some() { "last 10 s" } else { "since start" }, count(rows_d))
        } else {
            "Experts per token".into()
        },
    ));
    if rows_d > 0.0 && counted > 0.0 {
        let shares: Vec<(f64, Style)> = (0..5).map(|i| (part[i] / counted, color(t.sources[i]))).collect();
        let each = counted / rows_d;
        let mut line = vec![text(" ")];
        line.extend(bar(if w >= 110 { 40 } else { 20 }, &shares));
        line.extend([text("   "), label("hit rate"), text(" "), strong(pct(x.hit().unwrap_or(0.0), 1.0))]);
        if part[3] > 0.0 {
            line.extend([text("   "), label("with substitutes"), text(" "), strong(pct(counted - part[4], counted))]);
        }
        if each < per_row - 0.05 {
            line.extend([text("   "), label("counted"), text(format!(" {each:.1}/{}", s.experts_per_row))]);
        }
        l.push(Line::from(line));
        // a table: the sources' names, experts per token, shares
        let mut tab = [vec![text(format!(" {:10}", ""))], vec![text(" "), label("experts   ")], vec![text(" "), label("share     ")]];
        for i in 0..5 {
            let cw = SOURCE_NAMES[i].len() + 4;
            tab[0].extend([text("  "), Span::styled("■", color(t.sources[i])), text(" "), label(SOURCE_NAMES[i])]);
            tab[1].push(text(format!("{:>cw$.1}", part[i] / rows_d)));
            tab[2].push(text(format!("{:>cw$}", pct(part[i], counted))));
        }
        l.extend(tab.into_iter().map(Line::from));
        let (wall, gpu) = (x.seconds, x.gpu_seconds);
        let remaps = x.remaps as f64 / rows_d; // slots mapped for the GPU again: its memory was short
        let mut line = vec![text(" "), label("per token "), text(" "), label("SSD")];
        let rate = if wall > 0.0 { bytes / wall / 1e9 } else { 0.0 };
        line.push(text(format!(" {:.2} GB · {rate:.1} GB/s   ", bytes / rows_d / 1e9)));
        line.extend([label("wait"), text(format!(" {:.1} ms   ", x.read_wait_seconds / rows_d * 1e3))]);
        line.extend([label("GPU"), text(format!(" {:.1} ms · {} busy   ", gpu / rows_d * 1e3, pct(gpu, wall)))]);
        line.push(label("read ahead"));
        line.push(text(format!(
            " {:.1} · {:.1} cancelled",
            x.prefetches as f64 / rows_d,
            x.cancelled as f64 / rows_d
        )));
        if remaps >= 0.05 {
            line.extend([text("   "), Span::styled(format!("remapped {remaps:.1}"), color(t.sources[2]))]);
        }
        l.push(Line::from(line));
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
    let (experts_tone, rest_tone) = (t.sources[0], t.sources[1]);
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
                &[(share(mapped, lim), color(experts_tone)), (share(weights, lim), color(rest_tone)), (share(inuse - mine, lim), dim())],
            ));
            line.push(text("  "));
            line.extend(pad(fig, fw));
            line.extend([Span::styled("■", color(experts_tone)), text(" "), label("experts"), text(format!(" {mapped:.1} ")), slots]);
            line.extend([text("   "), Span::styled("■", color(rest_tone)), text(" "), label("weights")]);
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
        line.extend(bar(gw, &[(share(ex, ram), color(experts_tone)), (share(rest, ram), color(rest_tone))]));
        line.push(text("  "));
        line.extend(pad(vec![label("myds4"), text(" "), strong(format!("{foot:.1}")), text(" GiB")], fw));
        line.extend([Span::styled("■", color(experts_tone)), text(" "), label("experts"), text(format!(" {ex:.1}   "))]);
        line.extend([Span::styled("■", color(rest_tone)), text(" "), label("rest"), text(format!(" {rest:.1}   ")), label("slots")]);
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
        let swap0 = h.swapouts.unwrap_or(m.swapouts);
        let spare = (m.free.saturating_add(m.file).min(m.total)) as f64 / GIB;
        let file = m.file as f64 / GIB;
        let used = ram - spare;
        let level = if m.pressure >= 4 {
            Span::styled("critical", color(t.sources[4]))
        } else if m.pressure >= 2 {
            Span::styled("warning", color(t.sources[2]))
        } else {
            Span::styled("normal", color(t.requests[2]))
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

    // the log's last lines
    l.push(head(w, "Log"));
    let show = rows.saturating_sub(l.len());
    for line in &log[log.len().saturating_sub(show)..] {
        l.push(Line::from(Span::styled(format!(" {line}"), color(t.label))));
    }
    l.truncate(rows);
    l.into_iter().map(|line| fit(line, w)).collect()
}
