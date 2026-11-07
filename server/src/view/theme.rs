//! The live view's colors, in 24-bit RGB: where an expert was, the requests,
//! the phases, the accent and the grays, for a light or a dark terminal.
//! Tints (the chosen KV node's, the last minute's decoding phase) blend
//! toward the terminal's background. When the view starts, it
//! asks the terminal for that color (OSC 11, [`detect`]) and takes the light
//! or the dark colors by its brightness; without an answer it takes the light
//! ones. --tui-theme light or dark chooses them instead of asking.

use std::io::Write;
use std::os::fd::AsRawFd;
use std::sync::RwLock;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use ratatui::style::Color;

pub type Rgb = (u8, u8, u8);

/// The colors for one kind of background.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Theme {
    /// The terminal's background, which tints blend toward.
    pub ground: Rgb,
    /// Text that stands out: a session's name in the KV tree.
    pub ink: Rgb,
    pub label: Rgb,
    pub rule: Rgb,
    /// The idle phase in the last minute's chart.
    pub faint: Rgb,
    /// The KV tree's lines of sessions idle for under a minute, an hour, longer.
    pub idle: [Rgb; 3],
    pub accent: Rgb,
    /// Text on the accent: the tab shown, the request shown.
    pub on_accent: Rgb,
    pub prefill: Rgb,
    /// Where an expert was, in the order of the snapshot's sources: in
    /// memory, read ahead, late, substituted, from the SSD.
    pub sources: [Rgb; 5],
    /// The requests' colors, in the order they first appear.
    pub requests: [Rgb; 6],
}

pub const LIGHT: Theme = Theme {
    ground: (0xFF, 0xFF, 0xFF),
    ink: (0x1D, 0x1D, 0x1F),
    label: (0x6E, 0x6E, 0x73),
    rule: (0xC7, 0xC7, 0xCC),
    faint: (0xEC, 0xEC, 0xF1),
    idle: [(0x6E, 0x6E, 0x73), (0x9A, 0x9A, 0xA0), (0xBC, 0xBC, 0xC2)],
    accent: (0x58, 0x56, 0xD6),
    on_accent: (0xFF, 0xFF, 0xFF),
    prefill: (0x2F, 0xA7, 0xB8),
    sources: [(0x2F, 0x6F, 0xED), (0x8C, 0xC2, 0xFF), (0xF2, 0xA9, 0x3B), (0xA5, 0x7E, 0xF2), (0xEE, 0x5A, 0x52)],
    requests: [(0x58, 0x56, 0xD6), (0xF0, 0x8A, 0x24), (0x3B, 0xA5, 0x5C), (0xD6, 0x45, 0x7A), (0x9C, 0x7A, 0x4F), (0x8E, 0x8E, 0x93)],
};

pub const DARK: Theme = Theme {
    ground: (0x1E, 0x1E, 0x1E),
    ink: (0xF2, 0xF2, 0xF7),
    label: (0x98, 0x98, 0x9F),
    rule: (0x48, 0x48, 0x4A),
    faint: (0x2C, 0x2C, 0x2E),
    idle: [(0xAE, 0xAE, 0xB2), (0x7C, 0x7C, 0x80), (0x58, 0x58, 0x5C)],
    accent: (0x7D, 0x7A, 0xFF),
    on_accent: (0xFF, 0xFF, 0xFF),
    prefill: (0x40, 0xC8, 0xE0),
    sources: [(0x40, 0x8C, 0xFF), (0x9E, 0xCB, 0xFF), (0xFF, 0xB3, 0x40), (0xBF, 0x8C, 0xFF), (0xFF, 0x6B, 0x60)],
    requests: [(0x7D, 0x7A, 0xFF), (0xFF, 0x9F, 0x0A), (0x30, 0xD1, 0x58), (0xFF, 0x37, 0x5F), (0xAC, 0x8E, 0x68), (0x98, 0x98, 0x9D)],
};

static THEME: RwLock<Theme> = RwLock::new(LIGHT);
/// Whether the terminal shows 24-bit color; without it, colors are the
/// nearest of the 256 standard ones.
static TRUE_COLOR: AtomicBool = AtomicBool::new(true);

/// Whether the terminal says it shows 24-bit color (COLORTERM).
pub fn true_color_terminal() -> bool {
    std::env::var("COLORTERM").is_ok_and(|v| matches!(v.as_str(), "truecolor" | "24bit"))
}

pub fn set_true_color(on: bool) {
    TRUE_COLOR.store(on, Ordering::Relaxed);
}

pub fn true_color() -> bool {
    TRUE_COLOR.load(Ordering::Relaxed)
}

/// The nearest of the 256 standard colors: the 6 x 6 x 6 cube or the grays.
fn indexed(x: Rgb) -> u8 {
    const LEVELS: [u8; 6] = [0, 95, 135, 175, 215, 255];
    let near = |v: u8| (0..6).min_by_key(|&i| (LEVELS[i] as i32 - v as i32).abs()).unwrap_or(0);
    let d = |a: Rgb, b: Rgb| {
        let e = |p: u8, q: u8| (p as i32 - q as i32).pow(2);
        e(a.0, b.0) + e(a.1, b.1) + e(a.2, b.2)
    };
    let (r, g, b) = (near(x.0), near(x.1), near(x.2));
    let cube = (LEVELS[r], LEVELS[g], LEVELS[b]);
    let mean = (x.0 as i32 + x.1 as i32 + x.2 as i32) / 3;
    let k = ((mean - 8).max(0) / 10).min(23) as u8;
    let v = 8 + 10 * k;
    if d(x, (v, v, v)) < d(x, cube) { 232 + k } else { (16 + 36 * r + 6 * g + b) as u8 }
}

/// The colors in use.
pub fn get() -> Theme {
    *THEME.read().unwrap_or_else(|e| e.into_inner())
}

pub fn set(t: Theme) {
    *THEME.write().unwrap_or_else(|e| e.into_inner()) = t;
}

/// The colors for a terminal with background g: the light ones on a light
/// background, the dark ones on a dark one, their tints blending toward g itself.
pub fn for_ground(g: Rgb) -> Theme {
    let brightness = (0.299 * g.0 as f64 + 0.587 * g.1 as f64 + 0.114 * g.2 as f64) / 255.0;
    Theme { ground: g, ..if brightness >= 0.5 { LIGHT } else { DARK } }
}

pub fn c(x: Rgb) -> Color {
    if true_color() { Color::Rgb(x.0, x.1, x.2) } else { Color::Indexed(indexed(x)) }
}

/// a blended toward b: t = 0 is a, t = 1 is b.
pub fn mix(a: Rgb, b: Rgb, t: f64) -> Rgb {
    let m = |x: u8, y: u8| (x as f64 + (y as f64 - x as f64) * t.clamp(0.0, 1.0)).round() as u8;
    (m(a.0, b.0), m(a.1, b.1), m(a.2, b.2))
}

/// The terminal's background, as it answers OSC 11 within `wait`; None when
/// it does not answer. The terminal must be in raw mode, and the view must not
/// read the keys yet: the answer comes in as typed input. Keys typed during
/// the wait are dropped.
pub fn detect(wait: Duration) -> Option<Rgb> {
    let mut err = std::io::stderr();
    err.write_all(b"\x1b]11;?\x1b\\").ok()?;
    err.flush().ok()?;
    let fd = std::io::stdin().as_raw_fd();
    let until = Instant::now() + wait;
    let mut got = Vec::new();
    loop {
        let left = until.checked_duration_since(Instant::now())?;
        let mut p = libc::pollfd { fd, events: libc::POLLIN, revents: 0 };
        // Safety: one valid pollfd, and a buffer of the length read is given.
        if unsafe { libc::poll(&mut p, 1, left.as_millis().clamp(1, 1000) as i32) } <= 0 {
            return None;
        }
        let mut buf = [0u8; 64];
        let n = unsafe { libc::read(fd, buf.as_mut_ptr().cast(), buf.len()) };
        if n <= 0 {
            return None;
        }
        got.extend_from_slice(&buf[..n as usize]);
        if let Some(rgb) = parse_osc11(&got) {
            return Some(rgb);
        }
        if got.len() > 512 {
            return None;
        }
    }
}

/// The color in a terminal's answer to OSC 11, "ESC ] 11 ; rgb:R/G/B" ended
/// by BEL or ESC \, each channel 1 to 4 hexadecimal digits; None until a whole
/// answer is there.
pub fn parse_osc11(b: &[u8]) -> Option<Rgb> {
    let s = String::from_utf8_lossy(b);
    let at = s.find("]11;rgb:")? + "]11;rgb:".len();
    let rest = &s[at..];
    let end = rest.find(['\x07', '\x1b'])?;
    let mut parts = rest[..end].split('/');
    let mut channel = || -> Option<u8> {
        let h = parts.next()?;
        if h.is_empty() || h.len() > 4 {
            return None;
        }
        let v = u32::from_str_radix(h, 16).ok()?;
        let max = (1u32 << (4 * h.len())) - 1;
        Some(((v * 255 + max / 2) / max) as u8)
    };
    let rgb = (channel()?, channel()?, channel()?);
    parts.next().is_none().then_some(rgb)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_terminals_answer_gives_its_background() {
        assert_eq!(parse_osc11(b"\x1b]11;rgb:ffff/ffff/ffff\x1b\\"), Some((255, 255, 255)));
        assert_eq!(parse_osc11(b"\x1b]11;rgb:1e1e/1e1e/1e1e\x07"), Some((0x1E, 0x1E, 0x1E)));
        assert_eq!(parse_osc11(b"x\x1b]11;rgb:0/8/f\x07"), Some((0, 136, 255)));
        // not whole yet, or not a color
        assert_eq!(parse_osc11(b"\x1b]11;rgb:ffff/ff"), None);
        assert_eq!(parse_osc11(b"\x1b]11;rgb:ffff/ffff\x07"), None);
        assert_eq!(parse_osc11(b"\x1b]11;rgb:gg/00/00\x07"), None);
    }

    #[test]
    fn a_light_background_takes_the_light_colors_and_a_dark_one_the_dark() {
        assert_eq!(for_ground((0xFA, 0xF9, 0xF5)), Theme { ground: (0xFA, 0xF9, 0xF5), ..LIGHT });
        assert_eq!(for_ground((0x10, 0x12, 0x18)), Theme { ground: (0x10, 0x12, 0x18), ..DARK });
        // tints blend toward the background
        assert_eq!(mix((0, 0, 0), (200, 100, 50), 0.5), (100, 50, 25));
    }

    #[test]
    fn without_24_bit_color_a_color_is_the_nearest_standard_one() {
        assert_eq!(indexed((0, 0, 0)), 16);
        assert_eq!(indexed((255, 255, 255)), 231);
        assert_eq!(indexed((0xEE, 0x5A, 0x52)), 16 + 36 * 5 + 6 + 1); // the SSD's red: (255, 95, 95)
        assert_eq!(indexed((0x80, 0x80, 0x80)), 244); // a gray
    }
}
