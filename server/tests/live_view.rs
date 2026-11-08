//! The live view (--tui) with a model, in a pseudo-terminal of 160 x 46
//! cells that answers the view's question for its background as a dark
//! terminal would. While a request decodes, the test presses the tabs' keys,
//! scrolls a node's text with the wheel and clicks a tab, and checks what each
//! tab draws: the expert map's request, the KV tree and a node's text, the
//! last minute, log and memory tiles; then q, and the server's exit.
//!
//! It runs only when MYDS4_CHECKPOINT (the checkpoint directory) and
//! MYDS4_EXPERTS (a GGUF file) are set; MYDS4_SERVER picks the server
//! (default build/myds4-server). It loads the model once (one model process).

use std::fs::File;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::os::fd::{FromRawFd, OwnedFd};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("the repository")
        .to_path_buf()
}

/// A pseudo-terminal of cols x rows cells: its controlling side and the side
/// the server gets as its terminal.
fn pty(cols: u16, rows: u16) -> (File, OwnedFd) {
    let (mut m, mut s) = (0, 0);
    let mut size = libc::winsize {
        ws_row: rows,
        ws_col: cols,
        ws_xpixel: 0,
        ws_ypixel: 0,
    };
    // Safety: openpty fills two descriptors; no name or settings are asked for.
    let r = unsafe {
        libc::openpty(
            &mut m,
            &mut s,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            &mut size,
        )
    };
    assert_eq!(r, 0, "openpty");
    // Safety: both descriptors are new and owned here.
    unsafe { (File::from(OwnedFd::from_raw_fd(m)), OwnedFd::from_raw_fd(s)) }
}

fn http(port: u16, key: &str, method: &str, path: &str, body: &str) -> Option<String> {
    let mut s = TcpStream::connect(("127.0.0.1", port)).ok()?;
    s.set_read_timeout(Some(Duration::from_secs(600))).ok();
    let head = format!(
        "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: Bearer {key}\r\nContent-Type: application/json\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    s.write_all(head.as_bytes()).ok()?;
    s.write_all(body.as_bytes()).ok()?;
    let mut raw = String::new();
    s.read_to_string(&mut raw).ok()?;
    raw.starts_with("HTTP/1.1 200").then_some(raw)
}

#[test]
fn the_live_view_draws_every_tab_with_a_model() {
    let (Ok(checkpoint), Ok(experts)) = (
        std::env::var("MYDS4_CHECKPOINT"),
        std::env::var("MYDS4_EXPERTS"),
    ) else {
        eprintln!("skipped: set MYDS4_CHECKPOINT and MYDS4_EXPERTS");
        return;
    };
    let server = std::env::var("MYDS4_SERVER")
        .map(PathBuf::from)
        .unwrap_or_else(|_| root().join("build/myds4-server"));
    let port = TcpListener::bind("127.0.0.1:0")
        .and_then(|l| l.local_addr())
        .expect("a free port")
        .port();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    let key = format!("live-view-{}-{:x}", std::process::id(), nanos);
    let (mut master, slave) = pty(160, 46);
    let mut child = Command::new(&server)
        .args([
            "-m",
            &checkpoint,
            "--experts",
            &experts,
            "--cache",
            "40",
            "--kv-cache",
            "8",
            "--ctx",
            "32768",
        ])
        .args(["--serve", &port.to_string(), "--tui"])
        .env("MYDS4_API_KEY", &key)
        .env("TERM", "xterm-256color")
        .env("COLORTERM", "truecolor")
        .env_remove("NO_COLOR")
        .stdin(Stdio::from(slave.try_clone().expect("a descriptor")))
        .stdout(Stdio::from(slave.try_clone().expect("a descriptor")))
        .stderr(Stdio::from(slave))
        .spawn()
        .expect("start the server");

    // What the server writes to its terminal; the view's question for the
    // background is answered: a dark gray.
    let out = Arc::new(Mutex::new(Vec::<u8>::new()));
    let mut reader = master.try_clone().expect("a descriptor");
    let mut answer = master.try_clone().expect("a descriptor");
    let seen = out.clone();
    let pump = std::thread::spawn(move || {
        let (mut buf, mut answered) = ([0u8; 65536], false);
        while let Ok(n) = reader.read(&mut buf) {
            if n == 0 {
                break;
            }
            let mut o = seen.lock().unwrap();
            o.extend_from_slice(&buf[..n]);
            if !answered && o.windows(6).any(|w| w == b"\x1b]11;?") {
                let _ = answer.write_all(b"\x1b]11;rgb:1e1e/1e1e/1e1e\x1b\\");
                answered = true;
            }
        }
    });
    let since = |from: usize| String::from_utf8_lossy(&out.lock().unwrap()[from..]).into_owned();

    let ready = Instant::now();
    while http(port, &key, "GET", "/health", "").is_none() {
        assert!(
            ready.elapsed() < Duration::from_secs(600),
            "the server did not start"
        );
        std::thread::sleep(Duration::from_millis(500));
    }
    // a reply long enough to decode through the keys below
    let body = r#"{"model": "deepseek-v4.1-flash", "messages": [{"role": "user", "content": "Count from 1 to 300, one number per line."}],
                   "max_tokens": 1200, "temperature": 0, "thinking": false, "stream": false}"#;
    let (p, k) = (port, key.clone());
    let request = std::thread::spawn(move || http(p, &k, "POST", "/v1/chat/completions", body));
    std::thread::sleep(Duration::from_secs(4)); // its prefill, then decoding
    // the screen after each tab's keys and the mouse's reports (SGR, 1-based):
    // a wheel step down over the text panel, then a click on "4 Last minute",
    // which at 160 columns spans columns 43 to 57 of row 2
    let wheel = "\x1b[<65;121;20M";
    let click = "\x1b[<0;46;2M\x1b[<0;46;2m";
    let mut tabs = std::collections::HashMap::new();
    let steps = [
        ("experts", "3", 3),
        ("kv", "2", 2),
        ("down", "\x1b[B", 1),
        ("text", "\r", 3),
        ("wheel", wheel, 1),
        ("minute", click, 3),
        ("log", "5", 1),
        ("memory", "6", 1),
        ("overview", "1", 2),
    ];
    for (name, keys, wait) in steps {
        master.write_all(keys.as_bytes()).expect("a key");
        std::thread::sleep(Duration::from_secs(wait));
        tabs.insert(name, screen(&since(0), 160, 46).join("\n"));
    }
    let answer = request.join().expect("the request").expect("an answer");
    master.write_all(b"q").expect("q");
    let stop = Instant::now();
    let status = loop {
        if let Some(s) = child.try_wait().expect("the server") {
            break s;
        }
        assert!(
            stop.elapsed() < Duration::from_secs(60),
            "the server did not stop after q"
        );
        std::thread::sleep(Duration::from_millis(200));
    };
    drop(master);
    let _ = pump.join();
    let all = since(0);
    println!(
        "answer: {}",
        answer
            .lines()
            .last()
            .unwrap_or("")
            .chars()
            .take(120)
            .collect::<String>()
    );
    let checks = [
        (
            "the view asked for the background",
            all.contains("\x1b]11;?"),
        ),
        (
            "the dark colors (the dark accent)",
            all.contains("125;122;255"),
        ),
        (
            "the mouse reported, then given back",
            all.contains("\x1b[?1000h") && all.contains("\x1b[?1000l"),
        ),
        (
            "the tabs",
            tabs["overview"].contains("1 Overview") && tabs["overview"].contains("6 Memory"),
        ),
        (
            "the expert map's request",
            tabs["experts"].contains("Request") && tabs["experts"].contains("#1 · s1"),
        ),
        (
            "four experts a cell at 160 columns",
            tabs["experts"].contains("four experts"),
        ),
        (
            "the KV tree's axis and session",
            tabs["kv"].contains("tokens") && tabs["kv"].contains("s1"),
        ),
        (
            "a node's text",
            tabs["text"].contains("Count from 1 to 300"),
        ),
        (
            "the wheel scrolled the text",
            tabs["wheel"].contains("lines 4–"),
        ),
        (
            "a click chose the last minute",
            tabs["minute"].contains("phase") && tabs["minute"].contains("tokens/s"),
        ),
        ("the log", tabs["log"].contains("req 1:")),
        (
            "the shared memory budget and tile",
            tabs["memory"].contains("Shared KV budget") && tabs["memory"].contains("GiB tracked"),
        ),
        ("the server's exit status 0", status.success()),
    ];
    let failed: Vec<&str> = checks.iter().filter(|c| !c.1).map(|c| c.0).collect();
    for (name, ok) in &checks {
        println!("{} {name}", if *ok { "ok  " } else { "FAIL" });
    }
    if !failed.is_empty() {
        for (name, _, _) in steps {
            println!("--- the screen after {name}'s keys:\n{}", tabs[name]);
        }
    }
    assert!(failed.is_empty(), "not drawn or not done: {failed:?}");
}

/// What a terminal of cols x rows cells shows after `out`: the text written
/// where the cursor moves put it. Enough of a terminal for the view: cursor
/// moves, clears and wide characters; colors and modes are passed over.
fn screen(out: &str, cols: usize, rows: usize) -> Vec<String> {
    use unicode_width::UnicodeWidthChar;
    let mut grid = vec![vec![' '; cols]; rows];
    let (mut r, mut c) = (0usize, 0usize);
    let mut it = out.chars().peekable();
    while let Some(ch) = it.next() {
        match ch {
            '\x1b' => match it.next() {
                Some('[') => {
                    let (mut params, mut fin) = (String::new(), ' ');
                    for d in it.by_ref() {
                        if ('@'..='~').contains(&d) {
                            fin = d;
                            break;
                        }
                        params.push(d);
                    }
                    let n: Vec<usize> = params.split(';').map(|x| x.parse().unwrap_or(0)).collect();
                    match fin {
                        'H' | 'f' => {
                            r = n.first().copied().unwrap_or(1).max(1) - 1;
                            c = n.get(1).copied().unwrap_or(1).max(1) - 1;
                        }
                        'J' if params == "2" => grid = vec![vec![' '; cols]; rows],
                        'K' if r < rows => grid[r][c.min(cols)..].fill(' '),
                        _ => {}
                    }
                }
                Some(']') => {
                    while let Some(d) = it.next() {
                        if d == '\x07' || (d == '\x1b' && it.next_if_eq(&'\\').is_some()) {
                            break;
                        }
                    }
                }
                _ => {}
            },
            '\r' => c = 0,
            '\n' => r += 1,
            ch if !ch.is_control() => {
                let w = ch.width().unwrap_or(0);
                if r < rows && c < cols {
                    grid[r][c] = ch;
                    if w == 2 && c + 1 < cols {
                        grid[r][c + 1] = '\0';
                    }
                }
                c += w;
            }
            _ => {}
        }
    }
    grid.into_iter()
        .map(|row| row.into_iter().filter(|&ch| ch != '\0').collect())
        .collect()
}
