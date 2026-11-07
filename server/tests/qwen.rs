//! A Qwen3.8 model through HTTP (docs/server.md#qwen38-models): what the
//! worker keeps of a request that ends early, and the context's end.
//!
//! It runs only when MYDS4_QWEN names a Qwen3.8 checkpoint directory, one
//! model process at a time:
//!
//!     MYDS4_QWEN=DIR cargo test --release --manifest-path server/Cargo.toml \
//!       --target-dir build/cargo --test qwen -- --nocapture --test-threads 1
//!
//! MYDS4_SERVER picks the server binary (default build/myds4-server).

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use serde_json::{Value, json};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).parent().expect("the repository").to_path_buf()
}

/// Held while a test runs a server: one model process at a time.
static MODEL: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn one_model() -> std::sync::MutexGuard<'static, ()> {
    MODEL.lock().unwrap_or_else(|e| e.into_inner())
}

fn checkpoint() -> Option<String> {
    std::env::var("MYDS4_QWEN").ok().filter(|d| Path::new(d).join("config.json").exists())
}

struct Server {
    child: Child,
    port: u16,
    key: String,
}

impl Server {
    fn start(dir: &str, extra: &[&str]) -> Server {
        let port = TcpListener::bind("127.0.0.1:0").and_then(|l| l.local_addr()).expect("a free port").port();
        let key = format!("qwen-test-{}-{port}-key", std::process::id());
        let server = std::env::var("MYDS4_SERVER").map(PathBuf::from).unwrap_or_else(|_| root().join("build/myds4-server"));
        let mut args: Vec<String> = ["-m", dir, "--serve", &port.to_string(), "--kv-cache", "8"].map(String::from).to_vec();
        args.extend(extra.iter().map(|a| a.to_string()));
        let child = Command::new(server)
            .args(&args)
            .env("MYDS4_API_KEY", &key)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("start the server");
        let s = Server { child, port, key };
        let deadline = Instant::now() + Duration::from_secs(300);
        while Instant::now() < deadline {
            if s.healthy() {
                return s;
            }
            std::thread::sleep(Duration::from_millis(500));
        }
        panic!("the server did not start");
    }

    fn healthy(&self) -> bool {
        let Ok(mut s) = TcpStream::connect(("127.0.0.1", self.port)) else { return false };
        let mut raw = Vec::new();
        s.write_all(b"GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n").is_ok()
            && s.read_to_end(&mut raw).is_ok()
            && raw.starts_with(b"HTTP/1.1 200")
    }

    /// A request and its whole answer; with `leave`, the client closes its
    /// connection that long after sending the request.
    fn post(&self, body: &Value, leave: Option<Duration>) -> (u16, Vec<u8>) {
        let Ok(mut s) = TcpStream::connect(("127.0.0.1", self.port)) else { return (0, Vec::new()) };
        s.set_read_timeout(Some(Duration::from_secs(900))).ok();
        let payload = body.to_string();
        let head = format!(
            "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: Bearer {}\r\n\
             Content-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            self.key,
            payload.len()
        );
        if s.write_all(head.as_bytes()).and_then(|_| s.write_all(payload.as_bytes())).is_err() {
            return (0, Vec::new());
        }
        if let Some(after) = leave {
            std::thread::sleep(after);
            return (0, Vec::new()); // the stream drops here: the server sees the client leave
        }
        let mut raw = Vec::new();
        if s.read_to_end(&mut raw).is_err() {
            return (0, raw);
        }
        let Some(split) = raw.windows(4).position(|w| w == b"\r\n\r\n") else { return (0, raw) };
        let headers = String::from_utf8_lossy(&raw[..split]).to_lowercase();
        let status = headers.split_whitespace().nth(1).and_then(|c| c.parse().ok()).unwrap_or(0);
        let mut rest = raw[split + 4..].to_vec();
        if headers.contains("transfer-encoding: chunked") {
            let (mut out, mut at) = (Vec::new(), 0);
            while let Some(end) = rest[at..].windows(2).position(|w| w == b"\r\n") {
                let size = usize::from_str_radix(String::from_utf8_lossy(&rest[at..at + end]).trim(), 16).unwrap_or(0);
                if size == 0 {
                    break;
                }
                let start = at + end + 2;
                out.extend_from_slice(&rest[start..start + size]);
                at = start + size + 2;
            }
            rest = out;
        }
        (status, rest)
    }

    fn chat(&self, body: &Value) -> Value {
        let (status, answer) = self.post(body, None);
        assert_eq!(status, 200, "chat completion: {}", String::from_utf8_lossy(&answer));
        serde_json::from_slice(&answer).expect("a JSON answer")
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = Command::new("kill").args(["-TERM", &self.child.id().to_string()]).status();
        let deadline = Instant::now() + Duration::from_secs(60);
        while Instant::now() < deadline {
            if let Ok(Some(_)) = self.child.try_wait() {
                return;
            }
            std::thread::sleep(Duration::from_millis(200));
        }
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn long_system(notes: usize) -> String {
    "You are a careful assistant. ".to_string()
        + &(1..notes)
            .map(|i| format!("Note {i}: answer plainly, keep lists short, and say when you are not sure about item {i}."))
            .collect::<Vec<_>>()
            .join(" ")
}

fn body(messages: Value, max_tokens: u32, stream: bool) -> Value {
    json!({"model": "qwen", "messages": messages, "temperature": 0, "max_tokens": max_tokens,
           "reasoning_effort": "none", "stream": stream})
}

fn content(answer: &Value) -> String {
    answer["choices"][0]["message"]["content"].as_str().unwrap_or("").to_string()
}

/// A request whose client leaves during its prefill leaves a state that has
/// run only part of its prompt. A later prompt that extends the whole of the
/// first must resume where that state really ends, and get the reply that
/// the same conversation gets where nothing was cancelled.
#[test]
fn a_prefill_cancelled_midway_leaves_a_state_that_resumes_where_it_ends() {
    let Some(dir) = checkpoint() else {
        eprintln!("MYDS4_QWEN is not set: skipped");
        return;
    };
    let _one = one_model();
    // about 38,000 tokens: a prefill of about a minute, which the client leaves
    // after 2 s
    let system = long_system(1400);
    let first = json!([{"role": "system", "content": system}, {"role": "user", "content": "Name three fruits."}]);
    let second = json!([
        {"role": "system", "content": system},
        {"role": "user", "content": "Name three fruits."},
        {"role": "assistant", "content": "Apples, pears and plums."},
        {"role": "user", "content": "Now name three vegetables, in one line."}
    ]);
    let (n1, reference) = {
        let s = Server::start(&dir, &[]);
        let a = s.chat(&body(first.clone(), 1, false));
        (a["usage"]["prompt_tokens"].as_i64().unwrap(), s.chat(&body(second.clone(), 24, false)))
    };
    let s = Server::start(&dir, &[]);
    s.post(&body(first, 24, true), Some(Duration::from_secs(2)));
    std::thread::sleep(Duration::from_secs(20)); // the server notices, and ends the request between passes
    let warm = s.chat(&body(second, 24, false));
    let (n, hit) = (warm["usage"]["prompt_tokens"].as_i64().unwrap(), warm["usage"]["prompt_cache_hit_tokens"].as_i64().unwrap());
    eprintln!(
        "reference: {:?}\nafter the cancelled prefill: {:?} ({hit} of {n} prompt tokens reused; the first prompt has {n1})",
        content(&reference),
        content(&warm)
    );
    assert!(hit < n1, "the second prompt reused {hit} tokens, the whole first prompt ({n1}): its prefill was not cut short");
    assert_eq!(content(&warm), content(&reference), "the reply after a cancelled prefill");
}

/// A reply that runs to the context's end takes every position up to it:
/// ctx - prompt tokens.
#[test]
fn a_reply_takes_every_position_up_to_the_context_end() {
    let Some(dir) = checkpoint() else {
        eprintln!("MYDS4_QWEN is not set: skipped");
        return;
    };
    let _one = one_model();
    let s = Server::start(&dir, &["--ctx", "160"]);
    let answer = s.chat(&body(json!([{"role": "user", "content": "Count from 1 to 300, one number per line."}]), 1000, false));
    let (n, made) = (answer["usage"]["prompt_tokens"].as_i64().unwrap(), answer["usage"]["completion_tokens"].as_i64().unwrap());
    eprintln!("{n} prompt tokens, {made} completion tokens, finish {}", answer["choices"][0]["finish_reason"]);
    assert_eq!(answer["choices"][0]["finish_reason"], "length");
    assert_eq!(made, 160 - n, "completion tokens of a reply that reaches the context's end");
}
