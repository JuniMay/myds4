//! A Qwen3.8 model through HTTP (docs/server.md#models): what the worker
//! keeps of a request that ends early, the token budget and the context's end
//! with drafts and without, and a Qwen model served beside DeepSeek.
//!
//! It runs only when MYDS4_QWEN names a Qwen3.8 checkpoint directory, one
//! model process at a time:
//!
//!     MYDS4_QWEN=DIR cargo test --release --manifest-path server/Cargo.toml \
//!       --target-dir build/cargo --test qwen -- --nocapture --test-threads 1
//!
//! The test of both models in one process also needs MYDS4_CHECKPOINT (the
//! DeepSeek checkpoint directory) and MYDS4_EXPERTS (its GGUF experts).
//! MYDS4_SERVER picks the server binary (default build/myds4-server).

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
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
    /// The server's log (stderr), line by line.
    log: Arc<Mutex<Vec<String>>>,
}

impl Server {
    fn start(dir: &str, extra: &[&str]) -> Server {
        let mut args: Vec<String> = ["-m", dir, "--kv-cache", "8"].map(String::from).to_vec();
        args.extend(extra.iter().map(|a| a.to_string()));
        Server::spawn(args)
    }

    fn spawn(mut args: Vec<String>) -> Server {
        let port = TcpListener::bind("127.0.0.1:0").and_then(|l| l.local_addr()).expect("a free port").port();
        let key = format!("qwen-test-{}-{port}-key", std::process::id());
        let server = std::env::var("MYDS4_SERVER").map(PathBuf::from).unwrap_or_else(|_| root().join("build/myds4-server"));
        args.extend(["--serve".to_string(), port.to_string()]);
        let mut child = Command::new(server)
            .args(&args)
            .env("MYDS4_API_KEY", &key)
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .expect("start the server");
        let log = Arc::new(Mutex::new(Vec::new()));
        let (mut err, lines) = (child.stderr.take().expect("the server's stderr"), log.clone());
        std::thread::spawn(move || {
            let (mut raw, mut buf) = (Vec::new(), [0u8; 4096]);
            while let Ok(n) = err.read(&mut buf) {
                if n == 0 {
                    break;
                }
                raw.extend_from_slice(&buf[..n]);
                while let Some(end) = raw.iter().position(|&b| b == b'\n') {
                    lines.lock().unwrap().push(String::from_utf8_lossy(&raw[..end]).to_string());
                    raw.drain(..=end);
                }
            }
        });
        let s = Server { child, port, key, log };
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

    fn get(&self, path: &str) -> Value {
        let mut s = TcpStream::connect(("127.0.0.1", self.port)).expect("connect");
        let head =
            format!("GET {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: Bearer {}\r\nConnection: close\r\n\r\n", self.key);
        s.write_all(head.as_bytes()).expect("send");
        let mut raw = Vec::new();
        s.read_to_end(&mut raw).expect("read");
        let split = raw.windows(4).position(|w| w == b"\r\n\r\n").expect("headers");
        serde_json::from_slice(&raw[split + 4..]).expect("a JSON answer")
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

fn counting() -> Value {
    json!([{"role": "user", "content": "Count from 1 to 300, one number per line."}])
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
        let s = Server::start(&dir, &["--kv-cache", "12"]);
        let a = s.chat(&body(first.clone(), 1, false));
        (a["usage"]["prompt_tokens"].as_i64().unwrap(), s.chat(&body(second.clone(), 24, false)))
    };
    let s = Server::start(&dir, &["--kv-cache", "12"]);
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
    let answer = s.chat(&body(counting(), 1000, false));
    let (n, made) = (answer["usage"]["prompt_tokens"].as_i64().unwrap(), answer["usage"]["completion_tokens"].as_i64().unwrap());
    eprintln!("{n} prompt tokens, {made} completion tokens, finish {}", answer["choices"][0]["finish_reason"]);
    assert_eq!(answer["choices"][0]["finish_reason"], "length");
    assert_eq!(made, 160 - n, "completion tokens of a reply that reaches the context's end");
}

/// With MTP drafts, a verify pass can keep the token and its draft, and the
/// token sampled after them goes out without a pass. A reply still takes
/// exactly max_tokens tokens, and at the context's end every position up to
/// it, whichever parity the limit has.
#[test]
fn drafts_keep_to_max_tokens_and_the_context_end() {
    let Some(dir) = checkpoint() else {
        eprintln!("MYDS4_QWEN is not set: skipped");
        return;
    };
    let _one = one_model();
    for ctx in [160, 161] {
        let s = Server::start(&dir, &["--ctx", &ctx.to_string(), "--draft", "1"]);
        for max_tokens in [20, 21, 22, 23] {
            let answer = s.chat(&body(counting(), max_tokens, false));
            let made = answer["usage"]["completion_tokens"].as_i64().unwrap();
            eprintln!("--ctx {ctx}, max_tokens {max_tokens}: {made} completion tokens, finish {}", answer["choices"][0]["finish_reason"]);
            assert_eq!(answer["choices"][0]["finish_reason"], "length");
            assert_eq!(made, max_tokens as i64, "completion tokens of a reply with max_tokens {max_tokens}");
        }
        let answer = s.chat(&body(counting(), 1000, false));
        let (n, made) = (answer["usage"]["prompt_tokens"].as_i64().unwrap(), answer["usage"]["completion_tokens"].as_i64().unwrap());
        eprintln!("--ctx {ctx}: {n} prompt tokens, {made} completion tokens, finish {}", answer["choices"][0]["finish_reason"]);
        assert_eq!(answer["choices"][0]["finish_reason"], "length");
        assert_eq!(made, ctx - n, "completion tokens of a reply that reaches the context's end");
    }
}

/// DeepSeek and a Qwen model in one process: each answers, and while both
/// are idle, the log reports an expert cache that shrinks only when
/// DeepSeek's cache (the only one) really has fewer slots.
#[test]
fn deepseek_and_qwen_share_one_process() {
    let (Some(qwen), Ok(deepseek), Ok(experts)) =
        (checkpoint(), std::env::var("MYDS4_CHECKPOINT"), std::env::var("MYDS4_EXPERTS"))
    else {
        eprintln!("MYDS4_QWEN, MYDS4_CHECKPOINT or MYDS4_EXPERTS is not set: skipped");
        return;
    };
    let _one = one_model();
    let config = std::env::temp_dir().join(format!("myds4-two-models-{}.json", std::process::id()));
    let models = json!({"models": [
        {"model": deepseek, "experts": experts, "cache": 40, "kv_cache": 4},
        {"model": qwen, "ctx": 4096, "kv_cache": 4}
    ]});
    std::fs::write(&config, models.to_string()).expect("write the config file");
    let s = Server::spawn(vec!["--config".to_string(), config.display().to_string()]);
    let _ = std::fs::remove_file(&config);
    // the config's order: DeepSeek first
    let ids: Vec<String> =
        s.get("/v1/models")["data"].as_array().expect("models").iter().map(|m| m["id"].as_str().unwrap().to_string()).collect();
    assert_eq!(ids.len(), 2, "models: {ids:?}");
    for id in &ids {
        let mut b = body(json!([{"role": "user", "content": "Name three colors, in one line."}]), 16, false);
        b["model"] = json!(id);
        let answer = s.chat(&b);
        eprintln!("{id}: {:?}", content(&answer));
        assert!(!content(&answer).trim().is_empty(), "{id} gave no text");
    }
    std::thread::sleep(Duration::from_secs(4)); // the worker's idle calls run about once a second
    let slots =
        s.get(&format!("/v1/models/{}", ids[0]))["status"]["expert_cache"]["slots"].as_i64().expect("DeepSeek's expert cache slots");
    let lines: Vec<String> = s.log.lock().unwrap().iter().filter(|l| l.contains("memory short")).cloned().collect();
    eprintln!("DeepSeek's expert cache: {slots} slots; {} memory-short lines", lines.len());
    if let Some(last) = lines.last() {
        let after: i64 = last.trim_end_matches(" slots").rsplit(' ').next().and_then(|n| n.parse().ok()).expect("a slot count");
        assert_eq!(after, slots, "the log says the cache shrank to {after} slots ({last:?}); it has {slots}");
    }
}
