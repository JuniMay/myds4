//! A Qwen3.8 model through HTTP: what the worker
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
//! MYDS4_AUDIO_REQUEST supplies a JSON chat request containing input_audio
//! for the Transcribe reuse test; /dev/stdin accepts an in-memory fixture.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::{Value, json};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("the repository")
        .to_path_buf()
}

/// Held while a test runs a server: one model process at a time.
static MODEL: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn one_model() -> std::sync::MutexGuard<'static, ()> {
    MODEL.lock().unwrap_or_else(|e| e.into_inner())
}

fn checkpoint() -> Option<String> {
    std::env::var("MYDS4_QWEN")
        .ok()
        .filter(|d| Path::new(d).join("config.json").exists())
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

    fn spawn(args: Vec<String>) -> Server {
        Self::spawn_input(args, None)
    }

    fn with_config(config: Value) -> Server {
        Self::spawn_input(
            vec!["--config".into(), "/dev/stdin".into()],
            Some(config.to_string()),
        )
    }

    fn spawn_input(mut args: Vec<String>, input: Option<String>) -> Server {
        let port = TcpListener::bind("127.0.0.1:0")
            .and_then(|l| l.local_addr())
            .expect("a free port")
            .port();
        let key = format!("qwen-test-{}-{port}-key", std::process::id());
        let server = std::env::var("MYDS4_SERVER")
            .map(PathBuf::from)
            .unwrap_or_else(|_| root().join("build/myds4-server"));
        args.extend(["--serve".to_string(), port.to_string()]);
        let mut child = Command::new(server)
            .args(&args)
            .env("MYDS4_API_KEY", &key)
            .env("MYDS4_LOCK", root().join("build/myds4.lock"))
            .stdin(if input.is_some() {
                Stdio::piped()
            } else {
                Stdio::null()
            })
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .expect("start the server");
        if let Some(text) = input {
            let mut stdin = child.stdin.take().expect("config input");
            stdin.write_all(text.as_bytes()).expect("send the config");
        }
        let log = Arc::new(Mutex::new(Vec::new()));
        let (mut err, lines) = (
            child.stderr.take().expect("the server's stderr"),
            log.clone(),
        );
        std::thread::spawn(move || {
            let (mut raw, mut buf) = (Vec::new(), [0u8; 4096]);
            while let Ok(n) = err.read(&mut buf) {
                if n == 0 {
                    break;
                }
                raw.extend_from_slice(&buf[..n]);
                while let Some(end) = raw.iter().position(|&b| b == b'\n') {
                    lines
                        .lock()
                        .unwrap()
                        .push(String::from_utf8_lossy(&raw[..end]).to_string());
                    raw.drain(..=end);
                }
            }
        });
        let s = Server {
            child,
            port,
            key,
            log,
        };
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
        let Ok(mut s) = TcpStream::connect(("127.0.0.1", self.port)) else {
            return false;
        };
        let mut raw = Vec::new();
        s.write_all(b"GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n")
            .is_ok()
            && s.read_to_end(&mut raw).is_ok()
            && raw.starts_with(b"HTTP/1.1 200")
    }

    /// A request and its whole answer; with `leave`, the client closes its
    /// connection that long after sending the request.
    fn post(&self, body: &Value, leave: Option<Duration>) -> (u16, Vec<u8>) {
        let Ok(mut s) = TcpStream::connect(("127.0.0.1", self.port)) else {
            return (0, Vec::new());
        };
        s.set_read_timeout(Some(Duration::from_secs(900))).ok();
        let payload = body.to_string();
        let head = format!(
            "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: Bearer {}\r\n\
             Content-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            self.key,
            payload.len()
        );
        if s.write_all(head.as_bytes())
            .and_then(|_| s.write_all(payload.as_bytes()))
            .is_err()
        {
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
        let Some(split) = raw.windows(4).position(|w| w == b"\r\n\r\n") else {
            return (0, raw);
        };
        let headers = String::from_utf8_lossy(&raw[..split]).to_lowercase();
        let status = headers
            .split_whitespace()
            .nth(1)
            .and_then(|c| c.parse().ok())
            .unwrap_or(0);
        let mut rest = raw[split + 4..].to_vec();
        if headers.contains("transfer-encoding: chunked") {
            let (mut out, mut at) = (Vec::new(), 0);
            while let Some(end) = rest[at..].windows(2).position(|w| w == b"\r\n") {
                let size =
                    usize::from_str_radix(String::from_utf8_lossy(&rest[at..at + end]).trim(), 16)
                        .unwrap_or(0);
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
        assert_eq!(
            status,
            200,
            "chat completion: {}; recent server log: {:?}",
            String::from_utf8_lossy(&answer),
            self.log
                .lock()
                .unwrap()
                .iter()
                .rev()
                .take(24)
                .collect::<Vec<_>>()
        );
        serde_json::from_slice(&answer).expect("a JSON answer")
    }

    fn get(&self, path: &str) -> Value {
        let mut s = TcpStream::connect(("127.0.0.1", self.port)).expect("connect");
        let head = format!(
            "GET {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: Bearer {}\r\nConnection: close\r\n\r\n",
            self.key
        );
        s.write_all(head.as_bytes()).expect("send");
        let mut raw = Vec::new();
        s.read_to_end(&mut raw).expect("read");
        let split = raw
            .windows(4)
            .position(|w| w == b"\r\n\r\n")
            .expect("headers");
        serde_json::from_slice(&raw[split + 4..]).expect("a JSON answer")
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = Command::new("kill")
            .args(["-TERM", &self.child.id().to_string()])
            .status();
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
    answer["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("")
        .to_string()
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
        (
            a["usage"]["prompt_tokens"].as_i64().unwrap(),
            s.chat(&body(second.clone(), 24, false)),
        )
    };
    let s = Server::start(&dir, &["--kv-cache", "12"]);
    s.post(&body(first, 24, true), Some(Duration::from_secs(2)));
    std::thread::sleep(Duration::from_secs(20)); // the server notices, and ends the request between passes
    let warm = s.chat(&body(second, 24, false));
    let (n, hit) = (
        warm["usage"]["prompt_tokens"].as_i64().unwrap(),
        warm["usage"]["prompt_cache_hit_tokens"].as_i64().unwrap(),
    );
    eprintln!(
        "reference: {:?}\nafter the cancelled prefill: {:?} ({hit} of {n} prompt tokens reused; the first prompt has {n1})",
        content(&reference),
        content(&warm)
    );
    assert!(
        hit < n1,
        "the second prompt reused {hit} tokens, the whole first prompt ({n1}): its prefill was not cut short"
    );
    assert_eq!(
        content(&warm),
        content(&reference),
        "the reply after a cancelled prefill"
    );
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
    let (n, made) = (
        answer["usage"]["prompt_tokens"].as_i64().unwrap(),
        answer["usage"]["completion_tokens"].as_i64().unwrap(),
    );
    eprintln!(
        "{n} prompt tokens, {made} completion tokens, finish {}",
        answer["choices"][0]["finish_reason"]
    );
    assert_eq!(answer["choices"][0]["finish_reason"], "length");
    assert_eq!(
        made,
        160 - n,
        "completion tokens of a reply that reaches the context's end"
    );
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
    for (ctx, depth) in [(160, 1), (161, 3), (162, 5)] {
        let s = Server::start(
            &dir,
            &["--ctx", &ctx.to_string(), "--draft", &depth.to_string()],
        );
        for max_tokens in [20, 21, 22, 23] {
            let answer = s.chat(&body(counting(), max_tokens, false));
            let made = answer["usage"]["completion_tokens"].as_i64().unwrap();
            eprintln!(
                "--ctx {ctx}, max_tokens {max_tokens}: {made} completion tokens, finish {}",
                answer["choices"][0]["finish_reason"]
            );
            assert_eq!(answer["choices"][0]["finish_reason"], "length");
            assert_eq!(
                made, max_tokens as i64,
                "completion tokens of a reply with max_tokens {max_tokens}"
            );
        }
        let answer = s.chat(&body(counting(), 1000, false));
        let (n, made) = (
            answer["usage"]["prompt_tokens"].as_i64().unwrap(),
            answer["usage"]["completion_tokens"].as_i64().unwrap(),
        );
        eprintln!(
            "--ctx {ctx}: {n} prompt tokens, {made} completion tokens, finish {}",
            answer["choices"][0]["finish_reason"]
        );
        assert_eq!(answer["choices"][0]["finish_reason"], "length");
        assert_eq!(
            made,
            ctx - n,
            "completion tokens of a reply that reaches the context's end"
        );
    }
}

/// DeepSeek and a Qwen model in one process: each answers, and while both
/// are idle, the log reports an expert cache that shrinks only when
/// DeepSeek's cache (the only one) really has fewer slots.
#[test]
fn deepseek_and_qwen_share_one_process() {
    let (Some(qwen), Ok(deepseek), Ok(experts)) = (
        checkpoint(),
        std::env::var("MYDS4_CHECKPOINT"),
        std::env::var("MYDS4_EXPERTS"),
    ) else {
        eprintln!("MYDS4_QWEN, MYDS4_CHECKPOINT or MYDS4_EXPERTS is not set: skipped");
        return;
    };
    let _one = one_model();
    let models = json!({"memory": 64, "kv_cache": 8, "reserve": 10, "models": [
        {"model": deepseek, "experts": experts},
        {"model": qwen, "ctx": 4096}
    ]});
    let s = Server::with_config(models);
    // the config's order: DeepSeek first
    let ids: Vec<String> = s.get("/v1/models")["data"]
        .as_array()
        .expect("models")
        .iter()
        .map(|m| m["id"].as_str().unwrap().to_string())
        .collect();
    assert_eq!(ids.len(), 2, "models: {ids:?}");
    for id in &ids {
        let mut b = body(
            json!([{"role": "user", "content": "Name three colors, in one line."}]),
            16,
            false,
        );
        b["model"] = json!(id);
        let answer = s.chat(&b);
        eprintln!("{id}: {:?}", content(&answer));
        assert!(!content(&answer).trim().is_empty(), "{id} gave no text");
    }
    std::thread::sleep(Duration::from_secs(4)); // the worker's idle calls run about once a second
    let slots = s.get(&format!("/v1/models/{}", ids[0]))["status"]["expert_cache"]["slots"]
        .as_i64()
        .expect("DeepSeek's expert cache slots");
    let lines: Vec<String> = s
        .log
        .lock()
        .unwrap()
        .iter()
        .filter(|l| l.contains("memory short"))
        .cloned()
        .collect();
    eprintln!(
        "DeepSeek's expert cache: {slots} slots; {} memory-short lines",
        lines.len()
    );
    if let Some(last) = lines.last() {
        let after: i64 = last
            .trim_end_matches(" slots")
            .rsplit(' ')
            .next()
            .and_then(|n| n.parse().ok())
            .expect("a slot count");
        assert_eq!(
            after, slots,
            "the log says the cache shrank to {after} slots ({last:?}); it has {slots}"
        );
    }
}

/// A single tight pool must evict idle states across model boundaries,
/// free them through the right model, and never match another model's KV.
#[test]
fn models_share_one_kv_budget_and_one_reserve() {
    let (Some(qwen), Ok(deepseek)) = (checkpoint(), std::env::var("MYDS4_CHECKPOINT")) else {
        eprintln!("MYDS4_QWEN or MYDS4_CHECKPOINT is not set: skipped");
        return;
    };
    let _one = one_model();
    let s = Server::with_config(
        json!({"memory": 48, "reserve": 13, "kv_cache": 0.75, "models": [
            {"model": deepseek, "id": "deepseek-budget",  "ctx": 4096, "batch": 4},
            {"model": qwen, "id": "qwen-budget", "batch": 8}
        ]}),
    );
    let models = s.get("/v1/models");
    let q = models["data"]
        .as_array()
        .unwrap()
        .iter()
        .find(|m| m["id"] == "qwen-budget")
        .unwrap();
    assert_eq!(
        q["context_length"], 262144,
        "Qwen defaults to its checkpoint's 256K context"
    );
    let ask = |model: &str, text: &str| {
        let mut b = body(json!([{"role": "user", "content": text}]), 16, false);
        b["model"] = json!(model);
        let a = s.chat(&b);
        assert!(!content(&a).trim().is_empty());
        a
    };
    ask("deepseek-budget", "Name three colors, in one line.");
    ask("qwen-budget", "Give three prime numbers, in one line.");
    for id in ["deepseek-budget", "qwen-budget"] {
        let st = &s.get(&format!("/v1/models/{id}"))["status"];
        assert_eq!(st["reserve_gib"], 13.0);
        assert_eq!(st["kv_cache_limit_gib"], 0.75);
        assert_eq!(st["kv_cache_scope"], "server");
        assert!(st["kv_cache_total_gib"].as_f64().unwrap() <= 0.751);
        assert_eq!(st["model_memory"].as_array().unwrap().len(), 2);
    }
    // A fresh Qwen state and its snapshots cannot coexist with all the
    // old allocations under this budget. DeepSeek's oldest idle state goes.
    ask(
        "qwen-budget",
        "Write one Python expression that reverses a list.",
    );
    let ds = s.get("/v1/models/deepseek-budget");
    assert!(
        ds["status"]["sessions"].as_array().unwrap().is_empty(),
        "old DeepSeek state was not evicted: {ds}"
    );
    assert!(ds["status"]["kv_cache_total_gib"].as_f64().unwrap() <= 0.751);
    let again = ask("deepseek-budget", "Name three colors, in one line.");
    assert_eq!(
        again["usage"]["prompt_cache_hit_tokens"], 0,
        "evicted rows cannot be reused"
    );
    let ds = s.get("/v1/models/deepseek-budget");
    let qw = s.get("/v1/models/qwen-budget");
    let total = ds["status"]["kv_cache_total_gib"].as_f64().unwrap();
    let parts = ds["status"]["kv_cache_gib"].as_f64().unwrap()
        + qw["status"]["kv_cache_gib"].as_f64().unwrap();
    assert!((total - parts).abs() <= 0.002 && total <= 0.751);
    eprintln!("shared KV: {total:.3} GiB of 0.75; reserve 13 GiB; cross-model eviction passed");
}

#[test]
fn registered_models_have_no_loaded_weights_before_inference() {
    let (Some(qwen), Ok(deepseek)) = (checkpoint(), std::env::var("MYDS4_CHECKPOINT")) else {
        return;
    };
    let _one = one_model();
    let s = Server::with_config(
        json!({"memory": 48, "reserve": 10, "kv_cache": 2, "models": [
            {"model": deepseek, "id": "deepseek-idle",  "ctx": 4096},
            {"model": qwen, "id": "qwen-idle", "ctx": 4096}
        ]}),
    );
    let models = s.get("/v1/models");
    assert_eq!(models["data"].as_array().unwrap().len(), 2);
    for id in ["deepseek-idle", "qwen-idle"] {
        let status = s.get(&format!("/v1/models/{id}"));
        for model in status["status"]["model_memory"].as_array().unwrap() {
            assert_eq!(model["weights"], 0, "startup loaded weights: {model}");
            assert_eq!(
                model["experts"], 0,
                "startup allocated expert weights: {model}"
            );
        }
    }
}

#[test]
fn a_development_lock_does_not_block_normal_server_requests() {
    use std::os::fd::AsRawFd;
    let Some(directory) = checkpoint() else {
        return;
    };
    let _one = one_model();
    let development = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(root().join("build/myds4.lock"))
        .unwrap();
    assert_eq!(
        unsafe { libc::flock(development.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) },
        0
    );
    let server = Server::start(&directory, &["--memory", "32", "--draft", "0"]);
    let response = std::thread::scope(|scope| {
        let (send, receive) = std::sync::mpsc::channel();
        let server = &server;
        scope.spawn(move || {
            let answer = server.post(
                &body(json!([{"role":"user","content":"hello"}]), 1, false),
                None,
            );
            let _ = send.send(answer);
        });
        let answer = receive.recv_timeout(Duration::from_secs(8));
        drop(development);
        answer
    });
    assert!(
        response.is_ok(),
        "the server waited for the development lock: {:?}",
        server.log.lock().unwrap()
    );
    assert_eq!(response.unwrap().0, 200);
    assert_eq!(server.get("/health")["status"], "ok");
}

#[test]
fn unavailable_weights_do_not_stop_the_server() {
    let _one = one_model();
    let path = root().join("server/tests/fixtures/unavailable-model");
    let server = Server::start(path.to_str().unwrap(), &[]);
    let (status, answer) = server.post(
        &body(json!([{"role": "user", "content": "hello"}]), 1, false),
        None,
    );
    assert_eq!(
        status,
        503,
        "unavailable model response: {}",
        String::from_utf8_lossy(&answer)
    );
    assert_eq!(server.get("/health")["status"], "ok");
    assert_eq!(
        server.get("/v1/models")["data"].as_array().unwrap().len(),
        1
    );
}

#[test]
fn first_inference_loads_only_the_selected_model() {
    let (Some(qwen), Ok(deepseek)) = (checkpoint(), std::env::var("MYDS4_CHECKPOINT")) else {
        return;
    };
    let _one = one_model();
    let s = Server::with_config(
        json!({"memory": 48, "reserve": 10, "kv_cache": 2, "models": [
            {"model": deepseek, "id": "deepseek-lazy",  "ctx": 4096},
            {"model": qwen, "id": "qwen-lazy", "ctx": 4096, "draft": 1}
        ]}),
    );
    for (id, text) in [
        ("qwen-lazy", "Name one fruit."),
        ("deepseek-lazy", "Name one color."),
    ] {
        let mut request = body(json!([{"role": "user", "content": text}]), 8, false);
        request["model"] = json!(id);
        assert!(!content(&s.chat(&request)).trim().is_empty());
        let status = s.get(&format!("/v1/models/{id}"));
        assert_eq!(status["status"]["loaded"], true);
        if id == "qwen-lazy" {
            let other = s.get("/v1/models/deepseek-lazy");
            assert_eq!(other["status"]["loaded"], false);
            let models = other["status"]["model_memory"].as_array().unwrap();
            assert_eq!(
                models
                    .iter()
                    .find(|m| m["model"] == "deepseek-lazy")
                    .unwrap()["weights"],
                0
            );
        }
    }
    let mut image_request = body(
        json!([{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAQAAAAECAIAAAAmkwkpAAAAEElEQVR4nGP4z8AARwzEcQCukw/x0F8jngAAAABJRU5ErkJggg=="}},
            {"type": "text", "text": "What color fills this image? Reply with that color name only."}
        ]}]),
        16,
        false,
    );
    image_request["model"] = json!("deepseek-lazy");
    let answer = s.chat(&image_request);
    assert!(
        content(&answer).to_lowercase().contains("red"),
        "deferred DeepSeek vision after Qwen loaded: {}",
        content(&answer)
    );
}

#[test]
fn vision_distinguishes_images_and_reuses_a_conversation_with_mtp() {
    let Some(dir) = checkpoint() else { return };
    let _one = one_model();
    let server = Server::start(&dir, &["--ctx", "4096", "--draft", "1", "--kv-cache", "2"]);
    let red = "iVBORw0KGgoAAAANSUhEUgAAAAQAAAAECAIAAAAmkwkpAAAAEElEQVR4nGP4z8AARwzEcQCukw/x0F8jngAAAABJRU5ErkJggg==";
    let blue = "iVBORw0KGgoAAAANSUhEUgAAAAQAAAAECAIAAAAmkwkpAAAAEElEQVR4nGNgYPiPhIjiAACOsw/xs6MvMwAAAABJRU5ErkJggg==";
    let message = |image: &str| {
        json!({"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": format!("data:image/png;base64,{image}")}},
            {"type": "text", "text": "What single color fills this image? Reply with the color name only."}
        ]})
    };
    let red_message = message(red);
    let first = server.chat(&body(json!([red_message.clone()]), 16, false));
    let answer = content(&first);
    assert!(answer.to_lowercase().contains("red"), "red image: {answer}");
    let followup = server.chat(&body(
        json!([
            red_message,
            {"role": "assistant", "content": answer},
            {"role": "user", "content": "Repeat the color name."}
        ]),
        16,
        false,
    ));
    assert!(
        content(&followup).to_lowercase().contains("red"),
        "follow-up: {}",
        content(&followup)
    );
    assert!(
        followup["usage"]["prompt_cache_hit_tokens"]
            .as_u64()
            .unwrap_or(0)
            > 0
    );
    let other = server.chat(&body(json!([message(blue)]), 16, false));
    assert!(
        content(&other).to_lowercase().contains("blue"),
        "blue image: {}",
        content(&other)
    );

    // Chunked prefills must keep each queued request's own image embeddings.
    let ready = std::sync::Barrier::new(2);
    std::thread::scope(|scope| {
        let asks: Vec<_> = [(red, "red"), (blue, "blue")].into_iter().map(|(image, color)| {
            let server = &server;
            let ready = &ready;
            scope.spawn(move || {
                let message = json!({"role": "user", "content": [
                    {"type": "text", "text": "Ignore these filler labels: ".to_owned() + &"label ".repeat(1400)},
                    {"type": "image_url", "image_url": {"url": format!("data:image/png;base64,{image}")}},
                    {"type": "text", "text": "What single color fills the image? Reply with that color name only."}
                ]});
                ready.wait();
                let reply = server.chat(&body(json!([message]), 16, false));
                assert!(content(&reply).to_lowercase().contains(color), "queued {color} image: {}", content(&reply));
            })
        }).collect();
        for ask in asks {
            ask.join().unwrap();
        }
    });
}

#[test]
fn flash_images_and_reused_state() {
    let Ok(dir) = std::env::var("MYDS4_FLASH") else {
        return;
    };
    let _one = one_model();
    let server = Server::start(
        &dir,
        &[
            "--ctx",
            "4096",
            "--draft",
            "0",
            "--memory",
            "32",
            "--kv-cache",
            "2",
        ],
    );
    let red = "iVBORw0KGgoAAAANSUhEUgAAAAQAAAAECAIAAAAmkwkpAAAAEElEQVR4nGP4z8AARwzEcQCukw/x0F8jngAAAABJRU5ErkJggg==";
    let blue = "iVBORw0KGgoAAAANSUhEUgAAAAQAAAAECAIAAAAmkwkpAAAAEElEQVR4nGNgYPiPhIjiAACOsw/xs6MvMwAAAABJRU5ErkJggg==";
    for (image, color) in [(red, "red"), (blue, "blue")] {
        let message = json!({"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": format!("data:image/png;base64,{image}")}},
            {"type": "text", "text": "What single color fills this image? Reply with the color name only."}
        ]});
        let reply = server.chat(&body(json!([message.clone()]), 16, false));
        let answer = content(&reply);
        assert!(
            answer.to_lowercase().contains(color),
            "{color} image: {answer}"
        );
        let followup = server.chat(&body(
            json!([
                message, {"role": "assistant", "content": answer},
                {"role": "user", "content": "Repeat the color name."}
            ]),
            16,
            false,
        ));
        assert!(content(&followup).to_lowercase().contains(color));
        assert!(
            followup["usage"]["prompt_cache_hit_tokens"]
                .as_u64()
                .unwrap_or(0)
                > 0
        );
    }
}

#[test]
fn nemotron_and_qwen_reload_under_one_memory_limit() {
    let (Ok(nemotron), Ok(draft), Some(qwen)) = (
        std::env::var("MYDS4_NEMOTRON"),
        std::env::var("MYDS4_DFLASH"),
        checkpoint(),
    ) else {
        return;
    };
    let _one = one_model();
    let mut models = vec![
        json!({"id":"transcribe","model":nemotron,"draft":7,"draft_model":draft,"ctx":4096}),
        json!({"id":"qwen","model":qwen,"draft":0,"ctx":4096}),
    ];
    if let Ok(directory) = std::env::var("MYDS4_FLASH") {
        models.push(json!({"id":"flash","model":directory,"draft":0,"ctx":4096}));
    }
    let server =
        Server::with_config(json!({"memory":22,"reserve":10,"kv_cache":6,"models":models}));
    let mut request = body(
        json!([{"role":"user","content":"Name two colors."}]),
        24,
        false,
    );
    request["model"] = json!("transcribe");
    let first = server.chat(&request);
    let mut other = body(
        json!([{"role":"user","content":"What is two plus three? Give just the number."}]),
        8,
        false,
    );
    other["model"] = json!("qwen");
    assert!(content(&server.chat(&other)).contains('5'));
    let again = server.chat(&request);
    assert_eq!(content(&first), content(&again));
    assert!(
        again["usage"]["prompt_cache_hit_tokens"]
            .as_u64()
            .unwrap_or(0)
            > 0
    );
    std::thread::sleep(Duration::from_millis(350));
    let status = server.get("/v1/models/transcribe");
    let budget = &status["status"]["model_memory_budget"];
    assert_eq!(budget["limit"].as_u64(), Some(22 << 30));
    assert!(budget["allocated"].as_u64().unwrap() <= 22 << 30);
}

#[test]
fn transcribe_audio_reuses_its_prefix_and_continues_after_a_fork() {
    let (Ok(directory), Ok(fixture)) = (
        std::env::var("MYDS4_NEMOTRON"),
        std::env::var("MYDS4_AUDIO_REQUEST"),
    ) else {
        return;
    };
    let _one = one_model();
    let mut request: Value = serde_json::from_slice(&std::fs::read(fixture).unwrap()).unwrap();
    assert!(
        request["messages"]
            .as_array()
            .unwrap()
            .iter()
            .any(|message| {
                message["content"]
                    .as_array()
                    .is_some_and(|parts| parts.iter().any(|part| part["type"] == "input_audio"))
            })
    );
    request["temperature"] = json!(0);
    request["top_p"] = json!(1);
    request["top_k"] = json!(0);
    request["max_tokens"] = json!(256);
    request["reasoning_effort"] = json!("none");
    request["stream"] = json!(false);
    let draft = std::env::var("MYDS4_DFLASH").ok();
    let mut transcript = None;
    for depth in if draft.is_some() { vec![0, 7] } else { vec![0] } {
        let mut options = vec![
            "--memory",
            "28",
            "--kv-cache",
            "6",
            "--ctx",
            "4096",
            "--draft",
        ];
        options.push(if depth == 0 { "0" } else { "7" });
        if depth != 0 {
            options.extend(["--draft-model", draft.as_deref().unwrap()]);
        }
        let (continued, warm) = {
            let server = Server::start(&directory, &options);
            request["model"] = server.get("/v1/models")["data"][0]["id"].clone();
            let first = server.chat(&request);
            let answer = content(&first);
            assert!(!answer.trim().is_empty(), "empty transcript: {first}");
            if let Some(reference) = &transcript {
                assert_eq!(&answer, reference, "DFlash changed the greedy transcript");
            } else {
                eprintln!("audio transcript: {answer}");
                transcript = Some(answer.clone());
            }
            let repeated = server.chat(&request);
            assert_eq!(content(&repeated), answer);
            assert!(
                repeated["usage"]["prompt_cache_hit_tokens"]
                    .as_u64()
                    .unwrap()
                    > 0
            );
            let mut continued = request.clone();
            continued["messages"].as_array_mut().unwrap().extend([
                json!({"role":"assistant","content":answer}),
                json!({"role":"user","content":"Repeat the transcript exactly."}),
            ]);
            let warm = server.chat(&continued);
            assert!(warm["usage"]["prompt_cache_hit_tokens"].as_u64().unwrap() > 0);
            eprintln!(
                "audio draft={depth}: repeated {} reused; continuation {} reused",
                repeated["usage"]["prompt_cache_hit_tokens"],
                warm["usage"]["prompt_cache_hit_tokens"]
            );
            (continued, warm)
        };
        let cold = Server::start(&directory, &options).chat(&continued);
        assert_eq!(
            content(&warm),
            content(&cold),
            "audio continuation changed with cached history"
        );
    }
}

#[test]
fn disabled_drafts_leave_mtp_unloaded() {
    let Some(dir) = checkpoint() else { return };
    let _one = one_model();
    let server = Server::start(&dir, &["--ctx", "1024", "--draft", "0", "--kv-cache", "1"]);
    server.chat(&body(
        json!([{"role": "user", "content": "Name one color."}]),
        4,
        false,
    ));
    let loaded = server
        .log
        .lock()
        .unwrap()
        .iter()
        .any(|line| line.contains(", MTP head"));
    assert!(!loaded, "draft=0 still loaded and initialized the MTP head");
}
