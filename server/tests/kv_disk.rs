//! States saved on the SSD through HTTP (--kv-disk, --kv-dir,
//! --kv-disk-tokens): a state evicted from memory, or idle at quit, is
//! written to the SSD, and a later prompt that starts like it continues from
//! it as it would from memory, byte for byte in its greedy replies. A saved
//! state damaged after the start is refused when it is read: the prompt is
//! prefilled cold and the saved state goes. Saved states stay within their
//! budget, the most recently used kept. A second server on the same folder
//! runs without them.
//!
//! Set MYDS4_CHECKPOINT to the DeepSeek checkpoint directory and MYDS4_EXPERTS
//! optionally to GGUF experts. The tests run one model process at a time, in
//! temporary folders they remove:
//!
//!     MYDS4_CHECKPOINT=DIR MYDS4_EXPERTS=GGUF cargo test --release \
//!       --manifest-path server/Cargo.toml --test kv_disk -- --nocapture
//!
//! MYDS4_SERVER picks the server binary (default build/myds4-server).

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use serde_json::{Value, json};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("the repository")
        .to_path_buf()
}

/// Held while a test runs servers: one model process at a time.
static MODEL: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn one_model() -> std::sync::MutexGuard<'static, ()> {
    MODEL.lock().unwrap_or_else(|e| e.into_inner())
}

struct Setup {
    checkpoint: String,
    experts: Option<String>,
    server: PathBuf,
}

fn setup() -> Option<Setup> {
    let checkpoint = std::env::var("MYDS4_CHECKPOINT").ok()?;
    Some(Setup {
        checkpoint,
        experts: std::env::var("MYDS4_EXPERTS").ok(),
        server: std::env::var("MYDS4_SERVER")
            .map(PathBuf::from)
            .unwrap_or_else(|_| root().join("build/myds4-server")),
    })
}

/// A temporary folder for a test's saved states, removed when dropped.
struct Folder(PathBuf);

impl Folder {
    fn new(name: &str) -> Folder {
        let dir = std::env::temp_dir().join(format!("myds4-kv-disk-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        Folder(dir)
    }

    fn path(&self) -> &str {
        self.0.to_str().expect("a UTF-8 path")
    }

    /// The files in it whose names end in `suffix`.
    fn files(&self, suffix: &str) -> Vec<PathBuf> {
        std::fs::read_dir(&self.0)
            .map(|d| {
                d.flatten()
                    .map(|e| e.path())
                    .filter(|p| p.to_string_lossy().ends_with(suffix))
                    .collect()
            })
            .unwrap_or_default()
    }
}

impl Drop for Folder {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

// ---- HTTP/1.1, enough for the server's JSON answers ----

fn http(port: u16, key: &str, method: &str, path: &str, body: Option<&Value>) -> (u16, Vec<u8>) {
    let mut s = TcpStream::connect(("127.0.0.1", port)).expect("connect");
    s.set_read_timeout(Some(Duration::from_secs(1800))).ok();
    let payload = body.map(|b| b.to_string()).unwrap_or_default();
    let head = format!(
        "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nAuthorization: Bearer {key}\r\nContent-Type: application/json\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n",
        payload.len()
    );
    s.write_all(head.as_bytes())
        .and_then(|_| s.write_all(payload.as_bytes()))
        .expect("send");
    let mut raw = Vec::new();
    s.read_to_end(&mut raw).expect("read");
    let split = raw
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .expect("an HTTP answer");
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

struct Server {
    child: Child,
    port: u16,
    key: String,
}

impl Server {
    /// A server of the checkpoint with saved states in `dir` (none without),
    /// written for 256 tokens of their own; `extra` last.
    fn start(s: &Setup, dir: Option<&Folder>, extra: &[&str]) -> Server {
        let port = TcpListener::bind("127.0.0.1:0")
            .and_then(|l| l.local_addr())
            .expect("a free port")
            .port();
        let key = format!("kv-disk-test-{}-{port}-key", std::process::id());
        let mut args: Vec<String> = ["-m", &s.checkpoint, "--memory", "56", "--ctx", "32768", "--batch", "1"]
            .map(String::from)
            .to_vec();
        if let Some(experts) = &s.experts {
            args.extend(["--experts".to_string(), experts.clone()]);
        }
        match dir {
            Some(d) => args.extend(["--kv-dir", d.path(), "--kv-disk-tokens", "256"].map(String::from)),
            None => args.extend(["--kv-disk", "0"].map(String::from)),
        }
        args.extend(["--serve".to_string(), port.to_string()]);
        args.extend(extra.iter().map(|a| a.to_string()));
        let child = Command::new(&s.server)
            .args(&args)
            .env("MYDS4_API_KEY", &key)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("start the server");
        let server = Server { child, port, key };
        let deadline = Instant::now() + Duration::from_secs(900);
        while Instant::now() < deadline {
            if TcpStream::connect(("127.0.0.1", port)).is_ok() && server.get("/health").0 == 200 {
                return server;
            }
            std::thread::sleep(Duration::from_millis(500));
        }
        panic!("the server did not start");
    }

    fn get(&self, path: &str) -> (u16, Vec<u8>) {
        http(self.port, &self.key, "GET", path, None)
    }

    /// The served model's status: GET /v1/models/{id}'s "status".
    fn status(&self) -> Value {
        let models: Value = serde_json::from_slice(&self.get("/v1/models").1).unwrap_or(Value::Null);
        let id = models["data"][0]["id"].as_str().unwrap_or("").to_string();
        serde_json::from_slice::<Value>(&self.get(&format!("/v1/models/{id}")).1)
            .map(|v| v["status"].clone())
            .unwrap_or(Value::Null)
    }

    /// The status once `requests` requests have ended and the server is idle.
    fn settled(&self, requests: i64) -> Value {
        let deadline = Instant::now() + Duration::from_secs(60);
        loop {
            let v = self.status();
            let done = v["phase"] == "idle" && v["totals"]["requests"].as_i64().unwrap_or(0) >= requests;
            if done || Instant::now() > deadline {
                return v;
            }
            std::thread::sleep(Duration::from_millis(200));
        }
    }

    fn chat(&self, messages: &Value, max_tokens: u32) -> Value {
        let body = json!({"model": "deepseek-v4.1-flash", "messages": messages, "max_tokens": max_tokens,
                          "temperature": 0, "thinking": false, "stream": false});
        let (status, answer) = http(self.port, &self.key, "POST", "/v1/chat/completions", Some(&body));
        assert_eq!(status, 200, "chat completion: {}", String::from_utf8_lossy(&answer));
        serde_json::from_slice(&answer).unwrap_or(Value::Null)
    }

    /// Stops it as SIGTERM does and waits for it to exit (it saves its
    /// states first).
    fn stop(mut self) {
        let _ = Command::new("kill")
            .args(["-TERM", &self.child.id().to_string()])
            .status();
        let deadline = Instant::now() + Duration::from_secs(120);
        while Instant::now() < deadline {
            if let Ok(Some(_)) = self.child.try_wait() {
                return;
            }
            std::thread::sleep(Duration::from_millis(200));
        }
        panic!("the server did not stop within two minutes");
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        if let Ok(Some(_)) = self.child.try_wait() {
            return;
        }
        let _ = Command::new("kill")
            .args(["-TERM", &self.child.id().to_string()])
            .stderr(Stdio::null())
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

fn hit(answer: &Value) -> i64 {
    answer["usage"]["prompt_cache_hit_tokens"].as_i64().unwrap_or(-1)
}

fn prompt_tokens(answer: &Value) -> i64 {
    answer["usage"]["prompt_tokens"].as_i64().unwrap_or(-1)
}

fn content(answer: &Value) -> String {
    answer["choices"][0]["message"]["content"].as_str().unwrap_or("").to_string()
}

/// A conversation of about `chars` characters of document `doc` (docs/) as
/// its system prompt.
fn conversation(doc: &str, chars: usize, question: &str) -> Value {
    let text = std::fs::read_to_string(root().join("docs").join(doc)).expect("a document");
    let cut: String = text.chars().take(chars).collect();
    json!([{"role": "system", "content": format!("Answer from this reference.\n\n{cut}")},
           {"role": "user", "content": question}])
}

/// The conversation continued: the reply, and one more question.
fn continued(messages: &Value, answer: &Value, question: &str) -> Value {
    let mut m = messages.as_array().expect("messages").clone();
    m.push(json!({"role": "assistant", "content": content(answer)}));
    m.push(json!({"role": "user", "content": question}));
    Value::Array(m)
}

fn disk(status: &Value) -> (i64, i64, i64) {
    let d = &status["kv_disk"];
    let n = |k: &str| d[k].as_i64().unwrap_or(-1);
    (n("states"), n("saves"), n("loads"))
}

/// A state evicted for another one is saved; its conversation's next turn
/// reads it back and continues as the state kept in memory does.
#[test]
fn an_evicted_state_continues_from_the_ssd_as_from_memory() {
    let Some(s) = setup() else {
        eprintln!("skipped: set MYDS4_CHECKPOINT (and MYDS4_EXPERTS)");
        return;
    };
    let _one = one_model();
    let a = conversation("design.md", 6000, "Which parts of the engine run on the GPU?");
    let b = conversation("usage.md", 6000, "How is the server configured?");
    // the reference: the state stays in memory
    let (first, next, then, room) = {
        let server = Server::start(&s, None, &[]);
        let first = server.chat(&a, 24);
        let room = server.settled(1)["kv_cache_gib"].as_f64().expect("the cache's size");
        let next = continued(&a, &first, "And which run on the CPU?");
        let then = server.chat(&next, 24);
        (first, next, then, room)
    };
    // room for one such state: B's evicts A's, and A's next turn evicts B's
    let folder = Folder::new("evict");
    let limit = format!("{:.4}", room * 1.5);
    let server = Server::start(&s, Some(&folder), &["--kv-cache", &limit]);
    let first_again = server.chat(&a, 24);
    assert_eq!(content(&first_again), content(&first), "the first turn as on the reference server");
    server.chat(&b, 8);
    let after_b = server.settled(2);
    let then_again = server.chat(&next, 24);
    let end = server.settled(3);
    println!(
        "kv-cache {limit} GiB: after B {:?} (states, saves, loads); A's next turn reused {} of {} tokens (in memory: {}); {:?}",
        disk(&after_b),
        hit(&then_again),
        prompt_tokens(&then_again),
        hit(&then),
        disk(&end)
    );
    assert!(disk(&after_b).1 >= 1, "A's state was not saved when B's evicted it: {:?}", disk(&after_b));
    assert!(disk(&end).2 >= 1, "A's next turn did not read the saved state: {:?}", disk(&end));
    assert_eq!(hit(&then_again), hit(&then), "the saved state resumes where the state in memory does");
    assert_eq!(content(&then_again), content(&then), "the reply continues as from memory");
}

/// The states idle at quit are saved; after a restart the same prompt is
/// answered from its saved prompt end, without a prefill.
#[test]
fn a_state_saved_at_quit_answers_its_prompt_again_after_a_restart() {
    let Some(s) = setup() else {
        eprintln!("skipped: set MYDS4_CHECKPOINT (and MYDS4_EXPERTS)");
        return;
    };
    let _one = one_model();
    let folder = Folder::new("quit");
    let a = conversation("testing.md", 6000, "Which checks need a model?");
    let first = {
        let server = Server::start(&s, Some(&folder), &[]);
        let first = server.chat(&a, 24);
        server.settled(1);
        server.stop();
        first
    };
    assert_eq!(folder.files(".kv").len(), 1, "one saved state after the quit");
    let server = Server::start(&s, Some(&folder), &[]);
    let start = server.status();
    let again = server.chat(&a, 24);
    let end = server.settled(1);
    println!(
        "after the restart: {:?} (states, saves, loads); the prompt again reused {} of {} tokens; last prefill {}",
        disk(&start),
        hit(&again),
        prompt_tokens(&again),
        end["last_prefill"]["tokens"]
    );
    assert_eq!(disk(&start).0, 1, "the restarted server reads the saved state's index");
    assert_eq!(hit(&again), prompt_tokens(&again), "the whole prompt from the saved prompt end");
    assert_eq!(content(&again), content(&first), "the same reply as before the quit");
    assert_eq!(disk(&end).2, 1);
}

/// A part damaged after the start is refused when it is read: the prompt is
/// prefilled cold, gives the cold reply, and the saved state is deleted.
#[test]
fn a_saved_state_damaged_after_the_start_is_refused_when_read() {
    let Some(s) = setup() else {
        eprintln!("skipped: set MYDS4_CHECKPOINT (and MYDS4_EXPERTS)");
        return;
    };
    let _one = one_model();
    let folder = Folder::new("damage");
    let a = conversation("design.md", 5000, "What does the expert store do?");
    let first = {
        let server = Server::start(&s, Some(&folder), &[]);
        let first = server.chat(&a, 24);
        server.settled(1);
        server.stop();
        first
    };
    let server = Server::start(&s, Some(&folder), &[]);
    assert_eq!(disk(&server.status()).0, 1);
    for part in folder.files(".rows") {
        let mut bytes = std::fs::read(&part).expect("a part");
        bytes[16384 + 100] ^= 0x40; // in its first region's rows
        std::fs::write(&part, bytes).expect("damaged");
    }
    let again = server.chat(&a, 24);
    let end = server.settled(1);
    println!("damaged: reused {} of {}; {:?}", hit(&again), prompt_tokens(&again), disk(&end));
    assert_eq!(hit(&again), 0, "a cold prefill");
    assert_eq!(content(&again), content(&first), "the cold reply");
    assert_eq!(disk(&end).0, 0, "the damaged saved state is deleted");
    assert!(folder.files(".kv").is_empty());
}

/// Saved states stay within --kv-disk: at quit the most recently used are
/// written first, and an older one does not take their room.
#[test]
fn saved_states_stay_within_their_budget_most_recent_first() {
    let Some(s) = setup() else {
        eprintln!("skipped: set MYDS4_CHECKPOINT (and MYDS4_EXPERTS)");
        return;
    };
    let _one = one_model();
    let folder = Folder::new("budget");
    let prompts = [
        conversation("design.md", 4000, "Summarize the first section."),
        conversation("usage.md", 4000, "Summarize the first section."),
        conversation("testing.md", 4000, "Summarize the first section."),
    ];
    // What each saved state takes, all three written. They differ: the second
    // conversation also keeps a snapshot where it parts from the first. At
    // quit the most recently used is written first, so its number is the
    // lowest.
    let sizes: Vec<f64> = {
        let server = Server::start(&s, Some(&folder), &[]);
        for p in &prompts {
            server.chat(p, 1);
        }
        server.settled(3);
        server.stop();
        let mut bytes = std::collections::BTreeMap::<u64, u64>::new();
        for e in std::fs::read_dir(&folder.0).expect("the folder").flatten() {
            let name = e.file_name().to_string_lossy().into_owned();
            if let Some(id) = name.split(['-', '.']).next().and_then(|n| n.parse().ok()) {
                *bytes.entry(id).or_insert(0) += e.metadata().map_or(0, |m| m.len());
            }
        }
        let _ = std::fs::remove_dir_all(&folder.0);
        bytes.values().map(|&b| b as f64 / (1u64 << 30) as f64).collect()
    };
    assert_eq!(sizes.len(), 3, "all three saved: {sizes:?}");
    let limit = format!("{:.6}", sizes[0] + sizes[1] + sizes[2] / 2.0); // room for the two most recent
    {
        let server = Server::start(&s, Some(&folder), &["--kv-disk", &limit]);
        for p in &prompts {
            server.chat(p, 1);
        }
        server.settled(3);
        server.stop();
    }
    let server = Server::start(&s, Some(&folder), &["--kv-disk", &limit]);
    let start = server.status();
    let hits: Vec<i64> = prompts.iter().map(|p| hit(&server.chat(p, 1))).collect();
    println!("saved states of {sizes:.4?} GiB, --kv-disk {limit}: {:?} at the start; the prompts again reused {hits:?}", disk(&start));
    assert_eq!(disk(&start).0, 2, "two saved states fit");
    let gib = start["kv_disk"]["gib"].as_f64().unwrap_or(f64::NAN);
    assert!(gib <= start["kv_disk"]["limit_gib"].as_f64().unwrap_or(0.0), "{gib} GiB");
    // Each state's own tokens are nearly all of its prompt; the prompts share
    // only their first few tokens with each other.
    assert!(hits[1] >= 256 && hits[2] >= 256, "the two most recent are kept: {hits:?}");
    assert!(hits[0] < 256, "the oldest is not: {hits:?}");
}

/// One server uses a folder at a time: a second one there runs without
/// saved states. Neither loads its weights (no request runs).
#[test]
fn a_second_server_on_the_same_folder_runs_without_saved_states() {
    let Some(s) = setup() else {
        eprintln!("skipped: set MYDS4_CHECKPOINT (and MYDS4_EXPERTS)");
        return;
    };
    let _one = one_model();
    let folder = Folder::new("lock");
    let first = Server::start(&s, Some(&folder), &[]);
    let second = Server::start(&s, Some(&folder), &[]);
    assert!(first.status()["kv_disk"].is_object(), "the first server has its saved states");
    assert!(second.status()["kv_disk"].is_null(), "the second runs without them");
}
