//! The KV prefix cache through HTTP: the fork test's server
//! part, the cache's budget (--kv-cache), and a session kept while its idle
//! model unloads (--idle-unload). Forks of
//! a conversation reuse its history up to where they part from it, also while
//! the conversation is still decoding. The cache resumes each fork the
//! cheaper way: a replay at the shared prefix, or a snapshot before it that
//! leaves fewer rows to compute. So a fork reuses at least its shared prefix
//! less what a replay there computes again, and never more than the prefix.
//! Without --ced, each fork must give the tokens that the same prompt gives a
//! server that starts without a cache. With --ced, forks are approximate by
//! design: the replay is bounded to a ring, and their tokens are printed.
//!
//! Set MYDS4_CHECKPOINT to the checkpoint directory. MYDS4_EXPERTS optionally
//! selects GGUF experts; otherwise the tests use native MXFP4. The tests run
//! one model process at a time (they share a lock):
//!
//!     MYDS4_CHECKPOINT=DIR MYDS4_EXPERTS=GGUF cargo test --release \
//!       --manifest-path server/Cargo.toml --test prefix_fork -- --nocapture
//!
//! MYDS4_SERVER picks the server binary (default build/myds4-server) and
//! MYDS4_CLI the tokenizer's (default build/myds4). It prints every fork's
//! shared and reported reuse and fails if any lies outside those bounds.

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

/// At least the rows that an exact replay at k computes again, on average per
/// layer (the cache's cost of a replay, myds4_replay_rows): layer 39 starts a
/// ring (136 positions: the window and the rewind margin of 8, with the
/// default --lookup) before k, and each layer below at most 128 positions
/// earlier (127, plus a ratio-2 compressor's pair).
fn replay_rows(k: usize) -> usize {
    (0..40)
        .map(|j| k.min(137 + 128 * j))
        .sum::<usize>()
        .div_ceil(40)
}

/// What a bounded replay at k computes again (--ced): a ring's positions.
fn bounded_rows(k: usize) -> usize {
    k.min(136)
}

struct Setup {
    checkpoint: String,
    experts: Option<String>,
    server: PathBuf,
    cli: PathBuf,
}

fn setup() -> Option<Setup> {
    let checkpoint = std::env::var("MYDS4_CHECKPOINT").ok()?;
    let experts = std::env::var("MYDS4_EXPERTS").ok();
    let pick = |var: &str, default: &str| {
        std::env::var(var)
            .map(PathBuf::from)
            .unwrap_or_else(|_| root().join(default))
    };
    Some(Setup {
        checkpoint,
        experts,
        server: pick("MYDS4_SERVER", "build/myds4-server"),
        cli: pick("MYDS4_CLI", "build/myds4"),
    })
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
    fn start(s: &Setup, extra: &[&str]) -> Server {
        let port = TcpListener::bind("127.0.0.1:0")
            .and_then(|l| l.local_addr())
            .expect("a free port")
            .port();
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        let key = format!(
            "fork-test-{}-{:x}",
            std::process::id(),
            nanos ^ (port as u128) << 64
        );
        let mut args: Vec<String> = ["-m", &s.checkpoint, "--memory", "56", "--kv-cache", "8"]
            .iter()
            .map(|a| a.to_string())
            .collect();
        if let Some(experts) = &s.experts {
            args.extend(["--experts".to_string(), experts.clone()]);
        }
        args.extend(
            [
                "--ctx",
                "32768",
                "--serve",
                &port.to_string(),
                "--batch",
                "2",
            ]
            .map(String::from),
        );
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
    fn status(&self) -> (u16, Value) {
        let models: Value =
            serde_json::from_slice(&self.get("/v1/models").1).unwrap_or(Value::Null);
        let id = models["data"][0]["id"].as_str().unwrap_or("").to_string();
        let (code, body) = self.get(&format!("/v1/models/{id}"));
        (
            code,
            serde_json::from_slice::<Value>(&body)
                .map(|v| v["status"].clone())
                .unwrap_or(Value::Null),
        )
    }

    fn chat(&self, body: &Value) -> Value {
        let (status, answer) = http(
            self.port,
            &self.key,
            "POST",
            "/v1/chat/completions",
            Some(body),
        );
        let v: Value = serde_json::from_slice(&answer).unwrap_or(Value::Null);
        assert_eq!(
            status,
            200,
            "chat completion: {}",
            String::from_utf8_lossy(&answer)
        );
        v
    }
}

impl Drop for Server {
    fn drop(&mut self) {
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

// ---- the prompts' tokens, through the server's renderer and the CLI's tokenizer ----

fn tokens(s: &Setup, messages: &Value) -> Vec<i64> {
    rendered_tokens(s, messages, false)
}

/// The prompt's tokens, with thinking on or off.
fn rendered_tokens(s: &Setup, messages: &Value, thinking: bool) -> Vec<i64> {
    let case = std::env::temp_dir().join(format!("myds4-fork-{}.json", std::process::id()));
    let mode = if thinking { "thinking" } else { "chat" };
    std::fs::write(
        &case,
        json!({"messages": messages, "thinking_mode": mode}).to_string(),
    )
    .expect("a render case");
    let rendered = Command::new(&s.server)
        .arg("--render")
        .arg(&case)
        .output()
        .expect("render");
    let _ = std::fs::remove_file(&case);
    assert!(
        rendered.status.success(),
        "render: {}",
        String::from_utf8_lossy(&rendered.stderr)
    );
    let prompt = String::from_utf8(rendered.stdout).expect("UTF-8");
    let ids = Command::new(&s.cli)
        .args([
            "-m",
            &s.checkpoint,
            "--rendered",
            "--ctx",
            "32768",
            "--tokenize",
            "-p",
        ])
        .arg(prompt.trim_end_matches('\n'))
        .output()
        .expect("tokenize");
    assert!(
        ids.status.success(),
        "tokenize: {}",
        String::from_utf8_lossy(&ids.stderr)
    );
    String::from_utf8_lossy(&ids.stdout)
        .trim()
        .split(',')
        .filter_map(|t| t.parse().ok())
        .collect()
}

fn common(a: &[i64], b: &[i64]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

fn request(messages: &Value, max_tokens: u32) -> Value {
    body(messages, max_tokens, false)
}

fn body(messages: &Value, max_tokens: u32, thinking: bool) -> Value {
    json!({"model": "deepseek-v4.1-flash", "messages": messages, "max_tokens": max_tokens, "temperature": 0,
           "thinking": thinking, "stream": false})
}

fn hit(answer: &Value) -> i64 {
    answer["usage"]["prompt_cache_hit_tokens"]
        .as_i64()
        .unwrap_or(-1)
}

fn content(answer: &Value) -> String {
    answer["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("")
        .to_string()
}

/// The reply's reasoning and its content.
fn reply(answer: &Value) -> String {
    let m = &answer["choices"][0]["message"];
    format!(
        "{}|{}",
        m["reasoning_content"].as_str().unwrap_or(""),
        m["content"].as_str().unwrap_or("")
    )
}

/// The rows the server's last prefill computed (the model's status): the
/// prompt's new tokens, and with a replay also the earlier rows it computed
/// again.
fn last_prefill(server: &Server) -> i64 {
    let (code, v) = server.status();
    assert_eq!(code, 200, "GET /v1/models/{{id}}");
    v["last_prefill"]["tokens"].as_i64().unwrap_or(-1)
}

/// The conversation's text, from the repository's own documents: a long
/// system prompt and a long user turn.
fn texts() -> (String, String) {
    let design = std::fs::read_to_string(root().join("docs/design.md")).expect("docs/design.md");
    let usage = std::fs::read_to_string(root().join("docs/usage.md")).expect("docs/usage.md");
    let cut = |t: &str, n: usize| t.chars().take(n).collect::<String>();
    (
        format!(
            "You review this project's design. Reference:\n\n{}",
            cut(&design, 15000)
        ),
        cut(&usage, 4000),
    )
}

/// Character offsets: fixed ones near the ends and seeded random ones.
fn offsets(len: usize, random: usize, seed: u64) -> Vec<usize> {
    let mut v = vec![1, len / 3, len / 2, len.saturating_sub(2)];
    let mut x = seed;
    for _ in 0..random {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        v.push(1 + (x as usize) % len.saturating_sub(2).max(1));
    }
    v.sort();
    v.dedup();
    v
}

/// The conversation (a long system prompt and a long user turn) and its forks:
/// the user turn or the system prompt cut at an offset and continued
/// differently.
fn conversation_and_forks() -> (Value, Vec<(String, Value)>) {
    let (system, user) = texts();
    let tail =
        " Leave the rest aside and answer in one short sentence: what does this part describe?";
    let conversation =
        json!([{"role": "system", "content": system}, {"role": "user", "content": user}]);
    let mut forks: Vec<(String, Value)> = Vec::new();
    for c in offsets(user.chars().count(), 8, 0x9E37_79B9) {
        let cut: String = user.chars().take(c).collect();
        forks.push((
            format!("user turn at character {c}"),
            json!([{"role": "system", "content": system}, {"role": "user", "content": cut + tail}]),
        ));
    }
    for c in offsets(system.chars().count(), 4, 0x85EB_CA6B) {
        let cut: String = system.chars().take(c).collect();
        forks.push((
            format!("system prompt at character {c}"),
            json!([{"role": "system", "content": cut + tail}, {"role": "user", "content": user}]),
        ));
    }
    (conversation, forks)
}

/// A fork's outcome on the cached server.
struct Forked {
    what: String,
    messages: Value,
    shared: usize,
    reused: i64,
    text: String,
}

/// The forks, one after another on one server (with the extra options) while
/// the conversation decodes there. A fork whose reuse lies outside
/// [shared - rows(shared), shared] is added to `wrong`.
fn run_forks(
    s: &Setup,
    extra: &[&str],
    rows: fn(usize) -> usize,
    wrong: &mut Vec<String>,
) -> Vec<Forked> {
    let (conversation, forks) = conversation_and_forks();
    let source = tokens(s, &conversation);
    let mut results = Vec::new();
    let server = Server::start(s, extra);
    let port = server.port;
    let key = server.key.clone();
    let long = request(&conversation, 6000);
    let busy =
        std::thread::spawn(move || http(port, &key, "POST", "/v1/chat/completions", Some(&long)));
    std::thread::sleep(Duration::from_secs(5)); // past the conversation's prefill: it decodes
    for (what, messages) in forks {
        let shared = common(&tokens(s, &messages), &source);
        let answer = server.chat(&request(&messages, 16));
        let (reused, least) = (hit(&answer), shared - rows(shared));
        println!("{what}: shares {shared} tokens, reused {reused} (at least {least})");
        if reused < least as i64 || reused > shared as i64 {
            wrong.push(format!(
                "{what}: shares {shared} tokens, reused {reused}; at least {least} expected"
            ));
        }
        results.push(Forked {
            what,
            messages,
            shared,
            reused,
            text: content(&answer),
        });
    }
    drop(server); // ends the decoding request too
    let _ = busy.join();
    results
}

#[test]
fn forks_reuse_the_shared_prefix_and_give_the_cold_tokens() {
    let Some(s) = setup() else {
        eprintln!("skipped: set MYDS4_CHECKPOINT and MYDS4_EXPERTS");
        return;
    };
    let _one = one_model();
    let mut wrong = Vec::new();
    let results = run_forks(&s, &[], replay_rows, &mut wrong);
    // The same prompts on servers that start without a cache.
    for f in &results {
        let cold = Server::start(&s, &[]);
        let reference = content(&cold.chat(&request(&f.messages, 16)));
        drop(cold);
        let same = reference == f.text;
        println!(
            "{} (shares {}, reused {}): tokens {}",
            f.what,
            f.shared,
            f.reused,
            if same {
                "equal to the cold server's"
            } else {
                "DIFFERENT from the cold server's"
            }
        );
        if !same {
            wrong.push(format!(
                "{}: {:?} against the cold server's {reference:?}",
                f.what, f.text
            ));
        }
    }
    assert!(
        wrong.is_empty(),
        "forks that reused too little or changed the output:\n{}",
        wrong.join("\n")
    );
}

/// With --ced 128 a fork resumes by a bounded replay (a ring's positions
/// computed again with their windows cut off there) or a snapshot at most a
/// ring before its shared prefix.
#[test]
fn ced_forks_reuse_the_shared_prefix_less_a_ring() {
    let Some(s) = setup() else {
        eprintln!("skipped: set MYDS4_CHECKPOINT and MYDS4_EXPERTS");
        return;
    };
    let _one = one_model();
    let mut wrong = Vec::new();
    for f in run_forks(&s, &["--ced", "128"], bounded_rows, &mut wrong) {
        println!("{}: {:?}", f.what, f.text);
    }
    assert!(
        wrong.is_empty(),
        "forks that reused too little:\n{}",
        wrong.join("\n")
    );
}

/// A thinking prompt ends with <think>. The next turn renders that reply
/// with </think> there, so it shares all of the prompt but its last token.
/// It resumes from the turn's prompt-end snapshot one position back, without
/// computing any earlier rows again, with --ced too; without --ced it gives
/// the tokens that a server starting without a cache gives.
#[test]
fn a_thinking_turn_s_next_turn_resumes_one_position_before_its_prompt_end() {
    let Some(s) = setup() else {
        eprintln!("skipped: set MYDS4_CHECKPOINT and MYDS4_EXPERTS");
        return;
    };
    let _one = one_model();
    let (system, user) = texts();
    let first = json!([{"role": "system", "content": system}, {"role": "user", "content": user}]);
    let second = json!([{"role": "system", "content": system}, {"role": "user", "content": user},
                        {"role": "assistant", "content": "It describes how the server is started and checked."},
                        {"role": "user", "content": "Which of those options change the output?"}]);
    let (a, b) = (
        rendered_tokens(&s, &first, true),
        rendered_tokens(&s, &second, true),
    );
    let shared = common(&a, &b);
    assert_eq!(
        shared,
        a.len() - 1,
        "the next turn parts at the first turn's last token"
    );
    let mut wrong = Vec::new();
    for extra in [&[][..], &["--ced", "128"][..]] {
        let what = if extra.is_empty() {
            "without --ced"
        } else {
            "with --ced 128"
        };
        let server = Server::start(&s, extra);
        server.chat(&body(&first, 16, true));
        let answer = server.chat(&body(&second, 16, true));
        let (reused, computed) = (hit(&answer), last_prefill(&server));
        println!(
            "{what}: the next turn shares {shared} of its {} tokens, reused {reused}, computed {computed} rows",
            b.len()
        );
        if answer["usage"]["prompt_tokens"].as_i64() != Some(b.len() as i64) {
            wrong.push(format!(
                "{what}: the server's prompt has {} tokens, the test's {}",
                answer["usage"]["prompt_tokens"],
                b.len()
            ));
        }
        if reused != shared as i64 || computed != (b.len() - shared) as i64 {
            wrong.push(format!(
                "{what}: reused {reused} and computed {computed} rows; {shared} and {} expected",
                b.len() - shared
            ));
        }
        if extra.is_empty() {
            let text = reply(&answer);
            drop(server);
            let cold = Server::start(&s, &[]);
            let reference = reply(&cold.chat(&body(&second, 16, true)));
            println!(
                "{what}: tokens {}",
                if text == reference {
                    "equal to the cold server's"
                } else {
                    "DIFFERENT from the cold server's"
                }
            );
            if text != reference {
                wrong.push(format!(
                    "{what}: {text:?} against the cold server's {reference:?}"
                ));
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "the next thinking turn:\n{}",
        wrong.join("\n")
    );
}

// ---- an image in the shared prefix ----

fn base64(data: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::new();
    for c in data.chunks(3) {
        let n = (c[0] as u32) << 16
            | (*c.get(1).unwrap_or(&0) as u32) << 8
            | *c.get(2).unwrap_or(&0) as u32;
        for i in 0..4 {
            out.push(if i <= c.len() {
                ALPHABET[(n >> (18 - 6 * i) & 63) as usize] as char
            } else {
                '='
            });
        }
    }
    out
}

fn crc32(data: &[u8]) -> u32 {
    let mut c = !0u32;
    for &b in data {
        c ^= b as u32;
        for _ in 0..8 {
            c = if c & 1 != 0 {
                0xEDB8_8320 ^ (c >> 1)
            } else {
                c >> 1
            };
        }
    }
    !c
}

/// A small original image as a data URL: a 32 x 32 PNG in four colored
/// quarters (red, green, blue, yellow), its pixels in one stored deflate block.
fn image_url() -> String {
    let (w, h) = (32u32, 32u32);
    let mut pixels = Vec::new();
    for y in 0..h {
        pixels.push(0); // no filter
        for x in 0..w {
            pixels.extend(match (x < w / 2, y < h / 2) {
                (true, true) => [220, 40, 40],
                (false, true) => [40, 200, 40],
                (true, false) => [40, 60, 220],
                (false, false) => [230, 220, 30],
            });
        }
    }
    let (mut a, mut b) = (1u32, 0u32); // Adler-32
    for &p in &pixels {
        a = (a + p as u32) % 65521;
        b = (b + a) % 65521;
    }
    let mut zlib = vec![0x78, 0x01, 0x01];
    zlib.extend((pixels.len() as u16).to_le_bytes());
    zlib.extend((!(pixels.len() as u16)).to_le_bytes());
    zlib.extend(&pixels);
    zlib.extend((b << 16 | a).to_be_bytes());
    let chunk = |png: &mut Vec<u8>, kind: &[u8], data: &[u8]| {
        png.extend((data.len() as u32).to_be_bytes());
        let body: Vec<u8> = kind.iter().chain(data).copied().collect();
        png.extend(&body);
        png.extend(crc32(&body).to_be_bytes());
    };
    let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
    let mut header = Vec::new();
    header.extend(w.to_be_bytes());
    header.extend(h.to_be_bytes());
    header.extend([8, 2, 0, 0, 0]); // 8-bit RGB
    chunk(&mut png, b"IHDR", &header);
    chunk(&mut png, b"IDAT", &zlib);
    chunk(&mut png, b"IEND", &[]);
    format!("data:image/png;base64,{}", base64(&png))
}

/// A fork after an image reuses the rows through the image and gives the
/// tokens that a server starting without a cache gives: the image's rows are
/// the same, also those that an exact replay at the fork computes again.
#[test]
fn image_forks_give_the_cold_tokens() {
    let Some(s) = setup() else {
        eprintln!("skipped: set MYDS4_CHECKPOINT and MYDS4_EXPERTS");
        return;
    };
    let _one = one_model();
    let (system, _) = texts();
    let image = image_url();
    let turn = |question: &str| {
        json!([{"role": "system", "content": system},
               {"role": "user", "content": [{"type": "image_url", "image_url": {"url": image}},
                                            {"type": "text", "text": question}]}])
    };
    let first = turn("Describe the colors of this picture in one sentence.");
    let fork = turn("Which color is in the top left quarter? Answer in one word.");
    let (reused, text) = {
        let server = Server::start(&s, &[]);
        server.chat(&request(&first, 8));
        let answer = server.chat(&request(&fork, 16));
        (hit(&answer), content(&answer))
    };
    let reference = {
        let cold = Server::start(&s, &[]);
        content(&cold.chat(&request(&fork, 16)))
    };
    println!(
        "image fork: reused {reused}; tokens {}",
        if text == reference {
            "equal to the cold server's"
        } else {
            "DIFFERENT"
        }
    );
    assert!(reused > 0, "the fork after the image reused nothing");
    assert_eq!(
        text, reference,
        "the fork after the image gives other tokens than a cold server"
    );
}

// ---- the cache's budget (--kv-cache) ----

/// What the model's status tells of the cache: its charge and limit in GiB, the
/// phase, the requests that ended, and each state's positions.
struct Status {
    kv: f64,
    limit: f64,
    phase: String,
    requests: i64,
    sessions: Vec<i64>,
}

fn status(server: &Server) -> Status {
    let v = server.status().1;
    Status {
        kv: v["kv_cache_gib"].as_f64().unwrap_or(f64::NAN),
        limit: v["kv_cache_limit_gib"].as_f64().unwrap_or(f64::NAN),
        phase: v["phase"].as_str().unwrap_or("").to_string(),
        requests: v["totals"]["requests"].as_i64().unwrap_or(0),
        sessions: v["sessions"]
            .as_array()
            .map(|a| a.iter().filter_map(|s| s["tokens"].as_i64()).collect())
            .unwrap_or_default(),
    }
}

/// The status once `requests` requests have ended and the server is idle.
fn settled(server: &Server, requests: i64) -> Status {
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        let st = status(server);
        if (st.phase == "idle" && st.requests >= requests) || Instant::now() > deadline {
            return st;
        }
        std::thread::sleep(Duration::from_millis(200));
    }
}

/// A short conversation with a one-word answer.
fn short_prompt() -> Value {
    json!([{"role": "system", "content": "You answer in one word."}, {"role": "user", "content": "Name a color."}])
}

/// A retry of a saved prompt resumes at its prompt end from the snapshot
/// there and takes no snapshot of its own: it fits a cache that the first try
/// filled to within less than a snapshot (5.3 MiB) of --kv-cache.
#[test]
fn a_saved_prompt_again_fits_a_full_kv_cache() {
    let Some(s) = setup() else {
        eprintln!("skipped: set MYDS4_CHECKPOINT and MYDS4_EXPERTS");
        return;
    };
    let _one = one_model();
    let prompt = short_prompt();
    let n = tokens(&s, &prompt).len() as i64;
    // What the first try takes: its state and its prompt-end snapshot.
    let first = {
        let server = Server::start(&s, &[]);
        server.chat(&request(&prompt, 1));
        settled(&server, 1).kv
    };
    let limit = format!("{:.4}", first + 2.0 / 1024.0);
    let server = Server::start(&s, &["--kv-cache", &limit]);
    server.chat(&request(&prompt, 1));
    let again = server.chat(&request(&prompt, 1));
    let st = settled(&server, 2);
    println!(
        "saved prompt again: the first try took {first:.3} GiB, --kv-cache {limit}; the retry reused {} of {n}; the cache takes {:.3} GiB",
        hit(&again),
        st.kv
    );
    assert_eq!(hit(&again), n, "the retry did not resume at its prompt end");
    assert!(
        st.kv <= st.limit,
        "the cache takes {:.3} GiB of {:.3}",
        st.kv,
        st.limit
    );
}

/// A conversation with `lo` to `hi` prompt tokens: a cut of docs/design.md,
/// repeated as often as that size needs, as the system prompt, and a question
/// whose answer is long.
fn sized_prompt(s: &Setup, lo: usize, hi: usize) -> (Value, usize) {
    let text = std::fs::read_to_string(root().join("docs/design.md")).expect("docs/design.md");
    let conversation = |reference: &str| {
        json!([{"role": "system", "content": format!("You review this project's design. Reference:\n\n{reference}")},
               {"role": "user", "content": "List every heading of the reference, one per line, each with a one-sentence summary."}])
    };
    let once = tokens(s, &conversation(&text)).len();
    let design: Vec<char> = text.repeat(hi / once + 1).chars().collect();
    let make = |c: usize| conversation(&design[..c].iter().collect::<String>());
    let (mut a, mut b) = (0, design.len());
    while a < b {
        let c = (a + b) / 2;
        let m = make(c);
        let n = tokens(s, &m).len();
        if n < lo {
            a = c + 1;
        } else if n >= hi {
            b = c;
        } else {
            return (m, n);
        }
    }
    panic!("no cut of docs/design.md gives {lo} to {hi} prompt tokens");
}

/// A generation that takes its state past the room it started with (16,384
/// positions; the next 16,384 take 50 MiB more) is charged that growth before
/// the state grows: an idle state goes then, not after the generation, and
/// the cache stays within --kv-cache all along.
#[test]
fn a_growing_generation_frees_room_before_it_grows() {
    let Some(s) = setup() else {
        eprintln!("skipped: set MYDS4_CHECKPOINT and MYDS4_EXPERTS");
        return;
    };
    let _one = one_model();
    let short = short_prompt();
    let (long, n) = sized_prompt(&s, 16_330, 16_376);
    // What the short conversation takes alone.
    let f = {
        let server = Server::start(&s, &[]);
        server.chat(&request(&short, 1));
        settled(&server, 1).kv
    };
    // Room for the short conversation, the long prompt's state as it starts
    // (it takes no more than the short one's and a snapshot) and 25 MiB:
    // less than the long state's growth.
    let limit = format!("{:.4}", 2.0 * f + 25.0 / 1024.0);
    let server = Server::start(&s, &["--kv-cache", &limit]);
    server.chat(&request(&short, 1));
    let body = request(&long, (16_384 - n + 150) as u32);
    let (port, key) = (server.port, server.key.clone());
    let job =
        std::thread::spawn(move || http(port, &key, "POST", "/v1/chat/completions", Some(&body)));
    let mut samples = Vec::new();
    while !job.is_finished() {
        samples.push(status(&server));
        std::thread::sleep(Duration::from_millis(250));
    }
    let (code, answer) = job.join().expect("the long request");
    assert_eq!(
        code,
        200,
        "the long request: {}",
        String::from_utf8_lossy(&answer)
    );
    let answer: Value = serde_json::from_slice(&answer).unwrap_or(Value::Null);
    let generated = answer["usage"]["completion_tokens"].as_i64().unwrap_or(0) as usize;
    // Samples published after the long state grew: it holds more than 16,384 positions.
    let grown: Vec<&Status> = samples
        .iter()
        .filter(|st| st.phase == "decode" && st.sessions.iter().any(|&t| t > 16_384))
        .collect();
    let over = samples.iter().filter(|st| st.kv > st.limit).count();
    println!(
        "growing generation: {n} prompt tokens, {generated} generated, --kv-cache {limit} (the short one took {f:.3} GiB); \
         {} samples, {} while decoding past 16,384 positions, of which {} with the idle state still kept; {over} over the limit",
        samples.len(),
        grown.len(),
        grown.iter().filter(|st| st.sessions.len() > 1).count()
    );
    assert!(
        n + generated > 16_390,
        "the reply ended at {} positions, before the state grew",
        n + generated
    );
    assert!(
        !grown.is_empty(),
        "no status sample while it decoded past 16,384 positions"
    );
    assert!(
        grown.iter().all(|st| st.sessions.len() == 1),
        "the idle state stayed while the generation grew past its room"
    );
    assert_eq!(over, 0, "the cache took more than --kv-cache");
}

/// Quitting stops a long prompt's prefill: SIGTERM 5 s into the prefill of
/// about 15,000 tokens ends the server within 5 s, not when the prefill
/// ends, and the request's answer is the server's stop (503). A short
/// request loads the model first, so the long prompt's prefill starts as
/// soon as it arrives.
#[test]
fn quitting_stops_a_long_prefill() {
    let Some(s) = setup() else {
        eprintln!("skipped: set MYDS4_CHECKPOINT and MYDS4_EXPERTS");
        return;
    };
    let _one = one_model();
    let (long, n) = sized_prompt(&s, 15_000, 16_000);
    let mut server = Server::start(&s, &[]);
    server.chat(&request(&short_prompt(), 1));
    let body = request(&long, 8);
    let (port, key) = (server.port, server.key.clone());
    let job =
        std::thread::spawn(move || http(port, &key, "POST", "/v1/chat/completions", Some(&body)));
    std::thread::sleep(Duration::from_secs(5));
    let stop = Instant::now();
    let _ = Command::new("kill")
        .args(["-TERM", &server.child.id().to_string()])
        .status();
    let mut exit = None;
    while exit.is_none() && stop.elapsed() < Duration::from_secs(300) {
        exit = server.child.try_wait().ok().flatten();
        std::thread::sleep(Duration::from_millis(50));
    }
    let took = stop.elapsed().as_secs_f64();
    let (code, answer) = job.join().expect("the long request");
    println!(
        "quitting a {n}-token prefill: the server exited {took:.1} s after SIGTERM ({exit:?}); the request: {code} {}",
        String::from_utf8_lossy(&answer)
    );
    assert!(
        exit.is_some_and(|e| e.success()),
        "the server did not exit cleanly"
    );
    assert!(
        took < 5.0,
        "the server exited {took:.1} s after SIGTERM, during a {n}-token prefill"
    );
    assert_eq!(code, 503, "the stopped request's answer");
}

/// A model that goes its idle_unload time without requests gives back its
/// weights and experts: its status reads unloaded, the memory it held is back
/// in the engine's count and in the process's footprint, and its session
/// stays. It stays unloaded while the server keeps calling its idle hook (35
/// s: past DeepSeek's 32 calm calls, after which its cache raises its cap).
/// The same request then loads it again and continues that session with the
/// same greedy reply.
#[test]
fn an_idle_model_unloads_and_its_next_request_loads_it_again() {
    let Some(s) = setup() else {
        eprintln!("skipped: set MYDS4_CHECKPOINT and MYDS4_EXPERTS");
        return;
    };
    let _one = one_model();
    let server = Server::start(&s, &["--idle-unload", "3"]);
    let footprint = |v: &Value| v["memory"]["footprint_gib"].as_f64().unwrap_or(f64::MAX);
    let ask = request(&short_prompt(), 16);
    let first = content(&server.chat(&ask));
    let replied = Instant::now();
    let (_, before) = server.status();
    assert_eq!(before["loaded"], json!(true), "{before}");
    let bytes = |v: &Value| v["model_memory_budget"]["allocated"].as_u64().unwrap_or(0);
    let held = bytes(&before);
    // the server looks once a second: unloaded a little after the 3 s
    let mut after = before.clone();
    while after["loaded"] != json!(false) && replied.elapsed() < Duration::from_secs(30) {
        std::thread::sleep(Duration::from_millis(250));
        after = server.status().1;
    }
    let took = replied.elapsed().as_secs_f64();
    let back = held.saturating_sub(bytes(&after));
    // then it stays unloaded, and freed pages reach the counters
    let unloaded = Instant::now();
    let mut stayed = after["loaded"] == json!(false);
    while stayed && unloaded.elapsed() < Duration::from_secs(35) {
        std::thread::sleep(Duration::from_secs(1));
        after = server.status().1;
        stayed = after["loaded"] == json!(false);
    }
    let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
    println!(
        "unloaded {took:.1} s after the reply: {:.2} of {:.2} GiB of model memory back; \
         the process's footprint {:.1} GiB loaded, {:.1} after 35 s unloaded; sessions {}",
        gib(back),
        gib(held),
        footprint(&before),
        footprint(&after),
        after["sessions"]
    );
    assert!(
        took >= 2.5,
        "unloaded {took:.1} s after the reply, before 3 s"
    );
    assert!(stayed, "it did not stay unloaded: {after}");
    assert!(back * 10 >= held * 9, "{back} of {held} bytes back");
    // the pages go back to the system too, not only the engine's count; the
    // session, the workspace and the compiled kernels stay
    assert!(
        footprint(&after) <= footprint(&before) / 10.0,
        "the footprint fell from {:.1} to only {:.1} GiB",
        footprint(&before),
        footprint(&after)
    );
    assert!(
        after["sessions"].as_array().is_some_and(|l| !l.is_empty()),
        "the session stays: {after}"
    );
    let again = server.chat(&ask);
    assert_eq!(content(&again), first, "the reply after the reload");
    assert_eq!(server.status().1["loaded"], json!(true));
}
