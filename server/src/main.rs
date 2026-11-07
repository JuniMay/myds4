//! myds4-server: the OpenAI Chat Completions API over myds4's engine, with
//! DeepSeek's deepseek-recipe for requests, prompts and output
//! (docs/server.md). The engine (src/engine/engine.c behind
//! src/engine/engine.h, engine.rs) runs the model and owns the stop on SIGINT
//! or SIGTERM; the worker (worker.rs) owns the queue, the batching and the
//! requests' events, the KV prefix cache (kv.rs) the states kept between
//! requests; the live view (--tui) is view.rs.

mod chat;
mod engine;
mod http;
mod kv;
mod peers;
mod qwen;
mod view;
mod worker;

use std::io::IsTerminal;
use std::net::{Ipv4Addr, SocketAddr};
use std::process::exit;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::Router;
use axum::body::Body;
use axum::extract::{ConnectInfo, Request, State};
use axum::http::{HeaderMap, StatusCode, header};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use sha2::{Digest, Sha256};

use crate::chat::{Defaults, MODEL_ID};

const USAGE: &str = "usage: myds4-server -m MODEL --serve PORT [options]
  See docs/server.md.
  -m DIR          a checkpoint directory (DeepSeek V4.1 Flash or Qwen3.8 qwen3_5); give it again to serve
                  more models side by side, each in its own process on the next port, chosen by a
                  request's model field (see docs/server.md)
  --serve PORT    serve OpenAI Chat Completions at http://HOST:PORT/v1 (needs MYDS4_API_KEY or --api-key-file)
  --host ADDR     the IPv4 address to serve on (default 127.0.0.1)
  --api-key-file FILE  the API key clients must send (16 to 256 characters)
  --kv-cache GB   bound of the KV prefix cache: attention states and snapshots kept between requests (default 8)
  --batch N       requests decoding together at most, 1 to 6 (default 4)
  --tui           a live view in the terminal (requests, speed, experts, memory, the KV cache) instead of the
                  log on stderr; q stops the server, and the last log lines print on exit
  --tui-theme T   the live view's colors: auto (default: by the terminal's background, light without an
                  answer), light or dark
  -t T, --temperature T  the temperature of requests that set none (default 1; 0 is greedy)
  --top-p P       the top_p of requests that set none, 0 to 1 (default 1)
  --seed N        the seed of requests that set none (default: random for each)
  --think N       thinking effort 1-100 of requests that set none; 0 turns thinking off (default 75)
  --ctx N         context capacity in tokens, at most 1048576 (default 1048576)
  --ced off|N     approximate prefill: replay N decoder tokens, 128-1048576 (default off), plus the rewind margin (8)
  --prefill-prune  skip decoder batches outside required history; CED and substitution must be off (default off)
  --cache GB      expert cache budget (default: from available memory)
  --reserve GB    memory the expert cache leaves to other programs, 2-1024 (default 10)
  --cache-half-life N  score half-life in forward passes, 1-1000000 (default 50)
  --cache-read-cost   weight decode eviction priority by expert read bytes
  --experts GGUF  read the routed experts from a GGUF file or set of shards
  --substitute D  approximate: replace selected experts not in memory (see build/myds4)
  --draft N       verify up to N (1-5) DSpark drafts per pass
  --lookup off|N  verify up to N (1-15) context-copy drafts; --draft supplies fallback
  --runahead     overlap exact concurrent requests with expert reads
  --draft-cache N  with --draft: expert slots per stage, 15-128
  --online-predictor  CPU prefetch correction learned online
  --predictor-state FILE  online learning, its state loaded and saved
  --render FILE   print the prompt of a JSON test case of the model's encoding/tests, and exit
";

fn usage() -> ! {
    eprint!("{USAGE}");
    exit(2)
}

fn die(message: &str) -> ! {
    view::leave();
    eprintln!("myds4-server: {message}");
    exit(1)
}

fn value<T: std::str::FromStr>(flag: &str, v: &str) -> T {
    v.parse().unwrap_or_else(|_| die(&format!("{flag}: not a valid value: {v}")))
}

fn ced_depth(v: &str) -> Option<i32> {
    if v == "off" { Some(0) } else { v.parse().ok().filter(|n| (128..=1 << 20).contains(n)) }
}

#[test]
fn ced_accepts_off_or_a_bounded_replay_depth() {
    assert_eq!(ced_depth("off"), Some(0));
    assert_eq!(ced_depth("128"), Some(128));
    assert_eq!(ced_depth("512"), Some(512));
    assert_eq!(ced_depth("1048576"), Some(1048576));
    for v in ["", "on", "0", "127", "-1", "1048577", "128.0", "128x"] {
        assert_eq!(ced_depth(v), None, "{v}");
    }
}

struct Args {
    engine: engine::Config,
    /// Every -m, in order: the first runs in this process.
    models: Vec<String>,
    /// -t and --top-p as given (a Qwen model's own defaults stand otherwise).
    temperature: Option<f64>,
    top_p: Option<f64>,
    /// --parent PID: a process this one serves for, ending with it.
    parent: Option<i32>,
    port: Option<u16>,
    host: String,
    key_file: Option<String>,
    defaults: Defaults,
    render: Option<String>,
    tui: bool,
    tui_colors: view::Colors,
}

fn parse(args: &[String]) -> Args {
    let mut a = Args {
        engine: engine::Config { ctx: 1 << 20, kv_gb: 8.0, batch: 4, ..Default::default() },
        models: Vec::new(),
        temperature: None,
        top_p: None,
        parent: None,
        port: None,
        host: "127.0.0.1".into(),
        key_file: None,
        defaults: Defaults { temperature: 1.0, top_p: 1.0, seed: None, thinking: true, effort: 75 },
        render: None,
        tui: false,
        tui_colors: view::Colors::Auto,
    };
    let mut i = 0;
    while i < args.len() {
        let flag = args[i].as_str();
        if flag == "--tui" {
            a.tui = true;
            i += 1;
            continue;
        }
        let e = &mut a.engine;
        match flag {
            "--cache-read-cost" => e.cache_read_cost = true,
            "--runahead" => e.runahead = true,
            "--prefill-prune" => e.prefill_prune = true,
            "--online-predictor" => e.online_predictor = true,
            _ => {
                let Some(v) = args.get(i + 1) else { usage() };
                i += 1;
                match flag {
                    "-m" => {
                        if a.models.is_empty() {
                            e.model = v.clone();
                        }
                        a.models.push(v.clone());
                    }
                    "--parent" => a.parent = Some(value(flag, v)),
                    "--experts" => e.experts = Some(v.clone()),
                    "--ctx" => e.ctx = value(flag, v),
                    "--ced" => e.ced = ced_depth(v).unwrap_or_else(|| die("--ced takes off or 128 to 1048576 replay tokens")),
                    "--cache" => e.cache_gb = value(flag, v),
                    "--reserve" => e.reserve_gb = value(flag, v),
                    "--cache-half-life" => e.cache_half_life = value(flag, v),
                    "--substitute" => e.substitute = value(flag, v),
                    "--draft" => e.draft = value(flag, v),
                    "--lookup" => e.lookup = if v == "off" { 0 } else { value(flag, v) },
                    "--draft-cache" => e.draft_cache = value(flag, v),
                    "--predictor-state" => {
                        if v.is_empty() {
                            die("--predictor-state needs a nonempty path");
                        }
                        e.predictor_state = Some(v.clone());
                    }
                    "--kv-cache" => e.kv_gb = value(flag, v),
                    "--batch" => e.batch = value(flag, v),
                    "--serve" => a.port = Some(value(flag, v)),
                    "--host" => a.host = v.clone(),
                    "--api-key-file" => a.key_file = Some(v.clone()),
                    "-t" | "--temperature" => {
                        let t: f64 = value(flag, v);
                        if !t.is_finite() || t < 0.0 {
                            die("temperature must be finite and in [0, infinity)");
                        }
                        a.defaults.temperature = t;
                        a.temperature = Some(t);
                    }
                    "--top-p" => {
                        let p: f64 = value(flag, v);
                        if !(0.0..=1.0).contains(&p) {
                            die("top-p must be finite and in [0, 1]");
                        }
                        a.defaults.top_p = p;
                        a.top_p = Some(p);
                    }
                    "--seed" => a.defaults.seed = Some(v.parse().unwrap_or_else(|_| die("--seed needs a nonnegative integer"))),
                    "--think" => {
                        let n: u32 = v.parse().ok().filter(|n| *n <= 100).unwrap_or_else(|| die("--think takes 0 (off) or 1 to 100"));
                        a.defaults.thinking = n > 0;
                        a.defaults.effort = if n > 0 { n } else { 75 };
                    }
                    "--render" => a.render = Some(v.clone()),
                    "--tui-theme" => {
                        a.tui_colors = match v.as_str() {
                            "auto" => view::Colors::Auto,
                            "light" => view::Colors::Light,
                            "dark" => view::Colors::Dark,
                            _ => die("--tui-theme takes auto, light or dark"),
                        }
                    }
                    _ => usage(),
                }
            }
        }
        i += 1;
    }
    a
}

#[test]
fn prefill_and_draft_options_keep_the_sampling_defaults() {
    let defaults = parse(&[]);
    assert!(!defaults.engine.prefill_prune);
    assert_eq!(defaults.engine.ced, 0);
    assert_eq!(defaults.engine.lookup, 0);
    let args = ["--prefill-prune", "--lookup", "15", "--draft", "2"];
    let enabled = parse(&args.map(String::from));
    assert!(enabled.engine.prefill_prune);
    assert_eq!(enabled.engine.ced, 0);
    assert_eq!(enabled.engine.lookup, 15);
    assert_eq!(enabled.engine.draft, 2);
    assert_eq!(enabled.defaults.temperature, 1.0);
    assert_eq!(enabled.defaults.top_p, 1.0);
}

#[test]
fn runahead_is_explicit_and_preserves_sampling_defaults() {
    assert!(!parse(&[]).engine.runahead);
    let a = parse(&["--runahead".into(), "--batch".into(), "4".into()]);
    assert!(a.engine.runahead);
    assert_eq!(a.engine.batch, 4);
    assert_eq!((a.defaults.temperature, a.defaults.top_p), (1.0, 1.0));
    assert_eq!(a.engine.draft, 0);
    assert_eq!(a.engine.substitute, 0.0);
}

/// What the handlers share.
struct Shared {
    key: [u8; 32],  // the API key's SHA-256
    raw_key: String, // for the peers' servers
    defaults: Defaults,
    ctx: i64,
    started: u64,
    arrivals: AtomicU64,
    /// This process's model: DeepSeek's, or a Qwen model's.
    qwen: bool,
    id: String,
    /// The models other processes serve (peers.rs).
    peers: Vec<peers::Peer>,
}

/// Where a request goes: this process's model, or a peer's (its index).
enum Route {
    Local,
    Peer(usize),
}

/// The model a request names: this process's when it names none, or names
/// an unknown model while no other is served; an error when it names an
/// unknown one among several.
fn route(s: &Shared, body: &[u8]) -> Result<Route, Response> {
    #[derive(serde::Deserialize)]
    struct Named {
        model: Option<String>,
    }
    let Some(name) = serde_json::from_slice::<Named>(body).ok().and_then(|n| n.model) else { return Ok(Route::Local) };
    if name.eq_ignore_ascii_case(&s.id) {
        return Ok(Route::Local);
    }
    if let Some(i) = s.peers.iter().position(|p| name.eq_ignore_ascii_case(&p.id)) {
        return Ok(Route::Peer(i));
    }
    if s.peers.is_empty() {
        return Ok(Route::Local);
    }
    let known: Vec<&str> = std::iter::once(s.id.as_str()).chain(s.peers.iter().map(|p| p.id.as_str())).collect();
    Err(chat::error(
        StatusCode::NOT_FOUND,
        "invalid_request_error",
        &format!("no model {name:?} here; the models are {}", known.join(", ")),
    ))
}

type App = State<Arc<Shared>>;

/// Every request but GET /health carries the key, as Authorization: Bearer
/// KEY or x-api-key: KEY. The digests are compared in constant time, so the
/// time taken tells nothing about the key.
async fn authorize(State(s): App, ConnectInfo(peer): ConnectInfo<SocketAddr>, request: Request, next: Next) -> Response {
    if matches!(request.uri().path(), "/health" | "/v1/health") {
        return next.run(request).await;
    }
    let headers: &HeaderMap = request.headers();
    let key = headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.get(..7).filter(|b| b.eq_ignore_ascii_case("bearer ")).map(|_| v[7..].trim_start()))
        .or_else(|| headers.get("x-api-key").and_then(|v| v.to_str().ok()));
    let given: [u8; 32] = Sha256::digest(key.unwrap_or_default().as_bytes()).into();
    let same = given.iter().zip(s.key.iter()).fold(0u8, |d, (a, b)| d | (a ^ b)) == 0;
    if key.is_some() && same {
        return next.run(request).await;
    }
    engine::log(&format!(
        "{}: 401: {} {} with {}",
        peer.ip(),
        request.method(),
        request.uri().path(),
        if key.is_some() { "a wrong API key" } else { "no API key" }
    ));
    chat::error(StatusCode::UNAUTHORIZED, "authentication_error", "a valid API key is required (Authorization: Bearer KEY)")
}

fn json(body: String) -> Response {
    (StatusCode::OK, [(header::CONTENT_TYPE, "application/json")], body).into_response()
}

async fn health() -> Response {
    json(r#"{"status": "ok"}"#.into())
}

async fn models(State(s): App) -> Response {
    let entry = |id: &str, name: &str, ctx: i64| {
        serde_json::json!({
            "id": id, "object": "model", "created": s.started, "owned_by": "myds4",
            "name": name, "context_length": ctx, "max_model_len": ctx,
            "max_output_tokens": ctx.min(chat::DEFAULT_MAX_TOKENS),
        })
    };
    let local = if s.qwen { format!("{} (myds4)", s.id) } else { "DeepSeek V4.1 Flash (myds4)".into() };
    let mut data = vec![entry(&s.id, &local, s.ctx)];
    data.extend(s.peers.iter().map(|p| entry(&p.id, &p.name, p.ctx)));
    json(serde_json::json!({"object": "list", "data": data}).to_string())
}

async fn status(State(s): App) -> Response {
    if s.qwen { json(serde_json::json!({"model": s.id}).to_string()) } else { json(worker::status()) }
}

/// An error answered before the response starts, and its line in the log.
fn refused(peer: &str, e: chat::Error) -> Response {
    engine::log(&format!("{peer}: {}: {}", e.status.as_u16(), e.message));
    let kind = match e.status {
        StatusCode::TOO_MANY_REQUESTS => "rate_limit_error",
        StatusCode::BAD_REQUEST | StatusCode::REQUEST_TIMEOUT | StatusCode::PAYLOAD_TOO_LARGE => "invalid_request_error",
        _ => "server_error",
    };
    chat::error(e.status, kind, &e.message)
}

async fn completions(State(s): App, ConnectInfo(peer): ConnectInfo<SocketAddr>, body: Body) -> Response {
    let peer = peer.ip().to_string();
    let body = match http::read_body(body, http::LIMITS).await {
        Ok(body) => body,
        Err(e) => return refused(&peer, e),
    };
    let arrival = s.arrivals.fetch_add(1, Ordering::Relaxed) + 1;
    engine::log(&format!("http {arrival}: arrived from {peer}, {} body bytes", body.len()));
    match route(&s, &body) {
        Err(response) => return response,
        Ok(Route::Peer(i)) => {
            engine::log(&format!("http {arrival}: to {}", s.peers[i].id));
            return peers::forward(&s.peers[i], "POST", "/v1/chat/completions", &s.raw_key, body.into()).await;
        }
        Ok(Route::Local) => {}
    }
    if s.qwen {
        return qwen_completions(arrival, peer, body.to_vec()).await;
    }
    let defaults = s.defaults.clone();
    // Rendering and the engine's tokenization take a while for long prompts.
    let from = peer.clone();
    let queued = tokio::task::spawn_blocking(move || {
        let p = chat::prepare(&body, &defaults)?;
        let note = format!(
            "http {arrival} from {from}: {} messages, {} tools, {}; {}",
            p.messages,
            p.tools,
            if p.stream { "streamed" } else { "one response" },
            p.thinking()
        );
        match worker::submit(&p.engine_request(&note)) {
            Ok((cancel, rx)) => Ok((p, cancel, rx)),
            Err(worker::SubmitError::Busy(m)) => Err(chat::Error { status: StatusCode::TOO_MANY_REQUESTS, message: m }),
            Err(worker::SubmitError::Invalid(m)) => Err(chat::Error { status: StatusCode::BAD_REQUEST, message: m }),
        }
    })
    .await;
    let (p, cancel, rx) = match queued {
        Ok(Ok(queued)) => queued,
        Ok(Err(e)) => return refused(&peer, e),
        Err(e) => return chat::error(StatusCode::INTERNAL_SERVER_ERROR, "server_error", &e.to_string()),
    };
    if p.stream { chat::stream(p, cancel, rx) } else { chat::whole(p, cancel, rx).await }
}

/// A request for this process's Qwen model.
async fn qwen_completions(arrival: u64, peer: String, body: Vec<u8>) -> Response {
    let from = peer.clone();
    let queued = tokio::task::spawn_blocking(move || {
        let p = qwen::prepare(&body, &format!("http {arrival} from {from}"))?;
        let (p, request) = split(p);
        match qwen::worker::submit(request) {
            Ok((cancel, rx)) => Ok((p, cancel, rx)),
            Err(qwen::worker::SubmitError::Busy(m)) => Err(chat::Error { status: StatusCode::TOO_MANY_REQUESTS, message: m }),
            Err(qwen::worker::SubmitError::Invalid(m)) => Err(chat::Error { status: StatusCode::BAD_REQUEST, message: m }),
        }
    })
    .await;
    let (p, cancel, rx) = match queued {
        Ok(Ok(queued)) => queued,
        Ok(Err(e)) => return refused(&peer, e),
        Err(e) => return chat::error(StatusCode::INTERNAL_SERVER_ERROR, "server_error", &e.to_string()),
    };
    if p.stream { qwen::stream(p, cancel, rx) } else { qwen::whole(p, cancel, rx).await }
}

/// The prepared request's worker part, and the rest for its response.
fn split(mut p: qwen::Prepared) -> (qwen::Prepared, qwen::worker::Request) {
    let empty = qwen::worker::Request { prompt: Vec::new(), max_tokens: 0, sampling: p.request.sampling, note: String::new() };
    let request = std::mem::replace(&mut p.request, empty);
    (p, request)
}

async fn not_found() -> Response {
    chat::error(StatusCode::NOT_FOUND, "invalid_request_error", "no such endpoint")
}

async fn use_get() -> Response {
    chat::error(StatusCode::METHOD_NOT_ALLOWED, "invalid_request_error", "use GET")
}

async fn use_post() -> Response {
    chat::error(StatusCode::METHOD_NOT_ALLOWED, "invalid_request_error", "use POST")
}

fn main() {
    let a = parse(&std::env::args().skip(1).collect::<Vec<_>>());
    if let Some(file) = &a.render {
        exit(chat::render_file(file));
    }
    let (Some(port), false) = (a.port, a.engine.model.is_empty()) else { usage() };
    let key = match &a.key_file {
        Some(f) => std::fs::read_to_string(f).unwrap_or_else(|_| die(&format!("cannot open {f}"))),
        None => std::env::var("MYDS4_API_KEY").unwrap_or_default(),
    };
    let key = key.trim_end_matches(['\n', '\r', ' ']);
    if !(16..=256).contains(&key.len()) {
        die("the API key (MYDS4_API_KEY or --api-key-file) must have 16 to 256 characters");
    }
    let host: Ipv4Addr = if a.host == "localhost" {
        Ipv4Addr::LOCALHOST
    } else {
        a.host.parse().unwrap_or_else(|_| die(&format!("--host takes an IPv4 address: {}", a.host)))
    };
    // The port is taken before the model loads, so that a busy port fails at once.
    let listener = std::net::TcpListener::bind((host, port))
        .unwrap_or_else(|e| die(&format!("cannot listen on {host}:{port}: {e}")));
    listener.set_nonblocking(true).unwrap_or_else(|e| die(&format!("cannot configure the listening socket: {e}")));

    // The other models' servers load beside this one's model.
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut peer_list = Vec::new();
    for (i, dir) in a.models.iter().enumerate().skip(1) {
        let peer_port = port.checked_add(i as u16).unwrap_or_else(|| die("no port left for another model"));
        let (id, name, ctx) = if qwen::is_qwen(dir) {
            let id = qwen::model_id(dir);
            (id.clone(), format!("{id} (myds4)"), qwen::context(dir, a.engine.ctx as usize) as i64)
        } else {
            (MODEL_ID.to_string(), "DeepSeek V4.1 Flash (myds4)".to_string(), a.engine.ctx as i64)
        };
        if let Err(e) = peers::spawn(&args, dir, &id, peer_port, key) {
            die(&e);
        }
        engine::log(&format!("{id}: its own server at 127.0.0.1:{peer_port}, for requests that name it"));
        peer_list.push(peers::Peer { id, name, port: peer_port, ctx });
    }

    let mut config = a.engine;
    let is_qwen = qwen::is_qwen(&config.model);
    // The live view takes the log from the start; until it is on the screen,
    // the lines go to stderr as well.
    let tui = a.tui && std::io::stderr().is_terminal() && !is_qwen;
    if tui {
        config.on_log = Some(view::on_log);
        config.on_die = Some(view::on_die);
    }
    let (id, ctx) = if is_qwen {
        let ctx = qwen::context(&config.model, config.ctx as usize);
        let model = qwen::load(&config.model, ctx, a.temperature, a.top_p).unwrap_or_else(|e| die(&e));
        let q = qwen::worker::Config {
            dir: config.model.clone(),
            batch: config.batch.max(1) as usize,
            kv_bytes: (config.kv_gb * (1u64 << 30) as f64) as u64,
            ctx,
        };
        if !qwen::worker::start(&q) {
            exit(2);
        }
        if a.tui {
            engine::log("--tui: the live view shows DeepSeek's engine; the log stays on stderr");
        }
        (model.id.clone(), ctx)
    } else {
        if !worker::start(&config) {
            exit(2);
        }
        (MODEL_ID.to_string(), config.ctx as usize)
    };
    let mut title = format!(
        "myds4 · {host}:{port} · {MODEL_ID} · {} experts",
        if config.experts.is_some() { "GGUF" } else { "MXFP4" }
    );
    if config.substitute > 0.0 {
        title += &format!(" · --substitute {}", config.substitute);
    }
    if config.draft > 0 {
        title += &format!(" · --draft {}", config.draft);
    }
    for p in &peer_list {
        title += &format!(" · {}", p.id);
    }
    let live = if tui { view::start(title, a.tui_colors) } else { None };
    if a.tui && !is_qwen && live.is_none() {
        engine::log("--tui: stderr is not a terminal; the log stays on stderr");
    }
    if !host.is_loopback() {
        engine::log(&format!("warning: plain HTTP on {host}: keys and conversations cross the network unencrypted"));
    }
    if is_qwen {
        engine::log(&format!("serving {id} at http://{host}:{port}/v1 (context {ctx} tokens; Qwen3.8 engine)"));
    } else {
        engine::log(&format!(
            "serving {MODEL_ID} at http://{host}:{port}/v1 (context {} tokens{}{}; Rust server on deepseek-recipe)",
            config.ctx,
            if config.experts.is_some() { ", GGUF experts" } else { "" },
            if config.draft > 0 { ", drafts" } else { "" }
        ));
        let d = &a.defaults;
        engine::log(&format!(
            "request defaults: temperature {}, top_p {}, thinking {}, effort {}, max_tokens {}",
            d.temperature,
            d.top_p,
            if d.thinking { "on" } else { "off" },
            d.effort,
            chat::DEFAULT_MAX_TOKENS.min(config.ctx as i64)
        ));
    }
    let shared = Arc::new(Shared {
        key: Sha256::digest(key.as_bytes()).into(),
        raw_key: key.to_string(),
        defaults: a.defaults.clone(),
        ctx: ctx as i64,
        started: SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |t| t.as_secs()),
        arrivals: AtomicU64::new(0),
        qwen: is_qwen,
        id,
        peers: peer_list,
    });
    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/health", get(health))
        .route("/v1/models", get(models).fallback(use_get))
        .route("/models", get(models).fallback(use_get))
        .route("/v1/status", get(status).fallback(use_get))
        .route("/status", get(status).fallback(use_get))
        .route("/v1/chat/completions", post(completions).fallback(use_post))
        .route("/chat/completions", post(completions).fallback(use_post))
        .fallback(not_found)
        .layer(middleware::from_fn_with_state(shared.clone(), authorize))
        .with_state(shared);

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap_or_else(|e| die(&format!("cannot start the runtime: {e}")));
    runtime.block_on(async move {
        let listener = tokio::net::TcpListener::from_std(listener).unwrap_or_else(|e| die(&e.to_string()));
        // The engine owns the stop (SIGINT, SIGTERM, the live view's q through
        // request_stop): the server stops accepting, and the engine ends every
        // open response.
        // A peer's server ends with the server it serves for.
        let parent = a.parent;
        let stopped = async move {
            while !engine::stopping() && parent.is_none_or(peers::alive) {
                tokio::time::sleep(Duration::from_millis(250)).await;
            }
            peers::signal(); // the streams forwarded from them end with them
        };
        http::serve(listener, app, stopped, http::LIMITS).await;
    });
    if is_qwen {
        qwen::worker::stop();
    } else {
        worker::stop();
    }
    peers::stop();
    if let Some(live) = live {
        live.finish();
    }
}
