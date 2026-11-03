//! myds4-server: the OpenAI Chat Completions API over myds4's engine, with
//! DeepSeek's deepseek-recipe for requests, prompts and output
//! (docs/server.md). The engine (src/engine/engine.c behind
//! src/engine/engine.h) owns the model, the sessions, the queue, the batching
//! and the stop on SIGINT or SIGTERM; the live view (--tui) is view.rs.

mod chat;
mod engine;
mod http;
mod view;

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
  -m DIR          the checkpoint directory
  --serve PORT    serve OpenAI Chat Completions at http://HOST:PORT/v1 (needs MYDS4_API_KEY or --api-key-file)
  --host ADDR     the IPv4 address to serve on (default 127.0.0.1)
  --api-key-file FILE  the API key clients must send (16 to 256 characters)
  --kv-cache GB   bound of the conversations' attention state kept between requests (default 8)
  --batch N       requests decoding together at most, 1 to 6 (default 4)
  --tui           a live view in the terminal (requests, speed, experts, memory, sessions) instead of the
                  log on stderr; q stops the server, and the last log lines print on exit
  -t T, --temperature T  the temperature of requests that set none (default 1; 0 is greedy)
  --top-p P       the top_p of requests that set none, 0 to 1 (default 1)
  --seed N        the seed of requests that set none (default: random for each)
  --think N       thinking effort 1-100 of requests that set none; 0 turns thinking off (default 75)
  --ctx N         context capacity in tokens, at most 1048576 (default 1048576)
  --ced off|N     approximate prefill: replay N decoder tokens, 128-1048576 (default off), plus five rewind rows
  --cache GB      expert cache budget (default: from available memory)
  --reserve GB    memory the expert cache leaves to other programs, 2-1024 (default 10)
  --cache-half-life N  score half-life in forward passes, 1-1000000 (default 50)
  --cache-read-cost   weight decode eviction priority by expert read bytes
  --experts GGUF  read the routed experts from a GGUF file or set of shards
  --substitute D  approximate: replace selected experts not in memory (see build/myds4)
  --gpu-route     decoding: the GPU looks up each layer's experts itself
  --draft N       verify up to N (1-5) DSpark drafts per pass
  --draft-cache N  with --draft: expert slots per stage, 15-128
  --draft-trim    with --draft: compute only requested final-stage rows
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
    port: Option<u16>,
    host: String,
    key_file: Option<String>,
    defaults: Defaults,
    render: Option<String>,
    tui: bool,
}

fn parse() -> Args {
    let mut a = Args {
        engine: engine::Config { ctx: 1 << 20, kv_gb: 8.0, batch: 4, ..Default::default() },
        port: None,
        host: "127.0.0.1".into(),
        key_file: None,
        defaults: Defaults { temperature: 1.0, top_p: 1.0, seed: None, thinking: true, effort: 75 },
        render: None,
        tui: false,
    };
    let args: Vec<String> = std::env::args().skip(1).collect();
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
            "--gpu-route" => e.gpu_route = true,
            "--draft-trim" => e.draft_trim = true,
            "--online-predictor" => e.online_predictor = true,
            _ => {
                let Some(v) = args.get(i + 1) else { usage() };
                i += 1;
                match flag {
                    "-m" => e.model = v.clone(),
                    "--experts" => e.experts = Some(v.clone()),
                    "--ctx" => e.ctx = value(flag, v),
                    "--ced" => e.ced = ced_depth(v).unwrap_or_else(|| die("--ced takes off or 128 to 1048576 replay tokens")),
                    "--cache" => e.cache_gb = value(flag, v),
                    "--reserve" => e.reserve_gb = value(flag, v),
                    "--cache-half-life" => e.cache_half_life = value(flag, v),
                    "--substitute" => e.substitute = value(flag, v),
                    "--draft" => e.draft = value(flag, v),
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
                    }
                    "--top-p" => {
                        let p: f64 = value(flag, v);
                        if !(0.0..=1.0).contains(&p) {
                            die("top-p must be finite and in [0, 1]");
                        }
                        a.defaults.top_p = p;
                    }
                    "--seed" => a.defaults.seed = Some(v.parse().unwrap_or_else(|_| die("--seed needs a nonnegative integer"))),
                    "--think" => {
                        let n: u32 = v.parse().ok().filter(|n| *n <= 100).unwrap_or_else(|| die("--think takes 0 (off) or 1 to 100"));
                        a.defaults.thinking = n > 0;
                        a.defaults.effort = if n > 0 { n } else { 75 };
                    }
                    "--render" => a.render = Some(v.clone()),
                    _ => usage(),
                }
            }
        }
        i += 1;
    }
    a
}

/// What the handlers share.
struct Shared {
    key: [u8; 32],  // the API key's SHA-256
    defaults: Defaults,
    ctx: i64,
    started: u64,
    arrivals: AtomicU64,
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
    let output = s.ctx.min(chat::DEFAULT_MAX_TOKENS);
    json(
        serde_json::json!({"object": "list", "data": [{
            "id": MODEL_ID, "object": "model", "created": s.started, "owned_by": "myds4",
            "name": "DeepSeek V4.1 Flash (myds4)", "context_length": s.ctx, "max_model_len": s.ctx,
            "max_output_tokens": output,
        }]})
        .to_string(),
    )
}

async fn status() -> Response {
    json(engine::status())
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
        match engine::submit(&p.engine_request(&note)) {
            Ok((cancel, rx)) => Ok((p, cancel, rx)),
            Err(engine::SubmitError::Busy(m)) => Err(chat::Error { status: StatusCode::TOO_MANY_REQUESTS, message: m }),
            Err(engine::SubmitError::Invalid(m)) => Err(chat::Error { status: StatusCode::BAD_REQUEST, message: m }),
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
    let a = parse();
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

    let mut config = a.engine;
    // The live view takes the log from the start; until it is on the screen,
    // the lines go to stderr as well.
    let tui = a.tui && std::io::stderr().is_terminal();
    if tui {
        config.on_log = Some(view::on_log);
        config.on_die = Some(view::on_die);
    }
    if !engine::start(&config) {
        exit(2);
    }
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
    let live = if tui { view::start(title) } else { None };
    if a.tui && live.is_none() {
        engine::log("--tui: stderr is not a terminal; the log stays on stderr");
    }
    if !host.is_loopback() {
        engine::log(&format!("warning: plain HTTP on {host}: keys and conversations cross the network unencrypted"));
    }
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
    let shared = Arc::new(Shared {
        key: Sha256::digest(key.as_bytes()).into(),
        defaults: a.defaults.clone(),
        ctx: config.ctx as i64,
        started: SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |t| t.as_secs()),
        arrivals: AtomicU64::new(0),
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
        let stopped = async {
            while !engine::stopping() {
                tokio::time::sleep(Duration::from_millis(250)).await;
            }
        };
        http::serve(listener, app, stopped, http::LIMITS).await;
    });
    engine::stop();
    if let Some(live) = live {
        live.finish();
    }
}
