//! myds4-server: the OpenAI Chat Completions API over myds4's engine
//!, for one model or several in one process. The engine
//! (src/engine/engine.c behind src/engine/engine.h, engine.rs) runs the
//! models and owns the stop on SIGINT or SIGTERM; the worker (worker.rs)
//! owns the engine thread, every model's queue, batching and requests'
//! events, and each model's KV prefix cache (kv.rs) the states kept between
//! requests; each model family's chat format (format/) renders its requests
//! and reads its replies; the live view (--tui) is view.rs.

mod chat;
mod config;
mod engine;
mod http;
mod kv;
mod ui;
mod worker;

use std::io::IsTerminal;
use std::net::{Ipv4Addr, SocketAddr};
use std::process::exit;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use axum::Router;
use axum::body::Body;
use axum::extract::{ConnectInfo, Path, Request, State};
use axum::http::{HeaderMap, StatusCode, header};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::chat::{Family, deepseek, qwen};
use crate::config::ModelConfig;

const USAGE: &str = "usage: myds4-server [--config FILE | -m MODEL [model options]] [server options]
  Without --config or -m, read ~/.config/myds4/server.json. See docs/usage.md.
  -m DIR          a checkpoint directory (DeepSeek V4.1 Flash, or a Qwen3.8 qwen3_5 model), its options below
  --config FILE   server settings and models in JSON; a request's model field picks its model
  --serve PORT    override the JSON port; serve OpenAI Chat Completions at http://HOST:PORT/v1
  --host ADDR     the IPv4 address to serve on (default 127.0.0.1)
  --api-key-file FILE  the API key clients must send (16 to 256 characters)
  --reserve GB    one process-wide available-memory target, 2-1024 GiB (default 10)
  --kv-cache GB   total KV states and snapshots across all models, GiB (default 8)
  --tui           a live view in the terminal (requests, speed, experts, memory, the KV cache) instead of the
                  log on stderr; q stops the server, n shows the next model, and the last log lines print on exit
  --no-tui        use the log even when the configuration enables the live view
  --tui-theme T   the live view's colors: auto (default: by the terminal's background, light without an
                  answer), light or dark
  --render FILE   print the prompt of a JSON test case of DeepSeek's encoding/tests, and exit
 a model's options:
  --id NAME       the name requests give in model (default: deepseek-v4.1-flash, or a Qwen directory's name)
  --batch N       its requests running together at most, at most its rows per pass (default 4)
  -t T, --temperature T  the temperature of requests that set none (default: generation_config.json's, else 1)
  --top-p P       the top_p of requests that set none, 0 to 1 (default: generation_config.json's, else 1)
  --top-k K       the top_k of requests that set none, 0 for all (default: generation_config.json's, else 0)
  --seed N        the seed of requests that set none (default: random for each)
  --think N       thinking effort 1-100 of requests that set none; 0 turns thinking off
  --ctx N         context capacity in tokens, at most 1048576 and the model's own (default: the most)
  --draft N       verify up to N drafts per pass, the model's own (DeepSeek's DSpark 1-5, Qwen's MTP 1)
  --lookup off|N  verify up to N (1-15) context-copy drafts; --draft supplies fallback
  DeepSeek only:
  --ced off|N     approximate prefill: replay N decoder tokens, 128-1048576 (default off), plus the rewind margin (8)
  --prefill-prune  skip decoder batches outside required history; CED and substitution must be off (default off)
  --cache GB      expert cache budget (default: from available memory)
  --cache-half-life N  score half-life in forward passes, 1-1000000 (default 50)
  --cache-read-cost   weight decode eviction priority by expert read bytes
  --experts GGUF  read the routed experts from a GGUF file or set of shards
  --substitute D  approximate: replace selected experts not in memory (see build/myds4)
  --runahead     overlap exact concurrent requests with expert reads
  --pipeline     experimental native MXFP4 pipeline for concurrent requests (default off);
                 incompatible with --runahead, --draft, --experts, online prediction and substitution
  --draft-cache N  with --draft: expert slots per stage, 15-128
  --online-predictor  CPU prefetch correction learned online
  --predictor-state FILE  online learning, its state loaded and saved
";

fn usage() -> ! {
    eprint!("{USAGE}");
    exit(2)
}

fn die(message: &str) -> ! {
    ui::leave();
    eprintln!("myds4-server: {message}");
    exit(1)
}

struct Args {
    /// -m and its options, or --config's models.
    models: Vec<ModelConfig>,
    budgets: config::Budgets,
    server: config::Server,
    render: Option<String>,
}

fn parse(args: &[String]) -> Args {
    let mut a = Args {
        models: Vec::new(),
        budgets: config::Budgets::default(),
        server: config::Server::default(),
        render: None,
    };
    let mut one: Option<ModelConfig> = None;
    let mut options: Vec<(String, Value)> = Vec::new(); // -m's options, in order
    let mut budgets: Vec<(String, Value)> = Vec::new();
    let mut server: Vec<(String, Value)> = Vec::new();
    let mut file: Option<std::path::PathBuf> = None;
    let mut i = 0;
    while i < args.len() {
        let flag = args[i].as_str();
        let switch = |name: &str| (name.to_string(), Value::Bool(true));
        match flag {
            "--tui" | "--no-tui" => server.push(("tui".into(), Value::Bool(flag == "--tui"))),
            "--cache-read-cost" | "--runahead" | "--pipeline" | "--prefill-prune"
            | "--online-predictor" => options.push(switch(&flag[2..].replace('-', "_"))),
            _ => {
                let Some(v) = args.get(i + 1) else { usage() };
                i += 1;
                match flag {
                    "-m" => {
                        if one.is_some() {
                            die("-m serves one model; several go in a --config file");
                        }
                        one = Some(ModelConfig::new(v));
                    }
                    "--config" => file = Some(v.into()),
                    "--reserve" | "--kv-cache" => {
                        let name = flag[2..].replace('-', "_");
                        let value = serde_json::from_str::<Value>(v).unwrap_or(Value::Null);
                        budgets.push((name, value));
                    }
                    "--serve" => {
                        let port: u16 = v
                            .parse()
                            .unwrap_or_else(|_| die(&format!("--serve: not a port: {v}")));
                        server.push(("port".into(), Value::from(port)));
                    }
                    "--host" | "--api-key-file" | "--tui-theme" => {
                        server.push((flag[2..].replace('-', "_"), Value::String(v.clone())))
                    }
                    "--render" => a.render = Some(v.clone()),
                    _ => {
                        let name = match flag {
                            "-t" => "temperature".to_string(),
                            f if f.starts_with("--") => f[2..].replace('-', "_"),
                            _ => usage(),
                        };
                        // numbers as numbers, the rest (off, paths, names) as text
                        let value = serde_json::from_str::<Value>(v)
                            .ok()
                            .filter(Value::is_number)
                            .unwrap_or(Value::String(v.clone()));
                        options.push((name, value));
                    }
                }
            }
        }
        i += 1;
    }
    if one.is_none() && file.is_none() && a.render.is_none() {
        file = Some(config::default_path().unwrap_or_else(|e| die(&e)));
    }
    match (one, file) {
        (Some(_), Some(_)) => {
            die("-m and --config: give one model with -m, or several in the file")
        }
        (Some(mut m), None) => {
            for (name, v) in &options {
                m.set(name, v)
                    .unwrap_or_else(|e| die(&format!("--{}: {e}", name.replace('_', "-"))));
            }
            a.models.push(m);
        }
        (None, Some(f)) => {
            if let Some((name, _)) = options.first() {
                die(&format!(
                    "--{}: with --config, each model's options are in the file",
                    name.replace('_', "-")
                ));
            }
            let c = config::read(&f).unwrap_or_else(|e| die(&e));
            a.models = c.models;
            a.budgets = c.budgets;
            a.server = c.server;
        }
        (None, None) => {}
    }
    for (name, value) in budgets {
        a.budgets.set(&name, &value).unwrap_or_else(|e| die(&e));
    }
    for (name, value) in server {
        a.server.set(&name, &value).unwrap_or_else(|e| die(&e));
    }
    a
}

#[test]
fn one_model_takes_its_options_from_the_command_line() {
    let pipeline = parse(&["-m", "/m/d", "--pipeline", "--serve", "8008"].map(String::from));
    assert!(pipeline.models[0].engine.deepseek().unwrap().pipeline);
    assert!(!pipeline.models[0].engine.deepseek().unwrap().runahead);
    let a = parse(
        &[
            "-m",
            "/m/d",
            "--prefill-prune",
            "--lookup",
            "15",
            "--draft",
            "2",
            "--ced",
            "off",
            "--serve",
            "8008",
            "--tui",
        ]
        .map(String::from),
    );
    let m = &a.models[0];
    assert!(m.engine.deepseek().unwrap().prefill_prune);
    assert_eq!(
        (
            m.engine.lookup,
            m.engine.draft,
            m.engine.deepseek().unwrap().ced
        ),
        (15, 2, 0)
    );
    assert_eq!(
        (m.temperature, m.top_p, a.budgets.kv_gb, m.batch),
        (None, None, 8.0, 4)
    );
    assert_eq!((a.server.port, a.server.tui), (Some(8008), true));
    let a = parse(
        &[
            "-m",
            "/m/q",
            "-t",
            "0.6",
            "--top-k",
            "20",
            "--id",
            "qwen",
            "--kv-cache",
            "4",
        ]
        .map(String::from),
    );
    let m = &a.models[0];
    assert_eq!(
        (m.temperature, m.top_k, m.id.as_deref(), a.budgets.kv_gb),
        (Some(0.6), Some(20), Some("qwen"), 4.0)
    );
}

#[test]
fn command_line_server_settings_override_the_file_in_either_order() {
    let file = format!(
        "{}/tests/fixtures/default-home/.config/myds4/server.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let a = parse(&["--config".into(), file.clone()]);
    assert_eq!(
        (a.server.port, a.server.tui, a.server.tui_theme),
        (Some(8008), true, ui::Colors::Dark)
    );
    assert_eq!(a.server.api_key_file.as_deref(), Some("/dev/null"));
    for before in [false, true] {
        let flags: Vec<String> = [
            "--serve",
            "9008",
            "--host",
            "127.0.0.2",
            "--api-key-file",
            "override-key",
            "--no-tui",
            "--tui-theme",
            "light",
            "--reserve",
            "12",
            "--kv-cache",
            "16",
        ]
        .map(String::from)
        .into();
        let config = vec!["--config".into(), file.clone()];
        let a = parse(&if before {
            [flags, config].concat()
        } else {
            [config, flags].concat()
        });
        assert_eq!(
            (
                a.server.port,
                a.server.host.as_str(),
                a.server.api_key_file.as_deref()
            ),
            (Some(9008), "127.0.0.2", Some("override-key"))
        );
        assert_eq!(
            (a.server.tui, a.server.tui_theme),
            (false, ui::Colors::Light)
        );
        assert_eq!((a.budgets.reserve_gb, a.budgets.kv_gb), (12.0, 16.0));
    }
}

/// A served model: its name, its family's chat format and its settings.
struct Served {
    id: String,
    name: String,
    ctx: usize,
    tokenizer: engine::Tokenizer,
    format: Format,
}

enum Format {
    DeepSeek(deepseek::Defaults),
    Qwen(qwen::Settings),
}

/// What the handlers share.
struct Shared {
    key: [u8; 32], // the API key's SHA-256
    started: u64,
    arrivals: AtomicU64,
    models: Vec<Served>,
}

/// The model a request names (by name, in any case): the only model when it
/// names none, or names an unknown one while one model is served; an error
/// when it names an unknown one among several.
fn route(s: &Shared, body: &[u8]) -> Result<usize, Response> {
    #[derive(serde::Deserialize)]
    struct Named {
        model: Option<String>,
    }
    let name = serde_json::from_slice::<Named>(body)
        .ok()
        .and_then(|n| n.model);
    if let Some(name) = &name
        && let Some(i) = s
            .models
            .iter()
            .position(|m| name.eq_ignore_ascii_case(&m.id))
    {
        return Ok(i);
    }
    if s.models.len() == 1 {
        return Ok(0);
    }
    let known: Vec<&str> = s.models.iter().map(|m| m.id.as_str()).collect();
    let message = match name {
        Some(name) => format!(
            "no model {name:?} here; the models are {}",
            known.join(", ")
        ),
        None => format!(
            "the request names no model; the models are {}",
            known.join(", ")
        ),
    };
    Err(chat::error(
        StatusCode::NOT_FOUND,
        "invalid_request_error",
        &message,
    ))
}

type App = State<Arc<Shared>>;

/// Every request but GET /health carries the key, as Authorization: Bearer
/// KEY or x-api-key: KEY. The digests are compared in constant time, so the
/// time taken tells nothing about the key.
async fn authorize(
    State(s): App,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    request: Request,
    next: Next,
) -> Response {
    if matches!(request.uri().path(), "/health" | "/v1/health") {
        return next.run(request).await;
    }
    let headers: &HeaderMap = request.headers();
    let key = headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| {
            v.get(..7)
                .filter(|b| b.eq_ignore_ascii_case("bearer "))
                .map(|_| v[7..].trim_start())
        })
        .or_else(|| headers.get("x-api-key").and_then(|v| v.to_str().ok()));
    let given: [u8; 32] = Sha256::digest(key.unwrap_or_default().as_bytes()).into();
    let same = given
        .iter()
        .zip(s.key.iter())
        .fold(0u8, |d, (a, b)| d | (a ^ b))
        == 0;
    if key.is_some() && same {
        return next.run(request).await;
    }
    engine::log(&format!(
        "{}: 401: {} {} with {}",
        peer.ip(),
        request.method(),
        request.uri().path(),
        if key.is_some() {
            "a wrong API key"
        } else {
            "no API key"
        }
    ));
    chat::error(
        StatusCode::UNAUTHORIZED,
        "authentication_error",
        "a valid API key is required (Authorization: Bearer KEY)",
    )
}

fn json(body: String) -> Response {
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "application/json")],
        body,
    )
        .into_response()
}

async fn health() -> Response {
    json(r#"{"status": "ok"}"#.into())
}

/// A model's object as GET /v1/models lists it: OpenAI's fields, and the
/// context's length as servers commonly add it.
fn model_object(s: &Shared, m: &Served) -> Value {
    json!({
        "id": m.id, "object": "model", "created": s.started, "owned_by": "myds4",
        "name": m.name, "context_length": m.ctx, "max_model_len": m.ctx,
        "max_output_tokens": (m.ctx as i64).min(chat::DEFAULT_MAX_TOKENS),
    })
}

async fn models(State(s): App) -> Response {
    let data: Vec<Value> = s.models.iter().map(|m| model_object(&s, m)).collect();
    json(json!({"object": "list", "data": data}).to_string())
}

/// GET /v1/models/{id}: the model's object, and its live status under
/// "status" (what it is doing, its requests, KV prefix cache, counters,
/// expert cache and the process's memory).
async fn model(State(s): App, Path(id): Path<String>) -> Response {
    let Some(i) = s.models.iter().position(|m| id.eq_ignore_ascii_case(&m.id)) else {
        return chat::error(
            StatusCode::NOT_FOUND,
            "invalid_request_error",
            &format!("no model {id:?} here"),
        );
    };
    let mut v = model_object(&s, &s.models[i]);
    v["status"] = worker::status(i);
    json(v.to_string())
}

/// An error answered before the response starts, and its line in the log.
fn refused(peer: &str, e: chat::Error) -> Response {
    engine::log(&format!("{peer}: {}: {}", e.status.as_u16(), e.message));
    let kind = match e.status {
        StatusCode::TOO_MANY_REQUESTS => "rate_limit_error",
        StatusCode::BAD_REQUEST | StatusCode::REQUEST_TIMEOUT | StatusCode::PAYLOAD_TOO_LARGE => {
            "invalid_request_error"
        }
        _ => "server_error",
    };
    chat::error(e.status, kind, &e.message)
}

fn submitted(
    r: worker::Request,
) -> Result<
    (
        worker::Cancel,
        tokio::sync::mpsc::UnboundedReceiver<worker::Event>,
    ),
    chat::Error,
> {
    worker::submit(r).map_err(|e| match e {
        worker::SubmitError::Busy(m) => chat::Error {
            status: StatusCode::TOO_MANY_REQUESTS,
            message: m,
        },
        worker::SubmitError::Invalid(m) => chat::Error {
            status: StatusCode::BAD_REQUEST,
            message: m,
        },
    })
}

async fn completions(
    State(s): App,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    body: Body,
) -> Response {
    let peer = peer.ip().to_string();
    let body = match http::read_body(body, http::LIMITS).await {
        Ok(body) => body,
        Err(e) => return refused(&peer, e),
    };
    let arrival = s.arrivals.fetch_add(1, Ordering::Relaxed) + 1;
    engine::log(&format!(
        "http {arrival}: arrived from {peer}, {} body bytes",
        body.len()
    ));
    let i = match route(&s, &body) {
        Ok(i) => i,
        Err(response) => return response,
    };
    // Rendering and tokenization take a while for long prompts.
    let (from, shared) = (peer.clone(), s.clone());
    let queued = tokio::task::spawn_blocking(move || match &shared.models[i].format {
        Format::DeepSeek(d) => {
            let p = deepseek::prepare(&body, d, shared.models[i].tokenizer)?;
            let note = format!(
                "http {arrival} from {from}: {} messages, {} tools, {}; {}",
                p.messages,
                p.tools,
                if p.stream { "streamed" } else { "one response" },
                p.thinking()
            );
            let (cancel, rx) = submitted(p.engine_request(i, note)?)?;
            Ok::<_, chat::Error>((Prepared::DeepSeek(p), cancel, rx))
        }
        Format::Qwen(q) => {
            let mut p = qwen::prepare(&body, &format!("http {arrival} from {from}"), q, i)?;
            let request = std::mem::replace(
                &mut p.request,
                worker::Request {
                    model: i,
                    prompt: Vec::new(),
                    images: Vec::new(),
                    sampling: q.sampling,
                    max_tokens: 0,
                    note: String::new(),
                },
            );
            let (cancel, rx) = submitted(request)?;
            Ok((Prepared::Qwen(p), cancel, rx))
        }
    })
    .await;
    match queued {
        Ok(Ok((Prepared::DeepSeek(p), cancel, rx))) => {
            if p.stream {
                deepseek::stream(p, cancel, rx)
            } else {
                deepseek::whole(p, cancel, rx).await
            }
        }
        Ok(Ok((Prepared::Qwen(p), cancel, rx))) => {
            if p.stream {
                qwen::stream(p, cancel, rx)
            } else {
                qwen::whole(p, cancel, rx).await
            }
        }
        Ok(Err(e)) => refused(&peer, e),
        Err(e) => chat::error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "server_error",
            &e.to_string(),
        ),
    }
}

enum Prepared {
    DeepSeek(deepseek::Prepared),
    Qwen(qwen::Prepared),
}

async fn not_found() -> Response {
    chat::error(
        StatusCode::NOT_FOUND,
        "invalid_request_error",
        "no such endpoint",
    )
}

async fn use_get() -> Response {
    chat::error(
        StatusCode::METHOD_NOT_ALLOWED,
        "invalid_request_error",
        "use GET",
    )
}

async fn use_post() -> Response {
    chat::error(
        StatusCode::METHOD_NOT_ALLOWED,
        "invalid_request_error",
        "use POST",
    )
}

fn main() {
    let a = parse(&std::env::args().skip(1).collect::<Vec<_>>());
    if let Some(file) = &a.render {
        exit(deepseek::render_file(file));
    }
    let (Some(port), false) = (a.server.port, a.models.is_empty()) else {
        usage()
    };
    let key = match &a.server.api_key_file {
        Some(f) => std::fs::read_to_string(f).unwrap_or_else(|_| die(&format!("cannot open {f}"))),
        None => std::env::var("MYDS4_API_KEY").unwrap_or_default(),
    };
    let key = key.trim_end_matches(['\n', '\r', ' ']);
    if !(16..=256).contains(&key.len()) {
        die("the API key (MYDS4_API_KEY or --api-key-file) must have 16 to 256 characters");
    }
    let host: Ipv4Addr = if a.server.host == "localhost" {
        Ipv4Addr::LOCALHOST
    } else {
        a.server
            .host
            .parse()
            .unwrap_or_else(|_| die(&format!("host takes an IPv4 address: {}", a.server.host)))
    };
    // The port is taken before the models load, so that a busy port fails at once.
    let listener = std::net::TcpListener::bind((host, port))
        .unwrap_or_else(|e| die(&format!("cannot listen on {host}:{port}: {e}")));
    listener
        .set_nonblocking(true)
        .unwrap_or_else(|e| die(&format!("cannot configure the listening socket: {e}")));

    // Each model's family, name and context.
    let mut served = Vec::new();
    let mut families = Vec::new();
    for m in &a.models {
        let dir = &m.engine.model;
        let family = Family::of(dir).unwrap_or_else(|e| die(&e));
        let mut e = m.engine.clone();
        e.configure(family).unwrap_or_else(|error| die(&error));
        let id = m.id.clone().unwrap_or_else(|| match family {
            Family::DeepSeek => deepseek::MODEL_ID.into(),
            Family::Qwen => qwen::model_id(dir),
        });
        if family == Family::Qwen {
            e.ctx = qwen::context(dir, e.ctx as usize) as i32;
        }
        if served
            .iter()
            .any(|s: &worker::Served| s.id.eq_ignore_ascii_case(&id))
        {
            die(&format!("two models named {id}: give one an id"));
        }
        served.push(worker::Served {
            id,
            engine: e,
            batch: m.batch,
        });
        families.push(family);
    }
    // The live view takes the log from the start; until it is on the screen,
    // the lines go to stderr as well.
    let tui = a.server.tui && std::io::stderr().is_terminal();
    let (on_log, on_die): (
        Option<extern "C" fn(*const std::ffi::c_char)>,
        Option<extern "C" fn()>,
    ) = if tui {
        (Some(ui::on_log), Some(ui::on_die))
    } else {
        (None, None)
    };
    if !worker::start(served.clone(), a.budgets.clone(), on_log, on_die) {
        exit(2);
    }
    let loaded = worker::models();
    let mut served_models = Vec::new();
    for ((m, s), family) in a.models.iter().zip(&served).zip(&families) {
        let (ctx, id) = (s.engine.ctx as usize, s.id.clone());
        let format = match family {
            Family::DeepSeek => Format::DeepSeek(deepseek::Defaults {
                id: id.clone(),
                sampling: worker::Sampling {
                    temperature: m.temperature.unwrap_or(1.0),
                    top_p: m.top_p.unwrap_or(1.0),
                    top_k: m.top_k.unwrap_or(0),
                    seed: m.seed,
                },
                thinking: m.think != Some(0),
                effort: m.think.filter(|&n| n > 0).unwrap_or(75),
            }),
            Family::Qwen => {
                let mut sampling = qwen::sampling(&s.engine.model, m.temperature, m.top_p, m.top_k);
                sampling.seed = m.seed;
                let i = served_models.len();
                Format::Qwen(qwen::Settings {
                    id: id.clone(),
                    ctx,
                    tokenizer: loaded[i].1.tokenizer(),
                    sampling,
                    think: m.think,
                })
            }
        };
        let name = match family {
            Family::DeepSeek => "DeepSeek V4.1 Flash (myds4)".to_string(),
            Family::Qwen => format!("{id} (myds4)"),
        };
        engine::log(&format!(
            "serving {id} at http://{host}:{port}/v1 (context {ctx} tokens{}{})",
            if s.engine.deepseek().is_some_and(|d| d.experts.is_some()) {
                ", GGUF experts"
            } else {
                ""
            },
            if loaded[served_models.len()].1.draft_block() > 0 {
                ", drafts"
            } else {
                ""
            }
        ));
        served_models.push(Served {
            id,
            name,
            ctx,
            tokenizer: loaded[served_models.len()].1.tokenizer(),
            format,
        });
    }
    let title = format!(
        "myds4 · {host}:{port} · {}",
        served_models
            .iter()
            .map(|m| m.id.as_str())
            .collect::<Vec<_>>()
            .join(" · ")
    );
    let live = if tui {
        ui::start(title, a.server.tui_theme)
    } else {
        None
    };
    if a.server.tui && live.is_none() {
        engine::log("--tui: stderr is not a terminal; the log stays on stderr");
    }
    if !host.is_loopback() {
        engine::log(&format!(
            "warning: plain HTTP on {host}: keys and conversations cross the network unencrypted"
        ));
    }
    let shared = Arc::new(Shared {
        key: Sha256::digest(key.as_bytes()).into(),
        started: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |t| t.as_secs()),
        arrivals: AtomicU64::new(0),
        models: served_models,
    });
    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/health", get(health))
        .route("/v1/models", get(models).fallback(use_get))
        .route("/models", get(models).fallback(use_get))
        .route("/v1/models/{id}", get(model).fallback(use_get))
        .route("/models/{id}", get(model).fallback(use_get))
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
        let listener =
            tokio::net::TcpListener::from_std(listener).unwrap_or_else(|e| die(&e.to_string()));
        // The engine owns the stop (SIGINT, SIGTERM, the live view's q through
        // request_stop): the server stops accepting, and the engine ends every
        // open response.
        let stopped = async move {
            while !engine::stopping() {
                tokio::time::sleep(std::time::Duration::from_millis(250)).await;
            }
        };
        http::serve(listener, app, stopped, http::LIMITS).await;
    });
    worker::stop();
    if let Some(live) = live {
        live.finish();
    }
}
