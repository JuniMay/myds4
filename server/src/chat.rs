//! Chat Completions: a request's JSON through the compatibility steps into
//! recipe's conversion and V4.1 rendering, and the engine's events through
//! recipe's stream processor into response chunks (docs/server.md).

use std::ops::Range;
use std::pin::pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::body::Body;
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use deepseek_recipe::openai::ChatCompletionRequest;
use deepseek_recipe::openai::chat_completion::response::{
    ChatCompletionChunk, ChatCompletionResponse, ChatCompletionToolCallDelta,
};
use deepseek_recipe::request::{ConversationRequest, ConversionError, ConversionOptions, ProtocolRequest};
use deepseek_recipe::response::{Accumulate, ProtocolResponse};
use deepseek_recipe::stream::{InferenceChunk, InferenceFinishReason, PromptUsage, StreamProcessor};
use deepseek_recipe_core::messages::InputMessage;
use deepseek_recipe_core::multimodal::ImageSource;
use deepseek_recipe_core::special_tokens::{SpecialToken, SpecialTokens, SpecialTokensMap};
use deepseek_recipe_encoding::PromptEncoding;
use deepseek_recipe_encoding::v4::dsv41::DeepseekV41Encoding;
use serde_json::{Map, Value, json};
use tokio::sync::mpsc::UnboundedReceiver;
use tokio_stream::{Stream, StreamExt};

use crate::engine::{self, Event, Finish};

/// The model's name in responses and in GET /v1/models.
pub const MODEL_ID: &str = "deepseek-v4.1-flash";
/// The checkpoint's spelling of the image token (id 129264); recipe's
/// default, `<｜image｜>`, is not in its tokenizer.
const IMAGE: &str = "<｜deepseek_image｜>";
/// The model name with which recipe converts and renders for V4.1.
const RECIPE_MODEL: &str = "DeepSeek-V4.1-Flash";
/// `max_tokens` of a request that sets none, at most what the context leaves.
pub const DEFAULT_MAX_TOKENS: i64 = 262_144;

/// The server's defaults for requests that set no value.
#[derive(Clone)]
pub struct Defaults {
    pub temperature: f64,
    pub top_p: f64,
    pub seed: Option<u64>,
    pub thinking: bool,
    pub effort: u32,
}

/// A request the server answers with an error before its response starts.
pub struct Error {
    pub status: StatusCode,
    pub message: String,
}

fn bad(message: impl Into<String>) -> Error {
    Error { status: StatusCode::BAD_REQUEST, message: message.into() }
}

pub fn error_json(kind: &str, message: &str) -> String {
    json!({"error": {"message": message, "type": kind, "code": null}}).to_string()
}

pub fn error(status: StatusCode, kind: &str, message: &str) -> Response {
    (status, [(header::CONTENT_TYPE, "application/json")], error_json(kind, message)).into_response()
}

fn special_tokens() -> SpecialTokensMap {
    SpecialTokensMap { image: IMAGE.to_string(), ..SpecialTokensMap::default() }
}

fn encoding() -> &'static DeepseekV41Encoding {
    static ENCODING: OnceLock<DeepseekV41Encoding> = OnceLock::new();
    ENCODING.get_or_init(|| DeepseekV41Encoding::new().with_special_tokens(special_tokens()))
}

// ---- the compatibility steps ----

/// What the server reads from a request itself, before recipe converts the
/// rest.
struct Settings {
    temperature: f64,
    top_p: f64,
    seed: Option<u64>,
    max_tokens: i64,
    model: String,
    effort: Option<u32>,  // None: thinking off
}

/// C's atoi: optional blanks and sign, then digits.
fn atoi(s: &str) -> i64 {
    let s = s.trim_start();
    let (sign, digits) = match s.as_bytes().first() {
        Some(b'-') => (-1, &s[1..]),
        Some(b'+') => (1, &s[1..]),
        _ => (1, s),
    };
    let n = digits.bytes().take_while(u8::is_ascii_digit).fold(0i64, |n, d| n.saturating_mul(10).saturating_add((d - b'0') as i64));
    sign * n
}

/// A reasoning effort from 1 to 100: a number, or a level as recipe maps it
/// for V4.1 (minimal and low 50; medium, high and xhigh 75; ultra and max
/// 100); else the default.
fn effort_of(v: Option<&Value>, default: u32) -> u32 {
    let e = match v {
        Some(Value::Number(n)) => n.as_f64().map_or(0, |f| f.trunc() as i64),
        Some(Value::String(s)) => match s.as_str() {
            "minimal" | "low" => 50,
            "medium" | "high" | "xhigh" => 75,
            "ultra" | "max" => 100,
            s => atoi(s),
        },
        _ => 0,
    };
    if (1..=100).contains(&e) { e as u32 } else { default }
}

/// Rewrites what the C server accepts and recipe's schema refuses or reads
/// otherwise (docs/server.md, Compatibility steps), and takes out the
/// settings the server reads itself.
fn compatible(root: &mut Map<String, Value>, d: &Defaults) -> Result<Settings, Error> {
    let number = |v: Option<Value>, default: f64| match v {
        None | Some(Value::Null) => Ok(default),
        Some(Value::Number(n)) => n.as_f64().ok_or(()),
        Some(_) => Err(()),
    };
    let sampling = "temperature must be a finite nonnegative number and top_p must be in [0, 1]";
    let temperature = number(root.remove("temperature"), d.temperature).map_err(|()| bad(sampling))?;
    let top_p = number(root.remove("top_p"), d.top_p).map_err(|()| bad(sampling))?;
    if !temperature.is_finite() || temperature < 0.0 || !(0.0..=1.0).contains(&top_p) {
        return Err(bad(sampling));
    }
    let seed = match root.remove("seed") {
        None | Some(Value::Null) => d.seed,
        Some(Value::Number(n)) => Some(n.as_u64().ok_or_else(|| bad("seed must be a nonnegative integer"))?),
        Some(_) => return Err(bad("seed must be a nonnegative integer")),
    };
    let limit = |v: Option<Value>| match v {
        Some(Value::Number(n)) => n.as_f64().map(|f| f as i64),
        _ => None,
    };
    let completion = limit(root.remove("max_completion_tokens"));
    let max_tokens = completion.or(limit(root.remove("max_tokens"))).unwrap_or(DEFAULT_MAX_TOKENS);
    if max_tokens < 1 {
        return Err(bad("max_tokens must be at least 1"));
    }
    let model = root.get("model").and_then(Value::as_str).unwrap_or(MODEL_ID).to_string();
    root.insert("model".into(), RECIPE_MODEL.into());

    // A thinking setting decides; without one, an effort of "none" turns
    // thinking off and another effort turns it on.
    let thinking_field = root.remove("thinking");
    let effort_field = root.remove("reasoning_effort");
    let thinking = match &thinking_field {
        Some(Value::Bool(on)) => *on,
        Some(Value::Object(o)) => o.get("type").and_then(Value::as_str) != Some("disabled"),
        Some(Value::Null) | None => match &effort_field {
            Some(Value::String(s)) if s == "none" => false,
            Some(Value::Null) | None => d.thinking,
            Some(_) => true,
        },
        Some(_) => true,
    };
    let effort = effort_of(effort_field.as_ref(), d.effort);
    root.insert("thinking".into(), json!({"type": if thinking { "enabled" } else { "disabled" }}));
    if thinking {
        // recipe renders 50, 75 and 100; another number replaces the 75 after rendering
        let level = match effort {
            50 => "low",
            100 => "max",
            _ => "high",
        };
        root.insert("reasoning_effort".into(), level.into());
    }
    if root.get("stream") != Some(&Value::Bool(true)) {
        root.remove("stream_options");
    }

    if let Some(Value::Array(messages)) = root.get_mut("messages") {
        let mut calls: Vec<String> = Vec::new();  // the ids of the last assistant message's calls
        let mut answered = 0;
        for m in messages.iter_mut().filter_map(Value::as_object_mut) {
            match m.get("role").and_then(Value::as_str) {
                Some("developer") => {
                    m.insert("role".into(), "system".into());
                }
                Some("assistant") => {
                    calls.clear();
                    answered = 0;
                    match m.get_mut("tool_calls") {
                        Some(Value::Array(list)) if !list.is_empty() => {
                            for (k, call) in list.iter_mut().filter_map(Value::as_object_mut).enumerate() {
                                if !call.get("id").is_some_and(Value::is_string) {
                                    call.insert("id".into(), format!("call_{k}").into());
                                }
                                if !call.contains_key("type") {
                                    call.insert("type".into(), "function".into());
                                }
                                if let Some(Value::Object(f)) = call.get_mut("function") {
                                    if let Some(a) = f.get("arguments").filter(|a| !a.is_string()) {
                                        let text = a.to_string();
                                        f.insert("arguments".into(), text.into());
                                    }
                                }
                                calls.push(call["id"].as_str().unwrap_or_default().to_string());
                            }
                        }
                        Some(Value::Array(_)) | Some(Value::Null) => {
                            m.remove("tool_calls");
                        }
                        _ => {}
                    }
                }
                Some("tool") => {
                    if !m.get("tool_call_id").is_some_and(Value::is_string) {
                        if let Some(id) = calls.get(answered) {
                            m.insert("tool_call_id".into(), id.clone().into());
                        }
                    }
                    answered += 1;
                }
                _ => {}
            }
        }
    }
    Ok(Settings { temperature, top_p, seed, max_tokens, model, effort: thinking.then_some(effort) })
}

// ---- conversion and rendering ----

/// A request converted and rendered by recipe.
struct Rendered {
    request: ConversationRequest,
    prompt: String,
    specials: Vec<Range<usize>>,  // the explicit special tokens' bytes in prompt
    images: Vec<ImageSource>,     // in the order of their places
    settings: Settings,
}

fn conversion_error(e: ConversionError) -> Error {
    let status = match e {
        ConversionError::BadRequest { .. } => StatusCode::BAD_REQUEST,
        ConversionError::Internal { .. } => StatusCode::INTERNAL_SERVER_ERROR,
    };
    Error { status, message: e.to_string() }
}

/// recipe renders efforts 50, 75 and 100; for another number the server
/// converts with 75 and writes the number into the effort line, which follows
/// the begin and system tokens.
fn set_effort(prompt: &mut String, specials: &mut [Range<usize>], effort: u32) {
    let tokens = special_tokens();
    let at = tokens.token(SpecialToken::Bos).len() + tokens.token(SpecialToken::System).len();
    let line = "Reasoning Effort: 75 (range 1-100";
    if !prompt.get(at..).is_some_and(|rest| rest.starts_with(line)) {
        return;
    }
    let from = at + "Reasoning Effort: ".len();
    let number = effort.to_string();
    prompt.replace_range(from..from + 2, &number);
    let shift = |i: usize| (i as isize + number.len() as isize - 2) as usize;
    for r in specials.iter_mut().filter(|r| r.start >= from + 2) {
        *r = shift(r.start)..shift(r.end);
    }
}

fn render(root: Map<String, Value>, d: &Defaults) -> Result<Rendered, Error> {
    let mut root = root;
    let settings = compatible(&mut root, d)?;
    let request: ChatCompletionRequest =
        serde_json::from_value(Value::Object(root)).map_err(|e| bad(format!("invalid request: {e}")))?;
    let options = ConversionOptions::default()
        .with_special_tokens(special_tokens())
        .with_default_thinking_mode(d.thinking);
    let request = request.convert(options).map_err(conversion_error)?;
    let rendered = encoding().render_conversation(&request.conversation);
    let (mut prompt, mut specials) = (rendered.prompt, rendered.special_ranges);
    if let Some(effort) = settings.effort.filter(|e| ![50, 75, 100].contains(e)) {
        set_effort(&mut prompt, &mut specials, effort);
    }
    Ok(Rendered { request, prompt, specials, images: rendered.image_sources, settings })
}

/// A request ready for the engine.
pub struct Prepared {
    request: ConversationRequest,
    prompt: Vec<u8>,  // the rendered prompt, the byte 0xFF at each image's place
    images: Vec<Vec<u8>>,
    settings: Settings,
    pub stream: bool,
    pub messages: usize,
    pub tools: usize,
}

impl Prepared {
    pub fn engine_request<'a>(&'a self, note: &'a str) -> engine::Request<'a> {
        engine::Request {
            note,
            prompt: &self.prompt,
            images: &self.images,
            temperature: self.settings.temperature,
            top_p: self.settings.top_p,
            seed: self.settings.seed,
            max_tokens: self.settings.max_tokens,
        }
    }

    /// For the log: "thinking effort N" or "thinking off".
    pub fn thinking(&self) -> String {
        self.settings.effort.map_or("thinking off".into(), |e| format!("thinking effort {e}"))
    }
}

/// A request's body as the engine's request: checked, converted, rendered,
/// its images read.
pub fn prepare(body: &[u8], d: &Defaults) -> Result<Prepared, Error> {
    let Ok(Value::Object(root)) = serde_json::from_slice::<Value>(body) else {
        return Err(bad("the body is not a JSON object"));
    };
    let messages = root.get("messages").and_then(Value::as_array).map_or(0, Vec::len);
    let tools = root.get("tools").and_then(Value::as_array).map_or(0, Vec::len);
    if messages == 0 {
        return Err(bad("messages must be a non-empty array"));
    }
    let last = root["messages"].as_array().and_then(|m| m.last()).and_then(|m| m.get("role"));
    if last.and_then(Value::as_str) == Some("assistant") {
        return Err(bad("the last message must not be the assistant's"));
    }
    let r = render(root, d)?;
    let mut images = Vec::new();
    for source in &r.images {
        match source {
            ImageSource::DataUrl { data_url, .. } => {
                let url = data_url::DataUrl::process(data_url).map_err(|_| bad("an image's data: URL cannot be read"))?;
                let (bytes, _) = url.decode_to_vec().map_err(|_| bad("the image data is not base64"))?;
                images.push(bytes);
            }
            ImageSource::Bytes { data, .. } => images.push(data.clone()),
            ImageSource::Url { .. } => {
                return Err(bad(
                    "only data: URLs with base64 data are read (the server fetches no URLs and reads no files)",
                ));
            }
        }
    }
    let mut prompt = Vec::with_capacity(r.prompt.len());
    let mut at = 0;
    for s in r.specials.iter().filter(|s| &r.prompt[(*s).clone()] == IMAGE) {
        prompt.extend_from_slice(r.prompt[at..s.start].as_bytes());
        prompt.push(0xFF);
        at = s.end;
    }
    prompt.extend_from_slice(r.prompt[at..].as_bytes());
    Ok(Prepared { stream: r.request.stream, request: r.request, prompt, images, settings: r.settings, messages, tools })
}

// ---- responses ----

fn response_id() -> String {
    use rand::Rng;
    let s: String = rand::rng().sample_iter(&rand::distr::Alphanumeric).take(24).map(char::from).collect();
    format!("chatcmpl-{s}")
}

/// The engine's events as recipe's inference chunks. The end token goes on as
/// one token without text, as the C server counts it; the server's stop sets
/// `stopped` and ends the chunks without a finish.
fn inference(
    first: Event,
    mut rx: UnboundedReceiver<Event>,
    stopped: Arc<AtomicBool>,
) -> impl Stream<Item = InferenceChunk> + Send {
    async_stream::stream! {
        let mut next = Some(first);
        loop {
            let event = match next.take() {
                Some(event) => event,
                None => match rx.recv().await {
                    Some(event) => event,
                    None => break,
                },
            };
            match event {
                Event::Started { prompt, reused } => {
                    let prompt_usage = PromptUsage { prompt_tokens: prompt, prompt_cache_hit_tokens: reused };
                    yield InferenceChunk::Ready { system_fingerprint: None, prompt_usage };
                }
                Event::Tokens(token_ids) => yield InferenceChunk::Tokens { token_ids, logprobs: None },
                Event::Done { finish: Finish::Stop, .. } => {
                    yield InferenceChunk::Text { content: String::new(), token_count: 1 };
                    yield InferenceChunk::Finish { finish_reason: InferenceFinishReason::Stop };
                    break;
                }
                Event::Done { finish: Finish::Length, .. } => {
                    yield InferenceChunk::Finish { finish_reason: InferenceFinishReason::Length };
                    break;
                }
                Event::Done { finish: Finish::Shutdown, .. } => {
                    stopped.store(true, Ordering::Relaxed);
                    break;
                }
                Event::Done { .. } | Event::Refused(_) => break,
            }
        }
    }
}

/// recipe's processor over the request's events: its chunks, and whether the
/// server's stop ended them.
fn chunks(
    mut request: ConversationRequest,
    model: String,
    first: Event,
    rx: UnboundedReceiver<Event>,
) -> (impl Stream<Item = Result<ChatCompletionChunk, String>> + Send, Arc<AtomicBool>) {
    let stopped = Arc::new(AtomicBool::new(false));
    let generator = ChatCompletionRequest::chunk_generator(&request, response_id(), model);
    let options = std::mem::take(&mut request.parsing_options);
    let processor = StreamProcessor::new(generator, options).with_tokenizer(engine::Tokenizer);
    let chunks = processor
        .into_protocol_stream(inference(first, rx, stopped.clone()))
        .filter(|c| !c.as_ref().is_ok_and(empty))
        .map(|c| c.map_err(|e| e.to_string()));
    (chunks, stopped)
}

/// A chunk of an empty content delta and nothing else, such as the end
/// token's, which only adds to the usage.
fn empty(chunk: &ChatCompletionChunk) -> bool {
    !matches!(chunk.usage, Some(Some(_)))
        && chunk.choices.iter().all(|c| {
            c.finish_reason.is_none()
                && c.delta.role.is_none()
                && c.delta.tool_calls.is_none()
                && matches!(&c.delta.content, Some(Some(s)) if s.is_empty())
                && matches!(&c.delta.reasoning_content, None | Some(None))
        })
}

fn tool_calls(chunk: &ChatCompletionChunk) -> usize {
    chunk
        .choices
        .iter()
        .filter_map(|c| c.delta.tool_calls.as_ref())
        .flatten()
        .filter(|d| matches!(d, ChatCompletionToolCallDelta::Start { .. }))
        .count()
}

/// A streamed response: server-sent events from recipe's chunks, a comment
/// every 15 s while the request waits for the engine.
pub fn stream(p: Prepared, cancel: engine::Cancel, mut rx: UnboundedReceiver<Event>) -> Response {
    let id = cancel.id();
    let body = async_stream::stream! {
        let _cancel = cancel;  // dropped with the stream, polled or not
        let first = loop {
            match tokio::time::timeout(Duration::from_secs(15), rx.recv()).await {
                Ok(event) => break event,
                Err(_) => yield Ok::<String, std::io::Error>(": waiting\n\n".into()),
            }
        };
        let first = match first {
            Some(Event::Refused(message)) => {
                yield Ok(format!("data: {}\n\ndata: [DONE]\n\n", error_json("invalid_request_error", &message)));
                return;
            }
            Some(event) => event,
            None => return,
        };
        let (chunks, stopped) = chunks(p.request, p.settings.model, first, rx);
        let mut chunks = pin!(chunks);
        let mut calls = 0;
        while let Some(chunk) = chunks.next().await {
            if stopped.load(Ordering::Relaxed) {
                break;
            }
            match chunk.and_then(|c| {
                calls += tool_calls(&c);
                serde_json::to_string(&c).map_err(|e| e.to_string())
            }) {
                Ok(json) => yield Ok(format!("data: {json}\n\n")),
                Err(message) => {
                    yield Ok(format!("data: {}\n\n", error_json("server_error", &message)));
                    return;
                }
            }
        }
        if stopped.load(Ordering::Relaxed) {
            yield Ok(format!("data: {}\n\n", error_json("server_error", "the server is stopping")));
            return;
        }
        if calls > 0 {
            engine::log(&format!("req {id}: {calls} tool call{}", if calls == 1 { "" } else { "s" }));
        }
        yield Ok("data: [DONE]\n\n".into());
    };
    Response::builder()
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .body(Body::from_stream(body))
        .expect("a valid response")
}

/// A response that is not streamed: recipe's accumulated chat.completion.
pub async fn whole(p: Prepared, cancel: engine::Cancel, mut rx: UnboundedReceiver<Event>) -> Response {
    let id = cancel.id();
    let first = match rx.recv().await {
        Some(Event::Refused(message)) => return error(StatusCode::BAD_REQUEST, "invalid_request_error", &message),
        Some(event) => event,
        None => return error(StatusCode::SERVICE_UNAVAILABLE, "server_error", "the request ended without an answer"),
    };
    let created = SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs());
    let model = p.settings.model.clone();
    let (chunks, stopped) = chunks(p.request, p.settings.model, first, rx);
    let mut chunks = pin!(chunks);
    let mut response = ChatCompletionResponse::new(response_id(), model, created, 0, 0);
    let mut calls = 0;
    while let Some(chunk) = chunks.next().await {
        match chunk {
            Ok(chunk) => {
                calls += tool_calls(&chunk);
                response.accumulate(chunk);
            }
            Err(message) => return error(StatusCode::INTERNAL_SERVER_ERROR, "server_error", &message),
        }
    }
    if stopped.load(Ordering::Relaxed) {
        return error(StatusCode::SERVICE_UNAVAILABLE, "server_error", "the server is stopping");
    }
    if calls > 0 {
        engine::log(&format!("req {id}: {calls} tool call{}", if calls == 1 { "" } else { "s" }));
    }
    match serde_json::to_string(&response) {
        Ok(json) => (StatusCode::OK, [(header::CONTENT_TYPE, "application/json")], json).into_response(),
        Err(e) => error(StatusCode::INTERNAL_SERVER_ERROR, "server_error", &e.to_string()),
    }
}

// ---- --render ----

/// Replaces each image part's URL by a distinct data: URL, so that recipe
/// converts any test case; `urls` keeps the originals.
fn mark_images(v: &mut Value, urls: &mut Vec<String>) {
    match v {
        Value::Array(a) => a.iter_mut().for_each(|x| mark_images(x, urls)),
        Value::Object(o) => {
            if o.get("type").and_then(Value::as_str) == Some("image_url") {
                if let Some(Value::Object(iu)) = o.get_mut("image_url") {
                    let url = iu.get("url").and_then(Value::as_str).unwrap_or_default().to_string();
                    iu.insert("url".into(), format!("data:image/png;base64,{:08}", urls.len()).into());
                    urls.push(url);
                }
            }
            o.values_mut().for_each(|x| mark_images(x, urls));
        }
        _ => {}
    }
}

/// --render FILE: the prompt of a test case of the checkpoint's
/// encoding/tests (a message list, or an object with messages, tools,
/// thinking_mode and reasoning_effort), as the server renders it, on stdout;
/// on stderr each image's URL in the order of its place. A case that ends
/// with an assistant message renders without the next turn's start, as
/// encoding.py renders it (the server refuses such a request). Returns the
/// exit status.
pub fn render_file(path: &str) -> i32 {
    let case = match std::fs::read_to_string(path).map(|t| serde_json::from_str::<Value>(&t)) {
        Ok(Ok(case)) => case,
        _ => {
            eprintln!("myds4-server: {path}: not a JSON test case");
            return 1;
        }
    };
    let empty = Map::new();
    let (messages, fields) = match &case {
        Value::Array(_) => (case.clone(), &empty),
        Value::Object(o) => (o.get("messages").cloned().unwrap_or(Value::Null), o),
        _ => (Value::Null, &empty),
    };
    let mut root = Map::new();
    root.insert("model".into(), MODEL_ID.into());
    root.insert("messages".into(), messages);
    if let Some(tools) = fields.get("tools") {
        root.insert("tools".into(), tools.clone());
    }
    let thinking = fields.get("thinking_mode").and_then(Value::as_str) == Some("thinking");
    root.insert("thinking".into(), thinking.into());
    if let Some(effort) = fields.get("reasoning_effort") {
        root.insert("reasoning_effort".into(), effort.clone());
    }
    let mut urls = Vec::new();
    mark_images(&mut root["messages"], &mut urls);
    let defaults = Defaults { temperature: 0.0, top_p: 1.0, seed: None, thinking: false, effort: 75 };
    let r = match render(root, &defaults) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("myds4-server: {path}: {}", e.message);
            return 1;
        }
    };
    let mut prompt = r.prompt;
    if let Some(InputMessage::Assistant { prefix: false, .. }) = r.request.conversation.messages.last() {
        // the next turn's start: the assistant token and a thinking token
        if let [.., a, t] = r.specials.as_slice() {
            if a.end == t.start && t.end == prompt.len() {
                prompt.truncate(a.start);
            }
        }
    }
    print!("{prompt}");
    for source in &r.images {
        let url = match source {
            ImageSource::DataUrl { data_url, .. } => data_url
                .rsplit(',')
                .next()
                .and_then(|n| n.parse::<usize>().ok())
                .and_then(|n| urls.get(n).cloned())
                .unwrap_or_else(|| data_url.clone()),
            ImageSource::Url { url, .. } => url.clone(),
            ImageSource::Bytes { .. } => "(bytes)".into(),
        };
        eprintln!("image {url}");
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn defaults() -> Defaults {
        Defaults { temperature: 1.0, top_p: 1.0, seed: None, thinking: true, effort: 75 }
    }

    fn settings(v: Value) -> (Map<String, Value>, Settings) {
        let Value::Object(mut root) = v else { panic!("an object") };
        let s = compatible(&mut root, &defaults()).ok().expect("compatible");
        (root, s)
    }

    #[test]
    fn requests_take_the_c_servers_forms() {
        let (root, s) = settings(json!({
            "messages": [
                {"role": "developer", "content": "be brief"},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "f", "arguments": {"a": 1}}}]},
                {"role": "tool", "content": "done"},
                {"role": "assistant", "content": "ok", "tool_calls": []},
                {"role": "user", "content": "more"},
            ],
            "thinking": true, "reasoning_effort": 62, "temperature": 0, "top_p": 0, "seed": 7,
        }));
        assert_eq!((s.temperature, s.top_p, s.seed, s.effort), (0.0, 0.0, Some(7), Some(62)));
        assert_eq!(s.max_tokens, DEFAULT_MAX_TOKENS);
        assert_eq!(s.model, MODEL_ID);
        assert_eq!(root["model"], RECIPE_MODEL);
        assert_eq!(root["thinking"], json!({"type": "enabled"}));
        assert_eq!(root["reasoning_effort"], "high");
        assert!(!root.contains_key("temperature") && !root.contains_key("top_p") && !root.contains_key("seed"));
        let m = &root["messages"];
        assert_eq!(m[0]["role"], "system");
        assert_eq!(m[2]["tool_calls"][0]["id"], "call_0");
        assert_eq!(m[2]["tool_calls"][0]["type"], "function");
        assert_eq!(m[2]["tool_calls"][0]["function"]["arguments"], r#"{"a":1}"#);
        assert_eq!(m[3]["tool_call_id"], "call_0");
        assert!(m[4].get("tool_calls").is_none());
    }

    #[test]
    fn effort_none_turns_thinking_off() {
        let (root, s) = settings(json!({"messages": [{"role": "user", "content": "hi"}], "reasoning_effort": "none"}));
        assert_eq!(s.effort, None);
        assert_eq!(root["thinking"], json!({"type": "disabled"}));
        let (_, s) = settings(json!({"messages": [], "reasoning_effort": "max", "max_tokens": 5}));
        assert_eq!((s.effort, s.max_tokens), (Some(100), 5));
    }

    #[test]
    fn a_numeric_effort_is_written_into_the_prompt() {
        let request = json!({"messages": [{"role": "user", "content": "hi"}], "reasoning_effort": 62});
        let Value::Object(root) = request else { unreachable!() };
        let r = render(root, &defaults()).ok().expect("rendered");
        assert!(r.prompt.starts_with("<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: 62 (range 1-100"), "{}", r.prompt);
        for s in &r.specials {
            assert!(r.prompt[s.clone()].starts_with('<'), "{:?} in {}", s, r.prompt);
        }
    }

    /// The engine's events through recipe's processor, with the checkpoint's
    /// tokenizer (MYDS4_CHECKPOINT names its directory; skipped without it).
    #[tokio::test]
    async fn events_become_chunks() {
        let Ok(dir) = std::env::var("MYDS4_CHECKPOINT") else { return };
        engine::load_tokenizer(&dir);
        let think: Vec<u32> = engine::tokenize("Hmm.").iter().map(|&t| t as u32).collect();
        let answer: Vec<u32> = engine::tokenize("Hi there! 你好").iter().map(|&t| t as u32).collect();
        let request = json!({"messages": [{"role": "user", "content": "hi"}], "stream": true});
        let Value::Object(root) = request else { unreachable!() };
        let r = render(root, &defaults()).ok().expect("rendered");
        // the reasoning and </think> in one step, then the answer token by token
        let mut ids = think.clone();
        ids.push(128822);
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        tx.send(Event::Tokens(ids.clone())).unwrap();
        for &t in &answer {
            tx.send(Event::Tokens(vec![t])).unwrap();
        }
        tx.send(Event::Done { finish: Finish::Stop }).unwrap();
        let first = Event::Started { prompt: 10, reused: 4 };
        let (chunks, stopped) = chunks(r.request, MODEL_ID.into(), first, rx);
        let chunks: Vec<ChatCompletionChunk> = chunks.map(|c| c.expect("a chunk")).collect().await;
        assert!(!stopped.load(Ordering::Relaxed));
        let text = |f: fn(&ChatCompletionChunk) -> Option<String>| chunks.iter().filter_map(f).collect::<String>();
        let reasoning = text(|c| c.choices.first()?.delta.reasoning_content.clone().flatten());
        let content = text(|c| c.choices.first()?.delta.content.clone().flatten());
        assert_eq!(reasoning, "Hmm.");
        assert_eq!(content, "Hi there! 你好");
        let last = serde_json::to_value(chunks.last().unwrap()).unwrap();
        assert_eq!(last["choices"][0]["finish_reason"], "stop", "{last}");
        assert_eq!(last["usage"]["prompt_tokens"], 10);
        assert_eq!(last["usage"]["prompt_cache_hit_tokens"], 4);
        assert_eq!(last["usage"]["completion_tokens"], ids.len() + answer.len() + 1);
        let empty = chunks[1..chunks.len() - 1]
            .iter()
            .filter(|c| c.choices.first().is_some_and(|c| c.delta.content == Some(Some(String::new()))))
            .count();
        let all: Vec<String> = chunks.iter().map(|c| serde_json::to_string(c).unwrap()).collect();
        assert_eq!(empty, 0, "no empty deltas before the last chunk:\n{}", all.join("\n"));
    }
}
