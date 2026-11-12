//! DeepSeek V4.1's chat format: a Chat Completions or Responses request's
//! JSON through the compatibility steps into recipe's conversion and V4.1
//! rendering, its prompt tokens from the model's tokenizer, and the worker's
//! events through recipe's stream processor into the API's response chunks
//! or events.

use std::ops::Range;
use std::pin::pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::body::Body;
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use deepseek_recipe::openai::chat_completion::response::{
    ChatCompletionChunk, ChatCompletionToolCallDelta,
};
use deepseek_recipe::openai::responses::response::{ResponsesOutputItem, ResponsesStreamEvent};
use deepseek_recipe::openai::{ChatCompletionRequest, ResponsesRequest};
use deepseek_recipe::request::{
    ConversationRequest, ConversionError, ConversionOptions, ProtocolRequest,
};
use deepseek_recipe::response::{Accumulate, ProtocolResponse};
use deepseek_recipe::stream::{
    ChunkGenerator, InferenceChunk, InferenceFinishReason, PromptUsage, StreamProcessor,
};
use deepseek_recipe_core::messages::InputMessage;
use deepseek_recipe_core::multimodal::ImageSource;
use deepseek_recipe_core::special_tokens::{SpecialToken, SpecialTokens, SpecialTokensMap};
use deepseek_recipe_encoding::PromptEncoding;
use deepseek_recipe_encoding::v4::dsv41::DeepseekV41Encoding;
use serde::de::DeserializeOwned;
use serde_json::{Map, Value, json};
use tokio::sync::mpsc::UnboundedReceiver;
use tokio_stream::{Stream, StreamExt};

use super::{Api, DEFAULT_MAX_TOKENS, Error, bad, error, error_json, random_id};
use crate::engine;
use crate::worker::{self, Event, Finish, Sampling};

/// A DeepSeek checkpoint's name in responses and in GET /v1/models, unless
/// the server's configuration names it otherwise.
pub const MODEL_ID: &str = "deepseek-v4.1-flash";
/// The checkpoint's spelling of the image token (id 129264); recipe's
/// default, `<｜image｜>`, is not in its tokenizer.
pub const IMAGE: &str = "<｜deepseek_image｜>";
/// The model name with which recipe converts and renders for V4.1.
const RECIPE_MODEL: &str = "DeepSeek-V4.1-Flash";
/// The server's defaults for requests that set no value, and the model's name.
#[derive(Clone)]
pub struct Defaults {
    pub id: String,
    pub sampling: Sampling,
    pub thinking: bool,
    pub effort: u32,
}

fn special_tokens() -> SpecialTokensMap {
    SpecialTokensMap {
        image: IMAGE.to_string(),
        ..SpecialTokensMap::default()
    }
}

fn encoding() -> &'static DeepseekV41Encoding {
    static ENCODING: OnceLock<DeepseekV41Encoding> = OnceLock::new();
    ENCODING.get_or_init(|| DeepseekV41Encoding::new().with_special_tokens(special_tokens()))
}

// ---- the compatibility steps ----

/// What the server reads from a request itself, before recipe converts the
/// rest.
struct Settings {
    sampling: Sampling,
    max_tokens: i64,
    model: String,
    effort: Option<u32>, // None: thinking off
}

/// C's atoi: optional blanks and sign, then digits.
fn atoi(s: &str) -> i64 {
    let s = s.trim_start();
    let (sign, digits) = match s.as_bytes().first() {
        Some(b'-') => (-1, &s[1..]),
        Some(b'+') => (1, &s[1..]),
        _ => (1, s),
    };
    let n = digits
        .bytes()
        .take_while(u8::is_ascii_digit)
        .fold(0i64, |n, d| {
            n.saturating_mul(10).saturating_add((d - b'0') as i64)
        });
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
    if (1..=100).contains(&e) {
        e as u32
    } else {
        default
    }
}

/// Rewrites the request forms the server accepts that recipe's schema refuses
/// or reads otherwise, and takes out the
/// settings the server reads itself.
fn compatible(root: &mut Map<String, Value>, d: &Defaults) -> Result<Settings, Error> {
    let sampling = super::sampling(root, d.sampling)?;
    for key in ["temperature", "top_p", "top_k", "seed"] {
        root.remove(key);
    }
    let max_tokens = super::completion_limit(root, DEFAULT_MAX_TOKENS)?;
    super::stop_sequences(root)?;
    root.remove("max_completion_tokens");
    root.remove("max_tokens");
    let model = root
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or(&d.id)
        .to_string();
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
    root.insert(
        "thinking".into(),
        json!({"type": if thinking { "enabled" } else { "disabled" }}),
    );
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
        let mut calls: Vec<String> = Vec::new(); // the ids of the last assistant message's calls
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
                            for (k, call) in
                                list.iter_mut().filter_map(Value::as_object_mut).enumerate()
                            {
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
    Ok(Settings {
        sampling,
        max_tokens,
        model,
        effort: thinking.then_some(effort),
    })
}

/// A Responses request's settings that the server reads itself: sampling,
/// `max_output_tokens` and the reasoning effort, written back as the level
/// recipe renders (as for Chat Completions). The server keeps no responses,
/// so a request that continues a stored response or conversation is refused.
fn responses_compatible(root: &mut Map<String, Value>, d: &Defaults) -> Result<Settings, Error> {
    for key in ["previous_response_id", "conversation"] {
        if root.get(key).is_some_and(|v| !v.is_null()) {
            return Err(bad(format!(
                "{key} is not supported: the server keeps no responses, so send the whole conversation in input"
            )));
        }
    }
    let sampling = super::sampling(root, d.sampling)?;
    let max_tokens = match root.get("max_output_tokens") {
        None | Some(Value::Null) => DEFAULT_MAX_TOKENS,
        Some(value) => value
            .as_i64()
            .filter(|&n| n > 0)
            .ok_or_else(|| bad("max_output_tokens must be a positive integer"))?,
    };
    let model = root
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or(&d.id)
        .to_string();
    root.insert("model".into(), RECIPE_MODEL.into());

    // An effort of "none" turns thinking off and another effort, a level or
    // a number from 1 to 100, turns it on; without one, the model's default.
    let effort_field = root
        .get("reasoning")
        .and_then(|r| r.get("effort"))
        .filter(|e| !e.is_null())
        .cloned();
    let thinking = match &effort_field {
        None => d.thinking,
        Some(Value::String(s)) => s != "none",
        Some(_) => true,
    };
    let effort = effort_of(effort_field.as_ref(), d.effort);
    if thinking {
        let level = match effort {
            50 => "low",
            100 => "max",
            _ => "high",
        };
        match root.get_mut("reasoning") {
            Some(Value::Object(reasoning)) => {
                reasoning.insert("effort".into(), level.into());
            }
            _ => {
                root.insert("reasoning".into(), json!({"effort": level}));
            }
        }
    }
    Ok(Settings {
        sampling,
        max_tokens,
        model,
        effort: thinking.then_some(effort),
    })
}

/// The declared tools that recipe's Responses conversion drops: web search,
/// which the server cannot run, and tool types it does not know.
fn dropped_tools(root: &Map<String, Value>) -> usize {
    let known = ["function", "namespace", "custom", "tool_search"];
    root.get("tools")
        .and_then(Value::as_array)
        .map_or(0, |tools| {
            tools
                .iter()
                .filter(|t| !t.get("type").and_then(Value::as_str).is_some_and(|k| known.contains(&k)))
                .count()
        })
}

// ---- conversion and rendering ----

/// A request converted and rendered by recipe.
struct Rendered {
    request: ConversationRequest,
    prompt: String,
    specials: Vec<Range<usize>>, // the explicit special tokens' bytes in prompt
    images: Vec<ImageSource>,    // in the order of their places
    settings: Settings,
}

fn conversion_error(e: ConversionError) -> Error {
    let status = match e {
        ConversionError::BadRequest { .. } => StatusCode::BAD_REQUEST,
        ConversionError::Internal { .. } => StatusCode::INTERNAL_SERVER_ERROR,
    };
    Error {
        status,
        message: e.to_string(),
    }
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

/// The request through the API's compatibility steps, recipe's conversion
/// and V4.1 rendering.
fn render(root: Map<String, Value>, d: &Defaults, api: Api) -> Result<Rendered, Error> {
    let mut root = root;
    match api {
        Api::Chat => {
            let settings = compatible(&mut root, d)?;
            render_as::<ChatCompletionRequest>(root, settings, d)
        }
        Api::Responses => {
            let settings = responses_compatible(&mut root, d)?;
            render_as::<ResponsesRequest>(root, settings, d)
        }
    }
}

fn render_as<P: ProtocolRequest + DeserializeOwned>(
    root: Map<String, Value>,
    settings: Settings,
    d: &Defaults,
) -> Result<Rendered, Error> {
    let request: P = serde_json::from_value(Value::Object(root))
        .map_err(|e| bad(format!("invalid request: {e}")))?;
    let options = ConversionOptions::default()
        .with_special_tokens(special_tokens())
        .with_default_thinking_mode(d.thinking);
    let request = request.convert(options).map_err(conversion_error)?;
    let rendered = encoding().render_conversation(&request.conversation);
    let (mut prompt, mut specials) = (rendered.prompt, rendered.special_ranges);
    if let Some(effort) = settings.effort.filter(|e| ![50, 75, 100].contains(e)) {
        set_effort(&mut prompt, &mut specials, effort);
    }
    Ok(Rendered {
        request,
        prompt,
        specials,
        images: rendered.image_sources,
        settings,
    })
}

/// A request ready for the engine.
pub struct Prepared {
    tokenizer: engine::Tokenizer,
    request: ConversationRequest,
    prompt: Vec<u8>, // the rendered prompt, the byte 0xFF at each image's place
    images: Vec<Vec<u8>>,
    settings: Settings,
    pub api: Api,
    pub stream: bool,
    /// The request's messages, or its input items.
    pub messages: usize,
    pub tools: usize,
    /// Of the tools, those that conversion dropped (Responses web search and
    /// unknown types).
    pub dropped_tools: usize,
}

impl Prepared {
    /// The worker's request for model `model`: the rendered prompt's tokens,
    /// each image's place the image token, as many as the images.
    pub fn engine_request(&self, model: usize, note: String) -> Result<worker::Request, Error> {
        let prompt = self.tokenizer.tokenize_prompt(&self.prompt);
        let image = self.tokenizer.special(IMAGE);
        let places = prompt.iter().filter(|&&t| Some(t) == image).count();
        if prompt.is_empty() {
            return Err(bad("the prompt is empty"));
        }
        if places != self.images.len() {
            return Err(bad(format!(
                "the prompt has {places} image places and {} images",
                self.images.len()
            )));
        }
        let sampling = self.settings.sampling;
        Ok(worker::Request {
            model,
            prompt,
            media: self.images.clone(),
            sampling,
            max_tokens: self.settings.max_tokens,
            note,
        })
    }

    /// For the log: "thinking effort N" or "thinking off".
    pub fn thinking(&self) -> String {
        self.settings
            .effort
            .map_or("thinking off".into(), |e| format!("thinking effort {e}"))
    }
}

/// A request's body as the engine's request: checked, converted, rendered,
/// its images read; `tokenizer` is the model's.
pub fn prepare(
    body: &[u8],
    d: &Defaults,
    tokenizer: engine::Tokenizer,
    api: Api,
) -> Result<Prepared, Error> {
    let Ok(Value::Object(root)) = serde_json::from_slice::<Value>(body) else {
        return Err(bad("the body is not a JSON object"));
    };
    let tools = root
        .get("tools")
        .and_then(Value::as_array)
        .map_or(0, Vec::len);
    let (messages, dropped_tools) = match api {
        Api::Chat => {
            let messages = root
                .get("messages")
                .and_then(Value::as_array)
                .map_or(0, Vec::len);
            if messages == 0 {
                return Err(bad("messages must be a non-empty array"));
            }
            let last = root["messages"]
                .as_array()
                .and_then(|m| m.last())
                .and_then(|m| m.get("role"));
            if last.and_then(Value::as_str) == Some("assistant") {
                return Err(bad("the last message must not be the assistant's"));
            }
            (messages, 0)
        }
        Api::Responses => {
            let items = match root.get("input") {
                Some(Value::String(s)) if !s.is_empty() => 1,
                Some(Value::Array(items)) if !items.is_empty() => items.len(),
                _ => return Err(bad("input must be a non-empty string or array")),
            };
            (items, dropped_tools(&root))
        }
    };
    let r = render(root, d, api)?;
    let mut images = Vec::new();
    for source in &r.images {
        match source {
            ImageSource::DataUrl { data_url, .. } => {
                images.push(super::image_data(data_url)?);
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
    for s in r
        .specials
        .iter()
        .filter(|s| &r.prompt[(*s).clone()] == IMAGE)
    {
        prompt.extend_from_slice(r.prompt[at..s.start].as_bytes());
        prompt.push(0xFF);
        at = s.end;
    }
    prompt.extend_from_slice(r.prompt[at..].as_bytes());
    Ok(Prepared {
        tokenizer,
        stream: r.request.stream,
        request: r.request,
        prompt,
        images,
        settings: r.settings,
        api,
        messages,
        tools,
        dropped_tools,
    })
}

// ---- responses ----

/// The engine's events as recipe's inference chunks. The end token goes on as
/// one token without text, as the engine counts it; the server's stop sets
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
                Event::Done { .. } | Event::Refused(_) | Event::Failed(_) => break,
            }
        }
    }
}

/// A protocol's response chunk: a chat completion chunk or a Responses event.
type Chunk<P> =
    <<<P as ProtocolRequest>::Response as ProtocolResponse>::ChunkGenerator as ChunkGenerator>::Chunk;

/// What differs between the APIs' output.
trait Output: ProtocolRequest {
    /// A new response's id.
    fn id() -> String;
    /// Whether the chunk is sent.
    fn keep(chunk: &Chunk<Self>) -> bool;
    /// The tool calls the chunk starts.
    fn calls(chunk: &Chunk<Self>) -> usize;
    /// The stream's last event when the request fails: `kind` and `message`;
    /// `next` is the stream's next sequence number.
    fn failure(kind: &str, message: &str, id: &str, model: &str, next: usize) -> String;
}

impl Output for ChatCompletionRequest {
    fn id() -> String {
        random_id("chatcmpl-")
    }

    /// Not a chunk of an empty content delta and nothing else, such as the
    /// end token's, which only adds to the usage.
    fn keep(chunk: &ChatCompletionChunk) -> bool {
        matches!(chunk.usage, Some(Some(_)))
            || !chunk.choices.iter().all(|c| {
                c.finish_reason.is_none()
                    && c.delta.role.is_none()
                    && c.delta.tool_calls.is_none()
                    && matches!(&c.delta.content, Some(Some(s)) if s.is_empty())
                    && matches!(&c.delta.reasoning_content, None | Some(None))
            })
    }

    fn calls(chunk: &ChatCompletionChunk) -> usize {
        chunk
            .choices
            .iter()
            .filter_map(|c| c.delta.tool_calls.as_ref())
            .flatten()
            .filter(|d| matches!(d, ChatCompletionToolCallDelta::Start { .. }))
            .count()
    }

    fn failure(kind: &str, message: &str, _: &str, _: &str, _: usize) -> String {
        format!("data: {}\n\n", error_json(kind, message))
    }
}

impl Output for ResponsesRequest {
    fn id() -> String {
        random_id("resp_")
    }

    fn keep(_: &ResponsesStreamEvent) -> bool {
        true
    }

    fn calls(chunk: &ResponsesStreamEvent) -> usize {
        match chunk {
            ResponsesStreamEvent::OutputItemAdded { item, .. } => matches!(
                item,
                ResponsesOutputItem::FunctionCall { .. }
                    | ResponsesOutputItem::CustomToolCall { .. }
                    | ResponsesOutputItem::ToolSearchCall { .. }
            ) as usize,
            _ => 0,
        }
    }

    /// A `response.failed` event whose response carries the error.
    fn failure(kind: &str, message: &str, id: &str, model: &str, next: usize) -> String {
        let event = json!({
            "type": "response.failed",
            "sequence_number": next,
            "response": {
                "id": id,
                "object": "response",
                "created_at": now(),
                "status": "failed",
                "model": model,
                "output": [],
                "error": {"code": kind, "message": message},
                "incomplete_details": null,
            },
        });
        format!("event: response.failed\ndata: {event}\n\n")
    }
}

/// The time in Unix seconds.
fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// recipe's processor over the request's events: the API's chunks, and
/// whether the server's stop ended them.
fn chunks<P: Output>(
    mut request: ConversationRequest,
    id: String,
    model: String,
    tokenizer: engine::Tokenizer,
    first: Event,
    rx: UnboundedReceiver<Event>,
) -> (
    impl Stream<Item = Result<Chunk<P>, String>> + Send,
    Arc<AtomicBool>,
)
where
    Chunk<P>: Send,
{
    let stopped = Arc::new(AtomicBool::new(false));
    let generator = P::chunk_generator(&request, id, model);
    let options = std::mem::take(&mut request.parsing_options);
    let processor = StreamProcessor::new(generator, options).with_tokenizer(tokenizer);
    let chunks = processor
        .into_protocol_stream(inference(first, rx, stopped.clone()))
        .filter(|c| c.as_ref().map_or(true, P::keep))
        .map(|c| c.map_err(|e| e.to_string()));
    (chunks, stopped)
}

/// A server-sent event: the chunk's JSON, under the API's event name if it
/// names its events (Responses does).
fn sse<P: Output>(chunk: &Chunk<P>) -> Result<String, String>
where
    Chunk<P>: serde::Serialize,
{
    let json = serde_json::to_string(chunk).map_err(|e| e.to_string())?;
    Ok(match <P::Response as ProtocolResponse>::sse_event_name(chunk) {
        Some(name) => format!("event: {name}\ndata: {json}\n\n"),
        None => format!("data: {json}\n\n"),
    })
}

/// The API's last message of a stream that ended normally, if it has one
/// (`data: [DONE]` for Chat Completions).
fn done<P: Output>() -> String {
    <P::Response as ProtocolResponse>::sse_done_data().map_or(String::new(), |d| format!("data: {d}\n\n"))
}

fn log_calls(id: u64, calls: usize) {
    if calls > 0 {
        engine::log(&format!("req {id}: {calls} tool call{}", if calls == 1 { "" } else { "s" }));
    }
}

/// A streamed response: server-sent events from recipe's chunks, a comment
/// every 15 s while the request waits for the engine.
pub fn stream(p: Prepared, cancel: worker::Cancel, rx: UnboundedReceiver<Event>) -> Response {
    match p.api {
        Api::Chat => stream_as::<ChatCompletionRequest>(p, cancel, rx),
        Api::Responses => stream_as::<ResponsesRequest>(p, cancel, rx),
    }
}

fn stream_as<P: Output>(p: Prepared, cancel: worker::Cancel, mut rx: UnboundedReceiver<Event>) -> Response
where
    Chunk<P>: serde::Serialize + Send,
{
    let id = cancel.id();
    let response = P::id();
    let model = p.settings.model.clone();
    let body = async_stream::stream! {
        let _cancel = cancel;  // dropped with the stream, polled or not
        let first = loop {
            match tokio::time::timeout(Duration::from_secs(15), rx.recv()).await {
                Ok(event) => break event,
                Err(_) => yield Ok::<String, std::io::Error>(": waiting\n\n".into()),
            }
        };
        let first = match first {
            Some(Event::Failed(message)) => {
                yield Ok(P::failure("server_error", &message, &response, &model, 0) + &done::<P>());
                return;
            }
            Some(Event::Refused(message)) => {
                yield Ok(P::failure("invalid_request_error", &message, &response, &model, 0) + &done::<P>());
                return;
            }
            Some(event) => event,
            None => return,
        };
        let (chunks, stopped) = chunks::<P>(p.request, response.clone(), p.settings.model, p.tokenizer, first, rx);
        let mut chunks = pin!(chunks);
        let (mut calls, mut sent) = (0, 0);
        while let Some(chunk) = chunks.next().await {
            if stopped.load(Ordering::Relaxed) {
                break;
            }
            match chunk.and_then(|c| {
                calls += P::calls(&c);
                sse::<P>(&c)
            }) {
                Ok(event) => {
                    sent += 1;
                    yield Ok(event);
                }
                Err(message) => {
                    yield Ok(P::failure("server_error", &message, &response, &model, sent));
                    return;
                }
            }
        }
        if stopped.load(Ordering::Relaxed) {
            yield Ok(P::failure("server_error", "the server is stopping", &response, &model, sent));
            return;
        }
        log_calls(id, calls);
        yield Ok(done::<P>());
    };
    Response::builder()
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .body(Body::from_stream(body))
        .expect("a valid response")
}

/// A response that is not streamed: recipe's accumulated chat.completion or
/// Responses response object.
pub async fn whole(p: Prepared, cancel: worker::Cancel, rx: UnboundedReceiver<Event>) -> Response {
    match p.api {
        Api::Chat => whole_as::<ChatCompletionRequest>(p, cancel, rx).await,
        Api::Responses => whole_as::<ResponsesRequest>(p, cancel, rx).await,
    }
}

async fn whole_as<P: Output>(
    p: Prepared,
    cancel: worker::Cancel,
    mut rx: UnboundedReceiver<Event>,
) -> Response
where
    Chunk<P>: Send,
{
    let id = cancel.id();
    let first = match rx.recv().await {
        Some(Event::Failed(message)) => {
            return error(StatusCode::SERVICE_UNAVAILABLE, "server_error", &message);
        }
        Some(Event::Refused(message)) => {
            return error(StatusCode::BAD_REQUEST, "invalid_request_error", &message);
        }
        Some(event) => event,
        None => {
            return error(
                StatusCode::SERVICE_UNAVAILABLE,
                "server_error",
                "the request ended without an answer",
            );
        }
    };
    let (response_id, model) = (P::id(), p.settings.model.clone());
    let mut response =
        <P::Response as ProtocolResponse>::new(response_id.clone(), model, now(), 0, 0);
    let (chunks, stopped) = chunks::<P>(p.request, response_id, p.settings.model, p.tokenizer, first, rx);
    let mut chunks = pin!(chunks);
    let mut calls = 0;
    while let Some(chunk) = chunks.next().await {
        match chunk {
            Ok(chunk) => {
                calls += P::calls(&chunk);
                response.accumulate(chunk);
            }
            Err(message) => {
                return error(StatusCode::INTERNAL_SERVER_ERROR, "server_error", &message);
            }
        }
    }
    if stopped.load(Ordering::Relaxed) {
        return error(
            StatusCode::SERVICE_UNAVAILABLE,
            "server_error",
            "the server is stopping",
        );
    }
    log_calls(id, calls);
    match serde_json::to_string(&response) {
        Ok(json) => (
            StatusCode::OK,
            [(header::CONTENT_TYPE, "application/json")],
            json,
        )
            .into_response(),
        Err(e) => error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "server_error",
            &e.to_string(),
        ),
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
                    let url = iu
                        .get("url")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string();
                    iu.insert(
                        "url".into(),
                        format!("data:image/png;base64,{:08}", urls.len()).into(),
                    );
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
    let defaults = Defaults {
        id: MODEL_ID.into(),
        sampling: Sampling {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            seed: None,
        },
        thinking: false,
        effort: 75,
    };
    let r = match render(root, &defaults, Api::Chat) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("myds4-server: {path}: {}", e.message);
            return 1;
        }
    };
    let mut prompt = r.prompt;
    if let Some(InputMessage::Assistant { prefix: false, .. }) =
        r.request.conversation.messages.last()
    {
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
        Defaults {
            id: MODEL_ID.into(),
            sampling: Sampling {
                temperature: 1.0,
                top_p: 1.0,
                top_k: 0,
                seed: None,
            },
            thinking: true,
            effort: 75,
        }
    }

    fn settings(v: Value) -> (Map<String, Value>, Settings) {
        let Value::Object(mut root) = v else {
            panic!("an object")
        };
        let s = compatible(&mut root, &defaults()).ok().expect("compatible");
        (root, s)
    }

    #[test]
    fn requests_take_the_compatible_forms() {
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
        assert_eq!(
            (
                s.sampling.temperature,
                s.sampling.top_p,
                s.sampling.seed,
                s.effort
            ),
            (0.0, 0.0, Some(7), Some(62))
        );
        assert_eq!(s.max_tokens, DEFAULT_MAX_TOKENS);
        assert_eq!(s.model, MODEL_ID);
        assert_eq!(root["model"], RECIPE_MODEL);
        assert_eq!(root["thinking"], json!({"type": "enabled"}));
        assert_eq!(root["reasoning_effort"], "high");
        assert!(
            !root.contains_key("temperature")
                && !root.contains_key("top_p")
                && !root.contains_key("seed")
        );
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
        let (root, s) = settings(
            json!({"messages": [{"role": "user", "content": "hi"}], "reasoning_effort": "none"}),
        );
        assert_eq!(s.effort, None);
        assert_eq!(root["thinking"], json!({"type": "disabled"}));
        let (_, s) = settings(json!({"messages": [], "reasoning_effort": "max", "max_tokens": 5}));
        assert_eq!((s.effort, s.max_tokens), (Some(100), 5));
    }

    #[test]
    fn a_numeric_effort_is_written_into_the_prompt() {
        let request =
            json!({"messages": [{"role": "user", "content": "hi"}], "reasoning_effort": 62});
        let Value::Object(root) = request else {
            unreachable!()
        };
        let r = render(root, &defaults(), Api::Chat).ok().expect("rendered");
        assert!(
            r.prompt.starts_with(
                "<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: 62 (range 1-100"
            ),
            "{}",
            r.prompt
        );
        for s in &r.specials {
            assert!(
                r.prompt[s.clone()].starts_with('<'),
                "{:?} in {}",
                s,
                r.prompt
            );
        }
    }

    /// The engine's events through recipe's processor, with the checkpoint's
    /// tokenizer (MYDS4_CHECKPOINT names its directory; skipped without it).
    #[tokio::test]
    async fn events_become_chunks() {
        let Ok(dir) = std::env::var("MYDS4_CHECKPOINT") else {
            return;
        };
        let t = engine::Tokenizer::open(&dir, Some(IMAGE));
        let think: Vec<u32> = t.tokenize("Hmm.").iter().map(|&t| t as u32).collect();
        let answer: Vec<u32> = t
            .tokenize("Hi there! 你好")
            .iter()
            .map(|&t| t as u32)
            .collect();
        let request = json!({"messages": [{"role": "user", "content": "hi"}], "stream": true});
        let Value::Object(root) = request else {
            unreachable!()
        };
        let r = render(root, &defaults(), Api::Chat).ok().expect("rendered");
        // the reasoning and </think> in one step, then the answer token by token
        let mut ids = think.clone();
        ids.push(128822);
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        tx.send(Event::Tokens(ids.clone())).unwrap();
        for &t in &answer {
            tx.send(Event::Tokens(vec![t])).unwrap();
        }
        tx.send(Event::Done {
            finish: Finish::Stop,
        })
        .unwrap();
        let first = Event::Started {
            prompt: 10,
            reused: 4,
        };
        let (chunks, stopped) =
            chunks::<ChatCompletionRequest>(r.request, "chatcmpl-test".into(), MODEL_ID.into(), t, first, rx);
        let chunks: Vec<ChatCompletionChunk> = chunks.map(|c| c.expect("a chunk")).collect().await;
        assert!(!stopped.load(Ordering::Relaxed));
        let text = |f: fn(&ChatCompletionChunk) -> Option<String>| {
            chunks.iter().filter_map(f).collect::<String>()
        };
        let reasoning = text(|c| c.choices.first()?.delta.reasoning_content.clone().flatten());
        let content = text(|c| c.choices.first()?.delta.content.clone().flatten());
        assert_eq!(reasoning, "Hmm.");
        assert_eq!(content, "Hi there! 你好");
        let last = serde_json::to_value(chunks.last().unwrap()).unwrap();
        assert_eq!(last["choices"][0]["finish_reason"], "stop", "{last}");
        assert_eq!(last["usage"]["prompt_tokens"], 10);
        assert_eq!(last["usage"]["prompt_cache_hit_tokens"], 4);
        assert_eq!(
            last["usage"]["completion_tokens"],
            ids.len() + answer.len() + 1
        );
        let empty = chunks[1..chunks.len() - 1]
            .iter()
            .filter(|c| {
                c.choices
                    .first()
                    .is_some_and(|c| c.delta.content == Some(Some(String::new())))
            })
            .count();
        let all: Vec<String> = chunks
            .iter()
            .map(|c| serde_json::to_string(c).unwrap())
            .collect();
        assert_eq!(
            empty,
            0,
            "no empty deltas before the last chunk:\n{}",
            all.join("\n")
        );
    }

    fn responses_settings(v: Value) -> Result<(Map<String, Value>, Settings), Error> {
        let Value::Object(mut root) = v else {
            panic!("an object")
        };
        let s = responses_compatible(&mut root, &defaults())?;
        Ok((root, s))
    }

    #[test]
    fn responses_requests_take_the_server_settings() {
        let (root, s) = responses_settings(json!({
            "model": "deepseek-v4.1-flash", "input": "hi", "max_output_tokens": 300,
            "temperature": 0.5, "reasoning": {"effort": "xhigh", "summary": "auto"},
        }))
        .ok()
        .expect("settings");
        assert_eq!(root["model"], RECIPE_MODEL);
        assert_eq!(s.model, "deepseek-v4.1-flash");
        assert_eq!((s.effort, &root["reasoning"]["effort"]), (Some(75), &json!("high")));
        assert_eq!(root["reasoning"]["summary"], "auto");
        assert_eq!(s.max_tokens, 300);
        assert_eq!(s.sampling.temperature, 0.5);

        let effort = |v: Value| {
            let (root, s) = responses_settings(v).ok().expect("settings");
            (s.effort, root.get("reasoning").map(|r| r["effort"].clone()))
        };
        assert_eq!(effort(json!({"input": "hi", "reasoning": {"effort": "none"}})), (None, Some(json!("none"))));
        assert_eq!(effort(json!({"input": "hi", "reasoning": {"effort": "low"}})), (Some(50), Some(json!("low"))));
        assert_eq!(effort(json!({"input": "hi", "reasoning": {"effort": "max"}})), (Some(100), Some(json!("max"))));
        assert_eq!(effort(json!({"input": "hi", "reasoning": {"effort": 62}})), (Some(62), Some(json!("high"))));
        // without an effort, the model's default: thinking at 75 here
        assert_eq!(effort(json!({"input": "hi"})), (Some(75), Some(json!("high"))));
        assert_eq!(effort(json!({"input": "hi", "reasoning": null})), (Some(75), Some(json!("high"))));

        let (_, s) = responses_settings(json!({"input": "hi"})).ok().expect("settings");
        assert_eq!(s.max_tokens, DEFAULT_MAX_TOKENS);
        for key in ["previous_response_id", "conversation"] {
            let mut root = Map::new();
            root.insert("input".into(), "hi".into());
            root.insert(key.into(), "resp_1".into());
            let e = responses_compatible(&mut root, &defaults()).err().expect("refused");
            assert!(e.message.contains(key), "{}", e.message);
        }
        assert!(responses_settings(json!({"input": "hi", "max_output_tokens": 0})).is_err());
    }

    /// The shape of a request from Codex 0.167: developer and user messages, a
    /// function, a namespace of functions and web search (dropped), and a
    /// replayed tool call with its reasoning.
    #[test]
    fn codex_requests_render() {
        let request = json!({
            "model": "deepseek-v4.1-flash", "stream": true, "store": false, "tool_choice": "auto",
            "parallel_tool_calls": true, "include": ["reasoning.encrypted_content"],
            "prompt_cache_key": "k", "client_metadata": {"thread_id": "t"},
            "reasoning": {"effort": "xhigh", "summary": "auto"},
            "tools": [
                {"type": "function", "name": "exec_command", "description": "Runs a command.", "strict": false,
                 "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}},
                                "required": ["cmd"], "additionalProperties": false}},
                {"type": "namespace", "name": "multi_agent_v1", "description": "Agents.", "tools": [
                    {"type": "function", "name": "spawn_agent", "description": "Starts an agent.", "strict": false,
                     "parameters": {"type": "object", "properties": {"task": {"type": "string"}}, "required": ["task"]}}]},
                {"type": "web_search", "external_web_access": false},
            ],
            "input": [
                {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "You are a coding agent."}]},
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "List the files."}]},
                {"type": "reasoning", "summary": [], "encrypted_content": null,
                 "content": [{"type": "reasoning_text", "text": "I should run ls."}]},
                {"type": "function_call", "call_id": "call_0", "name": "exec_command", "arguments": "{\"cmd\":\"ls\"}"},
                {"type": "function_call_output", "call_id": "call_0", "output": "a.txt\nb.txt"},
            ],
        });
        let Value::Object(root) = request else {
            unreachable!()
        };
        assert_eq!(dropped_tools(&root), 1);
        let r = render(root, &defaults(), Api::Responses).ok().expect("rendered");
        assert!(r.request.stream);
        for text in ["You are a coding agent.", "exec_command", "multi_agent_v1::spawn_agent", "I should run ls.", "a.txt"] {
            assert!(r.prompt.contains(text), "{text} is not in the prompt:\n{}", r.prompt);
        }
        assert!(!r.prompt.contains("web_search"), "{}", r.prompt);
    }

    #[test]
    fn failures_end_each_api_its_way() {
        let chat = ChatCompletionRequest::failure("invalid_request_error", "too long", "chatcmpl-1", "m", 3);
        let data = chat.strip_prefix("data: ").expect("a data line");
        let v: Value = serde_json::from_str(data.trim_end()).unwrap();
        assert_eq!(v["error"]["message"], "too long");

        let event = ResponsesRequest::failure("invalid_request_error", "too long", "resp_1", "m", 3);
        let (name, data) = event.split_once('\n').expect("two lines");
        assert_eq!(name, "event: response.failed");
        let v: Value = serde_json::from_str(data.strip_prefix("data: ").unwrap().trim_end()).unwrap();
        assert_eq!(v["type"], "response.failed");
        assert_eq!(v["sequence_number"], 3);
        assert_eq!(v["response"]["id"], "resp_1");
        assert_eq!(v["response"]["status"], "failed");
        assert_eq!(v["response"]["error"]["message"], "too long");
        assert!(event.ends_with("\n\n"));
        assert_eq!(done::<ResponsesRequest>(), "", "Responses streams end without [DONE]");
        assert_eq!(done::<ChatCompletionRequest>(), "data: [DONE]\n\n");
    }

    /// The engine's events as Responses events, with the checkpoint's
    /// tokenizer (MYDS4_CHECKPOINT names its directory; skipped without it).
    #[tokio::test]
    async fn events_become_responses_events() {
        let Ok(dir) = std::env::var("MYDS4_CHECKPOINT") else {
            return;
        };
        let t = engine::Tokenizer::open(&dir, Some(IMAGE));
        let tokens = |s: &str| -> Vec<u32> { t.tokenize(s).iter().map(|&t| t as u32).collect() };
        let request = json!({"input": "hi", "stream": true});
        let Value::Object(root) = request else {
            unreachable!()
        };
        let r = render(root, &defaults(), Api::Responses).ok().expect("rendered");
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let mut ids = tokens("Hmm.");
        ids.push(128822); // </think>
        tx.send(Event::Tokens(ids)).unwrap();
        for t in tokens("Hi there! 你好") {
            tx.send(Event::Tokens(vec![t])).unwrap();
        }
        tx.send(Event::Done { finish: Finish::Stop }).unwrap();
        let first = Event::Started { prompt: 10, reused: 4 };
        let (events, stopped) = chunks::<ResponsesRequest>(r.request, "resp_1".into(), MODEL_ID.into(), t, first, rx);
        let events: Vec<ResponsesStreamEvent> = events.map(|e| e.expect("an event")).collect().await;
        assert!(!stopped.load(Ordering::Relaxed));
        let json: Vec<Value> = events.iter().map(|e| serde_json::to_value(e).unwrap()).collect();
        let types: Vec<&str> = json.iter().map(|e| e["type"].as_str().unwrap()).collect();
        assert_eq!(types.first(), Some(&"response.created"), "{types:?}");
        assert_eq!(types.last(), Some(&"response.completed"), "{types:?}");
        let text = |kind: &str| -> String {
            json.iter().filter(|e| e["type"] == kind).filter_map(|e| e["delta"].as_str()).collect()
        };
        assert_eq!(text("response.reasoning_text.delta"), "Hmm.");
        assert_eq!(text("response.output_text.delta"), "Hi there! 你好");
        let done = &json.last().unwrap()["response"];
        assert_eq!(done["id"], "resp_1");
        assert_eq!(done["model"], MODEL_ID);
        assert_eq!(done["usage"]["input_tokens"], 10, "{done}");
        assert_eq!(done["usage"]["input_tokens_details"]["cached_tokens"], 4, "{done}");
        let sse = sse::<ResponsesRequest>(&events[0]).unwrap();
        assert!(sse.starts_with("event: response.created\ndata: {"), "{sse}");
        // the sequence numbers count the events
        for (i, e) in json.iter().enumerate() {
            assert_eq!(e["sequence_number"], i, "{e}");
        }
    }
}
