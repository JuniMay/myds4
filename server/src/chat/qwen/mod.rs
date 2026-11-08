//! Qwen chat conversion: API messages become checkpoint-template tokens.
//! The response decoder separates reasoning, text and XML tool calls into
//! Chat Completions fields. Usage includes reused prompt tokens.

mod decoder;
mod prompt;

use std::path::Path;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::body::Body;
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use serde_json::{Map, Value, json};
use tokio::sync::mpsc::UnboundedReceiver;

use self::decoder::{Decoder, Piece};
use self::prompt::{self as template, Call, ImagePart, Message, Options};
use super::{Error, bad, error, error_json, response_id};
use crate::engine::Tokenizer;
use crate::worker::{self, Event, Finish, Sampling};

/// A served Qwen model's request defaults (the server's configuration, then
/// generation_config.json), its name, context and tokenizer.
pub struct Settings {
    pub id: String,
    pub ctx: usize,
    pub tokenizer: Tokenizer,
    pub sampling: Sampling,
    pub think: Option<u32>,
}

/// The context a model serves: the configured one, at most its
/// max_position_embeddings.
pub fn context(dir: &str, ctx: usize) -> usize {
    let config: Value = std::fs::read(Path::new(dir).join("config.json"))
        .ok()
        .and_then(|t| serde_json::from_slice(&t).ok())
        .unwrap_or(Value::Null);
    let text = config.get("text_config").unwrap_or(&config);
    let most = text
        .get("max_position_embeddings")
        .and_then(Value::as_u64)
        .map_or(ctx, |n| n as usize);
    ctx.min(most)
}

/// A model's default name: its directory's name, in lower case.
pub fn model_id(dir: &str) -> String {
    Path::new(dir.trim_end_matches('/'))
        .file_name()
        .map_or("qwen".into(), |n| n.to_string_lossy().to_lowercase())
}

/// The defaults of requests that set no sampling: the configured values where
/// given, else the checkpoint's generation_config.json.
pub fn sampling(
    dir: &str,
    temperature: Option<f64>,
    top_p: Option<f64>,
    top_k: Option<u32>,
) -> Sampling {
    let generation: Value = std::fs::read(Path::new(dir).join("generation_config.json"))
        .ok()
        .and_then(|t| serde_json::from_slice(&t).ok())
        .unwrap_or(Value::Null);
    let num = |k: &str, d: f64| generation.get(k).and_then(Value::as_f64).unwrap_or(d);
    Sampling {
        temperature: temperature.unwrap_or(num("temperature", 1.0)),
        top_p: top_p.unwrap_or(num("top_p", 1.0)),
        top_k: top_k.unwrap_or(num("top_k", 0.0) as u32),
        seed: None,
    }
}

fn content_of(content: Option<&Value>) -> Result<(String, Vec<ImagePart>), Error> {
    match content {
        None | Some(Value::Null) => Ok((String::new(), Vec::new())),
        Some(Value::String(s)) => Ok((s.clone(), Vec::new())),
        Some(Value::Array(parts)) => {
            let mut out = String::new();
            let mut images = Vec::new();
            for p in parts {
                match p.get("type").and_then(Value::as_str) {
                    Some("text") | None if p.get("text").is_some() => {
                        out.push_str(p.get("text").and_then(Value::as_str).unwrap_or(""))
                    }
                    Some("image_url") => {
                        let image = p
                            .get("image_url")
                            .ok_or_else(|| bad("image_url is missing"))?;
                        let url = image
                            .as_str()
                            .or_else(|| image.get("url").and_then(Value::as_str))
                            .ok_or_else(|| bad("image_url requires a URL"))?;
                        images.push(ImagePart {
                            offset: out.len(),
                            bytes: super::image_data(url)?,
                        });
                    }
                    Some(t) => return Err(bad(format!("unsupported content type {t}"))),
                    None => return Err(bad("a content part without a type")),
                }
            }
            Ok((out, images))
        }
        Some(_) => Err(bad(
            "a message's content must be a string or an array of parts",
        )),
    }
}

fn messages(list: &[Value]) -> Result<Vec<Message>, Error> {
    let mut out: Vec<Message> = Vec::new();
    for (i, m) in list.iter().enumerate() {
        let role = m
            .get("role")
            .and_then(Value::as_str)
            .ok_or(bad(format!("message {i} has no role")))?;
        let (content, images) = content_of(m.get("content"))?;
        if !images.is_empty() && matches!(role, "system" | "developer") {
            return Err(bad("system messages cannot contain images"));
        }
        let role = match role {
            // the template takes one system message, first: the leading
            // system and developer messages join, later ones speak as the user
            "system" | "developer" if out.iter().all(|x| x.role == "system") => {
                if let Some(first) = out.first_mut() {
                    first.content = format!("{}\n\n{}", first.content, content);
                    continue;
                }
                "system"
            }
            "system" | "developer" => "user",
            r @ ("user" | "assistant" | "tool") => r,
            r => return Err(bad(format!("message {i}: unsupported role {r:?}"))),
        };
        let reasoning = m
            .get("reasoning_content")
            .or_else(|| m.get("reasoning"))
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        let mut calls = Vec::new();
        for c in m
            .get("tool_calls")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            let f = c.get("function").unwrap_or(c);
            let name = f
                .get("name")
                .and_then(Value::as_str)
                .ok_or(bad(format!("message {i}: a tool call without a name")))?;
            let arguments = match f.get("arguments") {
                None | Some(Value::Null) => Map::new(),
                Some(Value::Object(o)) => o.clone(),
                Some(Value::String(s)) if s.trim().is_empty() => Map::new(),
                Some(Value::String(s)) => match serde_json::from_str::<Value>(s) {
                    Ok(Value::Object(o)) => o,
                    _ => {
                        return Err(bad(format!(
                            "message {i}: tool call {name}'s arguments are not a JSON object"
                        )));
                    }
                },
                Some(_) => {
                    return Err(bad(format!(
                        "message {i}: tool call {name}'s arguments are not a JSON object"
                    )));
                }
            };
            calls.push(Call {
                name: name.into(),
                arguments,
            });
        }
        out.push(Message {
            role: role.into(),
            content,
            images,
            reasoning,
            calls,
        });
    }
    Ok(out)
}

/// Thinking and its effort from the request: reasoning_effort (none turns
/// it off; low and minimal low; medium medium; high, xhigh, max and ultra
/// xhigh; a number 1-100 as DeepSeek's path takes it), `thinking` or
/// chat_template_kwargs.enable_thinking; else thinking at the template's
/// default effort, xhigh.
fn thinking(
    root: &Map<String, Value>,
    default: Option<u32>,
) -> Result<(bool, &'static str), Error> {
    let (mut on, mut effort) = default.map_or((true, "xhigh"), |n| by_number(n as f64));
    match root.get("reasoning_effort") {
        None | Some(Value::Null) => {}
        Some(Value::String(s)) => match s.to_ascii_lowercase().as_str() {
            "none" | "off" => on = false,
            "minimal" | "low" => effort = "low",
            "medium" => effort = "medium",
            "high" | "xhigh" | "max" | "ultra" => effort = "xhigh",
            other => match other.parse::<f64>() {
                Ok(n) => (on, effort) = by_number(n),
                Err(_) => return Err(bad(format!("unsupported reasoning_effort {s:?}"))),
            },
        },
        Some(Value::Number(n)) => (on, effort) = by_number(n.as_f64().unwrap_or(100.0)),
        Some(_) => return Err(bad("reasoning_effort must be a string or a number")),
    }
    if let Some(t) = root.get("thinking") {
        match t.get("type").and_then(Value::as_str).or(t.as_str()) {
            Some("disabled") => on = false,
            Some("enabled") => on = true,
            _ => {}
        }
        if let Some(b) = t.as_bool() {
            on = b;
        }
    }
    if let Some(b) = root
        .get("chat_template_kwargs")
        .and_then(|k| k.get("enable_thinking"))
        .and_then(Value::as_bool)
    {
        on = b;
    }
    Ok((on, effort))
}

fn by_number(n: f64) -> (bool, &'static str) {
    if n <= 0.0 {
        (false, "xhigh")
    } else if n <= 50.0 {
        (true, "low")
    } else if n <= 75.0 {
        (true, "medium")
    } else {
        (true, "xhigh")
    }
}

/// A request ready for the worker, and what its response needs.
pub struct Prepared {
    pub request: worker::Request,
    tokenizer: Tokenizer,
    pub stream: bool,
    pub usage_in_stream: bool,
    pub thinking: bool,
    pub tools: Vec<Value>,
    pub stop: Vec<String>,
    pub model: String,
}

/// A request's body as the request of the worker's model `slot` (`m`: its settings).
pub fn prepare(body: &[u8], note: &str, m: &Settings, slot: usize) -> Result<Prepared, Error> {
    let root: Map<String, Value> = serde_json::from_slice(body)
        .map_err(|e| bad(format!("the body is not a JSON object: {e}")))?;
    let list = root
        .get("messages")
        .and_then(Value::as_array)
        .ok_or(bad("messages must be an array"))?;
    let msgs = messages(list)?;
    let tools: Vec<Value> = match root.get("tools") {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::Array(t)) => t.clone(),
        Some(_) => return Err(bad("tools must be an array")),
    };
    let (on, effort) = thinking(&root, m.think)?;
    let prompt = template::render(
        &msgs,
        &Options {
            tools: &tools,
            thinking: on,
            effort,
        },
        &m.tokenizer,
    )
    .map_err(bad)?;
    let message_count = msgs.len();
    let images: Vec<_> = msgs
        .into_iter()
        .flat_map(|m| m.images)
        .map(|image| image.bytes)
        .collect();
    let max_tokens = super::completion_limit(&root, m.ctx as i64)?;
    let sampling = super::sampling(&root, m.sampling)?;
    let stop = super::stop_sequences(&root)?;
    let stream = root.get("stream").and_then(Value::as_bool).unwrap_or(false);
    let usage_in_stream = root
        .get("stream_options")
        .and_then(|o| o.get("include_usage"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let model = root
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or(&m.id)
        .to_string();
    let note = format!(
        "{note}: {} messages, {} tools, {}; thinking {}",
        message_count,
        tools.len(),
        if stream { "streamed" } else { "one response" },
        if on { effort } else { "off" }
    );
    let prompt: Vec<i32> = prompt.iter().map(|&t| t as i32).collect();
    Ok(Prepared {
        request: worker::Request {
            model: slot,
            prompt,
            images,
            sampling,
            max_tokens: max_tokens as i64,
            note,
        },
        tokenizer: m.tokenizer,
        stream,
        usage_in_stream,
        thinking: on,
        tools,
        stop,
        model,
    })
}

fn call_id() -> String {
    use rand::Rng;
    let s: String = rand::rng()
        .sample_iter(&rand::distr::Alphanumeric)
        .take(24)
        .map(char::from)
        .collect();
    format!("call_{s}")
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// The reply as it comes: decoded text through the protocol decoder, cut at
/// a stop string.
struct Reply {
    utf8: Utf8Stream,
    decoder: Decoder,
    stop: Vec<String>,
    text: String, // the reply's text that may begin a stop string, not yet released
    stopped: bool,
    prompt: usize,
    reused: usize,
    tokens: usize,
}

impl Reply {
    fn new(p: &Prepared) -> Reply {
        Reply {
            utf8: Utf8Stream::default(),
            decoder: Decoder::new(p.thinking, p.tools.clone()),
            stop: p.stop.clone(),
            text: String::new(),
            stopped: false,
            prompt: 0,
            reused: 0,
            tokens: 0,
        }
    }

    /// A token of the reply, its text from the model's tokenizer `tk`.
    fn token(&mut self, tk: &Tokenizer, id: u32) -> Vec<Piece> {
        self.tokens += 1;
        let t = tk.piece(id as i32);
        let text = if tk.marked_special(id as i32) {
            String::new()
        } else {
            self.utf8.push(t)
        };
        self.text(&text)
    }

    /// Text of the reply: released to the decoder but for a stop string and
    /// for the end of the text that may still begin one, which waits (in
    /// `text`) for what comes next.
    fn text(&mut self, text: &str) -> Vec<Piece> {
        if self.stopped || text.is_empty() {
            return Vec::new();
        }
        self.text.push_str(text);
        let cut = self
            .stop
            .iter()
            .filter(|s| !s.is_empty())
            .filter_map(|s| self.text.find(s.as_str()))
            .min();
        if let Some(at) = cut {
            self.stopped = true;
            let keep = self.text[..at].to_string();
            self.text.clear();
            return self.decoder.push(&keep);
        }
        // the longest end of the text that begins a stop string waits
        let wait = self
            .stop
            .iter()
            .flat_map(|s| {
                (1..s.len())
                    .filter(|&n| s.is_char_boundary(n))
                    .map(move |n| &s[..n])
            })
            .filter(|p| self.text.ends_with(p))
            .map(str::len)
            .max()
            .unwrap_or(0);
        let ready = self.text.len() - wait;
        let out = self.text[..ready].to_string();
        self.text.drain(..ready);
        if out.is_empty() {
            Vec::new()
        } else {
            self.decoder.push(&out)
        }
    }

    fn finish(&mut self, clean: bool) -> Vec<Piece> {
        let rest = self.utf8.finish();
        let mut out = self.text(&rest);
        if !self.stopped {
            let waiting = std::mem::take(&mut self.text);
            out.extend(self.decoder.push(&waiting));
        }
        out.extend(self.decoder.finish(clean || self.stopped));
        out
    }

    fn usage(&self) -> Value {
        json!({
            "prompt_tokens": self.prompt, "completion_tokens": self.tokens, "total_tokens": self.prompt + self.tokens,
            "prompt_cache_hit_tokens": self.reused, "prompt_cache_miss_tokens": self.prompt - self.reused,
            "prompt_tokens_details": {"cached_tokens": self.reused},
        })
    }
}

fn finish_reason(finish: Finish, calls: usize, stopped: bool) -> &'static str {
    if calls > 0 {
        "tool_calls"
    } else if stopped || finish == Finish::Stop {
        "stop"
    } else {
        "length"
    }
}

fn delta_of(p: Piece, calls: &mut usize) -> Value {
    match p {
        Piece::Text(t) => json!({"content": t}),
        Piece::Reasoning(t) => json!({"reasoning_content": t}),
        Piece::Call { name, arguments } => {
            let i = *calls;
            *calls += 1;
            json!({"tool_calls": [{"index": i, "id": call_id(), "type": "function",
                "function": {"name": name, "arguments": Value::Object(arguments).to_string()}}]})
        }
    }
}

/// A streamed response: server-sent events, a comment every 15 s while the
/// request waits.
pub fn stream(p: Prepared, cancel: worker::Cancel, mut rx: UnboundedReceiver<Event>) -> Response {
    let tokenizer = p.tokenizer;
    let body = async_stream::stream! {
        let _cancel = cancel;
        let id = response_id();
        let created = now();
        let mut reply = Reply::new(&p);
        let chunk = |delta: Value, finish: Option<&str>, usage: Option<Value>| {
            let mut c = json!({"id": id, "object": "chat.completion.chunk", "created": created, "model": p.model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]});
            if let Some(u) = usage {
                c["usage"] = u;
            }
            format!("data: {c}\n\n")
        };
        let mut calls = 0;
        let mut first = true;
        loop {
            let event = match tokio::time::timeout(Duration::from_secs(15), rx.recv()).await {
                Ok(Some(e)) => e,
                Ok(None) => return,
                Err(_) => {
                    yield Ok::<String, std::io::Error>(": waiting\n\n".into());
                    continue;
                }
            };
            match event {
                Event::Failed(message) => {
                    yield Ok(format!("data: {}\n\ndata: [DONE]\n\n", error_json("server_error", &message)));
                    return;
                }
                Event::Refused(m) => {
                    yield Ok(format!("data: {}\n\ndata: [DONE]\n\n", error_json("invalid_request_error", &m)));
                    return;
                }
                Event::Started { prompt, reused } => {
                    reply.prompt = prompt;
                    reply.reused = reused;
                    if first {
                        first = false;
                        yield Ok(chunk(json!({"role": "assistant", "content": ""}), None, None));
                    }
                }
                Event::Tokens(ts) => {
                    for t in ts {
                        for piece in reply.token(&tokenizer, t) {
                            yield Ok(chunk(delta_of(piece, &mut calls), None, None));
                        }
                        if reply.stopped {
                            break;
                        }
                    }
                    if reply.stopped {
                        break;  // dropping the receiver cancels the request
                    }
                }
                Event::Done { finish: Finish::Shutdown } => {
                    yield Ok(format!("data: {}\n\n", error_json("server_error", "the server is stopping")));
                    return;
                }
                Event::Done { finish } => {
                    for piece in reply.finish(finish == Finish::Stop) {
                        yield Ok(chunk(delta_of(piece, &mut calls), None, None));
                    }
                    let usage = Some(reply.usage());
                    yield Ok(chunk(json!({}), Some(finish_reason(finish, calls, reply.stopped)), usage));
                    yield Ok("data: [DONE]\n\n".into());
                    return;
                }
            }
        }
        for piece in reply.finish(true) {
            yield Ok(chunk(delta_of(piece, &mut calls), None, None));
        }
        let usage = p.usage_in_stream.then(|| reply.usage()).or(Some(reply.usage()));
        yield Ok(chunk(json!({}), Some(finish_reason(Finish::Stop, calls, true)), usage));
        yield Ok("data: [DONE]\n\n".into());
    };
    Response::builder()
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .body(Body::from_stream(body))
        .expect("a valid response")
}

/// A response that is not streamed: one chat.completion.
pub async fn whole(
    p: Prepared,
    cancel: worker::Cancel,
    mut rx: UnboundedReceiver<Event>,
) -> Response {
    let tokenizer = p.tokenizer;
    let mut reply = Reply::new(&p);
    let (mut content, mut reasoning, mut tool_calls) = (String::new(), String::new(), Vec::new());
    let mut take = |pieces: Vec<Piece>, tool_calls: &mut Vec<Value>| {
        for piece in pieces {
            match piece {
                Piece::Text(t) => content.push_str(&t),
                Piece::Reasoning(t) => reasoning.push_str(&t),
                Piece::Call { name, arguments } => {
                    tool_calls.push(json!({"id": call_id(), "type": "function",
                    "function": {"name": name, "arguments": Value::Object(arguments).to_string()}}))
                }
            }
        }
    };
    let finish = loop {
        match rx.recv().await {
            None => {
                return error(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "server_error",
                    "the request ended without an answer",
                );
            }
            Some(Event::Refused(m)) => {
                return error(StatusCode::BAD_REQUEST, "invalid_request_error", &m);
            }
            Some(Event::Failed(m)) => {
                return error(StatusCode::SERVICE_UNAVAILABLE, "server_error", &m);
            }
            Some(Event::Started { prompt, reused }) => {
                (reply.prompt, reply.reused) = (prompt, reused)
            }
            Some(Event::Tokens(ts)) => {
                for t in ts {
                    take(reply.token(&tokenizer, t), &mut tool_calls);
                    if reply.stopped {
                        break;
                    }
                }
                if reply.stopped {
                    break Finish::Stop;
                }
            }
            Some(Event::Done {
                finish: Finish::Shutdown,
            }) => {
                return error(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "server_error",
                    "the server is stopping",
                );
            }
            Some(Event::Done { finish }) => break finish,
        }
    };
    drop(cancel);
    take(reply.finish(finish == Finish::Stop), &mut tool_calls);
    let mut message = json!({"role": "assistant", "content": content});
    if !reasoning.is_empty() {
        message["reasoning_content"] = reasoning.into();
    }
    if !tool_calls.is_empty() {
        message["tool_calls"] = Value::Array(tool_calls.clone());
    }
    let body = json!({"id": response_id(), "object": "chat.completion", "created": now(), "model": p.model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason(finish, tool_calls.len(), reply.stopped)}],
        "usage": reply.usage()});
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "application/json")],
        body.to_string(),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reply_with_stops(stop: &[&str]) -> Reply {
        Reply {
            utf8: Utf8Stream::default(),
            decoder: Decoder::new(false, Vec::new()),
            stop: stop.iter().map(|s| s.to_string()).collect(),
            text: String::new(),
            stopped: false,
            prompt: 0,
            reused: 0,
            tokens: 0,
        }
    }

    #[test]
    fn utf8_stream_waits_for_whole_characters() {
        let mut s = Utf8Stream::default();
        assert_eq!(s.push(&[b'a', 0xf0]), "a");
        assert_eq!(s.push(&[0x9f, 0x99]), "");
        assert_eq!(s.push(&[0x82, b'b']), "🙂b");
        assert_eq!(s.finish(), "");
        assert_eq!(s.push(&[0xff, 0xe2]), "\u{FFFD}");
        assert_eq!(s.finish(), "\u{FFFD}");
    }

    fn texts(pieces: Vec<Piece>) -> String {
        pieces
            .into_iter()
            .map(|p| {
                if let Piece::Text(t) = p {
                    t
                } else {
                    String::new()
                }
            })
            .collect()
    }

    /// A stop string that arrives over several tokens ends the reply before
    /// it: none of it reaches the reply, also not its first tokens.
    #[test]
    fn a_stop_string_across_tokens_never_reaches_the_reply() {
        let mut r = reply_with_stops(&["\n\nUser:"]);
        let mut out = String::new();
        for piece in ["Hello", "\n\n", "User", ":", " more"] {
            out += &texts(r.text(piece));
        }
        out += &texts(r.finish(true));
        assert!(r.stopped);
        assert_eq!(out, "Hello");
        // text that only began like a stop string comes through whole
        let mut r = reply_with_stops(&["\n\nUser:"]);
        let mut out = String::new();
        for piece in ["Hello", "\n\n", "Users", " agree"] {
            out += &texts(r.text(piece));
        }
        out += &texts(r.finish(true));
        assert!(!r.stopped);
        assert_eq!(out, "Hello\n\nUsers agree");
        // a stop string inside one token
        let mut r = reply_with_stops(&["END"]);
        let out = texts(r.text("one ENDtwo")) + &texts(r.finish(true));
        assert_eq!(out, "one ");
    }

    #[test]
    fn effort_and_thinking_switches() {
        let r = |v: Value| thinking(v.as_object().unwrap(), None).unwrap();
        assert_eq!(r(json!({})), (true, "xhigh"));
        assert_eq!(r(json!({"reasoning_effort": "none"})).0, false);
        assert_eq!(r(json!({"reasoning_effort": "low"})), (true, "low"));
        assert_eq!(r(json!({"reasoning_effort": "high"})), (true, "xhigh"));
        assert_eq!(r(json!({"reasoning_effort": "medium"})), (true, "medium"));
        assert_eq!(r(json!({"reasoning_effort": 50})), (true, "low"));
        assert_eq!(
            r(json!({"chat_template_kwargs": {"enable_thinking": false}})).0,
            false
        );
        assert_eq!(r(json!({"thinking": {"type": "disabled"}})).0, false);
        assert!(
            thinking(
                json!({"reasoning_effort": "huge"}).as_object().unwrap(),
                None
            )
            .is_err()
        );
    }

    #[test]
    fn leading_system_and_developer_messages_join_and_later_ones_speak_as_the_user() {
        let m = messages(&[
            json!({"role": "system", "content": "A"}),
            json!({"role": "developer", "content": [{"type": "text", "text": "B"}]}),
            json!({"role": "user", "content": "hi"}),
            json!({"role": "system", "content": "C"}),
        ])
        .unwrap();
        assert_eq!(m.len(), 3);
        assert_eq!(
            (m[0].role.as_str(), m[0].content.as_str()),
            ("system", "A\n\nB")
        );
        assert_eq!((m[2].role.as_str(), m[2].content.as_str()), ("user", "C"));
        assert!(messages(&[json!({"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]})]).is_err());
    }

    #[test]
    fn tool_history_arguments_parse_as_objects() {
        let m = messages(&[json!({"role": "assistant", "content": null, "tool_calls": [
            {"id": "a", "type": "function", "function": {"name": "read", "arguments": "{\"path\": \"x\", \"n\": 2}"}},
            {"id": "b", "type": "function", "function": {"name": "list", "arguments": ""}}]})])
        .unwrap();
        assert_eq!(m[0].calls.len(), 2);
        assert_eq!(
            Value::Object(m[0].calls[0].arguments.clone()),
            json!({"path": "x", "n": 2})
        );
        assert!(m[0].calls[1].arguments.is_empty());
        assert!(messages(&[json!({"role": "assistant", "tool_calls": [{"function": {"name": "r", "arguments": "[1]"}}]})]).is_err());
    }

    /// Fixed renderings from the checkpoint's chat_template.jinja.
    #[test]
    fn renders_as_the_reference_template() {
        let cases: Vec<Value> =
            serde_json::from_str(include_str!("../../../tests/fixtures/chat/qwen_chats.json"))
                .unwrap();
        for (i, case) in cases.iter().enumerate() {
            let msgs = messages(case["messages"].as_array().unwrap()).unwrap();
            let on = case["enable_thinking"].as_bool().unwrap_or(true);
            let got = template::render_text(
                &msgs,
                &Options {
                    tools: &[],
                    thinking: on,
                    effort: "xhigh",
                },
            )
            .unwrap();
            assert_eq!(got, case["rendered"].as_str().unwrap(), "case {i}");
        }
    }

    /// Tool cases rendered with the checkpoint's own chat_template.jinja
    /// through jinja2 as Hugging Face renders it (trim_blocks, lstrip_blocks,
    /// its tojson): tool definitions with extra fields and non-ASCII text,
    /// calls with string, number and list arguments, consecutive tool
    /// responses, thinking off, and the low and medium efforts.
    #[test]
    fn renders_tools_as_the_checkpoint_template() {
        let cases: Vec<Value> = serde_json::from_str(include_str!(
            "../../../tests/fixtures/chat/qwen_tool_chats.json"
        ))
        .unwrap();
        for (i, case) in cases.iter().enumerate() {
            let msgs = messages(case["messages"].as_array().unwrap()).unwrap();
            let tools = case["tools"].as_array().cloned().unwrap_or_default();
            let on = case["enable_thinking"].as_bool().unwrap_or(true);
            let effort = case["reasoning_effort"].as_str().unwrap_or("xhigh");
            let got = template::render_text(
                &msgs,
                &Options {
                    tools: &tools,
                    thinking: on,
                    effort,
                },
            )
            .unwrap();
            assert_eq!(got, case["rendered"].as_str().unwrap(), "case {i}");
        }
    }
}

/// Bytes to text as tokens arrive: an incomplete UTF-8 sequence waits for
/// its next token.
#[derive(Default)]
pub struct Utf8Stream {
    pending: Vec<u8>,
}

impl Utf8Stream {
    pub fn push(&mut self, bytes: &[u8]) -> String {
        self.pending.extend_from_slice(bytes);
        let mut out = String::new();
        loop {
            match std::str::from_utf8(&self.pending) {
                Ok(text) => {
                    out.push_str(text);
                    self.pending.clear();
                    break;
                }
                Err(e) => {
                    let valid = e.valid_up_to();
                    out.push_str(
                        std::str::from_utf8(&self.pending[..valid]).expect("a valid prefix"),
                    );
                    self.pending.drain(..valid);
                    match e.error_len() {
                        Some(n) => {
                            out.push('\u{FFFD}');
                            self.pending.drain(..n);
                        }
                        None => break,
                    }
                }
            }
        }
        out
    }

    pub fn finish(&mut self) -> String {
        let rest = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        rest
    }
}
