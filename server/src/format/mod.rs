//! The models' chat formats (docs/server.md): a Chat Completions request's
//! JSON into a model's prompt tokens and sampling, and the worker's events
//! into the response. Each model family renders its own chat template and
//! reads its own reply protocol (reasoning, tool calls): DeepSeek V4.1 through
//! deepseek-recipe (`deepseek.rs`), Qwen3.8 through its checkpoint's template
//! and XML tool calls (`qwen.rs`). The errors and identifiers here are both
//! formats'.

pub mod deepseek;
pub mod qwen;
mod qwen_protocol;
mod qwen_template;

use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use serde_json::json;

/// `max_tokens` of a request that sets none, at most what the context leaves.
pub const DEFAULT_MAX_TOKENS: i64 = 262_144;

/// A request the server answers with an error before its response starts.
#[derive(Debug)]
pub struct Error {
    pub status: StatusCode,
    pub message: String,
}

pub fn bad(message: impl Into<String>) -> Error {
    Error { status: StatusCode::BAD_REQUEST, message: message.into() }
}

pub fn error_json(kind: &str, message: &str) -> String {
    json!({"error": {"message": message, "type": kind, "code": null}}).to_string()
}

pub fn error(status: StatusCode, kind: &str, message: &str) -> Response {
    (status, [(header::CONTENT_TYPE, "application/json")], error_json(kind, message)).into_response()
}

/// A response's id.
pub fn response_id() -> String {
    use rand::Rng;
    let s: String = rand::rng().sample_iter(&rand::distr::Alphanumeric).take(24).map(char::from).collect();
    format!("chatcmpl-{s}")
}

/// Which format a checkpoint's requests take: its family, from config.json's
/// model_type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Family {
    DeepSeek,
    Qwen,
}

impl Family {
    /// The family of a checkpoint directory, or why none.
    pub fn of(dir: &str) -> Result<Family, String> {
        let path = std::path::Path::new(dir).join("config.json");
        let text = std::fs::read(&path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        let config: serde_json::Value = serde_json::from_slice(&text).map_err(|e| format!("{}: {e}", path.display()))?;
        match config.get("model_type").and_then(serde_json::Value::as_str) {
            Some("deepseek_v41") => Ok(Family::DeepSeek),
            Some("qwen3_5" | "qwen3_5_text") => Ok(Family::Qwen),
            Some(t) => Err(format!("{}: model_type {t} is not one this server runs", path.display())),
            None => Err(format!("{}: no model_type", path.display())),
        }
    }
}
