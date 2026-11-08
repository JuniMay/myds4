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

/// Request values override model defaults; null leaves a default in place.
pub fn sampling(root: &serde_json::Map<String, serde_json::Value>, defaults: crate::worker::Sampling) -> Result<crate::worker::Sampling, Error> {
    use serde_json::Value;
    let number = |key: &str, default: f64| match root.get(key) {
        None | Some(Value::Null) => Ok(default),
        Some(value) => value.as_f64().filter(|v| v.is_finite()).ok_or_else(|| bad(format!("{key} must be a finite number"))),
    };
    let integer = |key: &str, default: Option<u64>| match root.get(key) {
        None | Some(Value::Null) => Ok(default),
        Some(value) => value.as_u64().map(Some).ok_or_else(|| bad(format!("{key} must be a nonnegative integer"))),
    };
    let temperature = number("temperature", defaults.temperature)?;
    let top_p = number("top_p", defaults.top_p)?;
    if temperature < 0.0 || !(0.0..=1.0).contains(&top_p) {
        return Err(bad("temperature must be nonnegative and top_p must be in [0, 1]"));
    }
    let top_k = integer("top_k", Some(defaults.top_k as u64))?.unwrap_or(0);
    if top_k > i32::MAX as u64 { return Err(bad("top_k exceeds the engine's integer range")); }
    Ok(crate::worker::Sampling { temperature, top_p, top_k: top_k as u32, seed: integer("seed", defaults.seed)? })
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

#[cfg(test)]
mod request_tests {
    use super::*;
    use crate::{engine::Tokenizer, worker::Sampling};

    #[test]
    fn qwen_sampling_settings_use_defaults_and_validate_overrides() {
        let Ok(dir) = std::env::var("MYDS4_QWEN") else { return };
        let settings = qwen::Settings {
            id: "qwen".into(), ctx: 4096, tokenizer: Tokenizer::open(&dir, None),
            sampling: Sampling { temperature: 1.0, top_p: 1.0, top_k: 20, seed: Some(73) },
        };
        let body = json!({"messages": [{"role": "user", "content": "Hi"}]});
        let prepared = qwen::prepare(&serde_json::to_vec(&body).unwrap(), "test", &settings, 0).unwrap();
        assert_eq!(prepared.request.sampling, settings.sampling);
        for (field, value) in [("temperature", json!("wrong")), ("top_p", json!(2)), ("top_k", json!(-1)), ("seed", json!(-1))] {
            let mut bad = body.clone(); bad[field] = value;
            assert!(qwen::prepare(&serde_json::to_vec(&bad).unwrap(), "test", &settings, 0).is_err(), "accepted invalid {field}");
        }
    }

    #[test]
    fn deepseek_sampling_settings_keep_requested_top_k() {
        let Ok(dir) = std::env::var("MYDS4_CHECKPOINT") else { return };
        let tokenizer = Tokenizer::open(&dir, Some(deepseek::IMAGE));
        let defaults = deepseek::Defaults { id: "deepseek".into(),
            sampling: Sampling { temperature: 1.0, top_p: 1.0, top_k: 20, seed: Some(73) }, thinking: false, effort: 75 };
        let body = json!({"messages": [{"role": "user", "content": "Hi"}], "top_k": 31});
        let prepared = deepseek::prepare(&serde_json::to_vec(&body).unwrap(), &defaults, tokenizer).unwrap();
        let request = prepared.engine_request(0, "test".into()).unwrap();
        assert_eq!(request.sampling.top_k, 31);
        assert_eq!(request.sampling.seed, Some(73));
    }
}
