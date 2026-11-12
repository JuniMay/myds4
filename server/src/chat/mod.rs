//! The models' chat formats: a Chat Completions request's
//! JSON into a model's prompt tokens and sampling, and the worker's events
//! into the response. Each model family renders its own chat template and
//! reads its own reply protocol (reasoning, tool calls): DeepSeek V4.1 through
//! deepseek-recipe (`deepseek.rs`), Qwen3.8 through its checkpoint's template
//! and XML tool calls (`chatml.rs`). The errors and identifiers here are both
//! formats'.

pub mod chatml;
pub mod deepseek;

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
    Error {
        status: StatusCode::BAD_REQUEST,
        message: message.into(),
    }
}

pub fn image_data(url: &str) -> Result<Vec<u8>, Error> {
    media_data(url)
}

pub fn media_data(url: &str) -> Result<Vec<u8>, Error> {
    let data =
        data_url::DataUrl::process(url).map_err(|_| bad("media require a base64 data URL"))?;
    data.decode_to_vec()
        .map(|(bytes, _)| bytes)
        .map_err(|_| bad("media data is not valid base64"))
}

/// Request values override model defaults; null leaves a default in place.
pub fn sampling(
    root: &serde_json::Map<String, serde_json::Value>,
    defaults: crate::worker::Sampling,
) -> Result<crate::worker::Sampling, Error> {
    use serde_json::Value;
    let number = |key: &str, default: f64| match root.get(key) {
        None | Some(Value::Null) => Ok(default),
        Some(value) => value
            .as_f64()
            .filter(|v| v.is_finite())
            .ok_or_else(|| bad(format!("{key} must be a finite number"))),
    };
    let integer = |key: &str, default: Option<u64>| match root.get(key) {
        None | Some(Value::Null) => Ok(default),
        Some(value) => value
            .as_u64()
            .map(Some)
            .ok_or_else(|| bad(format!("{key} must be a nonnegative integer"))),
    };
    let temperature = number("temperature", defaults.temperature)?;
    let top_p = number("top_p", defaults.top_p)?;
    if temperature < 0.0 || !(0.0..=1.0).contains(&top_p) {
        return Err(bad(
            "temperature must be nonnegative and top_p must be in [0, 1]",
        ));
    }
    let top_k = integer("top_k", Some(defaults.top_k as u64))?.unwrap_or(0);
    if top_k > i32::MAX as u64 {
        return Err(bad("top_k exceeds the engine's integer range"));
    }
    Ok(crate::worker::Sampling {
        temperature,
        top_p,
        top_k: top_k as u32,
        seed: integer("seed", defaults.seed)?,
    })
}

/// Both names accept positive integers. The completion limit takes precedence.
pub fn completion_limit(
    root: &serde_json::Map<String, serde_json::Value>,
    default: i64,
) -> Result<i64, Error> {
    use serde_json::Value;
    let read = |key: &str| match root.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => value
            .as_i64()
            .filter(|&n| n > 0)
            .map(Some)
            .ok_or_else(|| bad(format!("{key} must be a positive integer"))),
    };
    let completion = read("max_completion_tokens")?;
    Ok(completion.or(read("max_tokens")?).unwrap_or(default))
}

pub fn stop_sequences(
    root: &serde_json::Map<String, serde_json::Value>,
) -> Result<Vec<String>, Error> {
    use serde_json::Value;
    match root.get("stop") {
        None | Some(Value::Null) => Ok(Vec::new()),
        Some(Value::String(s)) => Ok(vec![s.clone()]),
        Some(Value::Array(values)) => values
            .iter()
            .map(|v| {
                v.as_str()
                    .map(String::from)
                    .ok_or_else(|| bad("stop must contain only strings"))
            })
            .collect(),
        _ => Err(bad("stop must be a string or an array of strings")),
    }
}

pub fn error_json(kind: &str, message: &str) -> String {
    json!({"error": {"message": message, "type": kind, "code": null}}).to_string()
}

pub fn error(status: StatusCode, kind: &str, message: &str) -> Response {
    (
        status,
        [(header::CONTENT_TYPE, "application/json")],
        error_json(kind, message),
    )
        .into_response()
}

/// The API a request came through: Chat Completions (`/v1/chat/completions`)
/// or Responses (`/v1/responses`, DeepSeek models only).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Api {
    Chat,
    Responses,
}

/// A chat completion's id.
pub fn response_id() -> String {
    random_id("chatcmpl-")
}

/// `prefix` and 24 random letters and digits.
pub fn random_id(prefix: &str) -> String {
    use rand::Rng;
    let s: String = rand::rng()
        .sample_iter(&rand::distr::Alphanumeric)
        .take(24)
        .map(char::from)
        .collect();
    format!("{prefix}{s}")
}

pub use crate::engine::Architecture as Family;

#[cfg(test)]
mod request_tests {
    use super::*;
    use crate::{engine::Tokenizer, worker::Sampling};

    #[test]
    fn qwen_sampling_settings_use_defaults_and_validate_overrides() {
        let Ok(dir) = std::env::var("MYDS4_QWEN") else {
            return;
        };
        let settings = chatml::Settings {
            id: "chatml".into(),
            ctx: 4096,
            think: None,
            variant: chatml::Variant::Qwen,
            tokenizer: Tokenizer::open(&dir, None),
            sampling: Sampling {
                temperature: 1.0,
                top_p: 1.0,
                top_k: 20,
                seed: Some(73),
            },
        };
        let body = json!({"messages": [{"role": "user", "content": "Hi"}]});
        let prepared =
            chatml::prepare(&serde_json::to_vec(&body).unwrap(), "test", &settings, 0).unwrap();
        assert_eq!(prepared.request.sampling, settings.sampling);
        for (field, value) in [
            ("temperature", json!("wrong")),
            ("top_p", json!(2)),
            ("top_k", json!(-1)),
            ("seed", json!(-1)),
        ] {
            let mut bad = body.clone();
            bad[field] = value;
            assert!(
                chatml::prepare(&serde_json::to_vec(&bad).unwrap(), "test", &settings, 0).is_err(),
                "accepted invalid {field}"
            );
        }
    }

    #[test]
    fn deepseek_sampling_settings_keep_requested_top_k() {
        let Ok(dir) = std::env::var("MYDS4_CHECKPOINT") else {
            return;
        };
        let tokenizer = Tokenizer::open(&dir, Some(deepseek::IMAGE));
        let defaults = deepseek::Defaults {
            id: "deepseek".into(),
            sampling: Sampling {
                temperature: 1.0,
                top_p: 1.0,
                top_k: 20,
                seed: Some(73),
            },
            thinking: false,
            effort: 75,
        };
        let body = json!({"messages": [{"role": "user", "content": "Hi"}], "top_k": 31});
        let prepared =
            deepseek::prepare(&serde_json::to_vec(&body).unwrap(), &defaults, tokenizer, Api::Chat).unwrap();
        let request = prepared.engine_request(0, "test".into()).unwrap();
        assert_eq!(request.sampling.top_k, 31);
        assert_eq!(request.sampling.seed, Some(73));
    }

    #[test]
    fn qwen_configured_thinking_is_used_unless_the_request_overrides_it() {
        let Ok(dir) = std::env::var("MYDS4_QWEN") else {
            return;
        };
        let settings = chatml::Settings {
            id: "chatml".into(),
            ctx: 4096,
            think: Some(0),
            variant: chatml::Variant::Qwen,
            tokenizer: Tokenizer::open(&dir, None),
            sampling: Sampling {
                temperature: 1.0,
                top_p: 1.0,
                top_k: 0,
                seed: None,
            },
        };
        let mut body = json!({"messages": [{"role": "user", "content": "Hi"}]});
        let prepare = |body: &serde_json::Value| {
            chatml::prepare(&serde_json::to_vec(body).unwrap(), "test", &settings, 0).unwrap()
        };
        assert!(
            !prepare(&body).thinking,
            "the configured think=0 must disable thinking"
        );
        body["thinking"] = json!(true);
        assert!(prepare(&body).thinking);
    }

    #[test]
    fn both_families_reject_malformed_completion_limits_and_stops() {
        let (Ok(qdir), Ok(ddir)) = (
            std::env::var("MYDS4_QWEN"),
            std::env::var("MYDS4_CHECKPOINT"),
        ) else {
            return;
        };
        let q = chatml::Settings {
            id: "chatml".into(),
            ctx: 4096,
            think: None,
            variant: chatml::Variant::Qwen,
            tokenizer: Tokenizer::open(&qdir, None),
            sampling: Sampling {
                temperature: 1.0,
                top_p: 1.0,
                top_k: 0,
                seed: None,
            },
        };
        let dt = Tokenizer::open(&ddir, Some(deepseek::IMAGE));
        let d = deepseek::Defaults {
            id: "deepseek".into(),
            sampling: q.sampling,
            thinking: false,
            effort: 75,
        };
        let mut accepted = Vec::new();
        for (key, value) in [
            ("max_tokens", json!(0)),
            ("max_tokens", json!(-1)),
            ("max_tokens", json!(1.5)),
            ("max_tokens", json!("12")),
            ("max_completion_tokens", json!(u64::MAX)),
            ("stop", json!(["end", 5])),
        ] {
            let mut body = json!({"messages": [{"role": "user", "content": "Hi"}]});
            body[key] = value;
            let bytes = serde_json::to_vec(&body).unwrap();
            if chatml::prepare(&bytes, "test", &q, 0).is_ok() {
                accepted.push(format!("Qwen accepted {body}"));
            }
            if deepseek::prepare(&bytes, &d, dt, Api::Chat).is_ok() {
                accepted.push(format!("DeepSeek accepted {body}"));
            }
        }
        assert!(accepted.is_empty(), "{}", accepted.join("; "));
    }

    #[test]
    fn qwen_image_parts_become_model_tokens_without_reinterpreting_text() {
        let Ok(dir) = std::env::var("MYDS4_QWEN") else {
            return;
        };
        let tokenizer = Tokenizer::open(&dir, None);
        let image = tokenizer.special("<|image_pad|>").unwrap();
        let settings = chatml::Settings {
            id: "chatml".into(),
            ctx: 4096,
            think: None,
            variant: chatml::Variant::Qwen,
            tokenizer,
            sampling: Sampling {
                temperature: 1.0,
                top_p: 1.0,
                top_k: 0,
                seed: Some(73),
            },
        };
        let body = json!({"messages": [{"role": "user", "content": [
            {"type": "text", "text": "Explain this image. The literal text <|image_pad|> is not an image."},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}}
        ]}]});
        let prepared =
            chatml::prepare(&serde_json::to_vec(&body).unwrap(), "test", &settings, 0).unwrap();
        assert_eq!(prepared.request.media.len(), 1);
        assert_eq!(
            prepared
                .request
                .prompt
                .iter()
                .filter(|&&t| t == image)
                .count(),
            1
        );
    }
}
