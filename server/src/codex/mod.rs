//! What Codex needs beyond the Responses API: its model catalog
//! (GET /v1/codex/models, Codex's `model_catalog_url`), and the service behind
//! its web tool (search.rs, POST /v1/alpha/search).
//!
//! For a model without catalog metadata, Codex assumes a context window of
//! 272,000 tokens, caps its `model_context_window` setting there, and offers
//! no `apply_patch` tool. The catalog gives each DeepSeek model its served
//! context, the reasoning levels the server maps to DeepSeek's efforts,
//! Codex's own instructions and the `apply_patch` tool. Every catalog entry
//! must carry instructions; Codex rejects a catalog with an entry without them.

pub mod search;

use serde_json::{Value, json};

/// Codex's instructions for models without their own: `prompt.md` of Codex
/// 0.167.2 (assets/codex/README.md says where it comes from and how to
/// update it).
pub const PROMPT: &str = include_str!("../../assets/codex/prompt.md");

/// A model the catalog lists: a DeepSeek model, which answers the Responses
/// API that Codex uses.
pub struct CatalogModel<'a> {
    pub id: &'a str,
    pub name: &'a str,
    pub ctx: usize,
    /// The model's default when a request names no effort: None for thinking
    /// off, else the effort from 1 to 100.
    pub effort: Option<u32>,
}

/// Codex's reasoning level for a DeepSeek effort: the server reads low as
/// 50, high as 75 and max as 100.
fn level(effort: Option<u32>) -> &'static str {
    match effort {
        None => "none",
        Some(e) if e <= 50 => "low",
        Some(e) if e >= 100 => "max",
        Some(_) => "high",
    }
}

/// The catalog entry of one model, in Codex's `ModelInfo` form.
pub fn entry(m: &CatalogModel) -> Value {
    json!({
        "slug": m.id,
        "display_name": m.name,
        "description": "Served by myds4",
        "default_reasoning_level": level(m.effort),
        "supported_reasoning_levels": [
            {"effort": "none", "description": "Thinking off"},
            {"effort": "low", "description": "DeepSeek reasoning effort 50"},
            {"effort": "high", "description": "DeepSeek reasoning effort 75"},
            {"effort": "max", "description": "DeepSeek reasoning effort 100"},
        ],
        // The fallback's values where the model gives no reason to differ.
        "shell_type": "unified_exec",
        "visibility": "list",
        "supported_in_api": true,
        "priority": 0,
        "availability_nux": null,
        "upgrade": null,
        "model_messages": {"instructions_template": PROMPT},
        "include_skills_usage_instructions": false,
        "include_plugin_usage_instructions": false,
        "include_apps_usage_instructions": false,
        // The server returns reasoning as text, never as summaries.
        "supports_reasoning_summary_parameter": true,
        "default_reasoning_summary": "none",
        "support_verbosity": false,
        "default_verbosity": null,
        // deepseek-recipe turns the custom apply_patch tool into a function
        // the model knows and returns its calls as custom tool calls.
        "apply_patch_tool_type": "freeform",
        "web_search_tool_type": "text",
        "truncation_policy": {"mode": "bytes", "limit": 10000},
        "supports_image_detail_original": false,
        "context_window": m.ctx,
        "max_context_window": m.ctx,
        "effective_context_window_percent": 95,
        "experimental_supported_tools": [],
        "input_modalities": ["text", "image"],
        "supports_parallel_tool_calls": true,
        "supports_search_tool": false,
    })
}

/// The catalog: `{"models": [...]}`.
pub fn catalog<'a>(models: impl IntoIterator<Item = CatalogModel<'a>>) -> Value {
    let models: Vec<Value> = models.into_iter().map(|m| entry(&m)).collect();
    json!({ "models": models })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entries_carry_the_served_context_the_instructions_and_apply_patch() {
        let c = catalog([CatalogModel {
            id: "deepseek-v4.1-flash",
            name: "DeepSeek V4.1 Flash (myds4)",
            ctx: 1_048_576,
            effort: Some(75),
        }]);
        let m = &c["models"][0];
        assert_eq!(m["slug"], "deepseek-v4.1-flash");
        assert_eq!((m["context_window"].as_u64(), m["max_context_window"].as_u64()), (Some(1_048_576), Some(1_048_576)));
        assert_eq!(m["apply_patch_tool_type"], "freeform");
        assert_eq!(m["default_reasoning_level"], "high");
        let template = m["model_messages"]["instructions_template"].as_str().unwrap();
        assert!(template.starts_with("You are a coding agent running in the Codex CLI"), "{}", &template[..80]);
        let levels: Vec<&str> = m["supported_reasoning_levels"]
            .as_array()
            .unwrap()
            .iter()
            .map(|l| l["effort"].as_str().unwrap())
            .collect();
        assert_eq!(levels, ["none", "low", "high", "max"]);
    }

    #[test]
    fn efforts_become_the_levels_the_server_reads_back() {
        assert_eq!(level(None), "none");
        assert_eq!(level(Some(50)), "low");
        assert_eq!(level(Some(30)), "low");
        assert_eq!(level(Some(75)), "high");
        assert_eq!(level(Some(62)), "high");
        assert_eq!(level(Some(100)), "max");
    }
}
