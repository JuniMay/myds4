//! The models the server serves and each one's settings (docs/server.md):
//! one model from `-m DIR` and its options on the command line, or several
//! from `--config FILE`, a JSON file that lists them, each with its own
//! options under the same names (without the dashes, `-` as `_`). The root
//! holds the process-wide reserve and shared KV budget.

use serde_json::{Map, Value};

use crate::engine;

/// One model's settings: the engine's, the worker's and its requests'
/// defaults.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelConfig {
    /// The name requests give in `model`; None: the family's default name.
    pub id: Option<String>,
    pub engine: engine::Options,
    /// Requests running together at most.
    pub batch: usize,
    /// Its requests' defaults where a request sets none; None: the
    /// checkpoint's generation_config.json, else the format's own.
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<u32>,
    pub seed: Option<u64>,
    /// Thinking effort 1-100, 0 off; None: the format's default.
    pub think: Option<u32>,
}

impl ModelConfig {
    pub fn new(model: &str) -> ModelConfig {
        ModelConfig {
            id: None,
            engine: engine::Options { model: model.into(), ctx: 1 << 20, ..Default::default() },
            batch: 4,
            temperature: None,
            top_p: None,
            top_k: None,
            seed: None,
            think: None,
        }
    }

    /// Sets an option from its name (as on the command line, without the
    /// dashes) and value; the option's message when either is invalid.
    pub fn set(&mut self, name: &str, v: &Value) -> Result<(), String> {
        let e = &mut self.engine;
        let num = || v.as_f64().ok_or(format!("{name} takes a number"));
        let int = || v.as_i64().filter(|n| i32::try_from(*n).is_ok()).map(|n| n as i32).ok_or(format!("{name} takes an integer"));
        let flag = || v.as_bool().ok_or(format!("{name} takes true or false"));
        let text = || v.as_str().filter(|s| !s.is_empty()).map(String::from).ok_or(format!("{name} takes a nonempty string"));
        match name {
            "model" => e.model = text()?,
            "id" => self.id = Some(text()?),
            "experts" => e.experts = Some(text()?),
            "ctx" => e.ctx = int()?,
            "ced" => {
                e.ced = match v {
                    Value::String(s) if s == "off" => 0,
                    _ => int().ok().filter(|n| (128..=1 << 20).contains(n)).ok_or("ced takes off or 128 to 1048576 replay tokens")?,
                }
            }
            "cache" => e.cache_gb = num()?,
            "cache_half_life" => e.cache_half_life = int()?,
            "cache_read_cost" => e.cache_read_cost = flag()?,
            "substitute" => e.substitute = num()? as f32,
            "draft" => e.draft = int()?,
            "lookup" => {
                e.lookup = match v {
                    Value::String(s) if s == "off" => 0,
                    _ => int()?,
                }
            }
            "draft_cache" => e.draft_cache = int()?,
            "online_predictor" => e.online_predictor = flag()?,
            "predictor_state" => e.predictor_state = Some(text()?),
            "prefill_prune" => e.prefill_prune = flag()?,
            "runahead" => e.runahead = flag()?,
            "pipeline" => e.pipeline = flag()?,
            "reserve" | "kv_cache" => return Err(format!("{name} is server-wide; move it outside models")),
            "batch" => self.batch = int().ok().filter(|n| *n >= 1).ok_or("batch takes a positive integer")? as usize,
            "temperature" => {
                self.temperature = Some(num().ok().filter(|t| t.is_finite() && *t >= 0.0).ok_or("temperature must be finite and nonnegative")?)
            }
            "top_p" => self.top_p = Some(num().ok().filter(|p| (0.0..=1.0).contains(p)).ok_or("top_p must be in [0, 1]")?),
            "top_k" => self.top_k = Some(v.as_u64().and_then(|n| u32::try_from(n).ok()).ok_or("top_k takes a nonnegative integer")?),
            "seed" => self.seed = Some(v.as_u64().ok_or("seed takes a nonnegative integer")?),
            "think" => self.think = Some(v.as_u64().filter(|n| *n <= 100).ok_or("think takes 0 (off) or 1 to 100")? as u32),
            _ => return Err(format!("no option {name}")),
        }
        Ok(())
    }
}

/// One process-wide reserve and one KV budget shared by every model.
#[derive(Clone, Debug, PartialEq)]
pub struct Budgets {
    pub reserve_gb: f64,
    pub kv_gb: f64,
}

impl Default for Budgets {
    fn default() -> Self { Self { reserve_gb: 10.0, kv_gb: 8.0 } }
}

impl Budgets {
    pub fn set(&mut self, name: &str, value: &Value) -> Result<(), String> {
        let n = value.as_f64().filter(|n| n.is_finite()).ok_or(format!("{name} takes a finite number"))?;
        match name {
            "reserve" if (2.0..=1024.0).contains(&n) => self.reserve_gb = n,
            "kv_cache" if n > 0.0 && n <= 1024.0 => self.kv_gb = n,
            "reserve" => return Err("reserve takes 2 to 1024 GiB".into()),
            "kv_cache" => return Err("kv_cache takes a positive budget up to 1024 GiB".into()),
            _ => return Err(format!("no server budget {name}")),
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Config {
    pub budgets: Budgets,
    pub models: Vec<ModelConfig>,
}

/// Reads process-wide budgets and the models they serve.
pub fn read(path: &str) -> Result<Config, String> {
    let text = std::fs::read(path).map_err(|e| format!("cannot read {path}: {e}"))?;
    let root: Value = serde_json::from_slice(&text).map_err(|e| format!("{path}: {e}"))?;
    parse(&root).map_err(|e| format!("{path}: {e}"))
}

pub fn parse(root: &Value) -> Result<Config, String> {
    let Some(obj) = root.as_object() else { return Err("not a JSON object".into()) };
    if let Some(k) = obj.keys().find(|k| !["models", "reserve", "kv_cache"].contains(&k.as_str())) {
        return Err(format!("no server setting {k}"));
    }
    let mut budgets = Budgets::default();
    for name in ["reserve", "kv_cache"] {
        if let Some(value) = obj.get(name) { budgets.set(name, value)?; }
    }
    let list = obj.get("models").and_then(Value::as_array).filter(|l| !l.is_empty()).ok_or("models must be a nonempty list")?;
    let mut out = Vec::new();
    for (i, m) in list.iter().enumerate() {
        let m: &Map<String, Value> = m.as_object().ok_or(format!("models[{i}] is not an object"))?;
        let dir = m.get("model").and_then(Value::as_str).ok_or(format!("models[{i}] has no model directory"))?;
        let mut c = ModelConfig::new(dir);
        for (k, v) in m {
            c.set(k, v).map_err(|e| format!("models[{i}]: {e}"))?;
        }
        out.push(c);
    }
    Ok(Config { budgets, models: out })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn reserve_and_kv_cache_are_server_wide() {
        let root = json!({"reserve": 10, "kv_cache": 12, "models": [
            {"model": "/models/deepseek", "cache": 56}, {"model": "/models/qwen"}
        ]});
        assert!(parse(&root).is_ok(), "one reserve and one KV budget must cover every model");
    }

    #[test]
    fn model_entries_cannot_create_separate_memory_budgets() {
        for setting in ["reserve", "kv_cache"] {
            let mut model = json!({"model": "/models/qwen"});
            model[setting] = json!(8);
            assert!(parse(&json!({"models": [model]})).is_err(), "{setting} belongs at the server level");
        }
    }

    #[test]
    fn a_file_gives_each_model_its_own_options() {
        let c = parse(&json!({"kv_cache": 16, "models": [
            {"model": "/m/deepseek", "cache": 72, "ced": 128, "batch": 4},
            {"model": "/m/qwen", "id": "qwen", "draft": 1, "ctx": 32768, "top_k": 20, "think": 0}
        ]}))
        .unwrap();
        let models = c.models;
        assert_eq!(models.len(), 2);
        let (d, q) = (&models[0], &models[1]);
        assert_eq!((d.engine.model.as_str(), d.engine.cache_gb, d.engine.ced, c.budgets.kv_gb), ("/m/deepseek", 72.0, 128, 16.0));
        assert_eq!((d.engine.draft, d.id.as_deref()), (0, None));
        assert_eq!((q.id.as_deref(), q.engine.draft, q.engine.ctx, q.top_k, q.think), (Some("qwen"), 1, 32768, Some(20), Some(0)));
        assert_eq!(q.engine.cache_gb, 0.0);
    }

    #[test]
    fn pipeline_is_explicit_and_per_model() {
        let models = parse(&json!({"models": [
            {"model": "/models/deepseek", "pipeline": true}, {"model": "/models/qwen"}
        ]})).unwrap().models;
        assert!(models[0].engine.pipeline);
        assert!(!models[0].engine.runahead);
        assert!(!models[1].engine.pipeline);
        assert!(parse(&json!({"models": [{"model": "/models/deepseek", "pipeline": "on"}]})).is_err());
    }

    #[test]
    fn unknown_or_invalid_options_are_refused() {
        assert!(parse(&json!({"models": []})).is_err());
        assert!(parse(&json!({"models": [{"cache": 1}]})).is_err());
        assert!(parse(&json!({"models": [{"model": "/m", "cahce": 1}]})).unwrap_err().contains("cahce"));
        assert!(parse(&json!({"models": [{"model": "/m", "ced": 64}]})).is_err());
        assert!(parse(&json!({"models": [{"model": "/m", "top_p": 2}]})).is_err());
        assert!(parse(&json!({"port": 8008, "models": [{"model": "/m"}]})).is_err());
        assert_eq!(parse(&json!({"models": [{"model": "/m", "ced": "off", "lookup": "off"}]})).unwrap().models[0].engine.ced, 0);
    }
}
