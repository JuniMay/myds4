//! The models the server serves and each one's settings:
//! one model from `-m DIR` and its options on the command line, or several
//! from `--config FILE`, a JSON file that lists them, each with its own
//! options under the same names (without the dashes, `-` as `_`). The root
//! holds the listener, authentication, live view and memory settings.

use serde_json::{Map, Value};
use std::path::{Path, PathBuf};

use crate::{engine, ui};

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
    /// Seconds without requests before its weights and experts unload, 0
    /// never; None: the server-wide setting.
    pub idle_unload: Option<f64>,
}

impl ModelConfig {
    pub fn new(model: &str) -> ModelConfig {
        ModelConfig {
            id: None,
            engine: engine::Options {
                model: model.into(),
                ctx: 1 << 20,
                ..Default::default()
            },
            batch: 4,
            temperature: None,
            top_p: None,
            top_k: None,
            seed: None,
            think: None,
            idle_unload: None,
        }
    }

    /// Sets an option from its name (as on the command line, without the
    /// dashes) and value; the option's message when either is invalid.
    pub fn set(&mut self, name: &str, v: &Value) -> Result<(), String> {
        let e = &mut self.engine;
        let num = || v.as_f64().ok_or(format!("{name} takes a number"));
        let int = || {
            v.as_i64()
                .filter(|n| i32::try_from(*n).is_ok())
                .map(|n| n as i32)
                .ok_or(format!("{name} takes an integer"))
        };
        let flag = || v.as_bool().ok_or(format!("{name} takes true or false"));
        let text = || {
            v.as_str()
                .filter(|s| !s.is_empty())
                .map(String::from)
                .ok_or(format!("{name} takes a nonempty string"))
        };
        match name {
            "model" => e.model = text()?,
            "draft_model" => e.set_draft_model(text()?)?,
            "id" => self.id = Some(text()?),
            "experts" => e.deepseek_mut()?.experts = Some(text()?),
            "ctx" => e.ctx = int()?,
            "ced" => {
                e.deepseek_mut()?.ced = match v {
                    Value::String(s) if s == "off" => 0,
                    _ => int()
                        .ok()
                        .filter(|n| (128..=1 << 20).contains(n))
                        .ok_or("ced takes off or 128 to 1048576 replay tokens")?,
                }
            }
            "cache" => return Err("cache is replaced by the server-wide memory setting".into()),
            "cache_half_life" => e.deepseek_mut()?.cache_half_life = int()?,
            "cache_read_cost" => e.deepseek_mut()?.cache_read_cost = flag()?,
            "draft" => e.draft = int()?,
            "lookup" => {
                e.lookup = match v {
                    Value::String(s) if s == "off" => 0,
                    _ => int()?,
                }
            }
            "draft_cache" => e.deepseek_mut()?.draft_cache = int()?,
            "online_predictor" => e.deepseek_mut()?.online_predictor = flag()?,
            "predictor_state" => e.deepseek_mut()?.predictor_state = Some(text()?),
            "prefill_prune" => e.deepseek_mut()?.prefill_prune = flag()?,
            "runahead" => e.deepseek_mut()?.runahead = flag()?,
            "pipeline" => e.deepseek_mut()?.pipeline = flag()?,
            "memory" | "reserve" | "kv_cache" | "kv_disk" | "kv_disk_tokens" | "kv_dir" | "port" | "host"
            | "api_key_file" | "tui" | "tui_theme" => {
                return Err(format!("{name} is server-wide; move it outside models"));
            }
            "batch" => {
                self.batch = int()
                    .ok()
                    .filter(|n| *n >= 1)
                    .ok_or("batch takes a positive integer")? as usize
            }
            "temperature" => {
                self.temperature = Some(
                    num()
                        .ok()
                        .filter(|t| t.is_finite() && *t >= 0.0)
                        .ok_or("temperature must be finite and nonnegative")?,
                )
            }
            "top_p" => {
                self.top_p = Some(
                    num()
                        .ok()
                        .filter(|p| (0.0..=1.0).contains(p))
                        .ok_or("top_p must be in [0, 1]")?,
                )
            }
            "top_k" => {
                self.top_k = Some(
                    v.as_u64()
                        .and_then(|n| u32::try_from(n).ok())
                        .ok_or("top_k takes a nonnegative integer")?,
                )
            }
            "seed" => self.seed = Some(v.as_u64().ok_or("seed takes a nonnegative integer")?),
            "think" => {
                self.think = Some(
                    v.as_u64()
                        .filter(|n| *n <= 100)
                        .ok_or("think takes 0 (off) or 1 to 100")? as u32,
                )
            }
            "idle_unload" => self.idle_unload = Some(idle_seconds(v)?),
            _ => return Err(format!("no option {name}")),
        }
        Ok(())
    }
}

/// Listener, authentication, terminal and saved-state settings for the whole
/// server.
#[derive(Clone, Debug, PartialEq)]
pub struct Server {
    pub port: Option<u16>,
    pub host: String,
    pub api_key_file: Option<String>,
    pub tui: bool,
    pub tui_theme: ui::Colors,
    /// The folder of the states saved on the SSD (server/src/disk.rs); None:
    /// [`default_kv_dir`].
    pub kv_dir: Option<String>,
}

impl Default for Server {
    fn default() -> Self {
        Self {
            port: None,
            host: "127.0.0.1".into(),
            api_key_file: None,
            tui: false,
            tui_theme: ui::Colors::Auto,
            kv_dir: None,
        }
    }
}

/// $HOME/.cache/myds4/kv: where states are saved unless kv_dir says otherwise.
pub fn default_kv_dir() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .filter(|s| !s.is_empty())
        .map(|home| PathBuf::from(home).join(".cache/myds4/kv"))
}

impl Server {
    pub fn set(&mut self, name: &str, value: &Value) -> Result<(), String> {
        let text = || {
            value
                .as_str()
                .filter(|s| !s.is_empty())
                .ok_or(format!("{name} takes nonempty text"))
        };
        match name {
            "port" => {
                self.port = Some(
                    value
                        .as_u64()
                        .and_then(|n| u16::try_from(n).ok())
                        .ok_or("port takes an integer from 0 to 65535")?,
                )
            }
            "host" => {
                let host = text()?;
                if host != "localhost" && host.parse::<std::net::Ipv4Addr>().is_err() {
                    return Err("host takes an IPv4 address or localhost".into());
                }
                self.host = host.into();
            }
            "api_key_file" => self.api_key_file = Some(text()?.into()),
            "kv_dir" => self.kv_dir = Some(text()?.into()),
            "tui" => self.tui = value.as_bool().ok_or("tui takes true or false")?,
            "tui_theme" => {
                self.tui_theme = match text()? {
                    "auto" => ui::Colors::Auto,
                    "light" => ui::Colors::Light,
                    "dark" => ui::Colors::Dark,
                    _ => return Err("tui_theme takes auto, light or dark".into()),
                }
            }
            _ => return Err(format!("no server setting {name}")),
        }
        Ok(())
    }
}

/// Seconds without requests before a model unloads: 0 (never) to 1,000,000.
fn idle_seconds(v: &Value) -> Result<f64, String> {
    v.as_f64()
        .filter(|n| (0.0..=1e6).contains(n))
        .ok_or("idle_unload takes 0 (never) to 1000000 seconds".into())
}

/// Global model-memory and KV budgets, the process-wide reserve, the
/// seconds a model may go without requests before it unloads, and the room
/// for saved states on the SSD.
#[derive(Clone, Debug, PartialEq)]
pub struct Budgets {
    pub memory_gb: f64,
    pub reserve_gb: f64,
    pub kv_gb: f64,
    /// 0: never; a model's own idle_unload setting overrides it.
    pub idle_unload_s: f64,
    /// What saved states may take on the SSD; 0: none are saved or read.
    pub kv_disk_gb: f64,
    /// A state is saved when it holds at least this many tokens that no
    /// other kept state holds.
    pub kv_disk_tokens: usize,
}

impl Default for Budgets {
    fn default() -> Self {
        Self {
            memory_gb: 0.0,
            reserve_gb: 10.0,
            kv_gb: 8.0,
            idle_unload_s: 600.0,
            kv_disk_gb: 32.0,
            kv_disk_tokens: 8192,
        }
    }
}

impl Budgets {
    pub fn set(&mut self, name: &str, value: &Value) -> Result<(), String> {
        if name == "idle_unload" {
            self.idle_unload_s = idle_seconds(value)?;
            return Ok(());
        }
        let n = value
            .as_f64()
            .filter(|n| n.is_finite())
            .ok_or(format!("{name} takes a finite number"))?;
        match name {
            "memory" if (0.0..=1024.0).contains(&n) => self.memory_gb = n,
            "memory" => {
                return Err(
                    "memory takes 0 (automatic) or a positive GiB budget up to 1024".into(),
                );
            }
            "reserve" if (2.0..=1024.0).contains(&n) => self.reserve_gb = n,
            "kv_cache" if n > 0.0 && n <= 1024.0 => self.kv_gb = n,
            "kv_disk" if (0.0..=65536.0).contains(&n) => self.kv_disk_gb = n,
            "kv_disk_tokens" if n >= 1.0 && n <= (1 << 20) as f64 && n.fract() == 0.0 => {
                self.kv_disk_tokens = n as usize
            }
            "reserve" => return Err("reserve takes 2 to 1024 GiB".into()),
            "kv_cache" => return Err("kv_cache takes a positive budget up to 1024 GiB".into()),
            "kv_disk" => return Err("kv_disk takes 0 (off) or a GiB budget up to 65536".into()),
            "kv_disk_tokens" => return Err("kv_disk_tokens takes 1 to 1048576 tokens".into()),
            _ => return Err(format!("no server budget {name}")),
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Config {
    pub server: Server,
    pub budgets: Budgets,
    pub models: Vec<ModelConfig>,
}

pub fn default_path() -> Result<PathBuf, String> {
    std::env::var_os("HOME")
        .filter(|s| !s.is_empty())
        .map(|home| PathBuf::from(home).join(".config/myds4/server.json"))
        .ok_or("HOME is unset; specify --config FILE or -m DIR".into())
}

/// Reads the server settings and the models they serve.
pub fn read(path: &Path) -> Result<Config, String> {
    let name = path.display();
    let text = std::fs::read(path).map_err(|e| format!("cannot read {name}: {e}"))?;
    let root: Value = serde_json::from_slice(&text).map_err(|e| format!("{name}: {e}"))?;
    parse(&root).map_err(|e| format!("{name}: {e}"))
}

pub fn parse(root: &Value) -> Result<Config, String> {
    let Some(obj) = root.as_object() else {
        return Err("not a JSON object".into());
    };
    if let Some(k) = obj.keys().find(|k| {
        ![
            "models",
            "memory",
            "reserve",
            "kv_cache",
            "kv_disk",
            "kv_disk_tokens",
            "kv_dir",
            "idle_unload",
            "port",
            "host",
            "api_key_file",
            "tui",
            "tui_theme",
        ]
        .contains(&k.as_str())
    }) {
        return Err(format!("no server setting {k}"));
    }
    let mut budgets = Budgets::default();
    for name in ["memory", "reserve", "kv_cache", "kv_disk", "kv_disk_tokens", "idle_unload"] {
        if let Some(value) = obj.get(name) {
            budgets.set(name, value)?;
        }
    }
    let mut server = Server::default();
    for name in ["port", "host", "api_key_file", "kv_dir", "tui", "tui_theme"] {
        if let Some(value) = obj.get(name) {
            server.set(name, value)?;
        }
    }
    let list = obj
        .get("models")
        .and_then(Value::as_array)
        .filter(|l| !l.is_empty())
        .ok_or("models must be a nonempty list")?;
    let mut out = Vec::new();
    for (i, m) in list.iter().enumerate() {
        let m: &Map<String, Value> = m
            .as_object()
            .ok_or(format!("models[{i}] is not an object"))?;
        let dir = m
            .get("model")
            .and_then(Value::as_str)
            .ok_or(format!("models[{i}] has no model directory"))?;
        let mut c = ModelConfig::new(dir);
        for (k, v) in m {
            c.set(k, v).map_err(|e| format!("models[{i}]: {e}"))?;
        }
        out.push(c);
    }
    Ok(Config {
        server,
        budgets,
        models: out,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn idle_unload_is_server_wide_and_a_model_can_set_its_own() {
        let c = parse(&json!({"idle_unload": 120, "models": [
            {"model": "/models/deepseek"}, {"model": "/models/qwen", "idle_unload": 0}
        ]}))
        .unwrap();
        assert_eq!(c.budgets.idle_unload_s, 120.0);
        assert_eq!(
            (c.models[0].idle_unload, c.models[1].idle_unload),
            (None, Some(0.0))
        );
        // ten minutes when nothing sets it
        let c = parse(&json!({"models": [{"model": "/models/deepseek"}]})).unwrap();
        assert_eq!(c.budgets.idle_unload_s, 600.0);
        for bad in [json!(-1), json!(2e6), json!("soon")] {
            assert!(parse(&json!({"idle_unload": bad, "models": [{"model": "/m"}]})).is_err());
            assert!(parse(&json!({"models": [{"model": "/m", "idle_unload": bad}]})).is_err());
        }
    }

    #[test]
    fn model_memory_is_global_and_separate_from_kv() {
        let root = json!({"memory": 64, "reserve": 10, "kv_cache": 12, "models": [
            {"model": "/models/deepseek"}, {"model": "/models/qwen"}
        ]});
        assert!(
            parse(&root).is_ok(),
            "one model-memory budget must serve every architecture"
        );
        for key in ["cache", "memory"] {
            let nested = json!({"models": [{"model": "/models/any-model", key: 32}]});
            assert!(
                parse(&nested).is_err(),
                "{key} must not create a per-model memory budget"
            );
        }
    }

    #[test]
    fn reserve_and_kv_cache_are_server_wide() {
        let root = json!({"memory": 56, "reserve": 10, "kv_cache": 12, "models": [
            {"model": "/models/deepseek"}, {"model": "/models/qwen"}
        ]});
        assert!(
            parse(&root).is_ok(),
            "one reserve and one KV budget must cover every model"
        );
    }

    #[test]
    fn model_entries_cannot_create_separate_memory_budgets() {
        for setting in ["memory", "reserve", "kv_cache"] {
            let mut model = json!({"model": "/models/qwen"});
            model[setting] = json!(8);
            assert!(
                parse(&json!({"models": [model]})).is_err(),
                "{setting} belongs at the server level"
            );
        }
    }

    #[test]
    fn server_settings_are_global_and_cli_values_can_override_them() {
        let mut c = parse(
            &json!({"port": 8008, "host": "localhost", "api_key_file": "key-file", "tui": true,
                                "tui_theme": "dark", "models": [{"model": "/models/deepseek"}]}),
        )
        .unwrap();
        assert_eq!(
            (
                c.server.port,
                c.server.host.as_str(),
                c.server.api_key_file.as_deref()
            ),
            (Some(8008), "localhost", Some("key-file"))
        );
        assert!(c.server.tui);
        assert_eq!(c.server.tui_theme, ui::Colors::Dark);
        c.server.set("port", &json!(9008)).unwrap();
        c.server.set("tui", &json!(false)).unwrap();
        c.server.set("tui_theme", &json!("light")).unwrap();
        assert_eq!(
            (c.server.port, c.server.tui, c.server.tui_theme),
            (Some(9008), false, ui::Colors::Light)
        );
        for (name, value) in [
            ("port", json!(-1)),
            ("port", json!(65536)),
            ("port", json!(1.5)),
            ("port", json!("8008")),
            ("host", json!("not-an-address")),
            ("api_key_file", json!("")),
            ("tui", json!(1)),
            ("tui_theme", json!("blue")),
        ] {
            assert!(c.server.set(name, &value).is_err(), "{name} {value}");
        }
    }

    #[test]
    fn a_file_gives_each_model_its_own_options() {
        let c = parse(&json!({"memory": 72, "kv_cache": 16, "models": [
            {"model": "/m/deepseek", "ced": 128, "batch": 4},
            {"model": "/m/qwen", "id": "qwen", "draft": 1, "ctx": 32768, "top_k": 20, "think": 0}
        ]}))
        .unwrap();
        let models = c.models;
        assert_eq!(models.len(), 2);
        let (d, q) = (&models[0], &models[1]);
        assert_eq!(
            (
                d.engine.model.as_str(),
                c.budgets.memory_gb,
                d.engine.deepseek().unwrap().ced,
                c.budgets.kv_gb
            ),
            ("/m/deepseek", 72.0, 128, 16.0)
        );
        assert_eq!((d.engine.draft, d.id.as_deref()), (0, None));
        assert_eq!(
            (
                q.id.as_deref(),
                q.engine.draft,
                q.engine.ctx,
                q.top_k,
                q.think
            ),
            (Some("qwen"), 1, 32768, Some(20), Some(0))
        );
        assert!(q.engine.deepseek().is_none());
    }

    #[test]
    fn pipeline_is_explicit_and_per_model() {
        let models = parse(&json!({"models": [
            {"model": "/models/deepseek", "pipeline": true}, {"model": "/models/qwen"}
        ]}))
        .unwrap()
        .models;
        assert!(models[0].engine.deepseek().unwrap().pipeline);
        assert!(!models[0].engine.deepseek().unwrap().runahead);
        assert!(models[1].engine.deepseek().is_none());
        assert!(
            parse(&json!({"models": [{"model": "/models/deepseek", "pipeline": "on"}]})).is_err()
        );
    }

    #[test]
    fn backend_options_must_match_the_checkpoint_family() {
        let mut models = parse(&json!({"models": [
            {"model": "/models/deepseek", "pipeline": true}, {"model": "/models/qwen", "draft": 1}
        ]}))
        .unwrap()
        .models;
        assert!(
            models[0]
                .engine
                .configure(engine::Architecture::Qwen)
                .is_err()
        );
        models[0]
            .engine
            .configure(engine::Architecture::DeepSeek)
            .unwrap();
        models[1]
            .engine
            .configure(engine::Architecture::Qwen)
            .unwrap();
        assert!(models[1].set("ced", &json!(128)).is_err());
        assert!(matches!(
            models[1].engine.backend,
            engine::BackendOptions::Qwen
        ));
    }

    #[test]
    fn separate_draft_path_belongs_to_nemotron() {
        let mut model = ModelConfig::new("/models/transcribe");
        model.set("draft_model", &json!("/models/dflash2")).unwrap();
        model.set("draft", &json!(7)).unwrap();
        model
            .engine
            .configure(engine::Architecture::Nemotron)
            .unwrap();
        assert!(
            matches!(&model.engine.backend, engine::BackendOptions::Nemotron { draft_model: Some(path) } if path == "/models/dflash2")
        );
        assert!(model.engine.configure(engine::Architecture::Qwen).is_err());
    }

    #[test]
    fn unknown_or_invalid_options_are_refused() {
        assert!(parse(&json!({"models": []})).is_err());
        assert!(parse(&json!({"models": [{"cache": 1}]})).is_err());
        assert!(
            parse(&json!({"models": [{"model": "/m", "cahce": 1}]}))
                .unwrap_err()
                .contains("cahce")
        );
        assert!(parse(&json!({"models": [{"model": "/m", "ced": 64}]})).is_err());
        assert!(parse(&json!({"models": [{"model": "/m", "top_p": 2}]})).is_err());
        assert!(parse(&json!({"prt": 8008, "models": [{"model": "/m"}]})).is_err());
        assert_eq!(
            parse(&json!({"models": [{"model": "/m", "ced": "off", "lookup": "off"}]}))
                .unwrap()
                .models[0]
                .engine
                .deepseek()
                .unwrap()
                .ced,
            0
        );
    }
}
