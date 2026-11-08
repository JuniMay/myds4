//! Startup settings without model loading. An empty key or an occupied
//! listener stops startup before the engine opens a checkpoint.
use std::io::Write;
use std::net::TcpListener;
use std::path::PathBuf;
use std::process::{Command, Stdio};

use serde_json::{Value, json};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_owned()
}

fn run(args: &[String], config: Option<Value>) -> String {
    let mut child = Command::new(env!("CARGO_BIN_EXE_myds4-server"))
        .args(args)
        .current_dir(root().join("server"))
        .env("HOME", root().join("server/tests/fixtures/default-home"))
        .env_remove("XDG_CONFIG_HOME")
        .env("MYDS4_API_KEY", "startup-check-key-not-a-secret")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("server starts");
    if let Some(c) = config {
        child
            .stdin
            .take()
            .unwrap()
            .write_all(c.to_string().as_bytes())
            .unwrap();
    } else {
        drop(child.stdin.take());
    }
    let output = child.wait_with_output().unwrap();
    assert!(
        !output.status.success(),
        "fixture must stop before model loading"
    );
    String::from_utf8(output.stderr).unwrap()
}

#[test]
fn a_config_file_supplies_the_listener_and_key_file() {
    let c = json!({"port": 8008, "api_key_file": "/dev/null", "tui": true,
                   "models": [{"model": "unused-model"}]});
    let error = run(&["--config".into(), "/dev/stdin".into()], Some(c));
    assert!(error.contains("must have 16 to 256 characters"), "{error}");
}

#[test]
fn no_arguments_loads_the_default_home_config() {
    let error = run(&[], None);
    assert!(error.contains("must have 16 to 256 characters"), "{error}");
}

#[test]
fn command_line_key_file_overrides_the_default_config() {
    let error = run(
        &["--api-key-file".into(), "missing-override-key".into()],
        None,
    );
    assert!(
        error.contains("cannot open missing-override-key"),
        "{error}"
    );
}

#[test]
fn explicit_config_and_cli_listener_override_stop_before_model_loading() {
    let guard = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = guard.local_addr().unwrap().port();
    let base = json!({"port": port, "host": "127.0.0.1", "tui": false, "tui_theme": "light",
                      "models": [{"model": "unused-model"}]});
    let error = run(
        &["--config".into(), "/dev/stdin".into()],
        Some(base.clone()),
    );
    assert!(
        error.contains(&format!("cannot listen on 127.0.0.1:{port}")),
        "{error}"
    );
    for before in [false, true] {
        let mut c = base.clone();
        c["port"] = json!(0);
        c["api_key_file"] = json!("missing-config-key");
        let cli = vec![
            "--serve".into(),
            port.to_string(),
            "--api-key-file".into(),
            "/dev/null".into(),
            "--no-tui".into(),
        ];
        let file = vec!["--config".into(), "/dev/stdin".into()];
        let args = if before {
            [cli, file].concat()
        } else {
            [file, cli].concat()
        };
        let error = run(&args, Some(c));
        assert!(error.contains("must have 16 to 256 characters"), "{error}");
    }
    let mut c = base;
    c["port"] = json!(0);
    let error = run(
        &[
            "--config".into(),
            "/dev/stdin".into(),
            "--serve".into(),
            port.to_string(),
        ],
        Some(c),
    );
    assert!(
        error.contains(&format!("cannot listen on 127.0.0.1:{port}")),
        "{error}"
    );
}

#[test]
fn an_explicit_model_uses_cli_settings_without_the_default_file() {
    let guard = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = guard.local_addr().unwrap().port();
    let error = run(
        &[
            "-m".into(),
            "unused-model".into(),
            "--serve".into(),
            port.to_string(),
        ],
        None,
    );
    assert!(
        error.contains(&format!("cannot listen on 127.0.0.1:{port}")),
        "{error}"
    );
}
