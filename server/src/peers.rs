//! Models served side by side (docs/server.md): the first `-m` runs in this
//! process, and each further one in a child process of this server, on the
//! next port, which this process starts, watches and stops. A request goes
//! to the model its `model` names; this process forwards the others' over
//! loopback HTTP and passes their responses back as they come.

use std::io::{BufRead, BufReader};
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;

use axum::body::{Body, Bytes};
use axum::http::StatusCode;
use axum::response::Response;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

use crate::engine;

/// A model another process serves.
pub struct Peer {
    pub id: String,
    pub name: String,
    pub port: u16,
    pub ctx: i64,
}

static CHILDREN: Mutex<Vec<Child>> = Mutex::new(Vec::new());

/// The child's arguments: this process's, but for the models, the port, the
/// host and the live view, which are its own.
fn child_args(args: &[String], model: &str, port: u16) -> Vec<String> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-m" | "--serve" | "--host" | "--tui-theme" | "--parent" | "--api-key-file" => i += 2,
            "--tui" => i += 1,
            _ => {
                out.push(args[i].clone());
                i += 1;
            }
        }
    }
    let pid = std::process::id().to_string();
    out.extend(["-m", model, "--serve", &port.to_string(), "--host", "127.0.0.1", "--parent", &pid].map(String::from));
    out
}

/// Starts a child process serving `model` on `port`, its log lines in this
/// one's log with the model's id in front.
pub fn spawn(args: &[String], model: &str, id: &str, port: u16, key: &str) -> Result<(), String> {
    let exe = std::env::current_exe().map_err(|e| format!("cannot find this executable: {e}"))?;
    let mut child = Command::new(exe)
        .args(child_args(args, model, port))
        .env("MYDS4_API_KEY", key)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("cannot start the server of {model}: {e}"))?;
    let stderr = child.stderr.take().expect("a piped stderr");
    let tag = id.to_string();
    std::thread::Builder::new()
        .name(format!("log {tag}"))
        .spawn(move || {
            for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                // its lines carry their own time; this log adds its own
                let text = line.split_once(' ').filter(|(t, _)| t.contains(':') && t.len() == 12).map_or(&line[..], |(_, r)| r);
                engine::log(&format!("[{tag}] {text}"));
            }
            engine::log(&format!("[{tag}] its server ended"));
        })
        .map_err(|e| e.to_string())?;
    CHILDREN.lock().unwrap_or_else(|e| e.into_inner()).push(child);
    Ok(())
}

/// Asks the child processes to stop (SIGTERM): their open responses end.
pub fn signal() {
    for c in CHILDREN.lock().unwrap_or_else(|e| e.into_inner()).iter() {
        unsafe { libc::kill(c.id() as i32, libc::SIGTERM) };
    }
}

/// Stops the child processes (SIGTERM) and waits for them.
pub fn stop() {
    let mut children = std::mem::take(&mut *CHILDREN.lock().unwrap_or_else(|e| e.into_inner()));
    for c in &children {
        unsafe { libc::kill(c.id() as i32, libc::SIGTERM) };
    }
    for c in &mut children {
        let _ = c.wait();
    }
}

/// Whether the process `pid` still runs (for --parent).
pub fn alive(pid: i32) -> bool {
    unsafe { libc::kill(pid, 0) == 0 || *libc::__error() == libc::EPERM }
}

fn gateway(message: &str) -> Response {
    crate::chat::error(StatusCode::BAD_GATEWAY, "server_error", message)
}

/// Forwards a request to the peer's server and passes its response back as
/// it comes. HTTP/1.0, so that the response's body ends with the
/// connection.
pub async fn forward(peer: &Peer, method: &str, path: &str, key: &str, body: Bytes) -> Response {
    let Ok(mut s) = TcpStream::connect(("127.0.0.1", peer.port)).await else {
        return gateway(&format!("{} is not reachable (it may still be loading)", peer.id));
    };
    let head = format!(
        "{method} {path} HTTP/1.0\r\nHost: 127.0.0.1:{}\r\nAuthorization: Bearer {key}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
        peer.port,
        body.len()
    );
    if s.write_all(head.as_bytes()).await.is_err() || s.write_all(&body).await.is_err() {
        return gateway(&format!("{} closed the connection", peer.id));
    }
    let mut buf = Vec::with_capacity(4096);
    let end = loop {
        if let Some(i) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            break i + 4;
        }
        let mut chunk = [0u8; 4096];
        match s.read(&mut chunk).await {
            Ok(0) | Err(_) => return gateway(&format!("{} ended the response before its head", peer.id)),
            Ok(n) => buf.extend_from_slice(&chunk[..n]),
        }
        if buf.len() > 64 << 10 {
            return gateway(&format!("{}'s response head is too long", peer.id));
        }
    };
    let head = String::from_utf8_lossy(&buf[..end]).into_owned();
    let mut lines = head.split("\r\n");
    let status = lines.next().and_then(|l| l.split(' ').nth(1)).and_then(|c| c.parse::<u16>().ok()).unwrap_or(502);
    let mut response = Response::builder().status(status);
    for line in lines {
        if let Some((name, value)) = line.split_once(':') {
            let name = name.trim();
            if name.eq_ignore_ascii_case("content-type") || name.eq_ignore_ascii_case("cache-control") {
                response = response.header(name, value.trim());
            }
        }
    }
    let first = Bytes::copy_from_slice(&buf[end..]);
    let rest = async_stream::stream! {
        if !first.is_empty() {
            yield Ok::<Bytes, std::io::Error>(first);
        }
        let mut chunk = vec![0u8; 16 << 10];
        loop {
            match s.read(&mut chunk).await {
                Ok(0) => break,
                Ok(n) => yield Ok(Bytes::copy_from_slice(&chunk[..n])),
                Err(e) => {
                    yield Err(e);
                    break;
                }
            }
        }
    };
    response.body(Body::from_stream(rest)).unwrap_or_else(|_| gateway("a malformed response"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_child_takes_the_flags_but_its_own_model_port_and_view() {
        let args: Vec<String> =
            ["-m", "a", "-m", "b", "--serve", "8008", "--tui", "--kv-cache", "16", "--api-key-file", "k", "--ctx", "65536"]
                .map(String::from)
                .to_vec();
        let got = child_args(&args, "b", 8009);
        let pid = std::process::id().to_string();
        assert_eq!(
            got,
            ["--kv-cache", "16", "--ctx", "65536", "-m", "b", "--serve", "8009", "--host", "127.0.0.1", "--parent", &pid]
        );
    }
}
