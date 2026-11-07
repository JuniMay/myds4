//! The server's connections and their limits: at most 64 open at once
//! (another is answered 503 and closed), a request's line and headers within
//! 60 s (a late head closes the connection) and 64 KB (431), each part of its
//! body within 60 s of the one before (408), a body of at most 32 MB (413),
//! and writes that a client may stall at most 60 s before its connection
//! closes, which cancels its request.
//!
//! Reads have no deadline of their own: hyper reads while a response is
//! pending, so that a client that closes its connection ends its request
//! whether or not the response is streamed.

use std::future::Future;
use std::io;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Duration;

use axum::body::{Body, Bytes, HttpBody};
use axum::extract::{ConnectInfo, Request};
use axum::http::StatusCode;
use axum::middleware::{self, Next};
use axum::response::Response;
use axum::{Extension, Router};
use http_body_util::BodyExt;
use hyper::server::conn::http1;
use hyper_util::rt::{TokioIo, TokioTimer};
use hyper_util::server::graceful::GracefulShutdown;
use hyper_util::service::TowerToHyperService;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt, ReadBuf};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::Semaphore;
use tokio::time::Sleep;

use crate::{engine, format};

/// The limits on connections and requests.
#[derive(Clone, Copy)]
pub struct Limits {
    /// Connections open at once; another is answered 503.
    pub connections: usize,
    /// For a request's head, for each part of its body, and for a write that
    /// the client does not take.
    pub timeout: Duration,
    /// A request's line and headers, in bytes (431 beyond). hyper's buffer
    /// for a head grows to this size and at most one read beyond it, which
    /// bounds its memory.
    pub head: usize,
    /// A request's body, in bytes (413 beyond).
    pub body: usize,
}

pub const LIMITS: Limits = Limits { connections: 64, timeout: Duration::from_secs(60), head: 64 << 10, body: 32 << 20 };

/// Serves app on listener until `stopped` completes; then it accepts no more
/// and waits for the open connections to end.
pub async fn serve(listener: TcpListener, app: Router, stopped: impl Future<Output = ()>, limits: Limits) {
    let slots = Arc::new(Semaphore::new(limits.connections));
    let graceful = GracefulShutdown::new();
    let mut builder = http1::Builder::new();
    builder.timer(TokioTimer::new()).header_read_timeout(limits.timeout).max_buf_size(limits.head);
    let app = app.layer(middleware::from_fn(move |r: Request, next: Next| async move {
        if head_size(&r) > limits.head {
            return large_head(&r, limits.head);
        }
        next.run(r).await
    }));
    tokio::pin!(stopped);
    loop {
        let (stream, peer) = tokio::select! {
            accepted = listener.accept() => match accepted {
                Ok(connection) => connection,
                Err(e) => {
                    engine::log(&format!("accept: {e}"));
                    tokio::time::sleep(Duration::from_millis(100)).await;
                    continue;
                }
            },
            () = &mut stopped => break,
        };
        let _ = stream.set_nodelay(true); // a token's chunk goes out at once
        let Ok(slot) = slots.clone().try_acquire_owned() else {
            engine::log(&format!("{}: 503: too many connections", peer.ip()));
            tokio::spawn(refuse(stream, limits));
            continue;
        };
        let service = TowerToHyperService::new(app.clone().layer(Extension(ConnectInfo(peer))));
        let connection = graceful.watch(builder.serve_connection(TokioIo::new(Stalls::new(stream, limits.timeout)), service));
        tokio::spawn(async move {
            let _ = connection.await;
            drop(slot);
        });
    }
    graceful.shutdown().await;
}

/// A request's line and headers in bytes, as hyper parsed them: its buffer
/// can take a head somewhat longer than its limit when the head arrives in
/// a few large reads.
fn head_size(r: &Request) -> usize {
    let target = r.uri().path_and_query().map_or(1, |p| p.as_str().len());
    let headers: usize = r.headers().iter().map(|(name, value)| name.as_str().len() + 2 + value.len() + 2).sum();
    r.method().as_str().len() + 1 + target + 1 + "HTTP/1.1\r\n".len() + headers + 2
}

fn large_head(r: &Request, limit: usize) -> Response {
    let peer = r.extensions().get::<ConnectInfo<std::net::SocketAddr>>().map(|c| c.0.ip().to_string());
    engine::log(&format!("{}: 431: a request's line and headers above {} KB", peer.unwrap_or_default(), limit >> 10));
    let message = format!("the request's line and headers are larger than {} KB", limit >> 10);
    format::error(StatusCode::REQUEST_HEADER_FIELDS_TOO_LARGE, "invalid_request_error", &message)
}

/// The answer to a connection beyond the limit, before its request is read.
/// What the client sent is then read and dropped until it closes (at most a
/// head's worth, for at most 2 s): closing with it unread would reset the
/// connection, which can discard the answer before the client reads it.
async fn refuse(mut stream: TcpStream, limits: Limits) {
    let body = format::error_json("server_error", "too many connections");
    let head = format!(
        "HTTP/1.1 503 Service Unavailable\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    let answered = tokio::time::timeout(limits.timeout, async {
        stream.write_all(head.as_bytes()).await?;
        stream.write_all(body.as_bytes()).await?;
        stream.shutdown().await
    })
    .await;
    if !matches!(answered, Ok(Ok(()))) {
        return;
    }
    let _ = tokio::time::timeout(Duration::from_secs(2), async {
        let (mut sink, mut left) = ([0u8; 4096], limits.head);
        while left > 0 {
            match stream.read(&mut sink).await {
                Ok(0) | Err(_) => break,
                Ok(n) => left = left.saturating_sub(n),
            }
        }
    })
    .await;
}

/// A request's body: at most limits.body bytes, each part arriving within
/// limits.timeout of the one before.
pub async fn read_body(mut body: Body, limits: Limits) -> Result<Bytes, format::Error> {
    let large = || format::Error {
        status: StatusCode::PAYLOAD_TOO_LARGE,
        message: format!("the request's body is larger than {} MB", limits.body >> 20),
    };
    if body.size_hint().lower() > limits.body as u64 {
        return Err(large()); // its Content-Length says so already
    }
    let mut out = Vec::new();
    loop {
        match tokio::time::timeout(limits.timeout, body.frame()).await {
            Err(_) => {
                let message = format!("the request's body stopped arriving for {} s", limits.timeout.as_secs());
                return Err(format::Error { status: StatusCode::REQUEST_TIMEOUT, message });
            }
            Ok(None) => return Ok(out.into()),
            Ok(Some(Err(_))) => {
                return Err(format::Error { status: StatusCode::BAD_REQUEST, message: "the request's body ended early".into() });
            }
            Ok(Some(Ok(frame))) => {
                if let Ok(data) = frame.into_data() {
                    if out.len() + data.len() > limits.body {
                        return Err(large());
                    }
                    out.extend_from_slice(&data);
                }
            }
        }
    }
}

/// A connection whose writes a client may stall at most `limit`: a write that
/// cannot go on for that long fails, and hyper closes the connection.
struct Stalls {
    stream: TcpStream,
    limit: Duration,
    stalled: Option<Pin<Box<Sleep>>>,
}

impl Stalls {
    fn new(stream: TcpStream, limit: Duration) -> Stalls {
        Stalls { stream, limit, stalled: None }
    }

    /// A write's outcome: one that waits starts the deadline, and one that
    /// goes on clears it.
    fn deadline<T>(&mut self, cx: &mut Context<'_>, outcome: Poll<io::Result<T>>) -> Poll<io::Result<T>> {
        if outcome.is_ready() {
            self.stalled = None;
            return outcome;
        }
        let limit = self.limit;
        let stalled = self.stalled.get_or_insert_with(|| Box::pin(tokio::time::sleep(limit)));
        match stalled.as_mut().poll(cx) {
            Poll::Ready(()) => Poll::Ready(Err(io::Error::new(io::ErrorKind::TimedOut, "the client stopped reading"))),
            Poll::Pending => Poll::Pending,
        }
    }
}

impl AsyncRead for Stalls {
    fn poll_read(self: Pin<&mut Self>, cx: &mut Context<'_>, buf: &mut ReadBuf<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.get_mut().stream).poll_read(cx, buf)
    }
}

impl AsyncWrite for Stalls {
    fn poll_write(self: Pin<&mut Self>, cx: &mut Context<'_>, buf: &[u8]) -> Poll<io::Result<usize>> {
        let s = self.get_mut();
        let outcome = Pin::new(&mut s.stream).poll_write(cx, buf);
        s.deadline(cx, outcome)
    }

    fn poll_write_vectored(self: Pin<&mut Self>, cx: &mut Context<'_>, bufs: &[io::IoSlice<'_>]) -> Poll<io::Result<usize>> {
        let s = self.get_mut();
        let outcome = Pin::new(&mut s.stream).poll_write_vectored(cx, bufs);
        s.deadline(cx, outcome)
    }

    fn is_write_vectored(&self) -> bool {
        self.stream.is_write_vectored()
    }

    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        let s = self.get_mut();
        let outcome = Pin::new(&mut s.stream).poll_flush(cx);
        s.deadline(cx, outcome)
    }

    fn poll_shutdown(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        let s = self.get_mut();
        let outcome = Pin::new(&mut s.stream).poll_shutdown(cx);
        s.deadline(cx, outcome)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::routing::{get, post};
    use std::net::SocketAddr;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Instant;
    use tokio::sync::oneshot;

    const QUICK: Limits = Limits { connections: 2, timeout: Duration::from_millis(300), head: 16 << 10, body: 1000 };

    /// Sets its flag when dropped: the handler or the response it belongs to ended.
    struct Ended(Arc<AtomicBool>);

    impl Drop for Ended {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }

    async fn start(app: Router, limits: Limits) -> (SocketAddr, oneshot::Sender<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let (stop, stopped) = oneshot::channel::<()>();
        tokio::spawn(serve(listener, app, async move { let _ = stopped.await; }, limits));
        (addr, stop)
    }

    fn app(flag: Arc<AtomicBool>) -> Router {
        let waits = flag.clone();
        Router::new()
            .route("/health", get(|| async { "ok" }))
            .route(
                "/body",
                post(|body: Body| async move {
                    match read_body(body, QUICK).await {
                        Ok(b) => (StatusCode::OK, format!("{} bytes", b.len())),
                        Err(e) => (e.status, e.message),
                    }
                }),
            )
            .route(
                "/wait",
                post(move || {
                    let ended = Ended(waits.clone());
                    async move {
                        tokio::time::sleep(Duration::from_secs(30)).await;
                        drop(ended);
                        "late"
                    }
                }),
            )
            .route(
                "/stream",
                get(move || {
                    let ended = Ended(flag.clone());
                    async move {
                        let chunks = async_stream::stream! {
                            let _ended = ended;
                            loop {
                                yield Ok::<_, io::Error>(Bytes::from(vec![b'x'; 64 << 10]));
                            }
                        };
                        Body::from_stream(chunks)
                    }
                }),
            )
    }

    async fn answer(stream: &mut TcpStream) -> String {
        let mut out = Vec::new();
        let _ = tokio::time::timeout(Duration::from_secs(5), stream.read_to_end(&mut out)).await;
        String::from_utf8_lossy(&out).into_owned()
    }

    async fn ended_within(flag: &AtomicBool, limit: Duration) -> Option<Duration> {
        let t = Instant::now();
        while t.elapsed() < limit {
            if flag.load(Ordering::SeqCst) {
                return Some(t.elapsed());
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        None
    }

    #[tokio::test]
    async fn connections_beyond_the_limit_get_503() {
        let (addr, _stop) = start(app(Arc::default()), Limits { timeout: Duration::from_secs(5), ..QUICK }).await;
        let first = TcpStream::connect(addr).await.unwrap();
        let _second = TcpStream::connect(addr).await.unwrap();
        tokio::time::sleep(Duration::from_millis(100)).await;
        let mut third = TcpStream::connect(addr).await.unwrap();
        // A client sends its request before it reads: closing a connection
        // with that request unread would reset it before the answer is read.
        third.write_all(b"GET /health HTTP/1.1\r\nHost: x\r\n\r\n").await.unwrap();
        tokio::time::sleep(Duration::from_millis(200)).await; // the answer arrives before the client reads it
        let mut refused = Vec::new();
        let read = tokio::time::timeout(Duration::from_secs(5), third.read_to_end(&mut refused)).await;
        assert!(matches!(read, Ok(Ok(_))), "the connection was reset or kept open: {read:?}");
        let refused = String::from_utf8_lossy(&refused);
        assert!(refused.starts_with("HTTP/1.1 503"), "{refused}");
        assert!(refused.contains("too many connections"), "{refused}");
        drop(first); // its slot comes back once the server sees it closed
        tokio::time::sleep(Duration::from_millis(200)).await;
        let mut fourth = TcpStream::connect(addr).await.unwrap();
        fourth.write_all(b"GET /health HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n").await.unwrap();
        let served = answer(&mut fourth).await;
        assert!(served.starts_with("HTTP/1.1 200"), "{served}");
    }

    #[tokio::test]
    async fn a_head_above_the_limit_gets_431() {
        let (addr, _stop) = start(app(Arc::default()), QUICK).await;
        for (pad, status) in [(15 << 10, "HTTP/1.1 200"), (17 << 10, "HTTP/1.1 431"), (40 << 10, "HTTP/1.1 431")] {
            let mut c = TcpStream::connect(addr).await.unwrap();
            let head = format!("GET /health HTTP/1.1\r\nHost: x\r\nConnection: close\r\nX-Pad: {}\r\n\r\n", "a".repeat(pad));
            let _ = c.write_all(head.as_bytes()).await;
            let got = answer(&mut c).await;
            assert!(got.starts_with(status), "{pad} bytes of padding: {got}");
        }
    }

    #[tokio::test]
    async fn a_late_head_closes_the_connection() {
        let (addr, _stop) = start(app(Arc::default()), QUICK).await;
        let mut c = TcpStream::connect(addr).await.unwrap();
        c.write_all(b"GET /health HTTP/1.1\r\nHost: x\r\n").await.unwrap();
        let t = Instant::now();
        let left = answer(&mut c).await;
        assert!(left.is_empty(), "{left}");
        assert!(t.elapsed() < Duration::from_secs(2), "closed after {:?}", t.elapsed());
    }

    #[tokio::test]
    async fn a_stalled_body_gets_408_and_a_large_one_413() {
        let (addr, _stop) = start(app(Arc::default()), QUICK).await;
        let mut c = TcpStream::connect(addr).await.unwrap();
        c.write_all(b"POST /body HTTP/1.1\r\nHost: x\r\nContent-Length: 10\r\n\r\nabc").await.unwrap();
        let late = answer(&mut c).await;
        assert!(late.starts_with("HTTP/1.1 408"), "{late}");
        let mut c = TcpStream::connect(addr).await.unwrap();
        c.write_all(b"POST /body HTTP/1.1\r\nHost: x\r\nContent-Length: 5000\r\nConnection: close\r\n\r\n").await.unwrap();
        let large = answer(&mut c).await;
        assert!(large.starts_with("HTTP/1.1 413"), "{large}");
        let mut c = TcpStream::connect(addr).await.unwrap();
        c.write_all(b"POST /body HTTP/1.1\r\nHost: x\r\nContent-Length: 3\r\nConnection: close\r\n\r\nabc").await.unwrap();
        let read = answer(&mut c).await;
        assert!(read.starts_with("HTTP/1.1 200") && read.ends_with("3 bytes"), "{read}");
    }

    #[tokio::test]
    async fn a_client_that_leaves_ends_its_request() {
        let flag = Arc::new(AtomicBool::new(false));
        let (addr, _stop) = start(app(flag.clone()), Limits { timeout: Duration::from_secs(30), ..QUICK }).await;
        let mut c = TcpStream::connect(addr).await.unwrap();
        c.write_all(b"POST /wait HTTP/1.1\r\nHost: x\r\nContent-Length: 0\r\n\r\n").await.unwrap();
        tokio::time::sleep(Duration::from_millis(200)).await;
        assert!(!flag.load(Ordering::SeqCst), "the request waits while its client stays");
        drop(c);
        let took = ended_within(&flag, Duration::from_secs(3)).await;
        assert!(took.is_some(), "the handler ran on after its client left");
    }

    #[tokio::test]
    async fn a_client_that_stops_reading_is_cut_off() {
        let flag = Arc::new(AtomicBool::new(false));
        let (addr, _stop) = start(app(flag.clone()), QUICK).await;
        let mut c = TcpStream::connect(addr).await.unwrap();
        c.write_all(b"GET /stream HTTP/1.1\r\nHost: x\r\n\r\n").await.unwrap();
        // The client reads nothing: once the socket's buffers are full, the
        // server's writes wait, and after the limit the response ends.
        let took = ended_within(&flag, Duration::from_secs(10)).await;
        assert!(took.is_some(), "the response kept waiting for a client that does not read");
        drop(c);
    }
}
