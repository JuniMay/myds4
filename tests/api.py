"""Bounded localhost smoke check for online learning across server requests.

Starts one owned model process. Request results and drained diagnostics stay in
RAM. This checks integration, batching and learner lifetime, not model quality.
"""

import argparse
import base64
import json
import math
import os
import re
import secrets
import socket
import struct
import subprocess
import threading
import time
import urllib.error
import urllib.request
import zlib
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def model_status(call):
    """The single model started by this check."""
    models = call("/v1/models")["data"]
    if len(models) != 1:
        raise RuntimeError("the predictor API check requires exactly one configured model")
    model = models[0]["id"]
    return call(f"/v1/models/{model}")["status"]


class Capture:
    def __init__(self, process):
        self.process = process
        self.lock = threading.Lock()
        self.tail = deque(maxlen=24)
        self.windows = deque(maxlen=1024)
        self.summary = {}
        self.sampling = []
        self.bytes = 0
        self.overflow = False
        self.batched = False
        self.reused = False
        self.threads = []
        for stream in (process.stdout, process.stderr):
            thread = threading.Thread(target=self.drain, args=(stream,), daemon=True)
            thread.start()
            self.threads.append(thread)

    def drain(self, stream):
        for line in iter(lambda: stream.readline(65537), ""):
            with self.lock:
                self.bytes += len(line.encode("utf-8"))
                self.overflow |= self.bytes > 8 * 1024 * 1024
                self.tail.append(line[-1200:].rstrip())
                self.batched |= bool(re.search(r"batch: [2-6] requests decoding together", line))
                self.reused |= bool(re.search(r", [1-9][0-9]* reused;", line))
                match = re.search(
                    r"sampling temperature ([\d.e+-]+), top_p ([\d.e+-]+); (\d+) draft passes", line
                )
                if match:
                    self.sampling.append((float(match[1]), float(match[2]), int(match[3])))
                if line.startswith('{"online_predictor_window":'):
                    try:
                        self.windows.append(json.loads(line)["online_predictor_window"])
                    except (ValueError, KeyError):
                        pass
                if line.startswith('{"online_predictor":'):
                    try:
                        self.summary = json.loads(line)["online_predictor"]
                    except (ValueError, KeyError):
                        pass
            if self.overflow:
                stop(self.process)
                break
        stream.close()

    def join(self):
        for thread in self.threads:
            thread.join(timeout=5)


def stop(process):
    if process.poll() is None:
        try:
            process.terminate()
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                process.kill()
            except ProcessLookupError:
                return
            process.wait(timeout=5)


def image_data_url():
    """An original 8-by-8 RGB image, encoded entirely in memory."""

    def chunk(kind, data):
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    scanlines = (b"\0" + bytes((48, 112, 208)) * 8) * 8
    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", 8, 8, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(scanlines)) + chunk(b"IEND", b"")
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--experts", type=Path)
    parser.add_argument(
        "--binary",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "build/myds4-server",
    )
    parser.add_argument("--cache", type=float, default=28)
    parser.add_argument("--context", type=int, default=1024)
    parser.add_argument("--requests", type=int, choices=(2, 3), default=3)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--draft", type=int, choices=range(6), default=2)
    parser.add_argument("--draft-cache", type=int, default=32)
    parser.add_argument("--substitute", type=float, default=0)
    parser.add_argument("--image-smoke", action="store_true")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top-p", type=float, default=1)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--sampling-check", action="store_true")
    parser.add_argument(
        "--predictor-state", type=Path, help="load or create state and verify its clean-stop save"
    )
    args = parser.parse_args()
    if not (math.isfinite(args.cache) and 1 <= args.cache <= 64):
        parser.error("--cache must be 1 to 64 GiB")
    if not (
        math.isfinite(args.temperature)
        and args.temperature >= 0
        and math.isfinite(args.top_p)
        and 0 <= args.top_p <= 1
        and 0 <= args.seed < 2**64
    ):
        parser.error("invalid sampling settings")
    if not (
        256 <= args.context <= 4096
        and 16 <= args.max_new_tokens <= 96
        and 30 <= args.timeout <= 600
    ):
        parser.error("bounds: context 256–4096, output 16–96 tokens, timeout 30–600 seconds")
    if not (
        15 <= args.draft_cache <= 128
        and math.isfinite(args.substitute)
        and 0 <= args.substitute <= 2
    ):
        parser.error("bounds: draft cache 15–128, substitute 0–2")
    return args


def main():
    args = arguments()
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    key = secrets.token_hex(24)
    env = dict(os.environ, MYDS4_API_KEY=key, MYDS4_ONLINE_WINDOW_PASSES="1")
    for name in ("MYDS4_PROFILE", "MYDS4_ONLINE_FREEZE_ROWS"):
        env.pop(name, None)
    cmd = [
        str(args.binary.resolve()),
        "-m",
        str(args.checkpoint.resolve()),
        "--serve",
        str(port),
        "--host",
        "127.0.0.1",
        "--ctx",
        str(args.context),
        "--cache",
        str(args.cache),
        "--kv-cache",
        "2",
        "--batch",
        str(args.requests),
        "--online-predictor",
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--seed",
        str(args.seed),
        "--think",
        "0",
    ]
    if args.experts:
        cmd += ["--experts", str(args.experts.resolve())]
    if args.draft:
        cmd += ["--draft", str(args.draft), "--draft-cache", str(args.draft_cache)]
    if args.substitute:
        cmd += ["--substitute", str(args.substitute)]
    state_existed = args.predictor_state is not None and args.predictor_state.exists()
    if args.predictor_state is not None:
        cmd += ["--predictor-state", str(args.predictor_state)]
    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    captured = Capture(process)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + args.timeout
    timer = threading.Timer(args.timeout, stop, args=(process,))
    timer.daemon = True
    timer.start()
    results, error, status = [], None, {}
    sampling_checks = {}
    image_result, image_window_start = None, 0

    def call(path, body=None):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("server check exceeded its time limit")
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}{path}",
            data=data,
            headers={
                "Authorization": "Bearer " + key,
                "Content-Type": "application/json",
            },
        )
        with opener.open(request, timeout=min(remaining, 120)) as response:
            payload = response.read(1024 * 1024 + 1)
            if response.status != 200 or len(payload) > 1024 * 1024:
                raise RuntimeError("unexpected HTTP status or response larger than 1 MiB")
            if body and body.get("stream"):
                text, reasoning, done = "", "", False
                for line in payload.splitlines():
                    if not line.startswith(b"data: "):
                        continue
                    if line[6:] == b"[DONE]":
                        done = True
                        continue
                    event = json.loads(line[6:])
                    for choice in event.get("choices", []):
                        delta = choice.get("delta", {})
                        text += delta.get("content") or ""
                        reasoning += delta.get("reasoning_content") or ""
                return {"stream_done": done, "content": text, "reasoning_content": reasoning}
            return json.loads(payload)

    def request(pair):
        a, b, answer = pair
        body = {
            "model": "deepseek-v4.1-flash",
            "max_tokens": args.max_new_tokens,
            "messages": [
                {"role": "system", "content": "Answer arithmetic questions clearly."},
                {
                    "role": "user",
                    "content": f"What is {a} times {b}? Start with the number, then explain the multiplication briefly.",
                },
            ],
        }
        reply = call("/v1/chat/completions", body)
        content = reply["choices"][0]["message"]["content"]
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("server returned empty or non-text content")
        return {
            "answer_present": str(answer) in content,
            "content": content,
            "completion_tokens": reply["usage"]["completion_tokens"],
        }

    try:
        while True:
            if process.poll() is not None:
                raise RuntimeError(f"server exited before readiness ({process.returncode})")
            try:
                call("/health")
                break
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                if time.monotonic() >= deadline:
                    raise TimeoutError("server did not become ready")
                time.sleep(0.2)
        pairs = [(17, 23, 391), (12, 14, 168), (19, 11, 209)][: args.requests]
        with ThreadPoolExecutor(max_workers=args.requests) as pool:
            results = list(pool.map(request, pairs))
        results.append(
            request(pairs[0])
        )  # Reuse the exact earlier prompt after its response ended.
        if args.sampling_check:
            body = {
                "model": "deepseek-v4.1-flash",
                "max_tokens": args.max_new_tokens,
                "messages": [
                    {"role": "user", "content": "Briefly explain how to merge sorted lists."}
                ],
            }
            for field, value in [
                ("temperature", -1),
                ("top_p", 1.1),
                ("temperature", "bad"),
                ("seed", -1),
            ]:
                try:
                    call("/v1/chat/completions", dict(body, **{field: value}))
                    sampling_checks["reject_" + field + "_" + str(value)] = False
                except urllib.error.HTTPError as exc:
                    sampling_checks["reject_" + field + "_" + str(value)] = exc.code == 400
            explicit = call(
                "/v1/chat/completions",
                dict(body, temperature=0.7, top_p=0.8, seed=76543, thinking=False),
            )
            repeated = call(
                "/v1/chat/completions",
                dict(body, temperature=0.7, top_p=0.8, seed=76543, thinking=False),
            )
            sampling_checks["request_override_completed"] = bool(
                explicit["choices"][0]["message"].get("content")
            )
            if not args.substitute:
                sampling_checks["seeded_override_repeats"] = (
                    explicit["choices"][0]["message"] == repeated["choices"][0]["message"]
                )
            streamed = call(
                "/v1/chat/completions",
                dict(body, temperature=0.7, top_p=0.8, seed=76543, thinking=False, stream=True),
            )
            sampling_checks["sampled_stream_completed"] = streamed["stream_done"] and bool(
                streamed["content"]
            )
            if not args.substitute:
                sampling_checks["sampled_stream_matches_nonstream"] = (
                    streamed["content"] == explicit["choices"][0]["message"]["content"]
                )
            reasoning = call("/v1/chat/completions", dict(body, thinking=True, reasoning_effort=16))
            sampling_checks["thinking_override_observed"] = bool(
                reasoning["choices"][0]["message"].get("reasoning_content")
            )
        if args.image_smoke:
            with captured.lock:
                image_window_start = max((w["window"] for w in captured.windows), default=0)
            reply = call(
                "/v1/chat/completions",
                {
                    "model": "deepseek-v4.1-flash",
                    "max_tokens": args.max_new_tokens,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {"url": image_data_url()},
                                },
                                {
                                    "type": "text",
                                    "text": "Describe this image briefly.",
                                },
                            ],
                        }
                    ],
                },
            )
            content = reply["choices"][0]["message"]["content"]
            if not isinstance(content, str) or not content.strip():
                raise RuntimeError("image request returned empty or non-text content")
            image_result = {
                "content": content,
                "completion_tokens": reply["usage"]["completion_tokens"],
                "scope": "image input and text output smoke; visual accuracy is not graded",
            }
        while True:
            status = model_status(call)
            if status.get("totals", {}).get("requests", 0) >= args.requests + 1 + args.image_smoke:
                break
            time.sleep(0.05)
    except (OSError, RuntimeError, ValueError, KeyError, IndexError, TypeError) as exc:
        error = f"{type(exc).__name__}: {exc}".replace(key, "<redacted>")
    finally:
        timer.cancel()
        stop(process)
        captured.join()
    windows = list(captured.windows)
    checks = {
        "requests_completed": len(results) == args.requests + 1,
        "arithmetic_answers_present": bool(results) and all(r["answer_present"] for r in results),
        "concurrent_batch_observed": captured.batched,
        "prompt_reuse_observed": captured.reused,
        "online_updates_observed": any(w.get("updates", 0) > 0 for w in windows),
        "one_learning_epoch": bool(windows) and all(w.get("learning_epoch") == 1 for w in windows),
        "no_invalid_ids": bool(windows) and all(w.get("invalid_native_ids") == 0 for w in windows),
        "no_epoch_skips": bool(windows) and all(w.get("epoch_skips") == 0 for w in windows),
        "bounded_diagnostics": not captured.overflow,
        "server_stopped": process.poll() is not None,
        "server_sampling_defaults_observed": len(captured.sampling) >= args.requests + 1
        and all(
            abs(t - args.temperature) < 1e-6 and abs(p - args.top_p) < 1e-6
            for t, p, _ in captured.sampling[: args.requests + 1]
        ),
    }
    if args.draft:
        checks["draft_verification_observed"] = any(n > 0 for _, _, n in captured.sampling)
    if args.sampling_check:
        sampling_checks["request_sampling_override_observed"] = any(
            abs(t - 0.7) < 1e-6 and abs(p - 0.8) < 1e-6 for t, p, _ in captured.sampling
        )
        checks.update(sampling_checks)
    if args.image_smoke:
        checks["image_request_completed"] = image_result is not None
        checks["updates_continue_after_image_request"] = any(
            w["window"] > image_window_start and w.get("updates", 0) > 0 for w in windows
        )
    if args.predictor_state is not None:
        checks["state_loaded_if_present"] = captured.summary.get("state_loads") == int(
            state_existed
        )
        checks["state_saved_on_clean_stop"] = (
            process.returncode == 0
            and captured.summary.get("state_saves") == 1
            and captured.summary.get("state_dirty") is False
            and args.predictor_state.is_file()
            and args.predictor_state.stat().st_size == captured.summary.get("state_file_bytes")
        )
    passed = error is None and all(checks.values())
    report = {
        "passed": passed,
        "checks": checks,
        "error": error,
        "responses": results,
        "image_response": image_result,
        "learning_windows": len(windows),
        "updates": sum(w.get("updates", 0) for w in windows),
        "saved_state": captured.summary if args.predictor_state is not None else None,
        "sampling": captured.sampling,
        "scope": "integration smoke only; approximate substitution can change outputs",
    }
    if not passed:
        report["diagnostic_tail"] = list(captured.tail)
    print(json.dumps(report, ensure_ascii=False).replace(key, "<redacted>"))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
