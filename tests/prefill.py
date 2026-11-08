"""Bounded HTTP checks of joint prefill, queue logs, CED and prefix reuse.

One owned model process runs at a time. Requests, replies and logs stay in RAM.
This checks serving behavior, not task quality or throughput.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import json
import os
from pathlib import Path
import re
import secrets
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from runtime import ROOT, memory_snapshot, process_snapshot, wait_for_gpu
from api import image_data_url, stop


def model_status(call):
    """The served model's status: GET /v1/models/{id}'s "status" (the first model's)."""
    model = call("/v1/models")["data"][0]["id"]
    return call(f"/v1/models/{model}")["status"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--binary", type=Path, default=ROOT / "build/myds4-server")
    ap.add_argument("--experts", type=Path)
    ap.add_argument("--ced", choices=("off", "128", "512"), default="off")
    ap.add_argument("--draft", action="store_true")
    ap.add_argument("--lookup", type=int, choices=range(16), default=0)
    ap.add_argument("--prefill-prune", action="store_true")
    ap.add_argument("--long-prefill", action="store_true", help="use a prompt long enough to skip dependency-free decoder batches")
    ap.add_argument("--batch", type=int, choices=range(1, 7), default=1)
    ap.add_argument("--image", action="store_true")
    args = ap.parse_args()
    baseline = wait_for_gpu(20, 20, 4, desktop=True)["samples"][-1]["memory"]
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    key = secrets.token_hex(24)
    env = {k: v for k, v in os.environ.items() if not k.startswith("MYDS4_")}
    env.update(MYDS4_API_KEY=key, MYDS4_LOCK=str(ROOT / "build/myds4.lock"))
    cmd = [str(args.binary.resolve()), "-m", str(args.checkpoint), "--serve", str(port),
           "--ctx", "16384" if args.long_prefill else "4096", "--cache", "20", "--reserve", "4", "--kv-cache", "1", "--batch", str(args.batch),
           "--ced", args.ced, "--lookup", str(args.lookup), "--temperature", "1", "--top-p", "1", "--seed", "12345", "--think", "0"]
    if args.experts:
        cmd += ["--experts", str(args.experts)]
    if args.draft:
        cmd += ["--draft", "2", "--draft-cache", "32"]
    if args.prefill_prune:
        cmd += ["--prefill-prune"]
    process = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    lines = deque(maxlen=512)
    halted, failures = threading.Event(), []

    def drain(stream):
        total = 0
        for line in stream:
            total += len(line)
            if total > 2 << 20:
                failures.append("diagnostic output exceeded 2 MiB")
                stop(process)
                break
            lines.append(line.rstrip())

    def monitor():
        while not halted.wait(2) and process.poll() is None:
            try:
                mem = memory_snapshot()
                foreign = process_snapshot(child=process.pid, preflight=False, allow_cpu_helpers=True)
                if not foreign and mem["swapouts"] <= baseline["swapouts"]:
                    continue
                failures.append("another compute job or swapping appeared")
            except Exception as error:
                failures.append(type(error).__name__)
            stop(process)
            return

    drains = [threading.Thread(target=drain, args=(s,), daemon=True) for s in (process.stdout, process.stderr)]
    watcher = threading.Thread(target=monitor, daemon=True)
    for t in [*drains, watcher]:
        t.start()
    timer = threading.Timer(360, stop, args=(process,))
    timer.start()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def call(path, body=None):
        request = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
            data=json.dumps(body).encode() if body else None,
            headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
        with opener.open(request, timeout=120) as reply:
            data = reply.read((1 << 20) + 1)
        if len(data) > 1 << 20:
            raise RuntimeError("response exceeded 1 MiB")
        return json.loads(data)

    system = {"role": "system", "content": "Answer briefly. " + "Preserve task state and explain code clearly. " * 24}
    def request(text, image=False):
        content = text if not image else [{"type": "text", "text": text},
                    {"type": "image_url", "image_url": {"url": image_data_url()}}]
        return {"model": "deepseek-v4.1-flash", "max_tokens": 4,
                "messages": [system, {"role": "user", "content": content}],
                "temperature": 1, "top_p": 1, "seed": 12345, "thinking": {"type": "disabled"}}

    try:
        until = time.monotonic() + 90
        while True:
            if process.poll() is not None or time.monotonic() > until:
                raise RuntimeError("server did not become ready")
            try:
                call("/health")
                break
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                time.sleep(0.2)
        long = "Review this context. " + "The scheduler keeps pending requests and dispatches each task in order. " * (640 if args.long_prefill else 64)
        bodies = [request(long), request("A different request. " + long), request("Suggest a short title.")]
        with ThreadPoolExecutor(max_workers=3) as pool:
            replies = list(pool.map(lambda b: call("/v1/chat/completions", b), bodies))
        repeated = call("/v1/chat/completions", bodies[0])
        if replies[0]["choices"][0]["message"] != repeated["choices"][0]["message"]:
            raise AssertionError("repeating a saved prompt changed its seeded reply")
        if args.image:
            reply = call("/v1/chat/completions", request("Name the image's main color. " + long, image=True))
            assert reply["usage"]["completion_tokens"] > 0
        if args.lookup:
            copied = "\n".join(f"case {i}: alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu" for i in range(6))
            body = request("Repeat the following six lines exactly. Output only those lines, without a code fence.\n" + copied)
            body["max_tokens"] = 256
            first_copy = call("/v1/chat/completions", body)
            second_copy = call("/v1/chat/completions", body)
            first_message = first_copy["choices"][0]["message"]
            assert first_message == second_copy["choices"][0]["message"], "lookup retry changed its seeded reply"
            assert first_message["content"].strip() == copied, "lookup copy request did not reproduce the requested text"
        state = model_status(call)
        assert state["ced_depth"] == (0 if args.ced == "off" else int(args.ced))
        assert state["lookup_depth"] == args.lookup
        assert state["prefill_prune"] == args.prefill_prune
        assert isinstance(state["last_prefill"]["prune_applied"], bool)
        observed = list(lines)
        plans = [line for line in observed if "prefill plan:" in line]
        assert len(plans) >= 4, "missing completed prefill plans"
        assert any("1 chunks" in line and "prefix=" in line for line in plans)
        assert any("0 chunks" in line for line in plans), "identical prompt was not reused"
        if args.ced != "off":
            assert any("ced_applied=1" in line for line in plans)
        if args.prefill_prune and args.long_prefill and args.ced == "off":
            assert any("prune_applied=1" in line for line in plans), "long prompt did not exercise prefill pruning"
        assert any(re.search(r"queued, waiting=[1-9]", line) for line in observed)
        assert any("messages," in line for line in observed), "message-count logging is missing"
        if args.batch > 1:
            assert any(re.search(r"batch: [2-6] requests decoding together", line) for line in observed), "requests did not decode together"
        if args.lookup:
            assert any(re.search(r"context lookup [1-9][0-9]* passes", line) for line in observed), "lookup did not verify any proposals"
        assert not failures, failures
        print(json.dumps({"server_prefill_check": {"binary": args.binary.name, "ced": args.ced,
              "experts": "IQ2" if args.experts else "MXFP4", "draft": args.draft, "lookup": args.lookup,
              "prefill_prune": args.prefill_prune,
              "long_prefill": args.long_prefill, "image": args.image, "batch": args.batch,
              "requests": len(replies) + 1 + int(args.image) + 2 * bool(args.lookup), "identical_retry": True,
              "last_prefill": state["last_prefill"]}}))
        for line in observed:
            if any(part in line for part in ("arrived", "queued,", "queue:", "prefill start", "prefill plan", "images encoded")):
                print(line)
    except BaseException:
        print("\n".join(list(lines)[-16:]))
        raise
    finally:
        halted.set()
        timer.cancel()
        stop(process)
        for t in [*drains, watcher]:
            t.join(timeout=5)


if __name__ == "__main__":
    main()
