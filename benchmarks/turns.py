"""Decoding after a long prompt in the same conversation, for server builds.

Each run starts one server and sends two turns of one conversation. The first
turn asks about a source excerpt; the second adds a long document, as a
tool's output would, and asks about it. For each turn the script reads
the model's status (GET /v1/models/{id}) when the first token streams (the prompt's prefill is done) and
when the reply ends, so the decode figures exclude the prefill: tokens per
second, expert-cache misses and read waits per decoded row. Runs alternate
between the builds given, with temperature and top-p 1 and matched seeds.
Prompts and replies stay in memory; --results writes counts, timings and
reply digests only.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import secrets
import signal
import socket
import subprocess
import sys
import time
import urllib.request

from runtime import ROOT, wait_for_gpu


def text_of(paths, chars):
    """The committed text of repository files: edits in the working tree do
    not change the prompts between runs."""
    text = "".join(subprocess.check_output(["git", "show", f"HEAD:{p}"], cwd=ROOT, text=True) for p in paths)
    return text[:chars]


def conversation(args):
    code = text_of(["src/engine/io.c"], args.first_chars)
    document = text_of(["docs/design.md", "docs/usage.md"], args.document_chars)
    first = [
        {"role": "system", "content": "You are a careful reviewer of C code for an inference engine."},
        {"role": "user", "content": "Here is part of the expert cache's source:\n\n" + code +
         "\n\nExplain how a slot is chosen for a new expert, and name one risk in that code."},
    ]
    second = ("The tool `cat docs/design.md docs/usage.md` printed:\n\n" + document +
              "\n\nUsing that output, explain how the server decides which earlier prompt it can reuse.")
    return first, second


def call(port, key, path, body=None, timeout=10):
    request = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=body,
                                     headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    return urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=timeout)


def status(port, key):
    """The served model's status: GET /v1/models/{id}'s "status"."""
    with call(port, key, "/v1/models") as response:
        models = json.loads(response.read(1 << 20))["data"]
        if len(models) != 1:
            raise RuntimeError("this comparison requires a server with one configured model")
        model = models[0]["id"]
    with call(port, key, f"/v1/models/{model}") as response:
        return json.loads(response.read(1 << 20))["status"]


def turn(port, key, messages, tokens, timeout, seed):
    """One streamed reply; the status at its first token and at its end."""
    with call(port, key, "/v1/models") as response:
        models = json.loads(response.read(1 << 20))["data"]
    if len(models) != 1:
        raise RuntimeError("this comparison requires one configured model")
    body = json.dumps({"model": models[0]["id"], "messages": messages, "max_tokens": tokens,
                       "temperature": 1, "top_p": 1, "seed": seed,
                       "stream": True, "stream_options": {"include_usage": True}}).encode()
    before = status(port, key)
    began, first, at_first, parts, usage = time.monotonic(), None, None, [], None
    with call(port, key, "/v1/chat/completions", body, timeout) as response:
        for raw in response:
            if not raw.startswith(b"data:"):
                continue
            raw = raw[5:].strip()
            if raw == b"[DONE]":
                break
            chunk = json.loads(raw)
            usage = chunk.get("usage") or usage
            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {})
                piece = (delta.get("reasoning_content") or "") + (delta.get("content") or "")
                if piece and first is None:
                    first, at_first = time.monotonic(), status(port, key)
                parts.append(piece)
    end = status(port, key)
    if at_first is None or usage is None:
        raise RuntimeError("the reply streamed no tokens")
    d0, d1 = at_first["model_decode"], end["model_decode"]
    c, c0, c1 = before["expert_cache"], at_first["expert_cache"], end["expert_cache"]
    rows = d1["computed_rows"] - d0["computed_rows"]
    seconds = d1["seconds"] - d0["seconds"]
    reply = "".join(parts)
    return reply, {
        "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
        "first_token_seconds": round(first - began, 3),
        "prefill_tokens": at_first["last_prefill"]["tokens"], "prefill_misses": c0["misses"] - c["misses"],
        "prefill_hits": c0["hits"] - c["hits"], "prefill_read_gb": round(c0["read_gb"] - c["read_gb"], 1),
        "decoded_rows": rows, "decode_seconds": round(seconds, 3),
        "decode_rows_per_second": round(rows / seconds, 3) if seconds > 0 else None,
        "misses_per_row": round((c1["misses"] - c0["misses"]) / rows, 3) if rows else None,
        "read_wait_ms_per_row": round(1000 * (d1["read_wait_seconds"] - d0["read_wait_seconds"]) / rows, 3) if rows else None,
        "requested_gb_per_row": round((d1["requested_expert_gb"] - d0["requested_expert_gb"]) / rows, 4) if rows else None,
        "reply_sha256": hashlib.sha256(reply.encode()).hexdigest(),
    }


def run(args, binary):
    guard = wait_for_gpu(args.cache, 60, args.reserve, desktop=True, quiet_cpu=True)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    key = secrets.token_hex(24)
    env = {k: v for k, v in os.environ.items() if not k.startswith("MYDS4_")}
    env.update(MYDS4_API_KEY=key)
    command = [str(binary), "-m", str(args.checkpoint), "--serve", str(port), "--ctx", str(args.context),
               "--cache", str(args.cache), "--reserve", str(args.reserve), "--kv-cache", "4", "--ced", args.ced]
    if args.experts:
        command += ["--experts", str(args.experts)]
    if args.draft:
        command += ["--draft", str(args.draft)]
    if args.lookup:
        command += ["--lookup", str(args.lookup)]
    log = None if args.server_log else subprocess.DEVNULL
    server = subprocess.Popen(command, env=env, stdout=subprocess.DEVNULL, stderr=log)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(1))  # stopping the script stops its server
    try:
        deadline = time.monotonic() + 180
        while True:
            if server.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError("the server did not become ready")
            try:
                status(port, key)
                break
            except OSError:
                time.sleep(.5)
        first, second = conversation(args)
        reply, one = turn(port, key, first, args.tokens, args.timeout, args.seed)
        messages = first + [{"role": "assistant", "content": reply}, {"role": "user", "content": second}]
        _, two = turn(port, key, messages, args.tokens, args.timeout, args.seed + 1)
        final = status(port, key)
        cache = final["expert_cache"]
        valid = cache["slots_released"] == 0 and cache["gib"] >= args.cache - .1
        return {"binary": binary.name, "date_utc": datetime.now(timezone.utc).isoformat(), "valid": valid,
                "first_turn": one, "second_turn": two, "cache_gib": cache["gib"],
                "available_gib_before": guard["samples"][-1]["memory"].get("available_gib")}
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--experts", type=Path, help="a GGUF expert file (any shard); the checkpoint's MXFP4 otherwise")
    ap.add_argument("--binaries", type=Path, nargs="+", required=True, help="server executables, run in --order")
    ap.add_argument("--order", type=int, nargs="+", help="indices into --binaries (default: 0 1 1 0)")
    ap.add_argument("--cache", type=float, default=40)
    ap.add_argument("--reserve", type=float, default=10)
    ap.add_argument("--context", type=int, default=32768)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--tokens", type=int, default=256, help="tokens generated in each turn")
    ap.add_argument("--first-chars", type=int, default=24000, help="source characters in the first turn")
    ap.add_argument("--document-chars", type=int, default=56000, help="document characters in the second turn")
    ap.add_argument("--ced", default="off")
    ap.add_argument("--draft", type=int, default=0)
    ap.add_argument("--lookup", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--server-log", action="store_true", help="pass the server's log through to this script's stderr")
    ap.add_argument("--results", type=Path)
    args = ap.parse_args()
    order = args.order or [0, 1, 1, 0][:2 * len(args.binaries)]
    if not all(0 <= i < len(args.binaries) for i in order):
        ap.error("--order names a binary that was not given")
    binaries = [b.resolve() for b in args.binaries]
    report = {"settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != "results"},
              "runs": []}
    replies = {}
    for i in order:
        record = run(args, binaries[i])
        digests = (record["first_turn"]["reply_sha256"], record["second_turn"]["reply_sha256"])
        record["replies_match"] = replies.setdefault("reference", digests) == digests
        report["runs"].append(record)
        print("TURN_RESULT " + json.dumps(record), flush=True)
        if args.results:
            args.results.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
