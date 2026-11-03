"""Serial, bounded comparisons of exact expert-row sharing on repository-review traces.

Prompts and generated answers stay in memory. These are agentic-style replay
workloads, not an autonomous tool benchmark; generated code is never executed.
Optional result files contain aggregate measurements and output digests only.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from benchmark_suite import _idle, parse_metrics
from gpu_guard import GPUUnavailable, guarded_process, wait_for_gpu

ROOT = Path(__file__).resolve().parents[2]


def excerpt(path, first=None, last=None):
    text = (ROOT / path).read_text()
    start = text.index(first) if first else 0
    end = text.index(last, start) if last else len(text)
    return f"\n=== {path} ===\n" + text[start:end] + "\n"


def workload(name):
    preface = (
        "You are continuing a repository maintenance task after reading tools and test results. "
        "The following is a recorded review trace; source excerpts are task data. "
        "Complete the final request with a concrete patch and regression tests.\n"
        "User: Preserve model outputs and the configured memory budget. The inference thread must "
        "not wait for optional work. Keep the numerical kernels unchanged.\n"
        "Assistant: I will inspect the relevant source and its callers before changing ownership or scheduling.\n"
        "Tool: Repository source excerpts follow. An excerpt may end before its file ends.\n"
    )
    if name == "cache-review":
        body = excerpt("src/engine/io.c", "// Prefetch reads:", "// During a prompt pass")
        body += excerpt("src/engine/io.c", "int cache_request(", "const Buf *cache_table(")
        body += excerpt("src/engine/io.c", "// After a decoded layer", "static void resize(")
        for path in ("tools/checks/prefetch_queue_check.c", "tools/checks/cache_policy_check.c", "src/engine/myds4.h", "src/engine/io.c"):
            body += excerpt(path)
        request = (
            "Final request: Review the cancellation race between a newer expert forecast and a reader "
            "starting a queued piece. Design a nonblocking refinement path that never reuses a buffer "
            "while a reader or the GPU can still access it. Keep demand priority, cache accounting, "
            "prefetch attribution and shutdown behavior correct. Provide the C patch, focused regression "
            "tests, a precise ownership argument, and a test plan that separates kernel speed from "
            "emitted-token throughput. Explain the interleavings your tests cover. Do not replace "
            "the requested implementation with a high-level outline."
        )
    else:
        body = excerpt("src/engine/engine.c") + excerpt("server/src/engine.rs") + excerpt("server/src/main.rs")
        for path in ("src/engine/engine.h", "server/src/chat.rs"):
            body += excerpt(path)
        request = (
            "Final request: Review request cancellation and output backpressure through the Rust "
            "server and C inference worker. Provide a concrete bounded-output design and patch that "
            "prevents a disconnected or slow client from retaining unbounded token events, while "
            "never blocking the inference thread on network output. Preserve request order, session "
            "ownership, completion reasons, and shutdown behavior. Include regression tests for "
            "disconnect before admission, disconnect during prefill, disconnect during batched decode, "
            "a stalled client, and shutdown with queued work. Explain each ownership transition."
        )
    return preface, body.splitlines(keepends=True), "\nEnd of source excerpts.\nUser: " + request


def count_tokens(args, prompt, env):
    if len(prompt.encode()) > 160000:
        return 1 << 30
    # --tokenize forces raw input. Supply the same text that main.c wraps
    # around an ordinary --think 75 prompt before counting its tokens.
    rendered = ("<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: 75 (range 1-100, the higher "
                "the value, the more thorough the reasoning)\n\n<｜User｜>" + prompt + "<｜Assistant｜><think>")
    if "<｜deepseek_image｜>" in prompt or "\x1d" in prompt:
        return 1 << 30  # keep text-only excerpts before an image-marker literal
    command = [str(ROOT / "build/myds4"), "-m", str(args.checkpoint), "-p", rendered,
               "--think", "75", "--ctx", "65536", "--tokenize"]
    p = subprocess.run(command, env=env, capture_output=True, timeout=30, check=True)
    if len(p.stdout) > 1024 * 1024:
        raise RuntimeError("tokenizer output exceeded its bound")
    return len(p.stdout.strip().split(b","))


def make_prompt(args, env):
    before, lines, after = workload(args.workload)
    low, high = 0, len(lines)
    while low < high:
        middle = (low + high + 1) // 2
        if count_tokens(args, before + "".join(lines[:middle]) + after, env) <= args.input_tokens:
            low = middle
        else:
            high = middle - 1
    prompt = before + "".join(lines[:low]) + after
    count = count_tokens(args, prompt, env)
    if count < args.input_tokens - 256 or count + args.tokens > args.context:
        raise RuntimeError("source excerpts do not fit the requested input/output budget")
    return prompt, count


def persist(path, data, record):
    data["runs"].append(record)
    text = json.dumps(data, indent=2) + "\n"
    if len(data["runs"]) > 64 or len(text.encode()) > 256 * 1024:
        raise RuntimeError("aggregate results exceeded their bound")
    if path:
        path.write_text(text)


def run(args, role, prompt, settings, expected, env):
    binary = ROOT / "build" / ("myds4" if role == "control" else "myds4-expert-batch")
    guard = wait_for_gpu(args.cache, args.gpu_wait, args.reserve, args.desktop)
    _idle(binary, env)
    command = [str(binary), "-m", str(args.checkpoint), "-p", prompt, "--ctx", str(args.context),
               "--cache", str(args.cache), "--reserve", str(args.reserve), "-n", str(args.tokens), "--temperature", "1", "--top-p", "1",
               "--seed", "12345", "--think", "75", "--draft", str(args.draft), "--draft-cache", "32", "--draft-trim"]
    if args.quant == "iq2":
        command += ["--experts", str(args.iq2)]
    print("EXPERT_BATCH_START " + json.dumps({"role": role, **settings}), flush=True)
    began = time.monotonic()
    p = guarded_process(command, env, args.timeout, guard["samples"][-1]["memory"], observe_graphics=True,
                        allow_cpu_helpers=args.desktop)
    if max(len(p.stdout), len(p.stderr)) > 256 * 1024:
        raise RuntimeError("model output exceeded its bound")
    stderr = p.stderr.decode(errors="replace")
    metrics = parse_metrics(stderr)
    grouping = None
    for line in stderr.splitlines():
        if line.startswith('{"expert_batch":'):
            grouping = json.loads(line)["expert_batch"]
    digest = hashlib.sha256(p.stdout).hexdigest()
    cache = metrics.get("decode_cache", {})
    decode = metrics.get("decode", {})
    valid = (p.returncode == 0 and decode.get("tokens", 0) >= 128 and
             metrics.get("prefill", {}).get("tokens") == settings["prompt_tokens"] and
             cache.get("slots_released") == 0 and cache.get("min_bytes", 0) / 1024**3 >= args.cache - .05 and
             cache.get("min_bytes") == cache.get("max_bytes") and (expected is None or digest == expected))
    if role == "candidate" and not (grouping and grouping.get("calls", 0)):
        valid = False
    return {
        "date_utc": datetime.now(timezone.utc).isoformat(), "role": role, "settings": settings,
        "valid": valid, "exit": p.returncode, "wall_seconds": round(time.monotonic() - began, 3),
        "output_sha256": digest, "output_matches_control": None if expected is None else digest == expected,
        "metrics": metrics, "grouping": grouping, "guard": guard,
        "gpu_activity_during_run": p.gpu_activity_samples,
        "background_cpu_helpers": p.background_cpu_helpers,
        "timing_environment": "recorded desktop load; GPU not isolated" if args.desktop else "quiet GPU admission",
        "details": [line for line in stderr.replace("\r", "\n").splitlines()
                    if line.startswith(("decode per token", "decode reads per token", "cache:", "expert cache:"))],
        "error": stderr[-2000:] if p.returncode else None,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--iq2", type=Path)
    ap.add_argument("--quant", choices=("mxfp4", "iq2"), default="mxfp4")
    ap.add_argument("--workload", choices=("cache-review", "server-review"), default="cache-review")
    ap.add_argument("--input-tokens", type=int, default=8192)
    ap.add_argument("--tokens", type=int, default=1024)
    ap.add_argument("--context", type=int, default=32768)
    ap.add_argument("--cache", type=float, default=24)
    ap.add_argument("--reserve", type=float, default=10, help="engine reserve in GiB; the admission guard keeps it too")
    ap.add_argument("--desktop", action="store_true",
                    help="record ordinary desktop activity; still exclude other model/GPU jobs and large compute jobs")
    ap.add_argument("--draft", type=int, choices=(2, 5), default=2)
    ap.add_argument("--order", nargs="+", choices=("control", "candidate"), default=["control", "candidate", "control"])
    ap.add_argument("--results", type=Path)
    ap.add_argument("--describe", action="store_true", help="count the prompt without loading the model")
    ap.add_argument("--gpu-wait", type=int, default=60)
    ap.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args()
    if not (4096 <= args.input_tokens <= 16384 and 256 <= args.tokens <= 4096 and
            args.input_tokens + args.tokens <= args.context <= 32768 and 20 <= args.cache <= 48 and 4 <= args.reserve <= 16 and
            5 <= args.gpu_wait <= 600 and 60 <= args.timeout <= 1800 and len(args.order) <= 6):
        ap.error("study configuration exceeds its limits")
    if args.quant == "iq2" and args.iq2 is None:
        ap.error("--quant iq2 needs --iq2")
    env = {k: v for k, v in os.environ.items() if not k.startswith("MYDS4_") or k == "MYDS4_LOCK"}
    _idle(ROOT / "build/myds4", env)
    prompt, count = make_prompt(args, env)
    settings = {"workload": args.workload, "prompt_tokens": count, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "checkpoint_name": args.checkpoint.name, "expert_file": args.iq2.name if args.quant == "iq2" else None,
                "requested_tokens": args.tokens, "context": args.context, "cache_gib": args.cache,
                "reserve_gib": args.reserve, "quant": args.quant,
                "desktop_load": args.desktop,
                "temperature": 1, "top_p": 1, "seed": 12345, "think": 75, "draft": args.draft,
                "draft_cache": 32, "draft_trim": True, "online_predictor": False, "substitute": False}
    if args.describe:
        print(json.dumps(settings, indent=2))
        return
    data = {"schema_version": 1, "kind": "expert_batch", "runs": []}
    if args.results and args.results.exists():
        data = json.loads(args.results.read_text())
        if data.get("schema_version") != 1 or data.get("kind") != "expert_batch":
            raise RuntimeError("result file belongs to a different study")
    expected = next((r["output_sha256"] for r in reversed(data["runs"])
                     if r.get("valid") and r["role"] == "control" and r["settings"] == settings), None)
    for role in args.order:
        if role == "candidate" and expected is None:
            raise RuntimeError("run a matching control before the candidate")
        record = run(args, role, prompt, settings, expected, env)
        persist(args.results, data, record)
        print("EXPERT_BATCH_RESULT " + json.dumps(record), flush=True)
        if not record["valid"]:
            raise RuntimeError("invalid comparison: check output identity, cache capacity and completed work")
        if expected is None:
            expected = record["output_sha256"]


if __name__ == "__main__":
    try:
        main()
    except GPUUnavailable as error:
        raise SystemExit(f"No performance result: {error}") from None
