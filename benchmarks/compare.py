"""Serial, bounded inference comparisons on repository and Harness prompts.

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
import resource
import shlex
import subprocess
import time

from evaluate import parse_metrics
from runtime import GPUUnavailable, guarded_process, wait_for_gpu

ROOT = Path(__file__).resolve().parents[1]
SOURCE_REVISION = None

def copy_commands():
    return "\n".join(f"python -m unittest tests.test_queue.QueueTests.test_case_{i:03d}" for i in range(48))


def excerpt(path, first=None, last=None):
    text = subprocess.check_output(["git", "show", f"{SOURCE_REVISION}:{path}"], cwd=ROOT, text=True) if SOURCE_REVISION else (ROOT / path).read_text()
    start = text.index(first) if first else 0
    end = text.index(last, start) if last else len(text)
    return f"\n=== {path} ===\n" + text[start:end] + "\n"


def workload(name):
    if name == "copy-commands":
        before = ("This is a text reproduction task. Do not execute commands.\n"
                  "The exact block to reproduce is delimited below.\nCOPY_START\n" + copy_commands() +
                  "\nCOPY_END\nThe following repository source is background context, not part of the block.\n")
        body = excerpt("src/engine/io.c") + excerpt("src/model/deepseek/model.c")
        after = ("\nFinal request: Repeat every line between COPY_START and COPY_END exactly, in the same order. "
                 "Output only those 48 command lines, without the delimiters, commentary, numbering or code fences.\n")
        return before, body.splitlines(keepends=True), after
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
        body += excerpt("src/engine/io.c", "int cache_request(", "void cache_load(")
        body += excerpt("src/engine/io.c", "// After a decoded layer", "static void resize(")
        for path in ("tests/unit/prefetch_queue.c", "tests/unit/cache_policy.c", "src/engine/myds4.h", "src/engine/io.c"):
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
    if len(prompt.encode()) > 900000:
        return 1 << 30
    # --tokenize forces raw input. Supply the same text that main.c wraps
    # around an ordinary --think 75 prompt before counting its tokens.
    rendered = prompt if args.harness_session else (
        "<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: 75 (range 1-100, the higher "
        "the value, the more thorough the reasoning)\n\n<｜User｜>" + prompt + "<｜Assistant｜><think>")
    if not args.harness_session and ("<｜deepseek_image｜>" in prompt or "\x1d" in prompt):
        return 1 << 30  # keep text-only excerpts before an image-marker literal
    command = [str(ROOT / "build/myds4"), "-m", str(args.checkpoint), "-p", rendered,
               "--think", "75", "--ctx", str(max(65536, args.context)), "--tokenize"]
    if args.harness_session:
        command += ["--rendered"]
    p = subprocess.run(command, env=env, capture_output=True, timeout=30, check=True)
    if len(p.stdout) > 1024 * 1024:
        raise RuntimeError("tokenizer output exceeded its bound")
    return len(p.stdout.strip().split(b","))


def make_prompt(args, env):
    if args.harness_session:
        from prefill import harness_request
        request = harness_request(args.harness_session, args.before_seq)
        payload = json.dumps(request, ensure_ascii=False).encode()
        if len(payload) > 2 * 1024 * 1024:
            raise RuntimeError("reconstructed request exceeds 2 MiB")
        rendered = subprocess.run([str(args.server.resolve()), "--render", "/dev/stdin"], input=payload,
                                  capture_output=True, check=True, timeout=30)
        prompt = rendered.stdout.decode()
        count = count_tokens(args, prompt, env)
        if count + args.tokens > args.context:
            raise RuntimeError("reconstructed request exceeds the context budget")
        args.harness_think = request["reasoning_effort"]
        return prompt, count
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


def draft_count(args, role):
    override = getattr(args, role + "_draft", None)
    return args.draft if override is None else override


def run(args, role, prompt, settings, expected, env):
    binary = args.control if role == "control" else args.candidate
    if hashlib.sha256(binary.read_bytes()).hexdigest() != settings[role + "_binary_sha256"]:
        raise RuntimeError("benchmark executable changed after the comparison was configured")
    guard = wait_for_gpu(args.memory + 2, args.gpu_wait, args.reserve, args.desktop, quiet_cpu=True)
    command = [str(binary), "-m", str(args.checkpoint), "-p", prompt, "--ctx", str(args.context),
               "--memory", str(args.memory), "--reserve", str(args.reserve), "-n", str(args.tokens), "--temperature", "1", "--top-p", "1",
               "--seed", str(getattr(args, "seed", 12345))]
    command += ["--rendered"] if args.harness_session else ["--think", "75"]
    drafts = draft_count(args, role)
    if drafts:
        command += ["--draft", str(drafts), "--draft-cache", "32"]
    command += shlex.split(args.control_args if role == "control" else args.candidate_args)
    if args.quant != "mxfp4":
        command += ["--experts", str(args.experts)]
    print("COMPARE_START " + json.dumps({"role": role, **settings}), flush=True)
    began = time.monotonic()
    cpu_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    p = guarded_process(command, env, args.timeout, guard["samples"][-1]["memory"], observe_graphics=True,
                        allow_cpu_helpers=args.desktop, abort_cpu_contention=True)
    cpu_after = resource.getrusage(resource.RUSAGE_CHILDREN)
    if max(len(p.stdout), len(p.stderr)) > 256 * 1024:
        raise RuntimeError("model output exceeded its bound")
    stderr = p.stderr.decode(errors="replace")
    metrics = parse_metrics(stderr)
    digest = hashlib.sha256(p.stdout).hexdigest()
    think_close = p.stdout.find(b"</think>")
    output_shape = {"bytes": len(p.stdout), "think_close_seen": think_close >= 0,
                    "bytes_before_think_close": think_close if think_close >= 0 else None,
                    "bytes_after_think_close": len(p.stdout) - think_close - len(b"</think>") if think_close >= 0 else None}
    copy_exact = None
    if args.workload == "copy-commands":
        body = p.stdout.decode(errors="replace").rsplit("</think>", 1)[-1].strip()
        copy_exact = body == copy_commands()
    cache = metrics.get("decode_cache", {})
    decode = metrics.get("decode", {})
    valid = (p.returncode == 0 and decode.get("tokens", 0) >= 128 and
             metrics.get("prefill", {}).get("tokens") == settings["prompt_tokens"] and
             cache.get("slots_released") == 0 and cache.get("min_bytes", 0) > 0 and
             cache.get("min_bytes") == cache.get("max_bytes") and (expected is None or digest == expected))
    if copy_exact is False:
        valid = False
    correctness_valid = valid
    cpu_contention = p.background_cpu_peak > 100
    if cpu_contention:
        valid = False
    return {
        "date_utc": datetime.now(timezone.utc).isoformat(), "role": role, "settings": settings,
        "valid": valid, "exit": p.returncode, "wall_seconds": round(time.monotonic() - began, 3),
        "correctness_valid": correctness_valid, "timing_valid": not cpu_contention,
        "timing_exclusion": "background CPU builds exceeded one core" if cpu_contention else None,
        "output_sha256": digest, "output_matches_control": None if expected is None else digest == expected,
        "output_shape": output_shape,
        "copy_exact": copy_exact,
        "metrics": metrics, "guard": guard,
        "memory_activity": p.memory_activity,
        "gpu_activity_during_run": p.gpu_activity_samples,
        "background_cpu_helpers": p.background_cpu_helpers,
        "background_cpu_peak": p.background_cpu_peak,
        "child_cpu_seconds": {
            "user": round(cpu_after.ru_utime - cpu_before.ru_utime, 6),
            "system": round(cpu_after.ru_stime - cpu_before.ru_stime, 6),
            "scope": "full model run plus diagnostic helper subprocesses; includes startup and prefill",
        },
        "timing_environment": "recorded desktop load; GPU not isolated" if args.desktop else "quiet GPU admission",
        "details": [line for line in stderr.replace("\r", "\n").splitlines()
                    if line.startswith(("decode per token", "decode reads per token", "cache:", "expert cache:"))],
        "error": stderr[-2000:] if p.returncode else None,
    }


def main():
    global SOURCE_REVISION
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--experts", type=Path, help="the GGUF expert file (any shard) for --quant iq2 or iq1")
    ap.add_argument("--quant", choices=("mxfp4", "iq2", "iq1"), default="mxfp4")
    ap.add_argument("--kind", choices=("memory", "representation", "lookup", "draft", "prefill", "cache"), default="draft")
    ap.add_argument("--control", type=Path, default=ROOT / "build/myds4")
    ap.add_argument("--candidate", type=Path, required=True)
    ap.add_argument("--control-args", default="", help="additional engine arguments, parsed without a shell")
    ap.add_argument("--candidate-args", default="", help="additional engine arguments, parsed without a shell")
    ap.add_argument("--workload", choices=("cache-review", "server-review", "copy-commands", "harness"), default="cache-review")
    ap.add_argument("--harness-session", type=Path)
    ap.add_argument("--before-seq", type=int)
    ap.add_argument("--server", type=Path, default=ROOT / "build/myds4-server")
    ap.add_argument("--input-tokens", type=int, default=8192)
    ap.add_argument("--tokens", type=int, default=1024)
    ap.add_argument("--context", type=int, default=32768)
    ap.add_argument("--memory", type=float, default=24)
    ap.add_argument("--reserve", type=float, default=10, help="engine reserve in GiB; the admission guard keeps it too")
    ap.add_argument("--desktop", action="store_true",
                    help="record ordinary desktop activity; still exclude other model/GPU jobs and large compute jobs")
    ap.add_argument("--draft", type=int, choices=range(6), default=2)
    ap.add_argument("--control-draft", type=int, choices=range(6), help="override --draft for the control")
    ap.add_argument("--candidate-draft", type=int, choices=range(6), help="override --draft for the candidate")
    ap.add_argument("--order", nargs="+", choices=("control", "candidate"), default=["control", "candidate", "control"])
    ap.add_argument("--results", type=Path)
    ap.add_argument("--describe", action="store_true", help="count the prompt without loading the model")
    ap.add_argument("--gpu-wait", type=int, default=60)
    ap.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args()
    SOURCE_REVISION = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if not (4096 <= args.input_tokens <= 114688 and 256 <= args.tokens <= 8192 and
            args.input_tokens + args.tokens <= args.context <= 131072 and 16 <= args.memory <= 80 and 4 <= args.reserve <= 16 and
            5 <= args.gpu_wait <= 600 and 60 <= args.timeout <= 1800 and len(args.order) <= 6):
        ap.error("study configuration exceeds its limits")
    if (args.quant != "mxfp4") != bool(args.experts):
        ap.error("--quant iq2 and iq1 need --experts, and mxfp4 takes none")
    if (args.workload == "harness") != bool(args.harness_session) or (args.harness_session and not args.before_seq):
        ap.error("--workload harness needs --harness-session and --before-seq")
    args.control = args.control.resolve()
    args.candidate = args.candidate.resolve()
    env = {k: v for k, v in os.environ.items() if not k.startswith("MYDS4_")}
    prompt, count = make_prompt(args, env)
    settings = {"workload": args.workload, "prompt_tokens": count, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "checkpoint_name": args.checkpoint.name, "expert_file": args.experts.name if args.experts else None,
                "requested_tokens": args.tokens, "context": args.context, "model_memory_gib": args.memory,
                "reserve_gib": args.reserve, "quant": args.quant,
                "kind": args.kind, "control_binary": args.control.name, "candidate_binary": args.candidate.name,
                "workload_revision": SOURCE_REVISION,
                "control_binary_sha256": hashlib.sha256(args.control.read_bytes()).hexdigest(),
                "candidate_binary_sha256": hashlib.sha256(args.candidate.read_bytes()).hexdigest(),
                "control_args": shlex.split(args.control_args), "candidate_args": shlex.split(args.candidate_args),
                "desktop_load": args.desktop,
                "temperature": 1, "top_p": 1, "seed": 12345,
                "think": args.harness_think if args.harness_session else 75, "draft": args.draft,
                "draft_cache": 32 if args.draft else 0, "online_predictor": False, "substitute": False}
    if args.control_draft is not None or args.candidate_draft is not None:
        by_role = {role: draft_count(args, role) for role in ("control", "candidate")}
        settings.update(draft=None, draft_cache=None, draft_by_role=by_role,
                        draft_cache_by_role={role: 32 if n else 0 for role, n in by_role.items()})
    if args.harness_session:
        settings.update(harness_session=args.harness_session.parent.name, before_seq=args.before_seq,
                        prompt_mode="reconstructed Harness request; recorded tool calls are not executed")
    if args.describe:
        print(json.dumps(settings, indent=2))
        return
    data = {"schema_version": 1, "kind": args.kind, "runs": []}
    if args.results and args.results.exists():
        data = json.loads(args.results.read_text())
        if data.get("schema_version") != 1 or data.get("kind") != args.kind:
            raise RuntimeError("result file belongs to a different study")
    expected = next((r["output_sha256"] for r in reversed(data["runs"])
                     if r.get("valid") and r["role"] == "control" and r["settings"] == settings), None)
    for role in args.order:
        if role == "candidate" and expected is None:
            raise RuntimeError("run a matching control before the candidate")
        record = run(args, role, prompt, settings, expected, env)
        persist(args.results, data, record)
        shown = record
        if args.results:
            shown = {"role": role, "valid": record["valid"], "timing_valid": record["timing_valid"],
                     "output_matches_control": record["output_matches_control"], "copy_exact": record["copy_exact"],
                     "metrics": record["metrics"],
                     "timing_exclusion": record["timing_exclusion"], "results": str(args.results)}
        print("COMPARE_RESULT " + json.dumps(shown), flush=True)
        if not record["valid"]:
            raise RuntimeError("invalid comparison: check output identity, cache capacity and completed work")
        if expected is None:
            expected = record["output_sha256"]


if __name__ == "__main__":
    try:
        main()
    except GPUUnavailable as error:
        raise SystemExit(f"No performance result: {error}") from None
