"""Bounded CED and DSpark comparisons using a reconstructed Harness request.

The server renders the request's prompt (build/myds4-server --render, through
a pipe). Session content and generated tokens remain in memory. Only aggregate
measurements are saved. Recorded tool calls are prompt data and are never
executed.
"""
import argparse
from compression import zstd
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from benchmark_suite import _idle
from gpu_guard import guarded_process, wait_for_gpu

ROOT = Path(__file__).resolve().parents[2]


def harness_request(path, before_seq):
    messages, header = [], None
    with zstd.open(path, "rt") as stream:
        for line in stream:
            event = json.loads(line)
            if event.get("seq", -1) >= before_seq:
                break
            if event["type"] == "request/header":
                header = event["data"]["header"]
            if "surfaceOp" not in event:
                continue
            if event["surfaceOp"] != "append":
                raise ValueError("This bounded reconstruction requires append-only message events")
            data = event["data"]
            message = data.get("message", data)
            blocks = message["content"]
            if any(b["type"] not in ("text", "reasoning", "tool-call") for b in blocks):
                raise ValueError("This study requires a text-only session prefix")
            role = message["role"]
            wire = {"role": "developer" if role == "system" else role,
                    "content": "".join(b["text"] for b in blocks if b["type"] == "text")}
            if role == "assistant":
                wire["reasoning_content"] = "\n".join(b["text"] for b in blocks if b["type"] == "reasoning")
                calls = [{"id": b["id"], "type": "function", "function": {
                    "name": b["name"], "arguments": json.dumps(json.loads(b["arguments"]), ensure_ascii=False,
                                                                 separators=(",", ":"))}}
                    for b in blocks if b["type"] == "tool-call"]
                if calls:
                    wire["tool_calls"] = calls
                if not wire["content"]:
                    wire["content"] = None
            if role == "tool":
                wire["tool_call_id"] = message["toolCallId"]
            messages.append(wire)
    if header is None or not messages:
        raise ValueError("No complete request header and message prefix found")
    # The inspected local OpenAI-compatible Harness adapter omits strict.
    tools = [{"type": "function", "function": {k: tool[k] for k in ("name", "description", "parameters")}}
             for tool in header["tools"]]
    return {"model": header["config"]["model"], "messages": messages, "tools": tools,
            "thinking_mode": "thinking", "reasoning_effort": header["config"]["reasoningEffort"],
            "temperature": 1, "top_p": 1}


def objects(text):
    return [json.loads(line) for line in text.splitlines() if line.startswith("{")]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--harness-session", type=Path, required=True)
    ap.add_argument("--before-seq", type=int, required=True)
    ap.add_argument("--expected-input-tokens", type=int, required=True)
    ap.add_argument("--q2", type=Path)
    ap.add_argument("--quant", choices=("mxfp4", "q2", "both"), default="mxfp4")
    ap.add_argument("--phase", choices=("validate", "prefill", "joint-prefill", "ced-generation", "dspark", "all"), default="all")
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--cache", type=float, default=20)
    ap.add_argument("--reserve", type=float, default=4)
    ap.add_argument("--binary", type=Path, default=ROOT / "build/ced-check")
    ap.add_argument("--server", type=Path, default=ROOT / "build/myds4-server", help="renders the request's prompt")
    ap.add_argument("--output", type=Path, default=ROOT / "data/benchmarks/ced_speed_results.json")
    args = ap.parse_args()
    if not 64 <= args.tokens <= 1024 or not 1 < args.expected_input_tokens < 16384:
        ap.error("Use 64–1024 generated tokens and fewer than 16,384 input tokens")
    if not 16 <= args.cache <= 80 or not 2 <= args.reserve <= 16:
        ap.error("Use a 16–80 GiB cache and 2–16 GiB reserve")
    if args.quant != "mxfp4" and args.q2 is None:
        ap.error("--q2 is required for IQ2 comparisons")
    request = harness_request(args.harness_session, args.before_seq)
    payload = json.dumps(request, ensure_ascii=False).encode()
    if len(payload) >= 2 * 1024 * 1024:
        raise ValueError("Reconstructed request exceeds 2 MiB")
    server = args.server.resolve()
    rendered = subprocess.run([str(server), "--render", "/dev/stdin"], input=payload, capture_output=True, check=True,
                              timeout=30)
    if any(line.startswith(b"image ") for line in rendered.stderr.splitlines()):
        raise ValueError("This study requires a text-only request")
    prompt = rendered.stdout
    binary = args.binary.resolve()
    command = [str(binary), str(args.checkpoint), "-", "--chat", "--steps", "16",
               "--cache", str(args.cache), "--reserve", str(args.reserve)]
    described = subprocess.run(command + ["--describe"], input=prompt, capture_output=True, check=True, timeout=30)
    setup = objects(described.stdout.decode())[0]["ced_setup"]
    if setup["prompt_tokens"] != args.expected_input_tokens:
        raise ValueError(f"The rendered prompt has {setup['prompt_tokens']} tokens; the recorded full input has "
                         f"{args.expected_input_tokens}")
    identity = {"binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
                "renderer_sha256": hashlib.sha256(server.read_bytes()).hexdigest(),
                "request_sha256": hashlib.sha256(payload).hexdigest(), "tokens": args.tokens,
                "prompt_tokens": setup["prompt_tokens"], "temperature": 1, "top_p": 1,
                "seed": 12345, "cache_gib": args.cache, "reserve_gib": args.reserve,
                "reasoning_effort": request["reasoning_effort"],
                "model": "deepseek-ai/DeepSeek-V4.1-Flash-1017"}
    report = json.loads(args.output.read_text()) if args.output.exists() else {
        "schema_version": 1, "scope": "Serial CED and DSpark model comparisons on reconstructed Harness requests",
        "private_prompt_or_generated_tokens_saved": False, "runs": []}
    if len(report["runs"]) > 48:
        raise ValueError("The aggregate already contains too many runs")
    plan = []
    if args.phase in ("all", "validate"):
        plan += [("validate-compact-128", ["--compare-compact", "128"]),
                 ("validate-compact-512", ["--compare-compact", "512"])]
    if args.phase in ("all", "prefill"):
        plan += [("prefill-control-a", ["--bench", "0"]),
                 ("prefill-128", ["--bench", "0", "--compact", "--keep", "128"]),
                 ("prefill-512", ["--bench", "0", "--compact", "--keep", "512"]),
                 ("prefill-control-b", ["--bench", "0"])]
    if args.phase == "joint-prefill":
        plan += [("joint-split-control-a", ["--bench", "0", "--prefill-plan", "split"]),
                 ("joint-off", ["--bench", "0", "--prefill-plan", "joint"]),
                 ("joint-ced-128", ["--bench", "0", "--prefill-plan", "joint", "--ced", "128"]),
                 ("joint-ced-512", ["--bench", "0", "--prefill-plan", "joint", "--ced", "512"]),
                 ("joint-split-control-b", ["--bench", "0", "--prefill-plan", "split"])]
    if args.phase == "ced-generation":
        plan += [("ced-generation-control-a", ["--bench", str(args.tokens)]),
                 ("ced-generation-128", ["--bench", str(args.tokens), "--compact", "--keep", "128"]),
                 ("ced-generation-512", ["--bench", str(args.tokens), "--compact", "--keep", "512"]),
                 ("ced-generation-control-b", ["--bench", str(args.tokens)])]
    if args.phase == "dspark":
        plan += [("dspark-control-a", ["--bench", str(args.tokens)]),
                 ("dspark-2", ["--bench", str(args.tokens), "--dspark", "2"]),
                 ("dspark-control-b", ["--bench", str(args.tokens)])]
    env = {k: v for k, v in os.environ.items() if not k.startswith("MYDS4_")}
    env.update(MYDS4_LOCK=str(ROOT / "build/myds4.lock"), MYDS4_PREFETCH_METRICS="1")
    for quant in (("mxfp4", "q2") if args.quant == "both" else (args.quant,)):
        def matching():
            return [r for r in report["runs"] if r["settings"] == identity and r["quant"] == quant and r["valid"]]
        for name, options in plan:
            if any(r["name"] == name for r in matching()):
                print("STUDY_REUSE " + json.dumps({"quant": quant, "name": name}), flush=True)
                continue
            if name.startswith("prefill-") and name not in ("prefill-control-a", "prefill-control-b"):
                required = "validate-compact-" + name.split("-")[-1]
                if not any(r["name"] == required for r in matching()):
                    raise ValueError("Run compact correctness validation with this build before timing it")
            if name in ("ced-generation-128", "ced-generation-512"):
                required = "validate-compact-" + name.split("-")[-1]
                if not any(r["name"] == required for r in matching()):
                    raise ValueError("Run compact correctness validation before the generation comparison")
            guard = wait_for_gpu(args.cache, 15, reserve_gib=args.reserve, desktop=True)
            _idle(ROOT / "build/myds4", env)
            cmd = command + options + (["--experts", str(args.q2)] if quant == "q2" else [])
            print("STUDY_START " + json.dumps({"quant": quant, "name": name}), flush=True)
            started = time.monotonic()
            p = guarded_process(cmd, env, 600, guard["samples"][-1]["memory"], observe_graphics=True,
                                allow_cpu_helpers=True, input_data=prompt)
            if len(p.stdout) + len(p.stderr) > 512 * 1024:
                raise RuntimeError("Diagnostic output exceeded its bound")
            records = objects(p.stdout.decode())
            bench = next((r["inference_bench"] for r in records if "inference_bench" in r), None)
            valid = p.returncode == 0
            error = p.stderr.decode(errors="replace")[-1500:] if not valid else None
            if bench:
                valid &= bench["cache_bytes_min"] == bench["cache_bytes_start"] == bench["cache_bytes_max"]
                valid &= bench["slots_released"] == 0
                valid &= abs(bench["cache_bytes_start"] / 1024**3 - args.cache) < 0.05
                if name.startswith("joint-") and name != "joint-split-control-a":
                    controls = [r["bench"] for r in matching() if r["name"] == "joint-split-control-a"]
                    if not controls:
                        raise ValueError("Missing split-prefill control")
                    equal = bench["prefill_logits_sha256"] == controls[0]["prefill_logits_sha256"]
                    bench["logits_match_split_prefill"] = equal
                    if name in ("joint-off", "joint-split-control-b") and not equal:
                        valid, error = False, "Joint or repeated split prefill changed logits"
                if name in ("dspark-2", "dspark-control-b"):
                    controls = [r["bench"] for r in matching() if r["name"] == "dspark-control-a"]
                    if not controls:
                        raise ValueError("Missing matched plain-decoding control")
                    valid &= (bench["output_token_sha256"], bench["generated"]) == (
                        controls[0]["output_token_sha256"], controls[0]["generated"])
                    if not valid:
                        error = "Output identity or fixed-cache check failed"
                if name.startswith("ced-generation-") and name != "ced-generation-control-a":
                    controls = [r["bench"] for r in matching() if r["name"] == "ced-generation-control-a"]
                    if not controls:
                        raise ValueError("Missing full-prefill generation control")
                    equal = (bench["output_token_sha256"], bench["generated"]) == (
                        controls[0]["output_token_sha256"], controls[0]["generated"])
                    bench["output_matches_full_prefill"] = equal
                    if name == "ced-generation-control-b" and not equal:
                        valid, error = False, "Repeated full-prefill control changed output"
            metrics = [x["prefetch_metrics"] for x in objects(p.stderr.decode(errors="replace")) if "prefetch_metrics" in x]
            record = {"settings": identity, "quant": quant, "name": name, "valid": bool(valid),
                      "exit_code": p.returncode, "elapsed_seconds": round(time.monotonic() - started, 3),
                      "bench": bench, "records": records if bench is None else [], "prefetch_metrics": metrics,
                      "guard": guard, "gpu_activity": p.gpu_activity_samples,
                      "background_cpu_helpers": p.background_cpu_helpers, "error": error}
            report["runs"].append(record)
            encoded = json.dumps(report, indent=2) + "\n"
            if len(report["runs"]) > 48 or len(encoded.encode()) > 512 * 1024:
                raise RuntimeError("Aggregate report exceeds its total size bound")
            args.output.write_text(encoded)
            print("STUDY_RESULT " + json.dumps({"quant": quant, "name": name, "valid": bool(valid),
                                                 "bench": bench, "error": error}), flush=True)
            if not valid:
                raise RuntimeError("Correctness or resource check failed; stopping the series")


if __name__ == "__main__":
    main()
