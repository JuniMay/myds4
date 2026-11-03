"""Serial live DSpark predictor trials with bounded, RAM-only output capture.

Shadow trials collect real inference signals and fit every ablation online,
but issue no additional prefetches. Live trials enable one candidate's reads.
Neither mode executes generated text or saves traces, logs or learned weights.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
from benchmark_suite import parse_metrics, _idle
from gpu_guard import GPUUnavailable, guarded_process, wait_for_gpu

ROOT = Path(__file__).resolve().parents[2]
VARIANTS = ("embedding", "hidden", "streams", "mix", "router", "attention", "kv", "history", "final", "all")
REASONING = """Develop a careful solution to this scheduling problem. A finite set of jobs forms a directed acyclic graph. Each job has a positive integer duration, and an edge means its source must complete before its destination starts. There are two identical workers. Explain why simply choosing the longest ready job is not always optimal. Construct and check a small counterexample, explain exact dynamic programming for small graphs, and derive a useful lower bound. Then give a practical deterministic scheduling algorithm with pseudocode and tests for dependencies, equal priorities, idle workers and failure propagation. Verify each example by listing start and finish times. Discuss which guarantees are proved and which choices are heuristics."""


def reasoning_prompt():
    fixtures = []
    for case in range(12):
        jobs = ", ".join(f"J{j}={1 + (case * 7 + j * 3) % 9}" for j in range(8))
        edges = [(j, k) for j in range(8) for k in range(j + 1, 8) if (j * 5 + k * 3 + case) % 7 < 2]
        links = ", ".join(f"J{a}->J{b}" for a, b in edges)
        fixtures.append(f"Fixture {case + 1}: durations {jobs}. Dependencies: {links}.")
    return REASONING + "\nUse these additional DAG fixtures to check your reasoning:\n" + "\n".join(fixtures)


def run(args, quant, prompt, mode, variant, expected):
    binary = ROOT / "build" / "myds4-predict-lab"
    env = {k: v for k, v in os.environ.items() if not k.startswith("MYDS4_") or k == "MYDS4_LOCK"}
    env.update(MYDS4_LAB_MODE=mode, MYDS4_LAB_VARIANT=variant,
               MYDS4_LAB_RANK=str(args.rank), MYDS4_LAB_WIDTH=str(args.width),
               MYDS4_LAB_GAIN=str(args.feature_gain),
               MYDS4_LAB_DISTANCE=str(args.distance), MYDS4_LAB_HORIZON=str(args.horizon),
               MYDS4_LAB_BASE=args.base, MYDS4_LAB_LEARN="0" if args.no_learn else "1",
               MYDS4_LAB_POLICY=args.policy,
               MYDS4_LAB_LONG="1" if args.long_range else "0",
               MYDS4_LAB_LONG_MISSES="1" if args.long_missing_first else "0",
               MYDS4_LAB_WASTE=str(args.waste_cost),
               MYDS4_LAB_VICTIMS=str(args.victim_candidates),
               MYDS4_PREFETCH_METRICS="1" if args.metrics else "0")
    gpu_guard = wait_for_gpu(args.cache, args.gpu_wait)
    try:
        _idle(binary, env)
    except RuntimeError as error:
        raise GPUUnavailable("another model acquired the lock after GPU inspection") from error
    command = [str(binary), "-m", str(args.checkpoint), "-p", prompt,
               "--ctx", "8192", "--cache", str(args.cache), "-n", str(args.tokens),
               "--temperature", "1", "--top-p", "1", "--seed", "12345", "--think", "75",
               "--draft", "2", "--draft-cache", "32"]
    if not args.native_router_only:
        command += ["--online-predictor"]
    if quant == "q2":
        command += ["--experts", str(args.q2)]
    started = time.monotonic()
    process = guarded_process(command, env, args.timeout, gpu_guard["samples"][-1]["memory"])
    if len(process.stdout) > 256 * 1024 or len(process.stderr) > 256 * 1024:
        raise RuntimeError("trial output exceeded its bound")
    stderr = process.stderr.decode(errors="replace")
    metrics = parse_metrics(stderr)
    learned = None
    for line in stderr.splitlines():
        if line.startswith('{"predictor_lab":'):
            learned = json.loads(line)["predictor_lab"]
    digest = hashlib.sha256(process.stdout).hexdigest()
    pf = metrics.get("prefetch_attribution", {}).get("decode", {})
    report = {
        "case": args.case, "quant": quant, "mode": mode, "variant": variant,
        "temperature": 1, "top_p": 1, "seed": 12345, "draft": 2,
        "cache_requested_gib": args.cache, "exit": process.returncode,
        "gpu_guard": gpu_guard,
        "base": args.base, "learning": not args.no_learn,
        "feature_gain": args.feature_gain,
        "long_range": args.long_range,
        "long_missing_first": args.long_missing_first, "waste_cost": args.waste_cost,
        "victim_candidates": args.victim_candidates,
        "policy": args.policy,
        "online_corrector": not args.native_router_only,
        "wall_seconds": round(time.monotonic() - started, 3),
        "output_equal_to_control": None if expected is None else digest == expected,
        "decode": metrics.get("decode"), "cache": metrics.get("cache"),
        "decode_cache": metrics.get("decode_cache"),
        "acceptance": metrics.get("draft"), "sharing": metrics.get("decode_experts"),
        "learner": learned,
        "attribution": {key: pf.get(key) for key in (
            "served", "admissions_by_issuer", "fresh_timely_first_uses_by_issuer",
            "first_uses_by_issuer", "target_pass_first_uses_by_issuer",
            "target_pass_first_uses_before_io_started_by_issuer",
            "target_pass_admission_to_first_use_ns_by_issuer", "prefetch_origin_read_bytes",
            "demand_origin_read_bytes", "unused_prefetch_read_bytes_removed",
            "prefetch_caused_evictions", "unconsumed_prefetch_read_bytes_live")},
        "memory": [s for s in stderr.splitlines() if s.startswith("cache:")],
        "error": stderr[-2000:] if process.returncode else None,
    }
    print("PREDICTOR_LAB_RESULT " + json.dumps(report), flush=True)
    if process.returncode:
        raise RuntimeError("model failed; stopping serial trials")
    if expected is not None and digest != expected:
        raise RuntimeError("generated output differs from the exact control")
    if metrics.get("cache", {}).get("gib_including_draft_at_end", 0) < args.cache - .2:
        raise RuntimeError("cache shrank; stop rather than compare unequal cache budgets")
    cache = metrics.get("decode_cache")
    if cache and (cache["slots_released"] or cache["min_bytes"] / 1024**3 < args.cache - .05):
        raise RuntimeError("decode did not hold the requested cache throughout")
    for line in report["memory"]:
        found = re.search(r"(\d+) pages swapped out and (\d+) compressed", line)
        if found and any(int(n) for n in found.groups()):
            raise RuntimeError("system swapped or compressed memory during the trial")
    return digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--q2", type=Path, required=True)
    parser.add_argument("--quant", choices=("both", "mxfp4", "q2"), default="both")
    parser.add_argument("--phase", choices=("shadow", "live"), default="shadow")
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=["embedding", "all"])
    parser.add_argument("--case", choices=("scheduler", "reasoning"), default="scheduler")
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--cache", type=float, default=28)
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--feature-gain", type=float, default=1)
    parser.add_argument("--long-range", action="store_true", help="forecast from actual backbone inputs four/eight layers ahead")
    parser.add_argument("--long-missing-first", action="store_true", help="skip cached candidates before choosing long-range admissions")
    parser.add_argument("--waste-cost", type=float, default=.5, help="cost assigned to unused completed prefetch bytes")
    parser.add_argument("--victim-candidates", type=int, default=8)
    parser.add_argument("--base", choices=("none", "embedding", "draft"), default="draft")
    parser.add_argument("--no-learn", action="store_true")
    parser.add_argument("--policy", choices=("rank", "miss", "bandit", "native", "native-control", "demand"), default="rank")
    parser.add_argument("--policies", nargs="+", choices=("rank", "miss", "bandit", "native", "native-control", "demand"),
                        help="compare these policies serially within the same controls")
    parser.add_argument("--native-router-only", action="store_true", help="use existing router forecasts without the online RLS corrector")
    parser.add_argument("--width", type=int, default=1)
    parser.add_argument("--distance", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--repeat-control", action="store_true")
    parser.add_argument("--repeat-variants", action="store_true")
    parser.add_argument("--metrics", action="store_true")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--gpu-wait", type=int, default=900, help="maximum seconds to wait for a quiet GPU before each run")
    args = parser.parse_args()
    if not (32 <= args.tokens <= 2048 and 20 <= args.cache <= 48 and 16 <= args.rank <= 256 and 0 < args.feature_gain <= 16
            and 1 <= args.width <= 6 and 1 <= args.distance <= args.horizon <= 40 and 1 <= args.victim_candidates <= 128
            and 0 <= args.waste_cost <= 8 and (not args.long_missing_first or args.long_range)
            and 30 <= args.timeout <= 1800 and 5 <= args.gpu_wait <= 3600):
        parser.error("configuration exceeds bounded study limits")
    prompt = (ROOT / "data/benchmarks/long_scheduler.txt").read_text() if args.case == "scheduler" else reasoning_prompt()
    for quant in (("mxfp4", "q2") if args.quant == "both" else (args.quant,)):
        expected = run(args, quant, prompt, "off", "embedding", None)
        if args.phase == "shadow":
            run(args, quant, prompt, "shadow", "sweep", expected)
        else:
            trials = [(policy, variant) for policy in (args.policies or [args.policy]) for variant in args.variants]
            if args.repeat_variants:
                trials += list(reversed(trials))
            for policy, variant in trials:
                args.policy = policy
                run(args, quant, prompt, "prefetch", variant, expected)
        if args.repeat_control:
            run(args, quant, prompt, "off", "embedding", expected)


if __name__ == "__main__":
    main()
