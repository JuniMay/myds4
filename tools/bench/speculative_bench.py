"""Compare plain, greedy-draft and rejection-sampled DSpark generation.

All runs use the same real Harness request, total expert-cache budget and
target sampler. Rejection sampling changes the seed-to-text mapping. This
runner measures performance and resource use; output digests are not an
accuracy test. Sampler distribution tests must pass before these runs.
Prompts and answers stay in memory, and generated tools are never executed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import expert_batch_bench as bench


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--iq2", type=Path)
    ap.add_argument("--harness-session", type=Path, required=True)
    ap.add_argument("--before-seq", type=int, required=True)
    ap.add_argument("--server", type=Path, default=bench.ROOT / "build/myds4-server")
    ap.add_argument("--control", type=Path, default=bench.ROOT / "build/myds4")
    ap.add_argument("--candidate", type=Path, default=bench.ROOT / "build/myds4-rejection-draft")
    ap.add_argument("--tokens", type=int, default=1024)
    ap.add_argument("--context", type=int, default=131072)
    ap.add_argument("--cache", type=float, default=56)
    ap.add_argument("--reserve", type=float, default=4)
    ap.add_argument("--draft", type=int, default=5)
    ap.add_argument("--seeds", type=int, nargs="+", default=[12345, 23456])
    ap.add_argument("--modes", nargs="+", choices=("plain", "greedy", "rejection"), default=["plain", "greedy", "rejection"])
    ap.add_argument("--desktop", action="store_true")
    ap.add_argument("--timeout", type=int, default=1200)
    ap.add_argument("--gpu-wait", type=int, default=60)
    ap.add_argument("--results", type=Path, required=True)
    args = ap.parse_args()
    if not (256 <= args.tokens <= 8192 and args.context <= 131072 and 20 <= args.cache <= 64 and
            4 <= args.reserve <= 16 and 1 <= args.draft <= 5 and 1 <= len(args.seeds) <= 4 and
            len(args.modes) <= 3 and 60 <= args.timeout <= 1800 and 5 <= args.gpu_wait <= 600):
        ap.error("configuration exceeds this study's bounds")
    # A failed probability check prevents every GPU run, including controls.
    subprocess.run([str(bench.ROOT / "build/speculative-sample-check")], check=True, timeout=30)
    args.checkpoint = args.checkpoint.resolve()
    args.control, args.candidate = args.control.resolve(), args.candidate.resolve()
    args.quant = "iq2" if args.iq2 else "mxfp4"
    args.workload, args.kind, args.input_tokens = "harness", "draft", 8192
    args.control_args = args.candidate_args = "--expert-batch --prefill-prune"
    env = {k: v for k, v in os.environ.items() if not k.startswith("MYDS4_") or k == "MYDS4_LOCK"}
    bench.SOURCE_REVISION = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=bench.ROOT, text=True).strip()
    prompt, count = bench.make_prompt(args, env)
    control, candidate, depth = args.control, args.candidate, args.draft
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (control, candidate)}
    report = {"schema_version": 1, "kind": "speculative_sampling", "runs": [],
              "scope": "live generation timing; distribution correctness has CPU tests, not a model quality benchmark",
              "quality_options": "CED and substitution off; temperature 1, top-p 1; fixed total expert cache",
              "same_seed_output_required": False}
    if args.results.exists():
        report = json.loads(args.results.read_text())
        if report.get("kind") != "speculative_sampling":
            raise RuntimeError("result file belongs to another study")
    for round_index, seed in enumerate(args.seeds):
        modes = args.modes if not round_index % 2 else list(reversed(args.modes))
        for mode in modes:
            args.control = args.candidate = candidate if mode == "rejection" else control
            args.draft, args.seed = (0 if mode == "plain" else depth), seed
            settings = {"mode": mode, "quant": args.quant, "before_seq": args.before_seq,
                        "harness_session": args.harness_session.parent.name,
                        "prompt_tokens": count, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                        "checkpoint_name": args.checkpoint.name, "expert_file": args.iq2.name if args.iq2 else None,
                        "requested_tokens": args.tokens, "context": args.context, "cache_gib": args.cache,
                        "reserve_gib": args.reserve, "temperature": 1, "top_p": 1, "seed": seed,
                        "draft": args.draft, "draft_cache": 32 if args.draft else 0, "draft_trim": bool(args.draft),
                        "expert_batch": True, "prefill_prune": True, "online_predictor": False, "substitute": False,
                        "control_binary_sha256": hashes[str(args.control)],
                        "candidate_binary_sha256": hashes[str(args.candidate)],
                        "binary": args.control.name, "workload_revision": bench.SOURCE_REVISION}
            record = bench.run(args, "candidate" if mode == "rejection" else "control", prompt, settings, None, env)
            record["mode"] = mode
            record["runtime_checks_valid"] = record.pop("correctness_valid")
            record["output_matches_control"] = None
            record["correctness_scope"] = "cache capacity and completed work; no output-equivalence assertion"
            if mode == "rejection" and not record["metrics"].get("rejection_sampling", {}).get("enabled"):
                record["valid"] = False
                record["error"] = "candidate did not report rejection sampling enabled"
            bench.persist(args.results, report, record)
            print("SPECULATIVE_RESULT " + json.dumps({"mode": mode, "seed": seed, "valid": record["valid"],
                                                      "metrics": record["metrics"], "results": str(args.results)}), flush=True)
            if not record["valid"]:
                raise RuntimeError("invalid run: inspect cache, completion and resource checks")


if __name__ == "__main__":
    main()
