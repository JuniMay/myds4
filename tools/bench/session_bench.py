"""Compare server throughput on concurrent DSH requests over HTTP.

One server runs at a time. Prompts, answers and diagnostics stay in memory;
the optional report stores counts, timings and response digests. Tool calls
produced by the model are never executed.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import socket
import subprocess
import threading
import time
import urllib.request

from ced_bench import harness_request
from gpu_guard import ROOT, guarded_process, memory_snapshot, process_snapshot, wait_for_gpu


def stop(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def requests(args):
    bodies, identities = [], []
    for seq in args.before_seq:
        body = harness_request(args.harness_session, seq)
        body.update(max_tokens=args.tokens, temperature=1, top_p=1, seed=12345,
                    stream=True, stream_options={"include_usage": True})
        payload = json.dumps(body, ensure_ascii=False).encode()
        if len(payload) > 2 << 20:
            raise ValueError("request exceeds 2 MiB")
        rendered = subprocess.run([str(args.binary), "--render", "/dev/stdin"], input=payload,
                                  capture_output=True, check=True, timeout=30).stdout
        counted = subprocess.run([str(ROOT / "build/myds4"), "-m", str(args.checkpoint),
                                  "--rendered", "--ctx", str(args.context), "--tokenize", "-p", rendered.decode()],
                                 capture_output=True, check=True, timeout=30).stdout
        n = len(counted.strip().split(b","))
        if n + args.tokens > args.context:
            raise ValueError("request exceeds the context budget")
        bodies.append(payload)
        identities.append({"before_seq": seq, "prompt_tokens": n, "request_sha256": hashlib.sha256(payload).hexdigest()})
    return bodies, identities


def run(args, mode, bodies, runahead=0):
    batch = int(mode)
    guard = wait_for_gpu(args.cache, 60, args.reserve, desktop=True, quiet_cpu=True)
    baseline = guard["samples"][-1]["memory"]
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    key = secrets.token_hex(24)
    env = {k: v for k, v in os.environ.items() if not k.startswith("MYDS4_")}
    env.update(MYDS4_API_KEY=key, MYDS4_LOCK=str(ROOT / "build/myds4.lock"))
    command = [str(args.binary), "-m", str(args.checkpoint), "--serve", str(port), "--ctx", str(args.context),
               "--cache", str(args.cache), "--reserve", str(args.reserve), "--kv-cache", "4", "--batch", str(batch),
               "--temperature", "1", "--top-p", "1", "--seed", "12345"]
    if runahead:
        command += ["--runahead"]
    if args.prefill_prune:
        command += ["--prefill-prune"]
    if args.experts:
        command += ["--experts", str(args.experts)]
    process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print("SESSION_PROCESS " + json.dumps({"runner_pid": os.getpid(), "server_pid": process.pid}), flush=True)
    done = threading.Event()
    failures, prefill_times, seen_batches = [], [], []
    counters = {"log_bytes": 0, "cpu_peak": 0.0, "maximum_cache_slots": 0, "cache_shrank": False,
                "request_running": False, "timing_cpu_peak": 0.0, "startup_cpu_peak": 0.0}
    grouping = {}
    lock = threading.Lock()

    def drain(stream):
        for line in iter(lambda: stream.readline(65537), ""):
            with lock:
                counters["log_bytes"] += len(line.encode())
                m = re.search(r"prefill of \d+ tokens in ([\d.]+) s", line)
                if m:
                    prefill_times.append(float(m[1]))
                m = re.search(r"batch: (\d+) requests decoding together", line)
                if m:
                    seen_batches.append(int(m[1]))
                if line.startswith('{"ready_groups":'):
                    grouping["ready"] = json.loads(line)["ready_groups"]
                overflow = counters["log_bytes"] > 4 << 20
            if overflow:
                failures.append("diagnostic output exceeded 4 MiB")
                stop(process)
                break
        stream.close()

    def call(path):
        request = urllib.request.Request(f"http://127.0.0.1:{port}{path}", headers={"Authorization": "Bearer " + key})
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=10) as response:
            payload = response.read((1 << 20) + 1)
        if len(payload) > 1 << 20:
            raise ValueError("status exceeds 1 MiB")
        return json.loads(payload)

    def monitor():
        began, last_progress = time.monotonic(), time.monotonic()
        while not done.wait(2) and process.poll() is None:
            try:
                helpers = []
                foreign = process_snapshot(child=process.pid, preflight=False, allow_cpu_helpers=True, background=helpers)
                mem = memory_snapshot()
                cpu = sum(p["cpu_percent"] for p in helpers)
                counters["cpu_peak"] = max(counters["cpu_peak"], cpu)
                phase_peak = "timing_cpu_peak" if counters["request_running"] else "startup_cpu_peak"
                counters[phase_peak] = max(counters[phase_peak], cpu)
                counters["compressed_input_gib"] = (mem["compressions"] - baseline["compressions"]) * mem["page_bytes"] / 1024**3
                counters["observed_peak_compressor_gib"] = max(counters.get("observed_peak_compressor_gib", 0), mem["compressor_gib"])
                if foreign or cpu > 100 or mem["swapouts"] > baseline["swapouts"]:
                    failures.append("competing compute, CPU contention or swapping")
                    stop(process)
                    return
                try:
                    state = call("/v1/status")
                    slots = state["expert_cache"]["slots"]
                    # Slots allocate lazily while the first prompt fills the
                    # cache. A smaller count afterward is a capacity change.
                    counters["cache_shrank"] |= slots < counters["maximum_cache_slots"]
                    counters["maximum_cache_slots"] = max(counters["maximum_cache_slots"], slots)
                except (OSError, ValueError, KeyError):
                    pass  # the listener opens after loading
                if time.monotonic() - last_progress >= 60:
                    print("SESSION_PROGRESS " + json.dumps({"mode": mode, "elapsed_seconds": round(time.monotonic() - began),
                          "free_gib": round(mem["free_gib"], 1), "cpu_percent": cpu}), flush=True)
                    last_progress = time.monotonic()
            except Exception as error:
                failures.append(type(error).__name__)
                stop(process)
                return

    def completion(payload):
        started = time.monotonic()
        request = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=payload,
                                        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
        pieces = {"content": [], "reasoning_content": []}
        calls, usage, first, total = {}, None, None, 0
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=args.timeout) as response:
            for raw in response:
                total += len(raw)
                if total > 2 << 20:
                    raise ValueError("one streamed response exceeded 2 MiB")
                if not raw.startswith(b"data:"):
                    continue
                raw = raw[5:].strip()
                if raw == b"[DONE]":
                    break
                chunk = json.loads(raw)
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for choice in chunk.get("choices", []):
                    delta = choice.get("delta", {})
                    if any(delta.get(k) for k in ("content", "reasoning_content", "tool_calls")) and first is None:
                        first = time.monotonic()
                    for name in pieces:
                        if delta.get(name):
                            pieces[name].append(delta[name])
                    for part in delta.get("tool_calls", []):
                        slot = calls.setdefault(part["index"], {"name": "", "arguments": ""})
                        function = part.get("function", {})
                        if function.get("name"):
                            slot["name"] = function["name"]
                        slot["arguments"] += function.get("arguments", "")
        if usage is None or not usage.get("completion_tokens"):
            raise ValueError("stream did not report completed-token usage")
        answer = {k: "".join(v) for k, v in pieces.items()}
        answer["tool_calls"] = [calls[k] for k in sorted(calls)]
        return {"seconds": time.monotonic() - started, "first_output_seconds": None if first is None else first - started,
                "usage": usage, "response_sha256": hashlib.sha256(json.dumps(answer, sort_keys=True).encode()).hexdigest()}

    readers = [threading.Thread(target=drain, args=(s,), daemon=True) for s in (process.stdout, process.stderr)]
    watcher = threading.Thread(target=monitor, daemon=True)
    timer = threading.Timer(args.timeout + 90, stop, args=(process,))
    for thread in [*readers, watcher]:
        thread.start()
    timer.start()
    try:
        deadline = time.monotonic() + 90
        while True:
            if process.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError("server failed to become ready")
            try:
                call("/health")
                initial = call("/v1/status")
                break
            except OSError:
                time.sleep(.2)
        print("SESSION_START " + json.dumps({"mode": mode, "requests": len(bodies),
              "binary": args.binary.name}), flush=True)
        counters["request_running"] = True
        began = time.monotonic()
        with ThreadPoolExecutor(max_workers=len(bodies)) as pool:
            outputs = list(pool.map(completion, bodies))
        elapsed = time.monotonic() - began
        counters["request_running"] = False
        final = call("/v1/status")
        generated = sum(r["usage"]["completion_tokens"] for r in outputs)
        prefill = sum(prefill_times)
        valid = (not failures and not counters["cache_shrank"] and
                 final["expert_cache"]["slots"] >= counters["maximum_cache_slots"] and
                 final["expert_cache"].get("slots_released") == initial["expert_cache"].get("slots_released") == 0 and
                 final["expert_cache"]["gib"] >= args.cache - .1 and generated >= 128 * len(bodies))
        if batch > 1 and len(bodies) > 1 and not seen_batches:
            valid = False
        if initial.get("runahead") != bool(runahead):
            failures.append("server did not apply the public runahead option")
            valid = False
        return {"mode": mode, "runahead": runahead, "valid": valid, "date_utc": datetime.now(timezone.utc).isoformat(),
                "generated_tokens": generated, "request_wall_seconds": elapsed, "reported_prefill_seconds": prefill,
                "aggregate_tokens_per_second": generated / elapsed,
                "aggregate_excluding_reported_prefill_tokens_per_second": generated / max(elapsed - prefill, 1e-9),
                "outputs": outputs, "seen_batches": sorted(set(seen_batches)), "grouping": grouping, "initial": initial, "final": final,
                "guard": guard, "cpu_peak": counters["cpu_peak"], "cache_shrank": counters["cache_shrank"],
                "compressed_input_gib": counters.get("compressed_input_gib", 0),
                "observed_peak_compressor_gib": counters.get("observed_peak_compressor_gib", baseline.get("compressor_gib", 0)),
                "timing_cpu_peak": counters["timing_cpu_peak"], "startup_cpu_peak": counters["startup_cpu_peak"],
                "maximum_cache_slots": counters["maximum_cache_slots"], "failures": failures}
    finally:
        done.set()
        timer.cancel()
        stop(process)
        for thread in [*readers, watcher]:
            thread.join(timeout=5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--harness-session", type=Path, required=True)
    parser.add_argument("--before-seq", type=int, nargs="+", default=[35, 88])
    parser.add_argument("--binary", type=Path, default=ROOT / "build/myds4-server")
    parser.add_argument("--run-binaries", type=Path, nargs="+",
                        help="one server executable per mode for matched HTTP comparisons; --binary renders requests")
    parser.add_argument("--experts", type=Path)
    parser.add_argument("--modes", nargs="+", default=["1", "2", "1"], help="the server's --batch for each run: 1, 2, 4 or 6")
    parser.add_argument("--runahead", type=int, nargs="+", help="one 0/1 value per mode: the server's --runahead")
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--context", type=int, default=65536)
    parser.add_argument("--cache", type=float, default=56)
    parser.add_argument("--reserve", type=float, default=4)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--prefill-prune", action="store_true")
    parser.add_argument("--describe", action="store_true")
    parser.add_argument("--results", type=Path)
    args = parser.parse_args()
    if args.runahead is None:
        args.runahead = [0] * len(args.modes)
    if not (1 <= len(args.before_seq) <= 6 and len(set(args.before_seq)) == len(args.before_seq)
            and 128 <= args.tokens <= 2048 and 16384 <= args.context <= 131072 and 16 <= args.cache <= 80
            and 4 <= args.reserve <= 16 and 120 <= args.timeout <= 1800 and len(args.modes) <= 6
            and len(args.runahead) == len(args.modes) and all(n in (0, 1) for n in args.runahead)
            and all(re.fullmatch(r"[1246]", s) for s in args.modes)):
        parser.error("configuration exceeds study bounds")
    if args.run_binaries and len(args.run_binaries) != len(args.modes):
        parser.error("--run-binaries needs one executable per HTTP mode")
    args.binary = args.binary.resolve()
    run_binaries = [p.resolve() for p in (args.run_binaries or [args.binary] * len(args.modes))]
    run_hashes = [hashlib.sha256(p.read_bytes()).hexdigest() for p in run_binaries]
    bodies, identities = requests(args)
    settings = {"requests": identities, "tokens": args.tokens, "context": args.context, "cache_gib": args.cache,
                "reserve_gib": args.reserve, "prefill_prune": args.prefill_prune, "temperature": 1, "top_p": 1,
                "seed": 12345, "binary_sha256": hashlib.sha256(args.binary.read_bytes()).hexdigest(),
                "checkpoint": args.checkpoint.name, "experts": args.experts.name if args.experts else None}
    if args.run_binaries:
        settings["run_binaries"] = [{"name": p.name, "sha256": digest}
                                    for p, digest in zip(run_binaries, run_hashes)]
    print("SESSION_SETTINGS " + json.dumps(settings), flush=True)
    if args.describe:
        return
    report = {"schema_version": 1, "scope": "Concurrent reconstructed DSH requests; generated tools not executed. "
              "Response identity covers returned text/reasoning/tool arguments, excluding generated call IDs. "
              "Decode estimate subtracts logged prefill wall time; it is not a GPU-only rate.", "settings": settings, "runs": []}
    if args.results and args.results.exists():
        old = json.loads(args.results.read_text())
        report["previous_attempts"] = old.get("previous_attempts", []) + old.get("runs", [])
    expected = None
    for mode, runahead, binary, digest in zip(args.modes, args.runahead, run_binaries, run_hashes):
        if hashlib.sha256(args.binary.read_bytes()).hexdigest() != settings["binary_sha256"]:
            raise RuntimeError("server executable changed during the comparison")
        if hashlib.sha256(binary.read_bytes()).hexdigest() != digest:
            raise RuntimeError("server executable changed during the comparison")
        run_args = argparse.Namespace(**vars(args))
        run_args.binary = binary
        record = run(run_args, mode, bodies, runahead)
        record["server_binary"] = {"name": binary.name, "sha256": digest}
        hashes = [r["response_sha256"] for r in record["outputs"]]
        record["responses_match_control"] = None if expected is None else hashes == expected
        record["valid"] &= expected is None or hashes == expected
        report["runs"].append(record)
        text = json.dumps(report, indent=2) + "\n"
        if len(text) > 256 << 10:
            raise RuntimeError("aggregate report exceeded 256 KiB")
        if args.results:
            args.results.write_text(text)
        print("SESSION_RESULT " + json.dumps({k: v for k, v in record.items()
              if k not in ("guard", "initial", "final", "gpu_activity_during_run")}), flush=True)
        if not record["valid"]:
            raise RuntimeError("invalid comparison; inspect resource guards, batching and response identity")
        if expected is None:
            expected = hashes


if __name__ == "__main__":
    main()
