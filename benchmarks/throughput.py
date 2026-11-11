"""Total throughput of concurrent requests, with a guard that they stay correct.

A case is a set of requests over one or more models, written as model:count
pairs (deepseek:12 or deepseek:3,qwen:3). Each mode serves a case on a fresh
server and reports the tokens per second over all its requests:

  old:1    the --old server, one request at a time per model (batch 1)
  old:b    the --old server, as many requests per model as its batch takes
  new      the --new server, every request admitted at once

The reference run sends each request alone to the --new server, so every
response of every mode must equal what that request returns alone: greedy
sampling makes an answer depend on nothing but the prompt and the model. A
response that differs fails the case, whatever its speed.

Each mode also reports what limits it: the steady rate (all requests decoding,
none finished yet), the expert bytes read and the time spent waiting for them
(the servers' own counters), the expert cache's hit rate and the GPU's mean use
(sampled from the system, so it includes the desktop's share).

Prompts stay in memory and the report holds only counts, timings and digests.
One server runs at a time; each run waits for an idle GPU and AC power.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request

from runtime import wait_for_gpu

TOPICS = ["a lighthouse keeper who finds a message in a bottle", "a clockmaker whose clocks run backwards",
          "a cartographer drawing a map of a city that moves", "a beekeeper who can hear the hive think",
          "a librarian cataloguing books that write themselves", "a sailor who collects fog in jars",
          "a gardener growing a forest in a single room", "a translator of languages nobody speaks yet",
          "a tailor sewing coats out of old letters", "a violin maker who repairs thunderstorms",
          "a baker whose bread remembers its recipe", "a postman delivering letters to the past",
          "a glassblower making windows onto other seasons", "a miner who finds a staircase of snow",
          "an astronomer counting stars that were never there", "a chess player whose pieces argue back",
          "a barber who cuts away bad memories", "a puppeteer whose puppets wake at noon",
          "a fisherman who catches songs in his nets", "a teacher who grades the weather",
          "a shepherd tending a flock of clouds", "a locksmith opening doors in dreams",
          "a potter shaping bowls that hold time", "a printer setting type for silent books"]


def prompt(index):
    return f"Write a short story about {TOPICS[index % len(TOPICS)]}, in the first person, and keep it under a page."


def serve(binary, models, batch, args, kv_gb=8):
    """A server on a free port; returns its process, base URL and key."""
    entries = []
    for name, count in models.items():
        entry = {"model": str(getattr(args, name)), "id": name, "batch": batch if batch else count,
                 "temperature": 0, "top_p": 1, "top_k": 1, "idle_unload": 0}
        if name == "deepseek" and args.experts:
            entry["experts"] = str(args.experts)
        for option in args.option or []:
            model, key, value = option.split(":", 2)
            if model == name:
                entry[key] = json.loads(value)
        entries.append(entry)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    config = {"memory": args.memory, "reserve": args.reserve, "kv_cache": kv_gb, "kv_disk": 0, "idle_unload": 0, "models": entries}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump(config, handle)
    key = secrets.token_hex(24)
    environment = {k: v for k, v in os.environ.items() if not k.startswith("MYDS4_") or k in (args.pass_env or [])}
    environment.update(MYDS4_API_KEY=key)
    log = sys.stderr if str(args.server_log) == "-" else open(args.server_log, "a") if args.server_log else subprocess.DEVNULL
    process = subprocess.Popen([str(binary), "--config", handle.name, "--serve", str(port), "--no-tui"], env=environment,
                               stdout=subprocess.DEVNULL, stderr=log, text=True)
    base = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 180
    while True:
        try:
            request = urllib.request.Request(base + "/health")
            urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=2).read()
            break
        except (OSError, urllib.error.URLError):
            if process.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError(f"{binary} did not start")
            time.sleep(.2)
    os.unlink(handle.name)
    return process, base, key


def complete(base, key, body):
    """One streamed chat completion: its answer's digest and its token count."""
    request = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    pieces = {"content": [], "reasoning_content": []}
    usage, first, started = None, None, time.monotonic()
    arrivals = []
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=1800) as response:
        for raw in response:
            if not raw.startswith(b"data:"):
                continue
            raw = raw[5:].strip()
            if raw == b"[DONE]":
                break
            chunk = json.loads(raw)
            if chunk.get("error"):
                raise RuntimeError("server error: " + json.dumps(chunk["error"]))
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {})
                if any(delta.get(k) for k in pieces):
                    arrivals.append(time.monotonic())
                    if first is None:
                        first = arrivals[-1]
                for name in pieces:
                    if delta.get(name):
                        pieces[name].append(delta[name])
    if not usage or not usage.get("completion_tokens"):
        raise RuntimeError("a response reported no tokens")
    answer = {k: "".join(v) for k, v in pieces.items()}
    text = answer["reasoning_content"] + answer["content"]
    return {"digest": hashlib.sha256(json.dumps(answer, sort_keys=True).encode()).hexdigest()[:16],
            "tokens": usage["completion_tokens"], "seconds": time.monotonic() - started,
            "first_token_seconds": None if first is None else first - started, "text": text, "arrivals": arrivals}


def requests_of(models, args):
    bodies = []
    index = 0
    for name, count in models.items():
        for _ in range(count):
            bodies.append({"model": name, "messages": [{"role": "user", "content": prompt(index)}],
                           "max_tokens": args.tokens, "temperature": 0, "stream": True,
                           "stream_options": {"include_usage": True}})
            index += 1
    return bodies


def counters(base, key, names):
    """Each model's expert cache and decode counters, from the server's status."""
    result = {}
    for name in names:
        request = urllib.request.Request(f"{base}/v1/models/{name}", headers={"Authorization": "Bearer " + key})
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=30) as response:
            status = json.loads(response.read())["status"]
        result[name] = {"read_gb": status["expert_cache"]["read_gb"], "hits": status["expert_cache"]["hits"],
                        "misses": status["expert_cache"]["misses"], "read_wait": status["model_decode"]["read_wait_seconds"]}
    return result


def gpu_use(stop_flag, samples):
    """Samples the GPU's device use once a second until stop_flag is set."""
    while not stop_flag.wait(1):
        output = subprocess.run(["ioreg", "-r", "-d", "1", "-c", "IOAccelerator"], capture_output=True, text=True).stdout
        at = output.find('"Device Utilization %"=')
        if at >= 0:
            samples.append(int(output[at + 23:].split(",")[0].split("}")[0]))


def steady_rate(outputs):
    """Tokens per second while every request is decoding: from the last first
    token to the first request's end, counting the responses' chunks."""
    begin = max(o["arrivals"][0] for o in outputs)
    end = min(o["arrivals"][-1] for o in outputs)
    if end <= begin:
        return None
    return sum(sum(begin < t <= end for t in o["arrivals"]) for o in outputs) / (end - begin)


def stop(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def measure(binary, models, batch, bodies, args, concurrent=True):
    """Runs the requests on a fresh server, all at once or one after the other."""
    wait_for_gpu(args.memory + 8, 900, args.reserve, desktop=True, quiet_cpu=True)
    process, base, key = serve(binary, models, batch, args)
    try:
        # Each model first answers something else: the weights load and the
        # expert cache fills before the clock starts, as on a server in use.
        for name in models:
            complete(base, key, {"model": name, "messages": [{"role": "user", "content": "Describe a quiet morning in a harbour town."}],
                                 "max_tokens": args.warmup, "temperature": 0, "stream": True, "stream_options": {"include_usage": True}})
        before = counters(base, key, models)
        stop_flag, samples = threading.Event(), []
        sampler = threading.Thread(target=gpu_use, args=(stop_flag, samples), daemon=True)
        sampler.start()
        began = time.monotonic()
        if concurrent:
            with ThreadPoolExecutor(max_workers=len(bodies)) as pool:
                outputs = list(pool.map(lambda body: complete(base, key, body), bodies))
        else:
            outputs = [complete(base, key, body) for body in bodies]
        wall = time.monotonic() - began
        stop_flag.set()
        sampler.join()
        time.sleep(.6)  # the worker publishes its counters four times a second
        after = counters(base, key, models)
        use = {}
        for name in models:
            b, a = before[name], after[name]
            hits, misses = a["hits"] - b["hits"], a["misses"] - b["misses"]
            use[name] = {"read_gb": round(a["read_gb"] - b["read_gb"], 1), "read_wait_seconds": round(a["read_wait"] - b["read_wait"], 1),
                         "hit_percent": round(100 * hits / max(1, hits + misses), 1)}
        return wall, outputs, use, (sum(samples) / len(samples) if samples else None)
    finally:
        stop(process)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--new", type=Path, required=True, help="the server under test")
    parser.add_argument("--old", type=Path, help="the server to compare with; omit to skip the old modes")
    parser.add_argument("--deepseek", type=Path)
    parser.add_argument("--experts", type=Path, help="DeepSeek's GGUF experts")
    parser.add_argument("--qwen", type=Path)
    parser.add_argument("--flash", type=Path)
    parser.add_argument("--nemotron", type=Path)
    parser.add_argument("--cases", nargs="+", required=True, help="model:count[,model:count ...] for each case")
    parser.add_argument("--tokens", type=int, default=192, help="tokens per response")
    parser.add_argument("--warmup", type=int, default=64, help="tokens of the first request to each model, before the clock starts")
    parser.add_argument("--memory", type=float, default=46, help="model memory, GiB")
    parser.add_argument("--reserve", type=float, default=10)
    parser.add_argument("--max-old-batch", type=int, default=6, help="the largest batch of the --old server")
    parser.add_argument("--modes", default="old:1,old:b,new", help="the modes to run, comma-separated")
    parser.add_argument("--results", type=Path, help="report file (JSON)")
    parser.add_argument("--server-log", type=Path, help="append the servers' messages to this file (-: to stderr)")
    parser.add_argument("--no-reference", action="store_true", help="skip the reference runs: no check of the responses")
    parser.add_argument("--option", action="append", help="a server configuration key of one model: model:key:json-value")
    parser.add_argument("--pass-env", nargs="*", help="MYDS4_ environment variables to pass to the servers")
    args = parser.parse_args()
    report = []
    failed = False
    for case in args.cases:
        models = {}
        for part in case.split(","):
            name, count = part.split(":")
            if not getattr(args, name, None):
                parser.error(f"case {case}: give --{name}")
            models[name] = int(count)
        bodies = requests_of(models, args)
        print(f"\ncase {case}: {len(bodies)} requests, {args.tokens} tokens each, memory {args.memory} GiB, greedy", flush=True)
        reference, alone = None, None
        if not args.no_reference:
            _, alone, _, _ = measure(args.new, models, 0, bodies, args, concurrent=False)
            reference = [o["digest"] for o in alone]
            print("  alone (reference): " + ", ".join(f"{o['tokens']} tokens" for o in alone[:4]) +
                  ("" if len(alone) <= 4 else " ..."), flush=True)
            for i, output in enumerate(alone[:2]):
                print(f"    request {i} starts: {output['text'][:80]!r}", flush=True)
        modes = []
        if args.old:
            modes.append(("old:1", args.old, 1))
            modes.append(("old:b", args.old, min(args.max_old_batch, max(models.values()))))
        modes.append(("new", args.new, max(models.values())))
        rows = []
        for name, binary, batch in modes:
            if name not in args.modes.split(","):
                continue
            wall, outputs, use, gpu = measure(binary, models, batch, bodies, args)
            tokens = sum(o["tokens"] for o in outputs)
            steady = steady_rate(outputs)
            different = [i for i, o in enumerate(outputs) if reference and o["digest"] != reference[i]]
            failed |= bool(different)
            row = {"mode": name, "batch": batch, "wall_seconds": wall, "tokens": tokens, "tokens_per_second": tokens / wall,
                   "first_token_seconds_max": max(o["first_token_seconds"] or 0 for o in outputs), "different_from_alone": different,
                   "steady_tokens_per_second": steady, "gpu_percent": gpu, "models": use}
            rows.append(row)
            print(f"  {name:6} batch {batch:<3} {tokens} tokens in {wall:7.1f} s: {tokens / wall:7.2f} tok/s; "
                  f"slowest first token {row['first_token_seconds_max']:.1f} s; "
                  f"{'same as alone' if not different else 'DIFFERS from alone: ' + str(different)}", flush=True)
            print(f"         steady {steady and round(steady, 2)} tok/s; GPU {gpu and round(gpu)}%; " +
                  "; ".join(f"{n}: {u['read_gb']} GB read, {u['read_wait_seconds']} s read wait, {u['hit_percent']}% hit" for n, u in use.items()), flush=True)
            for i in different:
                a, b = alone[i]["text"], outputs[i]["text"]
                at = next((k for k in range(min(len(a), len(b))) if a[k] != b[k]), min(len(a), len(b)))
                print(f"    request {i} first differs at character {at}: alone {a[max(0, at - 20):at + 40]!r}, "
                      f"concurrent {b[max(0, at - 20):at + 40]!r}", flush=True)
        report.append({"case": case, "tokens": args.tokens, "memory_gib": args.memory, "reference": reference, "runs": rows})
    if args.results:
        args.results.write_text(json.dumps(report, indent=2) + "\n")
    print("\n" + ("FAILED: a response differs from the same request alone." if failed else "Every response equals the same request alone."))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
