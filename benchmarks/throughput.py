"""Total throughput of concurrent requests, with a guard that they stay correct.

A case is a set of requests over one or more models, written as model:count
pairs (deepseek:12 or deepseek:3,qwen:3). Each mode serves a case on a fresh
server and reports the tokens per second over all its requests:

  old:1    the --old server, one request at a time per model (batch 1)
  old:b    the --old server, as many requests per model as its batch takes
  new      the --new server, every request admitted at once

--arm NAME [binary=PATH] [model:key:json-value ...] adds a mode: the --new
server (or PATH) with these configuration keys (for example
deepseek:rows_per_call:1), every request admitted at once. With --repeats R the modes run R times, every other time in
reverse order (A B C C B A for two), --pause seconds apart, all against one
reference.

--late-file FILE makes the last request of each case a long one: it asks for
a summary of the first --late-chars characters of FILE and is sent once every
other request has received --late-after tokens, so that its prefill runs while
the others decode.

The reference run sends each request alone to the --new server, so every
response of every mode must equal what that request returns alone: greedy
sampling makes an answer depend on nothing but the prompt and the model. A
response that differs fails the case, whatever its speed. --reference-cache
keeps those answers' digests, so that a later case with the same requests
runs only the new ones alone. The reference answers' share of 8-word windows
that occur nowhere before them is printed: greedy decoding that loops reads
few experts and runs fast.

Each mode reports each request's time to its first token (from when it was
sent) and its decode rate ((tokens - 1) over the time from its first token to
its last); the steady rate (every request decoding, none finished yet); the
expert bytes read, the expert cache's hit rate and the GPU's mean use (sampled
from the system, so it includes the desktop's share). A server that reports
engine_times also splits the steady window, and with --late-file the late
request's prefill: the GPU main queue's busy share and its idle time by what
the engine thread did meanwhile, the engine thread's time by its state, the
SSD's busy share, the reads in flight on average and the expert bytes read.

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
STATES = ("outside", "pass", "wait_gpu", "wait_read", "wait_other")


def prompt(index):
    return f"Write a short story about {TOPICS[index % len(TOPICS)]}, in the first person, and keep it under a page."


def serve(binary, models, batch, args, options=(), kv_gb=8):
    """A server on a free port; returns its process, base URL and key."""
    entries = []
    for name, count in models.items():
        entry = {"model": str(getattr(args, name)), "id": name, "batch": batch if batch else count,
                 "temperature": 0, "top_p": 1, "top_k": 1, "idle_unload": 0}
        if name == "deepseek" and args.experts:
            entry["experts"] = str(args.experts)
        for option in [*(args.option or []), *options]:
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


def complete(base, key, body, arrivals=None):
    """One streamed chat completion: its answer's digest, its token count and
    when it was sent; `arrivals` receives the time of each chunk as it comes."""
    request = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    pieces = {"content": [], "reasoning_content": []}
    usage, first, started = None, None, time.monotonic()
    arrivals = [] if arrivals is None else arrivals
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
            "tokens": usage["completion_tokens"], "prompt_tokens": usage.get("prompt_tokens"), "sent": started,
            "seconds": time.monotonic() - started, "first_token_seconds": None if first is None else first - started,
            "text": text, "arrivals": arrivals}


def requests_of(models, args):
    bodies = []
    index = 0
    for name, count in models.items():
        for _ in range(count):
            bodies.append({"model": name, "messages": [{"role": "user", "content": prompt(index)}],
                           "max_tokens": args.tokens, "temperature": 0, "stream": True,
                           "stream_options": {"include_usage": True}})
            index += 1
    if args.late_file:
        text = args.late_file.read_text()[:args.late_chars]
        bodies[-1]["messages"][0]["content"] = "Summarize the following text in one paragraph.\n\n" + text
    return bodies


def fresh_share(text, n=8):
    """The share of the text's n-word windows that occur nowhere before them (1: no repeats)."""
    words = text.split()
    seen, fresh = set(), 0
    for i in range(len(words) - n + 1):
        window = tuple(words[i:i + n])
        fresh += window not in seen
        seen.add(window)
    return fresh / max(1, len(words) - n + 1)


def status_of(base, key, name):
    request = urllib.request.Request(f"{base}/v1/models/{name}", headers={"Authorization": "Bearer " + key})
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=30) as response:
        return json.loads(response.read())["status"]


def counters(base, key, names):
    """Each model's expert cache and decode counters, from the server's status,
    and the process's engine_times (from the first model's status)."""
    result = {}
    for name in names:
        status = status_of(base, key, name)
        result[name] = {"read_gb": status["expert_cache"]["read_gb"], "hits": status["expert_cache"]["hits"],
                        "misses": status["expert_cache"]["misses"], "read_wait": status["model_decode"]["read_wait_seconds"],
                        "rows_per_call": status.get("rows_per_call"), "split": status.get("split_decoding_calls"),
                        "available_gib": status.get("memory", {}).get("available_gib"),
                        "cache_gib": status["expert_cache"].get("gib"), "slots": status["expert_cache"].get("slots")}
        result.setdefault("engine_times", status.get("engine_times"))
    return result


def sample(stop_flag, base, key, names, samples, gpu_samples):
    """Until stop_flag is set: the counters four times a second, and the GPU's
    device use from the system once a second."""
    next_gpu = time.monotonic()
    while not stop_flag.wait(.25):
        try:
            samples.append({"t": time.monotonic(), **counters(base, key, names)})
        except (OSError, ValueError, KeyError):
            pass
        if time.monotonic() >= next_gpu:
            next_gpu += 1
            output = subprocess.run(["ioreg", "-r", "-d", "1", "-c", "IOAccelerator"], capture_output=True, text=True).stdout
            at = output.find('"Device Utilization %"=')
            if at >= 0:
                gpu_samples.append(int(output[at + 23:].split(",")[0].split("}")[0]))


def window(samples, t0, t1, outputs, names):
    """Where the time went from t0 to t1, from the counter samples just outside
    them: shares of the counters' own span (the engine thread's total time)."""
    before = [s for s in samples if s["t"] <= t0]
    after = [s for s in samples if s["t"] >= t1]
    if t1 <= t0 or not before or not after or not before[-1].get("engine_times") or not after[0].get("engine_times"):
        return None
    a, b = before[-1], after[0]
    ea, eb = a["engine_times"], b["engine_times"]
    thread = {k: eb["thread_seconds"][k] - ea["thread_seconds"][k] for k in STATES}
    span = sum(thread.values())
    if span <= 0:
        return None
    idle = {k: eb["gpu_idle_seconds"][k] - ea["gpu_idle_seconds"][k] for k in STATES}
    busy = eb["gpu_busy_seconds"] - ea["gpu_busy_seconds"]
    tokens = sum(sum(t0 < t <= t1 for t in o["arrivals"]) for o in outputs)
    read_gb = sum(b[n]["read_gb"] - a[n]["read_gb"] for n in names)
    splits = [s[n]["split"] for s in samples if t0 <= s["t"] <= t1 for n in names if s[n].get("split") is not None]
    return {"seconds": round(t1 - t0, 2), "counter_span_seconds": round(span, 2), "tokens": tokens,
            "tokens_per_second": round(tokens / (t1 - t0), 2),
            "gpu_busy": round(busy / span, 3),
            "gpu_idle_while": {k: round(v / span, 3) for k, v in idle.items()},
            "gpu_idle_unattributed": round(1 - busy / span - sum(idle.values()) / span, 3),
            "thread": {k: round(v / span, 3) for k, v in thread.items()},
            "ssd_busy": round((eb["ssd_busy_seconds"] - ea["ssd_busy_seconds"]) / span, 3),
            "demand_reads_in_flight": round((eb["demand_read_seconds"] - ea["demand_read_seconds"]) / span, 2),
            "background_reads_in_flight": round((eb["background_read_seconds"] - ea["background_read_seconds"]) / span, 2),
            "read_gb_per_second": round(read_gb / span, 2),
            "read_mb_per_token": round(1e3 * read_gb / span / (tokens / (t1 - t0)), 1) if tokens else None,
            "split_share": round(sum(splits) / len(splits), 2) if splits else None}


def steady_rate(outputs):
    """Tokens per second while every request is decoding: from the last first
    token to the first request's end, counting the responses' chunks."""
    begin = max(o["arrivals"][0] for o in outputs)
    end = min(o["arrivals"][-1] for o in outputs)
    if end <= begin:
        return None
    return sum(sum(begin < t <= end for t in o["arrivals"]) for o in outputs) / (end - begin)


def decode_rate(o):
    """(tokens - 1) over the time from the request's first chunk to its last."""
    span = o["arrivals"][-1] - o["arrivals"][0]
    return (o["tokens"] - 1) / span if span > 0 else None


def stop(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def send_all(base, key, bodies, args, late):
    """Sends the requests at once; with `late`, the last one once every other
    request has received args.late_after chunks (or ended)."""
    arrivals = [[] for _ in bodies]
    with ThreadPoolExecutor(max_workers=len(bodies)) as pool:
        early = bodies[:-1] if late else bodies
        futures = [pool.submit(complete, base, key, body, arrivals[i]) for i, body in enumerate(early)]
        if late:
            while not all(len(arrivals[i]) >= args.late_after or f.done() for i, f in enumerate(futures)):
                for f in futures:
                    if f.done():
                        f.result()  # raises a request's error
                time.sleep(.005)
            futures.append(pool.submit(complete, base, key, bodies[-1], arrivals[-1]))
        return [f.result() for f in futures]


def measure(binary, models, batch, bodies, args, concurrent=True, options=()):
    """Runs the requests on a fresh server, all at once or one after the other."""
    wait_for_gpu(args.memory + 8, 900, args.reserve, desktop=True, quiet_cpu=True)
    process, base, key = serve(binary, models, batch, args, options)
    try:
        # Each model first answers something else: the weights load and the
        # expert cache fills before the clock starts, as on a server in use.
        for name in models:
            complete(base, key, {"model": name, "messages": [{"role": "user", "content": "Describe a quiet morning in a harbour town."}],
                                 "max_tokens": args.warmup, "temperature": 0, "stream": True, "stream_options": {"include_usage": True}})
        before = counters(base, key, models)
        stop_flag, samples, gpu_samples = threading.Event(), [{"t": time.monotonic(), **before}], []
        sampler = threading.Thread(target=sample, args=(stop_flag, base, key, list(models), samples, gpu_samples), daemon=True)
        sampler.start()
        began = time.monotonic()
        if concurrent:
            outputs = send_all(base, key, bodies, args, bool(args.late_file))
        else:
            outputs = [complete(base, key, body) for body in bodies]
        wall = time.monotonic() - began
        time.sleep(.6)  # the worker publishes its counters four times a second
        stop_flag.set()
        sampler.join()
        after = counters(base, key, models)
        samples.append({"t": time.monotonic(), **after})
        use = {}
        for name in models:
            b, a = before[name], after[name]
            hits, misses = a["hits"] - b["hits"], a["misses"] - b["misses"]
            use[name] = {"read_gb": round(a["read_gb"] - b["read_gb"], 1), "read_wait_seconds": round(a["read_wait"] - b["read_wait"], 1),
                         "hit_percent": round(100 * hits / max(1, hits + misses), 1), "rows_per_call": a["rows_per_call"],
                         "available_gib": a["available_gib"], "cache_gib": a["cache_gib"], "slots": a["slots"]}
        windows = {}
        if concurrent:
            windows["steady"] = window(samples, max(o["arrivals"][0] for o in outputs),
                                       min(o["arrivals"][-1] for o in outputs), outputs, models)
            if args.late_file:
                late = outputs[-1]
                windows["late_prefill"] = window(samples, late["sent"], late["arrivals"][0], outputs[:-1], models)
                windows["before_late"] = window(samples, max(o["arrivals"][0] for o in outputs[:-1]), late["sent"],
                                                outputs[:-1], models)
        return wall, outputs, use, (sum(gpu_samples) / len(gpu_samples) if gpu_samples else None), windows
    finally:
        stop(process)


def show_window(name, w):
    if not w:
        print(f"         {name}: no counters", flush=True)
        return
    idle = ", ".join(f"{k} {v:.0%}" for k, v in w["gpu_idle_while"].items() if v >= .005)
    thread = ", ".join(f"{k} {v:.0%}" for k, v in w["thread"].items())
    print(f"         {name} {w['seconds']} s: {w['tokens_per_second']} tok/s; GPU busy {w['gpu_busy']:.0%}, idle while "
          f"[{idle}], unattributed {w['gpu_idle_unattributed']:.0%}; engine thread [{thread}]; SSD busy {w['ssd_busy']:.0%}, "
          f"reads in flight {w['demand_reads_in_flight']} demand + {w['background_reads_in_flight']} background, "
          f"{w['read_gb_per_second']} GB/s, {w['read_mb_per_token']} MB/token"
          + (f"; requests split over calls {w['split_share']:.0%} of the time" if w.get("split_share") is not None else ""), flush=True)


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
    parser.add_argument("--modes", default="old:1,old:b,new", help="the built-in modes to run, comma-separated (none: only --arm)")
    parser.add_argument("--arm", nargs="+", action="append", metavar="NAME [binary=PATH] [model:key:json]",
                        help="a mode of the --new server (or PATH) with these configuration keys")
    parser.add_argument("--repeats", type=int, default=1, help="runs of every mode, every other time in reverse order")
    parser.add_argument("--pause", type=float, default=0, help="seconds between runs")
    parser.add_argument("--late-file", type=Path, help="the last request summarizes this file's start, sent while the others decode")
    parser.add_argument("--late-chars", type=int, default=32000, help="characters of --late-file in the late request")
    parser.add_argument("--late-after", type=int, default=32, help="tokens every other request has received when the late one is sent")
    parser.add_argument("--results", type=Path, help="report file (JSON)")
    parser.add_argument("--server-log", type=Path, help="append the servers' messages to this file (-: to stderr)")
    parser.add_argument("--no-reference", action="store_true", help="skip the reference runs: no check of the responses")
    parser.add_argument("--reference-cache", type=Path, help="keep the requests' answers alone (digests) here for later cases")
    parser.add_argument("--option", action="append", help="a server configuration key of one model, in every mode: model:key:json-value")
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
        if args.late_file and len(bodies) < 2:
            parser.error("--late-file needs at least two requests per case")
        print(f"\ncase {case}: {len(bodies)} requests, {args.tokens} tokens each, memory {args.memory} GiB, greedy"
              + (f", the last one late with {args.late_chars} characters of {args.late_file.name}" if args.late_file else ""), flush=True)
        reference, alone = None, {}
        if not args.no_reference:
            # A request alone answers the same whatever the case around it, so
            # the cache keeps each answer's digest (never its text) for later cases.
            cache = json.loads(args.reference_cache.read_text()) if args.reference_cache and args.reference_cache.exists() else {}
            keys = [hashlib.sha256(json.dumps({"model": str(getattr(args, b["model"])), "experts": str(args.experts),
                                               "options": sorted(args.option or []), "body": b}, sort_keys=True).encode()).hexdigest()
                    for b in bodies]
            missing = [i for i, k in enumerate(keys) if k not in cache]
            if missing:
                _, outputs, _, _, _ = measure(args.new, models, 0, [bodies[i] for i in missing], args, concurrent=False)
                for i, o in zip(missing, outputs):
                    alone[i] = o
                    cache[keys[i]] = {"digest": o["digest"], "tokens": o["tokens"], "prompt_tokens": o["prompt_tokens"],
                                      "fresh": fresh_share(o["text"]), "first_token_seconds": o["first_token_seconds"],
                                      "decode_tokens_per_second": decode_rate(o)}
                if args.reference_cache:
                    args.reference_cache.write_text(json.dumps(cache, indent=1) + "\n")
            known = [cache[k] for k in keys]
            reference = [c["digest"] for c in known]
            print(f"  alone (reference; {len(missing)} run now, {len(keys) - len(missing)} from the cache): " +
                  ", ".join(f"{c['tokens']} tokens" for c in known[:4]) + ("" if len(known) <= 4 else " ...") +
                  f"; fewest fresh 8-word windows {min(c['fresh'] for c in known):.2f}", flush=True)
            for i in sorted(alone)[:2]:
                print(f"    request {i} starts: {alone[i]['text'][:80]!r}", flush=True)
            if args.late_file:
                print(f"    late request alone: {known[-1]['prompt_tokens']} prompt tokens, first token after "
                      f"{known[-1]['first_token_seconds']:.1f} s, decode {known[-1]['decode_tokens_per_second']:.2f} tok/s", flush=True)
        modes = []
        if args.old:
            modes.append(("old:1", args.old, 1, ()))
            modes.append(("old:b", args.old, min(args.max_old_batch, max(models.values())), ()))
        modes.append(("new", args.new, max(models.values()), ()))
        modes = [m for m in modes if m[0] in args.modes.split(",")]
        for arm in args.arm or []:
            binary = next((Path(t[7:]) for t in arm[1:] if t.startswith("binary=")), args.new)
            modes.append((arm[0], binary, len(bodies), tuple(t for t in arm[1:] if not t.startswith("binary="))))
        order = [m for r in range(args.repeats) for m in (modes if r % 2 == 0 else modes[::-1])]
        rows = []
        for run, (name, binary, batch, options) in enumerate(order):
            if run and args.pause:
                time.sleep(args.pause)
            wall, outputs, use, gpu, windows = measure(binary, models, batch, bodies, args, options=options)
            tokens = sum(o["tokens"] for o in outputs)
            steady = steady_rate(outputs)
            different = [i for i, o in enumerate(outputs) if reference and o["digest"] != reference[i]]
            failed |= bool(different)
            requests = [{"first_token_seconds": o["first_token_seconds"] and round(o["first_token_seconds"], 2), "tokens": o["tokens"],
                         "decode_tokens_per_second": decode_rate(o) and round(decode_rate(o), 2),
                         "prompt_tokens": o["prompt_tokens"]} for o in outputs]
            row = {"mode": name, "binary": str(binary), "options": list(options), "batch": batch, "wall_seconds": wall, "tokens": tokens,
                   "tokens_per_second": tokens / wall, "first_token_seconds_max": max(o["first_token_seconds"] or 0 for o in outputs),
                   "different_from_alone": different, "steady_tokens_per_second": steady, "gpu_percent": gpu, "models": use,
                   "requests": requests, "windows": windows}
            rows.append(row)
            print(f"  {name:6} batch {batch:<3} {tokens} tokens in {wall:7.1f} s: {tokens / wall:7.2f} tok/s; "
                  f"slowest first token {row['first_token_seconds_max']:.1f} s; "
                  f"{'same as alone' if not different else 'DIFFERS from alone: ' + str(different)}", flush=True)
            print(f"         steady {steady and round(steady, 2)} tok/s; GPU {gpu and round(gpu)}%; " +
                  "; ".join(f"{n}: {u['read_gb']} GB read, {u['hit_percent']}% hit, rows per call {u['rows_per_call']}, "
                            f"cache {u['cache_gib']} GiB in {u['slots']} slots, {u['available_gib']} GiB available"
                            for n, u in use.items()), flush=True)
            print("         requests: " + "; ".join(f"{r['first_token_seconds']} s to first, {r['decode_tokens_per_second']} tok/s"
                                                    for r in requests), flush=True)
            for label, w in windows.items():
                show_window(label, w)
            for i in different:
                if i not in alone:
                    print(f"    request {i} differs from its cached reference", flush=True)
                    continue
                a, b = alone[i]["text"], outputs[i]["text"]
                at = next((k for k in range(min(len(a), len(b))) if a[k] != b[k]), min(len(a), len(b)))
                print(f"    request {i} first differs at character {at}: alone {a[max(0, at - 20):at + 40]!r}, "
                      f"concurrent {b[max(0, at - 20):at + 40]!r}", flush=True)
            report.append({"case": case, "tokens": args.tokens, "memory_gib": args.memory, "reference": reference, "run": row})
            if args.results:
                args.results.write_text(json.dumps(report, indent=2) + "\n")
    print("\n" + ("FAILED: a response differs from the same request alone." if failed else "Every response equals the same request alone."))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
