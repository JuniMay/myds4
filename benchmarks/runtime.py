"""GPU, memory and power guards for serial model runs.

wait_for_gpu waits until no competing compute process, GPU load or memory
shortage is visible and the machine is on AC power; guarded_process runs one
bounded process and discards the run if another workload or swapping
appears. In-memory compression is recorded and allowed. Neither is an exclusive lease.
"""
import json
import os
from pathlib import Path
import plistlib
import re
import signal
import subprocess
import threading
import time

ROOT = Path(__file__).resolve().parents[1]


class GPUUnavailable(RuntimeError):
    pass


def process_snapshot(child=None, preflight=True, allow_cpu_helpers=False, background=None, gpu_clients=None):
    """Read process names, never arbitrary arguments or prompts.

    Standalone Python/temporary/project compute jobs are conservatively treated
    as possible GPU users. Desktop mode records Python and compiler work
    without an observed GPU client as background CPU work. This is not an exclusive lease.
    """
    output = subprocess.run(["ps", "-axo", "pid=,ppid=,pcpu=,rss=,comm="],
                            capture_output=True, text=True, check=True, timeout=5).stdout
    processes = {}
    for line in output.splitlines():
        fields = line.strip().split(None, 4)
        if len(fields) == 5:
            pid, parent, cpu, rss, name = fields
            processes[int(pid)] = (int(parent), float(cpu), int(rss), name)
    excluded = set()
    pid = os.getpid()
    while pid and pid not in excluded:
        excluded.add(pid)
        pid = processes.get(pid, (0,))[0]
    children = {child} if child else set()
    while True:
        more = {pid for pid, row in processes.items() if row[0] in children}
        if more <= children:
            break
        children |= more
    excluded |= children
    cpu_tools = {"clang", "clang++", "metal", "mtlcompilerservice", "ninja", "cmake", "cargo", "rustc",
                 "iree-compile", "iree-opt", "ld", "ld64.lld", "lld"}
    foreign_cpu = [pid for pid, row in processes.items() if pid not in excluded and
                   (Path(row[3]).name.lower().startswith("python") or Path(row[3]).name.lower() in cpu_tools)]
    if allow_cpu_helpers and gpu_clients is None and foreign_cpu:
        gpu_clients = set(gpu_activity()["gpu_client_pids"])
    ownership_known = bool(gpu_clients) and (child is None or bool(children & set(gpu_clients)))
    gpu_ancestors = set(gpu_clients or ())
    for owner in list(gpu_ancestors):
        parent = processes.get(owner, (0,))[0]
        while parent and parent not in gpu_ancestors:
            gpu_ancestors.add(parent)
            parent = processes.get(parent, (0,))[0]
    competing = []
    for pid, (parent, cpu, rss, path) in processes.items():
        if pid in excluded:
            continue
        name = Path(path).name.lower()
        project_binary = path.startswith(str(ROOT / "build") + "/")
        temporary_binary = path.startswith(("/private/tmp/", "/tmp/", "/private/var/folders/", "/var/folders/"))
        named_probe = any(word in name for word in ("myds4", "bench", "iree", "gpu_probe", "ane_probe"))
        python_compute = name.startswith("python") and (cpu >= 2 or rss >= 256 * 1024)
        python_gpu = name.startswith("python") and pid in gpu_ancestors
        details = {"pid": pid, "name": Path(path).name, "cpu_percent": cpu, "rss_mib": round(rss / 1024, 1),
                   "parent_pid": parent, "parent_name": Path(processes.get(parent, (0, 0, 0, "unknown"))[3]).name,
                   "gpu_client_in_process_tree": pid in gpu_ancestors if ownership_known else None}
        # The native app can keep an idle Metal UI open without loading the
        # engine. A loaded model exceeds this small host-memory allowance;
        # child inference processes are checked independently below.
        if path.endswith("/MyDS4.app/Contents/MacOS/MyDS4") and rss < 256 * 1024:
            if background is not None:
                background.append({**details, "idle_app_shell": True})
            continue
        if allow_cpu_helpers and ownership_known and pid not in gpu_ancestors and (python_compute or name in cpu_tools):
            if background is not None and (cpu >= 2 or rss >= 256 * 1024):
                background.append(details)
            python_compute = False
            if name in cpu_tools:
                continue
        compiler = preflight and cpu >= 2 and name in cpu_tools
        if project_binary or temporary_binary or named_probe or python_compute or python_gpu or compiler:
            competing.append(details)
    return competing


def memory_snapshot():
    output = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True, timeout=5).stdout
    page_size = int(re.search(r"page size of (\d+) bytes", output)[1])
    def pages(name):
        return int(re.search(r"^" + re.escape(name) + r":\s+(\d+)", output, re.M)[1])
    free = pages("Pages free")
    # vm_stat has already subtracted speculative pages from its printed free
    # count. Its file-backed count includes them, so add that count unchanged.
    file_cache = pages("File-backed pages")
    return {"free_gib": free * page_size / 1024**3,
            "file_cache_gib": file_cache * page_size / 1024**3,
            "available_gib": (free + file_cache) * page_size / 1024**3,
            "page_bytes": page_size, "compressions": pages("Compressions"),
            "decompressions": pages("Decompressions"),
            "compressor_gib": pages("Pages occupied by compressor") * page_size / 1024**3,
            "compressed_logical_gib": pages("Pages stored in compressor") * page_size / 1024**3,
            "swapouts": pages("Swapouts")}


def gpu_activity():
    raw = subprocess.run(["ioreg", "-r", "-c", "IOAccelerator", "-a", "-l"],
                         capture_output=True, check=True, timeout=5).stdout
    if len(raw) > 8 * 1024**2:
        raise GPUUnavailable("GPU registry response exceeded its diagnostic bound")
    entries = plistlib.loads(raw)
    stats = [e.get("PerformanceStatistics", {}) for e in entries]
    if not stats or not any("Device Utilization %" in s for s in stats):
        raise GPUUnavailable("GPU activity could not be read")
    device = max(float(s.get("Device Utilization %", 0)) for s in stats)
    renderer = max(float(s.get("Renderer Utilization %", 0)) for s in stats)
    resident = sum(int(s.get("In use system memory", 0)) for s in stats) / 1024**3
    clients = set()
    def owners(node):
        creator = node.get("IOUserClientCreator", "")
        match = re.search(r"\bpid\s+(\d+)\b", creator) if isinstance(creator, str) else None
        if match:
            clients.add(int(match[1]))
        for child in node.get("IORegistryEntryChildren", []):
            owners(child)
    for entry in entries:
        owners(entry)
    return {"device_percent": device, "renderer_percent": renderer, "gpu_resident_gib": round(resident, 2),
            "gpu_client_pids": sorted(clients)}


def gpu_snapshot(memory_gib, reserve_gib=10, desktop=False, quiet_cpu=False):
    activity = gpu_activity()
    background = []
    busy = process_snapshot(allow_cpu_helpers=desktop, background=background,
                            gpu_clients=set(activity["gpu_client_pids"]))
    device, renderer, resident = (activity[k] for k in ("device_percent", "renderer_percent", "gpu_resident_gib"))
    memory = memory_snapshot()
    power = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True, check=True, timeout=5).stdout
    on_ac = "AC Power" in power.splitlines()[0]
    reasons = []
    if busy:
        reasons.append("another compute process is present")
    if quiet_cpu and sum(p["cpu_percent"] for p in background) > 100:
        reasons.append("background CPU builds exceed one CPU core")
    # Check device load as well as process ownership and the shared model lock.
    if device > 35:
        reasons.append("GPU device activity exceeds 35 percent")
    if resident > 8:
        reasons.append("another workload retains more than 8 GiB of GPU memory")
    # Callers include the expected KV and process overhead in memory_gib.
    required_available = memory_gib + reserve_gib
    if memory["available_gib"] < required_available:
        reasons.append("insufficient available memory for the process and reserve")
    if not on_ac:
        reasons.append("large model tests require AC power")
    return {"device_percent": device, "renderer_percent": renderer, "gpu_resident_gib": round(resident, 2),
            "desktop_load": desktop,
            "reserve_gib": reserve_gib, "required_available_gib": required_available,
            "memory": memory, "ac_power": on_ac, "competing_processes": busy,
            "background_cpu_helpers": background, "reasons": reasons}


def wait_for_gpu(memory_gib, timeout, reserve_gib=10, desktop=False, quiet_cpu=False):
    until = time.monotonic() + timeout
    quiet = []
    announced = False
    while True:
        try:
            state = gpu_snapshot(memory_gib, reserve_gib, desktop, quiet_cpu)
        except (subprocess.SubprocessError, OSError, ValueError) as error:
            raise GPUUnavailable("GPU preflight inspection failed") from error
        if state["reasons"]:
            quiet.clear()
            if not announced:
                print("GPU_WAIT " + json.dumps(state), flush=True)
                announced = True
        else:
            quiet.append(state)
            if len(quiet) == 3:
                print("GPU_READY " + json.dumps({"free_gib": quiet[-1]["memory"]["free_gib"],
                      "available_gib": quiet[-1]["memory"]["available_gib"],
                      "required_available_gib": quiet[-1]["required_available_gib"], "device_percent": quiet[-1]["device_percent"],
                      "desktop_load": desktop, "samples": len(quiet)}), flush=True)
                return {"method": "process scan and three GPU/memory samples; not an exclusive reservation", "samples": quiet}
        if time.monotonic() >= until:
            raise GPUUnavailable("GPU did not become available within the bounded wait")
        time.sleep(5 if state["reasons"] else 2)


def guarded_process(command, env, timeout, baseline, observe_graphics=False, allow_cpu_helpers=False,
                    input_data=None, abort_cpu_contention=False):
    process = subprocess.Popen(command, env=env, stdin=subprocess.PIPE if input_data is not None else subprocess.DEVNULL,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print("GPU_PROCESS " + json.dumps({"runner_pid": os.getpid(), "model_pid": process.pid}), flush=True)
    stop = threading.Event()
    interference = []
    activity_samples = []
    helper_samples = []
    helper_peak = 0.0
    memory_samples = [baseline]
    started = time.monotonic()
    def monitor():
        nonlocal helper_peak
        next_activity = started
        next_progress = started + 60
        while not stop.wait(2):
            if process.poll() is not None:
                return
            try:
                background = []
                busy = process_snapshot(child=process.pid, preflight=False,
                                        allow_cpu_helpers=allow_cpu_helpers, background=background)
                memory = memory_snapshot()
                memory_samples.append(memory)
                cpu_total = sum(row["cpu_percent"] for row in background)
                helper_peak = max(helper_peak, cpu_total)
                if background and len(helper_samples) < 32:
                    helper_samples.append({"elapsed_seconds": round(time.monotonic() - started, 3), "processes": background})
                if observe_graphics and time.monotonic() >= next_activity:
                    activity = gpu_activity()
                    visible = process.pid in activity.pop("gpu_client_pids")
                    activity_samples.append({"elapsed_seconds": round(time.monotonic() - started, 3),
                                             "model_gpu_client_visible": visible, **activity})
                    next_activity = time.monotonic() + 10
                if observe_graphics and time.monotonic() >= next_progress:
                    print("GPU_PROGRESS " + json.dumps({"elapsed_seconds": round(time.monotonic() - started),
                          "free_gib": round(memory["free_gib"], 1), "background_cpu_percent": cpu_total}), flush=True)
                    next_progress = time.monotonic() + 60
                if process.poll() is not None:
                    return  # snapshots after the child ended are not interference with its run
                if (busy or memory["swapouts"] > baseline["swapouts"] or
                        (abort_cpu_contention and cpu_total > 100)):
                    interference.append({"competing_processes": busy, "memory": memory,
                                         "background_cpu_percent": cpu_total})
                else:
                    continue
            except (subprocess.SubprocessError, OSError, ValueError, GPUUnavailable) as error:
                interference.append({"inspection_error": type(error).__name__})
            try:
                process.send_signal(signal.SIGINT)
            except ProcessLookupError:
                pass
            return
    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()
    try:
        stdout, stderr = process.communicate(input=input_data, timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()
        raise RuntimeError("bounded predictor trial timed out") from None
    except BaseException:
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
            try:
                process.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
        raise
    finally:
        stop.set()
        watcher.join(timeout=6)
    final_memory = memory_snapshot()
    memory_samples.append(final_memory)
    if final_memory["swapouts"] > baseline["swapouts"]:
        interference.append({"swapping_at_completion": final_memory})
    if interference:
        print("GPU_RUN_DISCARDED " + json.dumps(interference), flush=True)
        raise GPUUnavailable("another workload appeared, swapping occurred or monitoring failed; this run is not a performance result")
    result = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    result.gpu_activity_samples = activity_samples
    result.background_cpu_helpers = helper_samples
    result.background_cpu_peak = helper_peak
    result.memory_activity = {
        "compression_policy": "allowed without a compression-specific limit; swap-outs invalidate the run",
        "scope": "system-wide counters, sampled about every two seconds; not per-process attribution",
        "before": baseline, "after": final_memory,
        "observed_peak_compressor_gib": max(m.get("compressor_gib", 0) for m in memory_samples),
        "observed_peak_compressed_logical_gib": max(m.get("compressed_logical_gib", 0) for m in memory_samples),
        "observed_min_available_gib": min(m["available_gib"] for m in memory_samples),
        "compressed_input_gib": (final_memory["compressions"] - baseline["compressions"]) * final_memory["page_bytes"] / 1024**3,
        "decompressed_output_gib": (final_memory["decompressions"] - baseline.get("decompressions", final_memory["decompressions"])) * final_memory["page_bytes"] / 1024**3,
        "sample_count": len(memory_samples),
    }
    return result
