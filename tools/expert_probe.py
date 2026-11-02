"""Bounded in-memory expert-prediction captures and replay helpers.

The diagnostic executable writes a pipe, never a trace file. Capture timing
includes instrumentation and must not be reported as an inference speedup.
"""

from collections import deque
from dataclasses import dataclass, field
from types import SimpleNamespace
import fcntl
import os
from pathlib import Path
import struct
import subprocess
import threading
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HEADER = struct.Struct("<IIiiiiiid")
LAYERS, DIM, EXPERTS, TOPK = 40, 5120, 384, 6


@dataclass
class Pass:
    pos: int
    rows: int
    prefill: bool
    x: np.ndarray
    z: np.ndarray
    selected: np.ndarray
    tokens: np.ndarray
    times: np.ndarray
    accepted: int = 1
    draft: dict = field(default_factory=dict)
    early: dict = field(default_factory=dict)
    streams: dict = field(default_factory=dict)
    next_layer: int = 0
    original_pos: int | None = None
    original_rows: int | None = None
    future: dict = field(default_factory=dict)


def routing_view(passes, *, keep_draft=True):
    """Keep labels and scheduling data while dropping large main hidden arrays.

    Returned objects are routing-only views, not full Pass captures. They
    support cache replay and optional draft studies, but cannot be flattened
    for hidden-state training. Callers must also release separate flattened
    arrays and the original pass list to recover its memory.
    """
    return [
        SimpleNamespace(
            pos=p.pos,
            rows=p.rows,
            prefill=p.prefill,
            z=p.z,
            selected=p.selected,
            tokens=p.tokens,
            times=p.times,
            accepted=p.accepted,
            original_pos=getattr(p, "original_pos", None),
            original_rows=getattr(p, "original_rows", None),
            future=getattr(p, "future", {}),
            draft=p.draft if keep_draft else {},
            early={layer: (None, value[1], value[2]) for layer, value in p.early.items()},
        )
        for p in passes
    ]


def read_exact(stream, count, eof=False):
    parts, received = [], 0
    while received < count:
        value = stream.read(count - received)
        if not value:
            if not received and eof:
                return None
            raise EOFError(f"incomplete probe record: {received}/{count} bytes")
        parts.append(value)
        received += len(value)
    return b"".join(parts)


def records(stream):
    while (raw := read_exact(stream, HEADER.size, eof=True)) is not None:
        magic, kind, pos, layer, rows, dim, experts, ntokens, timestamp = HEADER.unpack(raw)
        if magic != 0x50524231 or not 1 <= kind <= 11:
            raise ValueError("unknown probe record")
        if not (
            0 <= rows <= 2048
            and 0 <= dim <= DIM
            and 0 <= experts <= EXPERTS
            and 0 <= ntokens <= 2048 * 7
        ):
            raise ValueError("probe dimensions exceed the diagnostic limits")
        feature_bytes, feature_type = (2, "<u2") if kind == 9 else (4, "<f4")
        x = np.frombuffer(read_exact(stream, rows * dim * feature_bytes), feature_type).reshape(
            rows, dim
        )
        z = np.frombuffer(read_exact(stream, rows * experts * 4), "<f4").reshape(rows, experts)
        tokens = np.frombuffer(read_exact(stream, ntokens * 4), "<i4")
        yield kind, pos, layer, rows, timestamp, x, z, tokens


def assert_model_idle(binary):
    path = Path(os.environ.get("MYDS4_LOCK", str(Path(binary).resolve().parent / "myds4.lock")))
    if not path.exists():
        return
    with path.open("rb") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                "another inference process holds the model lock; do not compete with it"
            ) from error
        fcntl.flock(stream, fcntl.LOCK_UN)


def capture(
    checkpoint,
    arguments,
    *,
    experts=None,
    cache_gb=30,
    timeout=300,
    max_rows=1024,
    binary=ROOT / "build" / "myds4-predict-probe",
    streams=False,
    retain_prefill_tail=None,
    retain_early_features=True,
):
    """Return passes and bounded diagnostics; do not write captures or output text.

    retain_prefill_tail keeps only the last N rows of each prefill pass, with
    matching positions, inputs, logits, native IDs and tokens. Original row
    counts remain in each Pass and metadata. Raw records are still validated
    in full and count against max_rows. Decode/verification passes stay whole.
    retain_early_features=False drops XP arrays but keeps native PC IDs and
    timestamps. Both defaults preserve the full capture.
    """
    if retain_prefill_tail is not None and (
        not isinstance(retain_prefill_tail, int) or not 1 <= retain_prefill_tail <= 2048
    ):
        raise ValueError("retain_prefill_tail must be 1..2048 or None")
    assert_model_idle(binary)
    read_fd, write_fd = os.pipe()
    env = dict(os.environ, MYDS4_PROBE_FD=str(write_fd))
    if streams:
        env["MYDS4_PROBE_STREAMS"] = "1"
    else:
        env.pop("MYDS4_PROBE_STREAMS", None)
    command = [str(binary), "-m", str(checkpoint), "--cache", str(cache_gb), "--ctx", "2048"]
    if experts:
        command += ["--experts", str(experts)]
    command += list(arguments)
    started = time.monotonic()
    process = subprocess.Popen(
        command, pass_fds=(write_fd,), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    os.close(write_fd)
    errors, output = deque(maxlen=80), bytearray()

    def drain_errors():
        for line in iter(process.stderr.readline, b""):
            errors.append(line.decode(errors="replace")[-8192:])

    def drain_output():
        for block in iter(lambda: process.stdout.read(4096), b""):
            if len(output) < 65536:
                output.extend(block[: 65536 - len(output)])

    readers = [
        threading.Thread(target=drain_errors, daemon=True),
        threading.Thread(target=drain_output, daemon=True),
    ]
    for reader in readers:
        reader.start()
    timed_out = threading.Event()

    def expire():
        timed_out.set()
        process.kill()

    timer = threading.Timer(timeout, expire)
    timer.start()
    passes, current, draft, total_rows, cache_metadata = [], None, {}, 0, {}
    raw_pos, raw_rows, raw_tokens, retained_start = None, 0, None, 0
    original_prefill_rows = 0
    early, early_pos = {}, None
    stream_context, stream_pos = {}, None
    try:
        with os.fdopen(read_fd, "rb", buffering=0) as stream:
            for kind, pos, layer, rows, stamp, x, z, ids in records(stream):
                if kind == 9:
                    if rows % 4 or x.shape[1] != DIM or z.shape[1] != 6:
                        raise ValueError("invalid four-stream snapshot")
                    context = (
                        x.reshape(rows // 4, 4, DIM).copy(),
                        z.reshape(rows // 4, 24).copy(),
                        stamp,
                    )
                    if layer >= LAYERS:
                        if draft.get("pos") != pos:
                            raise ValueError("draft streams have no seed")
                        draft["streams"][layer - LAYERS] = context
                    else:
                        if layer == 0:
                            stream_context, stream_pos = {}, pos
                        if stream_pos != pos:
                            raise ValueError("stream snapshot has no matching pass")
                        stream_context[layer] = context
                elif kind == 10:
                    if draft.get("pos") != pos or x.shape[1] != DIM:
                        raise ValueError("draft head has no matching seed")
                    draft["head"] = (x.copy(), stamp)
                elif kind == 11:
                    hops = min(2, LAYERS - 1 - layer)
                    if (
                        current is None
                        or current.prefill
                        or current.pos != pos
                        or current.rows != rows
                        or not 0 <= layer < LAYERS - 1
                        or current.next_layer != layer + 1
                        or layer in current.future
                        or x.shape != (rows, 0)
                        or z.shape != (rows, 0)
                        or len(ids) != hops * rows * TOPK
                    ):
                        raise ValueError("native future predictions have no matching source route")
                    forecasts = {}
                    for hop, values in enumerate(ids.reshape(hops, rows, TOPK), 1):
                        valid = TOPK if hop == 1 or np.all(values[:, 3:] >= 0) else 3
                        if (
                            np.any((values[:, :valid] < 0) | (values[:, :valid] >= EXPERTS))
                            or np.any(values[:, valid:] != -1)
                            or np.any(np.diff(np.sort(values[:, :valid], axis=1), axis=1) == 0)
                        ):
                            raise ValueError("invalid native future expert IDs or padding")
                        forecasts[hop] = (values[:, :valid].copy(), stamp)
                    current.future[layer] = forecasts
                elif kind == 7:
                    if x.shape != (LAYERS, EXPERTS) or len(ids) != 3:
                        raise ValueError("invalid cache metadata")
                    cache_metadata = {
                        "expert_read_bytes": x.copy(),
                        "cache_capacity": int(ids[2]),
                        "initial_cache_capacity": int(ids[0]),
                        "cache_slot_bytes": int(ids[1]),
                    }
                elif kind == 8:
                    if layer == 0:
                        early, early_pos = {}, pos
                    if pos != early_pos or x.shape != (rows, DIM) or len(ids) != rows * TOPK:
                        raise ValueError("invalid early residual prediction")
                    early[layer] = (
                        x.copy() if retain_early_features else None,
                        ids.reshape(rows, TOPK).copy(),
                        stamp,
                    )
                elif kind in (1, 2):
                    if (
                        x.shape != (rows, DIM)
                        or z.shape != (rows, EXPERTS)
                        or len(ids) != rows * (1 + TOPK)
                    ):
                        raise ValueError("target payload shape differs from the model")
                    if layer == 0:
                        if current is not None:
                            raise ValueError("new pass before the previous pass completed")
                        total_rows += rows
                        if total_rows > max_rows:
                            raise RuntimeError("capture exceeded its row budget")
                        raw_pos, raw_rows, raw_tokens = pos, rows, ids[:rows].copy()
                        kept = (
                            min(rows, retain_prefill_tail)
                            if kind == 2 and retain_prefill_tail is not None
                            else rows
                        )
                        retained_start = rows - kept
                        original_prefill_rows += rows if kind == 2 else 0
                        current = Pass(
                            pos + retained_start,
                            kept,
                            kind == 2,
                            np.empty((LAYERS, kept, DIM), np.float32),
                            np.empty((LAYERS, kept, EXPERTS), np.float32),
                            np.empty((LAYERS, kept, TOPK), np.int32),
                            raw_tokens[retained_start:].copy(),
                            np.empty(LAYERS),
                            accepted=kept if kind == 2 else 1,
                            draft=draft if kind == 1 and draft.get("pos") == pos else {},
                            early=early if kind == 1 and early_pos == pos else {},
                            streams=stream_context if stream_pos == pos else {},
                            original_pos=pos,
                            original_rows=rows,
                        )
                        if kind == 1:
                            draft = {}
                    if (
                        current is None
                        or raw_pos != pos
                        or raw_rows != rows
                        or not 0 <= layer < LAYERS
                        or current.prefill != (kind == 2)
                    ):
                        raise ValueError("target records are not a complete ordered pass")
                    if layer != current.next_layer or not np.array_equal(ids[:rows], raw_tokens):
                        raise ValueError(
                            "target layer order or input token IDs changed inside a pass"
                        )
                    current.x[layer], current.z[layer] = x[retained_start:], z[retained_start:]
                    current.selected[layer] = ids[rows:].reshape(rows, TOPK)[retained_start:]
                    current.times[layer] = stamp
                    current.next_layer += 1
                    if layer == LAYERS - 1:
                        if retained_start:
                            for source, (state, mixes, available) in current.streams.items():
                                if len(state) != raw_rows or len(mixes) != raw_rows:
                                    raise ValueError(
                                        "prefill stream rows do not match their original pass"
                                    )
                                current.streams[source] = (
                                    state[retained_start:].copy(),
                                    mixes[retained_start:].copy(),
                                    available,
                                )
                        passes.append(current)
                        current = None
                elif kind == 4:
                    draft = {
                        "pos": pos,
                        "seed": x.copy(),
                        "seed_time": stamp,
                        "stages": {},
                        "streams": {},
                        "stage_selected": {},
                    }
                elif kind == 3:
                    if draft.get("pos") != pos:
                        raise ValueError("draft stage has no matching seed")
                    draft["stages"][layer] = (x.copy(), z.copy(), stamp)
                    if len(ids):
                        if len(ids) != rows * 3:
                            raise ValueError("draft native labels have wrong shape")
                        draft["stage_selected"][layer] = ids.reshape(rows, 3).copy()
                elif kind == 5:
                    if draft.get("pos") != pos:
                        raise ValueError("draft proposal has no matching seed")
                    draft.update(tokens=ids.copy(), proposal_time=stamp)
                elif kind == 6:
                    if not passes or passes[-1].pos != pos or rows > passes[-1].rows:
                        raise ValueError("acceptance has no matching verification pass")
                    passes[-1].accepted = rows
        code = process.wait(timeout=10)
        for reader in readers:
            reader.join(timeout=2)
        if timed_out.is_set():
            raise TimeoutError("capture exceeded its time budget")
        if code or current is not None:
            raise RuntimeError(f"capture failed ({code}): {''.join(errors)[-3000:]}")
    finally:
        timer.cancel()
        if process.poll() is None:
            process.kill()
            process.wait()
    return passes, {
        "wall_seconds": time.monotonic() - started,
        "rows": total_rows,
        "retained_rows": sum(p.rows for p in passes),
        "original_prefill_rows": original_prefill_rows,
        "retained_prefill_rows": sum(p.rows for p in passes if p.prefill),
        "retain_prefill_tail": retain_prefill_tail,
        "retain_early_features": bool(retain_early_features),
        "passes": len(passes),
        "stderr": "".join(errors),
        "output": bytes(output),
        "note": "instrumented capture, not an inference benchmark",
        **cache_metadata,
    }


def flatten(passes):
    """Keep verification rows in one causal predictor-update group."""
    x = np.concatenate([p.x.transpose(1, 0, 2) for p in passes])
    z = np.concatenate([p.z.transpose(1, 0, 2) for p in passes])
    selected = np.concatenate([p.selected.transpose(1, 0, 2) for p in passes])
    groups = np.concatenate([np.full(p.rows, i, np.int32) for i, p in enumerate(passes)])
    warmup = sum(p.rows for p in passes if p.prefill)
    return x, z, selected, groups, warmup


def load_routers(checkpoint):
    import json
    from safetensors import safe_open

    checkpoint = Path(checkpoint)
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    weights = np.empty((LAYERS, EXPERTS, DIM), np.float32)
    bias = np.empty((LAYERS, EXPERTS), np.float32)
    for layer in range(LAYERS):
        for suffix, destination in (("weight", weights[layer]), ("bias", bias[layer])):
            key = f"layers.{layer}.ffn.gate.{suffix}"
            with safe_open(str(checkpoint / index[key]), framework="pt", device="cpu") as shard:
                destination[:] = shard.get_tensor(key).float().numpy()
    return weights, bias
