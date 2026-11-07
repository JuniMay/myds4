"""Cold SSD read throughput for expert-sized reads, bypassing the file cache.

Reads COUNT whole experts of the checkpoint's MXFP4 files (the w1, w2 and w3
weights and scales of one expert: about 18.8 MB in 6 reads), chosen at random
from the given layers, with THREADS readers at once, each read with
F_NOCACHE as the engine reads experts. Prints the bytes, the seconds and the
throughput. Pick layers that nothing read recently, or the numbers include
the file cache. Uses no GPU; reading does not wear the SSD.

usage: python3 tools/analysis/ssd_bench.py CHECKPOINT [--threads N] [--count N] [--layers A B]
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import random
import struct
import time

F_NOCACHE = 48  # macOS fcntl


def expert_spans(checkpoint, layers):
    """{(layer, expert): [(path, offset, length), ...]} from the shard headers."""
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())
    wanted = {}
    for name, shard in index["weight_map"].items():
        parts = name.split(".")
        if len(parts) > 4 and parts[0] == "layers" and parts[2:4] == ["ffn", "experts"] and int(parts[1]) in layers:
            wanted.setdefault(shard, []).append(name)
    spans = {}
    for shard, names in wanted.items():
        path = checkpoint / shard
        with open(path, "rb") as f:
            (n,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(n))
        for name in names:
            start, end = header[name]["data_offsets"]
            parts = name.split(".")
            spans.setdefault((int(parts[1]), int(parts[4])), []).append((path, 8 + n + start, end - start))
    return spans


def read_expert(pieces, fds):
    total = 0
    for path, offset, length in pieces:
        total += len(os.pread(fds[path], length, offset))
    return total


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--count", type=int, default=400)
    ap.add_argument("--layers", type=int, nargs=2, default=[30, 39], help="first and last layer to draw experts from")
    args = ap.parse_args()
    spans = expert_spans(args.checkpoint, range(args.layers[0], args.layers[1] + 1))
    chosen = random.Random(1).sample(sorted(spans), min(args.count, len(spans)))
    fds = {}
    for pieces in spans.values():
        for path, _, _ in pieces:
            if path not in fds:
                fds[path] = os.open(path, os.O_RDONLY)
                fcntl.fcntl(fds[path], F_NOCACHE, 1)
    began = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.threads) as pool:
        total = sum(pool.map(lambda key: read_expert(spans[key], fds), chosen))
    seconds = time.monotonic() - began
    for fd in fds.values():
        os.close(fd)
    print(json.dumps({"ssd_bench": {"experts": len(chosen), "threads": args.threads, "gb": round(total / 1e9, 2),
                                    "seconds": round(seconds, 2), "gb_per_s": round(total / 1e9 / seconds, 2)}}))


if __name__ == "__main__":
    main()
