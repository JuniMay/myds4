"""Bytes a decoded token reads, from the model files' headers alone.

Reads the checkpoint's safetensors headers and, optionally, GGUF expert files
(any shard of a set). Prints, as JSON: the checkpoint's dense tensors by group
(what one decoded token reads once, whatever the batch: attention, shared
experts, routers, norms; the output head; the embedding table, of which a
token reads one row; DSpark's stages, read only with --draft), and for each
expert source the bytes of one expert per layer. Loads no weights and uses
no GPU.

usage: python3 tools/analysis/model_bytes.py CHECKPOINT [GGUF ...]
"""
import json
from pathlib import Path
import re
import struct
import sys

# ggml type: (elements per block, bytes per block)
GGML = {0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20), 6: (32, 22), 7: (32, 24), 8: (32, 34), 9: (32, 36),
        10: (256, 84), 11: (256, 110), 12: (256, 144), 13: (256, 176), 14: (256, 210), 15: (256, 292),
        16: (256, 66), 17: (256, 74), 18: (256, 98), 19: (256, 50), 20: (32, 18), 21: (256, 110),
        22: (256, 82), 23: (256, 136), 29: (256, 56), 30: (1, 2), 39: (32, 17)}
NAMES = {10: "q2_k", 11: "q3_k", 12: "q4_k", 13: "q5_k", 14: "q6_k", 16: "iq2_xxs", 17: "iq2_xs", 18: "iq3_xxs",
         19: "iq1_s", 21: "iq3_s", 22: "iq2_s", 23: "iq4_xs", 29: "iq1_m", 39: "mxfp4", 0: "f32", 1: "f16", 30: "bf16"}


def safetensors_tensors(checkpoint):
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())
    tensors = {}
    for shard in sorted(set(index["weight_map"].values())):
        with open(checkpoint / shard, "rb") as f:
            (n,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(n))
        for name, info in header.items():
            if name != "__metadata__":
                start, end = info["data_offsets"]
                tensors[name] = end - start
    return tensors


def group(name):
    if ".ffn.experts." in name:
        return "experts"
    if name.startswith("mtp"):
        return "dspark"
    if "embed" in name:
        return "embedding"
    if name.startswith("head"):
        return "head"
    return "dense"


def gguf_tensors(path):
    """(name, type, elements) of every tensor in all shards of a GGUF set."""
    m = re.match(r"(.*-)(\d{5})-of-(\d{5})\.gguf$", path.name)
    shards = [path] if not m else [path.with_name(f"{m[1]}{i:05d}-of-{m[3]}.gguf") for i in range(1, int(m[3]) + 1)]
    out = []
    for shard in shards:
        if not shard.exists():  # a set may hold only the shards with experts
            continue
        with open(shard, "rb") as f:
            def u(fmt):
                return struct.unpack("<" + fmt, f.read(struct.calcsize(fmt)))[0]

            def string():
                return f.read(u("Q")).decode()

            def value(t):
                sizes = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
                if t == 8:
                    return string()
                if t == 9:
                    et, count = u("I"), u("Q")
                    for _ in range(count):
                        value(et)
                    return None
                return u(sizes[t])

            if f.read(4) != b"GGUF":
                raise ValueError(f"{shard} is not a GGUF file")
            u("I")
            tensors, kvs = u("Q"), u("Q")
            for _ in range(kvs):
                string()
                value(u("I"))
            for _ in range(tensors):
                name = string()
                dims = [u("Q") for _ in range(u("I"))]
                t = u("I")
                u("Q")
                elements = 1
                for d in dims:
                    elements *= d
                out.append((name, t, elements))
    return out


def gguf_experts(path):
    per_layer = {}
    head = None
    for name, t, elements in gguf_tensors(path):
        block, size = GGML[t]
        nbytes = elements // block * size
        m = re.match(r"blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$", name)
        if m:
            layer = per_layer.setdefault(int(m[1]), {"types": {}, "expert_bytes": 0})
            layer["types"][m[2]] = NAMES.get(t, str(t))
            layer["expert_bytes"] += nbytes // 384
        elif name == "output.weight":
            head = {"type": NAMES.get(t, str(t)), "bytes": nbytes}
    return per_layer, head


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    checkpoint = Path(sys.argv[1])
    groups, experts = {}, {}
    for name, nbytes in safetensors_tensors(checkpoint).items():
        g = group(name)
        groups[g] = groups.get(g, 0) + nbytes
        m = re.match(r"layers\.(\d+)\.ffn\.experts\.\d+\.", name)
        if m:
            experts[int(m[1])] = experts.get(int(m[1]), 0) + nbytes
    layers = sorted(experts)
    sources = {"checkpoint (MXFP4)": {"per_layer_expert_bytes": [experts[l] // 384 for l in layers]}}
    for arg in sys.argv[2:]:
        per_layer, head = gguf_experts(Path(arg))
        sources[Path(arg).name] = {"per_layer_expert_bytes": [per_layer[l]["expert_bytes"] for l in sorted(per_layer)],
                                   "per_layer_types": [per_layer[l]["types"] for l in sorted(per_layer)],
                                   "head": head}
    for s in sources.values():
        b = s["per_layer_expert_bytes"]
        s["expert_bytes_mean"] = sum(b) / len(b)
        s["slot_bytes_at_least"] = max(b)
        s["all_experts_gib"] = sum(b) * 384 / 2**30
        s["six_experts_all_layers_gb"] = 6 * sum(b) / 1e9
    print(json.dumps({"checkpoint_groups_gb": {g: round(n / 1e9, 3) for g, n in sorted(groups.items())},
                      "layers": len(layers), "sources": sources}, indent=1))


if __name__ == "__main__":
    main()
