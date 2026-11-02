"""Check a Core AI vision asset without loading the language model.

Uses checkpoint ViT weights and its CPU BF16 reference. Inputs are prepared
patch tensors, so no image or intermediate activation files are written.
"""

import argparse
import ctypes
import importlib.util
import json
import time
import types
from pathlib import Path

import numpy as np
import torch

from vision_ai import metrics, rope

ROOT = Path(__file__).resolve().parents[1]


def reference(checkpoint, layers):
    from safetensors import safe_open

    path = checkpoint / "inference" / "vision.py"
    spec = importlib.util.spec_from_file_location("checkpoint_vision", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg = json.loads((checkpoint / "inference" / "config.json").read_text())
    args = types.SimpleNamespace(**{k: v for k, v in cfg.items() if k.startswith("vision_")})
    args.vision_n_layers = layers
    old = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        model = module.ViT(args).eval()
    finally:
        torch.set_default_dtype(old)
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    with torch.no_grad():
        for name, param in model.named_parameters():
            key = "vision." + name
            with safe_open(str(checkpoint / index[key]), framework="pt", device="cpu") as shard:
                param.copy_(shard.get_tensor(key))
    return model


class Runtime:
    def __init__(self, library, asset, count):
        self.library = ctypes.CDLL(str(library))
        self.error = ctypes.create_string_buffer(2048)
        self.library.vision_ai_create.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_int32,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        self.library.vision_ai_create.restype = ctypes.c_void_p
        fp = ctypes.POINTER(ctypes.c_float)
        self.library.vision_ai_run.argtypes = [
            ctypes.c_void_p,
            fp,
            fp,
            fp,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        self.library.vision_ai_run.restype = ctypes.c_int32
        self.library.vision_ai_destroy.argtypes = [ctypes.c_void_p]
        self.library.vision_ai_destroy.restype = None
        self.handle = self.library.vision_ai_create(
            bytes(asset), b"vision", count, self.error, len(self.error)
        )
        if not self.handle:
            raise RuntimeError(self.error.value.decode(errors="replace"))

    def run(self, patches, rotary):
        patches = np.ascontiguousarray(patches, dtype=np.float32)
        rotary = np.ascontiguousarray(rotary, dtype=np.float32)
        result = np.empty((len(patches), 1024), dtype=np.float32)
        fp = ctypes.POINTER(ctypes.c_float)
        ok = self.library.vision_ai_run(
            self.handle,
            patches.ctypes.data_as(fp),
            rotary.ctypes.data_as(fp),
            result.ctypes.data_as(fp),
            self.error,
            len(self.error),
        )
        if not ok:
            raise RuntimeError(self.error.value.decode(errors="replace"))
        return torch.from_numpy(result)

    def close(self):
        if self.handle:
            self.library.vision_ai_destroy(self.handle)
            self.handle = None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("asset", type=Path)
    p.add_argument("--height", type=int, required=True)
    p.add_argument("--width", type=int, required=True)
    p.add_argument("--layers", type=int, default=32)
    p.add_argument("--library", type=Path, default=ROOT / "build" / "libmyds4_vision_ai.dylib")
    p.add_argument(
        "--min-snr",
        type=float,
        default=30.0,
        help="provisional backbone SNR threshold; the original vision check measures after the Aligner",
    )
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--threads", type=int, default=2)
    args = p.parse_args()
    if min(args.height, args.width, args.threads, args.repeats) < 1 or not 1 <= args.layers <= 32:
        p.error("dimensions, threads and repeats must be positive; layers must be 1..32")
    torch.set_num_threads(args.threads)
    torch.manual_seed(71)
    count = args.height * args.width
    cosine, sine = rope(args.height, args.width)
    rotary = torch.cat((cosine, sine), dim=1).reshape(64, count).T.numpy()
    started = time.perf_counter()
    runtime = Runtime(args.library.resolve(), args.asset.resolve(), count)
    print(
        f"Core AI load: {time.perf_counter() - started:.3f} s; ANE preference, placement not verified",
        flush=True,
    )
    try:
        model = reference(args.checkpoint, args.layers)
        passed = True
        for kind in ("random_rgb", "gradient_rgb"):
            patches = torch.zeros(count, 640)
            if kind == "random_rgb":
                pixels = torch.randint(0, 256, (count, 588)).float()
            else:
                pixels = torch.linspace(0, 255, count * 588).round().reshape(count, 588)
            # image_open maps eight-bit RGB values to [-1, 1], then BF16.
            patches[:, :588] = (pixels / 255.0 - 0.5) / 0.5
            patches = patches.bfloat16().float()
            with torch.inference_mode():
                want = model(
                    patches[:, :588].reshape(count, 3, 14, 14).bfloat16(), args.height, args.width
                ).float()
            started = time.perf_counter()
            got = runtime.run(patches.numpy(), rotary).bfloat16().float()
            first = time.perf_counter() - started
            result = metrics(want, got)
            passed &= result["finite"] and result["snr_db"] >= args.min_snr
            times = []
            for _ in range(args.repeats):
                started = time.perf_counter()
                runtime.run(patches.numpy(), rotary)
                times.append(time.perf_counter() - started)
            print(
                json.dumps({"case": kind, **result, "first_seconds": first, "warm_seconds": times}),
                flush=True,
            )
        if not passed:
            raise SystemExit(
                "Backbone diagnostic threshold not met; assess real-image post-Aligner agreement before deployment"
            )
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
