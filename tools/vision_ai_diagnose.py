"""Compare a real image's backbone and Aligner outputs across precisions.

This reports numerical differences, not task accuracy. It uses the
checkpoint's image preprocessing and does not load the language model.
"""

import argparse
import importlib.util
import io
import json
import sys
import time
import types
from pathlib import Path

import torch
from torch.nn import functional as F

from vision_ai import metrics, rope
from vision_ai_validate import ROOT, Runtime, reference


def module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def summary(want, got):
    result = metrics(want, got)
    similarity = F.cosine_similarity(want.float(), got.float(), dim=-1)
    signal = want.double().square().sum(-1)
    error = (want.double() - got.double()).square().sum(-1)
    row_snr = 10 * torch.log10(signal / error.clamp_min(1e-300))
    result.update(
        cosine_mean=similarity.mean().item(),
        cosine_min=similarity.min().item(),
        row_snr_median=row_snr.median().item(),
        row_snr_min=row_snr.min().item(),
        rows_below_cosine_099=int((similarity < 0.99).sum()),
        rows=len(similarity),
    )
    return result


def aligner(checkpoint, args, vision):
    from safetensors import safe_open

    old = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        model = vision.Aligner(args).eval()
    finally:
        torch.set_default_dtype(old)
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    with torch.no_grad():
        for name, param in model.named_parameters():
            key = "aligner." + name
            with safe_open(str(checkpoint / index[key]), framework="pt", device="cpu") as shard:
                param.copy_(shard.get_tensor(key))
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("image", type=Path)
    parser.add_argument("--asset", type=Path)
    parser.add_argument("--library", type=Path, default=ROOT / "build" / "libmyds4_vision_ai.dylib")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        help="fit the source image to this size in memory before normal preprocessing",
    )
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    config = json.loads((args.checkpoint / "inference" / "config.json").read_text())
    cfg = types.SimpleNamespace(
        dim=config["dim"], **{k: v for k, v in config.items() if k.startswith("vision_")}
    )
    processor = module_at(
        "diagnostic_image_processor", args.checkpoint / "inference" / "image_processor.py"
    )
    vision = module_at("diagnostic_vision_reference", args.checkpoint / "inference" / "vision.py")
    record = {"url": str(args.image)}
    if args.image_size:
        if min(args.image_size) < 1:
            parser.error("image dimensions must be positive")
        from PIL import Image, ImageOps

        with Image.open(args.image) as image:
            fitted = ImageOps.fit(image.convert("RGB"), tuple(args.image_size))
        data = io.BytesIO()
        fitted.save(data, format="PNG")
        record = {"data": data.getvalue()}
    patches, height, width, th, tw = processor.load_image(record, cfg)
    count = height * width
    print(json.dumps({"patch_grid": [height, width], "feature_grid": [th, tw]}), flush=True)
    model = reference(args.checkpoint, 32)
    projector = aligner(args.checkpoint, cfg, vision)
    with torch.inference_mode():
        started = time.perf_counter()
        want = model(patches, height, width)
        want_aligned = projector(want, height, width).float()
        print(json.dumps({"reference_bf16_seconds": time.perf_counter() - started}), flush=True)
        if args.asset:
            inputs = torch.zeros(count, 640)
            inputs[:, :588] = patches.flatten(1).float()
            cos, sin = rope(height, width)
            rotary = torch.cat((cos, sin), dim=1).reshape(64, count).T.numpy()
            runtime = Runtime(args.library.resolve(), args.asset.resolve(), count)
            try:
                raw = runtime.run(inputs.numpy(), rotary)
                repeated = runtime.run(inputs.numpy(), rotary)
                print(
                    json.dumps({"repeat_max_abs_error": (raw - repeated).abs().max().item()}),
                    flush=True,
                )
                got = raw.bfloat16()
            finally:
                runtime.close()
            got_aligned = projector(got, height, width).float()
            print(
                json.dumps(
                    {
                        "coreai_vs_bf16_backbone": summary(want, got),
                        "coreai_vs_bf16_aligned": summary(want_aligned, got_aligned),
                    }
                ),
                flush=True,
            )
        model.float()
        started = time.perf_counter()
        fp32 = model(patches.float(), height, width)
        fp32_aligned_same = projector(fp32.bfloat16(), height, width).float()
        print(
            json.dumps(
                {
                    "reference_fp32_seconds": time.perf_counter() - started,
                    "fp32_vs_bf16_backbone": summary(want, fp32),
                    "fp32_backbone_same_bf16_aligner": summary(want_aligned, fp32_aligned_same),
                }
            ),
            flush=True,
        )
        if args.asset:
            print(
                json.dumps(
                    {
                        "coreai_vs_fp32_backbone": summary(fp32, got),
                        "coreai_vs_fp32_same_bf16_aligner": summary(fp32_aligned_same, got_aligned),
                    }
                ),
                flush=True,
            )
        projector.float()
        fp32_aligned = projector(fp32, height, width)
        print(
            json.dumps({"full_fp32_vs_full_bf16_aligned": summary(want_aligned, fp32_aligned)}),
            flush=True,
        )


if __name__ == "__main__":
    main()
