"""Export the vision backbone through Apple's public Core AI tools.

The graph takes prepared patches and 2D rotary tables. Image decoding, the
3 x 3 patch merge and the Aligner remain in C. Shapes are fixed for each
export; no artificial patches enter attention.

The experimental reference-rounding variant requests BF16 boundaries around
FP16 projections. The current converter removes some of these round-trips,
so it is not a faithful BF16 emulation. Its distinct asset name prevents
the runtime from selecting it as a validated backbone.
"""

import argparse
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


def bf16_half(x):
    """Keep a checkpoint BF16 boundary between FP16 projections."""
    return x.to(torch.bfloat16).to(torch.float16)


class Projection(nn.Module):
    def __init__(self, inputs, outputs, bias=False, reference_rounding=False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(outputs, inputs, 1, 1))
        self.bias = nn.Parameter(torch.empty(outputs)) if bias else None
        self.reference_rounding = reference_rounding

    def forward(self, x):
        y = F.conv2d(x, self.weight, self.bias)
        return bf16_half(y) if self.reference_rounding else y


class Norm(nn.Module):
    def __init__(self, dim, reference_rounding=False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.dim = dim
        self.reference_rounding = reference_rounding
        self.export_impl = None

    def forward(self, x):
        if self.reference_rounding:
            x = x.float()
            y = x * torch.rsqrt(x.square().mean(1, keepdim=True) + 1e-6)
            return bf16_half(y * self.weight.float().view(1, self.dim, 1, 1))
        if self.export_impl is not None:
            return self.export_impl(x.transpose(1, 3), self.weight).transpose(1, 3)
        y = F.rms_norm(x.transpose(1, 3), (self.dim,), self.weight, 1e-6)
        return y.transpose(1, 3)


def rotary(x, cosine, sine, heads, reference_rounding=False):
    if reference_rounding:
        x, cosine, sine = x.float(), cosine.float(), sine.float()
    dim = x.shape[1] // heads
    parts = []
    for h in range(heads):
        a = x[:, h * dim : h * dim + dim // 2]
        b = x[:, h * dim + dim // 2 : (h + 1) * dim]
        parts.extend((a * cosine - b * sine, b * cosine + a * sine))
    y = torch.cat(parts, dim=1)
    return bf16_half(y) if reference_rounding else y


def attention(q, k, v, heads, chunk, reference_rounding=False):
    """Tile queries only: every query still attends to every real key."""
    if reference_rounding:
        q, k, v = q.float(), k.float(), v.float()
    dim, count = q.shape[1] // heads, q.shape[-1]
    outputs = []
    for h in range(heads):
        start, end = h * dim, (h + 1) * dim
        keys = k[:, start:end].permute(0, 2, 3, 1) * (dim**-0.5)
        values = v[:, start:end].permute(0, 2, 1, 3)
        pieces = []
        for first in range(0, count, chunk):
            query = q[:, start:end, :, first : first + chunk].permute(0, 2, 1, 3)
            scores = keys @ query
            # Keys occupy the channel axis for ANE softmax.
            probs = scores.transpose(1, 2).softmax(dim=1).transpose(1, 2)
            pieces.append((values @ probs).transpose(1, 2))
        outputs.append(torch.cat(pieces, dim=-1))
    y = torch.cat(outputs, dim=1)
    return bf16_half(y) if reference_rounding else y


class Block(nn.Module):
    def __init__(self, dim=1024, heads=16, intermediate=2816, chunk=256, reference_rounding=False):
        super().__init__()
        self.heads, self.chunk = heads, chunk
        self.reference_rounding = reference_rounding
        self.norm1, self.norm2 = Norm(dim, reference_rounding), Norm(dim, reference_rounding)
        self.qkv = Projection(dim, 3 * dim, True, reference_rounding)
        self.out = Projection(dim, dim, True, reference_rounding)
        self.up = Projection(dim, 2 * intermediate, reference_rounding=reference_rounding)
        self.down = Projection(intermediate, dim, reference_rounding=reference_rounding)

    def forward(self, x, cosine, sine):
        q, k, v = self.qkv(self.norm1(x)).chunk(3, dim=1)
        q = rotary(q, cosine, sine, self.heads, self.reference_rounding)
        k = rotary(k, cosine, sine, self.heads, self.reference_rounding)
        y = self.out(attention(q, k, v, self.heads, self.chunk, self.reference_rounding))
        x = bf16_half(x.float() + y.float()) if self.reference_rounding else x + y
        gate, up = self.up(self.norm2(x)).chunk(2, dim=1)
        if self.reference_rounding:
            activated = F.silu(gate.float()).to(torch.bfloat16).float()
            y = self.down(bf16_half(activated * up.float()))
            return bf16_half(x.float() + y.float())
        return x + self.down((gate * torch.sigmoid(gate)) * up)


class Backbone(nn.Module):
    def __init__(
        self,
        layers=32,
        chunk=256,
        dim=1024,
        heads=16,
        intermediate=2816,
        inputs=640,
        reference_rounding=False,
    ):
        super().__init__()
        self.patch = Projection(inputs, dim, True, reference_rounding)
        self.blocks = nn.ModuleList(
            Block(dim, heads, intermediate, chunk, reference_rounding) for _ in range(layers)
        )
        self.norm = Norm(dim, reference_rounding)

    def forward(self, patches, cosine, sine):
        x = self.patch(patches)
        for block in self.blocks:
            x = block(x, cosine, sine)
        return self.norm(x)


def load_backbone(checkpoint, layers, chunk, dtype=torch.float16, reference_rounding=False):
    from safetensors import safe_open

    checkpoint = Path(checkpoint)
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    if reference_rounding and dtype != torch.float16:
        raise ValueError("reference rounding requires FP16 projection weights")
    model = Backbone(layers, chunk, reference_rounding=reference_rounding).to(dtype=dtype).eval()
    if reference_rounding:
        for module in model.modules():
            if isinstance(module, Norm):
                module.float()
    names = {
        "patch.weight": "patch_embed.proj.weight",
        "patch.bias": "patch_embed.proj.bias",
        "norm.weight": "norm.weight",
    }
    for layer in range(layers):
        for target, source in (
            ("norm1", "norm1"),
            ("norm2", "norm2"),
            ("qkv", "attn.wqkv"),
            ("out", "attn.wo"),
            ("up", "mlp.w1"),
            ("down", "mlp.w2"),
        ):
            names[f"blocks.{layer}.{target}.weight"] = f"blocks.{layer}.{source}.weight"
        for target, source in (("qkv", "attn.wqkv"), ("out", "attn.wo")):
            names[f"blocks.{layer}.{target}.bias"] = f"blocks.{layer}.{source}.bias"
    with torch.no_grad():
        for target, param in model.named_parameters():
            key = "vision." + names[target]
            with safe_open(str(checkpoint / index[key]), framework="pt", device="cpu") as shard:
                value = shard.get_tensor(key).to(dtype=param.dtype)
            if not torch.isfinite(value).all():
                raise ValueError(f"{key} cannot be represented as {param.dtype}")
            if target == "patch.weight":
                param.zero_()
                param[:, :588, 0, 0].copy_(value.reshape(1024, 588))
            else:
                param.copy_(value.reshape(param.shape))
    return model


def rope(height, width, head_dim=64):
    inv = 1.0 / 10000 ** (torch.arange(0, head_dim // 2, 2).float() / (head_dim // 2))
    rows = torch.arange(height).repeat_interleave(width)
    cols = torch.arange(width).repeat(height)
    angles = torch.cat((rows[:, None] * inv, cols[:, None] * inv), dim=1)
    return tuple(v.T[None, :, None, :] for v in (angles.cos(), angles.sin()))


def metrics(want, got):
    want, got = want.double(), got.double()
    error = (want - got).square().sum().item()
    signal = want.square().sum().item()
    return {
        "snr_db": math.inf if error == 0 else 10 * math.log10(signal / error),
        "max_abs_error": (want - got).abs().max().item(),
        "finite": bool(torch.isfinite(got).all()),
    }


def export(args):
    from coreai_torch import ExternalizeSpec, TorchConverter, get_decomp_table
    from coreai_torch.composite_ops import RMSNormImpl

    reference_rounding = getattr(args, "reference_rounding", False)
    model = load_backbone(
        args.checkpoint, args.layers, args.chunk, reference_rounding=reference_rounding
    )
    externalized = []
    if not reference_rounding:
        for module in model.modules():
            if isinstance(module, Norm):
                module.export_impl = RMSNormImpl(eps=1e-6)
        externalized = [
            ExternalizeSpec(
                target_class=RMSNormImpl,
                composite_op_name="rms_norm",
                composite_attrs=["axes", "eps"],
            )
        ]
    n = args.height * args.width
    inputs = (
        torch.zeros(1, 640, 1, n, dtype=torch.float16),
        *(v if reference_rounding else v.half() for v in rope(args.height, args.width)),
    )
    converter = TorchConverter().add_pytorch_module(
        model,
        export_fn=lambda m: torch.export.export(m, inputs).run_decompositions(get_decomp_table()),
        externalize_modules=externalized,
        input_names=["patches", "cos", "sin"],
        output_names=["features"],
        entrypoint_name="vision",
    )
    program = converter.to_coreai()
    prefix = "vit-bf16" if reference_rounding else "vit"
    path = Path(args.output) / f"{prefix}-{args.layers}-{args.height}x{args.width}.aimodel"
    path.parent.mkdir(parents=True, exist_ok=True)
    # Core AI's serialization API is kept here so runtime code has no
    # dependency on Python, PyTorch or the export packages.
    program.save_asset(path)
    print(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--height", type=int, required=True, help="patch-grid height, not image pixels")
    p.add_argument("--width", type=int, required=True, help="patch-grid width")
    p.add_argument(
        "--layers",
        type=int,
        default=32,
        help="1 for a probe; only 32-layer exports are used by myds4",
    )
    p.add_argument(
        "--chunk", type=int, default=256, help="queries per attention tile; all keys remain visible"
    )
    p.add_argument(
        "--reference-rounding",
        action="store_true",
        help="experimental BF16 boundaries and FP32 elementwise math; writes a vit-bf16 asset",
    )
    p.add_argument("--threads", type=int, default=2)
    args = p.parse_args()
    if min(args.height, args.width, args.chunk, args.threads) < 1 or not 1 <= args.layers <= 32:
        p.error("dimensions, chunk and threads must be positive; layers must be 1..32")
    torch.set_num_threads(args.threads)
    export(args)


if __name__ == "__main__":
    main()
