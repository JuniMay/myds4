"""Compare a Flash-Next layer with Transformers using lazy, CPU-only checkpoint reads.

The native trace stays in memory. Only the requested layer's dense matrices
and its selected experts are reconstructed; no checkpoint conversion is saved.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from transformers.models.qwen4_exp.modeling_qwen4_exp import (
    Qwen4ExpTextConfig, Qwen4ExpTextDecoderLayer,
)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'benchmarks'))
from runtime import wait_for_gpu, guarded_process


class Checkpoint:
    def __init__(self, path):
        self.path = Path(path)
        self.files = json.loads((self.path / 'model.safetensors.index.json').read_text())['weight_map']

    def tensor(self, name):
        with safe_open(self.path / self.files[name], framework='pt', device='cpu') as handle:
            return handle.get_tensor(name)

    def scales(self, stem, rows, cols):
        name = stem + '.weight_scale'
        if name in self.files:
            return self.tensor(name).float()
        fixed = self.tensor(name + '.nvfp4_csf_fixed')
        base = fixed[:, :16].to(torch.int16)
        codes = fixed[:, 16:].reshape(rows // 16, 16, cols // 2)
        codes = torch.stack((codes & 15, codes >> 4), dim=-1).flatten(-2)
        values = (codes.to(torch.int16) + base[:, :, None]).to(torch.uint8).reshape(rows, cols)
        exceptions = self.tensor(name + '.nvfp4_csf_exceptions').to(torch.int64)
        values.flatten()[exceptions & 0xffffff] = (exceptions >> 24).to(torch.uint8)
        return values.view(torch.float8_e4m3fn).float()


def quantize(x, kind, scale):
    x = x.float()
    groups = x.reshape(-1, 32 if kind == 'm' else 16)
    if kind == 'm':
        amax = groups.abs().amax(-1, keepdim=True).clamp_min(2.0 ** -126)
        factor = 2.0 ** torch.ceil(torch.log2(amax / 448))
        result = (groups / factor).clamp(-448, 448).to(torch.float8_e4m3fn).float() * factor
    else:
        scaled = groups / scale
        factor = (scaled.abs().amax(-1, keepdim=True) / 6).clamp_max(448).to(torch.float8_e4m3fn).float()
        values = torch.where(factor > 0, scaled / factor, torch.zeros_like(scaled)).clamp(-6, 6)
        mid = torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5.0])
        levels = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6.0])
        magnitude = values.abs().contiguous()
        code = torch.bucketize(magnitude, mid)
        ties = (code < 7) & (magnitude == mid[code.clamp_max(6)]) & ((code & 1) != 0)
        code += ties.to(code.dtype)
        result = levels[code].copysign(values) * factor * scale
    return result.reshape(x.shape).to(torch.bfloat16).float()


class Linear(nn.Module):
    def __init__(self, checkpoint, stem, expert=False):
        super().__init__()
        self.kind, self.scale, self.input_scale = 'b', 1.0, 1.0
        raw = checkpoint.tensor(stem + '.weight')
        if raw.dtype == torch.uint8:
            self.kind = 'n' if expert else 'b'
            levels = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6., -0., -.5, -1, -1.5, -2, -3, -4, -6])
            codes = torch.stack((raw & 15, raw >> 4), -1).flatten(-2).long()
            scales = checkpoint.scales(stem, raw.shape[0], codes.shape[1] // 16)
            self.weight = levels[codes] * scales.repeat_interleave(16, -1)
            self.scale = float(checkpoint.tensor(stem + '.weight_scale_2'))
            if self.kind == 'n':
                self.input_scale = float(checkpoint.tensor(stem + '.input_scale'))
        elif raw.dtype == torch.float8_e4m3fn:
            scales = checkpoint.tensor(stem + '.weight_scale')
            self.weight = raw.float()
            if scales.numel() == 1 and scales.dtype == torch.float32:
                self.scale = float(scales)
            else:
                self.kind = 'm'
                self.weight *= (2.0 ** (scales.to(torch.int32) - 127)).repeat_interleave(32, -1)
        else:
            self.weight = raw.float()
        self.bias = checkpoint.tensor(stem + '.bias').float() if stem + '.bias' in checkpoint.files else None

    def forward(self, x):
        values = quantize(x, self.kind, self.input_scale) if self.kind in ('n', 'm') else x.float()
        output = F.linear(values, self.weight) * self.scale
        if self.bias is not None:
            output += self.bias
        return output.to(torch.bfloat16)


class Experts(nn.Module):
    def __init__(self, checkpoint, prefix):
        super().__init__()
        self.checkpoint, self.prefix = checkpoint, prefix

    def forward(self, x, selected, weights):
        out = torch.zeros_like(x)
        print('reference experts:', selected.tolist(), flush=True)
        for expert in selected.unique(sorted=True).tolist():
            token, rank = torch.where(selected == expert)
            stem = self.prefix + f'.{expert}'
            gate = Linear(self.checkpoint, stem + '.gate_proj', True)(x[token])
            up = Linear(self.checkpoint, stem + '.up_proj', True)(x[token])
            hidden = F.silu(gate) * up
            result = Linear(self.checkpoint, stem + '.down_proj', True)(hidden)
            out.index_add_(0, token, result * weights[token, rank, None])
        return out


class NgramTable(nn.Module):
    def __init__(self, checkpoint, prefix):
        super().__init__()
        self.checkpoint, self.prefix = checkpoint, prefix
        self.weight = torch.empty(0)
        name = prefix + '.shard_0.weight'
        with safe_open(checkpoint.path / checkpoint.files[name], framework='pt', device='cpu') as handle:
            rows, columns = handle.get_slice(name).get_shape()
        self.rows, self.width = rows, columns * 2
        self.scale = float(checkpoint.tensor(prefix + '.weight_scale_2'))

    def forward(self, ids):
        result = []
        levels = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6., -0., -.5, -1, -1.5, -2, -3, -4, -6])
        for index in ids.flatten().tolist():
            shard, row = divmod(index, self.rows)
            name = self.prefix + f'.shard_{shard}.weight'
            with safe_open(self.checkpoint.path / self.checkpoint.files[name], framework='pt', device='cpu') as handle:
                packed = handle.get_slice(name)[row:row + 1]
            name += '_scale'
            with safe_open(self.checkpoint.path / self.checkpoint.files[name], framework='pt', device='cpu') as handle:
                scales = handle.get_slice(name)[row:row + 1].float()
            code = torch.stack((packed & 15, packed >> 4), -1).flatten(-2).long()
            result.append((levels[code] * scales.repeat_interleave(16, -1) * self.scale).to(torch.bfloat16))
        return torch.cat(result).reshape(*ids.shape, self.width)


def assign(module, name, value):
    parent, _, key = name.rpartition('.')
    owner = module.get_submodule(parent) if parent else module
    setattr(owner, key, nn.Parameter(value, requires_grad=False))


def block(checkpoint, config, layer):
    prefix = f'model.language_model.layers.{layer}'
    with torch.device('meta'):
        result = Qwen4ExpTextDecoderLayer(config, layer)
    result.mlp.experts = Experts(checkpoint, prefix + '.mlp.experts')
    if result.ple is not None:
        result.ple.ple_embedding.ngram_embedding = NgramTable(checkpoint, prefix + '.ple.ple_embedding.ngram_embedding')
    for name, child in list(result.named_modules()):
        if isinstance(child, nn.Linear):
            parent, _, key = name.rpartition('.')
            owner = result.get_submodule(parent) if parent else result
            setattr(owner, key, Linear(checkpoint, prefix + '.' + name))
    for name, parameter in list(result.named_parameters()):
        if parameter.is_meta:
            assign(result, name, checkpoint.tensor(prefix + '.' + name))
    for name, value in list(result.named_buffers()):
        if value.is_meta and prefix + '.' + name in checkpoint.files:
            parent, _, key = name.rpartition('.')
            owner = result.get_submodule(parent) if parent else result
            setattr(owner, key, checkpoint.tensor(prefix + '.' + name))
    return result.eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--layer', type=int, default=0)
    parser.add_argument('--token', type=int, default=248044)
    args = parser.parse_args()
    torch.set_num_threads(4)
    config = Qwen4ExpTextConfig(**json.loads((args.checkpoint / 'config.json').read_text())['text_config'])
    config._attn_implementation = 'eager'
    ready = wait_for_gpu(27, 60, reserve_gib=10, desktop=True)
    import os
    result = guarded_process([str(ROOT / 'build/myds4'), '-m', str(args.checkpoint), '--memory', '24',
                              '--ctx', '4096', '--tokens', str(args.token), '--dump-fd', '1'],
                             os.environ.copy(), 180, ready['samples'][-1]['memory'], allow_cpu_helpers=True)
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors='replace'))
    width = config.hidden_size * config.hc_count
    expected_size = (config.num_hidden_layers * width + config.vocab_size) * 4
    if len(result.stdout) != expected_size:
        raise ValueError(f'trace has {len(result.stdout)} bytes; expected {expected_size}')
    trace = np.frombuffer(result.stdout, dtype='<f4')
    checkpoint = Checkpoint(args.checkpoint)
    if args.layer == 0:
        embedding = checkpoint.tensor('model.language_model.embed_tokens.weight')[args.token]
        x = embedding.repeat(config.hc_count).reshape(1, 1, width)
    else:
        x = torch.from_numpy(trace[(args.layer - 1) * width:args.layer * width].copy()).to(torch.bfloat16).reshape(1, 1, width)
    native = torch.from_numpy(trace[args.layer * width:(args.layer + 1) * width].copy())
    with torch.inference_mode():
        model = block(checkpoint, config, args.layer)
        output = model(x, position_embeddings=(torch.ones(1, 1, 64, dtype=torch.bfloat16), torch.zeros(1, 1, 64, dtype=torch.bfloat16)),
                       attention_mask=torch.zeros(1, 1, 1, 1), past_key_values=None, ple_input_ids=torch.tensor([[args.token]])).float().flatten()
    difference = output - native
    snr = 10 * torch.log10(output.square().sum() / difference.square().sum().clamp_min(1e-30))
    print(json.dumps({'layer': args.layer, 'max_abs': difference.abs().max().item(), 'snr_db': snr.item(),
                      'different': int((output != native).sum()), 'native_std': native.std().item(), 'reference_std': output.std().item()}))
    if snr.item() < 30:
        raise AssertionError('layer output differs beyond the BF16 reference allowance')


if __name__ == '__main__':
    main()
