"""Compare individual Transcribe backbone blocks with the upstream CPU model.

Weights retain their NVFP4 or FP8 quantization. Traces and reconstructed
weights stay in memory; the script reads only the selected block's experts.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.nemotron_h.modeling_nemotron_h import NemotronHConfig, NemotronHBlock
from flash import Checkpoint, Linear as StoredLinear, assign

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'benchmarks'))
from runtime import wait_for_gpu, guarded_process


class Linear(StoredLinear):
    def __init__(self, checkpoint, stem):
        super().__init__(checkpoint, stem, expert=True)
        raw = checkpoint.tensor(stem + '.weight')
        self.static_fp8 = raw.dtype == torch.float8_e4m3fn and self.kind != 'm'
        if self.static_fp8:
            self.input_scale = float(checkpoint.tensor(stem + '.input_scale'))

    def forward(self, x):
        if not self.static_fp8:
            return super().forward(x)
        values = (x.float() / self.input_scale).clamp(-448, 448).to(torch.float8_e4m3fn).float()
        result = F.linear(values, self.weight) * (self.scale * self.input_scale)
        if self.bias is not None:
            result += self.bias
        return result.to(torch.bfloat16)


class Experts(nn.Module):
    def __init__(self, checkpoint, prefix):
        super().__init__()
        self.checkpoint, self.prefix = checkpoint, prefix

    def forward(self, x, selected, weights):
        output = torch.zeros_like(x, dtype=torch.float32)
        for expert in selected.unique(sorted=True).tolist():
            token, rank = torch.where(selected == expert)
            prefix = self.prefix + f'.{expert}'
            values = Linear(self.checkpoint, prefix + '.up_proj')(x[token])
            values = F.relu(values).square()
            values = Linear(self.checkpoint, prefix + '.down_proj')(values)
            output.index_add_(0, token, values.float() * weights[token, rank, None])
        return output.to(x.dtype)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--layers', default='0,1,5')
    parser.add_argument('--token', type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(4)
    values = json.loads((args.checkpoint / 'config.json').read_text())['llm_config']
    values['time_step_limit'] = [0.0, float('inf')]
    config = NemotronHConfig(**values)
    config._attn_implementation = 'eager'
    ready = wait_for_gpu(24, 60, reserve_gib=10, desktop=True)
    result = guarded_process([str(ROOT / 'build/myds4'), '-m', str(args.checkpoint), '--memory', '24',
                              '--ctx', '4096', '--tokens', str(args.token), '--dump-fd', '1'],
                             os.environ.copy(), 180, ready['samples'][-1]['memory'], allow_cpu_helpers=True)
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors='replace'))
    width = config.hidden_size
    size = (config.num_hidden_layers * width + config.vocab_size) * 4
    if len(result.stdout) != size:
        raise ValueError(f'trace length {len(result.stdout)}, expected {size}')
    trace = np.frombuffer(result.stdout, dtype='<f4')
    checkpoint = Checkpoint(args.checkpoint)
    for layer in map(int, args.layers.split(',')):
        prefix = f'backbone.layers.{layer}'
        with torch.device('meta'):
            model = NemotronHBlock(config, layer)
        if config.layers_block_type[layer] == 'moe':
            model.mixer.experts = Experts(checkpoint, prefix + '.mixer.experts')
        for name, child in list(model.named_modules()):
            if isinstance(child, nn.Linear):
                parent, _, key = name.rpartition('.')
                setattr(model.get_submodule(parent) if parent else model, key, Linear(checkpoint, prefix + '.' + name))
        for name, parameter in list(model.named_parameters()):
            if parameter.is_meta:
                assign(model, name, checkpoint.tensor(prefix + '.' + name))
        for name, value in list(model.named_buffers()):
            if value.is_meta and prefix + '.' + name in checkpoint.files:
                parent, _, key = name.rpartition('.')
                setattr(model.get_submodule(parent) if parent else model, key, checkpoint.tensor(prefix + '.' + name))
        model.eval()
        if layer:
            inputs = torch.from_numpy(trace[(layer - 1) * width:layer * width].copy()).to(torch.bfloat16)
        else:
            inputs = checkpoint.tensor('backbone.embeddings.weight')[args.token]
        with torch.inference_mode():
            mask = torch.zeros(1, 1, 1, 1) if config.layers_block_type[layer] == 'full_attention' else None
            output = model(inputs.reshape(1, 1, width), attention_mask=mask).float().flatten()
        native = torch.from_numpy(trace[layer * width:(layer + 1) * width].copy())
        error = output - native
        snr = float(10 * torch.log10(output.square().sum() / error.square().sum().clamp_min(1e-30)))
        print(json.dumps({'layer': layer, 'snr_db': snr, 'max_abs': float(error.abs().max()),
                          'different': int((output != native).sum())}), flush=True)
        if snr < 30:
            raise AssertionError('Nemotron block differs beyond the BF16 reference tolerance')


if __name__ == '__main__':
    main()
