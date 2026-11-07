"""The reference for Qwen3.8 (qwen3_5) checkpoints: Hugging Face's own
modeling code (transformers' Qwen3_5ForCausalLM) on the CPU, in FP32, with
the checkpoint's NVFP4, FP8 and BF16 weights dequantized exactly as stored
(docs/qwen.md). It writes reference token ids, and compares its logits with
the engine's: the stream that `build/myds4 --tokens @IDS --perplexity
--prefill P --logits-fd FD` writes, whose row i holds the logits after
position P - 1 + i. With --prefill 1 every row comes from decoding passes;
with --prefill n - 1 the one row comes from prompt passes.

It needs torch and transformers, which the engine does not use; install them
in a virtual environment outside the repository:

    python3 -m venv VENV
    VENV/bin/pip install --no-cache-dir torch transformers safetensors

Examples (VENV/bin/python):

    qwen_hf.py --model DIR --text FILE --chars 4000 --write-tokens ids.txt
    qwen_hf.py --model DIR --chat "Write a short story." --write-tokens ids.txt
    qwen_hf.py --model DIR --tokens ids.txt --compare stream.bin --prefill 1

A forward pass takes about 20 GB of memory (the weights stay quantized and
each product dequantizes its matrix when it runs) and about a minute per
thousand tokens.
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # never fetch kernels or files
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy
import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextRotaryEmbedding

E2M1 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.float32)


class StoredLinear(torch.nn.Module):
    """A product with the checkpoint's weight as stored: NVFP4 (E2M1 codes,
    two a byte with the low nibble first, an E4M3 scale per 16 and an FP32
    scale), FP8 (E4M3 and an FP32 scale) or BF16. The weight is dequantized
    to FP32 when the product runs."""

    def __init__(self, tensors):
        super().__init__()
        self.t = tensors

    def weight_f32(self):
        w = self.t["weight"]
        if w.dtype == torch.uint8:
            codes = torch.stack((w & 15, w >> 4), dim=-1).reshape(w.shape[0], -1).long()
            scale = self.t["weight_scale"].float().repeat_interleave(16, dim=1)
            return E2M1[codes] * scale * self.t["weight_scale_2"].float()
        if w.dtype == torch.float8_e4m3fn:
            return w.float() * self.t["weight_scale"].float()
        return w.float()

    def forward(self, x):
        return F.linear(x.float(), self.weight_f32())


def load_model(model_dir):
    root = Path(model_dir)
    config = json.loads((root / "config.json").read_text())
    cfg = Qwen3_5TextConfig(**config.get("text_config", config))
    cfg._attn_implementation = "sdpa"
    with torch.device("meta"):
        model = Qwen3_5ForCausalLM(cfg)
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    files = {}

    def tensor(name):
        shard = index[name]
        if shard not in files:
            files[shard] = safe_open(root / shard, framework="pt")
        return files[shard].get_tensor(name)

    def stored(name):  # the checkpoint's name for a module of the causal LM
        return "lm_head" if name == "lm_head" else name.replace("model.", "model.language_model.", 1)

    for name in [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]:
        prefix = stored(name)
        parts = {p: tensor(f"{prefix}.{p}") for p in ("weight", "weight_scale", "weight_scale_2") if f"{prefix}.{p}" in index}
        parent, _, leaf = name.rpartition(".")
        setattr(model.get_submodule(parent) if parent else model, leaf, StoredLinear(parts))
    rest = {}
    for name, _ in model.named_parameters():
        module, _, leaf = name.rpartition(".")
        rest[name] = tensor(f"{stored(module)}.{leaf}").float()
    model.load_state_dict(rest, strict=False, assign=True)
    model.model.rotary_emb = Qwen3_5TextRotaryEmbedding(cfg)  # its buffers, computed on the CPU
    meta = [n for n, t in list(model.named_parameters()) + list(model.named_buffers()) if t.is_meta]
    if meta:
        sys.exit(f"not loaded: {meta[:5]}")
    return model.eval()


def read_ids(path):
    return [int(x) for x in Path(path).read_text().replace(",", " ").split()]


def tokenizer(model_dir):
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(Path(model_dir) / "tokenizer.json"))


def chat_ids(model_dir, message, thinking):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    text = tok.apply_chat_template(
        [{"role": "user", "content": message}], tokenize=False, add_generation_prompt=True, enable_thinking=thinking
    )
    return tokenizer(model_dir).encode(text, add_special_tokens=False).ids


def perplexity(logits, ids):
    """Of ids[1:] on logits[:-1] (rows of the positions before them)."""
    lp = F.log_softmax(logits.double(), -1)
    return math.exp(float(-lp.gather(1, torch.tensor(ids)[:, None]).mean()))


def compare(ref, eng, ids):
    """Rows of reference and engine logits at the same positions, and the
    tokens they score."""
    lr, le = F.log_softmax(ref.double(), -1), F.log_softmax(eng.double(), -1)
    kl = (lr.exp() * (lr - le)).sum(-1)
    noise = ((ref - eng).double() ** 2).sum(-1)
    snr = 10 * torch.log10((ref.double() ** 2).sum(-1) / noise.clamp(min=1e-300))
    top_r, top_e = ref.topk(10).indices, eng.topk(10).indices
    same1 = int((top_r[:, 0] == top_e[:, 0]).sum())
    overlap = sum(len(set(a.tolist()) & set(b.tolist())) for a, b in zip(top_r, top_e)) / len(ids)
    print(f"{len(ids)} rows: max |diff| {float((ref - eng).abs().max()):.4f}; KL mean {float(kl.mean()):.3e}, "
          f"max {float(kl.max()):.3e} nats; SNR mean {float(snr.mean()):.1f}, min {float(snr.min()):.1f} dB; "
          f"top-1 same at {same1}; top-10 overlap {overlap:.2f} of 10")
    print(f"perplexity of the {len(ids)} scored tokens: reference {perplexity(ref, ids):.4f}, "
          f"engine {perplexity(eng, ids):.4f}")


def main():
    a = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    a.add_argument("--model", required=True, help="the checkpoint directory")
    a.add_argument("--tokens", help="the prompt's token ids (commas, spaces or newlines between them)")
    a.add_argument("--text", help="a text file to tokenize with the checkpoint's tokenizer")
    a.add_argument("--chars", type=int, help="with --text: its first N characters")
    a.add_argument("--chat", help="a user message, rendered with the checkpoint's chat template")
    a.add_argument("--thinking", action="store_true", help="with --chat: the template's thinking mode")
    a.add_argument("--write-tokens", help="write the token ids of --text or --chat here")
    a.add_argument("--compare", help="the engine's --logits-fd stream for the same ids")
    a.add_argument("--prefill", type=int, default=1, help="with --compare: the engine's --prefill (default 1)")
    o = a.parse_args()

    if o.text or o.chat:
        if not o.write_tokens:
            sys.exit("--text and --chat need --write-tokens")
        if o.text:
            body = Path(o.text).read_text()
            ids = tokenizer(o.model).encode(body[: o.chars] if o.chars else body, add_special_tokens=False).ids
        else:
            ids = chat_ids(o.model, o.chat, o.thinking)
        Path(o.write_tokens).write_text(",".join(map(str, ids)) + "\n")
        print(f"{len(ids)} tokens -> {o.write_tokens}")
        if not o.tokens:
            return
    if not o.tokens:
        sys.exit("give --tokens (or --text/--chat with --write-tokens)")

    ids = read_ids(o.tokens)
    began = time.time()
    model = load_model(o.model)
    with torch.inference_mode():
        logits = model(input_ids=torch.tensor([ids]), use_cache=False).logits[0].float()
    print(f"{len(ids)} tokens: forward pass in {time.time() - began:.0f} s")
    if len(ids) > 1:
        print(f"perplexity over {len(ids) - 1} predictions: {perplexity(logits[:-1], ids[1:]):.4f}")
    top = logits[-1].topk(5)
    print("top next tokens:", [(int(i), round(float(v), 3)) for v, i in zip(top.values, top.indices)])
    if o.compare:
        vocab = logits.shape[1]
        eng = torch.from_numpy(numpy.fromfile(o.compare, dtype="float32"))
        if eng.numel() % vocab:
            sys.exit(f"{o.compare}: not rows of {vocab} floats")
        eng = eng.reshape(-1, vocab)
        first = o.prefill - 1  # row i: the logits after position first + i
        if first < 0 or first + len(eng) > len(ids) - 1:
            sys.exit(f"{o.compare}: {len(eng)} rows from position {first} past the prompt's {len(ids)} tokens")
        compare(logits[first : first + len(eng)], eng, ids[first + 1 : first + 1 + len(eng)])


if __name__ == "__main__":
    main()
