"""Checks myds4 against the checkpoint's own references.

  python tools/check.py tok CHECKPOINT     tokenizer against the tokenizers library
  python tools/check.py encoding CHECKPOINT
                                           prompts of the encoding's test cases and of 1,200
                                           generated conversations against encoding/encoding.py
  python tools/check.py vision CHECKPOINT IMAGE
                                           an image's feature rows (myds4 --vision-fd) against
                                           inference/vision.py's ViT and Aligner on the CPU
                                           (needs Pillow: pip install pillow)
  python tools/check.py model CHECKPOINT [CACHE]
                                           every layer's streams, the routing and the logits
                                           against DeepSeek's inference/model.py on the CPU;
                                           CACHE keeps the reference's results for reruns
  python tools/check.py idle URL KEYFILE   an idle server keeps its experts while no other
                                           program needs the memory
  python tools/check.py demand URL KEYFILE an idle server gives expert slots back when another
                                           program takes memory
  python tools/check.py placeholder URL KEYFILE
                                           a server reads a literal <｜deepseek_image｜> in
                                           message text as text
  python tools/check.py view               the live view (--tui) keeps its first line on screen
                                           when its lines fill the terminal (compiles
                                           tools/view_check.c with src/tui.c; no model)

Run from the repository root after `make`. The model check decodes a short
prompt one token at a time with both, holds everything in memory (about 30 GB
for the reference) and writes no files. The server checks wait for the
server's running request to end, or send a short one. The demand check
takes memory until only 7 GiB is free, for about 5 s.
"""

import importlib.util
import json
import math
import os
import subprocess
import sys
import time
import types
import urllib.error
import urllib.request
from pathlib import Path

TEXTS = [
    "Hello, world! It's 2026-10-28; price $1,234.56.",
    "  leading spaces\n\n\ttabs\r\n and   trailing   ",
    "def f(x):\n    return x**2  # square\n",
    "中文测试：你好世界！日本語のテキスト、カタカナ。한국어도 있어요.",
    "Emoji 🤖🚀 and accents: café naïve résumé Ångström",
    "<｜begin▁of▁sentence｜><｜User｜>Hi<｜Assistant｜></think>",
    "x = [1, 2, 3]; y = {'a': 1}\n\n\n",
    "１２３ full-width digits and ①②③ and ٣٤٥ arabic",
    "wait...what?!  --flag --other=val",
    "áb̀c combining",
    " \n \n",
    "Ünïcödé MIXED case WORDS",
    "ﬁ ligature, ½ fractions, ™ symbols, — dashes «quotes» ‘single’",
    "tab\tseparated\tvalues\n1\t2\t3",
    "    ",
    "12345678901234567890",
    "<think>reasoning</think>answer",
]


def tok(checkpoint):
    from tokenizers import Tokenizer
    reference = Tokenizer.from_file(f"{checkpoint}/tokenizer.json")
    failed = 0
    for text in TEXTS:
        want = reference.encode(text, add_special_tokens=False).ids
        out = subprocess.run(["build/myds4", "-m", checkpoint, "-p", text, "--tokenize"],
                             capture_output=True, text=True, check=True).stdout.strip()
        got = [int(v) for v in out.split(",")] if out else []
        if got != want:
            failed += 1
            print(f"mismatch for {text!r}\n  want {want}\n  got  {got}")
    print(f"tokenizer: {len(TEXTS) - failed} of {len(TEXTS)} texts match")
    return failed == 0


def torch_kernels(torch):
    """inference/kernel.py's operations in eager PyTorch, with its activation
    quantization: E4M3 with power-of-two scales per 32, E2M1 with
    power-of-two (indexer) or E4M3 (compressed keys) scales."""
    F = torch.nn.functional
    k = types.ModuleType("kernel")
    grid = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])

    def pow2_scale(amax, inv):  # fast_round_scale
        bits = (amax * torch.tensor(inv, dtype=torch.float32)).view(torch.int32)
        e = ((bits >> 23) & 255) - 127 + ((bits & 0x7FFFFF) != 0).int()
        return ((e + 127) << 23).view(torch.float32)

    def e2m1(v):  # round |v| <= 6 to E2M1, ties to even
        a = v.abs()
        e = (torch.frexp(a).exponent - 1).clamp(min=0).float()
        step = torch.exp2(e - 1)
        return torch.copysign(torch.round(a / step) * step, v)

    def act_quant(x, block_size=128, scale_fmt=None, scale_dtype=None, inplace=False):
        b = x.float().unflatten(-1, (-1, block_size))
        s = pow2_scale(b.abs().amax(-1, keepdim=True).clamp(min=1e-4), 1 / 448)
        y = ((b / s).clamp(-448, 448).to(torch.float8_e4m3fn).float() * s).flatten(-2).to(x.dtype)
        if inplace:
            return x.copy_(y)
        return y, None

    def fp4_act_quant(x, block_size=32, inplace=False, scale_dtype=torch.float8_e8m0fnu):
        b = x.float().unflatten(-1, (-1, block_size))
        amax = b.abs().amax(-1, keepdim=True)
        if scale_dtype == torch.float8_e4m3fn:
            s = (amax.clamp(min=6 * 2**-9) / 6).to(torch.float8_e4m3fn).float()
        else:
            s = pow2_scale(amax.clamp(min=6 * 2**-126), 1 / 6)
        y = (e2m1((b / s).clamp(-6, 6)) * s).flatten(-2).to(x.dtype)
        return x.copy_(y) if inplace else (y, None)

    def fp8_gemm(a, a_s, b, b_s, scale_dtype=None, block_size=128):
        w = b.float() * b_s.float().repeat_interleave(block_size, 0).repeat_interleave(block_size, 1)[: b.shape[0], : b.shape[1]]
        return F.linear(a.float(), w).to(torch.get_default_dtype())

    def fp4_gemm(a, a_s, b, b_s, scale_dtype=None, act_block_size=128):
        c = b.view(torch.uint8).long()
        values = torch.cat([grid, -grid])
        w = torch.stack([values[c & 15], values[c >> 4]], -1).flatten(-2)
        w = w * b_s.float().repeat_interleave(32, 1)
        return F.linear(a.float(), w).to(torch.get_default_dtype())

    def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
        # the kernel: FP32 scores, sink in the denominator, BF16 weights for the values
        rows = kv.float()[0, topk_idxs[0].clamp(min=0).long()]  # [m, k, d]
        s = torch.einsum("mhd,mkd->mhk", q[0].float(), rows) * softmax_scale
        s = s.masked_fill((topk_idxs[0] < 0).unsqueeze(1), float("-inf"))
        m = s.amax(-1, keepdim=True).clamp(min=-1e30)
        p = torch.exp(s - m)
        den = p.sum(-1, keepdim=True) + torch.exp(attn_sink.view(1, -1, 1) - m)
        o = torch.einsum("mhk,mkd->mhd", p.bfloat16().float(), rows) / den
        return o.unsqueeze(0).to(q.dtype)

    def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
        h = hc_mult
        pre = torch.sigmoid(mixes[..., :h] * hc_scale[0] + hc_base[:h]) + eps
        post = 2 * torch.sigmoid(mixes[..., h:2 * h] * hc_scale[1] + hc_base[h:2 * h])
        c = (mixes[..., 2 * h:] * hc_scale[2] + hc_base[2 * h:]).unflatten(-1, (h, h))
        c = c.softmax(-1) + eps
        c = c / (c.sum(-2, keepdim=True) + eps)
        for _ in range(sinkhorn_iters - 1):
            c = c / (c.sum(-1, keepdim=True) + eps)
            c = c / (c.sum(-2, keepdim=True) + eps)
        return pre, post, c

    for f in (act_quant, fp4_act_quant, fp8_gemm, fp4_gemm, sparse_attn, hc_split_sinkhorn):
        setattr(k, f.__name__, f)
    return k


def render(case, images=False):
    """myds4's prompt for a test case (the encoding/tests format), or None if it
    refused; with images, also the image URLs in placeholder order. The
    renderer is build/myds4, or the executable that MYDS4_RENDER names (the
    Rust server's build/myds4-server)."""
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".json", encoding="utf-8") as f:
        json.dump(case, f, ensure_ascii=False)
        f.flush()
        out = subprocess.run([os.environ.get("MYDS4_RENDER", "build/myds4"), "--render", f.name], capture_output=True)
    prompt = out.stdout.decode("utf-8") if out.returncode == 0 else None
    if not images:
        return prompt
    return prompt, [l[6:] for l in out.stderr.decode("utf-8").splitlines() if l.startswith("image ")]


def conversation(rng, with_images=False):
    """A random test case: tools, tool calls and results, reasoning, system
    messages at the start and mid-conversation, reminders, content parts,
    escapes and non-ASCII text, both thinking modes and every effort form;
    with_images, also image parts in user turns and tool results."""
    texts = ["Hello", "What's 2+2?", "多语言测试：你好，世界", "emoji 🙂🚀", 'quote " and backslash \\ here',
             "line1\nline2\n\n", "tab\tsep", "ctrl \u0001 char", "<think>literal</think>", "trailing space ",
             "  leading", "", "a" * 40, 'json {"a": 1}', "ü ñ é", "x</think>y", "Reasoning Effort: 3"]
    text = lambda: rng.choice(texts)
    images = [0]  # image parts get unique URLs, so that their order can be compared

    def part():
        if not with_images or rng.random() < 0.7:
            return {"type": "text", "text": text()}
        images[0] += 1
        return {"type": "image_url", "image_url": {"url": f"img_{images[0]}.png"}}
    content = lambda: [part() for _ in range(rng.randint(1, 3))] if rng.random() < (0.5 if with_images else 0.2) else text()

    def tool(i):
        props = {}
        for j in range(rng.randint(0, 4)):
            t = rng.choice(["string", "integer", "number", "boolean", "array", "object"])
            p = {"type": t}
            if rng.random() < 0.6:
                p["description"] = text()
            if t == "string" and rng.random() < 0.3:
                p["enum"] = ["a", "b", "ü"]
            if t == "array":
                p["items"] = {"type": "string"}
            if t == "object":
                p["properties"] = {"k": {"type": "integer", "minimum": 0}}
            if t == "number" and rng.random() < 0.5:
                p["default"] = 0.5
            props[f"p{j}_{t}"] = p
        f = {"name": f"tool_{i}", "parameters": {"type": "object", "properties": props, "required": list(props)[:1]}}
        if rng.random() < 0.8:
            f["description"] = text()
        return {"type": "function", "function": f}

    def arguments(t):
        values = {"string": text, "integer": lambda: rng.randint(-5, 1000), "number": lambda: rng.choice([0.5, 3.25, -1.0, 100]),
                  "boolean": lambda: rng.choice([True, False]), "array": lambda: ["x", "y ü"], "object": lambda: {"k": 3}}
        a = {k: values[p["type"]]() for k, p in t["function"]["parameters"]["properties"].items() if rng.random() < 0.7}
        return json.dumps(a, ensure_ascii=rng.random() < 0.5)

    tools = [tool(i) for i in range(rng.randint(1, 3))] if rng.random() < 0.5 else None
    msgs = [{"role": "system", "content": text() if with_images else content()}] if rng.random() < 0.7 else []
    turns = rng.randint(1, 4)
    for turn in range(turns):
        if rng.random() < 0.1:
            msgs.append({"role": "latest_reminder", "content": text()})
        msgs.append({"role": "user", "content": content()})
        if rng.random() < 0.12:
            msgs.append({"role": "system", "content": text()})
        if turn == turns - 1 and rng.random() < 0.7:
            break
        steps = rng.randint(1, 3)
        for s in range(steps):
            a = {"role": "assistant", "content": text()}
            if rng.random() < 0.6:
                a["reasoning_content"] = text()
            if not tools or s == steps - 1 or rng.random() >= 0.7:
                msgs.append(a)
                break
            a["tool_calls"] = [{"id": f"call_{turn}_{s}_{c}", "type": "function",
                                "function": {"name": t["function"]["name"], "arguments": arguments(t)}}
                               for c, t in enumerate(rng.choice(tools) for _ in range(rng.randint(1, 2)))]
            if rng.random() < 0.5:
                a["content"] = ""
            msgs.append(a)
            msgs += [{"role": "tool", "tool_call_id": c["id"], "content": content() if with_images else text()}
                     for c in a["tool_calls"]]
            if turn == turns - 1 and rng.random() < 0.3:
                break
    case = {"messages": msgs, "thinking_mode": rng.choice(["thinking", "chat"])}
    if tools:
        case["tools"] = tools
    effort = rng.choice([None, "low", "high", "max", 1, 25, 50, 62, 75, 99, 100])
    if effort is not None:
        case["reasoning_effort"] = effort
    return case


def encoding(checkpoint):
    """The prompts against encoding/encoding.py. A case's tools go into its
    first system message, which DeepSeek's request converter (deepseek-recipe)
    inserts empty when the conversation does not start with one; the
    encoding's own case loader would give them to the first message
    whatever its role, and render none for a user message."""
    import copy
    import random
    spec = importlib.util.spec_from_file_location("encoding", Path(checkpoint) / "encoding" / "encoding.py")
    enc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(enc)
    tests = Path(checkpoint) / "encoding" / "tests"
    failed = 0
    for i in (1, 2, 3, 5):  # case 4 uses the internal task field, which myds4 does not render
        case = json.loads((tests / f"test_input_{i}.json").read_text())
        if render(case if isinstance(case, dict) else {"messages": case}) != (tests / f"test_output_{i}.txt").read_text():
            failed += 1
            print(f"test case {i}: the prompts differ")
    same = 0
    for seed in (7, 8, 9):
        rng = random.Random(seed)
        for n in range(400):
            case = conversation(rng)
            c = copy.deepcopy(case)
            if "tools" in c:
                if c["messages"][0]["role"] != "system":
                    c["messages"].insert(0, {"role": "system", "content": ""})
                c["messages"][0]["tools"] = c["tools"]
            want = enc.encode_messages(c["messages"], thinking_mode=c["thinking_mode"],
                                       reasoning_effort=c.get("reasoning_effort"))
            got = render(case)
            if got == want:
                same += 1
            elif failed < 3:
                at = next((k for k in range(min(len(want), len(got or ""))) if want[k] != got[k]), 0)
                print(f"seed {seed} case {n} differs at {at}:\n  want {want[at:at + 120]!r}\n  got  {(got or '')[at:at + 120]!r}")
                failed += 1
            else:
                failed += 1
    same_images = 0
    for seed in (10, 11):  # image parts: the placeholders where encoding.py writes them, the images in its order
        rng = random.Random(seed)
        for n in range(300):
            case = conversation(rng, with_images=True)
            c = copy.deepcopy(case)
            if "tools" in c:
                if c["messages"][0]["role"] != "system":
                    c["messages"].insert(0, {"role": "system", "content": ""})
                c["messages"][0]["tools"] = c["tools"]
            want, media = enc.encode_messages(c["messages"], thinking_mode=c["thinking_mode"],
                                              reasoning_effort=c.get("reasoning_effort"), return_multi_modal_data=True)
            got, urls = render(case, images=True)
            if got == want and urls == [m.get("url") for m in media["images"]]:
                same_images += 1
            else:
                failed += 1
                if failed <= 3:
                    print(f"seed {seed} case {n} with images differs: images {urls} against {[m.get('url') for m in media['images']]}")
    print(f"encoding: test cases 1, 2, 3 and 5 checked; {same} of 1200 generated conversations and {same_images} of 600 "
          "with images equal byte for byte")
    return failed == 0


def reference(checkpoint, capacity):
    """DeepSeek's blocks with the checkpoint's weights, routed experts built
    when selected and Engram rows read when looked up."""
    import torch
    from safetensors import safe_open
    from tokenizers import Tokenizer
    torch.set_default_dtype(torch.bfloat16)
    inference = Path(checkpoint) / "inference"
    stubs = {"kernel": torch_kernels(torch), "vision": types.ModuleType("vision"),
             "image_processor": types.ModuleType("image_processor")}
    stubs["vision"].ViT = stubs["vision"].Aligner = None
    for i, name in enumerate(("IMAGE", "IMAGE_END", "IMAGE_NEW_LINE", "IMAGE_START")):
        setattr(stubs["image_processor"], name, i)
    sys.modules.update(stubs)
    mods = {}
    for name in ("engram", "model"):
        spec = importlib.util.spec_from_file_location(name, inference / f"{name}.py")
        mods[name] = sys.modules[name] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mods[name])
    model = mods["model"]
    model.default_dtype = torch.float8_e4m3fn
    import json
    cfg = json.loads((inference / "config.json").read_text())
    fields = {f.name for f in __import__("dataclasses").fields(model.ModelArgs)}
    args = model.ModelArgs(**{k: tuple(v) if isinstance(v, list) else v for k, v in cfg.items() if k in fields and k != "vision_n_layers"},
                           max_batch_size=1, max_seq_len=capacity, vision_n_layers=0)
    index = json.loads((Path(checkpoint) / "model.safetensors.index.json").read_text())["weight_map"]
    files = {}

    def tensor(name):
        f = index[name]
        if f not in files:
            files[f] = safe_open(f"{checkpoint}/{f}", framework="pt")
        t = files[f].get_tensor(name)
        return t.view(torch.float4_e2m1fn_x2) if t.dtype == torch.int8 else t

    class Experts:
        def __init__(self, stem):
            self.stem, self.built = stem, {}

        def __len__(self):
            return args.n_routed_experts

        def __getitem__(self, i):
            if i not in self.built:
                with torch.device("meta"):
                    e = model.Expert(args.dim, args.moe_inter_dim, dtype=torch.float4_e2m1fn_x2,
                                     swiglu_limit=args.swiglu_limit)
                for w in ("w1", "w2", "w3"):
                    lin = getattr(e, w)
                    lin.weight = torch.nn.Parameter(tensor(f"{self.stem}.{i}.{w}.weight"), requires_grad=False)
                    lin.scale = torch.nn.Parameter(tensor(f"{self.stem}.{i}.{w}.scale"), requires_grad=False)
                    lin.weight.scale = lin.scale
                if len(self.built) > 12:
                    self.built.pop(next(iter(self.built)))
                self.built[i] = e
            return self.built[i]

    class EngramRows:
        def __init__(self, stem):
            self.stem = stem

        def __call__(self, ids):
            rows = ids.reshape(-1).tolist()
            f = files.setdefault(index[self.stem + ".weight"], safe_open(f"{checkpoint}/{index[self.stem + '.weight']}", framework="pt"))
            w, s = f.get_slice(self.stem + ".weight"), f.get_slice(self.stem + ".scale")
            v = torch.stack([w[r:r + 1][0] for r in rows]).float().unflatten(-1, (-1, 32))
            v = v * torch.stack([s[r:r + 1][0] for r in rows]).float().unsqueeze(-1)
            return v.flatten(-2).to(torch.bfloat16).reshape(*ids.shape, -1)

    layout = model.EngramLayout.from_args(args)
    blocks = []
    for l in range(args.n_layers):
        with torch.device("meta"):
            b = model.Block(l, args, layout)
        del b.ffn._modules["experts"]
        if b.engram is not None:
            del b.engram._modules["embed"]
        b = b.to_empty(device="cpu")
        for name, p in list(b.named_parameters()):
            owner, _, leaf = name.rpartition(".")
            m = b.get_submodule(owner)
            t = tensor(f"layers.{l}.{name}")
            if leaf == "weight" and isinstance(m, model.Linear) and p.dtype != t.dtype:
                if t.dtype == torch.float8_e4m3fn:  # wo_a: dequantized to BF16 as convert.py stores it
                    s = tensor(f"layers.{l}.{owner}.scale").float()
                    t = (t.float() * s.repeat_interleave(32, 0).repeat_interleave(32, 1)).to(p.dtype)
                else:
                    t = t.to(p.dtype)
            elif p.dtype != t.dtype and t.dtype not in (torch.float8_e8m0fnu,):
                t = t.to(p.dtype)
            setattr(m, leaf, torch.nn.Parameter(t, requires_grad=False))
            if leaf == "scale" and isinstance(m, model.Linear):
                m.weight.scale = m.scale
        b.ffn.experts = Experts(f"layers.{l}.ffn.experts")
        if b.engram is not None:
            b.engram.embed = EngramRows(f"layers.{l}.engram.embed")
        for name, buf in list(b.named_buffers()):
            owner, _, leaf = name.rpartition(".")
            m = b.get_submodule(owner)
            if leaf == "freqs_cis":
                continue
            setattr(m, leaf, torch.full_like(buf, float("-inf")) if leaf == "score_state" else torch.zeros_like(buf))
        a = b.attn
        model.precompute_freqs_cis.cache_clear()
        orig, theta = (args.original_seq_len, args.compress_rope_theta) if a.compress_ratio else (0, args.rope_theta)
        a.freqs_cis = model.precompute_freqs_cis(args.rope_head_dim, capacity, orig, theta, args.rope_factor,
                                                 args.beta_fast, args.beta_slow)
        if a.indexer is not None:
            a.indexer.freqs_cis = None
        blocks.append(b)
        print(f"\rreference: built layer {l + 1}/{args.n_layers}", end="", file=sys.stderr)
    print(file=sys.stderr)
    backend = Tokenizer.from_file(f"{checkpoint}/tokenizer.json")

    class Shim:
        backend_tokenizer = backend

        def __len__(self):
            return backend.get_vocab_size(with_added_tokens=True)

    hashing = model.NgramHashState(args, layout, Shim())
    embed, norm, head = tensor("embed.weight"), tensor("norm.weight"), tensor("head.weight").float()
    return torch, model, args, blocks, hashing, embed, norm, head


def snr(want, got):
    err = ((want - got) ** 2).sum()
    return float("inf") if err == 0 else 10 * math.log10(float((want ** 2).sum() / err))


def vision(checkpoint, image):
    """The image's feature rows: inference/image_processor.py's patches through
    vision.py's ViT and Aligner (BF16 on the CPU, as the reference runs them)
    against myds4's (--vision-fd)."""
    import numpy as np
    import torch
    from safetensors import safe_open
    torch.set_default_dtype(torch.bfloat16)
    inference = Path(checkpoint) / "inference"
    mods = {}
    for name in ("image_processor", "vision"):
        spec = importlib.util.spec_from_file_location(name, inference / f"{name}.py")
        mods[name] = importlib.util.module_from_spec(spec)
        sys.modules[name] = mods[name]
        spec.loader.exec_module(mods[name])
    cfg = json.loads((inference / "config.json").read_text())
    args = types.SimpleNamespace(dim=cfg["dim"], **{k: v for k, v in cfg.items() if k.startswith("vision_")})
    vit, aligner = mods["vision"].ViT(args), mods["vision"].Aligner(args)
    index = json.loads((Path(checkpoint) / "model.safetensors.index.json").read_text())["weight_map"]
    for prefix, module in (("vision.", vit), ("aligner.", aligner)):
        state = {}
        for shard in sorted({f for k, f in index.items() if k.startswith(prefix)}):
            with safe_open(str(Path(checkpoint) / shard), "pt") as f:
                state.update({k[len(prefix):]: f.get_tensor(k) for k in f.keys() if k.startswith(prefix)})
        module.load_state_dict(state)
    patches, vh, vw, th, tw = mods["image_processor"].load_image({"url": str(image)}, args)
    with torch.inference_mode():
        want = aligner(vit(patches, vh, vw), vh, vw).float().numpy()
    r, w = os.pipe()
    proc = subprocess.Popen(["build/myds4", "-m", checkpoint, "--image", str(image), "--vision-fd", str(w)],
                            pass_fds=(w,), stderr=subprocess.DEVNULL)
    os.close(w)
    with os.fdopen(r, "rb") as f:
        raw = f.read()
    if proc.wait():
        sys.exit("myds4 failed")
    got = (np.frombuffer(raw, np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(-1, cfg["dim"])
    if got.shape != want.shape:
        sys.exit(f"myds4 wrote {got.shape[0]} rows, the reference {want.shape[0]} ({th} x {tw} tokens)")
    rows = [snr(want[i], got[i]) for i in range(want.shape[0])]
    print(f"vision: {vh} x {vw} patches, {th} x {tw} tokens; feature SNR {snr(want, got):.1f} dB "
          f"(rows min {min(rows):.1f}, median {np.median(rows):.1f}); {int((want == got).all(1).sum())} of "
          f"{want.shape[0]} rows equal")
    return snr(want, got) >= 30


def model_check(checkpoint, cache=None):
    """With `cache`, the reference's results are kept in that file and reused."""
    import numpy as np
    # a prompt and a teacher-forced continuation; both decode it one token at a time
    text = ("<｜begin▁of▁sentence｜><｜User｜>Explain in two sentences why the sky is blue."
            "<｜Assistant｜></think>The sky looks blue because sunlight")
    ids = subprocess.run(["build/myds4", "-m", checkpoint, "-p", text, "--tokenize"],
                         capture_output=True, text=True, check=True).stdout.strip()
    ids = [int(v) for v in ids.split(",")]
    n, L, S, V = len(ids), 40, 4 * 5120, 129280
    r, w = os.pipe()
    proc = subprocess.Popen(["build/myds4", "-m", checkpoint, "--tokens", ",".join(map(str, ids)),
                             "--cache", "16", "--dump-fd", str(w)], pass_fds=(w,), stderr=subprocess.DEVNULL)
    os.close(w)
    f = os.fdopen(r, "rb")
    got = {"attn": np.zeros((n, L, S), np.float32), "out": np.zeros((n, L, S), np.float32),
           "experts": np.zeros((n, L, 6), np.int32), "logits": np.zeros((n, V), np.float32),
           "engram": np.zeros((n, 2, S), np.float32)}
    for p in range(n):
        for l in range(L):
            if l in (1, 14):
                got["engram"][p, int(l == 14)] = np.frombuffer(f.read(S * 4), np.float32)
            got["attn"][p, l] = np.frombuffer(f.read(S * 4), np.float32)
            got["out"][p, l] = np.frombuffer(f.read(S * 4), np.float32)
            got["experts"][p, l] = np.frombuffer(f.read(24), np.int32)
            f.read(24)
        got["logits"][p] = np.frombuffer(f.read(V * 4), np.float32)
    if proc.wait():
        sys.exit("myds4 failed")
    if cache and os.path.exists(cache):
        want = dict(np.load(cache))
    else:
        want = reference_run(checkpoint, ids)
        if cache:
            np.savez_compressed(cache, **want)
    for e in range(2):
        print(f"Engram layer {(1, 14)[e]} SNR per position (dB):",
              " ".join(f"{snr(want['engram'][p, e], got['engram'][p, e]):.0f}" for p in range(n)))
    for l in range(3):
        for k in ("attn", "out"):
            print(f"layer {l} {k} SNR per position (dB):",
                  " ".join(f"{snr(want[k][p, l], got[k][p, l]):.0f}" for p in range(n)))
    print("layer  attention SNR (min/median dB)  layer SNR (min/median dB)  routing equal")
    for l in range(L):
        a = [snr(want["attn"][p, l], got["attn"][p, l]) for p in range(n)]
        o = [snr(want["out"][p, l], got["out"][p, l]) for p in range(n)]
        same = sum(sorted(want["experts"][p, l]) == sorted(got["experts"][p, l]) for p in range(n))
        print(f"{l:5d}  {min(a):12.1f} {np.median(a):7.1f}  {min(o):17.1f} {np.median(o):7.1f}  {same:6d}/{n}")
    agree = sum(int(want["logits"][p].argmax()) == int(got["logits"][p].argmax()) for p in range(n))
    print("logit SNR per position (dB):", " ".join(f"{snr(want['logits'][p], got['logits'][p]):.0f}" for p in range(n)))
    print(f"argmax agreement: {agree} of {n} positions")
    return agree >= n - 1


def reference_run(checkpoint, ids):
    import numpy as np
    torch, model, args, blocks, hashing, embed, norm, head = reference(checkpoint, 64)
    n, L, S = len(ids), 40, 4 * 5120
    want = {"attn": np.zeros((n, L, S), np.float32), "out": np.zeros((n, L, S), np.float32),
            "experts": np.zeros((n, L, 6), np.int32), "logits": np.zeros((n, 129280), np.float32),
            "engram": np.zeros((n, 2, S), np.float32)}
    seen = []
    post = model.Block.hc_post
    model.Block.hc_post = lambda self, *a: seen.append(post(self, *a)) or seen[-1]
    routes = {}
    for b in blocks:
        b.ffn.gate.register_forward_hook(lambda m, i, o, l=b.layer_id: routes.__setitem__(l, o[1]))
    with torch.no_grad():
        for pos, token in enumerate(ids):
            t = torch.tensor([[token]])
            h = embed[t].unsqueeze(2).repeat(1, 1, args.hc_mult, 1)
            pre = model.make_identity_pre_mix(h, args.hc_mult)
            hashes = hashing(t, pos)
            for b in blocks:
                if b.engram is not None:
                    e = args.engram_layer_ids.index(b.layer_id)
                    h = b.engram(h, hashes[:, :, e, :])
                    want["engram"][pos, e] = h.float().reshape(-1).numpy()
                seen.clear()
                h, pre = b(h, pos, pre, None)
                want["attn"][pos, b.layer_id] = seen[0].float().reshape(-1).numpy()
                want["out"][pos, b.layer_id] = h.float().reshape(-1).numpy()
                want["experts"][pos, b.layer_id] = routes[b.layer_id].reshape(-1).numpy()
            x = blocks[-1].hc_pre(h, pre).float()
            x = (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + args.norm_eps) * norm.float()).bfloat16()
            want["logits"][pos] = torch.nn.functional.linear(x.float(), head).reshape(-1).numpy()
            print(f"reference: position {pos + 1}/{n}", file=sys.stderr)
    return want


def server(url, key_file):
    """A function that calls a running server's API with its key."""
    key = Path(key_file).expanduser().read_text().strip()

    def call(path, body=None):
        r = urllib.request.Request(url.rstrip("/") + path, json.dumps(body).encode() if body else None,
                                   {"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
        with urllib.request.urlopen(r, timeout=7200) as f:
            return json.load(f)
    return call


def settled(call):
    """The server's status 3 to 4.5 s after a request ended, when its
    footprint has settled (it falls for about 2 s after a request). Sends a
    short request when none has just ended."""
    while True:
        s = call("/v1/status")
        done = s["totals"]["requests"]
        if s["phase"] != "idle" or s["waiting"] or (done and s["phase_seconds"] < 3):
            time.sleep(0.25)
        elif not done or s["phase_seconds"] >= 4.5:
            call("/v1/chat/completions", {"model": "deepseek-v4.1-flash", "max_tokens": 4,
                                          "messages": [{"role": "user", "content": "Say hello."}]})
        else:
            return s


def free_gib():
    """Free memory as myds4 counts it: free and speculative pages."""
    out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True).stdout
    page = int(out.split("page size of ")[1].split()[0])
    pages = {k: int(v.strip(" .")) for k, v in (l.split(":") for l in out.splitlines()[1:] if ":" in l)}
    return (pages["Pages free"] + pages["Pages speculative"]) * page / 2**30


def idle(url, key_file):
    """While no other program needs memory, an idle server keeps its experts:
    from 3 s to 10 s after a request, its footprint (the memory the kernel
    may not reclaim) and its expert slots stay as they were. Free memory must
    stay above the server's 10 GiB reserve meanwhile."""
    call = server(url, key_file)
    for _ in range(20):
        s = settled(call)
        time.sleep(10 - s["phase_seconds"])
        t = call("/v1/status")
        if t["phase"] != "idle" or t["waiting"] or t["totals"]["requests"] != s["totals"]["requests"]:
            continue  # another request came
        if free_gib() < 11:
            print(f"idle: only {free_gib():.1f} GiB free, near the reserve: no result")
            return False
        a, b = s["memory"]["footprint_gib"], t["memory"]["footprint_gib"]
        n, m = s["expert_cache"]["slots"], t["expert_cache"]["slots"]
        print(f"idle: footprint {a:.1f} GiB 3 s after a request, {b:.1f} GiB after 10 s; "
              f"expert slots {n}, then {m} (both expected to stay)")
        return a - b < 1.0 and m == n
    print("idle: the server did not stay idle for 10 s in 20 tries")
    return False


def demand(url, key_file):
    """When another program takes memory, an idle server gives expert slots
    back: the check takes memory until only 7 GiB is free, 3 GiB below the
    server's reserve, holds it for 5 s and lets it go. Meanwhile the server
    must give back at least 1 GiB of slots."""
    call = server(url, key_file)
    s = settled(call)
    take = int((free_gib() - 7) * 2**30)
    if take <= 0:
        print("demand: less than 7 GiB is free already")
        return False
    held = bytearray(take)
    held[::16384] = b"\x01" * len(range(0, take, 16384))  # every page written
    time.sleep(5)
    t = call("/v1/status")
    del held
    a, b = s["expert_cache"]["gib"], t["expert_cache"]["gib"]
    print(f"demand: took {take / 2**30:.1f} GiB; expert slots {s['expert_cache']['slots']} ({a:.1f} GiB), "
          f"5 s later {t['expert_cache']['slots']} ({b:.1f} GiB): {a - b:.1f} GiB given back, at least 1.0 expected")
    return a - b >= 1.0


def placeholder(url, key_file):
    """The image placeholder's text in a message or a tool result is text: the
    server answers such requests (it refused them), and the placeholder adds
    the tokens of its characters to the prompt, not one image token."""
    call = server(url, key_file)
    mark = "<｜deepseek_image｜>"

    def prompt_tokens(text):
        body = {"model": "deepseek-v4.1-flash", "max_tokens": 1, "reasoning_effort": "none",
                "messages": [{"role": "user", "content": "Read the file."},
                             {"role": "assistant", "content": "",
                              "tool_calls": [{"id": "c1", "type": "function",
                                              "function": {"name": "read", "arguments": "{}"}}]},
                             {"role": "tool", "tool_call_id": "c1", "content": text},
                             {"role": "user", "content": text}]}
        try:
            return call("/v1/chat/completions", body)["usage"]["prompt_tokens"]
        except urllib.error.HTTPError as e:
            print(f"placeholder: the server refused the request: {e.code} {e.read().decode(errors='replace')}")
            return None

    plain, marked = prompt_tokens("IMAGE = 'x'"), prompt_tokens(f"IMAGE = '{mark}'")
    if plain is None or marked is None:
        return False
    print(f"placeholder: {plain} prompt tokens with 'x', {marked} with the placeholder's text twice "
          f"(more than 2 more expected: its characters, not one image token each)")
    return marked - plain > 2


def view():
    """The live view writes at most a line break fewer than the terminal's
    rows: a line break after the last row scrolls the terminal, and the
    view's first line (the title) leaves the screen. Runs tools/view_check.c
    with src/tui.c in a 24-row pseudo-terminal, then presses q: the program
    must stop within 2 s with the normal screen back and the stop logged."""
    import fcntl, pty, select, struct, tempfile, termios, time
    root = Path(__file__).resolve().parent.parent
    flags = subprocess.run(["pkg-config", "--cflags", "yyjson", "libjpeg"], capture_output=True, text=True,
                           check=True).stdout.split()
    rows, cols = 24, 100
    with tempfile.TemporaryDirectory() as d:
        exe = str(Path(d) / "view_check")
        subprocess.run(["clang", "-std=c23", "-O1", "-Isrc", *flags, "tools/view_check.c", "src/tui.c", "src/util.c",
                        "-o", exe], cwd=root, check=True)
        master, slave = pty.openpty()
        fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))
        p = subprocess.Popen([exe], stdin=slave, stdout=slave, stderr=slave)
        os.close(slave)
        out, start, sent, at = b"", time.time(), None, 0
        while select.select([master], [], [], 5)[0]:
            try:
                chunk = os.read(master, 65536)
            except OSError:  # the program ended
                break
            if not chunk:
                break
            out += chunk
            if sent is None and time.time() - start > 1.3:  # two or three frames, then q
                os.write(master, b"q")
                sent, at = time.time(), len(out)
        os.close(master)
        p.wait()
        took = time.time() - sent if sent else float("inf")
    frames = [f.split(b"\x1b[?1049l")[0] for f in out.split(b"\x1b[H")[1:]]
    breaks = [f.count(b"\n") for f in frames]
    titled = all(f.startswith(b"\x1b[7m") for f in frames)
    stopped = p.returncode == 0 and took < 2 and b"\x1b[?1049l" in out[at:] and b"q: stopping" in out[at:]
    print(f"view: {len(frames)} frames in a {rows}-row terminal, line breaks per frame {sorted(set(breaks))} "
          f"(at most {rows - 1} expected); every frame starts with the title: {titled}; "
          f"q stopped it in {took:.2f} s with the normal screen back and the stop logged: {stopped}")
    return bool(frames) and max(breaks) <= rows - 1 and titled and stopped


if __name__ == "__main__":
    if len(sys.argv) in (3, 4) and sys.argv[1] == "model":
        sys.exit(0 if model_check(*sys.argv[2:]) else 1)
    if len(sys.argv) == 2 and sys.argv[1] == "view":
        sys.exit(0 if view() else 1)
    if len(sys.argv) == 4 and sys.argv[1] in ("idle", "demand", "vision", "placeholder"):
        sys.exit(0 if globals()[sys.argv[1]](*sys.argv[2:]) else 1)
    if len(sys.argv) != 3 or sys.argv[1] not in ("tok", "encoding"):
        sys.exit(__doc__)
    sys.exit(0 if globals()[sys.argv[1]](sys.argv[2]) else 1)
