# Qwen3.8 models

myds4 also runs Qwen3.8-27B in NVIDIA's NVFP4 build (the Model Optimizer
checkpoint of `Qwen/Qwen3.8-27B`, 20 GB, `model_type` `qwen3_5`), fully
resident on the GPU. It is one model family of the engine:
[`src/model/qwen.c`](../src/model/qwen.c) fills the model interface of
[`src/model/model.h`](../src/model/model.h), as DeepSeek's does, and its
kernels are in `src/kernels/` (`rows.metal`, `products.metal`,
`deltanet.metal`, `gqa.metal`), compiled into the one library with
DeepSeek's. `build/myds4 -m DIR` runs it; the server serves it alone or
beside DeepSeek ([the server](server.md#models)). Qwen3.8-Flash-Next
(`qwen4_exp`) is not supported.

## The model

| Part | Shape | Weights |
| --- | --- | --- |
| Embedding | 248,320 x 5,120 | BF16 |
| 64 layers | 48 gated DeltaNet layers, a full-attention layer every 4th (`layer_types`) | |
| Gated DeltaNet | 16 key heads and 48 value heads of 128; causal depthwise convolution (kernel 4) over q, k and v; FP32 state of 48 x 128 x 128 a layer | q/k/v and z projections, output: FP8; a and b projections: BF16 |
| Full attention | 24 query heads, 4 key-value heads of 256; per-head RMSNorm of q and k; rotary embedding over the first 64 dimensions (theta 10^7); output times sigmoid of a per-head gate | FP8 |
| MLP | SwiGLU, 17,408 wide | NVFP4 |
| Head | 248,320 x 5,120, untied | NVFP4 |
| MTP head | `fc` (10,240 to 5,120), one full-attention layer and SwiGLU MLP, its norm; the embedding and head are the model's | BF16 |

- **NVFP4:** E2M1 values (two a byte, low nibble first), an E4M3 scale per 16
  values, and an FP32 scale per matrix. **FP8:** E4M3 with an FP32 scale per
  matrix. The engine uses the weights as stored and keeps activations in
  FP32; it does not quantize activations (the checkpoint's `input_scale`
  values are not read).
- **Norms:** the checkpoint stores the RMSNorm weights centered at zero
  (input, post-attention, final, q and k norms, and the MTP head's): the
  engine adds 1 at load. The DeltaNet's gated output norm is used as stored.
- **The math** follows the reference implementation (MLX's `qwen3_5`, and
  Hugging Face's Qwen3.5 code for the MTP head):
  - DeltaNet: the convolution's output through SiLU; q and k L2-normalized
    (eps 10^-6 inside the root), q also divided by sqrt(128); beta =
    sigmoid(b); the decay g = exp(-exp(A_log) * softplus(a + dt_bias)); per
    token S = g S, then S += k (beta (v - S k)), output S q; then RMSNorm of
    each head times silu(z), and the output projection.
  - Attention: the query projection gives each head's query and gate; scores
    scaled by 1/16; the multimodal rotary sections reduce to plain rotary
    embedding for text, whose three position ids are equal.
  - A layer: h += mixer(norm(h)); h += down(silu(gate(n)) * up(n)).

## Sequences

A state (`qwen_state_new`) holds one sequence:

- the DeltaNet states and the convolution inputs (157 MB), twice: the
  current ones and a spare copy that a pass leaves after a marked row
  (`QwenRow.save`), which `qwen_state_back` returns to;
- the keys and values of the 16 attention layers and the MTP head's layer,
  FP16, 68 KB a position; their room doubles from 1,024 positions as the
  sequence grows (`qwen_state_reserve`);
- the hidden of its last row (and of the saved row), the MTP head's input.

A snapshot (`qwen_snap_take`, 157 MB) holds the recurrent part and the last
hidden at the state's position. A DeltaNet state cannot go back, so the
server resumes a prompt in place (the state the prompt extends), from a
snapshot (with the source's keys before it, `qwen_state_copy`), or from
nothing; see [the server](server.md#chat-formats).

## Passes

A pass runs rows of one or several states, a state's rows consecutive and at
its next positions, in one command buffer:

- **Decoding passes** (`qwen_forward`, 1 to 8 rows): the matrices run with
  one simdgroup a weight row (`mv1` for one row, `mvr` for 2 to 8, each
  decoded weight chunk serving every row). Every row's arithmetic is the
  same for any number of rows, so a row's logits do not depend on the pass:
  `build/myds4 --check-batch` finds no logit differing between two rows
  decoded together and each alone.
- **Prompt passes** (`qwen_prefill`, 1 to 512 rows): the matrices run on the
  GPU's matrix units (Metal tensor operations, as the DeepSeek engine's
  prompt kernels do), from BF16 tiles of the dequantized weights, which are
  exact in BF16, and of the inputs rounded to BF16, with FP32 sums. The
  units add in an order that does not depend on the tile or the number of
  rows, so a prompt's logits are the same however it is split into passes
  (`--check-prefill`: none differ between one prompt pass and two).
  Attention runs on the matrix units too when the rows are one state's
  consecutive positions (`gqa_prompt`): a threadgroup takes 32 rows of a
  query head and goes through the keys in chunks of 32 counted from
  position 0, with FP16 queries, keys, values and probabilities, FP32 sums
  and an online softmax per row. Prompt passes differ from decoding passes
  by these roundings: on a 1,293-token prompt the largest next-token logit
  difference was 0.10, with the same greedy token.
- **The rest of a layer**, in both: the convolution and the delta rule run
  each state's rows in order within the pass. The residual's add and the
  next RMSNorm run as one kernel (`add_norm`), and in decoding passes the
  MLP's gate and up products and SwiGLU run as one (`gate_up`); each gives
  the separate kernels' bits (a build without them gave the same logits,
  byte for byte, from both kinds of pass). A decoding pass's attention
  (and a prompt pass's whose rows are not one state's) runs in blocks of 256
  positions (`gqa_block`: each block's maximum score, sum and weighted
  values; `gqa_combine`: the blocks combined), 8 rows at a time; within a
  block each simdgroup reads whole key and value rows, its lanes 8
  dimensions each. A prompt pass's attention of one state's rows is
  `gqa_prompt`; its products are `gemm_nvfp4`, `gemm_fp8` and `gemm_bf16`
  on inputs that `x_bf16` rounds, and a decoding pass's are `mv1_*` and
  `mvr_*`.

## MTP drafts

The MTP head proposes the token after the next one. Its input at position p
is the hidden of position p (the last layer's output before the final norm)
and the token at p + 1, at p's rotary position; its keys go to its own
attention layer at p. Prompt passes give it the keys of their positions
(each pass's last hidden waits in the state for the next token).

Decoding with drafts (`--draft 1`; the model interface's draft, verify and
accept): the draft is the MTP head's most likely token; a verify pass runs
the sampled token (its state saved after it) and the draft; the target
samples from the first row's logits; if that token is the draft, the draft's
row is kept and the second row's logits give one more token; otherwise the
state returns to the saved row. The target samples every token it keeps
once, in order, from logits that equal one-row decoding's
(`build/myds4 --check-batch --draft 1`: a verify pass's first row against the
token alone), so greedy and seeded sampled replies are the same with drafts
or without.

While drafts are on, every state keeps the MTP head's keys complete: each
decoding pass and verify pass first runs the head over the keys the state
owes (its last row's hidden with the token now run, and after a kept draft
the saved row's too), and prompt passes fill the keys of their positions.
Without drafts the head does not run.

## Speed

Measured on 2026-11-07 with the passes and kernels described above, on the
M5 Max (128 GiB), on AC power with the GPU otherwise idle, 30 s apart
(back-to-back runs heat the GPU and slow by up to about 15%). Model:
Qwen3.8-27B NVFP4 (all weights resident). Prompts: the first 4,000
characters of `src/engine/io.c` (1,293 tokens) and 26,000 characters of
DeepSeek's model source (9,102 tokens), each continued as code, one request,
greedy; a request for a short story (28 tokens). The generated text was
checked. Through `build/myds4`, `--perplexity --prefill 1` on the io.c
prompt decodes one row a pass at 25.6 tokens/s, as in the table.

| Workload | Speed |
| --- | --- |
| Prompt passes, 1,293 tokens | 653-666 tokens/s |
| Prompt passes, 9,102 tokens | 650 tokens/s |
| Decoding, one row, short context | 27.1 tokens/s (36.8 ms a pass: 17.6 GB of weights, about 480 GB/s) |
| Decoding at 1.3K and 9.1K context | 25.6 and 25.9 tokens/s |
| Decoding passes of 2, 4, 8 rows | 40.9, 59.0, 122.3 ms (48.9, 67.7, 65.4 tokens/s together) |
| With MTP drafts: the io.c continuation, the story, the model.c continuation | 42.6, 37.3 and 38.7 tokens/s; 98%, 72% and 90% of drafts accepted (25.6, 26.4 and 25.9 without) |

A 9,164-token prompt that asks at its end for a fact given at its start got
the fact back.

A decoding pass's extra rows cost about 4 to 12 ms each: the matrix kernels
read each row's inputs for every weight row, so more rows soon cost more
arithmetic and cache traffic than the weights' reads.

## Checks and the reference

`build/myds4 -m DIR` takes the prompt as text (`-p` or `-f`, as one user
turn of the checkpoint's chat template, thinking off unless `--think` is 1
to 100; `--raw` for the text as is) or as token ids (`--tokens ID,ID,...`
or `--tokens @FILE`, commas, spaces or newlines between them). The checks:

| Option | Effect |
| --- | --- |
| `--check-batch [--draft 1]` | Two sequences' next tokens decoded together against each alone, and a verify pass's first row against the token alone: no logit may differ. |
| `--check-prefill` | Two prompt passes against one (no final logit may differ), and every position's prompt-pass logits against one-row decoding's (argmax agreement, SNR). |
| `--perplexity --prefill N [--logits-fd FD]` | The first N tokens in prompt passes, then one row a pass; the perplexity of the rest, and every scored row's logits on FD. |

The reference is Hugging Face's own model code (transformers'
`Qwen3_5ForCausalLM`) on the CPU in FP32, with the checkpoint's weights
dequantized exactly as stored: `tools/reference/qwen_hf.py` (torch and
transformers in a virtual environment outside the repository; its header
says how). It writes token ids from the checkpoint's tokenizer and chat
template, and compares its logits with the stream of `--perplexity
--logits-fd` at the same positions. On 2026-11-07, with the first 4,000
characters of `src/engine/io.c` (1,293 tokens):

| Engine passes | max diff | KL (nats) | SNR | top-1, top-10 |
| --- | ---: | ---: | ---: | --- |
| Decoding, at position 1,292 | 0.011 | 1.4e-6 | 65.0 dB | same, 10 of 10 |
| Decoding, every position 0 to 1,292 (the prompt and one more token) | 1.089 | mean 3.5e-7, max 9.1e-5 | mean 72.0, min 24.9 dB | same at all 1,293, 10 of 10 |
| Prompt passes, at position 1,292 | 0.107 | 8.6e-5 | 47.4 dB | same, 10 of 10 |

and for a 22-token chat prompt (a story request, thinking off): 0.0013,
2.1e-9 nats, 81.3 dB (decoding) and 0.056, 1.8e-6 nats, 48.4 dB (prompt
passes), the same top tokens. The perplexity of the 1,292 predictions is
4.5469 from decoding passes and 4.5471 from the reference. Decoding passes
differ from the reference by FP32 rounding; prompt passes round their inputs
to BF16 and attention's operands to FP16.

## Limitations

- Text only: the checkpoint's vision tower is not loaded, and requests with
  images are refused.
- The CLI's chat turn is one user message with thinking on (`--think` 1-100)
  or off (0), without tools or a system message.
- The MTP head drafts one token; drafts are used while one request decodes
  (see [the server](server.md#chat-formats)).
- While it serves, the model's weights stay resident on the GPU (20 GB, plus
  the states and snapshots). Those weights share the GPU residency budget
  with DeepSeek's mapped experts. Unmapped expert slots remain in RAM;
  allocated and GPU-mapped cache sizes differ (see [the server](server.md#models)). After
  60 s without
  requests, the server takes the model's buffers out of the GPU's residency
  set (`qwen_reside`; they stay in memory), and puts them back before the
  next pass: that took no measurable time, and the first reply after it
  decoded at 21.8 tokens/s.
