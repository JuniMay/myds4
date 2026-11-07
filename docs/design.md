# Design

## Files

The C sources are in five folders:

- `src/engine/`: what every model shares: the GPU layer, SSD reads and the
  routed-expert cache, the tokenizer, sampling, images, the JSON and
  safetensors readers, and the C ABI for the server (`engine.c`).
- `src/model/`: the models. Each family is one file (`deepseek.c`, `qwen.c`)
  that fills the model interface of `model.h`; `model.c` opens a checkpoint
  with the family whose model types include its `config.json`'s
  `model_type`. The CLI, the engine and the checks run every model through
  that interface.
- `src/kernels/`: every model's Metal kernels, compiled into one library per
  process (`library.c`).
- `src/cli/`: the command line (`main.c`) for generation, the checks and the
  benchmarks, for any model.
- `src/experimental/`: optional features. `build/myds4` and the server
  include the online predictor, context lookup, the prefill-pruning plan and
  runahead (an option of the server), each off unless its option is given.

The Makefile adds `-Isrc/engine`, so C files in every folder include the
engine's headers by name.

One diagnostic build remains beside the ordinary ones.
`build/myds4-predict-probe` (`make predict-probe`, switch
`MYDS4_PREDICT_PROBE`, capture code in `tools/experimental/predict_probe.c`)
is a diagnostic capture tool: it writes records of the routing, the
predictions, DSpark's stages and the cache to the file descriptor that
`MYDS4_PROBE_FD` names, normally a pipe, and `tools/analysis/expert_probe.py`
reads them. Its writes are synchronous and change the timing, so it is not
for speed measurements. Studies that were measured and removed are recorded
in [Removed experiments](experiments.md).

| File | Contents |
| --- | --- |
| `src/model/model.h`, `src/model/model.c` | The model interface: limits, states, snapshots, prefill, decoding passes, drafts and their verification, the CLI's chat turn and report; and opening a checkpoint by its `model_type`. |
| `src/model/deepseek.c` | DeepSeek V4.1 Flash: weights, request state, the forward pass for decoding and prefill, DSpark, the vision encoder, `--bench`. |
| `src/model/qwen.c`, `src/model/qwen.h` | Qwen3.8 (`qwen3_5`): weights, states with their DeltaNet and key-value rows, prompt and decoding passes, the MTP head (see [Qwen3.8 models](qwen.md)). |
| `src/engine/io.c` | SSD reads, the routed-expert cache, memory monitoring. |
| `src/engine/gpu.c` | Metal through the Objective-C runtime from C, the profiler. |
| `src/engine/tok.c` | A checkpoint's tokenizer, one per model: byte-level BPE after the `tokenizer.json`'s normalizer (none or NFC) and its sequence of Split regexes, which run in the system's ICU library (`libicucore`); a pipeline with other steps is refused. |
| `src/engine/util.c` | Errors, strings, a JSON parser for the model's own files, the safetensors index, GGUF headers. |
| `src/engine/vision.c` | Images: decoding, the reference's grid, Pillow's resize and padding, the vision encoder's patches and RoPE angles, an image's span of tokens. |
| `src/engine/sample.c` | Shared temperature/top-p sampling and the verification of drafts. |
| `src/engine/myds4.h` | The engine's shared declarations: the model's shape, the options, and the GPU, cache, tokenizer and model interfaces. |
| `src/experimental/context_lookup.c`, `src/experimental/context_lookup.h` | Bounded CPU history matching for the CLI and server's context-copy drafts. |
| `src/experimental/prefill_prune.h` | The dependency plan of `--prefill-prune`. |
| `src/experimental/runahead.c` | `--runahead`'s scheduler for concurrent requests; included by `deepseek.c`. |
| `src/experimental/moe_tasks.metal` | Standalone prototype that dispatches ready MXFP4 expert stages from different layers together; used only by `build/moe-tasks-check`. |
| `src/engine/engine.c`, `src/engine/engine.h` | The engine behind a C ABI for the server: loading, attention states and their snapshots, prefill (with a replay: exact, or bounded under `--ced`, and a capture position), decoding, drafts, sampling, context-copy drafts, images, statistics, the tokenizer and the log. Requests, their scheduling and the KV prefix cache are the server's (see [The server](server.md)). |
| `src/kernels/library.c` | Every kernel source in one string, compiled once per process (`gpu_start`). |
| `src/kernels/formats.metal` | The number formats every source decodes (E4M3, E2M1, BF16) and small shared functions. |
| `src/kernels/kernels.metal` | DeepSeek's kernels. |
| `src/kernels/gguf.metal`, `src/kernels/gguf_grids.metal` | The kernels for routed experts in GGUF codes (`--experts`), and their codebook grids, generated from llama.cpp's `ggml-common.h` (MIT License). |
| `src/kernels/vision.metal` | The vision encoder's kernels. |
| `src/kernels/rows.metal`, `products.metal`, `deltanet.metal`, `gqa.metal` | Operations on activation rows; products of matrices with one scale each (NVFP4, FP8, BF16); the gated DeltaNet; grouped-query attention over FP16 keys and values. |
| `src/cli/main.c` | Command line for any model: the model's chat turn, sampling, statistics, `--check-prefill`, `--check-batch`, `--perplexity`. |
| `src/experimental/online_predictor.c`, `src/experimental/online_predictor_math.h` | Online CPU prefetch correction, owned forecast/label records, recursive least-squares updates and learner metrics. |
| `server/` | The server, in Rust on `deepseek-recipe`: HTTP, request conversion, prompts, output and the live view (`--tui`; see [The server](server.md)). |
| `tools/checks/` | Regression checks: C programs that the Makefile builds (most need no model or GPU), Python checks of the command line and the server, and `regression.sh`, which compares two CLI builds' outputs and decode speed. |
| `tools/reference/` | `check.py`: myds4 against the checkpoint's references (tokenizer, the server's prompts, vision encoder, model), and checks of a running server. `gguf_decode_check.c`: IQ1_M decoding against llama.cpp's. |
| `tools/bench/` | Serial benchmarks: `benchmark_suite.py` (GSM8K, HumanEval and context prompts), `compare_bench.py` (a control against a candidate build), `ced_bench.py` (CED and DSpark studies), `session_bench.py` (concurrent requests through the server); and `gpu_guard.py`, the GPU, memory and power guard that they and the other model runners share. |
| `tools/analysis/` | `cachesim.py`, the expert-cache simulator for routing traces, and `expert_probe.py`, the reader of `build/myds4-predict-probe` captures. |
| `tools/data/` | `gen_data.py`, which generates the embedded tables in `data/` from the checkpoint. |
| `tools/experimental/` | Checks and tools of the optional code: the online predictor's checks, the context-lookup replay (`context_lookup_replay.c`), and the capture code of `build/myds4-predict-probe` (`predict_probe.c`). |

Metal is used from C: every message goes through `objc_msgSend` cast to the
method's exact type, so the project has no Objective-C or C++ sources besides
the Metal kernels. The kernels compile when the first model loads, all
sources as one library (the shared formats first), with the IEEE math mode
and Metal 4.1, which provides the tensor operations that drive the GPU's
matrix units. Kernel names are unique across the sources: a kernel that
computes something else under a name another source uses takes a name of its
own (`gemm_*` for products of matrices with one scale each, beside the
block-scaled `mm_fp8`).

## Weights

myds4 reads the checkpoint as DeepSeek stores it, without conversion:

| Tensors | Format | Where |
| --- | --- | --- |
| Attention projections, shared experts, indexer query projections, Engram's `wkv` | FP8 E4M3, one UE8M0 scale per 32 x 32 block | GPU memory, one buffer per tensor |
| Router, compressors, indexer key and weight projections, head | BF16 | GPU memory |
| Norms, sinks, router biases, hyper-connection weights | BF16 or FP32, kept as FP32 | GPU memory |
| Routed experts `w1`, `w2`, `w3` | MXFP4: two E2M1 codes per byte, low nibble first, one UE8M0 scale per 32 | read into cache slots on demand |
| Engram tables | E4M3 rows of 256 with UE8M0 scales | 24 rows per token per Engram layer, read on demand |
| Embedding | BF16 | one row per token, read on demand |
| DSpark stages `mtp.0` to `mtp.2` (with `--draft`) | as the layers above; 128 routed experts each | dense parts in GPU memory, routed experts in the cache (as layers 40 to 42) |

All reads use `F_NOCACHE` over aligned spans: unaligned uncached reads still
fill the page cache, so a payload lands in its buffer at the offset of its
alignment. An expert's three code tensors are contiguous in the file
(17.7 MB), as are its three scale tensors (1.1 MB): two 16 KiB-aligned reads
per expert, split into 2 MiB pieces that are read in parallel.

### Experts from GGUF files

With `--experts`, the 40 layers' routed experts come from GGUF files in
llama.cpp's layout instead of the checkpoint; the dense weights, the Engram
tables and the DSpark stages stay the checkpoint's. A layer's experts are
three tensors, `ffn_gate_exps` (`w1`), `ffn_up_exps` (`w3`) and
`ffn_down_exps` (`w2`), each holding all 384 experts' rows, an expert's rows
contiguous: three reads per expert. Each tensor is in one of llama.cpp's
codes of blocks of 256 values: Q2_K, IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S
or IQ3_XXS, a layer's `w1` and `w3` in the same one. unsloth's UD-IQ2_XXS
build uses all of them but IQ1_M, with experts of 7.6 to 12.1 MB;
ISTA-DASLab's GSQ-RCO IQ1_M build uses IQ1_S, IQ1_M, IQ2_XXS and IQ2_S,
with experts of 6.9 to 8.7 MB; MXFP4 experts are 18.9 MB. A cache slot
holds the largest of a file's experts. myds4 opens every shard of a split
set that is present, so the shard of Engram tables need not be downloaded.
If the file has an output head in Q6_K (`output.weight`, rows in the
checkpoint's token order), it replaces the checkpoint's BF16 head: 543 MB
read per decoded row instead of 1.32 GB, and 0.8 GB less memory. A head in
another code, such as the GSQ-RCO build's Q5_K, leaves the checkpoint's.

The kernels decode a sub-block of 32 values into the values that llama.cpp's
`dequantize_row_*` compute. A threadgroup first builds its code's tables in
threadgroup memory: the IQ2 and IQ3 grids with each value as half, and the
eight signs of each sign index (of each sign byte for IQ2_S) as half sign
bits, so that eight values take two or three table loads and an XOR. IQ1_S
and IQ1_M keep llama.cpp's grid of nibbles; masking turns their codes and
Q2_K's into half subnormals, a code n into n 2^-24 exactly. A code times
its scales is exact in FP32, so the decoded values equal llama.cpp's. IQ1_M
has no scale field: its FP16 block scale is the top nibbles of its four
16-bit scale words, whose low 12 bits hold a 3-bit scale for each 16 values,
and each 8 values take a nibble of `qh`, three index bits and the sign of
their offset. The kernels keep the
checkpoint's FP8 rounding of the experts' input and intermediate rows.
Decoding multiplies the decoded values with the input in FP32. A prompt's
kernels decode a tile of 128 rows (64 columns of `w1` and the same 64 of
`w3`, or 128 of `w2`) by 64 values into threadgroup memory as half, for the
matrix units, which read the pairs' rows of the input as half from device
memory, in pair order. A threadgroup takes up to 64 pairs of one expert, so
that each decoded tile serves all of them, in products of 16 pairs (the up
projection: `w1` and `w3` together) or 32 (`w2`); the threads load and
decode the next tile's codes into registers while the products of this one
are issued. The tile leaves 16 KB of the 32 KB of threadgroup memory, too
little for IQ2_S's tables (20 KB), so the prompt kernels keep IQ2_S's grid
in bytes.

The decoding down projection takes 8 rows of `w2` per threadgroup, two
simdgroups of 4 rows for each expert, so that each threadgroup's tables
serve twice as many rows. Lane l of a simdgroup sums sub-blocks l, l + 32
and, for l < 8, l + 64 of each of its 4 rows, in that order. Those last 8
sub-blocks of the 4 rows are decoded one per lane, row r's sub-block 64 + l
by lane 8 r + l, so that no lane decodes more than 9 sub-blocks of the 288.

Engram and embedding rows are small random reads, each in whole 4 KiB blocks
(the file system's block size, which uncached reads also take past the page
cache), from 64 lanes on GCD's global queue. Their speed is the number of
reads in flight, about 185,000 per second, so a row requested more than once
is read once and copied: a prompt repeats many of its n-grams (44% of the
rows of an 8,916-token source file).

## Numerics

With CED and substitution off and no GGUF override, the engine uses the
checkpoint's original stored weights, full layer stack and router-selected
experts. Prefetching changes when weights are read, not which experts run.
“Exact routing” describes that selection policy. A bitwise optimization check
compares two paths in this engine under the tested settings. Neither claim
establishes bit-for-bit equivalence to DeepSeek's official CUDA runtime.
The reference comparisons and precision limits below describe the known
numerical differences.

The kernels implement the operations in `inference/model.py`, including the
quantization the model was trained with: activations are quantized to E4M3
with power-of-two scales per 32 before every FP8 and FP4 product, window keys
to E4M3, compressed keys to E2M1 with E4M3 scales per 16, and indexer
queries and keys to E2M1 with power-of-two scales per 32. Values the
reference keeps in BF16 are rounded to BF16 where it rounds them; activations
are explicitly rounded at those boundaries. The four residual streams use
BF16 storage in both backbone and draft layers. Their mixing and reductions
still use FP32 arithmetic. Other activation workspaces retain FP32 storage.
The attention state's window keys, compressed
rows and index keys are stored as BF16, which holds their quantized values
exactly: an E2M1 code times an E4M3 scale has at most 6 significant bits.
Products accumulate in FP32. `tools/reference/activation_format.h` provides
CPU reference codecs for BF16 and 32-value E4M3 blocks with one E8M0 scale.
`make activation-format-check` checks finite code values, nearest-even
rounding, scale boundaries, signed zeros and the FP16 small-value limitation.
These codecs support representation checks and include the scaled-half
staging rule used by the native MXFP4 matrix kernels.

In prefill, the matrix units multiply BF16
tiles for the dense weights: the quantized activations and the dequantized
weights are both exact in BF16. For the routed experts they read the MXFP4
weights as stored. Their activation staging uses half values multiplied by
128; the matrix result is divided by 128 before BF16 rounding. The FP8
quantizer's scale floor gives a minimum step of `2^-31`, which becomes the
exact FP16 step `2^-24` after this scaling. Values up to 511.75 fit. Loading
checks the FFN RMSNorm weights against that upper bound, allowing for BF16
and FP8 rounding; the checkpoint's conservative bound is below 209. The
weighted SwiGLU output is bounded by 150 before FP8 quantization. This
preserves small activations while using Metal's supported half × MXFP4
matrix operation. GGUF matrix kernels retain their existing half staging.
Diagnostic stream dumps expand BF16 back to FP32 and keep their wire format.

`tools/reference/check.py model` decodes a 20-token prompt and continuation with
myds4 and with the reference (eager FP32 PyTorch on the CPU with the kernels'
arithmetic). The two differ by ulp-level differences between GPU and CPU
math, including the order in which products are summed. With the build of
2026-10-31, at layer 0 the streams match to a median SNR of 105 dB, the
worst position at 51.5 dB, where one BF16 or E4M3 value rounds the other
way; the routing is equal at all positions. Such a flip changes a quantized
key or an expert choice, and later layers amplify it, to 18.6 dB median
stream SNR at layer 39. The greedy tokens agree at 19 of 20 positions. Batched prefill and token-by-token
decoding (`--check-prefill`) diverge the same way: on the first 1,700
characters of the checkpoint's README (550 tokens), equal greedy tokens at
124 of the first 128 positions and 421 of the next 422, mean logit SNR 15.4
and 14.9 dB. A prompt run as two batches, the first of odd length (275 + 275
tokens), gives the same final logits as one batch, bit for bit.

## The forward pass

A batch is a run of consecutive token rows at one layer: one row when
decoding, up to 2048 rows of a prompt. Per layer:

1. Engram (layers 1 and 14): hash the n-grams ending at each token (2 to 4
   tokens, 8 heads, the table's primes and offsets), read the 24 table rows,
   dequantize, project with `wkv`, and add the gated value to the streams.
2. Hyper-connections: 24 mixes of the four streams, Sinkhorn-normalized,
   collapse the streams into the attention input (with the previous layer's
   mix) and expand the output back.
3. Attention: query and key projections, RoPE (YaRN for the compressed
   layers), the window of the last 128 keys, and for compressed layers the
   compressor (layers 2, 8, 14: groups of 2 pooled by a softmax gate; layer
   20: every position), the indexer (layers 2, 8, 14, 20, 24, 28, 32, 36:
   scores over all compressed positions, the top 512 by radix select) and
   sparse attention with sinks over the window and the selected rows; the
   other compressed layers reuse their source's rows and selection.
   Beyond 2,048 blocks of 8 positions (16,384 positions), the indexer has
   two levels (`candidate_source_layer` 20): layer 20 scores each block by
   its best position, keeps the 2,048 best and the block of the newest
   position (all those tied with the 2,048th as well), and the indexers of
   layers 24, 28, 32 and 36 select their 512 within those blocks only, as
   bits of a mask per row (`cand_topk`, then `index_topk`). Up to 2,048
   blocks every block is a candidate, so the two levels change nothing.
4. FFN: the router (sqrt-softplus scores, the bias selects 6, the unbiased
   scores weight them), the shared expert, the routed experts, and the sum
   in expert order, as the reference adds them.

The window keys of a layer live in a ring of `128 + max(8, lookup_depth)`
rows: the window plus positions a speculative pass may write for rejected
drafts. A prompt batch works on a window buffer filled from the ring and
written back. A KV source keeps 8 compressor rows normally and 32 for
wider lookup, from which a batch takes the pending group's first row.

### CED prefill

All new prompt rows run through layer 20, which produces full-context
global KV and index keys. Layers 21 through 39 run only the retained
suffix. Retained queries in layers 20 through 39 cannot read local keys
before the replay boundary. Discarded rows do not issue next-layer
expert prefetches; shorter retained batches use selective prefetching.

Ordinary builds expose `--ced off|N`, default off, through the CLI and both
servers. They retain N decoder rows plus the verification rewind margin
(`max(8, lookup_depth)`), and apply replay
only when the new input is longer than that sum. The replay boundary is
chosen across the whole forward call, including multi-chunk prompts. They
preserve all encoder/global KV work and prime DSpark only from computed
decoder inputs. Decoding runs normally after prefill. A compact prefill can
change local KV, routing and output; it is not an exact-output optimization.
The server's KV prefix cache resumes a prompt under CED by a bounded replay
(see The KV prefix cache).

`tools/checks/ced_check.c --compare-compact N` compares compact execution
with the same full-shape attention mask. `tools/checks/server_prefill_check.py`
checks the option and cache reuse through HTTP. Its shared guard in
`tools/bench/gpu_guard.py` checks GPU-client ownership, available memory, AC power
and competing processes, and stops its own workload on resource interference.
Requests, replies and diagnostic logs remain in memory.

## Decoding

A token runs as two command buffers per layer. The front holds the
attention, the FFN's mix and the router, which signals a shared event, then
the shared expert. Meanwhile the CPU waits for the signal, reads the routing,
starts reading the missing experts, starts reading the experts predicted for
the next two layers, waits for this layer's experts and encodes their
products. The next layer's front was committed before that: the experts'
command buffer was opened (enqueued) first, so it runs first, and the GPU
starts the next layer as soon as the experts are done.

A signal that a minute does not bring means that a command buffer failed
or the GPU stopped, and the results the CPU waits for are not there: the
program ends with the failed buffer's error, or with the value that was
not signaled.

When some of the layer's experts are still being read (of the token's, or of
any row of a verify pass or of several sequences), the others do not wait
for them. The experts' buffer gets the products of those in memory, the down
kernels writing each expert's part of each row's sum to a buffer of its own
instead of adding them up, and is committed at once, so the GPU works while
the SSD reads. A second command buffer, opened right after the first (and
committed empty when nothing was missing), gets the products of the experts
read meanwhile and the sum of all parts, added in expert order and rounded
as the down kernels do (`moe_parts`), so the result is the same bit for bit.

The token's other CPU work stays off the GPU's path. The embedding row is
read before layer 0, whose front embeds it. The Engram rows are read in the
background; the experts' command buffer of layer 0 (and of layer 13) is
committed only once Engram layer 1's (and layer 14's) rows are in, and
command buffers run in order, so the GPU cannot reach those layers before
their rows. The cache's bookkeeping between tokens runs while the GPU works
on layer 0's front.

Products of one row use matrix-vector kernels: a simdgroup computes 4 rows,
so that each input chunk serves 4 weight rows, and small matrices split K
over up to 8 simdgroups. With FP8 weights, each lane requests a chunk's four
weight rows before the chunk's inputs and scale, so that the reads from
memory go out first. Batches of up to 6 rows (a verify pass, a short prompt)
use their multi-row forms, which load each weight chunk once for all rows
and give each row the sums of the one-row kernels. Up to three products on
the same input that share K and the split run in one dispatch (the `*_3`
forms, one matrix per grid row, each with its own rows and output stride),
each output computed as the one-matrix kernel computes it: the router with
the two predictions, the shared expert's `w1` with its `w3`, the attention's
`wq_a` with `wkv`, and in the indexer's layers `wq_b` with the indexer's
`wq_b`. One dispatch likewise takes the three routings' top-k, a single
simdgroup picking each row's experts (the largest key, the lowest expert on
ties) without barriers. The mixes of a hyper-connection and the norm of its
input share one dispatch: the norm reads the pre weights computed for it
earlier, while a second threadgroup per row turns the mix's sums into this
block's weights, and for the attention also computes the residual part of
the FFN input. A decoded layer without a compressor or an indexer so runs 23
dispatches. The routed experts' up projection loads each input chunk once
for `w1` and `w3`, and a threadgroup, which computes 32 rows, quantizes them
to FP8 as one block; the down projection runs the 6 experts in parallel
simdgroups and sums them in expert order.

Attention of a decoded row sees up to 640 rows: the window, then the
selected compressed rows. All heads share the 512-wide key, which is also
the value. The rows form chunks of 64, in three dispatches: the scores and
each chunk's maximum per head, a threadgroup taking 4 heads and two chunks;
the probabilities against the row's maximum (summed as they are, rounded to
BF16 for the values) times the values, per chunk, a threadgroup taking 16
heads, two chunks and half of the columns; and the sum over the chunks, a
threadgroup taking one head and half of its columns. Each kernel loads into
registers ahead of use (the values' rows 8 at a time), since a load used as
soon as it is issued costs a whole memory latency (110 to 300 ns). Another
split of the work must keep the order of every sum, or the outputs change
(see Numerics): each score's lanes and `simd_sum`, a chunk's probability
sum as two halves of 32 rows, the values row by row, the chunks from the
first.

### CED and DSpark study tools

Build with `make build/ced-check build/dspark-state-check`. The CED tool
accepts `CHECKPOINT PROMPT_FILE|-`, using stdin for `-`. `--request` reads a
text-only Chat Completions request; `--chat` takes rendered prompt text.
Inputs are bounded below 16,384 tokens and remain in memory.

The default numerical mode compares full prefill with decoder local-window
masks over the same continuation tokens, with temperature/top-p 1 and seed
12345. It reports local KV error, global KV/index equality, router overlap,
logit and probability differences, and sample agreement. `--steps N` sets
1–64 continuation positions. `--compare-compact N` checks compact execution
against the corresponding full-shape mask; it does not assert equality to
full prefill.

`--bench N` measures prefill and up to N generated tokens; zero times only
prefill. `--ced off|N` uses production replay, including its verification rewind margin.
The diagnostic `--compact --keep N` instead retains exactly N decoder rows.
`--dspark N` enables 1–5 native drafts with 32 expert slots per stage.
`--cache GIB`, `--reserve GIB` and `--experts GGUF` choose cache bounds and
expert format. Sampling verification uses the main model's requested draws.

`tools/bench/ced_bench.py` reconstructs an append-only, text-only Harness
request from a supplied session path and event cutoff, using Python 3.14 or
later. Recorded tool calls are data and are never executed. It checks the
recorded input-token count, GPU ownership, memory and actual cache bounds, then
runs serial controls and candidates. Phases are `validate`, `prefill`,
`joint-prefill`, `ced-generation` and `dspark`; `all` runs numerical checks
and cold-prefill comparisons. Candidate timings require matching validation
where applicable. Output and diagnostics stay in RAM; aggregate reports are
bounded to 512 KiB and contain no private prompts or generated text.

`build/dspark-state-check CHECKPOINT [EXPERT_GGUF]` compares all three draft
windows after one through six accepted verification rows with sequential
execution of those same tokens. Its retained MXFP4 and IQ2 results matched
bit-for-bit. This checks state handling, not acceptance rate or task quality.

### Reading experts ahead

Three predictions name a layer's experts before its routing does, each
later and more accurate than the one before:

| Prediction | Made | Top-6 recall (story) |
| --- | --- | ---: |
| The layer's router on the FFN input two layers before | at that layer's routing | 69.6% |
| The layer's router on the FFN input one layer before | at that layer's routing | 75.9% |
| The layer's router on the residual part of its own FFN input | early in its front | 83.5% |

The residual part is what the FFN input holds before the attention adds to
it: with the attention's hyper-connection mix (pre a, post p, comb C), the
FFN input is the sum over streams k of a_k (p_k A + sum_j C_jk H_j), for
attention output A and incoming streams H, so its residual part is
sum_j (sum_k a_k C_jk) H_j. The front computes it, normalized with the FFN
norm, in the dispatch that ends the attention's mix, and signals the CPU
again before the attention.

Each prediction first cancels the read-aheads for its layer that no reader
has started and that are not among its top 6 (refinement, `cache_refine`),
then starts reading its top 3 that are not cached, in rank order, while
fewer than 6 read-aheads per row are reading (`cache_prefetch`, which
`cache_prefetch_from` calls with the prediction's name for the prefetch
metrics). A read-ahead is not evicted before its layer's routing; after it,
one the layer did not select keeps the score of its earlier selections
(see the cache below), and its read continues: `cache_demote` only lifts
its protection.

Refinement first filters candidate slots without locking, then rechecks
ownership and claims cancelled pieces under the reader queue's lock. It
skips the lock when there are no candidates. It removes every queued
reference before completing cancelled groups and making their buffers
reusable. Readers claim ownership under the same lock. A read that has
started stays resident, so refinement does
not wait for its I/O. Kept forecasts and held buffers remain protected.

The online CPU learner below corrects the one- and two-layer router
forecasts when enabled. Cache-retention settings operate independently of
the learner.

A prefetch's pieces wait in a queue for 4 reader threads, which start a
piece only while no demand piece is in flight: a demand read shares the SSD
with at most 4 pieces (8 MiB). A demand request for an expert still being
prefetched reads its pieces that no reader has started at once, and an
evicted slot's queued pieces are dropped.

Routing removes prefetch protection from unused experts of that layer even
when their reads are still pending. A later demand promotes their remaining
queued pieces to demand priority. If an optional prefetch's chosen victim is
still being read, the cache declines that admission. It leaves the victim
and the higher-scoring residents in place. A demand may wait for its victim's
read to finish before reusing the buffer.


### Online CPU predictor

The normal engine links the online predictor and Accelerate for single-thread
CPU math. `--online-predictor` enables a new in-memory learner.
`--predictor-state FILE` enables persistence and restores a compatible file
when present. With neither flag, the learner stays inactive. The main model
runs on Metal. The predictor works with exact and substitute routing, DSpark,
server requests, images, GGUF experts, profiling and diagnostic modes.
Those modes retain their existing constraints and execution paths.

There are 77 independent source/target heads: 39 one-layer and 38 two-layer
forecasts. A head receives 768 features: the normalized forecast logits and
the normalized current-layer router logits, 384 values each. Each block is
centered and divided by its root-mean-square deviation with epsilon 1e-6.
Its seeded projection A has shape 768 by 64 and remains fixed. A ReLU
activation produces 64 features; the learned matrix B maps them to a
384-value correction added to the baseline selection keys. The baseline
is the existing router forecast.

Recursive least squares fits B to the difference between target-layer and
forecast selection keys. Each head keeps a 64 by 64 covariance matrix,
initialized to I/16, with forgetting factor 0.995 per learned row. Batched
prefill does not train the learner. Small forward passes after the initial
single-token prompt, token-by-token prompt/debug paths and verification rows
can train it. The learned matrix starts at zero, which copies existing candidate
IDs. Learner initialization clears B, restores covariance and discards pending
records; an explicit learner reset does the same while retaining A. Learned
state belongs to the engine and survives KV resets, session switches and new
requests. Persistence is explicit through `--predictor-state`. Tensor payloads use
about 23 MiB, and owned prediction records use about 2.2 MB for up to 16
verification rows, below 160 KiB per supported row. Working buffers,
allocator overhead and library memory are additional.

At the routing synchronization point, the model thread copies forecast and
current-router logits and native expert IDs into CPU-owned storage. Per-head
records retain projected rows and baseline keys. Each record carries its
forward-pass epoch, row count and consumption state. Updates run
after the expert command buffers have been committed. Every head predicts
all rows of its pass before that head consumes the corresponding labels;
the head is used again only in a later pass. Labels from rows that verification
later rejects are still observed computations and can train the head.
Invalid auxiliary inputs use the existing candidate IDs, invalid candidate
IDs skip optional read-ahead, and invalid numerical updates are rejected.
Covariance guards can reset covariance while retaining learned output weights.
The output-weight update uses a separate loop from the covariance update.
Its nonoverlapping input and output arrays let the compiler vectorize the
elementwise fused multiply-adds. Both candidate arrays pass their numerical
checks before either becomes the learned state.

The server has one model worker and shares one learner across its requests.
A concurrent batch can contain unrelated sequences: each feature row is paired
with that same row's later native-router label within the same pass. No pending
record crosses a pass or session switch. Session KV snapshots do not contain
or restore learner state.

The engine passes the number of forecast horizons actually computed.
Multirow substitution computes only the one-layer forecast, so the learner
does not inspect or train on stale two-layer scratch. CPU substitution saves
native expert IDs before replacing selections.

The correction supplies only the candidate lists consumed by the existing
one- and two-layer prefetch scheduler. It does not overwrite the main router's
selected experts (`RT`), checkpoint weights, baseline prediction buffers or
current-layer residual forecast (`PC`). It uses backbone routing features;
support for DSpark verification does not mean it reads DSpark's internal
hidden states. Exact routing preserves the model output. With substitution,
prefetches can change cache residency, which can change substitutions and
output. All live learner math runs on the CPU.

#### Persistent state

`--predictor-state FILE` enables the learner and checks the path during model
loading. Its parent directory must exist and be writable. An absent file
starts a new learner. An existing file must pass all validation before any
live learned parameter changes; invalid files fail the load rather than
falling back to a new learner. Symbolic links and nonregular files are refused.

Version 1 uses a fixed little-endian layout of 24,090,040 bytes: a 128-byte
header and 77 head records. Each record contains its source/target mapping,
active flag, cumulative update count, fixed projection A, learned correction
B, covariance P and target-router bias. The header records dimensions,
completed learning-row progress, compatibility binding and an integrity
checksum. A checksum covers the entire file with its checksum field cleared.
The format stores FP32 tensor values without conversion.

The loader checks the version, dimensions, complete length, checksum and
finite tensor values. Covariance must be symmetric, within the live learner's
bounds and have a positive diagonal. These checks match the live numerical
invariants; loading preserves the saved covariance exactly. It also checks
active-head consistency, counters and target-router biases. The compatibility
binding includes the actual router weights, scales, text/image biases, model
dimensions, routed-expert quantization formats and output-head format. It
excludes paths and modification times. It does not cover every checkpoint
weight, so a matching binding is not a complete model-integrity check. MXFP4
and Q2 need separate state files.

Loading restores A, B, P, active flags, cumulative per-head updates and
learning-row progress. The configured freeze threshold is applied to the
restored progress. Transient source/label records are discarded; process
counters and timing start anew. Pending I/O, expert-cache slots and conversation
KV state are not part of this file. Shared server learning therefore survives
an engine restart without restoring conversations.

The model owner saves only at a completed forward-pass boundary. It writes
changed state on normal CLI completion or handled SIGINT/SIGTERM shutdown,
including a server stop. Signal handlers request shutdown; they do not read
or serialize learner arrays. The server does not write per request or on a
timer. A crash or SIGKILL bypasses the final save.

Saving creates a private sibling temporary file, writes the complete state,
flushes the file and atomically renames it over the destination. The writer
refuses unrelated destination files and checks for replacement races. A
failed save removes its temporary file and leaves the old destination intact.
One saved file occupies about 24.09 MB; replacement temporarily holds the old
and new copies, about 48.18 MB. Loading and saving each use a bounded additional
state buffer. No state file is written without `--predictor-state`.

#### Measurements and learning controls

The final `online_predictor` JSON report contains update/fallback/skip counts,
CPU forward/update/reset times, tensor/record sizes and raw top-three candidate
precision. Raw precision asks whether each candidate appears in that target
row's native top six. It includes resident experts and counts rows separately;
it is not a count of useful newly fetched experts. Native
[`prefetch_metrics`](#prefetch-attribution) supplies the distinct-demand,
readiness and first-use accounting for actual reads, within its stated scope.
The two reports have different denominators. Neither raw precision nor an
auxiliary CPU microbenchmark establishes an inference speedup.

The final `updates` count includes cumulative per-head updates restored from
disk. Timing, raw candidate accuracy and window update counts cover the current
process. `state_loads`, `state_saves`, `state_dirty` and `state_file_bytes`
describe persistence activity and its current status.

Two optional integer environment controls are read when the learner initializes:

| Control | Range and behavior |
| --- | --- |
| `MYDS4_ONLINE_FREEZE_ROWS` | Unset or 0 disables freezing; 1–1,048,576 sets a completed-row threshold that includes progress restored from a state file. |
| `MYDS4_ONLINE_WINDOW_PASSES` | Unset or 0 disables window reports; 1–4,096 sets the number of completed eligible passes per report. |

The freeze threshold counts each eligible computed forward/verification row once,
including rejected rows, rather than counting tokens emitted or updates
across all 77 heads. A pass that crosses the threshold completes all its
updates. The next pass freezes B and covariance updates. Forecasting and
raw-candidate scoring continue while frozen, so their costs remain. Reset
clears the learner's row threshold count and frozen state. KV resets and
new requests do not clear either.

`online_predictor_window` records contain per-window completed passes and
computed rows, cumulative learning-epoch rows, raw precision, CPU forward/update
times, update counts and frozen-pass counts. Windows never split a forward
pass. They are emitted after the final router's labels have been processed;
the last GPU commands can still be finishing. A final partial window is printed when unreported completed passes
remain. `last_pass_frozen` and `next_pass_frozen` distinguish a threshold-crossing
pass from the following frozen pass. The final report also gives
`computed_rows_completed`, `learning_epoch_rows`, `learner_resets`,
`completed_passes` and the configured thresholds. `state_scope` is `engine`;
window records identify the `learning_epoch`. Its field `verification_rows`
counts all eligible forward rows as well as verification rows. Update
counts count successful head/row updates, not emitted tokens. Window state
uses bounded aggregates; printing windows adds stderr work.

CPU validation is available independently of model execution:

| Check | Scope |
| --- | --- |
| `make build/online-predictor-unit`, then `build/online-predictor-unit` | 43 module checks: math, record ownership, native labels, available horizons, causal updates, changed row counts, reset, freeze boundaries and windows. |
| `.venv/bin/python -B tools/checks/online_predictor_cli_check.py` | 48 CLI checks of boolean enable, state paths, removed options, compatible modes and existing constraints, before model loading. Includes empty and nonempty profiler values and the `--reserve` range. |
| `make build/online-predictor-state-check`, then `build/online-predictor-state-check` | 42 CPU checks of round-trip and continued-learning identity, file validation, binding mismatch, unsafe paths and atomic-save failures. Temporary files are removed. |
| `make shutdown-check` | 8 CPU checks of signal handling and safe-boundary shutdown notification. |
| `make build/online-predictor-check`, then `build/online-predictor-check --rank 64 --pairs 77 --milliseconds 20` | 20 shared-math checks with 20,000 dead/repeated-feature update rows, followed by a bounded CPU microbenchmark. |

The cache checks listed under [read completion](#read-completion-and-cache-counters)
cover their own implementation behavior. Numerical and performance evidence
from model runs belongs in [Performance](performance.md).

### Approximate decoding and prefill

`--substitute D`, off by default, trades the model's exact output for fewer
reads. When a decoded layer selects experts that have no slot, the layer's
resident, fully read experts with the highest selection keys take their
places, if such a key is at most D below the missing expert's, starting
with the lowest-ranked missing expert. The selection is then the top six of
the keys with D added to those of the resident experts, the cache prior of
Skliar et al. (arXiv 2412.00099). The two selected experts with the highest
scores, which weigh most, are never replaced, as that paper keeps the top
two for DeepSeek-V2-Lite. The key is the sqrt-softplus score plus the
router's bias; a layer's 6th and 7th keys differ by about 0.03, its 6th and
12th by 0.03 to 0.27 depending on the layer (0.17 to 0.18 median). The six
weights then follow the router's formula for the experts used: each score
over their sum, times 1.5. The same rule applies to the backbone rows of a
verify pass. DSpark stages retain their native selections: their experts are
either preloaded or fetched on demand from their separate pools. The router's
own selection is what `--routes-fd` records.

A prompt's rows substitute as well. In a prompt batch, an expert without a
slot is read if some row cannot do without it: it is one of that row's two
highest scores, or no resident expert's key is within D of its key. The
batch's other selections of experts without a slot give way as a decoded
row's do, so an expert that is read anyway serves every row that selected
it. The CPU decides the rows in parallel, first which experts to read, then
the substitutions. Such a prompt reads ahead only each row's top 2
predictions for the next layer, and layer 0's experts once its routing is
known: the experts that most rows need are then in memory by the time the
next layer's rows substitute, and the others are not read.

Which experts are resident at a given moment depends on how far the SSD's
reads have come, so with `--substitute` the output varies from run to run,
also when greedy. [Performance](performance.md#approximate-decoding-and-prefill)
has the speed and the perplexity.

## Speculative decoding

With `--draft N`, decoding runs the checkpoint's DSpark draft
(`inference/model.py`, `forward_spec`) and verifies its tokens in one pass of
the main model:

1. The draft's input is the attention input of layers 37 to 39 (the mean of
   the four streams) at every position the main model ran. For each
   position, `mtp.0.main_proj` and `main_norm` turn it into `main_x`, and each
   of the three stages keeps the key of `main_x` (its `wkv`, `kv_norm`, RoPE,
   FP8) in a window of the last 128 positions.
2. For the token t at position p, the draft runs 5 rows at positions p to
   p + 4: t, then the noise token 128799. Each stage is a block like the main
   layers (hyper-connections, attention, 128 routed experts of which a row
   uses 3, a shared expert); its attention sees the stage's window and the
   keys of all 5 rows. After the stages, the head computes logits for the
   requested draft rows;
   the Markov head adds a bias for the token before (t, then the previous
   draft), and the argmax is the next draft. The bias and argmax chain runs
   on the GPU.
3. The verify pass runs t and the first N drafts through the main model.
   Each row is computed as decoding computes a token: the products read each
   weight once for all rows (`mvm_fp8`, `mvm_bf16`, whose sums for a row are
   those of the one-row kernels; 4 rows take about 1.6 times one row's
   time), and the attention and the routed experts run row by row with the
   decoding kernels. The target sampler uses the requested temperature and
   top-p on each row. A draft is kept while it equals that sampled token;
   the first mismatch becomes the next token without a second draw. If every
   draft matches, the final row supplies the next sample. A pass yields 1 to
   N + 1 tokens. EOS ends verification without drawing from later rows. When decoding exactly,
   each row also reads ahead as a decoded token does (see Reading experts
   ahead): the residual's prediction for its layer before the attention,
   the routers' for the next two layers after the routing, with up to 6
   reads ahead in flight per row. With `--substitute`, which waits for an
   expert still being read instead of replacing it, the rows read ahead
   only the next layer's top 3.

Every stage computes all five rows, whose keys every query attends to; the
head processes only the rows whose drafts the pass requested.

By default, each stage preloads all 128 MXFP4 experts. `--draft-cache N`
reserves 15–128 slots per stage within the total `--cache` budget. Values
0 and 128 select full preloading. Values 15–127 keep resident buffers but
load expert contents on demand, using least-recently-used replacement within
each stage's pool. Fifteen slots cover the maximum union of three experts
for each of five rows. Every currently requested expert is protected before
choosing a victim, and the previous GPU use of that stage's buffers finishes
before a slot is reused. Bounded pools fetch the native draft selections
even when backbone substitution is enabled.

Rejected rows leave no local attention state that a resumed position reads.
The window ring retains `128 + max(8, lookup_depth)` rows. Compressor state
retains 16 rows for the default margin and 32 for wider context-copy
verification. Compressed and index rows are rewritten before use, the
Engram history is rebuilt from the kept tokens, and the draft's windows
receive only the kept positions.

With exact expert routing, verification uses the main model's requested
sampling distribution. Cache settings change which experts must be read,
not the routing rule. With backbone substitution, output depends on cache state and timing.
Validation and measured performance are recorded in
[Performance](performance.md#speculative-decoding). Prompts of up to six
tokens use the same row kernels in prefill.

The checkpoint's additional DSpark weights contain about 14.23 billion
parameters: 13.59 billion routed-expert parameters and 0.636 billion other
parameters. Their stored size is 7.39 GiB, including 6.72 GiB of routed
experts and 0.66 GiB of other tensors. These counts exclude the backbone's
shared embedding and output head and count both FP4 values in each packed
byte. A 32-slot pool per stage reserves about 1.68 GiB of expert payload;
alignment and buffers add overhead. Without `--draft`, the loader skips
DSpark-specific weights and buffers, and the expert-cache budget serves the
backbone. Draft length controls proposal count; every enabled length loads
all three DSpark stages. The draft experts remain native MXFP4 even when
the backbone uses GGUF experts.

### Context-copy candidate generator

`--lookup off|N` enables context-copy drafting in the CLI and server, with
one to fifteen candidates and `off` as the default. The candidate generator
runs on the CPU. The caller appends committed token history to a bounded
per-sequence index.
Separate indexes link occurrences of twelve-token and four-token suffixes.
A query searches the twelve-token index first. After a successful draft,
it can search the four-token index if no longer match supplied a candidate.
Each index examines at most 64 occurrences and extends each suffix match
backward by at most 64 tokens. It prefers the longest match, then the longest available
continuation, then the most recent occurrence. Negative image-placeholder
IDs end matches and continuations.

The initial minimum match is 12 tokens. Two verifications that accept no
drafts raise it to 16. A fully accepted proposal allows a four-token match;
a partial acceptance returns to 12. Feedback counts actual offered tokens,
up to 15; an empty proposal does not change the feedback state. The caller
may request fewer candidates. Every candidate is copied from existing
history and verified by the target model. With `--draft`, DSpark supplies
candidates when lookup finds no eligible continuation. Without `--draft`,
that case uses ordinary decoding and does not load the DSpark weights.
When the server batches separate requests, it uses one ordinary row per
request and keeps each request's lookup history up to date.

The indexes allocate twelve bytes per configured history token plus at most
512 KiB for hash buckets. Reset reuses this allocation. The CPU fixture
compares incremental proposals against a direct scan over 32,768 prefixes
and checks search bounds, capacity failures, reset and feedback transitions.
It also checks that frequent short endings cannot hide a matching long suffix.
The model verifies up to sixteen rows, including the current token. Dense
products split these into groups of at most six to preserve the existing
scalar-equivalent arithmetic; routed experts request the union across all
rows. The local KV ring has `128 + max(8, lookup_depth)` rows. Wider lookup
uses 32 compressor-state rows so rejected drafts cannot overwrite state
needed to resume. DSpark retains its own five-position architecture.

The sampler checks up to 15 proposals, every rejection point, EOS and random
state preservation. GPU checks compare all sixteen rows with serial
decoding and resume after accepting 1, 2, 6, 15 or 16 rows. These checks
cover MXFP4 and IQ2, including an MXFP4 run with DSpark and online
prefetching. Candidate tests alone do not measure model acceptance or
throughput. The CLI prints exact lookup attempt, offered and accepted
counts; the server logs them per request and exposes `lookup_depth` in
status. Copy-heavy and reconstructed Harness timings are
reported separately in [Performance](performance.md#context-copy-drafting).
A consensus fallback for short matches, and studies of verification's
reads and cache credit built on it, were removed; see
[Removed experiments](experiments.md#consensus-fallback-for-context-copy-drafts).

### Temperature and nucleus sampling

CLI and server generation share the same sampler. Temperature zero selects
the lowest token ID among maximum logits and consumes no random draw.
Positive temperatures apply softmax to scaled logits. Top-p below one keeps
the smallest descending-probability prefix reaching that mass, retaining at
least one token; ties use ascending token ID. Top-p one uses the full
distribution in token order, avoids sorting, and does not apply a small-
probability cutoff. Each sampled token consumes one draw from its own
48-bit generator. Explicit integer seeds initialize its low 48 bits; absent
seeds use system randomness. Invalid sampling parameters are rejected.

DSpark's proposals remain deterministic. For a proposal d and target
distribution p, drawing y from p accepts d with probability p(d). On a
mismatch, y already follows the residual distribution conditioned on y != d.
This is the deterministic-proposal case of
[speculative sampling](https://arxiv.org/abs/2211.17192), derived by taking
the proposal distribution as a point mass. Only the matching prefix is
retained. No random draw is consumed for an unvisited future row, so ordinary
and speculative runs with identical logits and seeds retain the same random
stream. This does not require the draft model's softmax probabilities.

`--temperature`, `--top-p`, `--seed` and `--think` configure CLI generation
and server defaults. Request fields override each corresponding server
default. JSON numbers are validated in the server's raw-number representation.
With several active requests, the server samples one row per sequence from
the combined target pass; each sequence retains its own random state.

## Prefill

A prompt runs layer by layer over all of its rows, in batches of up to 2048,
so each layer's experts are read from the SSD at most once per pass. A pass
keeps every row's streams, selections and Engram rows (about 56 KB a row,
plus candidate masks), so a prompt longer
than 16,384 tokens runs as passes of 16,384, each reading the experts again.
Request buffers for residual streams, mixing coefficients, selections,
candidate masks, image flags and Engram rows grow with the prompt. The first
small forward pass afterward reduces them to `max(8, lookup_depth) + 1`
rows. It waits for submitted GPU work before releasing buffers.
Changing the number of rows within that small-batch capacity requires no
reallocation. Candidate masks reserve only these request rows; their width
still follows the configured context limit.

Temporary layer buffers also shrink for decoding: queries, attention
outputs, matrix staging, routed-expert activations and index scores retain
only the required small-batch capacity. A later prefill grows them again.
Draft-history projection can temporarily require up to 128 rows, and image
encoding reserves the full matrix staging capacity. Resizing waits for GPU
completion and updates cache residency. FP32 logits, persistent KV,
draft history and snapshots have independent lifetimes.

The index scores of a batch (a row of scores per row, as long as the
context) are computed in tiles of as many rows as 256 MB hold: the whole
batch up to a 32,768-token context, 64 rows at 1,048,576; tiles of any size
give the same scores and selections.
Products use the matrix units (Metal's `matmul2d`). On the development
machine a GPU core's products and its other work (conversions, decoding,
threadgroup memory) take turns rather than overlap, even between
threadgroups (see [Performance](performance.md#prefill-kernels)), so the
prompt kernels stage as little as they can. A dense product's input rows are
converted to BF16 once, and the matrix units read them from device memory;
the FP8 weights are dequantized into threadgroup memory in tiles of 64
columns, K tiles of 64, each tile serving 128 rows (two products of 64),
each thread loading its share of the next K tile before the products of this
one. The routed experts run as grouped products: the (token, expert) pairs
grouped by expert, each pair's row of the input copied once as half, in
pair order, which the matrix units read from device memory. Native MXFP4
staging multiplies activations by 128 and rescales the products as described
in Numerics. For the
checkpoint's MXFP4 experts a tile is up to 32 pairs of one expert, and each
simdgroup runs its own products of the tile's pairs with 32 columns, K tiles
of 128, the matrix units reading the MXFP4 weights from the slot: a
threadgroup takes 64 columns of `w1` and the same 64 of `w3`, two
simdgroups each (`w3`'s sums meet `w1`'s through threadgroup memory), or 128
columns of `w2`. The GGUF codes' tiles take up to 64 pairs (see
[Experts from GGUF files](#experts-from-gguf-files)). Attention also runs on
the matrix units: a threadgroup of 256 threads computes 32 heads of one row,
over chunks of 32 window or selected positions, with an online softmax. The
matrix units read the queries (with RoPE, as BF16) from device memory, and
each slice of 128 columns of a chunk's keys is loaded into registers while
the products of the slice before run. A product's sums depend only on its K
steps, not on its number of rows or columns (16 to 128 tested), its
simdgroups, the rows beside it or whether its half or BF16 operands come
from threadgroup or device memory, so how the kernels group rows and pairs
into tiles does not change their outputs.

The CPU feeds the pass so that the GPU rarely waits:

- The prompt's Engram rows are read in the background from its start, layer
  1's first; the pass waits for them only before layers 1 and 14.
- A prompt of 256 tokens or more selects nearly every expert of every
  layer: layer 0's are read right after the embedding, and after a layer's
  routing the CPU reads its missing experts and starts reading the next
  layer's. A prompt of more than 4,096 tokens (two batches), whose GPU work
  per layer takes about as long as the layer's reads, reads every expert of
  the next layer ahead, so that the GPU rarely waits. A shorter prompt, which
  waits for the SSD, reads ahead only the experts that its rows' top 8
  predictions name; the experts that no row selects are not read, and the
  few selected but not predicted are read after the routing. With
  `--substitute`, a prompt reads ahead less (see [Approximate decoding and
  prefill](#approximate-decoding-and-prefill)).
- The cache's bookkeeping for a layer (cooling finished layers, resizing)
  runs while the GPU works on the layer's first batch.

### Optional dependency-pruned prefill

`--prefill-prune` keeps all work in layers 0 through 20,
including every global KV and index-key producer. Layers 21 through 39
carry information between token positions only through their causal
128-token local windows. A change can move at most 127 positions forward
per layer. Token-local operations do not extend that distance.

The plan conservatively retains `2668 + max(8, lookup_depth)` positions
before the final state and before a capture's position (a fork's branch
point). It rounds the starts down and the capture's end up to the original 2,048-row matrix
batch boundaries, merges overlapping regions, and skips whole batches
outside them in layers 21 through 39. Those skipped rows cannot affect the
retained local state. Global attention still has the complete encoder KV.
The attention masks and numerical kernels remain unchanged.

The CLI and server expose the plan through `--prefill-prune`, off by
default. The engine's `myds4_config.prefill_prune` field selects the same
path. The plan applies only with substitution and CED off and without full-row
logits or activation/route dumps. Validation
(`tools/checks/prefill_prune_model_check.c`) compares logits, global and
local state, a capture at a prefix, a resume from it and DSpark history, as well
as checking the dependency plan on small causal graphs. See
[validation records](../data/benchmarks/prefill_prune_results.json).

Changing prefill's work changes the expert cache left for decoding. The
long MXFP4 comparison reduced total request time despite slower decoding.
The option therefore does not promise a decode-throughput improvement.
The server reports the configured `prefill_prune` flag and the last
prefill's `prune_applied` flag separately; short prompts or incompatible
diagnostics can leave pruning unused.

The CLI's `prefill_prune` JSON line reports the plan: `decoder_row_layers`
adds the rows actually executed across layers 21 through 39, and
`decoder_tokens` counts positions retained for the first of those layers.
A dependency interval that shrinks with decoder depth and outer chunks of
32,768 tokens were measured and removed; see
[Removed experiments](experiments.md#layer-dependent-prune-interval-and-32k-outer-chunks).

## Images

An image in a prompt takes the span of `inference/image_processor.py`:
`[IMAGE_START] + ([IMAGE] * tw + [IMAGE_NEW_LINE]) * th + [IMAGE_END]`, th
(tw + 1) + 2 positions, at most 1,024.

The image file is decoded as the reference's Pillow decodes it with
`convert("RGB")`: JPEG through libjpeg-turbo with its defaults, as Pillow
does, and other formats through macOS's ImageIO from the image's own pixels
(alpha dropped without compositing, gray repeated, palettes looked up,
16-bit channels by their high byte, except 16-bit gray without alpha, which
Pillow opens as `I;16` and clamps at 255; formats outside these are drawn
into sRGB). Images of more than 178,956,970 pixels are refused, Pillow's limit.
The grid follows `plan_image_grid`: an image of fewer than 544 x 544 pixels
is scaled up, its size is rounded up to multiples of 14, and an image whose
tokens would exceed 1,024 shrinks to the largest size that fits.
`ImageOps.pad` then resizes it to fit (`contain`, with Python's rounding of
halves to even) and centers it on gray (127). The resize ports Pillow's
bicubic filter (`libImaging/Resample.c`): fixed-point weights with 22
fraction bits, rows first, each pass rounded to 8 bits. The pixels become
float32 in [-1, 1], rounded to BF16, in patches of 3 x 14 x 14, zero-padded
to 640 values for the matrix units. On 2026-10-31, against Pillow 12.3.0 and
the reference's own grid functions, 49 generated images and five JPEG
photos gave the same grids, padded pixels and BF16 patches, byte for byte,
and eight 16-bit PNGs (gray, gray with alpha, RGB and RGBA) the same
pixels.
The generated images were RGB, RGBA, gray, gray with alpha, 8- and 4-bit
palette PNGs and JPEGs, from 1 x 1 to 4032 x 3024 pixels, including the
taller-than-100x case that Pillow resizes in two steps; the photos were
baseline, progressive, gray and 4:4:4 JPEGs.

The vision encoder (`vision.py`) runs 32 blocks of RMSNorm, attention and a
SwiGLU MLP over the image's patches, 1,024 wide with 16 heads of 64, as the
reference runs them in BF16. Every product's output (after its bias), the
norms (eps 1e-6), the 2D RoPE and the attention output are rounded to BF16.
The RoPE turns head values i and i + 32 by the patch's row for i < 16 and by
its column otherwise (theta 10,000). The products run on the matrix units
(`mm_bf16`, as the prompt's dense products). Attention covers all of the
image's patches, also on the matrix units: `vit_heads` copies each head's
queries, keys and values as BF16 (exact, as they are rounded to BF16
already), with zero rows up to a multiple of 64, and each simdgroup takes 16
queries of one head on its own (four to a threadgroup of 64 queries) over
blocks of 64 keys. The scores accumulate in FP32 and pass through an online
softmax in FP32, whose row maxima and sums come from the matrix units' row
reductions (`reduce_rows`); the probabilities are rounded to BF16 for their
product with the values, as PyTorch's CPU attention rounds them for the
reference's BF16 inputs (over blocks of 512 keys, as an emulation that
matched 97.5% of its outputs showed), and the outputs
accumulate in FP32. The probabilities pass from the scores' layout to the
second product's through `get_left_input_cooperative_tensor`, which needs a
source of its own element type (a float source passed this way gave wrong
rows), so a change of the products' shapes or types needs
`tools/reference/check.py vision` again. Each block is a command buffer of
its own, so that no long buffer holds up the display. The Aligner merges 3 x
3 patches (zero-padded at the edges) into rows of 9,216, then applies w1,
GELU and w2 to 5,120; the GELU (erf) runs on the CPU in double precision,
rounded to float, then to BF16. The weights (0.97 GB, BF16) load at the
first image and stay. Their resident memory counts toward the GPU's limit
like the others', and the cache maps fewer slots at its next check.

A rendered chat prompt marks each image part's place with the byte 0xFF
(`IMAGE_MARK`), which no valid UTF-8 text holds; the server's JSON parser
rejects invalid UTF-8. `tok_encode_chat` turns that byte into the image
token and tokenizes the placeholder's text `<｜deepseek_image｜>` as text, so
a message or a tool result that holds it (a file that mentions it, say)
reads as those characters. The reference's `prepare_vl_inputs` tokenizes
the placeholder's text as the image token and raises an error when the
count differs from the images, so it fails on such text. `--raw` prompts
keep the reference's tokenization: there each `<｜deepseek_image｜>` places
an `--image` image. `--render` writes the placeholder's text, as
`encoding.py` does.

Every position of an image's span carries a negative token derived from the
image file's bytes (FNV-1a, mixed with the position in the span), so that a
cache entry matches only the same image (see The KV prefix cache). The model takes such a
position as part of an image span, as `model.py` does: its embedding row is
the span's (the learned `image_start`, `image_end` and `image_newline` at
the delimiters, the Aligner's rows in reading order at the IMAGE slots); the
router selects with each layer's `bias_vl`; Engram leaves its stream
untouched (the gate is shut) and n-grams stop at it (the reference's DEAD
tokens). The server decodes a request's images when it renders the prompt,
and runs the vision encoder only for the images whose spans reach the part
of the prompt being prefilled; the others are in the rows the prompt resumes from.

Checked on 2026-10-31, besides the preprocessing (above): the placeholders'
places and the images' order in prompts (`tools/reference/check.py encoding`:
600 conversations with image parts, equal to `encoding.py` byte for byte), and
the encoder's features (`tools/reference/check.py vision`) for an 800 x 551
photo (40 x 58 patches, 280 feature rows): 33.2 dB SNR against the reference's
BF16 run (median row 38.3 dB, worst 17.1 dB; no row equal bit for bit). The
reference's own BF16 run is 28.6 dB from the same reference in FP32 (worst
row 10.1 dB) and myds4's is 29.7 dB (worst 13.9 dB): the differences are BF16
rounding with sums in another order, and the weakest rows are the same in
both. Greedy descriptions of two photos named their content correctly, and a
conversation's second turn, which resent its image, reused the image's rows
without encoding it again.

## The expert cache and memory

A slot is a Metal buffer holding one expert (18.9 MB with alignment slack).
The cache's budget is the available memory less a reserve for other
programs: 10 GiB, or 2 to 1,024 GiB with `--reserve`. The
available memory is the kernel's free pages plus reclaimable file-backed
pages. Speculative file pages are already in the free count and are not
counted twice. The kernel's file-cache target does not reduce availability,
including when that target is zero or unavailable. Cached files are not
reported as used memory in the TUI.

Initial expert capacity uses this available total minus the reserve and
reserved draft pools, bounded by the requested cache budget. File pages
remain available for macOS to reclaim, but this total is not a guarantee
that it will reclaim them before compressing or swapping anonymous pages.
The cache responds to actual memory pressure and swapping. Compression
in RAM has no separate limit. A requested budget remains an upper bound, because later pressure
can shrink the allocation.

The GPU also limits it. The GPU driver reports how much memory all
processes keep resident for the GPU ("In use system memory" in the
accelerator's statistics, read through IOKit). On the development machine,
past about 55 GiB of it the GPU stalls: decoding took 8 to 15 times its GPU
time while the GPU was only 17 to 46% busy, whatever the free memory or the
number of slots, and memory that was allocated but not resident did not
count (see [Performance](performance.md#gpu-resident-memory)). The cause is
presumably the IOGPU driver's wired-memory collector. The GPU reads a slot
only while the slot is mapped for it (in the residency set below), and the
cache keeps its mapped slots, with the weights beside them, below 41% of
the physical memory (52.5 GiB on 128 GiB), unless the collector is switched
off (`iogpu.disable_wired_collector`) or a low-water mark is set
(`iogpu.wired_lwm_mb`, which then is the limit); a margin of 3 GiB covers
slots taken out of the set, which the driver counts for about a second
more. The other slots stay in memory, so the cache's size follows free
memory and `--cache` alone. CPU reads fill unmapped buffers. `cache_get`
maps a slot when an encoder uses its expert; a forecast never used by the
GPU needs no mapping. Reusing a buffer for another expert removes its old
mapping before the CPU reads it. Committing a mapped slot into the set took
about 0.1 ms of CPU time for a 12 MB slot and 0.15 ms for an 18.9 MB one
(see [Performance](performance.md#gpu-resident-memory)), against about 1 and
1.4 ms to read the expert from the SSD. Between tokens,
and in a prompt pass for the layers done, the mapped slots with the lowest
scores leave the set until the rest fit; slots still being read or read
ahead, and those of the layer being served, stay. Past the
limit (other processes may take GPU memory) the cache maps exactly the
excess fewer, and it maps more again after 32 calm checks.

Every expert has a score: the sum of a growing weight added once per
distinct expert requested by a layer batch. Prefill contributes to this
history, and the history remains when an expert is evicted. The weight grows
by 2^(1/N) at each aging step, so an earlier request's relative contribution
halves after N steps. `--cache-half-life N` accepts 1–1,000,000; the default
is 50. A decode or verification pass advances the clock once, and prefill
advances it once at its end. Plain one-token decoding therefore advances
once per token. A verification pass still advances once when it accepts
several tokens. These are forward-pass units, not emitted-token units. The
growth ratio is computed at setup; occasional common rescaling keeps the
stored history and resident scores within range.

When decoding, the victim of an eviction is the eligible slot with the
lowest priority, by default its score; a prompt pass ranks by decoding's
part of the score (below). With
`--cache-read-cost`, off by default, the priority is the raw score
times the expert's aligned read bytes divided by the fixed backbone slot
size. Read bytes cover all of that expert's file spans rounded to the read
alignment. The cost factor is computed once from the known file layout;
raw score history and request increments remain unweighted. This weighting
does not change least-recently-used release under memory pressure, GPU
unmapping priorities or the draft pools' replacement rule.

Slots emptied by cancelled read-aheads go first. An empty entry is exempt
from current-layer usage protection, even if it retains that layer's usage
stamp. Held references still prevent reuse. When decoding, the experts
requested by the layer being served in the current pass and protected
read-aheads cannot be victims. A recency list orders the slots for giving
slots back.

Every slot contributes to the process's memory footprint. Mapped slots
are wired for the GPU; macOS may compress other slots in RAM. Compression
alone does not shrink the cache or prevent growth. The engine checks memory
at each decode or verification pass, during each prompt layer and every
second while the server idles. It shrinks when macOS reports pressure,
when swap-outs increase and the cache can release slots above its two-layer
minimum, or when available memory falls
below the reserve. It waits one second between releases unless pressure
is critical or swapping occurs.

After 32 calm checks, the cache can grow by a quarter of the headroom when
availability exceeds the reserve by 4 GiB, up to the requested budget.
For five minutes after swap-driven reclamation, growth uses free pages
alone. Compression, a swapper wakeup without swap-outs, and a global
swap-out while this cache is empty or at its minimum do not start that
cooldown. Actual pressure and a shortage against the reserve still limit
an empty cache's future allocation.
Within a prompt pass at layer l it releases and unmaps only slots of layers
before l - 1, which no GPU work in flight uses.

The mapped slots are in one residency set attached to the command queue,
so command buffers need no per-buffer residency. A slot mapped during a
token is committed with the set's other changes after the token, or before
the next layer of a prompt pass; until then the command buffers that use
the slot declare it themselves (`useResource`), and a buffer read that way
took the GPU about as long as one in the set. A slot leaves the set only at
those commits, when no GPU work in flight can read it.

`MYDS4_CACHE_TRACE=1` prints changes to allocation and mapping limits on
stderr. Each line includes the allocated and mapped slot counts, configured
maximum, memory-pressure level, new swap-outs, free and available bytes,
GPU headroom and whether growth uses free pages or available memory. The
trace is off by default. `build/cache-memory-check` exercises resize
decisions with controlled VM counters; `build/cache-mapping-check` checks
mapping at GPU use, including reuse of an old mapped slot. Both run on the
CPU without model weights or GPU commands.

With `--draft`, the default preloads the stages' 384 experts into dedicated
slots of their own size (7.3 GB in MXFP4, taken off the total cache budget).
Those experts remain resident. A smaller `--draft-cache` reserves three
pools of N slots and reads native draft experts on demand. The pool buffers
remain resident even when their expert contents change. They use MXFP4 slot
sizes with either backbone quantization, so a draft expert never occupies
a smaller GGUF backbone buffer. Both allocation modes reserve their bytes
before allocating the adaptive backbone cache. See
[Speculative decoding](#speculative-decoding) for pool limits and replacement.

During a prompt pass at layer l, the victim is a slot of a layer before l,
which the pass is done with, or else of a layer after l + 1; it is never of
layer l or l + 1, which the pass still needs. Among those it is the slot
whose expert decoding used least: each expert keeps, besides its score,
the part of it that decoding's requests gave (weighted by read cost as the
priority is). So the experts that only prompts needed go first, from
whichever eligible layer, and decoding after the prompt finds the experts it
used before. Among slots decoding used alike, as in a cache that has not
decoded yet, the least recently used goes, so that the experts of the
prompt's last layers and rows, which the decoding after the prompt needs
first, stay longest. The cache therefore needs room
for two backbone layers' experts (768 slots). A fixed
`--cache` budget must cover those slots plus the reserved draft pools. The
minimum byte budget depends on the backbone's slot size and draft settings.

### Read completion and cache counters

A slot's dispatch group tracks its outstanding reads. The slot keeps that
group until the consumer waits and releases it in `finish`, including when
the reads have already completed. `cache_ready` checks completion with a
zero-time group wait; `cache_slot` also accepts slots whose reads are still
pending. Completed experts can run in the ready batch and can serve as
substitution candidates under the existing selection rules. Exact routing
keeps its selected experts and weights.

The model thread owns slot allocation, reuse and group release. Read-only
slot queries also run in parallel substitution helpers while the model
thread waits for those helpers to return. They do not release groups. The
reader threads complete pending work. The last completed or cancelled piece
immediately releases the load's in-flight prefetch admission, before leaving
its dispatch group. The consumer later releases the group. The terminal view
and status endpoint read aggregate counts;
they do not query slot groups.

`cache_request` checks completion once for each requested expert and uses
that result for both its pending-read count and its late-prefetch count.
A prefetch hit means that the request found an expert marked as prefetched;
it can still be late. `prefetch_hits - late` counts those found complete at
that check. The CLI's overall hit percentage includes both kinds of
prefetch hit, so it measures requests that found an existing slot, including
unfinished prefetches, not the fraction that found their data ready.
`read_s` measures the time spent waiting for requested experts.

The global counters count the distinct experts requested by each layer
batch, combining its rows. With DSpark, they also include draft requests:
full preloading makes those requests hits, while bounded pools can add draft
misses and reads. The CLI, request-summary logs and status endpoint use those
global counts. The terminal view's decoded-row chart
uses backbone selections only, counted separately for each row, including
verification rows. Those two denominators are different.

CLI generation also reports `decode_experts` as one JSON line. `computed_rows`
counts backbone decode rows, including rejected verification rows. `selected`
is that count times 40 layers times six routed experts. `unique_layer_requests`
counts each served expert once per layer batch, combining duplicate selections
across rows. Thus `1 - unique_layer_requests / selected` measures within-batch
expert sharing. These counts exclude draft-stage experts and do not measure
cache hits, completed reads or useful prefetches.

A request consumes the slot's prefetch marker. Passing a layer also clears
the markers on its unselected experts, so later reuse of those experts is
counted as an ordinary hit by these legacy counters. Separate opt-in
attribution preserves the original read metadata, as described below.

The cache checks run without model weights or GPU execution. Controlled
read fixtures use `/dev/zero`:

| Command | Checks | Scope |
| --- | ---: | --- |
| `make cache-ready-check` | 45 | Completed/pending groups, readiness, late counts and consumer release. |
| `make draft-cache-check` | 49 | Separate stage pools, native demand admission, replacement and protected groups. |
| `make prefetch-metrics-check` | 54 | Fixed snapshots, first use, read origin, byte accounting and I/O timing. |
| `make cache-policy-check` | 58 | Aging ratios, victim rules in decoding and in a prompt pass (decoding's use first, then recency, also in a cache that has not decoded), aligned read costs, history, rescaling and reuse of refined empty slots. |
| `make prefetch-queue-check` | 14 | Admission release on completion or cancellation, pending ownership and delayed notifications. |
| `make prefetch-admission-check` | 19 | In-flight admission limits, promotion, repeated candidates and preservation of required data. |
| `make prefetch-demote-check` | 37 | Demotion after routing, which cancels no read in decoding or prefill; refinement, which cancels only wholly unstarted forecasts it excludes, keeping the queue's order, the accounting and the reuse of emptied slots; held and partly read forecasts. |
| `make prefetch-refine-race-check` | 8 | An actual reader winning ownership while refinement decides whether to cancel. |
| `make memory-available-check` | 11 | Reclaimable file pages, speculative-page accounting and independence from kernel cache targets. |
| `make cache-memory-check` | 8 | Initial capacity, calm growth with compression, and responses to pressure and swapping. |

### Prefetch attribution

`MYDS4_PREFETCH_METRICS=1` enables bounded in-memory observation; it is off
when unset or `0`. CLI generation, `--tokens` replay and `--perplexity`
write two JSON lines to stderr at completion. Each line has a
`prefetch_metrics` object with `version: 1` and `phase: "prefill"` or
`phase: "decode"`. The server's status endpoint and terminal view keep
their existing counters.

The observer handles backbone CPU routing, prefill and DSpark verification.
It excludes the draft stages. An independent monotonically increasing pass
ID names a forward batch; it is not an emitted-token ID. Decode row counts
include rejected verification rows and rows from concurrent sequences.
Within a layer batch, repeated use of an expert across rows counts once.

The model thread takes one readiness snapshot before substitution. It
records `exact_router` from the original choices and `served` from the
choices actually requested, both against that same snapshot. Each view has
`layer_batches`, `unique_demands` and `requested_payload_bytes`. The latter
two objects use the following mutually exclusive states:

| State | Meaning at the snapshot |
| --- | --- |
| `existing_ready` | A complete resident expert, including previously consumed prefetches and unused prefetches from an earlier target pass. |
| `fresh_timely_prefetch` | A newly admitted prefetch for this target pass, complete and not yet consumed. |
| `pending_prefetch` | An unfinished read originally admitted as a prefetch. |
| `pending_demand` | An unfinished read originally admitted on demand. |
| `absent` | No slot exists for the expert. |

Only the `served` view consumes first-use attribution. Read origin and
target-pass metadata survive clearing the eviction policy's prefetch marker.
An unused prefetch first requested in a later pass is an existing-ready hit
if complete, with a separate eventual-use count; it is not a fresh timely
hit for that later pass.

The remaining fields describe read admission, consumption and removal.
Issuer `pc` is the current layer's residual forecast; `h1` and `h2` forecast
one and two layers ahead. `prefill_bulk` admits the initial prefill layer's
experts together, and `other` covers unclassified callers.

| Field or group | Meaning |
| --- | --- |
| `passes`, `rows` | Forward batches and computed rows, split by execution phase. |
| `admissions_by_issuer` | New reads admitted by `pc`, `h1`, `h2`, `prefill_bulk` or `other`; predictions already in a slot do not create an admission. |
| `fresh_timely_first_uses_by_issuer` | Fresh timely served hits credited to their original issuer. |
| `first_uses_by_issuer`, `first_uses_before_io_started_by_issuer` | Observed first uses by origin, and the subset for which no read piece had started when consumption was recorded. Includes later-pass reuse. |
| `admission_to_first_use_ns_by_issuer` | Sum of admission-to-consumption intervals, in nanoseconds, for those first uses. Divide by the corresponding first-use count for a mean. |
| `target_pass_first_uses_by_issuer`, `target_pass_first_uses_before_io_started_by_issuer`, `target_pass_admission_to_first_use_ns_by_issuer` | The same counts and interval sums restricted to first use in the admission's intended forward pass. Excludes eventual reuse in later passes. |
| `io_timing` | Per-piece counts, summed admission-to-start time (`queue_ns`) and summed read-call duration (`service_ns`), split into `demand`, `background` and `promoted_prefetch`. |
| `prefetch_first_uses`, `eventual_first_uses_after_target_pass` | First observed consumption of prefetched reads, including late reads; the second field identifies use after the target pass. |
| `bytes_completed_at_first_use`, `bytes_remaining_at_first_use` | Completed and outstanding requested read bytes at the fixed first-use snapshot. |
| `prefetch_origin_read_bytes`, `demand_origin_read_bytes` | Completed requested bytes by original admission type; promoting a prefetch to the demand queue keeps its origin. |
| `unused_prefetches_removed`, `unused_prefetch_read_bytes_removed` | Prefetches removed before consumption, and their completed read bytes. |
| `unread_bytes_cancelled_on_removal` | Requested bytes cancelled before reading; separate from completed unused bytes. |
| `unconsumed_prefetch_read_bytes_live`, `prefetch_bytes_outstanding_live` | Unconsumed completed bytes and remaining prefetch bytes still in live slots at reporting time. Future use is unresolved. |
| `prefetch_caused_evictions` | Slots displaced by an admitted prefetch. |
| Each view's `rerequests_after_prefetch_eviction`, `absent_after_prefetch_eviction` | Later requests for those victims, with the absent subset separated. A served request clears the victim marker; an exact-router observation alone does not. |
| `first_uses_outside_observed_routes` | Previously tracked reads consumed outside those views, recorded separately without a timely-hit reward. |
| `scope`, `pollution_scope` | Descriptions of the population, byte units and limits of attribution. |

Payload-byte fields sum the expert's file spans. Read-byte fields count the
aligned lengths requested by completed read operations, not measured physical
SSD traffic. Reader threads update byte counters before leaving their groups,
so slot reuse cannot charge an old read to a new expert. Read totals follow
the admission phase; observed first uses follow the consumption phase. First
uses outside observed routes are attached to the original admission phase.
Live byte totals are asynchronous snapshots and are not automatically waste.

Readiness and completed-byte attribution use the fixed routing snapshot.
First-use timing and the no-piece-started count are sampled when the served
routes consume attribution, shortly after that snapshot. Timing sums are
diagnostic observations, not additional hit credits. A promoted prefetch's
queue interval starts at its original admission, not at promotion to demand.
Piece durations can overlap across readers and must not be added to infer
wall time. All timing hooks are disabled with the observer.

Eviction and rerequest fields are observations, not a counterfactual estimate
of lost baseline hits. They do not measure causal cache pollution or saved
wall time. Enabling observation adds CPU work; it does not install a predictor,
modify routing rules, or write per-token trace files. Approximate substitution
remains dependent on read timing.

## Scheduling and the environment

The process runs at the user-interactive QoS class. As a background process,
macOS gave its GPU work lower priority than the desktop's, which made
decoding up to 4 times slower.

The GPU lowers its clock within milliseconds of idling, and the work after
a pause runs up to 3 times slower. While a decoded token's layer waits for
experts being read, one simdgroup spins on a second command queue for about
1 ms per expert still being read (`gpu_keep_clock`): it occupies a sliver of
one core, and the clock stays up. A spin that outlasts the wait slows the
real work, so its length is the reads' estimate, not more.
Calibration warms the spin kernel, measures three completed dispatches with
Metal GPU start/end timestamps, and uses the fastest positive duration.
Host submission and scheduling delays do not enter the loop-rate estimate.
`make gpu-clock-check` injects host-clock delay and checks a requested 2 ms
GPU interval without loading model weights.

Demand expert reads run on GCD's global queue at the user-interactive class;
the prefetch reader threads and the row reads run at the user-initiated
class. No read may run at the utility class or below, where macOS throttles
disk reads.

## Server

The server (`build/myds4-server`, see [The server](server.md)) is written in
Rust with DeepSeek's `deepseek-recipe`: HTTP, request conversion, prompts,
output, the scheduler below (`server/src/worker.rs`) and the KV prefix
cache (`server/src/kv.rs`). The model runs behind the engine's C ABI
(`src/engine/engine.h`).

A chat completion is tokenized on its handler's thread and waits in a queue
(at most 16) for the model, which the worker's thread runs. A streamed
request gets its headers at once and a comment every 15 s while it waits.
Between tokens the worker admits waiting requests while fewer than
`--batch` decode: it resumes one's prompt from the KV prefix cache,
prefills the rest (the others wait) and samples its first token. Then
every request decoding advances by a token: alone, with drafts when
`--draft` is set; else in one step of `myds4_rows`, one row per request.
The worker sends each step's tokens to the request's handler, which writes
them to its client; when a client closes its end, the handler cancels its
request.

While no request runs, the experts stay in memory as the last request left
them. The worker wakes every second and resizes the cache as between tokens
(`cache_idle`): when other programs take memory, it gives slots back, and
the log notes it.

A step of several sequences runs like a verify pass (up to 6 rows): the
products load each weight once for all rows and give each row the sums of
the one-row kernels, the router, the experts (in expert order) and the head
run per row, and the attention of each row (its window key, compressor,
indexer and sparse attention) runs as a decoded row's, with that row's
state bound and its own position and Engram history. Each row's logits are
therefore those of its token decoded alone: `--check-batch` compares them,
and a verify pass's first row, with a token decoded alone (0 of 129,280
logits differed, with the GGUF experts and two drafts). With `--draft`, the
step then writes each row's draft window keys into its own sequence's
windows at its position (`draft_rows`), as decoding the sequence alone does,
so a sequence that drafts alone again later finds its windows whole
(`build/draft-rows-check` compares them byte for byte).

### Concurrent scheduling

The server's `--runahead`, off by default, lets ready requests advance
while other requests' expert reads run (`src/experimental/runahead.c`). It
applies to a step of two or more requests; a request alone keeps ordinary
decoding, including context lookup. The engine refuses it together with
DSpark, substitution, the online predictor or the routing and dump
diagnostics.

Each request is a job that advances through the layers on its own. A job
keeps what it needs across a pause between layers: its activations, router
output, candidate mask, index selection and Engram rows, and its
CPU-written descriptors and embedding rows; shared GPU scratch follows
command-queue order. After a job's layer front, the CPU waits for that
front's routing signal, reads the job's selected experts and requests them
with a reservation. During the step the engine reads only these selected
experts: it leaves out the residual and next-layer forecast projections
and starts no read-aheads. The scheduler runs a job whose experts are all
in memory, at the earliest such layer. Jobs ready at the same layer run
together: one command buffer runs each job's routed experts, each with its
own weight loads, and then the group's next layer front runs as one batch
in the small-batch kernels, which load each dense weight once for all
rows. When no job is ready, the scheduler waits for the job at the
earliest layer, the one with the fewest missing experts on a tie. Command
buffers retire once per step.

Reservations cover resident experts and pending reads; jobs that need the
same expert share its read. Each job releases its references only after
its GPU work finishes. Reserved entries cannot be evicted, and the cache
resizes only between steps, when no job holds a slot. When the CLI or the
server finishes, the `ready_groups` JSON line counts the groups and their
rows, and `forecast_work` the row-layers that ran without forecast
projections. The
[Bevy paper](https://arxiv.org/html/2611.04794v1) describes the broader
approach, including grouping ready requests at the same layer. Other
schedules, sharing expert weight loads within a group, and scheduling a
verify pass's rows the same way were measured and removed; see
[Removed experiments](experiments.md#runahead-schedules-other-than-the-public-one)
and [Causal verification](experiments.md#causal-verification).

`tools/bench/session_bench.py` compares server throughput on concurrent
reconstructed Harness requests over HTTP. It starts one server at a time,
with `--modes` giving each run's `--batch` (1, 2, 4 or 6; by default
`1 2 1`) and `--runahead` one 0 or 1 per mode; `--run-binaries` names one
server executable per mode. It compares returned text, reasoning and tool
arguments, and records prefill, client latency, model execution counters,
cache capacity and expert reads. It never executes generated tools. These
are workload replays rather than an autonomous agent quality benchmark.

### Experimental multi-layer expert work

`src/experimental/moe_tasks.metal` and `tools/experimental/moe_tasks_check.c`
test one building block for pipelined decoding. Each task describes a ready
gate/up stage or a ready down stage for one request and layer. A single
dispatch can contain both kinds of task from different layers. Each
threadgroup takes one task's tile, so requests keep separate activations,
outputs and expert tables. Tiles retain the ordinary scalar MXFP4 products,
SIMD reductions, BF16 rounding, activation quantization and ordered expert
sum. Down work uses six SIMD groups; the other two groups in the
256-thread launch participate in the barrier without reading experts.

The host must submit only independent ready tasks. A down task needs its
own completed gate/up output. The shader has no device-wide spin barrier
and does not wait for expert reads. The standalone check uses 24 resident
experts from four checkpoint layers and synthetic activations. It checks
all intermediate and final bytes, finite values, and unchanged outputs of
inactive jobs. It then times warmed kernels. It is not part of the default
Metal library, CLI or server. Measurements are in
[Performance](performance.md#multi-layer-expert-task-prototype).

Full pipelined decoding remains unimplemented. It needs request-owned
scratch for every stage that can overlap, a queue of ready stages with
explicit GPU completion and SSD-read dependencies, and expert reservations
that last through GPU use. The scheduler must preserve each request's
attention state, routing, reduction and sampling order. Model-level logit
and state comparisons and matched HTTP throughput tests are still required.
The current runahead scheduler above is a separate implementation.

### Prompts

The server renders prompts with recipe's `DeepseekV41Encoding`, after
DeepSeek's request converter ([Prompts and tokens](server.md#prompts-and-tokens));
[Prompts compared with encoding.py](server.md#prompts-compared-with-encodingpy)
lists where they differ from the checkpoint's `encoding/encoding.py`. The
checkpoint ships no Jinja chat template; its README names `encoding.py` as
the reference. Reasoning effort levels map as the converter maps them for
V4.1 (minimal and low 50; medium, high and xhigh 75; ultra and max 100;
numbers 1 to 100 as given), and an effort of `none` turns thinking off
unless the request sets `thinking`. Neither the converter nor `encoding.py`
adds a system prompt of its own.

recipe's stream processor parses the output: reasoning until `</think>`,
then content, then the DSML tool calls, each with its arguments as JSON. The
engine's tokenizer gives it each token's bytes; a character whose UTF-8
bytes are not all there yet waits for the next token.

The tokenizer loads added tokens as literal UTF-8 before converting the
byte-level BPE vocabulary. An added token that also appears in that
vocabulary keeps its literal spelling. Byte-alphabet decoding validates
the code point before indexing its table.

### The KV prefix cache

Each conversation's attention state lives in an engine state: the layers'
window rings, compressed rows, index keys and compressor rows, the draft's
windows and the Engram history. The compressed rows and index keys take
3,200 bytes per position (3.125 KiB) in BF16; a state starts with room for 16,384 positions
(55 MiB with its rolling rows, without `--draft`) and doubles its room when
a forward pass needs more, up to `--ctx` (3.13 GiB at 1,048,576), keeping
the rows it holds; a reset gives the room back. The KV prefix cache charges
a state these bytes (`myds4_state_bytes`), its snapshots apart. The layers use one state at a
time. A ring holds its layer's last `128 + max(8, lookup_depth)` positions.
The compressor keeps 16 rows normally and 32 for wider lookup, so a state
can continue at its position or up to `max(8, lookup_depth)` positions
before it (its rewind room), while its rings still hold those positions'
rows: see the rewind room below. The compressed rows and index keys of a position are never
rewritten once later positions ran, so another state can copy a state's
rows before its position (`myds4_state_copy`), also while a request still
decodes in it. The rows that later positions overwrite (the rings, the
compressor rows, the draft rings and the Engram history) are kept apart as
snapshots of 5.6 MiB each without `--draft` (5,832,716 bytes), plus 0.5 MB
of logits at a prompt's end (`myds4_snapshot_take`). A snapshot and its own state's rows before its
position together give the state at that position.

The storage rate follows the three ratio-2 KV sources and one ratio-1
source: `(3/2 + 1) * (512 + 128) * 2 = 3200` bytes per input token,
apart from rounding and guard rows. The values already obey the model's
FP4 quantization for global KV/index keys and FP8 for local windows;
BF16 holds those quantized values exactly. This format is the same for
MXFP4 and GGUF expert weights and for full or compact CED prefill.

DeepSeek's packed global layout instead needs 288 bytes per 512-value
main-KV group (256 code bytes and 32 E4M3 scales) and 68 bytes per
128-value index-key group (64 code bytes and four UE8M0 scales). The
amortized logical payload is therefore 890 bytes per input token. The
official DeepInfer index layout separates block-wide data and scale arrays;
padding depends on block alignment, rather than a universal 72-byte index
record. At 1,048,576 positions, packed logical global payload is about
0.869 GiB; this engine's BF16 global payload is 3.125 GiB, before rings and
saved-state allowances. These are layout calculations, not measured process
RSS. Model weights, expert buffers, positional tables and attention scratch
space are additional memory.

Packed KV storage is not implemented in this engine. Storing the existing
codes and scales could preserve its quantized values while reducing global
state memory by about 3.6x; load/dequantization kernels and state handling
would need numerical and performance validation. See
[FlashMLA's format](https://github.com/deepseek-ai/FlashMLA/blob/main/csrc/cuda_kernels/kv_cache_format.h)
and [DeepInfer's cache layouts](https://github.com/deepseek-ai/DeepInfer/blob/main/deepinfer/model_executor/layers/tilelang_ops/cache_common.py).

The server's KV prefix cache (`server/src/kv.rs`) keeps the states and their
snapshots between requests. Snapshots are taken at every prompt's end (with
its logits), at every generation's end, and at every fork's branch point:
where the prompt parts from every state, when that lies after the resume
point and before the prompt's end. The fork's prefill captures the rolling
rows there as it passes (`myds4_prefill` with a capture position), so a
later fork at the same point resumes there with nothing to replay. For a
prompt, the cache finds, among all states, idle or decoding, the longest
token prefix `k` that one of them shares with it, at any position and
whatever roles its tokens belong to, and picks the cheapest way to resume
there, counted in prefilled tokens. Without `--ced` every way is exact:

- **In place** (logged `kept`): an idle state that the prompt extends,
  whole or from up to its rewind room before its end, continues without a
  copy; its rows past the resume point and the snapshots past it are
  dropped. A state that a request decodes in is never continued in place.
- **Snapshot** (`saved` for the same prompt again, else `forked`): a new
  state copies the source's rows before a position `b <= k`, restores one
  of its snapshots at `b` or up to the snapshot's rewind room after `b`, and
  prefills from `b`. When `b` is the prompt's length, the snapshot is there,
  its logits are the prompt's and nothing is prefilled, so a retry decodes
  as the first try did. A thinking prompt ends with `<think>`, and the next
  turn renders that reply with `</think>` there: the next turn parts one
  position before the prompt-end snapshot and resumes from it this way,
  with nothing to replay.
- **Replay** (`forked`): a new state copies the source's rows before `k` and
  prefills from `k` with an exact replay (`myds4_prefill` with replay set).
  Layer 39 starts its window 136 positions before `k` (128 plus the rewind
  margin), and each layer below starts 127 positions earlier, about 5,100
  positions before `k` at layer 0: a layer's window keys come from the layer
  below's output there, which is right only a window after that layer's
  first row. So each layer rebuilds its rolling rows at `k` from the copied
  rows with the arithmetic of a cold prefill; the replay captures them as
  the new state's branch-point snapshot. It costs about 2,613 rows per layer
  beyond the prompt's new tokens for long prefixes, and never more than
  `k` (`myds4_replay_rows`).
- **Cold** (`new`): nothing is shared; a new state prefills from position 0.

The rewind room is `max(8, lookup_depth)` less the state's or snapshot's
stale rows: the margin's rows that rows written past where the state then
continued replaced. A ring is written at `p % ring` for position `p`, so
such a row takes the place of the margin's oldest row. A verify pass's rows
after the tokens it keeps (rejected drafts, or rows after the end token)
make as many stale rows as there are of them, unless more are stale
already. A prompt that continues a state before its end and stops short of
that end leaves one stale row for each old row after the prompt's end. Each
row written at a state's end replaces one. The cache counts them for every
state and snapshot (`stale` in `server/src/kv.rs`). Without the count, a
later resume within the margin would read a replaced row in every layer's
window, and its output would differ from a cold prefill's.

Ties go to the plan earlier in the list. A response's usage reports as
`prompt_cache_hit_tokens` (and `cached_tokens`) the prompt tokens whose rows
came from the cache: `k` for a replay, `b` for a snapshot, the resume point
in place. When a snapshot lies less than the replay's cost below `k`,
resuming there is cheaper, and the report is `b`; that fork captures its
branch point at `k` on the way.

With `--ced`, prefill is approximate already (CED prefill), and the replay
is bounded instead: every layer starts a ring before `k` (136 positions,
more with a wider `--lookup`), and the attention windows of the rows it
computes stop there. The compressed rows and index keys before `k` stay the
ones the new state copied; only the rolling rows at `k` are computed again,
from the copied global rows and the window that the replay covers, so they
differ from a cold prefill's. The replay costs at most a ring's rows
(`myds4_replay_rows`), so a snapshot wins only within a ring below `k`, or
within its rewind room after `k`, as at a thinking turn's prompt end. A
branch point's capture holds the encoder layers only (layers 0 through 20)
when layers 21 through 39 have not computed a ring before it in that
prefill: unless it lies at least a ring inside the retained decoder suffix.
The engine returns no snapshot then, and a later fork there replays again.

Every prefill batch runs the prefill (matrix) kernels, whatever its size,
so a row's bytes do not depend on where the prefill that computes it
starts: a resume over rows that came from prompts gives the same bytes as a
cold prefill of the whole prompt, in the logits, every state tensor and the
generated tokens. Prefill calls of 1 to 6 tokens therefore take 175-218 ms
instead of the 66-125 ms of the decode kernels (UD-IQ2_XXS). Decoded rows
are reused as they are: a resume over them gives what the conversation
itself continued there gives, which differs from a cold prefill of the same
tokens because decoding and prefilling round differently (on the fork
test's conversation, logits SNR 9.5-16.5 dB, the next token the same).

`--kv-cache` bounds what the states and snapshots take, apart from the
expert cache's budget. Before a request runs, the cache checks that the
room its prompt needs fits: its state at the prompt's end, the prompt end's
snapshot when it prefills (a retry of a saved prompt uses the snapshot
there and takes none), and its branch point's. It refuses the request
before anything is freed if that does not fit. Then it frees what is
needed: the spare snapshots first (all but each state's last prompt end and
last branch point), of the least recently used states first, then whole
idle states, least recently used first. A resume's source and every state a
request decodes in are kept, and a fork marks its source used. At most 256
states are kept.

A state that decodes past its room (16,384 positions at first, then each
doubling) is charged its new size before the pass that grows it, counting
the pass's draft and lookup rows (`kv::Cache::grow`), and the cache frees
room first in the same order. When only states that requests decode in are
left, the state grows past the limit, and the server logs by how much; new
requests are then refused until room is back. A generation's end snapshot
is taken only if it fits in the same way. `server/tests/prefix_fork.rs`
checks a retry at a full budget and a growing generation through HTTP.

The fork test checks this. `tools/checks/prefix_fork_check.c` forks a
conversation (a prompt, a decoded reply, a tool-like result, a second
decoded reply) at 26 position classes and 16 random positions, resumes each
fork by replay, from the latest snapshot and from the first snapshot after
it within the rewind margin, and compares the results with a cold prefill
and with the conversation continued at the fork. Each fork also resumes
once more from the branch-point snapshot it captured, which must give its
own bytes. It checks the stale-row count against the engine: a state with
3 stale rows (a short prompt continued before a state's end, or a verify
pass of 4 rows that keeps 1 token), resumed within the room the cache gives
it, must give the cold prefill's bytes, and so must one 3 tokens later with
the full margin; a resume past the room is printed. With `--ced N` it checks that a bounded replay
keeps the copied compressed rows and index keys byte for byte and gives
finite logits, and prints how far each fork lies from the cold prefill.
`server/tests/prefix_fork.rs` forks a conversation through HTTP while it
still decodes. Each fork must reuse at least its shared prefix less a
replay's cost, and no more than the prefix, and must give the tokens of a
server that starts without a cache; with `--ced 128` the reuse must be at
least the shared prefix less 136 ([Checks](usage.md#checks-and-diagnostics)).
A thinking turn's next turn must reuse all but the turn's last prompt token
and compute only its new tokens, with and without `--ced 128`, and without
`--ced` give the cold server's tokens.

### The live view

`--tui` gives stderr's alternate screen to a thread of the server
(`server/src/view.rs` and its tabs in `server/src/view/`, on ratatui and its
crossterm backend). Every 0.5 s it takes the worker's snapshot
(`worker::view`, as JSON) and keeps a sample of it; it draws the tab shown
with each snapshot, and at once after a key or a click that changes
something or when the answer to its question for a node's text comes. Input
that changes nothing draws nothing.
The engine writes the log's lines to the server's `on_log`, which keeps the
last 256 in a ring that the view shows; the lines logged before the view is
on the screen (while the model loads) also go to stderr. The snapshot takes
the queue's lock for the number of waiting requests and the status lock for
the requests, the totals and the cache's states and tree, which the worker
publishes there after each prefill and request end and once a second while
it decodes; it reads the engine's counters directly (`myds4_stats_get`), so
a long prefill's progress shows while it runs. The snapshot's JSON is built
under the status lock and written out as text after it. The tabs are
described in [Usage](usage.md#live-view).

**The last minute.** The view keeps 120 samples, one per snapshot, in its
own memory only: nothing is written to files or sent anywhere, and the
samples are gone when the view ends. A sample holds its time, the phase and
the engine's counters; each decoding request's tokens/s (over its last ten
snapshots); the requests that came since the sample before (numbers above
any seen); and, from the differences of the counters since the sample
before: while a prefill runs, the prompt rows computed per second; the
SSD's read rate; the decoded rows' hit rate; the GPU's busy share of the
decode steps' time; and the wait for reads per decoded row. The prompt rows
computed are the rows the engine counted (a chunk's when it ends) plus the
running chunk's rows times the share of its layers done, a count that only
grows, across chunks and between prefills. The read rate comes from the
count of every expert byte read, which grows as each read completes: the
prefill's own counter takes a chunk's reads only when the chunk ends.

The Overview's throughput takes each figure's mean over the last 10 samples
and over all of them, of the samples taken while it applies: decoding, a
prefill, or either for the SSD's reads. Its "since start" figures come from
counters taken together: the tokens generated (by the requests that ended
and those decoding, counted when the worker updates its status) over the
decode steps' time; and the prompt rows and expert bytes the engine had
counted when the latest prefill ended, over the prefills' time the worker
adds up then. A long prefill's chunks therefore join those figures when it
ends.

**The prefix tree** covers the 24 most recently used states. Depth first, a
node is the run of positions, from its parent's end, that all states of its
group share (compared 1,024 tokens at a time), and the states that go on
past it form groups by their token at its end. The tree holds at most 40
nodes. Each state keeps the positions up to which its last 32 reuses ran
(requests that continued it or resumed from a copy of it); a node counts
those that reach past its start, so a system prompt that many
conversations share collects the hits of all of them. A node also names a
state through it (`via`, the one used last), whose tokens hold its text.
The KV cache tab draws the nodes along the positions: a node's first child
goes on in its row and the others take the rows below its earlier subtrees.
The view knows the chosen node by a state through it (`via`) and its first
position. In each new tree it finds, on that state's path (from the state's
own node up), the node that holds the position, so the choice survives a
node that grows while its state decodes, a fork that splits it and an
eviction that merges it, while the state is in the tree.

**A node's text.** The view does not hold tokens; the KV prefix cache does,
on the worker's thread. The view posts its question, a state's number and
positions, beside the queue (`worker::ask_text`, which wakes an idle
worker); the worker answers between its steps with the tokens, at most
65,536. The view takes the answer that matches its question
(`worker::text_answer`) and turns the tokens into text on its own thread,
since the tokenizer serves any thread (`engine::text`): each token's bytes
(`myds4_piece`), the special tokens (the tokenizer's added ones,
`myds4_added`) marked for the accent color, and an image's run of keys as
"[image]". It wraps the text into lines once for the panel's width.

**The expert map's rows.** `routed_rows` reads each decoded row's router
result on the CPU anyway, to count its experts by where they are; for the
model's own layers in decode steps (not prefill, not the draft's stages) it
also keeps each row's experts and their places, with the step that chose
them (`model_routes`, `myds4_routes`). That is a copy of numbers already
read: decoding computes nothing more for it. With each snapshot the view
asks for the rows (`worker::ask_routes`, a flag). After its next decode step
the worker copies each decoding request's row of that step (row 0 of a step
or a verify pass, a state's row of `myds4_rows`) into the status, and the
next snapshot takes them, so the view gets each row once; without the view,
nothing asks and nothing is copied. The view draws a request's last 12.

The counters cost a few hundred additions per decoded token. A decoding
step adds the cache's statistics during the step to `decode_stats`, with its
rows, GPU time and wall time; a prefill adds them to `prefill_stats`. The
cache counts each expert once per layer and step for all rows, so
`routed_rows` also counts every row's selections, each by where its expert
is when the layer requests it (`cache_where`: in memory, read ahead, read
ahead and still being read, only on the SSD) or as a substitute. The
Overview's experts per token are the differences of the decoding counters
over the last 21 samples (10 s). Memory comes from `host_statistics64` (free,
file-backed, wired and compressor pages), `vm.vm_page_filecache_min`,
`vm.swapusage`, the accelerator's "In use system memory" (the GPU-resident
memory of all processes), and the cache's own counts. While the GPU idles,
that resident memory falls below myds4's mapped experts and weights: about
a minute after the last request it was 2.1 to 2.5 GiB against their 45.6
GiB, and wired memory fell from 56 to 8 GiB. The driver releases them until
the GPU runs again, so the view splits the resident memory into experts,
weights and the rest only while it holds them all.

**Colors.** The view draws in 24-bit color from one of two palettes, for a
light or a dark background. When it starts, before it reads keys, it asks
the terminal for its background (OSC 11; only when stdin is the terminal,
where the answer comes) and waits up to 0.2 s for the
answer on stdin: it takes the light palette for a bright background and the
dark one for a dark background, and tints (the selected node's, the last
minute's decoding phase) blend toward the background's own color. Without
an answer it takes the light palette; `--tui-theme light` or `dark` chooses
one without asking. When `COLORTERM` does not say the terminal shows 24-bit
color, each color becomes the nearest of the 256 standard ones.

A screen is drawn into ratatui's buffer from the chrome (the title and the
state, the tabs, the keys) and the tab; lines too long for the terminal end
after their last whole figure, with "…". The drawing also records the
places a click acts on: the tabs, the KV tree's nodes and the expert map's
requests. `make server-check` draws every tab from a fixed snapshot and a
generated minute on ratatui's test terminal at sizes from 220 x 60 to 12 x
4 and checks their parts, the keys and clicks, and the samples;
`server/tests/live_view.rs` runs the server with a model in a
pseudo-terminal.

While the view runs, the terminal is in raw mode, which neither echoes nor
buffers input and delivers Ctrl-C as a key, the cursor is hidden, and the
terminal reports the mouse's clicks and wheel (modes 1000 and 1006, without
the pointer's moves; `m` turns that off, so that the terminal can select
text). The view's thread waits for its next snapshot
by polling the keys and the mouse: `q`, or Ctrl-C, requests the stop that
SIGINT requests (`myds4_request_stop`) and logs "q: stopping after the
current step"; the server stops accepting within a second. SIGINT and
SIGTERM request that stop too. When the server ends, before a fatal error's
message (the engine's `on_die`, or the server's own) and after a panic, the
view stops the mouse's reports, leaves the alternate screen, restores the
terminal and writes the log lines that stderr has not shown yet, the last 30
at most. A frame being drawn finishes first. The view takes the snapshot
before the screen's lock, so an engine that fails while it holds the status
lock cannot keep the view from restoring the terminal. Without a terminal to
read keys from, the view draws, and the signals stop the server.


## Rationale

The decisions below were measured on the development machine on 2026-10-30
and 31. Runs of each comparison alternated, and other applications used 0
to 6% of the GPU unless a subsection says otherwise.

### Experts in memory before those being read

A decoded layer's experts used to run only once all of them were in memory,
so on a layer with a miss the GPU idled for the whole read, although most
of the layer's experts were resident. The experts in memory now run first
(see Decoding), at the cost of one more command buffer per layer and a
summing kernel; the result is the same bit for bit. Exact decoding with the
`UD-IQ2_XXS` experts, a 40 GiB cache, 150 tokens, two runs each; tokens/s,
the CPU's waits for the routing signals and the GPU's idle time per token:

| Run | All experts awaited | Experts in memory first |
| --- | ---: | ---: |
| Python, plain | 18.30 and 18.88; 27.3 and 29.3 ms; 24.4 and 24.5 ms | 19.20 and 19.21; 26.2 and 26.4 ms; 22.5 ms |
| Python, `--draft 2` | 18.61 and 18.84; 21.6 and 22.2 ms; 30.6 and 30.8 ms | 20.78 and 20.83; 13.5 ms; 25.2 and 25.4 ms |
| Story, `--draft 2` | 16.83 and 16.84; 28.8 and 28.9 ms; 27.4 and 27.5 ms | 18.56 and 18.61; 21.0 ms; 20.8 and 20.9 ms |

### Fewer dispatches per decoded layer

Each dispatch costs the GPU a few microseconds beyond its work, and a
decoded layer ran 32. Products of one shape on one input, the three
routings' top-k, and the hyper-connection split with the residual each
moved into one dispatch (24 per layer); later, products that share K and
the split moved into one dispatch with their own rows (23), and the one-row
FP8 kernel requests its weights before its inputs. Exact decoding, 150
tokens, two runs each:

| Run | 32 dispatches | 24 dispatches |
| --- | ---: | ---: |
| Story | 20.83 and 21.63 tokens/s; GPU 29.4 and 28.9 ms per token | 22.12 and 22.07; 27.2 and 27.3 ms |
| Python | 19.29 and 18.82; 28.9 and 29.1 ms | 19.85 and 19.22; 27.3 and 27.5 ms |
| Python, `--draft 2` | 20.85 and 19.97; 21.9 and 22.1 ms | 20.84 and 20.10; 21.3 ms |

The merged one-row products then took the profile (`MYDS4_PROFILE=1`,
story, 40 tokens) from 27.99 to 27.05 ms per token, the FP8 one-row
products from 14.78 to 13.96 ms; decoding ran at 21.74 to 21.79 tokens/s
against 21.22 to 21.54 (story) and at 19.22 to 19.25 against 18.67 to 19.12
(Python), three runs each. Variants of the one-row FP8 kernel that were
exact but not faster: loading the weights in rounds of 2 chunks (the shared
expert faster, `wkv` and the indexer's `wq_b` slower), rounds of 3 or 4, BF16
rounds of 2, 3 or 5, a rolling prefetch of the next chunk (slower
everywhere: its registers cost resident threadgroups) and threadgroups of
512 or 1,024 threads. `--bench` used to time `wq_a` first, on a cold GPU
clock (43.0 us against 14.7 us warm); it now times each matrix's 1,024-row
products first.

### Prompt kernels that stage less

On this GPU a core's matrix-unit products and its other work (decoding,
conversions, threadgroup memory) take turns rather than overlap. The
earlier tile kernels staged x and the weights in threadgroup memory per K
step: on IQ2_XXS codes the earlier `gg_up_m` spent 16.2 ms in its products,
11.3 ms decoding and about 6 ms staging x, 35.3 ms in all. Fewer columns per
threadgroup (32 or 16: 37.6 and 51.8 ms) and threadgroup memory padded to 32
KB (35.3 ms) did not make them overlap, so the kernels stage less (see
Prefill). Per batch of 2,048 rows:

| Kernel | Earlier | Current |
| --- | ---: | ---: |
| `gg_up_m`: Q2_K, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ1_S | 32.2, 34.2, 37.9, 36.7, 36.1, 33.6 ms | 19.7, 23.4, 23.3, 22.8, 23.6, 20.3 ms |
| `gg_down_m` with its quantization of h, the same codes | 14.2, 15.4, 17.6, 17.2, 16.1, 15.2 ms | 10.8, 11.7, 13.2, 13.2, 12.5, 11.7 ms |
| `mm_fp8` without its conversion: `wq_b`, `wo_b`, `sh1`, `sh2` | 5.06, 4.96, 1.51, 1.44 ms | 3.65, 3.57, 1.07, 1.06 ms |
| `attend_m` and `q_rope`, rows from position 4,000, 640 keys each | 25.2 ms | 15.8 ms |

Variants that were also exact but slower: 128 pairs per threadgroup
(`gg_up_m` on IQ2_XXS 25.4 instead of 21.6 ms without its gather,
`gg_down_m` 13.0 against 11.4 ms), products of 16 pairs for `w2` (11.8
against 11.4 ms), 64 heads per threadgroup in `attend_m` (17.2 against 15.8
ms), the softmax's maxima and exponentials on all threads (no change), and
dense tiles of 128 columns (0.70 to 1.02 times the earlier kernel's speed)
or of 256 rows (0.94 to 1.14 times). The matrix units reading x as FP32
changed 39% of the outputs and took 2.1 to 2.4 times as long. In alternated
pairs, the 8,912-token prefill took 19.8 to 24.9 s with the earlier kernels
and 14.8 to 18.4 s with these (1.19 to 1.40 times as fast, median 1.34); the
2,760-token prefill, which waits for the SSD, did not change.

### MXFP4 prompt tiles

The tile kernels of the checkpoint's MXFP4 experts (`moe_up_m`,
`moe_down_m`) used to stage the pairs' rows of x or h in threadgroup memory
as half for every K step of 128, between two barriers, and multiplied 32
pairs by 128 columns on four simdgroups together. They now read the rows as
half from device memory, copied once in pair order as for the GGUF codes,
and each simdgroup runs its own products of 32 pairs by 32 columns (see
Prefill); the outputs are the same bit for bit. Measured on 2026-10-31
while other applications used 0 to 34% of the GPU: on random inputs (the
harness of [Prefill kernels](performance.md#prefill-kernels), batches of
2,048 rows, best of 7 rounds, three runs) per batch, with the copy or
quantization of each kernel's input rows, and in the 8,912-token prefill's
profile (`MYDS4_PROFILE=prefill`, two runs each):

| Kernel | Staged rows | Rows from device memory |
| --- | ---: | ---: |
| `moe_up_m`, per batch | 23.7 to 24.2 ms | 21.7 to 21.8 ms |
| `moe_down_m` with its quantization of h, per batch | 13.1 to 13.3 ms | 10.8 to 11.1 ms |
| Both in the prefill's profile, with their conversions | 7.58 and 7.48 s | 6.23 and 6.26 s |

The 8,912-token prefill, which waits for the SSD (289 GB), took 22.74 and
23.42 s with the staged rows and 22.69 and 22.79 s with these, its GPU
working 13.84 and 18.50 s against 13.02 and 15.34 s; the 2,760-token prefill
took 20.34 and 20.46 s against 20.39 and 20.33 s, and a 15,965-token
prefill, which the GPU binds, 33.76 and 31.39 s against 30.21 and 29.78 s,
and later, while other applications used up to a third of the GPU, 36.18
and 38.18 s against 35.11 and 37.72 s. With the GGUF experts the 8,912-token
prefill did not change (16.18 and 16.40 s against 16.13 and 16.37 s).

Variants that were also exact but slower than these, per batch (the staged
kernels took 23.3 to 24.5 ms and 12.7 to 13.4 ms in the same runs): the rows
from device memory with the earlier products on four simdgroups (`moe_up_m`
30.4 ms, `moe_down_m` 11.5 ms), per-simdgroup products of 32 by 64 columns
(`moe_up_m` 49.9 ms, `moe_down_m` 23.5 ms), 32 by 128 (`moe_down_m` 91 to 94
ms), 64 by 32 (`moe_down_m` 24.5 ms) or 16 by 32 (`moe_up_m` 24.3 ms), `w1`
and `w3` in one simdgroup (23.8 ms), 64 or 128 pairs per tile (`moe_up_m`
23.9 to 25.8 ms), eight simdgroups per threadgroup (`moe_up_m` 22.5 to 23.7
ms), and the rows staged as before with per-simdgroup products (26.0 ms).
Two simdgroups per threadgroup were as fast for `moe_up_m` (22.1 ms), and so
were two or eight simdgroups per threadgroup, 64 pairs per tile and products
of 16 by 32 for `moe_down_m` (10.8 to 11.2 ms).

### Prompt read-ahead

A prompt batch of 256 rows or more used to read every expert of the next
layer ahead, also experts that no row selects, and prompts that wait for
the SSD waited for those reads too. Reading ahead only the rows' top 8
predictions for every prompt kept the outputs but slowed the 8,912-token
prompt by 7 and 18% in two pairs (from 14.89 to 15.95 s and from 19.35 to
22.88 s),
its GPU time growing as well, presumably because the short waits let the
GPU's clock drop. So only prompts of up to two batches (4,096 tokens) read
ahead less (see Prefill). Against reading every expert: 1,710 tokens took
10.53 and 10.52 s instead of 11.80 and 11.79 s (138 GB read instead of
155), 2,760 tokens 11.30 and 11.29 s instead of 12.21 and 12.19 s (145.1 GB
instead of 157.6), and 8,912 tokens 14.76 and 14.68 s against 14.67 and
14.78 s.

### Attention keys as BF16

The window keys, compressed rows and index keys were FP32, although each
value is an FP8 or FP4 code times a scale, which BF16 holds exactly (see
Numerics). As BF16, a session's state, which then also held two saved
copies of its rolling rows, took 66 MiB at first instead of 132, and
3.14 GiB at 1,048,576 positions instead of 6.28. The 8,912-token
prefill took 17.61 and 14.52 s against 18.48 and 15.48 s (`attend_m` in the
prefill profile 2,312 against 2,558 ms); decoding did not change.

### The substitution rule

The earlier rule let any missing selected expert give way, the
highest-ranked first, so the experts that weigh most were often replaced.
Skliar et al. (arXiv 2412.00099) add a bias to the logits of cached experts
(the cache prior), keep the top-ranked experts, and compare that with a
maximum-rank rule, a cumulative-sum threshold and pruning; in their
measurements the cache prior Pareto-dominates those three. They also scale
the bias per layer by a running average of the router's logit range.
myds4 applies the cache prior to DeepSeek's selection keys, keeps the two
highest scores, replaces the lowest-ranked missing expert first and uses
one D for every layer (see Approximate decoding and prefill). Replayed
offline on two 2,000-token routing traces of the `UD-IQ2_XXS` experts (a
story and a Python module, recorded on 2026-10-30 at a 40 GiB cache) with
a demand-only cache of 3,523 slots evicting by score; per token, the reads
and the routing weight replaced (the replaced experts' shares of their
layers' weights, summed):

| D | Story: earlier rule | Story: current | Python: earlier rule | Python: current |
| --- | ---: | ---: | ---: | ---: |
| 0 (exact) | 22.2 reads | | 32.6 reads | |
| 0.1 | 10.4, 1.33 | 10.8, 1.26 | 14.5, 2.70 | 15.9, 2.30 |
| 0.3 | 4.8, 2.62 | 5.6, 2.28 | 6.0, 5.23 | 8.4, 3.71 |
| 0.5 | 3.1, 3.85 | 4.2, 2.74 | 3.4, 7.45 | 6.7, 4.26 |
| 1.0 | 1.6, 9.43 | 3.6, 3.13 | 1.6, 14.43 | 5.9, 4.64 |

At an equal number of reads the current rule replaces about 3% less weight
on the story and 12% less on the Python module (at 5 to 6 reads per
token); keeping only the top one gave 7% less on the Python module, the new
order alone 2 to 5%. A given D now substitutes less: D = 0.5 reads about as
many experts as 0.3 did. Scaling D by each layer's gap between its 6th and
12th keys changed the routing weight replaced at a given number of reads by
1 to 4%, so D stays one number. The key gaps are the router's own and alike
on prose and code, so nothing is calibrated per task. In decoding the
README.md text at D = 0.3 (one run each, 649 positions), the earlier and
current rules gave a mean KL from exact decoding of 0.025 and 0.032 at 25.8
and 22.2 substitutions per token; runs vary by more than that.

### Substitution in prompts

With substitution only while decoding, a prompt read every expert its rows
selected, so on a server whose cache held the experts of earlier requests
a new prompt still waited for the SSD. Letting each row substitute by
itself would replace experts that other rows of the batch read anyway; a
batch therefore reads an expert if some row needs it, and only the other
missing selections give way (see Approximate decoding and prefill). With
`--substitute 0.5`, after a warm-up that prefilled 32 tokens of
`inference/model.py` and decoded the next 192, the decoded positions' mean
KL from exact decoding (in brackets, the positions with the same most likely
token):

| Prompt pass, then positions scored | Prompt exact | Prompt substituted |
| --- | ---: | ---: |
| README.md: 400 tokens, then 250 | 0.0079 (246) | 0.0113 (245) |
| `encoding.py`: 300 tokens, then 166 | 0.0416 (159) | 0.0517 (157) |

At D = 1.0 the same runs gave 0.0075 against 0.0099 and 0.0391 against
0.0523. The prompt passes took 5.9 and 6.5 s exactly and 2.3 and 2.7 s
substituted (78 and 86 GB read against 30 and 36). Through the server (one
request warming the cache, then new prompts; two runs each), new prompts of
105, 349 and 1,018 tokens took 3.5 to 3.6, 6.1 to 6.2 and 8.3 s with the
prompts exact and 1.5 to 1.6, 2.9 and 4.4 s substituted.

### Verify passes and the draft

Reading ahead for a verify pass's rows as for decoded rows (exact decoding,
2026-10-30) gave the Python prompt's `--draft 2` 18.73 to 18.95 tokens/s
against 18.26 to 18.29 and the story's 16.70 to 16.73 against 16.17 to
16.29, with 26.0 and 22.3 demand reads per token instead of 32.5 and 28.3.
With `--substitute 0.3` the same read-ahead gave the Python prompt 26.9
tokens/s against 28.8 to 29.4: more selected experts were still being
read, and so waited for instead of replaced (33 substitutions per token
instead of 38), so verify rows read ahead less there. Running the draft's
head only on the rows whose drafts a pass uses cut the drafting from 3.1 to
3.2 ms to 2.6 to 2.7 ms per token with `--draft 2`.

### The checkpoint's experts

With the checkpoint's MXFP4 experts on 2026-10-31 (a 36 GiB cache, two
runs each, alternated, other applications using 13 to 22% of the GPU), the
build of 2026-10-30 13:15 against that of 2026-10-31 08:39 (the same texts):

| Workload | 2026-10-30 13:15 | 2026-10-31 08:39 |
| --- | ---: | ---: |
| Decode, story prompt, 150 tokens | 7.41 and 7.38 tokens/s | 9.56 and 9.61 |
| Decode, Python function prompt, 150 tokens | 7.44 and 7.01 | 8.01 and 7.89 |
| Prefill, 2,760 tokens | 21.97 and 21.99 s | 20.48 and 20.31 s |
| Prefill, 8,912 tokens | 29.46 and 25.67 s | 25.02 and 23.22 s |

Between the two builds came every change above that applies to all expert
formats, and those of the evening of 2026-10-30 (every slot kept resident,
a faster decode attention, the GPU's resident-memory limit); the routed
experts' prompt kernels are the same in both.

### Experts mapped for the GPU on use

When every slot stayed mapped for the GPU, the GPU's limit on resident
memory (about 55 GiB for all processes on this machine) capped the cache
near 40 GiB, whatever `--cache` asked for. The build of 2026-10-30 13:15
took slots unused for 8 tokens out of the residency set, which kept the
resident memory low (36.3 GiB of 68.9 GiB allocated at a 52 GiB cache) but
remapped constantly: it decoded at 18.7 tokens/s at 52 GiB, against 19.8 to
20.3 at about 40 GiB with every slot mapped (the GGUF experts, 2026-10-30).
The cache now maps a slot when a layer needs it and takes slots out of the
set only between tokens, and only as many as exceed the limit (see The
expert cache and memory). Mapping a slot costs about 0.1 ms of CPU time
(the residency set's commit), a tenth of reading the expert from the SSD.
With the mapped slots capped at 868 (a test build), every regression check
gave the same outputs while 82 to 99 slots per token were mapped again, and
decoding ran within about 5% of its usual speed. At `--cache 64`, which
settled at 44 to 53 GiB, decoding was 4 to 11% faster than at 40 GiB with
the MXFP4 experts and 1 to 7% faster with the GGUF ones, exactly (see
[Performance](performance.md#expert-formats-compared)).

### The file cache as available memory

The engine and benchmark preflight count free pages plus reclaimable file
pages. For example, 58 GiB free and 36 GiB of file cache provide 94 GiB
available on a 128 GiB machine, so the TUI reports 34 GiB used. File cache
remains visible as a separate dim segment of the memory bar.

The CPU checks cover zero and changing kernel cache targets, failed target
queries, speculative-page accounting, initial allocation from file-cache
headroom and calm growth. They also verify that actual pressure, new
swapping still reduce expert capacity, while compression alone does not. The TUI regression
checks the displayed used and available totals.

### Eviction and read-ahead predictors

Replayed on two 2,000-token routing traces (a story and a Python module,
2026-10-30, 3,523 slots) without reading ahead, `tools/analysis/cachesim.py`
gave these demand reads per token:

| Policy | Story, 3,523 slots | Story, 4,400 | Python, 3,523 | Python, 4,400 |
| --- | ---: | ---: | ---: | ---: |
| Belady's optimal | 11.5 | 8.4 | 17.0 | 12.2 |
| Scores, half-life 50 one-token passes | 22.2 | 16.5 | 32.6 | 24.5 |
| Scores, half-life 25 / 100 / 200 | 23.7 / 22.2 / 22.9 | 17.4 / 15.5 / 16.0 | 34.5 / 33.8 / 35.4 | 25.9 / 24.4 / 25.5 |
| LRU / ARC / SLRU | 25.2 / 24.6 / 22.4 | 18.0 / 17.0 / 16.0 | 37.2 / 36.7 / 34.1 | 27.3 / 26.5 / 25.0 |

Adding a fraction of the weight to the experts the router ranked 7th to
12th (or lower) raised the reads by 0 to 24%. In the timing model (front
0.53 ms and experts 0.15 ms per layer, 0.86 ms per expert read), myds4's
reads ahead gave 23.9 and 20.0 tokens/s against 21.6 and 18.1 without
them. Other predictors, replayed on two 1,500-token test traces (the same
prompts) after fitting on three others (an explanation, a dialogue, a C
program): ridge maps from a layer's router logits (or its and the previous
layer's) to the router logits one, two and three layers on named 38 to 59%
of a layer's experts in their top 6, against 59 to 83% for myds4's
predictions; rank fusion of the predictions available for a layer, and a
map from layer 39 to the next token's first two layers (24 to 26% in its top
6), did not help either. In the timing model every combination of them
decoded 1 to 5% slower than myds4's three predictions, and weighting the
scores of layers 0 and 1 (whose misses are the most frequent) by 2 to 8
changed nothing in those replays. Current inference defaults to a score
half-life of 50 forward-pass advances and the three router forecasts.
`--cache-half-life` and `--cache-read-cost` configure retention;
`--online-predictor` enables learned corrections to the router forecasts. See
[The expert cache and memory](#the-expert-cache-and-memory) for the current
policy and its units.
