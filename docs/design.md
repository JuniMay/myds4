# Design

## Files

The C sources are in four folders:

- `src/engine/`: the engine. `build/libmyds4.a`, which the server links,
  holds these files and `src/experimental/online_predictor.c`.
- `src/cli/`: the command line (`main.c`) for generation, the checks and the
  benchmarks. `build/myds4` holds it, the engine's files without `engine.c`,
  and the online predictor.
- `src/kernels/`: the Metal kernels, which `src/engine/model.c` embeds.
- `src/experimental/`: opt-in predictor research code. `build/myds4` links the
  online predictor, which runs only with `--online-predictor`; only
  `build/myds4-predict-lab` links the predictor lab.

The Makefile adds `-Isrc/engine`, so C files in every folder include the
engine's headers by name.

| File | Contents |
| --- | --- |
| `src/engine/model.c` | Weights, request state, the forward pass for decoding and prefill, `--bench`. |
| `src/engine/io.c` | SSD reads, the routed-expert cache, memory monitoring. |
| `src/engine/gpu.c` | Metal through the Objective-C runtime from C, the profiler. |
| `src/engine/tok.c` | The tokenizer and the embedded tables. |
| `src/engine/util.c` | Errors, strings, a JSON parser for the model's own files, the safetensors index, GGUF headers. |
| `src/engine/vision.c` | Images: decoding, the reference's grid, Pillow's resize and padding, the vision encoder's patches and RoPE angles, an image's span of tokens. |
| `src/engine/sample.c` | Shared temperature/top-p sampling and DSpark prefix verification. |
| `src/engine/expert_batch.h` | Row grouping and GPU descriptors for the experimental shared-expert kernels. |
| `src/experimental/context_lookup.c`, `src/experimental/context_lookup.h` | Bounded CPU history matching for the CLI and server's context-copy drafts. |
| `src/engine/engine.c`, `src/engine/engine.h` | The engine behind a C ABI for the server: the queue, sessions, batching, the status, the live view's snapshot and the log, without HTTP or JSON (see [The server](server.md)). |
| `src/kernels/kernels.metal` | The GPU kernels. |
| `src/kernels/gguf.metal`, `src/kernels/gguf_grids.metal` | The kernels for routed experts in GGUF codes (`--experts`), and their codebook grids, generated from llama.cpp's `ggml-common.h` (MIT License). |
| `src/kernels/vision.metal` | The vision encoder's kernels. |
| `src/cli/main.c` | Command line, chat template, sampling, statistics, `--check-prefill`, `--perplexity`. |
| `src/experimental/online_predictor.c`, `src/experimental/online_predictor_math.h` | Online CPU prefetch correction, owned forecast/label records, recursive least-squares updates and learner metrics. |
| `src/experimental/predictor_lab.c`, `src/experimental/predictor_lab.h` | The predictor experiment build (see [Live predictor experiment build](#live-predictor-experiment-build)). |
| `server/` | The server, in Rust on `deepseek-recipe`: HTTP, request conversion, prompts, output and the live view (`--tui`; see [The server](server.md)). |
| `tools/checks/` | Regression checks: C programs that the Makefile builds (most need no model or GPU), and Python checks of the command line and the server. |
| `tools/reference/` | `check.py`: myds4 against the checkpoint's references (tokenizer, the server's prompts, vision encoder, model), and checks of a running server. `gguf_decode_check.c`: IQ1_M decoding against llama.cpp's. |
| `tools/bench/` | Serial benchmarks, and `gpu_guard.py`, the GPU, memory and power guard that they and the other model runners share. |
| `tools/analysis/` | `cachesim.py`, the expert-cache simulator for routing traces, and `expert_probe.py`, the reader of `build/myds4-predict-probe` captures. |
| `tools/data/` | `gen_data.py`, which generates the embedded tables in `data/` from the checkpoint. |
| `tools/experimental/` | Checks and runners of the experimental code: the online predictor's checks, the predictor lab's checks and runner (`predictor_lab.py`), and the capture code of `build/myds4-predict-probe` (`predict_probe.c`). |

Metal is used from C: every message goes through `objc_msgSend` cast to the
method's exact type, so the project has no Objective-C or C++ sources besides
the Metal kernels. The kernels compile at startup with the IEEE math mode and
Metal 4.1, which provides the tensor operations that drive the GPU's matrix
units.

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

The window keys of a layer live in a ring of `128 + max(5, lookup_depth)`
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
(`max(5, lookup_depth)`), and apply replay
only when the new input is longer than that sum. The replay boundary is
chosen across the whole forward call, including multi-chunk prompts. They
preserve all encoder/global KV work and prime DSpark only from computed
decoder inputs. Decoding runs normally after prefill. A compact prefill can
change local KV, routing and output; it is not an exact-output optimization.

The retained `tools/checks/prefill_prefix_check.c` compares compact execution
with the same full-shape attention mask, including DSpark state, prefix
restoration and chunk boundaries. `tools/checks/server_prefill_check.py`
checks the option and cache reuse through HTTP. Its shared guard in
`tools/bench/gpu_guard.py` checks GPU-client ownership, free memory, AC power
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

### Experimental sharing across expert rows

`--expert-batch` enables expert sharing in the CLI and server. The
`build/myds4-expert-batch` diagnostic build enables it by default. In
batches of two to sixteen rows, it groups up to three uses of the same
resident expert. Each group shares weight loads and decoding while keeping
an accumulator, routing weight and output position for each row. Fixed loop
sizes let the compiler keep those accumulators in registers. The available
paths cover MXFP4 and the Q2_K, IQ1_S, IQ2_XXS, IQ2_XS, IQ2_S and IQ3_XXS
codes used by the IQ2 expert build.

Rows retain their scalar lane assignment, order of additions, BF16 rounding
and FP8 activation quantization. Down products go into `PARTS`; the existing
`moe_parts` kernel sums each row's experts in expert order and performs its
hyper-connection expansion. A phase with no repeated expert uses the existing
per-row kernels. The single-row path and unsupported expert formats also use
those kernels.

The CPU groups entries after routing and substitution have selected the
experts. Ready experts run before the SSD wait; late experts run afterward.
Each layer has separate descriptors for those phases, held until the GPU
finishes the pass. The descriptor allocation is 480 KiB when sharing is enabled. This preserves the
existing overlap of expert computation and reads. The SSD request path
already requests the union of selected experts across rows; grouping targets
GPU weight traffic and decoding work.

The experimental executable prints aggregate `expert_batch` counters on
exit: grouped calls, row/expert pairs and weight groups. These describe
issued kernel work. They do not measure physical GPU memory traffic, SSD
traffic, cache hits or emitted-token speed.

The retained checks are:

| Target or tool | Scope |
| --- | --- |
| `build/expert-batch-check` | Synthetic GPU comparison against the scalar kernels, including all one-to-six-row sizes, three sharing patterns and whole/split/empty-first read phases. `--check-only` omits timing; `--format` selects a supported code. |
| `build/expert-batch-model-check` | Experimental engine: compare all six verification rows' logits with sequential decoding from the same saved state. Arguments are the checkpoint directory and, optionally, the expert GGUF file. |
| `build/expert-batch-model-control` | The same all-row check with the ordinary engine. |
| `tools/bench/expert_batch_bench.py` | Serial ordinary/grouped comparisons with identical prompts, sampling, draft settings and cache budgets. It checks output identity, actual cache bounds and that the candidate issued grouped work. |

The GPU checks take the same model lock as inference. The benchmark runner
also checks device activity, other compute processes, AC power and memory;
it interrupts only its own child when interference appears. Source excerpts
form long repository-review traces. The runner counts the complete chat
template and stops excerpts before image-marker literals. It keeps prompts
and generated answers in memory, and can save bounded aggregate results.
Generated code is not executed. These workloads measure inference on
agentic-style histories; they do not measure autonomous task completion.
Before loading, the runner requires the requested expert-cache budget,
14 GiB for weights and working buffers, and the explicit engine reserve to
be free. The reserve defaults to 10 GiB; `--reserve` sets it for every
comparison arm and for the guard. The native memory-pressure policy remains
active, and a run with cache shrinkage or new compression/swap is rejected.
The all-row model check requests a 20 GiB cache and a 4 GiB reserve; it
compares logits and makes no throughput claim.
The default timing admission limit is 20% device activity. Explicit
`--desktop` comparisons record ordinary desktop GPU activity without treating
its aggregate utilization as another model job. Process exclusion and the model
lock still apply. The runner records device/renderer activity during the
run. This does not establish exclusive GPU use; desktop-load
timings must be reported with that limitation and the repeat-control range.
In this desktop mode, Python jobs and known compiler/build tools are
logged as background CPU work when the driver exposes GPU-client
ownership and neither they nor their descendants own a GPU client. During
inference the model's own GPU client must also be visible for that exception
to apply. GPU clients, named model/probe jobs and the model lock
still block a run; memory bounds, cache shrinkage and new compression/swap
remain grounds to reject a comparison. CPU work does not need a separate
reservation. The default quiet mode retains the broader Python-job
exclusion. Background-helper observations are retained with the result.

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
prefill. `--prefill-plan split|joint` compares system-prefix scheduling.
`--ced off|N` uses production replay, including its verification rewind margin.
The diagnostic `--compact --keep N` instead retains exactly N decoder rows.
`--dspark N` enables 1–5 native drafts, 32 expert slots per stage and trimming.
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
has started and that are not among its top 6, then reads its top 3 that are
not cached, while fewer than 6 read-aheads are reading. A read-ahead is not
evicted before its layer's routing; after it, one the layer did not select
keeps the score of its earlier selections (see the cache below).

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
GPU routing, server requests, images, GGUF experts,
profiling and diagnostic modes. Those modes retain their existing constraints
and execution paths.

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
native expert IDs before replacing selections. GPU routing likewise uses
native `RT` labels rather than substituted `RU` selections. It copies features
before submitting the next layer, which reuses the forecast/router buffers,
and updates after submission. The GPU-resident route remains active; missing
experts use its existing CPU repair path.

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
| `.venv/bin/python -B tools/checks/online_predictor_cli_check.py` | 41 CLI checks of boolean enable, state paths, removed options, compatible modes and existing constraints, before model loading. Includes empty and nonempty profiler values and the `--reserve` range. |
| `make build/online-predictor-state-check`, then `build/online-predictor-state-check` | 42 CPU checks of round-trip and continued-learning identity, file validation, binding mismatch, unsafe paths and atomic-save failures. Temporary files are removed. |
| `make shutdown-check` | 8 CPU checks of signal handling and safe-boundary shutdown notification. |
| `make build/online-predictor-check`, then `build/online-predictor-check --rank 64 --pairs 77 --milliseconds 20` | 20 shared-math checks with 20,000 dead/repeated-feature update rows, followed by a bounded CPU microbenchmark. |

The cache checks listed under [read completion](#read-completion-and-cache-counters)
cover their own implementation behavior. Numerical and performance evidence
from model runs belongs in [Performance](performance.md).

### Live predictor experiment build

`make build/myds4-predict-lab` builds the optional experiment module in
`src/experimental/predictor_lab.c`. The ordinary executable does not link this
module. `tools/experimental/predictor_lab.py` runs serial CLI controls and
experiments with bounded output held in RAM. It does not save traces, model
output or learned weights. The runner stops on changed output, process failure,
cache shrinkage, compression or swap. A repeat control measures timing drift
across a trial.

Before every run, the runner requires three quiet GPU/memory samples. It
checks process names for other model, IREE, kernel-probe and substantial Python
workloads, requires GPU device activity at most 20%, resident GPU memory at
most 8 GiB, and free memory at least the requested cache plus 24 GiB for
runtime allocations and reserve. Large model runs require AC power. It waits
up to `--gpu-wait` seconds, default
900, without loading the model. The existing inference lock remains in use.
During inference it checks for other compute processes and new compression
or swap. On interference it sends SIGINT only to its own child and discards
that run. It never stops another agent's job. These checks are conservative
observations, not an exclusive GPU reservation; short activity between polls
can escape detection. Guard snapshots accompany new trial records.

The runner uses two DSpark drafts, 32 slots per draft stage, temperature 1,
top-p 1, seed 12,345, thinking effort 75 and context capacity 8,192. Its
default 512-token trials use a 28 GiB total expert-cache budget. It accepts
checkpoint and GGUF paths as arguments. The scheduler fixture and an original
scheduling-reasoning prompt provide distinct workloads; generated code is
never executed. Existing online correction is enabled unless
`--native-router-only` selects the engine's uncorrected router forecasts.

`--phase shadow` gathers signals from actual inference and fits ten
predictors on the same observations. It issues no additional reads. Its
timing includes all ten learners and is not a deployable throughput result.
`--phase live` enables one variant's reads in the actual cache. Its decode
time includes collection, prediction, training and cache scheduling.

Each joint predictor has a fixed projection from 584 features to 127 ReLU
features plus a constant. Recursive least squares learns a shared
128-by-15,360 output matrix for all 40 backbone routers. Rank is configurable
from 16 to 256; each model uses its configured rank instead of 128. Every
forward pass predicts all verification rows before consuming any of that
pass's labels. Updates occur after all 40 layer labels have been copied.
The first 16 passes train without scoring or additional read admissions.
Rejected verification rows are included, because the verifier computes them.
`--feature-gain` sets the norm of the nonconstant latent features; the default
is one. The constant remains one. This changes the strength of regularization
relative to the features and is recorded with each new experiment.

Input tensors are normalized and compressed into signed 64-value sketches.
These experiments test compressed features, not unrestricted full-tensor
neural encoders. Each named variant contains token embeddings and its named
extra group; `all` contains all nine groups:

| Variant | Additional input and availability |
| --- | --- |
| `embedding` | Current/proposed token embeddings from the engine's ordinary embedding read. |
| `hidden` | Normalized FFN inputs from all three draft stages, at existing synchronization points. |
| `streams` | All four residual streams at each draft stage's FFN input. |
| `mix` | Draft FFN mixing coefficients, stream norms and pairwise correlations. |
| `router` | All 128 router logits from each of the three draft stages. |
| `attention` | Draft attention queries and projected attention outputs. |
| `kv` | Four recency-bin summaries of the draft stages' cached window keys, at most 128 positions. These are derived from the last three backbone layers, not the whole model's compressed KV history. |
| `history` | Decayed routing labels of accepted rows, with previous accepted length. |
| `final` | The final normalized draft-head inputs and all four final residual streams. |

Verification proposal row r uses draft generator row r-1. The seed row uses
row zero as context and has an explicit seed flag. Position and row-offset
features are shared by the variants. History updates use the last accepted
row, excluding rejected rows from the history feature.

With `--base none`, the learner fits native top-six indicator vectors.
`--base embedding` or `--base draft` adds a router-informed forecast:
token embeddings, or the draft head state with its normalization weights
removed, are normalized with each target layer's FFN weights and projected
through its router on the GPU. The seed uses its embedding in both modes.
The learner then fits normalized routing-key residuals. These projections
approximate future FFN inputs; they do not compute those future inputs
exactly. Their GPU execution and synchronization are included in live timing.

The `rank` admission policy offers each row's leading predictions. The
`miss` policy ranks only uncached candidates, using the maximum score across
rows, and uses the ordinary lowest-score victim from an eligible shortlist.
The `bandit` policy learns fetch/abstain and eviction choices for that same
uncached candidate pool. Plans start at verification entry and refill while
the backbone advances, by default at distances two through eight. Existing
one-/two-layer forecasts continue to refine unstarted reads. The experiment
does not assume free I/O lead time before verification starts.

`--long-range` selects a separate set of 68 source/target heads for four-
and eight-layer backbone forecasts. The GPU projects the actual current
backbone FFN input through each available target router before the routing
signal. Each CPU head uses normalized forecast and current-router logits,
a fixed ReLU projection and online RLS output correction. Its latent feature
scaling follows the ordinary corrector; `--feature-gain` applies only to the
joint DSpark model. Labels are copied at the target layer, and updates occur
after that layer's expert commands are submitted. Source records are owned
and checked against the current pass and row count.

The eight-layer forecast can start a read. The four-layer forecast cancels
unstarted reads outside its top-six union. One-/two-layer engine forecasts
provide the later refinement. The first candidate can be admitted directly;
lower four-layer candidates require support from the preceding eight-layer
forecast. `--long-missing-first` skips cached candidates before selecting the
first offered expert from each row's top six. The shared in-flight limit
continues to apply. GPU projection, CPU fitting and admission costs are all
included in live timing. The long-range experiment requires CPU routing.

#### Cache-action controller

`--policy native --native-router-only` tests action learning using the
engine's existing residual and one-/two-layer forecast candidates. This mode
does not allocate or execute the joint DSpark predictor or its extra GPU
projections. `native-control` uses the same eligible victim shortlist without
learned admission or victim ranking.
`demand` disables optional backbone read admissions during decoded/verification
passes. It keeps mandatory expert reads, draft-stage reads and prefill behavior.
The engine still computes its native forecast tensors, so this is an admission
ablation rather than a measurement of removing all forecasting computation.

The controller is a contextual bandit with delayed feedback, not a
multi-step Q-learning agent. It estimates fetch value and retention value
with two 16-feature linear models fitted online by recursive least squares.
Features include candidate score or rank, forecast distance, forecast source,
pending prefetch count, expert bytes, prior routing frequency, resident age
and cache retention score. It fetches when estimated benefit exceeds the
selected victim's estimated retention cost; otherwise it abstains.

When a full cache needs a victim, the engine supplies entries with the lowest
ordinary retention scores: eight by default, or 1–128 with
`--victim-candidates`. Pending reads, pinned slots,
protected prefetches, current-layer experts in use and published GPU entries
are excluded. The controller must return a member of that shortlist. Demand
reads bypass its admission decisions. Exploration uses a separate random
generator, with at most one exploratory decision per verification pass.

Each admitted action has a unique ID attached to its slot. A later use gets
credit only if that same admission still owns the expert. Reader callbacks
accumulate its completed logical read bytes. At first use, its reward is the
fraction of the transfer already complete, scaled by payload bytes per 20 MB.
First use in a later pass discounts that value by one half per pass. Unused
admissions receive a penalty of `--waste-cost` times their completed transfer
cost, with a default of one half;
an unread cancellation therefore receives no transferred-byte penalty.
Reuse of an evicted victim during the remaining pass or either of the next
two passes adds a cost scaled by that victim's payload. Updates wait for
that observation window and for any still-owned read to finish. The bounded
table holds 512 outstanding actions; a full table causes abstention.
Unresolved final actions are not counted as settled outcomes. Empty slots
take precedence over occupied victims, including during exploration.

This reward is a read-progress/reuse proxy. It is not measured counterfactual
wall time. Actual read bytes, first-use attribution, eviction effects
and output throughput must therefore be evaluated separately. Neither a
positive proxy reward nor routing recall establishes a speedup.

The JSON `predictor_lab` report records mode, settings, scored rows, per-variant
accuracy, missing-at-forecast coverage, CPU times and action outcomes.
Missing-at-forecast coverage includes pending reads and is not a fresh-prefetch
hit rate. CLI `decode_cache` records the minimum and maximum allocated cache
bytes after forward passes and released-slot counts; the runner rejects any
decode that did not retain the requested budget throughout.
`prefetch_metrics` records actual admissions and first uses; issuer `draft`
identifies the extra DSpark plan. Native action-controlled reads retain their
original `pc`, `h1` or `h2` issuer. Existing residents receive no new-admission
credit. The runner compares generated output within each quantization.

`build/predictor-lab-check` checks signal wiring, causal updates, accepted-row
history and delayed action learning without a model. `build/cache-action-check`
checks victim eligibility, abstention, ownership and rejection of invalid
choices without GPU execution. Live results belong in Performance; the
experiment build is not a production speed recommendation.
The lab updates accepted-row history once for each observed DSpark pass.
Context-copy passes provide no DSpark features and do not reuse its previous
labels. The ordinary online predictor can still observe their target rows.

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

### Experts looked up by the GPU

`--gpu-route`, off by default, lets a decoded token's layers run without
waiting for the CPU between a layer's routing and its experts. It applies
to a token decoded alone, not to verify passes or to several sequences at
once. Each layer is one command buffer: the front, the `resolve` kernel, the
routing signal, the shared expert, then the routed experts as `resolve`
names them. The CPU encodes each layer's buffer two layers ahead, and the
head's in a buffer of its own, since the head's norm overwrites the FFN
input that a recomputation of the last layer's experts reads (see below).
It commits a layer's buffer as soon as the layer before has found all its
own experts in memory, or once it has had them computed again.

Before committing a layer's buffer, the CPU publishes the layer's table:
for each expert, the GPU addresses of its three spans if its slot is read in
full, 1 if it is still being read, else 0. It publishes the table before it
waits for the routing signal of the layer before, and again just before the
commit if experts were still being read. A ready slot that joined the
residency set during this token is committed into the set first, since the
GPU may use it only through `gpu_use` until then. Until the next layer's
table, no slot of the published layer is evicted or released.

`resolve` computes the selection keys as the route kernel does (with
`--substitute`, it substitutes an expert without a slot in the published
table by the CPU's rule),
writes the layer's six experts in expert order into entries of their own in
the experts' table, and reports the experts used and how many are not ready.
If one is not, every entry is null, the expert kernels return at once,
and the CPU computes the layer's experts as in the two-buffer path, in a
command buffer of its own with entries of its own. That buffer joins the
queue before the next layer's, so it runs after the layer's pass and before
the next layer. Each layer writes its routing and predictions into rows of
its own, which the CPU reads after the next layer's buffer has been
committed.

The GPU runs the same kernels on the same inputs in the same order as in
the two-buffer path, so the output is the same bit for bit
(`--check-batch --gpu-route` compares the two). It is 0.5 to 1.1% faster
(see [Performance](performance.md#experts-looked-up-by-the-gpu)): by the
gaps between command buffers, a layer whose experts are in memory already
starts them within 100 us of its front's end in the two-buffer path, and a
layer whose expert is still being read waits for the SSD either way.

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

`--draft-trim`, off by default, trims only the final stage's query/output
rows and feed-forward networks (FFNs) to the number of drafts requested by
that pass. Its attention still receives all five key/value rows, and the
first two stages still compute all five rows. The final requested count can
be smaller than `--draft N` near the generation or context limit. The head
already processes only the requested rows, with or without trimming.

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
The window ring retains `128 + max(5, lookup_depth)` rows. Compressor state
retains eight rows for native DSpark widths and 32 for wider context-copy
verification. Compressed and index rows are rewritten before use, the
Engram history is rebuilt from the kept tokens, and the draft's windows
receive only the kept positions.

With exact expert routing, verification uses the main model's requested
sampling distribution. Cache settings change which experts must be read; trimming changes
which unused final-stage outputs are computed. Neither changes the routing
rule. With backbone substitution, output depends on cache state and timing.
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
rows. The local KV ring has `128 + max(5, lookup_depth)` rows. Wider lookup
uses 32 compressor-state rows so rejected drafts cannot overwrite state
needed to resume. DSpark retains its own five-position architecture.

The sampler checks up to 15 proposals, every rejection point, EOS and random
state preservation. GPU checks compare all sixteen rows with serial
decoding and resume after accepting 1, 2, 6, 15 or 16 rows. These checks
cover MXFP4 and IQ2, including an MXFP4 run with DSpark, online prefetching
and GPU routing. Candidate tests alone do not measure model acceptance or
throughput. The CLI prints exact lookup attempt, offered and accepted
counts; the server logs them per request and exposes `lookup_depth` and
`expert_batch` in status. Copy-heavy and reconstructed Harness timings are
reported separately in [Performance](performance.md#context-copy-drafting).

The experimental `build/myds4-consensus-lookup` keeps the longer-match
path and adds a four-token fallback. It requires at least two earlier
occurrences and 80% agreement on the whole proposed continuation. Each
fallback supplies at most four tokens. Two failed verifications disable
this fallback until a longer match succeeds. The ordinary build uses the
long/short index without this consensus fallback.

Frequency-based selection and adaptive speculation lengths are also studied
in [SuffixDecoding](https://arxiv.org/html/2411.04975v3). This experiment
verifies one linear chain using the engine's existing causal attention.
CPU replay measures token agreement; live inference must establish whether
that agreement saves enough expert reads and computation.

The experimental `build/myds4-consensus-prefetch1` limits new prefetch
admissions during verification to forecasts from its first row. Refinement
still retains forecasts from every row, including the original offsets for
the next two layers. Ordinary concurrent requests and substitution retain
their existing policies. This experiment changes read admission only;
every selected expert needed for verification is still loaded.

The separate `build/myds4-consensus-row-credit` experiment adds cache
priority for repeated uses of an expert within a verification batch.
Reads remain deduplicated, and hit, miss and first-use counters retain
their existing definitions. Priority counts all computed verification
rows, including rows later rejected by sampling. The ordinary build does
not apply this extra priority.

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
small forward pass afterward reduces them to `max(5, lookup_depth) + 1`
rows. It waits for submitted GPU work before releasing buffers.
Changing the number of rows within that small-batch capacity requires no
reallocation. Candidate masks reserve only these request rows; their width
still follows the configured context limit.

Temporary layer buffers also shrink for decoding: queries, attention
outputs, matrix staging, routed-expert activations and index scores retain
only the required small-batch capacity. A later prefill grows them again.
Draft-history projection can temporarily require up to 128 rows, and image
encoding reserves the full matrix staging capacity. Resizing waits for GPU
completion and updates cache residency. Router tables retain separate slots
for queued layers when GPU routing is enabled. FP32 logits, persistent KV,
draft history and saved prefix state have independent lifetimes.

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

The plan conservatively retains `2668 + max(5, lookup_depth)` positions
before the final state and before the saved prefix checkpoint. It rounds
the starts down and the checkpoint end up to the original 2,048-row matrix
batch boundaries, merges overlapping regions, and skips whole batches
outside them in layers 21 through 39. Those skipped rows cannot affect the
retained local state. Global attention still has the complete encoder KV.
The attention masks and numerical kernels remain unchanged.

The CLI and server expose the plan through `--prefill-prune`, off by
default. The engine's `myds4_config.prefill_prune` field selects the same
path. `build/myds4-prefill-prune` enables it for diagnostic comparisons.
The paired benchmark builds
`build/myds4-prefix-prefill` and `build/myds4-prefix-prune` both save the
first-user prefix checkpoint.
The plan applies only with substitution and CED off and without full-row
logits or activation/route dumps. Validation compares
logits, global and local state, prefix restore and DSpark history, as well
as checking the dependency plan on small causal graphs. See
[validation records](../data/benchmarks/prefill_prune_results.json).

Changing prefill's work changes the expert cache left for decoding. The
long MXFP4 comparison reduced total request time despite slower decoding.
The option therefore does not promise a decode-throughput improvement.
The server reports the configured `prefill_prune` flag and the last
prefill's `prune_applied` flag separately; short prompts or incompatible
diagnostics can leave pruning unused.

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
session matches only the same image (see Sessions). The model takes such a
position as part of an image span, as `model.py` does: its embedding row is
the span's (the learned `image_start`, `image_end` and `image_newline` at
the delimiters, the Aligner's rows in reading order at the IMAGE slots); the
router selects with each layer's `bias_vl`; Engram leaves its stream
untouched (the gate is shut) and n-grams stop at it (the reference's DEAD
tokens). The server decodes a request's images when it renders the prompt,
and runs the vision encoder only for the images whose spans reach the part
of the prompt being prefilled; the others are in the session's rows.

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
available memory is the free memory (the kernel's free pages, which include
the file pages read ahead) and the file cache beyond the share that the
kernel keeps (`vm.vm_page_filecache_min`, 23.9 GiB with 128 GiB). While the
file cache is below that share, the kernel compresses applications' memory
rather than drop cached files: on the development machine it had
compressed 8.6 pages for every file page it dropped since boot. Beyond the
share it drops cached files first but may still compress some memory, so
the cache starts within the free memory less the reserve and takes more
only in its growth steps, which stop as soon as the kernel compresses
anything (see below).

Each available-memory calculation reads the current kernel file-cache floor.
When that query fails or returns zero, only free pages count toward available
memory. Later calculations retry the query, including after startup when the
kernel may not yet have published a nonzero floor.

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
memory and `--cache` alone: a slot whose expert a layer needs is mapped
again before the GPU uses it. Committing a mapped slot into the set took
about 0.1 ms of CPU time for a 12 MB slot and 0.15 ms for an 18.9 MB one
(see [Performance](performance.md#gpu-resident-memory)), against about 1 and
1.4 ms to read the expert from the SSD. Between tokens,
and in a prompt pass for the layers done, the mapped slots with the lowest
scores leave the set until the rest fit; slots still being read or read
ahead, and those of the layer being served or published, stay. Past the
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

Decode eviction chooses the lowest score by default. With
`--cache-read-cost`, off by default, the victim priority is the raw score
times the expert's aligned read bytes divided by the fixed backbone slot
size. Read bytes cover all of that expert's file spans rounded to the read
alignment. The cost factor is computed once from the known file layout;
raw score history and request increments remain unweighted. This policy
does not change prefill eviction, least-recently-used release under memory
pressure, GPU unmapping priorities or the draft pools' replacement rule.

Slots emptied by cancelled read-aheads go first. Experts requested by the
layer being served in the current pass, protected read-aheads, and experts
whose addresses are still published for GPU routing cannot be victims.
A recency list orders the slots for prompt passes and for giving slots back.

Every slot is in the memory the process accounts for (its footprint). The
slots mapped for the GPU are wired; the kernel could compress or swap the
others like any program's memory, so the cache gives memory back as soon
as the kernel starts compressing memory to make room. It checks memory at
each decode or verification pass, during each prompt layer and every second
while the server idles. It shrinks, least recently used slots first, when the kernel reports memory
pressure, when anything was swapped out or the swapper woke
(`vm.compressor.swapper.wakeups.total`), or when the available memory falls
below the reserve. It shrinks to the free memory less the reserve when the
kernel compresses anything while the free memory is below the reserve, or
when 64 MiB more of the process's own memory is compressed than at the
check before. Memory given back shows as free only after a while, so it
waits a second before shrinking again, unless the pressure is critical or
the system swaps. It grows again, by a quarter of the headroom, only after
32 checks in which the kernel compressed nothing and while the available
memory exceeds the reserve by 4 GiB, up to `--cache` when one
is given. For five minutes after a compression or swap that the cache could
have caused, the headroom is the free memory alone, without the file
cache.
Within a prompt pass at layer l it releases and unmaps only slots of layers
before l - 1, which no GPU work in flight uses.

The mapped slots are in one residency set attached to the command queue,
so command buffers need no per-buffer residency. A slot mapped during a
token is committed with the set's other changes after the token, or before
the next layer of a prompt pass; until then the command buffers that use
the slot declare it themselves (`useResource`), and a buffer read that way
took the GPU about as long as one in the set. A slot leaves the set only at
those commits, when no GPU work in flight can read it.

With `--draft`, the default preloads the stages' 384 experts into dedicated
slots of their own size (7.3 GB in MXFP4, taken off the total cache budget).
Those experts remain resident. A smaller `--draft-cache` reserves three
pools of N slots and reads native draft experts on demand. The pool buffers
remain resident even when their expert contents change. They use MXFP4 slot
sizes with either backbone quantization, so a draft expert never occupies
a smaller GGUF backbone buffer. Both allocation modes reserve their bytes
before allocating the adaptive backbone cache. See
[Speculative decoding](#speculative-decoding) for pool limits and replacement.

During a prompt pass the victim of an eviction is a slot of a layer already
done, never of the next layers, which the pass still needs. The cache
therefore needs room for two backbone layers' experts (768 slots). A fixed
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

The cache checks run without model weights, expert reads or GPU execution:

| Command | Checks | Scope |
| --- | ---: | --- |
| `make cache-ready-check` | 45 | Completed/pending groups, readiness, late counts and consumer release. |
| `make draft-cache-check` | 49 | Separate stage pools, native demand admission, replacement and protected groups. |
| `make prefetch-metrics-check` | 56 | Fixed snapshots, first use, read origin, byte accounting, I/O timing and unsupported coverage. |
| `make cache-policy-check` | 37 | Aging ratios, victim rules, aligned read costs, uniform costs, history and rescaling. |
| `make prefetch-queue-check` | 10 | Immediate admission release on completion or cancellation, with a delayed notification queue and reads from `/dev/zero`. |
| `make memory-available-check` | 7 | Changing kernel file-cache floors, conservative query failure and recovery. |

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
| `unsupported_gpu_route_passes`, `unsupported_gpu_route_rows` | Work that actually used the GPU lookup path and is excluded from the readiness views. |
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

GPU lookup uses a table published before routing. A later CPU completion
query does not establish whether that published table was ready, so its
passes explicitly report unsupported coverage. Verification passes still use
the supported CPU routing path even when `--gpu-route` is configured.

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

The server (`build/myds4-server`, see [The server](server.md)) handles HTTP,
request conversion, prompts and output in Rust with DeepSeek's
`deepseek-recipe`; the scheduler below runs in `src/engine/engine.c`, behind
a C ABI.

A chat completion waits in a queue (at most 16) for the model, which the
engine's thread runs. A streamed request gets its headers at once and a
comment every 15 s while it waits. Between tokens the worker admits waiting
requests while fewer than `--batch` decode: it tokenizes one's rendered
prompt, prefills it (the others wait) and samples its first token. Then
every request decoding advances by a token: alone, with drafts when
`--draft` is set; else in one step of `model_forward_rows`, one row per
request. The engine hands each step's tokens to the server, which writes
them to the request's client; when a client closes its end, the server
cancels its request.

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
logits differed, with the GGUF experts and two drafts). A sequence decoded
beside others does not update the draft's windows, so drafting it alone
later keeps fewer drafts until they refill.

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

### Sessions

Each conversation keeps its attention state in a session: the layers'
window rings, compressed rows, index keys and compressor rows, the draft's
windows and the Engram history. The compressed rows and index keys take
3,200 bytes per position (3.125 KiB) in BF16; a state starts with room for 16,384 positions
(66 MiB with its saved rows) and doubles its room when a forward pass
needs more, up to `--ctx` (3.14 GiB at 1,048,576), keeping the rows it
holds; a reset gives the room back. The layers use one session's state at a
time. A ring holds its layer's last `128 + max(5, lookup_depth)` positions.
The compressor keeps 8 rows normally and 32 for wider lookup. Server prefix
reuse can continue at any of the last 5 positions (BLOCK).
After each prompt's prefill the session saves those rows (slot 0, about 6
MB). A new conversation also saves the state before its first user turn
(slot 1), during the same prefill. Each layer copies the prefix's last ring
rows from its completed attention batch and preserves its compressor rows.
DSpark captures the prefix's final input rows separately, so saving the
prefix does not overwrite the current request's draft state. The compressed
rows and index keys of earlier
positions are never rewritten, so a saved slot and the state's later rows
together give the state at the saved position.

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

A request runs only the tokens a session does not hold:

- kept: a session whose tokens the prompt starts with, up to its last 5
  positions, continues (a tool loop's next request, a conversation's next
  turn);
- saved: a session whose saved prompt end the prompt shares restores slot
  0 and continues from there or up to 5 positions before it (a retry, or
  the next turn of a chat without tools, whose earlier reasoning is
  dropped);
- forked: a prompt that shares a session's first system part (another
  conversation with the same instructions and tools, such as an agent's
  subagent) gets a copy of that session from its slot 1, which leaves the
  session unchanged;
- new: else a free session, or the least recently used one emptied.

A fork sizes its destination from the saved prefix length, with the usual
initial capacity and growth rule. It copies the prefix's compressed global
rows and index keys, including each layout's guard row. The window rings,
compressor state, draft rings and Engram history come directly from slot 1.
The fork retains its own copy of that prefix snapshot. Its next completed
prompt creates a new slot-0 checkpoint. CPU host-buffer checks cover odd
and even prefix lengths, both compression ratios, draft state, capacity
reduction and source independence. Real-model resume validation has passed
the 257-token MXFP4 model check at a 128-token
prefix, with unchanged resumed logits and source state.

With CED, slot 1 holds encoder state through layer 20. Such a fork needs
more than the configured replay depth plus `max(5, lookup_depth)` new tokens, so replay
excludes missing decoder-prefix keys and rebuilds all rewind rows. Other
requests reuse a complete state or prefill from the beginning. A fork must
leave at least one new token to compute logits. Prefill retains the local
rows required by its configured verification margin and the five rows
needed before a saved server position.

A session in which a request decodes is never emptied, rewound or freed;
another request may fork from it. The prompt's last token always runs, for its logits, except when the prompt
is the session's saved prompt again: then the saved rows and the logits kept
with them (0.5 MB) continue the request, so that a retry decodes exactly as
the first try did; the prompt's last token in a decoding step would round
differently from the prefill's. Reuse checks every token, including the last,
and restores the saved window even if a short reply leaves the current
window within rewind range. `--kv-cache` bounds the sessions' memory,
which is apart from the expert cache's budget: before a
request runs, the least recently used other sessions are freed until the
chosen one fits with room for its prompt, and a request whose session alone
does not fit is refused; a session that grew while decoding pushes others
out after its request. Under memory pressure the least recently used
session other than the current one is freed too. The usage of a response reports the tokens a session supplied as
`cached_tokens`.

In a tool loop with the GGUF experts, the second request ran 26 of its 399
prompt tokens: the re-rendered tool call matched the model's own tokens,
so the state of the whole first turn was kept.

### The live view

`--tui` gives stderr's alternate screen to a thread of the server
(`server/src/view.rs`, on ratatui and its crossterm backend) that every
0.5 s takes the engine's snapshot (`myds4_view`, as JSON) and draws the frame
from it. The engine writes the log's lines to the server's `on_log`, which
keeps the last 256 in a ring that the frame shows; the lines logged before
the view is on the screen (while the model loads) also go to stderr. The
snapshot takes the queue's lock for the number of waiting requests and the
status lock for the requests, the totals and the sessions. Sessions are
freed and reassigned under the status lock, so the snapshot never reads a
freed session; a session's tokens may change while the snapshot reads them,
which at worst places a node wrongly for one frame.

The prefix tree covers the 24 most recently used sessions. Depth first, a
node is the run of positions, from its parent's end, that all sessions of
its group share (compared 1,024 tokens at a time with `memcmp`), and the
sessions that go on past it form groups by their token at its end. The
frame shows at most 40 nodes. Each session keeps the positions up to which
its last 32 reuses ran (requests that continued it, rewound to its saved
rows, or forked from it); a node counts those that reach past its start, so
a system prompt that many conversations share collects the hits of all of
them.

The counters cost a few hundred additions per decoded token. A decoding
step adds the cache's statistics during the step to `decode_stats`, with its
rows, GPU time and wall time; a prefill adds them to `prefill_stats`. The
cache counts each expert once per layer and step for all rows, so
`routed_rows` also counts every row's selections, each by where its expert
is when the layer requests it (`cache_where`: in memory, read ahead, read
ahead and still being read, only on the SSD) or as a substitute. The frame
keeps 21 snapshots of the decoding statistics and shows the differences
over the last 10 s. Memory comes from `host_statistics64` (free,
file-backed, wired and compressor pages), `vm.vm_page_filecache_min`,
`vm.swapusage`, the accelerator's "In use system memory" (the GPU-resident
memory of all processes), and the cache's own counts. While the GPU idles,
that resident memory falls below myds4's mapped experts and weights: about
a minute after the last request it was 2.1 to 2.5 GiB against their 45.6
GiB, and wired memory fell from 56 to 8 GiB. The driver releases them until
the GPU runs again, so the view splits the resident memory into experts,
weights and the rest only while it holds them all.

A frame is a list of lines, at most one per row, that ratatui draws in
place, so no line break scrolls the title off the screen; a line too long
for the terminal ends after its last whole figure, with "…". `make
server-check` draws frames from a fixed snapshot on ratatui's test terminal
and checks that the title comes first, that every line fits the width and
that the log fills the rows below the tree.

While the view runs, the terminal is in raw mode, which neither echoes nor
buffers input and delivers Ctrl-C as a key, and the cursor is hidden. The
view's thread waits for its next frame by polling the keys: `q`, or Ctrl-C,
requests the stop that SIGINT requests (`myds4_request_stop`) and logs "q:
stopping after the current step"; the server stops accepting within a
second. SIGINT and SIGTERM request that stop too. When the server ends,
before a fatal error's message (the engine's `on_die`, or the server's own)
and after a panic, the view leaves the alternate screen, restores the
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
Numerics). As BF16, a session's state takes 66 MiB at first instead of 132,
and 3.14 GiB at 1,048,576 positions instead of 6.28. The 8,912-token
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

At `--cache 64` the cache settled where the free memory met the 10 GiB
reserve, at 42 to 53 GiB, while macOS showed another 16 to 30 GiB as cached
files. The build of 2026-10-31 10:03 counted only free memory: the kernel's
free pages plus its read-ahead pages, which the free pages already include,
so those counted twice (1.1 to 1.2 GiB that day). It also started at the
full `--cache` size and shrank only after its first checks.

Counting the whole file cache was not tried. The kernel keeps a share of
the file cache (`vm.vm_page_filecache_min`, 23.9 GiB with 128 GiB) and,
while the file cache is below it, compresses applications' memory instead
of dropping cached files: since boot it had compressed 1.91 billion pages
and dropped 221 million file pages. The cache's unmapped slots are the
process's oldest untouched pages, which the kernel would compress first,
and quantized weights do not compress, so the compressor would fill and
then swap. The cache therefore counts only the file cache beyond the share,
starts within the free memory less the reserve, and stops growing, or gives
memory back, when the kernel compresses anything (see The expert cache and
memory).

Measured on 2026-10-31 with `--cache 64`, the story prompt, exact
decoding, the GGUF experts unless noted, each run a fresh process,
alternated, one run per row and rule (other applications used 0 to 22% of
the GPU). No run swapped anything out:

| State before the run | Cache, 10:03 rule | Cache, this rule | Tokens/s, 10:03 rule | Tokens/s, this rule |
| --- | ---: | ---: | ---: | ---: |
| 15.9 GiB of file cache, 300 tokens | 43.8 and 43.4 GiB | 44.4 and 42.7 GiB | 23.70 and 23.54 | 23.79 and 23.38 |
| The same, MXFP4 experts | 44.9 and 44.4 GiB | 43.7 and 42.3 GiB | 13.16 and 12.64 | 13.04 and 12.59 |
| 39.0 GiB of file cache, 600 tokens | 24.8 GiB | 34.1 GiB | 18.88 | 21.23 |
| Right after reading 30 GiB of other files, 600 tokens | 23.9 GiB | 20.2 GiB | 18.31 | 16.66 |

Below the share both rules held about the same cache, this one keeping
about 1 GiB more free since it counts the read-ahead pages once. With 39.0
GiB of file cache this rule took the free memory below the reserve (a
footprint of 43.7 GiB against 45.3 GiB free before the run), the file cache
beyond the share standing in for it, and the kernel dropped and compressed
nothing. Right after the 30 GiB had been read, the kernel compressed 40 MiB
during this rule's run and the cache stopped at 20.2 GiB. During the 10:03
rule's run, which came second (41.9 GiB of file cache against 46.1), the
kernel dropped 3.0 GiB of file pages and compressed 1.2 GiB, likely because
that rule started at the full 64 GiB and its first prompt took more than
the free memory.

The cost: with file cache beyond the share, the free memory can fall to 1
to 2 GiB, and a program that then takes memory gets it from the kernel's
reclaim, which drops cached files and may compress some memory, before the
cache gives slots back. The rule backs off at the first compression and
stays within the free memory for five minutes, which kept the cache at 20.2
GiB in the run right after the reads. Each state was measured once per
rule.

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
