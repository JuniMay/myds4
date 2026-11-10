# Design

## Responsibilities

| Area | Responsibility |
| --- | --- |
| `src/engine/` | C interface, Metal commands, file access, tokenization, sampling and media decoding. |
| `src/engine/expert_store.h`, `io.c` | The routed-expert store: slots read from the SSD, read-ahead and eviction by score, within the memory other programs leave. Each model with routed experts creates its own store from its layout (`cache_create`: its layers, their experts and each expert's file spans, up to eight), and every call takes it. DeepSeek and Nemotron use it. The stores share the reserve and the time of the last give-back, so that one shortfall goes back once. |
| `src/model/model.*` | Model registry and common execution interface. |
| `src/model/weights.*`, `experts.*`, `linear.*` | Reloadable weights, routed-expert batches and matrix products. |
| `src/model/deepseek/` | DeepSeek layers, experts, Engram, prefill planning and concurrent schedules. |
| `src/model/qwen/` | Qwen text layers, recurrent state, MTP and vision encoder. |
| `src/model/nemotron/` | Transcribe's language backbone, speech encoder and DFlash2 drafter. |
| `src/model/draft.*` | Context-copy proposal generation. |
| `src/kernels/` | Metal numerical kernels. |
| `server/src/chat/` | API request conversion, model prompt formatting and response decoding. |
| `server/src/ui/` | Terminal rendering and interaction. |
| `server/src/worker.rs`, `kv.rs` | Scheduling, model state ownership and the shared prefix-cache budget. |
| `server/src/control.rs` | Yield ownership, retry handling and temporary process-wide limits. |
| `tests/`, `server/tests/` | Maintained correctness checks and their inputs. |
| `benchmarks/` | Measurement drivers and evaluation datasets. |

The Rust server owns requests. One engine thread owns model execution.
Models are peers in the registry, with individual queues and settings.
The C configuration is tagged by architecture and holds the appropriate
option union. The Rust configuration uses the corresponding backend enum.

Registration reads metadata and tokenizers without loading weights.
An inference request triggers loading only for its selected model. The
process's memory follows the memory that other programs leave, less the
reserve (src/engine/memory.c, `memory_follow`): every addition of model
memory or KV is checked against the available memory, and idle model
memory goes back first when it would fall below the reserve; a load that
still does not fit is refused. The `memory` setting is an upper limit
within that. macOS counts a new buffer's pages as in use only once they are
written, while the engine charges the buffer when it allocates it. A reading
of the counters therefore counts the expert-cache bytes that no completed
read has written as in use (`memory_unwritten`; `cache_unwritten` in
src/engine/io.c): slots whose reads are under way, and the part of a slot
beyond a smaller expert. The expert cache's speculative growth stops while
one more slot would still fit above the reserve. That margin holds a pass's
small workspace buffers, which the workspace slot charges and releases on
every pass, so they do not take back an expert that the next miss would read
again. Memory messages go through the engine's log
(`say` in src/engine/util.c, which the server's live view receives) and
name only take-backs of 1 GiB or more, and refusals. Each loaded model owns
its tensor index; deferred media encoding uses that model's index.

Weights are read in 16 KiB-aligned spans into a page-aligned buffer and
copied out (`read_bytes`, src/model/weights.c): checkpoint files are opened
with `F_NOCACHE`, but unaligned uncached reads still fill the page cache,
which the kernel then keeps. Quitting stops a model's load (`model_load`)
between weights and between the 16 MiB pieces it reads: the load frees
what it took and fails, and the server ends the requests waiting for it as
a shutdown. Weights that a pass loads
complete with the pass. A pass that fails because its experts could not
load ends its requests: as a shutdown while quitting, otherwise with an
error, discarding their states. DeepSeek prefills a prompt in one engine
call, so its prefill (`prefill` in src/model/deepseek/model.c) checks for
quitting before each batch of rows. A stopped prefill waits for its GPU
work and its Engram row reads, returns no logits and no snapshot, and
leaves the state's new rows incomplete; the server then ends the request
as a shutdown and exits.

File validation errors during loading return a service error while the
server remains available. Fatal backend or device errors still terminate
the process. Model metadata remains registered. Weight groups can be unloaded under
memory pressure and restored from the same checkpoint. Dense weights are
one cache entry per model; expert entries depend on the MoE backbone, so
experts leave before routers, attention and shared weights.

Routed experts share the process memory budget and dependency rules. Flash-Next
and Nemotron also share the expert batch planner: it groups selected rows by
expert, holds every selection before loading misses, records cache use, and
retains the original expert order for each row's mixture. Their NVFP4 backend
executes small groups together and batches larger prompt groups into 16-row
matrix tiles. Quantized activations fit exactly in BF16; reductions retain
FP32. Workspaces remain charged to the model-memory pool.

Nemotron's and Flash-Next's routed experts live in expert stores, one per
model, as DeepSeek's do. Each model builds its store's layout from its
checkpoint. An expert's tensors that follow each other in a file share a
span. Transcribe's experts take two spans: both weight matrices, then both
scale matrices. In 24,563 of Flash-Next's 24,576 experts, the three weight
matrices and the fixed rows of their compressed scales form one 2.6 MB span,
and the scales' three exception lists a second. The layout records where
each expert's matrices lie in its slot. While the planner holds an expert,
the expert's matrices are views of its slot.

Flash-Next's checkpoint stores NVFP4 scales compressed (CSF): one code row
per 16-row slab, plus a list of exceptions. The store reads them as they
are. The reader that completes an expert's last read decodes its scales
(`weights_csf_decode`, 307 KB) into a region at the end of the slot, and
only then does the expert count as read. A slot is as large as the largest
expert's read bytes plus the decoded scales: about 3.1 MB, against the
2.76 MB that the expert's weights and decoded scales take alone. The
compressed scales and the 16 KiB-aligned reads stay in the slot, and the
largest exception lists (up to 112 KB, against 3.9 KB on average) set the
size of every slot.

While decoding, Flash-Next reads each layer's experts one layer ahead. A
layer that reads its experts (one that does not run through its table)
synchronizes with the GPU to read its routing. Before that point, the GPU
also applies the next layer's router to this layer's input. After the
layer holds its own experts, the store's background readers start reading
the next layer's predicted experts that are not in memory. The readers wait
while a layer's own misses are being read. The next layer's routing then
demotes the read-aheads it did not choose, so that they are evicted first.
A wrong prediction costs a read, never a result. A pass whose rows choose
at least as many experts as a layer has (a prompt of 52 rows or more)
needs nearly every expert and reads nothing ahead. Once a layer's expert
work is encoded, Flash-Next commits it, so that the GPU runs it while the
CPU encodes the next layer's work.

Every family with routed experts records, for each row of its latest
decoding pass, the experts chosen in each layer with routed experts and
where each one was then: in memory, read ahead, late (read ahead and still
being read), substituted (DeepSeek only) or from the SSD (`myds4_routes`).
Flash-Next and Transcribe take where each one was from the store just
before the layer acquires its experts; a layer reached through its table
had all of them in memory. The engine's statistics give the layers with
routed experts and each one's experts (`routed_layers`,
`experts_per_layer`), from which the live view draws its Experts tab.

When every expert of a layer is resident, GPU routing feeds the expert batch
planner directly. With a store, resident means in a slot, read, mapped for
the GPU and committed to the residency set. The engine rebuilds the layer's
weight-address table after any of its experts moves. Each layer's selections
stay in the pass's selection buffer until the pass completes. The engine
then collects the cache-use counters and marks the experts that each such
layer's rows chose as used, as it does for loaded selections, so that
eviction keeps them. A prompt's are marked before its pass ends, so that
they count toward the eviction priority of prompts, not of decoding (the
store's `decode_uses` counts decoding's). Layers with missing experts
return their selections to the engine for loading.

The expert batch stays in the current GPU command buffer with the following
layer work. Selected experts remain held until the next CPU routing read or
the end of the model pass. That synchronization completes the queued expert
work before the engine releases its selection and loads another. Resident
layers can run between these points without changing the held selection.
Kernels and per-row summation order remain unchanged. Nemotron's and
Flash-Next's stores advance their clocks after every pass, prompt chunks
included: the pass's GPU work has completed, so any slot that is not held
may go back, and slots mapped during the pass join the residency set for
the next one.

The shared weight loader gives each weight of a group its own Metal
allocation. A group is one memory entry: loading charges all of its
allocations together, and eviction releases them together.

The Metal backend distributes mapped allocations across residency sets.
Each set targets 5% of the device's recommended working-set size, with a
64 MiB minimum. The queue holds at most 32 sets; oversized allocations and
the set-count limit can exceed that per-set target. Views share the original
allocation's membership. Only changed sets are committed. This bounds the
amount of memory in each set while preserving buffer addresses and execution
order.

Command buffers run in the order they are opened, and a decode step opens a
layer's expert buffers before it commits the next layer's front behind
them. Until those opened buffers are committed, no thread may wait for a
command buffer committed after them: the GPU would run it only after buffers
that the waiting thread commits later. So an eviction in the middle of a
pass (an allocation that takes back an idle expert) commits the residency
change but does not wait for the GPU; the pass's own `gpu_sync` drains the
released buffers (`collect` and `reserved` in src/engine/gpu.c).

DeepSeek uses Metal's working-set recommendation to decide how many expert
buffers to map. Allocation follows the available memory and the reserve,
within the `memory` and KV settings. The engine does not change system
wired-memory settings.

Resident memory stays wired only while the GPU works. About half a second
after the last command buffer, the GPU driver starts taking the process's
resident memory out of wired memory, and once it is unwired macOS can swap
it. Measured on 2026-11-09 with DeepSeek idle: wired memory fell from 50 to
6 GiB within 3.5 s, stayed there while the server idled, and returned within
2 s of the next request. So an idle server's model memory can be swapped
under memory pressure. The engine never causes that pressure itself:
- every addition keeps the reserve free;
- a load first takes back what it needs;
- weights are read without filling the file cache.

The once-a-second maintenance gives memory back when other programs grow.
It also unloads a model that has gone its `idle_unload` time without
requests, with none waiting: after the GPU queues and expert reads drain,
`myds4_unload` releases every idle entry of that model's memory account,
each before the entries it depends on, so the expert cache goes before the
weights. The account's workspace and the model's KV states stay; the next
request's load reads the weights again. While a yield holds the memory,
nothing unloads for idleness.

The expert batch planner holds a layer's selected experts in the store
before any of them is read, then waits for the missing ones. The model waits
for these reads, so they go to the system's dispatch queues at once, ahead of
the store's four background readers. The selected experts stay held until
the next selection or the end of the pass.

Routing and numerical boundaries belong to the model. Flash-Next uses SwiGLU
and rounds each expert's contribution and each addition to BF16. Nemotron uses
squared ReLU, accumulates contributions in FP32, then rounds the mixture.
The shared backend selects separate compiled kernels for these rules.
The common matrix kernels can round their FP32 output to BF16 before storing
it. Callers select this at an existing rounding boundary. Small matrix-vector
products can compute chunks concurrently, then add them in lane-strided
order before the SIMD reduction. Both paths use FP32 output storage.

Flash-Next's router finds a top-k list within each SIMD group and merges those
lists. Equal scores retain the lower expert index. Softmax normalization uses
the full router vector. Selected weights are normalized in rank order and
rounded to BF16.
DeepSeek uses its MXFP4 and GGUF expert kernels, which already combine expert
work and overlap ready experts with pending reads.

## Scheduling and state

The control API queues yield and resume transitions for the engine thread.
Yield runs between worker turns. It drains both Metal queues and expert
reads, unpins weights, then reclaims model and KV allocations separately.
The worker pauses inference and holds the resulting limits until the owner
requests resume. Active request state and samplers stay alive. Resume
restores configured limits and loads any missing backbone needed by an
active request. Idle models load on demand. Control requests and status
queries continue on the HTTP threads throughout the transition.

The worker gives each model a turn. It admits requests up to that model's
batch limit, advances prefill, then advances decoding. Qwen prefill uses
bounded chunks. DeepSeek plans the remaining prompt in one call and adapts its internal
chunk size to the model-memory budget. Such a call can still delay another
model's turn. Qwen also adapts its prefill rows and workspace to the budget.

A request owns a state while it runs. The shared KV pool keeps completed
states and snapshots for prefix reuse. It compares token histories at any
shared prefix within the same model. Image positions use keys derived from
the image bytes, so different images do not reuse each other's state.

A state can continue in place, copy stable rows and restore a snapshot,
or use the model's supported replay operation. DeepSeek can reconstruct
rolling state around a branch. Qwen's recurrent DeltaNet state requires a
snapshot; copied attention keys alone cannot reconstruct that state.

The pool charges allocated capacity and pending growth before admission.
It also charges snapshots promised to an admitted prefill. Eviction first
removes spare snapshots, then idle states. Active states and a branch's
source remain protected. The configured limit may be exceeded by active
state growth when nothing idle remains to evict.

Each stored sequence owns its native state, snapshots and model handle.
Eviction drops those resources together. Rust ownership keeps native model handles on the engine thread. HTTP
handlers and the terminal receive copied statistics, without sharing the
native model handle or checking thread identity on each call.

Each prefill call receives its own image context. Image preprocessing and
feature storage belong to the request; model-specific position state is
kept in the sequence and its snapshots.

## DeepSeek V4.1 Flash

The supported checkpoint has 40 layers, width 5,120, four residual streams,
384 routed experts per layer and six selected experts per token. The
native experts are MXFP4; routers, normalization and other tensors retain
the checkpoint's stored representation. Optional GGUF experts change their
quantization and may also provide a different output head.

The engine stores residual streams as BF16, with FP32 accumulation where
required by its kernels. Attention includes local windows and compressed
history with an indexer. The local rolling buffers and compressor state
are part of a sequence's state and snapshots.

Engram uses normalized token IDs for n-gram hashing. The normalization map
is generated from the configured tokenizer at loading. The supported hash
layout is checked against the reference; it is not a tracked binary build
input. Image positions break n-grams rather than hashing image keys as text.

Expert weights are read into an adaptive RAM cache. Allocation and GPU
residency are separate: an allocated expert can remain in RAM while absent
from the GPU residency set. Routing fixes which experts exact inference
uses. Reads are split into bounded pieces, with demand work ahead of
background forecasts. Cache policy accounts for reuse and optional read
cost; speculative reads must not invalidate experts held by active work.

Ordinary execution overlaps ready expert work with remaining reads.
Runahead groups ready requests at the same layer. The experimental
pipeline keeps independent layer stages in flight and groups ready expert
stages across layers. Both preserve each request's ordered reduction.
There is still a token boundary before the next sampling step.

A prompt runs layer by layer over up to 16,384 rows, in batches of up to
8,192 rows. Each batch groups its routed (row, expert) pairs by expert into
tiles of up to 64 rows (32 for MXFP4), and each tile decodes its expert's
weights once. Larger batches therefore fill the tiles better: on a
5,772-token prompt the routed experts took about a quarter less GPU time
with 8,192-row batches than with 2,048. A batch needs about 0.9 MB of
workspace per row, about 7 GiB at 8,192 rows; prefill takes fewer rows when
the model-memory budget is short. Batch boundaries change no result. When a
layer's successor takes more than 4,096 rows, the layer reads all of the
next layer's experts ahead, since that many rows choose nearly every expert.

Dependency-pruned prefill skips decoder rows outside the history needed by
the final output and branch snapshots, in groups of 2,048 rows, and a batch
ends where a kept group ends. It retains required encoder work. CED instead
deliberately shortens decoder replay and is approximate. Either can change
cache warmth and subsequent decoding speed even when prefill is faster.

## Qwen3.8-27B

The supported NVFP4 checkpoint has 64 text layers: 48 gated DeltaNet layers
and 16 full-attention layers. Its text width is 5,120. MLP and output-head
weights use NVFP4; attention projections use FP8; other weights retain their
stored types. The engine currently keeps text activations in FP32. Prefill converts an
input to BF16 once for projections that share it and uses up to 1,024 rows
per chunk, reducing the chunk when the shared memory budget requires it.

DeltaNet keeps a recurrent matrix and convolution history. Full attention
keeps FP16 keys and values. Capacity starts at 1,024 positions and doubles
as required. A snapshot preserves recurrent state and the position data
needed to continue. The model's supported context is 262,144 tokens.

Decoding handles up to eight request rows. Prefill uses the Metal matrix
units and different operand rounding from one-row decoding. Therefore
reusing decoded state need not equal a cold prefill of the same text.

Three-row products use kernels with three accumulators per weight row.
The NVFP4 gate/up kernel uses the same fixed row count. This covers target
verification with two MTP proposals and avoids reserving accumulators for
eight rows. The arithmetic and reduction order remain unchanged.

With drafting enabled, the MTP head proposes up to five additional tokens.
The first proposal uses the target hidden after final normalization. Each
subsequent proposal uses the preceding MTP head output after its own final
normalization.
Target verification processes the proposals together with the current token.
It saves the recurrent state after each possible accepted prefix. Rejection
selects the matching checkpoint without replaying the target layers. The
draft head then replaces speculative keys with keys from accepted target
hiddens. Greedy verification keeps the target's token sequence. For sampled
requests, the head samples proposals using the request's temperature and
probability filters. Probability-ratio acceptance and residual correction
preserve the target distribution; proposal and acceptance draws change the
random stream relative to ordinary sampling.
With `draft: 0`, draft weights, attention rows and prefill work are omitted.
Multiple active requests use ordinary batched decoding instead. Larger draft
depths allocate more recurrent checkpoints in the separate KV budget.

The vision encoder has 27 blocks, width 1,152 and 16 attention heads of 72
values. It uses 16-pixel patches, a temporal patch size of two and a 2-by-2
spatial merger into the text width. A still image duplicates the temporal
input. Preprocessing resizes on a 32-pixel grid using the checkpoint's
limits and normalizes RGB channels to [-1, 1].

Learned image positions are interpolated on the patch grid. Vision RoPE
uses separate height and width coordinates. The text model uses interleaved
three-axis positions for image tokens and retains the resulting position
shift in sequence snapshots and draft rollback state.

Vision outputs use BF16 boundaries. Cross-device floating-point reductions
can produce different features. Functional image checks cover distinct
images, conversation reuse, MTP and simultaneous chunked requests. The
reference diagnostic retains a stricter 30 dB SNR and 0.999 cosine threshold;
not every image currently meets it. Image support should not be described
as bitwise equivalent to the CPU reference. Video input is not implemented.

## Qwen3.8 Flash-Next

Flash-Next uses 48 text layers, width 2,560 and four residual streams.
Gated hyper-connections mix those streams before attention and feed-forward
work. There are 36 DeltaNet layers and 12 sparse-attention layers. Each
layer selects 10 of 512 routed experts and also evaluates a shared expert.
The second layer adds the checkpoint's hashed n-gram embeddings and their
convolution state.

The engine reads NVFP4 expert weights and decodes their compressed scale
bytes losslessly. Dense MXFP8 projections use groups of 32 values. Routed
NVFP4 products quantize activations with the checkpoint's input scales.
Text buffers carry BF16-rounded values between operators; recurrent state
and reductions use FP32. The DeltaNet output gate is sigmoid.

Sparse attention scores complete groups of four historical keys. It
selects at most 512 groups and includes the incomplete causal tail. The
state retains BF16 attention keys and values, compressed index keys,
positions, recurrent matrices, convolution histories and n-gram history.
Snapshots retain the rolling state needed to fork a conversation.

Experts are independent entries in the shared model-memory pool. Selected
experts stay pinned through their GPU work; their backbone is retained
until its dependent entries have been released. Selected misses load before
each layer's routed work. These loads do not yet overlap expert computation. Prefill uses adaptive
chunks of up to 1,024 rows. Decoding batches up to eight request rows.
Flash-Next MTP and context-copy verification are not implemented.

The shared Qwen vision encoder reconstructs its quantized matrices in BF16
when an image first loads the encoder. Those weights and its workspace
remain inside the model-memory budget. MXFP8 activation quantization can
amplify small differences in BF16 rounding across blocks. Reference checks
therefore compare each block on identical inputs as well as final image
features. Image outputs are not bitwise equal to the CPU reference.

## Nemotron 3.5 Transcribe

The language backbone has 52 blocks: 23 Mamba2 blocks, 23 MoE blocks and
six full-attention blocks. Its hidden width is 2,688. Each MoE block selects
six of 128 experts and also runs a shared expert. Experts use two matrix
products with a squared ReLU between them. The expert store holds the 23 MoE
blocks' 2,944 routed experts, 5.35 MiB each, read in four pieces from two
spans. The attention blocks have 32
query heads and two key/value heads, each with 128 values. They do not
apply rotary positions; the Mamba blocks carry position-dependent history.

Mamba2 keeps FP32 recurrent state and convolution history. Attention keys
and values use BF16. The model keeps BF16 boundaries between operators
and FP32 routing scores. FP8 projections and NVFP4 products apply the
checkpoint's calibrated activation scales outside the matrix reduction.
This avoids rounding a global scale into each operand before the product.

The speech frontend decodes audio in memory to mono 16 kHz. It computes
128 log-mel features from 400-sample windows with a 160-sample hop and a
512-point Fourier transform. Its window and mel filters come from the
checkpoint. Eight frames form one encoder input. The ASR branch has 32
transformer blocks of width 1,280; the speaker branch has 31 blocks of
width 512. Their outputs are combined using the checkpoint's speaker
feature normalization and projected to the language model's width.

Long recordings use 500-token encoder windows with 50 tokens of left and
right context. Speaker history holds 188 selected feature rows and a
40-row queue. Confidence scores, recency boosts and silence embeddings
select the retained rows. The frontend normalizes ASR features over the
recording before windowing. Encoded audio spans participate in prefix
reuse through their input hashes, as images do.

DFlash2 is a separately loaded six-layer BF16 draft model. It receives
the target outputs of layers 1, 5, 19, 29, 41 and 51, counted from zero.
It projects those features into per-layer context keys and values. A
query contains the known next token and up to 15 masked positions.
Grouped convolutions and bidirectional attention predict candidates; a
small trained selector scores candidate transitions and chooses a path.
The drafter shares the target's embeddings and output head.

Target verification preserves a recurrent checkpoint after each possible
accepted prefix. Rejected rows leave no active target or draft history.
Verification uses the ordinary matrix-vector reduction order, sharing
weight reads across groups of up to eight rows. Deterministic proposals
consume the target sampler's draws in order. Drafting disabled omits its
weights, context keys, hidden capture and extra recurrent checkpoints.

The reference implementations are [NeMo Speech](https://github.com/NVIDIA-NeMo/Speech/tree/main/nemo/collections/speechlm2),
[Nemotron-H in Transformers](https://github.com/huggingface/transformers/tree/main/src/transformers/models/nemotron_h),
and [DFlash2 in vLLM](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen3_dflash2.py).

## Prompt formats and sampling

Chat adapters translate standard API messages and tools to each model's
prompt format and decode generated framing back into API fields. Client
harnesses receive ordinary text, reasoning and tool-call objects. Literal
special-token spellings inside message text remain text.

Qwen and Nemotron share ChatML framing and response parsing while retaining
their checkpoint-specific prompt templates. Nemotron's audio placement,
tool definitions and history reasoning rules differ from Qwen's.

Qwen's tool format and low/xhigh instructions are specified by the
checkpoint template. They are model instructions, not a client-specific
wire protocol. The adapters are checked against reference prompt fixtures.

A common sampler handles temperature, top-k, top-p and seeded randomness.
A request overrides configured defaults; absent and null values preserve
them. Deterministic proposals, including context lookup and DSpark, keep a
matching prefix of target samples. Qwen MTP can supply sampled proposals
and their probabilities. These use probability-ratio acceptance; a rejected
proposal samples from the positive part of the target distribution minus
the proposal distribution. Both methods preserve the target distribution.
Context lookup generates proposals from prior tokens and has no authority
to accept them.

## Numerical contract

Exact routing means using the selected experts without substitution. It
does not mean bit-identical output across frameworks or hardware. Storage,
accumulation order, prefill kernels and decoding kernels can round
differently. CED and expert substitution are additional explicit
approximations.

Correctness checks distinguish byte comparisons against the engine's
baseline, numerical comparisons against an upstream implementation, and
semantic model tests. SNR or matching a few outputs does not establish
benchmark quality. [Testing](testing.md) describes the validation layers.

Reference implementations: [DeepSeek](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017),
[Qwen in Transformers](https://github.com/huggingface/transformers/tree/main/src/transformers/models/qwen3_5).
