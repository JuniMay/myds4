# Design

## Responsibilities

| Area | Responsibility |
| --- | --- |
| `src/engine/` | C interface, Metal command handling, file access, tokenization, sampling and image decoding. |
| `src/model/model.*` | Model registry and common execution interface. |
| `src/model/deepseek/` | DeepSeek layers, experts, Engram, prefill planning and concurrent schedules. |
| `src/model/qwen/` | Qwen text layers, recurrent state, MTP and vision encoder. |
| `src/model/draft.*` | Context-copy proposal generation. |
| `src/kernels/` | Metal numerical kernels. |
| `server/src/chat/` | API request conversion, model prompt formatting and response decoding. |
| `server/src/ui/` | Terminal rendering and interaction. |
| `server/src/worker.rs`, `kv.rs` | Scheduling, model state ownership and the shared prefix-cache budget. |
| `tests/`, `server/tests/` | Maintained correctness checks and their inputs. |
| `benchmarks/` | Measurement drivers and evaluation datasets. |

The Rust server owns requests. One engine thread owns model execution.
Models are peers in the registry, with individual queues and settings.
The C configuration is tagged by architecture and holds the appropriate
option union. The Rust configuration uses the corresponding backend enum.

Registration reads metadata and tokenizers without loading weights.
An inference request triggers loading only for its selected model. The
engine's process lock covers both model families. Each loaded model owns
its tensor index; deferred vision loading uses that model's index.

File validation errors during loading return a service error while the
server remains available. Fatal backend or device errors still terminate
the process. Model metadata remains registered. Weight groups can be unloaded under
memory pressure and restored from the same checkpoint. Dense weights are
one cache entry per model; expert entries depend on the MoE backbone, so
experts leave before routers, attention and shared weights.

## Scheduling and state

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

Dependency-pruned prefill skips whole decoder batches outside the history
needed by the final output and branch snapshots. It retains required
encoder work and original batch boundaries. CED instead deliberately
shortens decoder replay and is approximate. Either can change cache warmth
and subsequent decoding speed even when prefill is faster.

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

With drafting enabled, the MTP head proposes up to five additional tokens.
Each draft head output supplies the hidden input for the next proposal.
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

## Prompt formats and sampling

Chat adapters translate standard API messages and tools to each model's
prompt format and decode generated framing back into API fields. Client
harnesses receive ordinary text, reasoning and tool-call objects. Literal
special-token spellings inside message text remain text.

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
