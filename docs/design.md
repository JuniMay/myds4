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
waits for a GPU signal after its routing to read it; the layer's shared
expert, which does not depend on the routing, follows the signal, so the
GPU runs it meanwhile. The signal also follows the previous layer's expert
work, whose experts' holds may then go. Before the signal, the GPU
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

Command buffers run in the order they are opened. No thread may wait for a
command buffer committed after one that is still open: the GPU would run it
only after the open one, which the waiting thread commits later. A decode
step therefore encodes the next layer's front into a detached command
buffer (`gpu_open_detached`), which joins the queue only when it is
committed, after the layer's expert buffer. An eviction in the middle of a
pass (an allocation that takes back an idle expert) commits the residency
change but does not wait for the GPU. Committed command buffers stay
retained until they are accounted for (`gpu_reap`, `gpu_sync`), so the
eviction also drains the released buffers, unless a command buffer is still
being encoded; the pass's own `gpu_sync` drains them then. `gpu_free_later`
frees a buffer that queued work may use once that work has finished, so
that replacing a workspace need not wait for the GPU.

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
batch limit, starts one prefill call and the decoding calls that have room,
and then advances all running calls together: the calls run on streams
(`myds4_stream`, built on passes), so that whichever stream can continue runs
while the others wait for the GPU or the SSD, for any mix of models.

A request plans which KV state its prefill resumes from when the prefills
before it have ended, so that it can resume from their states; until then it
waits among the model's admitted requests. Decoding requests run in calls of
at most the model's rows per call (6 for DeepSeek, 8 for Qwen3.8-27B,
Flash-Next and Nemotron; fewer with the model's `rows_per_call` setting),
because rows of one call share the GPU's work per layer. The decoding
requests run in as many full calls as they fill, the others waiting for a
call; or, while one call's waits for expert reads leave the GPU idle, they
split evenly over at least two calls, every request in one. A call takes the
requests that are free, the longest waiting first, so groups of requests
advance independently: each call's waits are filled with the other calls' GPU
work. The worker chooses between the two arrangements at run time
(`server/src/calls.rs`), from the GPU's idle time while the engine thread
waited for reads (the status's `engine_times`), read every quarter second:
one second after the decoding requests or the arrangement change, a
two-second window in which that idle time is above 12% splits the requests,
and one below 3% returns to full calls. Only windows in which two or more
requests of the model decode and nothing else runs count. Measured with
UD-IQ1_M experts at 50 GiB, one call left the GPU idle 24-26% of the time
while its rows waited for reads, two calls 5-9%, and two calls of about half
the requests were the fastest arrangement for 2 to 8 requests. With most
experts in memory the GPU is busy anyway: full calls read the dense weights
once per layer for all their rows. A request that has just finished its prefill waits for the
prefills still to come when they are short (at most 4,096 tokens in all) and
for at most 20 seconds, so that its first decoding calls start full and do
not take SSD reads and cache room from the prefills; a request that has
decoded before never waits for a prefill. A DeepSeek model with the `runahead`
or `pipeline` option runs one call at a time: those decoders keep their jobs
in model-wide state.

Qwen prefill uses bounded chunks. DeepSeek plans the remaining prompt in one
call and adapts its internal chunk size to the model-memory budget. Qwen also
adapts its prefill rows and workspace to the budget.

Work that does not run on a stream stops every stream for its duration:
loading a model, encoding images and audio, and the worker's state operations
(snapshots, copies and SSD saves, which also wait for the GPU work queued by
streams).

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
source remain protected. A prompt takes the cheapest plan that fits: each
plan keeps the state it reuses, so when that state does not fit beside the
prompt's own, the next plan is tried, down to a cold start, which keeps
nothing. The configured limit may be exceeded by active state growth when
nothing idle remains to evict.

Each stored sequence owns its native state, snapshots and model handle.
Eviction saves the state to the SSD when the rules below say so, then drops
those resources together; a state discarded after a failure is never saved.
Rust ownership keeps native model handles on the engine thread. HTTP
handlers and the terminal receive copied statistics, without sharing the
native model handle or checking thread identity on each call.

Each prefill call receives its own image context. Image preprocessing and
feature storage belong to the request; model-specific position state is
kept in the sequence and its snapshots.

### Passes

A pass is a piece of the engine's work that runs on its own stack and gives
the processor back where it would wait, so that one thread can advance
several passes (`src/engine/pass.c`). DeepSeek, Qwen3.8-27B, Flash-Next and
Nemotron can each run as passes (`ModelOps`: `pass_new`, `pass_begin`,
`pass_end`, `pass_release`). The worker runs its calls on streams
(`myds4_stream`): a stream is a pass that takes one engine call after another.
`myds4_run` advances the streams round-robin on the engine thread. It polls
what each waits for instead of blocking; after a moment without progress it
sleeps briefly between its looks, because a spinning core takes power that the
GPU would otherwise use. The command line runs its calls
directly, without passes, and `build/tests/passes` runs two passes of a model
interleaved.

A switch saves the callee-saved registers and the stack pointer, in a few
lines of AArch64 assembly. A yield and a resume together take about 30 ns.
A stack is address space with a guard page; only the pages a pass uses are
committed.

Inside a pass these waits yield: `gpu_sync`, `gpu_wait_signal`, the expert
store's `cache_wait` and the wait for DeepSeek's Engram reader. Each yield
carries the condition that ends it. The code that resumes the pass asks
whether the condition holds (`pass_ready`) or waits for it (`pass_block`). A
wait whose condition already holds does not yield. Outside a pass the same
calls block on the calling thread.

Other passes' GPU work can run only between a pass's command buffers, so no
command buffer may be open at a yield; the GPU layer stops the program if
one is. The detached front of the next layer is the exception, because it
has no place in the queue until it is committed.

These belong to one pass:
- The model-wide variables that a family lists with `pass_variables`. They
  are copied out when the pass yields and back when it resumes. The expert
  store's values for the pass's current layer and the layer it is making
  resident are among them (`cache_pass_variables`): the store applies the
  preferences of the pass that is running.
- Its workspace. Qwen3.8-27B, Flash-Next and Nemotron give each pass a whole
  workspace of its own: every activation buffer, the expert planner and, for
  Nemotron, DFlash2's buffers. The workspace is allocated at the size the
  pass asks for, and freed when the pass ends. DeepSeek's workspace is too
  large to duplicate, so a pass owns only the buffers whose contents are used
  across a yield, or that the CPU fills for GPU work that may still be queued:
  the routing and selection buffers, the logits, what the routed experts read
  from the layer's front (the residual streams `H1`, the FFN input `XQ` and the
  shared expert's output `SHO`), the tile tables and work lists of the expert
  kernels, and the draft's buffers (its inputs from layers 37 to 39, its
  stages' streams and its tokens). The other work buffers stay shared, because
  nothing in them is used across a yield; they only grow while passes are
  running.
- A Metal shared event for its signals. Signals fire in queue order, and
  each pass numbers its signals as it encodes them, so a shared numbering
  could report one pass's signal because of another pass's later work.

Holds protect the experts that queued GPU work reads. A pass holds the
experts of a layer until the GPU has finished the layer, because a store that
evicted them could hand their slots to another pass. A layer reached through
its table of resident experts is held as a whole (`cache_hold_layer`), since
the GPU chooses among all of its experts. The cache's resize, the eviction
for a new expert and the memory pool's eviction all skip held slots. An
expert that a pass needs may take any slot that is not held when the store's
preferences leave none; a read-ahead is dropped instead. Requests take their
holds one expert at a time, so a large request cannot evict an expert that
it requested a moment earlier. A pass that needs a slot while every slot is
held waits until the GPU work of another pass releases one.

Passes running together share the expert store without working against each
other: a read-ahead belongs to the pass that issued it, and only that pass
cancels or demotes it; each pass has its own limit of read-aheads in flight;
the store's minimum number of slots only rises while several passes run; and
background readers go on beside another pass's demand reads, which would
otherwise hold them back almost all the time. A pass waits for its own last
command buffer (`gpu_sync`). The GPU's clock-keeping spin (`gpu_keep_clock`)
runs only when no other pass has a call in progress, because the spin's
command buffer would hold back that pass's work.

### Saved states

States evicted from the pool, and those in memory at quit, are written to
the SSD (`server/src/disk.rs`, `src/engine/state_file.c`); the policy is in
`server/src/kv.rs`, with tests that need no model.

**Rows and snapshots.** Each family lists a state's append-only arrays as
regions (`ModelOps.state_regions`): row 0's address, the row's bytes, the
positions a row covers (`ratio`), whether a guard row follows, and a `lag`.
A state of p positions holds `(p - lag) / ratio + guard` rows; the rows
before `(p - lag) / ratio` never change at later positions. DeepSeek's
compressed rows and index keys have ratio 2 (layer 20: 1) and a guard row,
the row a compressed pair shares with the next position; Flash-Next's
index keys have ratio 4; Qwen 27B's MTP key at the last position waits for
the next token (lag 1, guard 1). `state_copy` copies exactly these regions,
so copying, saving and reading use one list. A snapshot's parts
(`ModelOps.snapshot_parts`) are its buffers, host arrays and integers; the
same list over a live state's rolling rows writes a state's end without
allocating a snapshot, and `snapshot_alloc` makes an empty one to read into.

**Files.** A part holds each region's rows from those that its first
position leaves unchanged to its end, at 16 KiB-aligned offsets, with a
CRC32C (ARM crc32c instructions) for each 4 MiB block and each region's
shape in its header. A saved state is a list of parts, the first from
position 0, each starting at or before the previous one's end. Reading a
prefix takes each row from the latest part that holds it, reads only the
blocks the prefix needs and checks each one; a shape that differs from the
model's, a short file or a bad block fails the read. Files move through
aligned buffers and `F_NOCACHE`, so they pass the page cache, which would
otherwise grow by their size and count as memory in use. An index
(`ID.kv`) holds the model's identity (`myds4_state_identity`: the family's
state format version, the checkpoint's files by name, size and first and
last 64 KiB, the expert and DFlash2 files, and the options that change a
state's rows or shapes), the tokens, the parts and snapshots, and a SHA-256
of the rest. Files are written under temporary names, synced and renamed,
the index last; the index's modification time is the saved state's last
use, for the order of eviction after a restart.

**Rules.** A state is saved when it holds at least `kv_disk_tokens` tokens
that no other kept state holds (in memory, or saved); at quit, when no
state stays in memory, states are taken most recently used first, so that
of a conversation's turns the newest is written and the older ones it holds
are not. Each in-memory state carries a link: the saved state whose rows its
own equal before some position, and that saved state's version then. A
read sets it, a copy of rows (`Snapshot`, `Replay`) passes it on lowered to
the copy's position, and continuing in place lowers it to where the state
continues. A save appends one part from the linked position while the saved
state is as the link knew it, holds fewer than `kv_disk_tokens` tokens after
the state's (an append ends it where the state ends), and is not the one an
admitted request reads next; otherwise, or when the saved state's replaced
rows would outnumber its own, it writes a new saved state. Rows computed by
decoding and by prefill differ in rounding, so only rows that are the same
bytes are joined: a link exists only where the rows were copied from the
saved state. A save also writes the newest prompt-end snapshot (with
logits), the newest branch snapshot and the state's end; saved states that
the new one covers (holding fewer than `kv_disk_tokens` tokens beyond it)
are deleted. Room within `kv_disk` comes from saved states of no model
here, then the least recently used, used before the state being saved, and
counts the files that the save replaces; the saved state that an admitted
request reads next is kept, and the room is taken only once the new saved
state is written. A read that finds a file damaged or missing deletes its
saved state, and the request is planned again without it; one that cannot
open a file at the moment passes the saved state over for that request
only. An index of another version of the store, with its files, is left as
it is.

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

An indexer scores the compressed positions that a row sees and selects up
to 512 of them. Beyond 16,384 positions, layer 20's indexer also keeps the
row's 2,048 best blocks of 8 positions as its candidates. The indexers of
the later layers score and select only within those blocks.

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

When decoding, the cache evicts the expert with the lowest priority: the
sum of its selections, each weighted so that it halves over the configured
half-life (`cache_half_life`, 50 forward passes by default), plus its near
misses. A near miss is one of the 10 experts with a decoded row's highest
router logits among those that no row of its pass selected; it counts half a
selection and halves over 12 forward passes. Such an expert is often
selected within a few tokens and seldom later, so it stays a little longer
without displacing the experts that are selected often.

A slot holds one expert in a buffer of its own. When every layer's experts
are within a sixteenth of the largest expert's size, every buffer has that
size, any expert fits any slot, and the cache's limits count slots: MXFP4
experts all take 18.9 MB. A GGUF expert set quantizes its layers
differently: a UD-IQ1_M expert takes 7.0 to 10.4 MB and a UD-IQ2_XXS expert
7.8 to 12.2 MB, depending on its layer. For such a set a slot's buffer has
the size of its layer's experts, and the limits count bytes, in units of the
largest size, so that the same memory holds more experts (`sized` in
src/engine/io.c). When decoding, an expert that needs a slot takes the
lowest-priority slot if its buffer fits: the buffer is large enough and at
most a sixteenth larger than the expert. Otherwise it takes the
lowest-priority slot with a fitting buffer among the 64 lowest priorities,
which the next misses would evict anyway. If there is none, the cache gives
buffers back, the lowest priorities first, until a buffer of the expert's
size fits in its bytes, and allocates that buffer (a conversion). A
conversion gives back no expert that was selected or read ahead in the
current token. A read-ahead gives back at most two buffers and a demand at
most four. If the memory pool admits no buffer even then, the read-ahead
gets no slot, and the demand reuses a buffer that is large enough, whatever
it wastes: among the slots with such buffers, the one that the eviction
order takes first. If no buffer is large enough, the demand's buffer is
required memory, for which the memory pool takes idle memory back. If the
pool cannot, because passes hold what it would take back, the demand gives
back the slots that nothing holds, the lowest priority first, until its
buffer fits; with none left, its pass waits for another pass to release a
slot, as a pass does when every slot of one size is held. A prompt pass
goes through every layer and needs a layer's experts only until the next
layer. Its new buffers therefore have the largest size and serve all its
layers, and it reuses any buffer that is large enough: on a cache that it
fills itself, it behaves as with slots of one size. Decoding then replaces
the pass's experts and converts their buffers one miss at a time.

While decoding, DeepSeek reads ahead the experts of three forecasts, each
a router applied to a row known earlier. Before a layer's attention, the
layer's own router runs on the residual part of its FFN input; the GPU
computes the logits and the CPU takes each row's top six from them. After a
layer's routing, the next two layers' routers run on its FFN input. The
forecasts read each row's top three experts that are not in memory. The
residual's forecast and the next layer's first cancel the layer's queued
read-aheads outside each row's top six. The background readers take queued
read-aheads from the lowest layer up, because a pass needs the nearest
layer's experts first. When a layer's routing is known, its queued
read-aheads that it did not select are cancelled, which leaves the SSD to
the next layers' reads, and the unselected ones already read become
evictable by the score that earlier selections gave them.

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

A request's passes run from none up to `draft` of the draft model's drafts,
because every verified row costs GPU work and expert reads whether it is
kept or not. The count is the one with the most expected tokens per second:
one plus, for each position up to the count, the measured rate at which the
drafts up to that position were all kept, over the measured time of a pass
with that many drafts. Every count is first measured three times, in rounds
from the most down; afterwards every eighth pass tries a count next to the
best, so that the figures follow the text. In the server only passes with
no other call of the model beside them are measured. The outputs do not
depend on the count. Context-lookup proposals are verified as proposed.

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
