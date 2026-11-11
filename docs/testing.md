# Testing

## CPU checks

From the repository root:

```sh
make test
```

This builds and runs the maintained CPU checks and the Rust server tests.
It also builds the GPU checks, the model checks and the benchmarks without
running them, so that a changed engine interface cannot leave one of them
unbuildable. It clears model-path environment variables for the Rust tests
so a normal CPU run does not load large checkpoints. Generated check executables live
under `build/tests/`; application binaries remain directly under `build/`.

Tests are grouped by role:

- `tests/unit/`: cache, state, sampler, metadata and planning invariants.
- `tests/gpu/`: small numerical kernel checks.
- `tests/model/`: complete-model state, prefill and verification checks.
- `tests/reference/`: comparisons with upstream implementations.
- `tests/fixtures/` and `server/tests/fixtures/`: fixed test inputs.
- `server/tests/`: HTTP, model lifecycle and terminal integration.

`build/tests/pass` checks the coroutine switch without the GPU: resuming and
yielding, waits that are over already, per-pass variables and many passes in
turn. It prints what a yield and a resume cost together.

Keep a regression when it proves a maintained behavior. Share fixture
setup within a subsystem; do not retain separate experiments solely to
preserve their former measurements.

Tokenizer cases use one reference runner across architectures. The fixtures
include added tokens, round trips and pieces longer than 1,024 bytes:

```sh
make build/tests/tokenizer-reference
build/tests/tokenizer-reference /path/to/deepseek tests/fixtures/tokenizer_deepseek_v41.json
build/tests/tokenizer-reference /path/to/qwen tests/fixtures/tokenizer_qwen3_5.json
build/tests/tokenizer-reference /path/to/transcribe tests/fixtures/tokenizer_nemotron.json
```

`build/tests/memory` checks the shared budget, unequal entry sizes, active
pins, expert-before-backbone eviction and separate KV storage.
It also checks reclaiming idle entries for the shared RAM reserve without
discarding pinned weights or inflating recorded peak memory, and unloading
one model's idle entries: experts before the backbone they depend on, with
pinned entries and other models' entries kept.
`build/tests/memory-budget` runs the engine's memory code on a simulated
Mac (128 GiB installed, other programs holding 28 GiB): model memory stays
within the installed memory less the reserve less other programs, never
takes the space that KV occupies, stays under a lower `memory` setting, a
load after other programs grew first takes back idle experts, KV growth
takes its space back from idle experts, and expert slots charged before
their reads write them are not room. It also checks that 20 passes of small
workspace buffers at the reserve take back no expert, and that memory
messages reach the engine's log hook only for take-backs of 1 GiB or more.
With the available memory at the reserve, expert slots within the memory a
model reserved for a pass's fewest slots are admitted, and slots beyond it
are refused.
Its simulated kernel counts memory as used once it is written, as macOS
does.
`build/tests/cache-memory` checks the cache's resizing under controlled
memory counters, including that a shortfall without memory pressure gives
back only the slots that restore the reserve. With a second model's store
beside the first and the same shortfall for both, the first gives the
shortfall back and the second gives nothing back until the memory readings
can show it; the second store's unwritten slots count as in use too. It also checks which slot
bytes count as not yet written: a slot whose read is under way, a load that
dropped a piece, and the part of a slot beyond a smaller expert.
`build/tests/cache-mapping` checks that CPU reads need no GPU residency and
that only an encoded use maps an expert. On a store of two layers of four
experts, it also checks when a layer can run through a table of its experts'
addresses: only once a hold of the whole layer has been read, reaped,
mapped and committed. It stops being ready when the residency limit unmaps
a slot or one of its experts leaves its slot. Experts that a table used count
as used without a read, and one that left its slot is skipped.
On a store whose layout has a transform (as Flash-Next's compressed scales
do), it checks that the transform runs once per expert, only after every
piece of the expert has been read and before the expert counts as read; that
it sees the read bytes at the spans' offsets; and that its bytes at the
slot's end fit in the slot's size. A store whose slot limit was set for one
row must hold a whole layer once a wider pass asks for its fewest slots
(`cache_working_rows`); the check runs that pass in a child process, which
ends with a message when the store cannot.
`build/tests/weights-read` checks that a model's weights are read in
16 KiB-aligned spans into aligned memory, so that a load does not fill the
file cache, and that each tensor's bytes arrive intact.
`build/tests/weights-stop` checks that quitting stops a model's load after
the weight being read, returns its memory and leaves it unloaded, while a
load that a pass makes completes.
`build/tests/state-file` checks the saved states' files with a fake model
family whose rows and snapshots live in plain memory: four regions like the
real ones (plain rows, compressed pairs with a guard row and more than one
4 MiB block, one row per four positions, and rows a position late, as
Qwen's MTP keys). Every prefix of one part, of a part and its append, and
of a part that starts earlier (a rewind) must read back the rows that a
state_copy of the live state holds; a prefix past the last part, parts out
of order, a damaged block that the prefix needs, a damaged header, a short
part and another shape are refused, and a damaged block the prefix does
not need is not read. A snapshot with logits and a live state's end come
back byte for byte, a damaged snapshot is refused, and a second stop
request ends a write. A checkpoint's identity changes with its
`config.json`, a weights file's first bytes or size, and any file of a GGUF
set, and not with documents or files whose names start with a dot. The
server's tests (`make server-check`) check the saved states' policy in
`kv.rs` (plans from saved states, which states are saved and how, links
through copies and continuations, room and coverage, and that a save
neither changes the saved state an admitted request reads next nor cuts
off a saved state's own tail) and the store in `disk.rs` (index round
trips and damage, the store's file names, which files a start keeps and
deletes, another version's index kept, the folder's lock, and, through a
fake state, that a failed save takes no room, that an append and a
replacement count what they replace as room, and that a file that cannot
be opened at the moment does not count as damage).
`build/tests/memory-model CHECKPOINT MEMORY_GIB` evicts a real model's
weights, retains its KV state and compares every next-token logit after
reloading. Its competing reservation is synthetic; it is a correctness
check, not a throughput benchmark.

`build/tests/qwen-mtp CHECKPOINT` checks the normalized hidden supplied to
the Qwen draft head against the target logits. It also verifies every
accepted prefix of two- through six-row batches, including the next target
logits and the repaired draft-head history.

`build/tests/nemotron` checks calibrated activation codes and Mamba2 state
against independent CPU calculations. `build/tests/nemotron-model TARGET
DFLASH2` compares all logits of a 16-row verification pass with sequential
decoding. It then checks several accepted prefixes, their next target
logits and their next draft proposals. Eviction takes the experts used
longest ago first, so each decoded row must mark exactly its chosen experts
used in the memory pool. Each decoded row must also leave its routes: each
MoE layer's chosen experts, each once and within the layer's experts, with
where each one was. The check then brings nearly every expert into
memory with 1,500 varied tokens; that prompt must count none of its expert
uses as decoding's, those of its layers that ran through their tables
included. The next row must mark its experts used too, and at least one
layer must reach its experts through its table of resident experts. The program prints hashes of its logits, so that two
builds can be compared, and the experts read and evicted. With
`--memory GIB` below its default 28, its experts are evicted and their
slots reused; no layer then holds all of its experts, so the resident-layer
check does not run. `tests/reference/nemotron.py`
compares selected backbone blocks with the upstream CPU implementation.

`build/tests/speech-model CHECKPOINT` reads bounded audio on stdin and
transcribes it. `--features` returns BF16 speech feature rows; `--decode`
returns a two-word header (sample rate and count) followed by mono FP32
samples. These modes let reference tests keep audio and features in memory.

The metadata check can additionally compare two checkpoint indexes and
DeepSeek's generated Engram map. It reads headers and tokenizer data, not
model weights:

```sh
make build/tests/metadata
build/tests/metadata /path/to/deepseek /path/to/qwen
```

`tests/reference/engram.py CHECKPOINT` computes the same layout through the
checkpoint's Python reference and prints its fingerprint. It writes a
binary only when an output path is explicitly requested.

## GPU and model checks

Run on AC power, with one model process at a time and no user server using
the GPU. The Python measurement drivers enforce process, memory and GPU
checks. Standalone C checks require the operator to provide that window.
Do not interpret a correctness check's elapsed time as a throughput result.

`make pass-eviction-check` needs no model. It opens two command buffers,
commits a third behind them, then allocates past a 64 MiB limit so that an
idle expert goes back. The GPU runs the third only after the two open ones,
so the eviction must not wait for the GPU; a wait that never ends fails the
check by its 20-second alarm.

`build/tests/passes CHECKPOINT --text FILE [--experts GGUF] [--memory GIB]
[--tokens N] [--decode N] [--draft N] [--draft-model DIR] [--no-garbage]
[--verbose]` runs any model family. It needs a text of at least twice
`--tokens` tokens. It runs two passes, each a prefill and greedy decoding
(with `--draft N`, drafting and verifying), alone and then interleaved at
every yield, and compares every logit vector and every proposed token byte
for byte. A buffer or variable that should belong to a pass but does not shows
up as a difference. It also prints how many layer passes went through tables
and how many experts were evicted while the passes were interleaved, so a run
at a memory size that evicts shows that layer holds were contested.
For DeepSeek it also fills the buffers that passes share with garbage between
the passes' turns; `--no-garbage` instead leaves GPU work queued across the
switches, so that the two passes' command buffers overlap in the queue, which
catches CPU-filled shared buffers. Run it with and without `--no-garbage` for
DeepSeek. The check prints how long the passes took alone and interleaved.
`--verbose` shows the memory messages that are usually suppressed. `--rows N`
decodes N sequences per pass, `--release` releases the state between calls as
a server stream does, and `--phases` orders the passes as the server does:
the first prompt alone, then the second prompt beside the first request's
decoding, each in a pass of its own.

`benchmarks/throughput.py --new SERVER [--old SERVER] [--deepseek DIR] [--experts
GGUF] [--qwen DIR] [--flash DIR] [--nemotron DIR] --cases deepseek:8
deepseek:3,qwen:3 ...` measures the total tokens per second of concurrent
requests, greedy, on fresh servers. For each case it reports the `--old`
server serving one request at a time (`old:1`) and with its batch decoding
(`old:b`), and the `--new` server serving every request at once (`new`); each
model first answers a short warm-up request, so the time counts neither its
load nor its coldest cache. For each mode it also reports the steady rate
(all requests decoding, none finished yet), the expert bytes read, the time
waiting for them, the expert cache's hit rate and the GPU's mean use (sampled
from the system, so it includes the desktop's share). It runs on AC power
with an idle GPU, as the other measurement drivers do. It also holds the
correctness guard for concurrency: each request first runs alone on the
`--new` server, and every response of every mode must equal that answer. A
response that differs fails the case whatever its speed, and the report names
the first differing character. `--modes` chooses the modes, `--no-reference`
skips the runs alone (and the check), `--option model:key:value` sets a
per-model configuration key (for instance `deepseek:runahead:true`), and
`--server-log -` prints the servers' messages.

`build/tests/gpu-steps` needs no model. A buffer that queued work uses must
stay allocated and charged until that work has finished (`gpu_free_later`,
`gpu_reap`), and one that no work uses must be freed at once.

`build/tests/experts` needs no model. It checks that the grouped NVFP4
expert kernels give each product, BF16 boundary and ordered mixture of
separate kernels, for SwiGLU and squared ReLU and for 1 to 32 rows, through
the planner's groups and through a layer's table. On a store of three small
experts read from a temporary file, it checks that the planner's held
selection survives memory pressure, that the table path's output equals the
acquired path's, that a new selection moves the holds, and that a release
lets every slot go back. It also compares the 16-row prompt matrix tile with
the 128-row tile for 1 to 64 rows.

```sh
make build/tests/vision
build/tests/vision
```

The vision check compares bounded activation and attention fixtures. Given
a Qwen checkpoint argument it reads an image on stdin and encodes it.
An optional output descriptor receives feature bytes; ordinary output is
a short shape report.

`build/tests/flash-model CHECKPOINT --long` checks a color response, complete
logits after state restoration, and state forks beyond the sparse indexer's
2,048-token selection budget. Each decoded row must mark exactly its chosen
experts used, plus the experts it read ahead that the next layer did not
choose (the store's read-ahead and read-ahead hit counts give that number).
When the row evicts, a later miss may take such a read-ahead's slot, which
then counts once for both; the count must then lie between the chosen
experts and the two together. Each decoded row must also leave its routes
(each layer's chosen experts, each once and within the layer's experts,
with where each one was), and a prompt's rows none. The program prints
hashes of its logits, so that two builds
can be compared, and the experts read and evicted. Within its default
32 GiB no layer holds all of its 512 experts, so its rows do not reach the
table path. `--memory GIB` sets a lower limit, under which experts are
evicted more often. `tests/reference/flash.py --checkpoint DIR
--layer N` compares a requested text block with Transformers using the same
input and only that block's selected experts.

`build/tests/state-file-model CHECKPOINT` checks saved states with a real
model of any family. A state's parts, read back for a prefix, must hold the
rows a state_copy of the live state held when they were written, also into
a state of another capacity, across an append and across a part from an
earlier position. A state read back with a saved snapshot, or with its end
written straight from the live state, must decode 24 rows with the same
logits, byte for byte, as the state itself; a damaged block is refused.
`--draft N`, `--lookup N`, `--experts GGUF` and `--draft-model DIR` set the
model's options, `--image` puts a generated test image in the prompt, and
`--unload` also writes a state while the model's weights are unloaded and
continues it after they are back. It writes its files to a temporary
folder, removed before it exits.

`build/tests/gguf-decode [GGUF]` compares GPU dequantization with llama.cpp's
scalar formulas. It covers Q2_K, IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S,
IQ3_XXS and the Q6_K output head, including both IQ2_S kernel layouts.
It checks random blocks and samples each supported expert tensor in the
supplied shard set. The tensor metadata chooses the format; a GGUF variant
can mix several formats. This checks stored-value decoding, not model quality.

The image reference mode in `tests/reference/qwen.py` also accepts Flash-Next.
It keeps per-block traces in memory, limits those fixtures to 1,024 patches,
and requires at least 45 dB SNR for each block on identical inputs. Its
complete quantized-encoder limits are 20 dB SNR and 0.995 cosine similarity.
These are numerical regression limits, not evidence of downstream image
quality. The existing Qwen3.8-27B limits remain 30 dB and 0.999.

For HTTP tests, set `MYDS4_QWEN` to Qwen's checkpoint and
`MYDS4_CHECKPOINT` to DeepSeek's. Set `MYDS4_EXPERTS` only for tests that
should use GGUF experts. Run the selected Rust integration target with
`--test-threads 1`; some cases deliberately load both architectures inside
one server process. The lifecycle checks verify metadata-only startup,
loading the requested model, and service errors for unavailable weights.

The `live_view` target needs `MYDS4_CHECKPOINT` and `MYDS4_EXPERTS`. It runs
the terminal interface in a 160 × 46 pseudo-terminal while a request
decodes, selects each tab by key, wheel and click, and checks what each tab
draws and the server's exit. Nothing but the interface may write to the
terminal while it is up: the engine's lines, such as a model's load, must
reach the Log tab instead, and the title and tabs must stay on the top rows.
Its terminal emulator scrolls, as a terminal does, when a line ends on the
bottom row. It prints every tab's screen when a check fails, or when
`MYDS4_PRINT_TABS` is set.

Set `MYDS4_FLASH` for Flash-Next's image and conversation-reuse check, and
for its saved-state check: a conversation's next turn after a restart
continues from the state saved at quit with the same reuse and reply as the
state kept in memory gives.
Set `MYDS4_NEMOTRON` and `MYDS4_DFLASH` for Transcribe's shared-memory
lifecycle check. The audio reuse test also reads a JSON chat request from
`MYDS4_AUDIO_REQUEST`; it must contain inline `input_audio` data. The path
can be `/dev/stdin` to keep the fixture in memory. That test compares a
repeated transcription and a continued conversation with fresh execution,
both without drafts and with DFlash when its checkpoint is supplied.

The `yield_` cases in the `qwen` HTTP integration target check ownership,
authentication, retries, disconnects and queued requests. With Qwen and
DeepSeek checkpoints configured, they pause two active requests, reclaim
weights, preserve their sampling and KV state, and compare complete replies
after resume. Omitting `MYDS4_EXPERTS` selects native MXFP4 for that check.
Positive-cap checks cross a maintenance cycle and verify lazy reload for
idle models. The development-lock case confirms that a normal server can
load a model while a development wrapper holds its own lock.

The `prefix_fork` target needs `MYDS4_CHECKPOINT`; `MYDS4_EXPERTS` is
optional there. Besides prefix reuse and the KV budget, it sends SIGTERM
5 s into a DeepSeek prefill of about 15,000 tokens and requires the server
to exit within 5 s and to answer that request with 503. With
`--idle-unload 3`, it requires the model to read unloaded at least 2.5 s
after a reply, with at least 90% of its model memory given back, the
process's footprint down by at least 90%, and its session kept. The model
must stay unloaded for 35 s while the server keeps calling its idle hook;
the same request must then load it again and give the same greedy reply. A
prompt that shares only its first tokens with the one idle state, in a cache
with room for one state, must be answered in that state's place.

The `kv_disk` target needs `MYDS4_CHECKPOINT` (DeepSeek; `MYDS4_EXPERTS`
optional) and saves states in temporary folders with `--kv-disk-tokens
256`. A state that another one evicts is saved, and its conversation's next
turn reads it back with the same reuse and reply as a server that kept it
in memory. A state saved at quit answers its prompt again after a restart
from its saved prompt end, without a prefill and with the same reply. A
part damaged after the start is refused when it is read: the prompt is
prefilled cold, with the cold reply, and the saved state is deleted.
Within a `--kv-disk` room for two of three states, the two most recently
used are kept. A second server on the same folder runs without saved
states; neither server loads its weights. The other HTTP targets start
their servers with `--kv-disk 0`, so that their cold baselines stay cold
and they write nothing to the SSD.

Model checks cover prefix forks, draft rollback, batched rows, compressed
history, dependency pruning and concurrent execution. Build the relevant
`build/tests/` target listed in the Makefile. Compare all relevant state and
logits for changes that claim numerical equivalence, not generated text
alone.

## Upstream comparisons

Reference scripts need a test environment with NumPy, PyTorch,
Transformers, safetensors, tokenizers and Pillow. These packages are not
runtime engine dependencies. They use local checkpoints and disable
online model downloads.

`tests/reference/deepseek.py` compares tokenization, prompt formatting,
vision and model intermediates. Its full-model check reports routing,
per-layer SNR, top-token agreement and KL divergence of the output
distributions. Its exit status checks top-token agreement only; inspect
the numerical measurements before claiming equivalence.
`tests/reference/qwen.py` compares text
logits and vision features with Transformers. Its `--image FILE` option
runs a CPU BF16 vision reference followed by the native encoder; `-` reads
image bytes from stdin. It reports SNR, cosine similarity and maximum
error. Its numerical tolerance is a diagnostic threshold, not a quality
benchmark.

Performance drivers and public evaluation inputs are under `benchmarks/`.
See [Performance](performance.md) before running or interpreting them.
