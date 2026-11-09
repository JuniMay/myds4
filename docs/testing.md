# Testing

## CPU checks

From the repository root:

```sh
make test
```

This builds and runs the maintained CPU checks and the Rust server tests.
It clears model-path environment variables for the Rust tests so a normal
CPU run does not load large checkpoints. Generated check executables live
under `build/tests/`; application binaries remain directly under `build/`.

Tests are grouped by role:

- `tests/unit/`: cache, state, sampler, metadata and planning invariants.
- `tests/gpu/`: small numerical kernel checks.
- `tests/model/`: complete-model state, prefill and verification checks.
- `tests/reference/`: comparisons with upstream implementations.
- `tests/fixtures/` and `server/tests/fixtures/`: fixed test inputs.
- `server/tests/`: HTTP, model lifecycle and terminal integration.

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
discarding pinned weights or inflating recorded peak memory.
`build/tests/memory-budget` runs the engine's memory code on a simulated
Mac (128 GiB installed, other programs holding 28 GiB): model memory stays
within the installed memory less the reserve less other programs, never
takes the space that KV occupies, stays under a lower `memory` setting, a
load after other programs grew first takes back idle experts, KV growth
takes its space back from idle experts, and expert slots charged before
their reads write them are not room. It also checks that 20 passes of small
workspace buffers at the reserve take back no expert, and that memory
messages reach the engine's log hook only for take-backs of 1 GiB or more.
Its simulated kernel counts memory as used once it is written, as macOS
does.
`build/tests/cache-memory` checks the cache's resizing under controlled
memory counters, including that a shortfall without memory pressure gives
back only the slots that restore the reserve. It also checks which slot
bytes count as not yet written: a slot whose read is under way, a load that
dropped a piece, and the part of a slot beyond a smaller expert.
`build/tests/weights-read` checks that a model's weights are read in
16 KiB-aligned spans into aligned memory, so that a load does not fill the
file cache, and that each tensor's bytes arrive intact.
`build/tests/weights-stop` checks that quitting stops a model's load after
the weight being read, returns its memory and leaves it unloaded, while a
load that a pass makes completes.
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
logits and their next draft proposals. `tests/reference/nemotron.py`
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

`make pass-eviction-check` needs no model. It opens two command buffers as
a decode step opens a layer's expert buffers, commits a third behind them,
then allocates past a 64 MiB limit so that an idle expert goes back. The
eviction must not wait for the GPU; a wait that never ends fails the check
by its 20-second alarm.

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
2,048-token selection budget. `tests/reference/flash.py --checkpoint DIR
--layer N` compares a requested text block with Transformers using the same
input and only that block's selected experts.

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

Set `MYDS4_FLASH` for Flash-Next's image and conversation-reuse check.
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
to exit within 5 s and to answer that request with 503.

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
