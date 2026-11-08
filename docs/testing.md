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

Tokenizer cases use one reference runner for both architectures. The fixtures
include added tokens, round trips and pieces longer than 1,024 bytes:

```sh
make build/tests/tokenizer-reference
build/tests/tokenizer-reference /path/to/deepseek tests/fixtures/tokenizer_deepseek_v41.json
build/tests/tokenizer-reference /path/to/qwen tests/fixtures/tokenizer_qwen3_5.json
```

`build/tests/memory` checks the shared budget, unequal entry sizes, active
pins, expert-before-backbone eviction and separate KV storage.
`build/tests/memory-model CHECKPOINT MEMORY_GIB` evicts a real model's
weights, retains its KV state and compares every next-token logit after
reloading. Its competing reservation is synthetic; it is a correctness
check, not a throughput benchmark.

The metadata check can additionally compare two checkpoint indexes and
DeepSeek's generated Engram map. It reads headers and tokenizer data, not
model weights:

```sh
make build/tests/metadata
MYDS4_LOCK=build/tests/runtime.lock build/tests/metadata /path/to/deepseek /path/to/qwen
```

`tests/reference/engram.py CHECKPOINT` computes the same layout through the
checkpoint's Python reference and prints its fingerprint. It writes a
binary only when an output path is explicitly requested.

## GPU and model checks

Run on AC power, with one model process at a time and no user server using
the GPU. The Python measurement drivers enforce process, memory and GPU
checks. Standalone C checks require the operator to provide that window.
Do not interpret a correctness check's elapsed time as a throughput result.

```sh
make build/tests/vision
build/tests/vision
```

The vision check compares bounded activation and attention fixtures. Given
a Qwen checkpoint argument it reads an image on stdin and encodes it.
An optional output descriptor receives feature bytes; ordinary output is
a short shape report.

For HTTP tests, set `MYDS4_QWEN` to Qwen's checkpoint and
`MYDS4_CHECKPOINT` to DeepSeek's. Set `MYDS4_EXPERTS` only for tests that
should use GGUF experts. Run the selected Rust integration target with
`--test-threads 1`; some cases deliberately load both architectures inside
one server process. The lifecycle checks verify metadata-only startup,
loading the requested model, and service errors for unavailable weights.

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
vision and model intermediates. `tests/reference/qwen.py` compares text
logits and vision features with Transformers. Its `--image FILE` option
runs a CPU BF16 vision reference followed by the native encoder; `-` reads
image bytes from stdin. It reports SNR, cosine similarity and maximum
error. Its numerical tolerance is a diagnostic threshold, not a quality
benchmark.

Performance drivers and public evaluation inputs are under `benchmarks/`.
See [Performance](performance.md) before running or interpreting them.
