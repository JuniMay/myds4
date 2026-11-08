# myds4

myds4 runs DeepSeek-V4.1-Flash-1017 (`deepseek-ai/DeepSeek-V4.1-Flash-1017`,
475 GiB) on one Apple silicon Mac with less memory than the model. The dense
weights (8 GiB) stay in memory; the routed experts, from the checkpoint (MXFP4,
289 GB) or from a smaller GGUF build, are read from the SSD into a cache whose
size follows the memory the rest of the system leaves free. The host is C, the
GPU kernels are Metal. It runs prompts from the command line and serves the
OpenAI Chat Completions API, for example to DeepSeek Harness. It also runs
Qwen3.8-27B (NVIDIA's NVFP4 build, resident on the GPU, with MTP drafts)
through the same engine, and the server serves several models in one process,
each request going to the model it names.

myds4 is inspired by [ds4](https://github.com/antirez/ds4) and mlx-stream.

- [Usage](docs/usage.md): every option, the server and its API, the live view,
  checks and diagnostics, limitations.
- [Design](docs/design.md): data flow, numerics, the expert cache and memory
  policy, decoding, prefill and images.
- [Performance](docs/performance.md): the machine's limits and measured results.
- [The server](docs/server.md): the HTTP server, in Rust over the C engine:
  its models and their chat formats (DeepSeek's on DeepSeek's
  `deepseek-recipe` crates), its C ABI, its live view, and where its prompts
  differ from `encoding.py`.
- [Qwen3.8 models](docs/qwen.md): the Qwen3.8-27B model, its passes, MTP
  drafts and speed, and its checks against Hugging Face's reference.
- [Removed experiments](docs/experiments.md): the studies that were
  measured and removed, how each worked and what it showed.

## Requirements

- macOS 27 on Apple silicon with Metal 4.1 (developed on an M5 Max with 128 GiB
  of memory and a 2 TB SSD).
- Xcode command line tools, and yyjson and libjpeg-turbo found through
  `pkg-config`; for the server, a Rust toolchain (`cargo`).
- The checkpoint (475 GiB) and, for speed, a GGUF expert build (121 to 165 GB)
  on a local SSD.

## Download

With the Hugging Face CLI (`hf`), the checkpoint and unsloth's `UD-IQ1_M`
experts:

```sh
CHECKPOINT=/path/to/DeepSeek-V4.1-Flash-1017
GGUF=/path/to/DeepSeek-V4.1-Flash-1017-GGUF
hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 --local-dir "$CHECKPOINT"
hf download unsloth/DeepSeek-V4.1-Flash-1017-GGUF \
  --include "UD-IQ1_M/*-0000[1-4]-of-00005.gguf" --local-dir "$GGUF"
```

myds4 reads every shard of the checkpoint: its MXFP4 experts share shards with
the other tensors. Of a GGUF build it reads only the routed experts (and a
Q6_K output head): `UD-IQ1_M` keeps them in shards 1 to 4 (140 GB), `UD-IQ2_XXS` in
shards 1 to 5 of 6 (165 GB, `--include "UD-IQ2_XXS/*-0000[1-5]-of-00006.gguf"`),
and ISTA-DASLab's GSQ-RCO `IQ1_M` in all five of its files (121 GB, from
`ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF` with
`--include "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ1_M-*.gguf"`).

## Build

```sh
brew install pkgconf yyjson jpeg-turbo
make
make server
```

`build/myds4` embeds the Metal kernels, which compile when the program starts.
It generates text and runs the checks and benchmarks; `build/myds4-server`
serves the API over the same engine.

## Run

```sh
EXPERTS="$GGUF/UD-IQ1_M/DeepSeek-V4.1-Flash-1017-UD-IQ1_M-00001-of-00005.gguf"
build/myds4 -m "$CHECKPOINT" --experts "$EXPERTS" -p "What is the capital of France?"
```

Without `--experts`, myds4 uses the checkpoint's native MXFP4 experts and
original output head. With CED and substitution off, it runs the full model
using the router's selected experts. Numerical differences from the reference
runtime remain; see [Numerics](docs/design.md#numerics).
The server needs an API key:

```sh
mkdir -p ~/.config/myds4 && openssl rand -hex 32 > ~/.config/myds4/api-key && chmod 600 ~/.config/myds4/api-key
build/myds4-server -m "$CHECKPOINT" --experts "$EXPERTS" --cache 56 \
  --serve 8008 --api-key-file ~/.config/myds4/api-key --tui
```

It serves `http://127.0.0.1:8008/v1`, uses the router's selected experts, samples at temperature
1 and top-p 1, and shows a live view that the `q` key stops: tabs for the
throughput, the KV prefix cache's tree with each node's text, each decoding
request's experts by layer, the last minute and the log ([live
view](docs/usage.md#live-view)). In DeepSeek
Harness, add a provider with **Custom model API**, that base URL, the OpenAI
Chat Completions protocol and the key file's contents (see
[Usage](docs/usage.md#deepseek-harness)). Each run takes most of the memory,
so one runs at a time: a second waits for the first's lock.

To serve DeepSeek and Qwen together, list both in `--config FILE` as shown
in [Models](docs/server.md#models). Put `reserve` and `kv_cache` at the JSON
root: they set one system headroom target and one KV budget for the whole
server. Each model keeps its own context, batch limit and sampling options.
Configuration changes require a restart. In the live view, `n` switches
models and `6` opens Memory, with allocation tiles for all loaded models.

The server and the CLI accept `--ced off|N` for approximate decoder replay
during prefill, with `off` as the default. For example, `--ced 128` retains
128 decoder tokens plus `max(8, lookup_depth)` rewind rows. This can change output; see the
[option and cache behavior](docs/usage.md) and the
[measured comparisons](docs/performance.md#joint-prefill-and-server-ced).
The server's KV prefix cache resumes each prompt at the longest prefix it
shares with any conversation the server holds, at any position. Without
`--ced` it is exact: over rows that came from prompts, the result is byte
for byte that of prefilling the whole prompt. With `--ced` it recomputes
the last shared ring of positions (136 with the default `--lookup`) with
their windows stopping there, which is approximate, as CED is
([design](docs/design.md#the-kv-prefix-cache)).
Arrival and prefill logs show request counts, queue depth and actual chunks
without logging conversation text.

`--prefill-prune` skips decoder prefill batches outside the dependencies
of the final state and of a fork's branch-point capture. It is off by default and
applies with CED and substitution off. The tested MXFP4 and IQ2 outputs
match full prefill. Long-prompt prefill improved, while subsequent decode
slowed in the measured MXFP4 case; see
[prefill measurements](docs/performance.md#dependency-pruned-prefill-checks).

`--lookup N` enables context-copy drafting with up to 15 candidates.
Every candidate is checked by the target model with the requested sampler.
Without `--draft`, lookup does not load DSpark weights. It is off
by default. Context copy speeds repeated text in the measured copy task,
but slows the measured long DSH continuation; see
[context-copy measurements](docs/performance.md#context-copy-drafting).

The server's `--runahead` option lets ready requests advance while other
requests wait for expert reads. It groups ready requests at the same layer;
each request loads its own experts. It applies when at least two requests
are decoding; a request alone keeps ordinary decoding. Omit DSpark,
substitution and the online predictor when using it. The option is off by
default. See the [server comparison](docs/performance.md#http-runahead-comparison)
for measured throughput and its limits.

The server also has an experimental `--pipeline` option for concurrent
native MXFP4 requests. It keeps request stages in flight and combines ready
expert work across layers. A lone request keeps ordinary decoding. It is
off by default and requires runahead, DSpark, substitution and online
prediction to be off. See the [pipeline design](docs/design.md#experimental-pipelined-decoding)
and [measurements](docs/performance.md#pipelined-server-decoding).

## Performance

Exact decoding (no `--substitute`), measured on 2026-11-01 on the M5 Max with
`--cache 56`: each run a fresh process, greedy, no expert file in the page
cache. The story prompt is "Write a short story about a lighthouse keeper who
discovers a message in a bottle." and the Python one "Write a Python function
that parses an ISO 8601 date string and returns a datetime, with docstring and
tests." The table gives tokens/s over 150 generated tokens (two runs each) and
the time to process a prompt from an empty cache:

| Experts | Story | Python function | 2,779-token prompt | 8,921-token prompt |
| --- | ---: | ---: | ---: | ---: |
| Checkpoint (MXFP4) | 13.09–13.12 | 10.74–10.85 | 20.3 s | 22.9 s |
| `UD-IQ2_XXS` | 22.12–22.29 | 20.37–20.40 | 11.4 s | 15.0 s |
| `UD-IQ1_M` | 23.75–23.89 | 22.43 | 9.7 s | 14.6 s |
| GSQ-RCO `IQ1_M` | 25.05–25.20 | 24.18–24.28 | 8.4 s | 14.7 s |

Over 1,000 tokens, the GGUF builds decoded the story at 25.1 to 27.5 tokens/s
and the Python function at 21.6 to 25.7. Speculative decoding with
`--draft 2`, which keeps the output exact, made the Python function 19 to 30%
faster with the GGUF builds; the story was 9 to 10% slower with the unsloth
builds. The vision encoder takes about 0.6 s for a letter-size page.
[Performance](docs/performance.md) has the details.

## Tool calls in DeepSeek Harness

This is my personal experience across several sessions, not a
tested model behavior. With both unsloth builds (`UD-IQ2_XXS` and
`UD-IQ1_M`), the model kept making parallel tool calls and sub-tool calls
(tools called from inside the harness's code tool, such as several
`tools.read_image` calls in one `Promise.all`). With GSQ-RCO `IQ1_M`, parallel
tool calls were similar, but it made sub-tool calls rarely. The examples above
use `UD-IQ1_M` for this reason.
