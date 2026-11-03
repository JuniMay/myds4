# Usage

How to run myds4 beyond the first steps in the [README](../README.md): the
command-line options, the server and its API, the live view, the checks and
diagnostics, the online prefetch predictor and the limitations. See
[Design](design.md) for how it works and [Performance](performance.md) for
measurements.

## Checks and data prerequisites

The checks and the tools that regenerate `data/` need Python 3 with `torch`,
`numpy`, `safetensors`, `tokenizers` and `sympy`, for example in a virtual
environment in `.venv`; the vision check also needs `pillow`. The program
itself needs no Python.

`build/myds4` embeds the kernels and two tables from `data/`, which
`tools/data/gen_data.py` generates from the checkpoint:

```sh
.venv/bin/python tools/data/gen_data.py /path/to/DeepSeek-V4.1-Flash-1017
```

- `data/engram.bin`: the Engram hashing constants and the compressed token
  map, computed by the checkpoint's `inference/engram.py`.
- `data/unicode.bin`: Unicode general-category runs for the tokenizer.

## Running the model

Each run that loads the model takes most of the memory, so only one runs at
a time: it holds an exclusive lock on `myds4.lock` beside the executable (or
on the file `MYDS4_LOCK` names) until it ends, and a second run waits for
it, saying so on stderr. `lockf(1)` takes the same lock.

| Option | Effect |
| --- | --- |
| `-p TEXT`, `-f FILE` | The user message (wrapped in the chat template). |
| `--raw` | Use the text as is, without the chat template. |
| `--think N` | Thinking effort 1 to 100; 0, the default, disables thinking. |
| `--image FILE` | An image (any format macOS reads; JPEG through libjpeg-turbo) before the prompt's text, up to 16. With `--raw`, each image takes a `<｜deepseek_image｜>` of the prompt instead. The vision encoder's weights (0.97 GB) load at the first image (see [Design](design.md#images)). |
| `--vision-fd FD` | With `--image`: write the images' feature rows (BF16, 5,120 per token) to the file descriptor FD and exit, for `tools/reference/check.py vision`. |
| `-n N` | Tokens to generate (default 128). |
| `-t T`, `--temperature T` | Finite nonnegative sampling temperature; 0, the default, is greedy. |
| `--top-p P` | Nucleus probability from 0 to 1; default 1 samples the full distribution. Zero retains only the highest-probability token. |
| `--seed N` | Nonnegative integer seed for repeatable sampling. Omit it to initialize each run from system randomness. |
| `--ctx N` | Context capacity in tokens, up to 1,048,576, the model's own (default 16,384). The attention state of a conversation takes about 3.1 KB per token and grows with it. |
| `--ced off\|N` | Approximate prefill, default `off`. A depth from 128 to 1,048,576 retains N decoder tokens plus five rows for cached-request rewinds. All new tokens run through the encoder and global KV projections; only the retained suffix runs through later decoder layers. Shorter inputs use full prefill. Decoding still uses all layers. Local KV and generated output can differ from full prefill. Available in the CLI and the server. |
| `--cache GB` | Upper bound on expert memory, including reserved draft pools. After reserving those pools, the budget must hold two backbone layers' experts. The cache adapts to available memory and keeps a reserve for other programs (`--reserve`); the requested budget is not a guaranteed allocation. Slots outside the GPU's mapping limit remain in memory and are mapped when needed. See [Design](design.md#the-expert-cache-and-memory). |
| `--reserve GB` | The memory the expert cache leaves to other programs, 2 to 1,024 GiB (default 10). The cache starts within the free memory less the reserve, gives slots back when the available memory falls below it, and grows only while the available memory exceeds it by 4 GiB. A smaller reserve keeps more experts in memory and leaves less to other applications; the cache still gives slots back when the kernel reports pressure, swaps or compresses memory while the free memory is below the reserve. |
| `--cache-half-life N` | Retention-score half-life, 1 to 1,000,000 forward-pass advances; default 50. A decode or verification pass advances once, and prefill advances once at its end. A DSpark pass can emit several tokens, so these are not emitted-token units. |
| `--cache-read-cost` | Off by default. Multiply backbone decode eviction priority by the expert's aligned read bytes divided by the fixed slot size. Raw score history, prefill eviction, memory-pressure release and GPU unmapping retain their existing rules. |
| `--online-predictor` | Learn a CPU correction to available one- and two-layer prefetch forecasts during inference. Included in the normal engine, off by default. Works with substitution, DSpark, GPU routing and serving. See [online CPU predictor](#online-cpu-prefetch-predictor) for learning and measurement scope. |
| `--predictor-state FILE` | Enable online learning, load compatible state if FILE exists, and save changed state on clean completion or handled shutdown. The parent directory must exist and be writable. See [persistent predictor state](#persistent-predictor-state). |
| `--draft N` | Speculative decoding with DSpark: each pass verifies up to N (1 to 5) proposed tokens using the main model's requested temperature and top-p. Supports greedy and sampled generation. By default, all 128 experts of each draft stage are preloaded within the total cache budget. See [Design](design.md#speculative-decoding). |
| `--draft-cache N` | With `--draft`, reserve N expert slots per stage, 15 to 128. The default, 0 or 128, preloads all draft experts. Values 15–127 load native selections on demand into separate stage pools, evicting their least recently used experts. The pools use MXFP4-sized slots even with a Q2 backbone and count against `--cache`. |
| `--draft-trim` | With `--draft`, compute queries, outputs and feed-forward networks (FFNs) for only the requested final-stage rows. All five key/value rows remain available to attention; the earlier stages still compute all five rows. Off by default. |
| `--substitute D` | Approximate decoding and prefill, off by default: selected experts that are not in memory give way to the layer's resident experts with the highest selection keys, if such a key is at most D below the missing one's, the lowest-ranked first; the two selected experts with the highest scores are always kept (0.1 to 1.0 measured). In a prompt, an expert that one of its rows cannot do without is read and serves every row that selected it; the rows' other missing experts give way. Faster; the output differs from the model's and varies from run to run (see [Performance](performance.md#approximate-decoding-and-prefill)). |
| `--gpu-route` | Exact decoding in which the GPU looks up each layer's experts. A layer whose published table does not mark every required expert ready is handled again through the CPU path. Off by default; earlier measurements found the same output and 0.5 to 1.1% higher throughput on the development machine (see [Design](design.md#experts-looked-up-by-the-gpu) and [Performance](performance.md#experts-looked-up-by-the-gpu)). Those timings predate the readiness correction. For a token decoded alone, not for verify passes or concurrent requests. With `--substitute`, the GPU applies the same rule to the table's publication-time state. |
| `--experts GGUF` | Read the routed experts from GGUF files in llama.cpp's layout (any shard of a split set; see [Design](design.md#experts-from-gguf-files)), for example unsloth's `UD-IQ2_XXS` build, whose experts are 7.6 to 12.1 MB instead of 18.9 MB, or ISTA-DASLab's GSQ-RCO `IQ1_M` build (6.9 to 8.7 MB), and a Q6_K output head if the file has one. Their codes approximate the checkpoint's weights. |
| `--perplexity` | Decode the text (without the chat template) one token at a time and print the perplexity of each next token, and how often it was the most likely one. |
| `--prefill N` | With `--perplexity`: prefill the text's first N tokens in one pass, and score only the tokens after them. |
| `--logits-fd FD` | With `--perplexity`: write the logits (129,280 float32) that each token is scored on to the file descriptor FD, for comparing two settings by KL divergence. |

Per-layer dumps (`--dump-fd`) and the full-row comparison (`--check-prefill`)
require `--ced off`. The retained CED checks compare compact execution with
its corresponding full-shape mask instead.

The program prints the generated text, then statistics: prefill and decode
speed, expert cache hits, logical expert-read bytes and time spent waiting for reads, GPU
time, the cache's size, the process footprint, the pages the system
swapped out and compressed during the run and the most of the process's
memory that was compressed, the expert reads per decoded token (on demand,
read ahead and used, still being read when requested, cancelled, and
substituted experts), and decode time per token by phase; with `--draft`,
also the passes and the drafts kept per pass.

## Server

`build/myds4-server --serve PORT` serves the OpenAI Chat Completions API on
`http://127.0.0.1:PORT/v1`, for clients such as the DeepSeek Harness app.
`make server` builds it (it needs `cargo`; see [The server](server.md#build)).
It takes the model and engine options of `build/myds4` above (`-m`,
`--experts`, `--ctx`, `--ced`, the cache, draft and predictor options,
`--substitute` and `--gpu-route`) and the server options below.
Every request but `GET /health` needs the API key, which the server reads
from `--api-key-file FILE` or from the `MYDS4_API_KEY` environment variable,
and which clients send as `Authorization: Bearer KEY` (or `x-api-key: KEY`).
The server refuses to start without a key of 16 to 256 characters.

The [README](../README.md#run) shows how to create a key file and start the server.

| Option | Effect |
| --- | --- |
| `--serve PORT` | Serve on this port. `--experts`, cache settings, draft settings, `--substitute`, `--gpu-route` and `--ctx` apply to every request. |
| `--host ADDR` | The IPv4 address to listen on (default `127.0.0.1`, this machine only). The server speaks plain HTTP: on another address, keys and conversations cross the network unencrypted. |
| `--api-key-file FILE` | The key clients must send. |
| `--kv-cache GB` | The memory the conversations' attention state may take (default 8 GiB, apart from `--cache`): a conversation takes 66 MiB at first and up to 3.1 GiB at 1,048,576 tokens. The least recently used conversations are dropped to make room; a request whose conversation alone needs more is refused. |
| `-t T`, `--temperature T` | The temperature of requests that set none (default 1, the checkpoint's recommendation; 0 is greedy). |
| `--top-p P` | The nucleus probability of requests that set none (default 1). |
| `--seed N` | The sampling seed of requests that set none. Without this option, each request receives a random seed. |
| `--think N` | The thinking default: 1–100 enables thinking at that effort; 0 disables it. Without this option, the server enables thinking at effort 75. Request fields take precedence. |
| `--batch N` | Requests decoded together at most, 1 to 6 (default 4). Requests that overlap then share each forward pass: more tokens per second in all, fewer for each request (see [Performance](performance.md#concurrent-requests)). A request alone runs the same at any N. |
| `--tui` | A live view on the terminal in place of the log (see below); the `q` key stops the server as Ctrl-C does. |
| `--render FILE` | Print the prompt of a JSON test case of the checkpoint's `encoding/tests` (or a request's messages, tools, thinking mode and effort) as the server renders it, and exit; for `tools/reference/check.py encoding`. |

Endpoints:

| Endpoint | Answer |
| --- | --- |
| `POST /v1/chat/completions` | A chat completion: streamed as server-sent events (`"stream": true`, with usage when `stream_options.include_usage` is set) or as one JSON response. Reasoning arrives in `reasoning_content`, tool calls in `tool_calls`. |
| `GET /v1/models` | The model `deepseek-v4.1-flash`, its context (`context_length`) and output cap (`max_output_tokens`). |
| `GET /v1/status` | What the server is doing: the phase (idle, prefill, decode) and its request, the waiting requests, the sessions, the KV and expert caches, memory, and totals. |
| `GET /health` | `{"status": "ok"}`, without a key. |

A request may set `messages` (roles `system`, `developer`, `user`,
`assistant`, `tool`), `tools`, `stream`, `max_tokens` or
`max_completion_tokens`, `temperature`, `top_p`, `seed`, `thinking`
(`{"type": "enabled"}`, `"adaptive"` or `"disabled"`, or `true` or
`false`) and `reasoning_effort` (1 to 100, or a level as DeepSeek's request
converter maps it for V4.1: `minimal` and `low` 50; `medium`, `high` and
`xhigh` 75; `ultra` and `max` 100; default 75). Without a `thinking`
setting, an explicit `reasoning_effort` of `none` turns thinking off and
another explicit effort enables it. If neither field is set, the server's
thinking default applies. Other fields are ignored. With `--draft`, both
greedy and sampled requests use speculative verification when decoding alone.

Without options the server decodes exactly (no `--substitute`), without
drafts or the online predictor, samples with temperature 1 and top-p 1 as the
checkpoint's README recommends (`--temperature 0` for greedy decoding), holds
a context of 1,048,576 tokens, thinks at effort 75 and decodes up to 4
requests together. These are defaults, not forced overrides: a request can supply its own
`temperature`, `top_p`, `seed`, `thinking` and `reasoning_effort`. Null sampling
fields use the server defaults. The log reports each request's effective
temperature/top-p and actual draft-pass count.

For a sampled server with persistent prefetch learning, set `CHECKPOINT`,
`API_KEY_FILE` and `PREDICTOR_STATE` to the desired paths, then run:

```sh
build/myds4-server -m "$CHECKPOINT" --serve 8008 --api-key-file "$API_KEY_FILE" \
  --draft 2 --draft-cache 32 \
  --predictor-state "$PREDICTOR_STATE"
```

The server receives prompts and images through requests. CLI `-f` loads a
single text prompt and is not needed for serving. `--think` sets the server
default only when supplied; request settings still take precedence.

User messages and tool results may hold images as content parts
(`{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}`,
or an `image` part with base64 `data` or a `source`), in any format macOS
reads. Only `data:` URLs are read: the server fetches no URLs and reads no
files for a client, and refuses a request with any other image source.
Images come only from image parts: the placeholder's text
`<｜deepseek_image｜>` in a message or a tool result (a file that mentions
it, say) is read as text. An image takes up to 1,024 prompt tokens. The
vision encoder runs for the images
of the part of the prompt that is prefilled; images in a conversation's
earlier turns, whose rows the session holds, are not encoded again. For the
DeepSeek Harness, the model's entry needs `image` among its inputs
(`input: [text, image]`).

Requests run in queue order. The server converts and tokenizes a request
before enqueueing it, so a short request can enter first even if another HTTP
body arrived earlier. A new request is prefilled while the ones
decoding wait, and then up to `--batch` requests decode together, a token
each per step; each is computed exactly as it would be alone, so a greedy
request's text does not depend on what else runs. With the GGUF experts and
exact decoding, two or four requests together decode at about the speed of
one in all (each row reads its own experts, and the SSD binds; see
[Performance](performance.md#concurrent-requests)). A request that
repeats a conversation's last prompt (a retry) continues from that prompt's
saved state and logits, so a greedy retry gives the same text.

Each request's system, user, tool and context messages form one prompt. The
server saves the system prefix during that prompt's prefill, without a
second pass at the first user turn. Prompts longer than 16,384 new tokens
still use bounded chunks. Separate requests keep separate attention states.
With CED enabled, a system-prefix snapshot may contain only encoder state.
The server forks that snapshot only when the new suffix exceeds the replay
depth plus five rows; otherwise it chooses a complete cached state or
prefills anew.

While no request runs, the server keeps its experts in memory, so the next
request finds them as the last one left them. Every second it checks memory
as it does between tokens: when other programs take memory (the kernel
reports pressure, the system swaps, the kernel compresses memory while the
free memory is below the reserve, 10 GiB unless `--reserve` sets it, or
the available memory falls
below the reserve), it gives expert slots back, as it does when the memory
that all processes keep resident for the GPU passes the GPU's limit. The
whole cache counts as the server's memory, for example in Activity Monitor.
The slots mapped for the GPU are wired; the kernel could compress or swap
the others like any program's memory, which is why the server gives slots
back as soon as the kernel compresses anything while the free memory is
below the reserve.

The log on stderr has a line per event: each HTTP request's arrival, its body
size, message and tool counts, queue admission, and waiting-request count.
Prefill logs show new and reused tokens, configured CED depth, actual chunk
count, encoder and decoder token counts, saved prefix, speed and expert reads.
Queue admission also reports requests taken together for subsequent prefills.
`GET /v1/status` includes `ced_depth` (0 means off) and `last_prefill` with the
last completed plan. These fields distinguish separate HTTP requests from
chunks inside one prompt. The log also shows the
decoding's progress every 10 s, and its end (tokens, speed, expert hits,
SSD reads, KV sessions, memory), expert slots given back while idle, and
refused or unauthorized requests. It never shows keys or conversation text.

### Live view

With `--tui`, when stderr is a terminal, the server shows a live view there
in place of the log, redrawn twice a second on the terminal's alternate
screen (ratatui on its crossterm backend, from the engine's snapshot,
`myds4_view`). The lines logged while the model loads stay on the normal
screen; the view starts once the model is loaded. It writes nothing to
files. Ctrl-C, SIGTERM or the `q` key stops the server after its current
step, about a second when idle; the terminal comes back with the last 30 log
lines. `q` also works in terminals that keep Ctrl-C for themselves. Without
a keyboard (no terminal to read keys from), the view draws and the signals
still stop the server. Figures are shown as dim labels before
bright numbers; a line too long for the terminal ends after its last whole
figure, with "…". A token here is one position that a decoding step
computes: a request's next token, or a draft being verified. The view has
five parts:

- **Now:** idle, prefilling (a gauge of the layers done), or decoding. For
  each request: its session, prompt and reused tokens, generated tokens and
  speed. A sparkline shows the total speed over the last 10 s (20 s on
  terminals wider than 100 columns).
- **Experts per token**, over the last 10 s (since start when nothing
  decoded in that time). A bar and a table split each token's 240 expert
  selections (40 layers of 6) by where the expert was, per token and as a
  share: in memory, read ahead in time, read ahead but still being read
  (late), a substitute for one not in memory (`--substitute`), or read from
  the SSD on demand. The hit rate counts the model's own choices found in
  memory (the first three); "with substitutes" adds the substitutes. Layers
  the GPU routes (`--gpu-route`) are not counted. The "per token" line: the
  SSD's bytes and rate, the wait for reads, the GPU's time and busy share,
  reads ahead started and cancelled, and slots mapped for the GPU again
  ("remapped") when there are any. The "prompts" line covers all prompts
  since start: tokens, selections substituted per token, the share of the
  experts their batches used that were in memory (a batch reads each expert
  once for all its rows), and bytes read.
- **Memory:** a row each with a gauge, the main figure and its parts.
  **GPU:** the driver's resident memory (all apps) against the GPU's limit,
  split into myds4's mapped experts, its weights and the rest while the GPU
  works; while it idles, the driver releases them ("idle; resident when
  busy"). **RAM:** myds4's footprint against the Mac's memory, split into
  its experts and the rest, and the expert slots against the cache's
  current limit and its largest. **SSD:** the experts only on the SSD (their
  bytes and count), evictions and slots given back for memory. **Mac:**
  memory in use against the total (the file cache dim), free memory,
  pressure, file cache, wired memory and swap; below it the compressor, the
  kernel's minimum for the file cache and swap-outs since the view started.
- **KV cache:** the sessions and their memory, the share of the finished
  requests' prompt tokens that sessions supplied, the requests by how they
  found their session (new, continued, saved rows, forked), and the
  sessions' prefix tree. A node is a run of positions that all its sessions
  share. `●` marks where a session ends, with its number, decoding or idle
  time, saved position and memory. A node's hits count the requests whose
  reused rows reach into it.
- **Log:** the latest log lines.

### DeepSeek Harness

In the app's Models settings, add a provider with **Custom model API**:

| Field | Value |
| --- | --- |
| Provider ID | `myds4` |
| Display name | `myds4 (local)` |
| Base URL | `http://127.0.0.1:8008/v1` (the port of `--serve`) |
| API protocol | OpenAI Chat Completions |
| API key | the contents of the key file |
| Models | **Fetch available models**, or **Add model** with the ID `deepseek-v4.1-flash`, a context window of 1,048,576 and at most 65,536 output tokens |

The app stores the key in its own credential file (`~/.dsh/.credentials.yaml`).

## Checks and diagnostics

| Command | What it checks |
| --- | --- |
| `.venv/bin/python tools/reference/check.py tok CHECKPOINT` | The tokenizer against the `tokenizers` library on 17 texts. |
| `.venv/bin/python tools/reference/check.py encoding CHECKPOINT` | The server's prompts (`build/myds4-server --render`; `MYDS4_RENDER` names another renderer) of the checkpoint's `encoding/tests` cases 1, 2, 3 and 5, and of 1,200 generated conversations (tools, tool calls and results, reasoning, system messages at the start and mid-conversation, content parts, escapes, both thinking modes, every effort form) and 600 with image parts in user turns and tool results (the placeholders' places and the images' order), against `encoding/encoding.py`, byte for byte. The server's renderer differs from `encoding.py` in known places ([Prompts compared with encoding.py](server.md#prompts-compared-with-encodingpy)), so the check reports the equal counts and exits with status 1. |
| `make server-check` | The server's tests without a model: the compatibility steps, a numeric effort in the prompt, the live view's frames on a test terminal (the title first, every line within the terminal's width, the log below the sessions' tree, the quit keys), and, with `MYDS4_CHECKPOINT=CHECKPOINT`, its output path through deepseek-recipe's stream processor with the checkpoint's tokenizer. |
| `.venv/bin/python tools/reference/check.py vision CHECKPOINT IMAGE [BINARY...]` | The image's feature rows (`--vision-fd`) against `inference/image_processor.py` and `vision.py` (ViT and Aligner in BF16 on the CPU): their SNR, overall and per row. With BINARY, each of those builds instead of `build/myds4`, against one reference. Needs `pillow`. |
| `.venv/bin/python tools/reference/check.py model CHECKPOINT [CACHE]` | Every layer's streams, the routing and the logits of 20 decoded positions against DeepSeek's `inference/model.py` on the CPU (about 12 minutes, up to 55 GB of memory; `CACHE` keeps the reference's results in a file for reruns). The reference reads the checkpoint through the page cache, which then holds tens of GB of it: until those pages are reclaimed, myds4 finds less free memory, and its expert reads may come from the page cache instead of the SSD. |
| `build/myds4 -m CHECKPOINT -p TEXT --check-prefill` | The prompt as two batches (the first of odd length) against one: the final logits must be equal. The batched prefill's logits at every position against token-by-token decoding. |
| `build/myds4 -m CHECKPOINT -p TEXT --check-batch` | The prompt's next token decoded alone and beside another sequence (`model_forward_rows`), and with `--draft N` in a verify pass: the logits must be equal. With `--gpu-route`, the token decoded alone takes the GPU-routed path and the others do not, so the check compares the two paths. |
| `make gguf-decode-check [GGUF=FILE]` | IQ1_M expert decoding on the GPU against llama.cpp's `dequantize_row_iq1_m`, value by value: 8,192 random blocks and, with `GGUF`, 768 blocks of each IQ1_M expert tensor of the file (any shard of a split set). No model is loaded. |
| `make cache-ready-check` | Cache readiness and prefetch accounting with completed, pending and released dispatch groups. Runs 45 checks without a model, expert reads or GPU execution. |
| `make draft-cache-check` | Runs 49 CPU checks of stage-pool bounds, native expert admission, least-recently-used replacement, protected requests and completion-group ownership. No model, expert reads or GPU execution. |
| `make prefetch-metrics-check` | Runs 56 CPU checks of fixed-snapshot route attribution, first-use accounting, origin preservation, completed/cancelled bytes, I/O timing and unsupported GPU-route coverage. Prints bounded JSON summaries; no model or GPU execution. |
| `make cache-policy-check` | Runs 37 CPU checks of configured score aging, victim eligibility, read-cost weighting, uniform-cost equivalence, persistent history and rescaling. No model, expert reads or GPU execution. |
| `make prefetch-queue-check` | Runs 10 CPU checks of immediate prefetch admission release, including completion and cancellation while notifications are delayed. Reads `/dev/zero`; no model, SSD files or GPU execution. |
| `make memory-available-check` | Runs 7 CPU checks of changing kernel file-cache floors and query recovery. No model or GPU execution. |
| `make gpu-clock-check` | Checks the GPU clock-keeper calibration with injected host scheduling delay. Uses a tiny Metal kernel and no model weights. |
| `make build/myds4-predict-lab` | Builds the optional live feature-ablation and cache-action experiment. The ordinary executable does not link its predictor. See [experiment design](design.md#live-predictor-experiment-build). |
| `make build/predictor-lab-check build/cache-action-check`, then run those two executables | Checks causal learning, signal wiring, victim eligibility and admission ownership without loading the model or running the GPU. |
| `python3 -B tools/experimental/predictor_lab.py --checkpoint CHECKPOINT --q2 GGUF_SHARD --phase shadow --metrics` | Runs exact MXFP4/Q2 controls and real-model signal ablations serially. Shadow mode issues no additional prefetches and is not a throughput benchmark of one predictor. |
| `python3 -B tools/experimental/predictor_lab.py --checkpoint CHECKPOINT --q2 GGUF_SHARD --phase live --policy native --native-router-only --variants embedding --metrics --repeat-control` | Tests learned fetch/abstain and eviction actions using existing router candidates, with a control before and after. No extra DSpark prediction network, traces or saved learner state. |
| `make build/online-predictor-unit`, then `build/online-predictor-unit` | Runs 43 CPU checks of learner math, owned records, pass boundaries, available forecast horizons, native labels, row-count changes, reset, freeze controls and reporting windows. No model or GPU execution. |
| `make sampling-check` | Runs 19 CPU checks of softmax/nucleus sampling, ties, seed consumption, EOS and sampled draft verification. No model or GPU. |
| `.venv/bin/python -B tools/checks/online_predictor_cli_check.py` | Runs 41 CLI compatibility and input-validation checks before model loading, including sampled drafts, temperature/top-p settings and the `--reserve` range. Build `build/myds4` first; another build can be passed as the argument. |
| `make build/online-predictor-check`, then `build/online-predictor-check --rank 64 --pairs 77 --milliseconds 20` | Runs 20 math checks, including 20,000 update rows with dead or repeated features, followed by a bounded CPU microbenchmark. Uses the live module's shared math; its times exclude inference, I/O and GPU scheduling. |
| `make build/online-predictor-state-check`, then `build/online-predictor-state-check` | Runs 42 CPU checks of state round trips, identical continued learning, compatibility binding, invalid-file rejection and atomic-save failure handling. Creates bounded temporary state files and removes them. |
| `make shutdown-check` | Runs 8 CPU checks of shutdown signal handling and safe-boundary notification. No model or GPU execution. |
| `make predict-probe` | Builds the diagnostic pipe-capture executable. `tools/analysis/expert_probe.py` provides its reusable Python capture reader. Captures stay in RAM; synchronous capture changes timing and is unsuitable for speed measurements. |
| `python3 -B tools/checks/online_predictor_server_check.py --checkpoint CHECKPOINT --temperature 1 --top-p 1 --draft 2 --sampling-check` | Checks concurrent requests, prefix reuse, effective sampler settings, real draft verification, request overrides and online learning. Add `--experts GGUF_SHARD` for Q2, `--image-smoke` for an image request, and `--predictor-state PATH` to check saving. Uses one local `build/myds4-server`, bounded RAM and no logs. |
| `python3 tools/reference/check.py idle URL KEYFILE` | A running server (for example `http://127.0.0.1:8008` with its key file) keeps its experts while idle: from 3 s to 10 s after a request, its footprint and expert slots must stay as they were, with more than 11 GiB free. It sends one short request when no request has just ended. |
| `python3 tools/reference/check.py placeholder URL KEYFILE` | A running server reads the placeholder's text `<｜deepseek_image｜>` in a tool result and a user message as text: it answers the request, and the placeholder adds the tokens of its characters, not one image token. Sends two requests of one generated token. |
| `python3 tools/reference/check.py demand URL KEYFILE` | An idle server gives expert slots back when another program takes memory: the check takes memory until only 7 GiB is free, for 5 s, and the server must give back at least 1 GiB of slots meanwhile. Run it only while no other large program runs. |
| `build/myds4 -m CHECKPOINT --bench` | Kernel bandwidth and matrix throughput on the model's matrices, the index selection against a CPU selection, the CPU-GPU round trip and the GPU's speed after idling. |
| `build/myds4 -m CHECKPOINT -p PROMPT -n TOKENS --routes-fd 3 3>TRACE`, then `.venv/bin/python tools/analysis/cachesim.py TRACE [--front MS --expert MS] SLOTS...` | On the routing trace of a generation, with myds4's predictions: reads per token of seven replacement policies, including Belady's optimal one and myds4's, and decode time of read-ahead policies in a timing model, including oracles that know the experts a given number of layers ahead. A 2,000-token trace is 140 MB. |
| `MYDS4_PROFILE=1` (or `prefill`) | Runs every dispatch alone and reports GPU time per kernel for decoding (or for the prefill). The `*_3` kernels are the products of up to three matrices in one dispatch. |
| `MYDS4_PREFETCH_METRICS=1` | Add two `prefetch_metrics` JSON summaries to stderr at the end of CLI generation, token replay or perplexity scoring. Separates backbone exact-router and served unions, ready/new/pending/absent requests, first-use bytes, queue/service timing and observed eviction/rerequest counts. See [schema and scope](design.md#prefetch-attribution). |

CLI generation's `decode_experts` JSON line counts computed backbone rows,
per-row expert selections and distinct expert requests within each layer
batch. It measures expert sharing across verification rows, including rejected
rows. It excludes draft-stage work and is separate from cache hit attribution.
The [sampled draft-length measurements](performance.md#sampled-draft-length-sweep)
compare zero through five drafts at temperature 1 and top-p 1.

Prefetch attribution is off by default and keeps bounded aggregates in memory.
It excludes draft experts and counts verification work, including rejected rows.
Its pass IDs are not emitted-token IDs. Passes that actually use GPU routing
report unsupported timing coverage. The eviction and rerequest counters record
observations; they do not establish causal cache pollution or saved time.
The server's status endpoint and terminal view retain their existing counters.

## Online CPU prefetch predictor

Build the engine and set `CHECKPOINT` to the model directory.

```sh
make build/myds4
MYDS4_PREFETCH_METRICS=1 build/myds4 -m "$CHECKPOINT" \
  -p "Explain how to merge two sorted lists." --ctx 2048 --cache 30 \
  -n 128 --draft 2 --draft-cache 32 --online-predictor
```

Either predictor option enables learning; without one it stays inactive. It works with
`--substitute`, DSpark and its cache/trim options, `--gpu-route`, serving,
images, GGUF experts, profiling and diagnostic modes.
DSpark verification uses the configured temperature and top-p. Add
`--experts GGUF_SHARD` for Q2 or other external backbone
experts.

The main model runs on Metal. The learner runs on the CPU, using completed
router results to improve later prefetches. GPU routing keeps its GPU path;
the engine copies its features before the next layer reuses them.

The learner has 77 independent rank-64 heads for the one- and two-layer
forecasts. Each head uses a fixed feature projection and learns an output
correction by recursive least squares. It trains during small forward and
verification passes, including rejected draft rows. Batched prefill and the
initial single-token prompt do not train it; later token-by-token prompt and
diagnostic passes can. Source features
and native target labels are copied before scratch reuse or substitution.
Updates run on the model thread after expert command buffers are submitted.
Multirow substitution computes only the next-layer forecast, so that path
trains and uses only the available horizon.

Learned corrections belong to the engine and remain in memory across requests,
KV-state resets and session switches. The server's single model worker shares
one learner across its requests; each row trains against its own native router
result. Without a state path, starting the engine initializes a new learner
and all learned state remains in RAM. Tensor state uses about 23 MiB and
prediction records about 0.8 MiB; working buffers and library memory are additional.

The learner changes prefetches, not checkpoint weights or the native routing
rule. Exact routing keeps the same output. With `--substitute`, changed cache
residency can change substituted experts and output, as with other cache
settings. This live learner runs on the CPU.

### Persistent predictor state

Set `PREDICTOR_STATE` to a file in an existing writable directory. Use a
separate file for each model and quantization. `--predictor-state` enables
the learner, so the standalone enable flag is unnecessary:

```sh
build/myds4 -m "$CHECKPOINT" -p "Explain merge sort." \
  --draft 2 --draft-cache 32 --predictor-state "$PREDICTOR_STATE"

build/myds4 -m "$CHECKPOINT" -p "Write a merge sort implementation." \
  --draft 2 --draft-cache 32 --predictor-state "$PREDICTOR_STATE"
```

The first run starts a new learner if the file is absent. A later run with
the same path loads the learned corrections and continues fitting. With Q2,
add `--experts "$Q2_SHARD"` and select a separate state path. A file with an
incompatible router/quantization binding, invalid format or corrupt contents
causes an error; the engine does not silently replace it with a new learner.

The engine saves changed state on normal CLI completion and on a handled
SIGINT or SIGTERM, including server shutdown. It saves when the running
computation reaches a safe boundary. The server shares and retains one learner
across requests, but writes neither periodically nor after each request.
Crashes and forced termination such as SIGKILL cannot trigger a save.

The file contains the fixed projection, learned correction weights, covariance,
router biases, active-head flags, cumulative head update counts and completed
learning-row progress. Loading restores enough state to continue fitting;
pending prediction records, cache contents, attention state and process timing
counters are not stored. The main model checkpoint is unchanged.

Each state file is **24,090,040 bytes** (about 24.09 MB). Saving writes a
private temporary sibling, flushes it and atomically replaces the destination.
Replacing an existing file temporarily requires about **48.18 MB** for the
old and new copies together. A save failure leaves the previous destination
intact and is reported as an error. The binding checks router parameters and
expert/head formats; it is not a digest of every model weight. Paths and file
modification times do not determine compatibility.

### Predictor measurements and learning controls

The `online_predictor` JSON report includes raw top-three candidate precision,
updates, skips and CPU component times. Raw precision counts resident experts
and repeated candidates across rows. Use the separate native
`prefetch_metrics` report for newly admitted reads, readiness and first-use
outcomes; raw candidate precision is not a cache-hit or speedup measurement.

Two optional environment controls apply to the learner:

| Variable | Effect |
| --- | --- |
| `MYDS4_ONLINE_FREEZE_ROWS=N` | Unset or 0 keeps learning. Values 1–1,048,576 freeze updates after at least N completed eligible rows, including progress restored from a state file. A pass that crosses the threshold finishes learning all its rows; freezing starts on the next pass. |
| `MYDS4_ONLINE_WINDOW_PASSES=N` | Unset or 0 disables window reports. Values 1–4,096 emit an `online_predictor_window` JSON record every N completed eligible passes, plus a final partial window when completed passes remain. |

Rows include rejected verification work and are not emitted-token counts.
While frozen, predictions and raw-precision scoring continue; learned output
weights and covariance stop updating. Only an explicit learner reset clears
the frozen state and row threshold count; KV resets and new requests retain
them. Window reporting uses bounded aggregates and adds
stderr output, which should be considered when comparing elapsed times.
See [Design](design.md#online-cpu-predictor) for features,
ownership and report semantics. Performance results are documented separately.

## Limitations

- Long prompts prefill slowly: the indexer's work grows with the square of
  the context, and a prompt over 16,384 tokens runs as several passes that
  each read the experts again. A 41,000-token prompt took about 80 s with
  the GGUF experts; a 1,048,576-token prompt would take hours. The
  server's sessions spare a conversation's earlier tokens this work.
- The server decodes up to 6 requests together and prefills one at a time;
  up to 16 more wait (more are refused with 429). Drafts (`--draft`) serve
  a request only while it decodes alone.
- The server gives one choice per request and ignores `stop`,
  `tool_choice`, `response_format` and `logprobs`.
- Draft acceptance, expert reads and
  cache allocation determine its throughput; see the recorded
  [performance results](performance.md#speculative-decoding).
- `--experts` decodes the GGUF codes as llama.cpp does but keeps the
  checkpoint's FP8 rounding of the experts' activations, so its output is
  not llama.cpp's.
- `--substitute` makes the output depend on the timing of the SSD's reads.
- Batched prefill and decoding of the same prompt agree on the greedy token
  at 97 to 100% of positions of a 550-token text, and decoding agrees with
  the reference at 19 of 20 positions (see [Design](design.md#numerics)).
