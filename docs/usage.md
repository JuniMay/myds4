# Usage

## Server configuration

`build/myds4-server` reads `~/.config/myds4/server.json`, using `HOME`.
`--config FILE` chooses another file. `-m DIR` instead serves one model
from command-line settings without reading the default configuration.
`--render FILE` is a diagnostic command and also bypasses the default file.

Use full paths in configuration files when the server may be started from
different directories. Relative paths use the working directory; JSON
strings do not expand `~`.

The root contains server settings and a nonempty `models` list:

| Setting | Meaning |
| --- | --- |
| `port` | HTTP listening port; `--serve PORT` overrides it. |
| `host` | IPv4 address or `localhost`; default `127.0.0.1`. |
| `api_key_file` | File containing a 16–256 character API key. Without a file, use `MYDS4_API_KEY`. |
| `tui` | Show the terminal interface; default false. `--tui` enables it and `--no-tui` disables it. |
| `tui_theme` | `auto`, `light` or `dark`; default `auto`. |
| `memory` | Shared model-memory ceiling in GiB: all loaded weights, experts, vision and model workspaces. KV is separate. Default 0 uses physical RAM minus `reserve` and `kv_cache`. |
| `reserve` | One process-wide target for available system memory, in GiB; default 10, range 2–1024. |
| `kv_cache` | One budget for all models' attention states and snapshots, in GiB; default 8. |
| `models` | Each model's checkpoint and inference settings. |

Explicit server flags override file settings regardless of argument order.
The model-specific settings stay in the file when using `--config`.
Configuration changes require a restart.

The server speaks plain HTTP. Listening beyond the local machine requires
a trusted network or a separately configured TLS proxy.

## Model settings

Every model entry has a `model` checkpoint directory. The engine identifies
its architecture from `config.json`. It currently supports one checkpoint
of each architecture per process.

| Setting | Meaning |
| --- | --- |
| `id` | Name clients send in `model`. Defaults to `deepseek-v4.1-flash` or the Qwen directory name in lower case. |
| `ctx` | Maximum context positions. The server defaults to the model's limit: 1,048,576 for DeepSeek Flash and 262,144 for the supported Qwen checkpoint. |
| `batch` | Maximum admitted requests for this model; default 4, capped at 6 for DeepSeek and 8 for Qwen. |
| `temperature`, `top_p`, `top_k`, `seed` | Defaults for requests that omit them or send null. Request values override them. `top_k: 0` uses the full vocabulary. |
| `think` | Default reasoning effort, 0 for off and 1–100 for enabled reasoning. |
| `draft` | Speculative proposals: DeepSeek DSpark 0–5; Qwen3.8-27B chained MTP 0–5; Nemotron DFlash2 0–15. Flash-Next requires 0. |
| `draft_model` | For Nemotron with `draft` enabled, the separate DFlash2 checkpoint directory. |
| `lookup` | `off` or 1–15 context-copy proposals, limited by the model's verification capacity. |

Qwen's generation configuration supplies sampling defaults when the entry
omits them. Set temperature and top-p explicitly to 1 when that is the
wanted policy. A batch limit is a concurrency cap, not a promise that all
requests fit at their maximum context length.

Flash-Next uses the same model entry and Qwen chat format. Its supported
checkpoint types are `qwen4_exp` and `qwen4_exp_text`. It supports images
when the checkpoint contains the vision encoder. Set `draft: 0` and
`lookup: "off"`; speculative verification is not implemented for this
architecture. Its routed experts share the global `memory` budget.

Nemotron 3.5 Transcribe uses checkpoint type `nemo_speechlm`. Its main model
and speech encoder load on demand. The DFlash2 checkpoint is BF16 and loads
only with `draft` enabled. Seven proposals are a useful starting point;
the best setting depends on acceptance and transcript length. Without
explicit sampling settings, Transcribe uses temperature 0 and reasoning
off. API requests can override those defaults.

DeepSeek-specific settings belong only to its entry:

| Setting | Meaning |
| --- | --- |
| `experts` | GGUF file or any shard of a set. Omit for the checkpoint's native MXFP4 experts and output head. |
| `cache_half_life` | Retention score half-life in forward passes; default 50. |
| `cache_read_cost` | Include aligned read cost in retention priority. |
| `substitute` | Approximate expert replacement within this router-score gap; 0 disables it. |
| `draft_cache` | Expert slots per DSpark stage, 15–128. Default 0 preloads all 128; these pools count against `cache`. |
| `ced` | `off` or a replay length of at least 128 tokens. Approximate decoder prefill; it can change output. |
| `prefill_prune` | Skip decoder batches outside required dependencies. Requires CED and substitution off. |
| `runahead` | Advance ready concurrent requests while others wait for experts. Requires native drafts, substitution and online prediction off. |
| `pipeline` | Experimental concurrent MXFP4 execution across layers. Requires GGUF experts, runahead, native drafts, substitution and online prediction off. |
| `online_predictor` | Enable the online CPU prefetch corrector. It changes read scheduling, not routed experts. |
| `predictor_state` | Load and save the online predictor at this path; also enables it. |

Substitution and CED deliberately approximate the model. Prefetching does
not change selected experts. Native drafts and context-copy proposals are
verified using the requested target sampler. They run when one request of
the model is decoding; ordinary batching or the selected concurrent
scheduler handles multiple requests. Pipeline therefore does not replace
`batch`. Pipeline can coexist with context lookup for a lone request.

## Memory

Available memory includes free and reclaimable file-backed pages. The
shared reserve is a headroom target. The model-memory pool holds variable-sized
entries. A dense model's weights form one entry. MoE experts are separate
entries that must be evicted before their dense backbone. Active requests
pin the weights they need. Unused weight groups can be unloaded and later
reloaded while their KV state remains in the separate pool.

The `memory` ceiling counts managed model allocations, including Metal
buffers and model-owned host buffers. Request text, tokenizer metadata,
framework and driver overhead also contribute to the process footprint.
`kv_cache` controls the shared KV pool independently. Inspect total process
memory when choosing the two budgets together.

Prefill uses chunks that fit available model workspace. A long prompt does
not require its entire working buffer at once. If the smallest supported
chunk and the required model weights cannot fit, the request reports a
memory error. The context and KV limits still apply.
In-memory compression is allowed. Benchmarks reject new swap-outs.

The KV pool evicts spare snapshots and then idle states across all models
in least-recently-used order. Prefix matching never crosses model
boundaries. Admission accounts for state growth and promised snapshots.
An active generation may exceed the budget when only active states remain;
this is logged. New requests must fit before they are admitted.

Context capacity is allocated as it grows. With MTP, a full 256K Qwen
state takes about 17.3 GiB before snapshots. Configuring eight concurrent
requests does not reserve or guarantee memory for eight such states.

## API

| Endpoint | Meaning |
| --- | --- |
| `GET /health` | Server health. Does not load model weights. |
| `GET /v1/models` | Registered model IDs and context limits. Does not load model weights. |
| `GET /v1/models/{id}` | Model metadata and status, including `loaded`, memory and queue information. |
| `POST /v1/chat/completions` | Generate with the model named in the request. Its first request loads its weights. |
| `GET /v1/control` | Process-wide execution state, effective budgets and retained model/KV bytes. |
| `POST /v1/control/yield` | Pause inference and reclaim to temporary budgets. |
| `POST /v1/control/resume` | Restore configured budgets and scheduling for the matching operation. |

Send the key as `Authorization: Bearer KEY` or `x-api-key: KEY`.
With several models, requests must name one. With one model, omitted or
unknown names route to that model for client compatibility.

Chat Completions accepts message history, tools, sampling settings,
`max_tokens` or `max_completion_tokens`, `stream`, and reasoning controls.
Tool calls are returned as API `tool_calls`; clients do not parse the
model's internal XML or special-token syntax.

Images use content parts of type `image_url`, with a base64 `data:` URL.
The server does not fetch remote image URLs or read client-supplied local
paths. DeepSeek and Qwen use their own image preprocessing and position
rules. Video requests are not implemented.

Transcribe accepts `input_audio` parts containing `data` (base64) and
`format`, or `audio_url` parts containing a base64 data URL. It decodes
formats supported by macOS, including WAV, FLAC and MP3, and resamples to
mono 16 kHz. Recordings may be up to one hour, subject to the context and
memory budgets. The encoder uses overlapping windows and carries speaker
history between them. Generated speaker tags are text; separate CTC word
alignment and timestamp metadata are not implemented. Remote audio URLs
and client-supplied file paths are not fetched by the server.
The HTTP request body, including base64 data, is limited to 32 MiB. Use a
compressed audio container or the CLI for recordings that exceed that limit.

### Yield and resume

A tool wrapper can yield the serving process before running resource-heavy
commands. These endpoints use the same API key as inference. They apply to
every model hosted by this process. Other programs run independently.

Send a new, caller-generated `operation_id` for each interval:

```json
{"operation_id":"build-001","memory_gib":16,"kv_cache_gib":2}
```

`POST /v1/control/yield` accepts these fields only:

| Field | Meaning |
| --- | --- |
| `operation_id` | Required nonblank string, 1–128 UTF-8 bytes, without control characters. |
| `memory_gib` | Optional model-memory allowance the server may retain. |
| `kv_cache_gib` | Optional separate KV allowance the server may retain. |

An omitted allowance keeps its configured limit. A positive allowance is a
strict cap, at most the configured limit. Zero requests the minimum
retainable memory: the server evicts every reclaimable entry, preserves
active-request state and required work buffers, and sets the temporary
limit to that remaining requirement. Zero therefore need not produce zero
retained bytes. KV reservations for active requests can make the effective
KV limit larger than its currently allocated bytes.

The server first reports `yielding`. At the next engine boundary it drains
both GPU queues and pending expert reads, releases weight pins, and
reclaims memory. Only then does it report `yielded`. A long prefill may
delay that boundary. New inference requests remain queued within the
normal queue limits. The health and control endpoints remain available.

A successful status has this form; sizes are integer bytes:

```json
{
  "state":"yielded",
  "operation_id":"build-001",
  "configured_limits":{"model_bytes":68719476736,"kv_bytes":8589934592},
  "effective_limits":{"model_bytes":17179869184,"kv_bytes":2147483648},
  "retained":{"model_bytes":1073741824,"kv_bytes":536870912},
  "error":null
}
```

`retained` is the worker's latest measurement of managed model and KV
allocations. It excludes HTTP, tokenizer, framework and driver overhead.
Transitional states can show the last completed measurement; `yielded`
confirms that drain and reclamation have finished. The temporary limits
stay in force until an explicit resume. Disconnects and elapsed time do
not resume inference.

`POST /v1/control/resume` takes only:

```json
{"operation_id":"build-001"}
```

It reports `resuming` while restoring the configured limits and reloading
backbones needed by paused requests. Idle weights and expert caches load
later on demand. `running` means normal scheduling has resumed. KV state
and per-request sampling state remain attached to paused requests.

Both POST endpoints return 202 when a transition is pending. Poll
`GET /v1/control`, or repeat the same POST with the same fields. Repeating
YIELD returns 200 once that operation is yielded. Repeating RESUME returns
200 once the operation has ended, with `resumed_operation_id` identifying
it. That acknowledgement leaves any newer owner's state unchanged.

Retries never create another interval. An active operation rejects another
owner, changed limits for its ID, or an unknown resume ID with HTTP 409.
A completed ID cannot start another yield; that retry returns
`operation_completed`. Operation history lasts for this server process.
A restart discards it, so a wrapper must reconfirm yielding after a restart.

Invalid JSON or limits return 400; missing or invalid authentication returns
401. A positive cap that cannot fit protected buffers fails asynchronously:
the server returns to `running` with configured limits and an `error` object
containing the operation ID, code `limits_unreachable`, and required bytes
in its message. Retrying that YIELD returns 409 with `yield_failed`.
Reclamation may already have evicted idle entries before this failure.

Wrappers must wait for their own `yielded` state before starting commands.
They own command timeouts and cancellation, including parallel children
and background jobs. Wait for every child to stop before resuming. Use a
separate cleanup deadline for resume; an aborted command's cancellation
signal must not cancel cleanup. Supply the whole work plan before yielding,
because inference on this server remains paused during the interval.
Development wrappers may serialize their outer intervals with their own
lock. The server does not acquire or inspect that lock.

## Terminal interface

`n` selects the next model. Keys 1–6 select Overview, KV cache, Experts,
Last minute, Log and Memory. The Memory tab shows all models' tracked
weights, experts and allocated KV, with the shared budgets shown once.
Process footprint also includes runtime allocations outside the tiles.

Arrow keys, Page Up/Down and the mouse navigate the selected panel.
`q` or Ctrl-C shuts down the server and restores the terminal. Use
`--no-tui` for ordinary logs.

## CLI and diagnostics

`build/myds4 -m DIR` handles a single request. Give `-p TEXT`, `-f FILE`,
or `--tokens IDS`. `--raw` bypasses chat formatting; `--rendered` accepts a
server-rendered prompt. The CLI defaults to 128 generated tokens, a 16K
context and greedy sampling. Use `-t 1 --top-p 1` for sampled generation.

The CLI accepts the relevant model options above, with hyphens instead of
underscores. `--image FILE` adds an image; `--audio FILE` adds a recording
for Transcribe. `--tokenize`, `--check-prefill`,
`--check-batch`, `--perplexity`, `--logits-fd`, `--routes-fd`, `--features-fd`
and `--bench` are developer diagnostics; see [Testing](testing.md).
