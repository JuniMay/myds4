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
| `memory` | Optional upper limit of model memory in GiB: all loaded weights, experts, vision and model workspaces; KV is separate. Default 0: no limit; the memory that other programs leave decides (see [Memory](#memory)). |
| `reserve` | Memory kept available to other programs, in GiB; default 10, range 2–1024. |
| `kv_cache` | One budget for all models' attention states and snapshots, in GiB; default 8. Experts use the part that KV does not occupy. |
| `kv_disk` | Room on the SSD for saved states, in GiB; default 32, 0 turns saving and reading off (see [Saved states](#saved-states)). |
| `kv_dir` | The saved states' folder; default `~/.cache/myds4/kv` (from `HOME`). |
| `kv_disk_tokens` | A state is saved only when it holds at least this many tokens that no other kept state holds; default 8,192. |
| `idle_unload` | Seconds a model may go without requests before its weights and experts unload; default 600, range 0–1,000,000, 0 never. A model's own `idle_unload` overrides it (see [Memory](#memory)). |
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
| `idle_unload` | Seconds this model may go without requests before its weights and experts unload, 0 never; default: the server-wide `idle_unload`. |

Qwen's generation configuration supplies sampling defaults when the entry
omits them. Set temperature and top-p explicitly to 1 when that is the
wanted policy. A batch limit is a concurrency cap, not a promise that all
requests fit at their maximum context length.

Flash-Next uses the same model entry and Qwen chat format. Its supported
checkpoint types are `qwen4_exp` and `qwen4_exp_text`. It supports images
when the checkpoint contains the vision encoder. Set `draft: 0` and
`lookup: "off"`; speculative verification is not implemented for this
architecture. Its routed experts are read into its expert store's slots as
passes need them, as DeepSeek's are. The store stays within the global
`memory` budget, keeps the experts used most, and gives slots back when
other programs need the memory. Each slot takes about 3.1 MB: an expert's
weights, its compressed scales as read, and the scales decoded. While
decoding, it reads each layer's predicted experts one layer ahead.

Nemotron 3.5 Transcribe uses checkpoint type `nemo_speechlm`. Its main model
and speech encoder load on demand. Its routed experts are read into the
expert store's slots as passes need them, as DeepSeek's are. The store stays
within the global `memory` budget, keeps the experts used most, and gives
slots back when other programs need the memory. The DFlash2 checkpoint is
BF16 and loads only with `draft` enabled. Seven proposals are a useful starting point;
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

The server's memory is what other programs leave: the installed memory less
`reserve` less what other programs hold at that moment. It follows them; it
is not a fixed number. Available memory means the free pages and the file
cache beyond the share that macOS keeps (`vm.vm_page_filecache_min`, about
24 GiB with 128 GiB). While its file cache is below that share, macOS
compresses and swaps programs' memory rather than drop cached files, so that
share never counts as available.

While the server idles, the GPU driver takes its model memory out of wired
memory within a few seconds, so macOS could swap it under memory pressure.
The rules below keep the server from causing that pressure (see
[Design](design.md)).

- **Additions keep the reserve.** Before model memory or KV is added, the
  engine checks the available memory: an addition may not take it below
  `reserve`. When it would, idle model memory goes back first. The expert
  cache grows only while one more expert slot would still fit above
  `reserve`, so the small buffers a pass takes and frees fit without taking
  an expert back. The log names take-backs of 1 GiB or more, and refusals;
  smaller take-backs are counted, not logged.
- **What goes back, and in what order.**
  - The model memory holds variable-sized entries. A dense model's weights
    form one entry, and MoE experts are separate entries that go back before
    their dense backbone.
  - The least recently used entries go back first.
  - The weights of a model that is serving are never taken: active requests
    pin the weights they need.
  - Unused weight groups can be unloaded and later reloaded while their KV
    state remains in the separate pool.
- **A load that does not fit is refused.** A model load that still does not
  fit is refused; the log gives the numbers, the request fails, and nothing
  is swapped.
- **When other programs grow.** The expert cache gives back the slots that
  bring the available memory back to `reserve`; under memory pressure or
  swapping it gives back a sixteenth of its slots at a time. Once a second,
  the server also takes back idle model memory, then idle KV, until the
  available memory is back at `reserve`.
- **An idle model unloads.** A model that has gone `idle_unload` seconds
  (default 600) without requests, with none waiting, gives back its expert
  cache and its weights; the log names it and the memory given back. Its
  KV states stay, and its next request loads it again: the weights are read
  from the SSD while every model's requests wait. When every model has gone
  its time, all their weights are back with other programs; their KV stays
  within `kv_cache`. A yield's interval has its own limits, so nothing
  unloads for idleness while it lasts.
- **KV comes first, within `kv_cache`.** The experts may use the space that
  KV does not occupy. When KV grows, idle experts give that space back
  before it is allocated. Experts never take the space that KV occupies.
- **`memory` is an upper limit, not a reservation.** It counts managed model
  allocations, including Metal buffers and model-owned host buffers. Without
  it, the available memory alone decides. Request text, tokenizer metadata,
  framework and driver overhead lie outside it, but they use available
  memory like anything else.
- **Reading the available memory.** macOS counts memory as in use only once
  its pages are written, and counts released memory a little late. The
  engine charges a buffer when it allocates it. So it adjusts its last
  reading by its own allocations and releases, and it counts expert slots
  that their reads have not written yet as in use. It reads again at each
  model load and at most once a second otherwise, but not within 1.5 s after
  it released memory unless the last reading is 5 s old.

The Memory tab and `/v1/models/{id}` show the current limit: the model memory
held plus the available memory above `reserve`, within `memory`.

Quitting (Ctrl-C, SIGTERM, or `q` in the terminal interface) stops a model's
load between weights; the load frees what it took, and requests waiting for
it end as a shutdown. Weights that a pass loads (routed experts, vision
weights) complete with their pass first. A DeepSeek prompt's prefill stops
before its next batch of up to 8,192 rows, and its request ends as a shutdown;
other models' prefills stop between the calls the server makes for them.
Then the server saves its states to the SSD (see [Saved states](#saved-states));
a second Ctrl-C, SIGTERM or `q` ends the saving, and the states not saved yet
are lost.

Prefill uses chunks that fit available model workspace. A long prompt does
not require its entire working buffer at once. If the smallest supported
chunk and the required model weights cannot fit, the request reports a
memory error. The context and KV limits still apply.
In-memory compression is allowed. Benchmarks reject new swap-outs.

The KV pool evicts spare snapshots and then idle states across all models
in least-recently-used order. Prefix matching never crosses model
boundaries. Admission accounts for state growth and promised snapshots.
An active generation may exceed the budget when only active states remain;
this is logged. New requests must fit before they are admitted. A prompt
whose cheapest resume would keep a state that does not fit beside its own
takes the next cheapest, down to a cold start in that state's place; only a
prompt that does not fit even then is refused.

Context capacity is allocated as it grows. With MTP, a full 256K Qwen
state takes about 17.3 GiB before snapshots. Configuring eight concurrent
requests does not reserve or guarantee memory for eight such states.

### Saved states

A state that leaves memory is saved to the SSD, in `kv_dir`, within
`kv_disk`; a later prompt that starts like it reads it back instead of
prefilling that part again.

- **When a state is saved.** When the KV pool evicts it, and at quit for
  every state still in memory, the most recently used first. A state is
  saved only when it holds at least `kv_disk_tokens` (8,192) tokens that no
  other state kept in memory and no saved state holds; an older turn of a
  conversation whose newer turn is kept is not written. A state whose
  prefill or decoding pass the stop cut short is not saved.
- **What is written.** The state's rows, its newest prompt-end snapshot
  (with the logits of the prompt's next token), its newest branch
  snapshot, and its end: the rolling rows where the conversation stands, so
  that the next turn continues there also for Qwen, Flash-Next and Nemotron,
  which keep no rewind margin. A state that was itself read from a saved
  state writes only its rows from where it parted from it, as one more part
  of that saved state, when the saved state holds fewer than
  `kv_disk_tokens` tokens after that point; otherwise it is written as a new
  saved state, and the old one keeps its own tokens. A saved state whose
  replaced rows would outnumber its own is written whole once more.
- **When a state is read.** A prompt resumes from a saved state as from a
  state in memory: at the longest prefix they share, from a snapshot there
  or (DeepSeek) with a replay. A state in memory, and a cold prefill, go
  first when they cost as much. The log names each read (`req N: P
  positions of saved state S read in T s`). A saved state that is damaged
  or missing a file is deleted, and the request is planned again without
  it; one whose files cannot be opened at the moment (their permissions,
  too many open files) stays, and only that request does without it.
- **Which states are read.** A saved state belongs to its model's checkpoint
  and settings: the `config.json`, `.safetensors` and `.gguf` files of the
  checkpoint folder (by name, size and their first and last 64 KiB), the
  `experts` GGUF set, the DFlash2 checkpoint, and `draft`, `lookup`, `ced`,
  `prefill_prune` and `substitute`. It is read after a rebuild of the server
  as long as these are unchanged; a rebuild whose kernels compute
  differently then continues it from rows the old build computed. Saved
  states of another checkpoint or other settings stay until their room is
  needed, and go first then.
- **Room.** The least recently used saved states go first to keep within
  `kv_disk`; a state is saved in the room of states used before it only,
  so a quit does not replace newer saved states with older ones; they are
  deleted once the new one is written, so a save that fails takes no room. A
  save that would leave less than 10 GiB free on the volume is skipped. The log
  names every save and skip with its tokens, GiB and seconds; the terminal
  interface's KV cache tab shows the saved states and the GiB written since
  the start.
- **SSD writes.** A state's rows take per position: DeepSeek 3.1 KiB, Qwen
  27B 68 KiB, Flash-Next 25.5 KiB, Nemotron 12 KiB with DFlash2; its end
  snapshot about 5.6 MiB (DeepSeek), 157 MB (Qwen 27B), 120 MB (Flash-Next)
  or 50 MB (Nemotron). A conversation's first save writes all its rows; its
  later saves write the rows added since and its snapshots. A quit writes
  at most what `kv_cache` holds. Files are written past the page cache
  (`F_NOCACHE`), on the engine thread between passes: decoding pauses while
  a state is written.
- **Measured speeds.** On an M5 Max with 128 GB, on AC power, one request at
  a time through the server with `--batch 1`:
  - Qwen 27B (`--memory 40 --draft 0`): a 14,533-token state wrote 1.18 GiB
    in 0.25 s (4.7 GiB/s) and read back 1.03 GiB in 0.12 s (8.6 GiB/s). Its
    prompt was answered 0.14 s after it arrived, against 22.9 s cold.
  - DeepSeek (UD-IQ1_M experts, `--memory 56`): a 14,255-token state wrote
    0.05 GiB in 0.02 s and read back in 0.01 s. Its prompt was answered in
    0.10 s, against 24.3 s cold.
  - Saved states cost decoding nothing while none is written or read. DeepSeek
    (the same settings, a 27-token prompt, 256 tokens) decoded at 22.3 and
    22.5 tokens/s with them on, and 22.2 and 22.5 with `kv_disk: 0`, with the
    same text.
- **The folder.** Each saved state is an index (`ID.kv`), its parts
  (`ID-N.rows`) and its snapshots (`ID-V-POS.snap`); each file is written
  under a temporary name, synced and renamed, the index last. At start the
  server reads the indexes and deletes damaged ones and files of these
  names that no index names; it deletes no other files, and keeps the
  indexes of another version of the server, with their files. One server uses a
  folder at a time (a lock on its `lock` file): another starts without saved
  states and logs why. The folder is made with mode 0700, the files 0600.
- **Privacy.** Saved states hold the token ids of the prompts and replies,
  so the conversations stay on the SSD as long as their saved states do. To
  remove them, stop the server and delete the folder; `kv_disk: 0` saves and
  reads nothing.

## API

| Endpoint | Meaning |
| --- | --- |
| `GET /health` | Server health. Does not load model weights. |
| `GET /v1/models` | Registered model IDs and context limits. Does not load model weights. |
| `GET /v1/models/{id}` | Model metadata and status, including `loaded`, memory and queue information, and `kv_disk`: the saved states' folder, count, GiB and budget and this run's loads, saves and GiB written (null without saved states). |
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

Keys 1–6 select Overview, KV cache, Experts, Last minute, Log and Memory.
With several models, the Overview, KV cache, Experts and Last minute tabs
show one model at a time, and `n` selects the next. The KV cache tab lists
every model with its sessions and their KV at its top; a click there shows
that model's sessions. Below it a line counts the requests by how they
resumed (new, continued, saved, forked, loaded from the SSD) and, with saved
states, another shows them: how many, their GiB of `kv_disk`, this run's
reads and saves, the GiB written, and the folder. The Log and Memory tabs
cover every model.

The Experts tab maps a decoding request's experts in each layer with routed
experts, for DeepSeek, Flash-Next and Transcribe alike: each expert of its
last sampled tokens in the color of where it was when last chosen (in
memory, read ahead, late, substituted or from the SSD), and at the right
each layer's experts of those tokens by where each one was. A cell holds
two experts where the terminal is wide enough, otherwise four or more.

The Memory tab draws the Mac's installed memory as a treemap: each
rectangle's area follows its bytes, and a character cell counts as twice
as tall as wide. Each model in memory is a rectangle under its name, split
into its weights, experts, workspace and allocated KV. Beside the models
are the server's other memory, other programs (the system's memory among
them), the file cache and free memory, with the `reserve` share of free
memory marked. A part too small to show its size is named on its model's
first row when it fits there, else below the map. Above the map are the
models' memory and its current limit, how many models are in memory, the
shared KV budget, the reserve, and the available memory.

Arrow keys, Page Up/Down and the mouse navigate the selected panel.
`q` or Ctrl-C shuts down the server and restores the terminal. Use
`--no-tui` for ordinary logs.

While the interface is up, only it draws on the terminal. Everything else
the server writes to stderr, such as the engine's lines about a model's
load, appears in the Log tab with its time. On exit the normal screen shows
the last 30 log lines.

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
