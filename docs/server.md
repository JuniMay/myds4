# The server

`server/` is myds4's HTTP server, written in Rust. The C and Metal code of
`src/` runs the models behind a C ABI,
[`src/engine/engine.h`](../src/engine/engine.h), and the Rust code handles
HTTP, each model family's chat format, scheduling and the live view. It
serves `POST /v1/chat/completions`, streamed or as one response,
`GET /v1/models`, `GET /v1/models/{id}` and `GET /health`; its options are in
[Usage](usage.md#server). It serves one model, or several in one process,
each request going to the model it names ([Models](#models)): DeepSeek V4.1
Flash, whose requests take DeepSeek's
[`deepseek-recipe`](https://github.com/deepseek-ai/deepseek-recipe) crates
(version 0.2.0, MIT License), and Qwen3.8 models ([Qwen3.8 models](qwen.md)).

## Parts

| Part | Language | Files |
| --- | --- | --- |
| HTTP: endpoints, the API key, routing by model, the options | Rust | `server/src/main.rs` |
| The models and their settings: `-m` and its options, or `--config` | Rust | `server/src/config.rs` |
| Connections: the accept loop, the limits and timeouts | Rust | `server/src/http.rs` |
| The chat formats: the errors both share (`format/mod.rs`); DeepSeek's compatibility steps, recipe's conversion and V4.1 rendering, images from `data:` URLs, recipe's stream processor (`format/deepseek.rs`); Qwen's conversion, chat template, reply protocol and responses (`format/qwen.rs`, `qwen_template.rs`, `qwen_protocol.rs`) | Rust | `server/src/format/` |
| The live view (`--tui`): the terminal, the keys and the mouse, the statistics windows and the tabs, on ratatui; in `view/`: the chrome, the colors, the last minute's samples and each tab (Overview, KV cache, Experts, Last minute) | Rust | `server/src/view.rs`, `server/src/view/` |
| The worker: the engine thread, every model's queue, scheduling and batching, prefills in calls, drafts, events, cancellation, each model's status and the live view's snapshot | Rust | `server/src/worker.rs` |
| The KV prefix cache: which attention states and snapshots of a model to keep, and how a prompt resumes from them | Rust | `server/src/kv.rs` |
| The engine's ABI in Rust: declarations and thin wrappers | Rust | `server/src/engine.rs` |
| The engine: the process (log, stop), and for each model its loading, states and snapshots, prefill (with a replay where the model has one, and a capture position), decoding, drafts, sampling, context-copy drafts, images, statistics and its tokenizer | C | `src/engine/engine.c` |
| The models behind one interface, the expert cache, the vision encoder, the tokenizer, signals | C, Metal | `src/model/`, `src/engine/io.c`, `src/engine/gpu.c`, `src/engine/vision.c`, `src/engine/tok.c`, `src/engine/util.c`, `src/kernels/` |

## Threads

- **The engine thread** (`worker::start` creates it with a 16 MiB stack;
  `myds4_init` gives it the user-interactive QoS class). It loads every
  model and then runs the worker loop ([Models](#models)). Every engine call
  that runs a model happens on this thread, so every model's passes go
  through one GPU command queue: `gpu.c` keeps its command buffers and an
  autorelease pool per thread, and each model family's work buffers are its
  own. While no model runs a request, it lets each model give memory back
  every second (`myds4_idle`: DeepSeek's expert cache resizes; a Qwen model's
  buffers leave the GPU's residency set after 60 s without passes).
- **The live view's thread** (`--tui`): twice a second it takes the worker's
  snapshot of the model it shows (`worker::view`) and keeps a sample of it
  for the last minute; it draws the tab shown on stderr's alternate screen,
  and again at once after a key or a click that changes it. A KV node's text
  is a question to the worker (`worker::ask_text`), which answers between its
  steps with the tokens the model's KV prefix cache holds
  (`worker::text_answer`); the view turns them into text with the model's
  tokenizer. With each snapshot the view asks for the decoding requests'
  routed rows (`worker::ask_routes`); the worker copies them
  (`myds4_routes`) after its next decoding pass of a model with routed
  experts, and the next snapshot takes them.
- **Tokio's threads** run HTTP: reading requests, converting, rendering and
  tokenizing them with their model's tokenizer (the engine's tokenizers take
  any thread), submitting them to the worker, and turning the worker's
  events into response chunks.

Signals belong to the engine: SIGINT and SIGTERM request the stop, and the
live view's `q` (or Ctrl-C, which the terminal's raw mode delivers as a key)
requests it through `worker::request_stop`, which also wakes the worker.
The server checks `myds4_stopping()` several times a second, stops accepting
connections, waits for the open responses to end (the worker ends every
request), and calls `worker::stop`, which waits for the engine thread to
let each model save what it keeps (`--predictor-state`, `myds4_finish`) and
end. The live view then gives the terminal back with the last log lines. It
also does so before a fatal error's message, the engine's (`on_die`) or the
server's, and after a panic; a panic on the engine thread ends the process,
since its requests could not end otherwise.

## Models

`-m DIR` serves one model, its options on the command line; `--config FILE`
serves the models a JSON file lists, each with its own options under the same
names without the dashes (`-` as `_`):

```json
{"models": [
  {"model": "/path/to/DeepSeek-V4.1-Flash-1017", "experts": "/path/to/UD-IQ1_M-00001-of-00004.gguf",
   "cache": 72, "ced": 128, "kv_cache": 16},
  {"model": "/path/to/Qwen3.8-27B-NVFP4", "draft": 1, "ctx": 65536, "kv_cache": 8}
]}
```

A checkpoint's `config.json` tells its family (`model_type` `deepseek_v41`,
or `qwen3_5`); a family refuses options it has no use for at startup (the
expert-cache, CED and DSpark options for a Qwen model), and the process
loads at most one model of each family. Every model has its own name
(`id`; by default `deepseek-v4.1-flash`, or a Qwen model's directory's name
in lower case), context, batch, KV prefix cache (`kv_cache`), drafts and
request defaults.

| Request | Answer |
| --- | --- |
| `GET /v1/models` | Every model's object: OpenAI's fields (`id`, `object`, `created`, `owned_by`) with its name, `context_length` and `max_model_len`. |
| `GET /v1/models/{id}` | The model's object and its live status under `status`: what it is doing and its request, the waiting requests, its KV prefix cache's states (`sessions`), its counters (`model_decode`, `last_prefill`), its expert cache where it has one, and the process's memory. |
| `"model"` names a served model (case does not matter) | That model answers. |
| `"model"` names none, or no model served | With one model served, that model answers; with several, 404 that lists them. |

**Scheduling.** The worker keeps a queue for each model (16 requests wait at
most; more get 429) and gives every model one step a turn, in turn. A step
admits the model's waiting requests while fewer than its batch run,
planning each against the model's KV prefix cache when it is admitted, so
that it resumes from what the prefills before it left; it runs one prefill
call of the model's oldest prefilling request; and it advances every
decoding request of the model by a token: alone with the model's drafts
(`draft`, `lookup`), or together in one decoding pass. One prefill call is
the rest of the prompt for a model that plans its prefill over the whole
prompt (DeepSeek: replay, CED and pruning; `myds4_prefill_rows` 0), and
otherwise up to the model's prefill rows (a Qwen model's 512 positions, 128
while others of its requests decode), so that a long prompt does not stall
the replies in progress. Passes of different models therefore alternate on
the GPU, a pass at a time.

**Memory.** All models share the process's memory: a Qwen model's weights
(20 GB) stay resident while it serves (after 60 s without passes its
buffers leave the GPU's residency set until its next pass), and DeepSeek's
expert cache adapts around them. Its slots follow the memory available
(with `reserve` left to other programs), and its GPU-mapped slots follow the
GPU's resident memory of all processes, which past about 55 GiB stalls the
GPU: past that point the cache unmaps the excess within about a second, also
while idle.

The allocated cache and its GPU-mapped part are separate. Qwen's resident
weights reduce the room for mapped experts, but an unmapped expert stays
in the cache. CPU reads fill unmapped buffers; GPU work maps an expert
when it uses it. `cache` is an upper bound. A swapper wakeup alone does not
reduce it. A global swap-out reduces allocation and starts a five-minute
growth cooldown only when the expert cache has slots above its minimum
to release. Actual pressure and the configured reserve still apply.

On 2026-11-07, a bounded memory check served DeepSeek V4.1 Flash with
`UD-IQ1_M` experts and Qwen3.8-27B NVFP4 in one process. DeepSeek used
`cache` 24, `reserve` 16 and a 4,096-token context. Three distinct code and
scheduler prompts of 596, 660 and 682 tokens each generated 64 tokens;
two Qwen prompts between them each generated 24 tokens. Temperature and
top-p were 1; drafts, CED and substitution were off. DeepSeek retained all
2,484 slots (24 GiB), released none for pressure, and the run recorded no
new swap-outs. Free memory stayed above 12.98 GiB. This checked memory
behavior, not comparative throughput.

Larger checks did not complete under the same 10 GiB free-memory guard.
The first 56 GiB attempt also recorded swap-outs while the engine still
reported about 33 GiB available. File-backed pages count as available, but
macOS need not reclaim them before swapping anonymous memory. A large
co-hosted allocation is therefore not validated by the available-memory
figure alone. A valid comparison of DeepSeek's speed alone and beside
Qwen has not been made.

Qwen3.8-27B with `draft` 1 decoded its 300-token reply to the same prompt at
40.7 tokens/s beside DeepSeek (147 of 151 drafts accepted). When both models
decode at once, their passes alternate on the GPU.

## Chat formats

Each model family has its chat format (`server/src/format/`): its request
conversion, its chat template, its prompt tokens and the reading of its
replies. Both answer with the same Chat Completions fields:
`reasoning_content`, `content`, `tool_calls`, and the usage with the prompt
tokens the KV prefix cache reused (`prompt_cache_hit_tokens`,
`prompt_tokens_details.cached_tokens`). DeepSeek's format is
[Requests](#requests), [Prompts and tokens](#prompts-and-tokens) and
[Responses](#responses) below. A Qwen model's:

1. **Conversion** (`format/qwen.rs`): the messages' text (content parts of
   type `text`; images are refused), the leading `system` and `developer`
   messages joined as the system message (a later one speaks as the user,
   since the template takes one system message, first), assistant
   `reasoning_content` (or `reasoning`) and `tool_calls` (their `arguments`
   parsed as a JSON object), tool messages, and `tools` as given.
2. **The chat template** (`format/qwen_template.rs`): the checkpoint's
   `chat_template.jinja` as code, checked against renderings of the template
   itself (`server/src/format/fixtures/`): every assistant turn keeps its
   reasoning, tools are listed as the request gives them, tool responses
   join in one user turn. Thinking and its effort:

   | Request | Template |
   | --- | --- |
   | none | thinking at effort `xhigh` (the template's default) |
   | `reasoning_effort` `none` or `off`, `thinking.type` `disabled`, `chat_template_kwargs.enable_thinking` false | thinking off |
   | `reasoning_effort` `minimal` or `low` (or 1-50) | `low` |
   | `reasoning_effort` `medium` (or 51-75) | `medium` |
   | `reasoning_effort` `high`, `xhigh`, `max`, `ultra` (or 76-100) | `xhigh` |

3. **Tokens**: the model's tokenizer in the engine (`src/engine/tok.c`): the
   checkpoint's byte-level BPE after NFC and its split regex. The template's
   markers become their token ids; text from messages is encoded as text
   alone, so a message that spells `<|im_start|>` stays text.
4. **The KV prefix cache** (`kv.rs`, with no rewind room and no replay): a
   prompt continues an idle state that it extends, resumes from a snapshot
   (the source's keys copied before it), or starts cold. Every prompt end
   gets a snapshot with its logits (an identical retry resumes at once), a
   prefill captures the point where its prompt parts from every state, and
   takes one at every multiple of 2,048 positions when the budget
   (`kv_cache`) allows, since without a replay a later prompt can only resume
   at a snapshot; a reply's end gets one when the budget allows. A state
   takes 314 MB plus 68 KiB a position (in rooms doubling from 1,024
   positions), a snapshot 157 MB.
5. **Drafts**: with `draft 1`, a request that decodes alone verifies the MTP
   head's draft in its pass (two rows); with several decoding, each takes one
   row. The engine keeps the MTP head's keys complete while drafts are on.
6. **Sampling**: `temperature`, `top_p`, `top_k` and `seed` from the
   request; otherwise the model's configured ones, else the checkpoint's
   `generation_config.json` (Qwen3.8-27B: temperature 1, top_p 0.95,
   top_k 20). The end tokens are `generation_config.json`'s.
7. **The reply** (`format/qwen_protocol.rs`): reasoning up to `</think>`
   streams as `reasoning_content`, then text as `content` (the template's
   newlines after `</think>` left out); a standalone `<tool_call>` line at
   the top level (not in a code block) starts the calls, which are parsed at
   the end: after an end token they become `tool_calls` (`finish_reason`
   `tool_calls`), each parameter a string where its schema takes one and
   JSON otherwise; calls that do not parse, or a reply cut short, come back
   as text. `stop` strings end the reply before them.

## A request's path

1. The server reads the request within the [connections' limits](#connections)
   and checks the API key: SHA-256 digests compared in constant time, from
   `Authorization: Bearer KEY` or `x-api-key`.
2. Its `model` picks the served model ([Models](#models)).
3. The model's chat format converts and renders it, and the model's
   tokenizer gives the prompt's tokens ([Chat formats](#chat-formats)). For
   DeepSeek: the [compatibility steps](#compatibility-steps), recipe's
   `ChatCompletionRequest::convert` and `DeepseekV41Encoding`'s rendering,
   each image place the byte 0xFF (`IMAGE_MARK`) in the rendered prompt and
   then the image token, and the images of `data:` URLs decoded to their
   files' bytes.
4. `worker::submit` queues the tokens with the model, the images, sampling
   settings and token limit. Invalid input (a 400) and a full queue (a 429,
   at 16 waiting for the model) are answered before the response starts.
5. The response starts: server-sent events for `"stream": true`, else one
   JSON body at the end. Each of the worker's events (the prefill's end, each
   step's tokens, the end, or a refusal) becomes the format's output:
   DeepSeek's through recipe's `StreamProcessor`, whose Chat Completions
   chunks are written as they come; a Qwen model's through its reply
   protocol. Every request gets exactly one last event, also one cancelled or
   stopped before its prefill.
6. When the client closes its connection or stops reading for 60 s, or the
   format stops early (a stop sequence), the server cancels the request:
   dropping its guard (`worker::Cancel`) sets its flag, which the worker reads
   before each step and when it admits requests.

## Connections

The server accepts at most 64 connections at once and answers another with
503 (`too many connections`) before reading its request. Each connection
sends without delay (`TCP_NODELAY`), so a token's chunk goes out at once.

| Limit | Beyond it |
| --- | --- |
| A request's line and headers arrive within 60 s, also on an idle connection between requests | The connection closes. |
| A request's line and headers take at most 64 KB | 431 |
| Each part of a body arrives within 60 s of the one before | 408 |
| A body takes at most 32 MB | 413 |
| The client takes each write within 60 s | The connection closes, and its request is cancelled. |

A connection that closes before its response ended (its client left, or a
write waited 60 s) cancels its request: a decoding request stops before its
next step, and a waiting one is dropped when the worker next admits
requests, without running. hyper notices a client that closes its
connection while the server computes the answer, streamed or not, because
it keeps reading the connection meanwhile; reads have no deadline then.

## The C ABI

[`src/engine/engine.h`](../src/engine/engine.h) is the whole interface. Its
version is `MYDS4_ABI = 8`. Callers must use the matching structure layouts
and rebuild with the engine. The engine runs the models; everything about
requests (their queues, scheduling, the KV prefix caches, the status and the
live view's data) is the server's. Every function that runs a model or
touches states, snapshots, samplers, lookups or images runs on one thread,
the engine thread; the log, the stop, the statistics and the tokenizers may
be called from any thread. Its parts:

- **The process.** `myds4_init(process)` installs two hooks, `on_log`, which
  receives the log's lines instead of stderr (the live view's pane), and
  `on_die`, which runs before a fatal error's message (the live view gives the
  terminal back), and the stop's signal handlers, on the calling thread,
  which from then on is the engine's. `myds4_stopping()` reports a requested
  stop, `myds4_request_stop()` requests one, and `myds4_log(line)` writes a
  line to the log with the time.
- **Models.** `myds4_load(config)` takes one model's options
  (`myds4_config`: the checkpoint, the context, the drafts and lookups, and
  DeepSeek's `experts`, cache settings, `substitute`, `draft_cache`, online
  predictor, `ced`, `prefill_prune` and `runahead`) and returns the model,
  its family from `config.json`; a family refuses options it has no use for,
  and the process holds one model of each family. A DeepSeek model waits for
  the model lock (`myds4.lock` beside the executable, or `$MYDS4_LOCK`).
  `myds4_finish(model)` saves what the model keeps (the predictor state), and
  `myds4_idle(model)` lets it give memory back while it runs no request.
- **Limits.** For each model: the rows of different states one decoding pass
  takes, the drafts per pass it makes itself, the rows of one verify pass,
  the positions a state can continue before its end (DeepSeek's
  `max(8, lookup_depth)`, a Qwen model's 0), the vocabulary, the most rows
  one prefill call should take (`myds4_prefill_rows`: 0 for the whole prompt,
  DeepSeek's), and whether a token ends a reply (`myds4_stop`).
- **States and snapshots**, each of one model: `myds4_state_new`,
  `myds4_state_free`, `myds4_state_bytes`; `myds4_state_copy(model, dst,
  src, positions)` copies the rows of a state that later positions do not
  change before a position into another state (DeepSeek's compressed rows and
  index keys, a Qwen model's keys and values), also while a request decodes
  in the source. `myds4_snapshot_take(model, state, pos, with_logits)` keeps
  a state's rolling part (DeepSeek's window rings, compressor rows, draft
  rings and Engram history; a Qwen model's DeltaNet and convolution states
  and last hidden) and optionally the last logits; `myds4_snapshot_restore`
  puts them back into a state that holds the snapshot state's rows before
  `pos`.
- **Running a model**, on explicit states.
  `myds4_prefill(model, state, tokens, n, from, replay, capture, &captured,
  &info)` prefills positions `from` to `n - 1` and returns the last one's
  logits. With `replay` (DeepSeek), the state holds only the copied rows
  before `from`, and the prefill starts earlier to rebuild the rolling rows
  at `from`. Without `--ced` the replay is exact: each layer starts where its
  window needs. With `--ced` it is bounded: every layer starts a ring before
  `from` (136 positions with the default `--lookup`), the windows of the rows
  it computes stop there, and the copied compressed rows and index keys stay
  as they are. A `capture` position after the prefill's first position and
  before `n` returns the rolling rows there as a snapshot (a fork's branch
  point), or none when they would hold the encoder layers only (under
  `--ced`, before the retained decoder suffix). `myds4_replay_rows(k)` (-1
  for a model without a replay) and `myds4_replay_start(k)` give a replay's
  cost and first position. `myds4_rows` (one token for each of several
  states), `myds4_draft`, `myds4_verify` and `myds4_accept` decode;
  `myds4_seek` sets DeepSeek's Engram history. Calls with invalid arguments
  return null instead of logits.
- **Sampling, context-copy drafts and images.** Samplers with the request's
  temperature, top_p, top_k and seed (`myds4_sample`, `myds4_sample_verify`
  for a verify pass, a model's end tokens ending the kept drafts); the
  lookup's history matcher; a prompt's images for a model with a vision
  encoder (`myds4_images_open`, `myds4_images_expand` to write each image's
  span of keys in place of its mark, `myds4_images_encode` for the images
  that a prefill reaches).
- **Statistics.** `myds4_stats_get(model, ...)` fills the model's counters
  (decoded and prompt rows, its expert cache's where it has one, the running
  prefill's progress) and the process's memory, which the status and the live
  view show. `myds4_routes(model, row, ...)` gives the experts that a row of
  the model's latest decoding pass chose in each layer and where each one
  was, for the live view's expert map: the engine keeps them when it counts
  each row's experts by where they are, a copy of numbers the CPU has
  already, so decoding computes nothing more.
- **Tokenizers.** `myds4_model_tokenizer(model)` gives a model's tokenizer,
  `myds4_tokenizer_open(dir, image)` a checkpoint's without its model (the
  tests'); `myds4_tokenize` (text alone, with the added tokens' spellings, or
  a rendered prompt), `myds4_piece` (a token's bytes), `myds4_added` (whether
  a token is an added token, and whether tokenizer.json marks it special) and
  `myds4_special` (an added token's id) serve the prompts and the output
  decoding. Each tokenizer has its own lock.

A request's system, user, tool and context messages arrive as one prompt.
The worker resumes it from the model's KV prefix cache (see
[The KV prefix cache](design.md#the-kv-prefix-cache)) and prefills the
rest: for DeepSeek in one call. The model divides more than 16,384
new tokens into chunks to bound working memory. The scheduler prefills
separate requests in queue order while other requests wait to decode.
Rendering and tokenization precede enqueueing, so HTTP arrival order and
queue order can differ. The two identifiers in the log expose that timing.

`--ced off|N` controls approximate decoder replay in the server and the
CLI. `off` is the default. N is 128 to 1,048,576; the engine retains N plus
`max(8, lookup_depth)` rewind rows. Shorter inputs run full prefill. CED preserves full-context
encoder/global KV work and changes the later decoder's local state, so it
can change generated output. It does not shorten decoding's layer stack.
With CED the KV prefix cache's replay is bounded: it computes a ring's
positions before the resume point again (136 with the default `--lookup`,
`128 + max(8, lookup_depth)` in general) with their windows stopping there,
so a cache hit is approximate too ([The KV prefix
cache](design.md#the-kv-prefix-cache)). `myds4_config.ced` is 0 for off.

`--prefill-prune` enables dependency-preserving decoder prefill, off by
default. All encoder/global KV work remains complete. The later layers
retain enough history for the final state and for a fork's branch-point
capture, then skip batches outside those dependencies. CED, substitution and
full-row diagnostics disable this plan. Short prompts may have nothing
to skip. It can improve prefill while changing the cache left for decoding;
the measured long MXFP4 request finished sooner but decoded more slowly.

`--lookup off|N` enables CPU context-copy proposals with a maximum of one
to fifteen tokens. The target model verifies every proposal using the
request's sampler. With one active request, lookup supplies eligible copied
continuations before `--draft` supplies any DSpark fallback. Without
`--draft`, the server does not load DSpark weights. With multiple active
requests, each request uses ordinary one-token decoding and retains its
own lookup history. Lookup is off by default.

`--runahead` changes how two or more active requests share the model. Each
request can advance to its next layer when its selected experts are ready.
The scheduler groups ready requests at the same layer and prioritizes
earlier layers; each request in a group runs its own experts, and the
group shares the next layer's dense products. Reads use actual router
selections; the concurrent path
does not compute or issue the unused native expert forecasts. Cache entries
remain reserved until every consuming job completes. Per-request sampling
and selected experts are preserved. A request running alone follows ordinary
decoding, including context lookup when configured.

Runahead is off by default. The engine refuses combinations with DSpark,
substitution or the online predictor. CED retains its configured prefill behavior. The model's
status (`GET /v1/models/{id}`) reports `runahead` and cumulative `model_decode`
counters: computed rows, model wall time, main-queue GPU time, expert-read
bytes and read-wait time. These times overlap and must not be added together.

The HTTP comparison uses a short off/on/off sequence, not a general
throughput guarantee, and it ran while runahead still shared expert weight
loads within a group. Its [recorded results](performance.md#http-runahead-comparison)
include the memory budget, prompt reuse and output equality.

Arrival logs carry an HTTP identifier, body size, and message/tool counts.
Queue logs show waiting requests and how many the scheduler admits for
prefill. Each request's start names how it resumed (`new`, `kept`, `saved`
or `forked`) and the tokens it reused; each completed prefill reports its
chunks, encoder/decoder tokens, CED application, its capture position
(`prefix=`, a fork's branch point, or 0) and, after a replay (`an exact
replay`, or `a bounded replay` with `--ced`), how many earlier rows it
computed again. These are counts, without conversation
text or API keys. The model's status includes configured `ced_depth`,
`lookup_depth`, `prefill_prune` and the `last_prefill` plan.
`last_prefill.prune_applied` reports whether that pass actually skipped
decoder work. A separate title request therefore has its own arrival
and prefill records rather than appearing as a chunk of the main request.

`expert_cache.slots` counts allocated expert slots. Allocation is lazy, so
it can be zero after startup and grow during the first prompt.
`expert_cache.slots_released` counts slots returned because the cache
shrunk for memory. Benchmark tools use that counter to detect capacity
changes even when the cache fills again before the next status sample.

`model_decode` reports cumulative model-forward work across active requests:
computed rows, elapsed seconds, main-queue GPU seconds, requested expert
gigabytes and read-wait seconds. It excludes prompt prefill and CPU response
streaming. Compare counter differences over a workload rather than comparing
the final values from servers with different histories.

## Requests

The 2026-11-03 joint-prefill integration checks ran the server with a 20 GiB
expert cache, a 4 GiB reserve, temperature/top-p 1 and seed 12345, the MXFP4
experts, `--ced 128`, two drafts with 32 slots per stage and `--draft-trim`
(an option since removed; see
[Removed experiments](experiments.md#--draft-trim-trimming-the-final-dspark-stage)),
and an image request. The checks cover simultaneous arrivals, queue and prefill-plan logs,
system-prefix forks, identical seeded replies on retry, zero recomputation
for that retry, and image embedding through the default vision path. They
do not score answer quality. Run `tools/checks/server_prefill_check.py` with
`--checkpoint CHECKPOINT`, `--binary EXECUTABLE`, `--ced off|128|512`, and
optional `--experts GGUF`, `--draft`, `--lookup N`, `--prefill-prune`,
`--batch N` and `--image`. Add `--long-prefill --prefill-prune` to require
that a prompt actually exercises dependency pruning. This longer server
case is provided for integration testing; it has not yet been run.
The tool checks GPU ownership,
bounds runtime and retains requests, replies and logs only in memory.

The KV prefix cache's decisions are tested without a model in
`server/src/kv.rs`: the longest match across busy and idle states, the
choice between continuing in place, a snapshot and a replay (and the
replay's cost, exact or bounded), the same prompt again, truncation in
place, a prompt that parts within the rewind margin before a snapshot (a
thinking turn's next turn), the stale rows that take a state's rewind room
(a short prompt continued before a state's end, a verify pass's rows after
its kept tokens) and the rows that give it back, the branch point a fork
leaves (after a replay, or past a snapshot
it resumed from), the refusal before any eviction, the eviction order and
the state bound; and the budget's other steps: a retry of a saved prompt at
a full budget, a decoding state charged its growth before it grows (idle
states and spare snapshots freed first, the limit exceeded only when busy
states alone hold it), and a generation-end snapshot kept only within the
limit. `tools/checks/state_fork_check.c` checks a fork's row
copy and snapshot restore on host buffers, and that the bytes the cache
charges a state are those its buffers take. With a model, the fork test
checks exactness: `tools/checks/prefix_fork_check.c` in the engine and
`server/tests/prefix_fork.rs` through HTTP (see [Checks](#checks)).
`tools/checks/attention_candidate_check.c` checks candidate-mask offsets after
decoder compaction and across prompt batches.

The server's defaults for requests that do not set a value:

| Setting | Default |
| --- | --- |
| `max_tokens` (`max_completion_tokens` first) | 262,144, at most what the context leaves after the prompt |
| `temperature` | 1 (`-t` sets another default; 0 is greedy) |
| `top_p` | 1 (`--top-p`) |
| `seed` | random for each request (`--seed`) |
| thinking | on, at effort 75 (`--think N`; 0 turns it off) |

No expert substitution, drafts or online predictor run unless their options
are given. `GET /v1/models` reports the same `max_tokens` default as
`max_output_tokens`.

### Compatibility steps

The server accepts some request forms that recipe's Chat Completions schema
refuses or reads differently. Before conversion, it rewrites them into
recipe's form:

- the role `developer` becomes `system`;
- `thinking` given as `true` or `false` becomes `{"type": "enabled"}` or
  `{"type": "disabled"}`, and an object without a `type` enables thinking;
- `reasoning_effort` as a number from 1 to 100 (or such a number as a
  string) is rendered as that number: recipe renders only the levels 50, 75
  and 100, so the server converts with `high` and writes the number into the
  prompt's effort line. Other unknown values give the default effort;
- any `model` name, or none, is served; recipe converts with its V4.1 name,
  and responses report the request's name (`deepseek-v4.1-flash` without
  one);
- `temperature`, `top_p` and `seed` are read by the server (a finite
  nonnegative temperature, top_p from 0 to 1, a nonnegative integer seed),
  and removed before conversion. recipe would
  refuse a temperature above 2 and a top_p of 0, and would set the
  temperature to 1 and top_p to at least 0.95 in thinking mode;
- `max_tokens` and `max_completion_tokens` are read by the server and
  removed before conversion, so that recipe's defaults (8,192; 65,536 or
  131,072 in thinking mode) and its upper bound do not apply;
- a tool call without an `id` gets `call_N`, N its place in its message,
  and a tool message without a `tool_call_id` answers the call at its place
  after the assistant message; an empty `tool_calls` list counts as none.

### Images

User messages and tool results may hold `image_url` parts with `data:` URLs
in base64. The server fetches no URLs and reads no files for a client: a
request with any other image source is refused. The engine decodes and
prepares the images (`src/engine/vision.c`, after the checkpoint's
`image_processor.py`).
recipe's own image crate prepares images with OpenCV and re-encodes them as
WebP; the server switches to it only once its patches equal those of the
checkpoint's `image_processor.py` (`tools/reference/check.py vision`).

## Prompts and tokens

recipe's `DeepseekV41Encoding` renders the prompt. The server gives it, and
the conversion, the checkpoint's spelling of the image token,
`<｜deepseek_image｜>` (id 129264); recipe's default spelling `<｜image｜>`
is not in the checkpoint's tokenizer.

The server tokenizes the rendered prompt on the request's thread with the
engine's tokenizer (`myds4_tokenize`, `src/engine/tok.c`): a special
token's spelling anywhere in the prompt is that token, and the byte 0xFF is
an image's place. The worker writes each image's span of keys at its place
(`myds4_images_expand`); the keys come from the image file's bytes, so a
cached prefix matches only the same image.

The model's output tokens go to recipe's `StreamProcessor` as token ids. Its
tokenizer is the engine's: a token's text is its bytes (`myds4_piece`), and
the special tokens are the tokenizer's added tokens (`myds4_added`), each of
which the processor decodes as a piece of its own, so that `</think>` and
the DSML tags count only as tokens. The end token is not passed on: the
worker's last event with the finish `Stop` becomes the processor's stop, and
the end token is counted as one completion token without text, as the
worker counts it.

## Responses

A streamed response is recipe's sequence of `chat.completion.chunk` events
and `data: [DONE]`. While a streamed request waits in the queue, the server
writes the comment `: waiting` every 15 s. A request the engine refuses after
its response started gets an `error` event and `[DONE]`; a request ended by
the server's stop gets an `error` event saying so. A response that is not streamed is recipe's
accumulated `chat.completion` object.

Usage reports the prompt tokens (with the images' spans), those whose rows
came from the KV prefix cache (`prompt_cache_hit_tokens`, `cached_tokens`;
see [The KV prefix cache](design.md#the-kv-prefix-cache)), and recipe's
completion and reasoning counts.

## Prompts compared with encoding.py

recipe follows DeepSeek's request converter, which differs from the
checkpoint's `encoding/encoding.py` in these places; there, the server's
prompts differ from `encoding.py`'s:

- Tool definitions: recipe lists each tool as `name`, `description` (empty
  when absent) and `parameters`, in that order; `encoding.py` writes the
  request's `function` object as given, with its key order and any other
  keys. The DeepSeek Harness's Chat Completions client adds `"strict":
  false` to every function, so the prompt of every harness request with
  tools differs from `encoding.py`'s.
- recipe drops a system message whose content is empty (an empty string, or
  only empty text parts); `encoding.py` writes its system token.
- A system message's own `tools` field (an `encoding.py` extension, not part
  of the Chat Completions API) is ignored by recipe.
- recipe keeps the first of two consecutive latest-reminder messages and
  drops the second.
- In thinking mode without tools, recipe drops the reasoning of every earlier
  assistant message; `encoding.py` keeps the reasoning of assistant messages
  after the last user turn.
- recipe validates more: a tool message must follow the assistant message
  whose call it answers, an assistant message needs content or tool calls,
  and image media types must be PNG, JPEG, GIF or WebP.

Output: recipe parses DSML tool calls with its own rules, counts reasoning
tokens by encoding the reasoning text (without `</think>`), and supports stop
sequences.

Measured on 2026-11-01 with `tools/reference/check.py encoding`; the causes
were found by rendering each differing conversation again with recipe's form
of the tool definitions and without its empty system messages:

| Prompts | Equal to `encoding.py`'s | The others differ by |
| --- | --- | --- |
| The checkpoint's test cases 1, 2, 3 and 5 | 1, 2 and 5 | case 3: its mid-conversation system message carries `tools` |
| 1,800 generated conversations (600 with images) | 746 | tool definitions: 887; tool definitions and an empty system message: 39; reasoning after the last user turn, thinking without tools: 127; an empty system message: 1 |

No generated conversation was refused. The generator writes a tool's
`description` after its `parameters` or leaves it out, so most of its
conversations with tools differ.

## Build

```sh
make server
```

builds `build/libmyds4.a` (the engines: the C sources but `src/cli/main.c`,
with the Qwen model of `src/model/qwen.c`),
then the crate with `cargo build --release --target-dir build/cargo`, and
copies the executable to `build/myds4-server`. Beside `build/myds4`, it takes the same lock,
`build/myds4.lock`, so only one of the two loads the model at a time. The
crate's `build.rs` builds the library through the same Makefile, so `cargo
build` in `server/` works too; `server/.cargo/config.toml` sets the same
target directory there, and cargo reads it only when it runs in `server/`.
Building needs a Rust toolchain: `cargo` on `PATH`, or `CARGO=` the path of
one for `make`. The first build downloads the crates of `server/Cargo.lock`
into cargo's home and writes about 0.5 GB to `build/cargo`. Through
`tokenizers`, which `deepseek-recipe` depends on, the build compiles
Oniguruma (C), `esaxx-rs` (C++) and `ring` (C and assembly); the server does
not call them: prompts and output use the engine's tokenizer.

## Checks

| Command | What it checks |
| --- | --- |
| `.venv/bin/python tools/reference/check.py encoding CHECKPOINT` | The server's prompts (`build/myds4-server --render FILE`) against `encoding.py` on the test cases and the generated conversations ([above](#prompts-compared-with-encodingpy)). |
| `make server-check` (`MYDS4_CHECKPOINT=CHECKPOINT` for the tokenizer test) | The models' settings: one model's options on the command line, a configuration file's models each with its own options, and unknown or invalid options refused. Qwen's format: the template's renderings against the checkpoint's own template (`server/src/format/fixtures/`: ten conversations, and five with tools, tool history, thinking off and the low and medium efforts, rendered with jinja2 as Hugging Face renders them), the reply's text as tokens arrive (whole UTF-8 characters), the reply's reasoning and calls at every chunk boundary, prose and code-block examples kept as text, calls that do not parse given back as text, a stop string across tokens, and the request conversion (roles, content parts, tool history, effort and thinking). DeepSeek's format: the compatibility steps, a numeric effort in the prompt, the worker's events through recipe's processor with the checkpoint's tokenizer (reasoning, content, finish, usage). The KV prefix cache's decisions (`server/src/kv.rs`), and the live view's tabs on a test terminal: the title and the tabs first, every line within the terminal's width at sizes from 220 x 60 down to 12 x 4, each tab's parts, the keys and clicks that choose tabs, nodes and requests, a node's text answered while its session grows and a chosen node kept through a fork that splits it, the last minute's samples (the counters' differences, arrivals, routed rows of the snapshot's size, the SSD's rate from the reads as they complete, a prefill's rate with no sample between two prefills), the throughput's windows and its since-start figures from counters counted together, per-request speeds, the colors (the terminal's answer for its background, light and dark, the 256 colors) and the quit keys; and the connections' limits on a local socket, with short timeouts: 503 beyond the connection limit, 431, a late head, 408 and 413, and a client that leaves or stops reading ending its request. No model is loaded; without `MYDS4_CHECKPOINT` the tokenizer test is skipped. With `MYDS4_EXPERTS` (a GGUF file) set too, it also runs the fork tests through HTTP (next row). The tokenizers' ids against the reference tokenizer's are a C check (`build/tokenizer-reference-check`, [Usage](usage.md#checks-and-diagnostics)). |
| `MYDS4_CHECKPOINT=CHECKPOINT MYDS4_EXPERTS=GGUF cargo test --release --manifest-path server/Cargo.toml --target-dir build/cargo --test prefix_fork -- --nocapture` | The fork test through HTTP (`server/tests/prefix_fork.rs`, about 12 minutes, one model process at a time): a conversation with a long system prompt and user turn decodes, and 20 forks of it, cut in the system prompt or the user turn and continued differently, must each reuse at least the prefix they share with it less what an exact replay there computes again, and no more than that prefix, and must give the tokens that a server starting without a cache gives. With `--ced 128` the same forks must each reuse at least the shared prefix less 136 tokens; their tokens are printed. A thinking turn's next turn, which shares all of the turn's prompt but its last token (`<think>` there, `</think>` in the next turn's history), must reuse that shared prefix and compute only its new tokens (`last_prefill.tokens` in the model's status, `GET /v1/models/{id}`), with and without `--ced 128`, and without `--ced` give the cold server's tokens. A fork after an image (a 32 x 32 PNG the test builds) must give the cold server's tokens. Two tests check the budget with a tight `--kv-cache` set from what a first server reports: a retry of a saved prompt must fit a cache filled to within 2 MiB of its limit, and a generation that takes its state past 16,384 positions must free an idle state before its state grows and keep the cache within its limit in every status sample. `MYDS4_SERVER` and `MYDS4_CLI` pick the binaries (default `build/myds4-server` and `build/myds4`). |
| `MYDS4_QWEN=DIR cargo test --release --manifest-path server/Cargo.toml --target-dir build/cargo --test qwen -- --nocapture --test-threads 1` | A Qwen3.8 model through HTTP (`server/tests/qwen.rs`, about 3 minutes, one model process at a time): a request whose client leaves during the prefill of a 38K-token prompt, and then a prompt that extends that whole prompt, which must reuse only what the cut-short state ran and give the reply the same conversation gets where nothing was cancelled; and a reply that runs to the context's end (`--ctx 160`) must take every position up to it. With `--draft 1`, replies must take exactly `max_tokens` tokens (20 to 23) and, at `--ctx` 160 and 161, every position up to the context's end. With `MYDS4_CHECKPOINT` and `MYDS4_EXPERTS` set too, DeepSeek and the Qwen model run in one process from a config file: each must answer, and while both are idle the log may report the expert cache shrinking only to the slots it really has. |
| `MYDS4_CHECKPOINT=CHECKPOINT MYDS4_EXPERTS=GGUF cargo test --release --manifest-path server/Cargo.toml --target-dir build/cargo --test live_view -- --nocapture` | The live view with a model (`server/tests/live_view.rs`, about 40 s, one model process): the server runs with `--tui` in a pseudo-terminal of 160 x 46 cells that answers the view's question for its background as a dark terminal would. While a request decodes, the test presses the tabs' keys and rebuilds the screen from the output: the dark colors, the expert map's request at four experts a cell, the KV tree, a node's text, the last minute and the log must be drawn, the mouse taken and given back, and the server must exit with status 0 after `q`. `MYDS4_SERVER` picks the server (default `build/myds4-server`). |
| `build/prefix-fork-check CHECKPOINT --text FILE [--rendered] [--limit N] [--experts GGUF] [--draft N] [--cache GB] [--ced N] --fork [--random N] [--at P]... [--prompt-only] [--suffix N]` | The fork test in the engine (`tools/checks/prefix_fork_check.c`): a conversation (the text's prompt, or its first `--limit` tokens, then a decoded reply, a tool-like result and a second decoded reply) forked at 26 position classes and N random positions (default 16). Where the history before a fork came from prompts, the exact replay, the latest snapshot and the first snapshot after the fork within the rewind margin (continued at the fork) must give a cold prefill's bytes in the logits, every state tensor and the next 16 greedy tokens; where it holds decoded rows, both snapshot resumes must give the bytes of the conversation computed again as it ran, and the replay must be about as close to the cold prefill as that conversation is. Each fork's branch-point snapshot, resumed over that fork's rows, must give the fork's own bytes. Without `--ced` it also checks the KV cache's stale-row count: a state with 3 stale rows (a short prompt continued before a state's end, or a verify pass of 4 rows that keeps 1 token), resumed within the room the cache gives it, and the state 3 tokens later, resumed the whole margin before its end, must give the cold prefill's bytes; resumes past the room are printed. With `--ced N` the replay is bounded: it must keep the copied compressed rows and index keys byte for byte and give finite logits, and each fork's distance from the cold prefill is printed. `--measure` prints how far decoding a reply and prefilling it differ. |

On 2026-11-04 (the UD-IQ2_XXS experts, `--cache 40`, `--kv-cache 8`) the
server, with the worker in Rust, reused 0 of a new prompt's 54 tokens, 54 of
the same prompt again (without a prefill, giving the same reply), 49 for
another user turn after the same system prompt (an exact replay), and 66 of
the first conversation's next turn's 75 (continued in place). SIGTERM during
a streamed request ended it with the `error` event `the server is
stopping`, and the server exited with status 0.

Later on 2026-11-04, with every fork capturing its branch point and the
bounded replay under `--ced` (same experts and options), the HTTP fork
tests passed in 12 minutes. Without `--ced` all 20 forks gave the tokens of
a server started without a cache. The first user-turn fork replayed at its
whole shared prefix (4,119 tokens); each later one resumed from the branch
point that the fork before it captured, for example sharing 4,256 tokens and
reusing 4,119. The 20 forks reused 4,417 fewer tokens than they shared, at
most 1,490 for one fork. With `--ced 128` every fork reused its whole
shared prefix by a bounded replay. The image fork reused 4,303 tokens and
gave the cold server's tokens. The engine's fork test on the first 5,600
tokens of `docs/design.md` (`--limit 5600 --fork --random 16`) passed: 40
forks, 0 failures, and all 47 resumes from a fork's own branch-point
snapshot gave that fork's bytes. With `--ced 128` it passed as well (40
forks, 0 failures): every bounded replay kept the copied compressed rows
and index keys byte for byte. Against the cold prefill (compacted too), the
bounded replays' logits had an SNR of 12.4 dB at the median (4.8 dB at
least) over prompt history and 10.9 dB (8.4 dB) over decoded history, with
the next token the same in 32 of 40 forks. Resumes from a snapshot under
CED, which the cache also used before, gave 12.7 dB (6.9 dB), the next
token the same in 13 of 15: CED's own result depends on where a prefill
starts, so no cache hit under CED gives the cold prefill's bytes.

After the budget fixes (2026-11-04, same experts), the two budget tests
passed in 57 s. A retry of a saved prompt fitted a cache filled to within
2 MiB of its limit: the first try took 0.065 GiB of `--kv-cache 0.0670`,
and the retry reused 15 of 15 tokens. A generation of 16,353 prompt tokens
and 181 generated tokens under `--kv-cache 0.1544` freed the idle state
before its own state grew past 16,384 positions: none of the 29 status
samples taken while it decoded past them still held the idle state, and no
sample exceeded the limit. The HTTP fork tests passed again (674 s) with
the same reuse.

On 2026-11-06 (the UD-IQ2_XXS experts, `--cache 40`), a thinking turn's
next turn shared 5,163 of its 5,186 tokens with the turn's prompt. Before
the cache resumed from a snapshot within its rewind room, that turn
reused 5,163 tokens by a replay and computed 5,110 rows without `--ced`
(an exact replay: about 5,087 earlier rows again) and 156 with `--ced
128` (a bounded one). Since then it resumes from the turn's prompt-end
snapshot one position back and computes 23 rows either way. Without `--ced`
its tokens equal the cold server's. The engine's fork test on the first
3,000 tokens of `docs/design.md` (`--limit 3000 --fork --random 0`) passed:
23 forks, 0 failures. The snapshot after a fork within the rewind margin,
continued at the fork, gave the cold prefill's bytes at 1 and 5 positions
before the prompt's end and at the reply's first position. Over decoded
history, continuing 1 position back, it gave the bytes of the conversation
continued there. With 3 stale rows (a short prompt continued before a
state's end, or a verify pass of 4 rows that kept 1 token), a resume 2
positions before the state's end gave the cold prefill's bytes. A resume 5
positions before it, which the cache gave before it counted stale rows,
differed in 867,679 and 863,467 bytes, and the next 16 tokens differed. 3
tokens decoded at the end gave the state back its whole margin: a resume 5
positions before it gave the cold prefill's bytes.

`tools/bench/turn_bench.py` timed the same kind of conversation through
the server (thinking on, greedy, `--cache 40`, without `--ced`, four
alternating runs, 2026-11-06). Its second turn, 8,891 tokens, shared 7,745
with the first. With the earlier build (`61cdf61`, which replays at the
shared prefix and predates the prefill eviction fix) it prefilled 6,233
rows and gave its first token after 10.64 and 10.77 s. With the snapshot
resume it prefilled 1,146 rows and gave it after 7.31 s both times. All
four runs gave the same replies.

The runs below were made with the scheduler in the engine's C code, before
it moved to `server/src/worker.rs`.

On 2026-11-01 the server (the UD-IQ1_M experts, `--cache 40`) answered eight
requests at temperature 0: thinking off; thinking at effort `high` and at
effort 62; a `developer` message; a tool call; a conversation's second turn
(24 of its 32 prompt tokens reused); the first prompt again, not streamed;
and a JPEG photo (4096 x 2160) with a question (960 prompt tokens).
`reasoning_tokens` counts the tokens of the reasoning text, without
`</think>` (43 and 104 for the two thinking requests).

The same run checked: `GET /health` without a key (200);
401 without a key and with a wrong one; `x-api-key`; 405, 404 and the 400s of
a body that is not JSON, a last message of the assistant, an `https` image
and a negative temperature; `/v1/models` and `/v1/models/{id}`; a client that
closes a stream after 12 chunks (the engine logged `client gone` after 12
tokens and was idle 0.2 s later); and SIGTERM during a stream (an `error`
event, no `[DONE]`, exit status 0 after 0.6 s).

With `--draft 2` (same options otherwise), the server answered the eight
greedy requests above, three sampled ones (temperature 1, top_p 1, seeds
1234, 7 and 42: a haiku, the primary colors at effort `high`, a five-sentence
story), and two pairs sent at once (two greedy, two sampled); for example,
the 67 tokens of the 1001 question took 25 passes with 41 drafts accepted.
The greedy answers equal those without drafts; the sampled answers equal
those without drafts, because verification consumes the sampler's draws as
plain decoding does; and each request of a pair, which decodes beside the
other without drafts, equals its answer alone.

On 2026-11-03 (the UD-IQ2_XXS experts, `--cache 40`), a run with `--tui` in a
pseudo-terminal of 120 x 40 kept the five lines logged while the model loaded
on the normal screen, then drew the view's five parts; while it answered a
greedy request of 48 tokens, its frames showed the request decoding, its
speed and its session's node. `q` stopped the server 0.9 s later with exit
status 0, the normal screen back and the 13 lines logged during the view
printed, the last `q: stopping after the current step`.
`tools/checks/online_predictor_server_check.py` with `--draft 2
--sampling-check` passed the same day.

The same day the server answered a 70 KB request head with 431 and, while
64 connections were open, a 65th with 503; once they closed, it served a
request. With `--batch 1`, a request that was not streamed stopped 0.05 s
after its client closed its connection while it decoded (`client gone`
after 256 tokens), and a waiting request whose client closed was dropped,
without running, when the first one ended (`client gone while waiting`).
SIGTERM during a streamed response ended it 0.13 s later with an `error`
event and no `[DONE]`, and the server exited with status 0 after 0.44 s;
`online_predictor_server_check.py` passed again. The 60 s timeouts were
checked with shorter ones in `make server-check`.

`--render FILE` renders a test case as the server renders a request with its
messages, tools, thinking mode and effort, with two differences that let it
take `encoding.py`'s cases: each image part's URL is replaced by a distinct
`data:` URL before conversion (the URLs are printed on stderr, one
`image URL` line each, in the order of their places), and a case that ends
with an assistant message is printed without the next turn's start, as
`encoding.py` renders it; the server refuses such a request.

## Not done yet

- The Responses and Messages endpoints, which recipe converts too.
- For Qwen models: images, log probabilities, `tool_choice` (the tools are
  always offered) and the live view.
- Log probabilities, and the constraints of structured output (recipe
  converts them; the engine does not enforce them).
- Prefilling several queued prompts in shared passes.
- Image preprocessing through recipe's image crate.
