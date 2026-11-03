# The Rust server

`server/` is a second HTTP server for myds4, written in Rust on DeepSeek's
[`deepseek-recipe`](https://github.com/deepseek-ai/deepseek-recipe) crates
(version 0.2.0, MIT License). It serves the same model with the same engine as
`build/myds4 --serve`: the C and Metal code of `src/` runs the model behind a
C ABI, [`src/engine.h`](../src/engine.h), and the Rust code handles HTTP,
request conversion, prompts and output. It takes the C server's options and
serves `POST /v1/chat/completions`, streamed or as one response, `GET
/v1/models`, `GET /v1/status` and `GET /health`.

The C server (`src/serve.c` and `src/chat.c`) stays the default until the Rust
server serves everything it serves; [Not done yet](#not-done-yet) lists what
is missing.

## Parts

| Part | Language | Files |
| --- | --- | --- |
| HTTP: endpoints, the API key, body limits, server-sent events, the options | Rust | `server/src/main.rs` |
| Requests: the compatibility steps, recipe's Chat Completions conversion, sampling settings, images from `data:` URLs | Rust | `server/src/chat.rs` |
| Prompts: recipe's V4.1 rendering | Rust | `server/src/chat.rs` |
| Output: recipe's stream processor (reasoning, content, tool calls, usage) and its Chat Completions chunks | Rust | `server/src/chat.rs` |
| The engine's ABI in Rust: declarations, events as messages, cancellation | Rust | `server/src/engine.rs` |
| The engine: the queue, sessions, batching, sampling, drafts, the status, the log | C | `src/engine.c` |
| The model, the expert cache, the vision encoder, the tokenizer, the live view, signals | C, Metal | `src/model.c`, `src/io.c`, `src/gpu.c`, `src/vision.c`, `src/tok.c`, `src/tui.c`, `src/util.c` |

`src/engine.c` holds the scheduler of `src/serve.c` without its HTTP and JSON:
the sessions and their reuse, the request queue, prefill, batched decoding,
drafts, the finish of a request and the live view's snapshot. Until the C
server is retired, both files hold that code.

## Threads

- **The engine's thread** (`myds4_start` creates it, with a 16 MB stack and
  the user-interactive QoS class, as `main.c` gives the C server's main
  thread). It loads the model and then runs the worker loop: it admits
  waiting requests while fewer than `--batch` decode, prefills each, advances
  every decoding request by a step and ends finished ones. All model work
  stays on this thread, because `gpu.c` keeps an autorelease pool per thread.
  While no request runs, it resizes the expert cache every second
  (`cache_idle`), as the C server does.
- **The live view's thread** (`--tui`), as in the C server.
- **Tokio's threads** run HTTP: reading requests, converting and rendering
  them, submitting them to the engine, and turning the engine's events into
  response chunks.

Signals belong to the engine: SIGINT, SIGTERM and the live view's `q` request
the stop, as in the C server. The Rust server checks `myds4_stopping()`
several times a second, stops accepting connections, waits for the open
responses to end (the engine ends every request), and calls `myds4_stop()`,
which saves the predictor state (`--predictor-state`) and returns.

## A request's path

1. The server reads the request (a body up to 32 MB) and checks the API
   key, as the C server checks it: SHA-256 digests compared in constant
   time, from `Authorization: Bearer KEY` or `x-api-key`.
2. It applies the [compatibility steps](#compatibility-steps) to the JSON,
   reads the sampling settings itself, and converts the rest with recipe's
   `ChatCompletionRequest::convert`.
3. recipe's `DeepseekV41Encoding` renders the conversation. Each image
   place becomes the byte 0xFF (`IMAGE_MARK`) in the prompt sent to the
   engine.
4. The images of `data:` URLs are decoded to their files' bytes.
5. `myds4_submit` queues the prompt, its images, sampling settings, token
   limit and an event callback. Invalid input (a 400) and a full queue (a
   429) are answered before the response starts.
6. The response starts: server-sent events for `"stream": true`, else one
   JSON body at the end. Each engine event becomes an input of recipe's
   `StreamProcessor`, whose Chat Completions chunks are written as they
   come.
7. When the client closes its connection, or the processor stops early (a
   stop sequence), the server cancels the request (`myds4_cancel`).

## The C ABI

[`src/engine.h`](../src/engine.h) is the whole interface. Its parts:

- **Start and stop.** `myds4_start(config)` takes the model and server
  options (`myds4_config`: the checkpoint, `--experts`, `--ctx`, the cache
  settings, `--substitute`, `--gpu-route`, the draft settings, the online
  predictor, `--ced`, `--kv-cache`, `--batch`, `--tui` and the address for the view's
  title). It waits for the model lock (`myds4.lock` beside the executable, or
  `$MYDS4_LOCK`), loads the model on the engine's thread and returns once
  requests are taken. `myds4_stopping()` reports a requested stop, and
  `myds4_stop()` requests one and waits for the engine's thread.
- **Requests.** `myds4_submit(request, &id, message, cap)` queues one request
  and returns at once: `MYDS4_QUEUED` with the request's id (also its number
  in the log), `MYDS4_BUSY` when 16 requests wait, or `MYDS4_INVALID` with a
  message. A request holds:
  - its whole prompt, as token ids or as rendered text that the engine
    tokenizes as the C server does (a special token's spelling is that token;
    the byte 0xFF is an image's place);
  - its images, as the bytes of image files, in the order of their places;
  - sampling: temperature (0 is greedy), top_p and an optional seed;
  - `max_tokens`, at least 1 (the context's end also stops a request);
  - the event callback and a pointer for it;
  - a note for the log, which the engine writes as `req ID NOTE` when it
    queues the request, before any line of its prefill.
- **Events.** The engine calls the request's callback on its own thread:
  `MYDS4_STARTED` after the prefill (prompt tokens with the images' spans,
  and the tokens a session held), `MYDS4_TOKENS` after each decoding step
  with the tokens that step kept (several when drafts are accepted; never
  the end token), and one `MYDS4_DONE` with the finish (`MYDS4_STOP` at the
  end token, `MYDS4_LENGTH` at `max_tokens` or the context's end,
  `MYDS4_CANCELLED`, `MYDS4_SHUTDOWN`) and the counts. A request refused
  before its prefill (the context or `--kv-cache` cannot hold it, or an image
  cannot be read) gets one `MYDS4_REFUSED` with a message instead. After
  `MYDS4_DONE` or `MYDS4_REFUSED` the engine never calls back for that
  request again. A callback must return quickly; it may call only
  `myds4_cancel` and `myds4_log`.
- **Cancel.** `myds4_cancel(id)` drops a waiting request or stops a running
  one before its next step. Ids of finished requests are ignored, so the
  caller may cancel whenever its response ends.
- **State.** `myds4_status()` returns the JSON of `GET /v1/status` (the
  phase, the requests, the sessions, the caches, memory and totals), and
  `myds4_log(line)` writes a line to the log: stderr with the time, or the
  live view's log pane.
- **Tokenizer.** `myds4_tokenizer(dir)` loads the checkpoint's tokenizer
  without the model; `myds4_tokenize`, `myds4_piece` (a token's bytes) and
  `myds4_added` (whether a token is one of the tokenizer's added tokens)
  serve the prompt checks and the output decoding below. The tokenizer keeps
  its work buffers in static memory, so the engine serializes its calls.

A request's system, user, tool and context messages arrive as one prompt.
The engine passes all its new tokens to the model together. It saves the
state before the first user turn during the same pass, for later forks
(see [Sessions](design.md#sessions)). The model divides more than 16,384
new tokens into chunks to bound working memory. The scheduler prefills
separate requests in queue order while other requests wait to decode.
Rendering and tokenization precede enqueueing, so HTTP arrival order and
queue order can differ. The two identifiers in the log expose that timing.

`--ced off|N` controls approximate decoder replay in both servers and the
CLI. `off` is the default. N is 128 to 1,048,576; the engine retains N plus
five rewind rows. Shorter inputs run full prefill. CED preserves full-context
encoder/global KV work and changes the later decoder's local state, so it
can change generated output. It does not shorten decoding's layer stack.
An encoder-only system-prefix snapshot can be forked only if the new suffix
exceeds N plus five rows and rebuilds the decoder window. The engine chooses
a complete cached state or cold prefill otherwise. C ABI version 3 adds the
`ced` configuration field; zero selects off.

Arrival logs carry an HTTP identifier, body size, and message/tool counts.
Queue logs show waiting requests and how many the scheduler admits for
prefill. Each completed prefill reports its chunks, encoder/decoder tokens,
CED application and saved prefix. These are counts, without conversation
text or API keys. `GET /v1/status` includes configured `ced_depth` and the
`last_prefill` plan. A separate title request therefore has its own arrival
and prefill records rather than appearing as a chunk of the main request.

## Requests

The 2026-11-03 joint-prefill integration checks used one owned server at a
time, a 20 GiB expert cache, a 4 GiB reserve, temperature/top-p 1 and seed
12345. Fourteen synthetic chat-completion requests passed across these configurations:

| Server | Experts | CED | DSpark | Image request |
| --- | --- | --- | --- | --- |
| Rust | MXFP4 | 128 | 2 drafts, 32 slots/stage, trim | yes |
| C | MXFP4 | off | off | no |
| C | mixed IQ2_XXS | 512 | 2 drafts, 32 slots/stage, trim | yes |

The checks cover simultaneous arrivals, queue and prefill-plan logs,
system-prefix forks, identical seeded replies on retry, zero recomputation
for that retry, and image embedding through the default vision path. They
do not score answer quality. Run `tools/server_prefill_check.py` with
`--checkpoint CHECKPOINT`, `--binary EXECUTABLE`, `--ced off|128|512`, and
optional `--experts GGUF`, `--draft` and `--image`; it checks GPU ownership,
bounds runtime and retains requests, replies and logs only in memory.

CPU regressions in `tools/prefill_schedule_check.c` cover one forward call
with a saved system prefix, exact retries after short replies, a changed
last prompt token, prefix-only requests and encoder-only fork eligibility.
`tools/attention_candidate_check.c` checks candidate-mask offsets after
decoder compaction and across prompt batches. The real-model
`tools/prefill_prefix_check.c` compares prefix state and resumed logits
at 640 and 16,521 tokens in both expert formats with DSpark enabled.

The server's defaults for requests that do not set a value:

| Setting | Default |
| --- | --- |
| `max_tokens` (`max_completion_tokens` first) | 262,144, at most what the context leaves after the prompt |
| `temperature` | 1 (`-t` sets another default; 0 is greedy) |
| `top_p` | 1 (`--top-p`) |
| `seed` | random for each request (`--seed`) |
| thinking | on, at effort 75 (`--think N`; 0 turns it off) |

No expert substitution, drafts or online predictor run unless their options
are given, as for the C server. `GET /v1/models` reports the same
`max_tokens` default as `max_output_tokens` (the C server reports 65,536).

### Compatibility steps

The C server accepts requests that recipe's Chat Completions schema refuses
or reads differently. Before conversion, the Rust server rewrites them into
recipe's form:

- the role `developer` becomes `system`;
- `thinking` given as `true` or `false` becomes `{"type": "enabled"}` or
  `{"type": "disabled"}`, and an object without a `type` enables thinking;
- `reasoning_effort` as a number from 1 to 100 (or such a number as a
  string) is rendered as that number: recipe renders only the levels 50, 75
  and 100, so the server converts with `high` and writes the number into the
  prompt's effort line. Other unknown values give the default effort, as in
  the C server;
- any `model` name, or none, is served; recipe converts with its V4.1 name,
  and responses report the request's name (`deepseek-v4.1-flash` without
  one);
- `temperature`, `top_p` and `seed` are read by the server, with the C
  server's rules (a finite nonnegative temperature, top_p from 0 to 1, a
  nonnegative integer seed), and removed before conversion. recipe would
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
in base64. As in the C server, the server fetches no URLs and reads no files
for a client: a request with any other image source is refused. The engine
decodes and prepares the images (`src/vision.c`) as the C server does.
recipe's own image crate prepares images with OpenCV and re-encodes them as
WebP; the server switches to it only once its patches equal those of the
checkpoint's `image_processor.py` (`tools/check.py vision`).

## Prompts and tokens

recipe's `DeepseekV41Encoding` renders the prompt. The server gives it, and
the conversion, the checkpoint's spelling of the image token,
`<｜deepseek_image｜>` (id 129264); recipe's default spelling `<｜image｜>`
is not in the checkpoint's tokenizer.

The engine tokenizes the rendered prompt with `src/tok.c`, as the C server
tokenizes its own: a special token's spelling anywhere in the prompt is that
token, and the byte 0xFF is an image's place. A prompt therefore gives the
same tokens in both servers whenever both render the same text.

The model's output tokens go to recipe's `StreamProcessor` as token ids. Its
tokenizer is the engine's: a token's text is its bytes (`myds4_piece`), and
the special tokens are the tokenizer's added tokens (`myds4_added`), each of
which the processor decodes as a piece of its own, so that `</think>` and
the DSML tags count only as tokens. The end token is not passed on: a
`MYDS4_DONE` with `MYDS4_STOP` becomes the processor's stop, and the end
token is counted as one completion token without text, as the C server
counts it.

## Responses

A streamed response is recipe's sequence of `chat.completion.chunk` events
and `data: [DONE]`. While a streamed request waits in the queue, the server
writes the comment `: waiting` every 15 s, as the C server does. A request
the engine refuses after its response started gets an `error` event and
`[DONE]`, as in the C server; a request ended by the server's stop gets an
`error` event saying so. A response that is not streamed is recipe's
accumulated `chat.completion` object.

Usage reports the engine's prompt tokens and the tokens a session held
(`prompt_cache_hit_tokens`, `cached_tokens`), and recipe's completion and
reasoning counts.

## Differences from the C server

recipe follows DeepSeek's request converter; the C server follows the
checkpoint's `encoding/encoding.py`. Where they differ, prompts differ:

- Tool definitions: recipe lists each tool as `name`, `description` (empty
  when absent) and `parameters`, in that order; the C server writes the
  request's `function` object as given, with its key order and any other
  keys. The DeepSeek Harness's Chat Completions client adds `"strict":
  false` to every function, so the two servers' prompts differ for every
  harness request with tools.
- recipe drops a system message whose content is empty (an empty string, or
  only empty text parts); the C server writes its system token.
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
tokens by encoding the reasoning text (the C server counted the tokens
generated up to `</think>`), and supports stop sequences, which the C server
ignores.

Measured on 2026-11-01 with `tools/check.py encoding` (`build/myds4` equals
`encoding.py` on all of them); the causes were found by rendering each
differing conversation again with recipe's form of the tool definitions and
without its empty system messages:

| Prompts | Equal to the C server's | The others differ by |
| --- | --- | --- |
| The checkpoint's test cases 1, 2, 3 and 5 | 1, 2 and 5 | case 3: its mid-conversation system message carries `tools` |
| 1,800 generated conversations (600 with images) | 746 | tool definitions: 887; tool definitions and an empty system message: 39; reasoning after the last user turn, thinking without tools: 127; an empty system message: 1 |

No generated conversation was refused. The generator writes a tool's
`description` after its `parameters` or leaves it out, so most of its
conversations with tools differ.

## Build

```sh
make build/myds4-server
```

builds `build/libmyds4.a` (the engine: the C sources but `main.c`,
`serve.c` and `chat.c`), then the crate with `cargo build --release
--target-dir build/cargo`, and copies the executable to
`build/myds4-server`. Beside `build/myds4`, it takes the same lock,
`build/myds4.lock`, so only one of the two loads the model at a time. The
crate's `build.rs` builds the library through the same Makefile, so `cargo
build` in `server/` works too; `server/.cargo/config.toml` sets the same
target directory there, and cargo reads it only when it runs in `server/`.
Building needs a Rust toolchain: `cargo` on `PATH`, or `CARGO=` the path of
one for `make`. The first build downloads the crates of `server/Cargo.lock`
(156 MB in cargo's home) and writes 0.5 GB to `build/cargo`. Through
`tokenizers`, which `deepseek-recipe` depends on, the build compiles
Oniguruma (C), `esaxx-rs` (C++) and `ring` (C and assembly); the server does
not call them: prompts and output use the engine's tokenizer.

## Checks

| Command | What it checks |
| --- | --- |
| `MYDS4_RENDER=build/myds4-server .venv/bin/python tools/check.py encoding CHECKPOINT` | The Rust server's prompts (`build/myds4-server --render FILE`) against `encoding.py` on the test cases and the generated conversations. |
| `MYDS4_CHECKPOINT=CHECKPOINT cargo test --release` in `server/` | The compatibility steps, a numeric effort in the prompt, and the engine's events through recipe's processor with the checkpoint's tokenizer (reasoning, content, finish, usage). No model is loaded; without `MYDS4_CHECKPOINT` the tokenizer test is skipped. |

On 2026-11-01 both servers ran one after the other with the same options
(the UD-IQ1_M experts, `--cache 40`) and answered the same eight requests at
temperature 0: thinking off; thinking at effort `high` and at effort 62; a
`developer` message; a tool call; a conversation's second turn (24 of its
32 prompt tokens reused); the first prompt again, not streamed; and a JPEG
photo (4096 x 2160) with a question (960 prompt tokens). The reasoning, the
content, the tool call, the finish reasons and the prompt, cached and
completion tokens were equal in all eight. `reasoning_tokens` was one lower
in the Rust server (43 and 104 against 44 and 105), because recipe counts
the tokens of the reasoning text, without `</think>`.

The same run of the Rust server checked: `GET /health` without a key (200);
401 without a key and with a wrong one; `x-api-key`; 405, 404 and the 400s of
a body that is not JSON, a last message of the assistant, an `https` image
and a negative temperature; `/v1/models` and `/v1/status`; a client that
closes a stream after 12 chunks (the engine logged `client gone` after 12
tokens and was idle 0.2 s later); and SIGTERM during a stream (an `error`
event, no `[DONE]`, exit status 0 after 0.6 s). A run with `--tui` in a
pseudo-terminal of 140 x 40 drew the live view, decoded the two thinking
requests together (`batch: 2 requests decoding together`) with the same
reasoning and content as each alone in the C server, and stopped 0.6 s after
`q` with exit status 0, the normal screen and the last log lines back.

With `--draft 2` (same options otherwise), both servers answered the eight
greedy requests above, three sampled ones (temperature 1, top_p 1, seeds
1234, 7 and 42: a haiku, the primary colors at effort `high`, a five-sentence
story), and two pairs sent at once (two greedy, two sampled). All 15
answers were equal between the servers, and every request's draft passes and
accepted drafts in the log were equal too (for example 25 passes with 41
drafts accepted for the 67 tokens of the 1001 question). The greedy answers
equal those without drafts; the sampled answers equal those of both servers
without drafts, because verification consumes the sampler's draws as plain
decoding does; and each request of a pair, which decodes beside the other
without drafts, equals its answer alone.

`--render FILE` renders a test case as the server renders a request with its
messages, tools, thinking mode and effort, with two differences that let it
take `encoding.py`'s cases: each image part's URL is replaced by a distinct
`data:` URL before conversion (the URLs are printed in the order of their
places, as `build/myds4 --render` prints them), and a case that ends with an
assistant message is printed without the next turn's start, as `encoding.py`
renders it; the server refuses such a request.

## Not done yet

- The C server's limits on connections: at most 64 at once, 60 s socket
  timeouts and 64 KB of headers.
- Noticing a client that leaves a request that is not streamed before its
  answer is written (the C server checks the connection every 16 tokens).
- The Responses and Messages endpoints, which recipe converts too.
- Log probabilities, and the constraints of structured output (recipe
  converts them; the engine does not enforce them).
- Prefilling several queued prompts in shared passes.
- Image preprocessing through recipe's image crate.
- Retiring `src/chat.c` and the C server's copy of the scheduler in
  `src/serve.c`.
