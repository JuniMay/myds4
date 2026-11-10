// myds4's inference engine behind a C ABI, for the server (server/, in Rust).
// The engine runs the models: every checkpoint the model interface of
// src/model/model.h runs (DeepSeek V4.1 Flash, Qwen3.8), several of them in
// one process, at most one of each family. For each model: loading, the
// attention states and their snapshots, prefill and decoding, the model's
// drafts and their verification, sampling, context-copy drafts, the vision
// encoder where it has one, and its tokenizer; for the process: the log and
// the stop on SIGINT or SIGTERM. The server owns everything about requests:
// their queue and scheduling, the KV prefix cache (which states and
// snapshots to keep, and how a prompt reuses them), HTTP, and the status it
// builds from myds4_stats_get.
//
// Threads: myds4_init, myds4_load and every function on models, states,
// snapshots, prefill, decoding, drafts, samplers, context lookups and images
// run on one thread, the server's engine thread: the GPU code keeps state per
// thread, and one command queue carries every model's passes. The tokenizer's
// functions, myds4_log, myds4_stopping and myds4_request_stop may be called
// from any thread. myds4_stats_get runs on the engine thread between passes;
// the caller publishes a copy for its HTTP handlers and terminal.
#pragma once
#include <stddef.h>
#include <stdint.h>

#define MYDS4_ABI 18

// ---- the process ----

typedef struct {
  int abi;  // MYDS4_ABI
  double memory_gb;  // upper limit of model memory, excluding KV; 0: none (the free memory decides)
  double reserve_gb; // memory kept available to other programs; 0: default 10 GiB
  // The log's lines, each with its time, go to on_log instead of stderr when
  // it is set. It runs on the thread that logs, one line at a time under the
  // log's lock: it must return quickly and must not call myds4_log.
  void (*on_log)(const char *line);
  // Runs before a fatal error's message on stderr and the process's exit
  // (the live view restores the terminal), on the failing thread; or null.
  void (*on_die)(void);
} myds4_process;

// Installs the log, the stop's signal handlers and the failure hook, on the
// calling thread, which from then on is the engine's: it needs a stack of at
// least 16 MiB, and it gets the user-interactive QoS class, so that the
// models' GPU work goes before background work. 0, or -1 when called twice or
// with another ABI.
int myds4_init(const myds4_process *p);

// Whether a stop was requested (SIGINT, SIGTERM or myds4_request_stop), and
// a request for one. Each request counts: myds4_stop_requests says how many
// there were (the server stops saving states at quit at the second).
int myds4_stopping(void);
void myds4_request_stop(void);
int myds4_stop_requests(void);

// A line for the log, with the time: on stderr, or to on_log.
void myds4_log(const char *line);

// ---- models ----

typedef enum { MYDS4_DEEPSEEK = 1, MYDS4_QWEN = 2, MYDS4_NEMOTRON = 3 } myds4_architecture;

typedef struct {
  const char *experts;       // GGUF file or shard; null uses checkpoint experts
  int cache_half_life;
  int cache_read_cost;
  float substitute;
  int draft;                // DSpark proposals, 0 to 5
  int draft_cache;
  int online_predictor;
  const char *predictor_state;
  int ced;
  int prefill_prune;
  int runahead;
  int pipeline;
} myds4_deepseek_config;

typedef struct { int draft; } myds4_qwen_config; // chained MTP proposals, 0 to 5
typedef struct { int draft; const char *draft_model; } myds4_nemotron_config;

// Common settings and one architecture's options. The tag must match the
// checkpoint. Each family validates its own option group.
typedef struct {
  int abi;
  const char *model;
  myds4_architecture architecture;
  int ctx;
  int lookup;
  union {
    myds4_deepseek_config deepseek;
    myds4_qwen_config qwen;
    myds4_nemotron_config nemotron;
  } options;
} myds4_config;

typedef struct myds4_model myds4_model;
typedef struct myds4_media myds4_media;
typedef struct myds4_sampler myds4_sampler;

// Registers a checkpoint and opens its tokenizer without loading weights.
// Null for invalid settings or a second registration of the same family.
myds4_model *myds4_open(const myds4_config *c);
// Makes the registered model resident on the engine thread; reloads evicted weights.
int myds4_load(myds4_model *m);
int myds4_loaded(const myds4_model *m);
void myds4_active(myds4_model *m, int active); // pin the model for admitted requests

// After the last request: what the model saves or reports on exit.
void myds4_finish(myds4_model *m);

// While the model runs no request, about once a second: memory goes back to
// other programs if they need it. The model's expert cache slots after it (0
// without one).
int myds4_idle(myds4_model *m);
// While the model runs no request: every idle part of its memory goes back,
// its expert cache before its weights. Its KV states stay, and the next
// myds4_load reads the weights again. Returns the bytes given back.
uint64_t myds4_unload(myds4_model *m);
uint64_t myds4_reclaim(uint64_t bytes);
// Process controls run on the engine thread, between completed model calls.
typedef struct {
  uint64_t limit, model_bytes, kv_bytes;
} myds4_memory_info;
myds4_memory_info myds4_memory(void);
void myds4_memory_limit(uint64_t bytes);
void myds4_drain(void); // engine-thread boundary: GPU queues and expert reads

// The model's limits: rows of different states that myds4_rows takes at
// once, the drafts per pass it makes itself (the selected model options' draft field, at most its
// draft head's), the most rows one verify pass takes, the positions a state
// can continue before its end (the rewind margin), and the vocabulary. And
// whether a token ends a reply.
int myds4_max_rows(const myds4_model *m);
int myds4_draft_block(const myds4_model *m);
int myds4_max_verify(const myds4_model *m);
int myds4_rewind(const myds4_model *m);
int myds4_vocab(const myds4_model *m);
int myds4_stop(const myds4_model *m, int32_t token);
// The most rows one myds4_prefill call should take, so that other requests'
// decoding passes run between a long prompt's calls (each from where the last
// ended); 0: the whole prompt in one call (DeepSeek, whose replay, CED and
// pruning plan the whole prompt).
int myds4_prefill_rows(const myds4_model *m);

// ---- attention states and snapshots ----

// One sequence's state, of one model.
typedef struct ModelState myds4_state;
myds4_state *myds4_state_new(myds4_model *m);  // empty
void myds4_state_free(myds4_model *m, myds4_state *s);
// The bytes a state (null: a new one) takes holding that many positions.
size_t myds4_state_bytes(const myds4_model *m, const myds4_state *s, int positions);
// The rows of src before positions that later positions do not change, into
// dst. src may be a running request's state.
void myds4_state_copy(myds4_model *m, myds4_state *dst, const myds4_state *src, int positions);

// A state's rolling part at one position: what later positions overwrite.
// A state continues at the position of a snapshot restored into it, or up to
// myds4_rewind() positions before it, after the rows before that position
// were copied into it from the snapshot's own state. Its rings hold
// myds4_rewind() positions' rows before each window for that. A row written
// past where the state then continues takes the place of one of them, the
// oldest: a verify pass's rows after the tokens myds4_accept keeps, and a
// state's rows after the end of a prompt that continued it before them. Each
// row written at the state's end replaces one again. A state (and a snapshot
// of it) continues at most myds4_rewind() less these stale rows before its
// position; the caller counts them (server/src/kv.rs).
typedef struct ModelSnapshot myds4_snapshot;
// s's rolling rows now, as at pos, and with with_logits the logits the last
// prefill or decoding pass returned (those pos's token is sampled from).
myds4_snapshot *myds4_snapshot_take(myds4_model *m, myds4_state *s, int pos, int with_logits);
void myds4_snapshot_restore(myds4_model *m, myds4_state *s, const myds4_snapshot *x);
void myds4_snapshot_free(myds4_model *m, myds4_snapshot *x);
size_t myds4_snapshot_bytes(const myds4_model *m, int with_logits);
const float *myds4_snapshot_logits(const myds4_model *m, const myds4_snapshot *x);  // null without

// ---- saved states (the server's SSD KV store, server/src/disk.rs) ----

// The server names, opens, syncs and renames the files; these calls write
// and read their contents (src/engine/state_file.c): uncached (F_NOCACHE),
// with a CRC32C for each 4 MiB block. A saved state is a list of parts.
// One part: the rows of s (holding at least `to` positions) that positions
// from `from` on wrote, up to `to` (the rows the first `from` positions leave
// unchanged are in earlier parts). The bytes written, or -1 (a write failed,
// or a second stop request ended it). With fd -1 nothing is written: the
// bytes it would take (the same for myds4_snapshot_write).
int64_t myds4_state_write(myds4_model *m, myds4_state *s, int from, int to, int fd);
// Fills s, a new state, with the rows before `positions` from a saved
// state's parts, in the order written (each part's rows replace the earlier
// parts' from its own start), as myds4_state_copy of that many positions
// would; a snapshot is restored into it next, or a prefill replays. 0, or -1
// when a part is short, damaged or of another shape, or the parts leave rows
// out (s is then of no use).
int myds4_state_read(myds4_model *m, myds4_state *s, int positions, const int *fds, int count);
// Snapshot x, or (x null) s's rolling rows as at pos (its end), without
// logits. The bytes written, or -1.
int64_t myds4_snapshot_write(myds4_model *m, const myds4_snapshot *x, myds4_state *s, int pos, int fd);
// A snapshot myds4_snapshot_write wrote; null when the file is short,
// damaged or of another shape.
myds4_snapshot *myds4_snapshot_read(myds4_model *m, int fd);
// What m's saved states depend on, available before its weights load: its
// family's state layout, its checkpoint's weights (the config.json,
// .safetensors and .gguf files of the folder, by name, size and first and
// last 64 KiB), the experts' GGUF set and DFlash2's checkpoint, and the
// options that change a state's rows or shapes (draft, lookup, ced,
// prefill_prune, substitute). A state saved with another identity is not
// read.
void myds4_state_identity(myds4_model *m, uint8_t out[32]);

// ---- running a model ----

typedef struct {
  int chunks, tokens, decoder_tokens, ced_applied, prune_applied, prefix_tokens;
} myds4_prefill_info;

// Prefills positions from .. n - 1 of a prompt in s (tokens: all n of them)
// and returns the last one's logits (vocabulary floats, valid until the next
// call that runs a model). Without replay the state holds every position
// before from (its own, or copied rows and a restored snapshot at from).
// With replay it holds only the rows myds4_state_copy copies before from,
// and the prefill rebuilds the rest (myds4_replay_rows -1: the model cannot;
// for DeepSeek, exact with --ced off: the prefill starts earlier, each layer
// where its window needs; bounded and approximate with --ced, as DeepInfer's
// encoder SWA bounded replay). With capture >= 0 (and captured not null),
// *captured gets the rolling rows at that position as a snapshot taken
// during the prefill: a replay's branch point (from), or a position after
// from and before n. It gets null when the model cannot take a whole state's
// snapshot there (DeepSeek under CED compaction). info (if not null) gets the
// prefill's plan. The server supplies validated token IDs, positions and row
// counts within the limits above. Prompt IDs may include the negative image
// keys from myds4_media_expand. Logit calls return null on cancellation or
// when the required model workspace cannot fit the shared budget.
const float *myds4_prefill(myds4_model *m, myds4_state *s, const int32_t *tokens, int n, int from, int replay,
                           int capture, myds4_snapshot **captured, myds4_prefill_info *info, const myds4_media *images);
// What a replay at k costs beyond the prompt's new rows: the rows before k it
// computes again, per layer on average (-1: the model cannot replay); and the
// first position it computes again, from which the prompt's images must be
// encoded (myds4_media_encode).
int myds4_replay_rows(const myds4_model *m, int k);
int myds4_replay_start(const myds4_model *m, int k);
// One token for each of count states at once (count <= myds4_max_rows(), no
// drafts): row r is tokens[r] at pos[r], states[r]'s next position. Rows of
// logits.
const float *myds4_rows(myds4_model *m, myds4_state **states, const int32_t *tokens, const int32_t *pos, int count);
// The model's drafts: n <= myds4_draft_block() drafts after token at pos of
// s. A verify pass of count rows from pos (the token, then its drafts):
// rows of logits. myds4_accept keeps the first n of the rows. Draft returns
// sampled proposal probabilities, or null for deterministic proposals.
const double *myds4_draft(myds4_model *m, myds4_state *s, int32_t token, int pos, int32_t *drafts, int n, myds4_sampler *sampler);
const float *myds4_verify(myds4_model *m, myds4_state *s, const int32_t *rows, int count, int pos);
void myds4_accept(myds4_model *m, myds4_state *s, const int32_t *rows, int n, int pos);
// s's token history at pos from the tokens before it (DeepSeek's Engram).
void myds4_seek(myds4_model *m, myds4_state *s, const int32_t *tokens, int pos);

// ---- sampling ----

// Null for invalid settings: temperature must be finite and nonnegative (0
// is greedy), top_p in [0, 1], top_k nonnegative (0: every token). Without
// seed_set the seed is random.
myds4_sampler *myds4_sampler_new(double temperature, double top_p, int top_k, uint64_t seed, int seed_set);
void myds4_sampler_free(myds4_sampler *s);
int32_t myds4_sample(const myds4_model *m, const float *logits, myds4_sampler *s);
// Rows of logits for the token and its n drafts: the next token, and in
// *accepted how many drafts it kept; a token that ends a reply ends the kept
// ones. q is the draft result, or null for deterministic context-copy proposals.
int32_t myds4_sample_verify(const myds4_model *m, const float *logits, const int32_t *drafts, int n, const double *q, myds4_sampler *s,
                            int *accepted);

// ---- context-copy drafts (lookup) ----

typedef struct ContextLookup myds4_lookup;
myds4_lookup *myds4_lookup_new(int capacity);  // null if it cannot hold that many tokens
void myds4_lookup_free(myds4_lookup *x);
int myds4_lookup_append(myds4_lookup *x, const int32_t *tokens, int n);  // 0 when full
int myds4_lookup_propose(const myds4_lookup *x, int max, int32_t *drafts);  // how many
void myds4_lookup_feedback(myds4_lookup *x, int offered, int accepted);

// ---- encoded media ----

// A prompt's media inputs: each placeholder becomes its feature span.
// Null with a message when an input cannot be decoded.
myds4_media *myds4_media_open(myds4_model *m, const uint8_t *const *data, const size_t *len, int n, char *message,
                                size_t cap);
// The prompt with the spans in place into out (at most max tokens): its
// length, or -1 if its places and images differ or it does not fit.
int myds4_media_expand(myds4_media *x, const int32_t *in, int n, int32_t *out, int max);
// Encodes inputs that positions from .. reach and gives all of them to
// the model's next prefill; the count encoded.
int myds4_media_encode(myds4_media *x, int from);
void myds4_media_close(myds4_media *x);  // after that prefill

// ---- statistics, for the server's status ----

typedef struct {
  uint64_t hits, misses, prefetch_hits, prefetches, bytes, evictions, late, cancelled, substituted, remaps, given_back;
  uint64_t row_own, row_ahead, row_late, row_sub, row_ssd;  // decoded rows' selections by where the expert was
  double read_s;                                           // waiting for reads
} myds4_cache_counts;

// A model's counters (its expert cache's: zero without one) and the
// process's memory.
typedef struct {
  myds4_cache_counts cache, decode, prefill;  // all of the cache's; of decoded rows; of prompt rows
  uint64_t decode_rows, prefill_rows;
  double decode_seconds, decode_gpu_seconds;
  int prefill_layer, prefill_rows_now;  // the prefill running: its layer (0: none) and rows
  int slots, mapped, cap, max_slots, experts, uncached;  // the expert cache's slots, and experts on the SSD only
  uint64_t slot_bytes, uncached_bytes, cache_bytes, dense_bytes;
  uint64_t footprint, gpu_in_use, available;
  int64_t gpu_limit;  // Metal's working-set recommendation; INT64_MAX before GPU initialization
  uint64_t total, free, file, file_kept, wired, compressor, compressed, swap_used, swapouts;
  int pressure;  // 1 normal, 2 warn, 4 critical
  int layers, experts_per_row;  // 0 without routed experts
  // the layers with routed experts (experts_per_row / routed_layers chosen
  // in each), and each one's experts
  int routed_layers, experts_per_layer;
  // model_memory_limit: the most model memory now, what it holds and the free
  // memory above the reserve, within --memory
  uint64_t model_memory_limit, model_memory_used, model_memory_committed, model_memory_peak;
  uint64_t kv_allocated, model_workspace;
} myds4_stats;
void myds4_refresh_stats(void); // sample process memory once for all models
void myds4_stats_get(const myds4_model *m, myds4_stats *s);
// The experts that row `row` of the model's latest decoding pass chose in
// each layer with routed experts, experts_per_row / routed_layers per layer,
// and where each one was: 0 in memory, 1 read ahead, 2 late, 3 substituted,
// 4 from the SSD. Returns 0,
// and leaves both unset, for a model without routed experts or a row that
// did not run every layer. From the engine thread, after the pass.
int myds4_routes(const myds4_model *m, int row, uint16_t *experts, uint8_t *sources);

// ---- tokenizers (tokenizer.json), from any thread ----

// A checkpoint's tokenizer: a model's own (myds4_model_tokenizer), or one opened
// from a checkpoint directory without its model, with `image` (or null) the
// spelling of the model's image token. A tokenizer.json the engine cannot run
// ends the process with a message.
typedef struct Tokenizer myds4_tokenizer;
myds4_tokenizer *myds4_model_tokenizer(const myds4_model *m);
myds4_tokenizer *myds4_tokenizer_open(const char *dir, const char *image);
// The token ids of len bytes of text, at most max: mode 0 text alone, 1 with
// the added tokens' spellings taken as those tokens, 2 a rendered prompt (as
// 1, and the byte 0xFF an image's place). Returns the count, or -1 if more
// than max.
int myds4_tokenize(myds4_tokenizer *t, const char *text, size_t len, int mode, int32_t *out, int max);
// A token's bytes (len 0 for an unknown id), whether it is one of the
// tokenizer's added tokens (2: one tokenizer.json marks special, 1: another,
// 0: none), and the id of an added token's spelling (-1 for none).
const char *myds4_piece(const myds4_tokenizer *t, int32_t id, size_t *len);
int myds4_added(const myds4_tokenizer *t, int32_t id);
int32_t myds4_special(const myds4_tokenizer *t, const char *text, size_t len);
