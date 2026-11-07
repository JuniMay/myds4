// myds4's inference engine behind a C ABI, for the server (server/, in Rust).
// The engine runs the model: loading, the attention states and their
// snapshots, prefill and decoding, DSpark drafts and their verification,
// sampling, context-copy drafts, the vision encoder, the tokenizer, the log
// and the stop on SIGINT or SIGTERM. The server owns everything about
// requests: their queue and scheduling, the KV prefix cache (which states and
// snapshots to keep, and how a prompt reuses them), HTTP, and the status and
// view JSON, which it builds from myds4_stats.
//
// Threads: myds4_load and every function on states, snapshots, prefill,
// decoding, drafts, samplers, context lookups and images run on one thread,
// the server's worker: the GPU code keeps state per thread, the model's work
// buffers are shared, and one attention state is in use at a time. The
// tokenizer's functions, myds4_log, myds4_stopping, myds4_request_stop and
// myds4_stats_get may be called from any thread; myds4_stats_get reads the
// counters as they are, without a lock, so a prefill's progress shows while
// it runs.
#pragma once
#include <stddef.h>
#include <stdint.h>

#define MYDS4_ABI 7

// Select concurrent runahead before loading. Zero uses ordinary batching.
// Returns -1 for invalid values or after loading. The config ABI is unchanged.
int myds4_set_runahead(int enabled);

// The engine's settings: the server's model and engine options.
typedef struct {
  int abi;                     // MYDS4_ABI
  const char *model;           // -m: the checkpoint directory
  const char *experts;         // --experts: a GGUF file or any shard of a set; null: the checkpoint's
  int ctx;                     // --ctx: context capacity in tokens, 1 to 1,048,576
  double cache_gb;             // --cache: expert cache budget in GB; 0: from available memory
  double reserve_gb;           // --reserve: memory the cache leaves to other programs in GiB; 0: 10
  int cache_half_life;         // --cache-half-life in forward passes; 0: the default, 50
  int cache_read_cost;         // --cache-read-cost
  float substitute;            // --substitute D; 0: exact
  int draft;                   // --draft N, 0 to 5
  int draft_cache;             // --draft-cache N; 0: all draft experts preloaded
  int online_predictor;        // --online-predictor
  const char *predictor_state; // --predictor-state FILE, or null
  int ced;                     // --ced off|N: 0 or 128 to 1048576 decoder replay tokens; approximate
  int lookup;                  // --lookup off|N: 0 or 1 to 15 context-copy drafts
  int prefill_prune;           // --prefill-prune: preserve dependencies while skipping unused decoder batches
  // The log's lines, each with its time, go to on_log instead of stderr when
  // it is set. It runs on the thread that logs, one line at a time under the
  // log's lock: it must return quickly and must not call myds4_log.
  void (*on_log)(const char *line);
  // Runs before a fatal error's message on stderr and the process's exit
  // (the live view restores the terminal), on the failing thread; or null.
  void (*on_die)(void);
} myds4_config;

// Loads the tokenizer and the model and installs the stop's signal handlers,
// on the calling thread, which from then on is the engine's: it needs a stack
// of at least 16 MiB, and it gets the user-interactive QoS class, so that the
// model's GPU work goes before background work. 0, or -1 with a message on
// stderr for invalid settings. A model that cannot be loaded ends the process
// (after on_die).
int myds4_load(const myds4_config *c);

// After the last request: saves --predictor-state and reports the online
// predictor.
void myds4_finish(void);

// Whether a stop was requested (SIGINT, SIGTERM or myds4_request_stop), and
// a request for one.
int myds4_stopping(void);
void myds4_request_stop(void);

// While no request runs, about once a second: the expert cache gives slots
// back if other programs need the memory. The slots it has afterwards.
int myds4_idle(void);

// A line for the log, with the time: on stderr, or to on_log.
void myds4_log(const char *line);

// The model's limits: rows of different states that myds4_rows takes at
// once (6), DSpark's drafts per pass, the most rows one verify pass takes,
// the rows a state can rewind (the rewind margin: 8, or --draft's or
// --lookup's drafts per pass when more), and the vocabulary.
int myds4_max_rows(void);
int myds4_draft_block(void);
int myds4_max_verify(void);
int myds4_rewind(void);
int myds4_vocab(void);

// ---- attention states and snapshots ----

// One sequence's attention state: the window rings, compressed rows, index
// keys and compressor rows of the layers, the drafts' windows and the Engram
// history. The layers work on one state at a time: the one in use.
typedef struct State myds4_state;
myds4_state *myds4_state_new(void);    // empty
void myds4_state_free(myds4_state *s);  // not the one in use
// The bytes a state (null: a new one) takes holding that many positions.
size_t myds4_state_bytes(const myds4_state *s, int positions);
// The global rows (compressed rows, index keys) of src before positions, into
// dst, which is not in use. src may be a running request's state: no later
// step changes the rows before its position.
void myds4_state_copy(myds4_state *dst, const myds4_state *src, int positions);
void myds4_state_use(myds4_state *s);
void myds4_reset(void);  // the state in use, empty

// A state's rolling part at one position: the rows that later positions
// overwrite (window rings, compressor rows, draft rings) and the Engram
// history. A state continues at the position of a snapshot restored into it,
// or up to myds4_rewind() positions before it, after the global rows before
// that position were copied into it from the snapshot's own state. Its rings
// hold myds4_rewind() positions' rows before each window for that. A row
// written past where the state then continues takes the place of one of
// them, the oldest: a verify pass's rows after the tokens myds4_accept keeps,
// and a state's rows after the end of a prompt that continued it before
// them. Each row written at the state's end replaces one again. A state (and
// a snapshot of it) continues at most myds4_rewind() less these stale rows
// before its position; the caller counts them (server/src/kv.rs).
typedef struct Snapshot myds4_snapshot;
// s's rolling rows now, as at pos, and with with_logits the logits the last
// prefill or step returned (those pos's token is sampled from).
myds4_snapshot *myds4_snapshot_take(myds4_state *s, int pos, int with_logits);
void myds4_snapshot_restore(myds4_state *s, const myds4_snapshot *x);
void myds4_snapshot_free(myds4_snapshot *x);
size_t myds4_snapshot_bytes(int with_logits);
const float *myds4_snapshot_logits(const myds4_snapshot *x);  // null without

// ---- running the model, in the state in use ----

typedef struct {
  int chunks, tokens, decoder_tokens, ced_applied, prune_applied, prefix_tokens;
} myds4_prefill_info;

// Prefills positions from .. n - 1 of a prompt (tokens: all n of them) and
// returns the last one's logits (vocabulary floats, valid until the next
// call that runs the model). Without replay the state holds every position
// before from (its own, or copied rows and a restored snapshot at from).
// With replay it holds only the global rows before from (myds4_state_copy),
// and the prefill rebuilds the rolling rows at from. With --ced off the
// replay is exact: the prefill starts earlier, each layer where its window
// needs. With --ced it is bounded and approximate, as DeepInfer's encoder SWA
// bounded replay: every layer computes again the ring's positions before from
// (136, with a wider --lookup more), its windows cut off at the first of them, and
// the copied global rows of groups completed before from stay as they are.
// With capture >= 0 (and captured not null), *captured gets the rolling rows
// at that position as a snapshot taken during the prefill: a replay's branch
// point (from), or a position after from and before n. It gets null when the
// capture holds only layers 0-20: under CED compaction, when the decoder
// layers start less than a ring before it. info (if not null) gets the
// prefill's plan. This and the calls below that return logits return null for
// invalid arguments: a token outside the vocabulary (a prompt may also hold
// the negative image keys of myds4_images_expand), positions outside the
// context, a capture outside the prefill, more rows than the limits above.
const float *myds4_prefill(const int32_t *tokens, int n, int from, int replay, int capture,
                           myds4_snapshot **captured, myds4_prefill_info *info);
// What a replay at k costs beyond the prompt's new rows: the rows before k it
// computes again, per layer on average (exact: about 2,613 for long prefixes,
// at most k; bounded, with --ced: the ring's 136, at most k); and the first
// position it computes again, from which the prompt's images must be encoded
// (myds4_images_encode). From the worker's thread.
int myds4_replay_rows(int k);
int myds4_replay_start(int k);
// One token at pos, decoded; its logits.
const float *myds4_step(int32_t token, int pos);
// One token for each of m states at once (m <= myds4_max_rows(), no
// drafts): row r is tokens[r] at pos[r] of states[r]. Rows of logits.
const float *myds4_rows(myds4_state **states, const int32_t *tokens, const int32_t *pos, int m);
// DSpark: n <= myds4_draft_block() drafts after token at pos.
void myds4_draft(int32_t token, int pos, int32_t *drafts, int n);
// A verify pass of m rows from pos (the token, then its drafts): rows of
// logits. myds4_accept keeps the first n of the rows.
const float *myds4_verify(const int32_t *rows, int m, int pos);
void myds4_accept(const int32_t *rows, int n, int pos);
// The Engram history at pos from the tokens before it.
void myds4_seek(const int32_t *tokens, int pos);

// ---- sampling ----

typedef struct myds4_sampler myds4_sampler;
// Null for invalid settings: temperature must be finite and nonnegative (0
// is greedy), top_p in [0, 1]. Without seed_set the seed is random.
myds4_sampler *myds4_sampler_new(double temperature, double top_p, uint64_t seed, int seed_set);
void myds4_sampler_free(myds4_sampler *s);
int32_t myds4_sample(const float *logits, myds4_sampler *s);
// Rows of logits for the token and its n drafts: the next token, and in
// *accepted how many drafts it kept; eos ends the kept ones.
int32_t myds4_sample_verify(const float *logits, const int32_t *drafts, int n, int32_t eos, myds4_sampler *s,
                            int *accepted);

// ---- context-copy drafts (--lookup) ----

typedef struct ContextLookup myds4_lookup;
myds4_lookup *myds4_lookup_new(int capacity);  // null if it cannot hold that many tokens
void myds4_lookup_free(myds4_lookup *x);
int myds4_lookup_append(myds4_lookup *x, const int32_t *tokens, int n);  // 0 when full
int myds4_lookup_propose(const myds4_lookup *x, int max, int32_t *drafts);  // how many
void myds4_lookup_feedback(myds4_lookup *x, int offered, int accepted);

// ---- images ----

// A prompt's images: each image place (the token MYDS4_IMAGE_TOKEN) becomes
// its image's span of tokens. Null with a message for an image that cannot be
// read.
#define MYDS4_IMAGE_TOKEN 129264
typedef struct myds4_images myds4_images;
myds4_images *myds4_images_open(const uint8_t *const *data, const size_t *len, int n, char *message, size_t cap);
// The prompt with the spans in place into out (at most max tokens): its
// length, or -1 if its places and images differ or it does not fit.
int myds4_images_expand(myds4_images *x, const int32_t *in, int n, int32_t *out, int max);
// Encodes the images that positions from .. reach and gives all of them to
// the next prefill; the count encoded.
int myds4_images_encode(myds4_images *x, int from);
void myds4_images_close(myds4_images *x);  // after that prefill

// ---- statistics, for the server's status and live view ----

typedef struct {
  uint64_t hits, misses, prefetch_hits, prefetches, bytes, evictions, late, cancelled, substituted, remaps, given_back;
  uint64_t row_own, row_ahead, row_late, row_sub, row_ssd;  // decoded rows' selections by where the expert was
  double read_s;                                           // waiting for reads
} myds4_cache_counts;

typedef struct {
  myds4_cache_counts cache, decode, prefill;  // all of the cache's; of decoded rows; of prompt rows
  uint64_t decode_rows, prefill_rows;
  double decode_seconds, decode_gpu_seconds;
  int prefill_layer, prefill_rows_now;  // the prefill running: its layer (0: none) and rows
  int slots, mapped, cap, max_slots, experts, uncached;  // the expert cache's slots, and experts on the SSD only
  uint64_t slot_bytes, uncached_bytes, cache_bytes, dense_bytes;
  uint64_t footprint, gpu_in_use, available;
  int64_t gpu_limit;  // INT64_MAX: none
  uint64_t total, free, file, file_kept, wired, compressor, compressed, swap_used, swapouts;
  int pressure;  // 1 normal, 2 warn, 4 critical
  int layers, experts_per_row;
} myds4_stats;
void myds4_stats_get(myds4_stats *s);
// For the live view: the experts that row `row` of the latest decode step
// (myds4_step: row 0; myds4_rows: the row of each state; myds4_verify: row 0
// is the token, the others its drafts) chose in each layer, 6 per layer,
// layer by layer (layers * 6 ids), and where each one was: 0 in memory, 1 read
// ahead, 2 late, 3 substituted, 4 from the SSD. Returns 0, and leaves both
// unset, when the row did not run every layer in that step. From the worker's
// thread, after the step.
int myds4_routes(int row, uint16_t *experts, uint8_t *sources);

// ---- the checkpoint's tokenizer (tokenizer.json), from any thread ----

// Loads it without the model (myds4_load loads it too) and returns 0; a
// tokenizer.json that cannot be read ends the process with a message.
int myds4_tokenizer(const char *dir);
// The token ids of text: chat 1 as a rendered prompt (a special token's
// spelling is that token, the byte 0xFF an image's place), chat 0 as text
// without special tokens. Returns the count, or -1 if more than max.
int myds4_tokenize(const char *text, size_t len, int chat, int32_t *out, int max);
// A token's bytes (len 0 for an unknown id), whether it is one of the
// tokenizer's added tokens (the special tokens among them), and the id of a
// special token's spelling (-1 for none).
const char *myds4_piece(int32_t id, size_t *len);
int myds4_added(int32_t id);
int32_t myds4_special(const char *text, size_t len);
