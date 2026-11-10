// Every model behind one interface. A model family (src/model/deepseek/model.c,
// src/model/qwen/model.c) fills a ModelOps table; model_open reads a checkpoint's
// config.json, picks the family that runs its model_type, and loads it. The
// engine (src/engine/engine.c), the CLI (src/cli/main.c) and the checks run
// any checkpoint through these calls.
//
// A process holds at most one model of each family, and every call on models
// and their states runs on one thread: the GPU code keeps state per thread,
// and a family's buffers are its own.
#pragma once
#include "myds4.h"
#include "../engine/memory.h"

// Common execution settings and family-specific options. The family is
// resolved from the checkpoint before loading.
typedef struct {
  int cache_half_life;  // score half-life in forward passes; 0 = default 50
  int cache_read_cost;  // weight decode eviction priorities by aligned read bytes / slot size
  int draft_cache;    // draft expert slots per stage; 0 = all 128 preloaded
  const char *predictor_state; // load or create online state; save on clean completion
  float substitute;            // > 0: replace a missing expert by a resident one up to this far below in key
  const char *experts;         // a GGUF file (set) whose routed experts replace the checkpoint's, or null
  int online_predictor;        // rank64 online prefetch corrector; 0 disables it
  int ced;                    // 0: no approximate replay; otherwise decoder replay tokens (at least WINDOW)
  int prefill_prune;          // skip decoder batches outside the final and saved-prefix dependencies
  int runahead;               // exact concurrent requests: group ready jobs while other expert reads run
  int pipeline;               // experimental native MXFP4 stages across concurrent requests
} DeepSeekOptions;
typedef struct { int reserved; } QwenOptions;
typedef struct { const char *draft_model; } NemotronOptions;
typedef struct {
  enum { MODEL_UNSPECIFIED, MODEL_DEEPSEEK, MODEL_QWEN, MODEL_NEMOTRON } family;
  int ctx;            // capacity in tokens
  int dump_fd;        // -1, or a pipe receiving per-layer streams
  int routes_fd;      // -1, or a pipe receiving every decoded token's routes, predictions and router logits
  int draft;          // enable the model's native draft head(s)
  int lookup;         // context-copy candidates; bounded by the model's verification capacity
  int quiet;                   // no progress lines on stderr
  union { DeepSeekOptions deepseek; QwenOptions qwen; NemotronOptions nemotron; };
} Options;

typedef struct ModelState ModelState;        // one sequence: a family's own state
typedef struct ModelSnapshot ModelSnapshot;  // a state's rolling part at one position

// A model's counters since it loaded, for the server's status. The expert
// cache's (io.c) belong to a model with routed experts (experts_per_row > 0).
typedef struct {
  uint64_t decode_rows, prefill_rows;  // rows of decoding and prompt passes
  double decode_seconds, decode_gpu_seconds;
  int prefill_layer, prefill_rows_now;  // the prompt pass running: its layer (0: none) and rows
  int layers, experts_per_row;          // 0 without routed experts
  uint64_t table_layers;                // layer passes that reached every expert through the layer's table
  uint64_t dense_bytes;                 // the weights always resident
  CacheStats decode, prefill;           // the expert cache's, of decoded rows and of prompt rows
  CacheStats cache;
  uint64_t cache_bytes, slot_bytes, uncached_bytes;
  int cache_slots, mapped, cap, max_slots, experts, uncached;
} ModelCounters;

// The last prefill's plan: its chunks, the rows it ran (from its first row),
// the rows its later layers ran, and whether approximate or pruned prefill
// applied (DeepSeek's CED and dependency pruning).
typedef struct {
  int chunks, tokens, decoder_tokens, ced_applied, prune_applied, prefix_tokens;
} ModelPlan;

typedef struct {
  const char *family;             // the family's name, for the log
  const char *const *model_types; // config.json model_type values it runs; null-terminated
  const char *media_token;        // the model's encoded-media placeholder, or null
  MediaKind media_kind;
  // Why the options do not suit the family (an option it has no use for), or
  // null; before load.
  const char *(*refuse)(const Options *o);
  int (*load)(const char *dir, const Options *o);
  MemorySlot *(*weight_slot)(void); // one reloadable dense-weight group
  void (*active)(int active); // pin request resources; release idle guarantees
  void (*counters)(ModelCounters *c);
  uint64_t (*cache_bytes)(void); // current expert payload; null for a dense model
  void (*plan)(ModelPlan *p);  // the last prefill's, when the family plans its own; or null
  // Limits after loading: the vocabulary; rows of different states one
  // decoding pass takes; rows one verify pass takes (the token and its
  // drafts); the drafts per pass the model makes itself (Options.draft, at
  // most what its draft head makes; 0 without one); and the positions a state
  // can continue before its end (its rewind margin).
  int (*vocab)(void);
  int (*max_rows)(void);
  int (*max_verify)(void);
  int (*draft_block)(void);
  int (*rewind)(void);
  int (*stop)(int token);  // whether a sampled token ends a reply
  // The most rows one prefill call should take, so that other requests'
  // decoding passes can run between a long prompt's calls (each call from
  // where the last ended); 0: the whole prompt in one call (a family that
  // plans its prefill over the whole prompt).
  int (*prefill_rows)(void);

  ModelState *(*state_new)(void);
  void (*state_free)(ModelState *s);
  size_t (*state_bytes)(const ModelState *s, int positions);  // s (null: a new state) holding that many
  // The rows of src before `positions` that later positions do not change,
  // into dst. A running src is fine.
  void (*state_copy)(ModelState *dst, const ModelState *src, int positions);

  ModelSnapshot *(*snapshot_take)(ModelState *s, int pos, int with_logits);
  void (*snapshot_restore)(ModelState *s, const ModelSnapshot *x);
  void (*snapshot_free)(ModelSnapshot *x);
  size_t (*snapshot_bytes)(int with_logits);
  const float *(*snapshot_logits)(const ModelSnapshot *x);  // null without

  // Prefills positions from .. n - 1 of a prompt (tokens: all n of them) in
  // s and returns the last one's logits. With replay the state holds only the
  // rows state_copy copies before from, and the prefill rebuilds the rest
  // (replay_rows: -1 when the model cannot). capture >= 0 takes a snapshot at
  // that position into *captured. every, when not null, gets the logits of
  // every position from .. n - 1.
  const float *(*prefill)(ModelState *s, const int *tokens, int n, int from, int replay, int capture,
                          ModelSnapshot **captured, float *every);
  int (*replay_rows)(int k);
  int (*replay_start)(int k);
  // One token for each of m states (m <= max_rows): row r is tokens[r] at
  // pos[r] of states[r], each state's next position. Rows of logits.
  const float *(*rows)(ModelState *const *states, const int *tokens, const int *pos, int m);
  // Drafts: up to draft_block() drafts after token at pos (every position
  // before it having run); a verify pass of m rows from pos (the token, then
  // its drafts) returns each row's logits, and accept keeps the first n.
  // Returns sampled proposal probabilities (n vocabulary rows), or null for deterministic proposals.
  const double *(*draft)(ModelState *s, int token, int pos, int *drafts, int n, Sampler *sampler);
  const float *(*verify)(ModelState *s, const int *tokens, int m, int pos);
  void (*accept)(ModelState *s, const int *tokens, int n, int pos);
  // The state's token history at pos from the tokens before it, where the
  // model keeps one (a no-op otherwise).
  void (*seek)(ModelState *s, const int *tokens, int pos);
  // While no request runs, about once a second: memory goes back to other
  // programs if they need it. Returns the model's expert cache slots (0
  // without one).
  int (*idle)(void);
  // At the end: what the model saves or reports on exit.
  void (*finish)(void);
  // The model's own lines of a CLI generation's report: at 0 when decoding
  // starts (it notes its counters), at 1 after it (it prints, per token of
  // `tokens`, what it measured meanwhile). May be null.
  void (*report)(FILE *out, int at, int tokens);
  // Prepare and encode model-specific media, then bind its feature spans
  // to the next prefill. Null callbacks mean the model accepts text only.
  int (*prepare_media)(const char *dir, Media *im, const void *data, size_t len, const char **error);
  int (*encode_media)(Media *im); // 0: the global model-memory pool cannot admit it
  void (*media)(const Media *im, int n);
  // The experts row `row` of the latest decoding pass chose and where each
  // was (myds4_routes); null without routed experts.
  int (*routes)(int row, uint16_t *id, uint8_t *from);
  // Render one CLI user turn with media placeholders and the chosen reasoning effort.
  void (*chat)(char *text, size_t cap, const char *prompt, int media_count, int think);
  // The CLI's --bench: the kernels timed on the model's own matrices; or null.
  void (*bench)(void);
} ModelOps;

typedef struct {
  const ModelOps *ops;
  const char *dir;
  MemoryAccount memory;
  MemorySlot work;
} Model;

// The family that runs the checkpoint directory dir (its config.json's
// model_type), and the model of dir, loaded. Both end the process with a
// message when no family runs it.
const ModelOps *model_find(const char *dir);
Model *model_create(const char *dir);
int model_load(Model *model, const Options *options);
void model_active(Model *model, int active);
Model *model_open(const char *dir, const Options *o);

extern const ModelOps deepseek_ops, qwen_ops, flash_ops, nemotron_ops;
