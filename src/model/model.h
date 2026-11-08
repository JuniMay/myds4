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

// Common execution settings and family-specific options. The family is
// resolved from the checkpoint before loading.
typedef struct {
  double budget_gb;   // expert cache budget, 0 = automatic
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
typedef struct {
  enum { MODEL_UNSPECIFIED, MODEL_DEEPSEEK, MODEL_QWEN } family;
  int ctx;            // capacity in tokens
  double reserve_gb;  // standalone CLI process reserve; server leaves 0 after setting its global reserve
  int dump_fd;        // -1, or a pipe receiving per-layer streams
  int routes_fd;      // -1, or a pipe receiving every decoded token's routes, predictions and router logits
  int draft;          // enable the model's native draft head(s)
  int lookup;         // context-copy candidates; bounded by the model's verification capacity
  int quiet;                   // no progress lines on stderr
  union { DeepSeekOptions deepseek; QwenOptions qwen; };
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
  uint64_t dense_bytes;                 // the weights always resident
  CacheStats decode, prefill;           // the expert cache's, of decoded rows and of prompt rows
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
  const char *image_token;        // the spelling of its image token (IMAGE_MARK's), or null
  // Why the options do not suit the family (an option it has no use for), or
  // null; before load.
  const char *(*refuse)(const Options *o);
  void (*load)(const char *dir, const Options *o);
  void (*counters)(ModelCounters *c);
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
  void (*draft)(ModelState *s, int token, int pos, int *drafts, int n);
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
  // The vision encoder, when the model has one (null otherwise): an image's
  // feature rows, and the images that the next prefill's image positions take.
  int (*prepare_image)(const char *dir, Image *im, const void *data, size_t len, const char **error);
  void (*image)(Image *im);
  void (*images)(const Image *im, int n);
  // The experts row `row` of the latest decoding pass chose and where each
  // was (myds4_routes); null without routed experts.
  int (*routes)(int row, uint16_t *id, uint8_t *from);
  // One user turn in the model's chat template, for the CLI: `images` image
  // places, then the prompt, with thinking at effort `think` (1 to 100) or
  // off (0). Null when the CLI cannot render this model's chat.
  void (*chat)(char *text, size_t cap, const char *prompt, int images, int think);
  // The CLI's --bench: the kernels timed on the model's own matrices; or null.
  void (*bench)(void);
} ModelOps;

typedef struct {
  const ModelOps *ops;
  const char *dir;
} Model;

// The family that runs the checkpoint directory dir (its config.json's
// model_type), and the model of dir, loaded. Both end the process with a
// message when no family runs it.
const ModelOps *model_find(const char *dir);
Model *model_open(const char *dir, const Options *o);

extern const ModelOps deepseek_ops, qwen_ops;
