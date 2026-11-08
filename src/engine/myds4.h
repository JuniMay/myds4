// myds4: DeepSeek-V4.1-Flash inference on Apple silicon, streaming routed
// experts from the checkpoint's safetensors files.
#pragma once
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

// ---- model shape (config.json of DeepSeek-V4.1-Flash) ----
enum {
  DIM = 5120, LAYERS = 40, VOCAB = 129280, HC = 4, MIX = 24,
  STREAM_BYTES = 2,  // residual streams store the reference's BF16 values
  HEADS = 64, HEAD_DIM = 512, ROPE_DIM = 64, QRANK = 1280, OGROUPS = 8, ORANK = 1024,
  WINDOW = 128, EXPERTS = 384, TOPK = 6, INTER = 2304,
  IDX_HEADS = 32, IDX_DIM = 128, IDX_TOPK = 512,
  ENGRAM_ROWS = 24, ENGRAM_DIM = 256,
  // The DSpark draft (inference/config.json: n_mtp_layers, dspark_*): three
  // stages after the backbone, each with 128 routed experts of which a token
  // uses 3, drafting BLOCK tokens from the attention inputs of the last three
  // layers (37 to 39); the drafted rows after the first are the NOISE token.
  DRAFTS = 3, DRAFT_EXPERTS = 128, DRAFT_TOPK = 3, BLOCK = 5, NOISE = 128799,
  VERIFY_DRAFTS = 15,  // target verification capacity, independent of DSpark
  // The positions a state can continue before its end or a snapshot's
  // position, for the KV prefix cache's resumes: each layer's window ring
  // keeps this many rows beyond its window. The rewind margin is at least
  // this, and at least the drafts or --lookup proposals that a verify pass
  // writes after the token (deepseek_rewind).
  REWIND = 8,
  BATCH_ROWS = 6,  // rows of different sequences that one decode pass takes (deepseek_forward_rows, --batch)
  DRAFT_CACHE_MIN = BLOCK * DRAFT_TOPK,  // one stage's largest union of native selections
  CACHE_LAYERS = LAYERS + DRAFTS,  // the expert cache's layers: the backbone's, then the stages'
};
#define NORM_EPS 1e-20f
#define HC_EPS 1e-6f

// One expert as stored: w1, w2, w3 codes back to back, then their scales.
#define EXPERT_W (3u * 5898240u)
#define EXPERT_S (3u * 368640u)

// ---- util.c ----
_Noreturn void die(const char *fmt, ...);
extern void (*die_hook)(void);  // runs before die's message (the server's live view restores the terminal)
double now(void);
#ifdef MYDS4_PREDICT_PROBE
// Diagnostic-only synchronous records on MYDS4_PROBE_FD. Pipe backpressure
// changes timing; these captures are not performance measurements.
int predict_probe_enabled(void);
int predict_probe_streams_enabled(void);
void predict_probe_emit(uint32_t kind, int pos, int layer, int rows,
                        int feature_dim, const float *features,
                        int logit_dim, const float *logits,
                        int token_count, const int *tokens);
void predict_probe_target(uint32_t kind, int pos, int layer, int rows,
                          int feature_dim, const float *features,
                          int logit_dim, const float *logits,
                          const int *tokens, const int *routes, int route_stride);
void predict_probe_streams(int pos, int layer, int rows, const float *streams, const float *mixes);
#endif
void *xmalloc(size_t n);
char *read_file(const char *path, size_t *len);
long b64_decode(const char *s, size_t n, uint8_t *out); // bytes written (at most 3 per 4 characters), -1 if not base64
void lock_model(void);     // waits until no other process holds myds4.lock beside the executable ($MYDS4_LOCK)
void shutdown_watch(void); // SIGINT/SIGTERM request a stop; handlers do no I/O
int shutdown_requested(void);
void shutdown_request(void); // a stop, as SIGINT and SIGTERM request one
// Sampling workspaces belong to the single model thread. Each non-greedy
// token consumes one draw, including tokens used to verify draft proposals.
typedef struct {
  double temp, top_p;
  unsigned short seed[3];
  int top_k;  // 0: every token
} Sampler;
int sampler_valid(const Sampler *s);
void sampler_seed(Sampler *s, uint64_t seed);
int sample_argmax(const float *logits, int count);
int sample_token(const float *logits, int count, Sampler *s);
// logits contains ndraft + 1 rows of count values; drafts contains ndraft
// tokens. The sampler is independent of the model's verification row limit.
// stop (or null): whether a token ends the reply, which also ends the kept drafts
int sample_verify(const float *logits, int count, const int *drafts, int ndraft, int (*stop)(int token), Sampler *s,
                  int *accepted);
typedef struct {
  char *p;
  size_t n, cap;
} Str; // a growing string, NUL-terminated
void str_put(Str *s, const char *t, size_t n);
void str_cat(Str *s, const char *t);
void str_printf(Str *s, const char *fmt, ...) __attribute__((format(printf, 2, 3)));

typedef struct Json Json;
struct Json {
  enum { J_NULL, J_BOOL, J_NUM, J_STR, J_ARR, J_OBJ } type;
  int n;           // children of arrays and objects, bytes of strings
  const char *key; // set on object members
  const char *s;
  double num;
  Json *kid;
};
Json *json_parse(char *text);
Json *json_get(const Json *obj, const char *key);
void json_free(Json *root); // releases the tree and its strings

typedef struct { // one tensor of the checkpoint
  int fd;
  uint64_t off, size;
  int dtype;  // 'b' bf16, 'f' f32, '8' fp8 e4m3, 'e' e8m0, 'i' packed e2m1 (I8), 'u' U8 (packed NVFP4)
  int64_t shape[4];
  int ndim;
} Tensor;
typedef struct TensorIndex TensorIndex;
TensorIndex *st_open(const char *dir);
Tensor st_get(const TensorIndex *index, const char *fmt, ...);
int st_has(const TensorIndex *index, const char *fmt, ...);
void st_close(TensorIndex *index); // only after every tensor user has finished

// GGUF files (llama.cpp's format), for --experts: a tensor's ggml type,
// dimensions (fastest first) and absolute offset. gguf_open takes one file
// or any shard of a split set (NAME-00001-of-0000N.gguf) and opens them all.
typedef struct { int fd, type; uint64_t off, ne[4]; } Gguf;
enum { GG_Q2_K = 10, GG_IQ2_XXS = 16, GG_IQ2_XS = 17, GG_IQ3_XXS = 18, GG_IQ1_S = 19, GG_IQ2_S = 22, GG_IQ1_M = 29 };
void gguf_open(const char *path);
int gguf_get(const char *name, Gguf *t);  // 0 if absent

// ---- gpu.c: Metal through the Objective-C runtime ----
typedef struct {
  void *obj;     // id<MTLBuffer>
  uint8_t *ptr;  // CPU address of offset 0
  uint64_t gpu;  // GPU address of offset 0
  size_t size;
  uint32_t off;  // byte offset of the payload inside the buffer
} Buf;
void gpu_init(const char *source, size_t len);
void gpu_start(void);  // gpu_init with every model's kernels (src/kernels/library.c), once per process
int gpu_kernel(const char *name);
Buf gpu_alloc(size_t size);  // resident from the start
Buf gpu_new(size_t size);    // outside the residency set
void gpu_free(Buf *b);
// Membership of a buffer in the residency set of every command buffer; the
// changes apply at gpu_residency_commit (gpu_alloc commits its own), which
// takes about 0.1 ms with thousands of buffers in the set.
void gpu_resident(const Buf *b, int on);
void gpu_residency_commit(void);
uint64_t gpu_working_set(void);
void *gpu_open(void);        // a command buffer that runs after those opened before
void gpu_encode(void *cmd);  // encode into it until gpu_commit or the next gpu_encode
void gpu_begin(void);        // gpu_encode(gpu_open())
void *gpu_close(void);       // ends the current command buffer's encoding without committing it
void gpu_submit(void *cmd);
void gpu_kern(int k);
void gpu_buf(int idx, const Buf *b, size_t off);
void gpu_bytes(int idx, const void *p, size_t n);
void gpu_use(const Buf *b);
void gpu_use_rw(const Buf *b);  // indirect read/write access, including hazard tracking
typedef struct { const Buf *from, *to; size_t from_offset, to_offset, bytes; } GpuCopy;
void gpu_copies(const GpuCopy *copies, int count);
void gpu_go(uint32_t gx, uint32_t gy, uint32_t gz, uint32_t tx, uint32_t ty, uint32_t tz);
uint64_t gpu_signal(void);  // ends the encoder, signals the returned value
void gpu_keep_clock(double seconds);  // keeps the GPU's clock up for about that long
void gpu_commit(void);
void gpu_wait_signal(uint64_t v);
uint64_t gpu_signal_value(void);  // latest completed signal, without waiting
void gpu_sync(void);
int gpu_threads(int k);  // max threads per threadgroup of a kernel
int gpu_profiling(void);       // MYDS4_PROFILE set: dispatches run one at a time, timed
void gpu_profile_reset(void);
extern double gpu_busy;  // seconds the GPU spent on committed command buffers
extern double gpu_gap;   // seconds between them, within each gpu_sync's batch
extern long gpu_gaps;    // of those gaps, the ones longer than 100 us

// ---- io.c: SSD reads, the expert cache and memory pressure ----
void io_read(int fd, uint64_t off, uint64_t len, Buf *dst);  // aligned, uncached
// Rows rows[i] of rowbytes each from a table at base, packed into dst.
void io_rows(int fd, uint64_t base, uint32_t rowbytes, const int64_t *rows, int n, uint8_t *dst);
uint64_t mem_available(void);
int mem_pressure(void);  // 1 normal, 2 warn, 4 critical
typedef struct {
  uint64_t total, free, file, file_kept;  // bytes: RAM, free pages, reclaimable files, diagnostic kernel cache target
  uint64_t wired, compressor, compressed;  // wired pages; the compressor's pages and the memory they hold
  uint64_t swap_used;                     // bytes in swap files
  uint64_t swapouts, compressions;        // pages since boot
  int pressure;                           // as mem_pressure
} MemInfo;
void mem_info(MemInfo *m);
int64_t mem_gpu_limit(void);  // the GPU's resident-memory limit (INT64_MAX: none)
uint64_t mem_gpu(void);       // the GPU driver's resident memory for all processes
int64_t mem_gpu_room(void);   // how far it is below the point where the GPU stalls (INT64_MAX: none)
uint64_t mem_footprint(void);  // of this process
uint64_t mem_swapouts(void);   // pages the system swapped out since boot
uint64_t mem_compressions(void);  // pages the system compressed since boot
uint64_t mem_compressed(int peak);  // bytes of this process in the compressor: now, or at most so far
typedef struct {
  uint64_t hits, misses, prefetch_hits, prefetches, bytes, evictions;
  uint64_t late, cancelled;  // prefetch hits still being read; read-aheads cancelled before reading
  uint64_t substituted;      // experts replaced by --substitute
  uint64_t remaps;           // slots in memory mapped for the GPU again
  uint64_t given_back;       // slots released when the cache shrank for memory
  // The decoded rows' selections (routed_rows), each counted for its row, by
  // where its expert was: in memory, read ahead, read ahead and still being
  // read, a substitute (--substitute), or only on the SSD.
  uint64_t row_own, row_ahead, row_late, row_sub, row_ssd;
  uint64_t row_unique; // backbone expert union across rows of each small forward pass
  double read_s;  // waiting for reads
} CacheStats;
// layers: the cache layers in use (LAYERS, or CACHE_LAYERS with the draft).
// The experts' spans (expert_span) of any layer that has none yet are the
// checkpoint's MXFP4 codes and scales.
// Before cache_init: 0 or 128 preloads every draft expert; 15..127 reserves
// that many MXFP4 slots per stage and reads its native selections on demand.
void cache_draft_limit(int slots);
int cache_draft_capacity(void);  // per stage; 128 is the preloaded default
enum { CACHE_HALF_LIFE_DEFAULT = 50, CACHE_HALF_LIFE_MAX = 1000000 };
// Before cache_init: 0 selects the default half-life. Aging advances once per
// decode/verify pass and after prefill, not once per emitted token. Read-cost
// weighting changes backbone decode eviction priority only; raw history stays.
void cache_retention(int half_life, int read_cost);
enum { CACHE_RESERVE_DEFAULT = 10, CACHE_RESERVE_MIN = 2, CACHE_RESERVE_MAX = 1024 };
// Before cache_init: the memory in GiB that the cache leaves available to
// other programs, 0 for the default. The cache starts within available memory
// less the reserve, gives slots back when the available memory falls below
// it and grows only while 4 GiB more than it is available.
void cache_reserve(double gib);
void cache_init(TensorIndex *index, uint64_t budget, int layers);
// Bounded draft pools are demand-only. Finish prior GPU use of a stage's
// buffers before cache_load/cache_request may replace their experts.
// The model thread owns cache mutations and slot lifetimes. Read-only
// queries (cache_slot, cache_ready, cache_where) may run in synchronous
// helpers while the model thread waits. Slots cannot change until they
// return. cache_get runs on the model thread and maps the expert for GPU
// use; at[k] is span k's offset, and *fresh requires gpu_use until the
// residency set is next committed.
const Buf *cache_get(int layer, int expert, uint32_t at[3], int *fresh);
void cache_load(int layer, const int *experts, int n);  // finishes their reads into memory
// Starts reading the experts into memory. Returns how many had pending reads
// when each was checked; prefetch lateness uses that same completion check.
int cache_request(int layer, const int *experts, int n);
void cache_wait(int layer, const int *experts, int n);     // waits for their reads
// Read the epoch before scanning pending requests, then wait only if no
// piece completed since that scan. A completion for any layer wakes it.
uint64_t cache_read_epoch(void);
void cache_wait_progress(uint64_t epoch);
int cache_slot(int layer, int expert);   // it has a slot: resident, or being read
int cache_hold(int layer, const int *experts, int n);
void cache_unhold(int layer, const int *experts, int n);
uint64_t deepseek_ready_groups(void);
int cache_ready(int layer, int expert);  // resident and read
int cache_where(int layer, int expert);  // 0 only on the SSD, 1 in memory, 2 read ahead, 3 read ahead and still being read
// Starts reading the n predicted experts that are not resident, in order,
// while fewer than `inflight` prefetches are reading.
void cache_prefetch(int layer, const int *experts, int n, int inflight);
// Optional MYDS4_PREFETCH_METRICS=1 observation. Pass IDs identify forward
// batches, not emitted tokens. Snapshot once before substitution, then observe
// the exact and served row unions from that same snapshot on the model thread.
// Only served routes consume first-use attribution. These hooks never change
// routing or cache admission.
enum { PF_PREFILL, PF_DECODE, PF_PHASES };
enum { PF_PC, PF_H1, PF_H2, PF_BULK, PF_OTHER, PF_ISSUERS };
enum { PF_EXISTING_READY, PF_FRESH_READY, PF_PENDING_PREFETCH, PF_PENDING_DEMAND, PF_ABSENT, PF_STATES };
typedef struct {
  int active, layer, phase;
  uint64_t pass;
  uint8_t state[EXPERTS], first[EXPERTS], ghost[EXPERTS];
  uint64_t completed[EXPERTS];
} CacheObservation;
void cache_metrics_begin(int phase, int rows);
void cache_metrics_end(void);
void cache_observe_snapshot(int layer, CacheObservation *snapshot);
void cache_observe_routes(const CacheObservation *snapshot, const int *ids, int rows, int stride, int served);
void cache_prefetch_from(int layer, const int *experts, int n, int inflight, int issuer);
void cache_metrics_report(FILE *out);
// Cancels the read-aheads for the layer that no reader has started and that keep[0..n) does not name.
void cache_refine(int layer, const int *keep, int n);
void cache_demote(int layer);  // after its routing: the layer's unselected read-aheads are evicted first
void cache_pass(int layer);  // a prompt pass is at this layer (-1: none): keep later layers' experts
void cache_adapt(void);             // between tokens
void cache_adapt_pass(int layer);   // within a prompt pass at this layer
void cache_idle(void);              // while no request runs: resizes as between tokens
uint64_t cache_bytes(void);
int cache_slots(void);
int cache_mapped(void);     // slots mapped for the GPU
int cache_cap(void);        // slots the cache may hold now
int cache_max(void);        // and at most
uint64_t cache_slot_bytes(void);
uint64_t cache_uncached_bytes(int *experts);  // the backbone's experts without a slot: bytes and count
extern CacheStats cache_stats;
// An expert's bytes in the files: an MXFP4 expert's codes (w1, w2, w3 back
// to back) and scales, or with --experts the GGUF rows of w1, w3 and w2.
typedef struct { int fd; uint64_t off, len; } FileSpan;
extern FileSpan expert_span[CACHE_LAYERS][EXPERTS][3];
extern int expert_spans[CACHE_LAYERS];  // spans per expert of the layer, 0: not set
// A routed expert as the expert kernels read it: the GPU addresses of its
// three matrices (null: no expert) and its routing weight.
typedef struct { uint64_t p[3]; float weight, pad; } ExpertRef;
static_assert(sizeof(ExpertRef) == 32, "Metal expert record layout");

// ---- tok.c: a checkpoint's tokenizer, one per model; any thread ----
typedef struct Tokenizer Tokenizer;
// The tokenizer of dir/tokenizer.json, with `image` (or null) the spelling of
// the model's image token; a pipeline it does not run ends the process.
Tokenizer *tok_open(const char *dir, const char *image);
int tok_vocab(const Tokenizer *t);  // its highest id plus one
// The ids of len bytes of text (NUL bytes included), at most max of them:
// TOK_TEXT, text alone, without the added tokens' spellings (a message's
// text); TOK_ADDED, text in which the added tokens' spellings are those
// tokens; TOK_PROMPT, a rendered chat prompt: as TOK_ADDED, and IMAGE_MARK,
// an image part's place, becomes the image token, whose own spelling (in a
// file a tool read, say) is text. The three below take a C string.
enum { TOK_TEXT, TOK_ADDED, TOK_PROMPT };
int tok_encode_bytes(Tokenizer *t, const char *text, size_t len, int mode, int *out, int max);
int tok_encode_text(Tokenizer *t, const char *text, int *out, int max);
int tok_encode(Tokenizer *t, const char *text, int *out, int max);
int tok_encode_chat(Tokenizer *t, const char *text, int *out, int max);
#define IMAGE_MARK "\xff"  // no valid UTF-8 text holds this byte
const char *tok_piece(const Tokenizer *t, int id, int *len);
int tok_added(const Tokenizer *t, int id);  // one of tokenizer.json's added tokens: 2 a special one, 1 another; 0
int tok_find(const Tokenizer *t, const char *special);              // its id; a spelling that is none ends the process
int tok_special(const Tokenizer *t, const char *text, size_t len);  // the id of an added token's spelling, -1 for none

// ---- vision.c: images for the vision encoder (inference/image_processor.py) ----
#define IMAGE_TOKEN 129264  // <｜deepseek_image｜>
#define VIT_K 640           // a patch's 3 x 14 x 14 values, zero-padded to a multiple of 64
typedef struct {
  int vh, vw;       // the ViT's grid of 14 x 14 pixel patches
  int th, tw;       // the tokens' grid after the aligner's 3 x 3 merge: th rows of tw
  uint16_t *patch;  // vh vw rows of VIT_K BF16 values
  uint64_t hash;    // of the image file's bytes
  int start;        // the position of its span in the prompt
  uint16_t *rows;   // once encoded (deepseek_image): th tw feature rows of DIM BF16 values
} Image;
// Decodes an image file (any format macOS reads) and prepares its patches;
// 0 with *err if it cannot.
int image_open(Image *im, const void *data, size_t len, const char **err);
// The image padded to the size the ViT takes (bw x bh RGB pixels) and its
// token grid, for checks; null with *err if it cannot be decoded.
uint8_t *image_rgb(const void *data, size_t len, int *bw, int *bh, int *th, int *tw, const char **err);
// Its span in a prompt: [IMAGE_START] + ([IMAGE] * tw + [IMAGE_NEW_LINE]) * th
// + [IMAGE_END], th (tw + 1) + 2 positions, each a negative token from the
// hash (image_keys), so that a session matches only the same image.
int image_span(const Image *im);
void image_keys(const Image *im, int *keys);
void image_rope(const Image *im, float *cs);  // per patch: the 32 cosines, then the 32 sines of its RoPE angles
// The tokens with each <｜deepseek_image｜> replaced by its image's span and
// the images' starts set: the new count (out holds up to max of them), or
// -1 if the tokens hold another number of placeholders than nim.
int image_expand(const int *in, int n, Image *im, int nim, int *out, int max);
void image_free(Image *im);

// ---- model.c ----
typedef struct {
  int ctx;            // capacity in tokens
  double budget_gb;   // expert cache budget, 0 = automatic
  double reserve_gb;  // standalone CLI process reserve; server leaves 0 after setting its global reserve
  int cache_half_life;  // score half-life in forward passes; 0 = default 50
  int cache_read_cost;  // weight decode eviction priorities by aligned read bytes / slot size
  int dump_fd;        // -1, or a pipe receiving per-layer streams
  int routes_fd;      // -1, or a pipe receiving every decoded token's routes, predictions and router logits
  int draft;          // keep the DSpark stages' windows, for deepseek_draft
  int lookup;         // 0: off; otherwise up to VERIFY_DRAFTS context-copy candidates
  int draft_cache;    // draft expert slots per stage; 0 = all 128 preloaded
  const char *predictor_state; // load or create online state; save on clean completion
  float substitute;            // > 0: replace a missing expert by a resident one up to this far below in key
  const char *experts;         // a GGUF file (set) whose routed experts replace the checkpoint's, or null
  int quiet;                   // no progress lines on stderr
  int online_predictor;        // rank64 online prefetch corrector; 0 disables it
  int ced;                    // 0: no approximate replay; otherwise decoder replay tokens (at least WINDOW)
  int prefill_prune;          // skip decoder batches outside the final and saved-prefix dependencies
  int runahead;               // exact concurrent requests: group ready jobs while other expert reads run
  int pipeline;               // experimental native MXFP4 stages across concurrent requests
} Options;
void online_predictor_init(int rank, const float *const biases[LAYERS]);
void online_predictor_reset(void);
void online_predictor_begin(int rows, int eligible);
const int *online_predictor_forecasts(int layer, int rows, int horizons, const float *gp, const float *gl,
                                      const int *native_pr, const int *native_rt);
void online_predictor_update(int layer);
void online_predictor_end(void);
void online_predictor_report(FILE *out);
void online_predictor_bind(const unsigned char identity[32]);
int online_predictor_load(const char *path, char *error, size_t capacity); // 1 loaded, 0 absent, -1 error
int online_predictor_save(const char *path, char *error, size_t capacity); // 1 saved, -1 error
int online_predictor_dirty(void);
void deepseek_save_predictor(void);
void deepseek_load(const char *dir, const Options *o);
// Runs the tokens at positions pos.. and returns the logits of the last one.
const float *deepseek_forward(const int *tokens, int n, int pos);
typedef struct {
  int tokens, chunks, decoder_tokens, ced_tokens, ced_applied, prune_applied;
  int decoder_row_layers; // token rows actually processed across layers 21 through 39
  int prefix_tokens, prefix_encoder_only;
} DeepSeekForwardInfo;
DeepSeekForwardInfo deepseek_forward_info(void);
// The DSpark draft for `token` at position pos, every position before it
// having run (with Options.draft): the tokens it drafts for pos + 1 .. pos + BLOCK.
void deepseek_draft(int token, int pos, int *drafts, int n); // the first n of BLOCK drafts
// Speculative decoding: runs the token at pos and its drafts (m rows) and
// returns each row's logits; deepseek_accept keeps the first n of the rows.
const float *deepseek_verify(const int *tokens, int m, int pos);
void deepseek_accept(const int *tokens, int n, int pos);
void deepseek_bench(void);
void deepseek_reset(void);
// One sequence's attention state: the window rings, compressed rows, index
// keys and compressor rows of the layers, the draft's windows and the
// Engram history. The layers work on one state at a time.
typedef struct State State;
State *deepseek_state_new(void);    // zeroed; the first takes over the state deepseek_load made
void deepseek_state_free(State *s);  // not the one in use
size_t deepseek_state_bytes(const State *s, int positions);  // s (null: a new state) holding that many
void deepseek_state_use(State *s);
// The rows of the state in use that later positions overwrite, saved in the
// state and restored (a rewind point, for the CLI's checks); a restored state
// continues at the saved position or up to deepseek_rewind() positions before it.
void deepseek_state_save(void);
void deepseek_state_restore(void);
// For the server's KV prefix cache. A snapshot holds a state's rolling part
// at one position: the window rings, compressor rows and draft rings that
// later positions overwrite, and the Engram history.
typedef struct Snapshot Snapshot;
Snapshot *deepseek_snapshot(State *s, int pos, int with_logits);  // s's rolling rows now, as at pos; and the last logits
void deepseek_snapshot_restore(State *s, const Snapshot *x);
void deepseek_snapshot_free(Snapshot *x);
size_t deepseek_snapshot_bytes(int with_logits);
int deepseek_snapshot_pos(const Snapshot *x);
int deepseek_snapshot_extent(const Snapshot *x);  // the last layer it holds: LAYERS - 1, or 20 (CED)
const float *deepseek_snapshot_logits(const Snapshot *x);  // null without
// The global rows (compressed rows, index keys) of src before positions, into
// dst, which is not in use (its rewind point is dropped). A running src is
// fine: no later step changes them.
void deepseek_state_copy(State *dst, const State *src, int positions);
// Prefills positions from .. n - 1 of a prompt (tokens: all n of them) in the
// state in use, and returns the last one's logits. Without replay, the state
// holds every position before from (its own, or a copy and a restored
// snapshot at from). With replay it holds only the global rows before from
// and the prefill rebuilds the rolling rows at from. With --ced off the
// replay is exact: the prefill starts earlier, each layer where its window
// needs. With --ced it is bounded and approximate (as DeepInfer's encoder
// SWA bounded replay): every layer computes again the ring's positions before
// from, with windows cut off at the first of them, and the copied global
// rows of groups completed before from stay. With capture >= 0, *captured
// gets the rolling rows at that position as a snapshot, taken during the
// prefill: after where it starts (from, or a replay's first row) and before
// n; a replay's branch point is from. Under CED compaction, unless the
// decoder layers start a ring or more before it, the capture holds layers
// 0-20 only (extent 20).
const float *deepseek_prefill(const int *tokens, int n, int from, int replay, int capture, Snapshot **captured);
// What a replay at k computes again before k, per layer on average (rounded
// up): with --ced off at most k, about 2,613 for long prefixes; with --ced
// the ring (WINDOW plus the rewind margin, 136 positions; at most k).
int deepseek_replay_rows(int k);
// The rewind margin: the positions beyond its window that each layer's ring
// keeps, and so how far a state can continue before its end. REWIND, or more
// with --draft or --lookup: a verify pass writes their rows after the token.
int deepseek_rewind(void);
// The first position a replay at k computes again (layer 0's), from which its
// prompt's images must be encoded.
int deepseek_replay_start(int k);
// The experts row `row` of the latest decode step chose in each layer (TOPK
// each, layer by layer) and where each one was (0 in memory, 1 read ahead,
// 2 late, 3 substituted, 4 from the SSD); 0 when the row did not run in it.
int deepseek_routes(int row, uint16_t *id, uint8_t *from);
// One token for each of m sequences at once (m <= BATCH_ROWS, no drafts):
// row r is tokens[r] at position pos[r] of state st[r], every position before
// it having run. Each row is computed as deepseek_forward computes a token
// alone. Returns the m rows' logits.
const float *deepseek_forward_rows(State **st, const int *tokens, const int *pos, int m);
// Continues the state in use at pos, given the tokens before it: at most
// deepseek_rewind() positions before the last one that ran, or after a reset at 0.
void deepseek_seek(const int *tokens, int pos);
// The vision encoder (inference/vision.py ViT and Aligner): the image's
// feature rows into im->rows. Weights load on first use.
void deepseek_image(Image *im);
// The images whose spans (their starts set) the prompts that deepseek_forward
// runs next hold as negative tokens (image_keys); a span's positions may be
// in any of its batches.
void deepseek_images(const Image *im, int n);

// model.c: the statistics of decoded rows and of prompt rows since start
extern CacheStats decode_stats, prefill_stats;
extern uint64_t decode_rows, prefill_rows;
extern double decode_gpu_s, decode_s;
extern int prefill_layer, prefill_n;  // a prompt pass's layer (0 between prompts) and rows
uint64_t deepseek_dense_bytes(void);
uint64_t deepseek_workspace_bytes(void); // resizable request/layer buffers; excludes fixed scratch, weights and KV
