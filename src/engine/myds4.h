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
} Sampler;
int sampler_valid(const Sampler *s);
void sampler_seed(Sampler *s, uint64_t seed);
int sample_argmax(const float *logits, int count);
int sample_token(const float *logits, int count, Sampler *s);
// logits contains ndraft + 1 rows of count values; drafts contains ndraft
// tokens. The sampler is independent of the model's verification row limit.
int sample_verify(const float *logits, int count, const int *drafts, int ndraft, int eos, Sampler *s, int *accepted);
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

typedef struct { // one tensor of the checkpoint
  int fd;
  uint64_t off, size;
  int dtype;  // 'b' bf16, 'f' f32, '8' fp8 e4m3, 'e' e8m0, 'i' packed e2m1
  int64_t shape[4];
  int ndim;
} Tensor;
void st_open(const char *dir);
Tensor st_get(const char *fmt, ...);
int st_has(const char *fmt, ...);

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
void *gpu_open_late(void);   // a command buffer that joins the queue when gpu_submit commits it
void *gpu_close(void);       // ends the current command buffer's encoding without committing it
void gpu_submit(void *cmd);
void gpu_kern(int k);
void gpu_buf(int idx, const Buf *b, size_t off);
void gpu_bytes(int idx, const void *p, size_t n);
void gpu_use(const Buf *b);
void gpu_go(uint32_t gx, uint32_t gy, uint32_t gz, uint32_t tx, uint32_t ty, uint32_t tz);
uint64_t gpu_signal(void);  // ends the encoder, signals the returned value
void gpu_keep_clock(double seconds);  // keeps the GPU's clock up for about that long
void gpu_commit(void);
void gpu_wait_signal(uint64_t v);
void gpu_spin_signal(uint64_t v);  // gpu_wait_signal, spinning
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
  uint64_t total, free, file, file_kept;  // bytes: RAM, free pages, file cache, the file cache macOS keeps
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
// other programs, 0 for the default. The cache starts within the free memory
// less the reserve, gives slots back when the available memory falls below
// it and grows only while 4 GiB more than it is available.
void cache_reserve(double gib);
void cache_init(uint64_t budget, int layers);
// Bounded draft pools are demand-only. Finish prior GPU use of a stage's
// buffers before cache_load/cache_request may replace their experts.
// The model thread owns cache mutations and slot lifetimes. Slot queries
// (cache_get, cache_slot, cache_ready, cache_where) may also run in synchronous
// parallel helpers while the model thread waits for them. No slot may be
// replaced or released until those helpers return.
// The slot of a resident expert; at[k]: where span k is in it; *fresh:
// resident only through gpu_use until the next token.
const Buf *cache_get(int layer, int expert, uint32_t at[3], int *fresh);
void cache_load(int layer, const int *experts, int n);  // makes them resident
// Starts making the experts resident. Returns how many had pending reads
// when each was checked; prefetch lateness uses that same completion check.
int cache_request(int layer, const int *experts, int n);
void cache_repeat_credit(int layer, int expert, int repeats); // extra row uses after one deduplicated request
void cache_wait(int layer, const int *experts, int n);     // waits for their reads
int cache_slot(int layer, int expert);   // it has a slot: resident, or being read
int cache_ready(int layer, int expert);  // resident and read
int cache_where(int layer, int expert);  // 0 only on the SSD, 1 in memory, 2 read ahead, 3 read ahead and still being read
// Starts reading up to `admit` of n predicted experts that are not resident,
// while fewer than `inflight` prefetches are reading.
void cache_prefetch(int layer, const int *experts, int n, int admit, int inflight);
// Optional MYDS4_PREFETCH_METRICS=1 observation. Pass IDs identify forward
// batches, not emitted tokens. Snapshot once before substitution, then observe
// the exact and served row unions from that same snapshot on the model thread.
// Only served routes consume first-use attribution. These hooks never change
// routing or cache admission. GPU-route passes report unsupported coverage.
enum { PF_PREFILL, PF_DECODE, PF_PHASES };
enum { PF_PC, PF_H1, PF_H2, PF_BULK, PF_OTHER, PF_DRAFT, PF_H4, PF_H8, PF_ISSUERS };
enum { PF_EXISTING_READY, PF_FRESH_READY, PF_PENDING_PREFETCH, PF_PENDING_DEMAND, PF_ABSENT, PF_STATES };
typedef struct {
  int active, layer, phase;
  uint64_t pass;
  uint8_t state[EXPERTS], first[EXPERTS], ghost[EXPERTS];
  uint64_t completed[EXPERTS];
} CacheObservation;
void cache_metrics_begin(int phase, int rows);
void cache_metrics_end(void);
void cache_metrics_unsupported(void);
void cache_observe_snapshot(int layer, CacheObservation *snapshot);
void cache_observe_routes(const CacheObservation *snapshot, const int *ids, int rows, int stride, int served);
void cache_prefetch_from(int layer, const int *experts, int n, int admit, int inflight, int issuer);
void cache_metrics_report(FILE *out);
// Cancels the read-aheads for the layer that no reader has started and that keep[0..n) does not name.
void cache_refine(int layer, const int *keep, int n);
void cache_demote(int layer);  // after its routing: the layer's unselected read-aheads are evicted first
void cache_pass(int layer);  // a prompt pass is at this layer (-1: none): keep later layers' experts
// --gpu-route: writes the layer's entries in the table of ready experts'
// span addresses (cache_table), which a command buffer committed next reads;
// from then on until cache_unpublish or the next layer's entries, its slots
// stay. Returns how many of its experts are still being read.
const Buf *cache_table(void);
int cache_publish(int layer);
void cache_unpublish(void);
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
extern uint64_t spec_fixups;  // --gpu-route: decoded layers whose experts the CPU computed again
// An expert's bytes in the files: an MXFP4 expert's codes (w1, w2, w3 back
// to back) and scales, or with --experts the GGUF rows of w1, w3 and w2.
typedef struct { int fd; uint64_t off, len; } FileSpan;
extern FileSpan expert_span[CACHE_LAYERS][EXPERTS][3];
extern int expert_spans[CACHE_LAYERS];  // spans per expert of the layer, 0: not set

// ---- tok.c ----
void tok_load(const char *dir);
int tok_encode(const char *text, int *out, int max);
// A rendered chat prompt: IMAGE_MARK, an image part's place, becomes the
// image token, and the placeholder's text <｜deepseek_image｜> (in a file a
// tool read, say) is text.
int tok_encode_chat(const char *text, int *out, int max);
#define IMAGE_MARK "\xff"  // no valid UTF-8 text holds this byte
const char *tok_piece(int id, int *len);
int tok_added(int id);  // one of tokenizer.json's added tokens (the special tokens among them)
int tok_find(const char *special);
// Engram hashing (data/engram.bin): multipliers [2][4], primes [2][3][8],
// offsets [2][24], the compressed pad id and token map.
extern const int32_t *engram_map;
extern const int64_t *engram_mult, *engram_primes, *engram_offsets;
int engram_pad(void);

// ---- vision.c: images for the vision encoder (inference/image_processor.py) ----
#define IMAGE_TOKEN 129264  // <｜deepseek_image｜>
#define VIT_K 640           // a patch's 3 x 14 x 14 values, zero-padded to a multiple of 64
typedef struct {
  int vh, vw;       // the ViT's grid of 14 x 14 pixel patches
  int th, tw;       // the tokens' grid after the aligner's 3 x 3 merge: th rows of tw
  uint16_t *patch;  // vh vw rows of VIT_K BF16 values
  uint64_t hash;    // of the image file's bytes
  int start;        // the position of its span in the prompt
  uint16_t *rows;   // once encoded (model_image): th tw feature rows of DIM BF16 values
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
  double reserve_gb;  // memory the expert cache leaves to other programs (GiB), 0 = CACHE_RESERVE_DEFAULT
  int cache_half_life;  // score half-life in forward passes; 0 = default 50
  int cache_read_cost;  // weight decode eviction priorities by aligned read bytes / slot size
  int dump_fd;        // -1, or a pipe receiving per-layer streams
  int routes_fd;      // -1, or a pipe receiving every decoded token's routes, predictions and router logits
  int draft;          // keep the DSpark stages' windows, for model_draft
  int lookup;         // 0: off; otherwise up to VERIFY_DRAFTS context-copy candidates
  int expert_batch;   // share repeated experts' weight loads across verification rows
  int draft_cache;    // draft expert slots per stage; 0 = all 128 preloaded
  int draft_trim;     // final draft stage: only requested query/FFN rows; keep all five keys
  const char *predictor_state; // load or create online state; save on clean completion
  float substitute;            // > 0: replace a missing expert by a resident one up to this far below in key
  int gpu_route;               // decoding: the GPU looks up each layer's experts (see step_spec)
  const char *experts;         // a GGUF file (set) whose routed experts replace the checkpoint's, or null
  int quiet;                   // no progress lines on stderr
  int online_predictor;        // rank64 online prefetch corrector; 0 disables it
  int ced;                    // 0: no approximate replay; otherwise decoder replay tokens (at least WINDOW)
  int prefill_prune;          // skip decoder batches outside the final and saved-prefix dependencies
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
void model_save_predictor(void);
void model_load(const char *dir, const Options *o);
// Runs the tokens at positions pos.. and returns the logits of the last one.
const float *model_forward(const int *tokens, int n, int pos);
// Run one prompt traversal while saving the state at prefix_pos in slot 1.
// An encoder-only CED snapshot requires more than ced + max(BLOCK, lookup)
// new tokens on reuse, so verification can rewind safely.
const float *model_forward_prefix(const int *tokens, int n, int pos, int prefix_pos, int *encoder_only);
typedef struct {
  int tokens, chunks, decoder_tokens, ced_tokens, ced_applied, prune_applied;
  int prefix_tokens, prefix_encoder_only;
} ModelForwardInfo;
ModelForwardInfo model_forward_info(void);
// The DSpark draft for `token` at position pos, every position before it
// having run (with Options.draft): the tokens it drafts for pos + 1 .. pos + BLOCK.
void model_draft(int token, int pos, int *drafts, int n); // the first n of BLOCK drafts
void model_draft_coupled(int token, int pos, int *drafts, int n, const Sampler *sampler); // experimental shared uniforms
// Speculative decoding: runs the token at pos and its drafts (m rows) and
// returns each row's logits; model_accept keeps the first n of the rows.
const float *model_verify(const int *tokens, int m, int pos);
void model_accept(const int *tokens, int n, int pos);
void model_bench(void);
void model_reset(void);
// One sequence's attention state: the window rings, compressed rows, index
// keys and compressor rows of the layers, the draft's windows and the
// Engram history. The layers work on one state at a time.
typedef struct State State;
State *model_state_new(void);    // zeroed; the first takes over the state model_load made
void model_state_free(State *s);  // not the one in use
// Fork slot 1 at its saved token count into an inactive destination. Copy
// only the prefix's global rows and restore its local state and history.
void model_state_fork(State *dst, const State *src, int positions);
size_t model_state_bytes(const State *s, int positions);  // s (null: a new state) holding that many
void model_state_use(State *s);
// The rows of the state in use that later positions overwrite, saved in
// slot k (0 or 1) and restored; a restored state continues at the saved
// position or up to BLOCK positions before it.
void model_state_save(int k);
void model_state_restore(int k);
// One token for each of m sequences at once (m <= BLOCK + 1, no drafts):
// row r is tokens[r] at position pos[r] of state st[r], every position before
// it having run. Each row is computed as model_forward computes a token
// alone. Returns the m rows' logits.
const float *model_forward_rows(State **st, const int *tokens, const int *pos, int m);
// Continues the state in use at pos, given the tokens before it: at most
// BLOCK positions before the last one that ran, or after a reset at 0.
void model_seek(const int *tokens, int pos);
// The vision encoder (inference/vision.py ViT and Aligner): the image's
// feature rows into im->rows. Weights load on first use.
void model_image(Image *im);
// The images whose spans (their starts set) the prompts that model_forward
// runs next hold as negative tokens (image_keys); a span's positions may be
// in any of its batches.
void model_images(const Image *im, int n);

// model.c: the statistics of decoded rows and of prompt rows since start
extern CacheStats decode_stats, prefill_stats;
extern uint64_t decode_rows, prefill_rows;
extern double decode_gpu_s, decode_s;
extern int prefill_layer, prefill_n;  // a prompt pass's layer (0 between prompts) and rows
uint64_t model_dense_bytes(void);
uint64_t model_workspace_bytes(void); // resizable request/layer buffers; excludes fixed scratch, weights and KV
