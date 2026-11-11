// Shared native services: files, memory, Metal, tokenization and images.
#pragma once
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

// ---- util.c ----
_Noreturn void die(const char *fmt, ...);
extern void (*die_hook)(void);  // runs before die's message (the server's live view restores the terminal)
// The log: say() writes one line with its time, to log_hook when it is set
// (the server's live view), else to stderr. Any thread may call it.
extern void (*log_hook)(const char *line);
void say(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
double now(void);
void *xmalloc(size_t n);
char *read_file(const char *path, size_t *len);
long b64_decode(const char *s, size_t n, uint8_t *out); // bytes written (at most 3 per 4 characters), -1 if not base64
void shutdown_watch(void); // SIGINT/SIGTERM request a stop; handlers do no I/O
int shutdown_requested(void);
int shutdown_requests(void); // how many stops were requested
void shutdown_request(void); // a stop, as SIGINT and SIGTERM request one
// Sampling workspaces belong to the single model thread. Each non-greedy
// ordinary token consumes one draw. Sampled proposals and acceptance use additional draws.
typedef struct {
  double temp, top_p;
  unsigned short seed[3];
  int top_k;  // 0: every token
} Sampler;
int sampler_valid(const Sampler *s);
void sampler_seed(Sampler *s, uint64_t seed);
int sample_argmax(const float *logits, int count);
int sample_token(const float *logits, int count, Sampler *s);
int sample_proposal(const float *logits, int count, Sampler *s, double *out);
int sample_verify_probabilities(const float *logits, int count, const int *drafts, int ndraft,
                                const double *q, int (*stop)(int token), Sampler *s, int *accepted);
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
  int dtype;  // 'b' BF16, 'f' F32, '8' E4M3, 'e' E8M0, 'i' I8, 'u' U8, 'U' U32, 'l' I64
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
struct MemorySlot;
typedef struct {
  void *obj;     // id<MTLBuffer>
  uint8_t *ptr;  // CPU address of offset 0
  uint64_t gpu;  // GPU address of offset 0
  size_t size;
  uint32_t off;  // byte offset of the payload inside the buffer
  struct MemorySlot *memory;
  size_t allocated;
  int kv;
} Buf;
void gpu_init(const char *source, size_t len);
const char *gpu_source(size_t *length);
void gpu_start(void);  // gpu_init with every model's kernels (src/kernels/library.c), once per process
int gpu_kernel(const char *name);
Buf gpu_alloc(size_t size);  // resident from the start
Buf gpu_new(size_t size);    // outside the residency set
Buf gpu_try_new(size_t size, int admission); // empty when the model-memory pool refuses it
Buf gpu_alloc_kv(size_t size); // charged only to the separate KV pool
size_t gpu_allocation_size(size_t size);
void gpu_free(Buf *b);
void gpu_free_later(Buf *b);  // gpu_free once the work committed or being encoded so far has finished
// Allocation membership in the main queue's bounded residency sets. Changes
// apply at gpu_residency_commit; gpu_alloc commits its own allocation.
void gpu_resident(const Buf *b, int on);
void gpu_residency_commit(void);
typedef struct {
  uint64_t bytes, largest_set, target_set;
  uint32_t sets, allocations;
} GpuResidencyInfo;
void gpu_residency_info(GpuResidencyInfo *info);
uint64_t gpu_working_set(void);
void *gpu_open(void);        // a command buffer that runs after those opened before
void *gpu_open_detached(void);  // one that joins the queue only when gpu_submit_detached commits it
void gpu_encode(void *cmd);  // encode into it until gpu_commit or the next gpu_encode
void gpu_begin(void);        // gpu_encode(gpu_open())
void gpu_encode_concurrent(void *cmd);  // gpu_encode whose dispatches may overlap unless a barrier lies between them
void gpu_barrier(void);                 // the dispatches encoded before it finish before those after it
void gpu_barrier_on(const Buf *const *buffers, int count);  // the same, for what they wrote to these buffers
void *gpu_close(void);       // ends the current command buffer's encoding without committing it
void gpu_submit(void *cmd);
void gpu_submit_detached(void *cmd);  // gpu_submit of a buffer from gpu_open_detached
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
extern int gpu_other_passes;  // passes with work besides the one running: their work keeps the clock up
void gpu_commit(void);
void gpu_wait_signal(uint64_t v);
uint64_t gpu_signal_value(void);  // latest completed signal, without waiting
// Inside a pass, its own event for its signals (see gpu_pass_variables, pass.h).
void gpu_pass_begin(void);
void gpu_pass_end(void);
void gpu_sync(void);
void gpu_reap(void);  // accounts for the command buffers that have finished, without waiting
void gpu_drain(void); // complete both the compute and clock-keeping queues
int gpu_threads(int k);  // max threads per threadgroup of a kernel
int gpu_profiling(void);       // MYDS4_PROFILE set: dispatches run one at a time, timed
void gpu_profile_reset(void);
extern double gpu_busy;  // seconds the GPU spent on committed command buffers
extern double gpu_gap;   // seconds between them, within each gpu_sync's batch
extern long gpu_gaps;    // of those gaps, the ones longer than 100 us
// What the engine thread does, for measuring where its time and the GPU's idle
// time go: outside myds4_run, running a pass, or in myds4_run with no pass
// ready while one waits for the GPU, else for expert or Engram reads, else for
// something else (a slot that another pass holds).
enum { THREAD_OUTSIDE, THREAD_PASS, THREAD_WAIT_GPU, THREAD_WAIT_READ, THREAD_WAIT_OTHER, THREAD_STATES };
void gpu_thread_state(int state);  // from the engine thread, when its state changes
// Since start: the engine thread's seconds in each state, the GPU main queue's
// busy seconds (gpu_busy) and its idle seconds between command buffers by
// the engine thread's state during them (counted when they are accounted for).
void gpu_times(double thread[THREAD_STATES], double *busy, double idle[THREAD_STATES]);

// ---- io.c: SSD reads, the expert cache and memory pressure ----
void io_read(int fd, uint64_t off, uint64_t len, Buf *dst);  // aligned, uncached
// Rows rows[i] of rowbytes each from a table at base, packed into dst.
void io_rows(int fd, uint64_t base, uint32_t rowbytes, const int64_t *rows, int n, uint8_t *dst);
uint64_t io_rows_workspace(int rows);
// Since start, for every expert store: the seconds with an expert read in
// flight (their union), and the demand and background reads' summed seconds.
void io_ssd_times(double *busy, double *demand, double *background);
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
// Model memory follows the available memory, keeping reserve_gib (0: the
// default) for other programs, within model_gib when it is above 0.
void memory_configure(double model_gib, double reserve_gib);
int64_t mem_gpu_limit(void);  // Metal's residency target (INT64_MAX before GPU initialization)
uint64_t mem_gpu(void);       // the GPU driver's resident memory for all processes
int64_t mem_gpu_room(void);   // target minus total driver residency (INT64_MAX before initialization)
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
  uint64_t decode_uses;      // uses counted toward decoding's eviction priority (a prompt's count none)
  // The decoded rows' selections (routed_rows), each counted for its row, by
  // where its expert was: in memory, read ahead, read ahead and still being
  // read, a substitute (--substitute), or only on the SSD.
  uint64_t row_own, row_ahead, row_late, row_sub, row_ssd;
  uint64_t row_unique; // backbone expert union across rows of each small forward pass
  double read_s;  // waiting for reads
} CacheStats;
enum { CACHE_RESERVE_DEFAULT = 10, CACHE_RESERVE_MIN = 2, CACHE_RESERVE_MAX = 1024 };
void cache_reserve(double gib);

// ---- tok.c: a checkpoint's tokenizer, one per model; any thread ----
typedef struct Tokenizer Tokenizer;
// The tokenizer of dir/tokenizer.json, with `image` (or null) the spelling of
// the model's image token; a pipeline it does not run ends the process.
Tokenizer *tok_open(const char *dir, const char *image);
void tok_close(Tokenizer *tokenizer);
int tok_vocab(const Tokenizer *t);  // its highest id plus one
// The ids of len bytes of text (NUL bytes included), at most max of them:
// TOK_TEXT, text alone, without the added tokens' spellings (a message's
// text); TOK_ADDED, text in which the added tokens' spellings are those
// tokens; TOK_PROMPT, a rendered chat prompt: as TOK_ADDED, and MEDIA_MARK,
// a media part's place, becomes the model's placeholder, whose own spelling (in a
// file a tool read, say) is text. The three below take a C string.
enum { TOK_TEXT, TOK_ADDED, TOK_PROMPT };
int tok_encode_bytes(Tokenizer *t, const char *text, size_t len, int mode, int *out, int max);
int tok_encode_text(Tokenizer *t, const char *text, int *out, int max);
int tok_encode(Tokenizer *t, const char *text, int *out, int max);
int tok_encode_chat(Tokenizer *t, const char *text, int *out, int max);
#define MEDIA_MARK "\xff"  // no valid UTF-8 text holds this byte
const char *tok_piece(const Tokenizer *t, int id, int *len);
int tok_added(const Tokenizer *t, int id);  // one of tokenizer.json's added tokens: 2 a special one, 1 another; 0
int tok_find(const Tokenizer *t, const char *special);              // its id; a spelling that is none ends the process
int tok_special(const Tokenizer *t, const char *text, size_t len);  // the id of an added token's spelling, -1 for none

// ---- media inputs and encoded feature rows ----
typedef enum { MEDIA_IMAGE, MEDIA_AUDIO } MediaKind;
typedef struct {
  MediaKind kind;
  int ntokens, embedding_width;
  union {
    struct { int vh, vw, th, tw, patch_stride; uint16_t *patch; };
    struct { int sample_rate, sample_count; float *samples; };
  };
  uint64_t hash;    // of the encoded file's bytes
  int start;        // the position of its span in the prompt
  uint16_t *rows;   // BF16 feature rows, after encoding
  Buf features;    // optional GPU storage owning rows
} Media;
// Decodes an image file (any format macOS reads) and prepares its patches;
// 0 with *err if it cannot.
int image_open(Media *im, const void *data, size_t len, const char **err);
int audio_open(Media *media, const void *data, size_t len, const char **error);
uint8_t *image_decode_rgb(const void *data, size_t len, int *width, int *height, const char **err);
uint8_t *image_resize_rgb(const uint8_t *pixels, int width, int height, int out_width, int out_height);
// The image padded to the size the ViT takes (bw x bh RGB pixels) and its
// token grid, for checks; null with *err if it cannot be decoded.
uint8_t *image_rgb(const void *data, size_t len, int *bw, int *bh, int *th, int *tw, const char **err);
// The model processor sets the span length. Its negative token keys depend
// on the media bytes, so prefix reuse distinguishes different inputs.
int media_span(const Media *im);
uint64_t media_embedding(const Media *media, int count, int position);
void media_keys(const Media *im, int *keys);
void image_rope(const Media *im, float *cs);  // per patch: the 32 cosines, then the 32 sines of its RoPE angles
// The tokens with each model-specific placeholder replaced by its feature span and
// the inputs' starts set: the new count (out holds up to max of them), or
// -1 if the tokens hold another number of placeholders than nim.
int media_expand(const int *in, int n, Media *im, int nim, int *out, int max, int media_token);
void media_free(Media *im);
