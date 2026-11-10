// DeepSeek V4.1 Flash shapes, execution and routed-expert cache.
#pragma once
#include "../model.h"
#include "../../engine/expert_store.h"

#define IMAGE_TOKEN 129264  // <｜deepseek_image｜>
#define VIT_K 640           // a patch's 3 x 14 x 14 values, zero-padded to a multiple of 64

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

uint64_t deepseek_ready_groups(void);
// An expert's bytes in the files: an MXFP4 expert's codes (w1, w2, w3 back
// to back) and scales, or with --experts the GGUF rows of w1, w3 and w2.
extern FileSpan expert_span[CACHE_LAYERS][EXPERTS][3];
extern int expert_spans[CACHE_LAYERS];  // spans per expert of the layer, 0: not set
// Every layer without spans yet (--experts sets the backbone's) gets the
// checkpoint's MXFP4 codes and scales.
static inline void deepseek_mxfp4_spans(TensorIndex *index) {
  for (int l = 0; l < CACHE_LAYERS; l++)
    for (int e = 0; !expert_spans[l] && e < (l < LAYERS ? EXPERTS : DRAFT_EXPERTS); e++) {
      Tensor w = l < LAYERS ? st_get(index, "layers.%d.ffn.experts.%d.w1.weight", l, e)
                            : st_get(index, "mtp.%d.ffn.experts.%d.w1.weight", l - LAYERS, e);
      Tensor s = l < LAYERS ? st_get(index, "layers.%d.ffn.experts.%d.w1.scale", l, e)
                            : st_get(index, "mtp.%d.ffn.experts.%d.w1.scale", l - LAYERS, e);
      expert_span[l][e][0] = (FileSpan){w.fd, w.off, EXPERT_W};
      expert_span[l][e][1] = (FileSpan){s.fd, s.off, EXPERT_S};
      if (e == (l < LAYERS ? EXPERTS : DRAFT_EXPERTS) - 1) expert_spans[l] = 2;
    }
}
// DeepSeek's expert store: the backbone's routed experts, then the DSpark
// stages'.
static inline ExpertStore *deepseek_cache_create(void) {
  return cache_create(&(ExpertLayout){.layers = CACHE_LAYERS, .scored_layers = LAYERS, .experts = EXPERTS,
                                      .chosen = TOPK, .stage_experts = DRAFT_EXPERTS, .stage_min = DRAFT_CACHE_MIN,
                                      .parts = 3, .span = &expert_span[0][0][0], .spans = expert_spans});
}
// With MYDS4_PREFETCH_METRICS: the expert store's read-ahead measurements.
void deepseek_metrics_report(FILE *out);
// A routed expert as the expert kernels read it: the GPU addresses of its
// three matrices (null: no expert) and its routing weight.
typedef struct { uint64_t p[3]; float weight, pad; } ExpertRef;
static_assert(sizeof(ExpertRef) == 32, "Metal expert record layout");


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
int deepseek_load(const char *dir, const Options *o);
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
int deepseek_image(Media *im);
// The images whose spans (their starts set) the prompts that deepseek_forward
// runs next hold as negative tokens (media_keys); a span's positions may be
// in any of its batches.
void deepseek_images(const Media *im, int n);

// model.c: the statistics of decoded rows and of prompt rows since start
extern CacheStats decode_stats, prefill_stats;
extern uint64_t decode_rows, prefill_rows;
extern double decode_gpu_s, decode_s;
extern int prefill_layer, prefill_n;  // a prompt pass's layer (0 between prompts) and rows
uint64_t deepseek_dense_bytes(void);
uint64_t deepseek_workspace_bytes(void); // resizable request/layer buffers; excludes fixed scratch, weights and KV
