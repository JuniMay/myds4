// DeepSeek-V4.1-Flash: weights, state and the forward pass.
//
// The forward pass runs a batch of consecutive token rows through one layer
// at a time: one row when decoding, up to MB rows of a prompt.
//
// A decoded token's layer runs as two command buffers. The front holds the
// attention, the FFN's hyper-connection mix and the router, which signals the
// CPU, then the shared expert. The CPU reads the routing, brings the routed
// experts into the cache, starts reading the experts predicted for the next
// layer and encodes the routed experts. The next layer's front is committed
// before that and waits in the queue behind them.
//
// A prompt runs layer by layer over all of its rows, so that each expert is
// read from the SSD at most once per prompt; its routed experts run as
// grouped products over the (token, expert) pairs.
#include "model.h"
#include <dispatch/dispatch.h>
#include <CommonCrypto/CommonDigest.h>
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "myds4.h"
#include "../model.h"

#include "engram.h"
#include "../weights.h"
#include "expert_tasks.h"
static Engram ngram;
static TensorIndex *checkpoint;
static WeightSet *core_weights, *vision_weights, *weight_group;
static Buf image_markers[3];
static int model_buffers_ready;
static uint64_t resizable_bytes(void);
#include "prefill.h"

typedef struct { Buf w, s; int n, k, group; } Lin;  // s.obj is null for BF16 weights; group: rows
                                                  // per input slice (wo_a)
typedef struct {
  int ratio, kv_source, index_source, src;  // src: the KV source whose rows it reads
  Lin wq_a, wq_b, wkv, wo_a, wo_b, gate, sh1, sh2, sh3;
  Buf attn_norm, ffn_norm, q_norm, kv_norm, sink, gate_bias, gate_bias_vl;  // bias_vl: inside image spans
  Buf hc_attn_fn, hc_attn_base, hc_attn_scale, hc_ffn_fn, hc_ffn_base, hc_ffn_scale;
  Lin c_wkv, c_wgate, i_wq_b, i_wproj, i_wk;  // compressor, indexer
  Buf c_norm, i_knorm;
  Lin e_wkv;  // Engram
  Buf e_qk;
  Tensor e_codes, e_scales;
  Buf ring;                // window keys [RING][512] in BF16, row p % RING
  Buf comp, keys, cstate;  // KV sources: compressed rows, index keys (BF16), compressor rows [PENDING][512] x 2 (FP32)
} Layer;

static Layer L[LAYERS];
static Layer D[DRAFTS];  // the DSpark stages; ring: the main model's keys (DSparkAttention's main_kv)
static Lin head, main_proj;
static Buf main_norm, draft_norm, markov_embed, markov_head;
static Buf final_norm, cs_base, cs_comp;
static Tensor embed_t;
static Options opt;
static int runahead_enabled;
static int pipeline_enabled;
static void pipeline_report(FILE *out);
static uint64_t scheduler_workspace_bytes;
static uint64_t runahead_group_count, runahead_group_rows;
static int runahead_route_only;
static uint64_t runahead_route_only_rows;
static int verifying;
// The rewind margin (deepseek_rewind): REWIND, or the most rows a verify pass
// writes after its token, when --draft or --lookup asks for more.
static int replay_margin(void) {
  int m = REWIND;
  if (opt.draft > m) m = opt.draft;
  if (opt.lookup > m) m = opt.lookup;
  return m;
}
// A prompt's rows run the prefill kernels however many rows a batch has, so
// that a row's result does not depend on where the prefill that computes it
// starts: decoded rows (one, or a verify pass's) run the decode kernels.
static int prompt_pass;
static int small_rows(int m) { return !prompt_pass && (m <= BATCH_ROWS || (verifying && m <= VERIFY_DRAFTS + 1)); }
static int prefill_prune_active, prefill_prune_end;

static uint32_t ced_first;
static int ced_active, ced_compact;
static DeepSeekForwardInfo forward_info;
// A bounded replay (deepseek_prefill's replay under --ced, as DeepInfer's
// encoder SWA bounded replay; approximate): every layer computes again from
// bounded_first, its windows cut off there, and the copied compressed rows
// and index keys of the groups completed before bounded_keep (the branch
// point) stay as they are.
static int bounded;
static uint32_t bounded_first, bounded_keep;
// The first position a row's window reads: CED's decoder layers from
// ced_first, every layer of a bounded replay from bounded_first.
static uint32_t window_lower(int layer, uint32_t pos) {
  uint32_t ced = ced_active && layer >= 20 && pos >= ced_first ? ced_first : 0;
  return bounded && bounded_first > ced ? bounded_first : ced;
}

#ifdef MYDS4_CED_CHECK
// The research tool selects a full-shape decoder SWA mask or compact execution.
// The retained tool owns these settings and observes routing after GPU completion.
static uint32_t ced_check_first;
static int ced_check_layer = 20, ced_check_compact;
static void ced_check_routes(int layer, int row, int rows, const int *routes);
#endif

// Rows of a prompt batch. A batch groups its (row, expert) pairs by expert
// into tiles, and each tile decodes its expert's weights once, so larger
// batches fill tiles better. Batch boundaries change no result.
#define MB 8192
// Dependency-pruned prefill keeps or skips decoder rows in groups of this
// many (prefill_prune_plan's block), independently of MB.
#define PRUNE_ROWS 2048
// A layer whose next layer takes more rows than this reads all of the next
// layer's experts ahead: so many rows choose nearly every expert.
#define BULK_ROWS 4096
#define KERNEL_ROWS 6  // the most rows the one-row kernels' multi-row forms (products, the head) take at once
#define TARGET 37   // the first of the layers whose attention input the draft reads
// A layer's window keys are a ring of RING rows (row p % RING), and a KV
// source keeps the compressor rows of its last PENDING positions (row p %
// PENDING). Beyond the window the ring keeps the rewind margin's rows, so
// that the rows a verify pass writes after the tokens it keeps (rejected
// drafts or proposals) do not overwrite rows later positions read, and so
// that a state can continue up to the margin before its end. A ratio-2
// compressor that continues at a position reads the two rows before it:
// PENDING exceeds the margin by at least two.
#define RING (WINDOW + replay_margin())
#define PENDING (replay_margin() + 2 > 16 ? 32 : 16)
#define PREDICT 16  // experts predicted per row for the next layer
#define ATT_ROWS (WINDOW + IDX_TOPK)  // the most rows a decoded row attends
#define FEW (VERIFY_DRAFTS + 1)  // maximum target verification rows
#define ATT_CHUNKS (ATT_ROWS / 64)    // of ATT_CHUNK rows in the kernels
#define GiB (1024.0 * 1024 * 1024)

// Per row of the request (prompt rows, or row 0 when decoding): the streams,
// the FFN mixes and the index selections.
static Buf HS, MF, SEL;
// Diagnostics retain their FP32 wire format while inference stores BF16.
// Callers consume this model-thread-owned scratch before the next call.
static const float *stream_floats(const Buf *b, size_t first, size_t count) {
  static float *values;
  static size_t capacity;
  if (count > capacity) {
    memory_host_free(values);
    values = memory_host_required(count * sizeof *values, 0);
    capacity = count;
  }
  const uint16_t *source = (const uint16_t *)b->ptr + first;
  for (size_t i = 0; i < count; i++) {
    uint32_t bits = (uint32_t)source[i] << 16;
    memcpy(values + i, &bits, sizeof bits);
  }
  return values;
}
static Buf IMG;  // per row: 1 inside an image span (a negative token), set by prefill
static int request_rows;
static int work_rows;
float *all_logits;  // when set, prefill writes every row's logits here
// BF16 rows for the matrix units: a product's input for mm_fp8 (x_bf16), up
// to MB rows, padded to 128, of up to XB_K values; or the queries with RoPE
// of up to QB_ROWS rows for attend_m (q_rope).
#define XB_K 8192
#define QB_ROWS 512
static_assert(QB_ROWS * HEADS * HEAD_DIM <= (MB + 128) * XB_K, "XB holds QB_ROWS rows of queries");
static Buf XB;
// Per row of a batch.
static Buf H1, MA, PART, X, XQ, QA, QR, QRQ, Q, O, OA, AO, KV, CK, CS, LAT, IK, QI, IW, SC, GL, RT, PR;
// The index scores of up to sc_rows rows at once (SC), and each row's
// candidate blocks (CAND, cand_words 32-bit words per prompt row).
static Buf CAND;
static int sc_rows, cand_words;
static Buf SHG, SHU, SHH, SHO, EROW, EC[2], ES[2], EX, EKV, MH, YK, TILES, PTOK, PW, TP, WIN, ETAB, logits;
// The draft: main_hidden rows (3 x 5120, of up to WINDOW positions), main_x,
// the drafted rows' streams, mixes, keys, logits, tokens and embeddings.
static Buf MAINH, MXA, MXN, MXQ, DHS, DMF, BLK, DSEL, DLOG, DTOK, DEROW;
// The residual prediction's input row, and this layer's top experts it
// predicts: KEEP when decoding (PC), PREDICT for the routes trace (PT).
static Buf XP, PT, PC, GP;  // GP: predicted logits, so that GL keeps the router's
// A decoded row's attention: scores, chunk maxima and chunk parts (see attend_one).
static Buf ASC, AMAX, APART;
static Buf PARTS;  // decoded rows' experts' down products, row r's at 8 DIM r, of expert e at 8 i + e

static int K_mm_fp8, K_mm_bf16, K_mv_fp8, K_mv_bf16, K_mvm_fp8, K_mvm_bf16, K_mv_fp8_3, K_mv_bf16_3, K_mvm_fp8_3,
    K_mvm_bf16_3, K_moe_up, K_moe_down, K_moe_up_m, K_moe_down_m, K_gg_gather, K_act_quant_h, K_gather_mxfp4, K_act_quant_mxfp4,
    K_moe_combine, K_moe_parts, K_act_quant, K_embed, K_rmsnorm, K_hc_mix, K_hc_pre_norm, K_hc_post, K_swiglu,
    K_kv_window, K_compress, K_index_key, K_index_score, K_index_topk, K_attend_scores, K_attend_values,
    K_attend_combine, K_attn_out, K_route,
    K_engram_rows, K_engram_gate, K_copy_rows, K_copy_keys, K_attend_m, K_hc_mean, K_markov, K_argmax, K_cand_topk,
    K_x_bf16, K_q_rope, K_vit_bias, K_vit_norm, K_vit_rope, K_vit_heads, K_vit_attn, K_vit_swiglu,
    K_vit_add, K_vit_unfold;

static uint64_t dense_bytes;
static size_t rows_bytes(int l, int cap, int keys);
static size_t saved_bytes(void);
static int start_cap(void);
static void base_state(void), reserve(int need, int keep), shrink(void);
// One sequence's attention state (see the state section at the end). A
// state holds compressed rows and index keys for cap positions: at first
// CAP (or the context, if less), doubling as positions need them.
#define CAP 16384
struct State {
  Buf ring[LAYERS], comp[LAYERS], keys[LAYERS], cstate[LAYERS], dring[DRAFTS];
  int32_t history[3];
  int cap;
  char *saved;  // the rolling rows deepseek_state_save copied, for deepseek_state_restore
};
static State *active;  // the state the layers work on

// A state's rolling part at one position: the rows that later positions
// overwrite (every window ring, the compressor rows, the draft rings) and
// the Engram history, in save_rows' layout (see deepseek_snapshot).
struct Snapshot {
  int pos;        // the position the state continues at
  int extent;     // the last layer whose rows it holds: LAYERS - 1, or 20 (CED's encoder-only capture)
  char *rows;     // saved_bytes()
  float *logits;  // the logits pos's token is sampled from, or null
};

typedef struct {
  int at, encoder_only, tap_first;
  uint8_t captured[LAYERS];
  char *ring[LAYERS], *cstate[LAYERS], *dring[DRAFTS], *history;
} PrefixCapture;
static PrefixCapture prefix_capture = {.at = -1};
static Buf prefix_taps, prefix_drings[DRAFTS];

// --experts: the 40 layers' routed experts from GGUF files in llama.cpp's
// layout, all of a layer's experts' rows of w1, w3 and w2 in one tensor each
// (ffn_gate_exps, ffn_up_exps, ffn_down_exps), in one of the codes of
// gguf.metal; the DSpark stages keep the checkpoint's. gk: a layer's kernels
// for its codes (gk[l].up 0: the checkpoint's MXFP4).
static const struct { int type; uint32_t bytes; const char *k[4]; } gg_codes[] = {
    {GG_Q2_K, 84, {"gg_up_q2_k", "gg_down_q2_k", "gg_up_m_q2_k", "gg_down_m_q2_k"}},
    {GG_IQ2_XXS, 66, {"gg_up_iq2_xxs", "gg_down_iq2_xxs", "gg_up_m_iq2_xxs", "gg_down_m_iq2_xxs"}},
    {GG_IQ2_XS, 74, {"gg_up_iq2_xs", "gg_down_iq2_xs", "gg_up_m_iq2_xs", "gg_down_m_iq2_xs"}},
    {GG_IQ2_S, 82, {"gg_up_iq2_s", "gg_down_iq2_s", "gg_up_m_iq2_s", "gg_down_m_iq2_s"}},
    {GG_IQ3_XXS, 98, {"gg_up_iq3_xxs", "gg_down_iq3_xxs", "gg_up_m_iq3_xxs", "gg_down_m_iq3_xxs"}},
    {GG_IQ1_S, 50, {"gg_up_iq1_s", "gg_down_iq1_s", "gg_up_m_iq1_s", "gg_down_m_iq1_s"}},
    {GG_IQ1_M, 56, {"gg_up_iq1_m", "gg_down_iq1_m", "gg_up_m_iq1_m", "gg_down_m_iq1_m"}}};
enum { GG_CODES = sizeof gg_codes / sizeof gg_codes[0] };
static struct {
  int up, down, up_m, down_m;
} gk[LAYERS];
static uint32_t expert_quant[LAYERS][3]; // 0 is native MXFP4; otherwise the GGUF type
// The routed experts' bytes in the files, which the expert store reads
// (deepseek_cache_create): the GGUF rows (gguf_experts), else the MXFP4
// codes and scales (deepseek_mxfp4_spans).
FileSpan expert_span[CACHE_LAYERS][EXPERTS][3];
int expert_spans[CACHE_LAYERS];
static ExpertStore *experts_store;  // the routed experts in memory
static CacheStats *expert_stats(void) { return cache_counters(experts_store); }
void deepseek_metrics_report(FILE *out) {
  if (experts_store) cache_metrics_report(experts_store, out);
}
static Buf head_q6;  // the GGUF file's output head in Q6_K, if it has one: used instead of the checkpoint's
static int K_gg_head1, K_gg_head6;
// A prompt batch's pairs' rows of XQ and MH, in pair order. Native MXFP4
// keeps half values multiplied by 128; GGUF keeps unscaled half values.
static Buf XH, MHH;
static Buf *const work_buffers[] = {
    &H1, &MA, &PART, &X, &XQ, &QA, &QR, &QRQ, &Q, &O, &OA, &AO, &KV, &CK, &CS, &LAT,
    &IK, &QI, &IW, &SC, &GL, &GP, &RT, &PR, &SHG, &SHU, &SHH, &SHO, &EROW, &EX, &EKV,
    &MH, &YK, &TILES, &XH, &MHH, &XB, &PTOK, &PW, &TP, &WIN};

static void gguf_experts(const char *path) {
  static const char *mat[3] = {"gate", "up", "down"};  // w1, w3, w2
  gguf_open(path);
  for (int l = 0; l < LAYERS; l++) {
    int code[3];
    for (int k = 0; k < 3; k++) {
      char name[64];
      snprintf(name, sizeof name, "blk.%d.ffn_%s_exps.weight", l, mat[k]);
      Gguf t;
      if (!gguf_get(name, &t)) die("%s: no tensor %s", path, name);
      uint64_t in = k < 2 ? DIM : INTER, out = k < 2 ? INTER : DIM;
      int c = 0;
      while (c < GG_CODES && gg_codes[c].type != t.type) c++;
      if (c == GG_CODES) die("%s: %s is in ggml type %d, which myds4 does not decode", path, name, t.type);
      if (t.ne[0] != in || t.ne[1] != out || t.ne[2] != EXPERTS) die("%s: %s is not [%llu, %llu, %d]", path, name, in, out, EXPERTS);
      uint64_t len = out * (in / 256) * gg_codes[c].bytes;
      for (int e = 0; e < EXPERTS; e++) expert_span[l][e][k] = (FileSpan){t.fd, t.off + e * len, len};
      code[k] = c;
      expert_quant[l][k] = (uint32_t)t.type;
    }
    if (code[0] != code[1]) die("%s: layer %d's w1 and w3 are in different codes", path, l);
    expert_spans[l] = 3;
    gk[l].up = gpu_kernel(gg_codes[code[0]].k[0]), gk[l].down = gpu_kernel(gg_codes[code[2]].k[1]);
    gk[l].up_m = gpu_kernel(gg_codes[code[0]].k[2]), gk[l].down_m = gpu_kernel(gg_codes[code[2]].k[3]);
  }
  Gguf h;
  if (gguf_get("output.weight", &h) && h.type == 14 && h.ne[0] == DIM && h.ne[1] == VOCAB) {  // Q6_K
    uint64_t len = (uint64_t)VOCAB * (DIM / 256) * 210;
    weights_remove(core_weights, &head.w);
    weights_tensor(core_weights, &head_q6, (Tensor){.fd = h.fd, .off = h.off, .size = len}, 1);
    K_gg_head1 = gpu_kernel("gg_head1"), K_gg_head6 = gpu_kernel("gg_head6");
  }
}


static void load(Buf *target, Tensor tensor) { weights_tensor(weight_group, target, tensor, 1); }
static void vec(Buf *target, Tensor tensor) { weights_vector(weight_group, target, tensor, 0, 0); }
static void lin(Lin *line, const char *stem) {
  char name[256];
  snprintf(name, sizeof name, "%s.weight", stem);
  Tensor tensor = st_get(checkpoint, "%s", name);
  *line = (Lin){.n = (int)tensor.shape[0], .k = (int)tensor.shape[1]};
  load(&line->w, tensor);
  if (tensor.dtype == '8') {
    snprintf(name, sizeof name, "%s.scale", stem);
    load(&line->s, st_get(checkpoint, "%s", name));
  }
}

static Buf zeros(size_t n) {
  Buf b = gpu_alloc(n);
  memset(b.ptr, 0, n);
  return b;
}

// Temporary layer buffers grow for prefill and shrink for small forward
// passes. Persistent KV, logits, request rows and draft history are separate.
static uint64_t work_plan(int rows, int allocate) {
  uint64_t bytes = 0;
#define W(buffer, amount) do { size_t size = (amount); bytes += gpu_allocation_size(size); if (allocate) buffer = zeros(size); } while (0)
  size_t m = (size_t)rows, p = m * TOPK;
  W(H1, m * HC * DIM * STREAM_BYTES); W(MA, m * MIX * 4); W(PART, m * 400 * 4);
  W(X, m * DIM * 4); W(XQ, m * DIM * 4); W(QA, m * QRANK * 4); W(QR, m * QRANK * 4);
  W(QRQ, m * QRANK * 4); W(Q, m * HEADS * HEAD_DIM * 4); W(O, m * HEADS * HEAD_DIM * 4);
  W(OA, m * OGROUPS * ORANK * 4); W(AO, m * DIM * 4); W(KV, m * HEAD_DIM * 4);
  W(CK, (m + 2) * HEAD_DIM * 4); W(CS, (m + 2) * HEAD_DIM * 4); W(LAT, m * HEAD_DIM * 4);
  W(IK, m * IDX_DIM * 4); W(QI, m * IDX_HEADS * IDX_DIM * 4); W(IW, m * IDX_HEADS * 4);
  int score_rows = (int)((256u << 20) / ((size_t)opt.ctx * 4));
  if (score_rows > rows) score_rows = rows;
  if (score_rows < 1) score_rows = 1;
  W(SC, (size_t)score_rows * opt.ctx * 4);
  size_t predicted = m > 2 * FEW ? m : 2 * FEW;
  size_t routed = m;
  W(GL, m * EXPERTS * 4); W(GP, predicted * EXPERTS * 4); W(RT, routed * 16 * 4);
  W(PR, (routed > predicted ? routed : predicted) * PREDICT * 4);
  W(SHG, m * INTER * 4); W(SHU, m * INTER * 4); W(SHH, m * INTER * 4); W(SHO, m * DIM * 4);
  W(EROW, m * DIM * 2); W(EX, m * ENGRAM_ROWS * ENGRAM_DIM * 4); W(EKV, m * (HC + 1) * DIM * 4);
  W(MH, p * INTER * 4); W(YK, p * DIM * 4); W(TILES, (p / 32 + EXPERTS) * 32);
  W(XH, (p + 64) * DIM * 2); W(MHH, (p + 64) * INTER * 2);
  size_t matrix = ((m + 127) / 128) * 128 * XB_K * 2;
  size_t queries = (m < QB_ROWS ? m : QB_ROWS) * HEADS * HEAD_DIM * 2;
  size_t draft = opt.draft ? 128u * 3 * DIM * 2 : 0;
  size_t xb = matrix > queries ? matrix : queries;
  W(XB, xb > draft ? xb : draft);
  W(PTOK, p * 4); W(PW, p * 4); W(TP, p * 4); W(WIN, (m + WINDOW) * HEAD_DIM * 2);
  if (allocate) sc_rows = score_rows;
#undef W
  return bytes;
}

static void work_capacity(int rows) {
  int small = replay_margin() + 1;
  if (rows > MB) rows = MB;
  if (rows < small) rows = small;
  if (rows <= work_rows && !(rows == small && work_rows > small)) return;
  int resize_cache = work_rows != 0;
  gpu_sync();
  for (size_t i = 0; i < sizeof work_buffers / sizeof *work_buffers; i++) gpu_free(work_buffers[i]);
  work_plan(rows, 1);
  work_rows = rows;
  if (resize_cache) cache_idle(experts_store); // capacity maintenance does not represent another token or pass
}

// RoPE angles: (cos, sin) for 32 frequencies per position; YaRN for the
// compressed layers, as precompute_freqs_cis computes them in FP32.
static Buf rope_table(int ctx, double base, int yarn) {
  float freq[32];
  for (int i = 0; i < 32; i++) freq[i] = 1.0f / powf((float)base, (float)(2 * i) / 64.0f);
  if (yarn) {
    double c32 = 64 * log(65536 / (32 * 2 * M_PI)) / (2 * log(base));
    double c1 = 64 * log(65536 / (1 * 2 * M_PI)) / (2 * log(base));
    double low = fmax(floor(c32), 0), high = fmin(ceil(c1), 63);
    for (int i = 0; i < 32; i++) {
      float ramp = fminf(fmaxf((float)((i - low) / fmax(high - low, 1e-3)), 0), 1);
      float smooth = 1 - ramp;
      freq[i] = freq[i] / 16 * (1 - smooth) + freq[i] * smooth;
    }
  }
  Buf b = gpu_alloc((size_t)ctx * 32 * 8);
  float *cs = (float *)b.ptr;
  for (int p = 0; p < ctx; p++)
    for (int i = 0; i < 32; i++) {
      float a = (float)p * freq[i];
      cs[(p * 32 + i) * 2] = cosf(a);
      cs[(p * 32 + i) * 2 + 1] = sinf(a);
    }
  return b;
}

static const int kv_sources[] = {2, 8, 14, 20}, index_sources[] = {2, 8, 14, 20, 24, 28, 32, 36};
static int is_in(int l, const int *set, int n) {
  for (int i = 0; i < n; i++)
    if (set[i] == l) return 1;
  return 0;
}

static uint64_t request_plan(int rows, int allocate) {
  uint64_t bytes = 0;
#define W(buffer, amount) do { size_t size = (amount); bytes += gpu_allocation_size(size); if (allocate) buffer = zeros(size); } while (0)
  W(HS, (size_t)rows * HC * DIM * STREAM_BYTES); W(MF, (size_t)rows * MIX * 4);
  W(SEL, (size_t)rows * (IDX_TOPK + 1) * 4);
  W(CAND, (size_t)rows * cand_words * 4);
  W(IMG, (size_t)rows);
  for (int e = 0; e < 2; e++) {
    W(EC[e], (size_t)rows * ENGRAM_ROWS * ENGRAM_DIM); W(ES[e], (size_t)rows * ENGRAM_ROWS * 8);
  }
#undef W
  return bytes;
}

static void request_capacity(int rows) {
  if (Q.obj) work_capacity(rows);
  // Small forward passes overwrite these rows and do not read the prompt's
  // workspace. Keep room for every verification row (the token and up to the
  // rewind margin's drafts) and every batched row without reallocating
  // between single-token and draft passes.
  int small = replay_margin() + 1 > BATCH_ROWS ? replay_margin() + 1 : BATCH_ROWS;
  if (rows <= small) rows = small;
  if (rows <= request_rows && !(rows == small && request_rows > small)) return;
  gpu_sync();
  gpu_free(&HS), gpu_free(&MF), gpu_free(&SEL);
  gpu_free(&CAND); gpu_free(&IMG);
  for (int e = 0; e < 2; e++) { gpu_free(&EC[e]); gpu_free(&ES[e]); }
  request_plan(rows, 1);
  request_rows = rows;
}

// The parts every block has: attention, hyper-connections, router and
// shared expert, of the tensors named prefix.*.
static void load_block(Layer *y, const char *prefix) {
  char s[128];
#define LIN(field, name) snprintf(s, sizeof s, "%s.%s", prefix, name), lin(&y->field, s)
#define VEC(field, name) vec(&y->field, st_get(checkpoint, "%s.%s", prefix, name))
  LIN(wq_a, "attn.wq_a"), LIN(wq_b, "attn.wq_b"), LIN(wkv, "attn.wkv");
  LIN(wo_a, "attn.wo_a"), LIN(wo_b, "attn.wo_b"), LIN(gate, "ffn.gate");
  y->wo_a.group = ORANK;
  LIN(sh1, "ffn.shared_experts.w1"), LIN(sh2, "ffn.shared_experts.w2"), LIN(sh3, "ffn.shared_experts.w3");
  VEC(attn_norm, "attn_norm.weight"), VEC(ffn_norm, "ffn_norm.weight");
  VEC(q_norm, "attn.q_norm.weight"), VEC(kv_norm, "attn.kv_norm.weight");
  VEC(sink, "attn.attn_sink"), VEC(gate_bias, "ffn.gate.bias");
  if (st_has(checkpoint, "%s.ffn.gate.bias_vl", prefix)) VEC(gate_bias_vl, "ffn.gate.bias_vl");
  else weights_alias(weight_group, &y->gate_bias_vl, &y->gate_bias);
  VEC(hc_attn_fn, "hc_attn_fn"), VEC(hc_attn_base, "hc_attn_base"), VEC(hc_attn_scale, "hc_attn_scale");
  VEC(hc_ffn_fn, "hc_ffn_fn"), VEC(hc_ffn_base, "hc_ffn_base"), VEC(hc_ffn_scale, "hc_ffn_scale");
}

// Bind saved learning to the actual routers and expert formats, without
// hashing hundreds of GB of expert data or depending on filesystem paths.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
static void identity_number(CC_SHA256_CTX *hash, uint64_t value) {
  uint8_t bytes[8];
  for (int i = 0; i < 8; i++) bytes[i] = (uint8_t)(value >> (8 * i));
  CC_SHA256_Update(hash, bytes, sizeof bytes);
}

static void predictor_identity(uint8_t result[32]) {
  CC_SHA256_CTX hash;
  CC_SHA256_Init(&hash);
  static const char version[] = "myds4 router/quantization identity v1";
  CC_SHA256_Update(&hash, version, sizeof version);
  identity_number(&hash, LAYERS);
  identity_number(&hash, DIM);
  identity_number(&hash, EXPERTS);
  identity_number(&hash, INTER);
  identity_number(&hash, head_q6.obj != NULL);
  for (int l = 0; l < LAYERS; l++) {
    Tensor gate = st_get(checkpoint, "layers.%d.ffn.gate.weight", l);
    identity_number(&hash, (uint64_t)gate.dtype);
    identity_number(&hash, gate.size);
    CC_SHA256_Update(&hash, L[l].gate.w.ptr + L[l].gate.w.off, (CC_LONG)gate.size);
    if (L[l].gate.s.obj) {
      Tensor scale = st_get(checkpoint, "layers.%d.ffn.gate.scale", l);
      identity_number(&hash, scale.size);
      CC_SHA256_Update(&hash, L[l].gate.s.ptr + L[l].gate.s.off, (CC_LONG)scale.size);
    } else identity_number(&hash, 0);
    CC_SHA256_Update(&hash, L[l].gate_bias.ptr, EXPERTS * sizeof(float));
    CC_SHA256_Update(&hash, L[l].gate_bias_vl.ptr, EXPERTS * sizeof(float));
    for (int k = 0; k < 3; k++) identity_number(&hash, expert_quant[l][k]);
  }
  CC_SHA256_Final(result, &hash);
}
#pragma clang diagnostic pop

uint64_t deepseek_ready_groups(void) { return runahead_group_count; }

void deepseek_save_predictor(void) {
  if (pipeline_enabled) pipeline_report(stderr);
  if (runahead_group_count)
    fprintf(stderr, "{\"ready_groups\":{\"groups\":%llu,\"rows\":%llu}}\n",
            (unsigned long long)runahead_group_count, (unsigned long long)runahead_group_rows);
  fprintf(stderr, "{\"forecast_work\":{\"route_only_row_layers\":%llu}}\n", (unsigned long long)runahead_route_only_rows);
  if (!opt.deepseek.predictor_state || !online_predictor_dirty()) return;
  char error[256];
  if (online_predictor_save(opt.deepseek.predictor_state, error, sizeof error) < 0)
    die("predictor state %s: %s", opt.deepseek.predictor_state, error);
  fprintf(stderr, "saved online predictor state: %s\n", opt.deepseek.predictor_state);
}

static uint64_t fixed_plan(int allocate) {
  uint64_t bytes = 2 * gpu_allocation_size((size_t)opt.ctx * 32 * 8);
  if (allocate) {
    cs_base = rope_table(opt.ctx, 10000, 0);
    cs_comp = rope_table(opt.ctx, 160000, 1);
  }
#define W(buffer, amount) do { size_t size = (amount); bytes += gpu_allocation_size(size); if (allocate) buffer = zeros(size); } while (0)
  size_t expert_tables = 2 * LAYERS * FEW + DRAFTS * BLOCK;
  W(ETAB, expert_tables * 256); W(logits, (size_t)FEW * VOCAB * 4);
  W(PARTS, FEW * DIM * 8 * 4);
  W(XP, FEW * DIM * 4); W(PT, PREDICT * 4); W(PC, FEW * PREDICT * 4);
  W(ASC, HEADS * ATT_ROWS * 4); W(AMAX, ATT_CHUNKS * HEADS * 4); W(APART, ATT_CHUNKS * HEADS * 514 * 4);
  if (opt.draft) {
    W(MAINH, WINDOW * 3 * DIM * 4); W(MXA, WINDOW * DIM * 4); W(MXN, WINDOW * DIM * 4);
    W(MXQ, WINDOW * DIM * 4); W(DHS, BLOCK * HC * DIM * STREAM_BYTES); W(DMF, BLOCK * MIX * 4);
    W(BLK, BLOCK * HEAD_DIM * 2); W(DSEL, (IDX_TOPK + 1) * 4); W(DLOG, (size_t)BLOCK * VOCAB * 4);
    W(DTOK, (BLOCK + 1) * 4); W(DEROW, BLOCK * DIM * 2);
    if (allocate) {
    int *sel = (int *)DSEL.ptr;  // every drafted row sees every drafted row's key
    sel[0] = BLOCK;
    for (int r = 0; r < BLOCK; r++) sel[1 + r] = r;
    int64_t noise[BLOCK - 1];
    for (int r = 0; r < BLOCK - 1; r++) noise[r] = NOISE;
    io_rows(embed_t.fd, embed_t.off, DIM * 2, noise, BLOCK - 1, DEROW.ptr + DIM * 2);
    }
  }
#undef W
  return bytes;
}

static Buf kv_zeros(size_t bytes) {
  Buf buffer = gpu_alloc_kv(bytes);
  memset(buffer.ptr, 0, bytes);
  return buffer;
}

static void prepare_weights(const char *dir, const Options *options) {
  if (options->deepseek.ced && options->dump_fd >= 0) die("per-layer dumps require --ced off");
  opt = *options;
  runahead_enabled = opt.deepseek.runahead;
  pipeline_enabled = opt.deepseek.pipeline;
  if (pipeline_enabled && (opt.deepseek.experts || opt.deepseek.runahead || opt.draft || opt.deepseek.online_predictor || opt.deepseek.predictor_state ||
                           opt.deepseek.substitute > 0 || opt.dump_fd >= 0 || opt.routes_fd >= 0 || getenv("MYDS4_PROFILE")))
    die("pipeline requires native MXFP4, exact routing, no runahead, DSpark, online predictor or dispatch profiling");
  if (pipeline_enabled) fprintf(stderr, "pipeline: experimental; native MXFP4 concurrent requests\n");
  if (runahead_enabled && (opt.draft || opt.deepseek.online_predictor || opt.deepseek.substitute > 0 ||
                          opt.dump_fd >= 0 || opt.routes_fd >= 0))
    die("runahead requires exact decoding without DSpark or online prediction");
  fprintf(stderr, "runahead: %s; concurrent requests only\n", runahead_enabled ? "on" : "off");
  if (opt.lookup < 0 || opt.lookup > VERIFY_DRAFTS) die("--lookup takes off or 1 to %d drafts", VERIFY_DRAFTS);
  if (opt.deepseek.ced && (opt.deepseek.ced < WINDOW || opt.deepseek.ced > 1 << 20)) die("--ced takes off or 128 to 1048576 replay tokens");
  if (opt.deepseek.predictor_state) opt.deepseek.online_predictor = 64;
  if (opt.deepseek.online_predictor && opt.deepseek.online_predictor != 64) die("online predictor requires rank 64");
  engram_init(&ngram, dir);
  gpu_start();
#define K(name) K_##name = gpu_kernel(#name)
  K(mm_fp8), K(mm_bf16), K(mv_fp8), K(mv_bf16), K(mvm_fp8), K(mvm_bf16), K(mv_fp8_3), K(mv_bf16_3), K(mvm_fp8_3);
  K(mvm_bf16_3), K(moe_up), K(moe_down), K(moe_up_m), K(moe_down_m), K(gg_gather), K(act_quant_h);
  K(gather_mxfp4), K(act_quant_mxfp4);
  K(moe_combine), K(moe_parts), K(act_quant), K(embed), K(rmsnorm), K(hc_mix), K(hc_pre_norm), K(hc_post);
  K(swiglu), K(kv_window), K(compress), K(index_key), K(index_score), K(index_topk), K(attn_out);
  K(attend_scores), K(attend_values), K(attend_combine);
  K(route), K(engram_rows), K(engram_gate), K(copy_rows), K(copy_keys), K(attend_m), K(hc_mean), K(markov), K(argmax);
  K(cand_topk), K(x_bf16), K(q_rope);
  K(vit_bias), K(vit_norm), K(vit_rope), K(vit_heads), K(vit_attn), K(vit_swiglu), K(vit_add), K(vit_unfold);
  checkpoint = st_open(dir);
  if (!memory_current()) memory_scope(memory_shared());
  core_weights = weights_new(memory_current()->account, "DeepSeek dense backbone");
  weight_group = core_weights;
  // the store reads the spans when it starts (cache_required), after they are set below
  experts_store = deepseek_cache_create();
  cache_memory(experts_store, weights_slot(core_weights));
  static const uint8_t ratios[LAYERS] = {0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  for (int l = 0; l < LAYERS; l++) {
    Layer *y = &L[l];
    char s[128], prefix[32];
    snprintf(prefix, sizeof prefix, "layers.%d", l);
    load_block(y, prefix);
    y->ratio = ratios[l];
    y->kv_source = is_in(l, kv_sources, 4);
    y->index_source = is_in(l, index_sources, 8);
    y->src = l;
    if (y->ratio)
      for (int i = 0; i < 4; i++)
        if (kv_sources[i] <= l) y->src = kv_sources[i];
    if (y->kv_source) {
      LIN(c_wkv, "attn.compressor.wkv");
      if (y->ratio == 2) LIN(c_wgate, "attn.compressor.wgate");
      VEC(c_norm, "attn.compressor.norm.weight");
      LIN(i_wk, "attn.indexer.wk");
      VEC(i_knorm, "attn.indexer.k_norm.weight");
    }
    if (y->index_source) LIN(i_wq_b, "attn.indexer.wq_b"), LIN(i_wproj, "attn.indexer.weights_proj");
    if (l == 1 || l == 14) {
      LIN(e_wkv, "engram.wkv");
      weights_product(core_weights, &y->e_qk,
                      st_get(checkpoint, "%s.engram.q_weight", prefix),
                      st_get(checkpoint, "%s.engram.k_weight", prefix));
      y->e_codes = st_get(checkpoint, "layers.%d.engram.embed.weight", l);
      y->e_scales = st_get(checkpoint, "layers.%d.engram.embed.scale", l);
    }
    if (!opt.quiet) fprintf(stderr, "\rloading layer %d/%d", l + 1, LAYERS);
  }
  lin(&head, "head");
  vec(&final_norm, st_get(checkpoint, "norm.weight"));
  embed_t = st_get(checkpoint, "embed.weight");
  if (opt.draft) {
    for (int d = 0; d < DRAFTS; d++) {
      char prefix[32];
      snprintf(prefix, sizeof prefix, "mtp.%d", d);
      load_block(&D[d], prefix);
    }
    lin(&main_proj, "mtp.0.main_proj");
    vec(&main_norm, st_get(checkpoint, "mtp.0.main_norm.weight"));
    vec(&draft_norm, st_get(checkpoint, "mtp.2.norm.weight"));
    load(&markov_embed, st_get(checkpoint, "mtp.2.markov_head.embed.weight"));
    load(&markov_head, st_get(checkpoint, "mtp.2.markov_head.head.weight"));
  }
  const char *markers[] = {"image_start", "image_end", "image_newline"};
  for (int i = 0; i < 3; i++)
    if (st_has(checkpoint, "%s", markers[i])) load(&image_markers[i], st_get(checkpoint, "%s", markers[i]));
  if (opt.deepseek.experts) gguf_experts(opt.deepseek.experts);
  deepseek_mxfp4_spans(checkpoint);
  cache_retention(experts_store, opt.deepseek.cache_half_life, opt.deepseek.cache_read_cost);
  cache_draft_limit(experts_store, opt.deepseek.draft_cache);
  cand_words = (opt.ctx / 8 + 31) / 32 + 1;
  cache_working_rows(experts_store, 1);
  cache_required(experts_store, opt.draft ? CACHE_LAYERS : LAYERS);
}

int deepseek_load(const char *dir, const Options *options) {
  double began = now();
  if (!checkpoint) prepare_weights(dir, options);
  MemorySlot *core = weights_slot(core_weights), *work = memory_current();
  memory_pin(core, 1);
  cache_active(experts_store, 1);
  int small = replay_margin() + 1 > BATCH_ROWS ? replay_margin() + 1 : BATCH_ROWS;
  cache_working_rows(experts_store, 1);
  uint64_t workspace = model_buffers_ready ? work->payload - resizable_bytes() + work_plan(small, 0) + request_plan(small, 0)
                                          : fixed_plan(0) + work_plan(small, 0) + request_plan(small, 0);
  uint64_t minimum = weights_required(core_weights) + workspace + cache_required(experts_store, opt.draft ? CACHE_LAYERS : LAYERS);
  if (!memory_minimum(core->account, minimum)) { memory_pin(core, 0); return 0; }
  if (model_buffers_ready) request_capacity(1);
  if (!weights_load(core_weights)) { memory_pin(core, 0); return 0; }
  for (int l = 0; l < LAYERS; l++) {
    Layer *y = &L[l];
    if (!opt.deepseek.experts) {
      float maximum = 0;
      for (int d = 0; d < DIM; d++) maximum = fmaxf(maximum, fabsf(((const float *)y->ffn_norm.ptr)[d]));
      // RMSNorm bounds a channel by sqrt(DIM)*max|weight|. The factor
      // 1.125 covers BF16 and E4M3 rounding before scaled-half staging.
      if (!(maximum * sqrtf(DIM) * 1.125f < 511.75f))
        die("layer %d FFN norm exceeds the MXFP4 activation-staging range", l);
    }
  }
  if (!model_buffers_ready) {
    for (int l = 0; l < LAYERS; l++) {
      Layer *y = &L[l];
      y->ring = kv_zeros(RING * HEAD_DIM * 2);
      if (y->kv_source) {
        y->comp = kv_zeros(rows_bytes(l, start_cap(), 0));
        y->keys = kv_zeros(rows_bytes(l, start_cap(), 1));
        y->cstate = kv_zeros(2 * PENDING * HEAD_DIM * 4);
      }
    }
    for (int d = 0; opt.draft && d < DRAFTS; d++) D[d].ring = kv_zeros(RING * HEAD_DIM * 2);
    request_capacity(1);
    work_capacity(small);
    fixed_plan(1);
  if (opt.deepseek.online_predictor) {
    const float *biases[LAYERS];
    for (int l = 0; l < LAYERS; l++) biases[l] = (const float *)L[l].gate_bias.ptr;
    online_predictor_init(opt.deepseek.online_predictor, biases);
    if (opt.deepseek.predictor_state) {
      uint8_t identity[32];
      predictor_identity(identity);
      online_predictor_bind(identity);
      char error[256];
      int loaded = online_predictor_load(opt.deepseek.predictor_state, error, sizeof error);
      if (loaded < 0) die("predictor state %s: %s", opt.deepseek.predictor_state, error);
      fprintf(stderr, "%s online predictor state: %s\n", loaded ? "loaded" : "new", opt.deepseek.predictor_state);
    }
  }
    cache_init(experts_store, 0, opt.draft ? CACHE_LAYERS : LAYERS);
    base_state();
    model_buffers_ready = 1;
  }
  dense_bytes = weights_resident_bytes(core_weights);
  fprintf(stderr, "loaded %.1f GiB of dense weights in %.1f s\n", dense_bytes / GiB, now() - began);
  return 1;
}

// ---- dispatch helpers ----

typedef struct { uint32_t n, k, group_rows, group_k, round, split; } MV;
typedef struct { MV mv; uint32_t m, lda, ldy; } MVM;
typedef struct { uint32_t m, n, k, round, lda, ldy; } MM;

static MV mv_params(const Lin *w, int round) {
  // split K over up to 8 simdgroups until there are about 1024 threadgroups
  uint32_t split = 1;
  while (split < 8 && (uint64_t)w->n * split / 32 < 1024 && w->k / 16 / (split * 2) >= 64) split *= 2;
  return (MV){w->n, w->k, w->group ? w->group : w->n, w->group ? w->k : 0, (uint32_t)round, split};
}

// A matrix of products() and where its outputs go (rows at stride ldy).
typedef struct { const Lin *w; const Buf *out; size_t off; uint32_t ldy; } Prod;
static void linear(const Lin *w, int m, const Buf *in, size_t in_off, uint32_t lda, const Buf *out,
                   size_t out_off, uint32_t ldy, int round);

// linear() with up to 3 matrices of one kind (FP8 or BF16), k and split on
// the same rows; with up to FEW rows (one when decoding; a verify pass's rows
// share each weight load, as decoding computes them), in one dispatch.
static void products(int count, const Prod *p, int m, const Buf *in, size_t in_off, uint32_t lda, int round) {
  if (!small_rows(m)) {
    for (int j = 0; j < count; j++) linear(p[j].w, m, in, in_off, lda, p[j].out, p[j].off, p[j].ldy, round);
    return;
  }
  // Keep the scalar-equivalent kernels' register bound (KERNEL_ROWS). Each
  // segment retains the same per-row product and accumulation order.
  if (m > KERNEL_ROWS) {
    for (int first = 0; first < m; first += KERNEL_ROWS) {
      int rows = m - first < KERNEL_ROWS ? m - first : KERNEL_ROWS;
      Prod part[3];
      for (int j = 0; j < count; j++) {
        part[j] = p[j];
        part[j].off += (size_t)first * p[j].ldy * sizeof(float);
      }
      products(count, part, rows, in, in_off + (size_t)first * lda * sizeof(float), lda, round);
    }
    return;
  }
  MVM q = {mv_params(p[0].w, round), (uint32_t)m, lda, p[0].ldy};
  uint32_t rows[3][2], width = 0;  // each product's rows and output stride, and the grid's width
  int fp8 = p[0].w->s.obj != 0, one = count == 1;
  gpu_kern(m == 1 ? fp8 ? one ? K_mv_fp8 : K_mv_fp8_3 : one ? K_mv_bf16 : K_mv_bf16_3
                  : fp8 ? one ? K_mvm_fp8 : K_mvm_fp8_3 : one ? K_mvm_bf16 : K_mvm_bf16_3);
  for (int j = 0; j < count; j++) {  // weights, scales, outputs: buffers 0, 1, 3, then 5 to 7 and 8 to 10
    if (mv_params(p[j].w, round).split != q.mv.split) die("products of different splits");
    rows[j][0] = p[j].w->n, rows[j][1] = p[j].ldy;
    uint32_t tgs = (p[j].w->n * q.mv.split + 31) / 32;
    width = tgs > width ? tgs : width;
    gpu_buf(j ? 3 * j + 2 : 0, &p[j].w->w, 0);
    if (fp8) gpu_buf(j ? 3 * j + 3 : 1, &p[j].w->s, 0);
    gpu_buf(j ? 3 * j + 4 : 3, p[j].out, p[j].off);
  }
  gpu_buf(2, in, in_off), gpu_bytes(4, &q, m == 1 ? sizeof q.mv : sizeof q);
  if (!one) gpu_bytes(11, rows, count * sizeof rows[0]);
  gpu_go(width, count, 1, 256, 1, 1);
}

// y = x W^T for m rows of x (row stride lda) into rows of y (stride ldy); a
// grouped W (wo_a) multiplies each input slice with its rows.
static void linear(const Lin *w, int m, const Buf *in, size_t in_off, uint32_t lda, const Buf *out,
                   size_t out_off, uint32_t ldy, int round) {
  if (small_rows(m)) {
    products(1, &(Prod){w, out, out_off, ldy}, m, in, in_off, lda, round);
    return;
  }
  int groups = w->group ? w->n / w->group : 1, n = w->n / groups, fp8 = w->s.obj != 0;
  if (fp8 && (size_t)(m + 127) / 128 * 128 * w->k * 2 > XB.size) die("linear: %d rows of %d", m, w->k);
  for (int g = 0; g < groups; g++) {
    MM p = {(uint32_t)m, (uint32_t)n, (uint32_t)w->k, (uint32_t)round, fp8 ? (uint32_t)w->k : lda, ldy};
    size_t row0 = (size_t)g * n, x = in_off + (size_t)g * w->k * 4;
    if (fp8) {  // the group's input as BF16 rows, which mm_fp8 reads from device memory
      uint32_t kl[2] = {(uint32_t)w->k, lda};
      gpu_kern(K_x_bf16);
      gpu_buf(0, in, x), gpu_buf(1, &XB, 0), gpu_bytes(2, kl, sizeof kl);
      gpu_go(w->k / 256, m, 1, 64, 1, 1);
    }
    gpu_kern(fp8 ? K_mm_fp8 : K_mm_bf16);
    gpu_buf(0, &w->w, row0 * w->k * (fp8 ? 1 : 2));
    if (fp8) gpu_buf(1, &w->s, row0 / 32 * (w->k / 32));
    gpu_buf(2, fp8 ? &XB : in, fp8 ? 0 : x);
    gpu_buf(3, out, out_off + row0 * 4);
    gpu_bytes(4, &p, sizeof p);
    gpu_go((n + 63) / 64, (m + (fp8 ? 127 : 63)) / (fp8 ? 128 : 64), 1, 128, 1, 1);
  }
}

// The logits of m rows of X (up to FEW) into y, rows of VOCAB.
static void head_out(int m, const Buf *y) {
  if (!head_q6.obj) {
    linear(&head, m, &X, 0, DIM, y, 0, VOCAB, 0);
    return;
  }
  for (int first = 0; first < m; first += KERNEL_ROWS) {
    uint32_t n = m - first < KERNEL_ROWS ? m - first : KERNEL_ROWS;
    gpu_kern(n == 1 ? K_gg_head1 : K_gg_head6);
    gpu_buf(0, &head_q6, 0), gpu_buf(1, &X, (size_t)first * DIM * 4);
    gpu_buf(2, y, (size_t)first * VOCAB * 4), gpu_bytes(3, &n, 4);
    gpu_go(VOCAB / 32, 1, 1, 256, 1, 1);
  }
}

static void quant(const Buf *in, const Buf *out, size_t n) {
  gpu_kern(K_act_quant);
  gpu_buf(0, in, 0);
  gpu_buf(1, out, 0);
  gpu_go(n / 256, 1, 1, 256, 1, 1);
}

// Rows [first, first + count) of 512 values between buffers whose row of
// position p is (p - base) % mod: kernel K_copy_rows for FP32 rows,
// K_copy_keys for BF16 (key) rows.
typedef struct { uint32_t first, count, src_base, src_mod, dst_base, dst_mod; } Copy;
static void copy_rows(const Buf *src, size_t src_off, const Buf *dst, size_t dst_off, Copy c, int k) {
  if (!c.count) return;
  gpu_kern(k);
  gpu_buf(0, src, src_off), gpu_buf(1, dst, dst_off), gpu_bytes(2, &c, sizeof c);
  gpu_go(2, c.count, 1, 256, 1, 1);
}

// Block y's hyper-connection mixes (of its attention, or its FFN) of streams
// h into mix, and in the dispatch of their second part, the block's input
// from h with the mix pre (see the forward pass) into X and XQ (hc_pre_norm);
// with xp, the attention's mixes also give the residual part of the FFN
// input into xp (see the residual function).
static void hc_mix(int m, const Buf *h, size_t h_off, const Layer *y, int ffn, const Buf *mix, size_t mix_off,
                   const Buf *pre, size_t pre_off, const Buf *xp) {
  uint32_t res = xp != 0;
  gpu_kern(K_hc_mix);
  gpu_buf(0, ffn ? &y->hc_ffn_fn : &y->hc_attn_fn, 0), gpu_buf(1, h, h_off), gpu_buf(2, &PART, 0);
  gpu_go(16, m, 1, 256, 1, 1);
  gpu_kern(K_hc_pre_norm);
  gpu_buf(0, h, h_off), gpu_buf(1, pre, pre_off), gpu_buf(2, ffn ? &y->ffn_norm : &y->attn_norm, 0);
  gpu_buf(3, &X, 0), gpu_buf(4, &XQ, 0), gpu_buf(5, &PART, 0), gpu_buf(6, ffn ? &y->hc_ffn_scale : &y->hc_attn_scale, 0);
  gpu_buf(7, ffn ? &y->hc_ffn_base : &y->hc_attn_base, 0), gpu_buf(8, mix, mix_off), gpu_bytes(11, &res, 4);
  if (xp) gpu_buf(9, &y->ffn_norm, 0), gpu_buf(10, xp, 0);
  gpu_go(m, 2, 1, 1024, 1, 1);
}

static void hc_pre_norm(int m, const Buf *h, size_t h_off, const Buf *mix, size_t mix_off, const Buf *w) {
  gpu_kern(K_hc_pre_norm);
  gpu_buf(0, h, h_off), gpu_buf(1, mix, mix_off), gpu_buf(2, w, 0), gpu_buf(3, &X, 0), gpu_buf(4, &XQ, 0);
  gpu_go(m, 1, 1, 1024, 1, 1);
}

// ---- Engram ----

double phase_s[7];  // decoding: rows read, routing waits, expert loads, prefetch, encoding, adapt, drafting
static int32_t history[3];  // compressed ids of the previous three tokens

// An image position (a negative token) takes part in no n-gram
// (NgramHashState's DEAD): its own n-gram is all pad, which engram() leaves
// unused, and the positions after it look back no further.
static void engram_hash(int token, int pos, int64_t rows[2][ENGRAM_ROWS]) {
  int64_t t[4];
  int dead = token < 0;
  t[0] = dead ? ngram.pad : ngram.map[token];
  for (int s = 1, stop = dead; s < 4; s++) {
    stop |= pos < s || history[s - 1] < 0;
    t[s] = stop ? ngram.pad : history[s - 1];
  }
  for (int e = 0; e < 2; e++) {
    int64_t roll = t[0] * ngram.multipliers[e * 4];
    for (int i = 1; i < 4; i++) {
      roll ^= t[i] * ngram.multipliers[e * 4 + i];
      for (int h = 0; h < 8; h++)
        rows[e][(i - 1) * 8 + h] =
            roll % ngram.primes[(e * 3 + i - 1) * 8 + h] + ngram.offsets[e * 24 + (i - 1) * 8 + h];
    }
  }
  history[2] = history[1], history[1] = history[0], history[0] = dead ? -1 : (int32_t)t[0];
}

// Reads the m tokens' table rows of Engram layer e into EC[e] and ES[e].
static void engram_read(int e, const int64_t (*rows)[2][ENGRAM_ROWS], int m) {
  Layer *y = &L[e ? 14 : 1];
  int64_t *r = memory_host_required((size_t)m * ENGRAM_ROWS * 8, 0);
  for (int i = 0; i < m; i++) memcpy(r + i * ENGRAM_ROWS, rows[i][e], ENGRAM_ROWS * 8);
  io_rows(y->e_codes.fd, y->e_codes.off, ENGRAM_DIM, r, m * ENGRAM_ROWS, EC[e].ptr);
  io_rows(y->e_scales.fd, y->e_scales.off, 8, r, m * ENGRAM_ROWS, ES[e].ptr);
  memory_host_free(r);
}

// ---- one layer for a batch of rows ----

typedef struct {
  int m, pos;  // rows, and the position of the first
  int row;     // its first row in the request buffers HS, MF and SEL
} Batch;

static size_t hs_off(const Batch *b) { return (size_t)b->row * HC * DIM * STREAM_BYTES; }
static size_t mf_off(const Batch *b) { return (size_t)b->row * MIX * 4; }

static void engram(const Batch *b, int l) {
  int e = l == 14, m = b->m;
  gpu_kern(K_engram_rows);
  gpu_buf(0, &EC[e], (size_t)b->row * ENGRAM_ROWS * ENGRAM_DIM), gpu_buf(1, &ES[e], (size_t)b->row * ENGRAM_ROWS * 8);
  gpu_buf(2, &EX, 0);
  gpu_go(m * ENGRAM_ROWS * ENGRAM_DIM / 256, 1, 1, 256, 1, 1);
  linear(&L[l].e_wkv, m, &EX, 0, ENGRAM_ROWS * ENGRAM_DIM, &EKV, 0, (HC + 1) * DIM, 1);
  gpu_kern(K_engram_gate);  // rows of image spans pass through (their gate is shut)
  gpu_buf(0, &HS, hs_off(b)), gpu_buf(1, &EKV, 0), gpu_buf(2, &L[l].e_qk, 0), gpu_buf(3, &IMG, b->row);
  gpu_go(HC, m, 1, 1024, 1, 1);
}

typedef struct { uint32_t pos, base, mod; } Win;
typedef struct { uint32_t ratio, start, row0; } Groups;
typedef struct { uint32_t pos, ratio, stride, cand; } Index;  // cand: words of a row's candidate bits, 0: none
// inference/config.json: candidate_source_layer, candidate_topk_blocks (of
// candidate_block_size 8 positions). Up to CAND_TOP blocks, all are candidates.
#define CAND_LAYER 20
#define CAND_TOP 2048
typedef struct { Win win; uint32_t first, comp_rows; } Att;

// Of the experts predicted for each of the next two layers, the top 3 are
// read ahead unless resident, while fewer than INFLIGHT prefetches are
// reading. The prediction one layer ahead, made a layer after the one two
// ahead, first cancels that layer's read-aheads not yet started outside its
// top KEEP.
#define MYDS4_PREFETCH_AHEAD 3
#define AHEAD MYDS4_PREFETCH_AHEAD
#define KEEP 6
#define INFLIGHT 6
static_assert(AHEAD >= 1 && AHEAD <= KEEP, "bounded native prefetch candidates");

// A routing of a batch's rows: their logits (rows of y's gate.n), the layer
// whose bias selects, where row 0's indices go (the rows' at stride), and the
// top k, weighted or not.
typedef struct { const Buf *lg; size_t lg_off; const Layer *y; const Buf *out; size_t off; uint32_t k, stride, weighted; } Routing;
static void route_top(int m, const Routing *r, int count, int row);

// Whether m rows read ahead as a decoded row does (see read_ahead): one row,
// or a verify pass's rows when decoding exactly. With --substitute an expert
// still being read is waited for, not replaced, so there a verify pass's
// rows read ahead only the next layer's (read_ahead_rows): reading ahead as
// many as for single rows doubled its reads ahead and took 8% of the speed.
static int native_forecasts(void) {
  return !runahead_route_only;
}

static int full_ahead(int m) { return !prompt_pass && native_forecasts() && (m == 1 || (small_rows(m) && opt.deepseek.substitute <= 0)); }

// Decoding (m <= FEW rows): layer l's top KEEP experts predicted from the
// residual part of each row's FFN input (XP, see hc_mix), known before its
// attention, into PC, row r's at r * PREDICT; for the routes trace of one
// row also its top PREDICT into PT.
static void predict_residual(int m, int l) {
  linear(&L[l].gate, m, &XP, 0, DIM, &GP, 0, EXPERTS, 0);
  Routing r[2] = {{&GP, 0, &L[l], &PC, 0, KEEP, PREDICT, 0}, {&GP, 0, &L[l], &PT, 0, PREDICT, PREDICT, 0}};
  route_top(m, r, m == 1 && opt.routes_fd >= 0 ? 2 : 1, 0);
}

// One row's attention when decoding: chunks of ATT_CHUNK of the up to
// ATT_ROWS rows it sees spread over threadgroups (see attend_scores): the
// scores of 4 heads and the values of 16 heads per pair of chunks, then the
// chunks' sums per head and half of its columns.
static void attend_one(const Buf *q, size_t q_off, const Buf *ring, const Buf *comp, const Buf *sel, size_t sel_off,
                       const Buf *sink, const Buf *cs, size_t cs_off, const Buf *o, size_t o_off, const Att *at) {
  gpu_kern(K_attend_scores);
  gpu_buf(0, q, q_off), gpu_buf(1, ring, 0), gpu_buf(2, comp, 0), gpu_buf(3, sel, sel_off), gpu_buf(4, cs, cs_off);
  gpu_buf(5, &ASC, 0), gpu_buf(6, &AMAX, 0), gpu_bytes(7, at, sizeof *at);
  gpu_go(HEADS / 4, (ATT_CHUNKS + 1) / 2, 1, 512, 1, 1);
  gpu_kern(K_attend_values);
  gpu_buf(0, ring, 0), gpu_buf(1, comp, 0), gpu_buf(2, sel, sel_off), gpu_buf(3, &ASC, 0), gpu_buf(4, &AMAX, 0);
  gpu_buf(5, &APART, 0), gpu_bytes(6, at, sizeof *at);
  gpu_go(HEADS / 16 * 2, (ATT_CHUNKS + 1) / 2, 1, 512, 1, 1);
  gpu_kern(K_attend_combine);
  gpu_buf(0, &APART, 0), gpu_buf(1, &AMAX, 0), gpu_buf(2, sink, 0), gpu_buf(3, o, o_off), gpu_buf(4, sel, sel_off);
  gpu_bytes(5, at, sizeof *at);
  gpu_go(HEADS, 2, 1, 128, 1, 1);
}

// Rows of several sequences at once (deepseek_forward_rows): row r is the token
// at position row_pos[r] of state row_state[r]. 0 rows: one sequence.
static State *row_state[FEW];
static int row_pos[FEW], nrows;
static void bind(const State *s);


// The attention of the batch's rows r0 .. r0 + m - 1, one sequence at
// positions pos ..: their keys into the window, a KV source's compressor,
// an index source's selection, sparse attention and the outputs' inverse
// RoPE. Their index selections go to SEL's rows from srow.
static void attend_rows(int l, int m, int pos, int r0, int srow) {
  Layer *y = &L[l], *src = &L[y->src];
  const Buf *cs = y->ratio ? &cs_comp : &cs_base;
  uint32_t p32 = pos;
  // The window keys: the ring when decoding; for a prompt batch, rows of WIN
  // from the earliest position any of its rows sees, loaded from the ring.
  const Buf *win = m == 1 ? &y->ring : &WIN;
  uint32_t base = m == 1 ? 0 : (pos >= WINDOW - 1 ? pos - (WINDOW - 1) : 0), mod = m == 1 ? RING : 1u << 30;
  size_t x = (size_t)r0 * DIM * 4, q = (size_t)r0 * HEADS * HEAD_DIM * 4;
  if (m > 1) copy_rows(&y->ring, 0, &WIN, 0, (Copy){base, pos - base, 0, RING, base, mod}, K_copy_keys);
  Win wp = {p32, base, mod};
  gpu_kern(K_kv_window);
  gpu_buf(0, &KV, (size_t)r0 * HEAD_DIM * 4), gpu_buf(1, &y->kv_norm, 0), gpu_buf(2, cs, 0), gpu_buf(3, win, 0);
  gpu_bytes(4, &wp, sizeof wp);
  gpu_go(m, 1, 1, 512, 1, 1);
  int visible = y->ratio ? (pos + m) / y->ratio : 0;  // compressed rows the last row sees
  if (y->kv_source) {
    // CK and CS rows 2.. are the batch's positions; rows 0, 1 the two before,
    // kept in cstate between batches.
    size_t row2 = 2 * HEAD_DIM * 4, half = PENDING * HEAD_DIM * 4;  // half: cstate's score rows
    Copy in = {pos >= 2 ? pos - 2 : 0, pos >= 2 ? 2 : pos, 0, PENDING, pos - 2, 1u << 30};
    linear(&y->c_wkv, m, &X, x, DIM, &CK, row2, HEAD_DIM, y->ratio == 1);
    uint32_t g0 = pos, groups = m;  // groups completed in this batch
    if (y->ratio == 2) {
      copy_rows(&y->cstate, 0, &CK, 0, in, K_copy_rows);
      copy_rows(&y->cstate, half, &CS, 0, in, K_copy_rows);
      linear(&y->c_wgate, m, &X, x, DIM, &CS, row2, HEAD_DIM, 0);
      g0 = pos & ~1u, groups = (pos + m - g0) / 2;
      uint32_t w = m < PENDING ? m : PENDING;
      Copy out = {pos + m - w, w, pos - 2, 1u << 30, 0, PENDING};
      copy_rows(&CK, 0, &y->cstate, 0, out, K_copy_rows);
      copy_rows(&CS, 0, &y->cstate, half, out, K_copy_rows);
    }
    if (bounded && groups) {  // the copied rows of groups completed before the branch point stay
      uint32_t r = (uint32_t)y->ratio, keep = bounded_keep / r * r;  // the first group ending at or after it
      if (keep > g0) {
        uint32_t skip = (keep - g0) / r;
        groups = skip < groups ? groups - skip : 0, g0 = keep;
      }
    }
    if (groups) {
      Groups g = {(uint32_t)y->ratio, g0, g0 - pos + 2};
      gpu_kern(K_compress);
      gpu_buf(0, &CK, 0), gpu_buf(1, &CS, 0), gpu_buf(2, &y->c_norm, 0), gpu_buf(3, cs, 0);
      gpu_buf(4, &LAT, 0), gpu_buf(5, &y->comp, 0), gpu_bytes(6, &g, sizeof g);
      gpu_go(groups, 1, 1, 512, 1, 1);
      linear(&y->i_wk, groups, &LAT, 0, HEAD_DIM, &IK, 0, IDX_DIM, 1);
      gpu_kern(K_index_key);
      gpu_buf(0, &IK, 0), gpu_buf(1, &y->i_knorm, 0), gpu_buf(2, cs, 0), gpu_buf(3, &y->keys, 0);
      gpu_bytes(4, &g, sizeof g);
      gpu_go(groups, 1, 1, 128, 1, 1);
    }
  }
  size_t sel_off = (size_t)srow * (IDX_TOPK + 1) * 4;
  if (y->index_source && visible > 0) {
    // index keys live with the KV source of the same ratio (layer 20 for
    // ratio 1). Beyond CAND_TOP blocks, layer 20 keeps its best blocks as
    // candidates, and the later ratio-1 indexers select within them.
    int cand = y->ratio == 1 && visible > CAND_TOP * 8;
    for (int t0 = 0; t0 < m; t0 += sc_rows) {  // rows in tiles, as many as SC holds
      uint32_t rows = m - t0 < sc_rows ? m - t0 : sc_rows, seen = (pos + t0 + rows) / y->ratio;
      Index ix = {p32 + t0, (uint32_t)y->ratio, (uint32_t)opt.ctx, cand && l > CAND_LAYER ? cand_words : 0};
      // Layer 20 writes once; later indexers reuse the same prompt row,
      // even when decoder compaction starts partway through a batch.
      size_t cand_off = (size_t)(srow + t0) * cand_words * 4;
      gpu_kern(K_index_score);
      gpu_buf(0, &QI, (size_t)(r0 + t0) * IDX_HEADS * IDX_DIM * 4), gpu_buf(1, &IW, (size_t)(r0 + t0) * IDX_HEADS * 4);
      gpu_buf(2, &src->keys, 0), gpu_buf(3, cs, 0), gpu_buf(4, &SC, 0), gpu_bytes(5, &ix, sizeof ix);
      gpu_go((seen + 255) / 256, rows, 1, 256, 1, 1);
      if (cand && l == CAND_LAYER) {
        Index cx = ix;
        cx.cand = cand_words;
        gpu_kern(K_cand_topk);
        gpu_buf(0, &SC, 0), gpu_buf(1, &CAND, cand_off), gpu_bytes(2, &cx, sizeof cx);
        gpu_go(rows, 1, 1, 1024, 1, 1);
      }
      gpu_kern(K_index_topk);
      gpu_buf(0, &SC, 0), gpu_buf(1, &SEL, sel_off + (size_t)t0 * (IDX_TOPK + 1) * 4), gpu_bytes(2, &ix, sizeof ix);
      gpu_buf(3, &CAND, cand_off);
      gpu_go(rows, 1, 1, 1024, 1, 1);
    }
  }
  if (small_rows(m)) {  // one row or a verify pass, each as decoding runs it
    for (int r = 0; r < m; r++) {
      Att ar = {{p32 + r, base, mod}, 0, visible > 0};
      ar.first = window_lower(l, p32 + r);
      attend_one(&Q, q + (size_t)r * HEADS * HEAD_DIM * 4, win, y->ratio ? &src->comp : win, &SEL,
                 sel_off + (size_t)r * (IDX_TOPK + 1) * 4, &y->sink, cs, 0, &O, q + (size_t)r * HEADS * HEAD_DIM * 4, &ar);
    }
  } else {  // many rows: on the matrix units, 32 heads per threadgroup, QB_ROWS rows at a time
    for (int r = 0; r < m;) {
      uint32_t rows = m - r < QB_ROWS ? m - r : QB_ROWS, p = p32 + r;
      size_t qr = q + (size_t)r * HEADS * HEAD_DIM * 4;
      Att ar = {{p, base, mod}, 0, visible > 0};
      // Prefix queries stay unchanged. Retained queries cannot read them.
      // Split at the boundary so first never exceeds a dispatched query position.
      if (ced_active && l >= 20 && p < ced_first && p + rows > ced_first) rows = ced_first - p;
      ar.first = window_lower(l, p);
      gpu_kern(K_q_rope);
      gpu_buf(0, &Q, qr), gpu_buf(1, cs, 0), gpu_buf(2, &XB, 0), gpu_bytes(3, &p, 4);
      gpu_go(HEADS * HEAD_DIM / 256, rows, 1, 256, 1, 1);
      gpu_kern(K_attend_m);
      gpu_buf(0, &XB, 0), gpu_buf(1, win, 0), gpu_buf(2, y->ratio ? &src->comp : win, 0);
      gpu_buf(3, &SEL, sel_off + (size_t)r * (IDX_TOPK + 1) * 4), gpu_buf(4, &y->sink, 0), gpu_buf(5, &O, qr);
      gpu_bytes(6, &ar, sizeof ar);
      gpu_go(HEADS / 32, rows, 1, 256, 1, 1);
      r += rows;
    }
  }
  if (m > 1) {  // the batch's last positions become the ring
    // Saved server states can rewind BLOCK positions before continuing.
    uint32_t keep = m < RING ? m : RING;
    copy_rows(&WIN, 0, &y->ring, 0, (Copy){pos + m - keep, keep, base, mod, 0, RING}, K_copy_keys);
  }
  gpu_kern(K_attn_out);
  gpu_buf(0, &O, q), gpu_buf(1, cs, 0), gpu_buf(2, &O, q), gpu_bytes(3, &p32, 4);
  gpu_go(m * HEADS * HEAD_DIM / 256, 1, 1, 256, 1, 1);
}

static void attention(const Batch *b, int l) {
  Layer *y = &L[l];
  int m = b->m;
  hc_mix(m, &HS, hs_off(b), y, 0, &MA, 0, &MF, mf_off(b), full_ahead(m) ? &XP : 0);
  if (full_ahead(m)) {  // decoding: this layer's experts predicted from the residual known before attention
    predict_residual(m, l);
    gpu_signal();
  }
  products(2, (Prod[]){{&y->wq_a, &QA, 0, QRANK}, {&y->wkv, &KV, 0, HEAD_DIM}}, m, &XQ, 0, DIM, 1);
  uint32_t n = QRANK;
  gpu_kern(K_rmsnorm);
  gpu_buf(0, &QA, 0), gpu_buf(1, &y->q_norm, 0), gpu_buf(2, &QR, 0), gpu_buf(3, &QRQ, 0), gpu_bytes(4, &n, 4);
  gpu_go(m, 1, 1, 1024, 1, 1);
  products(y->index_source ? 2 : 1, (Prod[]){{&y->wq_b, &Q, 0, HEADS * HEAD_DIM}, {&y->i_wq_b, &QI, 0, IDX_HEADS * IDX_DIM}},
           m, &QRQ, 0, QRANK, 1);
  if (y->index_source) linear(&y->i_wproj, m, &X, 0, DIM, &IW, 0, IDX_HEADS, 1);
  if (nrows) {  // each row with its own sequence's state
    for (int r = 0; r < m; r++) bind(row_state[r]), attend_rows(l, 1, row_pos[r], r, b->row + r);
    bind(active);
  } else {
    attend_rows(l, m, b->pos, 0, b->row);
  }
  linear(&y->wo_a, m, &O, 0, HEADS * HEAD_DIM, &OA, 0, OGROUPS * ORANK, 1);
  quant(&OA, &OA, (size_t)m * OGROUPS * ORANK);
  linear(&y->wo_b, m, &OA, 0, OGROUPS * ORANK, &AO, 0, DIM, 1);
  gpu_kern(K_hc_post);
  gpu_buf(0, &AO, 0), gpu_buf(1, &MA, 0), gpu_buf(2, &HS, hs_off(b)), gpu_buf(3, &H1, 0);
  gpu_go(DIM / 256, m, 1, 256, 1, 1);
}

// The top k of each routing's experts per row, up to 3 routings of the same
// gate.n in one dispatch; weighted: also their weights. Rows inside image
// spans (IMG from row) select with the layers' bias_vl.
static void route_top(int m, const Routing *r, int count, int row) {
  struct { uint32_t k, stride, n, weighted; } rp[3];
  gpu_kern(K_route);
  for (int j = 0; j < count; j++) {
    rp[j].k = r[j].k, rp[j].stride = r[j].stride, rp[j].n = (uint32_t)r[j].y->gate.n, rp[j].weighted = r[j].weighted;
    gpu_buf(3 * j, r[j].lg, r[j].lg_off), gpu_buf(3 * j + 1, &r[j].y->gate_bias, 0), gpu_buf(3 * j + 2, r[j].out, r[j].off);
    gpu_buf(10 + j, &r[j].y->gate_bias_vl, 0);
  }
  gpu_buf(13, &IMG, (size_t)row);
  gpu_bytes(9, rp, count * sizeof rp[0]);
  gpu_go(m, count, 1, r[0].y->gate.n, 1, 1);
}

// The FFN's mix, the router (RT: 6 indices, then 6 weights at 8, per row)
// and predictions of the next layers' experts (their routers applied to this
// layer's FFN input; prediction a of row r at PR row s + a m + r): the
// products in one dispatch and the top-k in another, when the rows are few.
// Decoding predicts the next layer and the one after, for more time to read.
static void ffn_front(const Batch *b, int l) {
  Layer *y = &L[l];
  int m = b->m;
  hc_mix(m, &H1, 0, y, 1, &MF, mf_off(b), &MA, 0, 0);
  Prod p[3] = {{&y->gate, &GL, 0, EXPERTS}};
  Routing r[3] = {{&GL, 0, y, &RT, 0, TOPK, 16, 1}};
  int n = 1, full = full_ahead(m);
  for (int a = 0; native_forecasts() && a < (full ? 2 : 1) && l + 1 + a < LAYERS; a++, n++) {
    uint32_t k = !full || opt.routes_fd >= 0 ? PREDICT : a ? AHEAD : KEEP;
    p[n] = (Prod){&L[l + 1 + a].gate, &GP, (size_t)a * m * EXPERTS * 4, EXPERTS};
    r[n] = (Routing){&GP, p[n].off, &L[l + 1 + a], &PR, (size_t)a * m * PREDICT * 4, k, PREDICT, 0};
  }
  products(n, p, m, &X, 0, DIM, 0);
  route_top(m, r, n, b->row);
  if (!native_forecasts()) runahead_route_only_rows += m;
}

static void shared_expert(int m, const Layer *y) {
  products(2, (Prod[]){{&y->sh1, &SHG, 0, INTER}, {&y->sh3, &SHU, 0, INTER}}, m, &XQ, 0, DIM, 1);
  gpu_kern(K_swiglu);
  gpu_buf(0, &SHG, 0), gpu_buf(1, &SHU, 0), gpu_buf(2, &SHH, 0);
  gpu_go(m * INTER / 256, 1, 1, 256, 1, 1);
  linear(&y->sh2, m, &SHH, 0, INTER, &SHO, 0, DIM, 1);
}

typedef struct { uint64_t p[3]; uint32_t start, count; } MoeTile;

// --substitute D (off by default; it changes the output): a decoded or
// prompt row's selected experts that have no slot give way to the block's
// resident, fully read experts with the highest selection keys, if such a key
// is at most D below the replaced one's, the lowest-ranked selected expert first:
// the selection becomes the top k of the keys with D added to those of the
// resident experts (the cache prior of Skliar et al., arXiv 2412.00099). The
// SUB_KEEP selected experts with the highest scores, which weigh most, are
// never replaced. The k weights then follow the router's formula for the
// experts used. z: the row's n router logits, idx: its k experts in key
// order (as the route kernel picks them); cl: the block's cache layer;
// read: experts that are read anyway, kept like those with a slot (or 0).
// Returns how many experts it replaced.
#define SUB_KEEP 2
static int substitute(int cl, const Layer *y, const float *z, int n, int *idx, float *wts, int k, const char *read, int vl) {
  int missing = 0;
  for (int j = 0; j < k; j++) missing |= !cache_slot(experts_store, cl, idx[j]) && !(read && read[idx[j]]);
  if (!missing) return 0;
  const float *bias = (const float *)(vl ? y->gate_bias_vl : y->gate_bias).ptr;
  float key[EXPERTS], sc[EXPERTS];
  int changed = 0, keep[8] = {0};
  for (int e = 0; e < n; e++) {  // as the route kernel computes them
    float u = expf(z[e]), v = 1 + u;
    sc[e] = sqrtf(z[e] > 20 ? z[e] : v == 1 ? u : logf(v) * u / (v - 1));
    key[e] = sc[e] + bias[e];
  }
  for (int p = 0; p < SUB_KEEP && p < k; p++) {
    int b = -1;
    for (int j = 0; j < k; j++)
      if (!keep[j] && (b < 0 || sc[idx[j]] > sc[idx[b]])) b = j;
    keep[b] = 1;
  }
  for (int j = k - 1; j >= 0; j--) {
    if (keep[j] || cache_slot(experts_store, cl, idx[j]) || (read && read[idx[j]])) continue;
    int best = -1;
    for (int e = 0; e < n; e++) {
      int taken = 0;
      for (int i = 0; i < k; i++) taken |= idx[i] == e;
      if (!taken && key[e] >= key[idx[j]] - opt.deepseek.substitute && (best < 0 || key[e] > key[best]) && cache_ready(experts_store, cl, e))
        best = e;
    }
    if (best >= 0) idx[j] = best, changed++;
  }
  if (!changed) return 0;
  float sum = 0;
  for (int j = 0; j < k; j++) sum += sc[idx[j]];
  for (int j = 0; j < k; j++) wts[j] = sc[idx[j]] / (sum + 1e-20f) * 1.5f;
  return changed;
}

// A prompt batch's rows: an expert without a slot is read if some row
// cannot do without it (one of the row's SUB_KEEP highest scores, or no
// resident expert within D of its key); the rows' other selections of
// experts without a slot give way as a decoded row's do. The rows run in
// parallel, first to find the experts to read, then to substitute.
static int sub_layer, sub_row, sub_pass, sub_count[MB];
static char sub_read[EXPERTS];
static void substitute_row(void *ctx, size_t r) {
  int *rt = (int *)RT.ptr + r * 16, idx[TOPK];
  float *rw = (float *)RT.ptr + r * 16 + 8, w[TOPK];
  const float *z = (const float *)GL.ptr + r * EXPERTS;
  int vl = ((const uint8_t *)IMG.ptr)[sub_row + r];
  if (sub_pass) {
    sub_count[r] = substitute(sub_layer, &L[sub_layer], z, EXPERTS, rt, rw, TOPK, sub_read, vl);
    return;
  }
  memcpy(idx, rt, sizeof idx), memcpy(w, rw, sizeof w);
  substitute(sub_layer, &L[sub_layer], z, EXPERTS, idx, w, TOPK, 0, vl);
  for (int k = 0; k < TOPK; k++)
    if (!cache_slot(experts_store, sub_layer, idx[k])) sub_read[idx[k]] = 1;  // rows may race here: each stores 1
}
static void substitute_batch(int l, int m, int row) {
  sub_layer = l, sub_row = row, memset(sub_read, 0, sizeof sub_read);
  for (sub_pass = 0; sub_pass < 2; sub_pass++)
    dispatch_apply_f(m, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), 0, substitute_row);
  for (int r = 0; r < m; r++) expert_stats()->substituted += sub_count[r];
}

// Row r's routed experts as its entries in ETAB (from eo) name them, in
// expert order (the reference sums them in that order), as a decoded row runs
// them; mf and hs: the row's offsets in MF and HS. With pm, only the experts
// with entries run, and their down products go to PARTS (see moe_parts).
static void experts_go(int l, int r, size_t eo, size_t mf, size_t hs, uint32_t pm) {
  gpu_kern(gk[l].up ? gk[l].up : K_moe_up);
  gpu_buf(0, &ETAB, eo), gpu_buf(1, &XQ, (size_t)r * DIM * 4), gpu_buf(2, &MH, (size_t)r * TOPK * INTER * 4);
  gpu_go(INTER / 32, TOPK, 1, 256, 1, 1);
  uint32_t n = TOPK, gg = gk[l].up != 0;  // GGUF: 8 rows of w2 per threadgroup, two simdgroups per expert
  gpu_kern(gg ? gk[l].down : K_moe_down);
  gpu_buf(0, &ETAB, eo), gpu_buf(1, &MH, (size_t)r * TOPK * INTER * 4), gpu_buf(2, &SHO, (size_t)r * DIM * 4);
  gpu_buf(3, &MF, mf), gpu_buf(4, &H1, (size_t)r * HC * DIM * STREAM_BYTES), gpu_buf(5, &HS, hs), gpu_bytes(6, &n, 4);
  gpu_bytes(7, &pm, 4), gpu_buf(8, &PARTS, (size_t)r * DIM * 8 * 4);
  gpu_go(DIM / (gg ? 8 : 4), 1, 1, 32 * TOPK * (gg ? 2 : 1), 1, 1);
}

// Row r's routed experts (resident) into ETAB entries from eo, in expert
// order (the reference sums them in that order); with `ready` (by expert),
// only those whose ready is want, the others null.
static void fill_entries(int l, size_t eo, const int *experts, const float *weights, const char *ready, int want) {
  int idx[TOPK];
  float wts[TOPK];
  memcpy(idx, experts, sizeof idx);
  memcpy(wts, weights, sizeof wts);
  for (int i = 1; i < TOPK; i++)
    for (int j = i; j > 0 && idx[j - 1] > idx[j]; j--) {
      int t = idx[j]; idx[j] = idx[j - 1], idx[j - 1] = t;
      float u = wts[j]; wts[j] = wts[j - 1], wts[j - 1] = u;
    }
  ExpertRef *ref = (ExpertRef *)(ETAB.ptr + eo);
  for (int k = 0; k < TOPK; k++) {
    uint32_t at[STORE_PARTS];
    int fresh;
    const Buf *b = !ready || ready[idx[k]] == want ? cache_get(experts_store, l, idx[k], at, &fresh) : 0;
    ref[k] = b ? (ExpertRef){{b->gpu + at[0], b->gpu + at[1], b->gpu + at[2]}, wts[k], 0} : (ExpertRef){{0, 0, 0}, 0, 0};
    if (b && fresh) gpu_use(b);
  }
}

// Row r's routed experts (resident) into its entries in ETAB, then their products.
static void encode_experts(int l, int r, const int *experts, const float *weights, size_t mf, size_t hs) {
  size_t eo = (size_t)(l * FEW + r) * 256;
  fill_entries(l, eo, experts, weights, 0, 0);
  experts_go(l, r, eo, mf, hs, 0);
}

// Row r's routed experts from their parts in PARTS, as the down kernels
// would sum them, into the streams.
static void experts_sum(int r, size_t mf, size_t hs) {
  uint32_t n = TOPK;
  gpu_kern(K_moe_parts);
  gpu_buf(0, &PARTS, (size_t)r * DIM * 8 * 4), gpu_buf(1, &SHO, (size_t)r * DIM * 4), gpu_buf(2, &MF, mf);
  gpu_buf(3, &H1, (size_t)r * HC * DIM * STREAM_BYTES), gpu_buf(4, &HS, hs), gpu_bytes(5, &n, 4);
  gpu_go(DIM / 256, 1, 1, 256, 1, 1);
}

// Where routed_rows writes the entries of row r's experts that were still
// being read while the others ran, apart from those of the others.
static size_t late_entries(int l, int r) { return (size_t)(LAYERS * FEW + DRAFTS * BLOCK + l * FEW + r) * 256; }

// The read-aheads of m <= FEW decoded rows after layer l's routing, from
// the predictions pr for the next two layers (see AHEAD; row r's at
// r PREDICT and (m + r) PREDICT), while fewer than m INFLIGHT are reading.
static void read_ahead(int l, int m, const int *pr) {
  int keep[FEW * KEEP], next[FEW * AHEAD], after[FEW * AHEAD], nk = 0, n = 0;
  for (int r = 0; r < m; r++) {
    for (int k = 0; k < KEEP; k++) keep[nk++] = pr[r * PREDICT + k];
    for (int k = 0; k < AHEAD; k++) next[n] = pr[r * PREDICT + k], after[n++] = pr[(m + r) * PREDICT + k];
  }
  int inflight = INFLIGHT * m;
  if (l + 1 < LAYERS) cache_refine(experts_store, l + 1, keep, nk);
  if (l + 1 < LAYERS) cache_prefetch_from(experts_store, l + 1, next, n, inflight, PF_H1);
  if (l + 2 < LAYERS) cache_prefetch_from(experts_store, l + 2, after, n, inflight, PF_H2);
}

static void read_residual_ahead(int layer, const int *keep, int nkeep, const int *first, int nfirst, int inflight) {
  cache_refine(experts_store, layer, keep, nkeep);
  cache_prefetch_from(experts_store, layer, first, nfirst, inflight, PF_PC);
}

// A batch's read-aheads after layer l's routing: the next layer's experts
// that the rows' top `ahead` predictions pr name (every expert if ahead >
// PREDICT), while fewer than `inflight` prefetches are reading.
static void read_ahead_rows(int m, int l, const int *pr, int ahead, int inflight) {
  if (l + 1 >= LAYERS) return;
  char want[EXPERTS] = {0};
  int list[EXPERTS], nl = 0;
  if (ahead > PREDICT) memset(want, 1, sizeof want);  // every expert
  for (int r = 0; r < m && ahead <= PREDICT; r++)
    for (int k = 0; k < ahead; k++) want[pr[r * PREDICT + k]] = 1;
  for (int e = 0; e < EXPERTS; e++)
    if (want[e]) list[nl++] = e;
  cache_prefetch_from(experts_store, l + 1, list, nl, inflight, PF_H1);
}


// The experts each decoded row chose in the model's layers, where each one
// was (0 in memory, 1 read ahead, 2 late, 3 substituted, 4 from the SSD) and
// the decode step that chose them, for the live view (deepseek_routes). The
// router's results are on the CPU already: recording them is a copy.
static uint16_t route_id[FEW][LAYERS][TOPK];
static uint8_t route_from[FEW][LAYERS][TOPK];
static uint64_t route_step[FEW][LAYERS], route_steps;

int deepseek_routes(int row, uint16_t *id, uint8_t *from) {
  if (row < 0 || row >= FEW || !route_steps) return 0;
  for (int l = 0; l < LAYERS; l++)
    if (route_step[row][l] != route_steps) return 0;  // the row did not run every layer in the latest step
  memcpy(id, route_id[row], sizeof route_id[row]);
  memcpy(from, route_from[row], sizeof route_from[row]);
  return 1;
}

// The routed experts of m <= FEW rows (RT), each as decoding runs it. While
// some are still being read, those in memory run first, their down products
// as parts (see moe_parts), in the command buffer open now, which is committed
// at once; the others and the rows' sums follow in `rest` once they are in.
// Returns whether it moved on to `rest`. approx: see substitute.
static int routed_rows(const Batch *b, int l, int ahead, int inflight, int approx, void *rest) {
  int m = b->m, *rt = (int *)RT.ptr, need[EXPERTS], nn = 0, nready = 0, chosen[FEW * TOPK];
  float *rw = (float *)RT.ptr;
  char used[EXPERTS] = {0}, ready[EXPERTS] = {0};
  CacheObservation observed;
  cache_observe_snapshot(experts_store, l, &observed);
  cache_observe_routes(experts_store, &observed, rt, m, 16, 0);
  int native_rt[FEW * 16];
  if (opt.deepseek.online_predictor && approx) memcpy(native_rt, rt, (size_t)m * 16 * sizeof *rt);
  for (int r = 0; approx && r < m; r++) {
    memcpy(chosen + r * TOPK, rt + r * 16, TOPK * sizeof *chosen);
    expert_stats()->substituted += substitute(l, &L[l], (const float *)GL.ptr + (size_t)r * EXPERTS, EXPERTS, rt + r * 16,
                                          rw + r * 16 + 8, TOPK, 0, ((const uint8_t *)IMG.ptr)[b->row + r]);
  }
  cache_observe_routes(experts_store, &observed, rt, m, 16, 1);
  uint64_t *tally[] = {&expert_stats()->row_ssd, &expert_stats()->row_own, &expert_stats()->row_ahead, &expert_stats()->row_late,
                       &expert_stats()->row_sub};
  int record = !prompt_pass && l < LAYERS;  // the model's own layers, in decode steps
  for (int r = 0; r < m; r++)
    for (int k = 0; k < TOPK; k++) { // each row's selections, by where their experts are now
      int e = rt[r * 16 + k];
      used[e] = 1;
      int where = approx && e != chosen[r * TOPK + k] ? 4 : cache_where(experts_store, l, e);
      (*tally[where])++;
      if (record && b->row + r < FEW) {  // tally's order (from the SSD first) to route_from's
        route_id[b->row + r][l][k] = (uint16_t)e, route_from[b->row + r][l][k] = (uint8_t)((where + 4) % 5);
        route_step[b->row + r][l] = route_steps;
      }
    }
  for (int e = 0; e < EXPERTS; e++)
    if (used[e]) need[nn++] = e;
  expert_stats()->row_unique += nn;
  double t0 = now();
  int reading = cache_request(experts_store, l, need, nn);  // this layer's reads first, then the predicted ones
  cache_demote(experts_store, l);
  if (reading) gpu_keep_clock(reading * 1e-3); // the GPU idles until they are in
  if (opt.deepseek.online_predictor) {
    const int *forecast =
        online_predictor_forecasts(l, m, full_ahead(m) ? 2 : 1, (const float *)GP.ptr, (const float *)GL.ptr,
                                   (const int *)PR.ptr, approx ? native_rt : rt);
    if (forecast) {
      if (full_ahead(m)) read_ahead(l, m, forecast);
      else read_ahead_rows(m, l, forecast, ahead, inflight);
    }
  } else {
    if (full_ahead(m)) read_ahead(l, m, (const int *)PR.ptr);
    else read_ahead_rows(m, l, (const int *)PR.ptr, ahead, inflight);
  }
  for (int i = 0; i < nn; i++) nready += ready[need[i]] = cache_ready(experts_store, l, need[i]);
  int split = rest && nready > 0 && nready < nn;
  for (int r = 0; split && r < m; r++) {
    size_t eo = (size_t)(l * FEW + r) * 256;
    fill_entries(l, eo, rt + r * 16, rw + r * 16 + 8, ready, 1);
    experts_go(l, r, eo, mf_off(b) + (size_t)r * MIX * 4, hs_off(b) + (size_t)r * HC * DIM * STREAM_BYTES, 1);
  }
  if (split) gpu_commit(), gpu_encode(rest);
  double t1 = now();
  cache_wait(experts_store, l, need, nn);
  phase_s[3] += t1 - t0, phase_s[2] += now() - t1, phase_s[4] -= now() - t0;
  for (int r = 0; r < m; r++) {
    size_t mf = mf_off(b) + (size_t)r * MIX * 4, hs = hs_off(b) + (size_t)r * HC * DIM * STREAM_BYTES;
    if (!split) {
      encode_experts(l, r, rt + r * 16, rw + r * 16 + 8, mf, hs);
      continue;
    }
    fill_entries(l, late_entries(l, r), rt + r * 16, rw + r * 16 + 8, ready, 0);
    experts_go(l, r, late_entries(l, r), mf, hs, 1);
    experts_sum(r, mf, hs);
  }
  return split;
}

// A batch's routed experts: the (row, expert) pairs grouped by expert, in
// tiles of up to 32 pairs of one expert (64 for the GGUF codes).
// The next layer's experts that the rows' top `ahead` predictions name are
// read ahead (see read_ahead_rows), while fewer than `inflight` prefetches
// are reading.
static void routed_batch(const Batch *b, int l, int ahead, int inflight, int approx) {
  if (small_rows(b->m)) {  // a short prompt: each row as decoding runs it
    routed_rows(b, l, ahead, inflight, approx, 0);
    return;
  }
  int m = b->m, count[EXPERTS] = {0}, start[EXPERTS], fill[EXPERTS] = {0}, need[EXPERTS], nn = 0;
  const int *rt = (const int *)RT.ptr;
  const float *rw = (const float *)RT.ptr;
  CacheObservation observed;
  cache_observe_snapshot(experts_store, l, &observed);
  cache_observe_routes(experts_store, &observed, rt, m, 16, 0);
  if (approx) substitute_batch(l, m, b->row);
  cache_observe_routes(experts_store, &observed, rt, m, 16, 1);
  for (int r = 0; r < m; r++)
    for (int k = 0; k < TOPK; k++) count[rt[r * 16 + k]]++;
  for (int e = 0, s = 0; e < EXPERTS; e++) {
    start[e] = s, s += count[e];
    if (count[e]) need[nn++] = e;
  }
  double t0 = now();
  cache_request(experts_store, l, need, nn);
  cache_demote(experts_store, l);
  if (full_ahead(m)) read_ahead(l, m, (const int *)PR.ptr);
  else {
    int skip = 0;
    // At the boundary, discarded rows cannot justify reads for the next layer.
    if (ced_active && ced_compact && l + 1 > 20 && ced_first > (uint32_t)b->pos)
      skip = ced_first - b->pos < (uint32_t)m ? (int)ced_first - b->pos : m;
    if (skip < m) read_ahead_rows(m - skip, l, (const int *)PR.ptr + skip * PREDICT, ahead, inflight);
  }
  double t1 = now();
  cache_wait(experts_store, l, need, nn);
  phase_s[3] += t1 - t0, phase_s[2] += now() - t1, phase_s[4] -= now() - t0;
  int *ptok = (int *)PTOK.ptr, *tp = (int *)TP.ptr;
  float *pw = (float *)PW.ptr;
  for (int r = 0; r < m; r++) {
    int ek[TOPK], qk[TOPK];
    for (int k = 0; k < TOPK; k++) {
      int e = rt[r * 16 + k], q = start[e] + fill[e]++;
      ptok[q] = r, pw[q] = rw[r * 16 + 8 + k];
      ek[k] = e, qk[k] = q;
    }
    for (int i = 1; i < TOPK; i++)  // the row's pairs in expert order
      for (int j = i; j > 0 && ek[j - 1] > ek[j]; j--) {
        int t = ek[j]; ek[j] = ek[j - 1], ek[j - 1] = t;
        t = qk[j]; qk[j] = qk[j - 1], qk[j - 1] = t;
      }
    memcpy(tp + r * TOPK, qk, sizeof qk);
  }
  MoeTile *tiles = (MoeTile *)TILES.ptr;
  int nt = 0, gg = gk[l].up != 0, per = gg ? 64 : 32;
  for (int i = 0; i < nn; i++) {
    int e = need[i], fresh;
    uint32_t at[STORE_PARTS];
    const Buf *slot = cache_get(experts_store, l, e, at, &fresh);
    if (fresh) gpu_use(slot);
    for (int c = 0; c < count[e]; c += per)
      tiles[nt++] = (MoeTile){{slot->gpu + at[0], slot->gpu + at[1], slot->gpu + at[2]}, (uint32_t)(start[e] + c),
                              (uint32_t)(count[e] - c < per ? count[e] - c : per)};
  }
  // Native MXFP4 scales by 128 for exact FP16 storage of small activations.
  uint32_t np = m * TOPK;
  gpu_kern(gg ? K_gg_gather : K_gather_mxfp4);
  gpu_buf(0, &XQ, 0), gpu_buf(1, &PTOK, 0), gpu_buf(2, &XH, 0);
  gpu_go(DIM / 4 / 256, np, 1, 256, 1, 1);
  gpu_kern(gg ? gk[l].up_m : K_moe_up_m);
  gpu_buf(0, &TILES, 0), gpu_buf(1, &XH, 0), gpu_buf(2, &PW, 0), gpu_buf(3, &MH, 0), gpu_bytes(4, &np, 4);
  gpu_go(nt, INTER / 64, 1, 128, 1, 1);
  gpu_kern(gg ? K_act_quant_h : K_act_quant_mxfp4);
  gpu_buf(0, &MH, 0), gpu_buf(1, &MHH, 0);
  gpu_go(np * INTER / 256, 1, 1, 256, 1, 1);
  gpu_kern(gg ? gk[l].down_m : K_moe_down_m);
  gpu_buf(0, &TILES, 0), gpu_buf(1, &MHH, 0), gpu_buf(2, &YK, 0), gpu_bytes(3, &np, 4);
  gpu_go(nt, DIM / 128, 1, 128, 1, 1);
  gpu_kern(K_moe_combine);
  gpu_buf(0, &YK, 0), gpu_buf(1, &TP, 0), gpu_buf(2, &SHO, 0), gpu_buf(3, &MF, mf_off(b));
  gpu_buf(4, &H1, 0), gpu_buf(5, &HS, hs_off(b));
  gpu_go(DIM / 256, m, 1, 256, 1, 1);
}

// ---- the draft's inputs ----

// The attention input of layers TARGET to TARGET + 2 (the mean of the
// streams), from max(batch start, end - keep) through the whole batch.
// Prefill retains its request tail. Decoding passes end=keep=1 to anchor
// storage at row zero while capturing every verification row.
static void capture(const Batch *b, int l, int end, int keep) {
  int first = end - keep, r0 = b->row > first ? b->row : first, r1 = b->row + b->m;
  if (!opt.draft || l < TARGET || r1 <= r0) return;
  uint32_t part = l - TARGET;
  gpu_kern(K_hc_mean);
  gpu_buf(0, &HS, (size_t)r0 * HC * DIM * STREAM_BYTES), gpu_buf(1, &MAINH, (size_t)(r0 - first) * 3 * DIM * 4);
  gpu_bytes(2, &part, 4);
  gpu_go(DIM / 256, r1 - r0, 1, 256, 1, 1);
}

// The draft stages' window keys come from MAINH: main_x =
// main_norm(main_proj(main_hidden)), then each stage's wkv, kv_norm, RoPE and
// FP8. draft_inputs: main_x of rows of MAINH into MXQ, in the open buffer.
static void draft_inputs(int rows) {
  quant(&MAINH, &MAINH, (size_t)rows * 3 * DIM);
  linear(&main_proj, rows, &MAINH, 0, 3 * DIM, &MXA, 0, DIM, 1);
  uint32_t n = DIM;
  gpu_kern(K_rmsnorm);
  gpu_buf(0, &MXA, 0), gpu_buf(1, &main_norm, 0), gpu_buf(2, &MXN, 0), gpu_buf(3, &MXQ, 0), gpu_bytes(4, &n, 4);
  gpu_go(rows, 1, 1, 1024, 1, 1);
}

// Rows r0 .. r0 + rows - 1 of KV into ring at positions p0 ...
static void draft_window(const Buf *ring, int r0, int rows, int p0, int d) {
  Win wp = {(uint32_t)p0, 0, WINDOW};
  gpu_kern(K_kv_window);
  gpu_buf(0, &KV, (size_t)r0 * HEAD_DIM * 4), gpu_buf(1, &D[d].kv_norm, 0), gpu_buf(2, &cs_base, 0), gpu_buf(3, ring, 0);
  gpu_bytes(4, &wp, sizeof wp);
  gpu_go(rows, 1, 1, 512, 1, 1);
}

// Each stage's window keys of positions p0 .. p0 + rows - 1 of the sequence
// in use, from MAINH's first rows.
static void draft_keys(int rows, int p0) {
  work_capacity(rows);
  gpu_begin();
  draft_inputs(rows);
  for (int d = 0; d < DRAFTS; d++) {
    linear(&D[d].wkv, rows, &MXQ, 0, DIM, &KV, 0, HEAD_DIM, 1);
    draft_window(&D[d].ring, 0, rows, p0, d);
  }
  gpu_commit();
  gpu_sync();
}

// The draft keys after a step of m sequences' rows (deepseek_forward_rows): row
// r's into st[r]'s rings at pos[r], each as draft_keys(1, pos[r]) writes them
// when the sequence decodes alone (the rows' products keep each row's order).
static void draft_rows(State *const *st, const int *pos, int m) {
  work_capacity(m);
  gpu_begin();
  draft_inputs(m);
  for (int d = 0; d < DRAFTS; d++) {
    linear(&D[d].wkv, m, &MXQ, 0, DIM, &KV, 0, HEAD_DIM, 1);
    for (int r = 0; r < m; r++) draft_window(&st[r]->dring[d], r, 1, pos[r], d);
  }
  gpu_commit();
  gpu_sync();
}

// ---- decoding: one token ----

// The tokens' embedding rows into EROW, then (embed_encode) into the streams;
// a position of an image span (a negative token, at position pos + i) takes
// its row from the image (image_row).
static const uint16_t *image_row(int pos);
static void embed_read(const int *tokens, int m, int row, int pos) {
  int64_t *t = memory_host_required(m * 8, 0);
  for (int i = 0; i < m; i++) t[i] = tokens[i] < 0 ? IMAGE_TOKEN : tokens[i];
  io_rows(embed_t.fd, embed_t.off, DIM * 2, t, m, EROW.ptr);
  memory_host_free(t);
  for (int i = 0; i < m; i++) {  // layer 0 collapses the identical streams with (1, 0, 0, 0)
    if (tokens[i] < 0) memcpy(EROW.ptr + (size_t)i * DIM * 2, image_row(pos + i), DIM * 2);
    float *pre = (float *)MF.ptr + (size_t)(row + i) * MIX;
    pre[0] = 1, pre[1] = pre[2] = pre[3] = 0;
  }
}

static void embed_encode(int m, int row) {
  gpu_kern(K_embed);
  gpu_buf(0, &EROW, 0), gpu_buf(1, &HS, (size_t)row * HC * DIM * STREAM_BYTES);
  gpu_go(DIM / 256, m, 1, 256, 1, 1);
}

static void embed_rows(const int *tokens, int m, int row, int pos) {
  embed_read(tokens, m, row, pos);
  gpu_begin();
  embed_encode(m, row);
  gpu_commit();
  gpu_sync();
}

// A prompt's or a step's Engram rows, read in the background, layer 1's
// first; each group is left when its table's rows are in.
typedef struct {
  const int64_t (*rows)[2][ENGRAM_ROWS];
  int n;
  dispatch_group_t read[2];
  MemorySlot *owner;
  uint64_t reserved;
} Rows;

static uint64_t row_scratch(int rows) {
  return (uint64_t)rows * ENGRAM_ROWS * 8 + io_rows_workspace(rows * ENGRAM_ROWS);
}

static void rows_begin(Rows *rows) {
  rows->owner = memory_current();
  rows->reserved = row_scratch(rows->n);
  if (rows->owner && !memory_claim(rows->owner, rows->reserved, MEMORY_REQUIRED))
    die("model-memory budget cannot fit Engram read workspace");
}

static void rows_end(Rows *rows) {
  if (rows->owner) memory_release(rows->owner, rows->reserved);
}

static void read_rows(void *ctx) {
  Rows *r = ctx;
  for (int e = 0; e < 2; e++) engram_read(e, r->rows, r->n), dispatch_group_leave(r->read[e]);
}

// Everything of layer l before its routed experts: the router signals the
// returned value, then the shared expert runs while the CPU routes.
static uint64_t layer_front(const Batch *b, int l) {
  gpu_begin();
  if (l == 0) embed_encode(b->m, b->row);  // the rows embed_read read
  if (l == 1 || l == 14) {
    engram(b, l);
    if (opt.dump_fd >= 0) {  // the streams after the Engram lookup
      gpu_commit();
      gpu_sync();
      if (write(opt.dump_fd, stream_floats(&HS, 0, HC * DIM), HC * DIM * 4) < 0) die("dump failed");
      gpu_begin();
    }
  }
  capture(b, l, 1, 1);
  attention(b, l);
  ffn_front(b, l);
  uint64_t v = gpu_signal();
  shared_expert(b->m, &L[l]);
  gpu_commit();
  return v;
}

// m rows at positions pos..: one token, or a token and its drafts.
static void step_rows(const int *tokens, int m, int pos) {
  for (int i = 0; i < m; i++)
    if (tokens[i] < 0) die("an image's token outside a prompt's batch (position %d)", pos + i);
  route_steps++;
  if (!nrows && pos + m > opt.ctx) die("context full (%d tokens)", opt.ctx);
  if (!nrows) reserve(pos + m, pos);
  request_capacity(m);
  double ts = now();
  int64_t rows[FEW][2][ENGRAM_ROWS];
  for (int i = 0; i < m; i++) {  // rows of several sequences hash with their own histories
    if (nrows) memcpy(history, row_state[i]->history, sizeof history);
    engram_hash(tokens[i], nrows ? row_pos[i] : pos + i, rows[i]);
    if (nrows) memcpy(row_state[i]->history, history, sizeof history);
  }
  // The Engram rows are read while the first layers run: the command buffers
  // of layers 1 and 14 run after the experts' of the layer before, which are
  // committed only when the table's rows are in.
  Rows rr = {.rows = rows, .n = m, .read = {dispatch_group_create(), dispatch_group_create()}};
  rows_begin(&rr);
  for (int e = 0; e < 2; e++) dispatch_group_enter(rr.read[e]);
  dispatch_async_f(dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), &rr, read_rows);
  embed_read(tokens, m, 0, pos);
  phase_s[0] += now() - ts;
  Batch b = {m, pos, 0};
  int ahead = opt.dump_fd < 0 && !gpu_profiling();
  // Pre-encode the next front while the current front runs. The ordinary
  // path reserves two earlier queue positions for experts. Completion
  // groups defer front submission until all expert groups are queued.
  uint64_t sig = layer_front(&b, 0), next = 0;
  ts = now();
  cache_adapt(experts_store);  // no slot is in use between tokens; layer 0's front uses none
  phase_s[5] += now() - ts;
  for (int l = 0; l < LAYERS; l++) {
    void *experts = gpu_open(), *rest = ahead ? gpu_open() : 0;  // rest: see routed_rows
    if (ahead && l + 1 < LAYERS) next = layer_front(&b, l + 1);
    double t0 = now();
    if (full_ahead(m)) {  // the residual's predictions, signaled before the attention
      int keep[FEW * KEEP], first[FEW * AHEAD], nk = 0, nf = 0;
      const int *pc = (const int *)PC.ptr;
      gpu_wait_signal(sig - 1);
      for (int r = 0; r < m; r++) {
        for (int k = 0; k < KEEP; k++) keep[nk++] = pc[r * PREDICT + k];
        for (int k = 0; k < AHEAD; k++) first[nf++] = pc[r * PREDICT + k];
      }
      read_residual_ahead(l, keep, nk, first, nf, INFLIGHT * m);
    }
    gpu_wait_signal(sig);
    phase_s[1] += now() - t0;
    t0 = now();
    if (opt.routes_fd >= 0 && (write(opt.routes_fd, RT.ptr, TOPK * 4) < 0 ||
                               write(opt.routes_fd, PR.ptr, 2 * PREDICT * 4) < 0 ||
                               write(opt.routes_fd, PT.ptr, PREDICT * 4) < 0 ||
                               write(opt.routes_fd, GL.ptr, EXPERTS * 4) < 0))
      die("routes: write failed");
    gpu_encode(experts);
    int split = routed_rows(&b, l, AHEAD, INFLIGHT, opt.deepseek.substitute > 0, rest);
    if (l == LAYERS - 1) {
      hc_pre_norm(m, &HS, 0, &MF, 0, &final_norm);
      head_out(m, &logits);
    }
    if (l == 0 || l == 13) {
      double tw = now();
      dispatch_group_wait(rr.read[l == 13], DISPATCH_TIME_FOREVER);
      phase_s[0] += now() - tw;
    }
    gpu_commit();
    if (rest && !split) gpu_submit(rest);  // empty
    phase_s[4] += now() - t0;
    // Labels and source records are CPU-owned. Both expert buffers are now
    // committed, so independent, already-used heads can learn behind GPU work.
    if (opt.deepseek.online_predictor) online_predictor_update(l);
    if (!ahead) {
      gpu_sync();  // the streams after attention and after the FFN, then the routing
      if (opt.dump_fd >= 0 &&
          (write(opt.dump_fd, stream_floats(&H1, 0, HC * DIM), HC * DIM * 4) < 0 ||
           write(opt.dump_fd, stream_floats(&HS, 0, HC * DIM), HC * DIM * 4) < 0 ||
           write(opt.dump_fd, RT.ptr, 48) < 0))
        die("dump failed");
      if (l + 1 < LAYERS) next = layer_front(&b, l + 1);
    }
    sig = next;
  }
  double ta = now();
  gpu_sync();
  phase_s[1] += now() - ta;
  for (int e = 0; e < 2; e++) dispatch_release(rr.read[e]);
  rows_end(&rr);
}

// ---- prefill: many tokens, layer by layer ----

// The prompt's Engram rows are read in the background (see Rows) while the
// GPU works on the layers before; the pass waits for each.

// The statistics of decoded rows (step) and prompt rows (prefill) since
// start, kept apart for the server's view.
CacheStats decode_stats, prefill_stats;
uint64_t decode_rows, prefill_rows;
double decode_gpu_s, decode_s;
int prefill_layer, prefill_n;

static void stats_add(CacheStats *sum, const CacheStats *before) {
  const CacheStats *c = expert_stats();
  sum->hits += c->hits - before->hits, sum->misses += c->misses - before->misses;
  sum->prefetch_hits += c->prefetch_hits - before->prefetch_hits, sum->prefetches += c->prefetches - before->prefetches;
  sum->bytes += c->bytes - before->bytes, sum->evictions += c->evictions - before->evictions;
  sum->late += c->late - before->late, sum->cancelled += c->cancelled - before->cancelled;
  sum->substituted += c->substituted - before->substituted, sum->remaps += c->remaps - before->remaps;
  sum->given_back += c->given_back - before->given_back, sum->read_s += c->read_s - before->read_s;
  sum->row_own += c->row_own - before->row_own, sum->row_ahead += c->row_ahead - before->row_ahead;
  sum->row_late += c->row_late - before->row_late, sum->row_sub += c->row_sub - before->row_sub;
  sum->row_ssd += c->row_ssd - before->row_ssd;
  sum->row_unique += c->row_unique - before->row_unique;
}

static void step(const int *tokens, int m, int pos) {
  if (opt.deepseek.online_predictor) {
    online_predictor_begin(m, nrows || pos > 0); // the initial single-token prompt stays cold
  }
  cache_metrics_begin(experts_store, PF_DECODE, m);
  CacheStats c0 = *expert_stats();
  double g0 = gpu_busy, t0 = now();
  step_rows(tokens, m, pos);
  if (opt.deepseek.online_predictor) online_predictor_end();
  cache_metrics_end(experts_store);
  stats_add(&decode_stats, &c0);
  decode_rows += m, decode_gpu_s += gpu_busy - g0, decode_s += now() - t0;
}

uint64_t deepseek_dense_bytes(void) { return weights_resident_bytes(core_weights) + weights_resident_bytes(vision_weights); }

uint64_t deepseek_workspace_bytes(void) {
  uint64_t bytes = HS.size + MF.size + SEL.size + CAND.size + IMG.size;
  for (int e = 0; e < 2; e++) bytes += EC[e].size + ES[e].size;
  for (size_t i = 0; i < sizeof work_buffers / sizeof *work_buffers; i++) bytes += work_buffers[i]->size;
  bytes += scheduler_workspace_bytes;
  return bytes;
}

static int prefix_batch(const Batch *b, int l) {
  return prefix_capture.at > b->pos && prefix_capture.at <= b->pos + b->m &&
         (!prefix_capture.encoder_only || l <= 20);
}

// The capture of the rolling rows at `at` during a prefill from pos, into dst
// (saved_bytes() in save_rows' layout).
static void prefix_prepare(int at, int pos, char *dst) {
  prefix_capture = (PrefixCapture){.at = at};
  if (at < 0) return;
  // Under CED compaction the decoder layers start at ced_first: they hold the
  // whole ring before `at` only if they started a ring or more before it.
  prefix_capture.encoder_only = ced_compact && ced_first + RING > (uint32_t)at;
  prefix_capture.tap_first = at > WINDOW ? at - WINDOW : 0;
  if (prefix_capture.tap_first < pos) prefix_capture.tap_first = pos;
  char *p = dst;
  memset(p, 0, saved_bytes());
  for (int l = 0; l < LAYERS; l++) {
    prefix_capture.ring[l] = p, p += L[l].ring.size;
    if (L[l].kv_source) prefix_capture.cstate[l] = p, p += L[l].cstate.size;
  }
  for (int d = 0; opt.draft && d < DRAFTS; d++) {
    prefix_capture.dring[d] = p, p += D[d].ring.size;
    if (!prefix_capture.encoder_only) {
      if (!prefix_drings[d].obj) prefix_drings[d] = zeros(D[d].ring.size);
      memcpy(prefix_drings[d].ptr, D[d].ring.ptr, D[d].ring.size);
    }
  }
  prefix_capture.history = p;
  if (opt.draft && !prefix_capture.encoder_only && !prefix_taps.obj)
    prefix_taps = zeros((size_t)WINDOW * 3 * DIM * 4);
}

// The previous chunk's attention is already synchronized. Its expert work
// may still run, but that work does not write these attention buffers.
static void prefix_before(const Batch *b, int l) {
  if (!prefix_batch(b, l)) return;
  memcpy(prefix_capture.ring[l], L[l].ring.ptr, L[l].ring.size);
  if (L[l].kv_source) memcpy(prefix_capture.cstate[l], L[l].cstate.ptr, L[l].cstate.size);
}

static void prefix_inputs(const Batch *b, int l) {
  if (prefix_capture.at < 0 || prefix_capture.encoder_only || !opt.draft || l < TARGET) return;
  int first = b->pos > prefix_capture.tap_first ? b->pos : prefix_capture.tap_first;
  int end = b->pos + b->m < prefix_capture.at ? b->pos + b->m : prefix_capture.at;
  if (first >= end) return;
  uint32_t part = l - TARGET;
  gpu_kern(K_hc_mean);
  gpu_buf(0, &HS, (size_t)(b->row + first - b->pos) * HC * DIM * STREAM_BYTES);
  gpu_buf(1, &prefix_taps, (size_t)(first - prefix_capture.tap_first) * 3 * DIM * 4);
  gpu_bytes(2, &part, sizeof part);
  gpu_go(DIM / 256, end - first, 1, 256, 1, 1);
}

static void prefix_after(const Batch *b, int l) {
  if (!prefix_batch(b, l)) return;
  int first = prefix_capture.at - RING;
  if (first < b->pos) first = b->pos;
  int base = b->pos >= WINDOW - 1 ? b->pos - (WINDOW - 1) : 0;
  for (int p = first; p < prefix_capture.at; p++) {
    const char *src = b->m == 1 ? (const char *)L[l].ring.ptr + (size_t)(p % RING) * HEAD_DIM * 2 :
                                 (const char *)WIN.ptr + (size_t)(p - base) * HEAD_DIM * 2;
    memcpy(prefix_capture.ring[l] + (size_t)(p % RING) * HEAD_DIM * 2, src, HEAD_DIM * 2);
  }
  if (L[l].kv_source && L[l].ratio == 2) {
    first = prefix_capture.at - PENDING;
    if (first < b->pos) first = b->pos;
    size_t half = (size_t)PENDING * HEAD_DIM * 4;
    for (int p = first; p < prefix_capture.at; p++) {
      size_t dst = (size_t)(p % PENDING) * HEAD_DIM * 4;
      size_t src = (size_t)(2 + p - b->pos) * HEAD_DIM * 4;
      memcpy(prefix_capture.cstate[l] + dst, CK.ptr + src, HEAD_DIM * 4);
      memcpy(prefix_capture.cstate[l] + half + dst, CS.ptr + src, HEAD_DIM * 4);
    }
  }
  prefix_capture.captured[l] = 1;
}

static void prefix_complete(void) {
  if (prefix_capture.at < 0) return;
  for (int l = 0; l < LAYERS; l++)
    if ((!prefix_capture.encoder_only || l <= 20) && !prefix_capture.captured[l])
      die("prefix snapshot missed layer %d", l);
  if (opt.draft && !prefix_capture.encoder_only) {
    Buf original_taps = MAINH, original_rings[DRAFTS];
    MAINH = prefix_taps;
    for (int d = 0; d < DRAFTS; d++) original_rings[d] = D[d].ring, D[d].ring = prefix_drings[d];
    prompt_pass = 1;
    draft_keys(prefix_capture.at - prefix_capture.tap_first, prefix_capture.tap_first);
    prompt_pass = 0;
    for (int d = 0; d < DRAFTS; d++) {
      memcpy(prefix_capture.dring[d], D[d].ring.ptr, D[d].ring.size);
      D[d].ring = original_rings[d];
    }
    MAINH = original_taps;
  }
  forward_info.prefix_tokens = prefix_capture.at;
  forward_info.prefix_encoder_only = prefix_capture.encoder_only;
  prefix_capture.at = -1;
}

// An exact replay at k (deepseek_prefill) rebuilds the rolling rows at k from the
// global rows a state copied: layer l must compute again, from replay_first[l],
// the window keys that its own ring at k and the layer above it need. Layer
// l's keys at a position come from layer l - 1's output there, and that
// output is right only a window (WINDOW - 1 positions) after layer l - 1's
// first row, so each layer below starts a window earlier. The rows a layer
// computes before its outputs are right only give their keys; nothing reads
// their outputs. A ratio-2 compressor starts on a pair.
static int replay_first[LAYERS], replaying;

static void replay_starts(int k) {
  int s = k - RING;
  for (int l = LAYERS - 1; l >= 0; l--) {
    if (s > k - RING) s = k - RING;
    if (s < 0) s = 0;
    if (L[l].kv_source && L[l].ratio == 2) s &= ~1;
    replay_first[l] = s;
    s -= WINDOW - 1;
  }
}

int deepseek_rewind(void) { return replay_margin(); }

int deepseek_replay_rows(int k) {
  if (k <= 0) return 0;
  if (opt.deepseek.ced) return k < RING ? k : RING;  // a bounded replay: the last ring's positions, in every layer
  replay_starts(k);  // deepseek_prefill sets them again for its own replay
  long rows = 0;
  for (int l = 0; l < LAYERS; l++) rows += k - replay_first[l];
  return (int)((rows + LAYERS - 1) / LAYERS);
}

int deepseek_replay_start(int k) {
  if (k <= 0) return 0;
  if (opt.deepseek.ced) return k > RING ? k - RING : 0;
  replay_starts(k);
  return replay_first[0];
}

static int decoder_begin(int n, int pos) {
  if (!ced_compact || ced_first <= (uint32_t)pos) return 0;
  return ced_first - pos < (uint32_t)n ? (int)ced_first - pos : n;
}

static PrefillPrune decoder_plan(int pos, int count, int layer) {
  int remaining = LAYERS - 21;
  return prefill_prune_plan(pos, count, prefill_prune_end, prefix_capture.at,
                           remaining, WINDOW, replay_margin(), PRUNE_ROWS);
}

// Runs the prompt's rows through every layer. Quitting stops it before its
// next batch of rows: it returns 0, and the state's rows from pos0 on are
// then incomplete.
static int prefill(const int *tokens, int n, int pos0) {
  if (pos0 + n > opt.ctx) die("context full (%d tokens)", opt.ctx);
  ced_active = ced_first != 0;
  prompt_pass = 1;
  int retained = n - decoder_begin(n, pos0);
  PrefillPrune plan = {.kept = n, .suffix_rows = n};
  if (prefill_prune_active) {
    plan = decoder_plan(pos0, n, 21);
    retained = plan.suffix_rows;
  }
  forward_info.chunks++, forward_info.decoder_tokens += prefill_prune_active ? plan.kept : retained;
  cache_metrics_begin(experts_store, PF_PREFILL, n);
  CacheStats c0 = *expert_stats();
  prefill_n = n, prefill_layer = 0;
  reserve(pos0 + n, pos0);
  request_capacity(n);
  for (int i = 0; i < n; i++) ((uint8_t *)IMG.ptr)[i] = tokens[i] < 0;
  int64_t (*rows)[2][ENGRAM_ROWS] = memory_host_required((size_t)n * sizeof *rows, 0);
  for (int i = 0; i < n; i++) {
    engram_hash(tokens[i], pos0 + i, rows[i]);
    if (prefix_capture.at == pos0 + i + 1) memcpy(prefix_capture.history, history, sizeof history);
  }
  Rows rr = {.rows = rows, .n = n, .read = {dispatch_group_create(), dispatch_group_create()}};
  rows_begin(&rr);
  for (int e = 0; e < 2; e++) dispatch_group_enter(rr.read[e]);
  dispatch_async_f(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), &rr, read_rows);
  for (int s = 0; s < n; s += MB) embed_rows(tokens + s, n - s < MB ? n - s : MB, s, pos0 + s);
  if (n >= 256 && opt.deepseek.substitute <= 0) {  // a large batch needs nearly every expert: read layer 0's, after the embedding's rows
    int all[EXPERTS];
    for (int e = 0; e < EXPERTS; e++) all[e] = e;
    cache_prefetch_from(experts_store, 0, all, EXPERTS, 2 * EXPERTS, PF_BULK);
  }
  int stopped = 0;
  for (int l = 0; l < LAYERS && !stopped; l++) {
    if (prefill_prune_active && l > 20) plan = decoder_plan(pos0, n, l);
    cache_pass(experts_store, l);
    if (l == 1 || l == 14) dispatch_group_wait(rr.read[l == 14], DISPATCH_TIME_FOREVER);
    int begin = 0, next_rows = n;
    if (ced_compact) {
      int first = decoder_begin(n, pos0);
      if (first && all_logits) die("compact prefill cannot return logits for discarded rows");
      // Layer 20 produces full-context global KV; later layers only consume it.
      if (l > 20) begin = first;
      if (l + 1 > 20) next_rows = n - first;
    } else if (prefill_prune_active && l + 1 > 20) {
      next_rows = decoder_plan(pos0, n, l + 1).kept;
    }
    if (replaying && replay_first[l] - pos0 > begin) begin = replay_first[l] - pos0 < n ? replay_first[l] - pos0 : n;
    int first_batch = 1, batch_rows = 0;
    for (int s = begin; s < n; s += batch_rows) {
      if (shutdown_requested()) {
        stopped = 1;
        break;
      }
      int end = n;
      if (prefill_prune_active && l > 20) {
        if (s < plan.prefix_begin) s = plan.prefix_begin;
        if (s >= plan.prefix_end && s < plan.tail_begin) s = plan.tail_begin;
        if (s >= n) break;
        if (s < plan.prefix_end) end = plan.prefix_end;  // a batch stays within a kept region
      }
      batch_rows = end - s < MB ? end - s : MB;
      Batch b = {batch_rows, pos0 + s, s};
      if (l > 20) forward_info.decoder_row_layers += b.m;
      prefix_before(&b, l);
      gpu_begin();
      if (l == 1 || l == 14) engram(&b, l);
      capture(&b, l, n, retained < WINDOW ? retained : WINDOW);
      prefix_inputs(&b, l);
      attention(&b, l);
      ffn_front(&b, l);
      shared_expert(b.m, &L[l]);
      gpu_commit();
      if (first_batch) cache_adapt_pass(experts_store, l);  // layer l - 1's experts may still be in use
      first_batch = 0;
      gpu_sync();
      prefix_after(&b, l);
#ifdef MYDS4_CED_CHECK
      ced_check_routes(l, pos0 + s, b.m, (const int *)RT.ptr);
#endif

      gpu_begin();
      int ahead = opt.deepseek.substitute > 0 ? SUB_KEEP : next_rows > BULK_ROWS ? EXPERTS : 8;
      if (prefill_prune_active && !next_rows) ahead = 0;
      routed_batch(&b, l, ahead, 2 * EXPERTS, opt.deepseek.substitute > 0);
      gpu_commit();
    }
    if (stopped) break;
    if (!opt.quiet) fprintf(stderr, "\rprefill: layer %d/%d", l + 1, LAYERS);
    prefill_layer = l + 1;
  }
  cache_pass(experts_store, -1);
  prompt_pass = 0;  // the head: one row, as decoding runs it
  if (!opt.quiet) fprintf(stderr, "\r                         \r");
  if (retained && !stopped) {
    gpu_begin();
    hc_pre_norm(1, &HS, (size_t)(n - 1) * HC * DIM * STREAM_BYTES, &MF, (size_t)(n - 1) * MIX * 4, &final_norm);
    head_out(1, &logits);
    gpu_commit();
  }
  gpu_sync();
  if (all_logits && !stopped) {  // every row's logits, for checks
    for (int r = 0; r < n; r++) {
      gpu_begin();
      hc_pre_norm(1, &HS, (size_t)r * HC * DIM * STREAM_BYTES, &MF, (size_t)r * MIX * 4, &final_norm);
      head_out(1, &logits);
      gpu_commit();
      gpu_sync();
      memcpy(all_logits + (size_t)r * VOCAB, logits.ptr, VOCAB * 4);
    }
    memcpy(logits.ptr, all_logits + (size_t)(n - 1) * VOCAB, VOCAB * 4);
  }
  // A stopped prefill may end before layers 1 and 14 waited for the rows:
  // the reader uses rr and rows until it leaves both groups.
  for (int e = 0; e < 2; e++) dispatch_group_wait(rr.read[e], DISPATCH_TIME_FOREVER), dispatch_release(rr.read[e]);
  rows_end(&rr);
  memory_host_free(rows);
  memset(IMG.ptr, 0, (size_t)n);  // decoding's rows are text
  if (opt.draft && retained && !stopped) {
    int keep = retained < WINDOW ? retained : WINDOW;
    prompt_pass = 1;
    draft_keys(keep, pos0 + n - keep);
    prompt_pass = 0;
  }
  cache_adapt(experts_store);
  stats_add(&prefill_stats, &c0);
  prefill_rows += n, prefill_layer = prefill_n = 0;
  cache_metrics_end(experts_store);
  ced_active = 0;
  return !stopped;
}


// A prefill pass keeps every row's streams, selections and Engram rows
// (about 56 KB plus candidate masks): longer prompts use several passes.
#define MYDS4_PREFILL_CHUNK 16384
#define CHUNK MYDS4_PREFILL_CHUNK
static_assert(CHUNK >= MB && CHUNK <= 32768 && CHUNK % MB == 0, "bounded prefill chunk with unchanged matrix batches");
static int prefill_chunk = CHUNK;

static uint64_t resizable_bytes(void) {
  uint64_t bytes = 0;
  for (size_t i = 0; i < sizeof work_buffers / sizeof *work_buffers; i++) bytes += work_buffers[i]->allocated;
  Buf *request[] = {&HS, &MF, &SEL, &CAND, &IMG, &EC[0], &EC[1], &ES[0], &ES[1]};
  for (size_t i = 0; i < sizeof request / sizeof *request; i++) bytes += request[i]->allocated;
  return bytes;
}

static int prepare_rows(int rows, int prompt) {
  static int previous_rows;
  static uint64_t previous_payload;
  MemorySlot *work = memory_current();
  if (!prompt && work && rows == previous_rows && work->payload == previous_payload) return rows;
  if (!work || !core_weights) return rows;
  int small = replay_margin() + 1 > BATCH_ROWS ? replay_margin() + 1 : BATCH_ROWS;
  uint64_t fixed = work->payload - resizable_bytes();
  int candidate = prompt && rows > prefill_chunk ? prefill_chunk : rows;
  for (;;) {
    int request_rows = candidate > small ? candidate : small;
    int layer_rows = request_rows < MB ? request_rows : MB;
    uint64_t workspace = fixed + request_plan(request_rows, 0) + work_plan(layer_rows, 0);
    uint64_t host = row_scratch(candidate) + (uint64_t)candidate * 2 * ENGRAM_ROWS * 8 + (uint64_t)layer_rows * 64;
    cache_working_rows(experts_store, candidate);
    uint64_t required = weights_required(core_weights) + workspace + host + cache_required(experts_store, opt.draft ? CACHE_LAYERS : LAYERS);
    MemoryAdmission mode = candidate > 1 && prompt ? MEMORY_WORKSPACE : MEMORY_REQUIRED;
    if (memory_minimum_mode(work->account, required, mode)) {
      previous_rows = prompt ? 0 : rows;
      previous_payload = work->payload;
      return candidate;
    }
    if (!prompt || candidate == 1) return 0;
    candidate = (candidate + 1) / 2;
  }
}

static const float *forward(const int *tokens, int n, int pos, int prefix_pos, char *prefix_dst, int prompt) {
  if (n < 1 || pos < 0 || n > opt.ctx - pos) die("invalid forward range");
  if (prefix_pos >= 0 && (prefix_pos <= pos || prefix_pos >= pos + n)) die("invalid prefix snapshot position");
  ced_first = 0, ced_compact = 0, ced_active = 0;
  if (opt.deepseek.ced && n > opt.deepseek.ced + replay_margin())
    ced_first = pos + n - (opt.deepseek.ced + replay_margin()), ced_compact = 1;
#ifdef MYDS4_CED_CHECK
  if (!opt.deepseek.ced) ced_first = ced_check_first, ced_compact = ced_check_compact;
#endif
  prefill_prune_active = opt.deepseek.prefill_prune && !ced_first && !opt.deepseek.ced && !all_logits &&
                         opt.deepseek.substitute <= 0 && opt.dump_fd < 0 && opt.routes_fd < 0;
  prefill_prune_end = pos + n;
  forward_info = (DeepSeekForwardInfo){.tokens = n, .ced_tokens = opt.deepseek.ced,
                                   .ced_applied = ced_compact && ced_first > (uint32_t)pos};
  prefix_prepare(prefix_pos, pos, prefix_dst);
  for (int done = 0, k; done < n; done += k) {
    k = prepare_rows(n - done < prefill_chunk ? n - done : prefill_chunk, prompt || n > 1);
    if (!k) return NULL;
    if ((k > 1 || prompt) && opt.dump_fd < 0) {
      float *every = all_logits;
      if (every) all_logits = every + (size_t)done * VOCAB;
      int completed = prefill(tokens + done, k, pos + done);
      all_logits = every;
      if (!completed) {  // quitting stopped it: no logits, and no snapshot of its rows
        prefix_capture.at = -1, prefill_prune_active = 0;
        return NULL;
      }
    } else {
      for (int i = done; i < done + k; i++) {
        step(tokens + i, 1, pos + i);
        if (opt.draft) draft_keys(1, pos + i);
        forward_info.chunks++, forward_info.decoder_tokens++;
        forward_info.decoder_row_layers += LAYERS - 21;
      }
    }
  }
  prefix_complete();
  forward_info.prune_applied = prefill_prune_active && forward_info.decoder_tokens < forward_info.tokens;
  prefill_prune_active = 0;
  if (opt.dump_fd >= 0 && write(opt.dump_fd, logits.ptr, VOCAB * 4) < 0) die("dump failed");
  return (const float *)logits.ptr;
}

const float *deepseek_forward(const int *tokens, int n, int pos) { return forward(tokens, n, pos, -1, 0, 0); }

const float *deepseek_prefill(const int *tokens, int n, int from, int replay, int capture, Snapshot **captured) {
  if (from < 0 || from >= n || n > opt.ctx) die("invalid prefill of positions %d to %d", from, n);
  if (!from) replay = 0;
  int start = from;
  if (replay && opt.deepseek.ced) {  // the bounded replay: the ring's positions before from, windows cut off there
    start = deepseek_replay_start(from);
    bounded = 1, bounded_first = (uint32_t)start, bounded_keep = (uint32_t)from;
  } else if (replay) {
    replay_starts(from), start = replay_first[0], replaying = 1;
  }
  if (capture >= 0 && (capture <= start || capture >= n || !captured))
    die("invalid capture at %d in a prefill of positions %d to %d", capture, start, n);
  // The capture also keeps, with --prefill-prune, the decoder rows before it
  // that its window needs.
  Snapshot *x = 0;
  if (capture >= 0) {
    x = xmalloc(sizeof *x);
    *x = (Snapshot){.pos = capture, .extent = LAYERS - 1, .rows = memory_host_required(saved_bytes(), 1)};
  }
  deepseek_seek(tokens, start);
  const float *lg = forward(tokens + start, n - start, start, capture, x ? x->rows : 0, 1);
  replaying = 0, bounded = 0;
  if (x && !lg) deepseek_snapshot_free(x), x = 0;  // a prefill that did not finish has no capture
  if (x) {
    if (forward_info.prefix_encoder_only) x->extent = 20;
    *captured = x;
  }
  return lg;
}

DeepSeekForwardInfo deepseek_forward_info(void) { return forward_info; }

// ---- the vision encoder (inference/vision.py ViT and Aligner) ----

typedef struct { Buf norm1, norm2, b_qkv, b_o; Lin qkv, o, w1, w2; } VitBlock;
static struct {
  int loaded;
  Lin patch, a1, a2;  // patch: proj with its 588 inputs zero-padded to VIT_K
  Buf patch_b, norm, a1_b, a2_b;
  VitBlock b[32];
} vit;
static const Media *imgs;  // the prompt's images (deepseek_images)
static int nimgs;

// The vision encoder's weights (0.97 GB) load at the first image.
static void vit_prepare(void) {
  if (vit.loaded) return;
  vision_weights = weights_new(memory_current()->account, "DeepSeek vision weights");
  weight_group = vision_weights;
  char s[96];
  Tensor t = st_get(checkpoint, "vision.patch_embed.proj.weight");
  vit.patch = (Lin){.n = 1024, .k = VIT_K};
  weights_padded(vision_weights, &vit.patch.w, t, VIT_K);
  vec(&vit.patch_b, st_get(checkpoint, "vision.patch_embed.proj.bias"));
  for (int l = 0; l < 32; l++) {
    VitBlock *b = &vit.b[l];
    snprintf(s, sizeof s, "vision.blocks.%d.attn.wqkv", l), lin(&b->qkv, s);
    snprintf(s, sizeof s, "vision.blocks.%d.attn.wo", l), lin(&b->o, s);
    snprintf(s, sizeof s, "vision.blocks.%d.mlp.w1", l), lin(&b->w1, s);
    snprintf(s, sizeof s, "vision.blocks.%d.mlp.w2", l), lin(&b->w2, s);
    vec(&b->b_qkv, st_get(checkpoint, "vision.blocks.%d.attn.wqkv.bias", l)), vec(&b->b_o, st_get(checkpoint, "vision.blocks.%d.attn.wo.bias", l));
    vec(&b->norm1, st_get(checkpoint, "vision.blocks.%d.norm1.weight", l)), vec(&b->norm2, st_get(checkpoint, "vision.blocks.%d.norm2.weight", l));
  }
  vec(&vit.norm, st_get(checkpoint, "vision.norm.weight"));
  lin(&vit.a1, "aligner.w1"), vec(&vit.a1_b, st_get(checkpoint, "aligner.w1.bias"));
  lin(&vit.a2, "aligner.w2"), vec(&vit.a2_b, st_get(checkpoint, "aligner.w2.bias"));
  vit.loaded = 1;
  weight_group = core_weights;
}

// y = x W^T for m rows, then bf(y + bias), or rounded to BF16 without a bias.
static void vit_linear(const Lin *w, int m, const Buf *x, uint32_t lda, const Buf *y, const Buf *bias) {
  uint32_t n = (uint32_t)w->n;
  linear(w, m, x, 0, lda, y, 0, n, !bias);
  if (!bias) return;
  gpu_kern(K_vit_bias);
  gpu_buf(0, y, 0), gpu_buf(1, bias, 0), gpu_bytes(2, &n, 4);
  gpu_go(n / 256, m, 1, 256, 1, 1);
}

static void vit_norm(const Buf *x, const Buf *w, const Buf *y, int m) {
  gpu_kern(K_vit_norm);
  gpu_buf(0, x, 0), gpu_buf(1, w, 0), gpu_buf(2, y, 0);
  gpu_go(m, 1, 1, 256, 1, 1);
}

static void vit_add(const Buf *x, const Buf *y, int m) {
  gpu_kern(K_vit_add);
  gpu_buf(0, x, 0), gpu_buf(1, y, 0);
  gpu_go(m * 1024 / 256, 1, 1, 256, 1, 1);
}

static float bf_cpu(float f) {  // round to BF16, ties to even
  uint32_t u;
  memcpy(&u, &f, 4);
  u = (u + 0x7FFFu + ((u >> 16) & 1u)) & 0xFFFF0000u;
  memcpy(&f, &u, 4);
  return f;
}

// The Aligner's GELU (exact, erf), on the CPU in double precision, rounded
// to float as the reference computes it, then to BF16.
static void vit_gelu_row(void *ctx, size_t r) {
  float *g = (float *)ctx + r * DIM;
  for (int i = 0; i < DIM; i++) g[i] = bf_cpu((float)(0.5 * g[i] * (1 + erf(g[i] / sqrt(2.0)))));
}

int deepseek_image(Media *im) {
  if (im->rows) return 1;
  vit_prepare();
  uint32_t n = (uint32_t)(im->vh * im->vw), L = (uint32_t)(im->th * im->tw);
  uint32_t np = (n + 63) / 64 * 64, dims[2] = {n, np};  // vit_attn's rows: the patches, padded to its tiles
  MemorySlot *work = memory_current(), *weights = weights_slot(vision_weights);
  uint64_t old_layer = 0;
  for (size_t i = 0; i < sizeof work_buffers / sizeof *work_buffers; i++) old_layer += work_buffers[i]->allocated;
  size_t sizes[] = {(size_t)n * VIT_K * 4, (size_t)n * 64 * 4,
    (size_t)n * 1024 * 4, (size_t)n * 1024 * 4, (size_t)n * 1024 * 4,
    (size_t)n * 3072 * 4, (size_t)np * 3072 * 2,
    (size_t)n * 5632 * 4, (size_t)n * 2816 * 4,
    (size_t)L * 9216 * 4, (size_t)L * DIM * 4, (size_t)L * DIM * 4};
  uint64_t temporary = (size_t)L * DIM * 2;
  for (size_t i = 0; i < sizeof sizes / sizeof *sizes; i++) temporary += gpu_allocation_size(sizes[i]);
  MemorySlot *entries[] = {work, weights};
  uint64_t bytes[] = {work->payload - old_layer + work_plan(MB, 0) + temporary, weights_required(vision_weights)};
  if (!memory_admit(entries, bytes, 2, MEMORY_REQUIRED)) return 0;
  memory_pin(weights, 1);
  work_capacity(MB);
  weights_load(vision_weights);
  im->rows = memory_host_required((size_t)L * DIM * 2, 0);
  Buf px = gpu_alloc((size_t)n * VIT_K * 4), cs = gpu_alloc((size_t)n * 64 * 4);
  for (size_t i = 0; i < (size_t)n * VIT_K; i++) ((uint32_t *)px.ptr)[i] = (uint32_t)im->patch[i] << 16;
  image_rope(im, (float *)cs.ptr);
  Buf y = gpu_alloc((size_t)n * 1024 * 4), x = gpu_alloc((size_t)n * 1024 * 4), t = gpu_alloc((size_t)n * 1024 * 4);
  Buf qkv = gpu_alloc((size_t)n * 3072 * 4), hv = gpu_alloc((size_t)np * 3072 * 2);
  Buf h = gpu_alloc((size_t)n * 5632 * 4), a = gpu_alloc((size_t)n * 2816 * 4);
  Buf u = gpu_alloc((size_t)L * 9216 * 4), g = gpu_alloc((size_t)L * DIM * 4), f = gpu_alloc((size_t)L * DIM * 4);
  gpu_begin();
  vit_linear(&vit.patch, n, &px, VIT_K, &x, &vit.patch_b);
  gpu_commit();
  for (int l = 0; l < 32; l++) {  // a command buffer per block: one long buffer would hold up the display
    VitBlock *b = &vit.b[l];
    gpu_begin();
    vit_norm(&x, &b->norm1, &y, n);
    vit_linear(&b->qkv, n, &y, 1024, &qkv, &b->b_qkv);
    gpu_kern(K_vit_rope);
    gpu_buf(0, &qkv, 0), gpu_buf(1, &cs, 0);
    gpu_go(1024 / 256, n, 1, 256, 1, 1);
    gpu_kern(K_vit_heads);
    gpu_buf(0, &qkv, 0), gpu_buf(1, &hv, 0), gpu_bytes(2, dims, sizeof dims);
    gpu_go(768 / 256, np, 1, 256, 1, 1);
    gpu_kern(K_vit_attn);
    gpu_buf(0, &hv, 0), gpu_buf(1, &y, 0), gpu_bytes(2, dims, sizeof dims);
    gpu_go(np / 64, 16, 1, 128, 1, 1);
    vit_linear(&b->o, n, &y, 1024, &t, &b->b_o);
    vit_add(&x, &t, n);
    vit_norm(&x, &b->norm2, &y, n);
    vit_linear(&b->w1, n, &y, 1024, &h, 0);
    gpu_kern(K_vit_swiglu);
    gpu_buf(0, &h, 0), gpu_buf(1, &a, 0);
    gpu_go(2816 / 256, n, 1, 256, 1, 1);
    vit_linear(&b->w2, n, &a, 2816, &t, 0);
    vit_add(&x, &t, n);
    gpu_commit();
  }
  uint32_t gr[3] = {(uint32_t)im->vh, (uint32_t)im->vw, (uint32_t)im->tw};
  gpu_begin();
  vit_norm(&x, &vit.norm, &y, n);
  gpu_kern(K_vit_unfold);
  gpu_buf(0, &y, 0), gpu_buf(1, &u, 0), gpu_bytes(2, gr, sizeof gr);
  gpu_go(9216 / 256, L, 1, 256, 1, 1);
  vit_linear(&vit.a1, L, &u, 9216, &g, &vit.a1_b);
  gpu_commit();
  gpu_sync();
  dispatch_apply_f(L, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), g.ptr, vit_gelu_row);
  gpu_begin();
  vit_linear(&vit.a2, L, &g, DIM, &f, &vit.a2_b);
  gpu_commit();
  gpu_sync();
  for (size_t i = 0; i < (size_t)L * DIM; i++) im->rows[i] = (uint16_t)(((uint32_t *)f.ptr)[i] >> 16);
  Buf *all[] = {&px, &cs, &x, &y, &t, &qkv, &hv, &h, &a, &u, &g, &f};
  for (size_t i = 0; i < sizeof all / sizeof *all; i++) gpu_free(all[i]);
  memory_pin(weights, 0);
  return 1;
}

void deepseek_images(const Media *im, int n) { imgs = im, nimgs = n; }

// The embedding of prompt position pos inside an image span
// (merge_image_embeddings): the delimiters' learned rows, or the image's
// feature rows in reading order.
static const uint16_t *image_row(int pos) {
  for (int i = 0; i < nimgs; i++) {
    const Media *im = &imgs[i];
    int j = pos - im->start, n = media_span(im);
    if (j < 0 || j >= n) continue;
    if (j == 0) return (const uint16_t *)(image_markers[0].ptr + image_markers[0].off);
    if (j == n - 1) return (const uint16_t *)(image_markers[1].ptr + image_markers[1].off);
    int r = (j - 1) / (im->tw + 1), c = (j - 1) % (im->tw + 1);
    if (c == im->tw) return (const uint16_t *)(image_markers[2].ptr + image_markers[2].off);
    if (!im->rows) die("the image at position %d was not encoded", im->start);
    return im->rows + ((size_t)r * im->tw + c) * DIM;
  }
  die("position %d: an image's token outside the prompt's images", pos);
}

// ---- the DSpark draft (inference/model.py forward_spec) ----


// One stage with m query/output rows and all BLOCK key/value rows at pos...
// Each query attends to the stage's window (the main model's keys through
// pos - 1) and every drafted row's key. Only the final stage may trim m.
static void draft_stage(int d, int pos) {
  const int m = BLOCK;  // the token's row and the noise tokens'
  Layer *y = &D[d];
  hc_mix(BLOCK, &DHS, 0, y, 0, &MA, 0, &DMF, 0, 0);
  products(2, (Prod[]){{&y->wq_a, &QA, 0, QRANK}, {&y->wkv, &KV, 0, HEAD_DIM}}, m, &XQ, 0, DIM, 1);
  uint32_t n = QRANK;
  gpu_kern(K_rmsnorm);
  gpu_buf(0, &QA, 0), gpu_buf(1, &y->q_norm, 0), gpu_buf(2, &QR, 0), gpu_buf(3, &QRQ, 0), gpu_bytes(4, &n, 4);
  gpu_go(m, 1, 1, 1024, 1, 1);
  linear(&y->wq_b, m, &QRQ, 0, QRANK, &Q, 0, HEADS * HEAD_DIM, 1);
  Win bw = {(uint32_t)pos, (uint32_t)pos, 1u << 30};
  gpu_kern(K_kv_window);
  gpu_buf(0, &KV, 0), gpu_buf(1, &y->kv_norm, 0), gpu_buf(2, &cs_base, 0), gpu_buf(3, &BLK, 0), gpu_bytes(4, &bw, sizeof bw);
  gpu_go(BLOCK, 1, 1, 512, 1, 1);
  Att at = {{(uint32_t)pos - 1, 0, WINDOW}, 0, 1};
  for (int r = 0; r < m; r++)  // the window up to pos - 1; the angle table from row r's position pos + r
    attend_one(&Q, (size_t)r * HEADS * HEAD_DIM * 4, &y->ring, &BLK, &DSEL, 0, &y->sink, &cs_base,
               (size_t)(r + 1) * 32 * 8, &O, (size_t)r * HEADS * HEAD_DIM * 4, &at);
  uint32_t p32 = pos;
  gpu_kern(K_attn_out);
  gpu_buf(0, &O, 0), gpu_buf(1, &cs_base, 0), gpu_buf(2, &O, 0), gpu_bytes(3, &p32, 4);
  gpu_go(m * HEADS * HEAD_DIM / 256, 1, 1, 256, 1, 1);
  linear(&y->wo_a, m, &O, 0, HEADS * HEAD_DIM, &OA, 0, OGROUPS * ORANK, 1);
  quant(&OA, &OA, (size_t)m * OGROUPS * ORANK);
  linear(&y->wo_b, m, &OA, 0, OGROUPS * ORANK, &AO, 0, DIM, 1);
  gpu_kern(K_hc_post);
  gpu_buf(0, &AO, 0), gpu_buf(1, &MA, 0), gpu_buf(2, &DHS, 0), gpu_buf(3, &H1, 0);
  gpu_go(DIM / 256, m, 1, 256, 1, 1);
  hc_mix(m, &H1, 0, y, 1, &DMF, 0, &MA, 0, 0);
  linear(&y->gate, m, &X, 0, DIM, &GL, 0, DRAFT_EXPERTS, 0);
  route_top(m, &(Routing){&GL, 0, y, &RT, 0, DRAFT_TOPK, 16, 1}, 1, 0);
  shared_expert(m, y);
  gpu_commit();
  gpu_sync();
  int idx[BLOCK][DRAFT_TOPK], need[BLOCK * DRAFT_TOPK], nn = 0;
  float wts[BLOCK][DRAFT_TOPK];
  for (int r = 0; r < m; r++) {  // each row's experts in expert order, as the reference sums them
    memcpy(idx[r], (int *)RT.ptr + r * 16, sizeof idx[r]);
    memcpy(wts[r], (float *)RT.ptr + r * 16 + 8, sizeof wts[r]);
    // A bounded draft cache loads native selections; missing draft slots
    // must not activate the backbone's approximate substitution rule.
    if (opt.deepseek.substitute > 0 && cache_draft_capacity(experts_store) == DRAFT_EXPERTS)
      expert_stats()->substituted += substitute(LAYERS + d, &D[d], (const float *)GL.ptr + (size_t)r * DRAFT_EXPERTS,
                                            DRAFT_EXPERTS, idx[r], wts[r], DRAFT_TOPK, 0, 0);
    for (int i = 1; i < DRAFT_TOPK; i++)
      for (int j = i; j > 0 && idx[r][j - 1] > idx[r][j]; j--) {
        int t = idx[r][j]; idx[r][j] = idx[r][j - 1], idx[r][j - 1] = t;
        float u = wts[r][j]; wts[r][j] = wts[r][j - 1], wts[r][j - 1] = u;
      }
    for (int k = 0; k < DRAFT_TOPK; k++) {
      int i = 0;
      while (i < nn && need[i] != idx[r][k]) i++;
      if (i == nn) need[nn++] = idx[r][k];
    }
  }
  cache_load(experts_store, LAYERS + d, need, nn);
  gpu_begin();
  for (int r = 0; r < m; r++) {
    size_t eo = (size_t)(LAYERS * FEW + d * BLOCK + r) * 256;
    ExpertRef *ref = (ExpertRef *)(ETAB.ptr + eo);
    for (int k = 0; k < DRAFT_TOPK; k++) {
      uint32_t at[STORE_PARTS];
      int fresh;
      const Buf *b = cache_get(experts_store, LAYERS + d, idx[r][k], at, &fresh);
      ref[k] = (ExpertRef){{b->gpu + at[0], b->gpu + at[1], b->gpu + at[2]}, wts[r][k], 0};
      if (fresh) gpu_use(b);
    }
    gpu_kern(K_moe_up);
    gpu_buf(0, &ETAB, eo), gpu_buf(1, &XQ, (size_t)r * DIM * 4), gpu_buf(2, &MH, (size_t)r * DRAFT_TOPK * INTER * 4);
    gpu_go(INTER / 32, DRAFT_TOPK, 1, 256, 1, 1);
  }
  for (int r = 0; r < m; r++) {
    n = DRAFT_TOPK;
    gpu_kern(K_moe_down);
    gpu_buf(0, &ETAB, (size_t)(LAYERS * FEW + d * BLOCK + r) * 256), gpu_buf(1, &MH, (size_t)r * DRAFT_TOPK * INTER * 4);
    gpu_buf(2, &SHO, (size_t)r * DIM * 4), gpu_buf(3, &DMF, (size_t)r * MIX * 4);
    gpu_buf(4, &H1, (size_t)r * HC * DIM * STREAM_BYTES), gpu_buf(5, &DHS, (size_t)r * HC * DIM * STREAM_BYTES), gpu_bytes(6, &n, 4);
    uint32_t pm = 0;
    gpu_bytes(7, &pm, 4), gpu_buf(8, &PARTS, 0);
    gpu_go(DIM / 4, 1, 1, 32 * DRAFT_TOPK, 1, 1);
  }
}

// The rows are the token at pos and the noise token; after the stages, each
// row's logits plus the Markov head's bias for the token before it (the
// token, then the previous draft) give the next draft, on the GPU. Every
// stage computes all BLOCK rows, whose keys every query sees; the head and
// bias use the first n rows, whose drafts are wanted.
void deepseek_draft(int token, int pos, int *drafts, int n) {
  if (token < 0) die("a draft after an image's token (position %d)", pos);
  if (n < 1 || n > BLOCK) die("deepseek_draft needs 1 to %d draft rows", BLOCK);
  work_capacity(BLOCK);
  double t0 = now();
  int64_t t = token;
  io_rows(embed_t.fd, embed_t.off, DIM * 2, &t, 1, DEROW.ptr);
  for (int r = 0; r < BLOCK; r++) {  // the first stage collapses the identical streams with (1, 0, 0, 0)
    float *pre = (float *)DMF.ptr + r * MIX;
    pre[0] = 1, pre[1] = pre[2] = pre[3] = 0;
  }
  ((int *)DTOK.ptr)[0] = token;
  gpu_begin();
  gpu_kern(K_embed);
  gpu_buf(0, &DEROW, 0), gpu_buf(1, &DHS, 0);
  gpu_go(DIM / 256, BLOCK, 1, 256, 1, 1);
  for (int d = 0; d < DRAFTS; d++) draft_stage(d, pos);
  hc_pre_norm(n, &DHS, 0, &DMF, 0, &draft_norm);
  head_out(n, &DLOG);
  uint32_t v = VOCAB;
  for (int r = 0; r < n; r++) {
    gpu_kern(K_markov);
    gpu_buf(0, &markov_embed, 0), gpu_buf(1, &markov_head, 0), gpu_buf(2, &DTOK, r * 4);
    gpu_buf(3, &DLOG, (size_t)r * VOCAB * 4);
    gpu_go(VOCAB / 8, 1, 1, 256, 1, 1);
    gpu_kern(K_argmax);
    gpu_buf(0, &DLOG, (size_t)r * VOCAB * 4), gpu_buf(1, &DTOK, (r + 1) * 4), gpu_bytes(2, &v, 4);
    gpu_go(1, 1, 1, 1024, 1, 1);
  }
  gpu_commit(), gpu_sync();
  memcpy(drafts, (int *)DTOK.ptr + 1, n * sizeof *drafts);
  phase_s[6] += now() - t0;
}


// A pass of the token at pos and its drafts. Rows after the accepted ones
// leave nothing later positions read (see RING and PENDING), except the
// Engram history, which deepseek_accept rebuilds, and the draft's windows, which
// it writes for the accepted rows only.
static int32_t verify_history[3];

const float *deepseek_verify(const int *tokens, int m, int pos) {
  if (m < 1 || m > replay_margin() + 1 || m > FEW) die("invalid target verification row count");
  memcpy(verify_history, history, sizeof history);
  verifying = 1;
  step(tokens, m, pos);
  verifying = 0;
  return (const float *)logits.ptr;
}


void deepseek_accept(const int *tokens, int n, int pos) {
  memcpy(history, verify_history, sizeof history);
  for (int i = 0; i < n; i++) history[2] = history[1], history[1] = history[0], history[0] = ngram.map[tokens[i]];
  if (opt.draft) {
    verifying = 1;
    draft_keys(n, pos);
    verifying = 0;
  }
}

// Clears the request's state: window rings, compressed rows, compressor
// rows and the Engram history. A state that grew shrinks back.
void deepseek_reset(void) {
  // The online learner belongs to the engine, not a conversation's KV state.
  shrink();
  gpu_sync();
  for (int d = 0; opt.draft && d < DRAFTS; d++) memset(D[d].ring.ptr, 0, D[d].ring.size);
  for (int l = 0; l < LAYERS; l++) {
    memset(L[l].ring.ptr, 0, L[l].ring.size);
    if (L[l].kv_source) {
      memset(L[l].comp.ptr, 0, L[l].comp.size);
      memset(L[l].keys.ptr, 0, L[l].keys.size);
      memset(L[l].cstate.ptr, 0, L[l].cstate.size);
    }
  }
  memset(history, 0, sizeof history);
}

// ---- kernel bandwidth on the model's own matrices ----

static int descending(const void *a, const void *b) {
  float x = *(const float *)a, y = *(const float *)b;
  return (x < y) - (x > y);
}

// One matrix per layer, so that no cache holds them; the best of three
// rounds after a warm-up round.
static double timed(void (*f)(int), int reps) {
  double best = 1e9;
  for (int round = 0; round < 4; round++) {
    double g = gpu_busy;
    gpu_begin();
    for (int r = 0; r < reps; r++) f(r);
    gpu_commit();
    gpu_sync();
    if (round && gpu_busy - g < best) best = gpu_busy - g;
  }
  return best / reps;
}

static size_t bench_field;
static const Buf *bench_in;
static int bench_m;
static void bench_mv(int r) {
  Lin *w = (Lin *)((char *)&L[r % LAYERS] + bench_field);
  linear(w, 1, bench_in, 0, w->k, &AO, 0, w->n, 1);
}
static void bench_mm(int r) {
  Lin *w = (Lin *)((char *)&L[r % LAYERS] + bench_field);
  linear(w, bench_m, bench_in, 0, w->k, &Q, 0, w->n, 1);
}
static void bench_head(int r) { head_out(1, &logits); }

static void bench_experts(int r) {
  int idx[TOPK] = {0, 1, 2, 3, 4, 5};
  float wts[TOPK] = {0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f};
  memcpy(RT.ptr, idx, sizeof idx), memcpy(RT.ptr + 32, wts, sizeof wts);
  routed_rows(&(Batch){1, 0, 0}, r % 8, AHEAD, INFLIGHT, 0, 0);
}

static void bench_hc(int r) { hc_mix(1, &HS, 0, &L[5], 0, &MA, 0, &MF, 0, 0); }
static void bench_attend(int r) {
  Att at = {{1000, 0, RING}, 0, 1};
  ((int *)SEL.ptr)[0] = 512;
  attend_one(&Q, 0, &L[5].ring, &L[2].comp, &SEL, 0, &L[5].sink, &cs_comp, 0, &O, 0, &at);
}

void deepseek_bench(void) {
  work_capacity(MB);
#define OFF(f) offsetof(Layer, f)
  struct { const char *name; size_t field; const Buf *in; } mats[] = {
      {"wq_a  1280x5120 fp8", OFF(wq_a), &XQ}, {"wq_b  32768x1280 fp8", OFF(wq_b), &QRQ},
      {"wkv   512x5120 fp8", OFF(wkv), &XQ}, {"wo_a  8192x4096 fp8", OFF(wo_a), &O},
      {"wo_b  5120x8192 fp8", OFF(wo_b), &OA}, {"sh1   2304x5120 fp8", OFF(sh1), &XQ},
      {"sh2   5120x2304 fp8", OFF(sh2), &SHH}, {"gate  384x5120 bf16", OFF(gate), &X}};
  for (size_t i = 0; i < sizeof mats / sizeof *mats; i++) {
    Lin *w = (Lin *)((char *)&L[0] + mats[i].field);
    bench_field = mats[i].field, bench_in = mats[i].in, bench_m = 1024;
    double tm = timed(bench_mm, LAYERS);  // first, so that its milliseconds of work raise the clock for one row's
    double t = timed(bench_mv, LAYERS), bytes = (double)w->n * w->k * (w->s.obj ? 1 : 2);
    fprintf(stderr, "%-24s %8.1f us %7.1f GB/s | 1024 rows %8.1f us %6.1f TFLOPS\n", mats[i].name, t * 1e6,
            bytes / t / 1e9, tm * 1e6, 2.0 * 1024 * w->n * w->k / tm / 1e12);
  }
  {  // products of 4 rows at once (verify passes) against each row alone, and their time
    const Lin *ws[] = {&L[3].wo_a, &L[3].wq_b, &L[3].gate};
    const char *names[] = {"wo_a (grouped fp8)", "wq_b (fp8)", "gate (bf16)"};
    for (int i = 0; i < 3; i++) {
      const Lin *w = ws[i];
      float *in = (float *)SC.ptr;
      for (size_t j = 0; j < 4 * (size_t)w->k; j++) in[j] = (float)((j * 2654435761u) % 2001) / 1000.0f - 1.0f;
      gpu_begin();
      for (int r = 0; r < 4; r++) linear(w, 1, &SC, (size_t)r * w->k * 4, w->k, &O, (size_t)r * w->n * 4, w->n, 1);
      linear(w, 4, &SC, 0, w->k, &Q, 0, w->n, 1);
      gpu_commit();
      gpu_sync();
      int diff = memcmp(O.ptr, Q.ptr, (size_t)4 * w->n * 4) != 0;
      bench_field = (size_t)((const char *)w - (const char *)&L[3]), bench_in = &SC;
      double t1 = timed(bench_mv, LAYERS), tm[3];
      for (int j = 0; j < 3; j++) bench_m = 2 << j, tm[j] = timed(bench_mm, LAYERS) / t1;
      fprintf(stderr, "%-24s 4 rows at once %s each row alone; 2, 4, 8 rows take %.2f, %.2f, %.2f x one row's time\n",
              names[i], diff ? "DIFFER from" : "equal", tm[0], tm[1], tm[2]);
    }
  }
  bench_field = OFF(wq_a), bench_in = &XQ;
  double t = timed(bench_head, 4);
  fprintf(stderr, "%-24s %8.1f us %7.1f GB/s\n", head_q6.obj ? "head  129280x5120 q6_k" : "head  129280x5120 bf16", t * 1e6,
          129280.0 * 5120 * (head_q6.obj ? 210.0 / 256 : 2) / t / 1e9);
  int idx[TOPK] = {0, 1, 2, 3, 4, 5};
  for (int l = 0; l < 8; l++) cache_load(experts_store, l, idx, TOPK);  // 48 distinct experts

  t = timed(bench_experts, 40);
  double ebytes = 0;  // one expert's bytes as stored, averaged over layers 0 to 7
  for (int l = 0; l < 8; l++)
    for (int k = 0; k < expert_spans[l]; k++) ebytes += expert_span[l][0][k].len / 8.0;
  fprintf(stderr, "%-24s %8.1f us %7.1f GB/s\n", "6 routed experts", t * 1e6, 6.0 * ebytes / t / 1e9);
  double t0 = now();  // CPU-GPU round trip: a tiny command buffer signals, the CPU sees it
  for (int r = 0; r < 200; r++) {
    gpu_begin();
    quant(&X, &XQ, DIM);
    uint64_t v = gpu_signal();
    gpu_commit();
    gpu_wait_signal(v);
  }
  gpu_sync();
  fprintf(stderr, "%-24s %8.1f us\n", "signal round trip", (now() - t0) / 200 * 1e6);
  {  // the top 512 of 3000 index scores, against a CPU selection; distinct, then tied scores
    for (int tied = 0; tied < 2; tied++) {
      const int n = 3000;
      float *sc = (float *)SC.ptr;
      for (int i = 0; i < n; i++) sc[i] = tied ? (float)((i * 7919) % 13) : (float)((i * 2654435761u) % 100003) - 50000;
      Index ix = {n - 1, 1, (uint32_t)opt.ctx, 0};
      gpu_begin();
      gpu_kern(K_index_topk);
      gpu_buf(0, &SC, 0), gpu_buf(1, &SEL, 0), gpu_bytes(2, &ix, sizeof ix), gpu_buf(3, &CAND, 0);
      gpu_go(1, 1, 1, 1024, 1, 1);
      gpu_commit();
      gpu_sync();
      const int *sel = (const int *)SEL.ptr;
      float kth;  // the 512th largest score
      float *sorted = xmalloc(n * 4);
      memcpy(sorted, sc, n * 4);
      for (int i = 0; i < 512; i++)
        for (int j = i + 1; j < n; j++)
          if (sorted[j] > sorted[i]) { float t = sorted[i]; sorted[i] = sorted[j]; sorted[j] = t; }
      kth = sorted[511];
      int above = 0, ok = sel[0] == 512;
      for (int i = 0; i < n; i++) above += sc[i] > kth;
      for (int k = 0, at_kth = 0; k < 512 && ok; k++) {
        int p = sel[1 + k];
        ok = p >= 0 && p < n && (k == 0 || p > sel[k]) && sc[p] >= kth;
        at_kth += sc[p] == kth;
        if (k == 511) ok = ok && at_kth == 512 - above;
      }
      fprintf(stderr, "index_topk of 3000 %s scores: %s\n", tied ? "tied" : "distinct", ok ? "matches" : "MISMATCH");
      free(sorted);
    }
  }
  {  // layer 20's candidate blocks of 40,000 scores, and the top 512 within them, against a CPU selection
    const int n = 40000, nb = (n + 7) / 8;
    float *sc = (float *)SC.ptr, *best = xmalloc(nb * 4), *sorted = xmalloc(n * 4);
    for (int i = 0; i < n; i++) sc[i] = (float)((i * 2654435761u) % 100003) - 50000;
    Index ix = {n - 1, 1, n, (nb + 31) / 32};
    gpu_begin();
    gpu_kern(K_cand_topk);
    gpu_buf(0, &SC, 0), gpu_buf(1, &CAND, 0), gpu_bytes(2, &ix, sizeof ix);
    gpu_go(1, 1, 1, 1024, 1, 1);
    gpu_kern(K_index_topk);
    gpu_buf(0, &SC, 0), gpu_buf(1, &SEL, 0), gpu_bytes(2, &ix, sizeof ix), gpu_buf(3, &CAND, 0);
    gpu_go(1, 1, 1, 1024, 1, 1);
    gpu_commit();
    gpu_sync();
    for (int b = 0; b < nb; b++) {  // a block's best score; the newest block always in
      best[b] = b == nb - 1 ? INFINITY : sc[b * 8];
      for (int i = b * 8 + 1; b < nb - 1 && i < b * 8 + 8; i++) best[b] = fmaxf(best[b], sc[i]);
    }
    memcpy(sorted, best, nb * 4);
    qsort(sorted, nb, 4, descending);
    const uint32_t *bits = (const uint32_t *)CAND.ptr;
    float kth = sorted[CAND_TOP - 1];  // the 2048th best block
    int kept = 0, ok = 1, m = 0;
    for (int b = 0; b < nb; b++) {
      int in = bits[b / 32] >> (b % 32) & 1;
      ok &= in == (best[b] >= kth), kept += in;
      for (int i = b * 8; in && i < b * 8 + 8 && i < n; i++) sorted[m++] = sc[i];  // the candidates' scores
    }
    qsort(sorted, m, 4, descending);
    const int *sel = (const int *)SEL.ptr;
    int good = sel[0] == 512, above = 0;
    for (int b = 0; b < nb; b++)
      for (int i = b * 8; (bits[b / 32] >> (b % 32) & 1) && i < b * 8 + 8 && i < n; i++) above += sc[i] > sorted[511];
    for (int k = 0, at = 0; k < 512 && good; k++) {
      int p = sel[1 + k];
      good = p >= 0 && p < n && (k == 0 || p > sel[k]) && (bits[p / 256] >> (p / 8 % 32) & 1) && sc[p] >= sorted[511];
      at += sc[p] == sorted[511];
      if (k == 511) good = good && at == 512 - above;
    }
    fprintf(stderr, "cand_topk of %d blocks: %d kept, %s; index_topk within them: %s\n", nb, kept,
            ok && kept == CAND_TOP ? "matches" : "MISMATCH", good ? "matches" : "MISMATCH");
    free(best), free(sorted);
  }
  // the same work after idle gaps, as decoding waits for the SSD between layers,
  // and after a gap in which gpu_keep_clock spins
  for (int mode = 0; mode < 3; mode++) {
    int gap = mode ? 3 : 0;
    double busy = 0;
    for (int r = 0; r < 40; r++) {
      if (mode == 2) gpu_keep_clock(2.8e-3);
      double a = now();
      while (now() - a < gap * 1e-3) {}
      double g = gpu_busy;
      gpu_begin();
      for (int k = 0; k < 6; k++) bench_mv(r * 6 + k);
      gpu_commit();
      gpu_sync();
      busy += gpu_busy - g;
    }
    fprintf(stderr, "6 x wq_a after %d ms %s %8.1f us\n", gap, mode == 2 ? "kept busy" : "idle     ", busy / 40 * 1e6);
  }
  struct { const char *name; void (*f)(int); } small[] = {{"hc mix, split, pre-norm", bench_hc},
                                                          {"attention (640 rows)", bench_attend}};
  for (size_t i = 0; i < sizeof small / sizeof *small; i++)
    fprintf(stderr, "%-24s %8.1f us\n", small[i].name, timed(small[i].f, 40) * 1e6);
}

// ---- the attention state of several sequences ----

// active: deepseek_load's state, or the one deepseek_state_use chose
static int taken;      // deepseek_load's state went to a caller of deepseek_state_new
static size_t state_rings[LAYERS], state_compressors[LAYERS], state_drafts[DRAFTS];

// Allocation layout belongs to the loaded model, not to whichever state
// happens to be bound. Eviction can leave the bindings empty.
static void remember_state_layout(void) {
  for (int l = 0; l < LAYERS; l++) {
    state_rings[l] = L[l].ring.size;
    state_compressors[l] = L[l].kv_source ? L[l].cstate.size : 0;
  }
  for (int d = 0; d < DRAFTS; d++) state_drafts[d] = opt.draft ? D[d].ring.size : 0;
}

static int start_cap(void) { return opt.ctx < CAP ? opt.ctx : CAP; }

static int grown(int cap, int need) {
  while (cap < need) cap *= 2;
  return cap < opt.ctx ? cap : opt.ctx;
}

static size_t rows_bytes(int l, int cap, int keys) {  // a KV source's compressed rows or index keys, BF16
  return (size_t)(cap / L[l].ratio + 1) * (keys ? IDX_DIM : HEAD_DIM) * 2;
}

// The rows later positions overwrite: every window ring, the compressor
// rows and the Engram history.
static size_t saved_bytes(void) {
  size_t n = sizeof history;
  for (int l = 0; l < LAYERS; l++) n += state_rings[l] + state_compressors[l];
  for (int d = 0; opt.draft && d < DRAFTS; d++) n += state_drafts[d];
  return n;
}

// The bytes a state owns once it holds `positions` (s null: a new state):
// its rolling buffers (saved_bytes() less the history, which lives in the
// State itself), its compressed rows and index keys, and the rewind copy
// that deepseek_state_save made, if any.
size_t deepseek_state_bytes(const State *s, int positions) {
  int cap = grown(s ? s->cap : start_cap(), positions);
  size_t n = saved_bytes() - sizeof history + (s && s->saved ? saved_bytes() : 0);
  for (int l = 0; l < LAYERS; l++)
    if (L[l].kv_source) n += rows_bytes(l, cap, 0) + rows_bytes(l, cap, 1);
  return n;
}

// s's compressed rows and index keys for cap positions, keeping those of
// the first `keep` positions.
static void resize(State *s, int cap, int keep) {
  gpu_sync();
  for (int l = 0; l < LAYERS; l++) {
    if (!L[l].kv_source) continue;
    for (int k = 0; k < 2; k++) {
      Buf *b = k ? &s->keys[l] : &s->comp[l], nb = gpu_alloc_kv(rows_bytes(l, cap, k));
      size_t used = rows_bytes(l, keep, k);
      memcpy(nb.ptr, b->ptr, used < b->size ? used : b->size);
      gpu_free(b);
      *b = nb;
    }
    if (s == active) L[l].comp = s->comp[l], L[l].keys = s->keys[l];
  }
  s->cap = cap;
}

static void reserve(int need, int keep) {  // the state in use holds need positions
  if (need > active->cap) resize(active, grown(active->cap, need), keep);
}

static void shrink(void) {  // the state in use back to its first capacity
  if (active->cap > start_cap()) resize(active, start_cap(), 0);
}

static void base_state(void) {  // deepseek_load's buffers as the state in use
  remember_state_layout();
  active = xmalloc(sizeof *active);
  for (int l = 0; l < LAYERS; l++) {
    active->ring[l] = L[l].ring;
    if (L[l].kv_source) active->comp[l] = L[l].comp, active->keys[l] = L[l].keys, active->cstate[l] = L[l].cstate;
  }
  for (int d = 0; opt.draft && d < DRAFTS; d++) active->dring[d] = D[d].ring;
  active->cap = start_cap();
}

State *deepseek_state_new(void) {
  if (!taken) return taken = 1, active;
  State *s = xmalloc(sizeof *s);
  s->cap = start_cap();
  for (int l = 0; l < LAYERS; l++) {
    s->ring[l] = kv_zeros(state_rings[l]);
    if (!L[l].kv_source) continue;
    s->comp[l] = kv_zeros(rows_bytes(l, s->cap, 0)), s->keys[l] = kv_zeros(rows_bytes(l, s->cap, 1));
    s->cstate[l] = kv_zeros(state_compressors[l]);
  }
  for (int d = 0; opt.draft && d < DRAFTS; d++) s->dring[d] = kv_zeros(state_drafts[d]);
  return s;
}

void deepseek_state_free(State *s) {
  if (s == active) die("the attention state in use cannot be freed");
  for (int l = 0; l < LAYERS; l++) {
    gpu_free(&s->ring[l]);
    if (L[l].kv_source) gpu_free(&s->comp[l]), gpu_free(&s->keys[l]), gpu_free(&s->cstate[l]);
  }
  for (int d = 0; opt.draft && d < DRAFTS; d++) gpu_free(&s->dring[d]);
  memory_host_free(s->saved);
  free(s);
}

static void bind(const State *s) {  // the layers' state buffers: s's, for the dispatches encoded next
  for (int l = 0; l < LAYERS; l++) {
    L[l].ring = s->ring[l];
    if (L[l].kv_source) L[l].comp = s->comp[l], L[l].keys = s->keys[l], L[l].cstate = s->cstate[l];
  }
  for (int d = 0; opt.draft && d < DRAFTS; d++) D[d].ring = s->dring[d];
}

void deepseek_state_use(State *s) {
  gpu_sync();
  memcpy(active->history, history, sizeof history);
  bind(s);
  memcpy(history, s->history, sizeof history);
  active = s;
}

// Ready-request scheduling.
enum { RUN_BUFFERS = 14 };
#define RUN_BUFFER_LIST &HS, &MF, &SEL, &CAND, &XQ, &H1, &SHO, &RT, &EC[0], &EC[1], &ES[0], &ES[1], &ETAB, &EROW
typedef struct {
  Buf saved[RUN_BUFFERS];
  State *state;
  int layer, pos, experts[TOPK];
} RunJob;
static RunJob run_jobs[FEW];
static float *run_logits;

static void run_bind(RunJob *job, Buf **live) {
  bind(job->state);
  for (int k = 0; k < RUN_BUFFERS; k++) *live[k] = job->saved[k];
}

static void run_issue(RunJob *job) {
  const int *rt = (const int *)RT.ptr;
  CacheObservation observation;
  cache_observe_snapshot(experts_store, job->layer, &observation);
  cache_observe_routes(experts_store, &observation, rt, 1, 16, 0);
  cache_observe_routes(experts_store, &observation, rt, 1, 16, 1);
  uint64_t *tally[] = {&expert_stats()->row_ssd, &expert_stats()->row_own, &expert_stats()->row_ahead, &expert_stats()->row_late};
  for (int k = 0; k < TOPK; k++) {
    job->experts[k] = rt[k];
    (*tally[cache_where(experts_store, job->layer, rt[k])])++;
  }
  for (int i = 1; i < TOPK; i++)
    for (int k = i; k > 0 && job->experts[k] < job->experts[k - 1]; k--) {
      int e = job->experts[k]; job->experts[k] = job->experts[k - 1], job->experts[k - 1] = e;
    }
  expert_stats()->row_unique += TOPK;
  cache_hold(experts_store, job->layer, job->experts, TOPK);
  cache_demote(experts_store, job->layer);
}

// Gather ready jobs at one layer into the existing small-batch kernels.
// Blit copies carry tracked source/destination dependencies across buffers.
static int run_group(RunJob **jobs, const int *ids, int count, Buf **live, const Buf *original, float *outputs) {
  const size_t bytes[RUN_BUFFERS] = {HC * DIM * STREAM_BYTES, MIX * 4, (IDX_TOPK + 1) * 4,
    (size_t)cand_words * 4, DIM * 4, HC * DIM * STREAM_BYTES, DIM * 4, 0,
    ENGRAM_ROWS * ENGRAM_DIM, ENGRAM_ROWS * ENGRAM_DIM, ENGRAM_ROWS * 8, ENGRAM_ROWS * 8, 0, 0};
  GpuCopy copies[FEW * RUN_BUFFERS];
  int ncopy = 0, layer = jobs[0]->layer;
  runahead_group_count++, runahead_group_rows += count;
  for (int k = 0; k < RUN_BUFFERS; k++) *live[k] = original[k];
  for (int r = 0; r < count; r++) {
    RunJob *job = jobs[r];
    cache_wait(experts_store, layer, job->experts, TOPK);
    for (int k = 0; k < RUN_BUFFERS; k++) if (bytes[k])
      copies[ncopy++] = (GpuCopy){&job->saved[k], live[k], 0, r * bytes[k], bytes[k]};
  }
  gpu_begin();
  gpu_copies(copies, ncopy);
  size_t entries = (size_t)layer * FEW * 256;
  for (int r = 0; r < count; r++) {
    const int *rt = (const int *)jobs[r]->saved[7].ptr;
    fill_entries(layer, entries + r * 256, rt, (const float *)rt + 8, NULL, 0);
  }
  for (int r = 0; r < count; r++)
    experts_go(layer, r, entries + r * 256, (size_t)r * MIX * 4, (size_t)r * HC * DIM * STREAM_BYTES, 0);
  uint64_t signal;
  if (layer == LAYERS - 1) {
    hc_pre_norm(count, &HS, 0, &MF, 0, &final_norm);
    head_out(count, &logits);
    signal = gpu_signal();
    gpu_commit();
  } else {
    gpu_commit();
    for (int r = 0; r < count; r++) row_state[r] = jobs[r]->state, row_pos[r] = jobs[r]->pos;
    nrows = count;
    signal = layer_front(&(Batch){count, jobs[0]->pos, 0}, layer + 1);
    nrows = 0;
  }
  gpu_wait_signal(signal);
  if (layer != LAYERS - 1) {
    ncopy = 0;
    for (int r = 0; r < count; r++) {
      RunJob *job = jobs[r];
      memcpy(job->saved[7].ptr, original[7].ptr + (size_t)r * 16 * 4, 16 * 4);
      for (int k = 0; k < 7; k++)
        copies[ncopy++] = (GpuCopy){live[k], &job->saved[k], r * bytes[k], 0, bytes[k]};
    }
    gpu_begin();
    gpu_copies(copies, ncopy);
    gpu_commit();
  }
  for (int r = 0; r < count; r++) {
    RunJob *job = jobs[r];
    cache_unhold(experts_store, layer, job->experts, TOPK);
    job->layer++;
    if (layer == LAYERS - 1) memcpy(outputs + (size_t)ids[r] * VOCAB, logits.ptr + (size_t)r * VOCAB * 4, VOCAB * 4);
    else RT = job->saved[7], run_issue(job);
  }
  RT = original[7];
  return layer == LAYERS - 1 ? count : 0;
}

static const float *runahead_rows(State **states, const int *tokens, const int *positions, int count) {
  if (nrows || verifying || count < 2 || count > FEW) die("invalid runahead batch");
  for (int r = 0; r < count; r++) if (tokens[r] < 0) die("runahead needs text decode tokens");
  runahead_route_only = 1;
  request_capacity(count);
  State *original_state = active;
  Buf *live[RUN_BUFFERS] = {RUN_BUFFER_LIST};
  Buf original[RUN_BUFFERS];
  for (int k = 0; k < RUN_BUFFERS; k++) original[k] = *live[k];
  if (!run_logits) {
    size_t bytes = (size_t)FEW * VOCAB * sizeof(float);
    run_logits = memory_host_required(bytes, 0), scheduler_workspace_bytes += bytes;
  }
  for (int r = 0; r < count; r++) {
    RunJob *job = &run_jobs[r];
    job->state = states[r], job->pos = positions[r], job->layer = 0;
    for (int k = 0; k < RUN_BUFFERS; k++) if (job->saved[k].size != original[k].size) {
      scheduler_workspace_bytes -= job->saved[k].size;
      gpu_free(&job->saved[k]);
      job->saved[k] = zeros(original[k].size);
      scheduler_workspace_bytes += job->saved[k].size;
    }
  }
  CacheStats before = *expert_stats();
  double began = now(), gpu_before = gpu_busy;
  cache_metrics_begin(experts_store, PF_DECODE, count);
  cache_adapt(experts_store); // no job holds a slot or a GPU command at a step boundary
  for (int r = 0; r < count; r++) {
    RunJob *job = &run_jobs[r];
    run_bind(job, live);
    int64_t rows[1][2][ENGRAM_ROWS];
    memcpy(history, job->state->history, sizeof history);
    engram_hash(tokens[r], job->pos, rows[0]);
    memcpy(job->state->history, history, sizeof history);
    // These small reads finish before switching EC/ES to another job.
    // Expert reads from preceding jobs remain in flight meanwhile.
    engram_read(0, rows, 1), engram_read(1, rows, 1);
    embed_read(tokens + r, 1, 0, job->pos);
    gpu_wait_signal(layer_front(&(Batch){1, job->pos, 0}, 0));
    run_issue(job);
  }
  int remaining = count;
  while (remaining) {
    int ready = -1, wait = -1, wait_misses = TOPK + 1;
    for (int r = 0; r < count; r++) {
      RunJob *job = &run_jobs[r];
      if (job->layer == LAYERS) continue;
      int missing = 0;
      for (int k = 0; k < TOPK; k++) missing += !cache_ready(experts_store, job->layer, job->experts[k]);
      if (!missing && (ready < 0 || job->layer < run_jobs[ready].layer)) ready = r;
      if (wait < 0 || job->layer < run_jobs[wait].layer ||
          (job->layer == run_jobs[wait].layer && missing < wait_misses)) wait = r, wait_misses = missing;
    }
    int r = ready >= 0 ? ready : wait;
    if (r < 0) die("runahead has no unfinished job");
    if (ready >= 0) {
      int ids[FEW], group = 0;
      RunJob *jobs[FEW];
      for (int j = 0; j < count; j++) if (run_jobs[j].layer == run_jobs[ready].layer) {
        int available = 1;
        for (int k = 0; k < TOPK; k++) available &= cache_ready(experts_store, run_jobs[j].layer, run_jobs[j].experts[k]);
        if (available) ids[group] = j, jobs[group++] = &run_jobs[j];
      }
      if (group > 1) {
        remaining -= run_group(jobs, ids, group, live, original, run_logits);
        continue;
      }
    }
    RunJob *job = &run_jobs[r];
    if (ready < 0) gpu_keep_clock(.001);
    cache_wait(experts_store, job->layer, job->experts, TOPK);
    run_bind(job, live);
    const int *rt = (const int *)RT.ptr;
    uint64_t signal = 0;
    gpu_begin();
    encode_experts(job->layer, 0, rt, (const float *)RT.ptr + 8, 0, 0);
    if (job->layer == LAYERS - 1) {
      hc_pre_norm(1, &HS, 0, &MF, 0, &final_norm);
      head_out(1, &logits);
      signal = gpu_signal();
    }
    gpu_commit();
    if (job->layer + 1 < LAYERS) signal = layer_front(&(Batch){1, job->pos, 0}, job->layer + 1);
    gpu_wait_signal(signal);
    cache_unhold(experts_store, job->layer, job->experts, TOPK);
    if (++job->layer == LAYERS) {
      memcpy(run_logits + (size_t)r * VOCAB, logits.ptr, VOCAB * sizeof(float));
      remaining--;
    } else run_issue(job);
  }
  gpu_sync();
  for (int k = 0; k < RUN_BUFFERS; k++) *live[k] = original[k];
  bind(original_state), memcpy(history, original_state->history, sizeof history);
  cache_metrics_end(experts_store);
  stats_add(&decode_stats, &before);
  decode_rows += count, decode_gpu_s += gpu_busy - gpu_before, decode_s += now() - began;
  runahead_route_only = 0;
  return run_logits;
}

// Layer-stage pipeline scheduling.
enum { PIPE_BUFFERS = 17, PIPE_BATCHES = 2 * LAYERS * BATCH_ROWS };
#define PIPE_BUFFER_LIST &HS, &MF, &SEL, &CAND, &XQ, &H1, &SHO, &RT, &EC[0], &EC[1], &ES[0], &ES[1], &ETAB, &EROW, &MH, &logits, &IMG
enum { PIPE_FRONT, PIPE_READ, PIPE_DOWN, PIPE_FINISH, PIPE_DONE };
typedef struct {
  Buf saved[PIPE_BUFFERS];
  State *state;
  int row, layer, pos, phase, held_layer, experts[TOPK];
  uint64_t signal, release_at;
} PipeJob;
static PipeJob pipe_jobs[BATCH_ROWS];
static Buf pipe_tasks;
static float *pipe_logits;
static int K_pipe = -1;
static struct {
  uint64_t steps, batches, tasks, mixed_stages, mixed_layers, fronts, max_inflight;
  double read_wait, gpu_wait;
} pipe_stats;

static void pipeline_report(FILE *out) {
  if (!pipe_stats.steps) return;
  fprintf(out, "{\"pipeline\":{\"steps\":%llu,\"stage_batches\":%llu,\"tasks\":%llu,\"mixed_stages\":%llu,"
               "\"mixed_layers\":%llu,\"fronts\":%llu,\"max_inflight\":%llu,\"read_wait_seconds\":%.6f,\"gpu_wait_seconds\":%.6f}}\n",
          (unsigned long long)pipe_stats.steps, (unsigned long long)pipe_stats.batches, (unsigned long long)pipe_stats.tasks,
          (unsigned long long)pipe_stats.mixed_stages, (unsigned long long)pipe_stats.mixed_layers,
          (unsigned long long)pipe_stats.fronts, (unsigned long long)pipe_stats.max_inflight,
          pipe_stats.read_wait, pipe_stats.gpu_wait);
}

static void pipe_bind(PipeJob *job, Buf **live) {
  bind(job->state);
  for (int k = 0; k < PIPE_BUFFERS; k++) *live[k] = job->saved[k];
}

static void pipe_front(PipeJob *job, Buf **live) {
  pipe_bind(job, live);
  job->phase = PIPE_FRONT;
  job->signal = layer_front(&(Batch){1, job->pos, 0}, job->layer);
  pipe_stats.fronts++;
}

static void pipe_issue(PipeJob *job) {
  const int *rt = (const int *)job->saved[7].ptr;
  CacheObservation observed;
  cache_observe_snapshot(experts_store, job->layer, &observed);
  cache_observe_routes(experts_store, &observed, rt, 1, 16, 0);
  cache_observe_routes(experts_store, &observed, rt, 1, 16, 1);
  uint64_t *tally[] = {&expert_stats()->row_ssd, &expert_stats()->row_own, &expert_stats()->row_ahead, &expert_stats()->row_late};
  for (int k = 0; k < TOPK; k++) {
    int e = rt[k], where = cache_where(experts_store, job->layer, e);
    job->experts[k] = e;
    (*tally[where])++;
    route_id[job->row][job->layer][k] = e;
    route_from[job->row][job->layer][k] = (where + 4) % 5;
    route_step[job->row][job->layer] = route_steps;
  }
  for (int i = 1; i < TOPK; i++)
    for (int k = i; k && job->experts[k] < job->experts[k - 1]; k--) {
      int e = job->experts[k]; job->experts[k] = job->experts[k - 1], job->experts[k - 1] = e;
    }
  expert_stats()->row_unique += TOPK;
  cache_hold(experts_store, job->layer, job->experts, TOPK);
  cache_demote(experts_store, job->layer);
  job->held_layer = job->layer;
  job->phase = PIPE_READ;
}

static void pipe_single(int kind, int layer, size_t entries) {
  gpu_kern(kind ? K_moe_down : K_moe_up);
  gpu_buf(0, &ETAB, entries);
  if (!kind) {
    gpu_buf(1, &XQ, 0), gpu_buf(2, &MH, 0);
    gpu_go(INTER / 32, TOPK, 1, 256, 1, 1);
  } else {
    uint32_t n = TOPK, pm = 0;
    gpu_buf(1, &MH, 0), gpu_buf(2, &SHO, 0), gpu_buf(3, &MF, 0);
    gpu_buf(4, &H1, 0), gpu_buf(5, &HS, 0), gpu_bytes(6, &n, 4), gpu_bytes(7, &pm, 4), gpu_buf(8, &PARTS, 0);
    gpu_go(DIM / 4, 1, 1, 32 * TOPK, 1, 1);
  }
}

static const float *pipeline_rows(State **states, const int *tokens, const int *positions, int count) {
  if (nrows || verifying || count < 2 || count > BATCH_ROWS) die("invalid pipeline batch");
  if (opt.deepseek.experts || opt.draft || opt.deepseek.substitute > 0 || opt.deepseek.online_predictor) die("pipeline needs native MXFP4 exact routing");
  request_capacity(1);
  State *original_state = active;
  Buf *live[PIPE_BUFFERS] = {PIPE_BUFFER_LIST}, original[PIPE_BUFFERS];
  for (int k = 0; k < PIPE_BUFFERS; k++) original[k] = *live[k];
  if (K_pipe < 0) K_pipe = gpu_kernel("moe_tasks");
  if (!pipe_tasks.obj) {
    pipe_tasks = zeros((size_t)PIPE_BATCHES * BATCH_ROWS * sizeof(MoeTask));
    pipe_logits = memory_host_required((size_t)BATCH_ROWS * VOCAB * 4, 0);
    scheduler_workspace_bytes += pipe_tasks.size + (size_t)BATCH_ROWS * VOCAB * 4;
  }
  for (int r = 0; r < count; r++) {
    PipeJob *job = pipe_jobs + r;
    job->state = states[r], job->row = r, job->pos = positions[r], job->layer = 0;
    job->held_layer = -1, job->release_at = 0;
    for (int k = 0; k < PIPE_BUFFERS; k++) {
      size_t bytes = k == 14 ? TOPK * INTER * 4 : k == 15 ? VOCAB * 4 : original[k].size;
      if (job->saved[k].size == bytes) continue;
      scheduler_workspace_bytes -= job->saved[k].size;
      gpu_free(job->saved + k);
      job->saved[k] = zeros(bytes);
      scheduler_workspace_bytes += bytes;
    }
    memset(job->saved[16].ptr, 0, job->saved[16].size);
  }
  CacheStats before = *expert_stats();
  double began = now(), gpu_before = gpu_busy;
  cache_metrics_begin(experts_store, PF_DECODE, count);
  cache_adapt(experts_store);
  runahead_route_only = 1;
  route_steps++;
  pipe_stats.steps++;
  for (int r = 0; r < count; r++) {
    PipeJob *job = pipe_jobs + r;
    if (tokens[r] < 0) die("pipeline needs decoded text tokens");
    pipe_bind(job, live);
    int64_t rows[1][2][ENGRAM_ROWS];
    memcpy(history, job->state->history, sizeof history);
    engram_hash(tokens[r], job->pos, rows[0]);
    memcpy(job->state->history, history, sizeof history);
    engram_read(0, rows, 1), engram_read(1, rows, 1);
    embed_read(tokens + r, 1, 0, job->pos);
    pipe_front(job, live);
  }
  int remaining = count, batch = 0;
  while (remaining) {
    uint64_t epoch = cache_read_epoch(experts_store), completed = gpu_signal_value(), wait = UINT64_MAX;
    PipeJob *ready[BATCH_ROWS];
    int kind[BATCH_ROWS], nr = 0, pending = 0;
    for (int r = 0; r < count; r++) {
      PipeJob *job = pipe_jobs + r;
      if (job->release_at && job->release_at <= completed) {
        cache_unhold(experts_store, job->held_layer, job->experts, TOPK);
        job->held_layer = -1, job->release_at = 0;
      }
      if (job->phase == PIPE_DONE) continue;
      if ((job->phase == PIPE_FRONT || job->phase == PIPE_FINISH) && job->signal <= completed) {
        if (job->phase == PIPE_FRONT) pipe_issue(job);
        else {
          memcpy(pipe_logits + (size_t)r * VOCAB, job->saved[15].ptr, VOCAB * 4);
          job->phase = PIPE_DONE, remaining--;
          continue;
        }
      }
      if (job->phase == PIPE_READ) {
        int all = 1;
        for (int k = 0; k < TOPK; k++) all &= cache_ready(experts_store, job->layer, job->experts[k]);
        if (all) ready[nr] = job, kind[nr++] = 0;
      } else if (job->phase == PIPE_DOWN) {
        ready[nr] = job, kind[nr++] = 1;
        pending += job->signal > completed;
      } else {
        pending++;
        if (job->signal < wait) wait = job->signal;
      }
    }
    if ((uint64_t)pending > pipe_stats.max_inflight) pipe_stats.max_inflight = pending;
    if (!remaining) break;
    if (!nr) {
      double start = now();
      if (wait != UINT64_MAX) gpu_wait_signal(wait), pipe_stats.gpu_wait += now() - start;
      else {
        gpu_keep_clock(.001);
        cache_wait_progress(experts_store, epoch);
        double elapsed = now() - start;
        pipe_stats.read_wait += elapsed, expert_stats()->read_s += elapsed;
      }
      continue;
    }
    if (batch == PIPE_BATCHES) die("pipeline stage count exceeded its bound");
    size_t offset = (size_t)batch++ * BATCH_ROWS * sizeof(MoeTask);
    MoeTask *tasks = (MoeTask *)(pipe_tasks.ptr + offset);
    uint32_t groups = 0;
    int mixed_kind = 0, mixed_layer = 0;
    gpu_begin();
    for (int i = 0; i < nr; i++) {
      PipeJob *job = ready[i];
      pipe_bind(job, live);
      size_t entries = (size_t)job->layer * FEW * 256;
      if (!kind[i]) {
        cache_wait(experts_store, job->layer, job->experts, TOPK);
        const int *rt = (const int *)RT.ptr;
        fill_entries(job->layer, entries, rt, (const float *)rt + 8, NULL, 0);
      } else {
        // A first-use mapping might not be committed to the residency set
        // yet. This new encoder must declare the down stage's resources too.
        for (int k = 0; k < TOPK; k++) {
          uint32_t at[STORE_PARTS]; int fresh;
          const Buf *b = cache_get(experts_store, job->layer, job->experts[k], at, &fresh);
          if (fresh) gpu_use(b);
        }
      }
      tasks[i] = (MoeTask){ETAB.gpu + entries, XQ.gpu, MH.gpu, SHO.gpu, MF.gpu, H1.gpu, HS.gpu,
                            groups, kind[i] ? DIM / 4 : TOPK * (INTER / 32), kind[i], TOPK};
      if (nr > 1) {
        gpu_use(&ETAB);
        if (!kind[i]) gpu_use(&XQ), gpu_use_rw(&MH);
        else {
          gpu_use(&MH), gpu_use(&SHO), gpu_use(&MF), gpu_use(&H1), gpu_use_rw(&HS);
        }
      }
      groups += tasks[i].groups;
      mixed_kind |= kind[i] != kind[0];
      mixed_layer |= job->layer != ready[0]->layer;
    }
    if (nr == 1) {
      pipe_bind(ready[0], live);
      pipe_single(kind[0], ready[0]->layer, (size_t)ready[0]->layer * FEW * 256);
    } else {
      uint32_t n = nr;
      gpu_kern(K_pipe), gpu_buf(0, &pipe_tasks, offset), gpu_bytes(1, &n, 4);
      gpu_go(groups, 1, 1, 256, 1, 1);
    }
    for (int i = 0; i < nr; i++) if (kind[i] && ready[i]->layer == LAYERS - 1) {
      pipe_bind(ready[i], live);
      hc_pre_norm(1, &HS, 0, &MF, 0, &final_norm);
      head_out(1, &logits);
    }
    uint64_t finished = gpu_signal();
    gpu_commit();
    pipe_stats.batches++, pipe_stats.tasks += nr;
    pipe_stats.mixed_stages += mixed_kind, pipe_stats.mixed_layers += mixed_layer;
    for (int i = 0; i < nr; i++) {
      PipeJob *job = ready[i];
      job->signal = finished;
      // The down stage only consumes GPU data. Its up stage is already
      // earlier on this queue, and declared resource dependencies order
      // their accesses. Do not wait for a CPU-visible completion here.
      if (!kind[i]) job->phase = PIPE_DOWN;
      else {
        job->release_at = finished;
        if (++job->layer < LAYERS) pipe_front(job, live);
        else job->phase = PIPE_FINISH;
      }
    }
  }
  gpu_sync();
  for (int k = 0; k < PIPE_BUFFERS; k++) *live[k] = original[k];
  bind(original_state), memcpy(history, original_state->history, sizeof history);
  runahead_route_only = 0;
  cache_metrics_end(experts_store);
  stats_add(&decode_stats, &before);
  decode_rows += count, decode_gpu_s += gpu_busy - gpu_before, decode_s += now() - began;
  return pipe_logits;
}


const float *deepseek_forward_rows(State **st, const int *tokens, const int *pos, int m) {
  if (m < 1 || m > BATCH_ROWS) die("%d rows of sequences at once: 1 to %d", m, BATCH_ROWS);
  for (int r = 0; r < m; r++) {
    if (pos[r] >= opt.ctx) die("context full (%d tokens)", opt.ctx);
    if (pos[r] + 1 > st[r]->cap) resize(st[r], grown(st[r]->cap, pos[r] + 1), pos[r]);
    row_state[r] = st[r], row_pos[r] = pos[r];
  }
  memcpy(active->history, history, sizeof history);
  if (pipeline_enabled && m > 1) return pipeline_rows(st, tokens, pos, m);
  if (runahead_enabled && m > 1) return runahead_rows(st, tokens, pos, m);
  nrows = m;
  step(tokens, m, pos[0]);
  nrows = 0;
  if (opt.draft) draft_rows(st, pos, m);
  memcpy(history, active->history, sizeof history);
  return (const float *)logits.ptr;
}

static void save_rows(int restore) {
  gpu_sync();
  if (!active->saved) {
    if (restore) die("no saved attention state");
    active->saved = memory_host_required(saved_bytes(), 1);
  }
  char *p = active->saved;
#define ROWS(b) (restore ? memcpy((b).ptr, p, (b).size) : memcpy(p, (b).ptr, (b).size), p += (b).size)
  for (int l = 0; l < LAYERS; l++) {
    ROWS(L[l].ring);
    if (L[l].kv_source) ROWS(L[l].cstate);
  }
  for (int d = 0; opt.draft && d < DRAFTS; d++) ROWS(D[d].ring);
#undef ROWS
  if (restore) memcpy(history, p, sizeof history);
  else memcpy(p, history, sizeof history);
}

void deepseek_state_save(void) { save_rows(0); }
void deepseek_state_restore(void) { save_rows(1); }

// ---- snapshots and row copies, for the server's KV prefix cache ----

// s's rolling rows to p, or (restore) p's to s.
static void rolling(State *s, char *p, int restore) {
#define ROLL(b) (restore ? memcpy((b).ptr, p, (b).size) : memcpy(p, (b).ptr, (b).size), p += (b).size)
  for (int l = 0; l < LAYERS; l++) {
    ROLL(s->ring[l]);
    if (L[l].kv_source) ROLL(s->cstate[l]);
  }
  for (int d = 0; opt.draft && d < DRAFTS; d++) ROLL(s->dring[d]);
#undef ROLL
  int32_t *h = s == active ? history : s->history;
  if (restore) memcpy(h, p, sizeof history);
  else memcpy(p, h, sizeof history);
}

Snapshot *deepseek_snapshot(State *s, int pos, int with_logits) {
  gpu_sync();
  Snapshot *x = xmalloc(sizeof *x);
  *x = (Snapshot){.pos = pos, .extent = LAYERS - 1, .rows = memory_host_required(saved_bytes(), 1)};
  rolling(s, x->rows, 0);
  if (with_logits) x->logits = memory_host_required(VOCAB * 4, 1), memcpy(x->logits, logits.ptr, VOCAB * 4);
  return x;
}

void deepseek_snapshot_restore(State *s, const Snapshot *x) {
  if (x->extent != LAYERS - 1) die("an encoder-only snapshot cannot restore a whole state");
  gpu_sync();
  rolling(s, x->rows, 1);
}

void deepseek_snapshot_free(Snapshot *x) {
  if (x) memory_host_free(x->rows), memory_host_free(x->logits), free(x);
}

size_t deepseek_snapshot_bytes(int with_logits) { return saved_bytes() + (with_logits ? VOCAB * 4 : 0); }
int deepseek_snapshot_pos(const Snapshot *x) { return x->pos; }
int deepseek_snapshot_extent(const Snapshot *x) { return x->extent; }
const float *deepseek_snapshot_logits(const Snapshot *x) { return x->logits; }

void deepseek_state_copy(State *dst, const State *src, int positions) {
  if (dst == active || dst == src) die("the attention state in use cannot be overwritten");
  if (positions < 0 || positions > src->cap) die("invalid state copy of %d positions", positions);
  gpu_sync();
  int cap = grown(start_cap(), positions);
  if (dst->cap != cap) resize(dst, cap, 0);
  memory_host_free(dst->saved), dst->saved = NULL;  // its rewind point belonged to its old rows
  // Global rows never change before a state's position. The row that a
  // position before `positions` shares with a later one (a compressed pair,
  // the guard row) is computed again by the prefill that continues there.
  for (int l = 0; l < LAYERS; l++) {
    if (!L[l].kv_source) continue;
    memcpy(dst->comp[l].ptr, src->comp[l].ptr, rows_bytes(l, positions, 0));
    memcpy(dst->keys[l].ptr, src->keys[l].ptr, rows_bytes(l, positions, 1));
  }
}

void deepseek_seek(const int *tokens, int pos) {
  for (int s = 0; s < 3; s++) {
    int t = pos > s ? tokens[pos - 1 - s] : 0;
    history[s] = t < 0 ? -1 : ngram.map[t];  // an image position: dead (see engram_hash)
  }
}

// ---- the model interface (model.h) ----

static int end_token = -1;  // config.json's eos_token_id

static int ds_load(const char *dir, const Options *o) {
  char path[1024];
  snprintf(path, sizeof path, "%s/config.json", dir);
  char *text = read_file(path, 0);
  const Json *eos = json_get(json_parse(text), "eos_token_id");
  if (!eos || eos->type != J_NUM) die("%s: no eos_token_id", path);
  end_token = (int)eos->num;
  free(text);
  return deepseek_load(dir, o);
}

// The state the layers work on: s, bound when it is not already.
static void use(ModelState *s) {
  if ((State *)s != active) deepseek_state_use((State *)s);
}

// Before a state in use is freed or overwritten: the layers bind an empty
// state, which nothing runs on until the next use binds a real one.
static State parked;
static void park(const ModelState *s) {
  if ((const State *)s == active) deepseek_state_use(&parked);
}

static int ds_vocab(void) { return VOCAB; }
static int ds_max_rows(void) { return BATCH_ROWS; }
static int ds_max_verify(void) { return VERIFY_DRAFTS + 1; }
static int ds_draft_block(void) { return opt.draft; }
static int ds_rewind(void) { return deepseek_rewind(); }
static int ds_stop(int token) { return token == end_token; }
static int ds_prefill_rows(void) { return 0; }  // replay, CED and pruning plan the whole prompt
static ModelState *ds_state_new(void) { return (ModelState *)deepseek_state_new(); }
static void ds_state_free(ModelState *s) {
  park(s);
  deepseek_state_free((State *)s);
}
static size_t ds_state_bytes(const ModelState *s, int positions) {
  return deepseek_state_bytes((const State *)s, positions);
}
static void ds_state_copy(ModelState *dst, const ModelState *src, int positions) {
  park(dst);
  deepseek_state_copy((State *)dst, (const State *)src, positions);
}
static ModelSnapshot *ds_snapshot_take(ModelState *s, int pos, int with_logits) {
  return (ModelSnapshot *)deepseek_snapshot((State *)s, pos, with_logits);
}
static void ds_snapshot_restore(ModelState *s, const ModelSnapshot *x) {
  deepseek_snapshot_restore((State *)s, (const Snapshot *)x);
}
static void ds_snapshot_free(ModelSnapshot *x) { deepseek_snapshot_free((Snapshot *)x); }
static size_t ds_snapshot_bytes(int with_logits) { return deepseek_snapshot_bytes(with_logits); }
static const float *ds_snapshot_logits(const ModelSnapshot *x) { return deepseek_snapshot_logits((const Snapshot *)x); }

static const float *ds_prefill(ModelState *s, const int *tokens, int n, int from, int replay, int capture,
                               ModelSnapshot **captured, float *every) {
  use(s);
  if (every && replay && from) die("every position's logits of a replay");
  Snapshot *x = 0;
  all_logits = every;
  const float *lg = deepseek_prefill(tokens, n, from, replay, capture, captured ? &x : 0);
  all_logits = 0;
  // a capture of layers 0-20 only (under CED compaction) restores no whole state
  if (x && deepseek_snapshot_extent(x) != LAYERS - 1) deepseek_snapshot_free(x), x = 0;
  if (captured) *captured = (ModelSnapshot *)x;
  return lg;
}

static const float *ds_rows(ModelState *const *states, const int *tokens, const int *pos, int m) {
  if (m > 1 && !prepare_rows(m, 0)) return NULL;
  if (m == 1) {  // one state's token: the decoding path with its read-aheads
    use(states[0]);
    return deepseek_forward(tokens, 1, pos[0]);
  }
  return deepseek_forward_rows((State **)states, tokens, pos, m);
}

static const double *ds_draft(ModelState *s, int token, int pos, int *drafts, int n, Sampler *sampler) {
  use(s);
  deepseek_draft(token, pos, drafts, n);
  return NULL;
}
static const float *ds_verify(ModelState *s, const int *tokens, int m, int pos) {
  if (!prepare_rows(m, 0)) return NULL;
  use(s);
  return deepseek_verify(tokens, m, pos);
}
static void ds_accept(ModelState *s, const int *tokens, int n, int pos) {
  use(s);
  deepseek_accept(tokens, n, pos);
}
static void ds_seek(ModelState *s, const int *tokens, int pos) {
  use(s);
  deepseek_seek(tokens, pos);
}
static MemorySlot *ds_weight_slot(void) { return core_weights ? weights_slot(core_weights) : NULL; }
static const char *ds_refuse(const Options *o) { return 0; }

static void ds_counters(ModelCounters *c) {
  *c = (ModelCounters){.decode_rows = decode_rows, .prefill_rows = prefill_rows, .decode_seconds = decode_s,
                       .decode_gpu_seconds = decode_gpu_s, .prefill_layer = prefill_layer, .prefill_rows_now = prefill_n,
                       .layers = LAYERS, .experts_per_row = LAYERS * TOPK, .dense_bytes = weights_resident_bytes(core_weights) + weights_resident_bytes(vision_weights),
                       .decode = decode_stats, .prefill = prefill_stats};
  c->experts = LAYERS * EXPERTS;
  if (!experts_store) return;  // registered, never loaded
  c->cache = *expert_stats();
  c->cache_bytes = cache_bytes(experts_store); c->slot_bytes = cache_slot_bytes(experts_store);
  c->cache_slots = cache_slots(experts_store); c->mapped = cache_mapped(experts_store);
  c->cap = cache_cap(experts_store); c->max_slots = cache_max(experts_store);
  c->uncached_bytes = cache_uncached_bytes(experts_store, &c->uncached);
}
// Before the first load there is no store: nothing to mark or count.
static void ds_active(int active) {
  if (experts_store) cache_active(experts_store, active);
}
static uint64_t ds_cache_bytes(void) { return experts_store ? cache_bytes(experts_store) : 0; }

static void ds_plan(ModelPlan *p) {
  DeepSeekForwardInfo f = deepseek_forward_info();
  *p = (ModelPlan){.chunks = f.chunks, .tokens = f.tokens, .decoder_tokens = f.decoder_tokens,
                   .ced_applied = f.ced_applied, .prune_applied = f.prune_applied, .prefix_tokens = f.prefix_tokens};
}

static int ds_idle(void) {
  cache_idle(experts_store);
  return cache_slots(experts_store);
}
static void ds_finish(void) {
  deepseek_save_predictor();
  online_predictor_report(stderr);
}

// A CLI generation's report: the prompt's plan after its prefill, and per
// decoded token the expert selections, the workspace, and where decoding's
// time went.
static void ds_report(FILE *out, int at, int tokens) {
  static uint64_t rows0, unique0, workspace;
  static double phase0[7], gap0;
  static long gaps0;
  if (at == 0) {
    DeepSeekForwardInfo f = deepseek_forward_info();
    fprintf(out,
            "{\"prefill_prune\":{\"applied\":%s,\"tokens\":%d,\"decoder_tokens\":%d,\"decoder_row_layers\":%d,"
            "\"chunks\":%d,\"prefix_tokens\":%d}}\n",
            f.prune_applied ? "true" : "false", f.tokens, f.decoder_tokens, f.decoder_row_layers, f.chunks,
            f.prefix_tokens);
    rows0 = decode_rows, unique0 = expert_stats()->row_unique, workspace = deepseek_workspace_bytes();
    memcpy(phase0, phase_s, sizeof phase0);
    gap0 = gpu_gap, gaps0 = gpu_gaps;
    return;
  }
  uint64_t computed = decode_rows - rows0;
  fprintf(out, "{\"decode_experts\":{\"computed_rows\":%llu,\"selected\":%llu,\"unique_layer_requests\":%llu}}\n",
          (unsigned long long)computed, (unsigned long long)(computed * LAYERS * TOPK),
          (unsigned long long)(expert_stats()->row_unique - unique0));
  fprintf(out, "{\"workspace\":{\"resizable_after_prefill_bytes\":%llu,\"resizable_after_decode_bytes\":%llu}}\n",
          (unsigned long long)workspace, (unsigned long long)deepseek_workspace_bytes());
  double per = tokens ? 1e3 / tokens : 0;
  fprintf(out,
          "decode per token (ms): rows %.1f, waits %.1f, loads %.1f, prefetch %.1f, encode %.1f, adapt %.1f, "
          "drafting %.1f; GPU idle between command buffers %.1f (%.1f gaps over 100 us)\n",
          (phase_s[0] - phase0[0]) * per, (phase_s[1] - phase0[1]) * per, (phase_s[2] - phase0[2]) * per,
          (phase_s[3] - phase0[3]) * per, (phase_s[4] - phase0[4]) * per, (phase_s[5] - phase0[5]) * per,
          (phase_s[6] - phase0[6]) * per, (gpu_gap - gap0) * per, (double)(gpu_gaps - gaps0) / (tokens ? tokens : 1));
  deepseek_metrics_report(out);
}

static void ds_chat(char *text, size_t cap, const char *prompt, int images, int think) {
  char user[16 * 32 + 1] = "";
  for (int i = 0; i < images && i < 16; i++) strcat(user, MEDIA_MARK "\n\n");  // each image's place, then the text
  if (think)
    snprintf(text, cap,
             "<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: %d (range 1-100, the higher "
             "the value, the more thorough the reasoning)\n\n<｜User｜>%s%s<｜Assistant｜><think>",
             think, user, prompt);
  else snprintf(text, cap, "<｜begin▁of▁sentence｜><｜User｜>%s%s<｜Assistant｜></think>", user, prompt);
}

static const char *const ds_types[] = {"deepseek_v41", 0};

static int ds_prepare_image(const char *dir, Media *im, const void *data, size_t len, const char **error) {
  return image_open(im, data, len, error);
}

const ModelOps deepseek_ops = {
    .family = "DeepSeek V4.1", .model_types = ds_types, .media_token = "<｜deepseek_image｜>", .refuse = ds_refuse,
    .load = ds_load, .weight_slot = ds_weight_slot, .active = ds_active, .counters = ds_counters, .cache_bytes = ds_cache_bytes, .plan = ds_plan, .routes = deepseek_routes, .vocab = ds_vocab,
    .max_rows = ds_max_rows, .max_verify = ds_max_verify, .draft_block = ds_draft_block, .rewind = ds_rewind,
    .prefill_rows = ds_prefill_rows,
    .stop = ds_stop, .state_new = ds_state_new, .state_free = ds_state_free, .state_bytes = ds_state_bytes,
    .state_copy = ds_state_copy, .snapshot_take = ds_snapshot_take, .snapshot_restore = ds_snapshot_restore,
    .snapshot_free = ds_snapshot_free, .snapshot_bytes = ds_snapshot_bytes, .snapshot_logits = ds_snapshot_logits,
    .prefill = ds_prefill, .replay_rows = deepseek_replay_rows, .replay_start = deepseek_replay_start,
    .rows = ds_rows, .draft = ds_draft, .verify = ds_verify, .accept = ds_accept, .seek = ds_seek, .idle = ds_idle,
    .finish = ds_finish, .report = ds_report, .prepare_media = ds_prepare_image, .encode_media = deepseek_image, .media = deepseek_images, .chat = ds_chat,
    .bench = deepseek_bench};
