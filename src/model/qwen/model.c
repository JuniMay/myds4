// The Qwen3.8 (qwen3_5) text model: Qwen3.8-27B in NVIDIA's NVFP4 build (see
// qwen.h and docs/design.md). A forward pass runs up to QWEN_ROWS token rows
// in one command buffer: per layer an RMSNorm, the mixer (a gated DeltaNet
// in three layers of four, gated full attention in the fourth), the
// residual, an RMSNorm and the SwiGLU MLP. The math follows the reference
// implementation (MLX's qwen3_5).
#include "model.h"
#include "vision.h"
#include "../weights.h"
#include "../linear.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "../../engine/myds4.h"
#include "../model.h"

#define ROWS QWEN_ROWS
#define PF QWEN_PREFILL
#define BLK 256  // positions per attention block (gqa.metal's BLK)

typedef Linear Mat;

typedef struct {
  int linear, index;  // DeltaNet or attention layer, and its index among those
  Buf in_norm, post_norm;  // FP32, the checkpoint's zero-centered weights plus 1
  Mat gate, up, down;
  // gated DeltaNet
  Mat qkv, z, a, b, out;
  Buf conv, A, dt_bias, norm;  // conv: [channels][4]; A = exp(A_log); norm: [value_dim], as stored
  // full attention
  Mat q, k, v, o;
  Buf q_norm, k_norm;  // plus 1
} Layer;

static TensorIndex *checkpoint;
static WeightSet *dense;
static QwenVision *vision;
static const Media *images;
static int image_count;
static struct {
  int layers, linear, attention, hidden, inter, vocab, nstop, stop[8];
  int heads, kv_heads, head_dim, rot, rope_h, rope_w;
  int key_heads, value_heads, key_dim, value_dim, channels;
  float eps, theta;
} c;

static Layer *L;
static Buf embed_w, final_norm;
static Mat head;
// The MTP head (mtp.*): pre-fc norms, fc, one full-attention layer (its keys
// after the model's, as attention layer c.attention), its norm; the
// embedding and head are the model's.
static int mtp, state_copies = 2;
static Layer M;
static Mat mtp_fc;
static Buf mtp_norm_e, mtp_norm_h, mtp_norm;
// Activations of up to PF rows; XB: a matrix's input as BF16 (prompt passes);
// SEQ: the rows' sequences.
static Buf TOK, EMB, X, XN, R, QKV, CO, Z, A_, B_, G, BETA, GO, GN, QG, KK, VV, Q, KR, AO, GT, UP, HH, LOGITS, XB, SEQ;
// The MTP head's rows: their hiddens, next tokens and sequences; its fc's input.
static Buf HB, TOKM, EMBM, SEQM, EH;
static Buf QH;  // a prompt pass's queries as FP16, by head
static Buf PART, ML;  // attention blocks' partial sums, for states of up to `blocks` * BLK positions
static int blocks;
static int workspace_rows;
static int K_gate_up[3], K_gate_up3, K_add_norm, K_concat, K_q_half, K_attend_prompt, K_embed, K_rms_norm, K_l2_norm, K_conv, K_gdn_gates, K_gdn, K_qk_rope, K_kv_store, K_attend_block,
    K_attend_combine, K_swiglu, K_add;

struct QwenState {
  Buf gdn[ROWS], conv[ROWS];  // current state and verified-prefix checkpoints
  Buf kv;               // keys then values: [attention layer][cap][kv head][head dim], FP16
  Buf hid[ROWS];         // the corresponding hidden vectors for the MTP head
  int cur, pos, cap, copies, verify_origin;
  int rope_delta;
  int saved_pos[ROWS], saved_rope[ROWS];
  int mtp_next; // first position whose draft-head key is not committed

};

struct QwenSnap {
  Buf gdn, conv, hid;
  int pos, rope_delta;
};

// A row's sequence, as deltanet.metal's Seq.
typedef struct {
  uint64_t gdn, gdn_save, conv, conv_save, k, v;
  uint32_t pos, cap;
  int32_t rope_t, rope_h, rope_w, pad;
} Seq;
static_assert(sizeof(Seq) == 72);

static uint64_t image_embedding(int pos) {
  return media_embedding(images, image_count, pos);
}

static void positions(Seq *row, const QwenState *state, int pos, int prompt) {
  qwen_image_positions(images, image_count, pos, state->rope_delta, prompt, &row->rope_t);
}

// The engine's buffers, so that an idle model can leave the GPU's residency
// set (qwen_reside) and come back before its next pass.
static Buf *owned;
static int nowned, owned_room, resident = 1;

static Buf own(Buf b) {
  if (nowned == owned_room) {
    owned_room = owned_room ? 2 * owned_room : 4096;
    owned = realloc(owned, (size_t)owned_room * sizeof *owned);
    if (!owned) die("out of memory");
  }
  owned[nowned++] = b;
  return b;
}

static void unown(Buf *b) {
  for (int i = 0; i < nowned; i++)
    if (owned[i].obj == b->obj) {
      owned[i] = owned[--nowned];
      break;
    }
  gpu_free(b);
}

void qwen_reside(int on) {
  if (on == resident) return;
  gpu_sync();
  for (int i = 0; i < nowned; i++) gpu_resident(&owned[i], on);
  weights_reside(dense, on);
  qwen_vision_reside(vision, on);
  gpu_residency_commit();
  resident = on;
}

static double num(const Json *o, const char *key) {
  const Json *v = json_get(o, key);
  if (!v || v->type != J_NUM) die("config.json: no number %s", key);
  return v->num;
}

static void upload(Buf *target, Tensor tensor) { weights_tensor(dense, target, tensor, 0); }

static void vec(Buf *target, const char *name, float add) {
  weights_vector(dense, target, st_get(checkpoint, "%s", name), add, WEIGHT_ADD);
}

static void mat(Mat *matrix, const char *stem) { linear_load(dense, matrix, checkpoint, stem); }

static Buf zeros(size_t n) {
  Buf b = own(gpu_alloc(n));
  memset(b.ptr, 0, n);
  return b;
}

static Buf kv_zeros(size_t bytes) {
  Buf buffer = own(gpu_alloc_kv(bytes));
  memset(buffer.ptr, 0, bytes);
  return buffer;
}

static void grow_blocks(int cap) {
  int need = (cap + BLK - 1) / BLK;
  if (need <= blocks) return;
  gpu_sync();
  unown(&PART), unown(&ML);
  blocks = need;
  PART = zeros((size_t)ROWS * c.heads * blocks * c.head_dim * 4);
  ML = zeros((size_t)ROWS * c.heads * blocks * 8);
}

static uint64_t workspace(int rows, int position, int allocate) {
  uint64_t bytes = 0;
  int capacity = (rows + 127) / 128 * 128;
  if (capacity < 128) capacity = 128;
  int keys = 4096;
  while (keys < position) keys *= 2;
  int parts = keys / BLK > blocks ? keys / BLK : blocks;
#define W(buffer, amount) do { size_t size = (amount); bytes += gpu_allocation_size(size); if (allocate && buffer.size != size) { unown(&buffer); buffer = zeros(size); } } while (0)
  size_t r = capacity * 4, vd = (size_t)c.value_heads * c.value_dim;
  W(TOK, capacity * 4); W(EMB, capacity * 8); W(EMBM, capacity * 8); W(SEQ, capacity * sizeof(Seq));
  int widest = c.inter > 2 * c.hidden ? c.inter : 2 * c.hidden;  // 2 * hidden: the MTP head's fc
  if (vd > (size_t)widest) widest = (int)vd;
  W(XB, (size_t)capacity * widest * 2);
  W(X, r * c.hidden); W(XN, r * c.hidden); W(R, r * c.hidden);
  W(QKV, r * c.channels); W(CO, r * c.channels); W(Z, r * vd); W(GO, r * vd); W(GN, r * vd);
  W(A_, r * c.value_heads); W(B_, r * c.value_heads); W(G, r * c.value_heads);
  W(BETA, r * c.value_heads);
  W(QG, r * 2 * c.heads * c.head_dim); W(KK, r * c.kv_heads * c.head_dim);
  W(VV, r * c.kv_heads * c.head_dim); W(Q, r * c.heads * c.head_dim); W(KR, r * c.kv_heads * c.head_dim);
  W(AO, r * c.heads * c.head_dim);
  W(GT, r * c.inter); W(UP, r * c.inter); W(HH, r * c.inter);
  W(LOGITS, (size_t)ROWS * 4 * c.vocab);
  W(HB, r * c.hidden); W(TOKM, capacity * 4); W(SEQM, capacity * sizeof(Seq)); W(EH, r * 2 * c.hidden);
  W(QH, (size_t)c.heads * capacity * c.head_dim * 2);
  W(PART, (size_t)ROWS * c.heads * parts * c.head_dim * 4);
  W(ML, (size_t)ROWS * c.heads * parts * 8);
  if (allocate) { blocks = parts; workspace_rows = capacity; }
#undef W
  return bytes;
}

static void prepare(const char *dir, int drafts) {
  char path[4096];
  snprintf(path, sizeof path, "%s/config.json", dir);
  char *text = read_file(path, 0);
  Json *root = json_parse(text);
  free(text);
  const Json *t = json_get(root, "text_config");
  if (!t) t = root;
  const Json *mt = json_get(t, "model_type");
  if (!mt || mt->type != J_STR || strcmp(mt->s, "qwen3_5_text")) die("%s: not a qwen3_5 text model", path);
  c.layers = (int)num(t, "num_hidden_layers"), c.hidden = (int)num(t, "hidden_size");
  c.inter = (int)num(t, "intermediate_size"), c.vocab = (int)num(t, "vocab_size");
  c.eps = (float)num(t, "rms_norm_eps");
  c.heads = (int)num(t, "num_attention_heads"), c.kv_heads = (int)num(t, "num_key_value_heads");
  c.head_dim = (int)num(t, "head_dim");
  c.rot = (int)(c.head_dim * num(t, "partial_rotary_factor"));
  const Json *rope = json_get(t, "rope_parameters");
  c.theta = (float)num(rope ? rope : t, "rope_theta");
  const Json *sections = json_get(rope, "mrope_section");
  c.rope_h = sections && sections->n == 3 ? (int)sections->kid[1].num : 11;
  c.rope_w = sections && sections->n == 3 ? (int)sections->kid[2].num : 10;
  c.key_heads = (int)num(t, "linear_num_key_heads"), c.value_heads = (int)num(t, "linear_num_value_heads");
  c.key_dim = (int)num(t, "linear_key_head_dim"), c.value_dim = (int)num(t, "linear_value_head_dim");
  c.channels = 2 * c.key_heads * c.key_dim + c.value_heads * c.value_dim;
  // The kernels' shapes: 128-value DeltaNet heads, at most 8 query heads per
  // key head and 256-value attention heads (one thread a value).
  if ((int)num(t, "linear_conv_kernel_dim") != 4 || c.key_dim != 128 || c.value_dim != 128 ||
      c.value_heads % c.key_heads || c.head_dim != 256 || c.heads % c.kv_heads || c.heads / c.kv_heads > 8)
    die("%s: a qwen3_5 shape this engine does not take", path);
  snprintf(path, sizeof path, "%s/generation_config.json", dir);
  Json *generation = NULL;
  if (!access(path, R_OK)) {
    text = read_file(path, 0);
    generation = json_parse(text);
    free(text);
  }
  const Json *eos = json_get(generation, "eos_token_id");
  if (!eos) eos = json_get(t, "eos_token_id");
  if (!eos) die("%s: no eos_token_id", dir);
  for (int i = 0; i < (eos->type == J_ARR ? eos->n : 1) && c.nstop < 8; i++)
    c.stop[c.nstop++] = (int)(eos->type == J_ARR ? eos->kid[i].num : eos->num);
  json_free(generation);
  const Json *types = json_get(t, "layer_types");
  if (!types || types->type != J_ARR || types->n != c.layers) die("%s: no layer_types", path);

  gpu_start();
  linear_init();
  K_gate_up[0] = gpu_kernel("gate_up_nvfp4"), K_gate_up[1] = gpu_kernel("gate_up_fp8"), K_gate_up[2] = gpu_kernel("gate_up_bf16");
  K_gate_up3 = gpu_kernel("gate_up3_nvfp4");
  K_add_norm = gpu_kernel("add_norm");
  K_concat = gpu_kernel("concat");
  K_q_half = gpu_kernel("gqa_q_half"), K_attend_prompt = gpu_kernel("gqa_prompt");
  K_embed = gpu_kernel("embed_rows"), K_rms_norm = gpu_kernel("rms_norm"), K_l2_norm = gpu_kernel("l2_norm");
  K_conv = gpu_kernel("conv"), K_gdn_gates = gpu_kernel("gdn_gates"), K_gdn = gpu_kernel("gdn");
  K_qk_rope = gpu_kernel("qk_rope"), K_kv_store = gpu_kernel("kv_store");
  K_attend_block = gpu_kernel("gqa_block"), K_attend_combine = gpu_kernel("gqa_combine");
  K_swiglu = gpu_kernel("silu_mul"), K_add = gpu_kernel("add");

  checkpoint = st_open(dir);
  dense = weights_new(memory_current()->account, "Qwen dense weights");
  vision = qwen_vision_open(checkpoint, json_get(root, "vision_config"));
  const char *m = "model.language_model";
  char p[256], n[256];
  upload(&embed_w, st_get(checkpoint, "%s.embed_tokens.weight", m));
  snprintf(n, sizeof n, "%s.norm.weight", m);
  vec(&final_norm, n, 1);
  mat(&head, "lm_head");
  L = calloc((size_t)c.layers, sizeof *L);
  for (int l = 0; l < c.layers; l++) {
    Layer *y = &L[l];
    const char *kind = types->kid[l].s;
    y->linear = !strcmp(kind, "linear_attention");
    if (!y->linear && strcmp(kind, "full_attention")) die("layer %d: unknown type %s", l, kind);
    y->index = y->linear ? c.linear++ : c.attention++;
    snprintf(n, sizeof n, "%s.layers.%d.input_layernorm.weight", m, l), vec(&y->in_norm, n, 1);
    snprintf(n, sizeof n, "%s.layers.%d.post_attention_layernorm.weight", m, l), vec(&y->post_norm, n, 1);
    snprintf(p, sizeof p, "%s.layers.%d.mlp.gate_proj", m, l), mat(&y->gate, p);
    snprintf(p, sizeof p, "%s.layers.%d.mlp.up_proj", m, l), mat(&y->up, p);
    snprintf(p, sizeof p, "%s.layers.%d.mlp.down_proj", m, l), mat(&y->down, p);
    if (y->linear) {
      snprintf(p, sizeof p, "%s.layers.%d.linear_attn.in_proj_qkv", m, l), mat(&y->qkv, p);
      snprintf(p, sizeof p, "%s.layers.%d.linear_attn.in_proj_z", m, l), mat(&y->z, p);
      snprintf(p, sizeof p, "%s.layers.%d.linear_attn.in_proj_a", m, l), mat(&y->a, p);
      snprintf(p, sizeof p, "%s.layers.%d.linear_attn.in_proj_b", m, l), mat(&y->b, p);
      snprintf(p, sizeof p, "%s.layers.%d.linear_attn.out_proj", m, l), mat(&y->out, p);
      snprintf(n, sizeof n, "%s.layers.%d.linear_attn.conv1d.weight", m, l), vec(&y->conv, n, 0);
      snprintf(n, sizeof n, "%s.layers.%d.linear_attn.A_log", m, l);
      weights_vector(dense, &y->A, st_get(checkpoint, "%s", n), 0, WEIGHT_ADD | WEIGHT_EXP);
      snprintf(n, sizeof n, "%s.layers.%d.linear_attn.dt_bias", m, l), vec(&y->dt_bias, n, 0);
      snprintf(n, sizeof n, "%s.layers.%d.linear_attn.norm.weight", m, l), vec(&y->norm, n, 0);
      if (y->qkv.rows != c.channels || y->z.rows != c.value_heads * c.value_dim || y->a.rows != c.value_heads)
        die("layer %d: DeltaNet projections of unexpected sizes", l);
    } else {
      snprintf(p, sizeof p, "%s.layers.%d.self_attn.q_proj", m, l), mat(&y->q, p);
      snprintf(p, sizeof p, "%s.layers.%d.self_attn.k_proj", m, l), mat(&y->k, p);
      snprintf(p, sizeof p, "%s.layers.%d.self_attn.v_proj", m, l), mat(&y->v, p);
      snprintf(p, sizeof p, "%s.layers.%d.self_attn.o_proj", m, l), mat(&y->o, p);
      snprintf(n, sizeof n, "%s.layers.%d.self_attn.q_norm.weight", m, l), vec(&y->q_norm, n, 1);
      snprintf(n, sizeof n, "%s.layers.%d.self_attn.k_norm.weight", m, l), vec(&y->k_norm, n, 1);
      if (y->q.rows != 2 * c.heads * c.head_dim || y->k.rows != c.kv_heads * c.head_dim)
        die("layer %d: attention projections of unexpected sizes", l);
    }
  }
  mtp = drafts && st_has(checkpoint, "mtp.fc.weight");
  if (mtp) {
    mat(&mtp_fc, "mtp.fc");
    vec(&mtp_norm_e, "mtp.pre_fc_norm_embedding.weight", 1), vec(&mtp_norm_h, "mtp.pre_fc_norm_hidden.weight", 1);
    vec(&mtp_norm, "mtp.norm.weight", 1);
    M = (Layer){.linear = 0, .index = c.attention};
    vec(&M.in_norm, "mtp.layers.0.input_layernorm.weight", 1);
    vec(&M.post_norm, "mtp.layers.0.post_attention_layernorm.weight", 1);
    mat(&M.q, "mtp.layers.0.self_attn.q_proj"), mat(&M.k, "mtp.layers.0.self_attn.k_proj");
    mat(&M.v, "mtp.layers.0.self_attn.v_proj"), mat(&M.o, "mtp.layers.0.self_attn.o_proj");
    vec(&M.q_norm, "mtp.layers.0.self_attn.q_norm.weight", 1), vec(&M.k_norm, "mtp.layers.0.self_attn.k_norm.weight", 1);
    mat(&M.gate, "mtp.layers.0.mlp.gate_proj"), mat(&M.up, "mtp.layers.0.mlp.up_proj");
    mat(&M.down, "mtp.layers.0.mlp.down_proj");
    if (mtp_fc.cols != 2 * c.hidden || mtp_fc.rows != c.hidden) die("mtp.fc: %d x %d", mtp_fc.rows, mtp_fc.cols);
  }
  json_free(root);
}

static uint64_t workspace_bytes(void) {
  const Buf *buffers[] = {&TOK, &EMB, &EMBM, &SEQ, &XB, &X, &XN, &R, &QKV, &CO, &Z, &GO, &GN,
    &A_, &B_, &G, &BETA, &QG, &KK, &VV, &Q, &KR, &AO, &GT, &UP, &HH, &LOGITS,
    &HB, &TOKM, &SEQM, &EH, &QH, &PART, &ML};
  uint64_t bytes = 0;
  for (size_t i = 0; i < sizeof buffers / sizeof *buffers; i++) bytes += buffers[i]->allocated;
  return bytes;
}

static int prepare_rows(int requested, int position) {
  MemorySlot *work = memory_current(), *core = weights_slot(dense);
  uint64_t fixed = work->payload - workspace_bytes();
  int rows = requested < PF ? requested : PF;
  for (;;) {
    uint64_t required = weights_required(dense) + fixed + workspace(rows, position, 0);
    MemoryAdmission mode = rows > 1 ? MEMORY_WORKSPACE : MEMORY_REQUIRED;
    if (memory_minimum_mode(core->account, required, mode)) {
      gpu_sync();
      workspace(rows, position, 1);
      return rows;
    }
    if (rows == 1) return 0;
    rows = (rows + 1) / 2;
  }
}

int qwen_load(const char *dir, int drafts) {
  double began = now();
  if (!checkpoint) prepare(dir, drafts);
  state_copies = mtp && drafts > 1 ? drafts + 1 : 2;
  MemorySlot *core = weights_slot(dense), *work = memory_current();
  memory_pin(core, 1);
  uint64_t required = weights_required(dense) + work->payload - workspace_bytes() + workspace(128, 0, 0);
  if (!memory_minimum(core->account, required)) { memory_pin(core, 0); return 0; }
  gpu_sync();
  workspace(128, 0, 1);
  if (!weights_load(dense)) { memory_pin(core, 0); return 0; }
  fprintf(stderr, "qwen: %d layers (%d DeltaNet, %d attention), %d hidden%s, loaded in %.1f s\n",
          c.layers, c.linear, c.attention, c.hidden, mtp ? ", MTP head" : "", now() - began);
  return 1;
}

int qwen_vocab(void) { return c.vocab; }

int qwen_stop(int token) {
  for (int i = 0; i < c.nstop; i++)
    if (c.stop[i] == token) return 1;
  return 0;
}

// ---- states ----

static size_t gdn_bytes(void) { return (size_t)c.linear * c.value_heads * c.value_dim * c.key_dim * 4; }
static size_t conv_bytes(void) { return (size_t)c.linear * 3 * c.channels * 4; }
static int kv_layers(void) { return c.attention + mtp; }
static size_t kv_bytes(int cap) { return (size_t)2 * kv_layers() * cap * c.kv_heads * c.head_dim * 2; }
static size_t hid_bytes(void) { return (size_t)c.hidden * 4; }

uint64_t qwen_state_bytes(int positions) {
  return state_copies * (gdn_bytes() + conv_bytes() + hid_bytes()) + kv_bytes(positions);
}
uint64_t qwen_snap_bytes(void) { return gdn_bytes() + conv_bytes() + hid_bytes(); }
int qwen_mtp_available(void) { return mtp; }

QwenState *qwen_state_new(void) {
  qwen_reside(1);
  QwenState *s = calloc(1, sizeof *s);
  s->copies = state_copies;
  for (int i = 0; i < s->copies; i++)
    s->gdn[i] = kv_zeros(gdn_bytes()), s->conv[i] = kv_zeros(conv_bytes()), s->hid[i] = kv_zeros(hid_bytes());
  s->cap = 1024;
  s->kv = own(gpu_alloc_kv(kv_bytes(s->cap)));
  for (int i = 0; i < s->copies; i++) s->saved_pos[i] = -1;
  grow_blocks(s->cap);
  return s;
}

void qwen_state_free(QwenState *s) {
  if (!s) return;
  gpu_sync();
  for (int i = 0; i < s->copies; i++) unown(&s->gdn[i]), unown(&s->conv[i]), unown(&s->hid[i]);
  unown(&s->kv);
  free(s);
}

int qwen_state_pos(const QwenState *s) { return s->pos; }
const float *qwen_state_hidden(const QwenState *s) { return (const float *)s->hid[s->cur].ptr; }

// Copies the keys and values of positions [0, n) of every attention layer
// between caches of capacities `from` and `to`.
static void kv_rows(uint8_t *dst, int to, const uint8_t *src, int from, int n) {
  size_t row = (size_t)c.kv_heads * c.head_dim * 2;
  for (int l = 0; l < 2 * kv_layers(); l++) memcpy(dst + l * to * row, src + l * from * row, n * row);
}

void qwen_state_reserve(QwenState *s, int positions) {
  qwen_reside(1);
  if (positions <= s->cap) return;
  int cap = s->cap;
  while (cap < positions) cap *= 2;
  gpu_sync();
  Buf kv = own(gpu_alloc_kv(kv_bytes(cap)));
  kv_rows(kv.ptr, cap, s->kv.ptr, s->cap, s->pos);
  unown(&s->kv);
  s->kv = kv, s->cap = cap;
  grow_blocks(cap);
}

void qwen_state_copy(QwenState *dst, const QwenState *src, int positions) {
  qwen_reside(1);
  if (dst == src) return;
  if (positions < 0 || positions > src->pos) die("qwen: a copy of %d positions from a state of %d", positions, src->pos);
  qwen_state_reserve(dst, positions > 0 ? positions : 1);
  gpu_sync();
  memcpy(dst->gdn[dst->cur].ptr, src->gdn[src->cur].ptr, gdn_bytes());
  memcpy(dst->conv[dst->cur].ptr, src->conv[src->cur].ptr, conv_bytes());
  memcpy(dst->hid[dst->cur].ptr, src->hid[src->cur].ptr, hid_bytes());
  kv_rows(dst->kv.ptr, dst->cap, src->kv.ptr, src->cap, positions);
  dst->pos = positions;
  for (int i = 0; i < dst->copies; i++) dst->saved_pos[i] = -1;
  dst->rope_delta = src->rope_delta;
  dst->mtp_next = positions > 0 ? positions - 1 : 0;  // the last key waits for the next token
}

void qwen_state_back(QwenState *s, int checkpoint) {
  s->cur = checkpoint;
  s->pos = s->saved_pos[checkpoint];
  s->rope_delta = s->saved_rope[checkpoint];
}

QwenSnap *qwen_snap_take(const QwenState *s) {
  qwen_reside(1);
  QwenSnap *n = calloc(1, sizeof *n);
  n->gdn = own(gpu_alloc_kv(gdn_bytes())), n->conv = own(gpu_alloc_kv(conv_bytes()));
  n->hid = own(gpu_alloc_kv(hid_bytes())), n->pos = s->pos, n->rope_delta = s->rope_delta;
  gpu_sync();
  memcpy(n->gdn.ptr, s->gdn[s->cur].ptr, gdn_bytes());
  memcpy(n->conv.ptr, s->conv[s->cur].ptr, conv_bytes());
  memcpy(n->hid.ptr, s->hid[s->cur].ptr, hid_bytes());
  return n;
}

void qwen_snap_restore(QwenState *s, const QwenSnap *n) {
  qwen_reside(1);
  if (n->pos > s->cap) die("qwen: a snapshot past its state's keys");
  gpu_sync();
  memcpy(s->gdn[s->cur].ptr, n->gdn.ptr, gdn_bytes());
  memcpy(s->conv[s->cur].ptr, n->conv.ptr, conv_bytes());
  memcpy(s->hid[s->cur].ptr, n->hid.ptr, hid_bytes());
  s->pos = n->pos;
  for (int i = 0; i < s->copies; i++) s->saved_pos[i] = -1;
  s->rope_delta = n->rope_delta;
  s->mtp_next = n->pos > 0 ? n->pos - 1 : 0;
}

int qwen_snap_pos(const QwenSnap *n) { return n->pos; }

void qwen_snap_free(QwenSnap *n) {
  if (!n) return;
  gpu_sync();
  unown(&n->gdn), unown(&n->conv), unown(&n->hid);
  free(n);
}

// ---- the forward pass ----

typedef struct { uint32_t rows, cols, n; float scale; } Mv;
typedef struct { uint32_t n; float eps, scale; uint32_t gate; } Norm;
typedef struct { uint32_t n, stride; float eps, scale; } L2p;
typedef struct { uint32_t channels, n, layer, pad; } Convp;
typedef struct { uint32_t n, heads, ratio, layer, channels, keys, pad0, pad1; } Gdnp;
typedef struct { uint32_t heads, dim, rot, stride; float theta, eps; uint32_t section_h, section_w; } Rope;
typedef struct { uint32_t n, width, layer, pad; } Kvp;
typedef struct { uint32_t heads, kv, dim, layer; float scale; uint32_t blocks, pad0, pad1; } Att;
typedef struct { uint32_t heads, kv, n, p0, layer, cap, rows, pad; float scale; uint32_t pad0, pad1, pad2; } AttPrompt;


typedef struct { uint32_t m, n, k, pad; float scale; uint32_t pad0, pad1, pad2; } MMp;

static Seq *seq;      // the pass's rows' sequences: SEQ's, or SEQM's for the MTP head
static Buf *seqbuf;  // their buffer
static int nrows, prompt_pass;

// y (rows of m->rows values) = m x for the n rows of x: in a prompt pass on
// the matrix units, else with the row kernels.
static void product(const Mat *m, const Buf *x, size_t xo, const Buf *y, size_t yo, int n, int staged) {
  linear_product(m, x, xo, y, yo, n, prompt_pass, &XB, staged, 0);
}

static void mv(const Mat *m, const Buf *x, size_t xo, const Buf *y, size_t yo, int n) {
  product(m, x, xo, y, yo, n, 0);
}

static void norm(const Buf *x, size_t xo, const Buf *w, const Buf *y, size_t yo, int rows, int n, const Buf *gate) {
  Norm p = {(uint32_t)n, c.eps, 1, gate != 0};
  gpu_kern(K_rms_norm);
  gpu_buf(0, x, xo), gpu_buf(1, w, 0), gpu_buf(2, y, yo), gpu_bytes(3, &p, sizeof p), gpu_buf(4, gate ? gate : x, 0);
  gpu_go((uint32_t)rows, 1, 1, n >= 1024 ? 1024 : 128, 1, 1);
}

static void elementwise(int k, const Buf *a, const Buf *b, const Buf *out, uint32_t n) {
  gpu_kern(k);
  gpu_buf(0, a, 0), gpu_buf(1, b, 0);
  if (out) gpu_buf(2, out, 0), gpu_bytes(3, &n, 4);
  else gpu_bytes(2, &n, 4);
  gpu_go((n + 255) / 256, 1, 1, 256, 1, 1);
}

// x += r for n rows of the hidden, then y = RMSNorm(x) * w.
static void add_norm(const Buf *w, int n) {
  Norm p = {(uint32_t)c.hidden, c.eps, 1, 0};
  gpu_kern(K_add_norm);
  gpu_buf(0, &X, 0), gpu_buf(1, &R, 0), gpu_buf(2, w, 0), gpu_buf(3, &XN, 0), gpu_bytes(4, &p, sizeof p);
  gpu_go((uint32_t)n, 1, 1, 1024, 1, 1);
}

// HH = silu(gate XN) * (up XN): in a decoding pass in one kernel, in a
// prompt pass the two products on the matrix units and then swiglu.
static void gate_up(const Layer *y, int n) {
  if (prompt_pass || y->gate.kind != y->up.kind || y->gate.rows != y->up.rows || y->gate.cols != y->up.cols) {
    mv(&y->gate, &XN, 0, &GT, 0, n), product(&y->up, &XN, 0, &UP, 0, n, 1);
    elementwise(K_swiglu, &GT, &UP, &HH, (uint32_t)(c.inter * n));
    return;
  }
  const Mat *g = &y->gate, *u = &y->up;
  int f = g->kind == 'n' ? 0 : g->kind == '8' ? 1 : 2;
  Mv p = {(uint32_t)g->rows, (uint32_t)g->cols, (uint32_t)n, g->scale};
  gpu_kern(n == 3 && f == 0 ? K_gate_up3 : K_gate_up[f]);
  gpu_buf(0, &g->w, 0), gpu_buf(1, g->kind == 'n' ? &g->s : &g->w, 0);
  gpu_buf(5, &u->w, 0), gpu_buf(6, u->kind == 'n' ? &u->s : &u->w, 0);
  gpu_buf(2, &XN, 0), gpu_buf(3, &HH, 0), gpu_bytes(4, &p, sizeof p), gpu_bytes(7, &u->scale, 4);
  gpu_go((uint32_t)(g->rows + 7) / 8, 1, 1, 256, 1, 1);
}


static void deltanet(const Layer *y) {
  int n = nrows;
  uint32_t keys = (uint32_t)(c.key_heads * c.key_dim);
  mv(&y->qkv, &XN, 0, &QKV, 0, n), product(&y->z, &XN, 0, &Z, 0, n, 1);
  product(&y->a, &XN, 0, &A_, 0, n, 1), product(&y->b, &XN, 0, &B_, 0, n, 1);
  Convp cp = {(uint32_t)c.channels, (uint32_t)n, (uint32_t)y->index, 0};
  gpu_kern(K_conv);
  gpu_buf(0, &QKV, 0), gpu_buf(1, &CO, 0), gpu_buf(2, &y->conv, 0), gpu_buf(3, seqbuf, 0);
  gpu_bytes(4, &cp, sizeof cp);
  gpu_go((uint32_t)(c.channels + 255) / 256, 1, 1, 256, 1, 1);
  L2p qn = {(uint32_t)c.key_dim, (uint32_t)c.channels, 1e-6f, 1.0f / sqrtf((float)c.key_dim)};
  L2p kn = {(uint32_t)c.key_dim, (uint32_t)c.channels, 1e-6f, 1.0f};
  gpu_kern(K_l2_norm);
  gpu_buf(0, &CO, 0), gpu_bytes(1, &qn, sizeof qn);
  gpu_go((uint32_t)c.key_heads, (uint32_t)n, 1, 128, 1, 1);
  gpu_buf(0, &CO, keys * 4), gpu_bytes(1, &kn, sizeof kn);
  gpu_go((uint32_t)c.key_heads, (uint32_t)n, 1, 128, 1, 1);
  uint32_t gp[2] = {(uint32_t)c.value_heads, (uint32_t)n};
  gpu_kern(K_gdn_gates);
  gpu_buf(0, &A_, 0), gpu_buf(1, &B_, 0), gpu_buf(2, &y->A, 0), gpu_buf(3, &y->dt_bias, 0), gpu_buf(4, &G, 0);
  gpu_buf(5, &BETA, 0), gpu_bytes(6, gp, sizeof gp);
  gpu_go((gp[0] * gp[1] + 63) / 64, 1, 1, 64, 1, 1);
  Gdnp dp = {(uint32_t)n, (uint32_t)c.value_heads, (uint32_t)(c.value_heads / c.key_heads), (uint32_t)y->index,
             (uint32_t)c.channels, keys, 0, 0};
  gpu_kern(K_gdn);
  gpu_buf(0, &CO, 0), gpu_buf(1, &G, 0), gpu_buf(2, &BETA, 0), gpu_buf(3, seqbuf, 0);
  gpu_buf(4, &GO, 0), gpu_bytes(5, &dp, sizeof dp);
  gpu_go((uint32_t)c.value_heads, (uint32_t)c.value_dim / 4, 1, 128, 1, 1);
  norm(&GO, 0, &y->norm, &GN, 0, n * c.value_heads, c.value_dim, &Z);
  mv(&y->out, &GN, 0, &R, 0, n);
}

// Whether the pass's rows are one sequence's consecutive positions (a prompt
// pass of one state, or the MTP head's rows of one).
static int one_sequence(int n) {
  for (int i = 1; i < n; i++)
    if (seq[i].k != seq[0].k || seq[i].pos != seq[0].pos + (uint32_t)i) return 0;
  return 1;
}

static void attention(const Layer *y) {
  int n = nrows;
  mv(&y->q, &XN, 0, &QG, 0, n), product(&y->k, &XN, 0, &KK, 0, n, 1), product(&y->v, &XN, 0, &VV, 0, n, 1);
  Rope qr = {(uint32_t)c.heads, (uint32_t)c.head_dim, (uint32_t)c.rot, (uint32_t)(2 * c.head_dim), c.theta, c.eps, (uint32_t)c.rope_h, (uint32_t)c.rope_w};
  Rope kr = {(uint32_t)c.kv_heads, (uint32_t)c.head_dim, (uint32_t)c.rot, (uint32_t)c.head_dim, c.theta, c.eps, (uint32_t)c.rope_h, (uint32_t)c.rope_w};
  gpu_kern(K_qk_rope);
  gpu_buf(0, &QG, 0), gpu_buf(1, &y->q_norm, 0), gpu_buf(2, &Q, 0), gpu_bytes(3, &qr, sizeof qr);
  gpu_buf(4, seqbuf, 0);
  gpu_go((uint32_t)c.heads, (uint32_t)n, 1, 256, 1, 1);
  gpu_buf(0, &KK, 0), gpu_buf(1, &y->k_norm, 0), gpu_buf(2, &KR, 0), gpu_bytes(3, &kr, sizeof kr);
  gpu_go((uint32_t)c.kv_heads, (uint32_t)n, 1, 256, 1, 1);
  uint32_t width = (uint32_t)(c.kv_heads * c.head_dim);
  Kvp kp = {(uint32_t)n, width, (uint32_t)y->index, 0};
  gpu_kern(K_kv_store);
  gpu_buf(0, &KR, 0), gpu_buf(1, &VV, 0), gpu_buf(2, seqbuf, 0), gpu_bytes(3, &kp, sizeof kp);
  gpu_go((width + 255) / 256, (uint32_t)n, 1, 256, 1, 1);
  Att ap = {(uint32_t)c.heads, (uint32_t)c.kv_heads, (uint32_t)c.head_dim, (uint32_t)y->index,
            1.0f / sqrtf((float)c.head_dim), (uint32_t)blocks, 0, 0};
  size_t q_row = (size_t)c.heads * c.head_dim * 4;
  if (prompt_pass && one_sequence(n)) {  // on the matrix units, 32 rows of a head a threadgroup
    AttPrompt pp = {(uint32_t)c.heads, (uint32_t)c.kv_heads, (uint32_t)n, seq[0].pos, (uint32_t)y->index, seq[0].cap,
                    (uint32_t)(n + 31) / 32 * 32, 0, 1.0f / sqrtf((float)c.head_dim), 0, 0, 0};
    gpu_kern(K_q_half);
    gpu_buf(0, &Q, 0), gpu_buf(1, &QH, 0), gpu_bytes(2, &pp, sizeof pp);
    gpu_go(1, pp.rows, (uint32_t)c.heads, 256, 1, 1);
    gpu_kern(K_attend_prompt);
    gpu_buf(0, &QH, 0), gpu_buf(1, seqbuf, 0), gpu_buf(3, &QG, 0), gpu_buf(4, &AO, 0), gpu_bytes(5, &pp, sizeof pp);
    gpu_go((uint32_t)c.heads, pp.rows / 32, 1, 256, 1, 1);
    mv(&y->o, &AO, 0, &R, 0, n);
    return;
  }
  for (int g = 0; g < n; g += ROWS) {  // ROWS rows at a time: the blocks' partial sums hold ROWS rows
    int k = n - g < ROWS ? n - g : ROWS;
    uint32_t last = 0;
    for (int i = g; i < g + k; i++) last = seq[i].pos > last ? seq[i].pos : last;
    gpu_kern(K_attend_block);
    gpu_buf(0, &Q, g * q_row), gpu_buf(1, seqbuf, g * sizeof(Seq)), gpu_buf(2, &PART, 0), gpu_buf(3, &ML, 0);
    gpu_bytes(4, &ap, sizeof ap);
    gpu_go((uint32_t)c.kv_heads, last / BLK + 1, (uint32_t)k, BLK, 1, 1);
    gpu_kern(K_attend_combine);
    gpu_buf(0, &PART, 0), gpu_buf(1, &ML, 0), gpu_buf(2, &QG, 2 * g * q_row), gpu_buf(3, &AO, g * q_row);
    gpu_buf(4, seqbuf, g * sizeof(Seq)), gpu_bytes(5, &ap, sizeof ap);
    gpu_go((uint32_t)c.heads, (uint32_t)k, 1, (uint32_t)c.head_dim, 1, 1);
  }
  mv(&y->o, &AO, 0, &R, 0, n);
}

// The MTP head over m rows: HB holds their hiddens, TOKM their next tokens,
// SEQM their sequences at the hiddens' positions. Their keys go to each
// sequence's MTP layer at those positions; with logits, the last `logits`
// rows' draft logits go to LOGITS.
static void mtp_rows(int m, int logits) {
  seqbuf = &SEQM, seq = (Seq *)SEQM.ptr, nrows = m;
  if (prompt_pass) for (int i = 0; i < image_count; i++) if (images[i].features.obj) gpu_use(&images[i].features);
  uint32_t hidden = (uint32_t)c.hidden;
  gpu_kern(K_embed);
  gpu_buf(0, &embed_w, 0), gpu_buf(1, &TOKM, 0), gpu_buf(2, &X, 0), gpu_bytes(3, &hidden, 4);
  gpu_buf(4, &EMBM, 0);
  gpu_go((hidden + 255) / 256, (uint32_t)m, 1, 256, 1, 1);
  norm(&X, 0, &mtp_norm_e, &XN, 0, m, c.hidden, 0);
  norm(&HB, 0, &mtp_norm_h, &HH, 0, m, c.hidden, 0);  // the target hidden after its final norm
  gpu_kern(K_concat);  // [embedding | hidden] a row
  gpu_buf(0, &XN, 0), gpu_buf(1, &HH, 0), gpu_buf(2, &EH, 0), gpu_bytes(3, &hidden, 4);
  gpu_go((2 * hidden + 255) / 256, (uint32_t)m, 1, 256, 1, 1);
  mv(&mtp_fc, &EH, 0, &X, 0, m);
  norm(&X, 0, &M.in_norm, &XN, 0, m, c.hidden, 0);
  attention(&M);
  add_norm(&M.post_norm, m);
  gate_up(&M, m);
  mv(&M.down, &HH, 0, &R, 0, m);
  elementwise(K_add, &X, &R, 0, hidden * m);
  if (logits) {
    size_t first = (size_t)(m - logits) * c.hidden * 4;
    norm(&X, first, &mtp_norm, &XN, first, logits, c.hidden, 0);
    int was = prompt_pass;
    prompt_pass = 0;
    mv(&head, &XN, first, &LOGITS, 0, logits);
    prompt_pass = was;
  }
}

// An MTP row's sequence: the state's keys, at the hidden's position.
static Seq mtp_seq(const QwenState *s, int pos, int prompt) {
  uint64_t kv = s->kv.gpu;
  Seq row = {.k = kv, .v = kv + kv_bytes(s->cap) / 2, .pos = (uint32_t)pos, .cap = (uint32_t)s->cap};
  positions(&row, s, pos, prompt);
  return row;
}

static const float *pass(const QwenRow *rows, int n, int logits, int prompt) {
  qwen_reside(1);
  int32_t *tok = (int32_t *)TOK.ptr;
  uint64_t *emb = (uint64_t *)EMB.ptr;
  seqbuf = &SEQ, seq = (Seq *)SEQ.ptr;
  prompt_pass = prompt;
  for (int i = 0, k = 0; i < n; i++) {
    QwenState *s = rows[i].state;
    k = i > 0 && rows[i - 1].state == s ? k + 1 : 0;
    int pos = s->pos + k;
    if (pos >= s->cap) qwen_state_reserve(s, pos + 1);
    tok[i] = rows[i].token < 0 ? 0 : rows[i].token;
    emb[i] = rows[i].token < 0 ? image_embedding(pos) : 0;
  }
  for (int i = 0, k = 0; i < n; i++) {  // after any reserve moved a state's keys
    QwenState *s = rows[i].state;
    k = i > 0 && rows[i - 1].state == s ? k + 1 : 0;
    uint64_t kv = s->kv.gpu;
    seq[i] = (Seq){.gdn = s->gdn[s->cur].gpu, .gdn_save = rows[i].save ? s->gdn[rows[i].save - 1].gpu : 0,
                   .conv = s->conv[s->cur].gpu, .conv_save = rows[i].save ? s->conv[rows[i].save - 1].gpu : 0,
                   .k = kv, .v = kv + kv_bytes(s->cap) / 2, .pos = (uint32_t)(s->pos + k), .cap = (uint32_t)s->cap};
    positions(&seq[i], s, s->pos + k, prompt);
  }
  nrows = n;
  uint32_t hidden = (uint32_t)c.hidden;
  gpu_begin();
  if (prompt) for (int i = 0; i < image_count; i++) if (images[i].features.obj) gpu_use(&images[i].features);
  gpu_kern(K_embed);
  gpu_buf(0, &embed_w, 0), gpu_buf(1, &TOK, 0), gpu_buf(2, &X, 0), gpu_bytes(3, &hidden, 4);
  gpu_buf(4, &EMB, 0);
  gpu_go((hidden + 255) / 256, (uint32_t)n, 1, 256, 1, 1);
  norm(&X, 0, &L[0].in_norm, &XN, 0, n, c.hidden, 0);
  for (int l = 0; l < c.layers; l++) {
    const Layer *y = &L[l];
    if (y->linear) deltanet(y);
    else attention(y);
    add_norm(&y->post_norm, n);
    gate_up(y, n);
    mv(&y->down, &HH, 0, &R, 0, n);
    if (l + 1 < c.layers) add_norm(&L[l + 1].in_norm, n);
    else elementwise(K_add, &X, &R, 0, hidden * n);
  }
  if (logits || mtp) {
    int normalized = mtp ? n : logits;
    size_t first = (size_t)(n - normalized) * c.hidden * 4;
    norm(&X, first, &final_norm, &XN, first, normalized, c.hidden, 0);
    if (logits) {
      int was = prompt_pass;
      prompt_pass = 0;
      mv(&head, &XN, (size_t)(n - logits) * c.hidden * 4, &LOGITS, 0, logits);
      prompt_pass = was;
    }
  }
  if (mtp) {
    static GpuCopy cp[2 * PF];
    size_t hb = hid_bytes();
    int m = 0, mm = 0;
    int32_t *tm = (int32_t *)TOKM.ptr;
    Seq *sm = (Seq *)SEQM.ptr;
    if (prompt) {
      // The MTP head's keys at the pass's positions but its last: each
      // position's hidden with the next position's token (the first takes
      // the state's last hidden from before the pass); the pass's last
      // hidden waits in the state for its next token.
      for (int i = 0; i < n;) {
        QwenState *s = rows[i].state;
        int j = i;
        while (j < n && rows[j].state == s) j++;
        int p0 = s->pos;
        if (p0 > 0) {
          cp[m] = (GpuCopy){&s->hid[s->cur], &HB, 0, (size_t)m * hb, hb};
          tm[m] = rows[i].token < 0 ? 0 : rows[i].token;
          ((uint64_t *)EMBM.ptr)[m] = rows[i].token < 0 ? image_embedding(p0) : 0;
          sm[m] = mtp_seq(s, p0 - 1, 1), m++;
        }
        for (int r = i; r + 1 < j; r++) {
          cp[m] = (GpuCopy){&XN, &HB, (size_t)r * hb, (size_t)m * hb, hb};
          tm[m] = rows[r + 1].token < 0 ? 0 : rows[r + 1].token;
          ((uint64_t *)EMBM.ptr)[m] = rows[r + 1].token < 0 ? image_embedding(p0 + r + 1 - i) : 0;
          sm[m] = mtp_seq(s, p0 + (r - i), 1), m++;
        }
        i = j;
      }
      if (m) gpu_copies(cp, m);
    }
    // Each target hidden is normalized before the MTP head consumes it.
    for (int i = 0; i < n; i++) {
      QwenState *s = rows[i].state;
      if (i + 1 == n || rows[i + 1].state != s) cp[mm++] = (GpuCopy){&XN, &s->hid[s->cur], (size_t)i * hb, 0, hb};
      if (rows[i].save) cp[mm++] = (GpuCopy){&XN, &s->hid[rows[i].save - 1], (size_t)i * hb, 0, hb};
    }
    gpu_copies(cp, mm);
    if (m) mtp_rows(m, 0);
  }
  gpu_commit();
  gpu_sync();
  for (int i = 0; i < n; i++) {
    QwenState *s = rows[i].state;
    const Seq *row = &((Seq *)SEQ.ptr)[i];
    int next = row->rope_t > row->rope_h ? row->rope_t : row->rope_h;
    if (row->rope_w > next) next = row->rope_w;
    s->rope_delta = next - (int)row->pos;
    s->pos++;
    if (rows[i].save) {
      int at = rows[i].save - 1;
      s->saved_pos[at] = s->pos;
      s->saved_rope[at] = s->rope_delta;
    }
    if (mtp && prompt) s->mtp_next = s->pos - 1;  // every key of the pass's positions but its last
  }
  return logits ? (const float *)LOGITS.ptr : 0;
}

// The MTP head's drafts: row i takes rows[i].token as the token after its
// state's last hidden (with `saved`, the saved row's), at that hidden's
// position, and leaves its keys there. A state's rows consecutive, at
// consecutive positions. Returns each row's draft logits.
const float *qwen_mtp(const QwenMtpRow *rows, int n, int logits) {
  qwen_reside(1);
  static GpuCopy cp[ROWS];
  size_t hb = hid_bytes();
  int32_t *tm = (int32_t *)TOKM.ptr;
  Seq *sm = (Seq *)SEQM.ptr;
  for (int i = 0; i < n; i++) {
    QwenState *s = rows[i].state;
    int hidden = rows[i].saved ? rows[i].saved - 1 : s->cur;
    int pos = rows[i].saved ? s->saved_pos[hidden] - 1 : s->pos - 1;
    cp[i] = (GpuCopy){&s->hid[hidden], &HB, 0, (size_t)i * hb, hb};
    tm[i] = rows[i].token, sm[i] = mtp_seq(s, pos, 0);
    ((uint64_t *)EMBM.ptr)[i] = 0;
  }
  prompt_pass = 0;
  gpu_begin();
  gpu_copies(cp, n);
  mtp_rows(n, logits);
  gpu_commit();
  gpu_sync();
  return logits ? (const float *)LOGITS.ptr : 0;
}

const float *qwen_forward(const QwenRow *rows, int n, int logits) { return pass(rows, n, logits, 0); }
const float *qwen_prefill(const QwenRow *rows, int n, int logits) { return pass(rows, n, logits, 1); }

// ---- the model interface (model.h) ----

static Options qopt;
static double *proposal_probabilities;
static size_t proposal_capacity;
static const float *last_logits;  // what the last call returned, for a snapshot's logits
static double last_pass;          // when the last pass ran (now())
enum { RELEASE_AFTER = 60 };      // seconds without passes before the buffers leave the residency set
static ModelCounters counted;     // decoding and prompt passes' rows and times

static const char *q_refuse(const Options *o) {
  if (o->deepseek.experts || o->deepseek.ced || o->deepseek.substitute > 0 || o->deepseek.online_predictor || o->deepseek.predictor_state || o->deepseek.prefill_prune ||
      o->deepseek.draft_cache || o->dump_fd >= 0 || o->routes_fd >= 0 || o->deepseek.runahead || o->deepseek.cache_half_life ||
      o->deepseek.cache_read_cost || o->deepseek.pipeline)
    return "experts, ced, substitute, online-predictor, predictor-state, prefill-prune, draft-cache, cache, "
           "cache-half-life, cache-read-cost, dump-fd, routes-fd, runahead and pipeline apply to DeepSeek's routed experts and "
           "layers, which a qwen3_5 model does not have";
  if (o->draft > 5) return "draft: Qwen supports up to five chained MTP proposals";
  return 0;
}

static int q_load(const char *dir, const Options *o) {
  qopt = *o;
  if (!qwen_load(dir, o->draft)) return 0;
  size_t needed = mtp ? (size_t)o->draft * c.vocab : 0;
  if (needed > proposal_capacity) {
    double *next = memory_host_resize(proposal_probabilities, needed * sizeof *next);
    if (!next) return 0;
    proposal_probabilities = next;
    proposal_capacity = needed;
  }
  last_pass = now();
  counted.dense_bytes = weights_resident_bytes(dense);
  return 1;
}

static void q_counters(ModelCounters *out) {
  *out = counted;
  out->dense_bytes = weights_resident_bytes(dense) + qwen_vision_bytes(vision);
}
static MemorySlot *q_weight_slot(void) { return dense ? weights_slot(dense) : NULL; }

// A decoding pass's rows and times into the counters.
static const float *decoded(const float *logits, int rows, double t0, double g0) {
  counted.decode_rows += rows, counted.decode_seconds += now() - t0, counted.decode_gpu_seconds += gpu_busy - g0;
  return logits;
}

static int q_vocab(void) { return c.vocab; }
static int q_max_rows(void) { return ROWS; }
static int q_max_verify(void) { return mtp ? state_copies : 1; }
static int q_draft_block(void) { return mtp ? qopt.draft : 0; }
static int q_rewind(void) { return 0; }
static int q_stop(int token) { return qwen_stop(token); }
static int q_prefill_rows(void) { return PF; }

static ModelState *q_state_new(void) { return (ModelState *)qwen_state_new(); }
static void q_state_free(ModelState *s) { qwen_state_free((QwenState *)s); }
static size_t q_state_bytes(const ModelState *s, int positions) {
  int cap = s ? ((const QwenState *)s)->cap : 1024;
  while (cap < positions) cap *= 2;
  return qwen_state_bytes(cap);
}
static void q_state_copy(ModelState *dst, const ModelState *src, int positions) {
  qwen_state_copy((QwenState *)dst, (const QwenState *)src, positions);
}

typedef struct {
  QwenSnap *snap;
  float *logits;  // or null
} Taken;

static ModelSnapshot *q_snapshot_take(ModelState *s, int pos, int with_logits) {
  QwenState *q = (QwenState *)s;
  if (pos != q->pos) die("qwen3_5: a snapshot at %d of a state at %d", pos, q->pos);
  Taken *t = xmalloc(sizeof *t);
  *t = (Taken){qwen_snap_take(q), 0};
  if (with_logits) {
    if (!last_logits) die("qwen3_5: a snapshot's logits before any pass");
    t->logits = memory_host_required((size_t)c.vocab * 4, 1), memcpy(t->logits, last_logits, (size_t)c.vocab * 4);
  }
  return (ModelSnapshot *)t;
}
static void q_snapshot_restore(ModelState *s, const ModelSnapshot *x) {
  qwen_snap_restore((QwenState *)s, ((const Taken *)x)->snap);
}
static void q_snapshot_free(ModelSnapshot *x) {
  if (!x) return;
  qwen_snap_free(((Taken *)x)->snap), memory_host_free(((Taken *)x)->logits), free(x);
}
static size_t q_snapshot_bytes(int with_logits) { return qwen_snap_bytes() + (with_logits ? (size_t)c.vocab * 4 : 0); }
static const float *q_snapshot_logits(const ModelSnapshot *x) { return ((const Taken *)x)->logits; }

// Complete the key at the last committed target hidden, paired with its next token.
static const float *mtp_fill(QwenState *s, int token, int logits) {
  if (!mtp || qopt.draft <= 0 || s->pos == 0) return NULL;
  if (s->mtp_next >= s->pos && !logits) return NULL;
  QwenMtpRow row = {s, token, 0};
  s->mtp_next = s->pos;
  return qwen_mtp(&row, 1, logits);
}

static const float *q_prefill(ModelState *st, const int *tokens, int n, int from, int replay, int capture,
                              ModelSnapshot **captured, float *every) {
  QwenState *s = (QwenState *)st;
  if (replay && from) die("qwen3_5: no replay (replay_rows)");
  if (from != s->pos) die("qwen3_5: a prefill from %d of a state at %d", from, s->pos);
  if (captured) *captured = 0;
  if (capture >= 0 && (capture <= from || capture >= n || !captured))
    die("qwen3_5: a capture at %d in a prefill of positions %d to %d", capture, from, n);
  static QwenRow rows[PF];
  // with every position's logits, passes of ROWS rows (the head takes at most
  // ROWS): a prompt's rows give the same logits however it is split
  int most = every ? ROWS : PF;
  const float *lg = 0;
  for (int at = from; at < n;) {
    int end = capture > at && capture < n ? capture : n;
    int wanted = end - at < most ? end - at : most;
    int k = prepare_rows(wanted, at + wanted);
    if (!k) return NULL;
    for (int i = 0; i < k; i++) rows[i] = (QwenRow){s, tokens[at + i], 0};
    counted.prefill_rows_now = k, counted.prefill_layer = 1;
    lg = qwen_prefill(rows, k, every ? k : 1);
    counted.prefill_rows += k, counted.prefill_rows_now = 0, counted.prefill_layer = 0;
    if (every) memcpy(every + (size_t)(at - from) * c.vocab, lg, (size_t)k * c.vocab * 4), lg += (size_t)(k - 1) * c.vocab;
    at += k;
    if (at == capture) *captured = q_snapshot_take(st, at, 0);
  }
  last_pass = now();
  return last_logits = lg;
}

static const float *q_rows(ModelState *const *states, const int *tokens, const int *pos, int m) {
  int end = 0;
  for (int r = 0; r < m; r++) if (pos[r] + 1 > end) end = pos[r] + 1;
  if ((m > workspace_rows || end > blocks * BLK) && !prepare_rows(m, end)) return NULL;
  QwenRow rows[ROWS];
  for (int r = 0; r < m; r++) {
    QwenState *s = (QwenState *)states[r];
    mtp_fill(s, tokens[r], 0);
    rows[r] = (QwenRow){s, tokens[r], 0};
  }
  double t0 = last_pass = now(), g0 = gpu_busy;
  return last_logits = decoded(qwen_forward(rows, m, m), m, t0, g0);
}

static const double *q_draft(ModelState *st, int token, int pos, int *drafts, int n, Sampler *sampler) {
  QwenState *s = (QwenState *)st;
  qwen_state_reserve(s, pos + n + 1);
  drafts[0] = sample_proposal(mtp_fill(s, token, 1), c.vocab, sampler, proposal_probabilities);
  for (int i = 1; i < n; i++) {
    // The preceding draft head's hidden predicts the next target hidden.
    GpuCopy copy = {&XN, &HB, 0, 0, hid_bytes()};
    ((int32_t *)TOKM.ptr)[0] = drafts[i - 1];
    ((uint64_t *)EMBM.ptr)[0] = 0;
    ((Seq *)SEQM.ptr)[0] = mtp_seq(s, pos + i - 1, 0);
    prompt_pass = 0;
    gpu_begin();
    gpu_copies(&copy, 1);
    mtp_rows(1, 1);
    gpu_commit(); gpu_sync();
    drafts[i] = sample_proposal((const float *)LOGITS.ptr, c.vocab, sampler, proposal_probabilities + (size_t)i * c.vocab);
  }
  return sampler->temp > 0 ? proposal_probabilities : NULL;
}

static const float *q_verify(ModelState *st, const int *tokens, int m, int pos) {
  QwenState *s = (QwenState *)st;
  if ((m > workspace_rows || pos + m > blocks * BLK) && !prepare_rows(m, pos + m)) return NULL;
  mtp_fill(s, tokens[0], 0);
  QwenRow rows[ROWS];
  s->verify_origin = s->cur;
  for (int i = 0; i < m; i++) {
    int save = i + 1 < m ? (s->cur + i + 1) % s->copies + 1 : 0;
    rows[i] = (QwenRow){s, tokens[i], save};
  }
  double t0 = last_pass = now(), g0 = gpu_busy;
  return last_logits = decoded(qwen_forward(rows, m, m), m, t0, g0);
}

static void q_accept(ModelState *st, const int *tokens, int n, int pos) {
  QwenState *s = (QwenState *)st;
  if (n < s->pos - pos) qwen_state_back(s, (s->verify_origin + n) % s->copies);
  // Replace speculative draft keys with keys from the accepted target hiddens.
  // This also leaves snapshots and copied prefixes with committed MTP history.
  if (mtp && n > 1) {
    QwenMtpRow rows[ROWS];
    for (int i = 0; i + 1 < n; i++) {
      int checkpoint = (s->verify_origin + i + 1) % s->copies;
      rows[i] = (QwenMtpRow){s, tokens[i + 1], checkpoint + 1};
    }
    qwen_mtp(rows, n - 1, 0);
  }
  s->mtp_next = s->pos > 0 ? s->pos - 1 : 0;
  last_logits = (const float *)LOGITS.ptr + (size_t)(n - 1) * c.vocab;
}

static void q_seek(ModelState *s, const int *tokens, int pos) {}
static int q_replay_rows(int k) { return -1; }  // a DeltaNet state cannot be rebuilt from copied rows
static int q_replay_start(int k) { return k; }

static int q_idle(void) {
  if (now() - last_pass >= RELEASE_AFTER) qwen_reside(0);
  return 0;
}

static int q_image(Media *image) {
  qwen_reside(1);
  return qwen_vision_encode(vision, image);
}

static void q_images(const Media *input, int count) {
  images = input;
  image_count = count;
}

static void q_chat(char *text, size_t cap, const char *prompt, int images, int think) {
  size_t at = (size_t)snprintf(text, cap, "<|im_start|>user\n");
  for (int i = 0; i < images && at < cap; i++)
    at += (size_t)snprintf(text + at, cap - at, "<|vision_start|>" MEDIA_MARK "<|vision_end|>");
  if (at < cap) snprintf(text + at, cap - at, "%s<|im_end|>\n<|im_start|>assistant\n%s", prompt,
                        think ? "<think>\n" : "<think>\n\n</think>\n\n");
}

static const char *const q_types[] = {"qwen3_5", "qwen3_5_text", 0};

const ModelOps qwen_ops = {
    .family = "Qwen3.8", .model_types = q_types, .media_token = "<|image_pad|>", .refuse = q_refuse, .load = q_load, .weight_slot = q_weight_slot, .counters = q_counters,
    .vocab = q_vocab, .max_rows = q_max_rows,
    .max_verify = q_max_verify, .draft_block = q_draft_block, .rewind = q_rewind, .stop = q_stop,
    .prefill_rows = q_prefill_rows,
    .state_new = q_state_new, .state_free = q_state_free, .state_bytes = q_state_bytes, .state_copy = q_state_copy,
    .snapshot_take = q_snapshot_take, .snapshot_restore = q_snapshot_restore, .snapshot_free = q_snapshot_free,
    .snapshot_bytes = q_snapshot_bytes, .snapshot_logits = q_snapshot_logits, .prefill = q_prefill,
    .replay_rows = q_replay_rows, .replay_start = q_replay_start, .rows = q_rows, .draft = q_draft, .verify = q_verify, .accept = q_accept,
    .seek = q_seek, .idle = q_idle, .chat = q_chat,
    .prepare_media = qwen_prepare_image, .encode_media = q_image, .media = q_images};
