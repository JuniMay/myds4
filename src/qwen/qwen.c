// The Qwen3.8 (qwen3_5) text model: Qwen3.8-27B in NVIDIA's NVFP4 build (see
// qwen.h and docs/qwen.md). A forward pass runs up to QWEN_ROWS token rows
// in one command buffer: per layer an RMSNorm, the mixer (a gated DeltaNet
// in three layers of four, gated full attention in the fourth), the
// residual, an RMSNorm and the SwiGLU MLP. The math follows the reference
// implementation (MLX's qwen3_5).
#include "qwen.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "../engine/myds4.h"

_Alignas(4) static const char kernel_source[] = {
#embed "qwen.metal"
    , 0};

#define ROWS QWEN_ROWS
#define PF QWEN_PREFILL
#define BLK 256  // positions per attention block (qwen.metal's BLK)

// A matrix: NVFP4 ('n': codes and E4M3 scales), FP8 ('8') or BF16 ('b'),
// times scale.
typedef struct {
  Buf w, s;
  float scale;
  int rows, cols, kind;
} Mat;

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

static struct {
  int layers, linear, attention, hidden, inter, vocab, nstop, stop[8];
  int heads, kv_heads, head_dim, rot;
  int key_heads, value_heads, key_dim, value_dim, channels;
  float eps, theta;
} c;

static Layer *L;
static Buf embed_w, final_norm;
static Mat head;
// The MTP head (mtp.*): pre-fc norms, fc, one full-attention layer (its keys
// after the model's, as attention layer c.attention), its norm; the
// embedding and head are the model's.
static int mtp;
static Layer M;
static Mat mtp_fc;
static Buf mtp_norm_e, mtp_norm_h, mtp_norm;
// Activations of up to PF rows; XB: a matrix's input as BF16 (prompt passes);
// SEQ: the rows' sequences.
static Buf TOK, X, XN, R, QKV, CO, Z, A_, B_, G, BETA, GO, GN, QG, KK, VV, Q, KR, AO, GT, UP, HH, LOGITS, XB, SEQ;
// The MTP head's rows: their hiddens, next tokens and sequences; its fc's input.
static Buf HB, TOKM, SEQM, EH;
static Buf QH;  // a prompt pass's queries as FP16, by head
static Buf PART, ML;  // attention blocks' partial sums, for states of up to `blocks` * BLK positions
static int blocks;
static int K_mv1[3], K_mvr[3], K_mm[3], K_gate_up[3], K_add_norm, K_x_bf16, K_concat, K_q_half, K_attend_prompt, K_embed, K_rms_norm, K_l2_norm, K_conv, K_gdn_gates, K_gdn, K_qk_rope, K_kv_store, K_attend_block,
    K_attend_combine, K_swiglu, K_add;

struct QwenState {
  Buf gdn[2], conv[2];  // the current recurrent state and the spare (a saved row's)
  Buf kv;               // keys then values: [attention layer][cap][kv head][head dim], FP16
  Buf hid[2];           // the last row's hidden (the MTP head's input), and the saved row's
  int cur, pos, cap;
  int saved;  // the position after the saved row that the spare holds (-1: none)
};

struct QwenSnap {
  Buf gdn, conv, hid;
  int pos;
};

// A row's sequence, as qwen.metal's Seq.
typedef struct {
  uint64_t gdn, gdn_save, conv, conv_save, k, v;
  uint32_t pos, cap;
} Seq;

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
  gpu_residency_commit();
  resident = on;
}

static double num(const Json *o, const char *key) {
  const Json *v = json_get(o, key);
  if (!v || v->type != J_NUM) die("config.json: no number %s", key);
  return v->num;
}

static Buf upload(Tensor t) {
  Buf b = own(gpu_alloc(t.size));
  size_t done = 0;
  while (done < t.size) {  // macOS refuses reads above INT_MAX bytes
    size_t want = t.size - done < (1u << 30) ? t.size - done : (1u << 30);
    ssize_t n = pread(t.fd, b.ptr + done, want, (off_t)(t.off + done));
    if (n <= 0) die("cannot read a tensor");
    done += (size_t)n;
  }
  return b;
}

// A BF16 or FP32 vector as FP32, plus `add`.
static Buf vec(const char *name, float add) {
  Tensor t = st_get("%s", name);
  int n = t.dtype == 'b' ? (int)(t.size / 2) : (int)(t.size / 4);
  if (t.dtype != 'b' && t.dtype != 'f') die("%s: not BF16 or FP32", name);
  Buf raw = upload(t), b = own(gpu_alloc((size_t)n * 4));
  float *f = (float *)b.ptr;
  for (int i = 0; i < n; i++)
    f[i] = (t.dtype == 'b' ? (union { uint32_t u; float f; }){(uint32_t)((uint16_t *)raw.ptr)[i] << 16}.f
                           : ((float *)raw.ptr)[i]) + add;
  unown(&raw);
  return b;
}

static float scalar(const char *name) {
  Tensor t = st_get("%s", name);
  float v;
  if (t.dtype != 'f' || t.size != 4 || pread(t.fd, &v, 4, (off_t)t.off) != 4) die("%s is not an FP32 scalar", name);
  return v;
}

static Mat mat(const char *stem) {
  char name[256];
  snprintf(name, sizeof name, "%s.weight", stem);
  Tensor t = st_get("%s", name);
  Mat m = {.rows = (int)t.shape[0], .scale = 1};
  if (t.dtype == 'u') {  // NVFP4: two codes a byte
    m.kind = 'n', m.cols = (int)t.shape[1] * 2;
    m.w = upload(t);
    snprintf(name, sizeof name, "%s.weight_scale", stem);
    m.s = upload(st_get("%s", name));
    snprintf(name, sizeof name, "%s.weight_scale_2", stem);
    m.scale = scalar(name);
  } else if (t.dtype == '8') {
    m.kind = '8', m.cols = (int)t.shape[1];
    m.w = upload(t);
    snprintf(name, sizeof name, "%s.weight_scale", stem);
    m.scale = scalar(name);
  } else if (t.dtype == 'b') {
    m.kind = 'b', m.cols = (int)t.shape[1];
    m.w = upload(t);
  } else {
    die("%s: unsupported weight type", name);
  }
  if (m.cols % 512 || m.rows % 4) die("%s: %d x %d, not 4-row and 512-column multiples", name, m.rows, m.cols);
  return m;
}

static Buf zeros(size_t n) {
  Buf b = own(gpu_alloc(n));
  memset(b.ptr, 0, n);
  return b;
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

void qwen_load(const char *dir) {
  char path[4096];
  snprintf(path, sizeof path, "%s/config.json", dir);
  Json *root = json_parse(read_file(path, 0));
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
  c.key_heads = (int)num(t, "linear_num_key_heads"), c.value_heads = (int)num(t, "linear_num_value_heads");
  c.key_dim = (int)num(t, "linear_key_head_dim"), c.value_dim = (int)num(t, "linear_value_head_dim");
  c.channels = 2 * c.key_heads * c.key_dim + c.value_heads * c.value_dim;
  // The kernels' shapes: 128-value DeltaNet heads, at most 8 query heads per
  // key head and 256-value attention heads (one thread a value).
  if ((int)num(t, "linear_conv_kernel_dim") != 4 || c.key_dim != 128 || c.value_dim != 128 ||
      c.value_heads % c.key_heads || c.head_dim != 256 || c.heads % c.kv_heads || c.heads / c.kv_heads > 8)
    die("%s: a qwen3_5 shape this engine does not take", path);
  snprintf(path, sizeof path, "%s/generation_config.json", dir);
  const Json *eos = access(path, R_OK) ? 0 : json_get(json_parse(read_file(path, 0)), "eos_token_id");
  if (!eos) eos = json_get(t, "eos_token_id");
  if (!eos) die("%s: no eos_token_id", dir);
  for (int i = 0; i < (eos->type == J_ARR ? eos->n : 1) && c.nstop < 8; i++)
    c.stop[c.nstop++] = (int)(eos->type == J_ARR ? eos->kid[i].num : eos->num);
  const Json *types = json_get(t, "layer_types");
  if (!types || types->type != J_ARR || types->n != c.layers) die("%s: no layer_types", path);

  gpu_init(kernel_source, sizeof kernel_source - 1);
  K_mv1[0] = gpu_kernel("mv1_nvfp4"), K_mv1[1] = gpu_kernel("mv1_fp8"), K_mv1[2] = gpu_kernel("mv1_bf16");
  K_mvr[0] = gpu_kernel("mvr_nvfp4"), K_mvr[1] = gpu_kernel("mvr_fp8"), K_mvr[2] = gpu_kernel("mvr_bf16");
  K_gate_up[0] = gpu_kernel("gate_up_nvfp4"), K_gate_up[1] = gpu_kernel("gate_up_fp8"), K_gate_up[2] = gpu_kernel("gate_up_bf16");
  K_add_norm = gpu_kernel("add_norm");
  K_mm[0] = gpu_kernel("mm_nvfp4"), K_mm[1] = gpu_kernel("mm_fp8"), K_mm[2] = gpu_kernel("mm_bf16");
  K_x_bf16 = gpu_kernel("x_bf16"), K_concat = gpu_kernel("concat");
  K_q_half = gpu_kernel("q_half"), K_attend_prompt = gpu_kernel("attend_prompt");
  K_embed = gpu_kernel("embed"), K_rms_norm = gpu_kernel("rms_norm"), K_l2_norm = gpu_kernel("l2_norm");
  K_conv = gpu_kernel("conv"), K_gdn_gates = gpu_kernel("gdn_gates"), K_gdn = gpu_kernel("gdn");
  K_qk_rope = gpu_kernel("qk_rope"), K_kv_store = gpu_kernel("kv_store");
  K_attend_block = gpu_kernel("attend_block"), K_attend_combine = gpu_kernel("attend_combine");
  K_swiglu = gpu_kernel("swiglu"), K_add = gpu_kernel("add");

  st_open(dir);
  double t0 = now();
  const char *m = "model.language_model";
  char p[256], n[256];
  embed_w = upload(st_get("%s.embed_tokens.weight", m));
  snprintf(n, sizeof n, "%s.norm.weight", m);
  final_norm = vec(n, 1);
  head = mat("lm_head");
  L = calloc((size_t)c.layers, sizeof *L);
  for (int l = 0; l < c.layers; l++) {
    Layer *y = &L[l];
    const char *kind = types->kid[l].s;
    y->linear = !strcmp(kind, "linear_attention");
    if (!y->linear && strcmp(kind, "full_attention")) die("layer %d: unknown type %s", l, kind);
    y->index = y->linear ? c.linear++ : c.attention++;
    snprintf(n, sizeof n, "%s.layers.%d.input_layernorm.weight", m, l), y->in_norm = vec(n, 1);
    snprintf(n, sizeof n, "%s.layers.%d.post_attention_layernorm.weight", m, l), y->post_norm = vec(n, 1);
    snprintf(p, sizeof p, "%s.layers.%d.mlp.gate_proj", m, l), y->gate = mat(p);
    snprintf(p, sizeof p, "%s.layers.%d.mlp.up_proj", m, l), y->up = mat(p);
    snprintf(p, sizeof p, "%s.layers.%d.mlp.down_proj", m, l), y->down = mat(p);
    if (y->linear) {
      snprintf(p, sizeof p, "%s.layers.%d.linear_attn.in_proj_qkv", m, l), y->qkv = mat(p);
      snprintf(p, sizeof p, "%s.layers.%d.linear_attn.in_proj_z", m, l), y->z = mat(p);
      snprintf(p, sizeof p, "%s.layers.%d.linear_attn.in_proj_a", m, l), y->a = mat(p);
      snprintf(p, sizeof p, "%s.layers.%d.linear_attn.in_proj_b", m, l), y->b = mat(p);
      snprintf(p, sizeof p, "%s.layers.%d.linear_attn.out_proj", m, l), y->out = mat(p);
      snprintf(n, sizeof n, "%s.layers.%d.linear_attn.conv1d.weight", m, l), y->conv = vec(n, 0);
      snprintf(n, sizeof n, "%s.layers.%d.linear_attn.A_log", m, l), y->A = vec(n, 0);
      for (int h = 0; h < c.value_heads; h++) ((float *)y->A.ptr)[h] = expf(((float *)y->A.ptr)[h]);
      snprintf(n, sizeof n, "%s.layers.%d.linear_attn.dt_bias", m, l), y->dt_bias = vec(n, 0);
      snprintf(n, sizeof n, "%s.layers.%d.linear_attn.norm.weight", m, l), y->norm = vec(n, 0);
      if (y->qkv.rows != c.channels || y->z.rows != c.value_heads * c.value_dim || y->a.rows != c.value_heads)
        die("layer %d: DeltaNet projections of unexpected sizes", l);
    } else {
      snprintf(p, sizeof p, "%s.layers.%d.self_attn.q_proj", m, l), y->q = mat(p);
      snprintf(p, sizeof p, "%s.layers.%d.self_attn.k_proj", m, l), y->k = mat(p);
      snprintf(p, sizeof p, "%s.layers.%d.self_attn.v_proj", m, l), y->v = mat(p);
      snprintf(p, sizeof p, "%s.layers.%d.self_attn.o_proj", m, l), y->o = mat(p);
      snprintf(n, sizeof n, "%s.layers.%d.self_attn.q_norm.weight", m, l), y->q_norm = vec(n, 1);
      snprintf(n, sizeof n, "%s.layers.%d.self_attn.k_norm.weight", m, l), y->k_norm = vec(n, 1);
      if (y->q.rows != 2 * c.heads * c.head_dim || y->k.rows != c.kv_heads * c.head_dim)
        die("layer %d: attention projections of unexpected sizes", l);
    }
  }
  mtp = st_has("mtp.fc.weight");
  if (mtp) {
    mtp_fc = mat("mtp.fc");
    mtp_norm_e = vec("mtp.pre_fc_norm_embedding.weight", 1), mtp_norm_h = vec("mtp.pre_fc_norm_hidden.weight", 1);
    mtp_norm = vec("mtp.norm.weight", 1);
    M = (Layer){.linear = 0, .index = c.attention};
    M.in_norm = vec("mtp.layers.0.input_layernorm.weight", 1);
    M.post_norm = vec("mtp.layers.0.post_attention_layernorm.weight", 1);
    M.q = mat("mtp.layers.0.self_attn.q_proj"), M.k = mat("mtp.layers.0.self_attn.k_proj");
    M.v = mat("mtp.layers.0.self_attn.v_proj"), M.o = mat("mtp.layers.0.self_attn.o_proj");
    M.q_norm = vec("mtp.layers.0.self_attn.q_norm.weight", 1), M.k_norm = vec("mtp.layers.0.self_attn.k_norm.weight", 1);
    M.gate = mat("mtp.layers.0.mlp.gate_proj"), M.up = mat("mtp.layers.0.mlp.up_proj");
    M.down = mat("mtp.layers.0.mlp.down_proj");
    if (mtp_fc.cols != 2 * c.hidden || mtp_fc.rows != c.hidden) die("mtp.fc: %d x %d", mtp_fc.rows, mtp_fc.cols);
  }
  size_t r = PF * 4, vd = (size_t)c.value_heads * c.value_dim;
  TOK = zeros(PF * 4), SEQ = zeros(PF * sizeof(Seq));
  int widest = c.inter > 2 * c.hidden ? c.inter : 2 * c.hidden;  // 2 * hidden: the MTP head's fc
  if (vd > (size_t)widest) widest = (int)vd;
  XB = zeros((size_t)PF * widest * 2);
  X = zeros(r * c.hidden), XN = zeros(r * c.hidden), R = zeros(r * c.hidden);
  QKV = zeros(r * c.channels), CO = zeros(r * c.channels), Z = zeros(r * vd), GO = zeros(r * vd), GN = zeros(r * vd);
  A_ = zeros(r * c.value_heads), B_ = zeros(r * c.value_heads), G = zeros(r * c.value_heads);
  BETA = zeros(r * c.value_heads);
  QG = zeros(r * 2 * c.heads * c.head_dim), KK = zeros(r * c.kv_heads * c.head_dim);
  VV = zeros(r * c.kv_heads * c.head_dim), Q = zeros(r * c.heads * c.head_dim), KR = zeros(r * c.kv_heads * c.head_dim);
  AO = zeros(r * c.heads * c.head_dim);
  GT = zeros(r * c.inter), UP = zeros(r * c.inter), HH = zeros(r * c.inter);
  LOGITS = zeros((size_t)ROWS * 4 * c.vocab);
  HB = zeros(r * c.hidden), TOKM = zeros(PF * 4), SEQM = zeros(PF * sizeof(Seq)), EH = zeros(r * 2 * c.hidden);
  QH = zeros((size_t)c.heads * PF * c.head_dim * 2);
  grow_blocks(4096);
  fprintf(stderr, "qwen: %d layers (%d DeltaNet, %d attention), %d hidden%s, loaded in %.1f s\n", c.layers, c.linear,
          c.attention, c.hidden, mtp ? ", MTP head" : "", now() - t0);
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
  return 2 * (gdn_bytes() + conv_bytes() + hid_bytes()) + kv_bytes(positions);
}
uint64_t qwen_snap_bytes(void) { return gdn_bytes() + conv_bytes() + hid_bytes(); }
int qwen_mtp_available(void) { return mtp; }

QwenState *qwen_state_new(void) {
  qwen_reside(1);
  QwenState *s = calloc(1, sizeof *s);
  for (int i = 0; i < 2; i++)
    s->gdn[i] = zeros(gdn_bytes()), s->conv[i] = zeros(conv_bytes()), s->hid[i] = zeros(hid_bytes());
  s->cap = 1024;
  s->kv = own(gpu_alloc(kv_bytes(s->cap)));
  s->saved = -1;
  grow_blocks(s->cap);
  return s;
}

void qwen_state_free(QwenState *s) {
  if (!s) return;
  gpu_sync();
  for (int i = 0; i < 2; i++) unown(&s->gdn[i]), unown(&s->conv[i]), unown(&s->hid[i]);
  unown(&s->kv);
  free(s);
}

int qwen_state_pos(const QwenState *s) { return s->pos; }

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
  Buf kv = own(gpu_alloc(kv_bytes(cap)));
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
  dst->pos = positions, dst->saved = -1;
}

void qwen_state_back(QwenState *s) {
  if (s->saved < 0) die("qwen: no saved row to return to");
  s->cur = !s->cur, s->pos = s->saved, s->saved = -1;
}

QwenSnap *qwen_snap_take(const QwenState *s) {
  qwen_reside(1);
  QwenSnap *n = calloc(1, sizeof *n);
  n->gdn = own(gpu_alloc(gdn_bytes())), n->conv = own(gpu_alloc(conv_bytes()));
  n->hid = own(gpu_alloc(hid_bytes())), n->pos = s->pos;
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
  s->pos = n->pos, s->saved = -1;
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
typedef struct { uint32_t heads, dim, rot, stride; float theta, eps; uint32_t pad0, pad1; } Rope;
typedef struct { uint32_t n, width, layer, pad; } Kvp;
typedef struct { uint32_t heads, kv, dim, layer; float scale; uint32_t blocks, pad0, pad1; } Att;
typedef struct { uint32_t heads, kv, n, p0, layer, cap, rows, pad; float scale; uint32_t pad0, pad1, pad2; } AttPrompt;


typedef struct { uint32_t m, n, k, pad; float scale; uint32_t pad0, pad1, pad2; } MMp;

static Seq *seq;      // the pass's rows' sequences: SEQ's, or SEQM's for the MTP head
static Buf *seqbuf;  // their buffer
static int nrows, prompt_pass;

// y (rows of m->rows values) = m x for the n rows of x: in a prompt pass on
// the matrix units, else with the row kernels.
static void mv(const Mat *m, const Buf *x, size_t xo, const Buf *y, size_t yo, int n) {
  int f = m->kind == 'n' ? 0 : m->kind == '8' ? 1 : 2;
  if (prompt_pass) {
    uint32_t k = (uint32_t)m->cols;
    gpu_kern(K_x_bf16);
    gpu_buf(0, x, xo), gpu_buf(1, &XB, 0), gpu_bytes(2, &k, 4);
    gpu_go((k / 4 + 255) / 256, (uint32_t)n, 1, 256, 1, 1);
    MMp p = {(uint32_t)n, (uint32_t)m->rows, k, 0, m->scale, 0, 0, 0};
    gpu_kern(K_mm[f]);
    gpu_buf(0, &m->w, 0), gpu_buf(1, m->kind == 'n' ? &m->s : &m->w, 0);
    gpu_buf(2, &XB, 0), gpu_buf(3, y, yo), gpu_bytes(4, &p, sizeof p);
    gpu_go((uint32_t)(m->rows + 63) / 64, (uint32_t)(n + 127) / 128, 1, 128, 1, 1);
    return;
  }
  Mv p = {(uint32_t)m->rows, (uint32_t)m->cols, (uint32_t)n, m->scale};
  gpu_kern(n == 1 ? K_mv1[f] : K_mvr[f]);
  gpu_buf(0, &m->w, 0), gpu_buf(1, m->kind == 'n' ? &m->s : &m->w, 0);
  gpu_buf(2, x, xo), gpu_buf(3, y, yo), gpu_bytes(4, &p, sizeof p);
  gpu_go((uint32_t)(m->rows + 7) / 8, 1, 1, 256, 1, 1);
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
    mv(&y->gate, &XN, 0, &GT, 0, n), mv(&y->up, &XN, 0, &UP, 0, n);
    elementwise(K_swiglu, &GT, &UP, &HH, (uint32_t)(c.inter * n));
    return;
  }
  const Mat *g = &y->gate, *u = &y->up;
  int f = g->kind == 'n' ? 0 : g->kind == '8' ? 1 : 2;
  Mv p = {(uint32_t)g->rows, (uint32_t)g->cols, (uint32_t)n, g->scale};
  gpu_kern(K_gate_up[f]);
  gpu_buf(0, &g->w, 0), gpu_buf(1, g->kind == 'n' ? &g->s : &g->w, 0);
  gpu_buf(5, &u->w, 0), gpu_buf(6, u->kind == 'n' ? &u->s : &u->w, 0);
  gpu_buf(2, &XN, 0), gpu_buf(3, &HH, 0), gpu_bytes(4, &p, sizeof p), gpu_bytes(7, &u->scale, 4);
  gpu_go((uint32_t)(g->rows + 7) / 8, 1, 1, 256, 1, 1);
}


static void deltanet(const Layer *y) {
  int n = nrows;
  uint32_t keys = (uint32_t)(c.key_heads * c.key_dim);
  mv(&y->qkv, &XN, 0, &QKV, 0, n), mv(&y->z, &XN, 0, &Z, 0, n);
  mv(&y->a, &XN, 0, &A_, 0, n), mv(&y->b, &XN, 0, &B_, 0, n);
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
  mv(&y->q, &XN, 0, &QG, 0, n), mv(&y->k, &XN, 0, &KK, 0, n), mv(&y->v, &XN, 0, &VV, 0, n);
  Rope qr = {(uint32_t)c.heads, (uint32_t)c.head_dim, (uint32_t)c.rot, (uint32_t)(2 * c.head_dim), c.theta, c.eps, 0, 0};
  Rope kr = {(uint32_t)c.kv_heads, (uint32_t)c.head_dim, (uint32_t)c.rot, (uint32_t)c.head_dim, c.theta, c.eps, 0, 0};
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
  uint32_t hidden = (uint32_t)c.hidden;
  gpu_kern(K_embed);
  gpu_buf(0, &embed_w, 0), gpu_buf(1, &TOKM, 0), gpu_buf(2, &X, 0), gpu_bytes(3, &hidden, 4);
  gpu_go((hidden + 255) / 256, (uint32_t)m, 1, 256, 1, 1);
  norm(&X, 0, &mtp_norm_e, &XN, 0, m, c.hidden, 0);
  norm(&HB, 0, &mtp_norm_h, &HH, 0, m, c.hidden, 0);  // the hidden before the final norm
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
static Seq mtp_seq(const QwenState *s, int pos) {
  uint64_t kv = s->kv.gpu;
  return (Seq){0, 0, 0, 0, kv, kv + kv_bytes(s->cap) / 2, (uint32_t)pos, (uint32_t)s->cap};
}

static const float *pass(const QwenRow *rows, int n, int logits, int prompt) {
  qwen_reside(1);
  if (n < 1 || n > (prompt ? PF : ROWS) || logits < 0 || logits > n || logits > ROWS)
    die("qwen: %d rows (%d with logits)", n, logits);
  int32_t *tok = (int32_t *)TOK.ptr;
  seqbuf = &SEQ, seq = (Seq *)SEQ.ptr;
  prompt_pass = prompt;
  for (int i = 0, k = 0; i < n; i++) {
    QwenState *s = rows[i].state;
    k = i > 0 && rows[i - 1].state == s ? k + 1 : 0;
    for (int j = 0; k == 0 && j < i; j++)
      if (rows[j].state == s) die("qwen: a state's rows apart in one pass");
    if (rows[i].token < 0 || rows[i].token >= c.vocab) die("qwen: token %d outside the vocabulary", rows[i].token);
    int pos = s->pos + k;
    if (pos >= s->cap) qwen_state_reserve(s, pos + 1);
    tok[i] = rows[i].token;
  }
  for (int i = 0, k = 0; i < n; i++) {  // after any reserve moved a state's keys
    QwenState *s = rows[i].state;
    k = i > 0 && rows[i - 1].state == s ? k + 1 : 0;
    uint64_t kv = s->kv.gpu;
    seq[i] = (Seq){s->gdn[s->cur].gpu, rows[i].save ? s->gdn[!s->cur].gpu : 0, s->conv[s->cur].gpu,
                   rows[i].save ? s->conv[!s->cur].gpu : 0, kv, kv + kv_bytes(s->cap) / 2,
                   (uint32_t)(s->pos + k), (uint32_t)s->cap};
  }
  nrows = n;
  uint32_t hidden = (uint32_t)c.hidden;
  gpu_begin();
  gpu_kern(K_embed);
  gpu_buf(0, &embed_w, 0), gpu_buf(1, &TOK, 0), gpu_buf(2, &X, 0), gpu_bytes(3, &hidden, 4);
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
  if (logits) {  // the head with the row kernels: at most ROWS rows
    size_t first = (size_t)(n - logits) * c.hidden * 4;
    norm(&X, first, &final_norm, &XN, first, logits, c.hidden, 0);
    int was = prompt_pass;
    prompt_pass = 0;
    mv(&head, &XN, first, &LOGITS, 0, logits);
    prompt_pass = was;
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
          tm[m] = rows[i].token, sm[m] = mtp_seq(s, p0 - 1), m++;
        }
        for (int r = i; r + 1 < j; r++) {
          cp[m] = (GpuCopy){&X, &HB, (size_t)r * hb, (size_t)m * hb, hb};
          tm[m] = rows[r + 1].token, sm[m] = mtp_seq(s, p0 + (r - i)), m++;
        }
        i = j;
      }
      if (m) gpu_copies(cp, m);
    }
    // each state's last hidden, and a saved row's, after the copies above read the old one
    for (int i = 0; i < n; i++) {
      QwenState *s = rows[i].state;
      if (i + 1 == n || rows[i + 1].state != s) cp[mm++] = (GpuCopy){&X, &s->hid[s->cur], (size_t)i * hb, 0, hb};
      if (rows[i].save) cp[mm++] = (GpuCopy){&X, &s->hid[!s->cur], (size_t)i * hb, 0, hb};
    }
    gpu_copies(cp, mm);
    if (m) mtp_rows(m, 0);
  }
  gpu_commit();
  gpu_sync();
  for (int i = 0; i < n; i++) {
    QwenState *s = rows[i].state;
    s->pos++;
    if (rows[i].save) s->saved = s->pos;
  }
  return logits ? (const float *)LOGITS.ptr : 0;
}

// The MTP head's drafts: row i takes rows[i].token as the token after its
// state's last hidden (with `saved`, the saved row's), at that hidden's
// position, and leaves its keys there. A state's rows consecutive, at
// consecutive positions. Returns each row's draft logits.
const float *qwen_mtp(const QwenMtpRow *rows, int n, int logits) {
  qwen_reside(1);
  if (!mtp) die("qwen: the checkpoint has no MTP head");
  if (n < 1 || n > ROWS || logits < 0 || logits > n) die("qwen: %d MTP rows (%d with logits)", n, logits);
  static GpuCopy cp[ROWS];
  size_t hb = hid_bytes();
  int32_t *tm = (int32_t *)TOKM.ptr;
  Seq *sm = (Seq *)SEQM.ptr;
  for (int i = 0; i < n; i++) {
    QwenState *s = rows[i].state;
    int pos = rows[i].saved ? s->saved - 1 : s->pos - 1;
    if (pos < 0) die("qwen: an MTP row without a hidden");
    if (rows[i].token < 0 || rows[i].token >= c.vocab) die("qwen: token %d outside the vocabulary", rows[i].token);
    cp[i] = (GpuCopy){&s->hid[rows[i].saved ? !s->cur : s->cur], &HB, 0, (size_t)i * hb, hb};
    tm[i] = rows[i].token, sm[i] = mtp_seq(s, pos);
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
