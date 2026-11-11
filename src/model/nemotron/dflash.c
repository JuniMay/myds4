#include "dflash.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

enum { HIDDEN = 2688, LAYERS = 6, HEADS = 32, KV_HEADS = 2, DIMENSION = 128, VOCAB = 131072, RANK = 256, TOP = 16 };
typedef struct {
  Buf base;
  Linear coefficients;
} Convolution;
typedef struct {
  Buf input_norm, output_norm, qnorm, knorm;
  Linear q, k, v, output, gate, up, down;
  Convolution attention_conv, mlp_conv;
} Block;
typedef struct {
  uint64_t state, saved, conv, saved_conv, k, v;
  uint32_t pos, cap;
  int32_t rope[4];
} Sequence;
enum {
  X,
  NORMAL,
  CONVOLVED,
  COEFFICIENTS,
  RESULT,
  Q,
  K,
  V,
  KR,
  ATTENTION,
  PARTS,
  MAXIMA,
  PACKED,
  GATE,
  UP,
  HIDDEN_PRODUCT,
  LOGITS,
  TOKENS,
  EMBEDDINGS,
  SEQUENCES,
  CONTEXT,
  CONDITION,
  BUFFERS
};
typedef struct {
  Buf buffer[BUFFERS];
  int rows, blocks;
} DFlashWork;
struct DFlash {
  TensorIndex *checkpoint;
  WeightSet *weights;
  MemorySlot workspace;
  const Buf *embedding;
  const Linear *head;
  Linear combine, condition;
  Buf hidden_norm, final_norm, predecessor, successor;
  Block layer[LAYERS];
  DFlashWork first, *work;  // the buffers in use: the first ones, or a pass's (src/engine/pass.h)
  int target_layers[LAYERS];
  float eps, theta;
  int capture, norm, embed, conv, rope, store, attention, sum, add, swiglu;
};
static float value(uint16_t v) {
  uint32_t u = (uint32_t)v << 16;
  float x;
  memcpy(&x, &u, 4);
  return x;
}
static float rounded(float x) {
  uint32_t u;
  memcpy(&u, &x, 4);
  u = (u + 0x7fff + ((u >> 16) & 1)) & 0xffff0000;
  memcpy(&x, &u, 4);
  return x;
}
static void vec(DFlash *d, TensorIndex *index, Buf *out, const char *name) {
  weights_vector(d->weights, out, st_get(index, "%s", name), 0, 0);
}
static void release(void *context) {
  DFlash *d = context;
  for (int i = 0; i < BUFFERS; i++)
    gpu_free(d->work->buffer + i);
  d->work->rows = d->work->blocks = 0;
  gpu_residency_commit();
}
DFlash *dflash_open(const char *directory, WeightSet *backbone, const Buf *embedding, const Linear *head) {
  DFlash *d = xmalloc(sizeof *d);
  d->work = &d->first;
  d->embedding = embedding;
  d->head = head;
  char path[4096], name[256];
  snprintf(path, sizeof path, "%s/config.json", directory);
  char *text = read_file(path, NULL);
  Json *root = json_parse(text);
  const Json *config = json_get(root, "dflash_config");
  const Json *layers = json_get(config, "target_layer_ids"), *rope = json_get(root, "rope_parameters");
  if (!layers || layers->n != 6 || json_get(root, "hidden_size")->num != HIDDEN ||
      json_get(root, "num_hidden_layers")->num != 6 || json_get(config, "selector_rank")->num != RANK ||
      json_get(config, "selector_top_k")->num != TOP || json_get(config, "conv_group_size")->num != 16 ||
      json_get(config, "conv_kernel_size")->num != 2)
    die("unsupported DFlash2 checkpoint dimensions");
  for (int i = 0; i < LAYERS; i++)
    d->target_layers[i] = (int)layers->kid[i].num;
  d->eps = (float)json_get(root, "rms_norm_eps")->num;
  d->theta = rope ? (float)json_get(rope, "rope_theta")->num : 10000;
  json_free(root);
  free(text);
  TensorIndex *index = st_open(directory);
  d->checkpoint = index;
  d->weights = weights_dependent(backbone, "DFlash2 weights");
  memory_slot_init(&d->workspace, weights_slot(backbone)->account, weights_slot(d->weights), "DFlash2 workspace",
                   release, d);
  linear_load(d->weights, &d->combine, index, "fc");
  linear_load(d->weights, &d->condition, index, "candidate_selector.hidden_projection");
  vec(d, index, &d->hidden_norm, "hidden_norm.weight");
  vec(d, index, &d->final_norm, "norm.weight");
  weights_tensor(d->weights, &d->predecessor, st_get(index, "candidate_selector.predecessor_codebook"), 0);
  weights_tensor(d->weights, &d->successor, st_get(index, "candidate_selector.successor_codebook"), 0);
  for (int l = 0; l < LAYERS; l++) {
    Block *b = d->layer + l;
#define P(field, suffix)                                                                                               \
  snprintf(name, sizeof name, "layers.%d.%s", l, suffix);                                                              \
  linear_load(d->weights, &b->field, index, name)
#define V(field, suffix)                                                                                               \
  snprintf(name, sizeof name, "layers.%d.%s", l, suffix);                                                              \
  vec(d, index, &b->field, name)
    P(q, "self_attn.q_proj");
    P(k, "self_attn.k_proj");
    P(v, "self_attn.v_proj");
    P(output, "self_attn.o_proj");
    P(gate, "mlp.gate_proj");
    P(up, "mlp.up_proj");
    P(down, "mlp.down_proj");
    V(input_norm, "input_layernorm.weight");
    V(output_norm, "post_attention_layernorm.weight");
    V(qnorm, "self_attn.q_norm.weight");
    V(knorm, "self_attn.k_norm.weight");
    P(attention_conv.coefficients, "attention_conv.kernel_projection");
    V(attention_conv.base, "attention_conv.base_kernel");
    P(mlp_conv.coefficients, "mlp_conv.kernel_projection");
    V(mlp_conv.base, "mlp_conv.base_kernel");
#undef P
#undef V
  }
  // Weight recipes retain file descriptors for subsequent reloads.
#define K(field, name) d->field = gpu_kernel(name)
  K(capture, "df_capture");
  K(norm, "group_rms_bf16");
  K(embed, "embed_rows");
  K(conv, "df_conv");
  K(rope, "df_rope");
  K(store, "nm_kv_store");
  K(attention, "nm_attention");
  K(sum, "nm_attention_sum");
  K(add, "nm_add");
  K(swiglu, "df_swiglu");
#undef K
  return d;
}
PassVariable dflash_pass_variable(DFlash *d) { return (PassVariable){&d->work, sizeof d->work}; }
void dflash_pass_begin(DFlash *d) {
  if (d) d->work = xmalloc(sizeof *d->work);
}
void dflash_pass_end(DFlash *d) {
  if (!d) return;
  for (int i = 0; i < BUFFERS; i++)
    gpu_free(d->work->buffer + i);
  free(d->work);
}
int dflash_load(DFlash *d) {
  return !d || weights_load(d->weights);
}
void dflash_active(DFlash *d, int active) {
  if (d) {
    memory_pin(weights_slot(d->weights), active);
    if (!active)
      memory_pin(&d->workspace, 0);
  }
}
uint64_t dflash_weight_bytes(const DFlash *d) {
  return d ? weights_resident_bytes(d->weights) : 0;
}
uint64_t dflash_required(const DFlash *d) {
  return d ? weights_required(d->weights) : 0;
}
uint64_t dflash_kv_bytes(int capacity) {
  return (uint64_t)2 * LAYERS * capacity * KV_HEADS * DIMENSION * 2;
}
int dflash_prepare(DFlash *d, int rows, int positions) {
  if (!d)
    return 1;
  int blocks = (positions + 255) / 256;
  if (blocks < 1)
    blocks = 1;
  if (rows < 16)
    rows = 16;
  size_t r = (size_t)rows * 4;
  size_t size[BUFFERS] = {r * HIDDEN,
                          r * HIDDEN,
                          r * HIDDEN,
                          r * 672,
                          r * HIDDEN,
                          r * 4096,
                          r * 256,
                          r * 256,
                          r * 256,
                          r * 4096,
                          16ull * HEADS * blocks * DIMENSION * 4,
                          16ull * HEADS * blocks * 8,
                          ((rows + 127) / 128) * 128ull * 16128 * 2,
                          r * 6144,
                          r * 6144,
                          r * 6144,
                          16ull * VOCAB * 4,
                          rows * 4ull,
                          rows * 8ull,
                          rows * sizeof(Sequence),
                          r * HIDDEN,
                          r * RANK};
  uint64_t bytes = 0;
  for (int i = 0; i < BUFFERS; i++)
    bytes += gpu_allocation_size(size[i]);
  MemorySlot *slots[] = {&d->workspace};
  if (!memory_admit(slots, &bytes, 1, MEMORY_REQUIRED))
    return 0;
  memory_pin(&d->workspace, 1);
  MemorySlot *before = memory_scope(&d->workspace);
  gpu_sync();
  for (int i = 0; i < BUFFERS; i++)
    if (d->work->buffer[i].size != size[i]) {
      gpu_free(d->work->buffer + i);
      d->work->buffer[i] = gpu_alloc(size[i]);
    }
  memory_scope(before);
  memory_trim(&d->workspace);
  d->work->rows = rows;
  d->work->blocks = blocks;
  return 1;
}
static void product(DFlash *d, const Linear *m, const Buf *x, size_t offset, const Buf *out, int rows, int rounding) {
  linear_product(m, x, offset, out, 0, rows, rows > 8, &d->work->buffer[PACKED], 0, rounding);
}
static void normalize(DFlash *d, const Buf *x, const Buf *weight, const Buf *out, int rows) {
  struct {
    uint32_t width, groups;
    float eps;
    uint32_t pad;
  } p = {HIDDEN, 1, d->eps, 0};
  gpu_kern(d->norm);
  gpu_buf(0, x, 0);
  gpu_buf(1, weight, 0);
  gpu_buf(2, out, 0);
  gpu_bytes(3, &p, sizeof p);
  gpu_go(rows, 1, 1, 256, 1, 1);
}
static void rotate(DFlash *d, const Buf *x, const Buf *weight, const Buf *out, int rows, int heads) {
  struct {
    uint32_t heads, dim, layer, base;
    float eps, theta;
    uint32_t pad[2];
  } p = {(uint32_t)heads, 128, 0, 0, d->eps, d->theta, {0}};
  gpu_kern(d->rope);
  gpu_buf(0, x, 0);
  gpu_buf(1, weight, 0);
  gpu_buf(2, out, 0);
  gpu_buf(3, &d->work->buffer[SEQUENCES], 0);
  gpu_bytes(4, &p, sizeof p);
  gpu_go(heads, rows, 1, 128, 1, 1);
}
static void sequences(DFlash *d, const DFlashRow *rows, int count) {
  Sequence *out = (Sequence *)d->work->buffer[SEQUENCES].ptr;
  for (int i = 0; i < count; i++)
    out[i] = (Sequence){.k = rows[i].kv->gpu,
                        .v = rows[i].kv->gpu + dflash_kv_bytes(rows[i].capacity) / 2,
                        .pos = (uint32_t)rows[i].position,
                        .cap = (uint32_t)rows[i].capacity};
}
static void store_keys(DFlash *d, int layer, int rows) {
  uint32_t p[2] = {256, (uint32_t)layer};
  gpu_kern(d->store);
  gpu_buf(0, &d->work->buffer[KR], 0);
  gpu_buf(1, &d->work->buffer[V], 0);
  gpu_buf(2, &d->work->buffer[SEQUENCES], 0);
  gpu_bytes(3, p, sizeof p);
  gpu_go(1, rows, 1, 256, 1, 1);
}
void dflash_capture(DFlash *d, int layer, const Buf *hidden, const Buf *captured, int rows) {
  if (!d)
    return;
  int at = -1;
  for (int i = 0; i < LAYERS; i++)
    if (d->target_layers[i] == layer)
      at = i;
  if (at < 0)
    return;
  uint32_t p[3] = {HIDDEN, LAYERS, (uint32_t)at};
  gpu_kern(d->capture);
  gpu_buf(0, hidden, 0);
  gpu_buf(1, captured, 0);
  gpu_bytes(2, p, sizeof p);
  gpu_go((HIDDEN + 255) / 256, rows, 1, 256, 1, 1);
}
void dflash_context(DFlash *d, const Buf *captured, const DFlashRow *rows, int count) {
  if (!d)
    return;
  sequences(d, rows, count);
  product(d, &d->combine, captured, 0, &d->work->buffer[CONTEXT], count, 1);
  normalize(d, &d->work->buffer[CONTEXT], &d->hidden_norm, &d->work->buffer[NORMAL], count);
  for (int l = 0; l < LAYERS; l++) {
    Block *b = d->layer + l;
    product(d, &b->k, &d->work->buffer[NORMAL], 0, &d->work->buffer[K], count, 1);
    product(d, &b->v, &d->work->buffer[NORMAL], 0, &d->work->buffer[V], count, 1);
    rotate(d, &d->work->buffer[K], &b->knorm, &d->work->buffer[KR], count, 2);
    store_keys(d, l, count);
  }
}
static void convolve(DFlash *d, const Convolution *conv, const Buf *x, const Buf *out, int rows, int side) {
  if (!side)
    product(d, &conv->coefficients, x, 0, &d->work->buffer[COEFFICIENTS], rows, 1);
  uint32_t p[4] = {HIDDEN, (uint32_t)rows, (uint32_t)side, 0};
  gpu_kern(d->conv);
  gpu_buf(0, x, 0);
  gpu_buf(1, &d->work->buffer[COEFFICIENTS], 0);
  gpu_buf(2, &conv->base, 0);
  gpu_buf(3, out, 0);
  gpu_bytes(4, p, sizeof p);
  gpu_go((HIDDEN + 255) / 256, rows, 1, 256, 1, 1);
}
static void add(DFlash *d, const Buf *out, const Buf *x, int rows) {
  uint32_t n = rows * HIDDEN;
  gpu_kern(d->add);
  gpu_buf(0, out, 0);
  gpu_buf(1, x, 0);
  gpu_bytes(2, &n, 4);
  gpu_go((n + 255) / 256, 1, 1, 256, 1, 1);
}
static void attention(DFlash *d, int layer, int rows, int length) {
  Block *b = d->layer + layer;
  product(d, &b->q, &d->work->buffer[CONVOLVED], 0, &d->work->buffer[Q], rows, 1);
  product(d, &b->k, &d->work->buffer[CONVOLVED], 0, &d->work->buffer[K], rows, 1);
  product(d, &b->v, &d->work->buffer[CONVOLVED], 0, &d->work->buffer[V], rows, 1);
  rotate(d, &d->work->buffer[Q], &b->qnorm, &d->work->buffer[Q], rows, 32);
  rotate(d, &d->work->buffer[K], &b->knorm, &d->work->buffer[KR], rows, 2);
  store_keys(d, layer, rows);
  struct {
    uint32_t heads, kv, dim, layer, blocks, causal, length, pad;
    float scale;
  } p = {32, 2, 128, (uint32_t)layer, (uint32_t)d->work->blocks, 0, (uint32_t)length, 0, 1 / sqrtf(128)};
  gpu_kern(d->attention);
  gpu_buf(0, &d->work->buffer[Q], 0);
  gpu_buf(1, &d->work->buffer[SEQUENCES], 0);
  gpu_buf(2, &d->work->buffer[PARTS], 0);
  gpu_buf(3, &d->work->buffer[MAXIMA], 0);
  gpu_bytes(4, &p, sizeof p);
  gpu_go(32, d->work->blocks, rows, 256, 1, 1);
  gpu_kern(d->sum);
  gpu_buf(0, &d->work->buffer[PARTS], 0);
  gpu_buf(1, &d->work->buffer[MAXIMA], 0);
  gpu_buf(2, &d->work->buffer[SEQUENCES], 0);
  gpu_buf(3, &d->work->buffer[ATTENTION], 0);
  gpu_bytes(4, &p, sizeof p);
  gpu_go(32, rows, 1, 128, 1, 1);
  product(d, &b->output, &d->work->buffer[ATTENTION], 0, &d->work->buffer[RESULT], rows, 1);
}
static void top_candidates(const float *logits, int *ids, float *scores) {
  for (int k = 0; k < TOP; k++)
    ids[k] = 0, scores[k] = -INFINITY;
  for (int token = 0; token < VOCAB; token++)
    if (logits[token] > scores[TOP - 1]) {
      int at = TOP - 1;
      while (at && logits[token] > scores[at - 1]) {
        scores[at] = scores[at - 1];
        ids[at] = ids[at - 1];
        at--;
      }
      scores[at] = logits[token];
      ids[at] = token;
    }
}
void dflash_predict(DFlash *d, const DFlashRow *state, int anchor, int *tokens, int count) {
  int rows = count + 1;
  DFlashRow sequence[16];
  if (!dflash_prepare(d, rows, state->position + rows))
    die("DFlash2 workspace does not fit model-memory budget");
  for (int i = 0; i < rows; i++) {
    ((int *)d->work->buffer[TOKENS].ptr)[i] = i ? 990 : anchor;
    ((uint64_t *)d->work->buffer[EMBEDDINGS].ptr)[i] = 0;
    sequence[i] = *state;
    sequence[i].position += i;
  }
  sequences(d, sequence, rows);
  gpu_begin();
  uint32_t hidden = HIDDEN;
  gpu_kern(d->embed);
  gpu_buf(0, d->embedding, 0);
  gpu_buf(1, &d->work->buffer[TOKENS], 0);
  gpu_buf(2, &d->work->buffer[X], 0);
  gpu_bytes(3, &hidden, 4);
  gpu_buf(4, &d->work->buffer[EMBEDDINGS], 0);
  gpu_go((HIDDEN + 255) / 256, rows, 1, 256, 1, 1);
  for (int l = 0; l < LAYERS; l++) {
    Block *b = d->layer + l;
    normalize(d, &d->work->buffer[X], &b->input_norm, &d->work->buffer[NORMAL], rows);
    convolve(d, &b->attention_conv, &d->work->buffer[NORMAL], &d->work->buffer[CONVOLVED], rows, 0);
    attention(d, l, rows, state->position + rows);
    convolve(d, &b->attention_conv, &d->work->buffer[RESULT], &d->work->buffer[NORMAL], rows, 1);
    add(d, &d->work->buffer[X], &d->work->buffer[NORMAL], rows);
    normalize(d, &d->work->buffer[X], &b->output_norm, &d->work->buffer[NORMAL], rows);
    convolve(d, &b->mlp_conv, &d->work->buffer[NORMAL], &d->work->buffer[CONVOLVED], rows, 0);
    product(d, &b->gate, &d->work->buffer[CONVOLVED], 0, &d->work->buffer[GATE], rows, 1);
    product(d, &b->up, &d->work->buffer[CONVOLVED], 0, &d->work->buffer[UP], rows, 1);
    uint32_t n = rows * 6144;
    gpu_kern(d->swiglu);
    gpu_buf(0, &d->work->buffer[GATE], 0);
    gpu_buf(1, &d->work->buffer[UP], 0);
    gpu_buf(2, &d->work->buffer[HIDDEN_PRODUCT], 0);
    gpu_bytes(3, &n, 4);
    gpu_go((n + 255) / 256, 1, 1, 256, 1, 1);
    product(d, &b->down, &d->work->buffer[HIDDEN_PRODUCT], 0, &d->work->buffer[RESULT], rows, 1);
    convolve(d, &b->mlp_conv, &d->work->buffer[RESULT], &d->work->buffer[NORMAL], rows, 1);
    add(d, &d->work->buffer[X], &d->work->buffer[NORMAL], rows);
  }
  normalize(d, &d->work->buffer[X], &d->final_norm, &d->work->buffer[NORMAL], rows);
  product(d, d->head, &d->work->buffer[NORMAL], HIDDEN * 4, &d->work->buffer[LOGITS], count, 1);
  product(d, &d->condition, &d->work->buffer[NORMAL], HIDDEN * 4, &d->work->buffer[CONDITION], count, 1);
  gpu_commit();
  gpu_sync();
  int candidates[15][TOP], previous[15][TOP];
  float unary[15][TOP], score[TOP] = {0};
  const float *condition = (const float *)d->work->buffer[CONDITION].ptr;
  const uint16_t *pre = (const uint16_t *)d->predecessor.ptr, *post = (const uint16_t *)d->successor.ptr;
  for (int step = 0; step < count; step++) {
    top_candidates((const float *)d->work->buffer[LOGITS].ptr + (size_t)step * VOCAB, candidates[step], unary[step]);
    float next[TOP];
    for (int current = 0; current < TOP; current++) {
      next[current] = -INFINITY;
      for (int old = 0; old < (step ? TOP : 1); old++) {
        int from = step ? candidates[step - 1][old] : anchor, to = candidates[step][current];
        float edge = 0;
        for (int k = 0; k < RANK; k++)
          edge += rounded(value(pre[(size_t)from * RANK + k]) * condition[step * RANK + k]) *
                  value(post[(size_t)to * RANK + k]);
        float total = (step ? score[old] : 0) + unary[step][current] + edge;
        if (total > next[current])
          next[current] = total, previous[step][current] = old;
      }
    }
    memcpy(score, next, sizeof score);
  }
  int best = 0;
  for (int k = 1; k < TOP; k++)
    if (score[k] > score[best])
      best = k;
  for (int step = count - 1; step >= 0; step--) {
    tokens[step] = candidates[step][best];
    best = previous[step][best];
  }
}
