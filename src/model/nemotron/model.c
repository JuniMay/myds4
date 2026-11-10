// Nemotron-H language backbone for the Transcribe checkpoint.
#include "../model.h"
#include "../experts.h"
#include "audio.h"
#include "dflash.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

enum { NM_ROWS = 256, NM_BATCH = 8 };
typedef struct {
  int kind, index, number, table_valid;
  int moe;  // the expert store's layer (MoE blocks only)
  uint64_t table_generation;
  Buf norm, conv, conv_bias, decay, dt, skip, gate_norm, correction;
  Linear input, output, q, k, v, router;
  Expert shared, *experts;
} Layer;
typedef struct {
  Buf state, conv, kv, draft_kv, saved_state[15], saved_conv[15];
  int pos, cap, verified;
} State;
typedef struct {
  Buf state, conv;
  float *logits;
  int pos;
} Snapshot;
typedef struct {
  uint64_t state, saved_state, conv, saved_conv, k, v;
  uint32_t pos, cap;
  int32_t rope_t, rope_h, rope_w, pad;
} Sequence;
typedef struct {
  State *state;
  int token, save;
} Row;
static struct {
  int ready, layers, recurrent, attention, hidden, intermediate, shared, experts, chosen;
  int heads, kv_heads, head_dim, ssm_heads, ssm_dim, ssm_state, groups, channels, projected, vocab;
  int normalize_topk, stop[8], nstop, rows, blocks, dump_fd, draft_depth, prompt_pass;
  float eps, factor, dt_minimum;
  TensorIndex *checkpoint;
  Layer *layer;
  WeightSet *core;
  // The routed experts: the store reads each expert's tensors (files: two
  // spans, weights then scales) into a slot.
  ExpertStore *store;
  int moe_layers, store_ready;
  ExpertFiles files;
  ExpertRoutes routes;  // each decoded row's experts in the MoE layers (myds4_routes)
  Buf embedding, final_norm;
  Linear head;
  Buf tokens, embeddings, sequences, x, norm, quant, packed, projected_values, convolved, scan, gated;
  Buf q, k, v, attended, part, maxima, result, shared_result, routing, selected, coefficients;
  Buf gathered, up, logits, captured;
  ExpertWork routed;
  Buf resident_table, resident_counts;
  ModelCounters counters;
  ModelPlan plan;
  const float *last_logits;
  SpeechEncoder *speech;
  DFlash *draft;
  const Media *media;
  int media_count;
  int embed_kernel, norm_kernel, round_kernel, quant_kernel, conv_kernel, scan_kernel, gate_kernel;
  int relu_kernel, add_kernel, route_kernel, gather_kernel, store_kernel, attention_kernel, sum_kernel;
} nm;

static int integer(const Json *j, const char *name, int fallback) {
  const Json *v = json_get(j, name);
  return v && v->type == J_NUM ? (int)v->num : fallback;
}
static float number(const Json *j, const char *name, float fallback) {
  const Json *v = json_get(j, name);
  return v && v->type == J_NUM ? (float)v->num : fallback;
}
static void vector(Buf *out, const char *name, int exponential) {
  weights_vector(nm.core, out, st_get(nm.checkpoint, "%s", name), 0, exponential ? WEIGHT_EXP : 0);
}
// The shared expert is part of the backbone.
static void shared_expert(Expert *e, const char *prefix) {
  char name[256];
  snprintf(name, sizeof name, "%s.up_proj", prefix);
  linear_load(nm.core, &e->up, nm.checkpoint, name);
  snprintf(name, sizeof name, "%s.down_proj", prefix);
  linear_load(nm.core, &e->down, nm.checkpoint, name);
}
// A routed expert's matrices (up and down; squared ReLU has no gate), and
// their tensors at index 1 and 2 of its three (experts_files).
static void routed_expert(Expert *e, const char *prefix, Tensor *weight, Tensor *scales, Tensor *exceptions) {
  char name[256];
  *e = (Expert){0};
  snprintf(name, sizeof name, "%s.up_proj", prefix);
  linear_describe(&e->up, nm.checkpoint, name, &weight[1], &scales[1], &exceptions[1]);
  snprintf(name, sizeof name, "%s.down_proj", prefix);
  linear_describe(&e->down, nm.checkpoint, name, &weight[2], &scales[2], &exceptions[2]);
}
// The store of the routed experts, from their tensors ([moe_layers][experts][3]).
static void store_layout(const Tensor *weight, const Tensor *scales, const Tensor *exceptions) {
  const Expert *experts[nm.moe_layers];
  for (int l = 0; l < nm.layers; l++)
    if (nm.layer[l].kind == 2)
      experts[nm.layer[l].moe] = nm.layer[l].experts;
  experts_files(&nm.files, weight, scales, exceptions, experts, nm.moe_layers, nm.experts);
  nm.store = cache_create(&(ExpertLayout){.layers = nm.moe_layers, .scored_layers = nm.moe_layers,
                                          .experts = nm.experts, .chosen = nm.chosen, .parts = nm.files.parts,
                                          .span = nm.files.span, .spans = nm.files.spans,
                                          .extra = nm.files.extra, .transform = nm.files.extra ? experts_decode : NULL,
                                          .context = &nm.files});
  cache_memory(nm.store, weights_slot(nm.core));
  experts_store(&nm.routed, nm.store, &nm.files);
}
static void prepare(const char *directory) {
  char path[4096], name[256];
  snprintf(path, sizeof path, "%s/config.json", directory);
  char *text = read_file(path, NULL);
  Json *root = json_parse(text);
  const Json *c = json_get(root, "llm_config");
  if (!c)
    c = root;
  const Json *types = json_get(c, "layers_block_type");
  nm.layers = types ? types->n : 0;
  nm.hidden = integer(c, "hidden_size", 0);
  nm.intermediate = integer(c, "moe_intermediate_size", 0);
  nm.shared = integer(c, "moe_shared_expert_intermediate_size", 0);
  nm.experts = integer(c, "n_routed_experts", 0);
  nm.chosen = integer(c, "num_experts_per_tok", 0);
  nm.heads = integer(c, "num_attention_heads", 0);
  nm.kv_heads = integer(c, "num_key_value_heads", 0);
  nm.head_dim = integer(c, "head_dim", 0);
  nm.ssm_heads = integer(c, "mamba_num_heads", 0);
  nm.ssm_dim = integer(c, "mamba_head_dim", 0);
  nm.ssm_state = integer(c, "ssm_state_size", 0);
  nm.groups = integer(c, "n_groups", 0);
  nm.vocab = integer(c, "vocab_size", 0);
  nm.factor = number(c, "routed_scaling_factor", 1);
  nm.eps = number(c, "layer_norm_epsilon", 1e-5f);
  const Json *normalize = json_get(c, "norm_topk_prob");
  nm.normalize_topk = normalize ? normalize->num != 0 : 1;
  nm.dt_minimum = number(c, "time_step_min", .001f);
  nm.channels = nm.ssm_heads * nm.ssm_dim + 2 * nm.groups * nm.ssm_state;
  nm.projected = nm.ssm_heads * nm.ssm_dim + nm.channels + nm.ssm_heads;
  if (nm.layers < 1 || nm.layers > 128 || nm.hidden != 2688 || nm.experts != 128 || nm.chosen < 1 || nm.chosen > 16 ||
      nm.head_dim != 128 || nm.heads != 32 || nm.kv_heads != 2 || nm.ssm_heads != 64 || nm.ssm_dim != 64 ||
      nm.ssm_state != 128 || nm.groups != 8 || integer(c, "conv_kernel", 4) != 4)
    die("unsupported Nemotron-H checkpoint dimensions");
  gpu_start();
  linear_init();
  experts_init(&nm.routed, nm.hidden, nm.intermediate, nm.experts, nm.chosen, EXPERT_RELU2, EXPERT_SUM_FP32);
#define K(field, name) nm.field = gpu_kernel(name)
  K(embed_kernel, "embed_rows");
  K(norm_kernel, "group_rms_bf16");
  K(round_kernel, "round_bf16");
  K(quant_kernel, "quantize_input");
  K(conv_kernel, "nm_conv");
  K(scan_kernel, "nm_scan");
  K(gate_kernel, "nm_gate_norm");
  K(relu_kernel, "nm_relu2");
  K(add_kernel, "nm_add");
  K(route_kernel, "nm_route");
  K(gather_kernel, "gather_rows");
  K(store_kernel, "nm_kv_store");
  K(attention_kernel, "nm_attention");
  K(sum_kernel, "nm_attention_sum");
#undef K
  nm.checkpoint = st_open(directory);
  nm.core = weights_new(memory_current()->account, "Nemotron backbone");
  nm.speech = speech_open(nm.checkpoint, root);
  weights_tensor(nm.core, &nm.embedding, st_get(nm.checkpoint, "backbone.embeddings.weight"), 0);
  vector(&nm.final_norm, "backbone.norm_f.weight", 0);
  linear_load(nm.core, &nm.head, nm.checkpoint, "lm_head");
  nm.layer = xmalloc((size_t)nm.layers * sizeof *nm.layer);
  int moe_layers = 0;
  for (int l = 0; l < nm.layers; l++)
    moe_layers += !strcmp(types->kid[l].s, "moe");
  size_t n_tensors = (size_t)moe_layers * nm.experts * 3 + 1;  // each expert's gate, up and down
  Tensor *weight = xmalloc(n_tensors * sizeof *weight), *scales = xmalloc(n_tensors * sizeof *scales),
         *exceptions = xmalloc(n_tensors * sizeof *exceptions);
  memset(weight, 0, n_tensors * sizeof *weight), memset(scales, 0, n_tensors * sizeof *scales);
  memset(exceptions, 0, n_tensors * sizeof *exceptions);
  for (int l = 0; l < nm.layers; l++) {
    Layer *y = nm.layer + l;
    y->number = l;
    const char *type = types->kid[l].s;
    y->kind = !strcmp(type, "mamba") || !strcmp(type, "linear_attention")     ? 0
              : !strcmp(type, "attention") || !strcmp(type, "full_attention") ? 1
              : !strcmp(type, "moe")                                          ? 2
                                                                              : -1;
    snprintf(name, sizeof name, "backbone.layers.%d.norm.weight", l);
    vector(&y->norm, name, 0);
#define MAT(field, suffix)                                                                                             \
  snprintf(name, sizeof name, "backbone.layers.%d.mixer.%s", l, suffix);                                               \
  linear_load(nm.core, &y->field, nm.checkpoint, name)
#define VEC(field, suffix, exp)                                                                                        \
  snprintf(name, sizeof name, "backbone.layers.%d.mixer.%s", l, suffix);                                               \
  vector(&y->field, name, exp)
    if (y->kind == 0) {
      y->index = nm.recurrent++;
      MAT(input, "in_proj");
      MAT(output, "out_proj");
      VEC(conv, "conv1d.weight", 0);
      VEC(conv_bias, "conv1d.bias", 0);
      VEC(decay, "A_log", 1);
      VEC(dt, "dt_bias", 0);
      VEC(skip, "D", 0);
      VEC(gate_norm, "norm.weight", 0);
    } else if (y->kind == 1) {
      y->index = nm.attention++;
      MAT(q, "q_proj");
      MAT(k, "k_proj");
      MAT(v, "v_proj");
      MAT(output, "o_proj");
    } else if (y->kind == 2) {
      MAT(router, "gate");
      VEC(correction, "gate.e_score_correction_bias", 0);
      snprintf(name, sizeof name, "backbone.layers.%d.mixer.shared_experts", l);
      shared_expert(&y->shared, name);
      y->moe = nm.moe_layers++;
      y->experts = xmalloc((size_t)nm.experts * sizeof *y->experts);
      for (int e = 0; e < nm.experts; e++) {
        snprintf(name, sizeof name, "backbone.layers.%d.mixer.experts.%d", l, e);
        size_t at = ((size_t)y->moe * nm.experts + e) * 3;
        routed_expert(y->experts + e, name, weight + at, scales + at, exceptions + at);
      }
    } else
      die("unsupported Nemotron layer type: %s", type);
#undef MAT
#undef VEC
  }
  store_layout(weight, scales, exceptions);
  experts_routes_init(&nm.routes, NM_BATCH, nm.moe_layers, nm.chosen);
  free(weight), free(scales), free(exceptions);
  snprintf(path, sizeof path, "%s/generation_config.json", directory);
  if (!access(path, R_OK)) {
    char *source = read_file(path, NULL);
    Json *generation = json_parse(source);
    const Json *stop = json_get(generation, "eos_token_id");
    for (int i = 0; stop && i < (stop->type == J_ARR ? stop->n : 1) && i < 8; i++)
      nm.stop[nm.nstop++] = (int)(stop->type == J_ARR ? stop->kid[i].num : stop->num);
    json_free(generation);
    free(source);
  }
  if (!nm.nstop)
    nm.stop[nm.nstop++] = integer(c, "eos_token_id", 2);
  json_free(root);
  free(text);
  nm.ready = 1;
}
static uint64_t recurrent_bytes(void) {
  return (uint64_t)nm.recurrent * nm.ssm_heads * nm.ssm_dim * nm.ssm_state * 4;
}
static uint64_t convolution_bytes(void) {
  return (uint64_t)nm.recurrent * nm.channels * 4 * 4;
}
static uint64_t kv_bytes(int cap) {
  return (uint64_t)2 * nm.attention * cap * nm.kv_heads * nm.head_dim * 2;
}
static Buf zero_buffer(size_t bytes, int kv) {
  Buf b = kv ? gpu_alloc_kv(bytes) : gpu_alloc(bytes);
  memset(b.ptr, 0, bytes);
  return b;
}
static uint64_t workspace(int rows, int blocks, int allocate) {
  size_t r = (size_t)rows * 4;
  uint64_t bytes = 0;
#define B(field, amount)                                                                                               \
  do {                                                                                                                 \
    size_t need = (amount);                                                                                            \
    bytes += gpu_allocation_size(need);                                                                                \
    if (allocate && nm.field.size != need) {                                                                           \
      gpu_free(&nm.field);                                                                                             \
      nm.field = gpu_alloc(need);                                                                                      \
    }                                                                                                                  \
  } while (0)
  B(resident_table, (size_t)nm.layers * nm.experts * sizeof(ExpertGroup));
  B(resident_counts, nm.layers * 4);
  B(tokens, rows * 4);
  B(embeddings, rows * 8);
  B(sequences, rows * sizeof(Sequence));
  B(x, r * nm.hidden);
  B(norm, r * nm.hidden);
  B(quant, r * nm.projected);
  B(packed, (size_t)((rows + 127) / 128) * 128 * nm.projected * 2);
  B(projected_values, r * nm.projected);
  B(convolved, r * nm.channels);
  B(scan, r * nm.ssm_heads * nm.ssm_dim);
  B(gated, r * nm.ssm_heads * nm.ssm_dim);
  B(q, r * nm.heads * nm.head_dim);
  B(k, r * nm.kv_heads * nm.head_dim);
  B(v, r * nm.kv_heads * nm.head_dim);
  B(attended, r * nm.heads * nm.head_dim);
  B(part, r * nm.heads * blocks * nm.head_dim);
  B(maxima, r * nm.heads * blocks * 2);
  B(result, r * nm.hidden);
  B(shared_result, r * nm.hidden);
  B(routing, r * nm.experts);
  B(selected, (size_t)nm.layers * rows * nm.chosen * 4);  // every layer's, kept for the pass (experts_selection)
  B(coefficients, r * nm.chosen);
  B(gathered, r * nm.hidden);
  B(up, r * (nm.shared > nm.intermediate ? nm.shared : nm.intermediate));
  B(logits, (size_t)(nm.draft_depth + 1 > NM_BATCH ? nm.draft_depth + 1 : NM_BATCH) * nm.vocab * 4);
  B(captured, nm.draft_depth ? r * nm.hidden * 6 : 0);
#undef B
  bytes += experts_workspace(&nm.routed, rows, allocate);
  if (allocate)
    nm.rows = rows, nm.blocks = blocks;
  return bytes;
}
static uint64_t workspace_bytes(void) {
  Buf *buffers[] = {&nm.resident_table, &nm.resident_counts, &nm.tokens,
                    &nm.embeddings,
                    &nm.sequences,
                    &nm.x,
                    &nm.norm,
                    &nm.quant,
                    &nm.packed,
                    &nm.projected_values,
                    &nm.convolved,
                    &nm.scan,
                    &nm.gated,
                    &nm.q,
                    &nm.k,
                    &nm.v,
                    &nm.attended,
                    &nm.part,
                    &nm.maxima,
                    &nm.result,
                    &nm.shared_result,
                    &nm.routing,
                    &nm.selected,
                    &nm.coefficients,
                    &nm.gathered,
                    &nm.up,
                    &nm.logits,
                    &nm.captured};
  uint64_t bytes = experts_bytes(&nm.routed);
  for (size_t i = 0; i < sizeof buffers / sizeof *buffers; i++)
    bytes += buffers[i]->allocated;
  return bytes;
}
static int prepare_rows(int rows, int positions) {
  uint64_t fixed = memory_current()->payload - workspace_bytes();
  int blocks = (positions + 255) / 256;
  if (blocks < 1)
    blocks = 1;
  for (;;) {
    cache_working_rows(nm.store, rows);  // the store's fewest slots: two layers' selections
    uint64_t experts = cache_required(nm.store, nm.moe_layers);
    if (memory_minimum(weights_slot(nm.core)->account, weights_required(nm.core) + dflash_required(nm.draft) + fixed +
                                                           workspace(rows, blocks, 0) + experts)) {
      gpu_sync();
      workspace(rows, blocks, 1);
      return rows;
    }
    if (rows == 1)
      return 0;
    rows = (rows + 1) / 2;
  }
}
static void reserve(State *s, int positions) {
  if (positions <= s->cap)
    return;
  int cap = s->cap ? s->cap : 1024;
  while (cap < positions)
    cap *= 2;
  gpu_sync();
  Buf next = gpu_alloc_kv(kv_bytes(cap));
  size_t row = (size_t)nm.kv_heads * nm.head_dim * 2;
  if (s->cap)
    for (int l = 0; l < 2 * nm.attention; l++)
      memcpy(next.ptr + (size_t)l * cap * row, s->kv.ptr + (size_t)l * s->cap * row, (size_t)s->pos * row);
  gpu_free(&s->kv);
  s->kv = next;
  if (nm.draft) {
    Buf draft = gpu_alloc_kv(dflash_kv_bytes(cap));
    if (s->cap)
      for (int l = 0; l < 12; l++)
        memcpy(draft.ptr + (size_t)l * cap * row, s->draft_kv.ptr + (size_t)l * s->cap * row, (size_t)s->pos * row);
    gpu_free(&s->draft_kv);
    s->draft_kv = draft;
  }
  s->cap = cap;
}
static ModelState *state_new(void) {
  State *s = xmalloc(sizeof *s);
  s->state = zero_buffer(recurrent_bytes(), 1);
  s->conv = zero_buffer(convolution_bytes(), 1);
  reserve(s, 1);
  for (int i = 0; i < nm.draft_depth; i++) {
    s->saved_state[i] = gpu_alloc_kv(recurrent_bytes());
    s->saved_conv[i] = gpu_alloc_kv(convolution_bytes());
  }
  return (ModelState *)s;
}
static void state_free(ModelState *state) {
  State *s = (State *)state;
  gpu_sync();
  gpu_free(&s->state);
  gpu_free(&s->conv);
  gpu_free(&s->kv);
  gpu_free(&s->draft_kv);
  for (int i = 0; i < nm.draft_depth; i++) {
    gpu_free(&s->saved_state[i]);
    gpu_free(&s->saved_conv[i]);
  }
  free(s);
}
static size_t state_bytes(const ModelState *state, int positions) {
  const State *s = (const State *)state;
  int cap = s && s->cap ? s->cap : 1024;
  while (cap < positions)
    cap *= 2;
  return (nm.draft_depth + 1) * (gpu_allocation_size(recurrent_bytes()) + gpu_allocation_size(convolution_bytes())) +
         gpu_allocation_size(kv_bytes(cap)) + (nm.draft ? gpu_allocation_size(dflash_kv_bytes(cap)) : 0);
}
static void state_copy(ModelState *destination, const ModelState *source, int positions) {
  State *d = (State *)destination;
  const State *s = (const State *)source;
  reserve(d, positions);
  gpu_sync();
  memcpy(d->state.ptr, s->state.ptr, recurrent_bytes());
  memcpy(d->conv.ptr, s->conv.ptr, convolution_bytes());
  size_t row = (size_t)nm.kv_heads * nm.head_dim * 2;
  for (int l = 0; l < 2 * nm.attention; l++)
    memcpy(d->kv.ptr + (size_t)l * d->cap * row, s->kv.ptr + (size_t)l * s->cap * row, (size_t)positions * row);
  if (nm.draft)
    for (int l = 0; l < 12; l++)
      memcpy(d->draft_kv.ptr + (size_t)l * d->cap * row, s->draft_kv.ptr + (size_t)l * s->cap * row,
             (size_t)positions * row);
  d->pos = positions;
}
static ModelSnapshot *snapshot_take(ModelState *state, int position, int logits) {
  State *s = (State *)state;
  Snapshot *x = xmalloc(sizeof *x);
  gpu_sync();
  x->state = gpu_alloc_kv(recurrent_bytes());
  x->conv = gpu_alloc_kv(convolution_bytes());
  memcpy(x->state.ptr, s->state.ptr, recurrent_bytes());
  memcpy(x->conv.ptr, s->conv.ptr, convolution_bytes());
  x->pos = position;
  if (logits) {
    x->logits = memory_host_required((size_t)nm.vocab * 4, 1);
    memcpy(x->logits, nm.last_logits, (size_t)nm.vocab * 4);
  }
  return (ModelSnapshot *)x;
}
static void snapshot_restore(ModelState *state, const ModelSnapshot *snapshot) {
  State *s = (State *)state;
  const Snapshot *x = (const Snapshot *)snapshot;
  gpu_sync();
  memcpy(s->state.ptr, x->state.ptr, recurrent_bytes());
  memcpy(s->conv.ptr, x->conv.ptr, convolution_bytes());
  s->pos = x->pos;
}
static void snapshot_free(ModelSnapshot *snapshot) {
  Snapshot *x = (Snapshot *)snapshot;
  gpu_free(&x->state);
  gpu_free(&x->conv);
  memory_host_free(x->logits);
  free(x);
}
static size_t snapshot_bytes(int logits) {
  return gpu_allocation_size(recurrent_bytes()) + gpu_allocation_size(convolution_bytes()) +
         (logits ? (size_t)nm.vocab * 4 : 0);
}
static const float *snapshot_logits(const ModelSnapshot *s) {
  return ((const Snapshot *)s)->logits;
}
static void unary(int kernel, const Buf *values, uint32_t count) {
  gpu_kern(kernel);
  gpu_buf(0, values, 0);
  gpu_bytes(1, &count, 4);
  gpu_go((count + 255) / 256, 1, 1, 256, 1, 1);
}
static void add(const Buf *target, const Buf *value, uint32_t count) {
  gpu_kern(nm.add_kernel);
  gpu_buf(0, target, 0);
  gpu_buf(1, value, 0);
  gpu_bytes(2, &count, 4);
  gpu_go((count + 255) / 256, 1, 1, 256, 1, 1);
}
static void norm(const Buf *input, size_t offset, const Buf *weight, const Buf *output, int rows) {
  struct {
    uint32_t width, groups;
    float eps;
    uint32_t pad;
  } p = {(uint32_t)nm.hidden, 1, nm.eps, 0};
  gpu_kern(nm.norm_kernel);
  gpu_buf(0, input, offset);
  gpu_buf(1, weight, 0);
  gpu_buf(2, output, 0);
  gpu_bytes(3, &p, sizeof p);
  gpu_go(rows, 1, 1, 256, 1, 1);
}
static void product(const Linear *matrix, const Buf *input, const Buf *output, int rows, int round, int quantize) {
  const Buf *x = input;
  Linear scaled = *matrix;
  if (quantize && matrix->kind != 'b') {
    struct {
      uint32_t count, bits;
      float scale;
      uint32_t pad;
    } p = {(uint32_t)(rows * matrix->cols),
           matrix->kind == 'n'   ? 4u
           : matrix->kind == 'm' ? 8u
                                 : 9u,
           matrix->input_scale, 0};
    gpu_kern(nm.quant_kernel);
    gpu_buf(0, input, 0);
    gpu_buf(1, &nm.quant, 0);
    gpu_bytes(2, &p, sizeof p);
    gpu_go((p.count + 255) / 256, 1, 1, 256, 1, 1);
    x = &nm.quant;
    if (matrix->kind != 'm')
      scaled.scale *= matrix->input_scale;
  }
  if (nm.prompt_pass || rows <= 8)
    linear_product(&scaled, x, 0, output, 0, rows, rows > 8, &nm.packed, 0, round);
  else
    for (int at = 0; at < rows; at += 8) {
      int count = rows - at < 8 ? rows - at : 8;
      linear_product(&scaled, x, (size_t)at * matrix->cols * 4, output, (size_t)at * matrix->rows * 4, count, 0,
                     &nm.packed, 0, round);
    }
}
static void recurrent(const Layer *y, int rows) {
  product(&y->input, &nm.norm, &nm.projected_values, rows, 1, 1);
  uint32_t cp[4] = {(uint32_t)rows, (uint32_t)nm.channels, (uint32_t)(nm.ssm_heads * nm.ssm_dim), (uint32_t)y->index};
  gpu_kern(nm.conv_kernel);
  gpu_buf(0, &nm.projected_values, 0);
  gpu_buf(1, &y->conv, 0);
  gpu_buf(2, &y->conv_bias, 0);
  gpu_buf(3, &nm.sequences, 0);
  gpu_buf(4, &nm.convolved, 0);
  gpu_bytes(5, cp, sizeof cp);
  gpu_go((nm.channels + 255) / 256, 1, 1, 256, 1, 1);
  struct {
    uint32_t rows, heads, groups, dim, state, layer, channels, projected;
    float minimum;
  } sp = {(uint32_t)rows,         64,           8, 64, 128, (uint32_t)y->index, (uint32_t)nm.channels,
          (uint32_t)nm.projected, nm.dt_minimum};
  gpu_kern(nm.scan_kernel);
  gpu_buf(0, &nm.convolved, 0);
  gpu_buf(1, &nm.projected_values, 0);
  gpu_buf(2, &y->decay, 0);
  gpu_buf(3, &y->skip, 0);
  gpu_buf(4, &y->dt, 0);
  gpu_buf(5, &nm.sequences, 0);
  gpu_buf(6, &nm.scan, 0);
  gpu_bytes(7, &sp, sizeof sp);
  gpu_go(64, 16, 1, 128, 1, 1);
  struct {
    uint32_t width, groups, stride, pad;
    float eps;
  } gp = {512, 8, (uint32_t)nm.projected, 0, nm.eps};
  gpu_kern(nm.gate_kernel);
  gpu_buf(0, &nm.scan, 0);
  gpu_buf(1, &nm.projected_values, 0);
  gpu_buf(2, &y->gate_norm, 0);
  gpu_buf(3, &nm.gated, 0);
  gpu_bytes(4, &gp, sizeof gp);
  gpu_go(rows * 8, 1, 1, 256, 1, 1);
  product(&y->output, &nm.gated, &nm.result, rows, 1, 1);
}
static void attention(const Layer *y, int rows) {
  product(&y->q, &nm.norm, &nm.q, rows, 1, 1);
  product(&y->k, &nm.norm, &nm.k, rows, 1, 1);
  product(&y->v, &nm.norm, &nm.v, rows, 1, 1);
  uint32_t kp[2] = {(uint32_t)(nm.kv_heads * nm.head_dim), (uint32_t)y->index};
  gpu_kern(nm.store_kernel);
  gpu_buf(0, &nm.k, 0);
  gpu_buf(1, &nm.v, 0);
  gpu_buf(2, &nm.sequences, 0);
  gpu_bytes(3, kp, sizeof kp);
  gpu_go((kp[0] + 255) / 256, rows, 1, 256, 1, 1);
  struct {
    uint32_t heads, kv, dim, layer, blocks, causal, length, pad;
    float scale;
  } ap = {(uint32_t)nm.heads,
          (uint32_t)nm.kv_heads,
          (uint32_t)nm.head_dim,
          (uint32_t)y->index,
          (uint32_t)nm.blocks,
          1,
          0,
          0,
          1 / sqrtf(nm.head_dim)};
  gpu_kern(nm.attention_kernel);
  gpu_buf(0, &nm.q, 0);
  gpu_buf(1, &nm.sequences, 0);
  gpu_buf(2, &nm.part, 0);
  gpu_buf(3, &nm.maxima, 0);
  gpu_bytes(4, &ap, sizeof ap);
  gpu_go(nm.heads, nm.blocks, rows, 256, 1, 1);
  gpu_kern(nm.sum_kernel);
  gpu_buf(0, &nm.part, 0);
  gpu_buf(1, &nm.maxima, 0);
  gpu_buf(2, &nm.sequences, 0);
  gpu_buf(3, &nm.attended, 0);
  gpu_bytes(4, &ap, sizeof ap);
  gpu_go(nm.heads, rows, 1, 128, 1, 1);
  product(&y->output, &nm.attended, &nm.result, rows, 1, 1);
}
static void feed_forward(const Expert *e, const Buf *input, const Buf *output, int rows) {
  product(&e->up, input, &nm.up, rows, 1, 1);
  unary(nm.relu_kernel, &nm.up, (uint32_t)(rows * e->up.rows));
  product(&e->down, &nm.up, output, rows, 1, 1);
}
static int routed(Layer *y, int rows) {
  size_t offset = (size_t)y->number * nm.experts * sizeof(ExpertGroup);
  uint64_t generation;
  int resident = (nm.prompt_pass || rows <= 8) && cache_layer_ready(nm.store, y->moe, &generation);
  if (resident && y->table_generation != generation) {
    y->table_valid = experts_table(&nm.routed, y->experts, y->moe, &nm.resident_table, offset);
    y->table_generation = generation;
  }
  resident &= y->table_valid;
  Buf selected = experts_selection(&nm.selected, y->number, rows, nm.chosen);
  feed_forward(&y->shared, &nm.norm, &nm.shared_result, rows);
  product(&y->router, &nm.norm, &nm.routing, rows, 0, 0);
  struct {
    uint32_t experts, chosen;
    float factor;
    uint32_t normalize;
  } p = {128, (uint32_t)nm.chosen, nm.factor, (uint32_t)nm.normalize_topk};
  gpu_kern(nm.route_kernel);
  gpu_buf(0, &nm.routing, 0);
  gpu_buf(1, &y->correction, 0);
  gpu_buf(2, &selected, 0);
  gpu_buf(3, &nm.coefficients, 0);
  gpu_bytes(4, &p, sizeof p);
  gpu_go(rows, 1, 1, 128, 1, 1);
  if (resident) {
    experts_resident(&nm.routed, &nm.resident_table, offset, &selected, &nm.coefficients, &nm.norm,
                      &nm.resident_counts, y->number * 4, rows, nm.prompt_pass);
    experts_sum(&nm.routed, &nm.result);
    add(&nm.result, &nm.shared_result, rows * nm.hidden);
    return 2;
  }
  gpu_commit();
  gpu_sync();
  if (!nm.prompt_pass)
    experts_routes_record(&nm.routes, nm.store, y->moe, (const uint32_t *)selected.ptr, rows, 0);
  experts_acquire(&nm.routed, y->experts, y->moe, (const uint32_t *)selected.ptr,
                  (const float *)nm.coefficients.ptr, rows, nm.prompt_pass, cache_counters(nm.store));
  gpu_begin();
  experts_encode(&nm.routed, y->experts, &nm.norm);
  for (int i = 0; i < nm.routed.active; i++) {
    int e = nm.routed.active_ids[i];
    int count = experts_count(&nm.routed, e);
    if (experts_grouped(&nm.routed, e)) continue;
    uint32_t shape[2] = {(uint32_t)nm.hidden, (uint32_t)count};
    gpu_kern(nm.gather_kernel);
    gpu_buf(0, &nm.norm, 0);
    gpu_buf(1, &nm.routed.entries, experts_entry_offset(&nm.routed, e));
    gpu_buf(2, &nm.gathered, 0);
    gpu_bytes(3, shape, sizeof shape);
    gpu_go((shape[0] * shape[1] + 255) / 256, 1, 1, 256, 1, 1);
    Buf out = experts_output(&nm.routed, e);
    feed_forward(y->experts + e, &nm.gathered, &out, count);
  }
  experts_sum(&nm.routed, &nm.result);
  add(&nm.result, &nm.shared_result, rows * nm.hidden);
  return 1;
}
static void add_stats(CacheStats *to, CacheStats before, CacheStats after) {
  to->hits += after.hits - before.hits;
  to->misses += after.misses - before.misses;
  to->bytes += after.bytes - before.bytes;
  to->read_s += after.read_s - before.read_s;
  to->row_own += after.row_own - before.row_own;
  to->row_ssd += after.row_ssd - before.row_ssd;
  to->row_unique += after.row_unique - before.row_unique;
  to->evictions += after.evictions - before.evictions;
}
static const float *pass(const Row *rows, int count, int logits, int prompt) {
  nm.prompt_pass = prompt;
  int maximum = 1;
  for (int r = 0, repeat = 0; r < count; r++) {
    State *s = rows[r].state;
    repeat = r && rows[r - 1].state == s ? repeat + 1 : 0;
    int positions = s->pos + repeat + 1;
    reserve(s, positions);
    if (positions > maximum)
      maximum = positions;
  }
  if ((nm.rows < count || nm.blocks < (maximum + 255) / 256) && prepare_rows(count, maximum) < count)
    return NULL;
  if (!dflash_prepare(nm.draft, count, maximum))
    return NULL;
  Sequence *seq = (Sequence *)nm.sequences.ptr;
  for (int r = 0, repeat = 0; r < count; r++) {
    State *s = rows[r].state;
    repeat = r && rows[r - 1].state == s ? repeat + 1 : 0;
    int position = s->pos + repeat;
    ((int *)nm.tokens.ptr)[r] = rows[r].token < 0 ? 0 : rows[r].token;
    ((uint64_t *)nm.embeddings.ptr)[r] = rows[r].token < 0 ? media_embedding(nm.media, nm.media_count, position) : 0;
    seq[r] = (Sequence){.state = s->state.gpu,
                        .conv = s->conv.gpu,
                        .k = s->kv.gpu,
                        .v = s->kv.gpu + kv_bytes(s->cap) / 2,
                        .pos = (uint32_t)(s->pos + repeat),
                        .cap = (uint32_t)s->cap};
    if (rows[r].save)
      seq[r].saved_state = s->saved_state[rows[r].save - 1].gpu,
      seq[r].saved_conv = s->saved_conv[rows[r].save - 1].gpu;
  }
  double began = now(), gpu = gpu_busy;
  CacheStats *stats = cache_counters(nm.store), before = *stats;
  if (!prompt)
    experts_routes_begin(&nm.routes);
  int resident_layers[nm.layers], resident_count = 0;
  gpu_begin();
  uint32_t hidden = (uint32_t)nm.hidden;
  if (prompt)
    for (int i = 0; i < nm.media_count; i++)
      gpu_use(&nm.media[i].features);
  gpu_kern(nm.embed_kernel);
  gpu_buf(0, &nm.embedding, 0);
  gpu_buf(1, &nm.tokens, 0);
  gpu_buf(2, &nm.x, 0);
  gpu_bytes(3, &hidden, 4);
  gpu_buf(4, &nm.embeddings, 0);
  gpu_go((hidden + 255) / 256, count, 1, 256, 1, 1);
  for (int l = 0; l < nm.layers; l++) {
    Layer *y = nm.layer + l;
    nm.counters.prefill_layer = prompt ? l + 1 : 0;
    norm(&nm.x, 0, &y->norm, &nm.norm, count);
    if (!y->kind)
      recurrent(y, count);
    else if (y->kind == 1)
      attention(y, count);
    else {
      if (prompt)
        cache_pass(nm.store, y->moe);  // a prompt's reads evict earlier layers' experts first
      if (routed(y, count) == 2) resident_layers[resident_count++] = l;
    }
    add(&nm.x, &nm.result, count * nm.hidden);
    dflash_capture(nm.draft, l, &nm.x, &nm.captured, count);
    if (nm.dump_fd >= 0) {
      gpu_commit();
      gpu_sync();
      size_t bytes = (size_t)count * nm.hidden * 4;
      if (write(nm.dump_fd, nm.x.ptr, bytes) != (ssize_t)bytes)
        die("Nemotron trace write failed");
      gpu_begin();
    }
  }
  norm(&nm.x, (size_t)(count - logits) * nm.hidden * 4, &nm.final_norm, &nm.norm, logits);
  product(&nm.head, &nm.norm, &nm.logits, logits, 0, 0);
  if (nm.draft) {
    DFlashRow context[NM_ROWS];
    for (int r = 0; r < count; r++)
      context[r] = (DFlashRow){&rows[r].state->draft_kv, rows[r].state->cap, (int)seq[r].pos};
    dflash_context(nm.draft, &nm.captured, context, count);
  }
  gpu_commit();
  gpu_sync();
  // The store counts the experts reached through tables as hits; a prompt's,
  // before its pass ends, as a prompt's use.
  for (int i = 0; i < resident_count; i++) {
    const Layer *y = nm.layer + resident_layers[i];
    Buf selected = experts_selection(&nm.selected, y->number, count, nm.chosen);
    stats->row_unique += experts_mark_used(&nm.routed, y->experts, y->moe, (const uint32_t *)selected.ptr, count);
    if (!prompt)
      experts_routes_record(&nm.routes, nm.store, y->moe, (const uint32_t *)selected.ptr, count, 1);
    stats->row_own += (uint64_t)count * nm.chosen;
  }
  if (prompt)
    cache_pass(nm.store, -1);
  nm.counters.table_layers += resident_count;
  experts_release(&nm.routed);
  // The pass's GPU work has completed and no expert is held: between passes
  // the store may evict any slot, its new slots join the residency set (the
  // next pass reaches them through tables), and the scores age once.
  cache_adapt(nm.store);
  if (nm.dump_fd >= 0) {
    size_t bytes = (size_t)logits * nm.vocab * 4;
    if (write(nm.dump_fd, nm.logits.ptr, bytes) != (ssize_t)bytes)
      die("Nemotron logit trace write failed");
  }
  for (int i = 0; i < count; i++)
    rows[i].state->pos++;
  if (prompt) {
    nm.counters.prefill_rows += count;
    add_stats(&nm.counters.prefill, before, *stats);
  } else {
    nm.counters.decode_rows += count;
    nm.counters.decode_seconds += now() - began;
    nm.counters.decode_gpu_seconds += gpu_busy - gpu;
    add_stats(&nm.counters.decode, before, *stats);
  }
  nm.counters.prefill_layer = 0;
  return nm.last_logits = (const float *)nm.logits.ptr;
}
static const float *prefill(ModelState *state, const int *tokens, int n, int from, int replay, int capture,
                            ModelSnapshot **captured, float *every) {
  State *s = (State *)state;
  Row rows[NM_ROWS];
  const float *logits = NULL;
  nm.plan = (ModelPlan){.tokens = n - from, .decoder_tokens = n - from};
  if (captured)
    *captured = NULL;
  for (int at = from; at < n;) {
    int end = capture > at && capture < n ? capture : n;
    int requested = end - at, limit = every ? NM_BATCH : NM_ROWS;
    if (requested > limit)
      requested = limit;
    int count = prepare_rows(requested, at + requested);
    if (!count)
      return NULL;
    for (int i = 0; i < count; i++)
      rows[i] = (Row){s, tokens[at + i], 0};
    logits = pass(rows, count, every ? count : 1, 1);
    if (!logits)
      return NULL;
    if (every) {
      memcpy(every + (size_t)(at - from) * nm.vocab, logits, (size_t)count * nm.vocab * 4);
      logits += (size_t)(count - 1) * nm.vocab;
    }
    at += count;
    nm.plan.chunks++;
    if (at == capture)
      *captured = snapshot_take(state, at, 0);
    if (shutdown_requested())
      return NULL;
  }
  return nm.last_logits = logits;
}
static const float *forward_rows(ModelState *const *states, const int *tokens, const int *positions, int count) {
  Row rows[NM_BATCH];
  for (int i = 0; i < count; i++)
    rows[i] = (Row){(State *)states[i], tokens[i], 0};
  return pass(rows, count, count, 0);
}
static const char *refuse(const Options *o) {
  if (o->lookup)
    return "Nemotron currently requires lookup off";
  if (o->draft < 0 || o->draft > 15)
    return "Nemotron draft takes 0 to 15 proposals";
  if (o->draft && !o->nemotron.draft_model)
    return "Nemotron drafting needs draft-model pointing to DFlash2";
  return NULL;
}
static int load(const char *directory, const Options *options) {
  nm.dump_fd = options->dump_fd;
  if (!nm.ready)
    prepare(directory);
  nm.draft_depth = options->draft;
  if (nm.draft_depth && !nm.draft)
    nm.draft = dflash_open(options->nemotron.draft_model, nm.core, &nm.embedding, &nm.head);
  memory_pin(weights_slot(nm.core), 1);
  dflash_active(nm.draft, 1);
  int ok = prepare_rows(1, 1) && weights_load(nm.core) && dflash_load(nm.draft);
  if (ok && !nm.store_ready) {  // the store's slots and readers start once; a reload refills the slots
    cache_init(nm.store, 0, nm.moe_layers);
    nm.store_ready = 1;
  }
  memory_pin(weights_slot(nm.core), 0);
  dflash_active(nm.draft, 0);
  return ok;
}
static MemorySlot *weight_slot(void) {
  return nm.core ? weights_slot(nm.core) : NULL;
}
static int vocab(void) {
  return nm.vocab;
}
static int max_rows(void) {
  return NM_BATCH;
}
static int max_verify(void) {
  return nm.draft_depth + 1;
}
static int draft_block(void) {
  return nm.draft_depth;
}
static int zero(void) {
  return 0;
}
static int prefill_rows(void) {
  return NM_ROWS;
}
static int stop(int token) {
  for (int i = 0; i < nm.nstop; i++)
    if (token == nm.stop[i])
      return 1;
  return 0;
}
static int replay_rows(int position) {
  return -1;
}
static int replay_start(int position) {
  return position;
}
static void seek(ModelState *s, const int *tokens, int position) {
}
// Before the first load there is no store, or no slot in it.
static int idle(void) {
  if (!nm.store_ready)
    return 0;
  cache_idle(nm.store);
  return cache_slots(nm.store);
}
static uint64_t store_bytes(void) {
  return nm.store_ready ? cache_bytes(nm.store) : 0;
}
static void counters(ModelCounters *out) {
  *out = nm.counters;
  out->layers = nm.layers;
  out->experts_per_row = (nm.layers - nm.recurrent - nm.attention) * nm.chosen;
  out->routed_layers = nm.moe_layers;
  out->experts_per_layer = nm.experts;
  out->dense_bytes = weights_resident_bytes(nm.core) + speech_weight_bytes(nm.speech) + dflash_weight_bytes(nm.draft);
  if (!nm.store)
    return;
  out->cache = *cache_counters(nm.store);
  out->cache_bytes = store_bytes();
  out->cache_slots = cache_slots(nm.store);
  out->mapped = cache_mapped(nm.store);
  out->cap = cache_cap(nm.store);
  out->max_slots = cache_max(nm.store);
  out->experts = nm.moe_layers * nm.experts;
  out->slot_bytes = cache_slot_bytes(nm.store);
}
static int routes(int row, uint16_t *id, uint8_t *from) {
  return experts_routes_get(&nm.routes, row, id, from);
}
static void plan(ModelPlan *out) {
  *out = nm.plan;
}
static void chat(char *text, size_t capacity, const char *prompt, int images, int thinking) {
  size_t at = snprintf(text, capacity, "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n%s", prompt);
  for (int i = 0; i < images && at < capacity; i++)
    at += snprintf(text + at, capacity - at, " " MEDIA_MARK);
  if (at < capacity)
    snprintf(text + at, capacity - at, "<|im_end|>\n<|im_start|>assistant\n%s",
             thinking ? "<think>\n" : "<think></think>");
}
static int encode_media(Media *media) {
  return speech_encode(nm.speech, media);
}
static void media(const Media *input, int count) {
  nm.media = input;
  nm.media_count = count;
}
static void active(int on) {
  speech_reside(nm.speech, on);
  dflash_active(nm.draft, on);
  if (nm.store)
    cache_active(nm.store, on);
}
static const double *draft(ModelState *state, int token, int position, int *tokens, int count, Sampler *sampler) {
  State *s = (State *)state;
  reserve(s, position + count + 1);
  DFlashRow row = {&s->draft_kv, s->cap, position};
  dflash_predict(nm.draft, &row, token, tokens, count);
  return NULL;
}
static const float *verify(ModelState *state, const int *tokens, int count, int position) {
  State *s = (State *)state;
  Row rows[16];
  for (int i = 0; i < count; i++)
    rows[i] = (Row){s, tokens[i], i + 1 < count ? i + 1 : 0};
  s->verified = count;
  return pass(rows, count, count, 0);
}
static void accept(ModelState *state, const int *tokens, int count, int position) {
  State *s = (State *)state;
  if (count < s->verified) {
    gpu_sync();
    memcpy(s->state.ptr, s->saved_state[count - 1].ptr, recurrent_bytes());
    memcpy(s->conv.ptr, s->saved_conv[count - 1].ptr, convolution_bytes());
  }
  s->pos = position + count;
  nm.last_logits = (const float *)nm.logits.ptr + (size_t)(count - 1) * nm.vocab;
}
static const char *const types[] = {"nemo_speechlm", NULL};
const ModelOps nemotron_ops = {.family = "Nemotron 3.5 Transcribe",
                               .model_types = types,
                               .refuse = refuse,
                               .load = load,
                               .weight_slot = weight_slot,
                               .media_token = "<|audio|>",
                               .media_kind = MEDIA_AUDIO,
                               .prepare_media = speech_prepare,
                               .encode_media = encode_media,
                               .media = media,
                               .active = active,
                               .counters = counters,
                               .routes = routes,
                               .cache_bytes = store_bytes,
                               .plan = plan,
                               .vocab = vocab,
                               .max_rows = max_rows,
                               .max_verify = max_verify,
                               .draft_block = draft_block,
                               .rewind = zero,
                               .stop = stop,
                               .prefill_rows = prefill_rows,
                               .state_new = state_new,
                               .state_free = state_free,
                               .state_bytes = state_bytes,
                               .state_copy = state_copy,
                               .snapshot_take = snapshot_take,
                               .snapshot_restore = snapshot_restore,
                               .snapshot_free = snapshot_free,
                               .snapshot_bytes = snapshot_bytes,
                               .snapshot_logits = snapshot_logits,
                               .prefill = prefill,
                               .replay_rows = replay_rows,
                               .replay_start = replay_start,
                               .rows = forward_rows,
                               .seek = seek,
                               .idle = idle,
                               .chat = chat,
                               .draft = draft,
                               .verify = verify,
                               .accept = accept};
