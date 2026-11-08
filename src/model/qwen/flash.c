// Qwen3.8 Flash-Next: gated residual streams, GDN, sparse attention and NVFP4 experts.
#include "../model.h"
#include "../linear.h"
#include "../experts.h"
#include "vision.h"
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#include <dispatch/dispatch.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

enum { FLASH_ROWS = 1024, FLASH_BATCH = 8, INDEX_DIM = 128, ATT_BLOCK = 256 };
typedef struct {
  Linear down, up, inject;
  Buf norm;
} Hyper;
typedef struct {
  Linear key, value;
  Buf norm_key, norm_query, norm_conv, conv;
  Tensor *table, *scales;
  int parts, head_dim;
  int64_t shard_rows, multipliers[3], primes[16], offsets[16];
  float scale;
} Ple;
typedef struct {
  int recurrent, index, ple_index, number, table_valid;
  WeightPresence presence;
  uint64_t table_generation;
  Hyper attention_hyper, mlp_hyper;
  Linear qkv, z, a, b, output, q, k, v, index_qk;
  Buf conv, decay, bias, norm, qnorm, knorm, iqnorm, iknorm;
  Linear router, shared_gate;
  Expert shared, *experts;
  Ple *ple;
} FlashLayer;
typedef struct {
  Buf gdn, conv, kv, ple;
  float *index, *tail;
  int32_t *positions;
  int pos, cap, rope_delta, previous[2];
} FlashState;
typedef struct {
  Buf gdn, conv, ple;
  float *tail, *logits;
  int pos, rope_delta, previous[2];
} FlashSnapshot;
typedef struct {
  uint64_t gdn, gdn_save, conv, conv_save, k, v;
  uint32_t pos, cap;
  int32_t rope_t, rope_h, rope_w, pad;
} FlashSeq;
static_assert(sizeof(FlashSeq) == 72);
typedef struct {
  FlashState *state;
  int token;
} Row;
static struct {
  int ready, layers, recurrent, attention, hidden, inter, streams, rank, experts, chosen, normalize_topk, dump_fd,
      output_gate;
  int heads, kv_heads, head_dim, key_heads, value_heads, key_dim, value_dim, channels, rotary, section_h, section_w;
  int index_heads, index_budget, index_ratio, ple_count, vocab, eos, nstop, stop[8], image_id;
  float eps, theta;
  TensorIndex *index;
  WeightSet *core;
  WeightCache cache;
  FlashLayer *layer;
  Hyper final;
  Buf embedding;
  Linear head;
  uint64_t expert_minimum;
  Buf tokens, images, sequences, ple_states, x, normalized, mixed, mix_up, mix_down, injection, result;
  Buf xb, quantized, qkv, convolved, z, a, b, decay, beta, gdn, gated, qg, k, v, q, kr, attention_out;
  Buf index_qk, shared, shared_gate, routing, selected, coefficients, gathered, gate, up, intermediate, logits;
  ExpertWork routed;
  Buf resident_table, resident_counts;
  Buf ngram, ple_key, ple_query, ple_value, ple_gated, ple_normalized, attention_ids, part, ml;
  int workspace_rows;
  ModelCounters counters;
  ModelPlan plan;
  const float *last_logits;
  QwenVision *vision;
  const Media *image_list;
  int image_count;
  int round, embed, norm_group, hc_silu, hc_mix, hc_inject, swiglu, shared_sum, route, gather, quant;
  int conv_kernel, l2, gates, delta, norm_kernel, rope, store, attend, combine, ple_gate, ple_conv;
} f;

static double number(const Json *j, const char *name, double fallback) {
  const Json *v = json_get(j, name);
  return v && v->type == J_NUM ? v->num : fallback;
}
static int boolean(const Json *j, const char *name, int fallback) {
  const Json *v = json_get(j, name);
  return v ? v->num != 0 : fallback;
}
static void read_tensor(Tensor t, void *out, size_t bytes, size_t offset) {
  size_t done = 0;
  while (done < bytes) {
    ssize_t n = pread(t.fd, (uint8_t *)out + done, bytes - done, t.off + offset + done);
    if (n <= 0)
      die("cannot read Flash-Next tensor");
    done += (size_t)n;
  }
}
static void vector(Buf *out, const char *name, int centered, int exponential) {
  weights_vector(f.core, out, st_get(f.index, "%s", name), centered ? 1 : 0,
                 (centered ? WEIGHT_ADD : 0) | (exponential ? WEIGHT_EXP : 0));
}
static void matrix(WeightSet *set, Linear *out, const char *name) {
  linear_load(set, out, f.index, name);
}
static void hyper(Hyper *h, const char *stem, int inject) {
  char name[384];
  snprintf(name, sizeof name, "%s.hc_norm.weight", stem);
  vector(&h->norm, name, 1, 0);
  snprintf(name, sizeof name, "%s.input_mix_weight_down", stem);
  matrix(f.core, &h->down, name);
  snprintf(name, sizeof name, "%s.input_mix_weight_up", stem);
  matrix(f.core, &h->up, name);
  if (inject) {
    snprintf(name, sizeof name, "%s.block_inject_weight", stem);
    matrix(f.core, &h->inject, name);
  }
}
static void expert(Expert *e, const char *stem, WeightPresence *presence) {
  int cached = presence != NULL;
  char name[384];
  e->weights = cached ? weights_cached(f.core, &f.cache, "Flash-Next expert") : f.core;
  if (cached) weights_track(e->weights, presence);
  snprintf(name, sizeof name, "%s.gate_proj", stem);
  matrix(e->weights, &e->gate, name);
  snprintf(name, sizeof name, "%s.up_proj", stem);
  matrix(e->weights, &e->up, name);
  snprintf(name, sizeof name, "%s.down_proj", stem);
  matrix(e->weights, &e->down, name);
  if (cached && weights_required(e->weights) > f.expert_minimum)
    f.expert_minimum = weights_required(e->weights);
}
static void ple_load(Ple *p, const char *stem, int parts) {
  char name[384];
  snprintf(name, sizeof name, "%s.key_proj", stem);
  matrix(f.core, &p->key, name);
  snprintf(name, sizeof name, "%s.value_proj", stem);
  matrix(f.core, &p->value, name);
  snprintf(name, sizeof name, "%s.norm_key.weight", stem);
  vector(&p->norm_key, name, 1, 0);
  snprintf(name, sizeof name, "%s.norm_query.weight", stem);
  vector(&p->norm_query, name, 1, 0);
  snprintf(name, sizeof name, "%s.norm_conv.weight", stem);
  vector(&p->norm_conv, name, 1, 0);
  snprintf(name, sizeof name, "%s.conv1d.weight", stem);
  vector(&p->conv, name, 0, 0);
  Tensor t = st_get(f.index, "%s.ple_embedding.layer_multipliers", stem);
  read_tensor(t, p->multipliers, sizeof p->multipliers, 0);
  t = st_get(f.index, "%s.ple_embedding.ngram_heads_vocab_sizes", stem);
  read_tensor(t, p->primes, sizeof p->primes, 0);
  t = st_get(f.index, "%s.ple_embedding.ngram_heads_offsets", stem);
  read_tensor(t, p->offsets, sizeof p->offsets, 0);
  t = st_get(f.index, "%s.ple_embedding.ngram_embedding.weight_scale_2", stem);
  read_tensor(t, &p->scale, 4, 0);
  p->parts = parts;
  p->table = xmalloc((size_t)parts * sizeof *p->table);
  p->scales = xmalloc((size_t)parts * sizeof *p->scales);
  for (int i = 0; i < parts; i++) {
    p->table[i] = st_get(f.index, "%s.ple_embedding.ngram_embedding.shard_%d.weight", stem, i);
    p->scales[i] = st_get(f.index, "%s.ple_embedding.ngram_embedding.shard_%d.weight_scale", stem, i);
  }
  p->shard_rows = p->table[0].shape[0];
  p->head_dim = (int)p->table[0].shape[1] * 2;
  if (p->head_dim * 16 != f.hidden || p->head_dim > 256)
    die("unsupported Flash-Next n-gram width");
}
static void prepare(const char *directory) {
  char path[1024], stem[320], name[384];
  snprintf(path, sizeof path, "%s/config.json", directory);
  char *text = read_file(path, NULL);
  Json *root = json_parse(text);
  const Json *c = json_get(root, "text_config");
  if (!c)
    c = root;
  f.layers = number(c, "num_hidden_layers", 0);
  f.hidden = number(c, "hidden_size", 0);
  f.inter = number(c, "moe_intermediate_size", 0);
  f.streams = number(c, "hc_count", 4);
  f.rank = number(c, "hc_lowrank", 320);
  f.experts = number(c, "num_experts", 0);
  f.chosen = number(c, "num_experts_per_tok", 0);
  f.normalize_topk = boolean(c, "norm_topk_prob", 1);
  const Json *gate_type = json_get(c, "output_gate_type");
  f.output_gate = gate_type && gate_type->type == J_STR && !strcmp(gate_type->s, "sigmoid") ? 2 : 1;
  f.vocab = number(c, "vocab_size", 0);
  f.eos = number(c, "eos_token_id", 248044);
  f.heads = number(c, "num_attention_heads", 0);
  f.kv_heads = number(c, "num_key_value_heads", 0);
  f.head_dim = number(c, "head_dim", 256);
  f.key_heads = number(c, "linear_num_key_heads", 0);
  f.value_heads = number(c, "linear_num_value_heads", 0);
  f.key_dim = number(c, "linear_key_head_dim", 128);
  f.value_dim = number(c, "linear_value_head_dim", 128);
  f.channels = (2 * f.key_heads + f.value_heads) * 128;
  f.index_heads = number(c, "indexer_n_heads", 4);
  f.index_budget = number(c, "indexer_budget", 2048);
  f.index_ratio = number(c, "indexer_compress_ratio", 4);
  f.eps = number(c, "rms_norm_eps", 1e-6);
  f.image_id = number(root, "image_token_id", 248056);
  const Json *rope = json_get(c, "rope_parameters");
  f.theta = number(rope, "rope_theta", 10000000);
  f.rotary = f.head_dim * number(rope, "partial_rotary_factor", .25);
  const Json *sections = json_get(rope, "mrope_section");
  f.section_h = sections ? sections->kid[1].num : 11;
  f.section_w = sections ? sections->kid[2].num : 10;
  if (f.layers < 1 || f.layers > 128 || f.hidden < 1 || f.inter < 1 || f.experts < 1 || f.chosen < 1 ||
      f.chosen > f.experts || f.heads < 1 || f.kv_heads < 1 || f.key_heads < 1 || f.value_heads < 1 ||
      f.value_heads % f.key_heads || f.hidden % 64 || f.streams != 4 || f.experts > 512 || f.chosen > 16 ||
      f.head_dim != 256 || f.heads / f.kv_heads > 12 || f.key_dim != 128 || f.value_dim != 128 || f.index_ratio != 4 ||
      f.index_heads != 4 || f.index_budget % 4 || f.rotary > 128)
    die("unsupported Flash-Next checkpoint dimensions");
  gpu_start();
  linear_init();
  experts_init(&f.routed, f.hidden, f.inter, f.experts, f.chosen, EXPERT_SWIGLU, EXPERT_SUM_BF16);
#define K(field, name) f.field = gpu_kernel(name)
  K(round, "round_bf16");
  K(embed, "flash_embed");
  K(norm_group, "group_rms_bf16");
  K(hc_silu, "flash_hc_silu");
  K(hc_mix, "flash_hc_mix");
  K(hc_inject, "flash_hc_inject");
  K(swiglu, "flash_swiglu");
  K(shared_sum, "flash_shared");
  K(route, "flash_route");
  K(gather, "gather_rows");
  K(quant, "quantize_input");
  K(conv_kernel, "conv");
  K(l2, "l2_norm");
  K(gates, "gdn_gates");
  K(delta, "gdn");
  K(norm_kernel, "rms_norm");
  K(rope, "flash_rope");
  K(store, "flash_kv_store");
  K(attend, "flash_attention_block");
  K(combine, "flash_attention_combine");
  K(ple_gate, "flash_ple_gate");
  K(ple_conv, "flash_ple_conv");
#undef K
  f.index = st_open(directory);
  f.core = weights_new(memory_current()->account, "Flash-Next backbone");
  f.vision = qwen_vision_open(f.index, json_get(root, "vision_config"));
  weights_tensor(f.core, &f.embedding, st_get(f.index, "model.language_model.embed_tokens.weight"), 0);
  matrix(f.core, &f.head, "lm_head");
  hyper(&f.final, "model.language_model.hyper_connection_mixer", 0);
  f.layer = xmalloc((size_t)f.layers * sizeof *f.layer);
  const Json *types = json_get(c, "layer_types"), *ple_ids = json_get(c, "ple_layer_ids");
  if (!types || types->n != f.layers)
    die("Flash-Next layer types do not match the configured layer count");
  for (int l = 0; l < f.layers; l++) {
    FlashLayer *y = f.layer + l;
    y->number = l;
    y->ple_index = -1;
    y->recurrent = !strcmp(types->kid[l].s, "linear_attention");
    y->index = y->recurrent ? f.recurrent++ : f.attention++;
    snprintf(stem, sizeof stem, "model.language_model.layers.%d.attn_hyper_connection", l);
    hyper(&y->attention_hyper, stem, 1);
    snprintf(stem, sizeof stem, "model.language_model.layers.%d.mlp_hyper_connection", l);
    hyper(&y->mlp_hyper, stem, 1);
    snprintf(stem, sizeof stem, "model.language_model.layers.%d.mlp.gate", l);
    matrix(f.core, &y->router, stem);
    snprintf(stem, sizeof stem, "model.language_model.layers.%d.mlp.shared_expert_gate", l);
    matrix(f.core, &y->shared_gate, stem);
    snprintf(stem, sizeof stem, "model.language_model.layers.%d.mlp.shared_expert", l);
    expert(&y->shared, stem, 0);
    y->experts = xmalloc((size_t)f.experts * sizeof *y->experts);
    for (int e = 0; e < f.experts; e++) {
      snprintf(stem, sizeof stem, "model.language_model.layers.%d.mlp.experts.%d", l, e);
      expert(y->experts + e, stem, &y->presence);
    }
    if (y->recurrent) {
#define MAT(field, suffix)                                                                                             \
  snprintf(stem, sizeof stem, "model.language_model.layers.%d.linear_attn.%s", l, suffix);                             \
  matrix(f.core, &y->field, stem)
      MAT(qkv, "in_proj_qkv");
      MAT(z, "in_proj_z");
      MAT(a, "in_proj_a");
      MAT(b, "in_proj_b");
      MAT(output, "out_proj");
#undef MAT
#define VEC(field, suffix, exp)                                                                                        \
  snprintf(name, sizeof name, "model.language_model.layers.%d.linear_attn.%s", l, suffix);                             \
  vector(&y->field, name, 0, exp)
      VEC(conv, "conv1d.weight", 0);
      VEC(decay, "A_log", 1);
      VEC(bias, "dt_bias", 0);
      VEC(norm, "norm.weight", 0);
#undef VEC
    } else {
#define MAT(field, suffix)                                                                                             \
  snprintf(stem, sizeof stem, "model.language_model.layers.%d.self_attn.%s", l, suffix);                               \
  matrix(f.core, &y->field, stem)
      MAT(q, "q_proj");
      MAT(k, "k_proj");
      MAT(v, "v_proj");
      MAT(output, "o_proj");
      MAT(index_qk, "indexer.index_qk_proj");
#undef MAT
#define VEC(field, suffix)                                                                                             \
  snprintf(name, sizeof name, "model.language_model.layers.%d.self_attn.%s", l, suffix);                               \
  vector(&y->field, name, 1, 0)
      VEC(qnorm, "q_norm.weight");
      VEC(knorm, "k_norm.weight");
      VEC(iqnorm, "indexer.q_layernorm.weight");
      VEC(iknorm, "indexer.k_layernorm.weight");
#undef VEC
    }
    for (int i = 0; ple_ids && i < ple_ids->n; i++)
      if ((int)ple_ids->kid[i].num == l + 1) {
        y->ple_index = f.ple_count++;
        y->ple = xmalloc(sizeof *y->ple);
        snprintf(stem, sizeof stem, "model.language_model.layers.%d.ple", l);
        ple_load(y->ple, stem, number(c, "split_ngram_parts", 512));
      }
  }
  snprintf(path, sizeof path, "%s/generation_config.json", directory);
  char *generation = read_file(path, NULL);
  Json *g = json_parse(generation);
  const Json *eos = json_get(g, "eos_token_id");
  for (int i = 0; eos && i < (eos->type == J_ARR ? eos->n : 1) && i < 8; i++)
    f.stop[f.nstop++] = (int)(eos->type == J_ARR ? eos->kid[i].num : eos->num);
  if (!f.nstop)
    f.stop[f.nstop++] = f.eos;
  json_free(g);
  free(generation);
  json_free(root);
  free(text);
  f.ready = 1;
}

static uint64_t gdn_bytes(void) {
  return (uint64_t)f.recurrent * f.value_heads * 128 * 128 * 4;
}
static uint64_t conv_bytes(void) {
  return (uint64_t)f.recurrent * 3 * f.channels * 4;
}
static uint64_t ple_bytes(void) {
  return (uint64_t)f.ple_count * 9 * f.streams * f.hidden * 4;
}
static uint64_t kv_bytes(int cap) {
  return (uint64_t)2 * f.attention * cap * f.kv_heads * f.head_dim * 2;
}
static uint64_t index_bytes(int cap) {
  return (uint64_t)f.attention * ((cap + 3) / 4) * 128 * 4;
}
static uint64_t tail_bytes(void) {
  return (uint64_t)f.attention * 4 * 128 * 4;
}
static Buf zero_buffer(size_t bytes, int kv) {
  Buf b = kv ? gpu_alloc_kv(bytes) : gpu_alloc(bytes);
  memset(b.ptr, 0, bytes);
  return b;
}
static uint64_t workspace(int n, int allocate) {
  uint64_t bytes = 0;
  size_t r = (size_t)n * 4, h = f.hidden, hc = h * f.streams, vd = f.value_heads * 128;
  size_t widest = hc > vd ? hc : vd;
  if ((size_t)f.inter > widest)
    widest = f.inter;
  int blocks = (f.index_budget + 3 + 255) / 256;
#define B(name, amount)                                                                                                \
  do {                                                                                                                 \
    size_t need = (amount);                                                                                            \
    bytes += gpu_allocation_size(need);                                                                                \
    if (allocate && f.name.size != need) {                                                                             \
      gpu_free(&f.name);                                                                                               \
      f.name = gpu_alloc(need);                                                                                        \
    }                                                                                                                  \
  } while (0)
  B(resident_table, (size_t)f.layers * f.experts * sizeof(ExpertGroup));
  B(resident_counts, f.layers * 4);
  B(tokens, n * 4);
  B(images, n * 8);
  B(sequences, n * sizeof(FlashSeq));
  B(ple_states, n * 8);
  B(x, r * hc);
  B(normalized, r * hc);
  B(mixed, r * h);
  B(mix_up, r * hc);
  B(mix_down, r * f.rank);
  B(injection, r * f.streams);
  B(result, r * h);
  B(xb, (size_t)((n + 127) / 128) * 128 * widest * 2);
  B(quantized, r * widest);
  B(qkv, r * f.channels);
  B(convolved, r * f.channels);
  B(z, r * vd);
  B(a, r * f.value_heads);
  B(b, r * f.value_heads);
  B(decay, r * f.value_heads);
  B(beta, r * f.value_heads);
  B(gdn, r * vd);
  B(gated, r * vd);
  B(qg, r * f.heads * f.head_dim * 2);
  B(k, r * f.kv_heads * f.head_dim);
  B(v, r * f.kv_heads * f.head_dim);
  B(q, r * f.heads * f.head_dim);
  B(kr, r * f.kv_heads * f.head_dim);
  B(attention_out, r * f.heads * f.head_dim);
  B(index_qk, r * 5 * 128);
  B(shared, r * h);
  B(shared_gate, r);
  B(routing, r * f.experts);
  B(selected, n * f.chosen * 4);
  B(coefficients, r * f.chosen);
  B(gathered, r * h);
  B(gate, r * f.inter);
  B(up, r * f.inter);
  B(intermediate, r * f.inter);
  B(logits, (size_t)FLASH_BATCH * f.vocab * 4);
  B(ngram, r * h);
  B(ple_key, r * hc);
  B(ple_query, r * hc);
  B(ple_value, r * h);
  B(ple_gated, r * hc);
  B(ple_normalized, r * hc);
  B(attention_ids, n * (f.index_budget + 4) * 4);
  B(part, (size_t)n * f.heads * blocks * f.head_dim * 4);
  B(ml, (size_t)n * f.heads * blocks * 8);
#undef B
  bytes += experts_workspace(&f.routed, n, allocate);
  if (allocate)
    f.workspace_rows = n;
  return bytes;
}
static uint64_t workspace_bytes(void) {
  Buf *buffers[] = {&f.resident_table, &f.resident_counts, &f.tokens,
                    &f.images,
                    &f.sequences,
                    &f.ple_states,
                    &f.x,
                    &f.normalized,
                    &f.mixed,
                    &f.mix_up,
                    &f.mix_down,
                    &f.injection,
                    &f.result,
                    &f.xb,
                    &f.quantized,
                    &f.qkv,
                    &f.convolved,
                    &f.z,
                    &f.a,
                    &f.b,
                    &f.decay,
                    &f.beta,
                    &f.gdn,
                    &f.gated,
                    &f.qg,
                    &f.k,
                    &f.v,
                    &f.q,
                    &f.kr,
                    &f.attention_out,
                    &f.index_qk,
                    &f.shared,
                    &f.shared_gate,
                    &f.routing,
                    &f.selected,
                    &f.coefficients,
                    &f.gathered,
                    &f.gate,
                    &f.up,
                    &f.intermediate,
                    &f.logits,
                    &f.ngram,
                    &f.ple_key,
                    &f.ple_query,
                    &f.ple_value,
                    &f.ple_gated,
                    &f.ple_normalized,
                    &f.attention_ids,
                    &f.part,
                    &f.ml};
  uint64_t n = experts_bytes(&f.routed);
  for (size_t i = 0; i < sizeof buffers / sizeof *buffers; i++)
    n += buffers[i]->allocated;
  return n;
}
static int prepare_rows(int n) {
  MemorySlot *work = memory_current();
  uint64_t fixed = work->payload - workspace_bytes();
  for (;;) {
    uint64_t experts = (uint64_t)(n * f.chosen < f.experts ? n * f.chosen : f.experts) * f.expert_minimum;
    if (memory_minimum(weights_slot(f.core)->account, weights_required(f.core) + fixed + workspace(n, 0) + experts)) {
      gpu_sync();
      workspace(n, 1);
      return n;
    }
    if (n == 1)
      return 0;
    n = (n + 1) / 2;
  }
}
static void reserve_state(FlashState *s, int positions) {
  if (positions <= s->cap)
    return;
  int cap = s->cap ? s->cap : 1024;
  while (cap < positions)
    cap *= 2;
  gpu_sync();
  Buf kv = gpu_alloc_kv(kv_bytes(cap));
  float *index = memory_host_required(index_bytes(cap), 1);
  int32_t *coords = memory_host_required((size_t)cap * 3 * 4, 1);
  if (s->cap) {
    size_t row = (size_t)f.kv_heads * f.head_dim * 2;
    for (int l = 0; l < 2 * f.attention; l++)
      memcpy(kv.ptr + (size_t)l * cap * row, s->kv.ptr + (size_t)l * s->cap * row, (size_t)s->pos * row);
    for (int l = 0; l < f.attention; l++)
      memcpy(index + (size_t)l * ((cap + 3) / 4) * 128, s->index + (size_t)l * ((s->cap + 3) / 4) * 128,
             (size_t)(s->pos / 4) * 128 * 4);
    memcpy(coords, s->positions, (size_t)s->pos * 3 * 4);
    gpu_free(&s->kv);
    memory_host_free(s->index);
    memory_host_free(s->positions);
  }
  s->kv = kv;
  s->index = index;
  s->positions = coords;
  s->cap = cap;
}
static ModelState *state_new(void) {
  FlashState *s = xmalloc(sizeof *s);
  s->gdn = zero_buffer(gdn_bytes(), 1);
  s->conv = zero_buffer(conv_bytes(), 1);
  s->ple = zero_buffer(ple_bytes(), 1);
  s->tail = memory_host_required(tail_bytes(), 1);
  s->previous[0] = s->previous[1] = f.eos;
  reserve_state(s, 1);
  return (ModelState *)s;
}
static void state_free(ModelState *state) {
  FlashState *s = (FlashState *)state;
  gpu_sync();
  gpu_free(&s->gdn);
  gpu_free(&s->conv);
  gpu_free(&s->ple);
  gpu_free(&s->kv);
  memory_host_free(s->index);
  memory_host_free(s->tail);
  memory_host_free(s->positions);
  free(s);
}
static size_t state_bytes(const ModelState *state, int positions) {
  const FlashState *s = (const FlashState *)state;
  int cap = s ? s->cap : 1024;
  while (cap < positions)
    cap *= 2;
  return gpu_allocation_size(gdn_bytes()) + gpu_allocation_size(conv_bytes()) + gpu_allocation_size(ple_bytes()) +
         gpu_allocation_size(kv_bytes(cap)) + index_bytes(cap) + tail_bytes() + (size_t)cap * 3 * 4;
}
static void state_copy(ModelState *to, const ModelState *from, int positions) {
  FlashState *d = (FlashState *)to;
  const FlashState *s = (const FlashState *)from;
  reserve_state(d, positions);
  gpu_sync();
  memcpy(d->gdn.ptr, s->gdn.ptr, gdn_bytes());
  memcpy(d->conv.ptr, s->conv.ptr, conv_bytes());
  memcpy(d->ple.ptr, s->ple.ptr, ple_bytes());
  memcpy(d->tail, s->tail, tail_bytes());
  size_t row = (size_t)f.kv_heads * f.head_dim * 2;
  for (int l = 0; l < 2 * f.attention; l++)
    memcpy(d->kv.ptr + (size_t)l * d->cap * row, s->kv.ptr + (size_t)l * s->cap * row, (size_t)positions * row);
  for (int l = 0; l < f.attention; l++)
    memcpy(d->index + (size_t)l * ((d->cap + 3) / 4) * 128, s->index + (size_t)l * ((s->cap + 3) / 4) * 128,
           (size_t)(positions / 4) * 128 * 4);
  memcpy(d->positions, s->positions, (size_t)positions * 3 * 4);
  d->pos = positions;
  d->rope_delta = s->rope_delta;
  memcpy(d->previous, s->previous, sizeof d->previous);
}
static ModelSnapshot *snapshot_take(ModelState *state, int pos, int with_logits) {
  FlashState *s = (FlashState *)state;
  FlashSnapshot *x = xmalloc(sizeof *x);
  x->gdn = gpu_alloc_kv(gdn_bytes());
  x->conv = gpu_alloc_kv(conv_bytes());
  x->ple = gpu_alloc_kv(ple_bytes());
  x->tail = memory_host_required(tail_bytes(), 1);
  gpu_sync();
  memcpy(x->gdn.ptr, s->gdn.ptr, gdn_bytes());
  memcpy(x->conv.ptr, s->conv.ptr, conv_bytes());
  memcpy(x->ple.ptr, s->ple.ptr, ple_bytes());
  memcpy(x->tail, s->tail, tail_bytes());
  x->pos = pos;
  x->rope_delta = s->rope_delta;
  memcpy(x->previous, s->previous, sizeof x->previous);
  if (with_logits) {
    x->logits = memory_host_required((size_t)f.vocab * 4, 1);
    memcpy(x->logits, f.last_logits, (size_t)f.vocab * 4);
  }
  return (ModelSnapshot *)x;
}
static void snapshot_restore(ModelState *state, const ModelSnapshot *snap) {
  FlashState *s = (FlashState *)state;
  const FlashSnapshot *x = (const FlashSnapshot *)snap;
  gpu_sync();
  memcpy(s->gdn.ptr, x->gdn.ptr, gdn_bytes());
  memcpy(s->conv.ptr, x->conv.ptr, conv_bytes());
  memcpy(s->ple.ptr, x->ple.ptr, ple_bytes());
  memcpy(s->tail, x->tail, tail_bytes());
  s->pos = x->pos;
  s->rope_delta = x->rope_delta;
  memcpy(s->previous, x->previous, sizeof s->previous);
}
static void snapshot_free(ModelSnapshot *snap) {
  FlashSnapshot *x = (FlashSnapshot *)snap;
  gpu_sync();
  gpu_free(&x->gdn);
  gpu_free(&x->conv);
  gpu_free(&x->ple);
  memory_host_free(x->tail);
  memory_host_free(x->logits);
  free(x);
}
static size_t snapshot_bytes(int logits) {
  return gpu_allocation_size(gdn_bytes()) + gpu_allocation_size(conv_bytes()) + gpu_allocation_size(ple_bytes()) +
         tail_bytes() + (logits ? (size_t)f.vocab * 4 : 0);
}
static const float *snapshot_logits(const ModelSnapshot *s) {
  return ((const FlashSnapshot *)s)->logits;
}

static void unary(int kernel, const Buf *x, uint32_t count) {
  gpu_kern(kernel);
  gpu_buf(0, x, 0);
  gpu_bytes(1, &count, 4);
  gpu_go((count + 255) / 256, 1, 1, 256, 1, 1);
}
static void product(const Linear *m, const Buf *x, size_t xo, const Buf *y, int n, int act4, int round) {
  Linear scaled = *m;
  const Buf *input = x;
  size_t offset = xo;
  if (m->kind == 'm' || (act4 && m->kind == 'n')) {
    struct {
      uint32_t count, bits;
      float scale;
      uint32_t pad;
    } p = {(uint32_t)(n * m->cols), m->kind == 'm' ? 8u : 4u, m->input_scale, 0};
    gpu_kern(f.quant);
    gpu_buf(0, x, xo);
    gpu_buf(1, &f.quantized, 0);
    gpu_bytes(2, &p, sizeof p);
    gpu_go((p.count + 255) / 256, 1, 1, 256, 1, 1);
    input = &f.quantized;
    offset = 0;
    if (m->kind == 'n') scaled.scale *= m->input_scale;
  }
  linear_product(&scaled, input, offset, y, 0, n, n > 8, &f.xb, 0);
  if (round) unary(f.round, y, (uint32_t)(n * m->rows));
}
static void norm_group(const Buf *x, const Buf *w, const Buf *y, int n, int width, int groups) {
  struct {
    uint32_t width, groups;
    float eps;
    uint32_t pad;
  } p = {(uint32_t)width, (uint32_t)groups, f.eps, 0};
  gpu_kern(f.norm_group);
  gpu_buf(0, x, 0);
  gpu_buf(1, w, 0);
  gpu_buf(2, y, 0);
  gpu_bytes(3, &p, sizeof p);
  gpu_go((uint32_t)(n * groups), 1, 1, 256, 1, 1);
}
static void mix(const Hyper *h, int n) {
  norm_group(&f.x, &h->norm, &f.normalized, n, f.hidden, f.streams);
  product(&h->down, &f.normalized, 0, &f.mix_down, n, 0, 1);
  uint32_t down[2] = {(uint32_t)(n * f.rank), (uint32_t)f.streams};
  gpu_kern(f.hc_silu);
  gpu_buf(0, &f.mix_down, 0);
  gpu_bytes(1, down, sizeof down);
  gpu_go((down[0] + 255) / 256, 1, 1, 256, 1, 1);
  product(&h->up, &f.mix_down, 0, &f.mix_up, n, 0, 1);
  uint32_t p[4] = {(uint32_t)f.hidden, (uint32_t)f.streams, (uint32_t)n, 0};
  gpu_kern(f.hc_mix);
  gpu_buf(0, &f.normalized, 0);
  gpu_buf(1, &f.mix_up, 0);
  gpu_buf(2, &f.mixed, 0);
  gpu_bytes(3, p, sizeof p);
  gpu_go((n * f.hidden + 255) / 256, 1, 1, 256, 1, 1);
  if (h->inject.rows)
    product(&h->inject, &f.normalized, 0, &f.injection, n, 0, 1);
}
static void inject(int n) {
  uint32_t p[4] = {(uint32_t)f.hidden, (uint32_t)f.streams, (uint32_t)n, 0};
  gpu_kern(f.hc_inject);
  gpu_buf(0, &f.x, 0);
  gpu_buf(1, &f.result, 0);
  gpu_buf(2, &f.injection, 0);
  gpu_bytes(3, p, sizeof p);
  gpu_go((n * f.hidden * f.streams + 255) / 256, 1, 1, 256, 1, 1);
}
static void feed_forward(const Expert *e, const Buf *x, const Buf *out, int n, int quantized) {
  product(&e->gate, x, 0, &f.gate, n, quantized, 1);
  product(&e->up, x, 0, &f.up, n, quantized, 1);
  uint32_t count = n * e->gate.rows;
  gpu_kern(f.swiglu);
  gpu_buf(0, &f.gate, 0);
  gpu_buf(1, &f.up, 0);
  gpu_buf(2, &f.intermediate, 0);
  gpu_bytes(3, &count, 4);
  gpu_go((count + 255) / 256, 1, 1, 256, 1, 1);
  product(&e->down, &f.intermediate, 0, out, n, quantized, 1);
}
static void recurrent(const FlashLayer *y, int n) {
  product(&y->qkv, &f.mixed, 0, &f.qkv, n, 0, 1);
  product(&y->z, &f.mixed, 0, &f.z, n, 0, 1);
  product(&y->a, &f.mixed, 0, &f.a, n, 0, 1);
  product(&y->b, &f.mixed, 0, &f.b, n, 0, 1);
  uint32_t cp[4] = {(uint32_t)f.channels, (uint32_t)n, (uint32_t)y->index, 1};
  gpu_kern(f.conv_kernel);
  gpu_buf(0, &f.qkv, 0);
  gpu_buf(1, &f.convolved, 0);
  gpu_buf(2, &y->conv, 0);
  gpu_buf(3, &f.sequences, 0);
  gpu_bytes(4, cp, sizeof cp);
  gpu_go((f.channels + 255) / 256, 1, 1, 256, 1, 1);
  struct {
    uint32_t n, stride;
    float eps, scale;
  } lp = {128, (uint32_t)f.channels, 1e-6f, 1 / sqrtf(128)};
  gpu_kern(f.l2);
  gpu_buf(0, &f.convolved, 0);
  gpu_bytes(1, &lp, sizeof lp);
  gpu_go(f.key_heads, n, 1, 128, 1, 1);
  lp.scale = 1;
  gpu_buf(0, &f.convolved, (size_t)f.key_heads * 128 * 4);
  gpu_bytes(1, &lp, sizeof lp);
  gpu_go(f.key_heads, n, 1, 128, 1, 1);
  uint32_t gates[2] = {(uint32_t)f.value_heads, (uint32_t)n};
  gpu_kern(f.gates);
  gpu_buf(0, &f.a, 0);
  gpu_buf(1, &f.b, 0);
  gpu_buf(2, &y->decay, 0);
  gpu_buf(3, &y->bias, 0);
  gpu_buf(4, &f.decay, 0);
  gpu_buf(5, &f.beta, 0);
  gpu_bytes(6, gates, sizeof gates);
  gpu_go((n * f.value_heads + 63) / 64, 1, 1, 64, 1, 1);
  unary(f.round, &f.beta, n * f.value_heads);
  uint32_t dp[8] = {(uint32_t)n,
                    (uint32_t)f.value_heads,
                    (uint32_t)(f.value_heads / f.key_heads),
                    (uint32_t)y->index,
                    (uint32_t)f.channels,
                    (uint32_t)(f.key_heads * 128),
                    0,
                    0};
  gpu_kern(f.delta);
  gpu_buf(0, &f.convolved, 0);
  gpu_buf(1, &f.decay, 0);
  gpu_buf(2, &f.beta, 0);
  gpu_buf(3, &f.sequences, 0);
  gpu_buf(4, &f.gdn, 0);
  gpu_bytes(5, dp, sizeof dp);
  gpu_go(f.value_heads, 32, 1, 128, 1, 1);
  unary(f.round, &f.gdn, n * f.value_heads * 128);
  struct {
    uint32_t n;
    float eps, scale;
    uint32_t gate;
  } np = {128, f.eps, 1, (uint32_t)f.output_gate};
  gpu_kern(f.norm_kernel);
  gpu_buf(0, &f.gdn, 0);
  gpu_buf(1, &y->norm, 0);
  gpu_buf(2, &f.gated, 0);
  gpu_bytes(3, &np, sizeof np);
  gpu_buf(4, &f.z, 0);
  gpu_go(n * f.value_heads, 1, 1, 128, 1, 1);
  unary(f.round, &f.gated, n * f.value_heads * 128);
  product(&y->output, &f.gated, 0, &f.result, n, 0, 1);
}
static float bf(float x) {
  uint32_t u;
  memcpy(&u, &x, 4);
  u = (u + 0x7fff + ((u >> 16) & 1)) & 0xffff0000u;
  memcpy(&x, &u, 4);
  return x;
}
static void normalize_rope(float *x, const float *w, const int32_t *pos) {
  float sum = 0;
  for (int i = 0; i < 128; i++)
    sum += x[i] * x[i];
  float inv = 1 / sqrtf(sum / 128 + f.eps);
  for (int i = 0; i < 128; i++)
    x[i] = bf(x[i] * inv * w[i]);
  for (int i = 0; i < f.rotary / 2; i++) {
    int axis = i % 3 == 1 && i < f.section_h * 3 ? 1 : i % 3 == 2 && i < f.section_w * 3 ? 2 : 0;
    float angle = pos[axis] * powf(f.theta, -2.0f * i / f.rotary), c = bf(cosf(angle)), s = bf(sinf(angle)), a = x[i],
          b = x[i + f.rotary / 2];
    x[i] = bf(bf(a * c) - bf(b * s));
    x[i + f.rotary / 2] = bf(bf(a * s) + bf(b * c));
  }
}
static int id_order(const void *a, const void *b) {
  int x = *(const int *)a, y = *(const int *)b;
  return (x > y) - (x < y);
}
static int worse(const float *s, int a, int b) {
  return s[a] < s[b] || (s[a] == s[b] && a > b);
}
static void top_blocks(const float *s, int count, int keep, int *out) {
  int n = 0;
  for (int i = 0; i < count; i++) {
    if (n < keep) {
      int k = n++;
      out[k] = i;
      while (k) {
        int p = (k - 1) / 2;
        if (!worse(s, out[k], out[p]))
          break;
        int v = out[k];
        out[k] = out[p];
        out[p] = v;
        k = p;
      }
    } else if (worse(s, out[0], i)) {
      out[0] = i;
      int k = 0;
      for (;;) {
        int a = k * 2 + 1;
        if (a >= keep)
          break;
        int b = a + 1;
        if (b < keep && worse(s, out[b], out[a]))
          a = b;
        if (!worse(s, out[a], out[k]))
          break;
        int v = out[k];
        out[k] = out[a];
        out[a] = v;
        k = a;
      }
    }
  }
  qsort(out, keep, sizeof *out, id_order);
}
static void select_attention(const FlashLayer *y, const Row *rows, int n) {
  const FlashSeq *seq = (const FlashSeq *)f.sequences.ptr;
  const float *projected = (const float *)f.index_qk.ptr;
  int stride = f.index_budget + 4;
  for (int r = 0; r < n; r++) {
    FlashState *s = rows[r].state;
    int pos = seq[r].pos, blocks = (pos + 1) / 4;
    float *tail = s->tail + (size_t)y->index * 4 * 128;
    memcpy(tail + (pos % 4) * 128, projected + (size_t)r * 640 + 512, 128 * 4);
    float *keys = s->index + (size_t)y->index * ((s->cap + 3) / 4) * 128;
    if (pos % 4 == 3) {
      float key[128];
      for (int i = 0; i < 128; i++)
        key[i] = bf((tail[i] + tail[128 + i] + tail[256 + i] + tail[384 + i]) * .25f);
      normalize_rope(key, (float *)y->iknorm.ptr, s->positions + (pos - 3) * 3);
      memcpy(keys + (size_t)(pos / 4) * 128, key, sizeof key);
    }
    int *out = (int *)f.attention_ids.ptr + (size_t)r * stride;
    int chosen = blocks < f.index_budget / 4 ? blocks : f.index_budget / 4;
    if (blocks <= chosen) {
      for (int i = 0; i < chosen * 4; i++)
        out[i + 1] = i;
    } else {
      float query[4 * 128];
      memcpy(query, projected + (size_t)r * 640, sizeof query);
      for (int h = 0; h < 4; h++)
        normalize_rope(query + h * 128, (float *)y->iqnorm.ptr, s->positions + pos * 3);
      float *scores = xmalloc((size_t)blocks * 5 * 4);
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 4, blocks, 128, 1, query, 128, keys, 128, 0, scores, blocks);
      float *sum = scores + 4 * blocks;
      for (int i = 0; i < blocks; i++)
        sum[i] = (fmaxf(scores[i], 0) + fmaxf(scores[blocks + i], 0) + fmaxf(scores[2 * blocks + i], 0) +
                  fmaxf(scores[3 * blocks + i], 0)) /
                 sqrtf(128);
      int *selected = xmalloc((size_t)chosen * 4);
      top_blocks(sum, blocks, chosen, selected);
      for (int i = 0; i < chosen; i++)
        for (int k = 0; k < 4; k++)
          out[1 + i * 4 + k] = selected[i] * 4 + k;
      free(selected);
      free(scores);
    }
    int used = chosen * 4;
    for (int i = blocks * 4; i <= pos; i++)
      out[++used] = i;
    out[0] = used;
  }
}
static void attention(const FlashLayer *y, const Row *rows, int n) {
  product(&y->index_qk, &f.mixed, 0, &f.index_qk, n, 0, 1);
  gpu_commit();
  gpu_sync();
  select_attention(y, rows, n);
  gpu_begin();
  product(&y->q, &f.mixed, 0, &f.qg, n, 0, 1);
  product(&y->k, &f.mixed, 0, &f.k, n, 0, 1);
  product(&y->v, &f.mixed, 0, &f.v, n, 0, 1);
  struct {
    uint32_t heads, dim, rot, stride;
    float theta, eps;
    uint32_t h, w;
  } rp = {(uint32_t)f.heads, 256,   (uint32_t)f.rotary,    512,
          f.theta,           f.eps, (uint32_t)f.section_h, (uint32_t)f.section_w};
  gpu_kern(f.rope);
  gpu_buf(0, &f.qg, 0);
  gpu_buf(1, &y->qnorm, 0);
  gpu_buf(2, &f.q, 0);
  gpu_bytes(3, &rp, sizeof rp);
  gpu_buf(4, &f.sequences, 0);
  gpu_go(f.heads, n, 1, 256, 1, 1);
  rp.heads = f.kv_heads;
  rp.stride = 256;
  gpu_buf(0, &f.k, 0);
  gpu_buf(1, &y->knorm, 0);
  gpu_buf(2, &f.kr, 0);
  gpu_bytes(3, &rp, sizeof rp);
  gpu_go(f.kv_heads, n, 1, 256, 1, 1);
  uint32_t kp[4] = {(uint32_t)n, (uint32_t)(f.kv_heads * 256), (uint32_t)y->index, 0};
  gpu_kern(f.store);
  gpu_buf(0, &f.kr, 0);
  gpu_buf(1, &f.v, 0);
  gpu_buf(2, &f.sequences, 0);
  gpu_bytes(3, kp, sizeof kp);
  gpu_go((kp[1] + 255) / 256, n, 1, 256, 1, 1);
  int maximum = 0;
  for (int i = 0; i < n; i++) {
    int count = ((int *)f.attention_ids.ptr)[i * (f.index_budget + 4)];
    if (count > maximum)
      maximum = count;
  }
  struct {
    uint32_t heads, kv, dim, layer;
    float scale;
    uint32_t blocks, stride, pad;
  } ap = {(uint32_t)f.heads,
          (uint32_t)f.kv_heads,
          256,
          (uint32_t)y->index,
          1.0f / 16,
          (uint32_t)((f.index_budget + 3 + 255) / 256),
          (uint32_t)(f.index_budget + 4),
          0};
  gpu_kern(f.attend);
  gpu_buf(0, &f.q, 0);
  gpu_buf(1, &f.sequences, 0);
  gpu_buf(2, &f.part, 0);
  gpu_buf(3, &f.ml, 0);
  gpu_bytes(4, &ap, sizeof ap);
  gpu_buf(5, &f.attention_ids, 0);
  gpu_go(f.kv_heads, (maximum + 255) / 256, n, 256, 1, 1);
  gpu_kern(f.combine);
  gpu_buf(0, &f.part, 0);
  gpu_buf(1, &f.ml, 0);
  gpu_buf(2, &f.qg, 0);
  gpu_buf(3, &f.attention_out, 0);
  gpu_buf(4, &f.attention_ids, 0);
  gpu_bytes(5, &ap, sizeof ap);
  gpu_go(f.heads, n, 1, 256, 1, 1);
  product(&y->output, &f.attention_out, 0, &f.result, n, 0, 1);
}
static float fp8(uint8_t b) {
  int exp = (b >> 3) & 15, m = b & 7;
  float v = exp ? ldexpf(1 + m * .125f, exp - 7) : ldexpf((float)m, -9);
  return b & 128 ? -v : v;
}
typedef struct {
  const Ple *p;
  const int *history;
  int n;
} Ngrams;
static void ngram_worker(void *context, size_t worker) {
  const Ngrams *j = context;
  const Ple *p = j->p;
  static const float values[8] = {0, .5f, 1, 1.5f, 2, 3, 4, 6};
  for (int item = (int)worker; item < j->n * 16; item += 8) {
    int row = item / 16, head = item % 16, degree = head < 8 ? 2 : 3;
    uint64_t bits = (uint64_t)j->history[row * 3] * (uint64_t)p->multipliers[0];
    for (int k = 1; k < degree; k++)
      bits ^= (uint64_t)j->history[row * 3 + k] * (uint64_t)p->multipliers[k];
    int64_t signed_bits;
    memcpy(&signed_bits, &bits, 8);
    int64_t id = signed_bits % p->primes[head];
    if (id < 0)
      id += p->primes[head];
    id += p->offsets[head];
    int shard = (int)(id / p->shard_rows);
    int64_t local = id % p->shard_rows;
    uint8_t codes[128], scales[16];
    read_tensor(p->table[shard], codes, p->head_dim / 2, local * (p->head_dim / 2));
    read_tensor(p->scales[shard], scales, p->head_dim / 16, local * (p->head_dim / 16));
    float *out = (float *)f.ngram.ptr + (size_t)row * f.hidden + head * p->head_dim;
    for (int i = 0; i < p->head_dim; i++) {
      unsigned nib = (codes[i / 2] >> (4 * (i % 2))) & 15;
      float value = values[nib & 7];
      if (nib & 8)
        value = -value;
      out[i] = bf(value * fp8(scales[i / 16]) * p->scale);
    }
  }
}
static void ple_forward(const FlashLayer *y, const int *history, int n) {
  const Ple *p = y->ple;
  gpu_commit();
  gpu_sync();
  Ngrams job = {p, history, n};
  dispatch_apply_f(8, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), &job, ngram_worker);
  gpu_begin();
  product(&p->key, &f.ngram, 0, &f.ple_key, n, 0, 1);
  norm_group(&f.ple_key, &p->norm_key, &f.ple_key, n, f.hidden, f.streams);
  norm_group(&f.x, &p->norm_query, &f.ple_query, n, f.hidden, f.streams);
  product(&p->value, &f.ngram, 0, &f.ple_value, n, 0, 1);
  uint32_t gp[2] = {(uint32_t)f.hidden, (uint32_t)f.streams};
  gpu_kern(f.ple_gate);
  gpu_buf(0, &f.ple_key, 0);
  gpu_buf(1, &f.ple_query, 0);
  gpu_buf(2, &f.ple_value, 0);
  gpu_buf(3, &f.ple_gated, 0);
  gpu_bytes(4, gp, sizeof gp);
  gpu_go(n * f.streams, 1, 1, 256, 1, 1);
  norm_group(&f.ple_gated, &p->norm_conv, &f.ple_normalized, n, f.hidden, f.streams);
  uint32_t cp[4] = {(uint32_t)(f.hidden * f.streams), (uint32_t)n, (uint32_t)y->ple_index, 0};
  gpu_kern(f.ple_conv);
  gpu_buf(0, &f.ple_normalized, 0);
  gpu_buf(1, &f.ple_gated, 0);
  gpu_buf(2, &p->conv, 0);
  gpu_buf(3, &f.ple_states, 0);
  gpu_buf(4, &f.x, 0);
  gpu_bytes(5, cp, sizeof cp);
  gpu_go((cp[0] + 255) / 256, 1, 1, 256, 1, 1);
}
static void routed_sum(int n) {
  experts_sum(&f.routed, &f.result);
  uint32_t dims[2] = {(uint32_t)f.hidden, (uint32_t)n};
  gpu_kern(f.shared_sum);
  gpu_buf(0, &f.result, 0);
  gpu_buf(1, &f.shared, 0);
  gpu_buf(2, &f.shared_gate, 0);
  gpu_bytes(3, dims, sizeof dims);
  gpu_go((dims[0] * dims[1] + 255) / 256, 1, 1, 256, 1, 1);
}
static int routed(FlashLayer *y, int n) {
  size_t offset = (size_t)y->number * f.experts * sizeof(ExpertGroup);
  int resident = y->presence.count == (unsigned)f.experts;
  if (resident && y->table_generation != y->presence.generation) {
    y->table_valid = experts_table(&f.routed, y->experts, &f.resident_table, offset);
    y->table_generation = y->presence.generation;
  }
  resident &= y->table_valid;
  feed_forward(&y->shared, &f.mixed, &f.shared, n, 0);
  product(&y->shared_gate, &f.mixed, 0, &f.shared_gate, n, 0, 1);
  product(&y->router, &f.mixed, 0, &f.routing, n, 0, 1);
  uint32_t p[4] = {(uint32_t)f.experts, (uint32_t)f.chosen, (uint32_t)f.normalize_topk, 0};
  gpu_kern(f.route);
  gpu_buf(0, &f.routing, 0);
  gpu_buf(1, &f.selected, 0);
  gpu_buf(2, &f.coefficients, 0);
  gpu_bytes(3, p, sizeof p);
  gpu_go(n, 1, 1, 512, 1, 1);
  if (resident) {
    experts_resident(&f.routed, &f.resident_table, offset, &f.selected, &f.coefficients, &f.mixed,
                      &f.resident_counts, y->number * 4, n, 1);
    routed_sum(n);
    return 2;
  }
  gpu_commit();
  gpu_sync();
  if (!experts_acquire(&f.routed, y->experts, (const uint32_t *)f.selected.ptr,
                       (const float *)f.coefficients.ptr, n, 1, &f.cache.stats))
    return 0;
  gpu_begin();
  experts_encode(&f.routed, y->experts, &f.mixed);
  for (int i = 0; i < f.routed.active; i++) {
    int e = f.routed.active_ids[i];
    int count = experts_count(&f.routed, e);
    if (experts_grouped(&f.routed, e)) continue;
    uint32_t dims[2] = {(uint32_t)f.hidden, (uint32_t)count};
    gpu_kern(f.gather);
    gpu_buf(0, &f.mixed, 0);
    gpu_buf(1, &f.routed.entries, experts_entry_offset(&f.routed, e));
    gpu_buf(2, &f.gathered, 0);
    gpu_bytes(3, dims, sizeof dims);
    gpu_go((dims[0] * dims[1] + 255) / 256, 1, 1, 256, 1, 1);
    Buf out = experts_output(&f.routed, e);
    feed_forward(y->experts + e, &f.gathered, &out, count, 1);
  }
  routed_sum(n);
  return 1;
}
static void add_stats(CacheStats *to, CacheStats before, CacheStats after) {
  to->hits += after.hits - before.hits;
  to->misses += after.misses - before.misses;
  to->bytes += after.bytes - before.bytes;
  to->evictions += after.evictions - before.evictions;
  to->read_s += after.read_s - before.read_s;
  to->row_own += after.row_own - before.row_own;
  to->row_ssd += after.row_ssd - before.row_ssd;
  to->row_unique += after.row_unique - before.row_unique;
}
static const float *pass(const Row *rows, int n, int logits, int prompt) {
  if (f.workspace_rows < n && prepare_rows(n) < n)
    return NULL;
  FlashSeq *seq = (FlashSeq *)f.sequences.ptr;
  int history[FLASH_ROWS * 3];
  for (int r = 0, repeat = 0; r < n; r++) {
    FlashState *s = rows[r].state;
    repeat = r && rows[r - 1].state == s ? repeat + 1 : 0;
    reserve_state(s, s->pos + repeat + 1);
  }
  for (int r = 0, repeat = 0; r < n; r++) {
    FlashState *s = rows[r].state;
    repeat = r && rows[r - 1].state == s ? repeat + 1 : 0;
    int pos = s->pos + repeat;
    int token = rows[r].token < 0 ? f.image_id : rows[r].token;
    ((int *)f.tokens.ptr)[r] = token;
    ((uint64_t *)f.images.ptr)[r] = rows[r].token < 0 ? media_embedding(f.image_list, f.image_count, pos) : 0;
    ((uint64_t *)f.ple_states.ptr)[r] = s->ple.gpu;
    seq[r] = (FlashSeq){.gdn = s->gdn.gpu,
                        .conv = s->conv.gpu,
                        .k = s->kv.gpu,
                        .v = s->kv.gpu + kv_bytes(s->cap) / 2,
                        .pos = (uint32_t)pos,
                        .cap = (uint32_t)s->cap,
                        .rope_t = pos + s->rope_delta,
                        .rope_h = pos + s->rope_delta,
                        .rope_w = pos + s->rope_delta};
    qwen_image_positions(f.image_list, f.image_count, pos, s->rope_delta, prompt, &seq[r].rope_t);
    memcpy(s->positions + pos * 3, &seq[r].rope_t, 3 * 4);
    history[r * 3] = token;
    history[r * 3 + 1] = s->previous[0];
    history[r * 3 + 2] = s->previous[1];
    s->previous[1] = s->previous[0];
    s->previous[0] = token;
    if (token == f.eos)
      s->previous[1] = f.eos;
  }
  double began = now(), gpu = gpu_busy;
  CacheStats before = f.cache.stats;
  int resident_layers[f.layers], resident_count = 0;
  gpu_begin();
  if (prompt)
    for (int i = 0; i < f.image_count; i++)
      gpu_use(&f.image_list[i].features);
  uint32_t ep[2] = {(uint32_t)f.hidden, (uint32_t)f.streams};
  gpu_kern(f.embed);
  gpu_buf(0, &f.embedding, 0);
  gpu_buf(1, &f.tokens, 0);
  gpu_buf(2, &f.x, 0);
  gpu_bytes(3, ep, sizeof ep);
  gpu_buf(4, &f.images, 0);
  gpu_go((f.hidden * f.streams + 255) / 256, n, 1, 256, 1, 1);
  for (int l = 0; l < f.layers; l++) {
    FlashLayer *y = f.layer + l;
    f.counters.prefill_layer = prompt ? l + 1 : 0;
    if (y->ple)
      ple_forward(y, history, n);
    mix(&y->attention_hyper, n);
    if (y->recurrent)
      recurrent(y, n);
    else
      attention(y, rows, n);
    inject(n);
    mix(&y->mlp_hyper, n);
    int result = routed(y, n);
    if (!result) return NULL;
    if (result == 2) resident_layers[resident_count++] = l;
    inject(n);
    if (f.dump_fd >= 0) {
      gpu_commit();
      gpu_sync();
      size_t bytes = (size_t)n * f.hidden * f.streams * 4;
      if (write(f.dump_fd, f.x.ptr, bytes) != (ssize_t)bytes)
        die("Flash-Next trace write failed");
      gpu_begin();
    }
  }
  mix(&f.final, n);
  product(&f.head, &f.mixed, (size_t)(n - logits) * f.hidden * 4, &f.logits, logits, 0, 0);
  gpu_commit();
  gpu_sync();
  for (int i = 0; i < resident_count; i++) {
    uint32_t unique = ((uint32_t *)f.resident_counts.ptr)[resident_layers[i]];
    f.cache.stats.hits += unique;
    f.cache.stats.row_unique += unique;
    f.cache.stats.row_own += (uint64_t)n * f.chosen;
  }
  experts_release(&f.routed);
  if (f.dump_fd >= 0) {
    size_t bytes = (size_t)logits * f.vocab * 4;
    if (write(f.dump_fd, f.logits.ptr, bytes) != (ssize_t)bytes)
      die("Flash-Next logit trace write failed");
  }
  for (int r = 0; r < n; r++) {
    int maximum = seq[r].rope_t > seq[r].rope_h ? seq[r].rope_t : seq[r].rope_h;
    if (seq[r].rope_w > maximum)
      maximum = seq[r].rope_w;
    rows[r].state->rope_delta = maximum - (int)seq[r].pos;
    rows[r].state->pos++;
  }
  if (prompt) {
    f.counters.prefill_rows += n;
    add_stats(&f.counters.prefill, before, f.cache.stats);
  } else {
    f.counters.decode_rows += n;
    f.counters.decode_seconds += now() - began;
    f.counters.decode_gpu_seconds += gpu_busy - gpu;
    add_stats(&f.counters.decode, before, f.cache.stats);
  }
  f.counters.prefill_layer = 0;
  return f.last_logits = (const float *)f.logits.ptr;
}

static const char *refuse(const Options *o) {
  if (o->draft)
    return "Flash-Next currently takes draft 0";
  if (o->lookup)
    return "Flash-Next currently takes lookup 0";
  if (o->deepseek.experts || o->deepseek.ced || o->deepseek.substitute || o->deepseek.online_predictor ||
      o->deepseek.pipeline || o->deepseek.runahead)
    return "these routed-expert options apply to DeepSeek";
  return NULL;
}
static int load(const char *dir, const Options *o) {
  f.dump_fd = o->dump_fd;
  if (!f.ready)
    prepare(dir);
  memory_pin(weights_slot(f.core), 1);
  int loaded = prepare_rows(1) && weights_load(f.core);
  memory_pin(weights_slot(f.core), 0);
  return loaded;
}
static MemorySlot *weight_slot(void) {
  return f.core ? weights_slot(f.core) : NULL;
}
static int vocab(void) {
  return f.vocab;
}
static int max_rows(void) {
  return FLASH_BATCH;
}
static int max_verify(void) {
  return 1;
}
static int drafts(void) {
  return 0;
}
static int rewind_rows(void) {
  return 0;
}
static int prefill_rows(void) {
  return FLASH_ROWS;
}
static int stop(int token) {
  for (int i = 0; i < f.nstop; i++)
    if (token == f.stop[i])
      return 1;
  return 0;
}
static const float *prefill(ModelState *state, const int *tokens, int n, int from, int replay, int capture,
                            ModelSnapshot **captured, float *every) {
  FlashState *s = (FlashState *)state;
  Row rows[FLASH_ROWS];
  f.plan = (ModelPlan){.tokens = n - from, .decoder_tokens = n - from};
  if (captured)
    *captured = NULL;
  const float *lg = NULL;
  for (int at = from; at < n;) {
    int end = capture > at && capture < n ? capture : n;
    int requested = end - at;
    if (requested > (every ? FLASH_BATCH : FLASH_ROWS))
      requested = every ? FLASH_BATCH : FLASH_ROWS;
    int count = prepare_rows(requested);
    if (!count)
      return NULL;
    for (int i = 0; i < count; i++)
      rows[i] = (Row){s, tokens[at + i]};
    lg = pass(rows, count, every ? count : 1, 1);
    if (!lg)
      return NULL;
    if (every) {
      memcpy(every + (size_t)(at - from) * f.vocab, lg, (size_t)count * f.vocab * 4);
      lg += (size_t)(count - 1) * f.vocab;
    }
    at += count;
    f.plan.chunks++;
    if (at == capture)
      *captured = snapshot_take(state, at, 0);
    if (shutdown_requested())
      return NULL;
  }
  return f.last_logits = lg;
}
static const float *forward_rows(ModelState *const *states, const int *tokens, const int *positions, int n) {
  Row rows[FLASH_BATCH];
  for (int i = 0; i < n; i++)
    rows[i] = (Row){(FlashState *)states[i], tokens[i]};
  return pass(rows, n, n, 0);
}
static int replay_rows(int position) {
  return -1;
}
static int replay_start(int position) {
  return position;
}
static void seek(ModelState *s, const int *tokens, int pos) {
}
static int idle(void) {
  return f.cache.slots;
}
static uint64_t cached_bytes(void) {
  return f.cache.bytes;
}
static void counters(ModelCounters *out) {
  *out = f.counters;
  out->layers = f.layers;
  out->experts_per_row = f.layers * f.chosen;
  out->dense_bytes = weights_resident_bytes(f.core) + qwen_vision_bytes(f.vision);
  out->cache = f.cache.stats;
  out->cache_bytes = f.cache.bytes;
  out->cache_slots = out->mapped = out->cap = f.cache.slots;
  out->max_slots = out->experts = f.cache.capacity;
  out->slot_bytes = f.cache.slots ? f.cache.bytes / f.cache.slots : f.expert_minimum;
}
static void plan(ModelPlan *out) {
  *out = f.plan;
}
static void chat(char *out, size_t cap, const char *prompt, int images, int think) {
  size_t at = snprintf(out, cap, "<|im_start|>user\n");
  for (int i = 0; i < images && at < cap; i++)
    at += snprintf(out + at, cap - at, "<|vision_start|>" MEDIA_MARK "<|vision_end|>");
  if (at < cap)
    snprintf(out + at, cap - at, "%s<|im_end|>\n<|im_start|>assistant\n%s", prompt,
             think ? "<think>\n" : "<think>\n\n</think>\n\n");
}
static int image(Media *im) {
  return qwen_vision_encode(f.vision, im);
}
static void images(const Media *input, int count) {
  f.image_list = input;
  f.image_count = count;
}
static void active(int on) {
  qwen_vision_reside(f.vision, on);
}
static const char *const model_types[] = {"qwen4_exp", "qwen4_exp_text", NULL};
const ModelOps flash_ops = {.family = "Qwen3.8 Flash-Next",
                            .model_types = model_types,
                            .media_token = "<|image_pad|>",
                            .active = active,
                            .prepare_media = qwen_prepare_image,
                            .encode_media = image,
                            .media = images,
                            .refuse = refuse,
                            .load = load,
                            .weight_slot = weight_slot,
                            .counters = counters,
                            .cache_bytes = cached_bytes,
                            .plan = plan,
                            .vocab = vocab,
                            .max_rows = max_rows,
                            .max_verify = max_verify,
                            .draft_block = drafts,
                            .rewind = rewind_rows,
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
                            .chat = chat};
