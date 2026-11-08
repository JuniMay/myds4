// Transcribe's mel frontend, speech transformer and speaker-feature encoder.
#include "audio.h"
#include "../linear.h"
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  Linear matrix;
  Buf bias;
} Projection;
typedef struct {
  Buf weight, bias;
} Norm;
typedef struct {
  Norm input, mlp, q, k;
  Projection qkv, output, up, down;
} SpeechLayer;
typedef struct {
  int width, heads, layers, rotary, qnorm;
  Projection stack;
  Norm input, final;
  SpeechLayer *layer;
} Transformer;
enum {
  X,
  NORMAL,
  QKV,
  Q,
  K,
  V,
  ATTENTION,
  PARTS,
  MAXIMA,
  SEQUENCES,
  UP,
  OUTPUT,
  PACKED,
  QUANT,
  ASR,
  ACTIVITY,
  SPEECH_BUFFERS
};
struct SpeechEncoder {
  TensorIndex *checkpoint;
  WeightSet *weights;
  MemorySlot workspace;
  Transformer asr, diar;
  Projection speaker_project, upsample, speaker_hidden, speakers, output;
  Norm asr_norm, diar_norm;
  Buf window, filter, speaker_kernel, silence;
  Buf buffer[SPEECH_BUFFERS];
  int rows, blocks, ready;
  int norm, bias, heads, attention, sum, add, gelu, relu, round, quant, conv_rows, activity, fuse, pack;
};
static float rounded(float value) {
  uint32_t bits;
  memcpy(&bits, &value, 4);
  bits = (bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000;
  memcpy(&value, &bits, 4);
  return value;
}
static void vector(SpeechEncoder *s, Buf *out, const char *name) {
  weights_vector(s->weights, out, st_get(s->checkpoint, "%s", name), 0, 0);
}
static void norm_load(SpeechEncoder *s, Norm *out, const char *prefix) {
  char name[320];
  snprintf(name, sizeof name, "%s.weight", prefix);
  vector(s, &out->weight, name);
  snprintf(name, sizeof name, "%s.bias", prefix);
  vector(s, &out->bias, name);
}
static void projection(SpeechEncoder *s, Projection *out, const char *prefix) {
  linear_load(s->weights, &out->matrix, s->checkpoint, prefix);
  if (st_has(s->checkpoint, "%s.bias", prefix)) {
    char name[320];
    snprintf(name, sizeof name, "%s.bias", prefix);
    vector(s, &out->bias, name);
  }
}
static void transformer(SpeechEncoder *s, Transformer *t, const char *prefix, int width, int heads, int layers,
                        int rotary, int qnorm) {
  *t = (Transformer){.width = width, .heads = heads, .layers = layers, .rotary = rotary, .qnorm = qnorm};
  char name[320];
  snprintf(name, sizeof name, "%s.pre_encode.proj", prefix);
  projection(s, &t->stack, name);
  snprintf(name, sizeof name, "%s.embed_norm", prefix);
  norm_load(s, &t->input, name);
  snprintf(name, sizeof name, "%s.final_norm", prefix);
  norm_load(s, &t->final, name);
  t->layer = xmalloc((size_t)layers * sizeof *t->layer);
  for (int i = 0; i < layers; i++) {
    SpeechLayer *l = t->layer + i;
#define N(field, suffix)                                                                                               \
  snprintf(name, sizeof name, "%s.layers.%d.%s", prefix, i, suffix);                                                   \
  norm_load(s, &l->field, name)
#define P(field, suffix)                                                                                               \
  snprintf(name, sizeof name, "%s.layers.%d.%s", prefix, i, suffix);                                                   \
  projection(s, &l->field, name)
    N(input, "norm1");
    N(mlp, "norm2");
    P(qkv, "attn.w_qkv");
    P(output, "attn.out_proj");
    P(up, "ffn.net.0");
    P(down, "ffn.net.3");
    if (qnorm) {
      N(q, "attn.q_norm");
      N(k, "attn.k_norm");
    }
#undef N
#undef P
  }
}
static void kernel_fill(Buf *out, const void *parameters) {
  float *values = (float *)out->ptr;
  for (int speaker = 0; speaker < 8; speaker++)
    for (int d = 0; d < 1280; d++) {
      float angle = speaker * expf((d / 2 * 2) * (-logf(10000) / 1280));
      values[speaker * 1280 + d] = rounded(d % 2 ? cosf(angle) : sinf(angle));
    }
}
static void release_workspace(void *context) {
  SpeechEncoder *s = context;
  for (int i = 0; i < SPEECH_BUFFERS; i++)
    gpu_free(s->buffer + i);
  s->rows = s->blocks = 0;
  gpu_residency_commit();
}
SpeechEncoder *speech_open(TensorIndex *checkpoint, const Json *config) {
  if (!json_get(config, "perception"))
    return NULL;
  SpeechEncoder *s = xmalloc(sizeof *s);
  s->checkpoint = checkpoint;
  s->weights = weights_new(memory_current()->account, "Transcribe speech weights");
  memory_slot_init(&s->workspace, memory_current()->account, weights_slot(s->weights), "speech workspace",
                   release_workspace, s);
  vector(s, &s->window, "perception.preprocessor.featurizer.window");
  vector(s, &s->filter, "perception.preprocessor.featurizer.fb");
  vector(s, &s->silence, "perception.encoder.diarization_model.sortformer_modules.learnable_sil_emb");
  transformer(s, &s->asr, "perception.encoder.asr_encoder", 1280, 10, 32, 64, 1);
  transformer(s, &s->diar, "perception.encoder.diarization_model.encoder", 512, 8, 31, 64, 0);
  projection(s, &s->speaker_project, "perception.encoder.diarization_model.sortformer_modules.encoder_proj");
  projection(s, &s->speaker_hidden, "perception.encoder.diarization_model.sortformer_modules.first_hidden_to_hidden");
  projection(s, &s->speakers, "perception.encoder.diarization_model.sortformer_modules.single_hidden_to_spks");
  Tensor up = st_get(checkpoint, "perception.encoder.diarization_model.sortformer_modules.subpixel_upsample.weight");
  s->upsample.matrix = (Linear){.rows = 1536, .cols = 576, .kind = 'b', .scale = 1, .input_scale = 1};
  weights_tensor(s->weights, &s->upsample.matrix.w, up, 0);
  vector(s, &s->upsample.bias, "perception.encoder.diarization_model.sortformer_modules.subpixel_upsample.bias");
  norm_load(s, &s->asr_norm, "perception.encoder.asr_norm");
  norm_load(s, &s->diar_norm, "perception.encoder.diar_norm");
  projection(s, &s->output, "perception.proj");
  weights_generated(s->weights, &s->speaker_kernel, 8 * 1280 * 4, kernel_fill, NULL, 0);
  return s;
}
int speech_prepare(const char *directory, Media *media, const void *data, size_t length, const char **error) {
  return audio_open(media, data, length, error);
}
static void initialize(SpeechEncoder *s) {
  if (s->ready)
    return;
  linear_init();
#define K(field, name) s->field = gpu_kernel(name)
  K(norm, "speech_norm");
  K(bias, "speech_bias");
  K(heads, "speech_heads");
  K(attention, "nm_attention");
  K(sum, "nm_attention_sum");
  K(add, "nm_add");
  K(gelu, "qv_gelu");
  K(relu, "speech_relu");
  K(round, "round_bf16");
  K(quant, "quantize_input");
  K(conv_rows, "speech_conv_rows");
  K(activity, "speech_activity");
  K(fuse, "speech_fuse");
  K(pack, "qv_pack");
#undef K
  s->ready = 1;
}
static int workspace(SpeechEncoder *s, int rows) {
  int blocks = (rows + 255) / 256;
  size_t sizes[SPEECH_BUFFERS] = {rows * 1536ull * 4,
                                  rows * 1536ull * 4,
                                  rows * 3840ull * 4,
                                  rows * 1280ull * 4,
                                  rows * 1280ull * 2,
                                  rows * 1280ull * 2,
                                  rows * 1280ull * 4,
                                  rows * 10ull * blocks * 128 * 4,
                                  rows * 10ull * blocks * 8,
                                  rows * 72ull,
                                  rows * 5120ull * 4,
                                  rows * 5120ull * 4,
                                  ((rows + 127) / 128) * 128ull * 5120 * 2,
                                  rows * 5120ull * 4,
                                  rows * 1280ull * 4,
                                  rows * 8ull * 4};
  uint64_t required = 0;
  for (int i = 0; i < SPEECH_BUFFERS; i++)
    required += gpu_allocation_size(sizes[i]);
  MemorySlot *slots[] = {&s->workspace};
  if (!memory_admit(slots, &required, 1, MEMORY_REQUIRED))
    return 0;
  memory_pin(&s->workspace, 1);
  MemorySlot *before = memory_scope(&s->workspace);
  gpu_sync();
  for (int i = 0; i < SPEECH_BUFFERS; i++)
    if (s->buffer[i].size != sizes[i]) {
      gpu_free(s->buffer + i);
      s->buffer[i] = gpu_alloc(sizes[i]);
    }
  memory_scope(before);
  memory_trim(&s->workspace);
  s->rows = rows;
  s->blocks = blocks;
  return 1;
}
static void unary(SpeechEncoder *s, int kernel, const Buf *x, int count) {
  uint32_t n = (uint32_t)count;
  gpu_kern(kernel);
  gpu_buf(0, x, 0);
  gpu_bytes(1, &n, 4);
  gpu_go((n + 255) / 256, 1, 1, 256, 1, 1);
}
static void product(SpeechEncoder *s, const Projection *p, const Buf *x, const Buf *out, int rows) {
  Linear matrix = p->matrix;
  const Buf *values = x;
  if (matrix.kind == '8') {
    struct {
      uint32_t count, bits;
      float scale;
      uint32_t pad;
    } q = {(uint32_t)(rows * matrix.cols), 9, matrix.input_scale, 0};
    gpu_kern(s->quant);
    gpu_buf(0, x, 0);
    gpu_buf(1, &s->buffer[QUANT], 0);
    gpu_bytes(2, &q, sizeof q);
    gpu_go((q.count + 255) / 256, 1, 1, 256, 1, 1);
    values = &s->buffer[QUANT];
    matrix.scale *= matrix.input_scale;
  }
  linear_product(&matrix, values, 0, out, 0, rows, 1, &s->buffer[PACKED], 0, 0);
  if (p->bias.size) {
    uint32_t shape[2] = {(uint32_t)matrix.rows, (uint32_t)rows};
    gpu_kern(s->bias);
    gpu_buf(0, out, 0);
    gpu_buf(1, &p->bias, 0);
    gpu_bytes(2, shape, sizeof shape);
    gpu_go((shape[0] * shape[1] + 255) / 256, 1, 1, 256, 1, 1);
  } else
    unary(s, s->round, out, rows * matrix.rows);
}
static void normalize(SpeechEncoder *s, const Norm *norm, const Buf *x, const Buf *out, int rows, int width) {
  struct {
    uint32_t width, rows, offset, stride;
    float eps;
    uint32_t round;
  } p = {(uint32_t)width, (uint32_t)rows, 0, (uint32_t)width, 1e-5f, 1};
  gpu_kern(s->norm);
  gpu_buf(0, x, 0);
  gpu_buf(1, &norm->weight, 0);
  gpu_buf(2, &norm->bias, 0);
  gpu_buf(3, out, 0);
  gpu_bytes(4, &p, sizeof p);
  gpu_go(rows, 1, 1, 256, 1, 1);
}
static void add(SpeechEncoder *s, const Buf *target, const Buf *value, int count) {
  uint32_t n = (uint32_t)count;
  gpu_kern(s->add);
  gpu_buf(0, target, 0);
  gpu_buf(1, value, 0);
  gpu_bytes(2, &n, 4);
  gpu_go((n + 255) / 256, 1, 1, 256, 1, 1);
}
static void attention(SpeechEncoder *s, const Transformer *t, const SpeechLayer *l, int rows) {
  int width = t->width, dim = width / t->heads;
  struct {
    uint32_t width, dim, rot, norm;
    float theta;
    uint32_t rows;
  } p = {(uint32_t)width, (uint32_t)dim, (uint32_t)t->rotary, (uint32_t)t->qnorm, 10000, (uint32_t)rows};
  gpu_kern(s->heads);
  gpu_buf(0, &s->buffer[QKV], 0);
  gpu_buf(1, t->qnorm ? &l->q.weight : &s->window, 0);
  gpu_buf(2, t->qnorm ? &l->q.bias : &s->window, 0);
  gpu_buf(3, t->qnorm ? &l->k.weight : &s->window, 0);
  gpu_buf(4, t->qnorm ? &l->k.bias : &s->window, 0);
  gpu_buf(5, &s->buffer[Q], 0);
  gpu_buf(6, &s->buffer[K], 0);
  gpu_buf(7, &s->buffer[V], 0);
  gpu_bytes(8, &p, sizeof p);
  gpu_go(t->heads * 3, rows, 1, 128, 1, 1);
  struct {
    uint32_t heads, kv, dim, layer, blocks, causal, length, pad;
    float scale;
  } a = {(uint32_t)t->heads, (uint32_t)t->heads, (uint32_t)dim, 0, (uint32_t)s->blocks, 0, (uint32_t)rows, 0,
         1 / sqrtf(dim)};
  gpu_kern(s->attention);
  gpu_buf(0, &s->buffer[Q], 0);
  gpu_buf(1, &s->buffer[SEQUENCES], 0);
  gpu_buf(2, &s->buffer[PARTS], 0);
  gpu_buf(3, &s->buffer[MAXIMA], 0);
  gpu_bytes(4, &a, sizeof a);
  gpu_go(t->heads, s->blocks, rows, 256, 1, 1);
  gpu_kern(s->sum);
  gpu_buf(0, &s->buffer[PARTS], 0);
  gpu_buf(1, &s->buffer[MAXIMA], 0);
  gpu_buf(2, &s->buffer[SEQUENCES], 0);
  gpu_buf(3, &s->buffer[ATTENTION], 0);
  gpu_bytes(4, &a, sizeof a);
  gpu_go(t->heads, rows, 1, 128, 1, 1);
}
static void run_transformer(SpeechEncoder *s, const Transformer *t, int rows, int stacked) {
  if (!stacked)
    product(s, &t->stack, &s->buffer[NORMAL], &s->buffer[X], rows);
  normalize(s, &t->input, &s->buffer[X], &s->buffer[X], rows, t->width);
  for (int i = 0; i < t->layers; i++) {
    const SpeechLayer *l = t->layer + i;
    normalize(s, &l->input, &s->buffer[X], &s->buffer[NORMAL], rows, t->width);
    product(s, &l->qkv, &s->buffer[NORMAL], &s->buffer[QKV], rows);
    attention(s, t, l, rows);
    product(s, &l->output, &s->buffer[ATTENTION], &s->buffer[OUTPUT], rows);
    add(s, &s->buffer[X], &s->buffer[OUTPUT], rows * t->width);
    normalize(s, &l->mlp, &s->buffer[X], &s->buffer[NORMAL], rows, t->width);
    product(s, &l->up, &s->buffer[NORMAL], &s->buffer[UP], rows);
    uint32_t shape[2] = {(uint32_t)(rows * l->up.matrix.rows), 1};
    gpu_kern(s->gelu);
    gpu_buf(0, &s->buffer[UP], 0);
    gpu_bytes(1, shape, sizeof shape);
    gpu_go((shape[0] + 255) / 256, 1, 1, 256, 1, 1);
    product(s, &l->down, &s->buffer[UP], &s->buffer[OUTPUT], rows);
    add(s, &s->buffer[X], &s->buffer[OUTPUT], rows * t->width);
  }
  normalize(s, &t->final, &s->buffer[X], &s->buffer[X], rows, t->width);
}
static float *mel_features(SpeechEncoder *s, const Media *media, int *frames) {
  *frames = media->sample_count / 160;
  float *mel = memory_host_required((size_t)*frames * 128 * 4, 0);
  float real[512], imag[512], fr[512], fi[512], power[257];
  vDSP_DFT_Setup setup = vDSP_DFT_zop_CreateSetup(NULL, 512, vDSP_DFT_FORWARD);
  const float *window = (const float *)s->window.ptr, *filter = (const float *)s->filter.ptr;
  for (int row = 0; row < *frames; row++) {
    if (!(row % 128) && shutdown_requested()) {
      vDSP_DFT_DestroySetup(setup);
      memory_host_free(mel);
      return NULL;
    }
    memset(real, 0, sizeof real);
    memset(imag, 0, sizeof imag);
    for (int j = 0; j < 400; j++) {
      int at = row * 160 + j - 200;
      float value =
          at >= 0 && at < media->sample_count ? media->samples[at] - (at ? .97f * media->samples[at - 1] : 0) : 0;
      real[j + 56] = value * window[j];
    }
    vDSP_DFT_Execute(setup, real, imag, fr, fi);
    for (int i = 0; i < 257; i++)
      power[i] = fr[i] * fr[i] + fi[i] * fi[i];
    for (int f = 0; f < 128; f++) {
      float sum = 0;
      for (int i = 0; i < 257; i++)
        sum += filter[f * 257 + i] * power[i];
      mel[(size_t)row * 128 + f] = logf(sum + 0x1p-24f);
    }
  }
  vDSP_DFT_DestroySetup(setup);
  return mel;
}
typedef struct {
  int cached, queued, compressed;
  float cache[188 * 512], cache_prob[188 * 8], fifo[40 * 512];
} SpeakerState;
typedef struct {
  float score;
  int index;
} RankedFrame;
static int descending(const void *a, const void *b) {
  const RankedFrame *x = a, *y = b;
  if (x->score != y->score)
    return x->score > y->score ? -1 : 1;
  return (x->index > y->index) - (x->index < y->index);
}
static int ascending_index(const void *a, const void *b) {
  int x = *(const int *)a, y = *(const int *)b;
  return (x > y) - (x < y);
}
static void compress_speakers(SpeechEncoder *s, SpeakerState *state, const float *embeddings,
                              const float *probabilities, int rows) {
  float *scores = memory_host_required((size_t)(rows + 1) * 8 * 4, 0);
  RankedFrame *ranked = memory_host_required((size_t)(rows + 1) * 8 * sizeof *ranked, 0);
  for (int r = 0; r < rows; r++) {
    float log_off[8], sum = 0;
    for (int speaker = 0; speaker < 8; speaker++) {
      log_off[speaker] = rounded(logf(fmaxf(rounded(1 - probabilities[r * 8 + speaker]), .25f)));
      sum += log_off[speaker];
    }
    sum = rounded(sum);
    for (int speaker = 0; speaker < 8; speaker++) {
      float on = rounded(logf(fmaxf(probabilities[r * 8 + speaker], .25f)));
      scores[r * 8 + speaker] = probabilities[r * 8 + speaker] > .5f
                                    ? rounded(rounded(rounded(on - log_off[speaker]) + sum) - logf(.5f))
                                    : -INFINITY;
    }
  }
  for (int speaker = 0; speaker < 8; speaker++) {
    int positive = 0;
    for (int r = 0; r < rows; r++)
      positive += scores[r * 8 + speaker] > 0;
    for (int r = 0; r < rows; r++) {
      if (positive >= 11 && scores[r * 8 + speaker] <= 0)
        scores[r * 8 + speaker] = -INFINITY;
      if (r >= 188)
        scores[r * 8 + speaker] = rounded(scores[r * 8 + speaker] + .05f);
    }
    for (int pass = 0; pass < 2; pass++) {
      for (int r = 0; r < rows; r++)
        ranked[r] = (RankedFrame){scores[r * 8 + speaker], r};
      qsort(ranked, rows, sizeof *ranked, descending);
      int count = pass ? 33 : 16;
      float boost = (pass ? 1 : 2) * logf(.5f);
      for (int r = 0; r < count; r++)
        scores[ranked[r].index * 8 + speaker] = rounded(ranked[r].score - boost);
    }
  }
  for (int speaker = 0; speaker < 8; speaker++)
    for (int r = 0; r <= rows; r++) {
      int at = speaker * (rows + 1) + r;
      ranked[at] = (RankedFrame){r == rows ? INFINITY : scores[r * 8 + speaker], at};
    }
  qsort(ranked, (size_t)(rows + 1) * 8, sizeof *ranked, descending);
  int selected[188];
  for (int i = 0; i < 188; i++)
    selected[i] = ranked[i].score == -INFINITY ? 99999 : ranked[i].index;
  qsort(selected, 188, sizeof *selected, ascending_index);
  for (int i = 0; i < 188; i++) {
    int row = selected[i] % (rows + 1), silent = selected[i] == 99999 || row == rows;
    memcpy(state->cache + i * 512, silent ? (const float *)s->silence.ptr : embeddings + (size_t)row * 512, 512 * 4);
    if (silent)
      memset(state->cache_prob + i * 8, 0, 8 * 4);
    else
      memcpy(state->cache_prob + i * 8, probabilities + row * 8, 8 * 4);
  }
  state->cached = 188;
  state->compressed = 1;
  memory_host_free(scores);
  memory_host_free(ranked);
}
static void update_speakers(SpeechEncoder *s, SpeakerState *state, const float *chunk, const float *predictions,
                            int left, int core) {
  int total = state->queued + core, pop = total - 40;
  if (pop < 300)
    pop = 300;
  if (pop > total)
    pop = total;
  float *pending = memory_host_required((size_t)total * 512 * 4, 0),
        *prob = memory_host_required((size_t)total * 8 * 4, 0);
  memcpy(pending, state->fifo, (size_t)state->queued * 512 * 4);
  memcpy(pending + (size_t)state->queued * 512, chunk + (size_t)left * 512, (size_t)core * 512 * 4);
  memcpy(prob, predictions + state->cached * 8, (size_t)state->queued * 8 * 4);
  memcpy(prob + state->queued * 8, predictions + (state->cached + state->queued + left) * 8, (size_t)core * 8 * 4);
  int count = state->cached + pop;
  float *candidate = memory_host_required((size_t)count * 512 * 4, 0),
        *candidate_prob = memory_host_required((size_t)count * 8 * 4, 0);
  memcpy(candidate, state->cache, (size_t)state->cached * 512 * 4);
  memcpy(candidate_prob, state->compressed ? state->cache_prob : predictions, (size_t)state->cached * 8 * 4);
  memcpy(candidate + (size_t)state->cached * 512, pending, (size_t)pop * 512 * 4);
  memcpy(candidate_prob + state->cached * 8, prob, (size_t)pop * 8 * 4);
  state->queued = total - pop;
  memcpy(state->fifo, pending + (size_t)pop * 512, (size_t)state->queued * 512 * 4);
  if (count > 188)
    compress_speakers(s, state, candidate, candidate_prob, count);
  else {
    state->cached = count;
    memcpy(state->cache, candidate, (size_t)count * 512 * 4);
    memcpy(state->cache_prob, candidate_prob, (size_t)count * 8 * 4);
  }
  memory_host_free(pending);
  memory_host_free(prob);
  memory_host_free(candidate);
  memory_host_free(candidate_prob);
}
static void set_sequences(SpeechEncoder *s, int rows) {
  for (int row = 0; row < rows; row++) {
    uint64_t *seq = (uint64_t *)(s->buffer[SEQUENCES].ptr + (size_t)row * 72);
    memset(seq, 0, 72);
    seq[4] = s->buffer[K].gpu;
    seq[5] = s->buffer[V].gpu;
    ((uint32_t *)seq)[12] = row;
    ((uint32_t *)seq)[13] = rows;
  }
}
static void speaker_head(SpeechEncoder *s, int rows, int tail) {
  product(s, &s->speaker_project, &s->buffer[X], &s->buffer[OUTPUT], rows);
  uint32_t cp[2] = {192, (uint32_t)rows};
  gpu_kern(s->conv_rows);
  gpu_buf(0, &s->buffer[OUTPUT], 0);
  gpu_buf(1, &s->buffer[NORMAL], 0);
  gpu_bytes(2, cp, sizeof cp);
  gpu_go((576 + 255) / 256, rows, 1, 256, 1, 1);
  product(s, &s->upsample, &s->buffer[NORMAL], &s->buffer[UP], rows);
  unary(s, s->relu, &s->buffer[UP], rows * 8 * 192);
  product(s, &s->speaker_hidden, &s->buffer[UP], &s->buffer[OUTPUT], rows * 8);
  unary(s, s->relu, &s->buffer[OUTPUT], rows * 8 * 192);
  product(s, &s->speakers, &s->buffer[OUTPUT], &s->buffer[UP], rows * 8);
  uint32_t p[2] = {(uint32_t)rows, (uint32_t)tail};
  gpu_kern(s->activity);
  gpu_buf(0, &s->buffer[UP], 0);
  gpu_buf(1, &s->buffer[ACTIVITY], 0);
  gpu_bytes(2, p, sizeof p);
  gpu_go((rows * 8 + 255) / 256, 1, 1, 256, 1, 1);
}
int speech_encode(SpeechEncoder *s, Media *media) {
  if (media->rows)
    return 1;
  initialize(s);
  memory_pin(weights_slot(s->weights), 1);
  if (!weights_load(s->weights)) {
    memory_pin(weights_slot(s->weights), 0);
    return 0;
  }
  int frames, total = media->ntokens;
  float *mel = mel_features(s, media, &frames);
  if (!mel) {
    memory_pin(weights_slot(s->weights), 0);
    return 0;
  }
  float mean[128], inverse[128];
  for (int f = 0; f < 128; f++) {
    double sum = 0;
    for (int r = 0; r < frames; r++)
      sum += mel[(size_t)r * 128 + f];
    mean[f] = (float)(sum / frames);
    double variance = 0;
    for (int r = 0; r < frames; r++) {
      double value = mel[(size_t)r * 128 + f] - mean[f];
      variance += value * value;
    }
    inverse[f] = 1 / (sqrtf((float)(frames > 1 ? variance / (frames - 1) : 0)) + 1e-5f);
  }
  SpeakerState *speakers = memory_host_required(sizeof *speakers, 0);
  media->embedding_width = 2688;
  media->features = gpu_alloc((size_t)total * 2688 * 2);
  for (int start = 0; start < total; start += 500) {
    int end = start + 500 < total ? start + 500 : total;
    int context_start = start > 50 ? start - 50 : 0, context_end = end + 50 < total ? end + 50 : total;
    int context = context_end - context_start, core = end - start, left = start - context_start;
    int combined = speakers->cached + speakers->queued + context;
    if (!workspace(s, combined)) {
      memory_host_free(speakers);
      memory_host_free(mel);
      memory_pin(weights_slot(s->weights), 0);
      return 0;
    }
    float *stack = (float *)s->buffer[NORMAL].ptr;
    for (int r = 0; r < context * 8; r++)
      for (int f = 0; f < 128; f++) {
        int at = context_start * 8 + r;
        stack[(size_t)r * 128 + f] = at < frames ? rounded((mel[(size_t)at * 128 + f] - mean[f]) * inverse[f]) : 0;
      }
    set_sequences(s, context);
    gpu_begin();
    run_transformer(s, &s->asr, context, 0);
    gpu_commit();
    gpu_sync();
    memcpy(s->buffer[ASR].ptr, s->buffer[X].ptr + (size_t)left * 1280 * 4, (size_t)core * 1280 * 4);
    for (int r = 0; r < context * 8; r++)
      for (int f = 0; f < 128; f++) {
        int at = context_start * 8 + r;
        stack[(size_t)r * 128 + f] = at < frames ? rounded(mel[(size_t)at * 128 + f]) : 0;
      }
    gpu_begin();
    product(s, &s->diar.stack, &s->buffer[NORMAL], &s->buffer[X], context);
    gpu_commit();
    gpu_sync();
    float *chunk = memory_host_required((size_t)context * 512 * 4, 0);
    memcpy(chunk, s->buffer[X].ptr, (size_t)context * 512 * 4);
    memcpy(s->buffer[X].ptr, speakers->cache, (size_t)speakers->cached * 512 * 4);
    memcpy(s->buffer[X].ptr + (size_t)speakers->cached * 512 * 4, speakers->fifo, (size_t)speakers->queued * 512 * 4);
    memcpy(s->buffer[X].ptr + (size_t)(speakers->cached + speakers->queued) * 512 * 4, chunk,
           (size_t)context * 512 * 4);
    set_sequences(s, combined);
    gpu_begin();
    run_transformer(s, &s->diar, combined, 1);
    int tail = context_end == total && frames % 8 ? frames % 8 : 8;
    speaker_head(s, combined, tail);
    gpu_commit();
    gpu_sync();
    float *probabilities = (float *)s->buffer[ACTIVITY].ptr;
    int first = speakers->cached + speakers->queued + left;
    float *activity = memory_host_required((size_t)core * 8 * 4, 0);
    for (int i = 0; i < core * 8; i++)
      activity[i] = probabilities[first * 8 + i] > .5f ? 1 : 0;
    if (end < total)
      update_speakers(s, speakers, chunk, probabilities, left, core);
    memory_host_free(chunk);
    memcpy(s->buffer[ACTIVITY].ptr, activity, (size_t)core * 8 * 4);
    memory_host_free(activity);
    gpu_begin();
    normalize(s, &s->diar_norm, &s->buffer[ACTIVITY], &s->buffer[ACTIVITY], core, 8);
    normalize(s, &s->asr_norm, &s->buffer[ASR], &s->buffer[ASR], core, 1280);
    uint32_t fp[2] = {1280, (uint32_t)core};
    gpu_kern(s->fuse);
    gpu_buf(0, &s->buffer[ASR], 0);
    gpu_buf(1, &s->buffer[ACTIVITY], 0);
    gpu_buf(2, &s->speaker_kernel, 0);
    gpu_bytes(3, fp, sizeof fp);
    gpu_go((core * 1280 + 255) / 256, 1, 1, 256, 1, 1);
    product(s, &s->output, &s->buffer[ASR], &s->buffer[OUTPUT], core);
    uint32_t pack[4] = {2688, 2688, (uint32_t)core, 0};
    gpu_kern(s->pack);
    gpu_buf(0, &s->buffer[OUTPUT], 0);
    gpu_buf(1, &media->features, (size_t)start * 2688 * 2);
    gpu_bytes(2, pack, sizeof pack);
    gpu_go((2688 + 255) / 256, core, 1, 256, 1, 1);
    gpu_commit();
    gpu_sync();
    if (shutdown_requested()) {
      memory_host_free(speakers);
      memory_host_free(mel);
      memory_pin(&s->workspace, 0);
      memory_pin(weights_slot(s->weights), 0);
      return 0;
    }
  }
  media->rows = (uint16_t *)media->features.ptr;
  memory_host_free(speakers);
  memory_host_free(mel);
  memory_pin(&s->workspace, 0);
  memory_pin(weights_slot(s->weights), 0);
  return 1;
}

uint64_t speech_weight_bytes(const SpeechEncoder *s) {
  return s ? weights_resident_bytes(s->weights) : 0;
}
void speech_reside(SpeechEncoder *s, int active) {
  if (s)
    weights_reside(s->weights, active);
}
