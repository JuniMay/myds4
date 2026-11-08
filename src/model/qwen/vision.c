// Qwen's patch encoder and spatial merger. Images use the checkpoint's
// resize limits; weights remain unloaded until an image is encoded.
#include "vision.h"
#include "../weights.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef struct { Buf w, b; int rows, cols, stride, quantize; float scale; } Linear;
typedef struct { Linear qkv, projection, up, down; Buf norm[2], bias[2]; } Block;
enum { VX, VN, VQKV, VA, VF, VP, VR, VH, VB, WORK_BUFFERS };
struct QwenVision {
  TensorIndex *index;
  int depth, width, intermediate, output, heads, head, side, loaded, capacity;
  Linear patch, merge[2];
  Buf position, norm, norm_bias;
  Block *blocks;
  WeightSet *weights;
  MemorySlot workspace;
  Buf work[WORK_BUFFERS];
  int pack, bias, layernorm, heads_kernel, attention, gelu, add, positions;
};

void qwen_image_positions(const Media *images, int count, int position, int delta, int prompt, int32_t out[3]) {
  if (prompt) {
    delta = 0;
    for (int i = 0; i < count; i++) {
      const Media *im = images + i;
      if (position < im->start) break;
      if (position < im->start + im->ntokens) {
        int at = position - im->start, base = im->start + delta;
        out[0] = base; out[1] = base + at / im->tw; out[2] = base + at % im->tw;
        return;
      }
      delta += (im->th > im->tw ? im->th : im->tw) - im->ntokens;
    }
  }
  out[0] = out[1] = out[2] = position + delta;
}

static int number(const Json *config, const char *key, int fallback) {
  const Json *v = json_get(config, key);
  return v && v->type == J_NUM ? (int)v->num : fallback;
}
static float value(uint16_t x) {
  uint32_t bits = (uint32_t)x << 16;
  float out;
  memcpy(&out, &bits, 4);
  return out;
}
static uint16_t rounded(float x) {
  uint32_t bits;
  memcpy(&bits, &x, 4);
  return (uint16_t)((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}

int qwen_prepare_image(const char *dir, Media *im, const void *data, size_t size, const char **error) {
  int w, h;
  uint8_t *pixels = image_decode_rgb(data, size, &w, &h, error);
  if (!pixels) return 0;
  if (w < 1 || h < 1 || (double)(w > h ? w : h) / (w < h ? w : h) > 200) {
    free(pixels); *error = "image aspect ratio exceeds 200"; return 0;
  }
  char path[4096];
  snprintf(path, sizeof path, "%s/preprocessor_config.json", dir);
  if (access(path, R_OK)) { free(pixels); *error = "checkpoint has no image processor configuration"; return 0; }
  char *text = read_file(path, NULL);
  Json *root = json_parse(text);
  int patch = number(root, "patch_size", 16), merge = number(root, "merge_size", 2);
  int temporal = number(root, "temporal_patch_size", 2);
  const Json *limits = json_get(root, "size");
  int minimum = number(limits, "shortest_edge", 65536), maximum = number(limits, "longest_edge", 16777216);
  json_free(root); free(text);
  if (patch != 16 || merge != 2 || temporal != 2 || minimum < 1 || maximum < minimum) {
    free(pixels); *error = "unsupported Qwen image processor settings"; return 0;
  }
  int factor = patch * merge;
  int rh = (int)rint((double)h / factor) * factor, rw = (int)rint((double)w / factor) * factor;
  if (rh < factor) rh = factor;
  if (rw < factor) rw = factor;
  double area = (double)rh * rw;
  if (area > maximum) {
    double ratio = sqrt((double)h * w / maximum);
    rh = (int)floor(h / ratio / factor) * factor;
    rw = (int)floor(w / ratio / factor) * factor;
  } else if (area < minimum) {
    double ratio = sqrt((double)minimum / ((double)h * w));
    rh = (int)ceil(h * ratio / factor) * factor;
    rw = (int)ceil(w * ratio / factor) * factor;
  }
  if (rh < factor || rw < factor) { free(pixels); *error = "image grid is empty"; return 0; }
  uint8_t *resized = image_resize_rgb(pixels, w, h, rw, rh);
  free(pixels);
  im->vh = rh / patch, im->vw = rw / patch;
  im->th = im->vh / merge, im->tw = im->vw / merge;
  im->ntokens = im->th * im->tw, im->patch_stride = 3 * temporal * patch * patch;
  im->embedding_width = 0, im->start = 0, im->rows = NULL, im->features = (Buf){0};
  im->hash = 1469598103934665603ull;
  for (size_t i = 0; i < size; i++) im->hash = (im->hash ^ ((const uint8_t *)data)[i]) * 1099511628211ull;
  im->patch = memory_host_alloc((size_t)im->vh * im->vw * im->patch_stride * 2, 0);
  if (!im->patch) { free(resized); *error = "model-memory budget cannot fit the image patches"; return 0; }
  for (int py = 0; py < im->vh; py++) for (int px = 0; px < im->vw; px++) {
    int row = ((py / merge) * im->tw + px / merge) * 4 + (py % merge) * 2 + px % merge;
    for (int c = 0; c < 3; c++) for (int t = 0; t < temporal; t++)
      for (int y = 0; y < patch; y++) for (int x = 0; x < patch; x++) {
        float pixel = ((float)resized[((size_t)(py * patch + y) * rw + px * patch + x) * 3 + c] / 255.0f - 0.5f) / 0.5f;
        im->patch[(size_t)row * im->patch_stride + ((c * temporal + t) * patch + y) * patch + x] = rounded(pixel);
      }
  }
  free(resized);
  return 1;
}

static void release_workspace(void *context) {
  QwenVision *vision = context;
  for (int i = 0; i < WORK_BUFFERS; i++) gpu_free(&vision->work[i]);
  vision->capacity = 0;
  gpu_residency_commit();
}

QwenVision *qwen_vision_open(TensorIndex *index, const Json *config) {
  if (!config) return NULL;
  QwenVision *v = xmalloc(sizeof *v);
  v->index = index;
  MemorySlot *owner = memory_current() ? memory_current() : memory_shared();
  v->weights = weights_new(owner->account, "Qwen vision weights");
  memory_slot_init(&v->workspace, owner->account, weights_slot(v->weights), "vision workspace", release_workspace, v);
  v->depth = number(config, "depth", 0), v->width = number(config, "hidden_size", 0);
  v->intermediate = number(config, "intermediate_size", 0), v->output = number(config, "out_hidden_size", 0);
  v->heads = number(config, "num_heads", 0);
  v->head = v->heads ? v->width / v->heads : 0;
  v->side = (int)sqrt((double)number(config, "num_position_embeddings", 0));
  if (v->depth < 1 || v->depth > 128 || v->width != 1152 || v->heads != 16 || v->head != 72 ||
      v->intermediate < 1 || v->output < 1 || v->side < 2 || number(config, "spatial_merge_size", 0) != 2 ||
      number(config, "patch_size", 0) != 16 || number(config, "temporal_patch_size", 0) != 2)
    die("unsupported Qwen vision shape");
  v->blocks = xmalloc((size_t)v->depth * sizeof *v->blocks);
  return v;
}

static void weight(QwenVision *v, Buf *target, const char *name, int stride) {
  Tensor tensor = st_get(v->index, "%s", name);
  if (tensor.dtype != 'b') die("%s: Qwen vision requires BF16 weights", name);
  if (stride) weights_padded(v->weights, target, tensor, stride);
  else weights_tensor(v->weights, target, tensor, 0);
}

static void linear(QwenVision *v, Linear *target, const char *prefix) {
  char name[256]; snprintf(name, sizeof name, "%s.weight", prefix);
  Tensor tensor = st_get(v->index, "%s", name);
  target->rows = (int)tensor.shape[0];
  target->cols = (int)(tensor.size / target->rows / (tensor.dtype == 'b' ? 2 : 1));
  if (tensor.dtype == 'u') target->cols *= 2;
  target->stride = (target->cols + 63) / 64 * 64;
  target->scale = 1;
  if (tensor.dtype == 'b') weight(v, &target->w, name, target->stride);
  else {
    Tensor scales = st_get(v->index, "%s.weight_scale", prefix);
    int kind = tensor.dtype == 'u' ? 'n' : scales.dtype == 'u' || scales.dtype == 'e' ? 'm' : '8';
    target->quantize = kind == 'm';
    if (kind == 'n') {
      Tensor global = st_get(v->index, "%s.weight_scale_2", prefix);
      if (pread(global.fd, &target->scale, 4, global.off) != 4) die("cannot read vision scale");
    } else if (kind == '8') {
      if (pread(scales.fd, &target->scale, 4, scales.off) != 4) die("cannot read vision scale");
    }
    weights_expanded(v->weights, &target->w, tensor, scales, kind, target->cols, target->stride);
  }
  snprintf(name, sizeof name, "%s.bias", prefix); weight(v, &target->b, name, 0);
}

static void load(QwenVision *v) {
  if (v->loaded) return;
  gpu_start();
  v->pack = gpu_kernel("qv_pack"), v->bias = gpu_kernel("qv_bias"), v->layernorm = gpu_kernel("qv_norm");
  v->heads_kernel = gpu_kernel("qv_heads"), v->attention = gpu_kernel("qv_attention");
  v->gelu = gpu_kernel("qv_gelu"), v->add = gpu_kernel("vit_add"), v->positions = gpu_kernel("qv_positions");
  linear(v, &v->patch, "model.visual.patch_embed.proj");
  weight(v, &v->position, "model.visual.pos_embed.weight", 0);
  char name[256];
  for (int i = 0; i < v->depth; i++) {
    Block *b = &v->blocks[i];
    snprintf(name, sizeof name, "model.visual.blocks.%d.attn.qkv", i); linear(v, &b->qkv, name);
    snprintf(name, sizeof name, "model.visual.blocks.%d.attn.proj", i); linear(v, &b->projection, name);
    snprintf(name, sizeof name, "model.visual.blocks.%d.mlp.linear_fc1", i); linear(v, &b->up, name);
    snprintf(name, sizeof name, "model.visual.blocks.%d.mlp.linear_fc2", i); linear(v, &b->down, name);
    for (int j = 0; j < 2; j++) {
      snprintf(name, sizeof name, "model.visual.blocks.%d.norm%d.weight", i, j + 1); weight(v, &b->norm[j], name, 0);
      snprintf(name, sizeof name, "model.visual.blocks.%d.norm%d.bias", i, j + 1); weight(v, &b->bias[j], name, 0);
    }
  }
  weight(v, &v->norm, "model.visual.merger.norm.weight", 0), weight(v, &v->norm_bias, "model.visual.merger.norm.bias", 0);
  linear(v, &v->merge[0], "model.visual.merger.linear_fc1"), linear(v, &v->merge[1], "model.visual.merger.linear_fc2");
  v->loaded = 1;
}


static void matrix(QwenVision *v, const Linear *l, const Buf *input, const Buf *out, int n, int packed) {
  if (!packed) {
    uint32_t shape[4] = {(uint32_t)l->cols, (uint32_t)l->stride, (uint32_t)n, (uint32_t)l->quantize};
    gpu_kern(v->pack); gpu_buf(0, input, 0); gpu_buf(1, &v->work[VB], 0); gpu_bytes(2, shape, sizeof shape);
    gpu_go((l->stride + 255) / 256, (n + 127) / 128 * 128, 1, 256, 1, 1);
  }
  struct { uint32_t m, n, k, pad; float scale; uint32_t pad2[3]; } p = {(uint32_t)n, (uint32_t)l->rows, (uint32_t)l->stride, 0, l->scale, {0}};
  gpu_kern(gpu_kernel("gemm_bf16")); gpu_buf(0, &l->w, 0); gpu_buf(1, &l->w, 0);
  gpu_buf(2, packed ? input : &v->work[VB], 0); gpu_buf(3, out, 0); gpu_bytes(4, &p, sizeof p);
  gpu_go((l->rows + 63) / 64, (n + 127) / 128, 1, 128, 1, 1);
  uint32_t shape[2] = {(uint32_t)l->rows, (uint32_t)n};
  gpu_kern(v->bias); gpu_buf(0, out, 0); gpu_buf(1, &l->b, 0); gpu_bytes(2, shape, sizeof shape);
  gpu_go((l->rows + 255) / 256, n, 1, 256, 1, 1);
}

static void norm(QwenVision *v, const Buf *input, const Buf *weight, const Buf *bias, const Buf *out, int n) {
  uint32_t width = (uint32_t)v->width;
  gpu_kern(v->layernorm); gpu_buf(0, input, 0); gpu_buf(1, weight, 0); gpu_buf(2, bias, 0); gpu_buf(3, out, 0);
  gpu_bytes(4, &width, 4); gpu_go(n, 1, 1, 256, 1, 1);
}
static void add(QwenVision *v, const Buf *out, const Buf *input, int n) {
  gpu_kern(v->add); gpu_buf(0, out, 0); gpu_buf(1, input, 0);
  gpu_go((uint32_t)n * v->width / 256, 1, 1, 256, 1, 1);
}
static void activate(QwenVision *v, const Buf *data, int n, int exact) {
  uint32_t p[2] = {(uint32_t)n, (uint32_t)exact};
  gpu_kern(v->gelu); gpu_buf(0, data, 0); gpu_bytes(1, p, sizeof p);
  gpu_go((n + 255) / 256, 1, 1, 256, 1, 1);
}

int qwen_vision_trace(QwenVision *v, Media *im, int trace_fd) {
  if (!v) die("checkpoint has no Qwen vision encoder");
  if (im->rows) return 1;
  load(v);
  int n = im->vh * im->vw, padded = (n + 255) / 256 * 256, width = v->width;
  int wide = v->intermediate > 4 * width ? v->intermediate : 4 * width;
  int capacity = padded > v->capacity ? padded : v->capacity;
  size_t sizes[WORK_BUFFERS] = {(size_t)capacity * width * 4, (size_t)capacity * width * 4,
    (size_t)capacity * width * 12, (size_t)capacity * width * 4, (size_t)capacity * wide * 4,
    (size_t)capacity * (width > v->output / 4 ? width : v->output / 4) * 4, (size_t)capacity * 72 * 4,
    (size_t)capacity * 16 * 96 * 6, (size_t)capacity * ((wide + 63) / 64 * 64) * 2};
  MemorySlot *owner = memory_current() ? memory_current() : memory_shared(), *weights = weights_slot(v->weights);
  uint64_t work_bytes = 0, feature_bytes = gpu_allocation_size((size_t)im->ntokens * v->output * 2);
  for (int i = 0; i < WORK_BUFFERS; i++) work_bytes += gpu_allocation_size(sizes[i]);
  MemorySlot *entries[] = {weights, &v->workspace, owner};
  uint64_t required[] = {weights_required(v->weights), work_bytes, owner->payload + feature_bytes};
  if (!memory_admit(entries, required, 3, MEMORY_REQUIRED)) return 0;
  memory_pin(weights, 1); memory_pin(&v->workspace, 1);
  if (!weights_load(v->weights)) die("reserved vision weights could not load");
  if (capacity != v->capacity) {
    gpu_sync();
    release_workspace(v);
    MemorySlot *previous = memory_scope(&v->workspace);
    for (int i = 0; i < WORK_BUFFERS; i++) { v->work[i] = gpu_alloc(sizes[i]); memset(v->work[i].ptr, 0, sizes[i]); }
    memory_scope(previous);
    v->capacity = capacity;
    memory_publish(&v->workspace, 1);
  }
  im->embedding_width = v->output;
  im->features = gpu_alloc((size_t)im->ntokens * v->output * 2);
  memcpy(v->work[VB].ptr, im->patch, (size_t)n * im->patch_stride * 2);
  float *positions = (float *)v->work[VP].ptr, *rope = (float *)v->work[VR].ptr;
  const uint16_t *grid = (const uint16_t *)v->position.ptr;
  for (int row = 0; row < n; row++) {
    int group = row / 4, y = group / im->tw * 2 + row % 4 / 2, x = group % im->tw * 2 + row % 2;
    float fy = (float)y * (v->side - 1) / (im->vh - 1), fx = (float)x * (v->side - 1) / (im->vw - 1);
    int y0 = (int)fy, x0 = (int)fx, y1 = y0 + 1 < v->side ? y0 + 1 : y0, x1 = x0 + 1 < v->side ? x0 + 1 : x0;
    float wy = fy - y0, wx = fx - x0;
    for (int d = 0; d < width; d++) {
      float p = value(grid[((size_t)y0 * v->side + x0) * width + d]) * (1 - wy) * (1 - wx)
              + value(grid[((size_t)y0 * v->side + x1) * width + d]) * (1 - wy) * wx
              + value(grid[((size_t)y1 * v->side + x0) * width + d]) * wy * (1 - wx)
              + value(grid[((size_t)y1 * v->side + x1) * width + d]) * wy * wx;
      positions[(size_t)row * width + d] = value(rounded(p));
    }
    for (int d = 0; d < 36; d++) {
      float angle = (float)(d < 18 ? y : x) / powf(10000.0f, (float)(2 * (d % 18)) / 36.0f);
      rope[row * 72 + d] = cosf(angle), rope[row * 72 + 36 + d] = sinf(angle);
    }
  }
  gpu_begin();
  matrix(v, &v->patch, &v->work[VB], &v->work[VX], n, 1);
  uint32_t count = (uint32_t)n * width;
  gpu_kern(v->positions); gpu_buf(0, &v->work[VX], 0); gpu_buf(1, &v->work[VP], 0); gpu_bytes(2, &count, 4);
  gpu_go((count + 255) / 256, 1, 1, 256, 1, 1);
  if (trace_fd >= 0) { gpu_commit(); gpu_sync(); write(trace_fd, v->work[VX].ptr, (size_t)n * width * 4); gpu_begin(); }
  for (int layer = 0; layer < v->depth; layer++) {
    Block *b = &v->blocks[layer];
    norm(v, &v->work[VX], &b->norm[0], &b->bias[0], &v->work[VN], n);
    matrix(v, &b->qkv, &v->work[VN], &v->work[VQKV], n, 0);
    uint32_t dims[2] = {(uint32_t)n, (uint32_t)padded};
    gpu_kern(v->heads_kernel); gpu_buf(0, &v->work[VQKV], 0); gpu_buf(1, &v->work[VR], 0); gpu_buf(2, &v->work[VH], 0); gpu_bytes(3, dims, sizeof dims);
    gpu_go(18, padded, 1, 256, 1, 1);
    gpu_kern(v->attention); gpu_buf(0, &v->work[VH], 0); gpu_buf(1, &v->work[VA], 0); gpu_bytes(2, dims, sizeof dims);
    gpu_go((n + 63) / 64, 16, 1, 128, 1, 1);
    matrix(v, &b->projection, &v->work[VA], &v->work[VN], n, 0); add(v, &v->work[VX], &v->work[VN], n);
    norm(v, &v->work[VX], &b->norm[1], &b->bias[1], &v->work[VN], n);
    matrix(v, &b->up, &v->work[VN], &v->work[VF], n, 0); activate(v, &v->work[VF], n * v->intermediate, 0);
    matrix(v, &b->down, &v->work[VF], &v->work[VN], n, 0); add(v, &v->work[VX], &v->work[VN], n);
    if (trace_fd >= 0) { gpu_commit(); gpu_sync(); write(trace_fd, v->work[VX].ptr, (size_t)n * width * 4); gpu_begin(); }
  }
  norm(v, &v->work[VX], &v->norm, &v->norm_bias, &v->work[VN], n);
  matrix(v, &v->merge[0], &v->work[VN], &v->work[VF], n / 4, 0); activate(v, &v->work[VF], n * width, 1);
  matrix(v, &v->merge[1], &v->work[VF], &v->work[VP], n / 4, 0);
  uint32_t shape[4] = {(uint32_t)v->output, (uint32_t)v->output, (uint32_t)im->ntokens, 0};
  gpu_kern(v->pack); gpu_buf(0, &v->work[VP], 0); gpu_buf(1, &im->features, 0); gpu_bytes(2, shape, sizeof shape);
  gpu_go((v->output + 255) / 256, im->ntokens, 1, 256, 1, 1);
  gpu_commit(); gpu_sync();
  im->rows = (uint16_t *)im->features.ptr;
  memory_pin(weights, 0); memory_pin(&v->workspace, 0);
  return 1;
}

int qwen_vision_encode(QwenVision *v, Media *im) { return qwen_vision_trace(v, im, -1); }

void qwen_vision_reside(QwenVision *v, int resident) {
  if (!v) return;
  weights_reside(v->weights, resident);
  for (int i = 0; i < WORK_BUFFERS; i++) if (v->work[i].obj) gpu_resident(&v->work[i], resident);
}
uint64_t qwen_vision_bytes(const QwenVision *v) { return v ? weights_resident_bytes(v->weights) : 0; }
void qwen_vision_close(QwenVision *v) {
  if (!v) return;
  release_workspace(v);
  memory_slot_remove(&v->workspace);
  weights_free(v->weights);
  free(v->blocks); free(v);
}
