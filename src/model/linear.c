#include "linear.h"
#include <math.h>
#include <stdio.h>
#include <unistd.h>

static int vector[4], few[4], three[4], matrix[4], small_matrix, convert, initialized;
void linear_init(void) {
  if (initialized)
    return;
  gpu_start();
  static const char *vectors[] = {"mv1_nvfp4", "mv1_fp8", "mv1_bf16", "mv1_mxfp8"};
  static const char *several[] = {"mvr_nvfp4", "mvr_fp8", "mvr_bf16", "mvr_mxfp8"};
  static const char *triples[] = {"mvr3_nvfp4", "mvr3_fp8", "mvr3_bf16"};
  static const char *matrices[] = {"gemm_nvfp4", "gemm_fp8", "gemm_bf16", "gemm_mxfp8"};
  for (int i = 0; i < 4; i++) {
    vector[i] = gpu_kernel(vectors[i]);
    few[i] = gpu_kernel(several[i]);
    three[i] = i < 3 ? gpu_kernel(triples[i]) : few[i];
    matrix[i] = gpu_kernel(matrices[i]);
  }
  convert = gpu_kernel("x_bf16");
  small_matrix = gpu_kernel("gemm16_nvfp4");
  initialized = 1;
}

static float scalar(Tensor tensor) {
  float value;
  if (tensor.dtype != 'f' || tensor.size != 4 || pread(tensor.fd, &value, 4, tensor.off) != 4 || !isfinite(value))
    die("invalid matrix scale");
  return value;
}

void linear_load(WeightSet *weights, Linear *m, TensorIndex *index, const char *stem) {
  Tensor w = st_get(index, "%s.weight", stem);
  if (w.ndim != 2 || w.shape[0] < 1 || w.shape[1] < 1)
    die("%s: matrix must have two positive dimensions", stem);
  *m = (Linear){.rows = (int)w.shape[0], .cols = (int)w.shape[1], .scale = 1, .input_scale = 1};
  weights_tensor(weights, &m->w, w, 0);
  if (w.dtype == 'u') {
    m->kind = 'n';
    m->cols *= 2;
    if (st_has(index, "%s.weight_scale", stem)) {
      Tensor s = st_get(index, "%s.weight_scale", stem);
      if (s.size != (size_t)m->rows * m->cols / 16)
        die("%s: NVFP4 scale dimensions differ", stem);
      weights_tensor(weights, &m->s, s, 0);
    } else {
      weights_csf(weights, &m->s, st_get(index, "%s.weight_scale.nvfp4_csf_fixed", stem),
                  st_get(index, "%s.weight_scale.nvfp4_csf_exceptions", stem), m->rows, m->cols / 16);
    }
    if (st_has(index, "%s.weight_scale_2", stem))
      m->scale = scalar(st_get(index, "%s.weight_scale_2", stem));
    if (st_has(index, "%s.input_scale", stem))
      m->input_scale = scalar(st_get(index, "%s.input_scale", stem));
  } else if (w.dtype == '8') {
    Tensor s = st_get(index, "%s.weight_scale", stem);
    if (s.dtype == 'f' && s.size == 4) {
      m->kind = '8';
      m->scale = scalar(s);
      if (st_has(index, "%s.input_scale", stem)) m->input_scale = scalar(st_get(index, "%s.input_scale", stem));
    } else {
      if ((s.dtype != 'e' && s.dtype != 'u') || s.size != (size_t)m->rows * m->cols / 32)
        die("%s: MXFP8 scales must hold one exponent per 32 values", stem);
      m->kind = 'm';
      weights_tensor(weights, &m->s, s, 0);
    }
  } else if (w.dtype == 'b')
    m->kind = 'b';
  else
    die("%s: unsupported matrix storage", stem);
  if (m->cols % 64)
    die("%s: matrix columns must be a multiple of 64", stem);
}

void linear_product(const Linear *m, const Buf *x, size_t xo, const Buf *y, size_t yo, int rows, int prompt,
                    const Buf *bf16, int staged) {
  int format = m->kind == 'n' ? 0 : m->kind == '8' ? 1 : m->kind == 'b' ? 2 : 3;
  const Buf *scales = m->s.obj ? &m->s : &m->w;
  if (prompt) {
    uint32_t k = (uint32_t)m->cols;
    if (!staged) {
      uint32_t strides[2] = {k, k};
      uint32_t threads = k % 256 == 0 ? 64 : k % 128 == 0 ? 32 : 16;
      gpu_kern(convert);
      gpu_buf(0, x, xo);
      gpu_buf(1, bf16, 0);
      gpu_bytes(2, strides, sizeof strides);
      gpu_go(k / (4 * threads), (uint32_t)rows, 1, threads, 1, 1);
    }
    struct {
      uint32_t m, n, k, pad;
      float scale;
      uint32_t pad0, pad1, pad2;
    } p = {(uint32_t)rows, (uint32_t)m->rows, k, 0, m->scale, 0, 0, 0};
    int tile = format == 0 && rows <= 64 ? 16 : 128;
    gpu_kern(tile == 16 ? small_matrix : matrix[format]);
    gpu_buf(0, &m->w, 0);
    gpu_buf(1, scales, 0);
    gpu_buf(2, bf16, 0);
    gpu_buf(3, y, yo);
    gpu_bytes(4, &p, sizeof p);
    gpu_go((m->rows + 63) / 64, (rows + tile - 1) / tile, 1, 128, 1, 1);
  } else {
    struct {
      uint32_t rows, cols, n;
      float scale;
    } p = {(uint32_t)m->rows, (uint32_t)m->cols, (uint32_t)rows, m->scale};
    gpu_kern(rows == 1 ? vector[format] : rows == 3 ? three[format] : few[format]);
    gpu_buf(0, &m->w, 0);
    gpu_buf(1, scales, 0);
    gpu_buf(2, x, xo);
    gpu_buf(3, y, yo);
    gpu_bytes(4, &p, sizeof p);
    gpu_go((m->rows + 7) / 8, 1, 1, 256, 1, 1);
  }
}
