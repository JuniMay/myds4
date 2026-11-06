// Dense decode memory-layout probe. Every variant retains the current
// products and reduction order. Generated weights stay in RAM and exceed
// 256 MiB per layout so that a small matrix cannot remain wholly cached.
#include "myds4.h"
#include "../reference/activation_format.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef struct { uint32_t n, k, group_rows, group_k, round, split; } MV;
typedef struct { MV mv; uint32_t stride; } PaddedMV;
typedef struct { int fp8, n, k, round; const char *name; } Shape;
enum { LAYOUTS = 4 };
static const int padding[LAYOUTS] = {0, 16, 32, 64};
static Buf weights[LAYOUTS], scales, x, out;
static int base_fp8, base_bf16, pad_fp8, pad_bf16;
static uint32_t rng = 79;
static uint32_t word(void) { rng ^= rng << 13; rng ^= rng >> 17; return rng ^= rng << 5; }

static MV parameters(Shape s) {
  uint32_t split = 1;
  while (split < 8 && (uint64_t)s.n * split / 32 < 1024 && s.k / 16 / (split * 2) >= 64) split *= 2;
  return (MV){s.n, s.k, s.n, 0, s.round, split};
}

static void encode(Shape s, int variant, int matrix) {
  int layout = variant < 0 ? 0 : variant, bytes = s.fp8 ? 1 : 2;
  PaddedMV p = {parameters(s), s.k + padding[layout]};
  gpu_kern(variant < 0 ? (s.fp8 ? base_fp8 : base_bf16) : (s.fp8 ? pad_fp8 : pad_bf16));
  gpu_buf(0, &weights[layout], (size_t)matrix * s.n * p.stride * bytes);
  if (s.fp8) gpu_buf(1, &scales, (size_t)matrix * (s.n / 32) * (s.k / 32));
  gpu_buf(2, &x, 0), gpu_buf(3, &out, 0);
  gpu_bytes(4, &p, variant < 0 ? sizeof(MV) : sizeof p);
  gpu_go((s.n * p.mv.split + 31) / 32, 1, 1, 256, 1, 1);
}

static double run(Shape s, int variant, int copies) {
  double before = gpu_busy;
  gpu_begin();
  for (int i = 0; i < copies; i++) encode(s, variant, i);
  gpu_commit(), gpu_sync();
  return (gpu_busy - before) * 1000 / copies;
}

static double median(double *a, int n) {
  for (int i = 1; i < n; i++)
    for (int j = i; j > 0 && a[j] < a[j - 1]; j--) { double t = a[j]; a[j] = a[j - 1], a[j - 1] = t; }
  return a[n / 2];
}

int main(void) {
  alarm(240);
  lock_model();
  static const char source[] = {
#embed "../../src/kernels/kernels.metal"
      , '\n',
#embed "../../src/experimental/dense_padding.metal"
  };
  gpu_init(source, sizeof source);
  base_fp8 = gpu_kernel("mv_fp8"), base_bf16 = gpu_kernel("mv_bf16");
  pad_fp8 = gpu_kernel("mv_fp8_padded"), pad_bf16 = gpu_kernel("mv_bf16_padded");
  const Shape shapes[] = {{1, 5120, 8192, 1, "output projection"},
                          {1, 32768, QRANK, 1, "query projection"},
                          {1, 2304, 5120, 1, "shared expert gate"},
                          {0, VOCAB, 5120, 0, "output head"}};
  int failures = 0, checks = 0;
  for (unsigned shape = 0; shape < sizeof shapes / sizeof *shapes; shape++) {
    Shape s = shapes[shape];
    size_t matrix_bytes = (size_t)s.n * s.k * (s.fp8 ? 1 : 2);
    int copies = (int)(((size_t)256 * 1024 * 1024 + matrix_bytes - 1) / matrix_bytes);
    if (copies < 1) copies = 1;
    size_t allocated = 0;
    for (int p = 0; p < LAYOUTS; p++) allocated += (size_t)copies * s.n * (s.k + padding[p]) * (s.fp8 ? 1 : 2);
    if (allocated > (UINT64_C(6) << 30)) die("dense layout fixture exceeds its memory bound");
    for (int p = 0; p < LAYOUTS; p++) {
      weights[p] = gpu_alloc((size_t)copies * s.n * (s.k + padding[p]) * (s.fp8 ? 1 : 2));
      memset(weights[p].ptr, 0xcd, weights[p].size);
    }
    if (s.fp8) {
      for (size_t i = 0; i < weights[0].size; i++) {
        uint8_t b = word(); weights[0].ptr[i] = (b & 127) == 127 ? b - 1 : b;
      }
      scales = gpu_alloc((size_t)copies * (s.n / 32) * (s.k / 32));
      for (size_t i = 0; i < scales.size; i++) scales.ptr[i] = 112 + word() % 9;
    } else {
      uint16_t *w = (uint16_t *)weights[0].ptr;
      for (size_t i = 0; i < weights[0].size / 2; i++) {
        uint32_t b = word(); w[i] = (b & 0x8000) | ((116 + ((b >> 7) & 7)) << 7) | (b & 127);
      }
    }
    for (int p = 1; p < LAYOUTS; p++)
      for (int row = 0; row < copies * s.n; row++)
        memcpy(weights[p].ptr + (size_t)row * (s.k + padding[p]) * (s.fp8 ? 1 : 2),
               weights[0].ptr + (size_t)row * s.k * (s.fp8 ? 1 : 2), (size_t)s.k * (s.fp8 ? 1 : 2));
    x = gpu_alloc((size_t)s.k * 4), out = gpu_alloc((size_t)s.n * 4);
    float *expected = xmalloc(out.size);
    size_t errors[LAYOUTS] = {0}, compared = 0;
    for (int pattern = 0; pattern < 3; pattern++) {
      for (int first = 0; first < s.k; first += 32) {
        float values[32];
        for (int i = 0; i < 32; i++) values[i] = pattern == 0 ? 0 : ((int)(word() % 257) - 128) * (pattern == 1 ? 0x1p-10f : .03125f);
        if (s.fp8) {
          RefFp8Block packed;
          if (!ref_fp8_pack(values, &packed)) die("invalid input fixture");
          ref_fp8_unpack(&packed, (float *)x.ptr + first);
        } else for (int i = 0; i < 32; i++) ((float *)x.ptr)[first + i] = ref_from_bf16(ref_bf16(values[i]));
      }
      for (int matrix = 0; matrix < copies; matrix++) {
        gpu_begin(), encode(s, -1, matrix), gpu_commit(), gpu_sync();
        memcpy(expected, out.ptr, out.size);
        for (int p = 0; p < LAYOUTS; p++) {
          gpu_begin(), encode(s, p, matrix), gpu_commit(), gpu_sync();
          for (int i = 0; i < s.n; i++) {
            float got = ((float *)out.ptr)[i];
            errors[p] += !isfinite(got) || memcmp(&got, expected + i, sizeof got) != 0;
          }
          checks++;
        }
        compared += s.n;
      }
    }
    double times[LAYOUTS + 1][9];
    for (int v = -1; v < LAYOUTS; v++) run(s, v, copies);
    for (int sample = 0; sample < 9; sample++)
      for (int step = 0; step <= LAYOUTS; step++) {
        int v = (sample + step) % (LAYOUTS + 1);
        times[v][sample] = run(s, v - 1, copies);
      }
    double control = median(times[0], 9);
    for (int p = 0; p < LAYOUTS; p++) {
      printf("{\"dense_padding\":{\"name\":\"%s\",\"fp8\":%s,\"n\":%d,\"k\":%d,\"copies\":%d,\"padding_elements\":%d,"
             "\"extra_weight_bytes\":%zu,\"compared_values\":%zu,\"different_values\":%zu,\"control_ms\":%.6f,\"candidate_ms\":%.6f}}\n",
             s.name, s.fp8 ? "true" : "false", s.n, s.k, copies, padding[p], weights[p].size - weights[0].size,
             compared, errors[p], control, median(times[p + 1], 9));
      failures += errors[p] != 0;
    }
    fflush(stdout);
    free(expected);
    for (int p = 0; p < LAYOUTS; p++) gpu_free(weights + p);
    gpu_free(&scales), gpu_free(&x), gpu_free(&out);
  }
  fprintf(stderr, "dense padding: %d comparisons, %d failing layout/shape combinations\n", checks, failures);
  return failures != 0;
}
