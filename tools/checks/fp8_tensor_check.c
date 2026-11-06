// Direct FP8 operands versus current BF16-staged FP8 dense products.
#include "myds4.h"
#include "../reference/activation_format.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef struct { uint32_t m, n, k, round, lda, ldy; } MM;
static Buf x, xb, codes, scales, w, ws, row_scales, out;
static int pack, bf_pack, control, native;
static int mode;
static const char *mode_name = "fp8-64";
static uint32_t rng = 13;
static uint32_t random_word(void) { rng ^= rng << 13; rng ^= rng >> 17; return rng ^= rng << 5; }

static double run(const MM *p, int candidate, int repeats) {
  uint32_t kl[2] = {p->k, p->k};
  double before = gpu_busy;
  gpu_begin();
  for (int r = 0; r < repeats; r++) {
    gpu_kern(candidate ? pack : bf_pack);
    gpu_buf(0, &x, 0);
    if (candidate && mode != 2) {
      gpu_buf(1, &codes, 0), gpu_buf(2, &scales, 0), gpu_bytes(3, kl, sizeof kl);
      gpu_go((p->k + 255) / 256, p->m, 1, 256, 1, 1);
    } else {
      gpu_buf(1, &xb, 0), gpu_bytes(2, kl, sizeof kl);
      gpu_go(p->k / 256, p->m, 1, 64, 1, 1);
    }
    gpu_kern(candidate ? native : control);
    gpu_buf(0, &w, 0), gpu_buf(1, candidate ? &row_scales : &ws, 0);
    if (candidate && mode != 2) {
      gpu_buf(2, &codes, 0), gpu_buf(3, &scales, 0), gpu_buf(4, &out, 0), gpu_bytes(5, p, sizeof *p);
      gpu_go((p->n + 63) / 64, mode == 0 ? (p->m + 63) / 64 : (p->m + 127) / 128, 1, 128, 1, 1);
    } else {
      gpu_buf(2, &xb, 0), gpu_buf(3, &out, 0), gpu_bytes(4, p, sizeof *p);
      gpu_go((p->n + 63) / 64, (p->m + 127) / 128, 1, 128, 1, 1);
    }
  }
  gpu_commit(), gpu_sync();
  return (gpu_busy - before) * 1000 / repeats;
}

int main(int argc, char **argv) {
  if (argc == 3 && !strcmp(argv[1], "--mode")) {
    mode_name = argv[2];
    if (!strcmp(mode_name, "fp8-128")) mode = 1;
    else if (!strcmp(mode_name, "half-128")) mode = 2;
    else if (strcmp(mode_name, "fp8-64")) die("invalid tensor probe mode");
  } else if (argc != 1) die("usage: fp8-tensor-check [--mode fp8-64|fp8-128|half-128]");
  alarm(180);
  lock_model();
  static const char source[] = {
#embed "../../src/kernels/kernels.metal"
      , '\n',
#embed "../../src/experimental/fp8_tensor.metal"
  };
  gpu_init(source, sizeof source);
  pack = gpu_kernel(mode == 2 ? "fp8_pack_half" : "fp8_pack_native"), bf_pack = gpu_kernel("x_bf16");
  control = gpu_kernel("mm_fp8");
  native = gpu_kernel(mode == 0 ? "mm_fp8_native" : mode == 1 ? "mm_fp8_native128" : "mm_fp8_half128");
  const int shapes[][3] = {{1, 32768, 1280}, {6, 32768, 1280}, {128, 5120, 8192}, {2048, 2304, 5120}, {2048, 5120, 8192}};
  int failures = 0;
  for (unsigned shape = 0; shape < sizeof shapes / sizeof *shapes; shape++) {
    MM p = {shapes[shape][0], shapes[shape][1], shapes[shape][2], 1, shapes[shape][2], shapes[shape][1]};
    size_t padded = ((p.m + 127) / 128) * 128;
    x = gpu_alloc((size_t)p.m * p.k * 4), xb = gpu_alloc(padded * p.k * 2);
    codes = gpu_alloc(padded * p.k), scales = gpu_alloc(padded * p.k / 32);
    w = gpu_alloc((size_t)p.n * p.k), ws = gpu_alloc((size_t)p.n * p.k / 1024);
    row_scales = gpu_alloc((size_t)p.n * p.k / 32);
    out = gpu_alloc((size_t)p.m * p.n * 4);
    memset(xb.ptr, 0, xb.size), memset(codes.ptr, 0, codes.size), memset(scales.ptr, 0, scales.size);
    for (size_t i = 0; i < w.size; i++) {
      uint8_t q = random_word();
      w.ptr[i] = (q & 127) == 127 ? q - 1 : q;
    }
    for (size_t i = 0; i < ws.size; i++) ws.ptr[i] = 112 + random_word() % 9;
    for (uint32_t row = 0; row < p.n; row++)
      memcpy(row_scales.ptr + (size_t)row * (p.k / 32), ws.ptr + (size_t)(row / 32) * (p.k / 32), p.k / 32);
    for (size_t first = 0; first < (size_t)p.m * p.k; first += 32) {
      float values[32];
      for (int i = 0; i < 32; i++) values[i] = ((int)(random_word() % 257) - 128) / 128.0f;
      RefFp8Block q;
      if (!ref_fp8_pack(values, &q)) die("invalid reference FP8 input");
      ref_fp8_unpack(&q, (float *)x.ptr + first);
    }
    float *reference = xmalloc(out.size);
    for (int rounding = 0; rounding < 2; rounding++) {
      p.round = rounding;
      run(&p, 0, 1); memcpy(reference, out.ptr, out.size);
      run(&p, 1, 1);
      size_t different = 0, packing = 0, nonfinite = 0;
      double signal = 0, error = 0, maximum = 0;
      for (size_t i = 0; i < (size_t)p.m * p.n; i++) {
        float got = ((float *)out.ptr)[i], want = reference[i];
        different += memcmp(&got, &want, 4) != 0;
        nonfinite += !isfinite(got) || !isfinite(want);
        double d = (double)got - want;
        signal += (double)want * want, error += d * d;
        maximum = fmax(maximum, fabs(d));
      }
      for (size_t i = 0; i < (size_t)p.m * p.k; i++) {
        float value;
        if (mode == 2) {
          _Float16 h;
          memcpy(&h, xb.ptr + i * 2, 2);
          value = (float)h * 0x1p-7f;
        } else value = ref_from_e4m3(codes.ptr[i]) * ref_bits_float((uint32_t)scales.ptr[i / 32] << 23);
        packing += memcmp(&value, (float *)x.ptr + i, 4) != 0;
      }
      run(&p, 0, p.m > 128 ? 4 : 32), run(&p, 1, p.m > 128 ? 4 : 32);
      double times[2][5];
      for (int trial = 0; trial < 5; trial++) {
        int first = trial & 1;
        times[first][trial] = run(&p, first, p.m > 128 ? 2 : 8);
        times[1 - first][trial] = run(&p, 1 - first, p.m > 128 ? 2 : 8);
      }
      for (int variant = 0; variant < 2; variant++)
        for (int i = 1; i < 5; i++)
          for (int j = i; j && times[variant][j] < times[variant][j - 1]; j--) {
            double t = times[variant][j]; times[variant][j] = times[variant][j - 1]; times[variant][j - 1] = t;
          }
      char snr[48] = "null";
      if (!nonfinite && error > 0 && signal > 0) snprintf(snr, sizeof snr, "%.4f", 10 * log10(signal / error));
      printf("{\"fp8_tensor\":{\"variant\":\"%s\",\"m\":%u,\"n\":%u,\"k\":%u,\"bf16_output\":%d,\"packing_differences\":%zu,"
             "\"output_differences\":%zu,\"nonfinite\":%zu,\"extra_scale_bytes\":%zu,\"maximum_finite_error\":%.9g,\"snr_db\":%s,\"control_ms\":%.6f,\"candidate_ms\":%.6f}}\n",
             mode_name, p.m, p.n, p.k, rounding, packing, different, nonfinite, row_scales.size - ws.size, isfinite(maximum) ? maximum : 0, snr,
             times[0][2], times[1][2]);
      if (packing || nonfinite || (rounding && different)) failures++;
    }
    free(reference);
    gpu_free(&x), gpu_free(&xb), gpu_free(&codes), gpu_free(&scales), gpu_free(&w), gpu_free(&ws), gpu_free(&row_scales), gpu_free(&out);
  }
  return failures != 0;
}
