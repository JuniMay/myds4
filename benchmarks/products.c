// Check matrix output rounding and time ordered vector reductions.
#include "../src/model/linear.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

static double run(int kernel, int split, int threads, int rows, int cols,
                  const Buf *w, const Buf *s, const Buf *x, const Buf *y, int repetitions) {
  struct { uint32_t rows, cols, n; float scale; } p = {rows, cols, 1, .03125f};
  double before = gpu_busy;
  gpu_begin(); gpu_kern(kernel); gpu_buf(0, w, 0); gpu_buf(1, s, 0);
  gpu_buf(2, x, 0); gpu_buf(3, y, 0); gpu_bytes(4, &p, sizeof p);
  for (int i = 0; i < repetitions; i++)
    gpu_go(split ? rows : (rows + 7) / 8, 1, 1, split ? threads : 256, 1, 1);
  gpu_commit(); gpu_sync();
  return (gpu_busy - before) * 1e6 / repetitions;
}
int main(void) {
  memory_configure(1, 10); gpu_start(); linear_init();
  const char *formats[] = {"nvfp4", "fp8", "bf16", "mxfp8"};
  static char names[4][2][32];
  int widths[] = {1, 4, 48, 320, 512}, inputs[] = {320, 2560, 10240, 32768};
  for (int f = 0; f < 4; f++) {
    snprintf(names[f][0], sizeof names[f][0], "mv1_%s", formats[f]);
    snprintf(names[f][1], sizeof names[f][1], "mv_split_%s", formats[f]);
    int ordinary = gpu_kernel(names[f][0]), split = gpu_kernel(names[f][1]);
    for (size_t i = 0; i < sizeof inputs / sizeof *inputs; i++) {
      int cols = inputs[i]; size_t count = (size_t)512 * cols;
      size_t bytes = f == 0 ? count / 2 : f == 2 ? count * 2 : count;
      Buf w = gpu_alloc(bytes), s = gpu_alloc(count / 16), x = gpu_alloc(32 * cols * 4), y = gpu_alloc(32 * 512 * 4);
      if (f == 2) for (size_t k = 0; k < count; k++) {
        float value = sinf(k * .029f) * 3; uint32_t bits; memcpy(&bits, &value, 4);
        ((uint16_t *)w.ptr)[k] = (bits + 0x7fff + ((bits >> 16) & 1)) >> 16;
      } else for (size_t k = 0; k < bytes; k++) w.ptr[k] = f ? (k * 73 + k / 19) % 120 : k * 71 + k / 93;
      for (size_t k = 0; k < s.size; k++) s.ptr[k] = f == 3 ? 112 + k % 16 : 32 + k % 40;
      for (int k = 0; k < 32 * cols; k++) ((float *)x.ptr)[k] = sinf(k * .031f) * 3;
      for (size_t j = 0; j < sizeof widths / sizeof *widths; j++) {
        int rows = widths[j];
        run(ordinary, 0, 256, rows, cols, &w, &s, &x, &y, 16);
        uint32_t expected[512]; memcpy(expected, y.ptr, rows * 4);
        for (int threads = 64; threads <= 256; threads *= 2) {
          double a = run(ordinary, 0, 256, rows, cols, &w, &s, &x, &y, 64);
          double b = run(split, 1, threads, rows, cols, &w, &s, &x, &y, 64);
          int equal = !memcmp(expected, y.ptr, rows * 4);
          double c = run(ordinary, 0, 256, rows, cols, &w, &s, &x, &y, 64);
          printf("{\"format\":\"%s\",\"rows\":%d,\"cols\":%d,\"threads\":%d,\"before_us\":%.3f,"
                 "\"candidate_us\":%.3f,\"after_us\":%.3f,\"equal\":%s}\n",
                 formats[f], rows, cols, threads, a, b, c, equal ? "true" : "false");
          fflush(stdout);
          if (!equal) return 1;
        }
      }
      if (cols == 2560) {
        Buf packed = gpu_alloc(128 * cols * 2), other = gpu_alloc(y.size);
        int counts[] = {1, 3, 8, 9, 32}, outputs[] = {4, 512};
        int round = gpu_kernel("round_bf16");
        for (int o = 0; o < 2; o++) for (int c = 0; c < 5; c++) {
          int n = counts[c], width = outputs[o];
          Linear m = {.w = w, .s = s, .scale = .03125f, .input_scale = 1,
                      .rows = width, .cols = cols, .kind = "n8bm"[f]};
          gpu_begin();
          linear_product(&m, &x, 0, &y, 0, n, n > 8, &packed, 0, 0);
          uint32_t length = n * width;
          gpu_kern(round); gpu_buf(0, &y, 0); gpu_bytes(1, &length, 4);
          gpu_go((length + 255) / 256, 1, 1, 256, 1, 1);
          gpu_commit(); gpu_sync();
          gpu_begin();
          linear_product(&m, &x, 0, &other, 0, n, n > 8, &packed, 0, 1);
          gpu_commit(); gpu_sync();
          if (memcmp(y.ptr, other.ptr, length * 4)) die("rounded product differs: format %d rows %d width %d", f, n, width);
        }
        printf("rounded %s: 10 vector and matrix cases match bit for bit\n", formats[f]);
        gpu_free(&packed); gpu_free(&other);
      }
      gpu_free(&w); gpu_free(&s); gpu_free(&x); gpu_free(&y);
    }
  }
  return 0;
}
