// Compile the shared kernel library, or encode one image supplied on stdin.
// Feature bytes go only to an explicitly supplied descriptor.
#include "../../src/model/qwen/vision.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static uint16_t bf16(float value) {
  uint32_t bits; memcpy(&bits, &value, 4);
  return (uint16_t)((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}
static float fp32(uint16_t value) {
  uint32_t bits = (uint32_t)value << 16; float out; memcpy(&out, &bits, 4); return out;
}

static int kernels(void) {
  enum { N = 8193, ROWS = 129, PAD = 256, HEADS = 16, DIM = 72, STRIDE = 96 };
  Buf activation = gpu_alloc(N * 4);
  float *a = (float *)activation.ptr;
  for (int mode = 0; mode < 2; mode++) {
    for (int i = 0; i < N; i++) a[i] = (float)i / (N - 1) * 200 - 100;
    uint32_t p[2] = {N, (uint32_t)mode};
    gpu_begin(); gpu_kern(gpu_kernel("qv_gelu")); gpu_buf(0, &activation, 0); gpu_bytes(1, p, sizeof p);
    gpu_go((N + 255) / 256, 1, 1, 256, 1, 1); gpu_commit(); gpu_sync();
    for (int i = 0; i < N; i++) {
      float x = (float)i / (N - 1) * 200 - 100;
      float g = mode ? erff(x * 0.7071067811865475f) : tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x));
      float expected = fp32(bf16(0.5f * x * (1 + g)));
      if (!isfinite(a[i]) || fabsf(a[i] - expected) > 1e-5f + fabsf(expected) / 128) die("GELU mode %d differs at %g", mode, x);
    }
  }
  gpu_free(&activation);
  Buf heads = gpu_alloc((size_t)3 * HEADS * PAD * STRIDE * 2), output = gpu_alloc(ROWS * HEADS * DIM * 4);
  memset(heads.ptr, 0, heads.size);
  uint16_t *h = (uint16_t *)heads.ptr;
  for (int part = 0; part < 3; part++) for (int head = 0; head < HEADS; head++)
    for (int row = 0; row < ROWS; row++) for (int d = 0; d < DIM; d++)
      h[((part * HEADS + head) * PAD + row) * STRIDE + d] = bf16(sinf((part * 31 + head * 17 + row * 11 + d) * 0.17f));
  uint32_t shape[2] = {ROWS, PAD};
  gpu_begin(); gpu_kern(gpu_kernel("qv_attention")); gpu_buf(0, &heads, 0); gpu_buf(1, &output, 0); gpu_bytes(2, shape, sizeof shape);
  gpu_go((ROWS + 63) / 64, HEADS, 1, 128, 1, 1); gpu_commit(); gpu_sync();
  double signal = 0, noise = 0, worst = 0;
  for (int q = 0; q < ROWS; q++) for (int head = 0; head < HEADS; head++) {
    double score[ROWS], maximum = -INFINITY, sum = 0;
    for (int k = 0; k < ROWS; k++) {
      double dot = 0;
      for (int d = 0; d < DIM; d++) dot += fp32(h[(head * PAD + q) * STRIDE + d]) * fp32(h[((HEADS + head) * PAD + k) * STRIDE + d]);
      score[k] = dot / sqrt(72.0); maximum = fmax(maximum, score[k]);
    }
    for (int k = 0; k < ROWS; k++) score[k] = exp(score[k] - maximum), sum += score[k];
    for (int d = 0; d < DIM; d++) {
      double want = 0;
      for (int k = 0; k < ROWS; k++) want += fp32(bf16((float)score[k])) * fp32(h[((2 * HEADS + head) * PAD + k) * STRIDE + d]);
      want = fp32(bf16((float)(want / sum)));
      float got = ((float *)output.ptr)[(q * HEADS + head) * DIM + d];
      if (!isfinite(got)) die("nonfinite attention fixture");
      double error = got - want; signal += want * want; noise += error * error; worst = fmax(worst, fabs(error));
    }
  }
  double snr = noise ? 10 * log10(signal / noise) : INFINITY;
  printf("vision kernels: GELU passed; padded attention max error %.6g, SNR %.2f dB\n", worst, snr);
  gpu_free(&heads); gpu_free(&output);
  return worst > 0.03 || snr < 45;
}

int main(int argc, char **argv) {
  gpu_start();
  if (argc == 1) return kernels();
  Str input = {0};
  char buffer[4096]; ssize_t n;
  while ((n = read(STDIN_FILENO, buffer, sizeof buffer)) > 0) {
    if (input.n + (size_t)n > 8u << 20) die("image fixture exceeds 8 MiB");
    str_put(&input, buffer, (size_t)n);
  }
  Image image = {0}; const char *error;
  if (!qwen_prepare_image(argv[1], &image, input.p, input.n, &error)) die("%s", error);
  char path[4096]; snprintf(path, sizeof path, "%s/config.json", argv[1]);
  char *text = read_file(path, NULL); Json *root = json_parse(text);
  TensorIndex *index = st_open(argv[1]);
  QwenVision *vision = qwen_vision_open(index, json_get(root, "vision_config"));
  json_free(root); free(text);
  qwen_vision_encode(vision, &image);
  if (image.ntokens != image.th * image.tw || image.embedding_width != 5120 || !image.rows) return 1;
  for (size_t i = 0; i < (size_t)image.ntokens * image.embedding_width; i++)
    if ((image.rows[i] & 0x7f80u) == 0x7f80u) die("nonfinite vision feature");
  if (argc == 3) {
    int fd = atoi(argv[2]); size_t bytes = (size_t)image.ntokens * image.embedding_width * 2;
    const char *p = (const char *)image.rows;
    while (bytes) { ssize_t done = write(fd, p, bytes); if (done <= 0) die("feature pipe closed"); p += done; bytes -= (size_t)done; }
  }
  printf("vision: %d x %d patches, %d feature rows, %d values per row\n", image.vh, image.vw, image.ntokens, image.embedding_width);
  image_free(&image); qwen_vision_close(vision); st_close(index); free(input.p);
  return 0;
}
