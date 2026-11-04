// Isolate the cost of reading packed FP8 activations in an MXFP4 decode kernel.
#include "myds4.h"
#include "../reference/activation_format.h"
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

enum { ROWS = 6, K = 5120, H = 2304, EXP = 6 };
typedef struct { uint64_t p[3]; float weight, pad; } Entry;
static Buf weights[EXP], entries, input, packed, scales, output;
static int ordinary_kernel, packed_kernel;
static uint32_t random_state = 7531;
static uint32_t random_bits(void) {
  random_state ^= random_state << 13;
  random_state ^= random_state >> 17;
  random_state ^= random_state << 5;
  return random_state;
}
static double run(int rows, int compact, int repeats) {
  double before = gpu_busy;
  gpu_begin();
  for (int n = 0; n < repeats; n++)
    for (int r = 0; r < rows; r++) {
      gpu_kern(compact ? packed_kernel : ordinary_kernel);
      gpu_buf(0, &entries, 0);
      if (compact) {
        gpu_buf(1, &packed, (size_t)r * K), gpu_buf(2, &scales, (size_t)r * K / 32);
        gpu_buf(3, &output, (size_t)r * EXP * H * 4);
      } else {
        gpu_buf(1, &input, (size_t)r * K * 4), gpu_buf(2, &output, (size_t)r * EXP * H * 4);
      }
      gpu_go(H / 32, EXP, 1, 256, 1, 1);
    }
  gpu_commit(), gpu_sync();
  return (gpu_busy - before) * 1000 / repeats;
}
static double median(double x[3]) {
  for (int i = 1; i < 3; i++) for (int j = i; j > 0 && x[j] < x[j - 1]; j--) {
    double value = x[j]; x[j] = x[j - 1]; x[j - 1] = value;
  }
  return x[1];
}

int main(void) {
  alarm(90);
  lock_model();
  static const char source[] = {
#embed "../../src/kernels/kernels.metal"
      , '\n',
#embed "packed_activation.metal"
  };
  gpu_init(source, sizeof source);
  ordinary_kernel = gpu_kernel("moe_up"), packed_kernel = gpu_kernel("moe_up_packed");
  entries = gpu_alloc(EXP * sizeof(Entry));
  for (int e = 0; e < EXP; e++) {
    weights[e] = gpu_alloc(EXPERT_W + EXPERT_S);
    for (size_t i = 0; i < EXPERT_W / 4; i++) ((uint32_t *)weights[e].ptr)[i] = random_bits();
    for (size_t i = 0; i < EXPERT_S; i++) weights[e].ptr[EXPERT_W + i] = 118 + random_bits() % 4;
    ((Entry *)entries.ptr)[e] = (Entry){{weights[e].gpu, weights[e].gpu + EXPERT_W, 0}, 0.1f + .02f * e, 0};
  }
  input = gpu_alloc(ROWS * K * 4), packed = gpu_alloc(ROWS * K), scales = gpu_alloc(ROWS * K / 32);
  output = gpu_alloc(ROWS * EXP * H * 4);
  for (int r = 0; r < ROWS; r++)
    for (int block = 0; block < K / 32; block++) {
      float values[32], decoded[32];
      for (int i = 0; i < 32; i++) values[i] = ((int)(random_bits() % 257) - 128) * (r % 3 == 0 ? 0x1p-27f : r % 3 == 1 ? 0x1p-9f : 0x1p-5f);
      RefFp8Block codes;
      if (!ref_fp8_pack(values, &codes)) die("invalid activation fixture");
      ref_fp8_unpack(&codes, decoded);
      memcpy(input.ptr + ((size_t)r * K + block * 32) * 4, decoded, sizeof decoded);
      memcpy(packed.ptr + (size_t)r * K + block * 32, codes.values, 32);
      scales.ptr[r * (K / 32) + block] = codes.scale;
    }
  void *expected = xmalloc(output.size);
  int failed = 0;
  run(ROWS, 0, 48);
  for (int rows = 1; rows <= ROWS; rows = rows == 1 ? 3 : rows + 3) {
    run(rows, 0, 1);
    memcpy(expected, output.ptr, (size_t)rows * EXP * H * 4);
    run(rows, 1, 1);
    int differences = 0;
    for (int i = 0; i < rows * EXP * H; i++)
      differences += memcmp((const float *)expected + i, (const float *)output.ptr + i, sizeof(float)) != 0;
    double ordinary[3], compact[3];
    for (int sample = 0; sample < 3; sample++) {
      if (sample & 1) {
        compact[sample] = run(rows, 1, 16);
        ordinary[sample] = run(rows, 0, 16);
      } else {
        ordinary[sample] = run(rows, 0, 16);
        compact[sample] = run(rows, 1, 16);
      }
    }
    printf("{\"packed_activation\":{\"rows\":%d,\"different_values\":%d,\"float_ms\":%.6f,\"packed_ms\":%.6f,"
           "\"float_input_bytes\":%d,\"packed_input_bytes\":%d}}\n",
           rows, differences, median(ordinary), median(compact), rows * K * 4, rows * K * 33 / 32);
    failed |= differences != 0;
  }
  free(expected);
  return failed;
}
