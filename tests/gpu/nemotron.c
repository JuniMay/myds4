// Mamba2 recurrence and saved prefixes against an independent CPU calculation.
#include "../../src/engine/myds4.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "../reference/activations.h"

static float bf(float x) {
  uint32_t bits;
  memcpy(&bits, &x, 4);
  bits = (bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000;
  memcpy(&x, &bits, 4);
  return x;
}
int main(void) {
  enum { ROWS = 3, HEADS = 2, DIM = 64, STATE = 128, GROUPS = 1, WIDTH = 128, CHANNELS = 384, PROJECTED = 514 };
  gpu_start();
  Buf input = gpu_alloc(256 * 4), codes = gpu_alloc(256 * 4);
  for (int i = 0; i < 256; i++)
    ((float *)input.ptr)[i] = bf(sinf(i * .13f) * 2);
  struct {
    uint32_t count, bits;
    float scale;
    uint32_t pad;
  } quant = {256, 9, .01337f, 0};
  gpu_begin();
  gpu_kern(gpu_kernel("quantize_input"));
  gpu_buf(0, &input, 0);
  gpu_buf(1, &codes, 0);
  gpu_bytes(2, &quant, sizeof quant);
  gpu_go(1, 1, 1, 256, 1, 1);
  gpu_commit();
  gpu_sync();
  int different = 0;
  for (int i = 0; i < 256; i++)
    different += ((float *)codes.ptr)[i] != ref_from_e4m3(ref_e4m3(((float *)input.ptr)[i] / quant.scale));
  printf("FP8 activation codes before the external global scale: %d differences\n", different);
  if (different)
    return 1;
  quant.bits = 4;
  gpu_begin();
  gpu_kern(gpu_kernel("quantize_input"));
  gpu_buf(0, &input, 0);
  gpu_buf(1, &codes, 0);
  gpu_bytes(2, &quant, sizeof quant);
  gpu_go(1, 1, 1, 256, 1, 1);
  gpu_commit();
  gpu_sync();
  static const float levels[] = {0, .5f, 1, 1.5f, 2, 3, 4, 6};
  for (int group = 0; group < 16; group++) {
    float maximum = 0;
    for (int i = 0; i < 16; i++)
      maximum = fmaxf(maximum, fabsf(((float *)input.ptr)[group * 16 + i] / quant.scale));
    float scale = ref_from_e4m3(ref_e4m3(maximum / 6));
    for (int i = 0; i < 16; i++) {
      float value = ((float *)input.ptr)[group * 16 + i] / quant.scale / scale;
      int code = 0;
      for (int k = 1; k < 8; k++) {
        float distance = fabsf(fabsf(value) - levels[k]), old = fabsf(fabsf(value) - levels[code]);
        if (distance < old || (distance == old && !(k & 1)))
          code = k;
      }
      float expected = copysignf(levels[code], value) * scale;
      different += ((float *)codes.ptr)[group * 16 + i] != expected;
    }
  }
  printf("NVFP4 activation groups before the external global scale: %d differences\n", different);
  if (different)
    return 1;
  Buf x = gpu_alloc(ROWS * CHANNELS * 4), projected = gpu_alloc(ROWS * PROJECTED * 4),
      state = gpu_alloc(HEADS * DIM * STATE * 4);
  Buf saved = gpu_alloc(ROWS * state.size), a = gpu_alloc(HEADS * 4), d = gpu_alloc(HEADS * 4),
      bias = gpu_alloc(HEADS * 4);
  Buf output = gpu_alloc(ROWS * WIDTH * 4), sequences = gpu_alloc(ROWS * 72);
  memset(state.ptr, 0, state.size);
  memset(projected.ptr, 0, projected.size);
  memset(sequences.ptr, 0, sequences.size);
  for (int i = 0; i < ROWS * CHANNELS; i++)
    ((float *)x.ptr)[i] = bf(sinf(i * .017f) * .2f);
  for (int h = 0; h < HEADS; h++)
    ((float *)a.ptr)[h] = .5f + h, ((float *)d.ptr)[h] = .75f, ((float *)bias.ptr)[h] = -1;
  for (int r = 0; r < ROWS; r++) {
    uint64_t *s = (uint64_t *)(sequences.ptr + r * 72);
    s[0] = state.gpu;
    s[1] = saved.gpu + r * state.size;
    for (int h = 0; h < HEADS; h++)
      ((float *)projected.ptr)[r * PROJECTED + WIDTH + CHANNELS + h] = bf(.1f * (r + h));
  }
  struct {
    uint32_t rows, heads, groups, dim, state, layer, channels, projected;
    float minimum;
  } p = {ROWS, HEADS, GROUPS, DIM, STATE, 0, CHANNELS, PROJECTED, .001f};
  gpu_begin();
  gpu_kern(gpu_kernel("nm_scan"));
  gpu_buf(0, &x, 0);
  gpu_buf(1, &projected, 0);
  gpu_buf(2, &a, 0);
  gpu_buf(3, &d, 0);
  gpu_buf(4, &bias, 0);
  gpu_buf(5, &sequences, 0);
  gpu_buf(6, &output, 0);
  gpu_bytes(7, &p, sizeof p);
  gpu_go(HEADS, DIM / 4, 1, 128, 1, 1);
  gpu_commit();
  gpu_sync();
  float *expected = calloc(HEADS * DIM * STATE, 4);
  double max_state = 0, max_output = 0;
  for (int r = 0; r < ROWS; r++)
    for (int h = 0; h < HEADS; h++) {
      float dt = log1pf(expf(((float *)projected.ptr)[r * PROJECTED + WIDTH + CHANNELS + h] + ((float *)bias.ptr)[h]));
      float decay = expf(-dt * ((float *)a.ptr)[h]);
      for (int dim = 0; dim < DIM; dim++) {
        float value = ((float *)x.ptr)[r * CHANNELS + h * DIM + dim], sum = 0;
        for (int k = 0; k < STATE; k++) {
          size_t at = ((size_t)h * DIM + dim) * STATE + k;
          float b = ((float *)x.ptr)[r * CHANNELS + WIDTH + k], c = ((float *)x.ptr)[r * CHANNELS + WIDTH + STATE + k];
          expected[at] = expected[at] * decay + dt * b * value;
          sum += expected[at] * c;
          max_state = fmax(max_state, fabs(expected[at] - ((float *)saved.ptr)[r * HEADS * DIM * STATE + at]));
        }
        float want = bf(sum + ((float *)d.ptr)[h] * value), got = ((float *)output.ptr)[r * WIDTH + h * DIM + dim];
        max_output = fmax(max_output, fabs(want - got));
      }
    }
  printf("Mamba2 saved-prefix state max error %.9g; output max error %.9g\n", max_state, max_output);
  return max_state > 1e-6 || max_output > .001;
}
