// Compare hierarchical top-k with the global-reduction reference, including ties.
#include "../../src/engine/myds4.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#define METAL(...) #__VA_ARGS__
static const char reference[] = METAL(
kernel void route_reference(device const float *logits [[buffer(0)]], device uint *ids [[buffer(1)]],
                             device float *weights [[buffer(2)]], constant FlashRoute &p [[buffer(3)]],
                             uint row [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
                             uint threads [[threads_per_threadgroup]]) {
  threadgroup float partial[32], selected[16];
  threadgroup uint indexes[32], winner;
  float key = t < p.experts ? logits[row * p.experts + t] : -INFINITY;
  float maximum = tg_max(key, partial, t, threads);
  float base = t < p.experts ? exp(key - maximum) : 0;
  base /= tg_sum(base, partial, t, threads);
  key = t < p.experts ? base : -INFINITY;
  for (uint k = 0; k < p.chosen; k++) {
    maximum = tg_max(key, partial, t, threads);
    uint index = simd_min(key == maximum ? t : UINT_MAX);
    if (t % 32 == 0) indexes[t / 32] = index;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!t) {
      uint best = UINT_MAX;
      for (uint s = 0; s < threads / 32; s++) best = min(best, indexes[s]);
      winner = best; ids[row * p.chosen + k] = best;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == winner) { selected[k] = base; key = -INFINITY; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (!t) {
    float total = 0;
    for (uint k = 0; k < p.chosen; k++) total += selected[k];
    for (uint k = 0; k < p.chosen; k++)
      weights[row * p.chosen + k] = bf(selected[k] / (p.normalized ? total : 1.0f));
  }
});
#undef METAL

static double run(int kernel, int experts, int chosen, int normalized, const Buf *x, const Buf *ids,
                  const Buf *weights, int rows, int repeat) {
  uint32_t p[4] = {experts, chosen, normalized, 0};
  int threads = (experts + 31) / 32 * 32;
  double before = gpu_busy;
  gpu_begin(); gpu_kern(kernel); gpu_buf(0, x, 0); gpu_buf(1, ids, 0); gpu_buf(2, weights, 0);
  gpu_bytes(3, p, sizeof p);
  for (int i = 0; i < repeat; i++) gpu_go(rows, 1, 1, threads, 1, 1);
  gpu_commit(); gpu_sync();
  return (gpu_busy - before) * 1e6 / repeat;
}
int main(void) {
  memory_configure(1, 10);
  size_t source_bytes;
  const char *source = gpu_source(&source_bytes);
  char *code = xmalloc(source_bytes + sizeof reference + 1);
  memcpy(code, source, source_bytes); code[source_bytes] = '\n';
  memcpy(code + source_bytes + 1, reference, sizeof reference);
  gpu_init(code, source_bytes + sizeof reference); free(code);
  int control = gpu_kernel("route_reference"), candidate = gpu_kernel("flash_route");
  enum { ROWS = 32 };
  Buf x = gpu_alloc(ROWS * 512 * 4), a = gpu_alloc(ROWS * 16 * 4), b = gpu_alloc(ROWS * 16 * 4);
  Buf wa = gpu_alloc(ROWS * 16 * 4), wb = gpu_alloc(ROWS * 16 * 4);
  int widths[] = {31, 32, 33, 64, 128, 384, 512}, choices[] = {1, 2, 6, 10, 16}, checks = 0;
  for (int kind = 0; kind < 5; kind++) for (int e = 0; e < 7; e++) {
    int experts = widths[e];
    for (int r = 0; r < ROWS; r++) for (int i = 0; i < experts; i++) {
      float v = kind == 0 ? 0 : kind == 1 ? i / 4 : kind == 2 ? sinf((i + r * 97) * .183f) * 8 :
                kind == 3 ? (i == experts - 1 ? 100 : -10000) : (i < 22 ? 10 : -3);
      ((float *)x.ptr)[r * experts + i] = v;
    }
    for (int c = 0; c < 5; c++) for (int norm = 0; norm < 2; norm++) {
      int k = choices[c]; size_t bytes = ROWS * k * 4;
      run(control, experts, k, norm, &x, &a, &wa, ROWS, 1);
      run(candidate, experts, k, norm, &x, &b, &wb, ROWS, 1);
      if (memcmp(a.ptr, b.ptr, bytes) || memcmp(wa.ptr, wb.ptr, bytes))
        die("routing differs: kind %d experts %d chosen %d normalized %d", kind, experts, k, norm);
      checks++;
    }
  }
  printf("routing: %d cases match IDs and weights bit for bit\n", checks);
  for (int rows = 1; rows <= ROWS; rows *= 8) {
    double before = run(control, 512, 10, 1, &x, &a, &wa, rows, 64);
    double middle = run(candidate, 512, 10, 1, &x, &b, &wb, rows, 64);
    double after = run(control, 512, 10, 1, &x, &a, &wa, rows, 64);
    printf("routing rows=%d: %.3f / %.3f / %.3f us\n", rows, before, middle, after);
  }
  return 0;
}
