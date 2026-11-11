// The prompt attention kernel against one that stages a slice of a chunk's
// keys at a time, for its scores and again for its values: every output must
// be the same bits.
#include "../../src/engine/myds4.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#define METAL(...) #__VA_ARGS__
static const char reference[] = METAL(
kernel void attend_m_reference(device const bfloat *qb [[buffer(0)]], device const bfloat *ring [[buffer(1)]],
                               device const bfloat *comp [[buffer(2)]], device const int *sel [[buffer(3)]],
                               device const float *sink [[buffer(4)]], device float *o [[buffer(5)]],
                               constant Att &at [[buffer(6)]], uint2 gr [[threadgroup_position_in_grid]],
                               uint t [[thread_index_in_threadgroup]]) {
  threadgroup bfloat kv[32 * 128], ps[32 * 32];
  threadgroup float ss[32 * 32], mx[32], sum[32], rescale[32];
  threadgroup uint rows[640];
  uint h0 = gr.x * 32, pos = at.win.pos + gr.y;
  o += gr.y * 32768, sel += gr.y * 513 + 1;
  uint lo = max(pos >= 127 ? pos - 127 : 0u, at.first);
  uint nwin = pos + 1 - lo, ncomp = at.comp_rows ? uint(sel[-1]) : 0, n = nwin + ncomp;
  for (uint r = t; r < n; r += 256) rows[r] = r < nwin ? win_row(lo + r, at.win) : 0x80000000u | uint(sel[r - nwin]);
  if (t < 32) mx[t] = -1e30f, sum[t] = 0;
  constexpr auto sdesc = matmul2d_descriptor(32, 32, 128, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  constexpr auto odesc = matmul2d_descriptor(32, 128, 32, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<sdesc, execution_simdgroups<8>> sop;
  matmul2d<odesc, execution_simdgroups<8>> oop;
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tq((device bfloat *)qb + gr.y * 32768 + h0 * 512, dextents<int32_t, 2>(512, 32));
  tensor<threadgroup bfloat, extents<int32_t, 128, 32>, tensor_inline> tk(kv, extents<int32_t, 128, 32>());
  tensor<threadgroup bfloat, extents<int32_t, 32, 32>, tensor_inline> tp(ps, extents<int32_t, 32, 32>());
  auto acc0 = oop.get_destination_cooperative_tensor<decltype(tp), decltype(tk), float>();
  auto acc1 = oop.get_destination_cooperative_tensor<decltype(tp), decltype(tk), float>();
  auto acc2 = oop.get_destination_cooperative_tensor<decltype(tp), decltype(tk), float>();
  auto acc3 = oop.get_destination_cooperative_tensor<decltype(tp), decltype(tk), float>();
  for (uint16_t i = 0; i < acc0.get_capacity(); i++)
    if (acc0.is_valid_element(i)) acc0[i] = acc1[i] = acc2[i] = acc3[i] = 0;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  Keys kc = keys_load(ring, comp, rows, 0, n, 0, t);
  for (uint r0 = 0; r0 < n; r0 += 32) {
    auto sc = sop.get_destination_cooperative_tensor<decltype(tq.slice(0, 0)), decltype(tk), float>();
    for (uint16_t i = 0; i < sc.get_capacity(); i++)
      if (sc.is_valid_element(i)) sc[i] = 0;
    for (uint ds = 0; ds < 4; ds++) {
      keys_store(kv, kc, t);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      kc = keys_load(ring, comp, rows, r0, n, (ds + 1) % 4, t);
      auto qd = tq.slice(ds * 128, 0);
      sop.run(qd, tk, sc);
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint16_t i = 0; i < sc.get_capacity(); i++)
      if (sc.is_valid_element(i)) {
        auto a = sc.get_multidimensional_index(i);
        ss[a[1] * 32 + a[0]] = r0 + a[0] < n ? sc[i] * 0.044194173824159216f : -INFINITY;
      }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t < 32) {
      float m = mx[t];
      for (uint j = 0; j < 32; j++) m = max(m, ss[t * 32 + j]);
      float f = exp(mx[t] - m), s2 = sum[t] * f;
      for (uint j = 0; j < 32; j++) {
        float e = exp(ss[t * 32 + j] - m);
        s2 += e;
        ps[t * 32 + j] = bfloat(e);
      }
      mx[t] = m, sum[t] = s2, rescale[t] = f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint16_t i = 0; i < acc0.get_capacity(); i++)
      if (acc0.is_valid_element(i)) {
        float f = rescale[acc0.get_multidimensional_index(i)[1]];
        acc0[i] *= f, acc1[i] *= f, acc2[i] *= f, acc3[i] *= f;
      }
    for (uint ds = 0; ds < 4; ds++) {
      keys_store(kv, kc, t);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (ds < 3) kc = keys_load(ring, comp, rows, r0, n, ds + 1, t);
      else if (r0 + 32 < n) kc = keys_load(ring, comp, rows, r0 + 32, n, 0, t);
      if (ds == 0) oop.run(tp, tk, acc0);
      else if (ds == 1) oop.run(tp, tk, acc1);
      else if (ds == 2) oop.run(tp, tk, acc2);
      else oop.run(tp, tk, acc3);
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
  for (uint16_t i = 0; i < acc0.get_capacity(); i++)
    if (acc0.is_valid_element(i)) {
      auto a = acc0.get_multidimensional_index(i);
      uint h = a[1], d = a[0];
      float den = sum[h] + exp(sink[h0 + h] - mx[h]);
      device float *out = o + (h0 + h) * 512 + d;
      out[0] = bf(acc0[i] / den), out[128] = bf(acc1[i] / den), out[256] = bf(acc2[i] / den), out[384] = bf(acc3[i] / den);
    }
});
#undef METAL

typedef struct { uint32_t pos, base, mod, first, comp_rows; } Att;
enum { ROWS = 512, COMP = 8192 };
static Buf QB, WIN, KEYS, SEL, SINK, OUT[2];
static uint32_t seed = 2463534242u;
static uint32_t next(void) { return seed = seed * 1664525u + 1013904223u; }
static uint16_t bf16(float v) {  // values of few bits are exact
  uint32_t u;
  memcpy(&u, &v, 4);
  return (uint16_t)(u >> 16);
}

static double run(int kernel, Att at, int rows, const Buf *out, int repeat) {
  double before = gpu_busy;
  gpu_begin();
  gpu_kern(kernel);
  gpu_buf(0, &QB, 0), gpu_buf(1, &WIN, 0), gpu_buf(2, &KEYS, 0), gpu_buf(3, &SEL, 0), gpu_buf(4, &SINK, 0);
  gpu_buf(5, out, 0), gpu_bytes(6, &at, sizeof at);
  for (int i = 0; i < repeat; i++) gpu_go(2, rows, 1, 256, 1, 1);
  gpu_commit();
  gpu_sync();
  return (gpu_busy - before) * 1e6 / repeat;
}

// Each row's selected compressed rows: `count(r)` of them, ascending.
static void select_rows(int rows, int (*count)(int)) {
  int *sel = (int *)SEL.ptr;
  for (int r = 0; r < rows; r++) {
    int n = count(r), at = (int)(next() % 7);
    sel[r * 513] = n;
    for (int i = 0; i < n; i++) sel[r * 513 + 1 + i] = at, at += 1 + (int)(next() % 15);
  }
}
static int all(int r) { return 512; }
static int none(int r) { return 0; }
static int varied(int r) { return (r * 37) % 513; }
static int few(int r) { return 100 + r % 60; }

int main(void) {
  memory_configure(1, 10);
  size_t source_bytes;
  const char *source = gpu_source(&source_bytes);
  char *code = xmalloc(source_bytes + sizeof reference + 1);
  memcpy(code, source, source_bytes), code[source_bytes] = '\n';
  memcpy(code + source_bytes + 1, reference, sizeof reference);
  gpu_init(code, source_bytes + sizeof reference), free(code);
  int before = gpu_kernel("attend_m_reference"), now_kernel = gpu_kernel("attend_m");
  QB = gpu_alloc((size_t)ROWS * 32768 * 2), WIN = gpu_alloc((size_t)(ROWS + 128) * 512 * 2);
  KEYS = gpu_alloc((size_t)COMP * 512 * 2), SEL = gpu_alloc((size_t)ROWS * 513 * 4), SINK = gpu_alloc(64 * 4);
  for (int i = 0; i < 2; i++) OUT[i] = gpu_alloc((size_t)ROWS * 32768 * 4);
  for (size_t i = 0; i < (size_t)ROWS * 32768; i++) ((uint16_t *)QB.ptr)[i] = bf16((float)((int)(next() >> 27) - 16) / 16);
  for (size_t i = 0; i < (size_t)(ROWS + 128) * 512; i++) ((uint16_t *)WIN.ptr)[i] = bf16((float)((int)(next() >> 27) - 16) / 16);
  for (size_t i = 0; i < (size_t)COMP * 512; i++) ((uint16_t *)KEYS.ptr)[i] = bf16((float)((int)(next() >> 27) - 16) / 16);
  for (int i = 0; i < 64; i++) ((float *)SINK.ptr)[i] = (float)(next() >> 8) / (1 << 22) - 2;
  struct { const char *name; uint32_t pos, first, comp; int (*count)(int); int rows; } cases[] = {
      {"a full window and 512 selected rows", 5000, 0, 1, all, 96},
      {"the first positions, no compressed rows", 0, 0, 0, none, 200},
      {"0 to 512 selected rows", 300, 0, 1, varied, 96},
      {"a window cut short, 100 to 159 selected rows", 5000, 4990, 1, few, 96},
  };
  for (size_t c = 0; c < sizeof cases / sizeof *cases; c++) {
    uint32_t pos = cases[c].pos;
    Att at = {pos, pos >= 127 ? pos - 127 : 0, 1u << 30, cases[c].first, cases[c].comp};
    select_rows(cases[c].rows, cases[c].count);
    memset(OUT[0].ptr, 0xa5, OUT[0].size), memset(OUT[1].ptr, 0xa5, OUT[1].size);
    run(before, at, cases[c].rows, &OUT[0], 1), run(now_kernel, at, cases[c].rows, &OUT[1], 1);
    const float *a = (const float *)OUT[0].ptr, *b = (const float *)OUT[1].ptr;
    size_t count = (size_t)cases[c].rows * 32768, nonzero = 0;
    for (size_t i = 0; i < count; i++) {
      if (memcmp(a + i, b + i, 4)) die("prompt attention differs: %s, row %zu, head %zu, column %zu: %g against %g", cases[c].name,
                                       i / 32768, i % 32768 / 512, i % 512, b[i], a[i]);
      if (!isfinite(a[i])) die("prompt attention is not finite: %s", cases[c].name);
      nonzero += a[i] != 0;
    }
    if (nonzero < count / 2) die("prompt attention is mostly zero: %s", cases[c].name);
    printf("prompt attention, %s: %zu outputs equal\n", cases[c].name, count);
  }
  Att at = {20000, 20000 - 127, 1u << 30, 0, 1};
  select_rows(ROWS, all);
  run(now_kernel, at, ROWS, &OUT[1], 1);  // raise the GPU's clock
  double first = run(before, at, ROWS, &OUT[0], 4), middle = run(now_kernel, at, ROWS, &OUT[1], 4);
  double last = run(before, at, ROWS, &OUT[0], 4);
  printf("prompt attention of %d rows over 640 keys: a slice at a time %.0f / %.0f us, the kernel %.0f us\n", ROWS, first, last,
         middle);
  return 0;
}
