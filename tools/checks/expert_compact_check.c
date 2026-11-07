// All partial masks, untouched slots and ordered sums; bounded resident timings.
#include "myds4.h"
#include "../../src/experimental/expert_compact.h"
#include "../reference/activation_format.h"
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

enum { K = 5120, H = 2304, N = 6, SETS = 8 };
static Buf w[SETS][N], refs, x, h, shared, mix, res, out, parts;
static int up, down, compact_up, compact_down, combine, checks, failures;
#ifdef MYDS4_PIPELINE_LIMIT
static const int limits = 1;
static const char *study = "pipeline_bound";
#else
static const int limits = 0;
static const char *study = "expert_compact";
#endif
static uint32_t random_state = 1;
static uint32_t bits(void) {
  uint32_t q = random_state;
  q ^= q << 13, q ^= q >> 17, q ^= q << 5;
  return random_state = q;
}
static void expect(const char *name, int ok) {
  checks++;
  if (!ok) failures++, fprintf(stderr, "FAIL: %s\n", name);
}
static size_t table(int set, int mask) {
  return (size_t)(set * 64 + mask) * 256;
}
static void batch(int compact, int set, int mask, int fused) {
  size_t eo = table(set, mask);
  int count = __builtin_popcount((unsigned)mask);
  int packed = compact && !limits;
  if (packed && !count) return;
  gpu_kern(compact ? compact_up : up);
  gpu_buf(0, &refs, eo), gpu_buf(1, &x, 0), gpu_buf(2, &h, 0);
  gpu_go(H / 32, packed ? count : N, 1, 256, 1, 1);
  gpu_kern(compact ? compact_down : down);
  gpu_buf(0, &refs, eo), gpu_buf(1, &h, 0);
  if (packed) gpu_buf(2, &parts, 0);
  else {
    uint32_t n = N, pm = !fused;
    gpu_buf(2, &shared, 0), gpu_buf(3, &mix, 0), gpu_buf(4, &res, 0), gpu_buf(5, &out, 0);
    gpu_bytes(6, &n, 4), gpu_bytes(7, &pm, 4), gpu_buf(8, &parts, 0);
  }
  gpu_go(K / 4, 1, 1, 32 * (packed ? count : N), 1, 1);
}
static void sum(void) {
  uint32_t n = N;
  gpu_kern(combine);
  gpu_buf(0, &parts, 0), gpu_buf(1, &shared, 0), gpu_buf(2, &mix, 0);
  gpu_buf(3, &res, 0), gpu_buf(4, &out, 0), gpu_bytes(5, &n, 4);
  gpu_go(K / 256, 1, 1, 256, 1, 1);
}
static double run(int compact, int mask, int full, int repeats) {
  double before = gpu_busy;
  gpu_begin();
  for (int i = 0; i < repeats; i++) {
    batch(compact, i % SETS, mask, 0);
    if (full) batch(compact, i % SETS, 63 ^ mask, 0), sum();
  }
  gpu_commit(), gpu_sync();
  return (gpu_busy - before) * 1000 / repeats;
}
static double median(double *v) {
  for (int i = 1; i < 7; i++) for (int j = i; j && v[j] < v[j - 1]; j--) {
    double t = v[j]; v[j] = v[j - 1], v[j - 1] = t;
  }
  return v[3];
}
int main(void) {
  alarm(90);
  lock_model();
  static const char source[] = {
#embed "../../src/kernels/kernels.metal"
    , '\n',
#embed "../../src/experimental/expert_compact.metal"
  };
  gpu_init(source, sizeof source);
  up = gpu_kernel("moe_up"), down = gpu_kernel("moe_down"), combine = gpu_kernel("moe_parts");
  compact_up = gpu_kernel("moe_up_active"), compact_down = gpu_kernel("moe_down_active");
#ifdef MYDS4_PIPELINE_LIMIT
  compact_up = up, compact_down = down;
  up = gpu_kernel_limited("moe_up", 0), down = gpu_kernel_limited("moe_down", 0);
  expect("bounded and default pipelines have distinct identities", up != compact_up && down != compact_down);
#endif
  refs = gpu_alloc(SETS * 64 * 256);
  x = gpu_alloc(K * 4), h = gpu_alloc(N * H * 4), shared = gpu_alloc(K * 4), mix = gpu_alloc(24 * 4);
  res = gpu_alloc(4 * K * STREAM_BYTES), out = gpu_alloc(res.size), parts = gpu_alloc(K * 8 * 4);
  for (int s = 0; s < SETS; s++) for (int e = 0; e < N; e++) {
    w[s][e] = gpu_alloc(EXPERT_W + EXPERT_S);
    uint32_t *p = (uint32_t *)w[s][e].ptr;
    for (size_t i = 0; i < EXPERT_W / 4; i++) p[i] = bits();
    for (size_t i = 0; i < EXPERT_S; i++) w[s][e].ptr[EXPERT_W + i] = 118 + bits() % 4;
  }
  for (int s = 0; s < SETS; s++) for (int mask = 0; mask < 64; mask++) {
    ExpertRef *r = (ExpertRef *)(refs.ptr + table(s, mask));
    memset(r, 0, 256);
    for (int e = 0; e < N; e++) if (mask & 1 << e)
      r[e] = (ExpertRef){{w[s][e].gpu, w[s][e].gpu + EXPERT_W, 0}, .05f + .03f * e, 0};
    expect("active list length", expert_active_slots(r) == __builtin_popcount((unsigned)mask));
  }
  for (int i = 0; i < K; i++) {
    ((float *)x.ptr)[i] = ((int)(bits() % 257) - 128) * .125f;
    ((float *)shared.ptr)[i] = ((int)(bits() % 257) - 128) / 256.f;
    for (int j = 0; j < 4; j++) ((uint16_t *)res.ptr)[j * K + i] = ref_bf16(((int)(bits() % 257) - 128) / 256.f);
  }
  for (int i = 0; i < 24; i++) ((float *)mix.ptr)[i] = bits() % 65 / 64.f;
  void *want_h = xmalloc(h.size), *want_parts = xmalloc(parts.size), *want_out = xmalloc(out.size);
  gpu_begin(), batch(0, 0, 63, 1), gpu_commit(), gpu_sync();
  memcpy(want_out, out.ptr, out.size);
  for (int mask = 0; mask < 64; mask++) {
    memset(h.ptr, 0xcd, h.size), memset(parts.ptr, 0xcd, parts.size);
    run(0, mask, 0, 1);
    memcpy(want_h, h.ptr, h.size), memcpy(want_parts, parts.ptr, parts.size);
    memset(h.ptr, 0xcd, h.size), memset(parts.ptr, 0xcd, parts.size);
    run(1, mask, 0, 1);
    expect("hidden values and inactive sentinels", !memcmp(want_h, h.ptr, h.size));
    expect("parts and inactive sentinels", !memcmp(want_parts, parts.ptr, parts.size));
    int untouched = 1;
    for (int e = 0; e < N; e++) if (!(mask & 1 << e)) {
      for (int i = 0; i < H; i++) untouched &= ((uint32_t *)h.ptr)[e * H + i] == 0xcdcdcdcd;
      for (int i = 0; i < K; i++) untouched &= ((uint32_t *)parts.ptr)[i * 8 + e] == 0xcdcdcdcd;
    }
    expect("inactive experts remain unwritten", untouched);
    run(1, mask, 1, 1);
    expect("complementary batches equal fused output", !memcmp(want_out, out.ptr, out.size));
  }
  printf("{\"%s_checks\":{\"passed\":%d,\"total\":%d,\"masks\":64}}\n", study, checks - failures, checks);
  if (failures) return 1;
  run(0, 7, 1, 64), run(1, 7, 1, 64);
  for (int n = 1; n <= 6; n += n < 3 ? 1 : 3) for (int full = 0; full < 2; full++) {
    double samples[2][7];
    for (int t = 0; t < 7; t++) for (int j = 0; j < 2; j++) {
      int v = (j + t) % 2;
      samples[v][t] = run(v, (1 << n) - 1, full, 64);
    }
    printf("{\"%s_timing\":{\"active\":%d,\"complete_split\":%s,\"sets\":%d,\"weight_bytes\":%llu,"
           "\"control_ms\":%.6f,\"candidate_ms\":%.6f}}\n", study, n, full ? "true" : "false", SETS,
           (unsigned long long)SETS * (full ? N : n) * (EXPERT_W + EXPERT_S), median(samples[0]), median(samples[1]));
  }
  free(want_h), free(want_parts), free(want_out);
  return 0;
}
