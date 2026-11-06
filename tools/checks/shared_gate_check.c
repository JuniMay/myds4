// Compare fused shared-expert activation with the engine's current two
// dispatches. The fixture stays in memory; no checkpoint is needed.
#include "myds4.h"
#include "../reference/activation_format.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

enum { K = 5120, N = 2304 };
typedef struct { uint32_t n, k, group_rows, group_k, round, split; } MV;
static Buf wg, sg, wu, su, x, gate, up, out;
static int mv, swiglu, act, fused[3];
static uint32_t rng = 524287;
static uint32_t word(void) { rng ^= rng << 13; rng ^= rng >> 17; return rng ^= rng << 5; }

static void dispatch(int variant) {
  if (!variant) {
    MV p = {N, K, N, 0, 1, 4};
    uint32_t rows[2][2] = {{N, N}, {N, N}};
    gpu_kern(mv);
    gpu_buf(0, &wg, 0), gpu_buf(1, &sg, 0), gpu_buf(2, &x, 0), gpu_buf(3, &gate, 0);
    gpu_bytes(4, &p, sizeof p), gpu_buf(5, &wu, 0), gpu_buf(6, &su, 0), gpu_buf(7, &up, 0);
    gpu_bytes(11, rows, sizeof rows);
    gpu_go(N * 4 / 32, 2, 1, 256, 1, 1);
    gpu_kern(swiglu);
    gpu_buf(0, &gate, 0), gpu_buf(1, &up, 0), gpu_buf(2, &out, 0);
    gpu_go(N / 256, 1, 1, 256, 1, 1);
  } else {
    gpu_kern(fused[variant - 1]);
    gpu_buf(0, &wg, 0), gpu_buf(1, &sg, 0), gpu_buf(2, &wu, 0), gpu_buf(3, &su, 0);
    gpu_buf(4, &x, 0), gpu_buf(5, &out, 0);
    gpu_go(variant == 3 ? N / 4 : N / 32, 1, 1, variant == 1 ? 256 : variant == 2 ? 1024 : 128, 1, 1);
    if (variant == 3) {
      gpu_kern(act);
      gpu_buf(0, &out, 0), gpu_buf(1, &out, 0);
      gpu_go(N / 256, 1, 1, 256, 1, 1);
    }
  }
}

static double run(int variant, int repeats) {
  double before = gpu_busy;
  gpu_begin();
  for (int i = 0; i < repeats; i++) dispatch(variant);
  gpu_commit(), gpu_sync();
  return (gpu_busy - before) * 1000 / repeats;
}

static double median(double *a, int n) {
  for (int i = 1; i < n; i++)
    for (int j = i; j && a[j] < a[j - 1]; j--) { double t = a[j]; a[j] = a[j - 1], a[j - 1] = t; }
  return a[n / 2];
}

int main(void) {
  alarm(120);
  lock_model();
  static const char source[] = {
#embed "../../src/kernels/kernels.metal"
    , '\n',
#embed "../../src/experimental/shared_gate.metal"
  };
  gpu_init(source, sizeof source);
  mv = gpu_kernel("mv_fp8_3"), swiglu = gpu_kernel("swiglu"), act = gpu_kernel("act_quant");
  fused[0] = gpu_kernel("shared_gate_fp8"), fused[1] = gpu_kernel("shared_gate_fp8_wide");
  fused[2] = gpu_kernel("shared_gate_fp8_quad");
  wg = gpu_alloc(N * K), wu = gpu_alloc(N * K);
  sg = gpu_alloc((N / 32) * (K / 32)), su = gpu_alloc(sg.size);
  x = gpu_alloc(K * 4), gate = gpu_alloc(N * 4), up = gpu_alloc(N * 4), out = gpu_alloc(N * 4);
  float reference[N];
  size_t different[3] = {0}, nonfinite[3] = {0};
  int checks = 0;
  for (int trial = 0; trial < 16; trial++) {
    for (size_t i = 0; i < wg.size; i++) {
      wg.ptr[i] = (word() & 0x80) | (word() % 127);
      wu.ptr[i] = (word() & 0x80) | (word() % 127);
    }
    for (size_t i = 0; i < sg.size; i++) {
      sg.ptr[i] = 105 + word() % 15;
      su.ptr[i] = 105 + word() % 15;
    }
    for (int first = 0; first < K; first += 32) {
      float a[32];
      for (int j = 0; j < 32; j++)
        a[j] = trial == 0 ? 0 : ldexpf(((int)(word() % 257) - 128) / 128.0f, trial - 10);
      RefFp8Block packed;
      if (!ref_fp8_pack(a, &packed)) die("bad fixture activation");
      ref_fp8_unpack(&packed, (float *)x.ptr + first);
    }
    run(0, 1), memcpy(reference, out.ptr, sizeof reference);
    for (int v = 1; v <= 3; v++) {
      run(v, 1);
      for (int j = 0; j < N; j++) {
        float got = ((float *)out.ptr)[j];
        different[v - 1] += memcmp(&got, reference + j, sizeof got) != 0;
        nonfinite[v - 1] += !isfinite(got);
      }
      checks++;
    }
  }
  double times[4][9];
  for (int v = 0; v < 4; v++) run(v, 32);
  for (int trial = 0; trial < 9; trial++)
    for (int i = 0; i < 4; i++) { int v = (trial + i) % 4; times[v][trial] = run(v, 32); }
  double control = median(times[0], 9);
  for (int v = 1; v <= 3; v++)
    printf("{\"shared_gate\":{\"variant\":%d,\"checked_values\":%d,\"different\":%zu,\"nonfinite\":%zu,"
           "\"control_ms\":%.6f,\"candidate_ms\":%.6f}}\n",
           v, 16 * N, different[v - 1], nonfinite[v - 1], control, median(times[v], 9));
  int bad = 0;
  for (int v = 0; v < 3; v++) bad |= different[v] || nonfinite[v];
  fprintf(stderr, "shared gate: %d comparisons; bitwise parity %s\n", checks, bad ? "FAIL" : "PASS");
  gpu_free(&wg), gpu_free(&wu), gpu_free(&sg), gpu_free(&su), gpu_free(&x), gpu_free(&gate), gpu_free(&up), gpu_free(&out);
  return bad;
}
