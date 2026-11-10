// What the pass scheduler's GPU steps depend on, measured on this machine:
// whether two independent kinds of work overlap inside one command buffer,
// and what it costs to cut a long run of work into short command buffers.
// Needs no model: the kernels are the matrix product of a prefill (gemm, on
// the GPU's matrix units) and the vector product of decoding (mv1, limited by
// memory bandwidth), on generated data. Run it on AC power with the GPU
// otherwise idle. Each line of output is a JSON object.
//
// usage: build/benchmarks/steps [--rounds N]
#include "../src/model/linear.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

// The matrix product: M rows of x through N weight rows of K columns, in one
// dispatch; for the cuts, UNIT_M rows through UNIT_N weight rows.
enum { N = 20480, K = 8192, M = 8192, UNIT_M = 128, UNIT_N = 5120 };
// The chain: CHAIN vector products, each over its own slice of ROWS x COLS
// weights, cycling through SLICES slices (2 GiB).
enum { ROWS = 1024, COLS = 32768, SLICES = 64, CHAIN = 320 };
#define SLICE ((size_t)ROWS * COLS)

static int k_gemm, k_mv;
static Buf mw, ms, mx, my, vw, vs, vx, vy;

static void matrix(uint32_t m, uint32_t n) {
  struct { uint32_t m, n, k, pad; float scale; uint32_t pad0, pad1, pad2; } p = {m, n, K, 0, .03125f, 0, 0, 0};
  gpu_kern(k_gemm);
  gpu_buf(0, &mw, 0), gpu_buf(1, &ms, 0), gpu_buf(2, &mx, 0), gpu_buf(3, &my, 0), gpu_bytes(4, &p, sizeof p);
  gpu_go((n + 63) / 64, (m + 127) / 128, 1, 128, 1, 1);
}

enum { NO_BARRIER, BARRIER_RESOURCE, BARRIER_ALL };

// Each product writes its own part of vy; the barriers order them as a pass
// orders its dispatches.
static void chain(int barrier) {
  struct { uint32_t rows, cols, n; float scale; } p = {ROWS, COLS, 1, .03125f};
  const Buf *out = &vy;
  for (int i = 0; i < CHAIN; i++) {
    size_t off = (size_t)(i % SLICES) * SLICE;
    if (i && barrier == BARRIER_RESOURCE) gpu_barrier_on(&out, 1);
    if (i && barrier == BARRIER_ALL) gpu_barrier();
    gpu_kern(k_mv);
    gpu_buf(0, &vw, off), gpu_buf(1, &vs, off / 16), gpu_buf(2, &vx, 0);
    gpu_buf(3, &vy, (size_t)i * ROWS * 4), gpu_bytes(4, &p, sizeof p);
    gpu_go((ROWS + 7) / 8, 1, 1, 256, 1, 1);
  }
}

enum { MATRIX, CHAIN_ALONE, SERIAL, TWO_ENCODERS, RESOURCE, ALL, NONE, NONE_CHAIN_FIRST, CHAIN_NONE, CHAIN_RESOURCE, MODES };
static const char *const mode_name[MODES] = {
  "matrix alone", "chain alone", "serial", "two encoders", "concurrent, barriers on the chain's output",
  "concurrent, barriers on all buffers", "concurrent, no barriers", "concurrent, no barriers, chain first",
  "chain alone, concurrent, no barriers", "chain alone, concurrent, barriers on its output"};
static int both_kinds(int mode) { return mode >= SERIAL && mode <= NONE_CHAIN_FIRST; }

static void run(int mode, double *gpu, double *wall) {
  double busy = gpu_busy, t0 = now();
  void *c = gpu_open();
  if (mode >= RESOURCE) gpu_encode_concurrent(c);
  else gpu_encode(c);
  switch (mode) {
    case MATRIX: matrix(M, N); break;
    case CHAIN_ALONE: chain(NO_BARRIER); break;
    case SERIAL: matrix(M, N); chain(NO_BARRIER); break;
    case TWO_ENCODERS: matrix(M, N); gpu_encode(c); chain(NO_BARRIER); break;
    case RESOURCE: matrix(M, N); chain(BARRIER_RESOURCE); break;
    case ALL: matrix(M, N); chain(BARRIER_ALL); break;
    case NONE: matrix(M, N); chain(NO_BARRIER); break;
    case NONE_CHAIN_FIRST: chain(NO_BARRIER); matrix(M, N); break;
    case CHAIN_NONE: chain(NO_BARRIER); break;
    case CHAIN_RESOURCE: chain(BARRIER_RESOURCE); break;
  }
  gpu_commit();
  gpu_sync();
  *gpu = gpu_busy - busy;
  *wall = now() - t0;
}

static int by_value(const void *a, const void *b) {
  double x = *(const double *)a, y = *(const double *)b;
  return (x > y) - (x < y);
}

typedef struct { double wall, busy, gap; long gaps; } Timed;

// `total` matrix products in command buffers of `per`, at most `ahead`
// buffers committed and not finished; each waits for its turn, as a
// scheduler's thread would.
static Timed cut(int total, int per, int ahead) {
  static uint64_t signals[16384];
  double busy = gpu_busy, gap = gpu_gap;
  long gaps = gpu_gaps;
  double t0 = now();
  int buffers = 0;
  for (int done = 0; done < total; buffers++) {
    if (buffers >= ahead) gpu_wait_signal(signals[buffers - ahead]);
    gpu_begin();
    for (int i = 0; i < per && done < total; i++, done++) matrix(UNIT_M, UNIT_N);
    signals[buffers] = gpu_signal();
    gpu_commit();
  }
  gpu_sync();
  return (Timed){now() - t0, gpu_busy - busy, gpu_gap - gap, gpu_gaps - gaps};
}

static void repeat(uint8_t *p, size_t size, size_t block) {
  for (size_t i = block; i < size; i += block) memcpy(p + i, p, size - i < block ? size - i : block);
}

static void weights(Buf *w, Buf *s) {
  size_t block = 1 << 20, n = w->size < block ? w->size : block;
  for (size_t k = 0; k < n; k++) w->ptr[k] = (k * 73 + k / 19) % 120;
  repeat((uint8_t *)w->ptr, w->size, n);
  n = s->size < block ? s->size : block;
  for (size_t k = 0; k < n; k++) s->ptr[k] = 32 + k % 40;
  repeat((uint8_t *)s->ptr, s->size, n);
}

int main(int argc, char **argv) {
  int rounds = argc == 3 && !strcmp(argv[1], "--rounds") ? atoi(argv[2]) : 5;
  if (rounds < 1 || rounds > 99) die("usage: steps [--rounds N]");
  memory_configure(8, 10);
  linear_init();
  k_gemm = gpu_kernel("gemm_fp8");
  k_mv = gpu_kernel("mv1_fp8");
  mw = gpu_alloc((size_t)N * K), ms = gpu_alloc((size_t)N * K / 16);
  mx = gpu_alloc((size_t)M * K * 2), my = gpu_alloc((size_t)M * N * 4);
  vw = gpu_alloc(SLICES * SLICE), vs = gpu_alloc(SLICES * SLICE / 16);
  vx = gpu_alloc((size_t)COLS * 4), vy = gpu_alloc((size_t)CHAIN * ROWS * 4);
  weights(&mw, &ms);
  weights(&vw, &vs);
  size_t values = (size_t)M * K, pattern = values < (1u << 19) ? values : (1u << 19);
  for (size_t k = 0; k < pattern; k++) {
    float value = sinf(k * .031f) * 3;
    uint32_t bits;
    memcpy(&bits, &value, 4);
    ((uint16_t *)mx.ptr)[k] = bits >> 16;
  }
  repeat((uint8_t *)mx.ptr, mx.size, pattern * 2);
  for (int k = 0; k < COLS; k++) ((float *)vx.ptr)[k] = sinf(k * .031f) * 3;

  // 1. Overlap: every mode in each round, the first round discarded (clocks
  // and first-use costs).
  static double gpu[MODES][100], wall[MODES][100];
  double ignored;
  for (int mode = 0; mode < MODES; mode++) run(mode, &ignored, &ignored);
  for (int r = 0; r < rounds; r++)
    for (int mode = 0; mode < MODES; mode++) run(mode, &gpu[mode][r], &wall[mode][r]);
  double median[MODES];
  for (int mode = 0; mode < MODES; mode++) {
    qsort(gpu[mode], rounds, sizeof(double), by_value);
    qsort(wall[mode], rounds, sizeof(double), by_value);
    median[mode] = gpu[mode][rounds / 2];
  }
  for (int mode = 0; mode < MODES; mode++) {
    printf("{\"overlap\":\"%s\",\"gpu_ms\":%.3f,\"gpu_min_ms\":%.3f,\"gpu_max_ms\":%.3f,\"wall_ms\":%.3f",
           mode_name[mode], median[mode] * 1e3, gpu[mode][0] * 1e3, gpu[mode][rounds - 1] * 1e3,
           wall[mode][rounds / 2] * 1e3);
    if (both_kinds(mode)) printf(",\"chain_hidden\":%.3f", (median[SERIAL] - median[mode]) / median[CHAIN_ALONE]);
    printf("}\n");
  }
  fflush(stdout);

  // 2. Cutting: about a second of matrix products, whole and in buffers of
  // the step's length, against the mean of the whole runs before and after.
  double busy = gpu_busy;
  gpu_begin();
  for (int i = 0; i < 64; i++) matrix(UNIT_M, UNIT_N);
  gpu_commit();
  gpu_sync();
  double unit = (gpu_busy - busy) / 64;
  int total = (int)(1.0 / unit);
  if (total > 16000) total = 16000;
  printf("{\"unit_dispatch_us\":%.2f,\"dispatches\":%d}\n", unit * 1e6, total);
  fflush(stdout);
  const double steps_ms[] = {0.5, 1, 2, 5, 10};
  for (size_t s = 0; s < sizeof steps_ms / sizeof *steps_ms; s++)
    for (int ahead = 1; ahead <= 3; ahead++) {
      int per = (int)lround(steps_ms[s] * 1e-3 / unit);
      if (per < 1) per = 1;
      double whole = cut(total, total, 1).wall;
      Timed t = cut(total, per, ahead);
      whole = (whole + cut(total, total, 1).wall) / 2;
      printf("{\"step_ms\":%.1f,\"per_buffer\":%d,\"ahead\":%d,\"whole_s\":%.4f,\"cut_s\":%.4f,\"cost_percent\":%.2f,"
             "\"gpu_idle_ms\":%.2f,\"gaps_over_100us\":%ld}\n",
             steps_ms[s], per, ahead, whole, t.wall, (t.wall / whole - 1) * 100, t.gap * 1e3, t.gaps);
      fflush(stdout);
    }
  return 0;
}
