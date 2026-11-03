// Exact expert-row checks and bounded GPU timings; synthetic weights, no SSD reads.
#include "myds4.h"
#include "expert_batch.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

enum { ROWS = 6, K = 5120, H = 2304, N = 6, UNIQUE = ROWS * N };
typedef ExpertRef Ref;
static Buf weights[UNIQUE], refs, groups, x, h, shared, mix, res, out, parts;
static int up, down, combine, up_rows, down_rows, counts[2], ids_by_slot[UNIQUE];
static int failures, checks;
static int code_bytes;
static const char *format = "mxfp4";
static uint32_t random_state = 1;

static uint32_t random_bits(void) {
  uint32_t x = random_state;
  x ^= x << 13, x ^= x >> 17, x ^= x << 5;
  return random_state = x;
}

static void setup(int m, int pattern) {
  int private_ids[UNIQUE], np = 0;
  for (int e = 0; e < UNIQUE; e++)
    if (e != 6 && e != 17 && e != 29) private_ids[np++] = e;
  Ref *table = (Ref *)refs.ptr;
  for (int r = 0; r < m; r++) {
    int ids[N];
    for (int k = 0; k < N; k++)
      ids[k] = pattern == 0 ? k : pattern == 2 ? r * N + k
               : k < 3 ? (int[]){6, 17, 29}[k] : private_ids[r * 3 + k - 3];
    for (int i = 1; i < N; i++)
      for (int j = i; j > 0 && ids[j - 1] > ids[j]; j--) {
        int t = ids[j]; ids[j] = ids[j - 1], ids[j - 1] = t;
      }
    for (int k = 0; k < N; k++) {
      uint64_t at = code_bytes ? (uint64_t)K * H / 256 * code_bytes : EXPERT_W;
      table[r * N + k] = (Ref){{weights[ids[k]].gpu, weights[ids[k]].gpu + at,
                               code_bytes ? weights[ids[k]].gpu + 2 * at : 0},
                              .05f + .03f * k + .01f * r, 0};
      ids_by_slot[r * N + k] = ids[k];
    }
  }
}

static void sum(int m) {
  uint32_t n = N;
  for (int r = 0; r < m; r++) {
    gpu_kern(combine);
    gpu_buf(0, &parts, r * K * 8 * 4), gpu_buf(1, &shared, r * K * 4), gpu_buf(2, &mix, r * 24 * 4);
    gpu_buf(3, &res, r * 4 * K * 4), gpu_buf(4, &out, r * 4 * K * 4), gpu_bytes(5, &n, 4);
    gpu_go(K / 256, 1, 1, 256, 1, 1);
  }
}

static void scalar(int m, int separated) {
  for (int r = 0; r < m; r++) {
    gpu_kern(up);
    gpu_buf(0, &refs, r * N * sizeof(Ref)), gpu_buf(1, &x, r * K * 4), gpu_buf(2, &h, r * N * H * 4);
    gpu_go(H / 32, N, 1, 256, 1, 1);
    uint32_t n = N, pm = separated;
    gpu_kern(down);
    gpu_buf(0, &refs, r * N * sizeof(Ref)), gpu_buf(1, &h, r * N * H * 4);
    gpu_buf(2, &shared, r * K * 4), gpu_buf(3, &mix, r * 24 * 4);
    gpu_buf(4, &res, r * 4 * K * 4), gpu_buf(5, &out, r * 4 * K * 4);
    gpu_bytes(6, &n, 4), gpu_bytes(7, &pm, 4), gpu_buf(8, &parts, r * K * 8 * 4);
    gpu_go(K / (code_bytes ? 8 : 4), 1, 1, N * 32 * (code_bytes ? 2 : 1), 1, 1);
  }
  if (separated) sum(m);
}

static void plan(int m, int mode) {
  counts[0] = counts[1] = 0;
  for (int phase = 0; phase < (mode == 2 ? 1 : 2); phase++) {
    Ref masked[UNIQUE];
    memcpy(masked, refs.ptr, m * N * sizeof(Ref));
    for (int q = 0; q < m * N; q++)
      if ((mode == 3 && ids_by_slot[q] % 2 != phase) || (mode == 4 && !phase)) masked[q].p[0] = 0;
    ExpertRows *g = (ExpertRows *)groups.ptr + phase * UNIQUE;
    counts[phase] = expert_batch_plan(g, masked, m, N, N * sizeof(Ref));
  }
}

static void grouped(int m) {
  for (int phase = 0; phase < 2; phase++) {
    if (!counts[phase]) continue;
    size_t off = phase * UNIQUE * sizeof(ExpertRows);
    gpu_kern(up_rows);
    gpu_buf(0, &groups, off), gpu_buf(1, &x, 0), gpu_buf(2, &h, 0);
    gpu_go(H / 32, counts[phase], 1, 256, 1, 1);
    gpu_kern(down_rows);
    gpu_buf(0, &groups, off), gpu_buf(1, &h, 0), gpu_buf(2, &parts, 0);
    gpu_go(K / 32, counts[phase], 1, 256, 1, 1);
  }
  sum(m);
}

static double run(int m, int mode, int repeats) {
  if (mode >= 2) plan(m, mode);
  double before = gpu_busy;
  gpu_begin();
  for (int i = 0; i < repeats; i++) {
    if (mode < 2) scalar(m, mode);
    else grouped(m);
  }
  gpu_commit(), gpu_sync();
  return (gpu_busy - before) * 1e3 / repeats;
}

static void compare(const char *what, const void *want, const void *got, size_t n, int stride, int used) {
  int different = 0;
  for (size_t i = 0; i < n; i++)
    if (i % stride < (size_t)used) different += memcmp((const char *)want + i * 4, (const char *)got + i * 4, 4) != 0;
  checks++, failures += different != 0;
  if (different) fprintf(stderr, "%s: %d values differ\n", what, different);
}

int main(int argc, char **argv) {
  int check_only = 0;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--check-only")) check_only = 1;
    else if (!strcmp(argv[i], "--format") && i + 1 < argc) format = argv[++i];
    else die("usage: expert-batch-check [--check-only] [--format FORMAT]");
  }
  const char *formats[] = {"mxfp4", "q2_k", "iq2_xxs", "iq2_xs", "iq2_s", "iq3_xxs", "iq1_s"};
  const int sizes[] = {0, 84, 66, 74, 82, 98, 50};
  int kind = 0;
  while (kind < 7 && strcmp(format, formats[kind])) kind++;
  if (kind == 7) die("unsupported expert fixture format");
  code_bytes = sizes[kind];
  alarm(90);
  lock_model();
  static const char source[] = {
#embed "../src/kernels.metal"
      , '\n',
#embed "../src/gguf_grids.metal"
      , '\n',
#embed "../src/gguf.metal"
  };
  gpu_init(source, sizeof source);
  char names[4][64]; // gpu_kernel keeps these names until process exit
  if (code_bytes) {
    snprintf(names[0], sizeof names[0], "gg_up_%s", format), snprintf(names[1], sizeof names[1], "gg_down_%s", format);
    snprintf(names[2], sizeof names[2], "gg_up_rows_%s", format), snprintf(names[3], sizeof names[3], "gg_down_rows_%s", format);
  } else {
    strcpy(names[0], "moe_up"), strcpy(names[1], "moe_down");
    strcpy(names[2], "moe_up_rows"), strcpy(names[3], "moe_down_rows");
  }
  up = gpu_kernel(names[0]), down = gpu_kernel(names[1]), combine = gpu_kernel("moe_parts");
  up_rows = gpu_kernel(names[2]), down_rows = gpu_kernel(names[3]);
  for (int e = 0; e < UNIQUE; e++) {
    size_t bytes = code_bytes ? (size_t)3 * K * H / 256 * code_bytes : EXPERT_W + EXPERT_S;
    weights[e] = gpu_alloc(bytes);
    uint32_t *p = (uint32_t *)weights[e].ptr;
    for (size_t i = 0; i < bytes / 4; i++) p[i] = random_bits();
    if (!code_bytes) {
      for (size_t i = 0; i < EXPERT_S; i++) weights[e].ptr[EXPERT_W + i] = 118 + random_bits() % 4;
    } else {
      for (size_t at = 0; at < bytes; at += code_bytes) {
        _Float16 d = (_Float16)(1.0f / (256 << (random_bits() % 3)));
        memcpy(weights[e].ptr + at + (kind == 1 ? 80 : 0), &d, 2);
        if (kind == 1) memcpy(weights[e].ptr + at + 82, &d, 2);
      }
    }
  }
  refs = gpu_alloc(ROWS * N * sizeof(Ref));
  groups = gpu_alloc(2 * UNIQUE * sizeof(ExpertRows));
  x = gpu_alloc(ROWS * K * 4), h = gpu_alloc(ROWS * N * H * 4);
  shared = gpu_alloc(ROWS * K * 4), mix = gpu_alloc(ROWS * 24 * 4);
  res = gpu_alloc(ROWS * 4 * K * 4), out = gpu_alloc(ROWS * 4 * K * 4), parts = gpu_alloc(ROWS * K * 8 * 4);
  for (int r = 0; r < ROWS; r++)
    for (int i = 0; i < K; i++) {
      const float scale[ROWS] = {.125f, 0x1p-16f, 4, 0, 0x1p-30f, 64};
      ((float *)x.ptr)[r * K + i] = ((int)(random_bits() % 257) - 128) * scale[r];
      ((float *)shared.ptr)[r * K + i] = ((int)(random_bits() % 257) - 128) / 256.0f;
      for (int j = 0; j < 4; j++)
        ((float *)res.ptr)[(r * 4 + j) * K + i] = ((int)(random_bits() % 257) - 128) / 256.0f;
    }
  for (int i = 0; i < ROWS * 24; i++) ((float *)mix.ptr)[i] = (random_bits() % 65) / 64.0f;
  float *want = xmalloc(out.size), *want_h = xmalloc(h.size), *want_parts = xmalloc(parts.size);
  setup(ROWS, 0);
  if (!check_only) run(ROWS, 0, 32); // keep startup clock changes out of the timing samples
  for (int pattern = 0; pattern < 3; pattern++)
    for (int m = 1; m <= ROWS; m++) {
      setup(m, pattern);
      run(m, 0, 1);
      memcpy(want, out.ptr, m * 4 * K * 4);
      run(m, 1, 1);
      compare("scalar parts expansion", want, out.ptr, m * 4 * K, 1, 1);
      memcpy(want_h, h.ptr, m * N * H * 4), memcpy(want_parts, parts.ptr, m * K * 8 * 4);
      for (int mode = 2; mode <= 4; mode++) {
        memset(h.ptr, 0xcd, h.size), memset(parts.ptr, 0xcd, parts.size), memset(out.ptr, 0xcd, out.size);
        run(m, mode, 1);
        compare("grouped up", want_h, h.ptr, m * N * H, 1, 1);
        compare("grouped down", want_parts, parts.ptr, m * K * 8, 8, N);
        compare("grouped expansion", want, out.ptr, m * 4 * K, 1, 1);
      }
      if (check_only) continue;
      double times[3][3];
      for (int t = 0; t < 3; t++)
        for (int mode = 0; mode < 3; mode++) times[mode][t] = run(m, mode, 8);
      for (int mode = 0; mode < 3; mode++)
        for (int i = 1; i < 3; i++)
          for (int j = i; j > 0 && times[mode][j] < times[mode][j - 1]; j--) {
            double a = times[mode][j]; times[mode][j] = times[mode][j - 1], times[mode][j - 1] = a;
          }
      printf("{\"expert_batch\":{\"format\":\"%s\",\"rows\":%d,\"sharing\":%d,\"groups\":%d,"
             "\"scalar_ms\":%.6f,\"parts_ms\":%.6f,\"grouped_ms\":%.6f}}\n", format, m, pattern, counts[0],
             times[0][1], times[1][1], times[2][1]);
    }
  free(want), free(want_h), free(want_parts);
  fprintf(stderr, "expert batch %s: %d/%d GPU checks passed\n", format, checks - failures, checks);
  return failures != 0;
}
