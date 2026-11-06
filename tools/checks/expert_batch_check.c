// Exact expert-row checks and bounded GPU timings; synthetic weights, no SSD reads.
#include "myds4.h"
#include "expert_batch.h"
#include "../reference/activation_format.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

enum { ROWS = 6, K = 5120, H = 2304, N = 6, UNIQUE = ROWS * N };
typedef ExpertRef Ref;
static Buf weights[UNIQUE], refs, groups, x, h, shared, mix, res, out, parts;
static int up, down, combine, up_rows, down_rows, counts[2], ids_by_slot[UNIQUE];
static int load_up, load_down;
static int specialize, specialized_up[3], specialized_down[3];
static const char *loads;
typedef struct { uint64_t p[3]; uint32_t start, count; } SingleTile;
static Buf single_tiles, single_x, single_h, single_ids, single_weights;
static int single_gather, single_up, single_quant, single_down, use_matrix, materialize_hidden;
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
    gpu_buf(3, &res, r * 4 * K * STREAM_BYTES), gpu_buf(4, &out, r * 4 * K * STREAM_BYTES), gpu_bytes(5, &n, 4);
    gpu_go(K / 256, 1, 1, 256, 1, 1);
  }
}

static void scalar(int m, int separated) {
  if (use_matrix) {
    SingleTile *tiles = (SingleTile *)single_tiles.ptr;
    int nt = 0;
    const Ref *table = (const Ref *)refs.ptr;
    for (int q = 0; q < m * N; q++)
      if (table[q].p[0]) tiles[nt++] = (SingleTile){{table[q].p[0],table[q].p[1],table[q].p[2]}, q, 1};
    for (int q = 0; q < m * N; q++) ((float *)single_weights.ptr)[q] = table[q].weight;
    if (nt) {
      uint32_t pairs = m * N;
      gpu_kern(single_gather);
      gpu_buf(0, &x, 0), gpu_buf(1, &single_ids, 0), gpu_buf(2, &single_x, 0);
      gpu_go(K / 4 / 256, pairs, 1, 256, 1, 1);
      gpu_kern(single_up);
      gpu_buf(0, &single_tiles, 0), gpu_buf(1, &single_x, 0), gpu_buf(2, &single_weights, 0), gpu_buf(3, &h, 0), gpu_bytes(4, &pairs, 4);
      gpu_go(nt, H / 64, 1, 128, 1, 1);
      gpu_kern(single_quant);
      gpu_buf(0, &h, 0), gpu_buf(1, &single_h, 0);
      gpu_go(pairs * H / 256, 1, 1, 256, 1, 1);
      gpu_kern(single_down);
      gpu_buf(0, &single_tiles, 0), gpu_buf(1, &single_h, 0), gpu_buf(2, &parts, 0), gpu_bytes(3, &pairs, 4);
      gpu_go(nt, K / 128, 1, 128, 1, 1);
      // Expose the same quantized hidden rows as the scalar path for checks.
      if (materialize_hidden) {
        gpu_kern(gpu_kernel("act_quant"));
        gpu_buf(0, &h, 0), gpu_buf(1, &h, 0);
        gpu_go(pairs * H / 256, 1, 1, 256, 1, 1);
      }
    }
    if (separated != 2) sum(m);
    return;
  }
  for (int r = 0; r < m; r++) {
    gpu_kern(up);
    gpu_buf(0, &refs, r * N * sizeof(Ref)), gpu_buf(1, &x, r * K * 4), gpu_buf(2, &h, r * N * H * 4);
    gpu_go(H / 32, N, 1, 256, 1, 1);
    uint32_t n = N, pm = separated != 0;
    gpu_kern(down);
    gpu_buf(0, &refs, r * N * sizeof(Ref)), gpu_buf(1, &h, r * N * H * 4);
    gpu_buf(2, &shared, r * K * 4), gpu_buf(3, &mix, r * 24 * 4);
    gpu_buf(4, &res, r * 4 * K * STREAM_BYTES), gpu_buf(5, &out, r * 4 * K * STREAM_BYTES);
    gpu_bytes(6, &n, 4), gpu_bytes(7, &pm, 4), gpu_buf(8, &parts, r * K * 8 * 4);
    gpu_go(K / (code_bytes ? 8 : 4), 1, 1, N * 32 * (code_bytes ? 2 : 1), 1, 1);
  }
  if (separated == 1) sum(m);
}

static void split_scalar(int m) {
  Ref original[UNIQUE];
  memcpy(original, refs.ptr, m * N * sizeof(Ref));
  for (int phase = 0; phase < 2; phase++) {
    memcpy(refs.ptr, original, m * N * sizeof(Ref));
    for (int q = 0; q < m * N; q++)
      if (ids_by_slot[q] % 2 != phase) ((Ref *)refs.ptr)[q].p[0] = 0;
    scalar(m, 2);
    // The next phase rewrites the shared descriptor buffer.
    gpu_commit(), gpu_sync(), gpu_begin();
  }
  memcpy(refs.ptr, original, m * N * sizeof(Ref));
  sum(m);
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
    if (specialize)
      for (int i = 1; i < counts[phase]; i++)
        for (int j = i; j > 0 && g[j].count < g[j - 1].count; j--) {
          ExpertRows t = g[j]; g[j] = g[j - 1], g[j - 1] = t;
        }
  }
}

static void grouped(int m) {
  for (int phase = 0; phase < 2; phase++) {
    if (!counts[phase]) continue;
    size_t off = phase * UNIQUE * sizeof(ExpertRows);
    if (specialize) {
      const ExpertRows *g = (const ExpertRows *)(groups.ptr + off);
      for (int first = 0, last; first < counts[phase]; first = last) {
        for (last = first + 1; last < counts[phase] && g[last].count == g[first].count; last++) {}
        int n = g[first].count;
        size_t at = off + (size_t)first * sizeof(ExpertRows);
        gpu_kern(specialized_up[n - 1]);
        gpu_buf(0, &groups, at), gpu_buf(1, &x, 0), gpu_buf(2, &h, 0);
        gpu_go(H / 32, last - first, 1, 256, 1, 1);
        gpu_kern(specialized_down[n - 1]);
        gpu_buf(0, &groups, at), gpu_buf(1, &h, 0), gpu_buf(2, &parts, 0);
        gpu_go(K / 32, last - first, 1, 256, 1, 1);
      }
      continue;
    }
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
  if (mode >= 2 && mode <= 4) plan(m, mode);
  int saved_up = up, saved_down = down;
  if (mode >= 5) up = load_up, down = load_down;
  use_matrix = mode >= 5 && loads && !strncmp(loads, "matrix", 6);
  materialize_hidden = repeats == 1;
  double before = gpu_busy;
  gpu_begin();
  for (int i = 0; i < repeats; i++) {
    if (mode == 7) split_scalar(m);
    else if (mode >= 5) scalar(m, mode - 5);
    else if (mode < 2) scalar(m, mode);
    else grouped(m);
  }
  gpu_commit(), gpu_sync();
  up = saved_up, down = saved_down;
  use_matrix = 0;
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
    else if (!strcmp(argv[i], "--loads") && i + 1 < argc) loads = argv[++i];
    else if (!strcmp(argv[i], "--specialize")) specialize = 1;
    else die("usage: expert-batch-check [--check-only] [--format FORMAT] [--loads early|pipeline|row|factor|matrix|matrix8] [--specialize]");
  }
  const char *formats[] = {"mxfp4", "q2_k", "iq2_xxs", "iq2_xs", "iq2_s", "iq3_xxs", "iq1_s"};
  const int sizes[] = {0, 84, 66, 74, 82, 98, 50};
  int kind = 0;
  while (kind < 7 && strcmp(format, formats[kind])) kind++;
  if (kind == 7) die("unsupported expert fixture format");
  code_bytes = sizes[kind];
  if (specialize && (code_bytes || loads)) die("--specialize needs native MXFP4 without --loads");
  if (loads && (code_bytes || (strcmp(loads, "early") && strcmp(loads, "pipeline") && strcmp(loads, "matrix") && strcmp(loads, "matrix8") &&
                             strcmp(loads, "row") && strcmp(loads, "factor"))))
    die("--loads takes early, pipeline, row, factor, matrix or matrix8 with native MXFP4 only");
  alarm(90);
  lock_model();
  static const char source[] = {
#embed "../../src/kernels/kernels.metal"
      , '\n',
#embed "../../src/kernels/gguf_grids.metal"
      , '\n',
#embed "../../src/kernels/gguf.metal"
      , '\n',
#embed "../../src/experimental/mxfp4_loads.metal"
      , '\n',
#embed "../../src/experimental/expert_rows_specialized.metal"
  };
  gpu_init(source, sizeof source);
  char names[6][64]; // gpu_kernel keeps these names until process exit
  if (code_bytes) {
    snprintf(names[0], sizeof names[0], "gg_up_%s", format), snprintf(names[1], sizeof names[1], "gg_down_%s", format);
    snprintf(names[2], sizeof names[2], "gg_up_rows_%s", format), snprintf(names[3], sizeof names[3], "gg_down_rows_%s", format);
  } else {
    strcpy(names[0], "moe_up"), strcpy(names[1], "moe_down");
    strcpy(names[2], "moe_up_rows"), strcpy(names[3], "moe_down_rows");
  }
  up = gpu_kernel(names[0]), down = gpu_kernel(names[1]), combine = gpu_kernel("moe_parts");
  up_rows = gpu_kernel(names[2]), down_rows = gpu_kernel(names[3]);
  if (specialize) {
    static const char *up_names[] = {"moe_up_rows_1", "moe_up_rows_2", "moe_up_rows_3"};
    static const char *down_names[] = {"moe_down_rows_1", "moe_down_rows_2", "moe_down_rows_3"};
    for (int i = 0; i < 3; i++) specialized_up[i] = gpu_kernel(up_names[i]), specialized_down[i] = gpu_kernel(down_names[i]);
  }
  if (loads && strncmp(loads, "matrix", 6)) {
    snprintf(names[4], sizeof names[4], "moe_up_%s", loads);
    snprintf(names[5], sizeof names[5], "moe_down_%s", loads);
    load_up = gpu_kernel(names[4]), load_down = gpu_kernel(names[5]);
  }
  if (loads && !strncmp(loads, "matrix", 6)) {
    single_tiles = gpu_alloc(UNIQUE * sizeof(SingleTile));
    single_x = gpu_alloc((UNIQUE + 64) * K * 2), single_h = gpu_alloc((UNIQUE + 64) * H * 2);
    single_ids = gpu_alloc(UNIQUE * 4), single_weights = gpu_alloc(UNIQUE * 4);
    memset(single_x.ptr, 0, single_x.size), memset(single_h.ptr, 0, single_h.size);
    for (int q = 0; q < UNIQUE; q++) ((int *)single_ids.ptr)[q] = q / N;
    int small = !strcmp(loads, "matrix8");
    single_gather = gpu_kernel("gather_mxfp4"), single_up = gpu_kernel(small ? "moe_up_small8" : "moe_up_m");
    single_quant = gpu_kernel("act_quant_mxfp4"), single_down = gpu_kernel(small ? "moe_down_small8" : "moe_down_single_m");
  }
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
  res = gpu_alloc(ROWS * 4 * K * STREAM_BYTES), out = gpu_alloc(ROWS * 4 * K * STREAM_BYTES), parts = gpu_alloc(ROWS * K * 8 * 4);
  for (int r = 0; r < ROWS; r++)
    for (int i = 0; i < K; i++) {
      const float scale[ROWS] = {.125f, 0x1p-16f, 4, 0, 0x1p-30f, 64};
      ((float *)x.ptr)[r * K + i] = ((int)(random_bits() % 257) - 128) * scale[r];
      ((float *)shared.ptr)[r * K + i] = ((int)(random_bits() % 257) - 128) / 256.0f;
      for (int j = 0; j < 4; j++)
        ((uint16_t *)res.ptr)[(r * 4 + j) * K + i] = ref_bf16(((int)(random_bits() % 257) - 128) / 256.0f);
    }
  if (loads && !strncmp(loads, "matrix", 6)) {
    // The native matrix path accepts FP8-quantized inputs within its checked
    // staging bound. The scalar-only fixture above deliberately exceeds it.
    for (int r = 0; r < ROWS; r++)
      for (int first = 0; first < K; first += REF_ACT_BLOCK) {
        float *block = (float *)x.ptr + r * K + first;
        for (int i = 0; i < REF_ACT_BLOCK; i++) block[i] = fmaxf(-128, fminf(128, block[i]));
        RefFp8Block packed;
        if (!ref_fp8_pack(block, &packed)) die("invalid matrix input fixture");
        ref_fp8_unpack(&packed, block);
      }
  }
  for (int i = 0; i < ROWS * 24; i++) ((float *)mix.ptr)[i] = (random_bits() % 65) / 64.0f;
  float *want = xmalloc(out.size), *want_h = xmalloc(h.size), *want_parts = xmalloc(parts.size);
  setup(ROWS, 0);
  if (!check_only) run(ROWS, 0, 32); // keep startup clock changes out of the timing samples
  for (int pattern = 0; pattern < 3; pattern++)
    for (int m = 1; m <= ROWS; m++) {
      setup(m, pattern);
      run(m, 0, 1);
      memcpy(want, out.ptr, m * 4 * K * STREAM_BYTES);
      run(m, 1, 1);
      compare("scalar parts expansion", want, out.ptr, m * K * STREAM_BYTES, 1, 1);
      memcpy(want_h, h.ptr, m * N * H * 4), memcpy(want_parts, parts.ptr, m * K * 8 * 4);
      for (int mode = 5; loads && mode <= 7; mode++) {
        memset(h.ptr, 0xcd, h.size), memset(parts.ptr, 0xcd, parts.size), memset(out.ptr, 0xcd, out.size);
        run(m, mode, 1);
        compare("load-scheduled up", want_h, h.ptr, m * N * H, 1, 1);
        if (mode >= 6) compare("load-scheduled down", want_parts, parts.ptr, m * K * 8, 8, N);
        compare("load-scheduled expansion", want, out.ptr, m * K * STREAM_BYTES, 1, 1);
      }
      for (int mode = 2; mode <= 4; mode++) {
        memset(h.ptr, 0xcd, h.size), memset(parts.ptr, 0xcd, parts.size), memset(out.ptr, 0xcd, out.size);
        run(m, mode, 1);
        compare("grouped up", want_h, h.ptr, m * N * H, 1, 1);
        compare("grouped down", want_parts, parts.ptr, m * K * 8, 8, N);
        compare("grouped expansion", want, out.ptr, m * K * STREAM_BYTES, 1, 1);
      }
      if (check_only) continue;
      if (specialize) {
        double times[2][7];
        for (int t = 0; t < 7; t++)
          for (int i = 0; i < 2; i++) {
            int v = (i + t) % 2;
            specialize = v;
            times[v][t] = run(m, 2, 16);
          }
        specialize = 1;
        for (int v = 0; v < 2; v++)
          for (int i = 1; i < 7; i++)
            for (int j = i; j > 0 && times[v][j] < times[v][j - 1]; j--) {
              double t = times[v][j]; times[v][j] = times[v][j - 1], times[v][j - 1] = t;
            }
        printf("{\"expert_specialize\":{\"rows\":%d,\"sharing\":%d,\"control_ms\":%.6f,\"candidate_ms\":%.6f}}\n",
               m, pattern, times[0][3], times[1][3]);
        continue;
      }
      if (loads) {
        double control[7], candidate[7];
        for (int t = 0; t < 7; t++) {
          if (t % 2) candidate[t] = run(m, 5, 16), control[t] = run(m, 0, 16);
          else control[t] = run(m, 0, 16), candidate[t] = run(m, 5, 16);
        }
        for (int i = 1; i < 7; i++) {
          for (int j = i; j > 0 && control[j] < control[j - 1]; j--) { double v = control[j]; control[j] = control[j - 1]; control[j - 1] = v; }
          for (int j = i; j > 0 && candidate[j] < candidate[j - 1]; j--) { double v = candidate[j]; candidate[j] = candidate[j - 1]; candidate[j - 1] = v; }
        }
        printf("{\"mxfp4_loads\":{\"variant\":\"%s\",\"rows\":%d,\"sharing\":%d,\"control_ms\":%.6f,\"candidate_ms\":%.6f}}\n",
               loads, m, pattern, control[3], candidate[3]);
        continue;
      }
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
