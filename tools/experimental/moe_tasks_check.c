// First pipeline building block: ready expert stages at different layers
// in one dispatch. Uses checkpoint weights, synthetic activations, no model
// state or SSD reads during timing. It is not a decoder throughput test.
#include "myds4.h"
#include "../../src/experimental/moe_tasks.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static const char source[] = {
#embed "../../src/kernels/formats.metal"
  , '\n',
#embed "../../src/kernels/kernels.metal"
  , '\n',
#embed "../../src/experimental/moe_tasks.metal"
};
typedef struct { uint64_t p[3]; float weight, pad; } Entry;
static_assert(sizeof(Entry) == 32);
enum { JOBS = 4, REPEATS = 48 };
typedef struct { Buf ex, x, h, shared, mix, res, out; } Job;
static Job jobs[JOBS];
static Buf tasks, dummy;
static int up, down, multi;

static Buf zeros(size_t n) {
  Buf b = gpu_alloc(n);
  memset(b.ptr, 0, n);
  return b;
}
static uint64_t address(Buf b) { return b.gpu + b.off; }
static void ordinary(int job, int kind) {
  Job *j = jobs + job;
  gpu_kern(kind ? down : up);
  gpu_buf(0, &j->ex, 0);
  if (!kind) {
    gpu_buf(1, &j->x, 0), gpu_buf(2, &j->h, 0);
    gpu_go(INTER / 32, TOPK, 1, 256, 1, 1);
  } else {
    uint32_t n = TOPK, pm = 0;
    gpu_buf(1, &j->h, 0), gpu_buf(2, &j->shared, 0), gpu_buf(3, &j->mix, 0);
    gpu_buf(4, &j->res, 0), gpu_buf(5, &j->out, 0);
    gpu_bytes(6, &n, 4), gpu_bytes(7, &pm, 4), gpu_buf(8, &dummy, 0);
    gpu_go(DIM / 4, 1, 1, TOPK * 32, 1, 1);
  }
}
static int kind_of(int pattern, int j) { return pattern == 2 ? j % 2 : pattern; }
static uint32_t plan(int n, int pattern) {
  MoeTask *t = (MoeTask *)tasks.ptr;
  uint32_t groups = 0;
  for (int i = 0; i < n; i++) {
    Job *j = jobs + i;
    int kind = kind_of(pattern, i);
    t[i] = (MoeTask){address(j->ex), address(j->x), address(j->h), address(j->shared), address(j->mix),
                  address(j->res), address(j->out), groups, kind ? DIM / 4 : TOPK * (INTER / 32), kind, TOPK};
    groups += t[i].groups;
  }
  return groups;
}
static void encode(int candidate, int n, int pattern, uint32_t groups) {
  if (!candidate) {
    for (int j = 0; j < n; j++) ordinary(j, kind_of(pattern, j));
  } else {
    uint32_t count = n;
    gpu_kern(multi), gpu_buf(0, &tasks, 0), gpu_bytes(1, &count, 4);
    gpu_go(groups, 1, 1, 256, 1, 1);
  }
}
static double timing(int candidate, int n, int pattern, uint32_t groups, double *wall) {
  double g = gpu_busy, start = now();
  gpu_begin();
  for (int i = 0; i < REPEATS; i++) encode(candidate, n, pattern, groups);
  gpu_commit(), gpu_sync();
  *wall = (now() - start) * 1e6 / REPEATS;
  return (gpu_busy - g) * 1e6 / REPEATS;
}
int main(int argc, char **argv) {
  if (argc != 2) die("usage: moe-tasks-check CHECKPOINT");
  if (getenv("MYDS4_PROFILE")) die("unset MYDS4_PROFILE for this check");
  alarm(120);
  gpu_init(source, sizeof source);
  up = gpu_kernel("moe_up"), down = gpu_kernel("moe_down"), multi = gpu_kernel("moe_tasks");
  TensorIndex *index = st_open(argv[1]);
  // Only 24 experts are allocated and read, about 453 MB. The cache limit
  // reserves metadata, not buffers for unused slots.
  cache_reserve(16), cache_init(index, 16ull << 30, LAYERS);
  const int layers[JOBS] = {3, 11, 23, 37};
  tasks = zeros(JOBS * sizeof(MoeTask)), dummy = zeros(16);
  for (int i = 0; i < JOBS; i++) {
    Job *j = jobs + i;
    j->ex = zeros(TOPK * sizeof(Entry)), j->x = zeros(DIM * 4), j->h = zeros(TOPK * INTER * 4);
    j->shared = zeros(DIM * 4), j->mix = zeros(MIX * 4), j->res = zeros(HC * DIM * 2), j->out = zeros(HC * DIM * 2);
    int ids[TOPK];
    for (int e = 0; e < TOPK; e++) ids[e] = 17 * i + e;
    cache_load(layers[i], ids, TOPK);
    Entry *entry = (Entry *)j->ex.ptr;
    for (int e = 0; e < TOPK; e++) {
      uint32_t at[3]; int fresh;
      const Buf *b = cache_get(layers[i], ids[e], at, &fresh);
      entry[e] = (Entry){{b->gpu + at[0], b->gpu + at[1], 0}, (float)(e + 1) / 32, 0};
    }
    for (int d = 0; d < DIM; d++) {
      ((float *)j->x.ptr)[d] = ((d * 17 + i * 11) % 33 - 16) / 16.0f;
      ((float *)j->shared.ptr)[d] = ((d * 13 + i * 3) % 17 - 8) / 16.0f;
    }
    for (int d = 0; d < HC * DIM; d++) {
      float f = ((d * 7 + i) % 17 - 8) / 16.0f;
      uint32_t u; memcpy(&u, &f, 4);
      ((uint16_t *)j->res.ptr)[d] = u >> 16;
    }
    for (int k = 0; k < MIX; k++) ((float *)j->mix.ptr)[k] = (k % 5 + 1) / 16.0f;
  }
  gpu_residency_commit();
  gpu_begin();
  for (int i = 0; i < JOBS; i++) ordinary(i, 0), ordinary(i, 1);
  gpu_commit(), gpu_sync();
  size_t hb = TOPK * INTER * 4, ob = HC * DIM * 2, errors = 0;
  uint8_t *ref = xmalloc(JOBS * (hb + ob));
  for (int j = 0; j < JOBS; j++) {
    memcpy(ref + j * (hb + ob), jobs[j].h.ptr, hb);
    memcpy(ref + j * (hb + ob) + hb, jobs[j].out.ptr, ob);
  }
  for (int n = 1; n <= JOBS; n *= 2) for (int pattern = 0; pattern < 3; pattern++) {
    if (n == 1 && pattern == 2) continue;
    uint32_t groups = plan(n, pattern);
    for (int j = 0; j < n; j++) {
      if (!kind_of(pattern, j)) memset(jobs[j].h.ptr, 0xa5, hb);
      else memset(jobs[j].out.ptr, 0xa5, ob);
    }
    gpu_begin(); encode(1, n, pattern, groups); gpu_commit(); gpu_sync();
    size_t different = 0, nonfinite = 0;
    for (int j = 0; j < JOBS; j++) {
      // Both buffers are checked: a task must leave its other output and
      // every inactive job intact, as well as compute its own result.
      for (size_t b = 0; b < hb; b++) different += jobs[j].h.ptr[b] != ref[j * (hb + ob) + b];
      for (size_t b = 0; b < ob; b++) different += jobs[j].out.ptr[b] != ref[j * (hb + ob) + hb + b];
      for (size_t b = 0; b < hb / 4; b++) nonfinite += !isfinite(((float *)jobs[j].h.ptr)[b]);
      for (size_t b = 0; b < ob / 2; b++) {
        uint32_t u = (uint32_t)((uint16_t *)jobs[j].out.ptr)[b] << 16;
        float f; memcpy(&f, &u, 4);
        nonfinite += !isfinite(f);
      }
    }
    errors += different + nonfinite;
    printf("{\"moe_tasks_check\":{\"jobs\":%d,\"pattern\":%d,\"different_bytes\":%zu,\"nonfinite\":%zu}}\n",
           n, pattern, different, nonfinite);
    if (different || nonfinite) continue;
    // Raise clocks with both paths before timing either. Data checking on
    // the CPU can leave the GPU idle between cases.
    double warm_wall;
    timing(0, n, pattern, groups, &warm_wall), timing(1, n, pattern, groups, &warm_wall);
    for (int trial = 0; trial < 3; trial++) {
      double wall[2], us[2];
      for (int pass = 0; pass < 2; pass++) {
        int candidate = (trial + pass) % 2;
        us[candidate] = timing(candidate, n, pattern, groups, wall + candidate);
      }
      printf("{\"moe_tasks_timing\":{\"jobs\":%d,\"pattern\":%d,\"trial\":%d,\"control_gpu_us\":%.3f,"
             "\"candidate_gpu_us\":%.3f,\"control_wall_us\":%.3f,\"candidate_wall_us\":%.3f}}\n",
             n, pattern, trial, us[0], us[1], wall[0], wall[1]);
    }
  }
  free(ref);
  return errors != 0;
}
