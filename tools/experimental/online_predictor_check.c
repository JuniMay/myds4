// Standalone CPU math checks and microbenchmarks. No inference/GPU hooks.
// Build: clang -std=c23 -O2 -framework Accelerate tools/experimental/online_predictor_check.c -o build/online_predictor_check
// Basis RNG: xorshift64*/Box-Muller, deliberately different from PyTorch.
// --export-a-fd N writes OPRA0001, uint32[pairs,F,R,4], then FP32 A[pairs,F,R]
// to an already-open pipe. Apple Silicon uses little endian. No files opened.
#include "../../src/experimental/online_predictor_math.h"

#define K OP_K
#define MAX_ROWS 3
#define COV_LIMIT OP_COV_LIMIT
typedef OpPair Pair;
typedef OpWork Work;
static int checks, failures;
static volatile double sink;
static void check(const char *name, bool ok) {
  checks++;
  if (!ok) {
    failures++;
    fprintf(stderr, "FAIL: %s\n", name);
  }
}

static void native_ids(Pair *p, const float *gp, int m, int *ids) {
  float scores[384];
  for (int row = 0; row < m; row++) {
    for (int j = 0; j < p->e; j++) scores[j] = op_key(gp[row * p->e + j], p->bias[j]);
    op_top6(scores, p->e, ids + row * K);
  }
}
static void double_forward(Pair *p, const float *gp, const float *gl, int m, double *out) {
  double feature[768], hidden[128];
  for (int row = 0; row < m; row++) {
    for (int part = 0; part < 2; part++) {
      const float *x = (part ? gl : gp) + row * p->e;
      double mean = 0, ss = 0;
      for (int j = 0; j < p->e; j++) mean += x[j];
      mean /= p->e;
      for (int j = 0; j < p->e; j++) ss += (x[j] - mean) * (x[j] - mean);
      for (int j = 0; j < p->e; j++) feature[part * p->e + j] = (x[j] - mean) / sqrt(ss / p->e + 1e-6);
    }
    for (int j = 0; j < p->r; j++) {
      double v = 0;
      for (int i = 0; i < p->f; i++) v += feature[i] * p->a[i * p->r + j];
      hidden[j] = fmax(v, 0);
    }
    for (int j = 0; j < p->e; j++) {
      double v = op_key_double(gp[row * p->e + j], p->bias[j]);
      for (int i = 0; i < p->r; i++) v += hidden[i] * p->b[i * p->e + j];
      out[row * p->e + j] = v;
    }
  }
}
static void double_update(int r, int e, double *p, double *b, const float *phi, const float *target) {
  double v[128], error[384], denominator = (double).995f;
  for (int i = 0; i < r; i++) {
    v[i] = 0;
    for (int j = 0; j < r; j++) v[i] += p[i * r + j] * phi[j];
    denominator += phi[i] * v[i];
  }
  for (int j = 0; j < e; j++) {
    double y = 0;
    for (int i = 0; i < r; i++) y += phi[i] * b[i * e + j];
    error[j] = target[j] - y;
  }
  for (int i = 0; i < r; i++) {
    for (int j = 0; j < e; j++) b[i * e + j] += v[i] / denominator * error[j];
    for (int j = 0; j < r; j++) p[i * r + j] = (p[i * r + j] - v[i] * v[j] / denominator) / (double).995f;
  }
}

static void validate(int stress_rank) {
  Pair p = op_create_pair(8, 4, 91);
  Work w = op_create_work(8, 4);
  uint64_t seed = 81;
  float gp[24], gl[24], labels[24];
  int native[18], out[18];
  double ref[24];
  for (int i = 0; i < 24; i++) {
    gp[i] = op_gaussian(&seed);
    gl[i] = op_gaussian(&seed);
    labels[i] = op_gaussian(&seed) * .1f;
  }
  native_ids(&p, gp, 3, native);
  check("B=0 preserves native candidate IDs",
        op_forward(&p, &w, gp, gl, 3, native, out) == 0 && !memcmp(out, native, sizeof out));
  for (int i = 0; i < 32; i++) p.b[i] = op_gaussian(&seed) * .03f;
  p.active = 1;
  float before_b[32], before_p[16];
  memcpy(before_b, p.b, sizeof before_b);
  memcpy(before_p, p.p, sizeof before_p);
  check("finite forward succeeds", op_forward(&p, &w, gp, gl, 3, native, out) == 0);
  double_forward(&p, gp, gl, 3, ref);
  double worst = 0;
  for (int i = 0; i < 24; i++) worst = fmax(worst, fabs(ref[i] - w.scores[i]));
  check("forward matches double reference", worst < 2e-5);
  bool ranked = true;
  for (int row = 0; row < 3; row++) {
    int ids[8];
    for (int i = 0; i < 8; i++) ids[i] = i;
    for (int i = 0; i < 8; i++)
      for (int j = i + 1; j < 8; j++)
        if (ref[row * 8 + ids[j]] > ref[row * 8 + ids[i]] ||
            (ref[row * 8 + ids[j]] == ref[row * 8 + ids[i]] && ids[j] < ids[i])) {
          int v = ids[i];
          ids[i] = ids[j];
          ids[j] = v;
        }
    for (int k = 0; k < K; k++) ranked &= ids[k] == out[row * K + k];
  }
  check("top6 matches independently sorted double scores", ranked);
  float extremes[] = {-FLT_MAX, -1000, -80, -20, 0, 20, 80, 1000, FLT_MAX};
  bool stable_keys = true;
  for (size_t i = 0; i < sizeof extremes / sizeof *extremes; i++) {
    double refkey = op_key_double(extremes[i], 0), got = op_key(extremes[i], 0);
    stable_keys &= isfinite(got) && fabs(got - refkey) <= 1e-6 * fmax(1, fabs(refkey));
  }
  check("sqrt-softplus remains stable at extreme finite logits", stable_keys);
  check("forward does not learn within a group",
        !memcmp(before_b, p.b, sizeof before_b) && !memcmp(before_p, p.p, sizeof before_p));
  check("forward IDs valid and unique", op_valid_ids(out, 8, 3));
  double dp[16], db[32];
  for (int i = 0; i < 16; i++) dp[i] = p.p[i];
  for (int i = 0; i < 32; i++) db[i] = p.b[i];
  float frozen_phi[12];
  memcpy(frozen_phi, w.phi, sizeof frozen_phi);
  for (int row = 0; row < 3; row++) {
    double_update(4, 8, dp, db, frozen_phi + row * 4, labels + row * 8);
    check("group-delayed update accepted", op_update(&p, &w, frozen_phi + row * 4, labels + row * 8) == 1);
  }
  double bw = 0, pw = 0;
  for (int i = 0; i < 32; i++) bw = fmax(bw, fabs(db[i] - p.b[i]));
  for (int i = 0; i < 16; i++) pw = fmax(pw, fabs(dp[i] - p.p[i]));
  check("B update matches double reference", bw < 2e-5);
  check("P update matches double reference", pw < 2e-5);
  check("update changes future parameters", memcmp(before_b, p.b, sizeof before_b) != 0);
  float tie[8] = {0};
  op_top6(tie, 8, out);
  bool stable = true;
  for (int i = 0; i < K; i++) stable &= out[i] == i;
  check("ties select lower expert ID", stable);
  float saved = gp[0];
  gp[0] = NAN;
  check("nonfinite auxiliary input uses native fallback",
        op_forward(&p, &w, gp, gl, 3, native, out) == 1 && !memcmp(native, out, sizeof out));
  int invalid[18];
  memcpy(invalid, native, sizeof invalid);
  invalid[1] = invalid[0];
  for (int i = 0; i < 18; i++) out[i] = -7;
  check("invalid fallback IDs refused", op_forward(&p, &w, gp, gl, 3, invalid, out) == -1 && out[0] == -7);
  gp[0] = saved;
  memcpy(before_b, p.b, sizeof before_b);
  labels[0] = NAN;
  check("nonfinite labels rejected without B changes",
        op_update(&p, &w, frozen_phi, labels) == 0 && !memcmp(before_b, p.b, sizeof before_b));
  printf("{\"math_check\":{\"forward_max_abs\":%.9g,\"B_update_max_abs\":%.9g,\"P_update_max_abs\":%.9g}}\n", worst, bw,
         pw);
  op_free_pair(&p);
  op_free_work(&w);
  p = op_create_pair(384, stress_rank, 19);
  w = op_create_work(384, stress_rank);
  float *zero = op_allocate((size_t)stress_rank * 4), *repeated = op_allocate((size_t)stress_rank * 4),
        *target = op_allocate(384 * 4);
  for (int i = 0; i < stress_rank * 384; i++) p.b[i] = .001f;
  p.active = 1;
  for (int i = 0; i < stress_rank; i++) repeated[i] = .5f + (i % 7) * .03f;
  for (int i = 0; i < 384; i++) target[i] = .1f;
  for (int i = 0; i < 10000; i++) op_update(&p, &w, zero, target);
  bool retained = true;
  for (int i = 0; i < stress_rank * 384; i++) retained &= p.b[i] == .001f;
  check("dead-feature guard retains B", retained);
  check("dead-feature covariance is bounded",
        p.covariance_resets > 0 && op_finite_array(p.p, (size_t)stress_rank * stress_rank));
  for (int i = 0; i < 10000; i++) op_update(&p, &w, repeated, target);
  check("repetitive-feature state stays finite",
        op_finite_array(p.b, (size_t)stress_rank * 384) && op_finite_array(p.p, (size_t)stress_rank * stress_rank));
  printf("{\"numeric_stress\":{\"rows\":20000,\"rank\":%d,\"covariance_limit\":%.0f,\"covariance_resets\":%llu,"
         "\"rejected_updates\":%llu,\"B_retained_during_dead_features\":%s}}\n",
         stress_rank, (double)COV_LIMIT, (unsigned long long)p.covariance_resets,
         (unsigned long long)p.rejected_updates, retained ? "true" : "false");
  free(zero);
  free(repeated);
  free(target);
  op_free_pair(&p);
  op_free_work(&w);
}

static void write_all(int fd, const void *bytes, size_t n) {
  while (n) {
    ssize_t k = write(fd, bytes, n);
    if (k < 0 && errno == EINTR) continue;
    if (k <= 0) op_die("A export pipe failed");
    bytes = (const char *)bytes + k;
    n -= (size_t)k;
  }
}
static void benchmark(int rank, int pairs, int milliseconds, int export_fd) {
  Pair *p = op_allocate((size_t)pairs * sizeof *p);
  Work w = op_create_work(384, rank);
  uint64_t seed = 191;
  size_t inputs = (size_t)pairs * MAX_ROWS * 384, phis = (size_t)pairs * MAX_ROWS * rank;
  float *gp = op_allocate(inputs * 4), *gl = op_allocate(inputs * 4), *labels = op_allocate(inputs * 4),
        *phi = op_allocate(phis * 4);
  int *native = op_allocate((size_t)pairs * MAX_ROWS * K * sizeof(int)), out[MAX_ROWS * K];
  for (int q = 0; q < pairs; q++) {
    p[q] = op_create_pair(384, rank, UINT64_C(921) + q * UINT64_C(0x9e3779b97f4a7c15));
    for (int i = 0; i < rank * 384; i++) p[q].b[i] = op_gaussian(&seed) * .01f;
    p[q].active = 1;
    for (int i = 0; i < MAX_ROWS * 384; i++) {
      size_t at = (size_t)q * MAX_ROWS * 384 + i;
      gp[at] = op_gaussian(&seed);
      gl[at] = op_gaussian(&seed);
      labels[at] = op_gaussian(&seed) * .1f;
    }
    native_ids(&p[q], gp + (size_t)q * MAX_ROWS * 384, MAX_ROWS, native + q * MAX_ROWS * K);
    if (op_forward(&p[q], &w, gp + (size_t)q * MAX_ROWS * 384, gl + (size_t)q * MAX_ROWS * 384, MAX_ROWS,
                   native + q * MAX_ROWS * K, out))
      op_die("benchmark setup forward failed");
    memcpy(phi + (size_t)q * MAX_ROWS * rank, w.phi, (size_t)MAX_ROWS * rank * 4);
  }
  if (export_fd >= 0) {
    uint32_t header[4] = {(uint32_t)pairs, 768, (uint32_t)rank, 4};
    write_all(export_fd, "OPRA0001", 8);
    write_all(export_fd, header, sizeof header);
    for (int q = 0; q < pairs; q++) write_all(export_fd, p[q].a, (size_t)768 * rank * 4);
  }
  size_t state = (size_t)pairs * (768 * rank + rank * 384 + rank * rank + 384) * 4;
  printf("{\"configuration\":{\"experts\":384,\"features\":768,\"rank\":%d,\"pairs\":%d,\"blas_threads\":1,\"state_"
         "bytes\":%zu,\"B_snapshot_copy_bytes\":%zu,\"forgetting\":0.995,\"ridge\":16,\"rng\":\"xorshift64star_"
         "BoxMuller_not_PyTorch\",\"normalization\":\"FP64_mean_RMS_then_FP32_features\",\"covariance_update\":"
         "\"symmetric_FP64_intermediate_then_FP32\"}}\n",
         rank, pairs, state, (size_t)pairs * rank * 384 * 4);
  for (int m = 1; m <= 3; m += 2)
    for (int all = 0; all < 2; all++)
      for (int updating = 0; updating < 2; updating++) {
        int count = all ? pairs : 1;
        double durations[5], sum = 0;
        uint64_t total_iterations = 0;
        for (int sample = 0; sample < 5; sample++) {
          double start = op_now(), elapsed;
          uint64_t iterations = 0;
          do {
            for (int q = 0; q < count; q++) {
              size_t xoff = (size_t)q * MAX_ROWS * 384, hoff = (size_t)q * MAX_ROWS * rank;
              if (updating) {
                for (int row = 0; row < m; row++)
                  op_update(&p[q], &w, phi + hoff + row * rank, labels + xoff + row * 384);
                sink += p[q].b[0];
              } else {
                if (op_forward(&p[q], &w, gp + xoff, gl + xoff, m, native + q * MAX_ROWS * K, out))
                  op_die("benchmark forward failed");
                sink += out[0];
              }
            }
            iterations++;
            elapsed = op_now() - start;
          } while (elapsed < milliseconds / 5000.0);
          durations[sample] = elapsed * 1000 / iterations;
          sum += elapsed;
          total_iterations += iterations;
        }
        for (int i = 0; i < 5; i++)
          for (int j = i + 1; j < 5; j++)
            if (durations[j] < durations[i]) {
              double v = durations[i];
              durations[i] = durations[j];
              durations[j] = v;
            }
        printf("{\"benchmark\":{\"operation\":\"%s\",\"rows\":%d,\"pairs\":%d,\"iterations\":%llu,\"mean_pass_ms\":%."
               "6f,\"median_pass_ms\":%.6f,\"median_pair_us\":%.6f}}\n",
               updating ? "RLS_update" : "forward_active", m, count, (unsigned long long)total_iterations,
               sum * 1000 / total_iterations, durations[2], durations[2] * 1000 / count);
        fflush(stdout);
      }
  uint64_t resets = 0, rejected = 0;
  for (int q = 0; q < pairs; q++) {
    resets += p[q].covariance_resets;
    rejected += p[q].rejected_updates;
    op_free_pair(&p[q]);
  }
  printf("{\"benchmark_guards\":{\"covariance_resets\":%llu,\"rejected_updates\":%llu}}\n", (unsigned long long)resets,
         (unsigned long long)rejected);
  free(p);
  free(gp);
  free(gl);
  free(labels);
  free(phi);
  free(native);
  op_free_work(&w);
}
int main(int argc, char **argv) {
  int rank = 64, pairs = 77, ms = 100, fd = -1;
  for (int i = 1; i < argc; i += 2) {
    if (i + 1 == argc) op_die("options require values");
    int v = atoi(argv[i + 1]);
    if (!strcmp(argv[i], "--rank")) rank = v;
    else if (!strcmp(argv[i], "--pairs")) pairs = v;
    else if (!strcmp(argv[i], "--milliseconds")) ms = v;
    else if (!strcmp(argv[i], "--export-a-fd")) fd = v;
    else op_die("unknown option");
  }
  if ((rank != 64 && rank != 128) || pairs < 1 || pairs > 77 || ms < 20 || ms > 500 || (fd >= 0 && fd < 3))
    op_die("bounds: rank64/128, pairs1..77, milliseconds20..500, export fd>=3");
  struct rlimit cpu = {60, 60};
  setrlimit(RLIMIT_CPU, &cpu);
  alarm(60);
  if (BLASSetThreading(BLAS_THREADING_SINGLE_THREADED)) op_die("cannot set Accelerate single-thread mode");
  validate(rank);
  printf("{\"checks\":%d,\"failures\":%d}\n", checks, failures);
  fflush(stdout);
  if (failures) return 1;
  benchmark(rank, pairs, ms, fd);
  puts("{\"scope\":\"CPU math only; includes normalization, keys and top6; excludes GPU snapshots, scheduling, I/O, "
       "publication and live inference\"}");
  return 0;
}
