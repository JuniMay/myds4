// Shared experimental math, derived from the checked standalone CPU probe.
#ifndef MYDS4_ONLINE_PREDICTOR_MATH_H
#define MYDS4_ONLINE_PREDICTOR_MATH_H
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#include <errno.h>
#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>
#include <unistd.h>

#define OP_K 6
#define OP_MAX_ROWS 6
#define OP_COV_LIMIT 1000000.0f
typedef struct {
  int e, f, r, active;
  float *a, *b, *p, *bias;
  uint64_t updates, covariance_resets, rejected_updates, fallback_rows;
} OpPair;
typedef struct {
  float *features, *phi, *keys, *scores, *vector, *error, *next_b, *next_p;
  int *ids;
} OpWork;
static inline void op_die(const char *s) {
  fprintf(stderr, "%s\n", s);
  exit(1);
}
static inline void *op_allocate(size_t n) {
  if (n > 128u * 1024 * 1024) op_die("individual allocation exceeds128 MiB");
  void *p = calloc(1, n ? n : 1);
  if (!p) op_die("allocation failed");
  return p;
}
static inline double op_now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC_RAW, &t);
  return t.tv_sec + t.tv_nsec * 1e-9;
}
static inline uint64_t op_random64(uint64_t *s) {
  *s ^= *s >> 12;
  *s ^= *s << 25;
  *s ^= *s >> 27;
  return *s * UINT64_C(2685821657736338717);
}
static inline double op_uniform(uint64_t *s) { return ((op_random64(s) >> 11) + .5) * 0x1p-53; }
static inline float op_gaussian(uint64_t *s) {
  return (float)(sqrt(-2 * log(op_uniform(s))) * cos(6.2831853071795864769 * op_uniform(s)));
}
static inline bool op_finite_array(const float *p, size_t n) {
  for (size_t i = 0; i < n; i++)
    if (!isfinite(p[i])) return false;
  return true;
}
static inline void op_covariance_reset(OpPair *p) {
  memset(p->p, 0, (size_t)p->r * p->r * 4);
  for (int j = 0; j < p->r; j++) p->p[j * p->r + j] = 1.0f / 16;
  p->covariance_resets++;
}
static inline OpPair op_create_pair(int e, int r, uint64_t seed) {
  OpPair p = {.e = e, .f = 2 * e, .r = r};
  p.a = op_allocate((size_t)p.f * r * 4);
  p.b = op_allocate((size_t)r * e * 4);
  p.p = op_allocate((size_t)r * r * 4);
  p.bias = op_allocate((size_t)e * 4);
  for (int i = 0; i < p.f * r; i++) p.a[i] = op_gaussian(&seed) / sqrtf((float)p.f);
  op_covariance_reset(&p);
  p.covariance_resets = 0;
  return p;
}
static inline void op_free_pair(OpPair *p) {
  free(p->a);
  free(p->b);
  free(p->p);
  free(p->bias);
}
static inline OpWork op_create_work(int e, int r) {
  OpWork w = {0};
  w.features = op_allocate((size_t)OP_MAX_ROWS * 2 * e * 4);
  w.phi = op_allocate((size_t)OP_MAX_ROWS * r * 4);
  w.keys = op_allocate((size_t)OP_MAX_ROWS * e * 4);
  w.scores = op_allocate((size_t)OP_MAX_ROWS * e * 4);
  w.vector = op_allocate((size_t)r * 4);
  w.error = op_allocate((size_t)e * 4);
  w.next_b = op_allocate((size_t)r * e * 4);
  w.next_p = op_allocate((size_t)r * r * 4);
  w.ids = op_allocate(OP_MAX_ROWS * OP_K * sizeof(int));
  return w;
}
static inline void op_free_work(OpWork *w) {
  free(w->features);
  free(w->phi);
  free(w->keys);
  free(w->scores);
  free(w->vector);
  free(w->error);
  free(w->next_b);
  free(w->next_p);
  free(w->ids);
}
static inline float op_key(float v, float bias) { return sqrtf(fmaxf(v, 0) + log1pf(expf(-fabsf(v)))) + bias; }
static inline double op_key_double(double v, double bias) { return sqrt(fmax(v, 0) + log1p(exp(-fabs(v)))) + bias; }
static inline void op_top6(const float *scores, int e, int *ids) {
  for (int k = 0; k < OP_K; k++) ids[k] = -1;
  for (int j = 0; j < e; j++) {
    int at = OP_K - 1;
    if (ids[at] >= 0 && scores[j] <= scores[ids[at]]) continue;
    while (at > 0 && (ids[at - 1] < 0 || scores[j] > scores[ids[at - 1]])) {
      ids[at] = ids[at - 1];
      at--;
    }
    ids[at] = j; // Ascending input IDs retain the lower ID on a tied score.
  }
}
static inline bool op_valid_ids(const int *ids, int e, int m) {
  for (int row = 0; row < m; row++)
    for (int k = 0; k < OP_K; k++) {
      int v = ids[row * OP_K + k];
      if (v < 0 || v >= e) return false;
      for (int j = 0; j < k; j++)
        if (ids[row * OP_K + j] == v) return false;
    }
  return true;
}
static inline void op_normalize(const float *x, float *out, int n) {
  double mean = 0, ss = 0;
  for (int j = 0; j < n; j++) mean += x[j];
  mean /= n;
  for (int j = 0; j < n; j++) {
    double v = x[j] - mean;
    ss += v * v;
  }
  double inverse = 1 / sqrt(ss / n + 1e-6);
  for (int j = 0; j < n; j++) out[j] = (float)((x[j] - mean) * inverse);
}
static inline int op_fallback(OpPair *p, const int *native, int m, int *out) {
  if (!native || !op_valid_ids(native, p->e, m)) return -1;
  memcpy(out, native, (size_t)m * OP_K * sizeof(int));
  p->fallback_rows += m;
  return 1;
}
// Forward never mutates learned parameters. Native IDs are copied verbatim
// for B=0 and on invalid auxiliary input; invalid fallback IDs are refused.
static inline int op_forward(OpPair *p, OpWork *w, const float *gp, const float *gl, int m, const int *native,
                             int *out) {
  if (m < 1 || m > OP_MAX_ROWS || !native || !op_valid_ids(native, p->e, m)) return -1;
  if (!op_finite_array(gp, (size_t)m * p->e) || !op_finite_array(gl, (size_t)m * p->e) ||
      !op_finite_array(p->bias, p->e))
    return op_fallback(p, native, m, out);
  for (int row = 0; row < m; row++) {
    op_normalize(gp + row * p->e, w->features + row * p->f, p->e);
    op_normalize(gl + row * p->e, w->features + row * p->f + p->e, p->e);
    for (int j = 0; j < p->e; j++) w->keys[row * p->e + j] = op_key(gp[row * p->e + j], p->bias[j]);
  }
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, p->r, p->f, 1, w->features, p->f, p->a, p->r, 0, w->phi,
              p->r);
  if (!op_finite_array(w->phi, (size_t)m * p->r)) return op_fallback(p, native, m, out);
  for (int j = 0; j < m * p->r; j++) w->phi[j] = fmaxf(w->phi[j], 0);
  memcpy(w->scores, w->keys, (size_t)m * p->e * 4);
  if (p->active)
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, p->e, p->r, 1, w->phi, p->r, p->b, p->e, 1, w->scores,
                p->e);
  if (!op_finite_array(w->scores, (size_t)m * p->e)) return op_fallback(p, native, m, out);
  if (!p->active) {
    memcpy(out, native, (size_t)m * OP_K * sizeof(int));
    return 0;
  }
  for (int row = 0; row < m; row++) op_top6(w->scores + row * p->e, p->e, w->ids + row * OP_K);
  memcpy(out, w->ids, (size_t)m * OP_K * sizeof(int));
  return 0;
}

// A caller must retain phi until all labels in its pass are available, then
// update in row order. Candidate parameters are validated before publication.
typedef struct { bool valid, nonzero; } OpWeightsCheck;

static inline OpWeightsCheck op_update_weights(const float *restrict b, const float *restrict error,
                                               const float *restrict vector, float *restrict next,
                                               int rows, int cols, double inv_den) {
  OpWeightsCheck check = {true, false};
  for (int i = 0; i < rows; i++) {
    float gain = (float)((double)vector[i] * inv_den);
    for (int j = 0; j < cols; j++) {
      float v = fmaf(gain, error[j], b[i * cols + j]);
      next[i * cols + j] = v;
      check.valid &= isfinite(v);
      check.nonzero |= v != 0;
    }
  }
  return check;
}

static inline int op_update(OpPair *p, OpWork *w, const float *phi, const float *target) {
  const float lambda = .995f;
  if (!op_finite_array(phi, p->r) || !op_finite_array(target, p->e)) {
    p->rejected_updates++;
    return 0;
  }
  bool bad = false;
  for (int i = 0; i < p->r; i++)
    bad |= !isfinite(p->p[i * p->r + i]) || p->p[i * p->r + i] <= 0 || p->p[i * p->r + i] > OP_COV_LIMIT;
  if (bad) op_covariance_reset(p);
  cblas_sgemv(CblasRowMajor, CblasNoTrans, p->r, p->r, 1, p->p, p->r, phi, 1, 0, w->vector, 1);
  double denominator = lambda;
  for (int i = 0; i < p->r; i++) denominator += (double)phi[i] * w->vector[i];
  if (!isfinite(denominator) || denominator < lambda * .5) {
    op_covariance_reset(p);
    p->rejected_updates++;
    return 0;
  }
  cblas_sgemv(CblasRowMajor, CblasTrans, p->r, p->e, 1, p->b, p->e, phi, 1, 0, w->error, 1);
  for (int j = 0; j < p->e; j++) w->error[j] = target[j] - w->error[j];
  if (!op_finite_array(w->error, p->e)) {
    p->rejected_updates++;
    return 0;
  }
  const double inv_den = 1.0 / denominator, inv_lambda = 1.0 / (double)lambda;
  OpWeightsCheck weights = op_update_weights(p->b, w->error, w->vector, w->next_b, p->r, p->e, inv_den);
  bool valid = weights.valid;
  for (int i = 0; i < p->r; i++) {
    for (int j = i; j < p->r; j++) {
      float v = (float)(((double)p->p[i * p->r + j] - (double)w->vector[i] * w->vector[j] * inv_den) * inv_lambda);
      w->next_p[i * p->r + j] = w->next_p[j * p->r + i] = v;
      valid &= isfinite(v) && fabsf(v) <= OP_COV_LIMIT && (i != j || v > 0);
    }
  }
  if (!valid) {
    op_covariance_reset(p);
    p->rejected_updates++;
    return 0;
  }
  memcpy(p->b, w->next_b, (size_t)p->r * p->e * 4);
  memcpy(p->p, w->next_p, (size_t)p->r * p->r * 4);
  p->active = weights.nonzero;
  p->updates++;
  return 1;
}

#endif
