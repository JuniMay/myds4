// Chen et al., "Accelerating Large Language Model Decoding with Speculative
// Sampling" (2023), modified rejection sampling. Target and proposal
// distributions include their temperature and nucleus filtering.
#include "speculative_sample.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

static const double *sorting;
static int order[VOCAB];
static double target[VOCAB], residual[VOCAB];

static int by_probability(const void *a, const void *b) {
  int i = *(const int *)a, j = *(const int *)b;
  if (sorting[i] != sorting[j]) return sorting[i] > sorting[j] ? -1 : 1;
  return (i > j) - (i < j);
}

void speculative_distribution(const float *logits, int count, const Sampler *s, double *p) {
  if (!sampler_valid(s)) die("invalid speculative sampler");
  int best = sample_argmax(logits, count);
  if (!s->temp) {
    memset(p, 0, count * sizeof *p);
    p[best] = 1;
    return;
  }
  double sum = 0;
  for (int i = 0; i < count; i++)
    sum += p[i] = logits[best] == INFINITY ? logits[i] == INFINITY : exp(((double)logits[i] - logits[best]) / s->temp);
  if (s->top_p < 1) {
    int n = 0;
    for (int i = 0; i < count; i++) if (p[i] > 0) order[n++] = i;
    sorting = p;
    qsort(order, n, sizeof *order, by_probability);
    double kept = 0, threshold = s->top_p * sum;
    int k = 0;
    do kept += p[order[k++]];
    while (k < n && kept < threshold);
    for (int i = k; i < n; i++) p[order[i]] = 0;
    sum = kept;
  }
  for (int i = 0; i < count; i++) p[i] /= sum;
}

int speculative_draw(const double *p, int count, double uniform) {
  if (count < 1 || count > VOCAB || !(uniform >= 0 && uniform < 1)) die("invalid speculative draw");
  int last = -1;
  for (int i = 0; i < count; i++) if (p[i] > 0) {
    last = i;
    if (uniform < p[i]) return i;
    uniform -= p[i];
  }
  if (last < 0) die("empty speculative distribution");
  return last;
}

double speculative_residual(const double *p, const double *q, int count, double *out) {
  double sum = 0;
  for (int i = 0; i < count; i++) sum += out[i] = fmax(0, p[i] - q[i]);
  if (sum > 0) for (int i = 0; i < count; i++) out[i] /= sum;
  return sum;
}

int speculative_verify(const float *logits, int count, const int *drafts, int ndraft,
                       const double *q, int eos, Sampler *s, int *accepted) {
  if (ndraft < 0 || ndraft > BLOCK || count < 1 || count > VOCAB) die("invalid rejection verification size");
  for (int r = 0; r < ndraft; r++) {
    speculative_distribution(logits + (size_t)r * count, count, s, target);
    const double *proposal = q + (size_t)r * count;
    int token = drafts[r];
    if (token < 0 || token >= count || !(proposal[token] > 0)) die("draft outside its proposal support");
    double u = erand48(s->seed);
    if (u * proposal[token] < target[token]) {
      if (token == eos) { *accepted = r; return eos; }
      continue;
    }
    double mass = speculative_residual(target, proposal, count, residual);
    if (!(mass > 0)) die("empty rejected-sample residual");
    *accepted = r;
    return speculative_draw(residual, count, erand48(s->seed));
  }
  *accepted = ndraft;
  return sample_token(logits + (size_t)ndraft * count, count, s);
}
