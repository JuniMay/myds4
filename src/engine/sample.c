// Target-model sampling shared by CLI, server and DSpark verification.
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "myds4.h"

int sampler_valid(const Sampler *s) {
  return isfinite(s->temp) && s->temp >= 0 && isfinite(s->top_p) && s->top_p >= 0 && s->top_p <= 1 && s->top_k >= 0;
}

void sampler_seed(Sampler *s, uint64_t seed) {
  for (int i = 0; i < 3; i++) s->seed[i] = (unsigned short)(seed >> (16 * i));
}

int sample_argmax(const float *logits, int count) {
  int best = 0;
  for (int i = 0; i < count; i++) {
    if (logits[i] > logits[best]) best = i;
  }
  return best;
}

// The workspace grows to the largest vocabulary sampled (any model's).
static double *probability, *target_probability;
static int *order, room;

static void make_room(int count) {
  if (count <= room) return;
  free(probability), free(target_probability), free(order);
  probability = xmalloc((size_t)count * sizeof *probability);
  target_probability = xmalloc((size_t)count * sizeof *target_probability);
  order = xmalloc((size_t)count * sizeof *order);
  room = count;
}

static int by_probability(const void *a, const void *b) {
  int i = *(const int *)a, j = *(const int *)b;
  if (probability[i] != probability[j]) return probability[i] > probability[j] ? -1 : 1;
  return (i > j) - (i < j);
}

// Whether token a ranks before token b: the higher logit, then the lower id.
static int ranks_before(const float *logits, int a, int b) {
  return logits[a] > logits[b] || (logits[a] == logits[b] && a < b);
}

typedef struct { int best, count, ranked; double sum; } Distribution;

// Keep the requested top-k and top-p support; probabilities remain unnormalized.
static Distribution distribution(const float *logits, int count, const Sampler *s) {
  int best = sample_argmax(logits, count);
  make_room(count);
  int n = count, ranked = 0;
  double sum = 0;
  if (s->top_k > 0 && s->top_k < count) {
    int k = s->top_k;
    n = 0;
    for (int i = 0; i < count; i++) {
      if (n == k && !ranks_before(logits, i, order[k - 1])) continue;
      int at = n < k ? n++ : k - 1;
      for (; at > 0 && ranks_before(logits, i, order[at - 1]); at--) order[at] = order[at - 1];
      order[at] = i;
    }
    for (int i = 0; i < n; i++) {
      int t = order[i];
      probability[t] = exp(((double)logits[t] - logits[best]) / s->temp);
      sum += probability[t];
    }
    ranked = 1;
  } else {
    for (int i = 0; i < count; i++) {
      probability[i] = exp(((double)logits[i] - logits[best]) / s->temp);
      sum += probability[i];
    }
    if (s->top_p < 1) {
      n = 0;
      for (int i = 0; i < count; i++) if (probability[i] > 0) order[n++] = i;
      qsort(order, n, sizeof *order, by_probability);
      ranked = 1;
    }
  }
  if (ranked) {
    double kept = 0, threshold = s->top_p * sum;
    int k = 0;
    do kept += probability[order[k++]];
    while (k < n && kept < threshold);
    n = k;
    sum = kept;
  }
  return (Distribution){best, n, ranked, sum};
}

static int draw(Distribution d, Sampler *s) {
  double r = erand48(s->seed) * d.sum;
  int last = d.best;
  for (int i = 0; i < d.count; i++) {
    int token = d.ranked ? order[i] : i;
    if (probability[token] == 0) continue;
    last = token;
    if (r < probability[token]) return token;
    r -= probability[token];
  }
  return last;
}

static void copy_distribution(Distribution d, int count, double *out) {
  if (d.ranked) {
    memset(out, 0, (size_t)count * sizeof *out);
    for (int i = 0; i < d.count; i++) out[order[i]] = probability[order[i]] / d.sum;
  } else for (int i = 0; i < count; i++) out[i] = probability[i] / d.sum;
}

int sample_token(const float *logits, int count, Sampler *s) {
  if (s->temp == 0) return sample_argmax(logits, count);
  return draw(distribution(logits, count, s), s);
}

int sample_proposal(const float *logits, int count, Sampler *s, double *out) {
  if (s->temp == 0) return sample_argmax(logits, count);
  Distribution d = distribution(logits, count, s);
  copy_distribution(d, count, out);
  return draw(d, s);
}

// Draft sources supply deterministic proposals. Sampling from the target and
// keeping a matching prefix preserves its distribution and random stream.
// A mismatch is already the next token: never draw it a second time.
int sample_verify(const float *logits, int count, const int *drafts, int ndraft, int (*stop)(int token), Sampler *s,
                  int *accepted) {
  int n = 0;
  for (;;) {
    int next = sample_token(logits + (size_t)n * count, count, s);
    if (n == ndraft || (stop && stop(next)) || next != drafts[n]) {
      *accepted = n;
      return next;
    }
    n++;
  }
}

// Accept q proposals with min(1, p/q); rejection samples normalized (p-q)+.
// The target distribution is unchanged. Proposal draws use their own RNG draws.
int sample_verify_probabilities(const float *logits, int count, const int *drafts, int ndraft,
                                const double *q, int (*stop)(int token), Sampler *s, int *accepted) {
  if (!q || s->temp == 0) return sample_verify(logits, count, drafts, ndraft, stop, s, accepted);
  for (int i = 0; i < ndraft; i++) {
    Distribution p = distribution(logits + (size_t)i * count, count, s);
    copy_distribution(p, count, target_probability);
    const double *proposal = q + (size_t)i * count;
    int token = drafts[i];
    if (erand48(s->seed) * proposal[token] < target_probability[token]) {
      if (stop && stop(token)) { *accepted = i; return token; }
      continue;
    }
    double sum = 0;
    for (int t = 0; t < count; t++) {
      probability[t] = fmax(0, target_probability[t] - proposal[t]);
      sum += probability[t];
    }
    *accepted = i;
    return draw((Distribution){p.best, count, 0, sum}, s);
  }
  *accepted = ndraft;
  return sample_token(logits + (size_t)ndraft * count, count, s);
}
