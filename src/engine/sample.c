// Target-model sampling shared by CLI, server and DSpark verification.
#include <math.h>
#include <stdlib.h>

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
static double *probability;
static int *order, room;

static void make_room(int count) {
  if (count <= room) return;
  free(probability), free(order);
  probability = xmalloc((size_t)count * sizeof *probability);
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

// From the top_k highest-ranked tokens at the temperature, cut to the top_p
// nucleus of their own distribution in rank order (top-k, then top-p, as the
// reference samplers apply them).
static int sample_top_k(const float *logits, int count, Sampler *s, int best) {
  int k = s->top_k, n = 0;
  for (int i = 0; i < count; i++) {  // one pass, each kept token inserted in rank order
    if (n == k && !ranks_before(logits, i, order[k - 1])) continue;
    int at = n < k ? n++ : k - 1;
    for (; at > 0 && ranks_before(logits, i, order[at - 1]); at--) order[at] = order[at - 1];
    order[at] = i;
  }
  double sum = 0;
  for (int i = 0; i < n; i++) {
    int t = order[i];
    probability[t] = exp(((double)logits[t] - logits[best]) / s->temp);
    sum += probability[t];
  }
  double kept = 0, threshold = s->top_p * sum;
  int m = 0;
  do kept += probability[order[m++]];
  while (m < n && kept < threshold);
  double r = erand48(s->seed) * kept;
  int last = best;
  for (int i = 0; i < m; i++) {
    int token = order[i];
    if (probability[token] == 0) continue;
    last = token;
    if (r < probability[token]) return token;
    r -= probability[token];
  }
  return last;
}

int sample_token(const float *logits, int count, Sampler *s) {
  int best = sample_argmax(logits, count);
  if (s->temp == 0) return best;
  make_room(count);
  if (s->top_k > 0 && s->top_k < count) return sample_top_k(logits, count, s, best);
  double sum = 0;
  for (int i = 0; i < count; i++) {
    double p = exp(((double)logits[i] - logits[best]) / s->temp);
    probability[i] = p;
    sum += p;
  }
  int n = count;
  if (s->top_p < 1) {
    n = 0;
    for (int i = 0; i < count; i++)
      if (probability[i] > 0) order[n++] = i;
    qsort(order, n, sizeof *order, by_probability);
    double kept = 0, threshold = s->top_p * sum;
    int k = 0;
    do kept += probability[order[k++]];
    while (k < n && kept < threshold);
    n = k;
    sum = kept;
  }
  // top-p=1 uses the full distribution in token order, without sorting or
  // discarding small positive probabilities. Nucleus sampling uses rank order.
  double r = erand48(s->seed) * sum;
  int last = best;
  for (int i = 0; i < n; i++) {
    int token = s->top_p < 1 ? order[i] : i;
    if (probability[token] == 0) continue;
    last = token;
    if (r < probability[token]) return token;
    r -= probability[token];
  }
  return last; // Floating-point accumulation can leave a final rounding residue.
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
