// Target-model sampling shared by CLI, server and DSpark verification.
#include <math.h>
#include <stdlib.h>

#include "myds4.h"

int sampler_valid(const Sampler *s) {
  return isfinite(s->temp) && s->temp >= 0 && isfinite(s->top_p) && s->top_p >= 0 && s->top_p <= 1;
}

void sampler_seed(Sampler *s, uint64_t seed) {
  for (int i = 0; i < 3; i++) s->seed[i] = (unsigned short)(seed >> (16 * i));
}

int sample_argmax(const float *logits, int count) {
  if (count < 1 || count > VOCAB) die("invalid sampling vocabulary size");
  int best = 0;
  for (int i = 0; i < count; i++) {
    if (isnan(logits[i])) die("cannot sample NaN logits");
    if (logits[i] > logits[best]) best = i;
  }
  if (logits[best] == -INFINITY) die("cannot sample an empty distribution");
  return best;
}

static double probability[VOCAB];
static int order[VOCAB];

static int by_probability(const void *a, const void *b) {
  int i = *(const int *)a, j = *(const int *)b;
  if (probability[i] != probability[j]) return probability[i] > probability[j] ? -1 : 1;
  return (i > j) - (i < j);
}

int sample_token(const float *logits, int count, Sampler *s) {
  if (!sampler_valid(s)) die("temperature must be finite and nonnegative; top-p must be in [0, 1]");
  int best = sample_argmax(logits, count);
  if (s->temp == 0) return best;
  double sum = 0;
  for (int i = 0; i < count; i++) {
    double p =
        logits[best] == INFINITY ? (logits[i] == INFINITY ? 1 : 0) : exp(((double)logits[i] - logits[best]) / s->temp);
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

// DSpark supplies deterministic proposals. Sampling from the target and
// keeping a matching prefix preserves its distribution and random stream.
// A mismatch is already the next token: never draw it a second time.
int sample_verify(const float *logits, int count, const int *drafts, int ndraft, int eos, Sampler *s, int *accepted) {
  if (ndraft < 0 || ndraft > BLOCK) die("invalid draft verification size");
  int n = 0;
  for (;;) {
    int next = sample_token(logits + (size_t)n * count, count, s);
    if (n == ndraft || next == eos || next != drafts[n]) {
      *accepted = n;
      return next;
    }
    n++;
  }
}
