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
static int *order, *order_moved, room;
static uint64_t *rank_key, *rank_key_moved;

static void make_room(int count) {
  if (count <= room) return;
  free(probability), free(target_probability), free(order), free(order_moved), free(rank_key), free(rank_key_moved);
  probability = xmalloc((size_t)count * sizeof *probability);
  target_probability = xmalloc((size_t)count * sizeof *target_probability);
  order = xmalloc((size_t)count * sizeof *order);
  order_moved = xmalloc((size_t)count * sizeof *order_moved);
  rank_key = xmalloc((size_t)count * sizeof *rank_key);
  rank_key_moved = xmalloc((size_t)count * sizeof *rank_key_moved);
  room = count;
}

// Whether token a ranks before token b: the higher logit, then the lower id.
static int ranks_before(const float *logits, int a, int b) {
  return logits[a] > logits[b] || (logits[a] == logits[b] && a < b);
}

typedef struct { int best, count, ranked; double sum; } Distribution;

// Sorts order[0..n), tokens with positive probabilities in increasing id
// order, as a full sort ranks them (the higher probability, then the lower
// id): a stable radix sort of the probabilities' bits, eight at a time from
// the lowest, keeps equal probabilities in id order. A positive double's bits
// order it as its value does; their complement orders it from the highest.
static void rank_by_probability(int n) {
  int *ids = order, *moved = order_moved;
  uint64_t *key = rank_key, *key_moved = rank_key_moved;
  for (int i = 0; i < n; i++) {
    uint64_t bits;
    memcpy(&bits, &probability[ids[i]], sizeof bits);
    key[i] = ~bits;
  }
  for (int shift = 0; shift < 64 && n > 1; shift += 8) {
    int start[257] = {0};
    for (int i = 0; i < n; i++) start[(key[i] >> shift & 255) + 1]++;
    if (start[(key[0] >> shift & 255) + 1] == n) continue;  // one value of this digit: nothing moves
    for (int d = 0; d < 256; d++) start[d + 1] += start[d];
    for (int i = 0; i < n; i++) {
      int at = start[key[i] >> shift & 255]++;
      key_moved[at] = key[i], moved[at] = ids[i];
    }
    uint64_t *k = key;
    key = key_moved, key_moved = k;
    int *m = ids;
    ids = moved, moved = m;
  }
  if (ids != order) memcpy(order, ids, (size_t)n * sizeof *order);
}

// The tokens that top-p may keep (probability[] holds every token's), in
// order[] as a full sort of every token with a positive probability ranks them
// (the higher probability, then the lower id); returns their count. Every token
// below a probability bound ranks after every token at or above it, so sorting
// only those at or above the bound gives the full sort's first tokens. The
// bound is the power of two at which the tokens' probabilities, summed by
// binary order of magnitude, first reach the threshold, lowered by one more
// order of magnitude because the nucleus sums them in another order. If the
// sorted tokens still do not reach the threshold, every token is sorted.
static int nucleus(int count, double threshold) {
  enum { ORDERS = 64 };
  double mass[ORDERS] = {0};
  for (int i = 0; i < count; i++) {
    uint64_t bits;
    memcpy(&bits, &probability[i], sizeof bits);
    int e = 1023 - (int)(bits >> 52 & 0x7ff);  // probability in [2^-e, 2^(1-e)); 1023: zero or subnormal
    mass[e <= 0 ? 0 : e < ORDERS ? e : ORDERS - 1] += probability[i];
  }
  double held = 0;
  int last = ORDERS;
  for (int e = 0; e < ORDERS && last == ORDERS; e++)
    if ((held += mass[e]) >= threshold) last = e;
  if (last + 1 < ORDERS - 1) {
    double bound = ldexp(1, -(last + 1));
    int n = 0;
    for (int i = 0; i < count; i++)
      if (probability[i] >= bound) order[n++] = i;
    rank_by_probability(n);
    double kept = 0;
    int k = 0;
    do kept += probability[order[k++]];
    while (k < n && kept < threshold);
    if (kept >= threshold) return n;
  }
  int n = 0;
  for (int i = 0; i < count; i++)
    if (probability[i] > 0) order[n++] = i;
  rank_by_probability(n);
  return n;
}

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
      n = nucleus(count, s->top_p * sum);
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
