#include "draft.h"

#include <stdlib.h>
#include <string.h>

struct ContextLookup {
  int capacity, count, buckets, hot, misses;
  int32_t storage[]; // tokens, two previous-occurrence arrays, two hash tables
};

enum { LONG_KEY = 12, SHORT_KEY = 4 };

static uint32_t bucket(const ContextLookup *s, const int32_t *tokens, int end, int width) {
  uint64_t x = UINT64_C(0x9e3779b97f4a7c15);
  for (int i = end - width + 1; i <= end; i++)
    x ^= (uint64_t)(uint32_t)tokens[i] + UINT64_C(0x9e3779b97f4a7c15) + (x << 6) + (x >> 2);
  x ^= x >> 33;
  x *= UINT64_C(0xff51afd7ed558ccd);
  x ^= x >> 33;
  x *= UINT64_C(0xc4ceb9fe1a85ec53);
  x ^= x >> 33;
  return (uint32_t)x & (uint32_t)(s->buckets - 1);
}

ContextLookup *context_lookup_new(int capacity) {
  if (capacity < 1 || capacity > 1048576) return NULL;
  int buckets = 64;
  while (buckets < capacity && buckets < 65536) buckets *= 2;
  ContextLookup *s = malloc(sizeof *s + (size_t)(3 * capacity + 2 * buckets) * sizeof(int32_t));
  if (!s) return NULL;
  s->capacity = capacity;
  s->buckets = buckets;
  context_lookup_reset(s);
  return s;
}

void context_lookup_free(ContextLookup *s) { free(s); }

void context_lookup_reset(ContextLookup *s) {
  s->count = s->hot = s->misses = 0;
  memset(s->storage + 3 * s->capacity, 0xFF, (size_t)2 * s->buckets * sizeof(int32_t));
}

int context_lookup_append(ContextLookup *s, const int32_t *tokens, int count) {
  if (!s || count < 0 || count > s->capacity - s->count || (count && !tokens)) return 0;
  int32_t *history = s->storage;
  for (int i = 0; i < count; i++) {
    int at = s->count++;
    history[at] = tokens[i];
    for (int index = 0; index < 2; index++) {
      int width = index ? SHORT_KEY : LONG_KEY, valid = at >= width - 1;
      int32_t *previous = history + (1 + index) * s->capacity;
      int32_t *heads = history + 3 * s->capacity + index * s->buckets;
      previous[at] = -1;
      // Negative IDs are image placeholders, not tokens that can be drafted.
      for (int j = 0; valid && j < width; j++) valid &= history[at - j] >= 0;
      if (valid) {
        uint32_t h = bucket(s, history, at, width);
        previous[at] = heads[h];
        heads[h] = at;
      }
    }
  }
  return 1;
}

static ContextDraft lookup_match(const ContextLookup *s, int max_drafts, int minimum, int allow_short) {
  ContextDraft out = {.source = -1};
  if (!s || s->count < SHORT_KEY || max_drafts <= 0) return out;
  if (max_drafts > CONTEXT_LOOKUP_DRAFTS) max_drafts = CONTEXT_LOOKUP_DRAFTS;
  const int32_t *history = s->storage;
  int end = s->count - 1;
  for (int index = 0; index < 2; index++) {
    // Search long suffixes first so frequent short endings cannot hide them.
    // The short index is only needed for the four-token fallback after a hit.
    if (index && (!allow_short || out.count)) break;
    int width = index ? SHORT_KEY : LONG_KEY, valid = s->count >= width;
    for (int j = 0; valid && j < width; j++) valid &= history[end - j] >= 0;
    if (!valid) continue;
    const int32_t *previous = history + (1 + index) * s->capacity;
    const int32_t *heads = history + 3 * s->capacity + index * s->buckets;
    int at = heads[bucket(s, history, end, width)], examined = 0;
    if (at == end) at = previous[at];
    for (; at >= 0 && examined < CONTEXT_LOOKUP_SEARCH; at = previous[at]) {
      examined++, out.examined++;
      if (memcmp(history + at - width + 1, history + end - width + 1, (size_t)width * sizeof *history)) continue;
      int matched = width;
      while (matched < CONTEXT_LOOKUP_MATCH && matched <= at && history[at - matched] >= 0 &&
             history[at - matched] == history[end - matched]) matched++;
      if (matched < minimum || matched < out.matched) continue;
      int count = 0;
      while (count < max_drafts && at + 1 + count < s->count && history[at + 1 + count] >= 0) count++;
      if (!count || (matched == out.matched && count <= out.count)) continue;
      out.matched = matched;
      out.source = at + 1;
      out.count = count;
      memcpy(out.tokens, history + out.source, (size_t)count * sizeof *out.tokens);
    }
  }
  return out;
}

ContextDraft context_lookup_propose(const ContextLookup *s, int max_drafts) {
  int minimum = s && s->hot ? SHORT_KEY : s && s->misses >= 2 ? 16 : LONG_KEY;
  return lookup_match(s, max_drafts, minimum, minimum == SHORT_KEY);
}

int context_lookup_feedback(ContextLookup *s, int offered, int accepted) {
  if (!s || offered < 0 || offered > CONTEXT_LOOKUP_DRAFTS || accepted < 0 || accepted > offered) return 0;
  if (!offered) return 1;
  s->hot = accepted == offered;
  s->misses = accepted ? 0 : s->misses < 2 ? s->misses + 1 : 2;
  return 1;
}

void draft_choice_init(DraftChoice *c) { memset(c, 0, sizeof *c); }

// A running mean of the first DRAFT_CHOICE_MEMORY observations, then an
// exponential one with that weight, so that a figure starts at its first
// value and follows changes of the text.
static void running(double *value, int *n, double x) {
  if (*n < DRAFT_CHOICE_MEMORY) ++*n;
  *value += (x - *value) / *n;
}

int draft_choice_count(DraftChoice *c, int most) {
  if (most > DRAFT_CHOICE_MAX) most = DRAFT_CHOICE_MAX;
  if (most <= 0) return 0;
  for (int round = 1; round <= DRAFT_CHOICE_WARM; round++)
    for (int k = most; k >= 0; k--)
      if (c->seconds_n[k] < round) return k;
  int best = 0;
  double tokens = 1, rate = 1 / c->seconds[0];
  for (int k = 1; k <= most; k++) {
    tokens += c->kept[k];
    if (tokens / c->seconds[k] > rate) best = k, rate = tokens / c->seconds[k];
  }
  if (++c->passes % DRAFT_CHOICE_EXPLORE) return best;
  int up = (c->passes / DRAFT_CHOICE_EXPLORE) % 2;  // alternately the count above and the one below
  return up ? (best < most ? best + 1 : best - 1) : (best > 0 ? best - 1 : best + 1);
}

void draft_choice_feedback(DraftChoice *c, int count, int kept, double seconds) {
  if (count < 0 || count > DRAFT_CHOICE_MAX || kept < 0 || kept > count || !(seconds > 0)) return;
  running(&c->seconds[count], &c->seconds_n[count], seconds);
  for (int j = 1; j <= count; j++) running(&c->kept[j], &c->kept_n[j], kept >= j);
}
