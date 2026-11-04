#include "context_lookup.h"

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

ContextDraft context_lookup_propose_consensus(const ContextLookup *s, int max_drafts) {
  ContextDraft out = lookup_match(s, max_drafts, s && s->misses >= 2 ? 16 : LONG_KEY, 0);
  if (out.count || !s || s->misses >= 2 || s->count < SHORT_KEY || max_drafts < 1) return out;
  const int32_t *history = s->storage, *previous = history + 2 * s->capacity;
  const int32_t *heads = history + 3 * s->capacity + s->buckets;
  int end = s->count - 1;
  for (int i = 0; i < SHORT_KEY; i++) if (history[end - i] < 0) return out;
  int sources[CONTEXT_LOOKUP_SEARCH], count = 0, checked = 0;
  int at = heads[bucket(s, history, end, SHORT_KEY)];
  if (at == end) at = previous[at];
  for (; at >= 0 && checked < CONTEXT_LOOKUP_SEARCH; at = previous[at]) {
    checked++, out.examined++;
    if (!memcmp(history + at - SHORT_KEY + 1, history + end - SHORT_KEY + 1, SHORT_KEY * sizeof *history))
      sources[count++] = at + 1;
  }
  int total = count;
  if (max_drafts > 4) max_drafts = 4;
  for (int depth = 0; count >= 2 && depth < max_drafts; depth++) {
    int32_t values[CONTEXT_LOOKUP_SEARCH];
    int votes[CONTEXT_LOOKUP_SEARCH] = {0}, distinct = 0, best = -1;
    for (int i = 0; i < count; i++) {
      int pos = sources[i] + depth;
      if (pos >= s->count || history[pos] < 0) continue;
      int k = 0;
      while (k < distinct && values[k] != history[pos]) k++;
      if (k == distinct) values[distinct++] = history[pos];
      votes[k]++;
    }
    for (int k = 0; k < distinct; k++)
      if (best < 0 || votes[k] > votes[best] || (votes[k] == votes[best] && values[k] < values[best])) best = k;
    // Support is measured against the original candidates for the whole
    // prefix, not renormalized after each token.
    if (best < 0 || votes[best] < 2 || 5 * votes[best] < 4 * total) break;
    out.tokens[out.count++] = values[best];
    int keep = 0;
    for (int i = 0; i < count; i++)
      if (sources[i] + depth < s->count && history[sources[i] + depth] == values[best]) sources[keep++] = sources[i];
    count = keep, out.source = sources[0], out.matched = SHORT_KEY;
  }
  return out;
}

int context_lookup_feedback(ContextLookup *s, int offered, int accepted) {
  if (!s || offered < 0 || offered > CONTEXT_LOOKUP_DRAFTS || accepted < 0 || accepted > offered) return 0;
  if (!offered) return 1;
  s->hot = accepted == offered;
  s->misses = accepted ? 0 : s->misses < 2 ? s->misses + 1 : 2;
  return 1;
}
