// Check indexed proposals against a direct scan of committed token history.
#include "../../src/model/draft.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int checks, failures;
static void expect(const char *name, int ok) {
  checks++;
  if (!ok) failures++, fprintf(stderr, "FAIL: %s\n", name);
}
static int pair(int32_t *tokens, int match) {
  for (int i = 0; i < match; i++) tokens[i] = 100 + i;
  for (int i = 0; i < 15; i++) tokens[match + i] = 1000 + i;
  tokens[match + 15] = 9999;
  memcpy(tokens + match + 16, tokens, (size_t)match * sizeof *tokens);
  return 2 * match + 16;
}

static ContextDraft direct(const int32_t *tokens, int n, int limit) {
  ContextDraft out = {.source = -1};
  if (n < 2 || tokens[n - 1] < 0 || tokens[n - 2] < 0) return out;
  if (limit > CONTEXT_LOOKUP_DRAFTS) limit = CONTEXT_LOOKUP_DRAFTS;
  for (int end = n - 2; end >= 1; end--) {
    int match = 0;
    while (match < 64 && match <= end && tokens[end - match] >= 0 &&
           tokens[end - match] == tokens[n - 1 - match]) match++;
    if (match < 12 || match < out.matched) continue;
    int count = 0;
    while (count < limit && end + 1 + count < n && tokens[end + 1 + count] >= 0) count++;
    if (!count || (match == out.matched && count <= out.count)) continue;
    out.matched = match, out.count = count, out.source = end + 1;
    memcpy(out.tokens, tokens + out.source, (size_t)count * sizeof *tokens);
  }
  return out;
}
static int same(ContextDraft a, ContextDraft b) {
  return a.count == b.count && a.matched == b.matched && a.source == b.source &&
         !memcmp(a.tokens, b.tokens, (size_t)a.count * sizeof *a.tokens);
}

int main(void) {
  alarm(10);
  int32_t tokens[512];
  ContextLookup *s = context_lookup_new(512);
  if (!s) return 2;
  expect("reject invalid capacities", !context_lookup_new(0) && !context_lookup_new(1048577));
  expect("empty history has no proposal", !context_lookup_propose(s, 15).count);
  expect("reject invalid append lengths", !context_lookup_append(s, NULL, 1) && !context_lookup_append(s, tokens, -1));
  int n = pair(tokens, 12);
  expect("append a complete committed history", context_lookup_append(s, tokens, n));
  ContextDraft p = context_lookup_propose(s, 15);
  expect("a twelve-token match supplies fifteen following tokens", p.count == 15 && p.matched == 12 && p.source == 12);
  expect("proposals copy the actual history continuation", !memcmp(p.tokens, tokens + 12, 15 * sizeof *tokens));
  expect("caller can limit verification width", context_lookup_propose(s, 2).count == 2);
  expect("zero verification capacity disables proposals", !context_lookup_propose(s, 0).count);
  expect("larger caller limits stay within fifteen drafts", context_lookup_propose(s, 99).count == 15);
  ContextDraft before = p;
  expect("capacity failure leaves the index unchanged", !context_lookup_append(s, tokens, 512) && same(before, context_lookup_propose(s, 15)));

  context_lookup_feedback(s, 15, 0), context_lookup_feedback(s, 15, 0);
  expect("two empty verifications require a longer match", !context_lookup_propose(s, 15).count);
  context_lookup_feedback(s, 15, 1);
  expect("some accepted tokens restore the normal threshold", context_lookup_propose(s, 15).count == 15);
  context_lookup_reset(s);
  n = pair(tokens, 4);
  context_lookup_append(s, tokens, n);
  expect("short matches are initially skipped", !context_lookup_propose(s, 15).count);
  expect("invalid feedback is rejected", !context_lookup_feedback(s, 2, 3) && !context_lookup_feedback(s, 16, 1));
  context_lookup_feedback(s, 15, 15);
  expect("a fully accepted proposal enables a four-token match", context_lookup_propose(s, 15).count == 15);
  context_lookup_feedback(s, 0, 0);
  expect("no offered draft is not a failed verification", context_lookup_propose(s, 15).count == 15);
  context_lookup_reset(s);
  n = pair(tokens, 16);
  context_lookup_append(s, tokens, n);
  context_lookup_feedback(s, 15, 0), context_lookup_feedback(s, 15, 0);
  expect("a sixteen-token match survives the stricter threshold", context_lookup_propose(s, 15).count == 15);

  context_lookup_reset(s);
  n = pair(tokens, 12);
  tokens[14] = -7;
  context_lookup_append(s, tokens, n);
  expect("proposals stop before image placeholders", context_lookup_propose(s, 15).count == 2);
  int32_t image = -9;
  context_lookup_append(s, &image, 1);
  expect("an image-ending query is not drafted", !context_lookup_propose(s, 15).count);

  context_lookup_reset(s);
  for (int i = 0; i < 512; i++) tokens[i] = 7;
  context_lookup_append(s, tokens, 512);
  p = context_lookup_propose(s, 15);
  expect("search and backward matching have fixed bounds", p.examined == 64 && p.matched == 64);
  expect("equal matches prefer a complete available continuation", p.count == 15 && p.source == 497);
  expect("proposals never read beyond committed history", p.source >= 0 && p.source + p.count <= 512);
  context_lookup_reset(s);
  expect("reset removes stale matches", !context_lookup_propose(s, 15).count);

  uint32_t random = 1234567;
  int indexed = 1, incremental = 1;
  for (int trial = 0; trial < 512; trial++) {
    context_lookup_reset(s);
    for (int i = 0; i < 64; i++) {
      random ^= random << 13, random ^= random >> 17, random ^= random << 5;
      tokens[i] = i >= 32 ? tokens[i - 32] : (int32_t)(random % 13);
      if (trial % 7 == 0 && i == 15) tokens[i] = -1;
      incremental &= context_lookup_append(s, tokens + i, 1);
      int limit = 1 + trial % 15;
      indexed &= same(context_lookup_propose(s, limit), direct(tokens, i + 1, limit));
    }
  }
  expect("incremental appends succeed within capacity", incremental);
  expect("indexed proposals match a full scan over 32768 prefixes", indexed);
  for (int suffix = 2; suffix <= 4; suffix += 2) {
    context_lookup_reset(s);
    int at = 0;
    for (int i = 0; i < 16; i++) tokens[at++] = 100 + i;
    for (int i = 0; i < 15; i++) tokens[at++] = 1000 + i;
    for (int i = 0; i < 80; i++) {
      tokens[at++] = 2000 + i;
      for (int j = 16 - suffix; j < 16; j++) tokens[at++] = 100 + j;
    }
    for (int i = 0; i < 16; i++) tokens[at++] = 100 + i;
    context_lookup_append(s, tokens, at);
    p = context_lookup_propose(s, 15);
    expect(suffix == 2 ? "common token pairs do not hide a matching long suffix"
                       : "common four-token endings do not hide a matching long suffix",
           p.count == 15 && p.matched == 16 && p.source == 16 && p.tokens[0] == 1000);
  }
  context_lookup_free(s);

  // The draft count: every count is measured DRAFT_CHOICE_WARM times, in
  // rounds from the most down; then the count with the most expected tokens
  // per second, and every DRAFT_CHOICE_EXPLORE-th pass a neighbour of it.
  DraftChoice c;
  draft_choice_init(&c);
  int order[4 * DRAFT_CHOICE_WARM], rounds = 1;
  for (int i = 0; i < 4 * DRAFT_CHOICE_WARM; i++)
    order[i] = draft_choice_count(&c, 3), draft_choice_feedback(&c, order[i], order[i], 0.04 + 0.01 * order[i]);
  for (int i = 0; i < 4 * DRAFT_CHOICE_WARM; i++) rounds &= order[i] == 3 - i % 4;
  expect("each count is measured DRAFT_CHOICE_WARM times, in rounds from the most down", rounds);
  expect("kept drafts make the most drafts the fastest", draft_choice_count(&c, 3) == 3);
  expect("a smaller limit bounds the count", draft_choice_count(&c, 2) == 2);
  for (int i = 0; i < 64; i++) draft_choice_feedback(&c, 3, 0, 0.07);
  expect("rejected drafts make no drafts the fastest", draft_choice_count(&c, 3) == 0);
  expect("a figure follows its last observations", c.kept[1] < 0.01 && c.kept_n[1] == DRAFT_CHOICE_MEMORY);
  draft_choice_init(&c);
  for (int w = 0; w < DRAFT_CHOICE_WARM; w++)
    for (int k = 0; k <= 3; k++) draft_choice_feedback(&c, k, k, 0.04 + 0.01 * k);
  int explored = 0, choices = 0;
  for (int i = 0; i < 4 * DRAFT_CHOICE_EXPLORE; i++, choices++) explored += draft_choice_count(&c, 3) != 3;
  expect("every DRAFT_CHOICE_EXPLORE-th pass tries a neighbour", explored == choices / DRAFT_CHOICE_EXPLORE);
  draft_choice_init(&c);
  draft_choice_feedback(&c, 2, 3, 0.05);
  draft_choice_feedback(&c, 16, 0, 0.05);
  draft_choice_feedback(&c, 1, 0, 0);
  expect("invalid feedback is ignored", !c.seconds_n[1] && !c.seconds_n[2] && !c.kept_n[1]);
  expect("no room means no drafts", draft_choice_count(&c, 0) == 0);

  fprintf(stderr, "context lookup and draft count: %d/%d CPU checks passed\n", checks - failures, checks);
  return failures != 0;
}
