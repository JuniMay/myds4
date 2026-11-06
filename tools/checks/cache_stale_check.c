// Experimental pass/layer-aware eviction, retaining live-work protection.
#define MYDS4_STALE_PRIORITY
#include "../../src/engine/io.c"

static int checks, failures;
static void expect(const char *name, int ok) {
  checks++;
  if (!ok) failures++, fprintf(stderr, "FAIL: %s\n", name);
}

static void setup(void) {
  static Slot memory[3];
  static double priority[3];
  slots = memory, score = priority;
  memset(slots, 0, sizeof memory);
  memset(where, 0xff, sizeof where);
  max_slots = nslots = cap = 3, head = 2, tail = 0;
  pass_layer = -1, serving = 5, published = -1, token = 7, weight = 2;
  for (int i = 0; i < 3; i++) {
    slots[i].expert = i, slots[i].mapped = 1, slots[i].demanded = 6;
    slots[i].prev = i == 2 ? -1 : i + 1, slots[i].next = i ? i - 1 : -1;
  }
  slots[0].layer = 2, score[0] = 100;
  slots[1].layer = 8, score[1] = 1;
  slots[2].layer = 3, score[2] = 2, slots[2].demanded = 7;
}

int main(void) {
  alarm(10);
  setup();
  expect("stale past layer precedes stale future layer", victim() == 0);
  slots[0].demanded = token;
  expect("stale future precedes actually used past expert", victim() == 1);
  slots[1].demanded = token;
  expect("fallback among used experts follows frequency", victim() == 1);
  published = 8;
  expect("published GPU table remains protected", victim() == 2);
  setup();
  slots[0].prefetched = 1;
  expect("protected prefetch remains ineligible", victim() == 1);
  slots[0].prefetched = 0, slots[0].used = token;
  expect("demoted unused forecast is not an actual demand", victim() == 0);
  slots[0].layer = serving;
  expect("current selected GPU work remains protected", victim() == 1);
  setup();
  slots[1].layer = 1;
  expect("frequency orders stale experts in the same class", victim() == 1);
  slots[2].expert = -1, score[2] = -INFINITY;
  expect("an empty slot precedes every occupied class", victim() == 2);
  setup();
  for (int i = 0; i < 3; i++) slots[i].layer = 0, where[0][i] = i;
  int selected = 1;
  expect("resident actual demand needs no I/O", cache_request(0, &selected, 1) == 0);
  expect("actual demand records this pass", slots[1].demanded == token);
  expect("unselected expert is not marked demanded", slots[0].demanded == 6);
  fprintf(stderr, "cache stale priority: %d/%d CPU checks passed\n", checks - failures, checks);
  return failures != 0;
}
