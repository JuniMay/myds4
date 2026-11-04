// Bulk prefill should not change the decode-frequency history in this experiment.
#define MYDS4_DECODE_ONLY_SCORE
#include "../../src/engine/io.c"

static int checks, failures;
static void expect(const char *name, int ok) {
  checks++;
  if (!ok) failures++, fprintf(stderr, "FAIL: %s\n", name);
}

int main(void) {
  alarm(10);
  static Slot memory[2];
  static double priorities[2];
  slots = memory, score = priorities;
  max_slots = cap = nslots = 2, head = 1, tail = 0, token = 7, weight = 2;
  memset(where, 0xff, sizeof where);
  for (int i = 0; i < 2; i++) {
    slots[i].layer = 0, slots[i].expert = i, slots[i].mapped = 1;
    slots[i].prev = i ? -1 : 1, slots[i].next = i ? 0 : -1;
    where[0][i] = i;
  }
  score[0] = escore[0] = 11, score[1] = escore[1] = 17;
  int ids[2] = {0, 1};
  cache_pass(0);
  int reading = 0;
  for (int batch = 0; batch < 4; batch++) reading += cache_request(0, ids, 2);
  expect("resident prefill requests perform no reads", !reading && !cache_stats.bytes && !cache_stats.misses);
  expect("prefill request accounting is retained", cache_stats.hits == 8);
  expect("prefill does not change prior decode history", escore[0] == 11 && escore[1] == 17);
  expect("resident priorities stay consistent with history", score[0] == 11 && score[1] == 17);
  expect("requests do not advance the score clock", token == 7 && weight == 2);
  cache_pass(-1);
  cache_request(0, ids, 1);
  expect("decoded requests still add their normal weight", escore[0] == 13 && score[0] == 13);
  expect("unselected history stays unchanged", escore[1] == 17);
  fprintf(stderr, "prefill priority: %d/%d CPU checks passed\n", checks - failures, checks);
  return failures != 0;
}
