#include "../../src/experimental/draft_queue.h"
#include <stdio.h>

static int checks, failures;
static void expect(const char *name, int yes) {
  checks++;
  if (!yes) failures++, fprintf(stderr, "FAIL: %s\n", name);
}

int main(void) {
  DraftQueue q = {0};
  const int original[5] = {20, 21, 22, 23, 24};
  int out[5];
  expect("set a complete draft block", draft_queue_set(&q, original, 5));
  expect("take only the verification prefix", draft_queue_take(&q, out, 2) == 2 && out[0] == 20 && out[1] == 21);
  expect("taking a prefix does not consume it", q.count == 5);
  expect("matching bonus leaves the later guesses", draft_queue_accept(&q, 2, 2, 22) && q.count == 2 && q.tokens[0] == 23);
  expect("end of the block clears the queue", draft_queue_accept(&q, 2, 2, 99) && !q.count);
  draft_queue_set(&q, original, 5);
  expect("rejected prefix clears later guesses", draft_queue_accept(&q, 2, 1, 99) && !q.count);
  draft_queue_set(&q, original, 5);
  expect("different bonus clears later guesses", draft_queue_accept(&q, 2, 2, 99) && !q.count);
  draft_queue_set(&q, original, 5);
  expect("invalid acceptance fails closed", !draft_queue_accept(&q, 2, 3, 22) && !q.count);
  expect("invalid block length fails closed", !draft_queue_set(&q, original, 6) && !q.count);
  draft_queue_set(&q, original, 5);
  expect("no remaining output space offers no guesses", !draft_queue_take(&q, out, 0) && q.count == 5);
  int all = 1;
  for (int chunk = 1; chunk <= 5; chunk++) for (int mismatch = 0; mismatch <= 5; mismatch++) {
    draft_queue_set(&q, original, 5);
    int offset = 0;
    while (q.count) {
      int offered = draft_queue_take(&q, out, chunk);
      int accepted = 0;
      while (accepted < offered && offset + accepted < mismatch) accepted++;
      int next_at = offset + accepted;
      int next = next_at == mismatch ? 999 : next_at < 5 ? original[next_at] : 888;
      all &= draft_queue_accept(&q, offered, accepted, next);
      if (accepted < offered || next_at == mismatch) {
        all &= !q.count;
        break;
      }
      offset += offered;
      if (offset < 5) offset++;
      all &= q.count == 5 - offset && !memcmp(q.tokens, original + offset, (size_t)q.count * sizeof *out);
    }
  }
  expect("every chunk width and rejection position preserves only a matching suffix", all);
  printf("draft queue: %d/%d CPU checks passed\n", checks - failures, checks);
  return failures != 0;
}
