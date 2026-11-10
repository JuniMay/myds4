// Check cache readiness and prefetch accounting without a model or GPU work.
// Run with `make cache-ready-check`. The fixture includes io.c so it can
// represent completed reads whose dispatch group has not yet been released.
#include "../../src/model/deepseek/model.h"
#include "../../src/engine/io.c"

static int checks, failures;

static void expect(const char *state, const char *field, uint64_t got, uint64_t want) {
  checks++;
  if (got == want) return;
  fprintf(stderr, "%s: %s: expected %llu, got %llu\n", state, field, (unsigned long long)want, (unsigned long long)got);
  failures++;
}

static void check_prefetch(const char *state, int has_group, int pending) {
  const int layer = 0, expert = 7;
  Slot slot = {.layer = layer, .expert = expert, .prev = -1, .next = -1, .prefetched = 1, .mapped = 1, .npiece = 1};
  double slot_score = 0;
  st->slots = &slot;
  st->score = &slot_score;
  st->head = st->tail = 0;
  st->token = 1;
  memset(st->where, 0xff, sizeof st->where);
  st->where[layer][expert] = 0;
  memset(&cache_stats, 0, sizeof cache_stats);
  // The read was claimed already. cache_request must not launch any I/O.
  atomic_store(&slot.taken[0], 1);
  if (has_group) {
    slot.loading = dispatch_group_create();
    dispatch_group_enter(slot.loading);
    if (!pending) dispatch_group_leave(slot.loading);
    expect(state, "group has pending work", dispatch_group_wait(slot.loading, DISPATCH_TIME_NOW) != 0, pending);
  }

  expect(state, "selected expert has a slot", cache_slot(layer, expert), 1);
  expect(state, "ready before request", cache_ready(layer, expert), !pending);
  expect(state, "prefetch classification", cache_where(layer, expert), pending ? 3 : 2);
  expect(state, "request's pending read count", cache_request(layer, &expert, 1), pending);
  expect(state, "prefetch hits", cache_stats.prefetch_hits, 1);
  expect(state, "late prefetch hits", cache_stats.late, pending);
  expect(state, "demand misses", cache_stats.misses, 0);
  expect(state, "bytes read", cache_stats.bytes, 0);
  expect(state, "selected expert keeps its slot", cache_slot(layer, expert), 1);

  if (pending) {
    dispatch_group_leave(slot.loading);
    expect(state, "ready after I/O completes", cache_ready(layer, expert), 1);
  }
  // Reap the completed handle, as the normal consumer does. No wait can block.
  cache_wait(layer, &expert, 1);
  expect(state, "ready after consumer wait", cache_ready(layer, expert), 1);
  expect(state, "consumer releases group", slot.loading == NULL, 1);
  expect(state, "missing expert has no slot", cache_slot(layer, expert + 1), 0);
  expect(state, "missing expert is not ready", cache_ready(layer, expert + 1), 0);
  expect(state, "missing expert classification", cache_where(layer, expert + 1), 0);
  st->slots = NULL;
  st->score = NULL;
}

int main(void) {
  check_prefetch("completed group", 1, 0);
  check_prefetch("pending group", 1, 1);
  check_prefetch("released group", 0, 0);
  fprintf(stderr, "cache readiness: %d/%d checks passed\n", checks - failures, checks);
  return failures != 0;
}
