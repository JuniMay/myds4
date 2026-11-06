// Exercise cross-stage borrowing with the same total slots, including
// pending reads, current-request protection and removal of old lookups.
// Reuse the host-only fixture; the ordinary cache check remains separate.
#define main ordinary_draft_cache_checks
#include "draft_cache_check.c"
#undef main

int main(void) {
  setup(16);
  uint64_t budget = cache_bytes();
  int ids[15];
  for (int i = 0; i < 15; i++) ids[i] = i;
  cache_load(LAYERS, ids, 15);
  for (int i = 0; i < 15; i++) ids[i] = 15 + i;
  cache_load(LAYERS, ids, 15);
  int resident = 0;
  for (int e = 0; e < 30; e++) resident += cache_ready(LAYERS, e);
  expect("a stage borrows spare slots from the total budget", resident, 30);
  expect("borrowing allocates no extra memory", cache_bytes(), budget);
  for (int i = 0; i < 15; i++) ids[i] = i;
  cache_load(LAYERS + 1, ids, 15);
  cache_load(LAYERS + 2, ids, 3);
  int oldest = where[LAYERS][0];
  uint64_t address = slots[oldest].b.gpu;
  int another = 3;
  cache_load(LAYERS + 2, &another, 1);
  expect("global least-recent expert gives way", cache_slot(LAYERS, 0), 0);
  expect("replacement uses the same buffer", where[LAYERS + 2][3], oldest);
  expect("new owner is recorded on the slot", slots[oldest].layer, LAYERS + 2);
  expect("native buffer address remains stable", slots[oldest].b.gpu, address);
  int current[] = {100, 1};
  cache_load(LAYERS, current, 2);
  expect("the entire current union is protected before replacement", cache_slot(LAYERS, 1), 1);
  expect("the next unprotected old expert gives way", cache_slot(LAYERS, 2), 0);
  expect("the new expert is ready", cache_ready(LAYERS, 100), 1);

  int pending = where[LAYERS][3];
  draft_age[pending - max_slots] = 0;
  slots[pending].loading = dispatch_group_create();
  dispatch_group_enter(slots[pending].loading);
  slots[pending].npiece = 1;
  atomic_store(&slots[pending].taken[0], 1);
  another = 4;
  cache_load(LAYERS + 2, &another, 1);
  expect("a pending read is never borrowed", where[LAYERS][3], pending);
  expect("pending group remains owned", slot_reading(pending), 1);
  int same = 3;
  expect("a pending hit reports its existing read", cache_request(LAYERS, &same, 1), 1);
  expect("a pending hit starts no duplicate read", atomic_load(&demand), 0);
  dispatch_group_leave(slots[pending].loading);
  cache_wait(LAYERS, &same, 1);
  expect("the consumer releases the completed group", slots[pending].loading == NULL, 1);

  int completed = where[LAYERS][5];
  draft_age[completed - max_slots] = 0;
  slots[completed].loading = dispatch_group_create();
  another = 5;
  cache_load(LAYERS + 2, &another, 1);
  expect("a completed group can be borrowed", where[LAYERS + 2][5], completed);
  expect("old lookup is cleared across stage boundaries", cache_slot(LAYERS, 5), 0);
  expect("replacement group is released", slots[completed].loading == NULL, 1);
  for (int i = max_slots; i < max_slots + npinned; i++) {
    Slot *s = &slots[i];
    expect("every occupied slot has one matching owner", where[s->layer][s->expert], i);
    s->used = 9999;
  }
  expect("all protected draft slots refuse a victim", draft_victim(LAYERS, 9999) == -1, 1);
  expect("the backbone pool is untouched", where[0][100], 0);
  expect("the total budget stays unchanged", cache_bytes(), budget);
  expect("the fixture performs no SSD reads", cache_stats.bytes, 0);
  clear_fixture();
  fprintf(stderr, "shared draft pool: %d/%d CPU checks passed\n", checks - failures, checks);
  return failures != 0;
}
