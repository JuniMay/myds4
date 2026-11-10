// Reservations must prevent reuse across independently advancing GPU jobs.
#include "../../src/model/deepseek/model.h"
#include "../../src/engine/io.c"

int main(void) {
  Slot fixture[4] = {0};
  double priorities[4] = {1, 2, 3, 4};
  st->slots = fixture, st->score = priorities, st->max_slots = st->cap = st->nslots = 4;
  memset(st->where, 0xff, sizeof st->where);
  st->head = 3, st->tail = 0, st->pass_layer = st->serving = -1;
  st->token = 10, st->weight = 1;
  for (int i = 0; i < 4; i++) {
    fixture[i].layer = 0, fixture[i].expert = i, fixture[i].mapped = 1;
    fixture[i].prev = i == 3 ? -1 : i + 1, fixture[i].next = i ? i - 1 : -1;
    st->where[0][i] = i, st->escore[i] = priorities[i];
  }
  int ids[] = {0, 1}, second[] = {1, 2}, failures = 0, checks = 0;
#define CHECK(x) do { checks++; if (!(x)) failures++, fprintf(stderr, "FAIL: %s\n", #x); } while (0)
  CHECK(cache_hold(0, ids, 2) == 0);
  st->serving = 3;
  CHECK(victim() == 2);
  CHECK(cache_hold(0, second, 2) == 0);
  st->serving = 3;
  CHECK(fixture[1].holds == 2 && victim() == 3);
  cache_unhold(0, ids, 2);
  CHECK(fixture[0].holds == 0 && fixture[1].holds == 1 && victim() == 0);
  CHECK(st->where[0][0] == 0 && st->where[0][1] == 1 && st->where[0][2] == 2);
  cache_unhold(0, second, 2);
  CHECK(fixture[1].holds == 0 && fixture[2].holds == 0 && st->held_references == 0);
  CHECK(cache_stats.misses == 0 && cache_stats.evictions == 0);
  fixture[3].loading = dispatch_group_create();
  dispatch_group_enter(fixture[3].loading);
  int pending[] = {3};
  CHECK(cache_hold(0, pending, 1) == 1 && cache_hold(0, pending, 1) == 1);
  CHECK(fixture[3].holds == 2 && st->where[0][3] == 3 && cache_stats.misses == 0);
  cache_unhold(0, pending, 1);
  CHECK(fixture[3].holds == 1 && !cache_ready(0, 3));
  dispatch_group_leave(fixture[3].loading);
  cache_wait(0, pending, 1);
  CHECK(cache_ready(0, 3) && fixture[3].holds == 1);
  cache_unhold(0, pending, 1);
  CHECK(fixture[3].holds == 0 && st->held_references == 0);
  printf("cache job reservations: %d/%d CPU checks passed\n", checks - failures, checks);
  return failures != 0;
}
