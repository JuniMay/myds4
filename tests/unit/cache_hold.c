// Reservations must prevent reuse across independently advancing GPU jobs.
#include "../../src/model/deepseek/model.h"
#include "../../src/engine/io.c"
#include "cache_layout.h"

int main(void) {
  st = deepseek_cache_create();
  Slot fixture[4] = {0};
  double priorities[4] = {1, 2, 3, 4};
  int free_slots[4];
  st->slots = fixture, st->score = priorities, st->max_slots = st->cap = st->nslots = 4, st->free_list = free_slots;
  memset(st->where, 0xff, WHERE_BYTES);
  st->head = 3, st->tail = 0, st->pass_layer = st->serving = -1;
  st->token = 10, st->weight = 1;
  for (int i = 0; i < 4; i++) {
    fixture[i].layer = 0, fixture[i].expert = i, fixture[i].mapped = 1;
    fixture[i].prev = i == 3 ? -1 : i + 1, fixture[i].next = i ? i - 1 : -1;
    WHERE(0, i) = i, st->escore[i] = priorities[i];
  }
  int ids[] = {0, 1}, second[] = {1, 2}, failures = 0, checks = 0;
#define CHECK(x) do { checks++; if (!(x)) failures++, fprintf(stderr, "FAIL: %s\n", #x); } while (0)
  CHECK(cache_hold(st, 0, ids, 2) == 0);
  st->serving = 3;
  CHECK(victim(st) == 2);
  CHECK(cache_hold(st, 0, second, 2) == 0);
  st->serving = 3;
  CHECK(fixture[1].holds == 2 && victim(st) == 3);
  cache_unhold(st, 0, ids, 2);
  CHECK(fixture[0].holds == 0 && fixture[1].holds == 1 && victim(st) == 0);
  CHECK(WHERE(0, 0) == 0 && WHERE(0, 1) == 1 && WHERE(0, 2) == 2);
  cache_unhold(st, 0, second, 2);
  CHECK(fixture[1].holds == 0 && fixture[2].holds == 0 && st->held_references == 0);
  CHECK(st->stats.misses == 0 && st->stats.evictions == 0);
  fixture[3].loading = dispatch_group_create();
  dispatch_group_enter(fixture[3].loading);
  int pending[] = {3};
  CHECK(cache_hold(st, 0, pending, 1) == 1 && cache_hold(st, 0, pending, 1) == 1);
  CHECK(fixture[3].holds == 2 && WHERE(0, 3) == 3 && st->stats.misses == 0);
  cache_unhold(st, 0, pending, 1);
  CHECK(fixture[3].holds == 1 && !cache_ready(st, 0, 3));
  dispatch_group_leave(fixture[3].loading);
  cache_wait(st, 0, pending, 1);
  CHECK(cache_ready(st, 0, 3) && fixture[3].holds == 1);
  cache_unhold(st, 0, pending, 1);
  CHECK(fixture[3].holds == 0 && st->held_references == 0);
  // A resize gives back unheld slots, the least recently used first, and
  // leaves held ones, whatever their age, for the work that holds them.
  int oldest[] = {0, 1};
  CHECK(cache_hold(st, 0, oldest, 2) == 0);
  st->cap = 2;
  cache_adapt_pass(st, 2);
  CHECK(st->nslots == 2 && fixture[0].holds == 1 && fixture[1].holds == 1);
  CHECK(WHERE(0, 0) == 0 && WHERE(0, 1) == 1 && WHERE(0, 2) < 0 && WHERE(0, 3) < 0);
  CHECK(st->held_references == 2);
  cache_unhold(st, 0, oldest, 2);
  printf("cache job reservations: %d/%d CPU checks passed\n", checks - failures, checks);
  return failures != 0;
}
