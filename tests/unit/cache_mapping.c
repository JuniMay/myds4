// CPU reads do not need GPU residency. Only an encoded use maps an expert.
#include "../../src/model/deepseek/model.h"
#include "myds4.h"
#include <stdlib.h>
#include <string.h>
static int added, removed;
static Buf fixture_new(size_t bytes) {
  void *p = calloc(1, bytes);
  return (Buf){.obj = p, .ptr = p, .size = bytes};
}
static void fixture_resident(const Buf *b, int on) { if (on) added++; else removed++; }
static void fixture_free(Buf *b) { free(b->ptr); *b = (Buf){0}; }
#define gpu_new fixture_new
#define gpu_resident fixture_resident
#define gpu_free fixture_free
#include "../../src/engine/io.c"
#undef gpu_new
#undef gpu_resident
#undef gpu_free

int main(void) {
  Slot fixture[2] = {0};
  double priorities[2] = {INFINITY, INFINITY};
  int unused[2] = {1, 0}, failures = 0, checks = 0;
#define CHECK(x) do { checks++; if (!(x)) failures++, fprintf(stderr, "FAIL: %s\n", #x); } while (0)
  st->slots = fixture, st->score = priorities, st->free_list = unused;
  st->max_slots = st->cap = st->nfree = 2, st->SLOT_SIZE = 64;
  st->head = st->tail = st->pass_layer = st->serving = -1;
  memset(st->where, 0xff, sizeof st->where);
  int a = take(0, 0, 1), b = take(1, 0, 0);
  CHECK(st->nslots == 2 && st->nmapped == 0 && added == 0);
  CHECK(!st->slots[a].mapped && !st->slots[b].mapped);
  int ids[] = {0};
  CHECK(cache_request(0, ids, 1) == 0);
  CHECK(st->nmapped == 0 && added == 0);
  uint32_t at[3];
  int fresh;
  CHECK(cache_get(0, 0, at, &fresh) == &st->slots[a].b);
  CHECK(st->nmapped == 1 && added == 1 && fresh && st->slots[a].mapped);
  CHECK(cache_stats.remaps == 0);
  CHECK(cache_get(0, 0, at, &fresh) == &st->slots[a].b);
  CHECK(st->nmapped == 1 && added == 1);
  // Reusing an old mapped buffer for a forecast must release its mapping.
  st->token++, st->serving = -1;
  st->score[a] = -100, st->score[b] = 100;
  int reused = take(2, 0, 1);
  CHECK(reused == a && st->nmapped == 0 && removed == 1 && !st->slots[a].mapped);
  CHECK(cache_get(2, 0, at, &fresh) == &st->slots[a].b);
  CHECK(st->nmapped == 1 && added == 2 && fresh);
  CHECK(cache_stats.remaps == 1);
  CHECK(st->nslots == 2 && st->where[0][0] == -1 && st->where[2][0] == a);
  for (int i = 0; i < 2; i++) fixture_free(&fixture[i].b);
  fprintf(stderr, "cache mapping: %d/%d CPU checks passed\n", checks - failures, checks);
  return failures != 0;
}
