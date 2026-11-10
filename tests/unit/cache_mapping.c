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
#include "cache_layout.h"
#undef gpu_new
#undef gpu_resident
#undef gpu_free

int main(void) {
  st = deepseek_cache_create();
  Slot fixture[2] = {0};
  double priorities[2] = {INFINITY, INFINITY};
  int unused[2] = {1, 0}, failures = 0, checks = 0;
#define CHECK(x) do { checks++; if (!(x)) failures++, fprintf(stderr, "FAIL: %s\n", #x); } while (0)
  st->slots = fixture, st->score = priorities, st->free_list = unused;
  st->max_slots = st->cap = st->nfree = 2, st->SLOT_SIZE = 64;
  st->head = st->tail = st->pass_layer = st->serving = -1;
  memset(st->where, 0xff, WHERE_BYTES);
  int a = take(st, 0, 0, 1), b = take(st, 1, 0, 0);
  CHECK(st->nslots == 2 && st->nmapped == 0 && added == 0);
  CHECK(!st->slots[a].mapped && !st->slots[b].mapped);
  int ids[] = {0};
  CHECK(cache_request(st, 0, ids, 1) == 0);
  CHECK(st->nmapped == 0 && added == 0);
  uint32_t at[STORE_PARTS];
  int fresh;
  CHECK(cache_get(st, 0, 0, at, &fresh) == &st->slots[a].b);
  CHECK(st->nmapped == 1 && added == 1 && fresh && st->slots[a].mapped);
  CHECK(st->stats.remaps == 0);
  CHECK(cache_get(st, 0, 0, at, &fresh) == &st->slots[a].b);
  CHECK(st->nmapped == 1 && added == 1);
  // Reusing an old mapped buffer for a forecast must release its mapping.
  st->token++, st->serving = -1;
  st->score[a] = -100, st->score[b] = 100;
  int reused = take(st, 2, 0, 1);
  CHECK(reused == a && st->nmapped == 0 && removed == 1 && !st->slots[a].mapped);
  CHECK(cache_get(st, 2, 0, at, &fresh) == &st->slots[a].b);
  CHECK(st->nmapped == 1 && added == 2 && fresh);
  CHECK(st->stats.remaps == 1);
  CHECK(st->nslots == 2 && WHERE(0, 0) == -1 && WHERE(2, 0) == a);
  for (int i = 0; i < 2; i++) fixture_free(&fixture[i].b);

  // A layer reaches its experts through a table of their addresses only while
  // each is in a slot, read, mapped and committed (not fresh); until one moves
  // or a slot is unmapped, it stays ready. A store of 2 layers of 4 experts.
  static FileSpan small_span[2][4][1];  // empty spans: their reads finish at once
  static int small_spans[2] = {1, 1};
  st = cache_create(&(ExpertLayout){.layers = 2, .scored_layers = 2, .experts = 4, .chosen = 2, .parts = 1,
                                    .span = &small_span[0][0][0], .spans = small_spans});
  cache_retention(st, 0, 0);
  Slot slots[8] = {0};
  double scores[8];
  int frees[8];
  for (int i = 0; i < 8; i++) scores[i] = INFINITY, frees[i] = 7 - i;
  st->slots = slots, st->score = scores, st->free_list = frees;
  st->max_slots = st->cap = st->mcap = st->nfree = 8, st->SLOT_SIZE = 64;
  st->head = st->tail = st->pass_layer = st->serving = -1;
  memset(st->where, 0xff, WHERE_BYTES);
  uint64_t g0, g1, g2, g3;
  int layer0[] = {0, 1, 2, 3};
  CHECK(!cache_layer_ready(st, 0, &g0));
  CHECK(cache_hold(st, 0, layer0, 4) == 0);  // a whole layer, more than a row chooses
  CHECK(!cache_layer_ready(st, 0, &g1) && g1 != g0);  // read but not reaped (cache_wait)
  cache_wait(st, 0, layer0, 4);
  CHECK(!cache_layer_ready(st, 0, &g1));  // not mapped
  for (int e = 0; e < 4; e++) CHECK(cache_get(st, 0, e, at, &fresh) && fresh);
  CHECK(!cache_layer_ready(st, 0, &g1) && st->any_fresh);
  cache_unhold(st, 0, layer0, 4);
  cache_adapt(st);
  CHECK(!st->any_fresh && cache_layer_ready(st, 0, &g2) && g2 == g1);
  CHECK(cache_layer_ready(st, 0, &g2) && !cache_layer_ready(st, 1, &g3));
  st->mcap = 3;  // the residency limit takes one slot back
  cache_adapt(st);
  CHECK(st->nmapped == 3 && !cache_layer_ready(st, 0, &g3) && g3 == g2);
  st->mcap = 8;
  for (int e = 0; e < 4; e++) cache_get(st, 0, e, at, &fresh);
  cache_adapt(st);
  CHECK(cache_layer_ready(st, 0, &g3));
  st->cap = 4;  // full: a layer-1 expert takes a layer-0 expert's slot
  take(st, 1, 0, 0);
  int gone = -1;
  for (int e = 0; e < 4; e++)
    if (WHERE(0, e) < 0) gone = e;
  CHECK(gone >= 0 && !cache_layer_ready(st, 0, &g3) && g3 != g2);
  // A table's experts count as used, without a read for one that left its slot.
  uint64_t hits = st->stats.hits, misses = st->stats.misses;
  int kept = gone ? 0 : 1, used[] = {gone, kept};
  cache_use(st, 0, used, 2);
  CHECK(WHERE(0, gone) < 0 && st->stats.misses == misses && st->stats.hits == hits + 1);
  CHECK(st->slots[WHERE(0, kept)].used == st->token && st->head == WHERE(0, kept));
  for (int i = 0; i < 8; i++) fixture_free(&slots[i].b);
  fprintf(stderr, "cache mapping: %d/%d CPU checks passed\n", checks - failures, checks);
  return failures != 0;
}
