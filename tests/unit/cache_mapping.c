// CPU reads do not need GPU residency. Only an encoded use maps an expert.
#include "../../src/model/deepseek/model.h"
#include "myds4.h"
#include <fcntl.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
static int added, removed;
static Buf fixture_new(size_t bytes) {  // filled with 0xaa, so that a read of zeros shows
  void *p = malloc(bytes);
  memset(p, 0xaa, bytes);
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

// The transform case's layout: each expert's two spans, and what its
// transform saw (readers may run two at once): calls, calls before every
// piece of the expert's slot was read, and bytes of the spans not read.
static const uint64_t read_len[2] = {3 * PIECE + 5, 77};
enum { EXTRA = 40 };
typedef struct {
  atomic_int calls, early, unread;
  uint32_t at[2][2];
  uint64_t size[2];
} Marks;
static void mark(void *context, int layer, int expert, uint8_t *slot, const uint32_t *at, uint64_t slot_bytes) {
  Marks *m = context;
  Slot *s = &st->slots[WHERE(layer, expert)];
  m->early += atomic_load(&s->read) != s->npiece;
  int unread = 0;
  for (int k = 0; k < 2; k++)
    for (uint64_t b = 0; b < read_len[k]; b++) unread += slot[at[k] + b] != 0;
  m->unread += unread;
  m->at[expert][0] = at[0], m->at[expert][1] = at[1], m->size[expert] = slot_bytes;
  memset(slot + slot_bytes - EXTRA, expert + 1, EXTRA);
  m->calls++;
}

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

  // A layout's transform runs once an expert's pieces are all read, before
  // the expert counts as read: it sees the read bytes at the spans' offsets
  // and writes the layout's extra bytes at the slot's end, which the slot's
  // size includes. One layer of 2 experts, each 2 spans (5 pieces) read from
  // /dev/zero.
  int zero = open("/dev/zero", O_RDONLY);
  static FileSpan read_span[1][2][2];
  static int read_spans[1] = {2};
  for (int e = 0; e < 2; e++)
    read_span[0][e][0] = (FileSpan){zero, SECTOR + 100 * e, read_len[0]},
    read_span[0][e][1] = (FileSpan){zero, 7, read_len[1]};
  static Marks marks;
  st = cache_create(&(ExpertLayout){.layers = 1, .scored_layers = 1, .experts = 2, .chosen = 1, .parts = 2,
                                    .span = &read_span[0][0][0], .spans = read_spans, .extra = EXTRA,
                                    .transform = mark, .context = &marks});
  cache_retention(st, 0, 0);
  cache_required(st, 1);
  CHECK(cache_slot_bytes(st) == EXTRA + REGION(read_len[0]) + REGION(read_len[1]));
  Slot read_slots[2] = {0};
  double read_scores[2] = {INFINITY, INFINITY};
  int read_frees[2] = {1, 0};
  st->slots = read_slots, st->score = read_scores, st->free_list = read_frees;
  st->max_slots = st->cap = st->mcap = st->nfree = 2;
  st->head = st->tail = st->pass_layer = st->serving = -1;
  memset(st->where, 0xff, WHERE_BYTES);
  int pair[] = {0, 1};
  cache_hold(st, 0, pair, 2);  // returns the reads still pending when checked, which varies
  cache_wait(st, 0, pair, 2);
  CHECK(marks.calls == 2 && marks.early == 0 && marks.unread == 0);
  for (int e = 0; e < 2; e++) {
    const Buf *slot = cache_get(st, 0, e, at, &fresh);
    int marked = slot != NULL;
    for (int b = 0; marked && b < EXTRA; b++) marked = slot->ptr[slot->size - EXTRA + b] == e + 1;
    CHECK(marked && marks.size[e] == slot->size && at[0] == marks.at[e][0] && at[1] == marks.at[e][1]);
    CHECK(at[1] >= REGION(read_len[0]) && at[1] + read_len[1] <= slot->size - EXTRA);
  }
  cache_unhold(st, 0, pair, 2);
  close(zero);
  for (int i = 0; i < 2; i++) fixture_free(&read_slots[i].b);

  // A pass's fewest slots (cache_working_rows) are room the model reserved
  // before the pass, so the slot limit rises to them. A store whose limit
  // was set for one row (4 slots) holds a whole layer of 8 experts in a
  // prompt pass of 4 rows (16 slots); without the room, no slot of the
  // layer may go and the process ends. In a child, which such an end fails.
  pid_t child = fork();
  if (!child) {
    static FileSpan wide_span[2][8][1];  // empty spans: their reads finish at once
    static int wide_spans[2] = {1, 1};
    st = cache_create(&(ExpertLayout){.layers = 2, .scored_layers = 2, .experts = 8, .chosen = 2, .parts = 1,
                                      .span = &wide_span[0][0][0], .spans = wide_spans});
    cache_retention(st, 0, 0);
    static Slot wide[16];
    static double wide_scores[16];
    static int wide_frees[16];
    for (int i = 0; i < 16; i++) wide_scores[i] = INFINITY, wide_frees[i] = 15 - i;
    st->slots = wide, st->score = wide_scores, st->free_list = wide_frees;
    st->max_slots = st->nfree = 16, st->SLOT_SIZE = 64;
    st->head = st->tail = st->pass_layer = st->serving = -1;
    memset(st->where, 0xff, WHERE_BYTES);
    cache_working_rows(st, 1);
    st->cap = st->mcap = st->minimum_slots;  // as cache_init sets them for one row, with little memory free
    cache_working_rows(st, 4);
    int layer[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    cache_pass(st, 0);
    cache_hold(st, 0, layer, 8);
    cache_wait(st, 0, layer, 8);
    int held = st->nslots == 8;
    for (int e = 0; e < 8; e++) held &= WHERE(0, e) >= 0;
    _exit(held ? 0 : 1);
  }
  int status = 0;
  CHECK(child > 0 && waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0);
  fprintf(stderr, "cache mapping: %d/%d CPU checks passed\n", checks - failures, checks);
  return failures != 0;
}
