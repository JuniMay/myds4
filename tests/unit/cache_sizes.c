// Slots sized per layer: when a store's layers differ in expert size, a slot's
// buffer has its layer's size and the cache's limits count bytes. No GPU or
// model: buffers are plain memory and the experts are read from /dev/zero.
#include "myds4.h"
#include "memory.h"
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
static int added, removed, freed;
static Buf fixture_new(size_t bytes) {
  void *p = malloc(bytes);
  memset(p, 0xaa, bytes);
  return (Buf){.obj = p, .ptr = p, .size = bytes, .allocated = bytes};
}
static void fixture_resident(const Buf *b, int on) { if (on) added++; else removed++; }
// With a memory pool: charged to the current owner as the engine's buffers are, or refused.
static Buf fixture_try_new(size_t bytes, int admission) {
  MemorySlot *owner = memory_current() ? memory_current() : memory_shared();
  if (!memory_claim(owner, bytes, (MemoryAdmission)admission)) return (Buf){0};
  Buf b = fixture_new(bytes);
  b.memory = owner;
  return b;
}
static void fixture_free(Buf *b) {
  if (b->memory) memory_release(b->memory, b->allocated);
  free(b->ptr);
  *b = (Buf){0};
  freed++;
}
static int pooled;  // the slots are entries of a memory pool (pool)
// As gpu_new: with a pool, memory that the pool must make room for.
static Buf fixture_required(size_t bytes) {
  if (!pooled) return fixture_new(bytes);
  Buf b = fixture_try_new(bytes, MEMORY_REQUIRED);
  if (!b.obj) fprintf(stderr, "the pool cannot accommodate %zu required bytes\n", bytes), abort();
  return b;
}
#define gpu_new fixture_required
#define gpu_try_new fixture_try_new
#define gpu_resident fixture_resident
#define gpu_free fixture_free
#include "../../src/engine/io.c"
#undef gpu_new
#undef gpu_try_new
#undef gpu_resident
#undef gpu_free

// Three layers of 8 experts: their experts take 16, 17 and 24 sectors.
enum { LAYERS3 = 3, EXPERTS8 = 8, SLOTS = 96 };
#define SMALL (16 * SECTOR)
#define MIDDLE (17 * SECTOR)
#define LARGE (24 * SECTOR)
static FileSpan span[LAYERS3][EXPERTS8][1];
static int spans[LAYERS3] = {1, 1, 1};
static ExpertStore *st;
static Slot slots[SLOTS];
static double scores[SLOTS];
static int frees[SLOTS], failures, checks;
#define CHECK(x) do { checks++; if (!(x)) failures++, fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); } while (0)

static MemoryAccount account;
static MemorySlot backbone;

// A pass that demands one expert (for a demand that has to wait).
static int wanted[2], got = -9;
static void demand(void *argument) {
  (void)argument;
  got = take(st, wanted[0], wanted[1], 0);
}

// A store whose layers' experts have these payload bytes, with room for
// `units` slots of the largest size.
static void setup(uint64_t len0, uint64_t len1, uint64_t len2, int units) {
  static int zero = -1;
  if (zero < 0) zero = open("/dev/zero", O_RDONLY);
  if (!st) {
    st = cache_create(&(ExpertLayout){.layers = LAYERS3, .scored_layers = LAYERS3, .experts = EXPERTS8, .chosen = 2, .parts = 1,
                                      .span = &span[0][0][0], .spans = spans});
    cache_retention(st, 0, 0);
  }
  for (int i = 0; st->slots && i < SLOTS; i++) {
    finish(st, i);
    if (slots[i].b.obj) fixture_free(&slots[i].b);
  }
  if (pooled) {
    for (int i = 0; i < SLOTS; i++) memory_slot_remove(&slots[i].memory);
    memory_slot_remove(&backbone);
    cache_memory(st, NULL);
    pooled = 0;
  }
  uint64_t len[LAYERS3] = {len0, len1, len2};
  for (int l = 0; l < LAYERS3; l++)
    for (int e = 0; e < EXPERTS8; e++) span[l][e][0] = (FileSpan){zero, 0, len[l]};
  st->layout_ready = 0, st->SLOT_SIZE = 0, st->sized = 0;  // the layout is read again
  memset(st->layer_bytes, 0, LAYERS3 * sizeof *st->layer_bytes);
  cache_required(st, LAYERS3);
  memset(slots, 0, sizeof slots);
  for (int i = 0; i < SLOTS; i++) scores[i] = INFINITY, frees[i] = SLOTS - 1 - i;
  st->slots = slots, st->score = scores, st->free_list = frees;
  st->max_slots = st->nfree = SLOTS, st->cap = st->mcap = st->max_cap = units;
  st->nslots = st->nmapped = 0, st->bytes = st->mapped_bytes = st->conversions = 0;
  st->head = st->tail = st->pass_layer = st->serving = -1;
  st->cache_busy = 0;
  st->token = 10, st->weight = st->near_weight = 1;
  memset(st->where, 0xff, WHERE_BYTES);
  memset(st->escore, 0, SCORE_BYTES), memset(st->dscore, 0, SCORE_BYTES), memset(st->nscore, 0, SCORE_BYTES);
  memset(&st->stats, 0, sizeof st->stats);
  added = removed = freed = 0;
}

// The store's slots in a memory pool of `limit` bytes, as cache_init registers them (setup takes
// them out again).
static void pool(uint64_t limit) {
  static int registered;
  memory_init(limit);
  if (!registered) memory_account_init(&account, "model"), registered = 1;
  memory_slot_init(&backbone, &account, NULL, "backbone", NULL, NULL);
  cache_memory(st, &backbone);
  for (int i = 0; i < SLOTS; i++) {
    slots[i].store = st;
    memory_slot_init(&slots[i].memory, &account, &backbone, "routed expert", pool_evict, &slots[i]);
    slots[i].memory.can_evict = pool_can_evict, slots[i].memory.automatic = 1;
  }
  pooled = 1;
}

static uint64_t held_bytes(void) {  // what the slots' buffers take, counted from the slots
  uint64_t bytes = 0;
  for (int i = 0; i < st->max_slots; i++)
    if (slots[i].b.obj) bytes += slots[i].b.size;
  return bytes;
}

int main(void) {
  // The layers' sizes: a span's aligned cover (REGION), the largest for the limits.
  setup(220000, 236000, 350000, 2);
  CHECK(st->sized && st->SLOT_SIZE == LARGE);
  CHECK(st->layer_bytes[0] == SMALL && st->layer_bytes[1] == MIDDLE && st->layer_bytes[2] == LARGE);
  CHECK(slot_need(st, 0) == SMALL && slot_need(st, 2) == LARGE);
  CHECK(slot_fits(MIDDLE, SMALL) && !slot_fits(SMALL, MIDDLE) && !slot_fits(LARGE, SMALL) && slot_fits(SMALL, SMALL));

  // Bytes, not slots: a limit of two of the largest slots (48 sectors) holds three small experts.
  int a = take(st, 0, 0, 0), b = take(st, 0, 1, 0), c = take(st, 0, 2, 0);
  CHECK(st->nslots == 3 && st->bytes == 3 * SMALL && held_bytes() == st->bytes);
  CHECK(slots[a].b.size == SMALL && slots[b].b.size == SMALL && slots[c].b.size == SMALL);
  CHECK(cache_bytes(st) == 3 * SMALL && cache_slot_bytes(st) == SMALL && cache_slots(st) == 3);
  CHECK(cache_cap(st) == 3 && cache_max(st) == 3);  // no further slot of the largest size fits
  // A fourth takes the lowest-priority slot: its buffer has the size.
  st->score[a] = 3, st->score[b] = 1, st->score[c] = 2;
  int d = take(st, 0, 3, 0);
  CHECK(d == b && st->nslots == 3 && st->bytes == 3 * SMALL && WHERE(0, 1) < 0 && WHERE(0, 3) == b && st->conversions == 0 && freed == 0);
  // A buffer a sixteenth larger than the expert is reused; a larger expert does not go into a smaller buffer.
  setup(220000, 236000, 350000, 2);
  a = take(st, 1, 0, 0), b = take(st, 1, 1, 0);  // two middle buffers, 34 of 48 sectors
  st->score[a] = 1, st->score[b] = 2;
  c = take(st, 0, 0, 0);  // a small expert: 16 sectors more do not fit, the lowest middle buffer does
  CHECK(c == a && slots[c].b.size == MIDDLE && st->nslots == 2 && st->bytes == 2 * MIDDLE && st->conversions == 0);

  // A conversion: no buffer holds the large expert, so the two lowest priorities give theirs back.
  setup(220000, 236000, 350000, 2);
  a = take(st, 0, 0, 0), b = take(st, 0, 1, 0), c = take(st, 0, 2, 0);
  st->score[a] = 5, st->score[b] = 1, st->score[c] = 2;
  d = take(st, 2, 0, 0);
  CHECK(st->conversions == 1 && freed == 2 && slots[d].b.size == LARGE);
  CHECK(WHERE(0, 0) == a && WHERE(0, 1) < 0 && WHERE(0, 2) < 0 && WHERE(2, 0) == d);
  CHECK(st->nslots == 2 && st->bytes == SMALL + LARGE && held_bytes() == st->bytes && st->stats.evictions == 2);
  // The other way: a large buffer would waste a third of itself on a small expert. With the small
  // buffer in use (held), the large one gives way to a small one, and the bytes left over hold a
  // second small expert without another eviction.
  st->score[a] = 9, st->score[d] = 1;
  slots[a].holds = 1;
  int e = take(st, 0, 3, 0);
  CHECK(st->conversions == 2 && slots[e].b.size == SMALL && WHERE(2, 0) < 0 && st->bytes == 2 * SMALL);
  int f = take(st, 0, 4, 0);
  CHECK(st->conversions == 2 && st->nslots == 3 && st->bytes == 3 * SMALL && WHERE(0, 0) == a && WHERE(0, 3) == e && WHERE(0, 4) == f);
  slots[a].holds = 0;

  // Among the lowest priorities a fitting buffer is reused rather than a buffer converted: 70 large
  // buffers and one small one, in a cache of 71 of the largest.
  for (int near = 0; near < 2; near++) {
    setup(220000, 236000, 350000, 71);
    for (int k = 0; k < 70; k++) {
      int i = take(st, 2, k % EXPERTS8, 0);
      place(st, 2, k % EXPERTS8, -1);  // 70 buffers of the large layer; which of its experts they hold does not matter
      slots[i].expert = -1, st->score[i] = k + 1;
    }
    int small = take(st, 0, 0, 0);
    // near: 50 of the large buffers are lower, fewer than FIT_RANK, so the small buffer is taken.
    // Otherwise all 70 are lower, and the lowest large buffer is given back for a small one.
    st->score[small] = near ? 50.5 : 100;
    int before = st->nslots;
    int got = take(st, 0, 1, 0);
    if (near) CHECK(got == small && st->conversions == 0 && WHERE(0, 0) < 0 && st->nslots == before);
    else CHECK(got != small && st->conversions == 1 && WHERE(0, 0) == small && slots[got].b.size == SMALL && freed == 1 && st->nslots == before);
  }

  // What a conversion may give back when decoding: not an expert used or read ahead this token.
  setup(220000, 236000, 350000, 2);
  a = take(st, 0, 0, 0), b = take(st, 0, 1, 0), c = take(st, 0, 2, 0);
  st->score[a] = 1, st->score[b] = 2, st->score[c] = 3;
  slots[a].used = st->token;  // the lowest was used this token (by a layer already served)
  d = take(st, 2, 0, 0);
  CHECK(st->conversions == 1 && WHERE(0, 0) == a && WHERE(0, 1) < 0 && WHERE(0, 2) < 0 && slots[d].b.size == LARGE);
  // A read-ahead gives at most two buffers back, and none that is still protected.
  setup(220000, 236000, 350000, 2);
  a = take(st, 0, 0, 0), b = take(st, 0, 1, 0), c = take(st, 0, 2, 0);
  st->score[a] = 1, st->score[b] = 2, st->score[c] = 3;
  slots[b].used = slots[c].used = st->token;
  CHECK(take(st, 2, 0, 1) == -1 && WHERE(0, 1) == b && WHERE(0, 2) == c);  // one freeable buffer is too little
  CHECK(st->bytes == 2 * SMALL && st->nslots == 2);  // what it gave back stays free for the next request
  slots[b].used = slots[c].used = st->token - 1;
  d = take(st, 2, 0, 1);
  CHECK(d >= 0 && slots[d].b.size == LARGE && st->bytes == SMALL + LARGE);
  // A demand that can give nothing back reuses the lowest buffer that is large enough.
  setup(220000, 236000, 350000, 2);
  a = take(st, 2, 0, 0), b = take(st, 2, 1, 0);
  st->score[a] = 1, st->score[b] = 2;
  slots[a].used = slots[b].used = st->token, st->serving = -1;
  c = take(st, 0, 0, 0);
  CHECK(c == a && slots[c].b.size == LARGE && st->conversions == 0 && WHERE(2, 0) < 0 && WHERE(0, 0) == a);
  // Among the buffers that are large enough it keeps the victim's preferences: a read-ahead that is
  // still protected stays while another slot will do, although its priority is lower.
  setup(220000, 236000, 350000, 2);
  a = take(st, 2, 0, 0), b = take(st, 2, 1, 0);
  st->score[a] = 1, st->score[b] = 2;
  slots[a].used = slots[b].used = st->token, slots[a].prefetched = 1;
  c = take(st, 0, 0, 0);
  CHECK(c == b && WHERE(2, 0) == a && WHERE(2, 1) < 0 && WHERE(0, 0) == b && st->conversions == 0);
  // When only the read-ahead's buffer is large enough, the demand takes it.
  setup(220000, 236000, 350000, 2);
  a = take(st, 0, 0, 0), b = take(st, 2, 0, 0);  // 40 of 48 sectors
  slots[a].used = slots[b].used = st->token, slots[b].prefetched = 1;
  c = take(st, 1, 0, 0);  // a middle expert: the small buffer cannot hold it and nothing can be given back
  CHECK(c == b && WHERE(2, 0) < 0 && WHERE(0, 0) == a && WHERE(1, 0) == b && st->conversions == 0 && st->nslots == 2);

  // A prompt pass allocates buffers of the largest size, which serve all its layers; decoding its
  // experts' sizes.
  setup(220000, 236000, 350000, 3);
  cache_pass(st, 0);
  a = take(st, 0, 0, 0);
  cache_pass(st, -1);
  b = take(st, 0, 1, 0);
  CHECK(slots[a].b.size == LARGE && slots[b].b.size == SMALL && st->bytes == LARGE + SMALL);
  // A prompt pass at layer 2 over a cache that decoding filled. Of the slots it may take (the layers it
  // is done with, all of one priority before any decoding), the least recently used whose buffer is
  // large enough goes first.
  setup(220000, 236000, 350000, 3);
  a = take(st, 0, 0, 0), b = take(st, 1, 0, 0), c = take(st, 0, 1, 0), d = take(st, 1, 1, 0);  // 66 of 72 sectors
  cache_pass(st, 2);
  int m = take(st, 1, 2, 0);  // a middle expert: the oldest slot (a) is too small, the oldest large enough is b
  CHECK(m == b && WHERE(1, 0) < 0 && WHERE(0, 0) == a && st->conversions == 0 && st->bytes == 2 * SMALL + 2 * MIDDLE);
  // An expert of the pass's own layer finds no buffer large enough. Buffers go back, the least recently
  // used first, but none of the layer before the pass's own.
  int big = take(st, 2, 0, 0);
  CHECK(st->conversions == 1 && slots[big].b.size == LARGE && WHERE(0, 0) < 0 && WHERE(0, 1) < 0);
  CHECK(WHERE(1, 1) == d && WHERE(1, 2) == b && st->bytes == 2 * MIDDLE + LARGE && held_bytes() == st->bytes);
  // A smaller expert of the pass goes into the large buffer it is done with: no buffer is given back.
  cache_pass(st, 3);  // (the store has three layers; the pass's layer only orders the victims here)
  int again = take(st, 0, 5, 0);
  CHECK(st->conversions == 1 && st->bytes == 2 * MIDDLE + LARGE && slots[again].b.size >= SMALL);
  cache_pass(st, -1);
  // A prompt pass that can give nothing back reuses a buffer that is large enough from a layer it is
  // done with, not one of its own layer whose expert decoding used less.
  setup(220000, 236000, 350000, 3);
  a = take(st, 0, 0, 0);  // a small buffer, as decoding allocates it
  cache_pass(st, 0);
  b = take(st, 0, 1, 0);  // the pass's buffers have the largest size
  cache_pass(st, 1);
  c = take(st, 1, 0, 0);  // 64 of 72 sectors
  st->dscore[0 * EXPERTS8 + 1] = 5, st->score[b] = 5;  // decoding used b's expert, and not c's
  d = take(st, 1, 1, 0);  // a (the victim) is too small, and no buffer of layers 0 and 1 may be given back
  CHECK(d == b && WHERE(0, 1) < 0 && WHERE(1, 0) == c && WHERE(0, 0) == a && WHERE(1, 1) == b);
  CHECK(st->conversions == 0 && st->bytes == SMALL + 2 * LARGE && freed == 0);
  cache_pass(st, -1);
  // Resizing counts bytes: the limit falls to one largest slot, and the least recently used leave until
  // the rest fit; the mapped bytes follow their own limit.
  setup(220000, 236000, 350000, 3);
  int held[4];
  for (int k = 0; k < 4; k++) held[k] = take(st, 0, k, 0);  // 64 of 72 sectors
  uint32_t at[STORE_PARTS];
  int fresh;
  for (int k = 0; k < 4; k++) cache_get(st, 0, k, at, &fresh);
  CHECK(st->nmapped == 4 && st->mapped_bytes == 4 * SMALL);
  st->cap = 1;
  for (int i = st->tail; over_cap(st) && i >= 0;) {  // as resize gives slots back
    int prev = slots[i].prev;
    release(st, i);
    i = prev;
  }
  CHECK(st->nslots == 1 && st->bytes == SMALL && st->nmapped == 1 && st->mapped_bytes == SMALL && WHERE(0, 3) == held[3]);
  CHECK(used_units(st) == 1 && mapped_units(st) == 1 && !over_cap(st) && !over_mcap(st));
  st->mcap = 0;
  CHECK(over_mcap(st));

  // Experts are read into buffers of their own size: the reads stay inside them.
  setup(220000, 236000, 350000, 4);
  int ids[] = {0, 1};
  cache_load(st, 0, ids, 2);
  cache_load(st, 2, ids, 2);
  for (int k = 0; k < 2; k++) {
    Slot *s0 = &slots[WHERE(0, k)], *s2 = &slots[WHERE(2, k)];
    CHECK(s0->b.size == SMALL && s2->b.size == LARGE && s0->extent <= SMALL && s2->extent <= LARGE);
    CHECK(s0->b.ptr[s0->at[0]] == 0 && s0->b.ptr[s0->at[0] + 219999] == 0 && s2->b.ptr[s2->at[0] + 349999] == 0);
  }
  cache_drain();

  // Layers within a sixteenth of the largest keep slots of one size, the largest.
  setup(220000, 236000, 236000, 2);
  CHECK(!st->sized && st->SLOT_SIZE == MIDDLE && st->layer_bytes[0] == SMALL && slot_need(st, 0) == MIDDLE);
  a = take(st, 0, 0, 0), b = take(st, 1, 0, 0);
  CHECK(slots[a].b.size == MIDDLE && slots[b].b.size == MIDDLE && st->nslots == 2);
  st->score[a] = 2, st->score[b] = 1;
  CHECK(take(st, 0, 1, 0) == b && st->nslots == 2 && st->conversions == 0);
  // A store whose layers have one size keeps slots: its limits count them.
  setup(220000, 220000, 220000, 2);
  CHECK(!st->sized && st->SLOT_SIZE == SMALL && slot_need(st, 1) == SMALL);
  a = take(st, 0, 0, 0), b = take(st, 1, 0, 0);
  st->score[a] = 1, st->score[b] = 2;
  c = take(st, 2, 0, 0);
  CHECK(c == a && st->nslots == 2 && st->conversions == 0 && cache_cap(st) == 2 && cache_bytes(st) == 2 * SMALL);

  // In a memory pool that admits 72 sectors, under a cache limit that is not reached: the pool's
  // refusal ends the growth, and a conversion gives back only what the new buffer needs.
  setup(220000, 236000, 350000, 8);
  pool(72 * SECTOR);
  int four[5];
  for (int k = 0; k < 4; k++) four[k] = take(st, 0, k, 0), st->score[four[k]] = k + 1;
  CHECK(st->nslots == 4 && st->bytes == 4 * SMALL && memory_usage().allocated == 4 * SMALL);
  four[4] = take(st, 0, 4, 0);  // the pool refuses a fifth buffer: the lowest slot's fits
  CHECK(four[4] == four[0] && st->nslots == 4 && st->conversions == 0 && memory_usage().allocated == 4 * SMALL);
  st->score[four[0]] = 9;
  big = take(st, 2, 0, 0);  // one small buffer back makes the room: 3 small and 1 large are the 72 sectors
  CHECK(st->conversions == 1 && freed == 1 && slots[big].b.size == LARGE && st->nslots == 4 && WHERE(0, 1) < 0);
  CHECK(st->bytes == 3 * SMALL + LARGE && memory_usage().allocated == st->bytes && held_bytes() == st->bytes);
  // A pool over its limit admits no further buffer. A demand gives at most four buffers back. No
  // buffer is large enough for it, so it takes its buffer as required memory, and the pool makes the
  // room by taking two further slots back. The buffer belongs to the slot it is charged to.
  setup(220000, 236000, 350000, 12);
  pool(128 * SECTOR);
  for (int k = 0; k < 8; k++) st->score[take(st, 0, k, 0)] = k + 1;
  CHECK(st->nslots == 8 && memory_usage().allocated == 8 * SMALL);
  memory_set_limit(64 * SECTOR);
  big = take(st, 2, 0, 0);
  CHECK(freed == CONVERT_DEMAND + 2 && st->stats.given_back == 2 && st->nslots == 3 && st->nfree == SLOTS - 3);
  CHECK(slots[big].b.size == LARGE && WHERE(2, 0) == big && slots[big].b.memory == &slots[big].memory);
  CHECK(st->bytes == 2 * SMALL + LARGE && held_bytes() == st->bytes && memory_usage().allocated == st->bytes);
  for (int k = 0; k < CONVERT_DEMAND; k++) CHECK(WHERE(0, k) < 0);  // the lowest priorities went first
  for (int i = 0; i < SLOTS; i++) CHECK(slots[i].memory.payload == (slots[i].b.obj ? slots[i].b.size : 0));
  // A read-ahead in a shortage, with the large buffer in use, gives two back and gets no slot.
  memory_set_limit(16 * SECTOR);
  freed = 0, slots[big].holds = 1;
  CHECK(take(st, 2, 1, 1) == -1 && freed == CONVERT_OPTIONAL && st->nslots == 1 && st->bytes == LARGE);
  slots[big].holds = 0;

  // A busy cache whose pool has no room even for required memory: two slots are held and two were
  // used this token, so the pool may take none back. The demand gives back what nothing holds, the
  // lowest priority first, until its buffer fits; the program does not stop.
  setup(220000, 236000, 350000, 4);
  pool(64 * SECTOR);
  for (int k = 0; k < 4; k++) four[k] = take(st, 0, k, 0), st->score[four[k]] = k + 1;
  slots[four[0]].holds = slots[four[1]].holds = 1;
  slots[four[2]].used = slots[four[3]].used = st->token;
  cache_active(st, 1);
  big = take(st, 2, 0, 0);
  CHECK(slots[big].b.size == LARGE && WHERE(2, 0) == big && slots[big].b.memory == &slots[big].memory);
  CHECK(WHERE(0, 0) == four[0] && WHERE(0, 1) == four[1] && WHERE(0, 2) < 0 && WHERE(0, 3) < 0);
  CHECK(st->nslots == 3 && st->bytes == 2 * SMALL + LARGE && memory_usage().allocated == st->bytes && held_bytes() == st->bytes);
  slots[four[0]].holds = slots[four[1]].holds = 0;
  // With every other slot held, the demand has given back what nothing held and still has no room:
  // its pass waits, as a pass does when every slot is held, until another pass releases a slot.
  setup(220000, 236000, 350000, 4);
  pool(40 * SECTOR);
  a = take(st, 2, 0, 0), b = take(st, 0, 0, 0);  // the pool's 40 sectors
  slots[a].holds = 1, slots[b].used = st->token;
  cache_active(st, 1);
  wanted[0] = 2, wanted[1] = 1;
  Pass *waiting = pass_new(demand, NULL, 1 << 20);
  CHECK(!pass_resume(waiting) && got == -9 && !pass_ready(waiting));  // the pass yields
  CHECK(WHERE(0, 0) < 0 && WHERE(2, 0) == a && st->nslots == 1 && st->bytes == LARGE);
  slots[a].holds = 0;  // the other pass's GPU work has finished
  CHECK(pass_ready(waiting));
  CHECK(pass_resume(waiting) && got == a && WHERE(2, 1) == a && WHERE(2, 0) < 0 && st->nslots == 1 && st->bytes == LARGE);
  pass_free(waiting);
  cache_active(st, 0);

  printf("cache slot sizes: %d/%d checks passed\n", checks - failures, checks);
  return failures != 0;
}
