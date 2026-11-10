// Bounded draft-cache checks without a model, files, or GPU execution.
// Empty read spans exercise demand admission and group ownership without I/O.
#include "../../src/model/deepseek/model.h"
#include <sys/wait.h>
#include "../../src/engine/io.c"

static int checks, failures;
static Slot fixture[2 + DRAFTS * DRAFT_EXPERTS];
static double fixture_scores[2 + DRAFTS * DRAFT_EXPERTS];

static void expect(const char *name, uint64_t got, uint64_t want) {
  checks++;
  if (got == want) return;
  fprintf(stderr, "%s: expected %llu, got %llu\n", name, (unsigned long long)want, (unsigned long long)got);
  failures++;
}

static void clear_fixture(void) {
  if (st->slots)
    for (int i = 0; i < st->max_slots + st->npinned; i++) finish(i);
  st->slots = NULL;
  st->score = NULL;
}

static void setup(int capacity) {
  clear_fixture();
  cache_draft_limit(capacity);
  memset(fixture, 0, sizeof fixture);
  memset(fixture_scores, 0, sizeof fixture_scores);
  memset(st->where, 0xff, sizeof st->where);
  memset(st->escore, 0, sizeof st->escore);
  memset(&cache_stats, 0, sizeof cache_stats);
  st->slots = fixture, st->score = fixture_scores;
  st->max_slots = st->cap = st->nslots = 2;
  st->npinned = draft_pool_count(CACHE_LAYERS);
  st->SLOT_SIZE = 4096; // backbone's smaller slots must not be reused for drafts
  st->pinned_bytes = (uint64_t)st->npinned * 8192;
  st->token = 1, st->draft_clock = st->draft_touch_clock = 0, st->weight = 1;
  memset(st->draft_age, 0, sizeof st->draft_age);
  atomic_store(&st->prefetching, 0), atomic_store(&st->demand, 0);
  st->qhead = st->qtail = 0;
  for (int i = 0; i < st->max_slots; i++) {
    st->slots[i].layer = 0, st->slots[i].expert = 100 + i;
    st->where[0][100 + i] = i;
  }
  for (int i = st->max_slots; i < st->max_slots + st->npinned; i++) {
    Slot *s = &st->slots[i];
    s->layer = LAYERS + (i - st->max_slots) / st->draft_capacity;
    s->expert = st->draft_capacity == DRAFT_EXPERTS ? (i - st->max_slots) % st->draft_capacity : -1;
    s->pinned = 1;
    s->b.obj = (void *)1; // preallocated fixture storage; no Metal operations use it
    s->b.size = 8192;
    s->b.gpu = 0x100000 + (uint64_t)i * 8192;
    if (s->expert >= 0) st->where[s->layer][s->expert] = i;
  }
  for (int d = 0; d < DRAFTS; d++) expert_spans[LAYERS + d] = 0;
}

static void fill_stage(void) {
  int ids[16];
  for (int i = 0; i < 16; i++) ids[i] = i;
  cache_load(LAYERS, ids, 16);
  for (int e = 0; e < 16; e++) st->slots[st->where[LAYERS][e]].used = e + 1;
  st->draft_clock = 100;
}

static void invalid_limit(void) { cache_draft_limit(14); }
static void excessive_union(void) {
  int ids[17];
  for (int i = 0; i < 17; i++) ids[i] = i;
  cache_request(LAYERS, ids, 17);
}
static void invalid_expert(void) {
  int e = DRAFT_EXPERTS;
  cache_request(LAYERS, &e, 1);
}

static void rejected(const char *name, void (*operation)(void)) {
  pid_t pid = fork();
  if (pid < 0) die("cannot fork draft-cache check");
  if (!pid) operation(), _exit(0);
  int status;
  if (waitpid(pid, &status, 0) != pid) die("cannot wait for draft-cache check");
  expect(name, WIFEXITED(status) && WEXITSTATUS(status) != 0, 1);
}

int main(void) {
  cache_draft_limit(0);
  expect("default retains 128 experts", cache_draft_capacity(), 128);
  expect("no draft reserves no slots", draft_pool_count(LAYERS), 0);
  expect("default reserves all stages", draft_pool_count(CACHE_LAYERS), 384);
  rejected("capacity below complete row union rejected", invalid_limit);
  for (int c = 32; c <= 128; c *= 2) {
    cache_draft_limit(c);
    expect("stage capacity reserved separately", draft_pool_count(CACHE_LAYERS), 3 * c);
  }

  setup(16);
  expect("cache bytes include separate larger draft buffers", cache_bytes(), 2 * 4096 + 48 * 8192);
  rejected("oversized union rejected before admission", excessive_union);
  rejected("out-of-range expert rejected", invalid_expert);
  int first[] = {1, 2, 3};
  expect("empty-span demand reads finish immediately", cache_request(LAYERS, first, 3), 0);
  expect("first demands are misses", cache_stats.misses, 3);
  expect("draft misses are not prefetches", cache_stats.prefetches, 0);
  expect("draft reads do not enter prefetch queue", st->qtail, 0);
  expect("no test bytes read", cache_stats.bytes, 0);
  for (int e = 1; e <= 3; e++) {
    int at = st->where[LAYERS][e], fresh;
    uint32_t spans[3];
    const Buf *buffer = cache_get(LAYERS, e, spans, &fresh);
    expect("draft uses its own stage pool", at >= 2 && at < 18, 1);
    expect("draft buffer keeps larger format size", buffer->size, 8192);
    expect("resident pool does not need remapping", fresh, 0);
    expect("completed group is ready before release", cache_ready(LAYERS, e), 1);
  }
  cache_wait(LAYERS, first, 3);
  expect("consumer releases completed group", st->slots[st->where[LAYERS][1]].loading == NULL, 1);

  setup(16);
  fill_stage();
  int wanted[] = {16, 0}; // new expert appears before an old needed LRU expert
  cache_load(LAYERS, wanted, 2);
  expect("entire current union protected", cache_slot(LAYERS, 0), 1);
  expect("unneeded least-recent expert evicted", cache_slot(LAYERS, 1), 0);
  expect("new native expert loaded", cache_ready(LAYERS, 16), 1);
  expect("other stage is untouched", cache_slot(LAYERS + 1, 16), 0);
  expect("backbone slot is untouched", st->where[0][100], 0);
  int skip = 120;
  cache_prefetch(LAYERS, &skip, 1, 6);
  expect("bounded draft prefetch is disabled", cache_slot(LAYERS, skip), 0);

  int pending = st->where[LAYERS][2];
  st->slots[pending].used = 0;
  st->draft_age[pending - st->max_slots] = 0;
  st->slots[pending].loading = dispatch_group_create();
  dispatch_group_enter(st->slots[pending].loading);
  st->slots[pending].npiece = 1;
  atomic_store(&st->slots[pending].taken[0], 1);
  int another = 17;
  cache_load(LAYERS, &another, 1);
  expect("pending read is not a victim", st->where[LAYERS][2], pending);
  expect("pending group remains owned", slot_reading(pending), 1);
  int same = 2;
  expect("request reports pending draft hit", cache_request(LAYERS, &same, 1), 1);
  expect("pending draft hit starts no duplicate read", atomic_load(&st->demand), 0);
  dispatch_group_leave(st->slots[pending].loading);
  cache_wait(LAYERS, &same, 1);
  expect("pending group released only by consumer", st->slots[pending].loading == NULL, 1);

  for (int i = st->max_slots; i < st->max_slots + st->draft_capacity; i++) st->slots[i].used = 9999;
  expect("victim cannot escape its protected stage", draft_victim(LAYERS, 9999) == -1, 1);

  setup(16);
  fill_stage();
  int completed = st->where[LAYERS][4];
  uint64_t address = st->slots[completed].b.gpu;
  st->slots[completed].used = 0;
  st->draft_age[completed - st->max_slots] = 0;
  st->slots[completed].loading = dispatch_group_create();
  dispatch_group_enter(st->slots[completed].loading);
  dispatch_group_leave(st->slots[completed].loading);
  int replacement = 100;
  cache_load(LAYERS, &replacement, 1);
  expect("completed unreleased slot can be reused", st->where[LAYERS][replacement], completed);
  expect("reused buffer keeps GPU address", st->slots[completed].b.gpu, address);
  expect("old expert lookup removed", cache_slot(LAYERS, 4), 0);
  expect("replacement read group released", st->slots[completed].loading == NULL, 1);

  int stage_ids[16];
  for (int i = 0; i < 16; i++) stage_ids[i] = i;
  cache_load(LAYERS + 1, stage_ids, 16);
  int touch_order[] = {7, 2};
  cache_load(LAYERS + 1, touch_order, 2);
  for (int e = 0; e < 16; e++)
    if (e != 7 && e != 2) st->slots[st->where[LAYERS + 1][e]].used = 9999;
  expect("LRU retains request order within a pass", draft_victim(LAYERS + 1, 9999), st->where[LAYERS + 1][7]);

  setup(128);
  int resident = 7;
  expect("full pinned path remains ready", cache_request(LAYERS, &resident, 1), 0);
  expect("full pinned path keeps its hit accounting", cache_stats.hits, 1);
  expect("full pinned path has no demand miss", cache_stats.misses, 0);
  expect("full pinned index layout unchanged", st->where[LAYERS][resident], st->max_slots + resident);
  clear_fixture();
  fprintf(stderr, "draft cache: %d/%d checks passed\n", checks - failures, checks);
  return failures != 0;
}
