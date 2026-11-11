// Retention policy checks use empty reads and in-memory slots; no GPU or model.
#include "../../src/model/deepseek/model.h"
#include <sys/wait.h>
#include "../../src/engine/io.c"
#include "cache_layout.h"

static Slot fixture[3];
static double fixture_score[3];
static int checks, failures;

static void expect(const char *name, int condition) {
  checks++;
  if (condition) return;
  fprintf(stderr, "%s failed\n", name);
  failures++;
}

static int near(double a, double b) { return fabs(a - b) <= 1e-12 * fmax(fabs(b), 1e-300); }

static void setup(int half_life, int cost) {
  if (st->slots)
    for (int i = 0; i < st->max_slots; i++) finish(st, i);
  st->slots = NULL;
  cache_retention(st, half_life, cost);
  memset(fixture, 0, sizeof fixture);
  memset(st->where, 0xff, WHERE_BYTES);
  memset(st->escore, 0, SCORE_BYTES);
  memset(st->dscore, 0, SCORE_BYTES);
  memset(st->nscore, 0, SCORE_BYTES);
  st->near_weight = 1;
  memset(expert_spans, 0, sizeof expert_spans);
  memset(&st->stats, 0, sizeof st->stats);
  st->slots = fixture, st->score = fixture_score;
  st->max_slots = st->cap = st->nslots = 3, st->npinned = st->nfree = 0;
  st->head = 2, st->tail = 0, st->token = 10, st->weight = 1;
  st->pass_layer = st->serving = -1;
  st->pf_enabled = 0;
  for (int i = 0; i < 3; i++) {
    st->slots[i].layer = 0, st->slots[i].expert = i, st->slots[i].mapped = 1;
    st->slots[i].prev = i == 2 ? -1 : i + 1;
    st->slots[i].next = i == 0 ? -1 : i - 1;
    WHERE(0, i) = i;
    st->score[i] = st->escore[i] = i + 1;
    st->reload_cost[i] = 1;
  }
}

static void bad_half(void) {
  st->slots = NULL;
  cache_retention(st, -1, 0);
}
static void bad_cost(void) {
  st->slots = NULL;
  cache_retention(st, 50, 2);
}
static void too_large_half(void) {
  st->slots = NULL;
  cache_retention(st, CACHE_HALF_LIFE_MAX + 1, 0);
}
static void too_late(void) { cache_retention(st, 10, 0); }

static void rejected(const char *name, void (*operation)(void)) {
  pid_t pid = fork();
  if (pid < 0) die("cannot fork cache-policy check");
  if (!pid) operation(), _exit(0);
  int status;
  if (waitpid(pid, &status, 0) != pid) die("cannot wait for cache-policy check");
  expect(name, WIFEXITED(status) && WEXITSTATUS(status) != 0);
}

// A reader has started the cheapest victim's last piece, but routing has
// demoted its prefetch. Optional work must leave it alone rather than wait
// for that read or evict a more valuable resident. Demand can wait and reuse it.
static void *complete_read(void *group) {
  usleep(100000);
  dispatch_group_leave(group);
  return NULL;
}

static void busy_victim(int optional, int prefill) {
  setup(10, 0);
  st->pass_layer = prefill ? 1 : -1;
  st->slots[0].loading = dispatch_group_create();
  st->slots[0].npiece = 1;
  atomic_store(&st->slots[0].taken[0], 1);
  dispatch_group_enter(st->slots[0].loading);
  pthread_t reader;
  if (pthread_create(&reader, NULL, complete_read, st->slots[0].loading)) die("cannot start fixture reader");
  double began = now();
  int got = take(st, 2, 0, optional);
  double elapsed = now() - began;
  pthread_join(reader, NULL);
  fprintf(stderr, "cache policy: %s %s with busy victim: slot %d, %.3f ms\n",
          prefill ? "prefill" : "decode", optional ? "prefetch" : "demand", got, elapsed * 1000);
  if (optional) {
    expect("prefetch declines a victim still being read", got == -1);
    expect("declined prefetch preserves all residents", WHERE(0, 0) == 0 && WHERE(0, 1) == 1 &&
           WHERE(0, 2) == 2 && WHERE(2, 0) == -1 && st->stats.evictions == 0);
    finish(st, 0);
    expect("prefetch can use the same victim once its read ends", take(st, 2, 0, 1) == 0);
  } else {
    expect("demand still waits for and reuses the cheapest victim", got == 0 && !st->slots[0].loading &&
           WHERE(0, 0) == -1 && WHERE(2, 0) == 0 && st->stats.evictions == 1);
  }
}

// A prediction for layer 1 is refined while layer 0 is still serving.
// Its canceled slot must remain reusable when actual routing advances to 1.
static void refined_empty_setup(void) {
  setup(10, 0);
  memset(st->where, 0xff, WHERE_BYTES);
  st->slots[0].layer = 1, st->slots[0].expert = 0;
  st->slots[1].layer = 0, st->slots[1].expert = 1;
  st->slots[2].layer = 1, st->slots[2].expert = 2;
  WHERE(1, 0) = 0, WHERE(0, 1) = 1, WHERE(1, 2) = 2;
  st->score[0] = st->escore[EXPERTS] = 1;
  st->score[2] = st->escore[EXPERTS + 2] = 3;
  st->slots[0].prefetched = 1, st->slots[0].used = st->token;
  st->slots[0].npiece = 1, st->slots[0].background_load = 1;
  atomic_store(&st->slots[0].taken[0], 0);
  atomic_store(&st->slots[0].unfinished, 1);
  atomic_store(&st->prefetching, 1);
  st->slots[0].loading = dispatch_group_create();
  dispatch_group_enter(st->slots[0].loading);
  st->serving = 0;
  int keep = 2;
  cache_refine(st, 1, &keep, 1);
  expect("refinement leaves an empty completed slot", WHERE(1, 0) == -1 && st->slots[0].expert == -1 &&
         st->score[0] == -INFINITY && !st->slots[0].loading && !atomic_load(&st->prefetching));
  // This is the next operation in read_ahead/read_residual_ahead. Already
  // resident candidates do not consume the slot before actual routing.
  cache_prefetch(st, 1, &keep, 1, 6);
  expect("resident forecast leaves the refined slot available", st->slots[0].expert == -1 && WHERE(1, 2) == 2);
}

static void refined_empty_reuse(void) {
  refined_empty_setup();
  int routed[] = {2, 3};
  cache_request(st, 1, routed, 2);
  expect("actual routing reuses its layer's refined empty slot", WHERE(1, 3) == 0);
  expect("refined empty reuse preserves another layer's ready expert", WHERE(0, 1) == 1);
  expect("refined empty reuse preserves the selected resident", WHERE(1, 2) == 2 && st->slots[2].used == st->token);
  expect("refinement regression performs no I/O", st->stats.bytes == 0);

  refined_empty_setup();
  st->serving = 1;
  expect("the serving layer permits a refined empty slot", victim(st) == 0);
  st->slots[0].holds = 1;
  expect("a reservation still excludes an otherwise empty slot", victim(st) == 1);
  st->slots[0].holds = 0;
  expect("releasing the reservation restores empty-slot eligibility", victim(st) == 0);
}

int main(void) {
  st = deepseek_cache_create();
  alarm(20);
  setup(0, 0);
  expect("zero selects default half-life", st->retention_half_life == 50);
  expect("default ratio retains original constant", st->retention_growth == exp2(1.0 / 50));
  for (int i = 0; i < 50; i++) advance_score_clock(st);
  expect("default clock doubles weight over 50 advances", near(st->weight, 2));
  expect("clock counts forward advances", st->token == 60);
  expect("aging preserves raw history between rescalings", st->escore[0] == 1);
  expect("aging refreshes a slot's priority from its history", near(st->score[0], 0.5));
  rejected("negative half-life rejected", bad_half);
  rejected("excessive half-life rejected", too_large_half);
  rejected("invalid weighting flag rejected", bad_cost);
  rejected("live policy mutation rejected", too_late);

  setup(10, 0);
  for (int i = 0; i < 10; i++) advance_score_clock(st);
  expect("ten-pass clock doubles weight", near(st->weight, 2));
  expect("unweighted priority is raw score", eviction_priority(st, 1) == st->score[1]);
  expect("default victim is lowest raw score", victim(st) == 0);
  st->slots[0].prefetched = 1;
  expect("prefetch protection unchanged", victim(st) == 1);
  st->serving = 0, st->slots[1].used = st->token;
  expect("current union protection unchanged", victim(st) == 2);
  st->serving = -1, st->slots[0].prefetched = 0;
  st->score[0] = st->score[1] = 2, st->score[2] = 3;
  expect("equal priorities retain physical-slot tie break", victim(st) == 0);
  st->score[2] = -INFINITY, st->slots[2].expert = -1;
  expect("empty slots still go first", victim(st) == 2);
  st->score[2] = INFINITY, st->slots[2].layer = 32767, st->slots[2].expert = 32767;
  expect("unallocated slots stay ineligible", victim(st) == 0);

  setup(10, 1);
  st->reload_cost[0] = 1, st->reload_cost[1] = .25, st->reload_cost[2] = 1;
  expect("cost weighting retains expensive lower-frequency expert", victim(st) == 1);
  expect("priority multiplication leaves raw resident score", st->score[1] == 2);
  expect("priority multiplication leaves raw ghost history", st->escore[1] == 2);
  for (int i = 0; i < 3; i++) st->reload_cost[i] = .5;
  expect("uniform read costs preserve victim", victim(st) == 0);

  // A near miss counts NEAR_CREDIT of a selection at once and fades with a
  // half-life of NEAR_HALF_LIFE steps against the selections' 50: slot 0's
  // expert, selected once, outranks slot 1's, selected twice, while its three
  // near misses are fresh, and no longer NEAR_HALF_LIFE steps later.
  setup(0, 0);
  st->score[0] = st->escore[0] = 1, st->score[1] = st->escore[1] = 2, st->score[2] = st->escore[2] = 4;
  expect("without near misses the least selected expert goes", victim(st) == 0);
  for (int k = 0; k < 3; k++) cache_near(st, 0, (int[]){0}, 1);
  expect("near misses raise a resident expert's priority at once", near(st->score[0], 1 + 3 * NEAR_CREDIT));
  expect("fresh near misses keep an expert", victim(st) == 1);
  for (int k = 0; k < NEAR_HALF_LIFE; k++) advance_score_clock(st);
  double fade = exp2(-(double)NEAR_HALF_LIFE / CACHE_HALF_LIFE_DEFAULT);
  expect("aging refreshes both parts of the priorities",
         near(st->score[0], fade + 3 * NEAR_CREDIT / 2) && near(st->score[1], 2 * fade));
  expect("near misses fade faster than selections", victim(st) == 0);
  WHERE(0, 3) = -1;
  cache_near(st, 0, (int[]){3}, 1);
  expect("an expert without a slot keeps its near misses", st->nscore[3] > 0);
  cache_near(st, st->scored_layers, (int[]){0}, 1);
  expect("layers after the scored ones take no near misses", st->nscore[(size_t)st->scored_layers * st->experts] == 0);

  // A prompt pass keeps what decoding uses most: its victim is the slot of a
  // layer the pass is done with whose expert decoding used least, not the
  // least recently used one, else of a layer after the next. Slot 0 is the
  // least recently used, and the one decoding uses most.
  setup(10, 0);
  st->slots[0].layer = 0, st->slots[1].layer = 1, st->slots[2].layer = 3;
  st->score[0] = 9, st->score[1] = 2, st->score[2] = 1;
  st->dscore[0] = 9, st->dscore[EXPERTS + 1] = 2, st->dscore[3 * EXPERTS + 2] = 1;
  st->pass_layer = 2;
  expect("prefill evicts the done layers' slot decoding used least", victim(st) == 1);
  st->pass_layer = 0;
  expect("prefill fallback protects current and next layer", victim(st) == 2);
  st->slots[2].layer = 1;
  expect("prefill finds no victim in its current and next layers", victim(st) == -1);
  st->pass_layer = -1;

  // A first prompt, with no decoding history: the prompt credits an expert
  // once for each of its batches that selects it, so under --ced the encoder
  // layers' experts (every batch) score above the decoder layers' (the last
  // ring's one batch). A prompt pass ranks by decoding's use alone, and among
  // equals takes the least recently used slot: the earlier layer's slot 0,
  // not the later layer's lower-scored slot 1, whose expert the decoding
  // after the prompt needs.
  setup(10, 0);
  st->slots[0].layer = 0, st->slots[1].layer = 1, st->slots[2].layer = 3;
  st->score[0] = st->escore[0] = 4, st->score[1] = st->escore[EXPERTS + 1] = 1, st->score[2] = 1;
  st->pass_layer = 2;
  expect("a first prompt's pass evicts the least recently used slot", victim(st) == 0);
  st->pass_layer = -1;

  // Between slots decoding used alike, recency decides, not the read cost.
  setup(10, 0);
  st->slots[2].layer = 3;
  st->score[0] = st->score[1] = 5, st->score[2] = 1;
  expert_spans[0] = 1;
  expert_span[0][0][0].len = 200, expert_span[0][1][0].len = 100;
  st->pass_layer = 2;
  expect("prefill breaks a decoding tie by recency", victim(st) == 0);
  expert_spans[0] = 0, st->pass_layer = -1;

  // Decoding's use outranks recency: an older slot decoding used stays.
  setup(10, 0);
  st->slots[0].layer = 0, st->slots[1].layer = 1, st->slots[2].layer = 3;
  st->dscore[0] = 3;
  st->pass_layer = 2;
  expect("prefill keeps an older slot that decoding used", victim(st) == 1);
  st->pass_layer = -1;

  setup(10, 1);
  st->SLOT_SIZE = 65536, expert_spans[0] = 2;
  for (int e = 0; e < EXPERTS; e++) {
    expert_span[0][e][0] = (FileSpan){.off = 1, .len = 16384};
    expert_span[0][e][1] = (FileSpan){.off = 32768, .len = 1};
  }
  prepare_reload_cost(st, 1);
  expect("cost uses aligned covers of all spans", st->reload_cost[0] == .75);
  expect("uniform aligned costs preserve victim", victim(st) == 0);
  expert_span[0][1][0] = (FileSpan){.off = 0, .len = 1};
  prepare_reload_cost(st, 1);
  expect("each expert gets its own read cost", st->reload_cost[1] == .5 && st->reload_cost[0] == .75);

  setup(10, 1);
  st->max_slots = st->cap = st->nslots = 2, st->head = 1, st->tail = 0;
  WHERE(0, 2) = -1, st->escore[2] = 0;
  st->slots[1].prev = -1;
  st->score[0] = st->escore[0] = 2, st->score[1] = st->escore[1] = 5;
  st->reload_cost[0] = .25, st->reload_cost[1] = .5, st->reload_cost[2] = .25;
  int incoming = 2;
  cache_load(st, 0, &incoming, 1);
  expect("eviction preserves absent expert history", WHERE(0, 0) == -1 && st->escore[0] == 2);
  expect("new request adds raw unweighted count", st->escore[2] == 1 && st->score[WHERE(0, 2)] == 1);
  advance_score_clock(st);
  double before = st->weight;
  incoming = 0;
  cache_load(st, 0, &incoming, 1);
  expect("reentry restores prior history before increment", near(st->escore[0], 2 + before));
  expect("resident priority comes from the restored history", near(st->score[WHERE(0, 0)], st->escore[0] / st->weight));
  expect("weighted reentry priority uses static cost",
         near(eviction_priority(st, WHERE(0, 0)), st->escore[0] / st->weight * .25));
  expect("empty-span fixture performs no file reads", st->stats.bytes == 0);

  st->weight = 0x1p500;
  st->escore[3] = 7;
  double scale = st->weight * st->retention_growth, old = st->escore[0];
  advance_score_clock(st);
  expect("overflow guard resets global weight", st->weight == 1);
  expect("overflow guard rescales resident history", near(st->escore[0], old / scale));
  expect("overflow guard rescales ghost history", near(st->escore[3], 7 / scale));
  expect("resident and historical scores stay consistent", st->score[WHERE(0, 0)] == st->escore[0]);
  busy_victim(1, 0);
  busy_victim(1, 1);
  busy_victim(0, 0);
  refined_empty_reuse();
  for (int i = 0; i < st->max_slots; i++) finish(st, i);
  fprintf(stderr, "cache policy: %d/%d checks passed\n", checks - failures, checks);
  return failures != 0;
}
