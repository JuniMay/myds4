// Retention policy checks use empty reads and in-memory slots; no GPU or model.
#include <sys/wait.h>
#include "../../src/engine/io.c"

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
  if (slots)
    for (int i = 0; i < max_slots; i++) finish(i);
  slots = NULL;
  cache_retention(half_life, cost);
  memset(fixture, 0, sizeof fixture);
  memset(where, 0xff, sizeof where);
  memset(escore, 0, sizeof escore);
  memset(expert_spans, 0, sizeof expert_spans);
  memset(&cache_stats, 0, sizeof cache_stats);
  slots = fixture, score = fixture_score;
  max_slots = cap = nslots = 3, npinned = nfree = 0;
  head = 2, tail = 0, token = 10, weight = 1;
  pass_layer = serving = published = -1;
  pf_enabled = 0;
  for (int i = 0; i < 3; i++) {
    slots[i].layer = 0, slots[i].expert = i, slots[i].mapped = 1;
    slots[i].prev = i == 2 ? -1 : i + 1;
    slots[i].next = i == 0 ? -1 : i - 1;
    where[0][i] = i;
    score[i] = escore[i] = i + 1;
    reload_cost[i] = 1;
  }
}

static void bad_half(void) {
  slots = NULL;
  cache_retention(-1, 0);
}
static void bad_cost(void) {
  slots = NULL;
  cache_retention(50, 2);
}
static void too_large_half(void) {
  slots = NULL;
  cache_retention(CACHE_HALF_LIFE_MAX + 1, 0);
}
static void too_late(void) { cache_retention(10, 0); }

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
  pass_layer = prefill ? 1 : -1;
  slots[0].loading = dispatch_group_create();
  slots[0].npiece = 1;
  atomic_store(&slots[0].taken[0], 1);
  dispatch_group_enter(slots[0].loading);
  pthread_t reader;
  if (pthread_create(&reader, NULL, complete_read, slots[0].loading)) die("cannot start fixture reader");
  double began = now();
  int got = take(2, 0, optional);
  double elapsed = now() - began;
  pthread_join(reader, NULL);
  fprintf(stderr, "cache policy: %s %s with busy victim: slot %d, %.3f ms\n",
          prefill ? "prefill" : "decode", optional ? "prefetch" : "demand", got, elapsed * 1000);
  if (optional) {
    expect("prefetch declines a victim still being read", got == -1);
    expect("declined prefetch preserves all residents", where[0][0] == 0 && where[0][1] == 1 &&
           where[0][2] == 2 && where[2][0] == -1 && cache_stats.evictions == 0);
    finish(0);
    expect("prefetch can use the same victim once its read ends", take(2, 0, 1) == 0);
  } else {
    expect("demand still waits for and reuses the cheapest victim", got == 0 && !slots[0].loading &&
           where[0][0] == -1 && where[2][0] == 0 && cache_stats.evictions == 1);
  }
}

int main(void) {
  alarm(20);
  setup(0, 0);
  expect("zero selects default half-life", retention_half_life == 50);
  expect("default ratio retains original constant", retention_growth == exp2(1.0 / 50));
  for (int i = 0; i < 50; i++) advance_score_clock();
  expect("default clock doubles weight over 50 advances", near(weight, 2));
  expect("clock counts forward advances", token == 60);
  expect("aging preserves raw history between rescalings", escore[0] == 1 && score[0] == 1);
  slots[0].used = token;
  CacheStats stats_before = cache_stats;
  double added = weight * 2;
  cache_repeat_credit(0, 0, 2);
  expect("repeated row uses update resident and persistent priority", near(score[0], 1 + added) && score[0] == escore[0]);
  expect("repeated row uses add no read or cache events", !memcmp(&stats_before, &cache_stats, sizeof stats_before));
  expect("repeated use protects the more frequent expert", victim() == 1);
  cache_repeat_credit(0, 0, 0);
  expect("zero repeated uses leave priority unchanged", near(score[0], 1 + added));
  rejected("negative half-life rejected", bad_half);
  rejected("excessive half-life rejected", too_large_half);
  rejected("invalid weighting flag rejected", bad_cost);
  rejected("live policy mutation rejected", too_late);

  setup(10, 0);
  for (int i = 0; i < 10; i++) advance_score_clock();
  expect("ten-pass clock doubles weight", near(weight, 2));
  expect("unweighted priority is raw score", eviction_priority(1) == score[1]);
  expect("default victim is lowest raw score", victim() == 0);
  slots[0].prefetched = 1;
  expect("prefetch protection unchanged", victim() == 1);
  serving = 0, slots[1].used = token;
  expect("current union protection unchanged", victim() == 2);
  published = 0;
  expect("published layer protection unchanged", victim() == -1);
  published = serving = -1, slots[0].prefetched = 0;
  score[0] = score[1] = 2, score[2] = 3;
  expect("equal priorities retain physical-slot tie break", victim() == 0);
  score[2] = -INFINITY, slots[2].expert = -1;
  expect("empty slots still go first", victim() == 2);
  score[2] = INFINITY, slots[2].layer = 32767, slots[2].expert = 32767;
  expect("unallocated slots stay ineligible", victim() == 0);

  setup(10, 1);
  reload_cost[0] = 1, reload_cost[1] = .25, reload_cost[2] = 1;
  expect("cost weighting retains expensive lower-frequency expert", victim() == 1);
  expect("priority multiplication leaves raw resident score", score[1] == 2);
  expect("priority multiplication leaves raw ghost history", escore[1] == 2);
  for (int i = 0; i < 3; i++) reload_cost[i] = .5;
  expect("uniform read costs preserve victim", victim() == 0);
  slots[0].layer = 0, slots[1].layer = 1, slots[2].layer = 3;
  pass_layer = 1;
  expect("prefill uses left-layer LRU", victim() == 0);
  pass_layer = 0;
  expect("prefill fallback protects current and next layer", victim() == 2);

  setup(10, 1);
  SLOT_SIZE = 65536, expert_spans[0] = 2;
  for (int e = 0; e < EXPERTS; e++) {
    expert_span[0][e][0] = (FileSpan){.off = 1, .len = 16384};
    expert_span[0][e][1] = (FileSpan){.off = 32768, .len = 1};
  }
  prepare_reload_cost(1);
  expect("cost uses aligned covers of all spans", reload_cost[0] == .75);
  expect("uniform aligned costs preserve victim", victim() == 0);
  expert_span[0][1][0] = (FileSpan){.off = 0, .len = 1};
  prepare_reload_cost(1);
  expect("each expert gets its own read cost", reload_cost[1] == .5 && reload_cost[0] == .75);

  setup(10, 1);
  max_slots = cap = nslots = 2, head = 1, tail = 0;
  where[0][2] = -1, escore[2] = 0;
  slots[1].prev = -1;
  score[0] = escore[0] = 2, score[1] = escore[1] = 5;
  reload_cost[0] = .25, reload_cost[1] = .5, reload_cost[2] = .25;
  int incoming = 2;
  cache_load(0, &incoming, 1);
  expect("eviction preserves absent expert history", where[0][0] == -1 && escore[0] == 2);
  expect("new request adds raw unweighted count", escore[2] == 1 && score[where[0][2]] == 1);
  advance_score_clock();
  double before = weight;
  incoming = 0;
  cache_load(0, &incoming, 1);
  expect("reentry restores prior history before increment", near(escore[0], 2 + before));
  expect("resident score keeps raw reentry history", score[where[0][0]] == escore[0]);
  expect("weighted reentry priority uses static cost", near(eviction_priority(where[0][0]), escore[0] * .25));
  expect("empty-span fixture performs no file reads", cache_stats.bytes == 0);

  weight = 0x1p500;
  escore[3] = 7;
  double scale = weight * retention_growth, old = escore[0];
  advance_score_clock();
  expect("overflow guard resets global weight", weight == 1);
  expect("overflow guard rescales resident history", near(escore[0], old / scale));
  expect("overflow guard rescales ghost history", near(escore[3], 7 / scale));
  expect("resident and historical scores stay consistent", score[where[0][0]] == escore[0]);
  busy_victim(1, 0);
  busy_victim(1, 1);
  busy_victim(0, 0);
  for (int i = 0; i < max_slots; i++) finish(i);
  fprintf(stderr, "cache policy: %d/%d checks passed\n", checks - failures, checks);
  return failures != 0;
}
