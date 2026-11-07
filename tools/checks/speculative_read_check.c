// Selected speculative experts use the background queue. Confirming a row
// promotes its existing reads once, without changing ownership or attribution.
// /dev/zero supplies the bytes; no model, GPU or scratch file is needed.
#include <dispatch/dispatch.h>
#include <fcntl.h>
#include <stdlib.h>
#include <unistd.h>

static dispatch_queue_t deferred;
#ifndef MYDS4_RUNAHEAD
#define MYDS4_RUNAHEAD
#endif
#define MYDS4_CAUSAL_PRIORITY_READS
#define dispatch_get_global_queue(qos, flags) deferred
#include "../../src/engine/io.c"
#undef dispatch_get_global_queue

static Slot fixture;
static PrefetchLoad observation;
static double priority;
static int checks, failures;
static void expect(const char *name, int ok) {
  checks++;
  if (!ok) failures++, fprintf(stderr, "FAIL: %s\n", name);
}
static void nothing(void *unused) {}

static void setup(int fd) {
  memset(&fixture, 0, sizeof fixture), memset(&observation, 0, sizeof observation);
  memset(where, 0xff, sizeof where), memset(escore, 0, sizeof escore);
  memset(&cache_stats, 0, sizeof cache_stats), memset(pf_stats, 0, sizeof pf_stats);
  fixture.layer = 0, fixture.expert = 0, fixture.prev = fixture.next = -1, fixture.mapped = 1;
  fixture.b.ptr = calloc(1, 65536), fixture.b.size = 65536;
  slots = &fixture, score = &priority, pf_loads = &observation;
  priority = 0;
  max_slots = cap = nslots = 1, npinned = nfree = 0, held_references = 0;
  head = tail = 0, token = 10, weight = 1, pass_layer = -1;
  where[0][0] = 0;
  expert_spans[0] = 1, expert_span[0][0][0] = (FileSpan){fd, 0, 16384};
  pf_enabled = pf_active = 1, pf_phase = PF_DECODE, pf_pass = 1, pf_rows = 1;
  qhead = qtail = 0;
  atomic_store(&demand, 0), atomic_store(&prefetching, 0);
  dispatch_suspend(deferred);
}

static void reap(void) {
  if (slot_reading(0)) claim(0);
  dispatch_resume(deferred);
  dispatch_sync_f(deferred, NULL, nothing);
  finish(0);
  expect("all read work is reaped", !atomic_load(&demand) && !atomic_load(&prefetching) && !fixture.loading);
  expect("all reservations are released", !held_references && !fixture.holds);
  free(fixture.b.ptr);
  slots = NULL, pf_loads = NULL;
}

int main(void) {
  alarm(20);
  int fd = open("/dev/zero", O_RDONLY);
  if (fd < 0) return 1;
  deferred = dispatch_queue_create("myds4.check.speculative-priority", DISPATCH_QUEUE_SERIAL);
  int expert = 0;

  setup(fd);
  where[0][0] = -1, fixture.expert = -1;
  expect("a missing speculative expert is admitted", cache_hold_speculative(0, &expert, 1) == 1);
  expect("it uses the background queue", atomic_load(&demand) == 0 && atomic_load(&prefetching) == 1 && qtail == 1);
  expect("router selection is not attributed to a predictor", observation.tracked && !observation.origin_prefetch && !fixture.prefetched);
  expect("admission records one miss and one reservation", cache_stats.misses == 1 && fixture.holds == 1 && held_references == 1);
  expect("a second speculative consumer shares the read", cache_hold_speculative(0, &expert, 1) == 1 && fixture.holds == 2);
  expect("a speculative hit does not promote the read", atomic_load(&demand) == 0 && !atomic_load(&fixture.taken[0]));
  uint64_t hits = cache_stats.hits, misses = cache_stats.misses;
  double score_before = priority;
  cache_promote_selected(0, &expert, 1);
  expect("confirmation promotes the queued piece", atomic_load(&demand) == 1 && atomic_load(&fixture.taken[0]) == 1);
  cache_promote_selected(0, &expert, 1);
  expect("repeated confirmation starts no duplicate read", atomic_load(&demand) == 1 && qtail == 1);
  expect("promotion changes no request count or score", hits == cache_stats.hits && misses == cache_stats.misses && priority == score_before);
  expect("promotion changes no reservation count", fixture.holds == 2 && held_references == 2);
  cache_unhold(0, &expert, 1), cache_unhold(0, &expert, 1);
  expect("cancelled consumers leave pending bytes owned", slot_reading(0) && where[0][0] == 0);
  reap();
  expect("shared consumers read one copy", cache_stats.bytes == 16384);

  setup(fd);
  start(0, 1);
  cache_hold_speculative(0, &expert, 1);
  expect("a speculative hit does not demote a demand", atomic_load(&demand) == 1 && !fixture.background_load && !atomic_load(&prefetching));
  cache_unhold(0, &expert, 1);
  reap();

  setup(fd);
  start(0, 2);
  cache_hold(0, &expert, 1);
  expect("another required row promotes a speculative read", atomic_load(&demand) == 1 && fixture.holds == 1);
  cache_unhold(0, &expert, 1);
  reap();

  setup(fd);
  start(0, 0);
  expect("native forecasts retain their own attribution", observation.tracked && observation.origin_prefetch);
  reap();

  dispatch_release(deferred), close(fd);
  fprintf(stderr, "speculative read priority: %d/%d CPU checks passed\n", checks - failures, checks);
  return failures != 0;
}
