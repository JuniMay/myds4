// Exercise the bounded forecast segment with real cache operations.
// Demand callbacks are held; no background readers or GPU work run.
#include <dispatch/dispatch.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

static dispatch_queue_t deferred;
#define MYDS4_PREFETCH_POOL 4
#define dispatch_get_global_queue(qos, flags) deferred
#include "../../src/engine/io.c"
#undef dispatch_get_global_queue

enum { N = 16, BYTES = 65536 };
static Slot fixture[N];
static double priorities[N];
static int available[N], checks, failures;

static void expect(const char *name, int ok) {
  checks++;
  if (!ok) failures++, fprintf(stderr, "FAIL: %s\n", name);
}

static void nothing(void *unused) {}

static void check_pool(int fd, int limit) {
  memset(fixture, 0, sizeof fixture);
  memset(where, 0xff, sizeof where);
  memset(escore, 0, sizeof escore);
  memset(&cache_stats, 0, sizeof cache_stats);
  pool_limit = limit, pool_members = pool_peak = 0;
  pool_admissions = pool_declined = pool_promotions = pool_deferred = pool_setup_evictions = 0;
  slots = fixture, score = priorities, free_list = available;
  nslots = cap = max_slots = nmapped = N, nfree = npinned = 0;
  head = N - 1, tail = 0, token = 10, weight = 1;
  pass_layer = published = -1, serving = 1;
  qhead = qtail = 0;
  atomic_store(&demand, 0), atomic_store(&prefetching, 0);
  expert_spans[2] = expert_spans[3] = 1;
  uint8_t *buffers[N];
  for (int i = 0; i < N; i++) {
    Slot *s = &fixture[i];
    s->b.ptr = buffers[i] = xmalloc(BYTES), s->b.size = BYTES;
    memset(s->b.ptr, 0xa5, BYTES);
    s->layer = 0, s->expert = i, s->mapped = 1;
    s->prev = i == N - 1 ? -1 : i + 1, s->next = i ? i - 1 : -1;
    where[0][i] = i, priorities[i] = escore[i] = i + 1;
  }
  for (int l = 2; l <= 3; l++)
    for (int e = 100; e < 100 + N; e++) expert_span[l][e][0] = (FileSpan){fd, 0, 16384};
  dispatch_suspend(deferred);
  int forecast[8];
  for (int k = 0; k < limit; k++) forecast[k] = 100 + k;
  cache_prefetch(2, forecast, limit, limit, 16);
  expect("forecast segment reaches its bound", pool_members == limit && pool_peak == limit);
  expect("segment uses the existing allocation", nslots == N && nfree == 0);
  expect("initial segment cost is counted", pool_setup_evictions == (uint64_t)limit);
  int next = 100 + limit;
  cache_prefetch(2, &next, 1, 1, 16);
  expect("protected full segment declines another forecast", where[2][next] == -1 && pool_declined == 1);
  expect("declining preserves a main resident", where[0][limit] == limit);
  int selected = 100;
  expect("selected forecast remains pending", cache_request(2, &selected, 1) == 1);
  expect("demand promotes without moving the buffer", where[2][selected] == 0 && fixture[0].b.ptr == buffers[0] &&
         !fixture[0].probation && fixture[0].loading);
  expect("promotion demotes one main entry without discarding it", fixture[limit].probation &&
         where[0][limit] == limit && fixture[limit].b.ptr == buffers[limit]);
  expect("promotion preserves the segment bound", pool_members == limit && pool_promotions == 1);
  int future = 104;
#ifdef MYDS4_RUNAHEAD
  fixture[limit].holds = 1;
  cache_prefetch(3, &future, 1, 1, 16);
  expect("held segment entries remain protected", where[3][future] == -1 && where[0][limit] == limit);
  fixture[limit].holds = 0;
#endif
  published = 0;
  cache_prefetch(3, &future, 1, 1, 16);
  expect("published segment entries remain protected", where[3][future] == -1 && where[0][limit] == limit);
  published = -1;
  cache_prefetch(3, &future, 1, 1, 16);
  expect("new forecast replaces only the demoted entry", where[3][future] == limit && where[0][limit] == -1);
  int survivors = 0;
  for (int i = limit + 1; i < N; i++) survivors += where[0][i] == i;
  expect("remaining main entries survive", survivors == N - limit - 1);
  expect("steady forecasts add no main-segment displacement", pool_setup_evictions == (uint64_t)limit);
  cache_refine(2, forecast + 1, limit - 2);
  expect("cancelled forecast leaves a reusable segment slot", fixture[limit - 1].expert == -1 &&
         fixture[limit - 1].probation && pool_members == limit);
  int refill = 105;
  cache_prefetch(3, &refill, 1, 1, 16);
  expect("another forecast reuses that empty slot", where[3][refill] == limit - 1 && nslots == N);
  pass_layer = 2;
  int prompt_expert = 101;
  cache_request(2, &prompt_expert, 1);
  expect("prefill reclaims segment membership without another demotion", !fixture[1].probation &&
         pool_members == limit - 1 && pool_promotions == 1);
  release(2);
  expect("pressure release updates segment ownership", pool_members == limit - 2 && !fixture[2].probation &&
         nslots == N - 1 && nfree == 1 && available[0] == 2);
  dispatch_resume(deferred);
  dispatch_sync_f(deferred, NULL, nothing);
  expect("demand callbacks keep the promoted identities", where[2][100] == 0 && where[2][101] == 1 &&
         cache_ready(2, 100) && cache_ready(2, 101));
  expect("only demanded payloads were read", cache_stats.bytes == 2 * 16384);
  for (int i = 0; i < N; i++) cancel(i), free(buffers[i]);
  expect("all admission ownership completes", !atomic_load(&prefetching) && !atomic_load(&demand));
  slots = NULL, score = NULL, free_list = NULL;
}

int main(void) {
  alarm(20);
  int fd = open("/dev/zero", O_RDONLY);
  if (fd < 0) return 1;
  deferred = dispatch_queue_create("myds4.check.prefetch-pool", DISPATCH_QUEUE_SERIAL);
  check_pool(fd, 4), check_pool(fd, 8);
  dispatch_release(deferred);
  close(fd);
  printf("prefetch pool: %d/%d CPU checks passed\n", checks - failures, checks);
  return failures != 0;
}
