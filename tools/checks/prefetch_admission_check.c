// Check forecast admission ownership with controlled read completion.
// This is a CPU cache fixture, not a model-speed measurement.
// No reader threads or GPU buffers are created; reads use /dev/zero.
#include <dispatch/dispatch.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

static dispatch_queue_t deferred;
#define dispatch_get_global_queue(qos, flags) deferred
#include "../../src/engine/io.c"
#undef dispatch_get_global_queue

static void nothing(void *unused) {}

int main(void) {
  alarm(20);
  enum { CURRENT = 6, COUNT = CURRENT + 1 };
  Slot fixture[COUNT] = {0};
  double priorities[COUNT];
  int ids[CURRENT], future = 0, checks = 0, failures = 0;
#define CHECK(x) do { checks++; if (!(x)) failures++, fprintf(stderr, "FAIL: %s\n", #x); } while (0)
  int fd = open("/dev/zero", O_RDONLY);
  if (fd < 0) return 1;
  deferred = dispatch_queue_create("myds4.check.prefetch-admission", DISPATCH_QUEUE_SERIAL);
  dispatch_suspend(deferred);
  slots = fixture, score = priorities;
  max_slots = cap = nslots = COUNT, nfree = npinned = 0;
  head = COUNT - 1, tail = 0, token = 10, weight = 1;
  pass_layer = serving = -1;
  memset(where, 0xff, sizeof where);
  expert_spans[0] = expert_spans[2] = 1;
  expert_span[2][future][0] = (FileSpan){fd, 0, 16384};
  for (int i = 0; i < COUNT; i++) {
    fixture[i].b.ptr = calloc(1, 65536), fixture[i].b.size = 65536;
    if (!fixture[i].b.ptr) die("fixture allocation failed");
    fixture[i].mapped = 1, fixture[i].layer = 0;
    fixture[i].expert = i < CURRENT ? i : -1;
    fixture[i].prev = i == COUNT - 1 ? -1 : i + 1;
    fixture[i].next = i ? i - 1 : -1;
    priorities[i] = i < CURRENT ? 0 : -INFINITY;
    if (i == CURRENT) continue;
    ids[i] = i, where[0][i] = i;
    expert_span[0][i][0] = (FileSpan){fd, 0, 16384};
    fixture[i].prefetched = 1;
    start(i, 0);
  }
  CHECK(atomic_load(&prefetching) == CURRENT && atomic_load(&demand) == 0);
  CHECK(cache_request(0, ids, CURRENT) == CURRENT);
  cache_demote(0);
  CHECK(atomic_load(&demand) == CURRENT);
  CHECK(atomic_load(&prefetching) == CURRENT); // Promotion has not freed admissions.
  CHECK(cache_stats.prefetch_hits == CURRENT && cache_stats.late == CURRENT);
  cache_prefetch(2, &future, 1, CURRENT);
  CHECK(where[2][future] == -1 && cache_stats.prefetches == 0);

  dispatch_resume(deferred);
  dispatch_sync_f(deferred, NULL, nothing);
  cache_wait(0, ids, CURRENT);
  CHECK(atomic_load(&prefetching) == 0 && atomic_load(&demand) == 0);
  CHECK(where[2][future] == -1); // Free capacity does not retry the skipped forecast.
  CHECK(cache_stats.bytes == (uint64_t)CURRENT * 16384);

  cache_prefetch(2, &future, 1, CURRENT); // Offer exactly the same candidate again.
  CHECK(where[2][future] == CURRENT && atomic_load(&prefetching) == 1);
  CHECK(cache_stats.prefetches == 1);
  cache_prefetch(2, &future, 1, CURRENT);
  CHECK(cache_stats.prefetches == 1 && atomic_load(&prefetching) == 1);
  for (int i = 0; i < CURRENT; i++) CHECK(where[0][i] == i && cache_ready(0, i));
  cancel(CURRENT);
  CHECK(atomic_load(&prefetching) == 0 && !fixture[CURRENT].loading);
  for (int i = 0; i < COUNT; i++) free(fixture[i].b.ptr);
  dispatch_release(deferred);
  close(fd);
  slots = NULL, score = NULL;
  printf("prefetch admissions: %d/%d CPU checks passed\n", checks - failures, checks);
  return failures != 0;
}
