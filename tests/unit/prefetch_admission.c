// Check forecast admission ownership with controlled read completion.
// This is a CPU cache fixture, not a model-speed measurement.
// No reader threads or GPU buffers are created; reads use /dev/zero.
#include "../../src/model/deepseek/model.h"
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
  st->slots = fixture, st->score = priorities;
  st->max_slots = st->cap = st->nslots = COUNT, st->nfree = st->npinned = 0;
  st->head = COUNT - 1, st->tail = 0, st->token = 10, st->weight = 1;
  st->pass_layer = st->serving = -1;
  memset(st->where, 0xff, sizeof st->where);
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
    ids[i] = i, st->where[0][i] = i;
    expert_span[0][i][0] = (FileSpan){fd, 0, 16384};
    fixture[i].prefetched = 1;
    start(i, 0);
  }
  CHECK(atomic_load(&st->prefetching) == CURRENT && atomic_load(&st->demand) == 0);
  CHECK(cache_request(0, ids, CURRENT) == CURRENT);
  cache_demote(0);
  CHECK(atomic_load(&st->demand) == CURRENT);
  CHECK(atomic_load(&st->prefetching) == CURRENT); // Promotion has not freed admissions.
  CHECK(cache_stats.prefetch_hits == CURRENT && cache_stats.late == CURRENT);
  cache_prefetch(2, &future, 1, CURRENT);
  CHECK(st->where[2][future] == -1 && cache_stats.prefetches == 0);

  dispatch_resume(deferred);
  dispatch_sync_f(deferred, NULL, nothing);
  cache_wait(0, ids, CURRENT);
  CHECK(atomic_load(&st->prefetching) == 0 && atomic_load(&st->demand) == 0);
  CHECK(st->where[2][future] == -1); // Free capacity does not retry the skipped forecast.
  CHECK(cache_stats.bytes == (uint64_t)CURRENT * 16384);

  cache_prefetch(2, &future, 1, CURRENT); // Offer exactly the same candidate again.
  CHECK(st->where[2][future] == CURRENT && atomic_load(&st->prefetching) == 1);
  CHECK(cache_stats.prefetches == 1);
  cache_prefetch(2, &future, 1, CURRENT);
  CHECK(cache_stats.prefetches == 1 && atomic_load(&st->prefetching) == 1);
  for (int i = 0; i < CURRENT; i++) CHECK(st->where[0][i] == i && cache_ready(0, i));
  cancel(CURRENT);
  CHECK(atomic_load(&st->prefetching) == 0 && !fixture[CURRENT].loading);
  for (int i = 0; i < COUNT; i++) free(fixture[i].b.ptr);
  dispatch_release(deferred);
  close(fd);
  st->slots = NULL, st->score = NULL;
  printf("prefetch admissions: %d/%d CPU checks passed\n", checks - failures, checks);
  return failures != 0;
}
