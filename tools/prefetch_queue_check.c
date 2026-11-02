// Exercise read admission when completion callbacks are delayed. No GPU or SSD files.
#include <dispatch/dispatch.h>
#include <fcntl.h>
#include <stdlib.h>
#include <unistd.h>

static dispatch_queue_t deferred;
#define dispatch_get_global_queue(qos, flags) deferred
#include "../src/io.c"
#undef dispatch_get_global_queue

static int checks, failures;
static Slot fixture;

static void expect(const char *name, int ok) {
  checks++;
  if (!ok) failures++, fprintf(stderr, "FAIL: %s\n", name);
}

static void nothing(void *context) {}

static void setup(int fd) {
  memset(&fixture, 0, sizeof fixture);
  fixture.layer = 0;
  fixture.expert = 0;
  fixture.b.ptr = calloc(1, 65536);
  fixture.b.size = 65536;
  slots = &fixture;
  expert_spans[0] = 1;
  expert_span[0][0][0] = (FileSpan){fd, 0, 16384};
  qhead = qtail = 0;
  atomic_store(&demand, 0);
  atomic_store(&prefetching, 0);
  memset(&cache_stats, 0, sizeof cache_stats);
  dispatch_suspend(deferred);
}

static void teardown(void) {
  dispatch_resume(deferred);
  dispatch_sync_f(deferred, NULL, nothing);
  finish(0);
  expect("all work is reaped", !fixture.loading && !atomic_load(&demand) && !atomic_load(&prefetching));
  free(fixture.b.ptr);
  slots = NULL;
}

int main(void) {
  alarm(20);
  int fd = open("/dev/zero", O_RDONLY);
  if (fd < 0) return 1;
  deferred = dispatch_queue_create("myds4.check.deferred-completion", DISPATCH_QUEUE_SERIAL);

  setup(fd);
  start(0, 0);
  expect("queued prefetch occupies one admission", atomic_load(&prefetching) == 1 && slot_reading(0));
  atomic_store(&fixture.taken[0], 1);
  atomic_store(&demand, 1);
  read_demand(NULL); // Promotion completes its one piece while the notification queue cannot run.
  expect("completed data is ready", !slot_reading(0));
  expect("completed prefetch immediately releases admission", atomic_load(&prefetching) == 0);
  expect("promoted read remains a single read", cache_stats.bytes == 16384);
  teardown();

  setup(fd);
  start(0, 0);
  cancel(0);
  expect("cancelled prefetch has no read", cache_stats.bytes == 0 && !fixture.loading);
  expect("cancelled prefetch immediately releases admission", atomic_load(&prefetching) == 0);
  teardown();

  setup(fd);
  start(0, 1);
  expect("demand reads do not consume prefetch admissions", atomic_load(&prefetching) == 0);
  teardown();

  dispatch_release(deferred);
  close(fd);
  fprintf(stderr, "prefetch queue: %d/%d CPU checks passed\n", checks - failures, checks);
  return failures != 0;
}
