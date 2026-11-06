// Pending urgent work must retain priority and complete each owned piece once.
// GCD submissions are captured so queue admission can be checked before reads.
#include <dispatch/dispatch.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
static void capture(void *context, void (*work)(void *));
#define MYDS4_URGENT_WINDOW 4
#define dispatch_async_f(queue, context, work) capture(context, work)
#include "../../src/engine/io.c"
#undef dispatch_async_f

enum { NSLOT = 4, NPIECE = 8, LENGTH = 16384 };
static Slot fixture[NSLOT];
static struct { void *context; void (*work)(void *); } jobs[64];
static int njobs, checks, failures;

static void capture(void *context, void (*work)(void *)) {
  if (njobs >= 64) die("too many fixture submissions");
  jobs[njobs].context = context, jobs[njobs++].work = work;
}

static void expect(const char *name, int ok) {
  checks++;
  if (!ok) failures++, fprintf(stderr, "FAIL: %s\n", name);
}

static void run_case(int fd, int limit) {
  memset(fixture, 0, sizeof fixture);
  memset(&cache_stats, 0, sizeof cache_stats);
  slots = fixture;
  urgent_window = limit, urgent_workers = urgent_peak_workers = urgent_peak_pending = 0;
  urgent_submissions = urgent_worker_launches = 0;
  urgent_head = urgent_tail = UINT32_MAX - 4;
  atomic_store(&demand, 0), atomic_store(&prefetching, 2);
  njobs = 0;
  for (int i = 0; i < NSLOT; i++) {
    Slot *s = &fixture[i];
    s->b.ptr = xmalloc(NPIECE * LENGTH), s->b.size = NPIECE * LENGTH;
    memset(s->b.ptr, 0xa5, s->b.size);
    s->npiece = NPIECE, s->background_load = i < 2;
    s->loading = dispatch_group_create();
    atomic_store(&s->unfinished, NPIECE);
    for (int k = 0; k < NPIECE; k++) {
      s->piece[k] = (Piece){fd, (uint64_t)k * LENGTH, LENGTH, s->b.ptr + k * LENGTH};
      dispatch_group_enter(s->loading);
    }
    claim(i);
  }
  expect("worker submissions respect the bound", njobs == limit && urgent_workers == (unsigned)limit);
  expect("pending work suppresses background reads", atomic_load(&demand) == NSLOT * NPIECE);
  expect("all pieces are queued before callbacks run", urgent_tail - urgent_head == NSLOT * NPIECE && !cache_stats.bytes);
  for (int i = 0; i < NSLOT; i++) claim(i);
  expect("repeat promotion does not duplicate work", urgent_submissions == NSLOT * NPIECE && njobs == limit);
  int at = 0;
  for (uint32_t q = urgent_head; q != urgent_tail; q++, at++)
    expect("FIFO identity survives counter wrap", urgent_queue[q % QUEUE] == (uint32_t)((at / NPIECE) << 4 | at % NPIECE));
  for (int j = 0; j < njobs; j++) jobs[j].work(jobs[j].context);
  expect("all worker reservations are released", urgent_workers == 0 && urgent_head == urgent_tail);
  expect("all urgency and prefetch admissions finish", !atomic_load(&demand) && !atomic_load(&prefetching));
  expect("aligned payload is read exactly once", cache_stats.bytes == (uint64_t)NSLOT * NPIECE * LENGTH);
  expect("peak worker count remains bounded", urgent_peak_workers == (unsigned)limit);
  for (int i = 0; i < NSLOT; i++) {
    expect("each group is complete before release", !slot_reading(i) && atomic_load(&fixture[i].unfinished) == 0);
    size_t wrong = 0;
    for (size_t k = 0; k < fixture[i].b.size; k++) wrong += fixture[i].b.ptr[k] != 0;
    expect("all requested destinations were filled", wrong == 0);
    finish(i);
    free(fixture[i].b.ptr);
  }
  slots = NULL;
}

int main(void) {
  alarm(20);
  int fd = open("/dev/zero", O_RDONLY);
  if (fd < 0) return 1;
  run_case(fd, 4), run_case(fd, 8), run_case(fd, 16);
  close(fd);
  printf("urgent read window: %d/%d CPU checks passed\n", checks - failures, checks);
  return failures != 0;
}
