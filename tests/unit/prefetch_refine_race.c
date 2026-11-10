// Make a real reader win ownership while cache_refine is deciding whether
// to cancel. Legacy refinement waits for the held read; a locked decision
// must leave it resident and return. No timing threshold, GPU or disk files.
#include "../../src/model/deepseek/model.h"
#include <dispatch/dispatch.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

static int watched_lock(pthread_mutex_t *lock);
static char watched_exchange(_Atomic char *object, char value);
static ssize_t watched_pread(int fd, void *buffer, size_t length, off_t offset);
static long watched_wait(dispatch_group_t group, dispatch_time_t timeout);
static void watched_leave(dispatch_group_t group);

#define pthread_mutex_lock(lock) watched_lock(lock)
#undef atomic_exchange
#define atomic_exchange(object, value) watched_exchange(object, value)
#define pread(fd, buffer, length, offset) watched_pread(fd, buffer, length, offset)
#define dispatch_group_wait(group, timeout) watched_wait(group, timeout)
#define dispatch_group_leave(group) watched_leave(group)
#include "../../src/engine/io.c"
#include "cache_layout.h"
#undef pthread_mutex_lock
#undef atomic_exchange
#undef pread
#undef dispatch_group_wait
#undef dispatch_group_leave

enum { CONTROLLER, REFINER, READER };
static _Thread_local int role;
static pthread_mutex_t control = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t changed = PTHREAD_COND_INITIALIZER;
static int armed, injection, reader_go, reader_abort, reader_in_io, allow_io, refine_done, refine_waited;
static Slot fixture[2];
static double priorities[2];
static dispatch_group_t watched_group;
static int checks, failures;

static void expect(const char *name, int ok) {
  checks++;
  if (!ok) failures++, fprintf(stderr, "FAIL: %s\n", name);
}

// Both versions receive the same ownership outcome. The current code
// reaches the exchange after its unlocked scan. A locked implementation
// reaches qlock first, so the reader wins before that transaction starts.
static void let_reader_win(int point) {
  if (!armed) return;
  armed = 0, injection = point;
  pthread_mutex_lock(&control);
  reader_go = 1;
  pthread_cond_broadcast(&changed);
  while (!reader_in_io) pthread_cond_wait(&changed, &control);
  pthread_mutex_unlock(&control);
}

static int watched_lock(pthread_mutex_t *lock) {
  if (role == REFINER && lock == &st->qlock) let_reader_win(1);
  return pthread_mutex_lock(lock);
}

static char watched_exchange(_Atomic char *object, char value) {
  if (role == REFINER && object == &fixture[0].taken[0]) let_reader_win(2);
  return atomic_exchange_explicit(object, value, memory_order_seq_cst);
}

static ssize_t watched_pread(int fd, void *buffer, size_t length, off_t offset) {
  if (role == READER && buffer == fixture[0].piece[0].dst) {
    pthread_mutex_lock(&control);
    reader_in_io = 1;
    pthread_cond_broadcast(&changed);
    while (!allow_io) pthread_cond_wait(&changed, &control);
    pthread_mutex_unlock(&control);
  }
  return pread(fd, buffer, length, offset);
}

static long watched_wait(dispatch_group_t group, dispatch_time_t timeout) {
  if (role == REFINER && group == watched_group && timeout == DISPATCH_TIME_FOREVER) {
    pthread_mutex_lock(&control);
    if (reader_in_io && !allow_io) {
      refine_waited = 1;
      pthread_cond_broadcast(&changed);
    }
    pthread_mutex_unlock(&control);
  }
  return dispatch_group_wait(group, timeout);
}

static void watched_leave(dispatch_group_t group) {
  dispatch_group_leave(group);
  // This is the real reader's final operation on its claimed piece. Stop
  // this fixture thread before it loops back to service unrelated entries.
  if (role == READER && group == watched_group) pthread_exit(NULL);
}

static void *reader_thread(void *unused) {
  role = READER;
  pthread_mutex_lock(&control);
  while (!reader_go) pthread_cond_wait(&changed, &control);
  int abort = reader_abort;
  pthread_mutex_unlock(&control);
  if (abort) return NULL;
  return reader(NULL);
}

static void *refine_thread(void *unused) {
  role = REFINER;
  cache_refine(0, NULL, 0);
  pthread_mutex_lock(&control);
  refine_done = 1;
  pthread_cond_broadcast(&changed);
  pthread_mutex_unlock(&control);
  return NULL;
}

int main(void) {
  deepseek_cache_layout();
  alarm(10); // only a deadlock backstop; assertions do not depend on elapsed time
  int fd = open("/dev/zero", O_RDONLY);
  if (fd < 0) return 1;
  st->slots = fixture, st->score = priorities;
  st->max_slots = st->cap = st->nslots = 2, st->nfree = st->npinned = 0;
  st->head = 1, st->tail = 0, st->token = 10, st->weight = 1;
  st->pass_layer = st->serving = -1;
  memset(st->where, 0xff, WHERE_BYTES);
  expert_spans[0] = 1;
  for (int i = 0; i < 2; i++) {
    Slot *s = &fixture[i];
    s->b.ptr = xmalloc(65536), s->b.size = 65536;
    memset(s->b.ptr, 0xa5, s->b.size);
    s->layer = 0, s->expert = i, s->mapped = 1, s->prefetched = 1, s->used = st->token;
    s->prev = i ? -1 : 1, s->next = i ? 0 : -1;
    WHERE(0, i) = i, priorities[i] = st->escore[i] = i + 1;
    expert_span[0][i][0] = (FileSpan){fd, 0, 16384};
    start(i, 0);
  }
  watched_group = fixture[0].loading;
  armed = 1;
  pthread_t reading, refining;
  if (pthread_create(&reading, NULL, reader_thread, NULL) ||
      pthread_create(&refining, NULL, refine_thread, NULL)) die("cannot start refine race fixture");
  pthread_mutex_lock(&control);
  while (!refine_done && !refine_waited) pthread_cond_wait(&changed, &control);
  int returned = refine_done, waited = refine_waited;
  pthread_mutex_unlock(&control);

  expect("reader won ownership during refinement", injection && reader_in_io && atomic_load(&fixture[0].taken[0]));
  expect("refinement returns before the reader is released", returned);
  expect("refinement does not wait on the unfinished reader", !waited);
  expect("started data remains cached while I/O is held", returned && WHERE(0, 0) == 0 &&
         fixture[0].loading == watched_group && atomic_load(&fixture[0].unfinished) == 1);
  expect("wholly unstarted obsolete entry still cancels", returned && WHERE(0, 1) == -1 &&
         fixture[1].expert == -1 && !slot_reading(1));

  pthread_mutex_lock(&control);
  if (!reader_go) reader_abort = reader_go = 1; // report missing instrumentation without stranding the worker
  allow_io = 1;
  pthread_cond_broadcast(&changed);
  pthread_mutex_unlock(&control);
  pthread_join(reading, NULL);
  pthread_join(refining, NULL);
  expect("reader completion preserves the started expert", WHERE(0, 0) == 0 && cache_ready(0, 0));
  expect("only the started piece was read", cache_stats.bytes == 16384);
  expect("all read admissions finish", !atomic_load(&st->prefetching) && !atomic_load(&st->demand));
  for (int i = 0; i < 2; i++) cancel(i), free(fixture[i].b.ptr);
  close(fd);
  st->slots = NULL, st->score = NULL;
  printf("refine ownership race: %d/%d CPU checks passed (reader injected at %s)\n",
         checks - failures, checks, injection == 1 ? "qlock" : injection == 2 ? "piece claim" : "no hook");
  return failures != 0;
}
