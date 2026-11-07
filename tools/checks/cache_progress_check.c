// A completion from any expert must wake a waiter, including a completion
// between its readiness scan and the wait. No weights or GPU are loaded.
#include <pthread.h>
#include <stdatomic.h>
static _Atomic int waiting, wait_result, wait_calls, returned;
static int fixture_wait(pthread_cond_t *cond, pthread_mutex_t *lock, const struct timespec *until) {
  atomic_store(&waiting, 1);
  atomic_fetch_add(&wait_calls, 1);
  int result = pthread_cond_timedwait(cond, lock, until);
  atomic_store(&wait_result, result);
  return result;
}
#define pthread_cond_timedwait fixture_wait
#include "../../src/engine/io.c"
#undef pthread_cond_timedwait
static void *waiter(void *context) {
  uint64_t epoch = *(uint64_t *)context;
  cache_wait_progress(epoch);
  atomic_store(&returned, 1);
  return NULL;
}
int main(void) {
  alarm(5);
  int checks = 0, failures = 0;
#define CHECK(x) do { checks++; if (!(x)) failures++, fprintf(stderr, "FAIL: %s\n", #x); } while (0)
  Slot a = {0}, b = {0};
  Slot *slots[] = {&a, &b};
  for (int i = 0; i < 2; i++) {
    slots[i]->loading = dispatch_group_create();
    dispatch_group_enter(slots[i]->loading);
    atomic_store(&slots[i]->unfinished, 1);
  }
  uint64_t epoch = cache_read_epoch();
  pthread_t thread;
  pthread_create(&thread, NULL, waiter, &epoch);
  while (!atomic_load(&waiting)) sched_yield();
  // The second expert finishes first; the first expert remains pending.
  piece_finished(&b, 0);
  pthread_join(thread, NULL);
  CHECK(atomic_load(&returned));
  CHECK(atomic_load(&wait_result) == 0);
  CHECK(dispatch_group_wait(a.loading, DISPATCH_TIME_NOW) != 0);
  CHECK(dispatch_group_wait(b.loading, DISPATCH_TIME_NOW) == 0);
  CHECK(cache_read_epoch() > epoch);
  epoch = cache_read_epoch();
  piece_finished(&a, 0);
  CHECK(cache_read_epoch() > epoch);
  // This must observe the completed epoch without sleeping for the timeout.
  int calls = atomic_load(&wait_calls);
  cache_wait_progress(epoch);
  CHECK(atomic_load(&wait_calls) == calls);
  CHECK(dispatch_group_wait(a.loading, DISPATCH_TIME_NOW) == 0);
  dispatch_release(a.loading), dispatch_release(b.loading);
  fprintf(stderr, "cache completion notifications: %d/%d checks passed\n", checks - failures, checks);
  return failures != 0;
}
