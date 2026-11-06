// A completion wait must wake for any selected expert, not just the first.
// Groups stand in for held I/O; no model, files or GPU commands are used.
#define MYDS4_EXPERT_WAVES
#include "../../src/engine/io.c"
#include <sched.h>

static Slot fixture[3];
static int ids[] = {4, 7, 9};
static _Atomic int returned;
static int checks, failures;

static void expect(const char *name, int ok) {
  checks++;
  if (!ok) failures++, fprintf(stderr, "FAIL: %s\n", name);
}

static void *waiter(void *unused) {
  cache_wait_any(0, ids, 3);
  atomic_store(&returned, 1);
  return NULL;
}

int main(void) {
  alarm(10);
  slots = fixture;
  memset(where, 0xff, sizeof where);
  for (int i = 0; i < 3; i++) {
    fixture[i].layer = 0, fixture[i].expert = ids[i], fixture[i].npiece = 1;
    fixture[i].loading = dispatch_group_create();
    dispatch_group_enter(fixture[i].loading);
    atomic_store(&fixture[i].unfinished, 1);
    atomic_store(&fixture[i].taken[0], 1);
    where[0][ids[i]] = i;
  }
  pthread_t t;
  if (pthread_create(&t, NULL, waiter, NULL)) return 1;
  while (!atomic_load(&ready_waiting)) sched_yield();
  expect("no incomplete expert is reported ready", !atomic_load(&returned));
  piece_finished(&fixture[1], 0);
  pthread_join(t, NULL);
  expect("later expert completion releases the wait", atomic_load(&returned));
  expect("first expert remains pending", slot_reading(0));
  expect("third expert remains pending", slot_reading(2));
  expect("completed group remains owned by the consumer", fixture[1].loading != NULL && !slot_reading(1));
  expect("wait registration is cleared", !atomic_load(&ready_waiting));
  cache_wait_any(0, ids, 3);
  expect("an already complete expert returns immediately", !atomic_load(&ready_waiting));
  cache_wait(0, ids + 1, 1);
  expect("consumer reaps only its completed expert", !fixture[1].loading && fixture[0].loading && fixture[2].loading);
  piece_finished(&fixture[0], 0), piece_finished(&fixture[2], 0);
  cache_wait(0, ids, 3);
  expect("all groups can be reaped", !fixture[0].loading && !fixture[2].loading);
  expect("completion waiting performs no reads", cache_stats.bytes == 0);
  slots = NULL;
  printf("expert completion: %d/%d CPU checks passed\n", checks - failures, checks);
  return failures != 0;
}
