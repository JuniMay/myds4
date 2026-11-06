// A selected pending read must be promoted before victim reclamation waits.
// Capture urgent callbacks and complete the held victim at that wait boundary.
#include <dispatch/dispatch.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
static long watched_wait(dispatch_group_t, dispatch_time_t);
static void capture(void *, void (*)(void *));
#define dispatch_group_wait(group, timeout) watched_wait(group, timeout)
#define dispatch_async_f(queue, context, work) capture(context, work)
#include "../../src/engine/io.c"
#undef dispatch_group_wait
#undef dispatch_async_f

static Slot fixture[2];
static double priorities[2];
static dispatch_group_t victim_group;
static struct { void *context; void (*work)(void *); } jobs[8];
static int njobs, observed_wait, checks, failures;
static void expect(const char *name, int ok) {
  checks++;
  if (!ok) failures++, fprintf(stderr, "FAIL: %s\n", name);
}
static void capture(void *context, void (*work)(void *)) {
  if (njobs >= 8) die("too many promotion fixture submissions");
  jobs[njobs].context = context, jobs[njobs++].work = work;
}
static long watched_wait(dispatch_group_t group, dispatch_time_t timeout) {
  if (group == victim_group && timeout == DISPATCH_TIME_FOREVER && !observed_wait) {
    observed_wait = 1;
    expect("later selected prefetch is urgent before reclamation waits", atomic_load(&fixture[1].taken[0]) && njobs == 1);
    expect("selected pending buffer stays protected", fixture[1].used == token && where[0][1] == 1);
    read_piece(&fixture[0].piece[0], 0);
    cache_stats.bytes += fixture[0].piece[0].len;
    piece_finished(&fixture[0], 0);
  }
  return dispatch_group_wait(group, timeout);
}
int main(void) {
  alarm(10);
  int fd = open("/dev/zero", O_RDONLY);
  if (fd < 0) return 1;
  slots = fixture, score = priorities;
  max_slots = cap = nslots = 2, nfree = npinned = 0;
  head = 1, tail = 0, token = 10, weight = 1;
  pass_layer = serving = published = -1;
  memset(where, 0xff, sizeof where);
  expert_spans[0] = 1;
  for (int i = 0; i < 2; i++) {
    Slot *s = &fixture[i];
    s->b.ptr = xmalloc(65536), s->b.size = 65536;
    memset(s->b.ptr, 0xa5, s->b.size);
    s->mapped = 1, s->used = token - 1;
    s->prev = i ? -1 : 1, s->next = i ? 0 : -1;
    priorities[i] = i + 1;
    expert_span[0][i][0] = (FileSpan){fd, 0, 16384};
  }
  fixture[0].layer = 1, fixture[0].expert = 7, where[1][7] = 0;
  fixture[0].piece[0] = (Piece){fd, 0, 16384, fixture[0].b.ptr};
  fixture[0].npiece = 1, fixture[0].loading = dispatch_group_create();
  atomic_store(&fixture[0].taken[0], 1), atomic_store(&fixture[0].unfinished, 1);
  dispatch_group_enter(fixture[0].loading);
  victim_group = fixture[0].loading;
  fixture[1].layer = 0, fixture[1].expert = 1, fixture[1].prefetched = 1, where[0][1] = 1;
  start(1, 0);
  int selected[] = {0, 1};
  expect("both selected reads are pending", cache_request(0, selected, 2) == 2);
  expect("victim reclamation reached the held read", observed_wait);
  expect("each selected piece has one urgent owner", njobs == 2 && atomic_load(&demand) == 2);
  expect("hit and late attribution is unchanged", cache_stats.misses == 1 && cache_stats.prefetch_hits == 1 && cache_stats.late == 1);
  for (int j = 0; j < njobs; j++) jobs[j].work(jobs[j].context);
  cache_wait(0, selected, 2);
  expect("selected experts are ready", cache_ready(0, 0) && cache_ready(0, 1));
  expect("only victim and selected bytes were read", cache_stats.bytes == 3 * 16384);
  expect("all read admissions finish", !atomic_load(&demand) && !atomic_load(&prefetching));
  expect("victim mapping is retired", where[1][7] == -1);
  for (int i = 0; i < 2; i++) {
    size_t wrong = 0;
    for (int k = 0; k < 16384; k++) wrong += fixture[i].b.ptr[k] != 0;
    expect("selected destination contains the requested bytes", !wrong);
    finish(i), free(fixture[i].b.ptr);
  }
  close(fd);
  printf("selected prefetch promotion: %d/%d CPU checks passed\n", checks - failures, checks);
  return failures != 0;
}
