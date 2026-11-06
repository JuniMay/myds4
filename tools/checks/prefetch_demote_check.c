// Actual routing may discard only wholly unstarted, unused forecasts.
// Deferred demand callbacks and a manually claimed piece keep ownership
// transitions deterministic. No reader threads, GPU buffers or files are made.
#include <dispatch/dispatch.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

static dispatch_queue_t deferred;
#define MYDS4_PREFETCH_DEMOTE_CANCEL
#define dispatch_get_global_queue(qos, flags) deferred
#include "../../src/engine/io.c"
#undef dispatch_get_global_queue

enum { NSLOT = 7, LAYER = 3, BUFSIZE = 131072, QMAX = 32 };
static Slot fixture[NSLOT];
static double priorities[NSLOT];
static int checks, failures;

static void expect(const char *name, int ok) {
  checks++;
  if (!ok) failures++, fprintf(stderr, "FAIL: %s\n", name);
}

static void nothing(void *unused) {}

static void complete_piece(int slot, int piece) {
  Slot *s = &fixture[slot];
  uint64_t began = metrics_io_begin(s, 0);
  read_piece(&s->piece[piece], 0);
  metrics_io_end(s, 0, began);
  __atomic_fetch_add(&cache_stats.bytes, s->piece[piece].len, __ATOMIC_RELAXED);
  metrics_piece_done(s, s->piece[piece].len);
  piece_finished(s, piece);
}

static int snapshot(uint32_t out[QMAX]) {
  uint32_t n = qtail - qhead;
  if (n > QMAX) die("fixture queue exceeded its bound");
  for (uint32_t k = 0; k < n; k++) out[k] = queue[(qhead + k) % QUEUE];
  return (int)n;
}

static int same_queue(const uint32_t *expected, int count) {
  uint32_t actual[QMAX];
  return snapshot(actual) == count && !memcmp(actual, expected, (size_t)count * sizeof *actual);
}

static void setup(int fd, int prefill) {
  memset(fixture, 0, sizeof fixture);
  memset(where, 0xff, sizeof where);
  memset(escore, 0, sizeof escore);
  memset(&cache_stats, 0, sizeof cache_stats);
  slots = fixture, score = priorities;
  max_slots = cap = nslots = NSLOT, nfree = npinned = 0;
  head = NSLOT - 1, tail = 0, token = 10, weight = 1;
  pass_layer = prefill ? LAYER : -1, serving = published = -1;
  pf_enabled = pf_active = 0;
  qhead = qtail = 0;
  atomic_store(&demand, 0), atomic_store(&prefetching, 0);
  dispatch_suspend(deferred);
  // 0 selected, 1/6 unused and unstarted, 2 partially started, 5 ready.
  // 3/4 belong to other layers and must retain their queue positions.
  const int layers[NSLOT] = {LAYER, LAYER, LAYER, LAYER + 1, LAYER - 1, LAYER, LAYER};
  const int experts[NSLOT] = {0, 1, 2, 0, 0, 3, 4};
  for (int l = LAYER - 1; l <= LAYER + 2; l++) expert_spans[l] = 2;
  for (int i = 0; i < NSLOT; i++) {
    Slot *s = &fixture[i];
    s->b.ptr = xmalloc(BUFSIZE), s->b.size = BUFSIZE;
    memset(s->b.ptr, 0xa5, BUFSIZE);
    s->mapped = 1, s->layer = layers[i], s->expert = experts[i], s->prefetched = 1, s->used = token;
    s->prev = i == NSLOT - 1 ? -1 : i + 1, s->next = i ? i - 1 : -1;
    where[s->layer][s->expert] = i;
    priorities[i] = escore[s->layer * EXPERTS + s->expert] = 10 + i;
    for (int k = 0; k < 2; k++)
      expert_span[s->layer][s->expert][k] = (FileSpan){fd, (uint64_t)k * 32768, 16384};
    start(i, 0);
  }
  for (int k = 0; k < 2; k++)
    expert_span[LAYER + 2][0][k] = (FileSpan){fd, (uint64_t)k * 32768, 16384};

  int selected = 0;
  expect("selected forecast is pending", cache_request(LAYER, &selected, 1) == 1);
  atomic_store(&fixture[2].taken[0], 1); // a reader owns this piece, but has not completed it
  for (int k = 0; k < 2; k++) {
    atomic_store(&fixture[5].taken[k], 1);
    complete_piece(5, k);
  }
  finish(5);
  fixture[5].prefetched = 0, fixture[5].used = token - 1;
  priorities[5] = escore[LAYER * EXPERTS + 3] = 0; // empty slots must beat this eligible low-priority resident

  // Interleave slots, retain claimed entries, and include a duplicate old
  // reference to slot 1. Both the ring index and uint32 counter wrap.
  const uint32_t order[] = {0x30, 0x10, 0x00, 0x20, 0x60, 0x40, 0x50,
                            0x31, 0x11, 0x01, 0x21, 0x61, 0x41, 0x51, 0x10};
  qhead = UINT32_MAX - 4;
  qtail = qhead + sizeof order / sizeof *order;
  for (size_t k = 0; k < sizeof order / sizeof *order; k++) queue[(qhead + (uint32_t)k) % QUEUE] = order[k];
}

static void teardown(void) {
  // Finish the simulated in-flight background read, then let the selected
  // demand callbacks finish. No callback may outlive a fixture buffer.
  if (fixture[2].loading) complete_piece(2, 0);
  dispatch_resume(deferred);
  dispatch_sync_f(deferred, NULL, nothing);
  while (qhead != qtail) {
    uint32_t q = queue[qhead++ % QUEUE];
    Slot *s = &fixture[q >> 4];
    if (!atomic_exchange(&s->taken[q & 15], 1)) complete_piece(q >> 4, q & 15);
  }
  for (int i = 0; i < NSLOT; i++) {
    finish(i);
    free(fixture[i].b.ptr);
  }
  expect("all admissions and demand callbacks finish", !atomic_load(&prefetching) && !atomic_load(&demand));
  slots = NULL, score = NULL;
}

int main(void) {
  alarm(20);
  int fd = open("/dev/zero", O_RDONLY);
  if (fd < 0) return 1;
  deferred = dispatch_queue_create("myds4.check.prefetch-demote", DISPATCH_QUEUE_SERIAL);
  for (int prefill = 0; prefill <= 1; prefill++) {
    setup(fd, prefill);
    uint32_t before[QMAX], expected[QMAX];
    int count = snapshot(before), kept = 0;
    for (int k = 0; k < count; k++)
      if (prefill || ((before[k] >> 4) != 1 && (before[k] >> 4) != 6)) expected[kept++] = before[k];
    int admissions = atomic_load(&prefetching);
    uint64_t bytes = cache_stats.bytes, cancelled = cache_stats.cancelled;
    uint8_t *selected_buffer = fixture[0].b.ptr, *partial_buffer = fixture[2].b.ptr, *reuse_buffer = fixture[1].b.ptr;
    cache_demote(LAYER);

    expect("selected pending expert stays protected", where[LAYER][0] == 0 && fixture[0].b.ptr == selected_buffer &&
           fixture[0].loading && !fixture[0].prefetched && atomic_load(&fixture[0].unfinished) == 2);
    expect("partially started forecast can still finish", where[LAYER][2] == 2 && fixture[2].b.ptr == partial_buffer &&
           fixture[2].loading && atomic_load(&fixture[2].unfinished) == 2 &&
           atomic_load(&fixture[2].taken[0]) == 1 && atomic_load(&fixture[2].taken[1]) == 0);
    expect("already ready expert remains cached", where[LAYER][3] == 5 && !slot_reading(5));
    expect("other layers remain forecast entries", where[LAYER + 1][0] == 3 && fixture[3].prefetched &&
           where[LAYER - 1][0] == 4 && fixture[4].prefetched);
    expect("demotion performs no I/O", cache_stats.bytes == bytes);
    expect("remaining queue keeps FIFO order across wrap", same_queue(expected, kept));
    expect("only canceled admissions are released", atomic_load(&prefetching) == admissions - (prefill ? 0 : 2));
    expect("cancellations counted once", cache_stats.cancelled == cancelled + (prefill ? 0 : 2));
    if (prefill) {
      expect("prefill leaves unused reads queued", where[LAYER][1] == 1 && where[LAYER][4] == 6 &&
             fixture[1].loading && fixture[6].loading && !fixture[1].prefetched && !fixture[6].prefetched);
    } else {
      expect("wholly unstarted unused entries are empty", where[LAYER][1] == -1 && where[LAYER][4] == -1 &&
             fixture[1].expert == -1 && fixture[6].expert == -1 && !slot_reading(1) && !slot_reading(6));
      cache_demote(LAYER);
      expect("repeated demotion is idempotent", same_queue(expected, kept) && cache_stats.cancelled == cancelled + 2);
      int incoming = 0;
      cache_prefetch(LAYER + 2, &incoming, 1, 1, 100);
      expect("empty slot can be reused without allocation", where[LAYER + 2][0] == 1 && fixture[1].b.ptr == reuse_buffer);
      expected[kept++] = 0x10, expected[kept++] = 0x11;
      expect("reused slot has only its new queue references", same_queue(expected, kept));
      expect("reuse does not displace selected or partial data", where[LAYER][0] == 0 && where[LAYER][2] == 2);
    }
    teardown();
  }

  setup(fd, 0);
  int keep = 1;
  cache_refine(LAYER, &keep, 1);
  expect("refinement retains a kept unstarted forecast", where[LAYER][1] == 1 && fixture[1].loading &&
         fixture[1].prefetched && !atomic_load(&fixture[1].taken[0]) && !atomic_load(&fixture[1].taken[1]));
  expect("refinement cancels an excluded unstarted forecast", where[LAYER][4] == -1 &&
         fixture[6].expert == -1 && !slot_reading(6));
  expect("refinement preserves selected and partially started reads", where[LAYER][0] == 0 &&
         where[LAYER][2] == 2 && atomic_load(&fixture[2].taken[0]) && !atomic_load(&fixture[2].taken[1]));
  teardown();

  setup(fd, 0);
  published = LAYER;
  uint32_t before[QMAX];
  int count = snapshot(before);
  uint64_t cancelled = cache_stats.cancelled;
  cache_demote(LAYER);
  expect("published layer retains its pending buffers", where[LAYER][1] == 1 && where[LAYER][4] == 6 &&
         fixture[1].loading && fixture[6].loading && !atomic_load(&fixture[1].taken[0]) &&
         !atomic_load(&fixture[6].taken[0]));
  expect("published layer does not alter the read queue", same_queue(before, count) &&
         cache_stats.cancelled == cancelled);
  teardown();

#ifdef MYDS4_RUNAHEAD
  setup(fd, 0);
  fixture[1].holds = 1;
  fixture[5].prefetched = 1;
  cache_demote(LAYER);
  expect("held forecast keeps its pending buffer", where[LAYER][1] == 1 && fixture[1].holds == 1 &&
         fixture[1].loading && !atomic_load(&fixture[1].taken[0]) && !atomic_load(&fixture[1].taken[1]));
  expect("held forecast does not protect an unrelated unused entry", where[LAYER][4] == -1 &&
         fixture[6].expert == -1 && !slot_reading(6));
  expect("completed forecast remains available", where[LAYER][3] == 5 && !slot_reading(5));
  teardown();
#endif
  dispatch_release(deferred);
  close(fd);
  printf("prefetch demotion: %d/%d CPU checks passed\n", checks - failures, checks);
  return failures != 0;
}
