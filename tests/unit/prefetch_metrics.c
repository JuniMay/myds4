// In-memory checks of optional attribution; no model, expert files or GPU work.
#include "../../src/model/deepseek/model.h"
#include "../../src/engine/io.c"
#include "cache_layout.h"

static int checks, failures;
static Slot fixture[8];
static PrefetchLoad metadata[8];
static double fixture_scores[8];

static void expect(const char *name, uint64_t got, uint64_t want) {
  checks++;
  if (got == want) return;
  fprintf(stderr, "%s: expected %llu, got %llu\n", name, (unsigned long long)want, (unsigned long long)got);
  failures++;
}

static void load(int i, int layer, int expert, int prefetch, int issuer, uint64_t done, int pending) {
  Slot *s = &fixture[i];
  s->layer = layer, s->expert = expert, s->mapped = 1, s->prefetched = prefetch;
  s->npiece = 1, s->piece[0].len = 100, s->taken[0] = 1;
  WHERE(layer, expert) = i;
  st->pf_issuer = issuer;
  metrics_start(st, i, !prefetch);
  s->loading = dispatch_group_create();
  dispatch_group_enter(s->loading);
  uint64_t began = done ? metrics_io_begin(st, s, !prefetch) : 0;
  metrics_piece_done(st, s, done);
  metrics_io_end(st, s, !prefetch, began);
  if (!pending) dispatch_group_leave(s->loading);
}

int main(void) {
  st = deepseek_cache_create();
  memset(st->where, 0xff, WHERE_BYTES);
  st->slots = fixture, st->score = fixture_scores, st->pf_loads = metadata, st->max_slots = 8, st->nslots = 8;
  st->pf_enabled = 1;
  expert_spans[4] = 1;
  for (int e = 0; e < EXPERTS; e++) expert_span[4][e][0].len = 100;
  st->token = 77;
  cache_metrics_begin(st, PF_PREFILL, 2);
  load(4, 4, 4, 1, PF_H1, 100, 0);
  cache_metrics_end(st);
  cache_metrics_begin(st, PF_DECODE, 3);
  expect("forward IDs are independent of cache aging", st->token, 77);
  expect("one verify batch is one metric pass", st->pf_stats[PF_DECODE].passes, 1);
  expect("verify rows count work", st->pf_stats[PF_DECODE].rows, 3);
  load(0, 4, 0, 0, PF_OTHER, 100, 0);
  load(1, 4, 1, 1, PF_H2, 100, 0);
  load(2, 4, 2, 1, PF_PC, 40, 1);
  load(3, 4, 3, 0, PF_OTHER, 20, 1);
  load(5, 4, 6, 0, PF_OTHER, 100, 0);
  load(6, LAYERS, 1, 0, PF_OTHER, 100, 0);
  expect("demand piece timing excludes draft experts", atomic_load(&st->pf_stats[PF_DECODE].io_pieces[0]), 3);
  expect("background pieces keep prefetch origin", atomic_load(&st->pf_stats[PF_DECODE].io_pieces[1]), 2);
  expect("draft bytes excluded", atomic_load(&st->pf_stats[PF_DECODE].demand_read_bytes), 220);
  cache_demote(st, 4);
  expect("demotion clears policy marker", st->slots[1].prefetched, 0);
  expect("demotion preserves read origin", metadata[1].origin_prefetch, 1);
  CacheObservation snapshot;
  cache_observe_snapshot(st, 4, &snapshot);
  int exact[2][6] = {{0, 1, 2, 3, 4, 5}, {5, 4, 3, 2, 1, 0}};
  cache_observe_routes(st, &snapshot, &exact[0][0], 2, 6, 0);
  PrefetchStats *d = &st->pf_stats[PF_DECODE];
  uint64_t want[PF_STATES] = {2, 1, 1, 1, 1};
  for (int k = 0; k < PF_STATES; k++) {
    expect("exact union state", d->states[0][k], want[k]);
    expect("payload follows distinct demand", d->payload[0][k], want[k] * 100);
  }
  expect("exact view does not consume prefetch", metadata[1].used, 0);
  expect("exact view has no first-use reward", d->first_uses, 0);
  metrics_piece_done(st, &st->slots[2], 60), dispatch_group_leave(st->slots[2].loading);
  metrics_piece_done(st, &st->slots[3], 80), dispatch_group_leave(st->slots[3].loading);
  expect("I/O can finish after snapshot", cache_ready(st, 4, 2), 1);
  int served[2][6] = {{0, 1, 2, 3, 4, 6}, {6, 4, 3, 2, 1, 0}};
  cache_observe_routes(st, &snapshot, &served[0][0], 2, 6, 1);
  expect("served union omits replaced absent expert", d->states[1][PF_ABSENT], 0);
  expect("served union includes replacement ready expert", d->states[1][PF_EXISTING_READY], 3);
  expect("served retains snapshot pending state", d->states[1][PF_PENDING_PREFETCH], 1);
  expect("served first-use count", d->first_uses, 3);
  expect("older prefetch is eventual reuse", d->eventual_first_uses, 1);
  expect("timely reward follows original H2 issuer", d->timely_first[PF_H2], 1);
  expect("late PC is not a timely reward", d->timely_first[PF_PC], 0);
  expect("bytes at first demand use fixed snapshot", d->first_completed_bytes, 240);
  expect("remaining bytes at first demand", d->first_remaining_bytes, 60);
  expect("completed bytes include late origin reads", atomic_load(&d->prefetch_read_bytes), 200);
  expect("admission phase kept across later consumption", atomic_load(&st->pf_stats[PF_PREFILL].prefetch_read_bytes), 100);
  cache_observe_snapshot(st, 4, &snapshot);
  cache_observe_routes(st, &snapshot, &served[0][0], 2, 6, 1);
  expect("repeat use receives no first-use reward", d->first_uses, 3);
  expect("consumed prefetch becomes existing-ready", snapshot.state[1], PF_EXISTING_READY);

  Slot *s = &st->slots[7];
  s->layer = 4, s->expert = 7, s->npiece = 2, s->piece[0].len = 30, s->piece[1].len = 70;
  WHERE(4, 7) = 7, s->taken[0] = 1, s->taken[1] = 0;
  st->pf_issuer = PF_H1;
  metrics_start(st, 7, 0);
  s->loading = dispatch_group_create();
  dispatch_group_enter(s->loading), dispatch_group_enter(s->loading);
  metrics_piece_done(st, s, 30), dispatch_group_leave(s->loading);
  atomic_store(&s->unfinished, 1); // one pending piece in this synthetic load
  cancel(st, 7);
  metrics_retire(st, 7, 1);
  expect("partial unused read bytes retired", d->unused_removed_bytes, 30);
  expect("unread cancelled bytes are separate", d->unused_cancelled_bytes, 70);
  expect("observed prefetch eviction counted", d->prefetch_evictions, 1);
  WHERE(4, 7) = -1, s->expert = -1;
  int rerequest[6] = {0, 1, 2, 3, 4, 7};
  cache_observe_snapshot(st, 4, &snapshot);
  cache_observe_routes(st, &snapshot, rerequest, 1, 6, 0);
  expect("exact rerequest leaves ghost intact", GHOST(4, 7), 1);
  cache_observe_routes(st, &snapshot, rerequest, 1, 6, 1);
  expect("served rerequest clears ghost", GHOST(4, 7), 0);
  expect("observed absent rerequest counted", d->absent_after_eviction[1], 1);
  cache_observe_snapshot(st, LAYERS, &snapshot);
  expect("draft route observations excluded", snapshot.active, 0);
  cache_metrics_end(st);
  cache_observe_snapshot(st, 4, &snapshot);
  expect("outside supported forward pass is inactive", snapshot.active, 0);
  cache_metrics_begin(st, PF_DECODE, 1);
  load(7, 4, 7, 1, PF_PC, 100, 0);
  st->slots[7].prev = st->slots[7].next = -1;
  cache_metrics_end(st);
  cache_observe_snapshot(st, 4, &snapshot);
  expect("observations inactive once the pass ends", snapshot.active, 0);
  int unobserved = 7;
  cache_request(st, 4, &unobserved, 1);
  expect("unobserved served request is not later called unused", metadata[7].used, 1);
  expect("unobserved request gets no timely reward", d->timely_first[PF_PC], 0);
  expect("unobserved consumption is reported separately", d->unobserved_first_uses, 1);
  finish(st, 7);
  WHERE(4, 7) = -1;
  cache_metrics_begin(st, PF_DECODE, 1);
  load(7, 4, 8, 1, PF_H2, 0, 1);
  uint64_t before_started = d->unstarted_at_use[PF_H2];
  int queued_use[6] = {0, 1, 2, 3, 4, 8};
  cache_observe_snapshot(st, 4, &snapshot);
  cache_observe_routes(st, &snapshot, queued_use, 1, 6, 1);
  expect("first use identifies a prefetch that has not started I/O", d->unstarted_at_use[PF_H2], before_started + 1);
  expect("one new first use is attributed to its source", d->used_by_issuer[PF_H2], 2);
  expect("same-pass timing retains intended-target uses", d->target_uses[PF_H2], 2);
  expect("same-pass timing excludes later reuse", d->target_uses[PF_H1], 0);
  uint64_t began = metrics_io_begin(st, &st->slots[7], 1);
  metrics_piece_done(st, &st->slots[7], 100);
  metrics_io_end(st, &st->slots[7], 1, began);
  dispatch_group_leave(st->slots[7].loading);
  expect("promoted prefetch has a separate timing bucket", atomic_load(&d->io_pieces[2]), 1);
  expect("first I/O start is recorded", atomic_load(&metadata[7].first_io_ns) > 0, 1);
  cache_metrics_end(st);
  st->pf_enabled = 0;
  uint64_t before = st->pf_pass;
  cache_metrics_begin(st, PF_DECODE, 6);
  expect("disabled metrics do not advance passes", st->pf_pass, before);
  expect("disabled I/O timing needs no clock read", metrics_io_begin(st, &st->slots[7], 1), 0);
  st->pf_enabled = 1;
  for (int i = 0; i < 8; i++) finish(st, i);
  cache_metrics_report(st, stdout);
  fprintf(stderr, "prefetch metrics: %d/%d checks passed\n", checks - failures, checks);
  return failures != 0;
}
