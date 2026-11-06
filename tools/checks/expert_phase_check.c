// Read partitions and group lifetimes; no model files, disk writes or GPU.
#define MYDS4_EXPERT_PHASED
#include "../../src/engine/io.c"

int main(void) {
  alarm(20);
  Slot fixture = {0};
  Piece original[16];
  slots = &fixture;
  fixture.b.ptr = xmalloc(REGION(EXPERT_W) + REGION(EXPERT_S));
  fixture.layer = 0, fixture.expert = 0;
  expert_spans[0] = 2;
  cache_phase_reads(1);
  int checks = 0, failures = 0;
#define CHECK(x) do { checks++; if (!(x)) failures++, fprintf(stderr, "FAIL: %s\n", #x); } while (0)
  for (uint64_t offset = 0; offset < SECTOR; offset += 511) {
    FileSpan *w = &expert_span[0][0][0], *sc = &expert_span[0][0][1];
    *w = (FileSpan){7, 2 * SECTOR + offset, EXPERT_W};
    *sc = (FileSpan){7, 32ull << 20, EXPERT_S};
    int n = split(fixture.piece, w->fd, w->off, w->len, fixture.b.ptr, &fixture.at[0]);
    n += split(fixture.piece + n, sc->fd, sc->off, sc->len, fixture.b.ptr + REGION(EXPERT_W), &fixture.at[1]);
    memcpy(original, fixture.piece, n * sizeof *original);
    int first = phase_pieces(&fixture, n), matched = 0;
    CHECK(first > 0 && first < n);
    for (int k = 0; k < n; k++) {
      const Piece *p = &fixture.piece[k];
      for (int j = 0; j < n; j++)
        matched += p->fd == original[j].fd && p->off == original[j].off &&
                   p->len == original[j].len && p->dst == original[j].dst;
      int down_only = p->dst < fixture.b.ptr + REGION(EXPERT_W) &&
                      p->off >= w->off + EXPERT_W / 3 && p->off + p->len <= w->off + 2 * (EXPERT_W / 3);
      CHECK((k >= first) == down_only);
      CHECK(p->off % SECTOR == 0 && p->len % SECTOR == 0);
    }
    CHECK(matched == n); // Every original range appears once, with the same destination.
    cache_phase_reads(0);
    CHECK(phase_pieces(&fixture, n) == 0);
    cache_phase_reads(1);
    expert_spans[0] = 3;
    CHECK(phase_pieces(&fixture, n) == 0); // GGUF does not use native MXFP4 ranges.
    expert_spans[0] = 2;
  }
  memset(where, 0xff, sizeof where);
  where[0][0] = 0;
  for (int late_first = 0; late_first < 2; late_first++) {
    fixture.npiece = 4, fixture.up_pieces = 2;
    fixture.loading = dispatch_group_create(), fixture.up_loading = dispatch_group_create();
    atomic_store(&fixture.unfinished, 4);
    fixture.background_load = 1;
    atomic_store(&prefetching, 1);
    for (int k = 0; k < 4; k++) dispatch_group_enter(fixture.loading), atomic_store(&fixture.taken[k], 1);
    for (int k = 0; k < 2; k++) dispatch_group_enter(fixture.up_loading);
    int base = late_first ? 2 : 0;
    piece_finished(&fixture, base + 1), piece_finished(&fixture, base);
    CHECK(slot_reading(0) && atomic_load(&prefetching) == 1);
    CHECK((dispatch_group_wait(fixture.up_loading, DISPATCH_TIME_NOW) == 0) == !late_first);
    if (!late_first) {
      int expert = 0;
      cache_wait_up(0, &expert, 1);
      CHECK(slot_reading(0)); // Early readiness does not expose the down matrix.
    }
    piece_finished(&fixture, 3 - base), piece_finished(&fixture, 2 - base);
    CHECK(!slot_reading(0) && atomic_load(&prefetching) == 0);
    finish(0);
    CHECK(!fixture.loading && !fixture.up_loading && fixture.up_pieces == 0);
  }
  fixture.npiece = 4, fixture.up_pieces = 2;
  fixture.loading = dispatch_group_create(), fixture.up_loading = dispatch_group_create();
  atomic_store(&fixture.unfinished, 4), atomic_store(&prefetching, 1);
  for (int k = 0; k < 4; k++) dispatch_group_enter(fixture.loading), atomic_store(&fixture.taken[k], 0);
  for (int k = 0; k < 2; k++) dispatch_group_enter(fixture.up_loading);
  cancel(0);
  CHECK(!fixture.loading && !fixture.up_loading && atomic_load(&prefetching) == 0);
  CHECK(cache_stats.bytes == 0);
  free(fixture.b.ptr);
  printf("expert read phases: %d/%d CPU checks passed\n", checks - failures, checks);
  return failures != 0;
}
