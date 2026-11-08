// Fork a state from a snapshot with host buffers, as the KV prefix cache
// does. No model weights or GPU are used.
#include "../../src/model/deepseek/model.h"
#include "myds4.h"
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static size_t allocated, peak;
static int pending, unsafe_frees;
static Buf fixture_alloc(size_t n) {
  if (allocated + n > (64u << 20)) die("state fixture exceeded 64 MiB");
  void *p = xmalloc(n);
  memset(p, 0xD7, n);
  allocated += n;
  if (allocated > peak) peak = allocated;
  return (Buf){.obj = p, .ptr = p, .size = n};
}
static void fixture_free(Buf *b) {
  if (!b->obj) return;
  unsafe_frees += pending;
  allocated -= b->size;
  free(b->obj);
  *b = (Buf){0};
}
static void fixture_sync(void) { pending = 0; }

#define gpu_alloc fixture_alloc
#define gpu_free fixture_free
#define gpu_sync fixture_sync
#include "../../src/model/deepseek/model.c"

static int checks, failures;
static void expect(const char *name, int ok) {
  checks++;
  if (!ok) failures++, fprintf(stderr, "FAIL: %s\n", name);
}
static int filled(const Buf *b, size_t first, unsigned char value) {
  for (size_t i = first; i < b->size; i++) if (b->ptr[i] != value) return 0;
  return 1;
}
static State *make_state(int cap) {
  State *s = xmalloc(sizeof *s);
  memset(s, 0, sizeof *s);
  s->cap = cap;
  for (int l = 0; l < LAYERS; l++) {
    s->ring[l] = fixture_alloc(16);
    if (!L[l].kv_source) continue;
    s->cstate[l] = fixture_alloc(32);
    s->comp[l] = fixture_alloc(rows_bytes(l, cap, 0));
    s->keys[l] = fixture_alloc(rows_bytes(l, cap, 1));
  }
  for (int d = 0; opt.draft && d < DRAFTS; d++) s->dring[d] = fixture_alloc(16);
  return s;
}

// The bytes a state's buffers take, and its rewind copy's, if it has one.
static size_t owned(const State *s) {
  size_t n = s->saved ? saved_bytes() : 0;
  for (int l = 0; l < LAYERS; l++) n += s->ring[l].size + s->cstate[l].size + s->comp[l].size + s->keys[l].size;
  for (int d = 0; opt.draft && d < DRAFTS; d++) n += s->dring[d].size;
  return n;
}

// The source's rolling rows (and history) set to `value`, as at some position.
static void rolling_fill(State *src, unsigned char value) {
  for (int l = 0; l < LAYERS; l++) memset(src->ring[l].ptr, value, src->ring[l].size);
  memset(src->cstate[2].ptr, value, src->cstate[2].size);
  for (int d = 0; opt.draft && d < DRAFTS; d++) memset(src->dring[d].ptr, value, src->dring[d].size);
}

// A fork as the server's KV cache makes it: a snapshot of the source at the
// fork, the source going on past it, then the source's global rows before the
// fork copied into another state and the snapshot restored there.
static void run(int ratio, int positions, int large, int draft) {
  memset(L, 0, sizeof L);
  memset(D, 0, sizeof D);
  opt.ctx = large ? 32768 : 128;
  opt.draft = draft;
  L[2].kv_source = 1;
  L[2].ratio = ratio;
  State *src = make_state(opt.ctx), *dst = make_state(start_cap());
  active = src;
  bind(src);
  remember_state_layout(); // the fixture supplies the model's small rolling-buffer layout
  // What the server's KV prefix cache charges for a state (myds4_state_bytes).
  expect("a state is charged the bytes it owns", deepseek_state_bytes(src, src->cap) == owned(src));
  State *fresh = make_state(grown(start_cap(), positions));
  expect("a new state is charged what it will own", deepseek_state_bytes(NULL, positions) == owned(fresh));
  deepseek_state_free(fresh);
  rolling_fill(src, 0x42);
  int32_t fork_history[3] = {11, 12, 13};
  memcpy(history, fork_history, sizeof history);  // the state in use keeps its history in `history`
  Snapshot *x = deepseek_snapshot(src, positions, 0);
  rolling_fill(src, 0x39);
  for (int i = 0; i < 3; i++) history[i] = src->history[i] = 70 + i;
  memset(src->comp[2].ptr, 0x31, src->comp[2].size);
  memset(src->keys[2].ptr, 0x32, src->keys[2].size);
  size_t saved = saved_bytes();
  dst->saved = xmalloc(saved);  // a rewind point of the destination's old rows
  memset(dst->saved, 0x67, saved);
  expect("a state is charged its rewind copy", deepseek_state_bytes(dst, dst->cap) == owned(dst));
  pending = 1;
  deepseek_state_copy(dst, src, positions);
  deepseek_snapshot_restore(dst, x);
  size_t main_used = rows_bytes(2, positions, 0), keys_used = rows_bytes(2, positions, 1);
  expect("capacity follows the prefix, not the source allocation", dst->cap == grown(start_cap(), positions));
  expect("required global KV rows and guard row are copied", !memcmp(src->comp[2].ptr, dst->comp[2].ptr, main_used));
  expect("required index rows and guard row are copied", !memcmp(src->keys[2].ptr, dst->keys[2].ptr, keys_used));
  expect("global KV beyond the prefix is not copied", filled(&dst->comp[2], main_used, 0xD7));
  expect("index keys beyond the prefix are not copied", filled(&dst->keys[2], keys_used, 0xD7));
  int local = 1;
  for (int l = 0; l < LAYERS; l++) local &= filled(&dst->ring[l], 0, 0x42);
  local &= filled(&dst->cstate[2], 0, 0x42);
  for (int d = 0; opt.draft && d < DRAFTS; d++) local &= filled(&dst->dring[d], 0, 0x42);
  expect("local and draft state comes from the snapshot, not the source as it is now", local);
  expect("the snapshot's history is restored", !memcmp(dst->history, fork_history, sizeof fork_history));
  expect("the destination's old rewind point is dropped", dst->saved == NULL);
  expect("active source and its state are unchanged", active == src && history[0] == 70 && filled(&src->ring[0], 0, 0x39));
  expect("submitted work completes before replacing buffers", !pending && !unsafe_frees);
  if (large)
    fprintf(stderr, "prefix copy: %zu required global bytes; %zu bytes in source capacity\n",
            main_used + keys_used, src->comp[2].size + src->keys[2].size);
  deepseek_snapshot_free(x);
  active = NULL;
  deepseek_state_free(dst), deepseek_state_free(src);
}

static void recreate_after_eviction(void) {
  memset(L, 0, sizeof L), memset(D, 0, sizeof D);
  opt.ctx = 128, opt.draft = 0;
  L[2].kv_source = 1, L[2].ratio = 2;
  State *seed = make_state(128);
  bind(seed);
  base_state();
  free(seed); // base_state owns these same fixture buffers now
  taken = 1;
  State *first = active;
  size_t expected = owned(first), ring = first->ring[0].size, local = first->cstate[2].size;
  ds_state_free((ModelState *)first);
  expect("an evicted active state leaves the model parked", active == &parked);
  expect("new-state accounting survives an empty binding", deepseek_state_bytes(NULL, 128) == expected);
  State *next = deepseek_state_new();
  expect("new window buffers survive eviction of the active state", next->ring[0].size == ring && ring != 0);
  expect("new compressor buffers survive eviction of the active state", next->cstate[2].size == local && local != 0);
  expect("recreated state owns its complete allocation", owned(next) == expected);
  ds_state_free((ModelState *)next);
}

int main(void) {
  alarm(15);
  run(2, 33, 1, 1);
  for (int ratio = 1; ratio <= 2; ratio++)
    for (int positions = 31; positions <= 33; positions++) run(ratio, positions, 0, positions & 1);
  recreate_after_eviction();
  expect("all host buffers are released", allocated == 0);
  fprintf(stderr, "state fork: %d/%d CPU checks passed; peak=%zu bytes\n", checks - failures, checks, peak);
  return failures != 0;
}
