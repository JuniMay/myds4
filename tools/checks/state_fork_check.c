// Fork a saved prefix with host buffers. No model weights or GPU are used.
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
#include "../../src/engine/model.c"

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

// Exercise the same saved-prefix operation used by the server.
static void fork_prefix(State *dst, State *src, int positions) {
  model_state_fork(dst, src, positions);
}

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
  for (int i = 0; i < 3; i++) history[i] = src->history[i] = 70 + i;
  for (int l = 0; l < LAYERS; l++) memset(src->ring[l].ptr, 0x39, src->ring[l].size);
  memset(src->comp[2].ptr, 0x31, src->comp[2].size);
  memset(src->keys[2].ptr, 0x32, src->keys[2].size);
  size_t saved = saved_bytes();
  src->saved[0] = xmalloc(saved), src->saved[1] = xmalloc(saved), dst->saved[0] = xmalloc(saved);
  memset(src->saved[0], 0x53, saved);
  memset(src->saved[1], 0x42, saved);
  memset(dst->saved[0], 0x67, saved);
  int32_t prefix_history[3] = {11, 12, 13};
  memcpy(src->saved[1] + saved - sizeof prefix_history, prefix_history, sizeof prefix_history);
  pending = 1;
  fork_prefix(dst, src, positions);
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
  expect("local and draft state comes from the saved prefix", local);
  expect("prefix history is restored", !memcmp(dst->history, prefix_history, sizeof prefix_history));
  expect("saved prefix remains available for later forks", dst->saved[1] && !memcmp(dst->saved[1], src->saved[1], saved));
  expect("later prompt checkpoint is not inherited", dst->saved[0] == NULL);
  expect("active source and its state are unchanged", active == src && history[0] == 70 && filled(&src->ring[0], 0, 0x39));
  expect("submitted work completes before replacing buffers", !pending && !unsafe_frees);
  if (large)
    fprintf(stderr, "prefix copy: %zu required global bytes; %zu bytes in source capacity\n",
            main_used + keys_used, src->comp[2].size + src->keys[2].size);
  active = NULL;
  model_state_free(dst), model_state_free(src);
}

int main(void) {
  alarm(15);
  run(2, 33, 1, 1);
  for (int ratio = 1; ratio <= 2; ratio++)
    for (int positions = 31; positions <= 33; positions++) run(ratio, positions, 0, positions & 1);
  expect("all host buffers are released", allocated == 0);
  fprintf(stderr, "state fork: %d/%d CPU checks passed; peak=%zu bytes\n", checks - failures, checks, peak);
  return failures != 0;
}
