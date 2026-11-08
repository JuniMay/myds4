// Compare reported Qwen state bytes with the buffers actually allocated.
// Small model dimensions keep this CPU-only fixture below 40 MiB.
#include "myds4.h"
#include <stdlib.h>
#include <string.h>
static size_t allocated, peak;
static Buf fixture_alloc(size_t n) {
  if (allocated + n > (40u << 20)) die("Qwen state fixture exceeded 40 MiB");
  void *p = calloc(1, n ? n : 1);
  if (!p) die("fixture allocation failed");
  allocated += n;
  if (allocated > peak) peak = allocated;
  return (Buf){.obj = p, .ptr = p, .gpu = (uint64_t)p, .size = n};
}
static void fixture_free(Buf *b) {
  if (b->obj) allocated -= b->size, free(b->obj);
  *b = (Buf){0};
}
static void fixture_sync(void) {}
#define gpu_alloc fixture_alloc
#define gpu_alloc_kv fixture_alloc
#define gpu_free fixture_free
#define gpu_sync fixture_sync
#include "../../src/model/qwen/model.c"

static size_t owned_bytes(const QwenState *s) {
  size_t n = s->kv.size;
  for (int i = 0; i < s->copies; i++) n += s->gdn[i].size + s->conv[i].size + s->hid[i].size;
  return n;
}
int main(void) {
  c.linear = c.attention = 1;
  c.hidden = 8, c.heads = 4, c.kv_heads = 2, c.head_dim = 4;
  c.value_heads = 2, c.value_dim = c.key_dim = 4, c.channels = 16;
  mtp = 1;
  int checks = 0, failures = 0;
#define CHECK(x) do { checks++; if (!(x)) failures++, fprintf(stderr, "FAIL: %s\n", #x); } while (0)
  Options chained = {.draft = 3, .dump_fd = -1, .routes_fd = -1};
  CHECK(q_refuse(&chained) == NULL);
  for (int copies = 2; copies <= 6; copies += 2) {
  state_copies = copies;
  QwenState *s = qwen_state_new();
  CHECK(q_state_bytes(NULL, 1) == owned_bytes(s));
  CHECK(q_state_bytes((ModelState *)s, 0) == owned_bytes(s));
  for (int positions = 1025; positions <= 65537; positions = positions * 2 - 1) {
    size_t promised = q_state_bytes((ModelState *)s, positions);
    qwen_state_reserve(s, positions);
    CHECK(promised == owned_bytes(s));
    CHECK(q_state_bytes((ModelState *)s, 1) == owned_bytes(s));
  }
  qwen_state_free(s);
  }
  unown(&PART), unown(&ML);
  free(owned);
  CHECK(allocated == 0);
  fprintf(stderr, "Qwen state accounting: %d/%d checks passed; peak=%zu bytes\n", checks - failures, checks, peak);
  return failures != 0;
}
