// Exact masked top-k selection, including empty masks, ties and partial blocks.
#include "myds4.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

enum { MAX_N = 131072, WORDS = MAX_N / 256 + 1, GUARD = 16, OUT = 513 };
typedef struct { uint32_t pos, ratio, stride, cand; } Index;
typedef struct { uint32_t key, index; } Ranked;
static Buf scores, mask, selected;
static int original, candidate;
static uint32_t rng = 12345;
static uint32_t random_word(void) { rng ^= rng << 13; rng ^= rng >> 17; return rng ^= rng << 5; }
static uint32_t order(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  return u & 0x80000000u ? ~u : u | 0x80000000u;
}
static int rank_order(const void *a, const void *b) {
  const Ranked *x = a, *y = b;
  return x->key != y->key ? (x->key < y->key ? 1 : -1) : (x->index > y->index) - (x->index < y->index);
}
static int position_order(const void *a, const void *b) {
  int x = *(const int *)a, y = *(const int *)b;
  return (x > y) - (x < y);
}

static double run(int kernel, const Index *ix, int repeats) {
  memset(selected.ptr, 0x5a, selected.size);
  double before = gpu_busy;
  gpu_begin();
  for (int i = 0; i < repeats; i++) {
    gpu_kern(kernel);
    gpu_buf(0, &scores, 0), gpu_buf(1, &selected, GUARD * 4), gpu_bytes(2, ix, sizeof *ix), gpu_buf(3, &mask, 0);
    gpu_go(1, 1, 1, 1024, 1, 1);
  }
  gpu_commit(), gpu_sync();
  return (gpu_busy - before) * 1000 / repeats;
}

int main(void) {
  alarm(120);
  lock_model();
  static const char source[] = {
#embed "../../src/kernels/kernels.metal"
      , '\n',
#embed "../../src/experimental/index_prune.metal"
  };
  gpu_init(source, sizeof source);
  original = gpu_kernel("index_topk"), candidate = gpu_kernel("index_topk_masked");
  scores = gpu_alloc(MAX_N * 4), mask = gpu_alloc(WORDS * 4), selected = gpu_alloc((OUT + 2 * GUARD) * 4);
  Ranked *ranked = xmalloc(MAX_N * sizeof *ranked);
  int expected[OUT], control[OUT], checks = 0, failures = 0;
  const int lengths[] = {17, 511, 512, 513, 16383, 16384, 16385, 20000, 65536, 107934};
  for (unsigned c = 0; c < sizeof lengths / sizeof *lengths; c++)
    for (int pattern = 0; pattern < 4; pattern++)
      for (int masks = 0; masks < 7; masks++) {
        int n = lengths[c], k = n < 512 ? n : 512;
        uint32_t *bits = (uint32_t *)mask.ptr;
        for (int w = 0; w < WORDS; w++)
          bits[w] = masks <= 1 ? UINT32_MAX : masks == 2 ? 0 : masks == 3 ? (w == 0 ? 1 : 0)
                  : masks == 4 ? (w < 2 ? UINT32_MAX : 0) : masks == 5 ? 0x01010101u : random_word();
        for (int i = 0; i < n; i++) {
          float f = pattern == 0 ? 0 : pattern == 1 ? (float)(i % 8192 - 4096) / 128
                    : (float)((int)(random_word() % 257) - 128) / 16;
          if (pattern == 3) {
            const uint32_t exceptional[] = {0, 0x80000000u, 0x7f800000u, 0xff800000u, 0x7fc00001u, 0xffc00001u, 0x3f800000u, 0xbf800000u};
            memcpy(&f, &exceptional[i % 8], 4);
          }
          ((float *)scores.ptr)[i] = f;
          int allowed = masks == 0 || ((bits[i >> 8] >> ((i >> 3) & 31)) & 1);
          ranked[i] = (Ranked){order(allowed ? f : -INFINITY), (uint32_t)i};
        }
        qsort(ranked, n, sizeof *ranked, rank_order);
        expected[0] = k;
        for (int i = 0; i < k; i++) expected[i + 1] = ranked[i].index;
        qsort(expected + 1, k, sizeof(int), position_order);
        Index ix = {(uint32_t)n - 1, 1, MAX_N, masks ? WORDS : 0};
        run(original, &ix, 1);
        memcpy(control, selected.ptr + GUARD * 4, (k + 1) * 4);
        run(candidate, &ix, 1);
        int ok = !memcmp(control, expected, (k + 1) * 4) && !memcmp(control, selected.ptr + GUARD * 4, (k + 1) * 4);
        for (int i = 0; i < GUARD; i++)
          ok &= ((uint32_t *)selected.ptr)[i] == 0x5a5a5a5au && ((uint32_t *)selected.ptr)[GUARD + OUT + i] == 0x5a5a5a5au;
        checks++;
        if (!ok) failures++, fprintf(stderr, "top-k mismatch: n=%d scores=%d mask=%d\n", n, pattern, masks);
        if (!ok || pattern != 1 || masks != 5 || n < 16385) continue;
        double a[7], b[7];
        run(original, &ix, 8), run(candidate, &ix, 8);
        for (int r = 0; r < 7; r++) {
          if (r & 1) b[r] = run(candidate, &ix, 16), a[r] = run(original, &ix, 16);
          else a[r] = run(original, &ix, 16), b[r] = run(candidate, &ix, 16);
        }
        for (int i = 1; i < 7; i++) {
          for (int j = i; j && a[j] < a[j - 1]; j--) { double v = a[j]; a[j] = a[j - 1]; a[j - 1] = v; }
          for (int j = i; j && b[j] < b[j - 1]; j--) { double v = b[j]; b[j] = b[j - 1]; b[j - 1] = v; }
        }
        printf("{\"index_mask_kernel\":{\"positions\":%d,\"control_ms\":%.6f,\"candidate_ms\":%.6f}}\n", n, a[3], b[3]);
      }
  fprintf(stderr, "masked index top-k: %d/%d GPU checks passed\n", checks - failures, checks);
  free(ranked);
  return failures != 0;
}
