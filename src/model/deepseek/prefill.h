#ifndef MYDS4_PREFILL_PRUNE_H
#define MYDS4_PREFILL_PRUNE_H

#include <stdint.h>

typedef struct { int prefix_begin, prefix_end, tail_begin, kept, suffix_rows; } PrefillPrune;

// Decoder stages mix positions only through a causal window. Keep a
// conservative dependency halo before the end and before a saved prefix.
// Boundaries retain the original matrix batches to preserve their shapes.
static inline PrefillPrune prefill_prune_plan(int start, int count, int end, int prefix,
                                             int stages, int window, int rewind, int block) {
  if (count <= 0) return (PrefillPrune){0};
  if (stages < 0 || window < 1 || rewind < 0 || block < 1)
    return (PrefillPrune){.kept = count, .suffix_rows = count};
  int64_t halo = (int64_t)window + rewind + ((int64_t)stages + 1) * (window - 1);
  int64_t tail = (int64_t)end - halo - start, saved = (int64_t)prefix - start, before = saved - halo;
  int first = tail <= 0 ? 0 : tail >= count ? count : (int)(tail / block * block);
  int last = prefix < 0 || saved <= 0 ? 0 : saved >= count ? count
             : (int)((saved + block - 1) / block * block);
  if (last > count) last = count;
  int begin = !last || before <= 0 ? 0 : before >= count ? count : (int)(before / block * block);
  if (begin == last) begin = last = 0;
  if (last && last >= first) {
    if (begin < first) first = begin;
    begin = last = 0;
  }
  return (PrefillPrune){begin, last, first, last - begin + count - first, count - first};
}

#endif
