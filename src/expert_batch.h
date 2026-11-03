#ifndef MYDS4_EXPERT_BATCH_H
#define MYDS4_EXPERT_BATCH_H

#include <stddef.h>
#include <stdint.h>

typedef struct { uint64_t p[3]; float weight, pad; } ExpertRef;
// Up to three token rows share one expert's weight loads. slot preserves
// each row's expert-sorted position for the final sum.
typedef struct {
  uint64_t p[3];
  uint32_t row[3], slot[3];
  float weight[3];
  uint32_t count;
} ExpertRows;
static_assert(sizeof(ExpertRef) == 32 && sizeof(ExpertRows) == 64, "Metal expert record layout");

// Caller supplies space for m*n records. Null entries belong to the other
// read phase. Tables have already been sorted in each row's summation order.
static inline int expert_batch_plan(ExpertRows *groups, const void *table, int m, int n, size_t stride) {
  int ng = 0;
  for (int r = 0; r < m; r++) {
    const ExpertRef *ref = (const ExpertRef *)((const uint8_t *)table + r * stride);
    for (int k = 0; k < n; k++) {
      if (!ref[k].p[0]) continue;
      int g = 0;
      while (g < ng && (groups[g].count == 3 || groups[g].p[0] != ref[k].p[0] ||
             groups[g].p[1] != ref[k].p[1] || groups[g].p[2] != ref[k].p[2])) g++;
      if (g == ng) groups[ng++] = (ExpertRows){.p = {ref[k].p[0], ref[k].p[1], ref[k].p[2]}};
      uint32_t at = groups[g].count++;
      groups[g].row[at] = r, groups[g].slot[at] = k, groups[g].weight[at] = ref[k].weight;
    }
  }
  return ng;
}

#endif
