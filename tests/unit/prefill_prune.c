// Check the dependency plan on causal graphs, not on model numerics.
#include "../../src/model/deepseek/prefill.h"
#include <stdio.h>
#include <string.h>
#include <unistd.h>

int main(void) {
  alarm(20);
  unsigned long cases = 0, pruned = 0;
  for (int schedule = 0; schedule < 2; schedule++)
   for (int layers = 1; layers <= 5; layers++)
    for (int window = 1; window <= 8; window++)
      for (int rewind = 0; rewind <= 3; rewind++)
        for (int block = 1; block <= 4; block *= 2)
          for (int chunks = 1; chunks <= 3; chunks += 2)
            for (int start = 0; start <= 9; start += 9)
              for (int n = 1; n <= 127; n += 7)
                for (int which = 0; which < 3; which++) {
                  int end = start + n, prefix = which ? start + n * which / 3 : -1;
                  int chunk = block * chunks, kept = 0;
                  unsigned char output[6][256] = {{0}}, key[6][256] = {{0}}, run[6][256] = {{0}};
                  memset(output[0], 1, sizeof output[0]);
                  for (int l = 1; l <= layers; l++)
                   for (int p = start; p < end; p += chunk) {
                    int size = end - p < chunk ? end - p : chunk;
                    PrefillPrune plan = prefill_prune_plan(p, size, end, prefix, schedule ? layers - l : layers, window, rewind, block);
                    if (plan.prefix_begin < 0 || plan.prefix_begin > plan.prefix_end || plan.prefix_end > plan.tail_begin || plan.tail_begin > size ||
                        plan.kept != plan.prefix_end - plan.prefix_begin + size - plan.tail_begin ||
                        (plan.prefix_begin != size && plan.prefix_begin % block) ||
                        (plan.prefix_end != size && plan.prefix_end % block) ||
                        (plan.tail_begin != size && plan.tail_begin % block)) return 2;
                    kept += plan.kept;
                    for (int r = 0; r < size; r++) run[l][p + r] = (r >= plan.prefix_begin && r < plan.prefix_end) || r >= plan.tail_begin;
                  }
                  for (int l = 1; l <= layers; l++)
                    for (int p = 0; p < end; p++) {
                      if (p < start) { key[l][p] = output[l][p] = 1; continue; }
                      if (!run[l][p]) continue;
                      key[l][p] = output[l - 1][p];
                      output[l][p] = output[l - 1][p];
                      int first = p - window + 1;
                      if (first < 0) first = 0;
                      for (int q = first; q <= p; q++) output[l][p] &= key[l][q];
                    }
                  int valid = output[layers][end - 1];
                  for (int l = 1; l <= layers; l++) {
                    int first = end - window - rewind;
                    if (first < 0) first = 0;
                    for (int p = first; p < end; p++) valid &= key[l][p];
                    if (prefix > start) {
                      first = prefix - window - rewind;
                      if (first < 0) first = 0;
                      for (int p = first; p < prefix; p++) valid &= key[l][p];
                      valid &= output[l][prefix - 1];
                    }
                  }
                  if (!valid) {
                    fprintf(stderr, "dependency failure: schedule=%d layers=%d window=%d rewind=%d start=%d n=%d prefix=%d block=%d chunk=%d\n",
                            schedule, layers, window, rewind, start, n, prefix, block, chunk);
                    return 1;
                  }
                  cases++, pruned += kept < n * layers;
                }
  PrefillPrune p = prefill_prune_plan(0, 16384, 16384, 1024, 19, 128, 15, 2048);
  if (p.prefix_begin || p.prefix_end != 2048 || p.tail_begin != 12288 || p.kept != 6144 || p.suffix_rows != 4096 || !pruned) return 1;
  p = prefill_prune_plan(0, 16384, 16384, 8192, 19, 128, 15, 2048);
  if (p.prefix_begin != 4096 || p.prefix_end != 8192 || p.tail_begin != 12288 || p.kept != 8192) return 1;
  printf("prefill dependency plan: %lu causal graphs passed; %lu pruned cases; prefix and final state retained\n", cases, pruned);
  return 0;
}
