// Replay recorded token IDs through the CPU matcher. This does not run a model.
// stdin: native uint32 prompt count, continuation count, then int32 token IDs.
#include "../../src/experimental/context_lookup.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main(int argc, char **argv) {
  if (argc > 1) return 2;
  alarm(20);
  uint32_t shape[2];
  if (fread(shape, sizeof *shape, 2, stdin) != 2 || !shape[0] || !shape[1] ||
      shape[0] > 262144 || shape[1] > 32768 || shape[0] + shape[1] > 262144) return 2;
  uint32_t total = shape[0] + shape[1];
  int32_t *tokens = malloc((size_t)total * sizeof *tokens);
  if (!tokens || fread(tokens, sizeof *tokens, total, stdin) != total) return 2;
  ContextLookup *s = context_lookup_new((int)total);
  if (!s || !context_lookup_append(s, tokens, (int)shape[0])) return 2;
  uint64_t calls = 0, passes = 0, offered = 0, accepted = 0, gap = 0, examined = 0;
  for (uint32_t at = shape[0]; at < total;) {
    if (!context_lookup_append(s, tokens + at, 1)) return 2;
    int room = (int)(total - at - 1);
    int limit = room < 15 ? room : 15;
    ContextDraft p = context_lookup_propose(s, limit);
    calls++, examined += p.examined;
    int kept = 0;
    while (kept < p.count && tokens[at + 1 + kept] == p.tokens[kept]) kept++;
    if (p.count) {
      passes++, offered += p.count, accepted += kept;
      gap += at + 1 - (uint32_t)p.source;
      context_lookup_feedback(s, p.count, kept);
    }
    if (kept && !context_lookup_append(s, tokens + at + 1, kept)) return 2;
    at += 1 + kept;
  }
  printf("{\"prompt_tokens\":%u,\"continuation_tokens\":%u,\"calls\":%llu,"
         "\"verification_passes\":%llu,\"offered\":%llu,\"accepted\":%llu,"
         "\"computed_rows_proxy\":%llu,\"examined\":%llu,\"mean_source_gap\":%.3f,"
         "\"scope\":\"offline recorded-token replay; no GPU, expert cache, model sampling or throughput measurement\"}\n",
         shape[0], shape[1], (unsigned long long)calls, (unsigned long long)passes,
         (unsigned long long)offered, (unsigned long long)accepted,
         (unsigned long long)(shape[1] + offered - accepted), (unsigned long long)examined,
         passes ? (double)gap / passes : 0);
  context_lookup_free(s);
  free(tokens);
  return 0;
}
