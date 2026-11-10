// Real Flash-Next smoke, state restoration and expert recency checks; no
// throughput claims. The checks print hashes of their logits, so that two
// builds can be compared.
#include "recency.h"
#include <ctype.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x)                                                                                                       \
  do {                                                                                                                 \
    if (!(x))                                                                                                          \
      die("Flash-Next check: %s", #x);                                                                                 \
  } while (0)

int main(int argc, char **argv) {
  // --memory: a lower model-memory limit makes the checks evict and reuse expert memory.
  int long_check = 0;
  double memory = 32;
  for (int i = 2; i < argc; i++)
    if (!strcmp(argv[i], "--long"))
      long_check = 1;
    else if (!strcmp(argv[i], "--memory") && i + 1 < argc && atof(argv[i + 1]) > 0)
      memory = atof(argv[++i]);
    else
      argc = 0;
  if (argc < 2)
    die("usage: flash-model CHECKPOINT [--long] [--memory GIB]");
  memory_configure(memory, 10);
  Options options = {.ctx = 4096, .dump_fd = -1, .routes_fd = -1, .quiet = 1};
  Model *model = model_open(argv[1], &options);
  CHECK(model != NULL);
  const ModelOps *ops = model->ops;
  ModelCounters counters;
  Tokenizer *tokenizer = tok_open(argv[1], NULL);
  char prompt[256];
  ops->chat(prompt, sizeof prompt, "Name two colors.", 0, 0);
  int tokens[128], count = tok_encode_chat(tokenizer, prompt, tokens, 128);
  ModelState *state = ops->state_new();
  const float *logits = ops->prefill(state, tokens, count, 0, 0, -1, NULL, NULL);
  CHECK(logits != NULL);
  uint64_t prefill_hash = logits_hash(logits, (size_t)ops->vocab() * 4);
  ModelSnapshot *start = ops->snapshot_take(state, count, 0);
  int token = sample_argmax(logits, ops->vocab()), first = token;
  char output[1024] = {0};
  size_t used = 0;
  float *expected = xmalloc((size_t)ops->vocab() * 4);
  int rows = 0, ready = 0;
  uint64_t guessed = 0;
  int bounded = 0;  // rows that evicted: their count is bounded, not exact
  ops->counters(&counters);
  uint64_t tables = counters.table_layers;
  for (int i = 0; i < 16 && !ops->stop(token); i++) {
    int length;
    const char *piece = tok_piece(tokenizer, token, &length);
    CHECK(used + length < sizeof output);
    memcpy(output + used, piece, length);
    used += length;
    int position = count + i;
    CacheStats before = counters.cache;
    double began = now();
    logits = ops->rows(&state, &token, &position, 1);
    int touched = touched_experts(began, &ready);
    // A read-ahead marks its expert used when its read starts; the row's own
    // read-aheads that its next layer did not choose add to its experts.
    // When the row evicts, a later miss may take such a read-ahead's slot,
    // which then counts once for both: the count lies between the two.
    ops->counters(&counters);
    int chosen = counters.experts_per_row, evicted = counters.cache.evictions != before.evictions;
    int unchosen = (int)((counters.cache.prefetches - before.prefetches) -
                         (counters.cache.prefetch_hits - before.prefetch_hits));
    if (evicted ? touched < chosen || touched > chosen + unchosen : touched != chosen + unchosen)
      die("Flash-Next check: decoded row %d marked %d experts used, not its %d and %s %d read ahead unchosen", i,
          touched, chosen, evicted ? "at most" : "exactly", unchosen);
    guessed += unchosen;
    bounded += evicted;
    rows++;
    if (!i)
      memcpy(expected, logits, (size_t)ops->vocab() * 4);
    token = sample_argmax(logits, ops->vocab());
  }
  ops->counters(&counters);
  printf("each of %d decoded rows marked its %d chosen experts used, and its read-aheads (%llu unchosen in all; "
         "%d rows evicted, bounded; %d experts in memory, %llu layer passes through tables)\n",
         rows, counters.experts_per_row, (unsigned long long)guessed, bounded, ready,
         (unsigned long long)(counters.table_layers - tables));
  for (size_t i = 0; i < used; i++)
    output[i] = (char)tolower((unsigned char)output[i]);
  static const char *colors[] = {"red",    "blue",  "green", "yellow", "orange",
                                 "purple", "black", "white", "pink",   "brown"};
  int found = 0;
  for (size_t i = 0; i < sizeof colors / sizeof *colors; i++)
    found += strstr(output, colors[i]) != NULL;
  printf("color response: %s\n", output);
  printf("prefill logits %016llx, first decoded logits %016llx, response %016llx\n", (unsigned long long)prefill_hash,
         (unsigned long long)logits_hash(expected, (size_t)ops->vocab() * 4),
         (unsigned long long)logits_hash(output, used));
  CHECK(found >= 2);
  ops->snapshot_restore(state, start);
  logits = ops->rows(&state, &first, &count, 1);
  CHECK(!memcmp(logits, expected, (size_t)ops->vocab() * 4));
  ops->snapshot_free(start);
  ops->state_free(state);
  tok_close(tokenizer);
  free(expected);
  puts("Flash-Next colors and complete restored-state logits passed");
  if (long_check) {
    enum { N = 2071 };
    int *history = xmalloc(N * sizeof *history);
    int words[32],
        nwords = tok_encode(tokenizer = tok_open(argv[1], NULL), "red blue green yellow white black ", words, 32);
    for (int i = 0; i < N; i++)
      history[i] = words[i % nwords];
    state = ops->state_new();
    logits = ops->prefill(state, history, N, 0, 0, -1, NULL, NULL);
    CHECK(logits != NULL);
    token = sample_argmax(logits, ops->vocab());
    start = ops->snapshot_take(state, N, 0);
    ModelState *fork = ops->state_new();
    ops->state_copy(fork, state, N);
    expected = xmalloc((size_t)ops->vocab() * 4);
    int position = N;
    memcpy(expected, ops->rows(&state, &token, &position, 1), (size_t)ops->vocab() * 4);
    printf("logits after %d positions %016llx\n", N,
           (unsigned long long)logits_hash(expected, (size_t)ops->vocab() * 4));
    CHECK(!memcmp(expected, ops->rows(&fork, &token, &position, 1), (size_t)ops->vocab() * 4));
    ops->snapshot_restore(state, start);
    CHECK(!memcmp(expected, ops->rows(&state, &token, &position, 1), (size_t)ops->vocab() * 4));
    ops->snapshot_free(start);
    ops->state_free(state);
    ops->state_free(fork);
    free(expected);
    free(history);
    tok_close(tokenizer);
    puts("Flash-Next sparse selection, fork and restoration beyond 2048 positions passed");
  }
  ops->counters(&counters);
  printf("within %.0f GiB: %llu experts read, %llu evicted\n", memory, (unsigned long long)counters.cache.misses,
         (unsigned long long)counters.cache.evictions);
  return 0;
}
