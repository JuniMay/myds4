// DFlash2 verification must retain every accepted prefix of target state, and
// a pass must mark the routed experts it uses. The checks print hashes of
// their logits, so that two builds can be compared.
#include "recency.h"
#include <stdlib.h>
#include <string.h>

#define CHECK(x)                                                                                                       \
  do {                                                                                                                 \
    if (!(x))                                                                                                          \
      die("Nemotron state check: %s", #x);                                                                             \
  } while (0)

// The latest decoded row's routes, by where each expert was (routes_valid).
static int row_sources[5];

// Decodes one row and requires it to mark exactly its chosen experts used,
// and to leave its routes. *tables: the layers that reached their experts
// through their tables.
static const float *decoded_row(const ModelOps *ops, ModelState **state, int token, int position, int *ready,
                                int *tables) {
  ModelCounters counters;
  ops->counters(&counters);
  uint64_t before = counters.table_layers;
  double began = now();
  const float *logits = ops->rows(state, &token, &position, 1);
  CHECK(logits);
  int touched = touched_experts(began, ready);
  ops->counters(&counters);
  *tables = (int)(counters.table_layers - before);
  if (touched != counters.experts_per_row)
    die("Nemotron state check: the row at position %d marked %d experts used, not its %d (%d layers through their "
        "tables)",
        position, touched, counters.experts_per_row, *tables);
  memset(row_sources, 0, sizeof row_sources);
  if (!routes_valid(ops, &counters, row_sources))
    die("Nemotron state check: the row at position %d's routes are missing or invalid", position);
  return logits;
}

// A layer whose experts are all in memory reaches them through its table for
// up to eight rows; its experts count as used there too. Varied tokens bring
// nearly every routed expert into memory: with this seed, 2,844 of
// Transcribe's 2,944, and some layers then hold all of theirs. The check
// fails when no layer reaches its experts through its table.
static void resident_check(const ModelOps *ops, uint64_t *hash) {
  enum { WARM = 1500 };
  int *tokens = xmalloc(WARM * sizeof *tokens), ready;
  uint32_t seed = 12345;
  for (int i = 0; i < WARM; i++) {
    seed = seed * 1664525u + 1013904223u;
    tokens[i] = 1000 + (int)((seed >> 8) % 99000);
  }
  ModelState *state = ops->state_new();
  CHECK(ops->prefill(state, tokens, WARM, 0, 0, -1, NULL, NULL));
  ModelCounters counters;
  ops->counters(&counters);
  printf("%d of %d experts in memory after %d varied tokens\n", counters.cache_slots, counters.experts, WARM);
  int tables;
  *hash = logits_hash(decoded_row(ops, &state, tokens[WARM - 1], WARM, &ready, &tables), (size_t)ops->vocab() * 4);
  printf("the row after them marked exactly its %d chosen experts used; %d layers reached them through their "
         "tables; its routes: %d in memory, %d read ahead, %d late, %d from the SSD\n",
         counters.experts_per_row, tables, row_sources[0], row_sources[1], row_sources[2], row_sources[4]);
  if (!tables)
    die("Nemotron state check: no layer reached its experts through its table; nothing was checked there");
  ops->state_free(state);
  free(tokens);
}

int main(int argc, char **argv) {
  // --memory: a lower model-memory limit makes the checks evict and reuse expert memory; no
  // layer then holds all of its experts, so the resident-layer check does not run.
  double memory = 28;
  if (argc == 5 && !strcmp(argv[3], "--memory") && atof(argv[4]) > 0)
    memory = atof(argv[4]);
  else if (argc != 3)
    die("usage: nemotron-model TARGET DFLASH2 [--memory GIB]");
  memory_configure(memory, 10);
  Options options = {.ctx = 4096,
                     .dump_fd = -1,
                     .routes_fd = -1,
                     .draft = 15,
                     .family = MODEL_NEMOTRON,
                     .nemotron.draft_model = argv[2]};
  Model *model = model_open(argv[1], &options);
  CHECK(model);
  const ModelOps *ops = model->ops;
  Tokenizer *tokenizer = tok_open(argv[1], NULL);
  char prompt[512];
  ops->chat(prompt, sizeof prompt, "Explain why clear speech helps transcription.", 0, 0);
  int input[256], n = tok_encode_chat(tokenizer, prompt, input, 256), vocab = ops->vocab(), ready = 0;
  ModelState *reference = ops->state_new(), *candidate = ops->state_new();
  const float *logits = ops->prefill(reference, input, n, 0, 0, -1, NULL, NULL);
  CHECK(logits);
  uint64_t prefill_hash = logits_hash(logits, (size_t)vocab * 4);
  ops->state_copy(candidate, reference, n);
  ModelSnapshot *start = ops->snapshot_take(reference, n, 0);
  enum { ROWS = 16 };
  int tokens[ROWS + 1];
  tokens[0] = sample_argmax(logits, vocab);
  float *expected = xmalloc((size_t)(ROWS + 1) * vocab * 4);
  int tables = 0;
  for (int i = 0; i <= ROWS; i++) {
    int row_tables;
    logits = decoded_row(ops, &reference, tokens[i], n + i, &ready, &row_tables);
    tables += row_tables;
    memcpy(expected + (size_t)i * vocab, logits, (size_t)vocab * 4);
    if (i < ROWS)
      tokens[i + 1] = sample_argmax(logits, vocab);
  }
  printf("each of %d decoded rows marked exactly its chosen experts used (%d experts in memory, %d layer passes "
         "through tables)\n",
         ROWS + 1, ready, tables);
  printf("prefill logits %016llx, %d decoded rows' logits %016llx, tokens %016llx\n", (unsigned long long)prefill_hash,
         ROWS + 1, (unsigned long long)logits_hash(expected, (size_t)(ROWS + 1) * vocab * 4),
         (unsigned long long)logits_hash(tokens, sizeof tokens));
  int cases[] = {1, 4, 8, 15, 16};
  for (size_t c = 0; c < sizeof cases / sizeof *cases; c++) {
    int kept = cases[c];
    ops->snapshot_restore(candidate, start);
    logits = ops->verify(candidate, tokens, ROWS, n);
    CHECK(logits);
    int different = 0;
    for (size_t i = 0; i < (size_t)ROWS * vocab; i++)
      different += logits[i] != expected[i];
    printf("verify 16 rows: %d differing logits\n", different);
    CHECK(different == 0);
    ops->accept(candidate, tokens, kept, n);
    int position = n + kept;
    logits = ops->rows(&candidate, tokens + kept, &position, 1);
    CHECK(!memcmp(logits, expected + (size_t)kept * vocab, (size_t)vocab * 4));
    ops->snapshot_restore(reference, start);
    for (int i = 0; i <= kept; i++) {
      position = n + i;
      ops->rows(&reference, tokens + i, &position, 1);
    }
    int forecast[7], control[7], anchor = sample_argmax(logits, vocab);
    Sampler greedy = {.temp = 0, .top_p = 1};
    ops->draft(candidate, anchor, n + kept + 1, forecast, 7, &greedy);
    ops->draft(reference, anchor, n + kept + 1, control, 7, &greedy);
    CHECK(!memcmp(forecast, control, sizeof forecast));
    printf("accepted prefix %d: target logits and next DFlash2 proposals match\n", kept);
  }
  ops->snapshot_free(start);
  ops->state_free(candidate);
  ops->state_free(reference);
  tok_close(tokenizer);
  free(expected);
  puts("Nemotron accepted-prefix checks passed");
  if (argc == 3) {
    uint64_t resident_hash;
    resident_check(ops, &resident_hash);
    printf("logits with every expert in memory %016llx\n", (unsigned long long)resident_hash);
    puts("Nemotron experts used in resident layers count as used");
  }
  ModelCounters counters;
  ops->counters(&counters);
  printf("within %.0f GiB: %llu experts read, %llu evicted\n", memory, (unsigned long long)counters.cache.misses,
         (unsigned long long)counters.cache.evictions);
  return 0;
}
