// Accepted DSpark history must match a sequential run of the same tokens.
#include "../../src/model/deepseek.c"

int main(int argc, char **argv) {
  if (argc < 2 || argc > 3) die("usage: dspark-state-check CHECKPOINT [EXPERT_GGUF]");
  alarm(300);
  tok_load(argv[1]);
  int tokens[2048];
  int n = tok_encode_chat("<｜begin▁of▁sentence｜><｜User｜>Write a short deterministic task scheduler. "
                         "Preserve dependency order and cancel queued work when a required input fails."
                         "<｜Assistant｜></think>", tokens, 2048);
  Options o = {.ctx = 2048, .budget_gb = 20, .reserve_gb = 4, .draft = 1, .draft_cache = 32,
               .experts = argc == 3 ? argv[2] : NULL, .dump_fd = -1, .routes_fd = -1, .quiet = 1};
  deepseek_load(argv[1], &o);
  const float *lg = deepseek_forward(tokens, n, 0);
  int rows[BLOCK + 1];
  rows[0] = sample_argmax(lg, VOCAB);
  deepseek_state_save();
  size_t bytes = D[0].ring.size;
  void *expected = xmalloc((size_t)(BLOCK + 1) * DRAFTS * bytes);
  for (int r = 0; r <= BLOCK; r++) {
    lg = deepseek_forward(rows + r, 1, n + r);
    if (r < BLOCK) rows[r + 1] = sample_argmax(lg, VOCAB);
    for (int d = 0; d < DRAFTS; d++)
      memcpy((char *)expected + ((size_t)r * DRAFTS + d) * bytes, D[d].ring.ptr, bytes);
  }
  int failed = 0;
  for (int kept = 1; kept <= BLOCK + 1; kept++) {
    deepseek_state_restore();
    deepseek_verify(rows, BLOCK + 1, n);
    deepseek_accept(rows, kept, n);
    uint64_t different = 0;
    for (int d = 0; d < DRAFTS; d++) {
      const uint16_t *ref = (const uint16_t *)((const char *)expected + ((size_t)(kept - 1) * DRAFTS + d) * bytes);
      const uint16_t *actual = (const uint16_t *)D[d].ring.ptr;
      for (size_t j = 0; j < bytes / sizeof(uint16_t); j++) different += actual[j] != ref[j];
    }
    printf("{\"dspark_state_check\":{\"accepted_rows\":%d,\"compared_values\":%zu,\"different_values\":%llu}}\n",
           kept, DRAFTS * bytes / sizeof(uint16_t), (unsigned long long)different);
    fflush(stdout);
    failed |= different != 0;
  }
  free(expected);
  return failed;
}
