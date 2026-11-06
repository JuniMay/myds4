// Compare every verification row with sequential decoding from the same state.
#include "myds4.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main(int argc, char **argv) {
  if (argc < 2 || argc > 3) die("usage: expert-batch-model-check CHECKPOINT [EXPERT_GGUF]");
  alarm(300);
  Options opt = {.ctx = 2048, .budget_gb = 20, .reserve_gb = 4, .draft = 5, .draft_cache = 32,
                 .draft_trim = 1, .experts = argc == 3 ? argv[2] : NULL, .dump_fd = -1, .routes_fd = -1};
  tok_load(argv[1]);
  int tokens[2048];
  int n = tok_encode_chat("<｜begin▁of▁sentence｜><｜User｜>Review a deterministic job scheduler. "
                         "Explain how a cancelled queue entry can safely coexist with a reader that already started, "
                         "and how to preserve the order of results.<｜Assistant｜></think>", tokens, 2048);
  model_load(argv[1], &opt);
  int rows[BLOCK + 1];
  rows[0] = sample_argmax(model_forward(tokens, n, 0), VOCAB);
  model_state_save();
  model_draft(rows[0], n, rows + 1, BLOCK);
  float *batched = xmalloc((size_t)(BLOCK + 1) * VOCAB * sizeof(float));
  memcpy(batched, model_verify(rows, BLOCK + 1, n), (size_t)(BLOCK + 1) * VOCAB * sizeof(float));
  model_state_restore();
  int failed = 0;
  for (int r = 0; r <= BLOCK; r++) {
    const float *sequential = model_forward(rows + r, 1, n + r);
    int different = 0, nonfinite = 0;
    float error = 0;
    for (int v = 0; v < VOCAB; v++) {
      float a = batched[(size_t)r * VOCAB + v], b = sequential[v];
      different += memcmp(&a, &b, sizeof a) != 0;
      nonfinite += !isfinite(a) || !isfinite(b);
      if (fabsf(a - b) > error) error = fabsf(a - b);
    }
    printf("{\"verify_row\":%d,\"logits\":%d,\"different_bits\":%d,\"nonfinite\":%d,\"max_abs\":%.9g}\n",
           r, VOCAB, different, nonfinite, error);
    failed |= different || nonfinite;
  }
  free(batched);
  return failed;
}
