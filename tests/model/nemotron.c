// DFlash2 verification must retain every accepted prefix of target state.
#include "../../src/model/model.h"
#include <stdlib.h>
#include <string.h>

#define CHECK(x)                                                                                                       \
  do {                                                                                                                 \
    if (!(x))                                                                                                          \
      die("Nemotron state check: %s", #x);                                                                             \
  } while (0)
int main(int argc, char **argv) {
  if (argc != 3)
    die("usage: nemotron-model TARGET DFLASH2");
  memory_configure(28, 10);
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
  int input[256], n = tok_encode_chat(tokenizer, prompt, input, 256), vocab = ops->vocab();
  ModelState *reference = ops->state_new(), *candidate = ops->state_new();
  const float *logits = ops->prefill(reference, input, n, 0, 0, -1, NULL, NULL);
  CHECK(logits);
  ops->state_copy(candidate, reference, n);
  ModelSnapshot *start = ops->snapshot_take(reference, n, 0);
  enum { ROWS = 16 };
  int tokens[ROWS + 1];
  tokens[0] = sample_argmax(logits, vocab);
  float *expected = xmalloc((size_t)(ROWS + 1) * vocab * 4);
  for (int i = 0; i <= ROWS; i++) {
    int position = n + i;
    logits = ops->rows(&reference, tokens + i, &position, 1);
    CHECK(logits);
    memcpy(expected + (size_t)i * vocab, logits, (size_t)vocab * 4);
    if (i < ROWS)
      tokens[i + 1] = sample_argmax(logits, vocab);
  }
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
  return 0;
}
