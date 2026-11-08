// Verify every accepted prefix, its next target logits and its MTP history.
#include "../../src/model/model.h"
#include "../../src/model/qwen/model.h"
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) die("Qwen MTP check: %s", #x); } while (0)

int main(int argc, char **argv) {
  if (argc != 2) die("usage: qwen-mtp CHECKPOINT");
  memory_configure(32, 8, 10);
  Options options = {.ctx = 4096, .draft = 5, .dump_fd = -1, .routes_fd = -1, .quiet = 1};
  Model *model = model_open(argv[1], &options);
  CHECK(model != NULL);
  const ModelOps *ops = model->ops;
  CHECK(ops->draft_block() == 5 && ops->max_verify() == 6);
  Tokenizer *tokenizer = tok_open(argv[1], NULL);
  int prompt[128];
  int count = tok_encode(tokenizer, "Write a Python function that counts the words in a line.", prompt, 128);
  ModelState *state = ops->state_new();
  const float *logits = ops->prefill(state, prompt, count, 0, 0, -1, NULL, NULL);
  size_t row_bytes = (size_t)ops->vocab() * sizeof(float);
  float *target = xmalloc(7 * row_bytes), *draft = xmalloc(7 * row_bytes);
  int tokens[8];
  tokens[0] = sample_argmax(logits, ops->vocab());
  ModelSnapshot *start = ops->snapshot_take(state, count, 0);
  for (int i = 0; i < 7; i++) {
    int position = count + i;
    logits = ops->rows(&state, tokens + i, &position, 1);
    memcpy(target + (size_t)i * ops->vocab(), logits, row_bytes);
    tokens[i + 1] = sample_argmax(logits, ops->vocab());
    QwenMtpRow row = {(QwenState *)state, tokens[i + 1], 0};
    logits = qwen_mtp(&row, 1, 1);
    memcpy(draft + (size_t)i * ops->vocab(), logits, row_bytes);
  }
  int checks = 0;
  for (int rows = 2; rows <= 6; rows++) {
    for (int accepted = 1; accepted <= rows; accepted++) {
      ops->snapshot_restore(state, start);
      int proposals[5];
      ops->draft(state, tokens[0], count, proposals, rows - 1);
      logits = ops->verify(state, tokens, rows, count);
      CHECK(!memcmp(logits, target, (size_t)rows * row_bytes));
      ops->accept(state, tokens, accepted, count);
      CHECK(qwen_state_pos((QwenState *)state) == count + accepted);
      QwenMtpRow next = {(QwenState *)state, tokens[accepted], 0};
      logits = qwen_mtp(&next, 1, 1);
      CHECK(!memcmp(logits, draft + (size_t)(accepted - 1) * ops->vocab(), row_bytes));
      int position = count + accepted;
      logits = ops->rows(&state, tokens + accepted, &position, 1);
      CHECK(!memcmp(logits, target + (size_t)accepted * ops->vocab(), row_bytes));
      checks++;
    }
  }
  printf("Qwen MTP: %d accepted-prefix cases matched every target and draft logit\n", checks);
  ops->snapshot_free(start);
  ops->state_free(state);
  tok_close(tokenizer);
  free(target); free(draft);
  return 0;
}
