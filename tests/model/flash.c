// Real Flash-Next smoke and state restoration checks; no throughput claims.
#include "../../src/model/model.h"
#include <ctype.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x)                                                                                                       \
  do {                                                                                                                 \
    if (!(x))                                                                                                          \
      die("Flash-Next check: %s", #x);                                                                                 \
  } while (0)

int main(int argc, char **argv) {
  if (argc < 2 || argc > 3)
    die("usage: flash-model CHECKPOINT [--long]");
  memory_configure(32, 10);
  Options options = {.ctx = 4096, .dump_fd = -1, .routes_fd = -1, .quiet = 1};
  Model *model = model_open(argv[1], &options);
  CHECK(model != NULL);
  const ModelOps *ops = model->ops;
  Tokenizer *tokenizer = tok_open(argv[1], NULL);
  char prompt[256];
  ops->chat(prompt, sizeof prompt, "Name two colors.", 0, 0);
  int tokens[128], count = tok_encode_chat(tokenizer, prompt, tokens, 128);
  ModelState *state = ops->state_new();
  const float *logits = ops->prefill(state, tokens, count, 0, 0, -1, NULL, NULL);
  CHECK(logits != NULL);
  ModelSnapshot *start = ops->snapshot_take(state, count, 0);
  int token = sample_argmax(logits, ops->vocab()), first = token;
  char output[1024] = {0};
  size_t used = 0;
  float *expected = xmalloc((size_t)ops->vocab() * 4);
  for (int i = 0; i < 16 && !ops->stop(token); i++) {
    int length;
    const char *piece = tok_piece(tokenizer, token, &length);
    CHECK(used + length < sizeof output);
    memcpy(output + used, piece, length);
    used += length;
    int position = count + i;
    logits = ops->rows(&state, &token, &position, 1);
    if (!i)
      memcpy(expected, logits, (size_t)ops->vocab() * 4);
    token = sample_argmax(logits, ops->vocab());
  }
  for (size_t i = 0; i < used; i++)
    output[i] = (char)tolower((unsigned char)output[i]);
  static const char *colors[] = {"red",    "blue",  "green", "yellow", "orange",
                                 "purple", "black", "white", "pink",   "brown"};
  int found = 0;
  for (size_t i = 0; i < sizeof colors / sizeof *colors; i++)
    found += strstr(output, colors[i]) != NULL;
  printf("color response: %s\n", output);
  CHECK(found >= 2);
  ops->snapshot_restore(state, start);
  logits = ops->rows(&state, &first, &count, 1);
  CHECK(!memcmp(logits, expected, (size_t)ops->vocab() * 4));
  ops->snapshot_free(start);
  ops->state_free(state);
  tok_close(tokenizer);
  free(expected);
  puts("Flash-Next colors and complete restored-state logits passed");
  if (argc == 3 && !strcmp(argv[2], "--long")) {
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
  return 0;
}
