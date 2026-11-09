// Verify every accepted prefix, its next target logits and its MTP history.
#include "../../src/model/model.h"
#include "../../src/model/qwen/model.h"
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define CHECK(x) do { if (!(x)) die("Qwen MTP check: %s", #x); } while (0)

static Buf read_tensor(Tensor tensor) {
  Buf result = gpu_alloc(tensor.size);
  size_t done = 0;
  while (done < tensor.size) {
    ssize_t n = pread(tensor.fd, result.ptr + done, tensor.size - done, tensor.off + done);
    CHECK(n > 0);
    done += (size_t)n;
  }
  return result;
}

static void check_hidden(const char *directory, ModelState *state, const float *expected) {
  TensorIndex *index = st_open(directory);
  Tensor weight = st_get(index, "lm_head.weight");
  Buf w = read_tensor(weight), scale = read_tensor(st_get(index, "lm_head.weight_scale"));
  Buf factor = read_tensor(st_get(index, "lm_head.weight_scale_2"));
  unsigned rows = (unsigned)weight.shape[0], cols = (unsigned)weight.shape[1] * 2;
  Buf x = gpu_alloc(cols * sizeof(float)), y = gpu_alloc(rows * sizeof(float));
  memcpy(x.ptr, qwen_state_hidden((QwenState *)state), cols * sizeof(float));
  struct { uint32_t rows, cols, n; float scale; } args = {rows, cols, 1, *(float *)factor.ptr};
  gpu_begin(); gpu_kern(gpu_kernel("mv1_nvfp4"));
  gpu_buf(0, &w, 0); gpu_buf(1, &scale, 0); gpu_buf(2, &x, 0); gpu_buf(3, &y, 0);
  gpu_bytes(4, &args, sizeof args);
  gpu_go((rows + 7) / 8, 1, 1, 256, 1, 1);
  gpu_commit(); gpu_sync();
  size_t different = 0;
  for (unsigned i = 0; i < rows; i++) different += memcmp(y.ptr + i * 4, expected + i, 4) != 0;
  printf("MTP hidden through the output head: %zu/%u logits differ\n", different, rows);
  CHECK(different == 0);
  gpu_free(&w); gpu_free(&scale); gpu_free(&factor); gpu_free(&x); gpu_free(&y);
  st_close(index);
}

int main(int argc, char **argv) {
  if (argc != 2) die("usage: qwen-mtp CHECKPOINT");
  memory_configure(32, 10);
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
  check_hidden(argv[1], state, logits);
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
      Sampler greedy = {.temp = 0, .top_p = 1};
      ops->draft(state, tokens[0], count, proposals, rows - 1, &greedy);
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
