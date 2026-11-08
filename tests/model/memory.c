// Real weight eviction/reload with a retained KV state. No throughput claim.
#include "../../src/model/model.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <unistd.h>

#define CHECK(test) do { if (!(test)) die("memory model check failed: %s", #test); } while (0)

int main(int argc, char **argv) {
  if (argc < 2 || argc > 3) die("usage: memory-model CHECKPOINT [MEMORY_GIB]");
  memory_init((uint64_t)((argc == 3 ? strtod(argv[2], NULL) : 24) * 1073741824.0));
  Options options = {.ctx = 4096, .dump_fd = -1, .routes_fd = -1, .quiet = 1};
  Model *model = model_open(argv[1], &options);
  CHECK(model && model->ops->weight_slot);
  const ModelOps *ops = model->ops;
  ModelState *state = ops->state_new();
  CHECK(memory_usage().kv > 0);
  Tokenizer *tokenizer = tok_open(argv[1], NULL);
  int tokens[128], n = tok_encode(tokenizer, "Describe a blue triangle in one sentence.", tokens, 128);
  const float *prefill = ops->prefill(state, tokens, n, 0, 0, -1, NULL, NULL);
  CHECK(prefill != NULL);
  if (ops->plan) { ModelPlan plan; ops->plan(&plan); printf("prefill chunks: %d for %d input tokens\n", plan.chunks, n); }
  for (int i = 0; i < ops->vocab(); i++) CHECK(isfinite(prefill[i]));
  int token = sample_argmax(prefill, ops->vocab()), position = n;
  ModelSnapshot *snapshot = ops->snapshot_take(state, n, 0);
  const float *row = ops->rows(&state, &token, &position, 1);
  size_t bytes = (size_t)ops->vocab() * sizeof(float);
  float *expected = xmalloc(bytes);
  memcpy(expected, row, bytes);
  ops->snapshot_restore(state, snapshot);
  ops->seek(state, tokens, n);

  MemorySlot *weights = ops->weight_slot();
  uint64_t kv = memory_usage().kv, weight_bytes = weights->bytes;
  MemoryAccount competitor;
  MemorySlot competing_weights;
  memory_account_init(&competitor, "synthetic competing model");
  memory_slot_init(&competing_weights, &competitor, NULL, "dense weights", NULL, NULL);
  uint64_t fixed = memory_usage().allocated - model->memory.bytes + model->work.payload;
  uint64_t amount = memory_usage().limit - fixed;
  MemorySlot *request[] = {&competing_weights};
  CHECK(!memory_admit(request, &amount, 1, MEMORY_REQUIRED));
  CHECK(memory_ready(weights));
  model_active(model, 0);
  uint64_t footprint_before = mem_footprint();
  CHECK(memory_admit(request, &amount, 1, MEMORY_REQUIRED));
  CHECK(!memory_ready(weights) && weights->payload == 0);
  CHECK(memory_usage().kv == kv);
  uint64_t footprint_after = mem_footprint();
  // Metal's physical-memory ledger catches up after buffer release.
  uint64_t retained = footprint_before - weight_bytes + (1ull << 30);
  for (int i = 0; i < 100 && footprint_after > retained; i++) {
    usleep(10000);
    footprint_after = mem_footprint();
  }
  printf("process footprint around eviction: %.3f -> %.3f GiB\n", footprint_before / 1073741824.0, footprint_after / 1073741824.0);
  CHECK(footprint_after <= retained);
  memory_slot_remove(&competing_weights);
  CHECK(model_load(model, &options));
  model_active(model, 1);
  CHECK(memory_ready(weights) && memory_usage().kv == kv);
  row = ops->rows(&state, &token, &position, 1);
  CHECK(row != NULL && !memcmp(row, expected, bytes));
  printf("weight reload: all %d logits identical; %.3f GiB weights evicted together; %.3f GiB KV retained\n",
         ops->vocab(), weight_bytes / 1073741824.0, kv / 1073741824.0);
  ops->snapshot_free(snapshot);
  ops->state_free(state);
  free(expected);
  tok_close(tokenizer);
  return 0;
}
