// Compare complete logits and state with one selected expert per GPU wave.
#include "../../src/engine/model.c"

static size_t different(const void *a, const void *b, size_t n) {
  const uint8_t *x = a, *y = b;
  size_t count = 0;
  for (size_t i = 0; i < n; i++) count += x[i] != y[i];
  return count;
}

int main(int argc, char **argv) {
  if (argc < 2 || argc > 4) die("usage: expert-waves-model-check CHECKPOINT [EXPERT_GGUF] [--long]");
  alarm(600);
  int length = 257;
  const char *experts = NULL;
  for (int i = 2; i < argc; i++) {
    if (!strcmp(argv[i], "--long")) length = 16521;
    else if (!experts) experts = argv[i];
    else die("invalid expert wave check argument");
  }
  enum { STEPS = 8, CONTEXT = 32768 };
  int tokens[CONTEXT], pattern[128];
  tok_load(argv[1]);
  int np = tok_encode(" Review the scheduler and preserve the order of results. Explain its regression tests. alpha beta 12345.\n", pattern, 128);
  for (int i = 0; i < length; i++) tokens[i] = pattern[i % np];
  Options o = {.ctx = CONTEXT, .budget_gb = 20, .reserve_gb = 4, .quiet = 1, .dump_fd = -1, .routes_fd = -1,
               .experts = experts};
  model_load(argv[1], &o);
  expert_waves_enabled = 0;
  int next = sample_argmax(model_forward(tokens, length, 0), VOCAB);
  Snapshot *prefix = model_snapshot(active, length, 0);
  float *reference = xmalloc((size_t)STEPS * VOCAB * sizeof(float));
  for (int r = 0; r < STEPS; r++) {
    tokens[length + r] = next;
    const float *lg = model_forward(tokens + length + r, 1, length + r);
    memcpy(reference + (size_t)r * VOCAB, lg, VOCAB * sizeof(float));
    next = sample_argmax(lg, VOCAB);
  }
  size_t bytes = saved_bytes();
  uint8_t *state = xmalloc(bytes), *global[LAYERS][2] = {{0}};
  size_t sizes[LAYERS][2] = {{0}};
  rolling(active, (char *)state, 0);
  for (int l = 0; l < LAYERS; l++) if (L[l].kv_source)
    for (int k = 0; k < 2; k++) {
      Buf *b = k ? &L[l].keys : &L[l].comp;
      sizes[l][k] = b->size, global[l][k] = xmalloc(b->size);
      memcpy(global[l][k], b->ptr, b->size);
    }
  model_snapshot_restore(active, prefix);
  model_snapshot_free(prefix);
  expert_waves_enabled = 1, expert_waves_limit = 1;
  size_t logit_diff = 0, global_diff = 0;
  int nonfinite = 0;
  for (int r = 0; r < STEPS; r++) {
    const float *lg = model_forward(tokens + length + r, 1, length + r);
    logit_diff += different(reference + (size_t)r * VOCAB, lg, VOCAB * sizeof(float));
    for (int j = 0; j < VOCAB; j++) nonfinite += !isfinite(lg[j]);
  }
  Snapshot *actual = model_snapshot(active, length + STEPS, 0);
  size_t state_diff = different(state, actual->rows, bytes);
  model_snapshot_free(actual);
  for (int l = 0; l < LAYERS; l++) if (L[l].kv_source)
    for (int k = 0; k < 2; k++) {
      Buf *b = k ? &L[l].keys : &L[l].comp;
      if (b->size != sizes[l][k]) die("state capacity changed");
      global_diff += different(global[l][k], b->ptr, b->size);
      free(global[l][k]);
    }
  printf("{\"expert_waves_model\":{\"quant\":\"%s\",\"prompt_tokens\":%d,\"steps\":%d,\"max_batches\":%d,"
         "\"layers\":%llu,\"batches\":%llu,\"logit_bytes_differ\":%zu,\"local_bytes_differ\":%zu,"
         "\"global_bytes_differ\":%zu,\"nonfinite\":%d}}\n", experts ? "iq2" : "mxfp4", length, STEPS, expert_wave_cap,
         (unsigned long long)expert_wave_layers, (unsigned long long)expert_wave_batches,
         logit_diff, state_diff, global_diff, nonfinite);
  free(reference), free(state);
  int groups = expert_wave_cap && expert_wave_cap < TOPK ? expert_wave_cap : TOPK;
  return logit_diff || state_diff || global_diff || nonfinite ||
         expert_wave_layers != STEPS * LAYERS || expert_wave_batches != (uint64_t)STEPS * LAYERS * groups;
}
