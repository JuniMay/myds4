// Compare complete logits and state with compact, disjoint partial batches.
#include "../../src/engine/model.c"
#include "model_state_fixture.h"

static size_t different(const void *a, const void *b, size_t n) {
  const uint8_t *x = a, *y = b;
  size_t count = 0;
  for (size_t i = 0; i < n; i++) count += x[i] != y[i];
  return count;
}

int main(int argc, char **argv) {
  if (argc < 2 || argc > 5) die("usage: expert-compact-model-check CHECKPOINT [EXPERT_GGUF] [--long] [--all-missing]");
  alarm(600);
  int length = 257, all_missing = 0, waves = 0, preencode = 0, limits = 0;
  const char *experts = NULL;
  for (int i = 2; i < argc; i++) {
    if (!strcmp(argv[i], "--long")) length = 16521;
    else if (!strcmp(argv[i], "--all-missing")) all_missing = 1;
    else if (!experts) experts = argv[i];
    else die("invalid expert compaction check argument");
  }
  enum { STEPS = 8, CONTEXT = 32768 };
  int tokens[CONTEXT], pattern[128];
  tok_load(argv[1]);
  int np = tok_encode(" Review the scheduler and preserve the order of results. Explain its regression tests. alpha beta 12345.\n", pattern, 128);
  for (int i = 0; i < length; i++) tokens[i] = pattern[i % np];
  Options o = {.ctx = CONTEXT, .budget_gb = 20, .reserve_gb = 4, .quiet = 1, .dump_fd = -1, .routes_fd = -1,
               .experts = experts};
  model_load(argv[1], &o);
  expert_compact_enabled = 0;
#ifdef MYDS4_LATE_PREENCODE
  late_preencode_enabled = 0, preencode = 1;
#endif
#ifdef MYDS4_EXPERT_WAVES
  expert_waves_enabled = 0, waves = 1;
  if (all_missing) die("--all-missing is for the ordinary two-batch check");
#endif
#ifdef MYDS4_PIPELINE_LIMIT
  int limited_up = K_moe_up, limited_down = K_moe_down;
  K_moe_up = gpu_kernel_limited("moe_up", 0), K_moe_down = gpu_kernel_limited("moe_down", 0);
  limits = 1;
#endif
  int next = sample_argmax(model_forward(tokens, length, 0), VOCAB);
  FixtureState *prefix = fixture_capture();
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
  fixture_rolling(state, 0);
  for (int l = 0; l < LAYERS; l++) if (L[l].kv_source)
    for (int k = 0; k < 2; k++) {
      Buf *b = k ? &L[l].keys : &L[l].comp;
      sizes[l][k] = b->size, global[l][k] = xmalloc(b->size);
      memcpy(global[l][k], b->ptr, b->size);
    }
  fixture_restore(prefix);
  fixture_free(prefix);
  expert_compact_enabled = 1, expert_compact_force_split = all_missing ? 2 : 1;
  expert_compact_calls = expert_compact_active = 0;
#ifdef MYDS4_LATE_PREENCODE
  late_preencode_enabled = 1, late_preencode_mixed = late_preencode_missing = 0;
#endif
#ifdef MYDS4_EXPERT_WAVES
  expert_waves_enabled = 1, expert_waves_limit = 1;
  expert_wave_layers = expert_wave_batches = 0;
#endif
#ifdef MYDS4_PIPELINE_LIMIT
  K_moe_up = limited_up, K_moe_down = limited_down;
#endif
  size_t logit_diff = 0, global_diff = 0;
  int nonfinite = 0;
  for (int r = 0; r < STEPS; r++) {
    const float *lg = model_forward(tokens + length + r, 1, length + r);
    logit_diff += different(reference + (size_t)r * VOCAB, lg, VOCAB * sizeof(float));
    for (int j = 0; j < VOCAB; j++) nonfinite += !isfinite(lg[j]);
  }
  void *actual = xmalloc(bytes);
  fixture_rolling(actual, 0);
  size_t state_diff = different(state, actual, bytes);
  free(actual);
  for (int l = 0; l < LAYERS; l++) if (L[l].kv_source)
    for (int k = 0; k < 2; k++) {
      Buf *b = k ? &L[l].keys : &L[l].comp;
      if (b->size != sizes[l][k]) die("state capacity changed");
      global_diff += different(global[l][k], b->ptr, b->size);
      free(global[l][k]);
    }
  printf("{\"expert_compact_model\":{\"quant\":\"%s\",\"prompt_tokens\":%d,\"steps\":%d,"
         "\"waves\":%d,\"preencode\":%d,\"pipeline_limits\":%d,\"all_missing\":%d,"
         "\"batches\":%llu,\"active_experts\":%llu,\"logit_bytes_differ\":%zu,\"local_bytes_differ\":%zu,"
         "\"global_bytes_differ\":%zu,\"nonfinite\":%d}}\n", experts ? "iq2" : "mxfp4", length, STEPS,
         waves, preencode, limits, all_missing,
         (unsigned long long)expert_compact_calls, (unsigned long long)expert_compact_active,
         logit_diff, state_diff, global_diff, nonfinite);
  free(reference), free(state);
  uint64_t batches = (uint64_t)STEPS * LAYERS * (waves ? TOPK : all_missing ? 0 : 2);
  uint64_t contributions = all_missing ? 0 : (uint64_t)STEPS * LAYERS * TOPK;
  return logit_diff || state_diff || global_diff || nonfinite ||
         (!experts && (expert_compact_calls != batches || expert_compact_active != contributions));
}
