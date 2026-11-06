// Compare all logits and retained state with phased reads on and off.
#include "../../src/engine/model.c"
#include "model_state_fixture.h"

static size_t different(const void *a, const void *b, size_t n) {
  const uint8_t *x = a, *y = b;
  size_t count = 0;
  for (size_t i = 0; i < n; i++) count += x[i] != y[i];
  return count;
}

int main(int argc, char **argv) {
  if (argc < 2 || argc > 4) die("usage: expert-phase-model-check CHECKPOINT [EXPERT_GGUF] [--long]");
  alarm(600);
  int length = 257;
  const char *experts = NULL;
  for (int i = 2; i < argc; i++) {
    if (!strcmp(argv[i], "--long")) length = 16521;
    else if (!experts) experts = argv[i];
    else die("invalid expert-phase-model-check arguments");
  }
  enum { STEPS = 16, CONTEXT = 32768 };
  int tokens[CONTEXT], pattern[128];
  tok_load(argv[1]);
  int np = tok_encode(" Review the job scheduler and preserve the order of results. Explain the regression tests. alpha beta 12345.\n", pattern, 128);
  for (int i = 0; i < length; i++) tokens[i] = pattern[i % np];
  Options o = {.ctx = CONTEXT, .budget_gb = 20, .reserve_gb = 8, .quiet = 1, .dump_fd = -1, .routes_fd = -1,
               .experts = experts, .draft = 2, .draft_cache = 32, .draft_trim = 1};
  model_load(argv[1], &o);
  expert_phase_enabled = 0;
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
  int draft[2], actual_draft[2];
  model_draft(next, length + STEPS, draft, 2);
  fixture_restore(prefix);
  expert_phase_enabled = 1;
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
  model_draft(next, length + STEPS, actual_draft, 2);
  int draft_diff = memcmp(draft, actual_draft, sizeof draft) != 0;
  printf("{\"expert_phase_model\":{\"quant\":\"%s\",\"prompt_tokens\":%d,\"continuation_tokens\":%d,"
         "\"phased_layers\":%llu,\"different_logit_bytes\":%zu,\"different_state_bytes\":%zu,"
         "\"different_global_bytes\":%zu,\"drafts_differ\":%s,\"nonfinite\":%d}}\n",
         experts ? "iq2" : "mxfp4", length, STEPS, (unsigned long long)expert_phase_layers,
         logit_diff, state_diff, global_diff, draft_diff ? "true" : "false", nonfinite);
  fixture_free(prefix), free(reference), free(state);
  return logit_diff || state_diff || global_diff || draft_diff || nonfinite || (!experts && !expert_phase_layers);
}
