// Compare late pre-encoding with ordinary submission and full model state.
#include "myds4.h"
static void checked_wait(int layer, const int *experts, int n);
#define cache_wait checked_wait
#include "../../src/engine/model.c"
#include "model_state_fixture.h"
#undef cache_wait
static uint64_t unsubmitted_checks, premature_submissions;
static void checked_wait(int layer, const int *experts, int n) {
  uint64_t encoded = late_preencode_mixed + late_preencode_missing;
  if (late_preencode_enabled && encoded > unsubmitted_checks) {
    // gpu_open enqueues the command; status 1 means it is not committed.
    premature_submissions += gpu_command_status() != 1;
    unsubmitted_checks++;
  }
  cache_wait(layer, experts, n);
}

static size_t different(const void *a, const void *b, size_t n) {
  const uint8_t *x = a, *y = b;
  size_t count = 0;
  for (size_t i = 0; i < n; i++) count += x[i] != y[i];
  return count;
}

int main(int argc, char **argv) {
  if (argc < 2 || argc > 5) die("usage: late-preencode-model-check CHECKPOINT [EXPERT_GGUF] [--long] [--all-missing]");
  alarm(600);
  int length = 257, all_missing = 0;
  const char *experts = NULL;
  for (int i = 2; i < argc; i++) {
    if (!strcmp(argv[i], "--all-missing")) all_missing = 1;
    else if (!strcmp(argv[i], "--long")) length = 16521;
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
  expert_compact_enabled = 0, late_preencode_enabled = 0;
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
  late_preencode_enabled = 1, expert_compact_force_split = all_missing ? 2 : 1;
  late_preencode_mixed = late_preencode_missing = 0;
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
  printf("{\"late_preencode_model\":{\"quant\":\"%s\",\"prompt_tokens\":%d,\"steps\":%d,"
         "\"mixed\":%llu,\"all_missing\":%llu,\"unsubmitted_checks\":%llu,\"premature_submissions\":%llu,"
         "\"logit_bytes_differ\":%zu,\"local_bytes_differ\":%zu,\"global_bytes_differ\":%zu,\"nonfinite\":%d}}\n",
         experts ? "iq2" : "mxfp4", length, STEPS,
         (unsigned long long)late_preencode_mixed, (unsigned long long)late_preencode_missing,
         (unsigned long long)unsubmitted_checks, (unsigned long long)premature_submissions,
         logit_diff, state_diff, global_diff, nonfinite);
  free(reference), free(state);
  return logit_diff || state_diff || global_diff || nonfinite || premature_submissions ||
         (!experts && unsubmitted_checks != STEPS * LAYERS);
}
