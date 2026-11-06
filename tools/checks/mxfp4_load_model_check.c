// Compare experimental MXFP4 decode kernels with the ordinary kernels on
// the same loaded model, prefix state and forced continuation.
#define MYDS4_FP4_LOADS 4
#include "../../src/engine/model.c"
#include "model_state_fixture.h"

static size_t different(const void *a, const void *b, size_t n) {
  const uint8_t *x = a, *y = b;
  size_t count = 0;
  for (size_t i = 0; i < n; i++) count += x[i] != y[i];
  return count;
}

int main(int argc, char **argv) {
  if (argc < 2 || argc > 3 || (argc == 3 && strcmp(argv[2], "--stdin")))
    die("usage: mxfp4-load-model-check CHECKPOINT [--stdin]");
  alarm(600);
  enum { STEPS = 16, CONTEXT = 32768 };
  int *tokens = xmalloc(CONTEXT * sizeof *tokens), n = 8192;
  tok_load(argv[1]);
  if (argc == 3) {
    size_t limit = 512u << 10;
    char *text = xmalloc(limit + 1);
    size_t bytes = fread(text, 1, limit, stdin);
    if (!bytes || bytes == limit || ferror(stdin)) die("rendered input exceeds its bound");
    text[bytes] = 0;
    n = tok_encode_chat(text, tokens, CONTEXT);
    free(text);
  } else {
    int pattern[128];
    int count = tok_encode(" Review this source code and preserve cancellation order. Explain the regression tests. alpha beta gamma 12345.\n", pattern, 128);
    for (int i = 0; i < n; i++) tokens[i] = pattern[i % count];
  }
  if (n < 1 || n + STEPS >= CONTEXT) die("invalid model-check context length");
  Options o = {.ctx = CONTEXT, .budget_gb = 20, .reserve_gb = 4, .quiet = 1, .dump_fd = -1, .routes_fd = -1};
  model_load(argv[1], &o);
  int ordinary_up = gpu_kernel("moe_up"), ordinary_down = gpu_kernel("moe_down");
  K_moe_up = ordinary_up, K_moe_down = ordinary_down;
  const float *lg = model_forward(tokens, n, 0);
  int next = sample_argmax(lg, VOCAB);
  FixtureState *prefix = fixture_capture();
  float *reference = xmalloc((size_t)STEPS * VOCAB * sizeof(float));
  for (int i = 0; i < STEPS; i++) {
    tokens[n + i] = next;
    lg = model_forward(tokens + n + i, 1, n + i);
    memcpy(reference + (size_t)i * VOCAB, lg, VOCAB * sizeof(float));
    next = sample_argmax(lg, VOCAB);
  }
  size_t state_bytes = saved_bytes();
  uint8_t *state = xmalloc(state_bytes);
  fixture_rolling(state, 0);
  uint8_t *global[LAYERS][2] = {{0}};
  size_t sizes[LAYERS][2] = {{0}};
  for (int l = 0; l < LAYERS; l++) if (L[l].kv_source)
    for (int k = 0; k < 2; k++) {
      Buf *buffer = k ? &L[l].keys : &L[l].comp;
      sizes[l][k] = buffer->size;
      global[l][k] = xmalloc(buffer->size);
      memcpy(global[l][k], buffer->ptr, buffer->size);
    }
  const char *names[][3] = {{"row", "moe_up_row", "moe_down_row"}, {"factor", "moe_up_factor", "moe_down_factor"}};
  int failures = 0;
  for (int variant = 0; variant < 2; variant++) {
    fixture_restore(prefix);
    K_moe_up = gpu_kernel(names[variant][1]), K_moe_down = gpu_kernel(names[variant][2]);
    size_t logit_bytes = 0, state_diff = 0, global_diff = 0;
    int nonfinite = 0;
    for (int i = 0; i < STEPS; i++) {
      lg = model_forward(tokens + n + i, 1, n + i);
      logit_bytes += different(reference + (size_t)i * VOCAB, lg, VOCAB * sizeof(float));
      for (int j = 0; j < VOCAB; j++) nonfinite += !isfinite(lg[j]) || !isfinite(reference[(size_t)i * VOCAB + j]);
    }
    void *actual = xmalloc(state_bytes);
    fixture_rolling(actual, 0);
    state_diff = different(state, actual, state_bytes);
    free(actual);
    for (int l = 0; l < LAYERS; l++) if (L[l].kv_source)
      for (int k = 0; k < 2; k++) {
        Buf *buffer = k ? &L[l].keys : &L[l].comp;
        if (buffer->size != sizes[l][k]) die("state capacity changed during comparison");
        global_diff += different(global[l][k], buffer->ptr, buffer->size);
      }
    printf("{\"mxfp4_load_model\":{\"variant\":\"%s\",\"prompt_tokens\":%d,\"continuation_tokens\":%d,"
           "\"different_logit_bytes\":%zu,\"different_state_bytes\":%zu,\"different_global_bytes\":%zu,\"nonfinite_logits\":%d}}\n",
           names[variant][0], n, STEPS, logit_bytes, state_diff, global_diff, nonfinite);
    failures += logit_bytes || state_diff || global_diff || nonfinite;
  }
  for (int l = 0; l < LAYERS; l++) for (int k = 0; k < 2; k++) free(global[l][k]);
  fixture_free(prefix), free(reference), free(state), free(tokens);
  return failures != 0;
}
