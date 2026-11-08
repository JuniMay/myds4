// Compare the runahead or pipeline scheduler with ordinary multi-request decoding.
#include "../../src/model/deepseek/model.h"
#include "../../src/model/deepseek/model.c"
static Tokenizer *tk;  // the checkpoint's tokenizer

static size_t differences(const void *a, const void *b, size_t bytes) {
  const uint8_t *x = a, *y = b;
  size_t count = 0;
  for (size_t i = 0; i < bytes; i++) count += x[i] != y[i];
  return count;
}

int main(int argc, char **argv) {
  if (argc < 2) die("usage: runahead-model-check CHECKPOINT [--pipeline] [--long] [--requests N] [--experts GGUF]");
  int count = 4, length = 257, pipeline = 0;
  Options o = {.ctx = 32768, .deepseek.budget_gb = 20, .reserve_gb = 4, .quiet = 1, .dump_fd = -1, .routes_fd = -1};
  for (int i = 2; i < argc; i++) {
    if (!strcmp(argv[i], "--long")) length = 16521;
    else if (!strcmp(argv[i], "--pipeline")) pipeline = 1;
    else if (!strcmp(argv[i], "--requests") && i + 1 < argc) count = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--experts") && i + 1 < argc) o.deepseek.experts = argv[++i];
    else die("invalid runahead check option");
  }
  if (count < (pipeline ? 1 : 2) || count > BATCH_ROWS) die("invalid scheduler request count");
  o.deepseek.pipeline = pipeline;
  alarm(600);
  tk = tok_open(argv[1], deepseek_ops.image_token);
  deepseek_load(argv[1], &o);
  enum { STEPS = 4 };
  State *states[FEW];
  Snapshot *prefix[FEW];
  int positions[FEW], rows[STEPS + 1][FEW], pattern[128];
  int pn = tok_encode(tk, " Review the queue, preserve cancellation, and explain the regression tests. alpha beta gamma 12345.\n", pattern, 128);
  int *prompt = xmalloc((length + FEW * 17) * sizeof(int));
  for (int r = 0; r < count; r++) {
    states[r] = deepseek_state_new();
    deepseek_state_use(states[r]);
    positions[r] = length + r * 17;
    for (int i = 0; i < positions[r]; i++) prompt[i] = pattern[(i + r * 3) % pn];
    rows[0][r] = sample_argmax(deepseek_forward(prompt, positions[r], 0), VOCAB);
    prefix[r] = deepseek_snapshot(states[r], positions[r], 0);
  }
  float *reference = xmalloc((size_t)STEPS * count * VOCAB * sizeof(float));
  runahead_enabled = pipeline_enabled = 0;
  for (int step = 0; step < STEPS; step++) {
    const float *p = deepseek_forward_rows(states, rows[step], positions, count);
    memcpy(reference + (size_t)step * count * VOCAB, p, (size_t)count * VOCAB * sizeof(float));
    for (int r = 0; r < count; r++) rows[step + 1][r] = sample_argmax(p + (size_t)r * VOCAB, VOCAB), positions[r]++;
  }
  size_t saved = saved_bytes();
  uint8_t *local[FEW], *global[FEW][LAYERS][2] = {{{0}}};
  for (int r = 0; r < count; r++) {
    deepseek_state_use(states[r]);
    local[r] = xmalloc(saved);
    rolling(states[r], (char *)local[r], 0);
    for (int l = 0; l < LAYERS; l++) if (L[l].kv_source)
      for (int k = 0; k < 2; k++) {
        Buf *b = k ? &L[l].keys : &L[l].comp;
        global[r][l][k] = xmalloc(b->size);
        memcpy(global[r][l][k], b->ptr, b->size);
      }
    deepseek_snapshot_restore(states[r], prefix[r]);
    deepseek_snapshot_free(prefix[r]);
    positions[r] -= STEPS;
  }
  runahead_enabled = !pipeline, pipeline_enabled = pipeline;
  size_t logit_diff = 0, local_diff = 0, global_diff = 0;
  int nonfinite = 0, active_changed = 0;
  for (int step = 0; step < STEPS; step++) {
    deepseek_state_use(states[0]);
    State *previous_active = active;
    const float *p = deepseek_forward_rows(states, rows[step], positions, count);
    active_changed += active != previous_active;
    logit_diff += differences(reference + (size_t)step * count * VOCAB, p, (size_t)count * VOCAB * sizeof(float));
    for (size_t i = 0; i < (size_t)count * VOCAB; i++) nonfinite += !isfinite(p[i]);
    for (int r = 0; r < count; r++) positions[r]++;
  }
  for (int r = 0; r < count; r++) {
    deepseek_state_use(states[r]);
    Snapshot *actual = deepseek_snapshot(states[r], positions[r], 0);
    local_diff += differences(local[r], actual->rows, saved);
    deepseek_snapshot_free(actual);
    for (int l = 0; l < LAYERS; l++) if (L[l].kv_source)
      for (int k = 0; k < 2; k++) {
        Buf *b = k ? &L[l].keys : &L[l].comp;
        global_diff += differences(global[r][l][k], b->ptr, b->size);
        free(global[r][l][k]);
      }
    free(local[r]);
  }
  printf("{\"%s_check\":{\"requests\":%d,\"minimum_prompt_tokens\":%d,\"steps\":%d,\"logit_bytes_differ\":%zu,"
         "\"local_bytes_differ\":%zu,\"global_bytes_differ\":%zu,\"nonfinite\":%d,\"active_state_changes\":%d,\"ready_groups\":%llu}}\n",
         pipeline ? "pipeline" : "runahead", count, length, STEPS, logit_diff, local_diff, global_diff, nonfinite, active_changed,
         (unsigned long long)runahead_group_count);
  if (pipeline) printf("{\"pipeline_stages\":{\"tasks\":%llu,\"batches\":%llu,\"mixed_stages\":%llu,\"mixed_layers\":%llu,\"max_inflight\":%llu}}\n",
      (unsigned long long)pipe_stats.tasks, (unsigned long long)pipe_stats.batches,
      (unsigned long long)pipe_stats.mixed_stages, (unsigned long long)pipe_stats.mixed_layers,
      (unsigned long long)pipe_stats.max_inflight);
  free(reference), free(prompt);
  return logit_diff || local_diff || global_diff || nonfinite || active_changed ||
         (pipeline ? pipe_stats.tasks != (count > 1 ? (uint64_t)count * LAYERS * 2 * STEPS : 0) : !runahead_group_count);
}
