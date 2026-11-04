// Compare dependency-pruned prefill with the full model at identical batch boundaries.
#include "../../src/engine/model.c"

static size_t bytes_differ(const void *a, const void *b, size_t count) {
  const uint8_t *x = a, *y = b;
  size_t different = 0;
  for (size_t i = 0; i < count; i++) different += x[i] != y[i];
  return different;
}

int main(int argc, char **argv) {
  if (argc < 2) die("usage: prefill-prune-model-check CHECKPOINT [--stdin|--long] [--prefix N] [--wide] [--resume] [--no-draft] [--gpu-route] [--experts GGUF]");
  Options o = {.ctx = 32768, .budget_gb = 20, .reserve_gb = 4, .draft = 2, .draft_cache = 32,
               .draft_trim = 1, .expert_batch = 1, .quiet = 1, .dump_fd = -1, .routes_fd = -1};
  int longer = 0, resume = 0, prefix = 1024, prefix_set = 0, input = 0;
  for (int i = 2; i < argc; i++) {
    if (!strcmp(argv[i], "--long")) longer = 1;
    else if (!strcmp(argv[i], "--prefix") && i + 1 < argc) prefix = atoi(argv[++i]), prefix_set = 1;
    else if (!strcmp(argv[i], "--stdin")) input = 1;
    else if (!strcmp(argv[i], "--wide")) o.lookup = 15;
    else if (!strcmp(argv[i], "--resume")) resume = 1;
    else if (!strcmp(argv[i], "--no-draft")) o.draft = o.draft_cache = o.draft_trim = 0;
    else if (!strcmp(argv[i], "--gpu-route")) o.gpu_route = 1;
    else if (!strcmp(argv[i], "--experts") && i + 1 < argc) o.experts = argv[++i];
    else die("invalid pruning check option");
  }
  alarm(600);
  tok_load(argv[1]);
  int pattern[128], n = longer ? CHUNK + 137 : 8192;
  int *tokens = xmalloc((size_t)o.ctx * sizeof *tokens);
  if (input) {
    size_t limit = 512u << 10;
    char *text = xmalloc(limit + 1);
    size_t count = fread(text, 1, limit, stdin);
    if (!count || count == limit || ferror(stdin)) die("invalid rendered input");
    text[count] = 0;
    n = tok_encode_chat(text, tokens, o.ctx);
    free(text);
    if (!prefix_set) {
      int user = tok_find("<｜User｜>");
      prefix = 0;
      while (prefix < n && tokens[prefix] != user) prefix++;
    }
  } else {
    int pn = tok_encode(" Review this code, preserve cancellation order, and explain the tests. alpha beta gamma 12345.\n", pattern, 128);
    for (int i = 0; i < n; i++) tokens[i] = pattern[i % pn];
  }
  if (prefix < 1 || prefix >= n || n + 8 > o.ctx) die("invalid prefix or input length");
  model_load(argv[1], &o);
  opt.prefill_prune = 0;
  float *reference = xmalloc(VOCAB * sizeof(float));
  memcpy(reference, model_forward_prefix(tokens, n, 0, prefix, NULL), VOCAB * sizeof(float));
  model_state_save(0);
  size_t state_size = saved_bytes();
  uint8_t *state = xmalloc(state_size), *saved = xmalloc(state_size);
  memcpy(state, active->saved[0], state_size), memcpy(saved, active->saved[1], state_size);
  uint8_t *globals[LAYERS][2] = {{0}};
  size_t sizes[LAYERS][2] = {{0}};
  for (int l = 0; l < LAYERS; l++) if (L[l].kv_source)
    for (int k = 0; k < 2; k++) {
      Buf *b = k ? &L[l].keys : &L[l].comp;
      sizes[l][k] = b->size;
      globals[l][k] = xmalloc(b->size);
      memcpy(globals[l][k], b->ptr, b->size);
    }
  int draft_reference[2] = {0}, first = sample_argmax(reference, VOCAB);
  if (o.draft) model_draft(first, n, draft_reference, 2);
  enum { STEPS = 8 };
  int continuation[STEPS];
  float *future = xmalloc((size_t)STEPS * VOCAB * sizeof(float));
  const float *lg = reference;
  for (int i = 0; i < STEPS; i++) {
    continuation[i] = sample_argmax(lg, VOCAB);
    lg = model_forward(continuation + i, 1, n + i);
    memcpy(future + (size_t)i * VOCAB, lg, VOCAB * sizeof(float));
  }
  model_reset();
  opt.prefill_prune = 1;
  int encoder_only = -1;
  lg = model_forward_prefix(tokens, n, 0, prefix, &encoder_only);
  ModelForwardInfo info = model_forward_info();
  size_t logit_diff = bytes_differ(reference, lg, VOCAB * sizeof(float));
  for (int i = 0; i < VOCAB; i++) logit_diff += !isfinite(reference[i]) || !isfinite(lg[i]);
  model_state_save(0);
  size_t state_diff = bytes_differ(state, active->saved[0], state_size);
  size_t prefix_diff = bytes_differ(saved, active->saved[1], state_size), global_diff = 0;
  for (int l = 0; l < LAYERS; l++) if (L[l].kv_source)
    for (int k = 0; k < 2; k++) {
      Buf *b = k ? &L[l].keys : &L[l].comp;
      if (sizes[l][k] != b->size) die("global capacity changed between comparisons");
      global_diff += bytes_differ(globals[l][k], b->ptr, b->size);
    }
  int drafts[2] = {0}, draft_diff = 0;
  if (o.draft) model_draft(first, n, drafts, 2), draft_diff = memcmp(drafts, draft_reference, sizeof drafts) != 0;
  size_t continuation_diff = 0;
  for (int i = 0; i < STEPS; i++)
    continuation_diff += bytes_differ(future + (size_t)i * VOCAB, model_forward(continuation + i, 1, n + i), VOCAB * sizeof(float));
  size_t resume_diff = 0;
  if (resume) {
    opt.prefill_prune = 0;
    model_state_restore(1), model_seek(tokens, prefix);
    memcpy(reference, model_forward(tokens + prefix, n - prefix, prefix), VOCAB * sizeof(float));
    opt.prefill_prune = 1;
    model_state_restore(1), model_seek(tokens, prefix);
    resume_diff = bytes_differ(reference, model_forward(tokens + prefix, n - prefix, prefix), VOCAB * sizeof(float));
  }
  printf("{\"prefill_prune_check\":{\"rendered_input\":%s,\"prompt_tokens\":%d,\"prefix_tokens\":%d,\"decoder_tokens\":%d,"
         "\"chunks\":%d,\"wide\":%d,\"draft\":%d,\"different_logit_bytes\":%zu,\"different_state_bytes\":%zu,"
         "\"different_prefix_bytes\":%zu,\"different_global_bytes\":%zu,\"different_continuation_bytes\":%zu,"
         "\"different_drafts\":%d,\"different_resume_bytes\":%zu,\"encoder_only\":%d}}\n",
         input ? "true" : "false", n, prefix, info.decoder_tokens, info.chunks, o.lookup, o.draft, logit_diff, state_diff,
         prefix_diff, global_diff, continuation_diff, draft_diff, resume_diff, encoder_only);
  return logit_diff || state_diff || prefix_diff || global_diff || continuation_diff || draft_diff || resume_diff ||
         encoder_only != 0 || !info.prune_applied || info.decoder_tokens >= n;
}
