// Compare dependency-pruned prefill with the full model at identical batch boundaries.
#include "../../src/model/deepseek/model.h"
#include "../../src/model/deepseek/model.c"
static Tokenizer *tk;  // the checkpoint's tokenizer

static size_t bytes_differ(const void *a, const void *b, size_t count) {
  const uint8_t *x = a, *y = b;
  size_t different = 0;
  for (size_t i = 0; i < count; i++) different += x[i] != y[i];
  return different;
}

int main(int argc, char **argv) {
  if (argc < 2) die("usage: prefill-prune-model-check CHECKPOINT [--stdin|--long] [--tokens N] [--chunk32] [--prefix N] [--wide] [--resume] [--no-draft] [--experts GGUF]");
  double model_memory_gib = 20;
  double model_reserve_gib = 4;
  Options o = {.ctx = 131072, .draft = 2, .deepseek.draft_cache = 32,
               .quiet = 1, .dump_fd = -1, .routes_fd = -1};
  int longer = 0, resume = 0, prefix = 1024, prefix_set = 0, input = 0, chunk32 = 0, count = 0;
  for (int i = 2; i < argc; i++) {
    if (!strcmp(argv[i], "--long")) longer = 1;
    else if (!strcmp(argv[i], "--chunk32")) chunk32 = 1;
    else if (!strcmp(argv[i], "--tokens") && i + 1 < argc) count = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--prefix") && i + 1 < argc) prefix = atoi(argv[++i]), prefix_set = 1;
    else if (!strcmp(argv[i], "--stdin")) input = 1;
    else if (!strcmp(argv[i], "--wide")) o.lookup = 15;
    else if (!strcmp(argv[i], "--resume")) resume = 1;
    else if (!strcmp(argv[i], "--no-draft")) o.draft = o.deepseek.draft_cache = 0;
    else if (!strcmp(argv[i], "--experts") && i + 1 < argc) o.deepseek.experts = argv[++i];
    else die("invalid pruning check option");
  }
  alarm(600);
  tk = tok_open(argv[1], deepseek_ops.media_token);
  int pattern[128], n = count ? count : longer ? 16384 + 137 : 8192;
  if (n < 2 || n + 8 > o.ctx) die("invalid fixture token count");
  int *tokens = xmalloc((size_t)o.ctx * sizeof *tokens);
  if (input) {
    size_t limit = 512u << 10;
    char *text = xmalloc(limit + 1);
    size_t count = fread(text, 1, limit, stdin);
    if (!count || count == limit || ferror(stdin)) die("invalid rendered input");
    text[count] = 0;
    n = tok_encode_chat(tk, text, tokens, o.ctx);
    free(text);
    if (!prefix_set) {
      int user = tok_find(tk, "<｜User｜>");
      prefix = 0;
      while (prefix < n && tokens[prefix] != user) prefix++;
    }
  } else {
    int pn = tok_encode(tk, " Review this code, preserve cancellation order, and explain the tests. alpha beta gamma 12345.\n", pattern, 128);
    for (int i = 0; i < n; i++) tokens[i] = pattern[i % pn];
  }
  if (prefix < 1 || prefix >= n || n + 8 > o.ctx) die("invalid prefix or input length");
  memory_configure(model_memory_gib, model_reserve_gib);
  deepseek_load(argv[1], &o);
  prefill_chunk = 16384;
  opt.deepseek.prefill_prune = 0;
  float *reference = xmalloc(VOCAB * sizeof(float));
  Snapshot *captured_reference = 0, *captured = 0;  // the rolling rows at prefix, taken during each prefill
  memcpy(reference, deepseek_prefill(tokens, n, 0, 0, prefix, &captured_reference), VOCAB * sizeof(float));
  deepseek_state_save();
  size_t state_size = saved_bytes();
  uint8_t *state = xmalloc(state_size);
  memcpy(state, active->saved, state_size);
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
  if (o.draft) deepseek_draft(first, n, draft_reference, 2);
  enum { STEPS = 8 };
  int continuation[STEPS];
  float *future = xmalloc((size_t)STEPS * VOCAB * sizeof(float));
  const float *lg = reference;
  for (int i = 0; i < STEPS; i++) {
    continuation[i] = sample_argmax(lg, VOCAB);
    lg = deepseek_forward(continuation + i, 1, n + i);
    memcpy(future + (size_t)i * VOCAB, lg, VOCAB * sizeof(float));
  }
  deepseek_reset();
  opt.deepseek.prefill_prune = 1;
  prefill_chunk = chunk32 ? 32768 : 16384;
  lg = deepseek_prefill(tokens, n, 0, 0, prefix, &captured);
  int encoder_only = deepseek_snapshot_extent(captured) != LAYERS - 1;
  DeepSeekForwardInfo info = deepseek_forward_info();
  int candidates = n;
  if (n > CAND_TOP * 8) {
    const uint32_t *bits = (const uint32_t *)CAND.ptr + (size_t)((n - 1) % prefill_chunk) * cand_words;
    candidates = 0;
    for (int i = 0; i < n; i++) candidates += (bits[i >> 8] >> ((i >> 3) & 31)) & 1;
  }
  size_t logit_diff = bytes_differ(reference, lg, VOCAB * sizeof(float));
  for (int i = 0; i < VOCAB; i++) logit_diff += !isfinite(reference[i]) || !isfinite(lg[i]);
  deepseek_state_save();
  size_t state_diff = bytes_differ(state, active->saved, state_size);
  size_t prefix_diff = bytes_differ(captured_reference->rows, captured->rows, state_size), global_diff = 0;
  for (int l = 0; l < LAYERS; l++) if (L[l].kv_source)
    for (int k = 0; k < 2; k++) {
      Buf *b = k ? &L[l].keys : &L[l].comp;
      if (sizes[l][k] != b->size) die("global capacity changed between comparisons");
      global_diff += bytes_differ(globals[l][k], b->ptr, b->size);
    }
  int drafts[2] = {0}, draft_diff = 0;
  if (o.draft) deepseek_draft(first, n, drafts, 2), draft_diff = memcmp(drafts, draft_reference, sizeof drafts) != 0;
  size_t continuation_diff = 0;
  for (int i = 0; i < STEPS; i++)
    continuation_diff += bytes_differ(future + (size_t)i * VOCAB, deepseek_forward(continuation + i, 1, n + i), VOCAB * sizeof(float));
  size_t resume_diff = 0;
  if (resume) {  // the state's rows before prefix are the same; its rolling rows come from each capture
    opt.deepseek.prefill_prune = 0;
    prefill_chunk = 16384;
    deepseek_snapshot_restore(active, captured_reference);
    memcpy(reference, deepseek_prefill(tokens, n, prefix, 0, -1, 0), VOCAB * sizeof(float));
    opt.deepseek.prefill_prune = 1;
    prefill_chunk = chunk32 ? 32768 : 16384;
    deepseek_snapshot_restore(active, captured);
    resume_diff = bytes_differ(reference, deepseek_prefill(tokens, n, prefix, 0, -1, 0), VOCAB * sizeof(float));
  }
  deepseek_snapshot_free(captured_reference), deepseek_snapshot_free(captured);
  printf("{\"prefill_prune_check\":{\"rendered_input\":%s,\"last_query_candidates\":%d,\"context\":%d,\"prompt_tokens\":%d,\"prefix_tokens\":%d,\"decoder_tokens\":%d,\"decoder_row_layers\":%d,\"chunk_rows\":%d,"
         "\"chunks\":%d,\"wide\":%d,\"draft\":%d,\"different_logit_bytes\":%zu,\"different_state_bytes\":%zu,"
         "\"different_prefix_bytes\":%zu,\"different_global_bytes\":%zu,\"different_continuation_bytes\":%zu,"
         "\"different_drafts\":%d,\"different_resume_bytes\":%zu,\"encoder_only\":%d}}\n",
         input ? "true" : "false", candidates, o.ctx, n, prefix,
         info.decoder_tokens, info.decoder_row_layers, prefill_chunk,
         info.chunks, o.lookup, o.draft, logit_diff, state_diff,
         prefix_diff, global_diff, continuation_diff, draft_diff, resume_diff, encoder_only);
  return logit_diff || state_diff || prefix_diff || global_diff || continuation_diff || draft_diff || resume_diff ||
         encoder_only != 0 || !info.prune_applied || info.decoder_tokens >= n;
}
