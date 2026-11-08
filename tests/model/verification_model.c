// Wide target verification and rejected-row rollback against serial decoding.
#include "../../src/model/deepseek/model.h"
#include "myds4.h"
#include "../../src/model/model.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
static Tokenizer *tk;  // the checkpoint's tokenizer

static int compare(const float *a, const float *b) {
  int errors = 0;
  for (int i = 0; i < VOCAB; i++) errors += !isfinite(a[i]) || !isfinite(b[i]) || memcmp(a + i, b + i, sizeof(float)) != 0;
  return errors;
}

int main(int argc, char **argv) {
  if (argc < 2) die("usage: verification-model-check CHECKPOINT [--long] [--draft] [--online] [--experts GGUF]");
  double model_memory_gib = 20;
  double model_reserve_gib = 4;
  Options o = {.ctx = 2048, .lookup = VERIFY_DRAFTS,
               .dump_fd = -1, .routes_fd = -1, .quiet = 1};
  int longer = 0;
  for (int i = 2; i < argc; i++) {
    if (!strcmp(argv[i], "--long")) longer = 1, o.ctx = 32768;
    else if (!strcmp(argv[i], "--draft")) o.draft = 2, o.deepseek.draft_cache = 32;
    else if (!strcmp(argv[i], "--online")) o.deepseek.online_predictor = 64;
    else if (!strcmp(argv[i], "--experts") && i + 1 < argc) o.deepseek.experts = argv[++i];
    else die("invalid verification check option");
  }
  alarm(420);
  tk = tok_open(argv[1], deepseek_ops.image_token);
  int seed[128], n = longer ? 16521 : 257;
  int *tokens = xmalloc((size_t)n * sizeof *tokens);
  int length = tok_encode(tk, " Review the queue, keep duplicate jobs, and preserve cancellation order.\n", seed, 128);
  if (!length) die("empty fixture prompt");
  for (int i = 0; i < n; i++) tokens[i] = seed[i % length];
  memory_configure(model_memory_gib, 8, model_reserve_gib);
  deepseek_load(argv[1], &o);
  int rows[VERIFY_DRAFTS + 2];
  rows[0] = sample_argmax(deepseek_forward(tokens, n, 0), VOCAB);
  for (int i = 1; i < VERIFY_DRAFTS + 2; i++) rows[i] = seed[(i * 7) % length];
  deepseek_state_save();
  float *reference = xmalloc((size_t)(VERIFY_DRAFTS + 2) * VOCAB * sizeof(float));
  int draft_reference[VERIFY_DRAFTS + 2][2];
  for (int r = 0; r < VERIFY_DRAFTS + 2; r++) {
    if (o.draft) deepseek_draft(rows[r], n + r, draft_reference[r], 2);
    memcpy(reference + (size_t)r * VOCAB, deepseek_forward(rows + r, 1, n + r), VOCAB * sizeof(float));
  }
  deepseek_state_restore();
  const float *got = deepseek_verify(rows, VERIFY_DRAFTS + 1, n);
  int failed = 0;
  for (int r = 0; r <= VERIFY_DRAFTS; r++) {
    int different = compare(got + (size_t)r * VOCAB, reference + (size_t)r * VOCAB);
    printf("{\"verify_row\":%d,\"different_logits\":%d}\n", r, different);
    failed |= different != 0;
  }
  fflush(stdout);
  const int kept_cases[] = {1, 2, 6, 15, 16};
  for (size_t i = 0; i < sizeof kept_cases / sizeof *kept_cases; i++) {
    int kept = kept_cases[i], drafts[2], draft_diff = 0;
    deepseek_state_restore();
    deepseek_verify(rows, VERIFY_DRAFTS + 1, n);
    deepseek_accept(rows, kept, n);
    if (o.draft) {
      deepseek_draft(rows[kept], n + kept, drafts, 2);
      draft_diff = memcmp(drafts, draft_reference[kept], sizeof drafts) != 0;
    }
    got = deepseek_forward(rows + kept, 1, n + kept);
    int different = compare(got, reference + (size_t)kept * VOCAB);
    printf("{\"rollback_kept\":%d,\"different_logits\":%d,\"different_drafts\":%d}\n", kept, different, draft_diff);
    fflush(stdout);
    failed |= different != 0 || draft_diff;
  }
  free(reference), free(tokens);
  return failed;
}
