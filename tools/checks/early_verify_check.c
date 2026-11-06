// Early target decisions and state rollback against ordinary serial
// decoding, with real logits and the same target random stream.
#include "myds4.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int different(const float *a, const float *b) {
  int bad = 0;
  for (int i = 0; i < VOCAB; i++) bad += !isfinite(a[i]) || memcmp(a + i, b + i, sizeof(float)) != 0;
  return bad;
}

int main(int argc, char **argv) {
  if (argc < 2) die("usage: early-verify-check CHECKPOINT [--experts GGUF] [--long]");
  alarm(420);
  Options o = {.ctx = 2048, .budget_gb = 20, .reserve_gb = 4, .draft = 2, .draft_cache = 32,
               .draft_trim = 1, .lookup = VERIFY_DRAFTS, .expert_batch = 1,
               .dump_fd = -1, .routes_fd = -1, .quiet = 1};
  int n = 257;
  for (int i = 2; i < argc; i++) {
    if (!strcmp(argv[i], "--experts") && i + 1 < argc) o.experts = argv[++i];
    else if (!strcmp(argv[i], "--long")) n = 16521, o.ctx = 32768;
    else die("invalid early verification check argument");
  }
  tok_load(argv[1]);
  int seed[128], *tokens = xmalloc((size_t)n * sizeof(int));
  int length = tok_encode(" Review the queue and preserve cancellation order while sharing expert weights.\n", seed, 128);
  if (!length) die("empty fixture");
  for (int i = 0; i < n; i++) tokens[i] = seed[i % length];
  model_load(argv[1], &o);
  Sampler sampler = {.temp = 1, .top_p = 1};
  sampler_seed(&sampler, 12345);
  enum { ROWS = VERIFY_DRAFTS + 1 };
  int chain[ROWS + 2], proposal[ROWS], expected_drafts[ROWS + 1][2];
  chain[0] = sample_token(model_forward(tokens, n, 0), VOCAB, &sampler);
  Sampler initial = sampler;
  model_state_save();
  float *reference = xmalloc((size_t)(ROWS + 1) * VOCAB * sizeof(float));
  for (int r = 0; r <= ROWS; r++) {
    model_draft(chain[r], n + r, expected_drafts[r], 2);
    const float *lg = model_forward(chain + r, 1, n + r);
    memcpy(reference + (size_t)r * VOCAB, lg, VOCAB * sizeof(float));
    chain[r + 1] = sample_token(lg, VOCAB, &sampler);
  }
  int failures = 0;
  const int cuts[] = {0, 1, 2, 5, 14, 15};
  for (unsigned i = 0; i < sizeof cuts / sizeof *cuts; i++) {
    int cut = cuts[i], expected_count, accepted, next, next_drafts[2];
    memcpy(proposal, chain, sizeof proposal);
    if (cut < VERIFY_DRAFTS) proposal[cut + 1] = (proposal[cut + 1] + 1) % VOCAB;
    Sampler expected = initial, actual = initial;
    int expected_next = sample_verify(reference, VOCAB, proposal + 1, VERIFY_DRAFTS, -1, &expected, &expected_count);
    if (expected_count != cut) die("fixture failed to construct rejection at %d", cut);
    model_state_restore();
    EarlyVerifyStats before = model_early_stats();
    const float *got = model_verify_early(proposal, ROWS, n, -1, &actual, &next, &accepted);
    EarlyVerifyStats after = model_early_stats();
    int logits_bad = 0;
    for (int r = 0; r <= accepted; r++) logits_bad += different(got + (size_t)r * VOCAB, reference + (size_t)r * VOCAB);
    int sampling_bad = accepted != expected_count || next != expected_next || memcmp(actual.seed, expected.seed, sizeof actual.seed);
    model_accept(proposal, accepted + 1, n);
    model_draft(next, n + accepted + 1, next_drafts, 2);
    int draft_bad = memcmp(next_drafts, expected_drafts[accepted + 1], sizeof next_drafts) != 0;
    got = model_forward(&next, 1, n + accepted + 1);
    int continuation_bad = different(got, reference + (size_t)(accepted + 1) * VOCAB);
    printf("{\"early_verify_check\":{\"accepted\":%d,\"sampling_mismatch\":%d,\"different_logits\":%d,"
           "\"different_continuation_logits\":%d,\"different_drafts\":%d,\"cancelled_rows\":%llu,"
           "\"skipped_expert_row_layers\":%llu}}\n", accepted, sampling_bad, logits_bad, continuation_bad, draft_bad,
           (unsigned long long)(after.cancelled_rows - before.cancelled_rows),
           (unsigned long long)(after.skipped_expert_row_layers - before.skipped_expert_row_layers));
    fflush(stdout);
    failures += sampling_bad || logits_bad || continuation_bad || draft_bad;
  }
  EarlyVerifyStats totals = model_early_stats();
  if (!totals.cancelled_rows || !totals.skipped_expert_row_layers) failures++;
  printf("{\"early_verify_total\":{\"cancelled_rows\":%llu,\"skipped_expert_row_layers\":%llu,\"failures\":%d}}\n",
         (unsigned long long)totals.cancelled_rows, (unsigned long long)totals.skipped_expert_row_layers, failures);
  free(reference), free(tokens);
  return failures != 0;
}
