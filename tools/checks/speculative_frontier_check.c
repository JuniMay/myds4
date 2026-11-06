// Asynchronous readiness must preserve the target sampler and its accepted
// prefix. Unfinished rows contain NaNs and must never be sampled.
#include "../../src/experimental/speculative_frontier.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

enum { V = 7, ROWS = VERIFY_DRAFTS + 1 };
static int checks, failures, cases;
static void expect(const char *what, int ok) {
  checks++;
  if (!ok) failures++, fprintf(stderr, "FAIL: %s\n", what);
}

static void run(int n, int cut, int eos_at, int order, double temperature, double top_p, uint64_t seed) {
  float logits[ROWS * V], available[ROWS * V];
  int drafts[VERIFY_DRAFTS], sampled[ROWS], sequence[ROWS], size = 0;
  Sampler initial = {.temp = temperature, .top_p = top_p};
  sampler_seed(&initial, seed);
  Sampler original = initial;
  cases++;
  for (int r = 0; r < ROWS; r++)
    for (int k = 0; k < V; k++) {
      logits[r * V + k] = ((r * 9 + k * 3) % 17 - 8) * .17f;
      available[r * V + k] = NAN;
    }
  Sampler proposals = initial;
  for (int r = 0; r <= n; r++) sampled[r] = sample_token(logits + r * V, V, &proposals);
  memcpy(drafts, sampled, (size_t)n * sizeof(int));
  if (cut < n) drafts[cut] = (drafts[cut] + 1) % V;
  int eos = eos_at >= 0 ? sampled[eos_at] : -1;
  Sampler expected = initial;
  int accepted;
  int next = sample_verify(logits, V, drafts, n, eos, &expected, &accepted);
  SpeculativeFrontier f;
  expect("valid setup", speculative_frontier_start(&f, drafts, n, eos, &initial));
  if (order == 0) {
    for (int r = 0; r <= n; r++) sequence[size++] = r;
  } else if (order == 1) {
    for (int r = n; r >= 0; r--) sequence[size++] = r;
  } else {
    for (int r = 1; r <= n; r += 2) sequence[size++] = r;
    for (int r = 0; r <= n; r += 2) sequence[size++] = r;
  }
  uint32_t ready = 0;
  for (int i = 0; i < size; i++) {
    int r = sequence[i];
    memcpy(available + r * V, logits + r * V, V * sizeof(float));
    ready |= UINT32_C(1) << r;
    speculative_frontier_poll(&f, available, V, ready);
    uint32_t consumed = (UINT32_C(1) << f.sampled) - 1;
    expect("only a ready contiguous prefix is sampled", (ready & consumed) == consumed);
    expect("caller sampler is unchanged", !memcmp(initial.seed, original.seed, sizeof initial.seed));
  }
  expect("verification completes", f.done);
  expect("accepted prefix matches ordinary verification", f.accepted == accepted);
  expect("next token matches ordinary verification", f.next == next);
  expect("random state matches ordinary verification", !memcmp(f.sampler.seed, expected.seed, sizeof expected.seed));
  expect("no samples follow the first rejection or EOS", f.sampled == accepted + 1);
  Sampler ended = f.sampler;
  for (int i = 0; i < ROWS * V; i++) available[i] = NAN;
  expect("completed polling is idempotent", speculative_frontier_poll(&f, available, V, UINT32_MAX));
  expect("completed polling consumes no randomness", !memcmp(f.sampler.seed, ended.seed, sizeof ended.seed));
}

int main(void) {
  const int depths[] = {0, 1, 2, 5, VERIFY_DRAFTS};
  for (unsigned d = 0; d < sizeof depths / sizeof *depths; d++) {
    int n = depths[d];
    for (int cut = 0; cut <= n; cut++)
      for (int order = 0; order < 3; order++)
        for (int greedy = 0; greedy < 2; greedy++)
          for (int nucleus = 0; nucleus < 2; nucleus++)
            for (int seed = 0; seed < 8; seed++)
              run(n, cut, -1, order, greedy ? 0 : 1, nucleus ? .6 : 1, (uint64_t)seed);
    for (int eos = 0; eos <= n; eos++)
      run(n, n, eos, 2, 1, 1, 12345);
  }
  fprintf(stderr, "speculative frontier: %d cases, %d/%d CPU checks passed\n", cases, checks - failures, checks);
  return failures != 0;
}
