// CPU checks of target sampling and deterministic-proposal verification.
#include "../../src/engine/myds4.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int checks, failures;
static void expect(const char *name, int ok) {
  checks++;
  if (!ok) failures++, fprintf(stderr, "FAIL: %s\n", name);
}

static void frequency(double temp, double top_p, const double expected[3], const char *name) {
  float logits[3] = {logf(.6f), logf(.3f), logf(.1f)};
  Sampler s = {.temp = temp, .top_p = top_p};
  sampler_seed(&s, 982451653);
  int count[3] = {0};
  for (int i = 0; i < 100000; i++) count[sample_token(logits, 3, &s)]++;
  int ok = 1;
  for (int i = 0; i < 3; i++) ok &= fabs(count[i] / 100000.0 - expected[i]) < .006;
  expect(name, ok);
}

int main(void) {
  alarm(30);
  Sampler s = {.temp = 1, .top_p = 1};
  expect("temperature one and top-p one are valid", sampler_valid(&s));
  s.temp = -1;
  expect("negative temperature is invalid", !sampler_valid(&s));
  s.temp = NAN;
  expect("NaN temperature is invalid", !sampler_valid(&s));
  s.temp = INFINITY;
  expect("infinite temperature is invalid", !sampler_valid(&s));
  s.temp = 1;
  s.top_p = 1.1;
  expect("top-p above one is invalid", !sampler_valid(&s));
  s.top_p = -.1;
  expect("negative top-p is invalid", !sampler_valid(&s));
  s.top_p = NAN;
  expect("NaN top-p is invalid", !sampler_valid(&s));
  s.top_p = 0;
  expect("zero top-p keeps the highest-probability token", sampler_valid(&s));
  float tie[4] = {0, 1, 1, -1};
  s.temp = 0;
  sampler_seed(&s, 1234);
  Sampler before = s;
  expect("greedy chooses the lower ID on a tie", sample_token(tie, 4, &s) == 1);
  expect("greedy consumes no random draw", !memcmp(s.seed, before.seed, sizeof s.seed));
  s.temp = 1;
  s.top_p = 1;
  sampler_seed(&s, 0);
  float tiny[2] = {-30.5f, 0};
  expect("top-p one retains tiny positive probabilities", sample_token(tiny, 2, &s) == 0);
  frequency(1, 1, (double[]){.6, .3, .1}, "full softmax frequencies match analytic probabilities");
  frequency(1, .7, (double[]){2.0 / 3, 1.0 / 3, 0}, "nucleus renormalizes the smallest sufficient prefix");
  frequency(1, 0, (double[]){1, 0, 0}, "zero top-p retains one token");
  double sum = sqrt(.6) + sqrt(.3) + sqrt(.1);
  frequency(2, 1, (double[]){sqrt(.6) / sum, sqrt(.3) / sum, sqrt(.1) / sum},
            "temperature scales logits before sampling");

  enum { N = 7, VERIFY_DRAFTS = 15 };
  float logits[(VERIFY_DRAFTS + 1) * N];
  for (int r = 0; r <= VERIFY_DRAFTS; r++)
    for (int t = 0; t < N; t++) logits[r * N + t] = (float)((r * 17 + t * 13) % 19 - 9) / 3;
  int matches = 1, rng_matches = 1, eos_stops = 1, greedy_matches = 1;
  for (int trial = 0; trial < 1000; trial++)
    for (int k = 0; k <= VERIFY_DRAFTS; k++) {
      Sampler original = {.temp = trial % 3 ? 1 : 0, .top_p = trial % 2 ? 1 : .8};
      sampler_seed(&original, 98317u * (unsigned)trial + 12457);
      Sampler serial = original, after[VERIFY_DRAFTS + 1];
      int expected[VERIFY_DRAFTS + 1];
      for (int r = 0; r <= k; r++) {
        expected[r] = sample_token(logits + r * N, N, &serial);
        after[r] = serial;
      }
      for (int mismatch = 0; mismatch <= k; mismatch++) {
        int drafts[VERIFY_DRAFTS];
        for (int r = 0; r < k; r++) drafts[r] = expected[r];
        if (mismatch < k) drafts[mismatch] = (drafts[mismatch] + 1) % N;
        Sampler speculative = original;
        int accepted;
        int next = sample_verify(logits, N, drafts, k, -1, &speculative, &accepted);
        matches &= accepted == mismatch && next == expected[mismatch];
        rng_matches &= !memcmp(speculative.seed, after[mismatch].seed, sizeof speculative.seed);
        if (original.temp == 0) greedy_matches &= accepted == mismatch && next == expected[mismatch];
      }
      if (k > 0) {
        int drafts[VERIFY_DRAFTS];
        for (int r = 0; r < k; r++) drafts[r] = expected[r];
        Sampler speculative = original;
        int accepted;
        int next = sample_verify(logits, N, drafts, k, expected[0], &speculative, &accepted);
        eos_stops &=
            next == expected[0] && accepted == 0 && !memcmp(speculative.seed, after[0].seed, sizeof speculative.seed);
      }
    }
  expect("every rejection position and all-accepted bonus match serial tokens", matches);
  expect("verification preserves the serial random stream", rng_matches);
  expect("EOS stops verification without consuming later draws", eos_stops);
  expect("greedy verification remains exact", greedy_matches);
  int coupled_equal = 1, coupled_exact = 1;
  for (int trial = 0; trial < 256; trial++) {
    Sampler initial = {.temp = 1, .top_p = trial % 2 ? 1 : .8};
    sampler_seed(&initial, 98317u * (unsigned)trial + 91);
    for (int changed = 0; changed < 2; changed++) {
      Sampler proposal = initial, target = initial, reference = initial;
      int drafts[5], accepted, expected[6];
      Sampler after[6];
      for (int r = 0; r < 6; r++) {
        expected[r] = sample_token(logits + r * N, N, &reference);
        after[r] = reference;
        if (r < 5) {
          float q[N];
          for (int i = 0; i < N; i++) q[i] = logits[r * N + (changed ? (i + r + 1) % N : i)];
          drafts[r] = sample_token(q, N, &proposal);
        }
      }
      int next = sample_verify(logits, N, drafts, 5, -1, &target, &accepted);
      if (!changed) coupled_equal &= accepted == 5;
      for (int r = 0; r < accepted; r++) coupled_exact &= drafts[r] == expected[r];
      coupled_exact &= next == expected[accepted] && !memcmp(target.seed, after[accepted].seed, sizeof target.seed);
    }
  }
  expect("shared uniforms accept an identical draft distribution", coupled_equal);
  expect("shared uniforms preserve target tokens and RNG with a different draft distribution", coupled_exact);
  fprintf(stderr, "sampling: %d/%d CPU checks passed\n", checks - failures, checks);
  return failures != 0;
}
