#include "myds4.h"
#include "../../src/experimental/speculative_sample.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

static int checks, failures;
static void expect(const char *name, int okay) {
  checks++;
  if (!okay) failures++, fprintf(stderr, "FAIL: %s\n", name);
}

int main(void) {
  enum { N = 4, TRIALS = 200000 };
  const double cases[][2][N] = {
    {{.1, .2, .3, .4}, {.4, .3, .2, .1}},
    {{.1, .2, .3, .4}, {.1, .2, .3, .4}},
    {{0, .4, .6, 0}, {.5, 0, 0, .5}},
    {{.2, .3, .5, 0}, {0, 0, 1, 0}},
    {{.5, 0, .5, 0}, {.6, .2, .1, .1}},
  };
  for (unsigned c = 0; c < sizeof cases / sizeof *cases; c++) {
    float logits[2 * N];
    for (int i = 0; i < N; i++) logits[i] = logits[N + i] = cases[c][0][i] ? log(cases[c][0][i]) : -INFINITY;
    Sampler s = {.temp = 1, .top_p = 1};
    sampler_seed(&s, 23456 + c);
    double p[N], q[N], r[N], reconstructed[N];
    speculative_distribution(logits, N, &s, p);
    memcpy(q, cases[c][1], sizeof q);
    double mass = speculative_residual(p, q, N, r), maximum = 0;
    for (int i = 0; i < N; i++) {
      reconstructed[i] = fmin(p[i], q[i]) + mass * r[i];
      maximum = fmax(maximum, fabs(reconstructed[i] - p[i]));
    }
    expect("accepted and residual mass reconstruct target exactly", maximum < 1e-14);
    int counts[N] = {0}, accepted_total = 0;
    for (int trial = 0; trial < TRIALS; trial++) {
      int draft = speculative_draw(q, N, erand48(s.seed)), accepted;
      int next = speculative_verify(logits, N, &draft, 1, q, -1, &s, &accepted);
      counts[accepted ? draft : next]++;
      accepted_total += accepted;
    }
    double observed = 0;
    for (int i = 0; i < N; i++) observed = fmax(observed, fabs((double)counts[i] / TRIALS - p[i]));
    expect("sampled first token has target frequencies", observed < .005);
    expect("acceptance equals distribution overlap", fabs((double)accepted_total / TRIALS - (1 - mass)) < .005);
  }
  Sampler s = {.temp = 1, .top_p = .5};
  const float ties[] = {0, 0, 0, -INFINITY};
  double p[4];
  speculative_distribution(ties, 4, &s, p);
  expect("nucleus retains ranked prefix with stable ties", p[0] == .5 && p[1] == .5 && p[2] == 0 && p[3] == 0);
  s.temp = 0;
  speculative_distribution(ties, 4, &s, p);
  expect("zero temperature uses one greedy token", p[0] == 1 && p[1] == 0 && p[2] == 0 && p[3] == 0);
  const float inf[] = {INFINITY, 0, INFINITY, -INFINITY};
  s.temp = s.top_p = 1;
  speculative_distribution(inf, 4, &s, p);
  expect("positive infinities share all probability mass", p[0] == .5 && p[1] == 0 && p[2] == .5 && p[3] == 0);
  float logits[12] = {0, -INFINITY, -INFINITY, -INFINITY, -INFINITY, 0, -INFINITY, -INFINITY,
                      -INFINITY, -INFINITY, 0, -INFINITY};
  double q[8] = {1, 0, 0, 0, 0, 1, 0, 0};
  int drafts[] = {0, 1}, accepted;
  sampler_seed(&s, 45678);
  int next = speculative_verify(logits, 4, drafts, 2, q, -1, &s, &accepted);
  expect("all accepted rows use the target bonus distribution", accepted == 2 && next == 2);
  next = speculative_verify(logits, 4, drafts, 2, q, 1, &s, &accepted);
  expect("accepted EOS ends before later draft rows", accepted == 1 && next == 1);
  next = speculative_verify(logits, 4, drafts, 0, q, -1, &s, &accepted);
  expect("zero drafts samples the first target row", accepted == 0 && next == 0);
  // Check a two-token joint distribution, including conditional proposals
  // and a first-token rejection followed by ordinary target continuation.
  const double initial[] = {.1, .2, .3, .4}, proposal0[] = {.4, .3, .2, .1};
  int joint[4][4] = {{0}};
  sampler_seed(&s, 56789);
  for (int trial = 0; trial < TRIALS; trial++) {
    drafts[0] = speculative_draw(proposal0, 4, erand48(s.seed));
    memcpy(q, proposal0, 4 * sizeof *q);
    for (int i = 0; i < 4; i++) {
      q[4 + i] = proposal0[(i + drafts[0]) % 4];
      logits[i] = log(initial[i]);
      logits[4 + i] = log(initial[(i + drafts[0]) % 4]);
      logits[8 + i] = 0;
    }
    drafts[1] = speculative_draw(q + 4, 4, erand48(s.seed));
    next = speculative_verify(logits, 4, drafts, 2, q, -1, &s, &accepted);
    int a = accepted ? drafts[0] : next, b;
    if (accepted == 2) b = drafts[1];
    else if (accepted == 1) b = next;
    else {
      for (int i = 0; i < 4; i++) p[i] = initial[(i + a) % 4];
      b = speculative_draw(p, 4, erand48(s.seed));
    }
    joint[a][b]++;
  }
  double error = 0;
  for (int a = 0; a < 4; a++) for (int b = 0; b < 4; b++)
    error = fmax(error, fabs((double)joint[a][b] / TRIALS - initial[a] * initial[(a + b) % 4]));
  expect("two-token joint frequencies preserve conditional target", error < .004);
  printf("speculative rejection sampling: %d/%d CPU checks passed\n", checks - failures, checks);
  return failures != 0;
}
