// Check admission limits separately from retention, without model or GPU work.
#include "myds4.h"
#include <string.h>

static struct { int layer, count, admit, inflight, issuer, ids[96]; } observed[4];
static int calls, retained, retain_ids[96];
static void spy_refine(int layer, const int *ids, int count) {
  retained = count;
  memcpy(retain_ids, ids, (size_t)count * sizeof *ids);
}
static void spy_prefetch(int layer, const int *ids, int count, int admit, int inflight, int issuer) {
  observed[calls].layer = layer, observed[calls].count = count;
  observed[calls].admit = admit, observed[calls].inflight = inflight, observed[calls].issuer = issuer;
  memcpy(observed[calls].ids, ids, (size_t)count * sizeof *ids);
  calls++;
}
#define MYDS4_VERIFY_PREFETCH_ROWS 1
#define cache_refine spy_refine
#define cache_prefetch_from spy_prefetch
#include "../../src/engine/model.c"

static int checks, failures;
static void expect(const char *name, int yes) {
  checks++;
  if (!yes) failures++, fprintf(stderr, "FAIL: %s\n", name);
}
int main(void) {
  int pr[8 * PREDICT], original[8 * PREDICT];
  for (int i = 0; i < 8 * PREDICT; i++) pr[i] = i;
  memcpy(original, pr, sizeof pr);
  verifying = 1;
  read_ahead(5, 4, pr);
  expect("both future layers receive a forecast", calls == 2 && observed[0].layer == 6 && observed[1].layer == 7);
  expect("admission uses one verification row", observed[0].count == AHEAD && observed[1].count == AHEAD && observed[0].inflight == INFLIGHT);
  expect("second-layer forecasts retain the original full-row stride", observed[1].ids[0] == pr[4 * PREDICT]);
  expect("retention keeps forecasts from every row", retained == 4 * KEEP && retain_ids[3 * KEEP] == pr[3 * PREDICT]);
  expect("forecast arrays are unchanged", !memcmp(original, pr, sizeof pr));
  calls = 0;
  int keep[4 * KEEP], first[4 * AHEAD];
  for (int i = 0; i < 4 * KEEP; i++) keep[i] = i;
  for (int i = 0; i < 4 * AHEAD; i++) first[i] = 100 + i;
  read_residual_ahead(5, keep, 4 * KEEP, first, 4 * AHEAD, 4 * INFLIGHT);
  expect("residual admission is capped without shrinking retention", calls == 1 && retained == 4 * KEEP && observed[0].count == AHEAD && observed[0].ids[0] == 100);
  calls = 0;
  verifying = 0;
  read_ahead(5, 4, pr);
  expect("ordinary concurrent rows retain their full admission", observed[0].count == 4 * AHEAD && observed[0].inflight == 4 * INFLIGHT);
  calls = 0;
  verifying = 1, opt.substitute = .1f;
  read_residual_ahead(5, keep, 4 * KEEP, first, 4 * AHEAD, 4 * INFLIGHT);
  expect("substitution retains its existing read-ahead policy", observed[0].count == 4 * AHEAD && observed[0].inflight == 4 * INFLIGHT);
  fprintf(stderr, "verification prefetch: %d/%d CPU checks passed\n", checks - failures, checks);
  return failures != 0;
}
