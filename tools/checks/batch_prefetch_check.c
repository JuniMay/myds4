// Bound speculative admission across requests without changing verification.
#include "myds4.h"
#include <string.h>
#include <unistd.h>

static struct { int layer, count, admit, inflight, issuer, ids[96]; } calls[4];
static int ncalls, kept;
static int credit_total[EXPERTS], credit_layer;
static void spy_credit(int layer, int expert, int repeats) {
  credit_layer = layer;
  credit_total[expert] += repeats;
}
static void spy_refine(int layer, const int *experts, int n) { kept = n; }
static void spy_prefetch(int layer, const int *experts, int n, int admit, int inflight, int issuer) {
  if (ncalls >= 4 || n > 96) die("prefetch fixture overflow");
  int at = ncalls++;
  calls[at].layer = layer, calls[at].count = n, calls[at].admit = admit;
  calls[at].inflight = inflight, calls[at].issuer = issuer;
  memcpy(calls[at].ids, experts, n * sizeof(int));
}
#define MYDS4_BATCH_PREFETCH_STUDY
#define cache_refine spy_refine
#define cache_prefetch_from spy_prefetch
#define cache_repeat_credit spy_credit
#include "../../src/engine/model.c"

static int checks, failures;
static void expect(const char *name, int ok) {
  checks++;
  if (!ok) failures++, fprintf(stderr, "FAIL: %s\n", name);
}

int main(void) {
  alarm(10);
  int predictions[4 * PREDICT];
  for (int r = 0; r < 4; r++)
    for (int k = 0; k < PREDICT; k++) predictions[r * PREDICT + k] = 10 * (r + 1) + k;
  nrows = 2, batch_prefetch_limit = -1;
  read_ahead(3, 2, predictions);
  expect("default keeps per-row budgets", ncalls == 2 && calls[0].admit == 6 && calls[0].inflight == 12);
  expect("default keeps row ordering", calls[0].ids[1] == 11 && calls[0].ids[3] == 20);
  ncalls = 0, batch_prefetch_limit = 1;
  read_ahead(3, 2, predictions);
  const int expected[] = {10, 20, 11, 21, 12, 22};
  expect("one shared budget still considers every candidate", ncalls == 2 && calls[0].count == 6 && calls[0].admit == 3 && calls[0].inflight == 6);
  expect("rank ordering covers both requests", !memcmp(expected, calls[0].ids, sizeof expected));
  expect("refinement keeps both requests' predictions", kept == 12);
  expect("second horizon has the same bound", calls[1].admit == 3 && calls[1].inflight == 6 && calls[1].ids[1] == 40);
  ncalls = 0, batch_prefetch_limit = 0;
  read_ahead(3, 2, predictions);
  expect("zero budget admits no new reads", calls[0].admit == 0 && calls[1].inflight == 0);
  ncalls = 0, nrows = 0, batch_prefetch_limit = 1;
  read_ahead(3, 2, predictions);
  expect("verification retains its original budget", calls[0].admit == 6 && calls[0].inflight == 12 && calls[0].ids[1] == 11);
  ncalls = 0, nrows = 1;
  read_ahead(3, 1, predictions);
  expect("single-session decoding retains its original budget", calls[0].admit == 3 && calls[0].inflight == 6);
  int first[] = {10, 11, 12, 20, 21, 22}, keep[12] = {0};
  ncalls = 0, nrows = 2;
  read_residual_ahead(3, keep, 12, first, 6, 12);
  expect("residual forecasts use the shared admission and rank order", ncalls == 1 && calls[0].admit == 3 && calls[0].inflight == 6 && !memcmp(expected, calls[0].ids, sizeof expected));
  ncalls = 0;
  read_ahead(LAYERS - 1, 2, predictions);
  expect("last layer has no future reads", ncalls == 0);
  int routes[3 * 16] = {0};
  for (int r = 0; r < 3; r++) for (int k = 0; k < TOPK; k++) routes[r * 16 + k] = r * TOPK + k;
  routes[16] = routes[32] = 0; // one expert serves every request
  routes[33] = 1; // another serves two requests
  nrows = 3, batch_usage_credit = 0;
  credit_batch_uses(7, routes, 3);
  expect("default keeps distinct-expert accounting", credit_total[0] == 0);
  batch_usage_credit = 1;
  credit_batch_uses(7, routes, 3);
  expect("three concurrent uses receive two extra credits", credit_layer == 7 && credit_total[0] == 2);
  expect("two uses receive one extra credit", credit_total[1] == 1);
  expect("private experts receive no extra credit", credit_total[2] == 0 && credit_total[7] == 0);
  memset(credit_total, 0, sizeof credit_total);
  nrows = 0;
  credit_batch_uses(7, routes, 3);
  expect("speculative rows receive no concurrent-use credit", credit_total[0] == 0);
  nrows = 1;
  credit_batch_uses(7, routes, 1);
  expect("one request keeps existing cache scores", credit_total[0] == 0);
  fprintf(stderr, "batch prefetch budget: %d/%d CPU checks passed\n", checks - failures, checks);
  return failures != 0;
}
