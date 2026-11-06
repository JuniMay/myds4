// Prompt usage changes retention only: no new reads, routing or reservations.
#define MYDS4_PREFILL_CREDIT_ROWS 0
#include "../../src/engine/io.c"

int main(void) {
  Slot fixture[12] = {0};
  double priorities[12];
  int experts[12], routes[4 * 16] = {0}, saved_routes[4 * 16];
  slots = fixture, score = priorities, max_slots = cap = nslots = 12;
  memset(where, 0xff, sizeof where);
  head = 11, tail = 0, pass_layer = 0, serving = published = -1;
  token = 10, weight = 2;
  for (int i = 0; i < 12; i++) {
    fixture[i].layer = 0, fixture[i].expert = experts[i] = i, fixture[i].mapped = 1;
    fixture[i].prev = i == 11 ? -1 : i + 1, fixture[i].next = i ? i - 1 : -1;
    where[0][i] = i, escore[i] = priorities[i] = 4;
  }
  for (int r = 0; r < 4; r++)
    for (int k = 0; k < TOPK; k++) routes[r * 16 + k] = r == 3 ? k + TOPK : k;
  memcpy(saved_routes, routes, sizeof routes);
  int checks = 0, failures = 0;
#define CHECK(x) do { checks++; if (!(x)) failures++, fprintf(stderr, "FAIL: %s\n", #x); } while (0)
  CHECK(cache_request(0, experts, 12) == 0);
  CacheStats before = cache_stats;
  cache_prefill_credit(0, experts, 12, routes, 4, 0);
  double sum = 0;
  for (int e = 0; e < 12; e++) {
    CHECK(score[e] == (e < TOPK ? 5.5 : 4.5) && score[e] == escore[e]);
    CHECK(where[0][e] == e && slots[e].used == token);
    sum += score[e] - 4;
  }
  CHECK(sum == weight * TOPK);
  CHECK(!memcmp(&before, &cache_stats, sizeof before));
  CHECK(!memcmp(routes, saved_routes, sizeof routes));
  pass_layer = -1;
  CHECK(victim() == -1);  // The requested union remains protected this token.
  serving = 1;
  CHECK(victim() == TOPK);
  pass_layer = 0;
  for (int e = 0; e < 12; e++) score[e] = escore[e] = 4;
  cache_request(0, experts, 12);
  before = cache_stats;
  cache_prefill_credit(0, experts, 12, routes, 4, 1);
  for (int e = 0; e < 12; e++) CHECK(score[e] == (e < TOPK ? 4 : 6) && score[e] == escore[e]);
  CHECK(!memcmp(&before, &cache_stats, sizeof before));
  pass_layer = -1, serving = 1;
  CHECK(victim() == 0);
  int selected = 0;
  cache_request(0, &selected, 1);
  CHECK(score[0] == 6 && escore[0] == 6);  // Decode still receives a whole use.
  pass_layer = 0;
  for (int e = 0; e < 12; e++) score[e] = escore[e] = 4;
  cache_request(0, experts, 12);
  cache_prefill_credit(0, experts, 12, routes, 4, 256);
  for (int e = 0; e < 12; e++) CHECK(score[e] == (e < TOPK ? 5.5 : 4.5));
  printf("prefill usage credit: %d/%d CPU checks passed\n", checks - failures, checks);
  return failures != 0;
}
