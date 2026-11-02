// Check eligible victim selection and admission ownership without GPU work.
#define MYDS4_PREDICT_LAB
#include "../src/io.c"
#include <sys/wait.h>

static int checks, failures, choice_mode, suppress, victim_limit = 8;
static void expect(const char *name, int ok) {
  checks++;
  if (!ok) failures++, fprintf(stderr, "FAIL: %s\n", name);
}
int predictor_lab_cache_policy(void) { return 1; }
int predictor_lab_suppress_prefetch(void) { return suppress; }
int predictor_lab_victim_limit(void) { return victim_limit; }
void predictor_lab_native_candidates(int l, const int *ids, int n, int issuer) {}
void predictor_lab_read(uint64_t id, uint64_t bytes) {}
int predictor_lab_choose(int layer, int e, uint64_t bytes, int pending,
                         const LabVictim *v, int n, uint64_t *id) {
  if (!choice_mode) return -2;
  *id = 42;
  return choice_mode == 2 ? 100000 : v[0].slot;
}

int main(void) {
  alarm(20);
  max_slots = nslots = cap = 22;
  slots = calloc(max_slots, sizeof *slots);
  score = calloc(max_slots, sizeof *score);
  memset(where, 0xff, sizeof where);
  token = 20, weight = 1, serving = 0, published = 2;
  head = 0, tail = max_slots - 1;
  for (int i = 0; i < max_slots; i++) {
    slots[i].layer = i, slots[i].expert = 0, slots[i].mapped = 1;
    slots[i].used = token;
    slots[i].prev = i - 1, slots[i].next = i + 1 < max_slots ? i + 1 : -1;
    score[i] = 100 - i;
    where[i][0] = i;
    expert_spans[i] = 1, expert_span[i][0][0].len = 100;
  }
  slots[1].prefetched = 1, slots[3].pinned = 1;
  slots[4].loading = dispatch_group_create();
  dispatch_group_enter(slots[4].loading);
  score[5] = INFINITY;
  LabVictim list[128];
  for (int i = 0; i < 128; i++) list[i].slot = -900;
  int n = action_victims(list);
  expect("shortlist fills eight entries", n == 8);
  for (int i = 0; i < n; i++) expect("shortlist is the lowest eight eligible scores", list[i].slot == 21 - i);
  expect("unfinished, pinned, prefetched and current work excluded", list[7].slot >= 14);
  pf_issuer = PF_DRAFT;
  int before = cache_stats.evictions;
  expect("abstention does not take a slot", take(30, 5, 1) == -1 && where[30][5] == -1);
  expect("abstention does not evict", cache_stats.evictions == (uint64_t)before);
  choice_mode = 1;
  int at = take(30, 5, 1);
  expect("accepted action chooses an eligible slot", at == 21 && where[30][5] == 21);
  expect("action ownership is attached to admitted expert", cache_action_state(30, 5, 42) == 2);
  expect("another admission receives no credit", cache_action_state(30, 5, 43) == 0);
  slots[at].loading = dispatch_group_create();
  dispatch_group_enter(slots[at].loading);
  expect("owned pending read remains late", cache_action_state(30, 5, 42) == 1);
  dispatch_group_leave(slots[at].loading);
  finish(at);
  pid_t child = fork();
  if (!child) { choice_mode = 2; take(31, 5, 1); _exit(0); }
  int status;
  waitpid(child, &status, 0);
  expect("an out-of-shortlist victim is rejected", WIFEXITED(status) && WEXITSTATUS(status) != 0);
  dispatch_group_leave(slots[4].loading), finish(4);
  where[6][0] = -1, slots[6].expert = -1, score[6] = -INFINITY;
  n = action_victims(list);
  expect("an empty cancelled slot comes before occupied victims", n && list[0].slot == 6);
  expect("empty-slot features remain finite", n && isfinite(list[0].frequency) && list[0].frequency == 0);
  suppress = 1;
  int unused = 123;
  uint64_t admissions = cache_stats.prefetches;
  cache_prefetch_from(31, &unused, 1, 1, 6, PF_H2);
  expect("demand control does not admit optional reads", cache_stats.prefetches == admissions && where[31][123] == -1);
  victim_limit = 16;
  n = action_victims(list);
  expect("wider victim shortlist has the configured size", n == 16 && list[0].slot == 6);
  int safe = 1;
  for (int i = 0; i < n; i++) safe &= list[i].slot != 0 && list[i].slot != 1 && list[i].slot != 2 &&
      list[i].slot != 3 && list[i].slot != 5;
  expect("wider shortlist keeps the same protection rules", safe);
  fprintf(stderr, "cache actions: %d/%d checks passed\n", checks - failures, checks);
  return failures != 0;
}
