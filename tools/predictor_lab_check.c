// Causal learning and signal wiring checks; no model, GPU or read files.
#include "../src/predictor_lab.c"

static int checks, failures, scheduled, wrong_schedule, long_scheduled, long_refined, long_bad, early_long_update;
static void expect(const char *name, int ok) {
  checks++;
  if (!ok) failures++, fprintf(stderr, "FAIL: %s\n", name);
}
int cache_ready(int layer, int expert) { return !(expert & 1); }
int cache_slot(int layer, int expert) { return cache_ready(layer, expert); }
int cache_action_state(int layer, int expert, uint64_t id) { return id ? 2 : 0; }
void cache_prefetch_from(int layer, const int *ids, int n, int admit, int inflight, int issuer) {
  if (issuer == PF_H4 || issuer == PF_H8) {
    long_scheduled++;
    long_bad |= layer < (issuer == PF_H4 ? 4 : 8) || layer >= LAYERS || admit != 3 || inflight != 18;
    for (int k = 0; k < n; k++) long_bad |= ids[k] < 0 || ids[k] >= EXPERTS;
    if (lab.long_missing_first) for (int k = 0; k < n; k++) long_bad |= cache_slot(layer, ids[k]);
    return;
  }
  scheduled++;
  wrong_schedule |= layer < 1 || layer > 7 || issuer != PF_DRAFT || admit != 3 || inflight != 18;
  for (int k = 0; k < n; k++) {
    wrong_schedule |= ids[k] < 0 || ids[k] >= EXPERTS;
    for (int j = 0; j < k; j++) wrong_schedule |= ids[k] == ids[j];
  }
}
void cache_refine(int layer, const int *ids, int n) {
  long_refined++;
  long_bad |= layer < 4 || layer >= LAYERS || n != 3 * TOPK;
}

static float x[BLOCK * DIM], streams[BLOCK * HC * DIM], mix[BLOCK * MIX];
static float router[BLOCK * DRAFT_EXPERTS], query[BLOCK * HEADS * HEAD_DIM];
static uint16_t embeddings[(BLOCK + 1) * DIM], keys[WINDOW * HEAD_DIM];
static float long_gp[3 * EXPERTS], label_logits[3 * EXPERTS], label_bias[EXPERTS];

static void prepare(int pos) {
  predictor_lab_begin(pos, 2, embeddings);
  for (int d = 0; d < DRAFTS; d++)
    predictor_lab_stage(d, BLOCK, x, streams, mix, router, query, x, keys);
  predictor_lab_final(2, streams, x);
  predictor_lab_embeddings(3, embeddings);
  predictor_lab_predict(3);
}

static void labels(void) {
  int routes[3][16] = {{0}}, before[3][16];
  for (int l = 0; l < LAYERS; l++) {
    for (int r = 0; r < 3; r++) for (int k = 0; k < TOPK; k++) routes[r][k] = (l * 13 + r * 37 + k) % EXPERTS;
    memcpy(before, routes, sizeof routes);
    predictor_lab_routes(l, 3, routes[0], label_logits, label_bias);
    if (predictor_lab_long_active()) {
      predictor_lab_long_forecast(l, 3, long_gp, long_gp, long_gp);
      predictor_lab_long_update(l);
      if (l < 4 && lab.longer[0][0].model.updates) early_long_update = 1;
    }
    if (memcmp(before, routes, sizeof routes)) failures++;
  }
}

int main(void) {
  alarm(60);
  setenv("MYDS4_LAB_MODE", "shadow", 1);
  setenv("MYDS4_LAB_VARIANT", "sweep", 1);
  setenv("MYDS4_LAB_RANK", "16", 1);
  setenv("MYDS4_LAB_WARMUP", "1", 1);
  setenv("MYDS4_LAB_BASE", "none", 1);
  setenv("MYDS4_LAB_LEARN", "1", 1);
  setenv("MYDS4_LAB_POLICY", "rank", 1);
  setenv("MYDS4_LAB_GAIN", "1", 1);
  setenv("MYDS4_LAB_LONG", "0", 1);
  setenv("MYDS4_LAB_LONG_MISSES", "0", 1);
  setenv("MYDS4_LAB_WASTE", "0.5", 1);
  setenv("MYDS4_LAB_VICTIMS", "8", 1);
  for (size_t i = 0; i < sizeof x / sizeof *x; i++) x[i] = sinf(i * .013f);
  for (size_t i = 0; i < sizeof streams / sizeof *streams; i++) streams[i] = cosf(i * .017f);
  for (size_t i = 0; i < sizeof mix / sizeof *mix; i++) mix[i] = .1f + (i % 7) * .1f;
  for (size_t i = 0; i < sizeof router / sizeof *router; i++) router[i] = sinf(i * .03f);
  for (size_t i = 0; i < sizeof query / sizeof *query; i++) query[i] = cosf(i * .02f);
  for (size_t i = 0; i < sizeof keys / sizeof *keys; i++) keys[i] = 0x3f00 + i % 128;
  for (size_t i = 0; i < sizeof embeddings / sizeof *embeddings; i++) embeddings[i] = 0x3f00 + i % 128;
  expect("sweep initializes", predictor_lab_enabled() && lab.count == LAB_VARIANTS);
  expect("all specified feature groups collected", lab.collect_mask == (1u << LAB_GROUPS) - 1);
  prepare(16);
  expect("a complete forecast precedes labels", lab.active && lab.v[0].model.updates == 0);
  expect("cold prediction is untrained", lab.v[0].scores[0] == 0);
  labels();
  expect("labels do not update the current prediction", lab.v[0].scores[0] == 0 && lab.v[0].model.updates == 0);
  predictor_lab_schedule(-1);
  expect("shadow mode issues no reads", scheduled == 0);
  predictor_lab_end();
  expect("all rows learn after the pass", lab.v[0].model.updates == 3 && !lab.active);
  expect("warmup is excluded from accuracy", lab.v[0].raw_total == 0);
  predictor_lab_accept(2);
  expect("history uses accepted row", lab.prior_labels[37] == .1f);
  expect("history excludes rejected row", lab.prior_labels[74] == 0);
  prepare(18);
  for (int i = 1; i < LAB_VARIANTS; i++)
    expect(names[i], memcmp(lab.v[0].phi, lab.v[i].phi, 3 * lab.rank * sizeof(float)) != 0);
  float before = lab.v[0].scores[37];
  labels();
  expect("second labels cannot change pending scores", lab.v[0].scores[37] == before && lab.v[0].model.updates == 3);
  expect("raw counts include every layer and verification row", lab.v[0].raw_total == LAYERS * 3 * TOPK);
  expect("missing coverage excludes ready experts", lab.v[0].missing > 0 && lab.v[0].missing < lab.v[0].raw_total);
  lab.live = 1;
  predictor_lab_schedule(-1);
  expect("live schedule respects range, origin, union and budget", scheduled == 7 && !wrong_schedule);
  predictor_lab_end();
  expect("second pass updates each variant", lab.v[9].model.updates == 6);
  expect("no rejected numerical updates", lab.v[9].model.rejected_updates == 0);
  predictor_lab_accept(1);
  predictor_lab_embeddings(1, embeddings);
  predictor_lab_predict(1);
  expect("a final plain step does not reuse draft features", !lab.active);
  lab.policy = 2;
  create_actions();
  prepare(20);
  LabVictim victims[2] = {{7, 4, 52, 100, 0, 20000000}, {8, 4, 52, 0, 100, 20000000}};
  for (int r = 0; r < lab.rows; r++) lab.v[0].scores[r * LAB_OUTPUTS + 3 * EXPERTS + 39] = 3;
  uint64_t action_id = 0;
  int victim = predictor_lab_choose(3, 39, 20000000, 0, victims, 2, &action_id);
  expect("action policy chooses from supplied victims", victim == 8 && action_id > 0);
  predictor_lab_read(action_id, 12000000);
  predictor_lab_read(action_id, 8000000);
  expect("action accumulates completed read bytes", atomic_load(&lab.action[(action_id - 1) % ACTION_RECORDS].transferred) == 20000000);
  expect("action parameters stay fixed before outcomes", lab.fetch_value.updates == 0);
  labels();
  predictor_lab_end();
  predictor_lab_accept(2);
  expect("policy waits for its reuse horizon", lab.fetch_value.updates == 0);
  prepare(22), labels(), predictor_lab_end(), predictor_lab_accept(2);
  prepare(24), labels(), predictor_lab_end(), predictor_lab_accept(2);
  expect("delayed action outcome trains fetch value", lab.fetch_value.updates == 1);
  expect("delayed action outcome trains retention value", lab.retain_value.updates == 1);
  expect("owned timely fetch is credited once", lab.action_used == 1 && lab.action_timely == 1);
  expect("needed victim is penalized", lab.victim_reused == 1);
  prepare(26);
  memset(lab.fetch_value.b, 0, ACTION_FEATURES * sizeof(float));
  lab.fetch_value.b[0] = -100;
  for (uint64_t seed = 1;; seed++) {
    uint64_t trial = seed;
    if (op_uniform(&trial) >= .02) { lab.action_rng = seed; break; }
  }
  action_id = 123;
  expect("negative-value action abstains without ownership", predictor_lab_choose(3, 39, 20000000, 0,
          victims, 2, &action_id) == -2 && action_id == 0);
  labels(), predictor_lab_end(), predictor_lab_accept(1);
  lab.policy = 3, lab.count = 0, lab.base = 0;
  prepare(28);
  int native[3] = {13, 14, 15};
  predictor_lab_native_candidates(1, native, 3, PF_H2);
  expect("native controller uses supplied candidate ranks", lab.native_rank[1][13] == 1 && lab.native_rank[1][15] < .34f);
  expect("native controller needs no auxiliary projection", !predictor_lab_projection());
  lab.policy = 4;
  action_id = 123;
  expect("matched native control uses the first eligible victim", predictor_lab_choose(1, 13, 20000000, 0,
          victims, 2, &action_id) == 7 && action_id == 0);
  labels(), predictor_lab_end(), predictor_lab_accept(1);
  lab.policy = 3;
  prepare(30);
  lab.explored = 1;
  memset(lab.fetch_value.b, 0, ACTION_FEATURES * sizeof(float));
  memset(lab.retain_value.b, 0, ACTION_FEATURES * sizeof(float));
  lab.fetch_value.b[0] = 100, lab.retain_value.b[0] = -100;
  LabVictim empty[2] = {{7, 4, 52, 0, 100, 20000000}, {9, 4, -1, 0, 0, 0}};
  expect("a free slot is preferred even if retention estimates are negative",
         predictor_lab_choose(1, 13, 20000000, 0, empty, 2, &action_id) == 9);
  labels(), predictor_lab_end(), predictor_lab_accept(1);
  lab.policy = 5, lab.active = 1;
  expect("demand control suppresses decode prefetches", predictor_lab_suppress_prefetch());
  lab.active = 0;
  expect("demand control keeps prefill behavior", !predictor_lab_suppress_prefetch());
  lab.policy = 0, lab.count = 1, lab.gain = 8;
  prepare(32);
  double norm = 0;
  for (int j = 0; j < lab.rank - 1; j++) norm += lab.v[0].phi[j] * lab.v[0].phi[j];
  expect("feature gain sets latent norm", fabs(norm - 64) < .001);
  expect("feature gain does not scale the intercept", lab.v[0].phi[lab.rank - 1] == 1);
  labels(), predictor_lab_end(), predictor_lab_accept(1);
  lab.count = 0, lab.long_enabled = 1, lab.rank = 16, lab.policy = 0;
  const float *biases[LAYERS];
  for (int l = 0; l < LAYERS; l++) biases[l] = label_bias;
  for (int i = 0; i < 3 * EXPERTS; i++) label_logits[i] = .001f * i, long_gp[i] = sinf(i * .03f);
  predictor_lab_long_init(biases);
  prepare(34);
  labels();
  expect("long forecasts wait for target labels", !early_long_update);
  expect("four-layer head learns all rows", lab.longer[0][0].model.updates == 3);
  expect("eight-layer head learns all rows", lab.longer[0][1].model.updates == 3);
  expect("long correction weights actually change", lab.longer[0][0].model.active);
  expect("unavailable horizon is not initialized or trained", lab.longer[36][0].model.updates == 0);
  expect("long heads score every valid pair", lab.long_total[0] == 36 * 3 * TOPK && lab.long_total[1] == 32 * 3 * TOPK);
  expect("long read plans cover valid targets", long_scheduled == 68 && !long_bad);
  expect("four-layer refinements preserve top-six union", long_refined == 36);
  predictor_lab_end(), predictor_lab_accept(1);
  lab.long_missing_first = 1;
  prepare(36), labels(), predictor_lab_end(), predictor_lab_accept(1);
  expect("missing-first long plans only offer uncached experts", long_scheduled == 136 && !long_bad);
  predictor_lab_report(stdout);
  fprintf(stderr, "predictor lab: %d/%d checks passed\n", checks - failures, checks);
  return failures != 0;
}
