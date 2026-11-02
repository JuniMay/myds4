// Joint DSpark-to-backbone feature ablations, evaluated before each pass learns.
// Models share observations, never labels from their current prediction pass.
// No checkpoint weights, native routes, sampler state or saved online weights change.
#include "predictor_lab.h"
#include "online_predictor_math.h"
#include <stdatomic.h>

enum { LAB_GROUPS = 9, LAB_SKETCH = 64, LAB_FEATURES = LAB_GROUPS * LAB_SKETCH + 8,
       LAB_OUTPUTS = LAYERS * EXPERTS, LAB_VARIANTS = 10 };
enum { EMBED, HIDDEN, STREAMS, MIXES, ROUTER, ATTENTION, KV_HISTORY, HISTORY, FINAL };
enum { ACTION_FEATURES = 16, ACTION_RECORDS = 512 };
typedef struct {
  uint64_t id, pass;
  uint64_t requested;
  _Atomic uint64_t transferred;
  int active, layer, expert, victim_layer, victim_expert, used, timely, victim_used;
  float fetch[ACTION_FEATURES], retain[ACTION_FEATURES], cost, victim_scale, progress;
} CacheAction;
static const char *const names[LAB_VARIANTS] = {
  "embedding", "hidden", "streams", "mix", "router", "attention", "kv", "history", "final", "all"
};
typedef struct {
  OpPair model;
  float *projection, *phi, *scores;
  int top[LAYERS][BLOCK + 1][TOPK];
  uint64_t raw_hits, raw_total, union_hits, union_predictions, missing, missing_hits;
  uint64_t layer_hits[LAYERS], layer_total[LAYERS];
  double predict_s, update_s;
  unsigned mask;
  int name;
} LabVariant;
typedef struct {
  OpPair model;
  uint64_t pass;
  int valid, learned, rows;
  float phi[OP_MAX_ROWS * 256], keys[OP_MAX_ROWS * EXPERTS];
  int ids[OP_MAX_ROWS * TOPK], baseline[OP_MAX_ROWS * TOPK];
  uint8_t missing[EXPERTS];
} LongHead;
static struct {
  int initialized, enabled, live, count, rank, pos, rows, pending, prepared, active;
  int warmup, width, horizon, distance, base, learn, previous_accepted, policy, completed, issuer, explored, victim_limit;
  unsigned collect_mask;
  float gain;
  float waste_cost;
  int long_enabled, long_initialized, long_missing_first;
  LongHead longer[LAYERS][2];
  OpWork long_work;
  float long_label[OP_MAX_ROWS * EXPERTS];
  int long_ids[OP_MAX_ROWS * TOPK], long_label_layer;
  uint64_t long_hits[2], long_base_hits[2], long_total[2], long_missing[2], long_missing_hits[2];
  double long_forward_s, long_update_s;
  uint64_t passes, scored_passes, rows_seen, baseline_hits, baseline_total, schedules;
  LabVariant v[LAB_VARIANTS];
  OpWork work;
  float stage[BLOCK][LAB_GROUPS][LAB_SKETCH];
  float embedding[BLOCK + 1][LAB_SKETCH];
  float history[LAB_SKETCH], prior_labels[LAB_OUTPUTS];
  float features[BLOCK + 1][LAB_FEATURES];
  float head[BLOCK][DIM], baseline[BLOCK + 1][LAB_OUTPUTS];
  float native_rank[LAYERS][EXPERTS];
  float target[BLOCK + 1][LAB_OUTPUTS];
  uint8_t labels[BLOCK + 1][LAB_OUTPUTS];
  uint8_t seen[LAYERS], ready_before[LAYERS][EXPERTS];
  double collect_s, schedule_s;
  OpPair fetch_value, retain_value;
  OpWork action_work;
  CacheAction action[ACTION_RECORDS];
  uint64_t action_id, action_attempts, action_admitted, action_abstained, action_settled, action_rng;
  uint64_t action_used, action_timely, victim_reused;
  double action_reward, action_decision_s, action_fit_s;
} lab;

static void create_actions(void) {
  lab.fetch_value = op_create_pair(1, ACTION_FEATURES, 731);
  lab.retain_value = op_create_pair(1, ACTION_FEATURES, 733);
  lab.action_work = op_create_work(1, ACTION_FEATURES);
  lab.fetch_value.b[0] = .2f, lab.fetch_value.b[1] = .8f;
  lab.retain_value.b[0] = .1f, lab.retain_value.b[4] = .3f, lab.retain_value.b[5] = .2f;
  lab.action_rng = 2601101;
}

static unsigned variant_mask(int i) { return i == LAB_VARIANTS - 1 ? (1u << LAB_GROUPS) - 1 : 1u | (1u << i); }

static int integer_env(const char *key, int fallback, int low, int high) {
  const char *s = getenv(key);
  if (!s) return fallback;
  char *end;
  errno = 0;
  long n = strtol(s, &end, 10);
  if (errno || !*s || *end || n < low || n > high) die("%s must be %d..%d", key, low, high);
  return (int)n;
}

static uint32_t hash32(uint32_t n) {
  n ^= n >> 16; n *= 0x7feb352d; n ^= n >> 15; n *= 0x846ca68b; return n ^ (n >> 16);
}

static float bf16_value(uint16_t x) {
  uint32_t bits = (uint32_t)x << 16;
  float f;
  memcpy(&f, &bits, 4);
  return f;
}

// Each source tensor is normalized independently. Signed sketches retain all
// coordinates but compress them; they are not full-tensor neural encoders.
static void sketch(float *out, const float *x, int n, uint32_t tag) {
  double ss = 0;
  for (int i = 0; i < n; i++) ss += (double)x[i] * x[i];
  float scale = 1 / sqrtf((float)ss + 1e-12f);
  for (int i = 0; i < n; i++) {
    uint32_t h = hash32((uint32_t)i + tag * 0x9e3779b9u);
    out[h % LAB_SKETCH] += (h & 0x80000000u ? -scale : scale) * x[i];
  }
}

static void sketch_bf16(float *out, const uint16_t *x, int n, uint32_t tag) {
  double ss = 0;
  for (int i = 0; i < n; i++) { float v = bf16_value(x[i]); ss += (double)v * v; }
  float scale = 1 / sqrtf((float)ss + 1e-12f);
  for (int i = 0; i < n; i++) {
    uint32_t h = hash32((uint32_t)i + tag * 0x9e3779b9u);
    out[h % LAB_SKETCH] += (h & 0x80000000u ? -scale : scale) * bf16_value(x[i]);
  }
}

static void create_variant(int at, int name) {
  LabVariant *v = &lab.v[at];
  v->name = name;
  v->mask = variant_mask(name);
  lab.collect_mask |= v->mask;
  v->model = (OpPair){.e = LAB_OUTPUTS, .r = lab.rank};
  v->model.b = op_allocate((size_t)lab.rank * LAB_OUTPUTS * 4);
  v->model.p = op_allocate((size_t)lab.rank * lab.rank * 4);
  op_covariance_reset(&v->model);
  v->model.covariance_resets = 0;
  v->projection = op_allocate((size_t)LAB_FEATURES * (lab.rank - 1) * 4);
  v->phi = op_allocate((size_t)(BLOCK + 1) * lab.rank * 4);
  v->scores = op_allocate((size_t)(BLOCK + 1) * LAB_OUTPUTS * 4);
  // Every feature ablation uses the same projection. Missing groups are zero.
  uint64_t seed = 2601101;
  for (int i = 0; i < LAB_FEATURES * (lab.rank - 1); i++)
    v->projection[i] = op_gaussian(&seed) / sqrtf(LAB_FEATURES);
}

int predictor_lab_enabled(void) {
  if (lab.initialized) return lab.enabled;
  lab.initialized = 1;
  const char *mode = getenv("MYDS4_LAB_MODE");
  if (!mode || !strcmp(mode, "off")) return 0;
  if (strcmp(mode, "shadow") && strcmp(mode, "prefetch")) die("MYDS4_LAB_MODE must be off, shadow or prefetch");
  lab.live = !strcmp(mode, "prefetch");
  const char *base = getenv("MYDS4_LAB_BASE");
  if (!base || !strcmp(base, "none")) lab.base = 0;
  else if (!strcmp(base, "embedding")) lab.base = 1;
  else if (!strcmp(base, "draft")) lab.base = 2;
  else die("MYDS4_LAB_BASE must be none, embedding or draft");
  lab.learn = integer_env("MYDS4_LAB_LEARN", 1, 0, 1);
  lab.long_enabled = integer_env("MYDS4_LAB_LONG", 0, 0, 1);
  lab.long_missing_first = integer_env("MYDS4_LAB_LONG_MISSES", 0, 0, 1);
  lab.waste_cost = .5f;
  const char *waste = getenv("MYDS4_LAB_WASTE");
  if (waste) {
    char *end;
    double value = strtod(waste, &end);
    if (!*waste || *end || !isfinite(value) || value < 0 || value > 8) die("MYDS4_LAB_WASTE must be in [0,8]");
    lab.waste_cost = (float)value;
  }
  const char *policy = getenv("MYDS4_LAB_POLICY");
  if (!policy || !strcmp(policy, "rank")) lab.policy = 0;
  else if (!strcmp(policy, "miss")) lab.policy = 1;
  else if (!strcmp(policy, "bandit")) lab.policy = 2;
  else if (!strcmp(policy, "native")) lab.policy = 3;
  else if (!strcmp(policy, "native-control")) lab.policy = 4;
  else if (!strcmp(policy, "demand")) lab.policy = 5;
  else die("MYDS4_LAB_POLICY must be rank, miss, bandit, native, native-control or demand");
  if (lab.policy == 2 || lab.policy == 3) create_actions();
  lab.rank = integer_env("MYDS4_LAB_RANK", 128, 16, 256);
  const char *gain = getenv("MYDS4_LAB_GAIN");
  lab.gain = 1;
  if (gain) {
    char *end;
    double value = strtod(gain, &end);
    if (!*gain || *end || !isfinite(value) || value <= 0 || value > 16) die("MYDS4_LAB_GAIN must be in (0,16]");
    lab.gain = (float)value;
  }
  lab.warmup = integer_env("MYDS4_LAB_WARMUP", 16, 0, 256);
  lab.victim_limit = integer_env("MYDS4_LAB_VICTIMS", 8, 1, 128);
  lab.width = integer_env("MYDS4_LAB_WIDTH", 1, 1, TOPK);
  lab.horizon = integer_env("MYDS4_LAB_HORIZON", 8, 2, LAYERS);
  lab.distance = integer_env("MYDS4_LAB_DISTANCE", 2, 1, lab.horizon);
  const char *which = getenv("MYDS4_LAB_VARIANT");
  if (!which) which = lab.live ? "all" : "sweep";
  if (lab.long_enabled && (lab.policy == 1 || lab.policy == 2)) die("long forecasts use rank, native, native-control or demand policy");
  if (lab.policy >= 3 || lab.long_enabled) {
    // This trial uses existing engine forecasts as candidate actions. No
    // auxiliary DSpark prediction network or GPU projections are evaluated.
    lab.base = 0;
  } else if (!strcmp(which, "sweep")) {
    if (lab.live) die("a live prefetch run needs one predictor variant");
    for (int i = 0; i < LAB_VARIANTS; i++) create_variant(lab.count++, i);
  } else {
    int name = 0;
    while (name < LAB_VARIANTS && strcmp(which, names[name])) name++;
    if (name == LAB_VARIANTS) die("unknown MYDS4_LAB_VARIANT");
    create_variant(lab.count++, name);
  }
  if (BLASSetThreading(BLAS_THREADING_SINGLE_THREADED)) die("cannot set predictor lab BLAS threads");
  if (lab.count) lab.work = op_create_work(LAB_OUTPUTS, lab.rank);
  lab.enabled = 1;
  return 1;
}

void predictor_lab_begin(int pos, int drafts, const uint16_t *seed) {
  if (!predictor_lab_enabled()) return;
  if (drafts < 1 || drafts > BLOCK || lab.active) die("invalid predictor lab draft boundary");
  lab.pos = pos, lab.rows = drafts + 1, lab.pending = 1, lab.prepared = 0;
  lab.explored = 0;
  memset(lab.stage, 0, sizeof lab.stage);
  memset(lab.embedding, 0, sizeof lab.embedding);
  double t = now();
  if (lab.count) sketch_bf16(lab.embedding[0], seed, DIM, 1);
  lab.collect_s += now() - t;
}

void predictor_lab_stage(int d, int rows, const float *x, const float *streams,
                         const float *mix, const float *router, const float *query,
                         const float *attention, const uint16_t *ring) {
  if (!lab.enabled || !lab.pending) return;
  double t = now();
  if (d < 0 || d >= DRAFTS || rows < 1 || rows > BLOCK) die("invalid predictor lab stage");
  float context[LAB_SKETCH] = {0};
  if (lab.collect_mask & (1u << KV_HISTORY)) {
    int n = lab.pos < WINDOW ? lab.pos : WINDOW;
    float pooled[4][HEAD_DIM] = {{0}};
    for (int j = 0; j < n; j++) {
      int at = (lab.pos - 1 - j) % WINDOW, bin = j * 4 / n;
      for (int c = 0; c < HEAD_DIM; c++) pooled[bin][c] += bf16_value(ring[at * HEAD_DIM + c]);
    }
    for (int bin = 0; bin < 4; bin++) sketch(context, pooled[bin], HEAD_DIM, 200 + d * 4 + bin);
  }
  for (int r = 0; r < rows; r++) {
    float (*f)[LAB_SKETCH] = lab.stage[r];
    if (lab.collect_mask & (1u << HIDDEN)) sketch(f[HIDDEN], x + r * DIM, DIM, 10 + d);
    if (lab.collect_mask & (1u << STREAMS))
      for (int h = 0; h < HC; h++) sketch(f[STREAMS], streams + (r * HC + h) * DIM, DIM, 20 + 4 * d + h);
    if (lab.collect_mask & (1u << MIXES)) {
      sketch(f[MIXES], mix + r * MIX, MIX, 40 + d);
      float norms[HC], cosine[HC * (HC - 1) / 2];
      for (int h = 0; h < HC; h++) {
        const float *v = streams + (r * HC + h) * DIM;
        norms[h] = sqrtf(cblas_sdot(DIM, v, 1, v, 1) / DIM + 1e-12f);
      }
      for (int h = 0, n = 0; h < HC; h++) for (int k = h + 1; k < HC; k++)
        cosine[n++] = cblas_sdot(DIM, streams + (r * HC + h) * DIM, 1,
                                 streams + (r * HC + k) * DIM, 1) / (DIM * norms[h] * norms[k]);
      for (int h = 0; h < HC; h++) norms[h] = log1pf(norms[h]);
      sketch(f[MIXES], norms, HC, 50 + d), sketch(f[MIXES], cosine, 6, 60 + d);
    }
    if (lab.collect_mask & (1u << ROUTER)) sketch(f[ROUTER], router + r * DRAFT_EXPERTS, DRAFT_EXPERTS, 70 + d);
    if (lab.collect_mask & (1u << ATTENTION)) {
      sketch(f[ATTENTION], query + (size_t)r * HEADS * HEAD_DIM, HEADS * HEAD_DIM, 80 + d);
      sketch(f[ATTENTION], attention + r * DIM, DIM, 90 + d);
    }
    if (lab.collect_mask & (1u << KV_HISTORY))
      for (int j = 0; j < LAB_SKETCH; j++) f[KV_HISTORY][j] += context[j];
  }
  lab.collect_s += now() - t;
}

void predictor_lab_final(int rows, const float *streams, const float *head) {
  if (!lab.enabled) return;
  if (lab.base == 2) memcpy(lab.head, head, (size_t)rows * DIM * sizeof(float));
  if (!(lab.collect_mask & (1u << FINAL))) return;
  double t = now();
  for (int r = 0; r < rows; r++) {
    sketch(lab.stage[r][FINAL], head + r * DIM, DIM, 101);
    for (int h = 0; h < HC; h++) sketch(lab.stage[r][FINAL], streams + (r * HC + h) * DIM, DIM, 102 + h);
  }
  lab.collect_s += now() - t;
}

void predictor_lab_embeddings(int rows, const uint16_t *embedding) {
  if (!lab.enabled || !lab.pending) return;
  double t = now();
  if (rows != lab.rows) die("predictor lab embedding rows differ");
  for (int r = 1; lab.count && r < rows; r++) sketch_bf16(lab.embedding[r], embedding + r * DIM, DIM, 1);
  lab.collect_s += now() - t;
  lab.prepared = 1;
}

int predictor_lab_projection(void) { return lab.enabled && lab.prepared && lab.base; }

void predictor_lab_projection_input(int rows, const uint16_t *embedding, const float *norm, float *out) {
  if (!predictor_lab_projection() || rows != lab.rows) die("invalid lab projection input");
  for (int r = 0; r < rows; r++) for (int i = 0; i < DIM; i++)
    out[r * DIM + i] = lab.base == 2 && r > 0
        ? (fabsf(norm[i]) > 1e-6f ? lab.head[r - 1][i] / norm[i] : 0)
        : bf16_value(embedding[r * DIM + i]);
}

static void normalized_keys(float *out, const float *logits, const float *bias) {
  float keys[EXPERTS];
  for (int e = 0; e < EXPERTS; e++) keys[e] = op_key(logits[e], bias[e]);
  op_normalize(keys, out, EXPERTS);
}

void predictor_lab_baseline(int l, int rows, const float *logits, const float *bias) {
  if (!predictor_lab_projection() || l < 0 || l >= LAYERS || rows != lab.rows) die("invalid lab baseline");
  for (int r = 0; r < rows; r++)
    normalized_keys(lab.baseline[r] + l * EXPERTS, logits + r * EXPERTS, bias);
}

void predictor_lab_predict(int rows) {
  if (!lab.enabled || !lab.prepared) return;
  if (rows != lab.rows || lab.active) die("invalid predictor lab prediction boundary");
  lab.active = 1, lab.pending = 0, lab.prepared = 0;
  memset(lab.seen, 0, sizeof lab.seen);
  memset(lab.target, 0, sizeof lab.target);
  memset(lab.labels, 0, sizeof lab.labels);
  memset(lab.native_rank, 0, sizeof lab.native_rank);
  if (lab.long_enabled) for (int l = 0; l < LAYERS; l++) for (int h = 0; h < 2; h++)
    lab.longer[l][h].valid = lab.longer[l][h].learned = 0;
  for (int l = 0; l < LAYERS; l++) for (int e = 0; e < EXPERTS; e++) lab.ready_before[l][e] = cache_ready(l, e);
  for (int vi = 0; vi < lab.count; vi++) {
    LabVariant *v = &lab.v[vi];
    double t = now();
    memset(lab.features, 0, sizeof lab.features);
    for (int r = 0; r < rows; r++) {
      for (int g = 0; g < LAB_GROUPS; g++) if (v->mask & (1u << g)) {
        // Proposal row r comes from draft head row r-1. The seed uses row 0
        // as context, with a separate seed flag; no future labels are read.
        const float *from = g == EMBED ? lab.embedding[r] : g == HISTORY ? lab.history : lab.stage[r ? r - 1 : 0][g];
        float *to = lab.features[r] + g * LAB_SKETCH;
        double ss = 0;
        for (int j = 0; j < LAB_SKETCH; j++) ss += (double)from[j] * from[j];
        float scale = 1 / sqrtf((float)ss + 1e-12f);
        for (int j = 0; j < LAB_SKETCH; j++) to[j] = from[j] * scale;
      }
      float *meta = lab.features[r] + LAB_GROUPS * LAB_SKETCH;
      meta[0] = r == 0; meta[1] = r / (float)BLOCK; meta[2] = log1pf(lab.pos) / 10;
      meta[3] = rows / (float)(BLOCK + 1);
      if (v->mask & (1u << HISTORY)) meta[4] = lab.previous_accepted / (float)(BLOCK + 1);
    }
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, rows, lab.rank - 1, LAB_FEATURES,
                1, lab.features[0], LAB_FEATURES, v->projection, lab.rank - 1, 0, v->phi, lab.rank);
    for (int r = 0; r < rows; r++) {
      double ss = 0;
      for (int j = 0; j < lab.rank - 1; j++) { float x = fmaxf(v->phi[r * lab.rank + j], 0); v->phi[r * lab.rank + j] = x; ss += x * x; }
      float scale = lab.gain / sqrtf((float)ss + 1e-12f);
      for (int j = 0; j < lab.rank - 1; j++) v->phi[r * lab.rank + j] *= scale;
      v->phi[r * lab.rank + lab.rank - 1] = 1;
    }
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, rows, LAB_OUTPUTS, lab.rank,
                1, v->phi, lab.rank, v->model.b, LAB_OUTPUTS, 0, v->scores, LAB_OUTPUTS);
    if (lab.base) for (int r = 0; r < rows; r++) for (int j = 0; j < LAB_OUTPUTS; j++)
      v->scores[r * LAB_OUTPUTS + j] += lab.baseline[r][j];
    if (!op_finite_array(v->scores, (size_t)rows * LAB_OUTPUTS)) die("nonfinite predictor lab scores");
    for (int l = 0; l < LAYERS; l++) for (int r = 0; r < rows; r++)
      op_top6(v->scores + (size_t)r * LAB_OUTPUTS + l * EXPERTS, EXPERTS, v->top[l][r]);
    v->predict_s += now() - t;
  }
}

void predictor_lab_routes(int l, int rows, const int *routes, const float *logits, const float *bias) {
  if (!lab.active) return;
  if (l < 0 || l >= LAYERS || lab.seen[l] || rows != lab.rows) die("invalid predictor lab labels");
  lab.seen[l] = 1;
  if (lab.long_enabled) {
    memcpy(lab.long_label, logits, (size_t)rows * EXPERTS * sizeof(float));
    for (int r = 0; r < rows; r++) memcpy(lab.long_ids + r * TOPK, routes + r * 16, TOPK * sizeof(int));
    lab.long_label_layer = l;
  }
  uint8_t target_union[EXPERTS] = {0};
  for (int r = 0; r < rows; r++) for (int k = 0; k < TOPK; k++) {
    int e = routes[r * 16 + k];
    if (e < 0 || e >= EXPERTS) die("invalid native label in predictor lab");
    lab.labels[r][l * EXPERTS + e] = 1;
    target_union[e] = 1;
  }
  for (int a = 0; (lab.policy == 2 || lab.policy == 3) && a < ACTION_RECORDS; a++) {
    CacheAction *c = &lab.action[a];
    if (!c->active) continue;
    if (!c->used && c->layer == l && target_union[c->expert]) {
      int state = cache_action_state(l, c->expert, c->id);
      c->used = state > 0, c->timely = state == 2 && c->pass == lab.passes;
      float fraction = state == 2 ? 1 : state ? fminf(1, atomic_load(&c->transferred) / (float)c->requested) : 0;
      c->progress = fraction * powf(.5f, (float)(lab.passes - c->pass));
    }
    if (c->victim_layer == l && c->victim_expert >= 0 && target_union[c->victim_expert]) c->victim_used = 1;
  }
  for (int r = 0; r < rows; r++) {
    float *target = lab.target[r] + l * EXPERTS;
    if (lab.base) {
      normalized_keys(target, logits + r * EXPERTS, bias);
      for (int e = 0; e < EXPERTS; e++) target[e] -= lab.baseline[r][l * EXPERTS + e];
    } else for (int e = 0; e < EXPERTS; e++) target[e] = lab.labels[r][l * EXPERTS + e];
  }
  if (lab.passes < (uint64_t)lab.warmup) return;
  if (lab.base) for (int r = 0; r < rows; r++) {
    int ids[TOPK];
    op_top6(lab.baseline[r] + l * EXPERTS, EXPERTS, ids);
    for (int k = 0; k < TOPK; k++) lab.baseline_hits += lab.labels[r][l * EXPERTS + ids[k]], lab.baseline_total++;
  }
  for (int vi = 0; vi < lab.count; vi++) {
    LabVariant *v = &lab.v[vi];
    uint8_t predicted[EXPERTS] = {0};
    for (int r = 0; r < rows; r++) for (int k = 0; k < TOPK; k++) {
      int e = v->top[l][r][k];
      int hit = lab.labels[r][l * EXPERTS + e] != 0;
      v->raw_hits += hit, v->raw_total++, v->layer_hits[l] += hit, v->layer_total[l]++;
      predicted[e] = 1;
    }
    for (int e = 0; e < EXPERTS; e++) {
      v->union_predictions += predicted[e];
      v->union_hits += predicted[e] && target_union[e];
      if (target_union[e] && !lab.ready_before[l][e]) v->missing++, v->missing_hits += predicted[e];
    }
  }
}

void predictor_lab_schedule(int completed) {
  if (!lab.active || !lab.live || lab.passes < (uint64_t)lab.warmup) return;
  double t = now();
  lab.completed = completed;
  if (lab.policy >= 3 || lab.long_enabled) return;
  int first = completed + lab.distance, last = completed + lab.horizon;
  if (first < 0) first = 0;
  if (last >= LAYERS) last = LAYERS - 1;
  for (int l = first; l <= last; l++) {
    int ids[(BLOCK + 1) * TOPK], n = 0;
    uint8_t seen[EXPERTS] = {0};
    if (!lab.policy) {
      for (int k = 0; k < lab.width; k++) for (int r = 0; r < lab.rows; r++) {
        int e = lab.v[0].top[l][r][k];
        if (!seen[e]) seen[e] = 1, ids[n++] = e;
      }
    } else {
      // The admission-policy comparison has the same uncached candidate pool.
      // Max across rows avoids multiplying correlated row probabilities.
      float score[EXPERTS];
      for (int e = 0; e < EXPERTS; e++) {
        score[e] = -INFINITY;
        if (!cache_slot(l, e)) for (int r = 0; r < lab.rows; r++)
          score[e] = fmaxf(score[e], lab.v[0].scores[(size_t)r * LAB_OUTPUTS + l * EXPERTS + e]);
      }
      for (int k = 0; k < lab.width * lab.rows; k++) {
        int best = -1;
        for (int e = 0; e < EXPERTS; e++) if (isfinite(score[e]) && (best < 0 || score[e] > score[best])) best = e;
        if (best < 0) break;
        ids[n++] = best, score[best] = -INFINITY;
      }
    }
    cache_prefetch_from(l, ids, n, lab.rows * lab.width, 6 * lab.rows, PF_DRAFT);
  }
  lab.schedules++;
  lab.schedule_s += now() - t;
}

void predictor_lab_end(void) {
  if (!lab.active) return;
  for (int l = 0; l < LAYERS; l++) if (!lab.seen[l]) die("missing predictor lab target layer");
  for (int vi = 0; vi < lab.count; vi++) {
    LabVariant *v = &lab.v[vi];
    double t = now();
    for (int r = 0; lab.learn && r < lab.rows; r++) op_update(&v->model, &lab.work, v->phi + r * lab.rank, lab.target[r]);
    v->update_s += now() - t;
  }
  double action_start = now();
  for (int i = 0; (lab.policy == 2 || lab.policy == 3) && i < ACTION_RECORDS; i++) {
    CacheAction *a = &lab.action[i];
    if (!a->active || lab.passes < a->pass + 2) continue;
    // Reader callbacks update only transferred. A pending owned group keeps
    // the record alive; group completion/cancellation precedes slot reuse.
    if (cache_action_state(a->layer, a->expert, a->id) == 1) continue;
    // Delayed contextual-bandit feedback. These are readiness/reuse proxies,
    // not a counterfactual measurement of milliseconds saved by this action.
    float transferred = fminf(1, atomic_load(&a->transferred) / (float)a->requested);
    float fetch_reward = (a->used ? a->progress : -lab.waste_cost * transferred) * a->cost;
    float victim_cost = a->victim_used ? a->victim_scale : 0;
    if (lab.learn) {
      op_update(&lab.fetch_value, &lab.action_work, a->fetch, &fetch_reward);
      if (a->victim_expert >= 0) op_update(&lab.retain_value, &lab.action_work, a->retain, &victim_cost);
    }
    lab.action_settled++, lab.action_used += a->used, lab.action_timely += a->timely;
    lab.victim_reused += a->victim_used, lab.action_reward += fetch_reward - victim_cost;
    a->active = 0;
  }
  lab.action_fit_s += now() - action_start;
  lab.scored_passes += lab.passes >= (uint64_t)lab.warmup;
  lab.passes++, lab.rows_seen += lab.rows;
  lab.active = 0;
}

void predictor_lab_accept(int rows) {
  if (!lab.enabled) return;
  if (rows < 1 || rows > lab.rows || lab.active) die("invalid predictor lab accepted prefix");
  memset(lab.history, 0, sizeof lab.history);
  for (int i = 0; i < LAB_OUTPUTS; i++) lab.prior_labels[i] = .9f * lab.prior_labels[i] + .1f * lab.labels[rows - 1][i];
  sketch(lab.history, lab.prior_labels, LAB_OUTPUTS, 301);
  lab.previous_accepted = rows;
}

int predictor_lab_cache_policy(void) {
  return lab.active && lab.live && lab.passes >= (uint64_t)lab.warmup ? lab.policy : 0;
}

int predictor_lab_suppress_prefetch(void) { return lab.active && lab.live && lab.policy == 5; }
int predictor_lab_victim_limit(void) { return lab.victim_limit; }

void predictor_lab_native_candidates(int layer, const int *ids, int n, int issuer) {
  if (!lab.active || lab.policy < 3 || layer < 0 || layer >= LAYERS) return;
  lab.issuer = issuer;
  if (lab.long_enabled && (issuer == PF_H4 || issuer == PF_H8)) return; // supplied with all six source ranks
  for (int e = 0; e < EXPERTS; e++) lab.native_rank[layer][e] = 0;
  for (int i = 0; i < n; i++) {
    if (ids[i] < 0 || ids[i] >= EXPERTS) die("invalid action candidate");
    // Current PC/h1/h2 callers supply the top three of each row.
    float rank = issuer == PF_H4 || issuer == PF_H8 ? 1 : 1.0f / (1 + i % 3);
    lab.native_rank[layer][ids[i]] = fmaxf(lab.native_rank[layer][ids[i]], rank);
  }
}

static void action_features(float *f, int layer, int expert, uint64_t bytes, int pending, double frequency, double age) {
  memset(f, 0, ACTION_FEATURES * sizeof *f);
  f[0] = 1;
  if (layer < 0 || layer >= LAYERS || expert < 0 || expert >= EXPERTS) return;
  if (lab.policy >= 3) f[1] = lab.native_rank[layer][expert];
  else {
    float score = -INFINITY;
    for (int r = 0; r < lab.rows; r++) score = fmaxf(score, lab.v[0].scores[(size_t)r * LAB_OUTPUTS + layer * EXPERTS + expert]);
    f[1] = tanhf(score * .5f);
  }
  f[2] = fminf(1, fmaxf(0, (layer - lab.completed) / 8.0f));
  f[3] = f[2] * f[2];
  f[4] = tanhf(log1pf((float)frequency) / 3);
  f[5] = 1 / (1 + (float)age / 10);
  f[6] = lab.prior_labels[layer * EXPERTS + expert];
  f[7] = fminf(1, pending / (6.0f * lab.rows));
  f[8] = bytes / 20000000.0f;
  f[9] = lab.rows / 6.0f;
  f[10] = f[1] * f[2], f[11] = f[1] * f[7], f[12] = f[1] * f[6];
  f[13] = layer / 40.0f, f[14] = f[4] * f[5];
  f[15] = lab.policy >= 3 ? lab.issuer / 2.0f : 1;
}

int predictor_lab_choose(int layer, int expert, uint64_t bytes, int pending,
                         const LabVictim *victims, int count, uint64_t *action_id) {
  *action_id = 0;
  if (!predictor_lab_cache_policy() || (lab.policy != 2 && lab.policy != 3)) return count ? victims[0].slot : -1;
  double started = now();
  lab.action_attempts++;
  float fetch[ACTION_FEATURES], keep[ACTION_FEATURES] = {0}, candidate[ACTION_FEATURES];
  action_features(fetch, layer, expert, bytes, pending, 0, 0);
  float benefit = cblas_sdot(ACTION_FEATURES, lab.fetch_value.b, 1, fetch, 1);
  float cost = 0;
  int chosen = -1;
  for (int i = 0; i < count; i++) {
    if (victims[i].expert < 0) { chosen = i, cost = 0; memset(keep, 0, sizeof keep); break; }
    action_features(candidate, victims[i].layer, victims[i].expert, victims[i].bytes, pending, victims[i].frequency, victims[i].age);
    float value = fmaxf(0, cblas_sdot(ACTION_FEATURES, lab.retain_value.b, 1, candidate, 1));
    if (chosen < 0 || value < cost) chosen = i, cost = value, memcpy(keep, candidate, sizeof keep);
  }
  // A separate deterministic RNG cannot change target-model sampling.
  int explore = !lab.explored && op_uniform(&lab.action_rng) < .02;
  if (explore) lab.explored = 1;
  if (benefit <= fmaxf(cost, 0) && !explore) {
    lab.action_abstained++, lab.action_decision_s += now() - started;
    return -2;
  }
  if (explore && count && victims[chosen].expert >= 0) {
    chosen = (int)(op_random64(&lab.action_rng) % (unsigned)count);
    action_features(keep, victims[chosen].layer, victims[chosen].expert, victims[chosen].bytes,
                    pending, victims[chosen].frequency, victims[chosen].age);
  }
  int free_record = 0;
  while (free_record < ACTION_RECORDS && lab.action[free_record].active) free_record++;
  if (free_record == ACTION_RECORDS) {
    lab.action_abstained++, lab.action_decision_s += now() - started;
    return -2;
  }
  CacheAction *a = &lab.action[free_record];
  *a = (CacheAction){.id = ++lab.action_id * ACTION_RECORDS + free_record + 1, .pass = lab.passes,
      .requested = bytes, .active = 1, .layer = layer, .expert = expert,
      .victim_layer = chosen < 0 ? -1 : victims[chosen].layer,
      .victim_expert = chosen < 0 ? -1 : victims[chosen].expert, .cost = bytes / 20000000.0f,
      .victim_scale = chosen < 0 ? 0 : victims[chosen].bytes / 20000000.0f};
  memcpy(a->fetch, fetch, sizeof fetch), memcpy(a->retain, keep, sizeof keep);
  *action_id = a->id;
  lab.action_admitted++;
  lab.action_decision_s += now() - started;
  return chosen < 0 ? -1 : victims[chosen].slot;
}

void predictor_lab_read(uint64_t id, uint64_t bytes) {
  if (!id) return;
  CacheAction *a = &lab.action[(id - 1) % ACTION_RECORDS];
  if (a->id != id) die("read outlived its cache-action record");
  atomic_fetch_add_explicit(&a->transferred, bytes, memory_order_relaxed);
}

int predictor_lab_long_enabled(void) { return lab.enabled && lab.long_enabled; }
int predictor_lab_long_active(void) { return lab.active && lab.long_enabled; }

void predictor_lab_long_init(const float *const biases[LAYERS]) {
  if (!predictor_lab_long_enabled()) return;
  if (lab.long_initialized) die("long predictor initialized twice");
  for (int source = 0; source < LAYERS; source++) for (int h = 0; h < 2; h++) {
    int target = source + (h ? 8 : 4);
    if (target >= LAYERS) continue;
    LongHead *p = &lab.longer[source][h];
    p->model = op_create_pair(EXPERTS, lab.rank, 2001 + source * 13 + h);
    memcpy(p->model.bias, biases[target], EXPERTS * sizeof(float));
  }
  lab.long_work = op_create_work(EXPERTS, lab.rank);
  lab.long_initialized = 1;
}

void predictor_lab_long_forecast(int source, int rows, const float *gp4, const float *gp8, const float *current) {
  if (!predictor_lab_long_active()) return;
  if (!lab.long_initialized || rows != lab.rows) die("invalid long forecast state");
  double began = now();
  for (int h = 0; h < 2; h++) {
    int target = source + (h ? 8 : 4);
    if (target >= LAYERS) continue;
    LongHead *p = &lab.longer[source][h];
    const float *gp = h ? gp8 : gp4;
    float keys[EXPERTS];
    for (int r = 0; r < rows; r++) {
      for (int e = 0; e < EXPERTS; e++) keys[e] = op_key(gp[r * EXPERTS + e], p->model.bias[e]);
      op_top6(keys, EXPERTS, p->baseline + r * TOPK);
    }
    int status = op_forward(&p->model, &lab.long_work, gp, current, rows, p->baseline, p->ids);
    p->valid = status >= 0, p->learned = status != 0;
    p->pass = lab.passes, p->rows = rows;
    if (status < 0) continue;
    memcpy(p->phi, lab.long_work.phi, (size_t)rows * lab.rank * sizeof(float));
    memcpy(p->keys, lab.long_work.keys, (size_t)rows * EXPERTS * sizeof(float));
    for (int e = 0; e < EXPERTS; e++) p->missing[e] = !cache_ready(target, e);
    if (!lab.live || lab.passes < (uint64_t)lab.warmup) continue;
    int keep[OP_MAX_ROWS * TOPK], ids[OP_MAX_ROWS * TOPK], nk = 0, n = 0;
    LongHead *earlier = !h && source >= 4 ? &lab.longer[source - 4][1] : NULL;
    memset(lab.native_rank[target], 0, sizeof lab.native_rank[target]);
    for (int r = 0; r < rows; r++) for (int k = 0; k < TOPK; k++) {
      int e = p->ids[r * TOPK + k];
      lab.native_rank[target][e] = fmaxf(lab.native_rank[target][e], 1.0f / (k + 1));
    }
    for (int r = 0; r < rows; r++) for (int k = 0; k < TOPK; k++) keep[nk++] = p->ids[r * TOPK + k];
    if (!h) cache_refine(target, keep, nk);
    for (int r = 0; r < rows; r++) {
      int offered = 0;
      for (int k = 0; k < (lab.long_missing_first ? TOPK : lab.width) && offered < lab.width; k++) {
        int e = p->ids[r * TOPK + k];
        if (lab.long_missing_first && cache_slot(target, e)) continue;
        int agree = lab.long_missing_first ? offered == 0 : k == 0;
        if (earlier && earlier->valid && earlier->pass == lab.passes && earlier->rows == rows)
          for (int j = 0; j < TOPK; j++) agree |= earlier->ids[r * TOPK + j] == e;
        if (agree) ids[n++] = e, offered++;
      }
    }
    cache_prefetch_from(target, ids, n, rows * lab.width, 6 * rows, h ? PF_H8 : PF_H4);
  }
  lab.long_forward_s += now() - began;
}

void predictor_lab_long_update(int layer) {
  if (!predictor_lab_long_active()) return;
  if (lab.long_label_layer != layer) die("long predictor label was overwritten");
  double began = now();
  for (int h = 0; h < 2; h++) {
    int source = layer - (h ? 8 : 4);
    if (source < 0) continue;
    LongHead *p = &lab.longer[source][h];
    if (!p->valid || p->learned || p->pass != lab.passes || p->rows != lab.rows) continue;
    p->learned = 1;
    if (lab.passes >= (uint64_t)lab.warmup) {
      uint8_t actual[EXPERTS] = {0}, predicted[EXPERTS] = {0};
      for (int r = 0; r < lab.rows; r++) for (int k = 0; k < TOPK; k++) {
        int got = 0, base = 0;
        for (int j = 0; j < TOPK; j++) {
          got |= p->ids[r * TOPK + k] == lab.long_ids[r * TOPK + j];
          base |= p->baseline[r * TOPK + k] == lab.long_ids[r * TOPK + j];
        }
        lab.long_hits[h] += got, lab.long_base_hits[h] += base, lab.long_total[h]++;
        actual[lab.long_ids[r * TOPK + k]] = 1, predicted[p->ids[r * TOPK + k]] = 1;
      }
      for (int e = 0; e < EXPERTS; e++) if (actual[e] && p->missing[e]) {
        lab.long_missing[h]++, lab.long_missing_hits[h] += predicted[e];
      }
    }
    if (lab.learn) for (int r = 0; r < lab.rows; r++) {
      float residual[EXPERTS];
      for (int e = 0; e < EXPERTS; e++)
        residual[e] = op_key(lab.long_label[r * EXPERTS + e], p->model.bias[e]) - p->keys[r * EXPERTS + e];
      op_update(&p->model, &lab.long_work, p->phi + r * lab.rank, residual);
    }
  }
  lab.long_update_s += now() - began;
}

void predictor_lab_report(FILE *out) {
  if (!lab.enabled) return;
  fprintf(out, "{\"predictor_lab\":{\"mode\":\"%s\",\"rank\":%d,\"feature_gain\":%.4f,\"joint_predictor\":%s,\"passes\":%llu,\"rows\":%llu,"
          "\"warmup_passes\":%d,\"scored_passes\":%llu,\"width\":%d,\"distance\":%d,\"horizon\":%d,"
          "\"collect_ms\":%.3f,\"schedule_ms\":%.3f,\"base\":%d,\"learning\":%s,"
          "\"baseline_raw_hits\":%llu,\"baseline_raw_total\":%llu,\"policy\":%d,\"victim_candidates\":%d,\"variants\":[",
          lab.live ? "prefetch" : "shadow", lab.rank, lab.gain, lab.count ? "true" : "false", (unsigned long long)lab.passes,
          (unsigned long long)lab.rows_seen, lab.warmup, (unsigned long long)lab.scored_passes,
          lab.width, lab.distance, lab.horizon, 1000 * lab.collect_s, 1000 * lab.schedule_s,
          lab.base, lab.learn ? "true" : "false", (unsigned long long)lab.baseline_hits,
          (unsigned long long)lab.baseline_total, lab.policy, lab.victim_limit);
  for (int vi = 0; vi < lab.count; vi++) {
    LabVariant *v = &lab.v[vi];
    fprintf(out, "%s{\"name\":\"%s\",\"raw_hits\":%llu,\"raw_total\":%llu,\"union_hits\":%llu,"
            "\"union_predictions\":%llu,\"missing_at_forecast\":%llu,\"missing_at_forecast_hits\":%llu,"
            "\"predict_ms\":%.3f,\"update_ms\":%.3f,\"updates\":%llu,\"rejected_updates\":%llu,\"resets\":%llu}",
            vi ? "," : "", names[v->name], (unsigned long long)v->raw_hits, (unsigned long long)v->raw_total,
            (unsigned long long)v->union_hits, (unsigned long long)v->union_predictions,
            (unsigned long long)v->missing, (unsigned long long)v->missing_hits,
            1000 * v->predict_s, 1000 * v->update_s, (unsigned long long)v->model.updates,
            (unsigned long long)v->model.rejected_updates, (unsigned long long)v->model.covariance_resets);
  }
  fprintf(out, "],\"long_forecasts\":{\"enabled\":%s,\"missing_first\":%s,\"forward_ms\":%.3f,\"update_ms\":%.3f,\"horizons\":[",
          lab.long_enabled ? "true" : "false", lab.long_missing_first ? "true" : "false",
          1000 * lab.long_forward_s, 1000 * lab.long_update_s);
  for (int h = 0; h < 2; h++) fprintf(out, "%s{\"distance\":%d,\"raw_hits\":%llu,\"baseline_hits\":%llu,\"raw_total\":%llu,"
      "\"missing\":%llu,\"missing_hits\":%llu}", h ? "," : "", h ? 8 : 4,
      (unsigned long long)lab.long_hits[h], (unsigned long long)lab.long_base_hits[h],
      (unsigned long long)lab.long_total[h], (unsigned long long)lab.long_missing[h], (unsigned long long)lab.long_missing_hits[h]);
  fprintf(out, "]},\"actions\":{\"attempted\":%llu,\"admitted\":%llu,\"abstained\":%llu,\"settled\":%llu,"
          "\"used\":%llu,\"timely\":%llu,\"victims_reused_within_two_passes\":%llu,\"proxy_reward\":%.3f,"
          "\"decision_ms\":%.3f,\"fit_ms\":%.3f,\"features\":16,\"reward_model\":\"progress_bytes_v1\","
          "\"fetch_updates\":%llu,\"retain_updates\":%llu,\"rejected_updates\":%llu,\"covariance_resets\":%llu,\"waste_cost\":%.4f},",
          (unsigned long long)lab.action_attempts, (unsigned long long)lab.action_admitted,
          (unsigned long long)lab.action_abstained, (unsigned long long)lab.action_settled,
          (unsigned long long)lab.action_used, (unsigned long long)lab.action_timely,
          (unsigned long long)lab.victim_reused, lab.action_reward, 1000 * lab.action_decision_s, 1000 * lab.action_fit_s,
          (unsigned long long)lab.fetch_value.updates, (unsigned long long)lab.retain_value.updates,
          (unsigned long long)(lab.fetch_value.rejected_updates + lab.retain_value.rejected_updates),
          (unsigned long long)(lab.fetch_value.covariance_resets + lab.retain_value.covariance_resets), lab.waste_cost);
  fputs("\"scope\":\"causal predictions and action outcomes; first-use timeliness is in prefetch_metrics; "
        "computed rows include rejected drafts; shadow accuracy is not a throughput result\"}}\n", out);
}
