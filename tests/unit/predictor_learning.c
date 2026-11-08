// CPU-only math, ownership and causal-lifecycle checks for the opt-in module.
#include "../../src/model/deepseek/model.h"
#include "../../src/model/deepseek/predictor.c"
#include <sys/wait.h>

static int checks, failures;
static void expect(const char *name, int ok) {
  checks++;
  if (!ok) failures++, fprintf(stderr, "FAIL: %s\n", name);
}
static void fill(int layer, int rows, float *gp, float *gl, int *pr, int *rt) {
  memset(pr, 0, 2 * OP_MAX_ROWS * ONLINE_PREDICT * sizeof *pr);
  memset(rt, 0, OP_MAX_ROWS * 16 * sizeof *rt);
  for (int row = 0; row < rows; row++) {
    float score[EXPERTS];
    int ids[OP_K];
    for (int e = 0; e < EXPERTS; e++) gl[row * EXPERTS + e] = .2f + (e % 13) * .03f + layer * .001f + row * .01f;
    for (int e = 0; e < EXPERTS; e++) score[e] = op_key(gl[row * EXPERTS + e], 0);
    op_top6(score, EXPERTS, ids);
    memcpy(rt + row * 16, ids, sizeof ids);
    for (int h = 0; h < 2; h++) {
      for (int e = 0; e < EXPERTS; e++) {
        float value = (e % 11) * .02f + layer * .002f + row * .01f + h * .02f;
        gp[(h * rows + row) * EXPERTS + e] = value;
        score[e] = op_key(value, 0);
      }
      op_top6(score, EXPERTS, ids);
      memcpy(pr + (h * rows + row) * ONLINE_PREDICT, ids, (h ? 3 : 6) * sizeof *ids);
    }
  }
}
static void math_checks(void) {
  OpPair p = op_create_pair(8, 4, 91);
  OpWork w = op_create_work(8, 4);
  float gp[6 * 8], gl[6 * 8], bias[8] = {0};
  int native[6 * OP_K], actual[6 * OP_K];
  for (int row = 0; row < 6; row++) {
    float keys[8];
    for (int e = 0; e < 8; e++) gp[row * 8 + e] = (e - 4) * .1f + row * .01f, gl[row * 8 + e] = (e % 3) * .2f;
    for (int e = 0; e < 8; e++) keys[e] = op_key(gp[row * 8 + e], bias[e]);
    op_top6(keys, 8, native + row * OP_K);
  }
  expect("six-row cold prediction preserves native IDs",
         op_forward(&p, &w, gp, gl, 6, native, actual) == 0 && !memcmp(native, actual, sizeof native));
  float phi[4] = {.2f, .3f, 0, .1f}, target[8];
  double denominator = (double).995f;
  for (int i = 0; i < 4; i++) denominator += phi[i] * phi[i] / 16.0;
  for (int e = 0; e < 8; e++) target[e] = (e - 3) * .01f;
  expect("checked RLS update accepted", op_update(&p, &w, phi, target));
  double error = 0;
  for (int i = 0; i < 4; i++)
    for (int e = 0; e < 8; e++) error = fmax(error, fabs(p.b[i * 8 + e] - (phi[i] / 16.0 / denominator) * target[e]));
  expect("RLS B matches double reference", error < 1e-7);
  expect("RLS P remains finite", op_finite_array(p.p, 16));
  gp[0] = NAN;
  expect("nonfinite input falls back to native IDs",
         op_forward(&p, &w, gp, gl, 6, native, actual) == 1 && !memcmp(native, actual, sizeof native));
  native[1] = native[0];
  expect("duplicate fallback IDs refused", op_forward(&p, &w, gp, gl, 6, native, actual) < 0);
  float saved[32];
  memcpy(saved, p.b, sizeof saved);
  memset(phi, 0, sizeof phi);
  p.p[0] = INFINITY;
  op_update(&p, &w, phi, target);
  expect("covariance reset retains finite B", p.covariance_resets > 0 && !memcmp(saved, p.b, sizeof saved));
  op_free_pair(&p);
  op_free_work(&w);
}

static void environment_checks(void) {
  const char *name = "MYDS4_ONLINE_FREEZE_ROWS";
  unsetenv(name);
  unsetenv("MYDS4_ONLINE_WINDOW_PASSES");
  expect("absent learning controls default to zero", online_env_integer(name, 1048576) == 0);
  setenv(name, "512", 1);
  expect("freeze threshold accepts a bounded integer", online_env_integer(name, 1048576) == 512);
  const char *bad[] = {"", "-1", "1x", "1048577", "18446744073709551616"};
  int rejected = 1;
  for (size_t i = 0; i < sizeof bad / sizeof *bad; i++) {
    pid_t child = fork();
    if (!child) {
      setenv(name, bad[i], 1);
      online_env_integer(name, 1048576);
      _exit(0);
    }
    if (child < 0) {
      rejected = 0;
      continue;
    }
    int status;
    if (waitpid(child, &status, 0) != child || !WIFEXITED(status) || WEXITSTATUS(status) == 0) rejected = 0;
  }
  expect("malformed and out-of-range freeze controls are rejected", rejected);
  unsetenv(name);
  setenv("MYDS4_ONLINE_WINDOW_PASSES", "4096", 1);
  expect("window interval accepts its upper bound", online_env_integer("MYDS4_ONLINE_WINDOW_PASSES", 4096) == 4096);
  unsetenv("MYDS4_ONLINE_WINDOW_PASSES");
}

int main(void) {
  alarm(60);
  environment_checks();
  if (BLASSetThreading(BLAS_THREADING_SINGLE_THREADED)) return 1;
  math_checks();
  static float bias[LAYERS][EXPERTS];
  const float *biases[LAYERS];
  for (int l = 0; l < LAYERS; l++) biases[l] = bias[l];
  online_predictor_init(64, biases);
  expect("all 77 heads initialized", online.count == 77);
  float *a_before = op_allocate(768 * ONLINE_RANK * 4);
  memcpy(a_before, online.models[0].a, 768 * ONLINE_RANK * 4);
  float gp[2 * OP_MAX_ROWS * EXPERTS], gl[OP_MAX_ROWS * EXPERTS];
  int pr[2 * OP_MAX_ROWS * ONLINE_PREDICT], rt[OP_MAX_ROWS * 16];
  online_predictor_begin(3, 0);
  fill(0, 3, gp, gl, pr, rt);
  expect("cold prefill keeps original forecasts", online_predictor_forecasts(0, 3, 2, gp, gl, pr, rt) == pr);
  online_predictor_update(0);
  online_predictor_end();
  expect("cold prefill does not train", online.models[0].updates == 0);
  online_predictor_reset();
  online_predictor_begin(6, 1);
  int unchanged = 1, valid = 1;
  for (int layer = 0; layer < LAYERS; layer++) {
    fill(layer, 6, gp, gl, pr, rt);
    const int *out = online_predictor_forecasts(layer, 6, 2, gp, gl, pr, rt);
    int hops = layer + 2 < LAYERS ? 2 : layer + 1 < LAYERS;
    if (!out) {
      valid = 0;
      continue;
    }
    for (int h = 0; h < hops; h++)
      for (int row = 0; row < 6; row++) {
        unchanged &= !memcmp(out + (h * 6 + row) * ONLINE_PREDICT, pr + (h * 6 + row) * ONLINE_PREDICT,
                             (h ? 3 : 6) * sizeof(int));
        int ids[OP_K];
        memcpy(ids, out + (h * 6 + row) * ONLINE_PREDICT, sizeof ids);
        valid &= op_valid_ids(ids, EXPERTS, 1);
      }
    // Simulate GPU scratch reuse before the delayed target update.
    for (size_t i = 0; i < sizeof gp / sizeof *gp; i++) gp[i] = NAN;
    for (size_t i = 0; i < sizeof gl / sizeof *gl; i++) gl[i] = NAN;
    for (size_t i = 0; i < sizeof rt / sizeof *rt; i++) rt[i] = -1;
    online_predictor_update(layer);
  }
  online_predictor_end();
  expect("first whole pass uses cold snapshots", unchanged);
  expect("corrected IDs remain valid and unique", valid);
  int trained = 1;
  for (int p = 0; p < online.count; p++) trained &= online.models[p].updates == 6;
  expect("every head trains only after its target using owned copies", trained);
  expect("raw candidate denominator counts each row once", online.candidates == 77u * 6u * 3u);
  online_predictor_begin(3, 1);
  valid = 1;
  for (int layer = 0; layer < LAYERS; layer++) {
    fill(layer, 3, gp, gl, pr, rt);
    const int *out = online_predictor_forecasts(layer, 3, 2, gp, gl, pr, rt);
    int hops = layer + 2 < LAYERS ? 2 : layer + 1 < LAYERS;
    if (!out) valid = 0;
    for (int h = 0; out && h < hops; h++)
      for (int row = 0; row < 3; row++) valid &= op_valid_ids(out + (h * 3 + row) * ONLINE_PREDICT, EXPERTS, 1);
    online_predictor_update(layer);
  }
  online_predictor_end();
  expect("learned forecasts remain valid after verification row count changes", valid);
  trained = 1;
  for (int p = 0; p < online.count; p++) trained &= online.models[p].updates == 9;
  expect("changed row count updates each new row once", trained);
  uint64_t before_h1 = online.models[online.pair[0][0]].updates;
  uint64_t before_h2 = online.models[online.pair[0][1]].updates;
  uint64_t before_invalid = online.invalid_native;
  uint64_t before_fallback = online.models[online.pair[0][1]].fallback_rows;
  online_predictor_begin(3, 1);
  for (int layer = 0; layer < LAYERS; layer++) {
    fill(layer, 3, gp, gl, pr, rt);
    for (int i = 3 * EXPERTS; i < 6 * EXPERTS; i++) gp[i] = NAN;
    for (int i = 3 * ONLINE_PREDICT; i < 6 * ONLINE_PREDICT; i++) pr[i] = -1;
    const int *out = online_predictor_forecasts(layer, 3, 1, gp, gl, pr, rt);
    if (!out) valid = 0;
    // Served IDs may change after observation; native labels remain owned.
    memset(rt, 0xff, sizeof rt);
    online_predictor_update(layer);
  }
  online_predictor_end();
  expect("one-horizon passes ignore poisoned h2 inputs",
         valid && online.invalid_native == before_invalid &&
             online.models[online.pair[0][1]].fallback_rows == before_fallback);
  expect("one-horizon pass trains h1 from copied native labels",
         online.models[online.pair[0][0]].updates == before_h1 + 3);
  expect("one-horizon pass never trains stale h2 records", online.models[online.pair[0][1]].updates == before_h2);
  online_predictor_begin(1, 1);
  for (int layer = 0; layer < LAYERS; layer++) {
    fill(layer, 1, gp, gl, pr, rt);
    online_predictor_forecasts(layer, 1, 2, gp, gl, pr, rt);
    online_predictor_update(layer);
  }
  online_predictor_end();
  expect("h2 learning resumes when the next pass computes it",
         online.models[online.pair[0][1]].updates == before_h2 + 1);
  uint64_t before = online.models[0].updates;
  online_predictor_begin(1, 1);
  fill(1, 1, gp, gl, pr, rt);
  online_predictor_forecasts(1, 1, 2, gp, gl, pr, rt);
  online_predictor_update(1);
  expect("missing source epoch cannot train old records", online.models[0].updates == before);
  uint64_t candidates = online.candidates;
  online_predictor_update(1);
  expect("repeated target cannot consume twice", online.candidates == candidates);
  online_predictor_end();
  online_predictor_reset();
  int cleared = 1;
  for (int p = 0; p < online.count; p++) {
    for (int i = 0; i < ONLINE_RANK * EXPERTS; i++) cleared &= online.models[p].b[i] == 0;
    for (int i = 0; i < ONLINE_RANK; i++) cleared &= online.models[p].p[i * ONLINE_RANK + i] == .0625f;
  }
  expect("learner reset clears B and restores P", cleared);
  expect("learner reset keeps the seeded A", !memcmp(a_before, online.models[0].a, 768 * ONLINE_RANK * 4));
  expect("prediction records stay below 160 KiB per supported row",
         ONLINE_PAIRS * sizeof(OnlineRecord) < (size_t)OP_MAX_ROWS * 160 * 1024);
  online_predictor_begin(3, 1);
  fill(0, 3, gp, gl, pr, rt);
  gp[0] = NAN;
  const int *fallback = online_predictor_forecasts(0, 3, 2, gp, gl, pr, rt);
  expect("live nonfinite forecast preserves native first three", fallback && !memcmp(fallback, pr, 3 * sizeof(int)));
  expect("invalid forward cannot feed learning", !online.records[online.pair[0][0]].learnable);
  fill(0, 3, gp, gl, pr, rt);
  pr[1] = pr[0];
  expect("invalid native candidate skips optional read-ahead",
         online_predictor_forecasts(0, 3, 2, gp, gl, pr, rt) == NULL);
  online_predictor_end();
  online_predictor_reset();
  online.freeze_rows = 7;
  online.window_passes = 2;
  uint64_t initial_updates[ONLINE_PAIRS];
  for (int p = 0; p < online.count; p++) initial_updates[p] = online.models[p].updates;
  int crossing_unfrozen = 1;
  for (int pass = 0; pass < 2; pass++) {
    int rows = pass ? 3 : 6;
    online_predictor_begin(rows, 1);
    for (int layer = 0; layer < LAYERS; layer++) {
      fill(layer, rows, gp, gl, pr, rt);
      online_predictor_forecasts(layer, rows, 2, gp, gl, pr, rt);
      online_predictor_update(layer);
      crossing_unfrozen &= !online.frozen;
    }
    online_predictor_end();
  }
  expect("threshold crossing never freezes part of a pass", crossing_unfrozen && online.epoch_rows == 9);
  trained = 1;
  for (int p = 0; p < online.count; p++) trained &= online.models[p].updates == initial_updates[p] + 9;
  expect("threshold-crossing pass trains every head and row", trained);
  expect("window reports only completed whole passes",
         online.window_index == 1 && online.window.passes == online.completed_passes);
  float frozen_b[ONLINE_RANK * EXPERTS], frozen_p[ONLINE_RANK * ONLINE_RANK];
  memcpy(frozen_b, online.models[0].b, sizeof frozen_b);
  memcpy(frozen_p, online.models[0].p, sizeof frozen_p);
  uint64_t guesses_before = online.candidates;
  online_predictor_begin(3, 1);
  expect("pass following threshold starts frozen", online.frozen);
  for (int layer = 0; layer < LAYERS; layer++) {
    fill(layer, 3, gp, gl, pr, rt);
    online_predictor_forecasts(layer, 3, 2, gp, gl, pr, rt);
    online_predictor_update(layer);
  }
  online_predictor_end();
  expect("frozen pass preserves B and P", !memcmp(frozen_b, online.models[0].b, sizeof frozen_b) &&
                                              !memcmp(frozen_p, online.models[0].p, sizeof frozen_p));
  expect("frozen pass still evaluates every candidate", online.candidates - guesses_before == 77u * 3u * 3u);
  expect("frozen row skips are counted separately", online.frozen_skips == 77u * 3u);
  online_window_report(1);
  expect("final partial window advances its snapshot once",
         online.window_index == 2 && online.window.passes == online.completed_passes);
  online_window_report(1);
  expect("repeated report cannot duplicate the final window", online.window_index == 2);
  online_predictor_reset();
  online_predictor_begin(1, 1);
  expect("new learner resets freeze state and completed-row threshold",
         !online.frozen && online.epoch_rows == 0 && online.window_index == 0);
  expect("explicit learner reset clears source epochs", !online.records[0].observed);
  online_predictor_end();
  online_predictor_report(stdout);
  free(a_before);
  for (int p = 0; p < online.count; p++) op_free_pair(&online.models[p]);
  op_free_work(&online.work);
  free(online.models);
  free(online.records);
  fprintf(stderr, "online predictor: %d/%d CPU checks passed\n", checks - failures, checks);
  return failures != 0;
}
