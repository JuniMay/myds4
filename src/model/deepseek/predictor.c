// Prefetch-only rank64 RLS, owned by the model thread.
// Main routing, expert weights and residual-PC predictions are untouched.
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#include <errno.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>
#include <unistd.h>
#include "model.h"
#include "myds4.h"
#include "learning.h"
#include <CommonCrypto/CommonDigest.h>
#include <fcntl.h>
#include <stdarg.h>
#include <sys/stat.h>

enum { ONLINE_PAIRS = 77, ONLINE_RANK = 64, ONLINE_PREDICT = 16 };
typedef struct {
  uint64_t epoch;
  int rows, observed, learnable, consumed;
  float phi[OP_MAX_ROWS * ONLINE_RANK], keys[OP_MAX_ROWS * EXPERTS];
  int native3[OP_MAX_ROWS * 3], predicted3[OP_MAX_ROWS * 3];
} OnlineRecord;
typedef struct {
  uint64_t passes, rows, candidates, baseline_hits, corrected_hits, updates, frozen_passes;
  double forward_s, update_s;
} OnlineWindow;
static struct {
  int enabled, active, rows, count, labels_layer, labels_consumed, labels_valid;
  uint64_t epoch, label_epoch;
  int pair[LAYERS][2];
  OpPair *models;
  OpWork work;
  OnlineRecord *records;
  float gp[2 * OP_MAX_ROWS * EXPERTS], gl[OP_MAX_ROWS * EXPERTS];
  int label_ids[OP_MAX_ROWS * OP_K], output[2 * OP_MAX_ROWS * ONLINE_PREDICT];
  uint64_t passes, rows_seen, resets, forecasts, learning_skips, epoch_skips, invalid_native;
  uint64_t candidates, baseline_hits, corrected_hits;
  uint64_t completed_passes, computed_rows, epoch_rows, learned_rows, frozen_passes, frozen_skips;
  uint64_t freeze_rows, window_passes, window_index;
  int frozen;
  OnlineWindow window;
  double forward_s, update_s, reset_s;
  unsigned char identity[32];
  int bound, dirty;
  uint64_t state_loads, state_saves;
} online;

static uint64_t online_env_integer(const char *name, uint64_t limit) {
  const char *value = getenv(name);
  if (!value) return 0;
  char *end;
  errno = 0;
  unsigned long long n = strtoull(value, &end, 10);
  if (!*value || *value < '0' || *value > '9' || *end || errno || n > limit) {
    char error[160];
    snprintf(error, sizeof error, "%s must be an integer from 0 to %llu", name, (unsigned long long)limit);
    op_die(error);
  }
  return n;
}

static OnlineWindow online_snapshot(void) {
  return (OnlineWindow){online.completed_passes, online.computed_rows,  online.candidates,
                        online.baseline_hits,    online.corrected_hits, online.learned_rows,
                        online.frozen_passes,    online.forward_s,      online.update_s};
}

// Counts describe candidate accuracy. Cache readiness and newly useful reads
// remain separate native cache metrics. A window never splits a forward pass.
static void online_window_report(int partial) {
  OnlineWindow a = online_snapshot(), b = online.window;
  if (!online.window_passes || a.passes == b.passes) return;
  uint64_t guesses = a.candidates - b.candidates;
  fprintf(stderr,
          "{\"online_predictor_window\":{\"window\":%llu,\"learning_epoch\":%llu,\"partial\":%s,"
          "\"passes\":%llu,\"computed_rows\":%llu,\"learning_epoch_rows\":%llu,"
          "\"raw_top3_predictions\":%llu,\"raw_baseline_top3_hits\":%llu,\"raw_corrected_top3_hits\":%llu,"
          "\"raw_baseline_top3_precision\":%.9f,\"raw_corrected_top3_precision\":%.9f,"
          "\"forward_ms\":%.6f,\"update_ms\":%.6f,\"updates\":%llu,\"frozen_passes\":%llu,"
          "\"last_pass_frozen\":%s,\"next_pass_frozen\":%s,\"freeze_after_computed_rows\":%llu,"
          "\"invalid_native_ids\":%llu,\"epoch_skips\":%llu,"
          "\"scope\":\"computed rows include rejected verification rows; raw candidates include residents; not "
          "new-read usefulness\"}}\n",
          (unsigned long long)++online.window_index, (unsigned long long)online.resets, partial ? "true" : "false",
          (unsigned long long)(a.passes - b.passes), (unsigned long long)(a.rows - b.rows),
          (unsigned long long)online.epoch_rows, (unsigned long long)guesses,
          (unsigned long long)(a.baseline_hits - b.baseline_hits),
          (unsigned long long)(a.corrected_hits - b.corrected_hits),
          guesses ? (double)(a.baseline_hits - b.baseline_hits) / guesses : 0,
          guesses ? (double)(a.corrected_hits - b.corrected_hits) / guesses : 0, (a.forward_s - b.forward_s) * 1000,
          (a.update_s - b.update_s) * 1000, (unsigned long long)(a.updates - b.updates),
          (unsigned long long)(a.frozen_passes - b.frozen_passes), online.frozen ? "true" : "false",
          online.freeze_rows && online.epoch_rows >= online.freeze_rows ? "true" : "false",
          (unsigned long long)online.freeze_rows, (unsigned long long)online.invalid_native,
          (unsigned long long)online.epoch_skips);
  online.window = a;
}

void online_predictor_init(int rank, const float *const biases[LAYERS]) {
  if (!rank) return;
  if (rank != ONLINE_RANK || online.enabled) op_die("online predictor requires one rank64 initialization");
  online.freeze_rows = online_env_integer("MYDS4_ONLINE_FREEZE_ROWS", 1048576);
  online.window_passes = online_env_integer("MYDS4_ONLINE_WINDOW_PASSES", 4096);
  if (BLASSetThreading(BLAS_THREADING_SINGLE_THREADED)) op_die("cannot set single-thread Accelerate mode");
  memset(online.pair, 0xff, sizeof online.pair);
  online.models = op_allocate(ONLINE_PAIRS * sizeof *online.models);
  online.records = op_allocate(ONLINE_PAIRS * sizeof *online.records);
  online.work = op_create_work(EXPERTS, rank);
  for (int source = 0; source < LAYERS; source++)
    for (int h = 1; h <= 2 && source + h < LAYERS; h++) {
      int p = online.count++;
      online.pair[source][h - 1] = p;
      online.models[p] = op_create_pair(EXPERTS, rank, UINT64_C(921) + (uint64_t)p * UINT64_C(0x9e3779b97f4a7c15));
      if (!op_finite_array(biases[source + h], EXPERTS)) op_die("nonfinite target-router bias");
      memcpy(online.models[p].bias, biases[source + h], EXPERTS * 4);
    }
  if (online.count != ONLINE_PAIRS) op_die("unexpected online predictor head count");
  online.enabled = 1;
  online_predictor_reset();
  fprintf(stderr, "online predictor: rank64,77 heads, one CPU owner; shared engine learning, prefetch only\n");
}

void online_predictor_reset(void) {
  if (!online.enabled) return;
  double start = op_now();
  online.active = 0;
  online.frozen = 0;
  online.epoch_rows = 0;
  online.window_index = 0;
  online.window = online_snapshot();
  online.labels_consumed = 1;
  memset(online.records, 0, ONLINE_PAIRS * sizeof *online.records);
  for (int p = 0; p < online.count; p++) {
    OpPair *m = &online.models[p];
    uint64_t numerical = m->covariance_resets;
    memset(m->b, 0, (size_t)m->r * m->e * 4);
    op_covariance_reset(m);
    m->covariance_resets = numerical;
    m->active = 0;
  }
  online.resets++;
  online.dirty = 1;
  online.reset_s += op_now() - start;
}

void online_predictor_begin(int rows, int eligible) {
  if (!online.enabled) return;
  if (rows < 1 || rows > OP_MAX_ROWS || online.epoch == UINT64_MAX) op_die("invalid online predictor pass");
  online.epoch++;
  online.rows = rows;
  online.active = eligible;
  // A threshold-crossing pass learns every row. Only the following pass freezes.
  if (eligible) online.frozen = online.freeze_rows && online.epoch_rows >= online.freeze_rows;
  online.labels_consumed = 1;
  if (eligible) online.passes++, online.rows_seen += rows;
}

void online_predictor_end(void) { online.active = 0; }

// h2 has only three native IDs. Fill the unused suffix for the checked math
// API, but never admit that suffix: read_ahead uses only its first three.
static int native_six(const int *pr, int rows, int known, int *out) {
  for (int row = 0; row < rows; row++) {
    for (int k = 0; k < known; k++) {
      int e = pr[row * ONLINE_PREDICT + k];
      if (e < 0 || e >= EXPERTS) return 0;
      for (int j = 0; j < k; j++)
        if (out[row * OP_K + j] == e) return 0;
      out[row * OP_K + k] = e;
    }
    int used = known;
    for (int e = 0; used < OP_K && e < EXPERTS; e++) {
      int found = 0;
      for (int k = 0; k < used; k++) found |= out[row * OP_K + k] == e;
      if (!found) out[row * OP_K + used++] = e;
    }
  }
  return 1;
}

const int *online_predictor_forecasts(int layer, int rows, int horizons, const float *gp, const float *gl,
                                      const int *native_pr, const int *native_rt) {
  if (!online.enabled || !online.active) return native_pr;
  if (layer < 0 || layer >= LAYERS || rows != online.rows || horizons < 0 || horizons > 2) {
    online.epoch_skips++;
    return native_pr;
  }
  double start = op_now();
  int available = layer + 2 < LAYERS ? 2 : layer + 1 < LAYERS;
  int hops = horizons < available ? horizons : available;
  // Approximate multirow routing computes only h1. Never reuse an h2
  // source record or read the uncomputed suffix of GPU scratch.
  for (int h = hops; h < available; h++) online.records[online.pair[layer][h]].observed = 0;
  memcpy(online.gl, gl, (size_t)rows * EXPERTS * 4);
  if (hops) memcpy(online.gp, gp, (size_t)hops * rows * EXPERTS * 4);
  for (int row = 0; row < rows; row++) memcpy(online.label_ids + row * OP_K, native_rt + row * 16, OP_K * sizeof(int));
  online.labels_valid =
      op_finite_array(online.gl, (size_t)rows * EXPERTS) && op_valid_ids(online.label_ids, EXPERTS, rows);
  online.labels_layer = layer;
  online.label_epoch = online.epoch;
  online.labels_consumed = 0;
  int native[2][OP_MAX_ROWS * OP_K];
  for (int h = 0; h < hops; h++)
    if (!native_six(native_pr + h * rows * ONLINE_PREDICT, rows, h ? 3 : 6, native[h])) {
      for (int k = 0; k < hops; k++) online.records[online.pair[layer][k]].observed = 0;
      online.invalid_native++;
      online.forward_s += op_now() - start;
      return NULL; // Skip optional read-ahead rather than pass invalid IDs.
    }
  memset(online.output, 0, sizeof online.output);
  for (int h = 0; h < hops; h++) {
    int p = online.pair[layer][h], ids[OP_MAX_ROWS * OP_K];
    OpPair *model = &online.models[p];
    OnlineRecord *record = &online.records[p];
    int status =
        op_forward(model, &online.work, online.gp + (size_t)h * rows * EXPERTS, online.gl, rows, native[h], ids);
    record->epoch = online.epoch;
    record->rows = rows;
    record->observed = status >= 0;
    record->learnable = status == 0;
    record->consumed = 0;
    if (status < 0) {
      online.invalid_native++;
      online.forward_s += op_now() - start;
      return NULL;
    }
    if (record->learnable) {
      memcpy(record->phi, online.work.phi, (size_t)rows * ONLINE_RANK * 4);
      memcpy(record->keys, online.work.keys, (size_t)rows * EXPERTS * 4);
    }
    for (int row = 0; row < rows; row++) {
      memcpy(online.output + (h * rows + row) * ONLINE_PREDICT, ids + row * OP_K, OP_K * sizeof(int));
      memcpy(record->native3 + row * 3, native[h] + row * OP_K, 3 * sizeof(int));
      memcpy(record->predicted3 + row * 3, ids + row * OP_K, 3 * sizeof(int));
    }
    online.forecasts += rows;
  }
  online.forward_s += op_now() - start;
  return hops ? online.output : native_pr;
}

void online_predictor_update(int layer) {
  if (!online.enabled || !online.active) return;
  if (online.labels_consumed || online.label_epoch != online.epoch || online.labels_layer != layer) {
    online.epoch_skips++;
    return;
  }
  online.labels_consumed = 1;
  double start = op_now();
  for (int h = 1; h <= 2 && layer - h >= 0; h++) {
    int p = online.pair[layer - h][h - 1];
    OnlineRecord *record = &online.records[p];
    OpPair *model = &online.models[p];
    if (!record->observed) continue;
    if (record->consumed || record->epoch != online.epoch || record->rows != online.rows) {
      online.epoch_skips++;
      continue;
    }
    record->consumed = 1;
    if (!online.labels_valid) {
      online.learning_skips += online.rows;
      continue;
    }
    for (int row = 0; row < online.rows; row++) {
      for (int k = 0; k < 3; k++) {
        int baseline = 0, corrected = 0;
        for (int j = 0; j < OP_K; j++) {
          baseline |= record->native3[row * 3 + k] == online.label_ids[row * OP_K + j];
          corrected |= record->predicted3[row * 3 + k] == online.label_ids[row * OP_K + j];
        }
        online.candidates++, online.baseline_hits += baseline, online.corrected_hits += corrected;
      }
      if (online.frozen) {
        online.frozen_skips++;
        continue;
      }
      if (!record->learnable) {
        online.learning_skips++;
        continue;
      }
      float residual[EXPERTS];
      for (int e = 0; e < EXPERTS; e++)
        residual[e] = op_key(online.gl[row * EXPERTS + e], model->bias[e]) - record->keys[row * EXPERTS + e];
      uint64_t resets_before = model->covariance_resets;
      int changed = op_update(model, &online.work, record->phi + row * ONLINE_RANK, residual);
      if (!changed) online.learning_skips++;
      else online.learned_rows++;
      if (changed || model->covariance_resets != resets_before) online.dirty = 1;
    }
  }
  online.update_s += op_now() - start;
  if (layer == LAYERS - 1) {
    online.completed_passes++;
    online.computed_rows += online.rows;
    online.epoch_rows += online.rows;
    online.dirty = 1;
    online.frozen_passes += online.frozen;
    if (online.window_passes && online.completed_passes - online.window.passes >= online.window_passes)
      online_window_report(0);
  }
}

// Files use little-endian IEEE binary32 values. The explicit header never
// contains a C struct, pointers, padding or an in-flight source/label record.
enum {
  STATE_HEADER = 128,
  STATE_META = 24,
  STATE_DIGEST = 88,
  STATE_A = 768 * ONLINE_RANK,
  STATE_B = ONLINE_RANK * EXPERTS,
  STATE_P = ONLINE_RANK * ONLINE_RANK,
  STATE_FLOATS = STATE_A + STATE_B + STATE_P + EXPERTS,
  STATE_HEAD_BYTES = STATE_META + STATE_FLOATS * 4,
  STATE_BYTES = STATE_HEADER + ONLINE_PAIRS * STATE_HEAD_BYTES
};
static const unsigned char state_magic[8] = {'M', 'Y', 'D', 'S', '4', 'O', 'N', '1'};
_Static_assert(sizeof(float) == 4 && FLT_RADIX == 2 && FLT_MANT_DIG == 24, "state requires IEEE binary32");
_Static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__, "state float encoding requires little endian");

static int state_error(char *error, size_t cap, const char *format, ...) {
  if (error && cap) {
    va_list ap;
    va_start(ap, format);
    vsnprintf(error, cap, format, ap);
    va_end(ap);
  }
  return -1;
}

static void state_u32(unsigned char *p, uint32_t v) {
  for (int i = 0; i < 4; i++) p[i] = (unsigned char)(v >> (i * 8));
}
static void state_u64(unsigned char *p, uint64_t v) {
  for (int i = 0; i < 8; i++) p[i] = (unsigned char)(v >> (i * 8));
}
static uint32_t state_get32(const unsigned char *p) {
  uint32_t v = 0;
  for (int i = 0; i < 4; i++) v |= (uint32_t)p[i] << (i * 8);
  return v;
}
static uint64_t state_get64(const unsigned char *p) {
  uint64_t v = 0;
  for (int i = 0; i < 8; i++) v |= (uint64_t)p[i] << (i * 8);
  return v;
}
static void state_digest(unsigned char *data, unsigned char digest[32]) {
  unsigned char previous[32], result[32];
  memcpy(previous, data + STATE_DIGEST, 32);
  memset(data + STATE_DIGEST, 0, 32);
  CC_SHA256(data, STATE_BYTES, result);
  memcpy(data + STATE_DIGEST, previous, 32);
  memcpy(digest, result, 32);
}
static int state_read(int fd, void *data, size_t size) {
  unsigned char *p = data;
  while (size) {
    ssize_t n = read(fd, p, size);
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) return 0;
    p += n;
    size -= (size_t)n;
  }
  return 1;
}
static int state_write(int fd, const void *data, size_t size) {
  const unsigned char *p = data;
  while (size) {
    ssize_t n = write(fd, p, size);
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) return 0;
    p += n;
    size -= (size_t)n;
  }
  return 1;
}

// The model owner binds already-loaded router bytes and expert format metadata.
// File names and machine-specific paths are deliberately outside this identity.
void online_predictor_bind(const unsigned char identity[32]) {
  if (!online.enabled || online.active || !identity) op_die("cannot bind online predictor state during a pass");
  if (online.bound && memcmp(identity, online.identity, 32)) op_die("cannot change online predictor state identity");
  memcpy(online.identity, identity, 32);
  online.bound = 1;
}

int online_predictor_dirty(void) { return online.enabled && online.dirty; }

static int state_ready(const char *path, char *error, size_t cap) {
  if (!online.enabled || !online.bound)
    return state_error(error, cap, "predictor state requires an initialized, bound learner");
  if (online.active) return state_error(error, cap, "predictor state requires a completed forward pass");
  if (!path || !*path) return state_error(error, cap, "predictor state path is empty");
  return 1;
}

static int state_parent_writable(const char *path, char *error, size_t cap) {
  const char *slash = strrchr(path, '/');
  size_t length = slash ? (size_t)(slash - path) : 1;
  if (slash == path) length = 1;
  char *parent = op_allocate(length + 1);
  if (slash) memcpy(parent, path, length);
  else parent[0] = '.';
  struct stat st;
  int ok = !stat(parent, &st) && S_ISDIR(st.st_mode) && !access(parent, W_OK | X_OK);
  free(parent);
  return ok ? 1 : state_error(error, cap, "predictor state parent directory must exist and be writable");
}

static int state_header(const unsigned char *data, char *error, size_t cap) {
  if (memcmp(data, state_magic, 8) || state_get32(data + 8) != 1 || state_get32(data + 12) != STATE_HEADER ||
      state_get64(data + 16) != STATE_BYTES - STATE_HEADER || state_get32(data + 24) != ONLINE_PAIRS ||
      state_get32(data + 28) != EXPERTS || state_get32(data + 32) != 768 || state_get32(data + 36) != ONLINE_RANK ||
      state_get32(data + 40) != 1 || state_get32(data + 44) || state_get64(data + 120))
    return state_error(error, cap, "unsupported predictor state format or dimensions");
  if (memcmp(data + 56, online.identity, 32))
    return state_error(error, cap, "predictor state belongs to different routers or expert formats");
  if (state_get64(data + 48) > UINT64_MAX - OP_MAX_ROWS)
    return state_error(error, cap, "predictor state row count is out of range");
  return 1;
}

static int state_covariance(const float *p) {
  // Match the live update guards. FP32 rounding can make a long-running
  // covariance fail strict Cholesky even while these invariants hold; saving
  // must neither discard that running state nor silently change its next fit.
  for (int i = 0; i < ONLINE_RANK; i++) {
    if (p[i * ONLINE_RANK + i] <= 0) return 0;
    for (int j = 0; j < ONLINE_RANK; j++)
      if (!isfinite(p[i * ONLINE_RANK + j]) || fabsf(p[i * ONLINE_RANK + j]) > OP_COV_LIMIT ||
          p[i * ONLINE_RANK + j] != p[j * ONLINE_RANK + i])
        return 0;
  }
  return 1;
}

static int state_validate(unsigned char *data, char *error, size_t cap) {
  if (state_header(data, error, cap) < 0) return -1;
  unsigned char digest[32];
  state_digest(data, digest);
  if (memcmp(digest, data + STATE_DIGEST, 32)) return state_error(error, cap, "predictor state checksum mismatch");
  int index = 0;
  for (int source = 0; source < LAYERS; source++)
    for (int h = 1; h <= 2 && source + h < LAYERS; h++, index++) {
      const unsigned char *head = data + STATE_HEADER + index * STATE_HEAD_BYTES;
      const float *a = (const float *)(head + STATE_META), *b = a + STATE_A, *p = b + STATE_B, *bias = p + STATE_P;
      int active = state_get32(head + 8), nonzero = 0;
      if (state_get32(head) != (uint32_t)source || state_get32(head + 4) != (uint32_t)(source + h) || active > 1 ||
          state_get32(head + 12) || state_get64(head + 16) > UINT64_MAX / ONLINE_PAIRS)
        return state_error(error, cap, "invalid predictor head metadata at head %d", index);
      if (!op_finite_array(a, STATE_FLOATS) || memcmp(bias, online.models[index].bias, EXPERTS * 4))
        return state_error(error, cap, "nonfinite parameters or incompatible bias at predictor head %d", index);
      for (int k = 0; k < STATE_B; k++) nonzero |= b[k] != 0;
      if (active != nonzero || !state_covariance(p))
        return state_error(error, cap, "invalid active flag or covariance at predictor head %d", index);
    }
  return 1;
}

static unsigned char *state_encode(void) {
  unsigned char *data = op_allocate(STATE_BYTES);
  memcpy(data, state_magic, 8);
  state_u32(data + 8, 1);
  state_u32(data + 12, STATE_HEADER);
  state_u64(data + 16, STATE_BYTES - STATE_HEADER);
  state_u32(data + 24, ONLINE_PAIRS);
  state_u32(data + 28, EXPERTS);
  state_u32(data + 32, 768);
  state_u32(data + 36, ONLINE_RANK);
  state_u32(data + 40, 1); // Feature normalization, selection-key function and RLS algorithm version.
  state_u64(data + 48, online.epoch_rows);
  memcpy(data + 56, online.identity, 32);
  int index = 0;
  for (int source = 0; source < LAYERS; source++)
    for (int h = 1; h <= 2 && source + h < LAYERS; h++, index++) {
      unsigned char *head = data + STATE_HEADER + index * STATE_HEAD_BYTES;
      OpPair *m = &online.models[index];
      state_u32(head, source);
      state_u32(head + 4, source + h);
      state_u32(head + 8, m->active);
      state_u64(head + 16, m->updates);
      unsigned char *values = head + STATE_META;
      memcpy(values, m->a, STATE_A * 4);
      memcpy(values + STATE_A * 4, m->b, STATE_B * 4);
      memcpy(values + (STATE_A + STATE_B) * 4, m->p, STATE_P * 4);
      memcpy(values + (STATE_A + STATE_B + STATE_P) * 4, m->bias, EXPERTS * 4);
    }
  state_digest(data, data + STATE_DIGEST);
  return data;
}

// Return 0 only for an absent file. Every other failure leaves all live
// parameters, learning progress, pending records and dirty state unchanged.
int online_predictor_load(const char *path, char *error, size_t cap) {
  if (state_ready(path, error, cap) < 0) return -1;
  int fd = open(path, O_RDONLY | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK);
  if (fd < 0) {
    if (errno == ENOENT) return state_parent_writable(path, error, cap) < 0 ? -1 : 0;
    return state_error(error, cap, "open predictor state: %s", strerror(errno));
  }
  struct stat st;
  if (fstat(fd, &st) || !S_ISREG(st.st_mode) || st.st_size != STATE_BYTES) {
    close(fd);
    return state_error(error, cap, "predictor state must be a regular file of %d bytes", STATE_BYTES);
  }
  unsigned char *data = op_allocate(STATE_BYTES);
  int ok = state_read(fd, data, STATE_BYTES);
  close(fd);
  if (!ok) {
    free(data);
    return state_error(error, cap, "cannot read complete predictor state");
  }
  if (state_validate(data, error, cap) < 0) {
    free(data);
    return -1;
  }
  for (int index = 0; index < ONLINE_PAIRS; index++) {
    const unsigned char *head = data + STATE_HEADER + index * STATE_HEAD_BYTES;
    const unsigned char *values = head + STATE_META;
    OpPair *m = &online.models[index];
    memcpy(m->a, values, STATE_A * 4);
    memcpy(m->b, values + STATE_A * 4, STATE_B * 4);
    memcpy(m->p, values + (STATE_A + STATE_B) * 4, STATE_P * 4);
    m->active = state_get32(head + 8);
    m->updates = state_get64(head + 16);
  }
  online.epoch_rows = state_get64(data + 48);
  online.frozen = online.freeze_rows && online.epoch_rows >= online.freeze_rows;
  online.labels_consumed = 1;
  memset(online.records, 0, ONLINE_PAIRS * sizeof *online.records);
  online.window = online_snapshot();
  online.window_index = 0;
  online.dirty = 0;
  online.state_loads++;
  free(data);
  return 1;
}

static int state_target(const char *path, struct stat *st, char *error, size_t cap) {
  if (lstat(path, st))
    return errno == ENOENT ? 0 : state_error(error, cap, "inspect predictor state destination: %s", strerror(errno));
  if (!S_ISREG(st->st_mode))
    return state_error(error, cap, "predictor state destination must be a regular file, not a symlink");
  int fd = open(path, O_RDONLY | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK);
  unsigned char header[STATE_HEADER];
  struct stat opened;
  int ok = fd >= 0 && !fstat(fd, &opened) && opened.st_dev == st->st_dev && opened.st_ino == st->st_ino &&
           opened.st_size == STATE_BYTES && state_read(fd, header, sizeof header);
  if (fd >= 0) close(fd);
  if (!ok) return state_error(error, cap, "refusing to replace an unrelated or changed predictor state destination");
  return state_header(header, error, cap);
}

int online_predictor_save(const char *path, char *error, size_t cap) {
  if (state_ready(path, error, cap) < 0) return -1;
  struct stat before, current;
  int existed = state_target(path, &before, error, cap);
  if (existed < 0) return -1;
  unsigned char *data = state_encode();
  if (state_validate(data, error, cap) < 0) {
    free(data);
    return -1;
  }
  size_t n = strlen(path);
  char *temporary = op_allocate(n + sizeof ".tmp.XXXXXX");
  snprintf(temporary, n + sizeof ".tmp.XXXXXX", "%s.tmp.XXXXXX", path);
  int fd = mkstemp(temporary), ok = fd >= 0;
  if (!ok) state_error(error, cap, "create predictor state temporary file: %s", strerror(errno));
  if (ok && (!state_write(fd, data, STATE_BYTES) || fsync(fd))) {
    ok = 0;
    state_error(error, cap, "write predictor state: %s", strerror(errno));
  }
  if (fd >= 0 && close(fd) && ok) {
    ok = 0;
    state_error(error, cap, "close predictor state: %s", strerror(errno));
  }
  if (ok) {
    int still_exists = lstat(path, &current) == 0;
    if ((existed && (!still_exists || current.st_dev != before.st_dev || current.st_ino != before.st_ino ||
                     current.st_size != before.st_size || current.st_mtimespec.tv_sec != before.st_mtimespec.tv_sec ||
                     current.st_mtimespec.tv_nsec != before.st_mtimespec.tv_nsec)) ||
        (!existed && (still_exists || errno != ENOENT))) {
      ok = 0;
      state_error(error, cap, "predictor state destination changed while saving");
    }
  }
  if (ok && rename(temporary, path)) {
    ok = 0;
    state_error(error, cap, "publish predictor state: %s", strerror(errno));
  }
  if (!ok && fd >= 0) unlink(temporary);
  free(temporary);
  free(data);
  if (!ok) return -1;
  online.dirty = 0;
  online.state_saves++;
  return 1;
}

void online_predictor_report(FILE *out) {
  if (!online.enabled) return;
  online_window_report(1);
  uint64_t updates = 0, resets = 0, rejected = 0, fallback = 0;
  for (int p = 0; p < online.count; p++) {
    updates += online.models[p].updates;
    resets += online.models[p].covariance_resets;
    rejected += online.models[p].rejected_updates;
    fallback += online.models[p].fallback_rows;
  }
  size_t state =
      ONLINE_PAIRS * (768 * ONLINE_RANK + ONLINE_RANK * EXPERTS + ONLINE_RANK * ONLINE_RANK + EXPERTS) * sizeof(float);
  fprintf(out,
          "{\"online_predictor\":{\"rank\":64,\"pairs\":77,\"state_scope\":\"engine\",\"batched_prefill_training\":"
          "false,\"passes\":%llu,\"verification_rows\":%llu,\"forecast_rows\":%llu,\"updates\":%llu,\"learning_skips\":"
          "%llu,\"epoch_skips\":%llu,\"invalid_native_ids\":%llu,\"fallback_rows\":%llu,\"covariance_resets\":%llu,"
          "\"rejected_updates\":%llu,\"learner_resets\":%llu,\"forward_ms\":%.6f,\"update_ms\":%.6f,\"reset_ms\":%.6f,"
          "\"state_bytes\":%zu,\"record_bytes\":%zu,\"raw_top3_predictions\":%llu,\"raw_baseline_top3_hits\":%llu,"
          "\"raw_corrected_top3_hits\":%llu,\"raw_baseline_top3_precision\":%.9f,\"raw_corrected_top3_precision\":%.9f,"
          "\"basis\":\"C_xorshift64star_BoxMuller_not_PyTorch\",\"scope\":\"prefetch only; computed rows include "
          "rejected verification rows; raw per-row candidate precision includes resident experts and is not new-read "
          "usefulness; small-pass learning; native RT/PR diagnostics unchanged\",",
          (unsigned long long)online.passes, (unsigned long long)online.rows_seen, (unsigned long long)online.forecasts,
          (unsigned long long)updates, (unsigned long long)online.learning_skips,
          (unsigned long long)online.epoch_skips, (unsigned long long)online.invalid_native,
          (unsigned long long)fallback, (unsigned long long)resets, (unsigned long long)rejected,
          (unsigned long long)online.resets, online.forward_s * 1000, online.update_s * 1000, online.reset_s * 1000,
          state, ONLINE_PAIRS * sizeof(OnlineRecord), (unsigned long long)online.candidates,
          (unsigned long long)online.baseline_hits, (unsigned long long)online.corrected_hits,
          online.candidates ? (double)online.baseline_hits / online.candidates : 0,
          online.candidates ? (double)online.corrected_hits / online.candidates : 0);
  fprintf(out,
          "\"state_loads\":%llu,\"state_saves\":%llu,\"state_dirty\":%s,\"state_file_bytes\":%d,"
          "\"freeze_after_computed_rows\":%llu,\"window_passes\":%llu,\"completed_passes\":%llu,"
          "\"computed_rows_completed\":%llu,\"learning_epoch_rows\":%llu,\"frozen\":%s,\"frozen_update_rows_skipped\":%"
          "llu}}\n",
          (unsigned long long)online.state_loads, (unsigned long long)online.state_saves,
          online.dirty ? "true" : "false", STATE_BYTES, (unsigned long long)online.freeze_rows,
          (unsigned long long)online.window_passes, (unsigned long long)online.completed_passes,
          (unsigned long long)online.computed_rows, (unsigned long long)online.epoch_rows,
          online.frozen ? "true" : "false", (unsigned long long)online.frozen_skips);
}
