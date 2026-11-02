// Diagnostic records for the optional MYDS4_PREDICT_PROBE build. No file is
// opened: the caller supplies MYDS4_PROBE_FD, normally a pipe. Writes are
// synchronous and may stall inference, so their timing is not a benchmark.
// Kinds: 1 decoded/verification target, 2 prefill target, 3 DSpark stage,
// 4 seed embedding, 5 seed and proposals, 6 accepted prefix, 7 cache metadata,
// 8 same-layer residual prediction, 9 four-stream state and FFN mixes,
// 10 final normalized draft-head input, 11 native future expert predictions.
// Kind 3 appends the three native selected expert IDs for each stage row.
// Kind 7 features are aligned SSD bytes [LAYERS][EXPERTS]; its three int32
// values are initial capacity, uniform slot bytes, and maximum capacity.
// Kind 8 carries residual features and six native predicted expert IDs per
// row, with no logits. It is emitted only when full_ahead() runs.
// Kind 9 requires MYDS4_PROBE_STREAMS (nonempty and not "0"). Its features
// are BF16 uint16 bits, not FP32: [m,4,5120], with header rows=4*m. The
// following FP32 payload holds the raw [m,24] mixes, counted in the header
// as [4*m,6]; it does not mean six coefficients per stream. Batches above
// 128 rows are omitted, as are main layers outside 0,1,4,8,12,20,28,35.
// Kind 9 layers 40..42 are stage inputs. Layer 43 is final DHS and DMF for
// all computed rows: five by default, or only requested rows for stage 42
// and final layer 43 with --draft-trim. Kind 10, also layer 43, contains FP32
// head inputs for only the requested draft rows. These final records require
// MYDS4_PROBE_STREAMS and follow the existing synchronization AFTER the head;
// their timestamps do not establish availability before head execution.
// Kind 11 follows a kind 1 target record for full-ahead decode/verification.
// It has no float payload: int32 IDs are [valid hops, rows, 6] from PR, in
// h1/h2 order. h1 has six IDs; h2 normally has three and a -1 suffix. Layer
// 38 has only h1, and layer 39 emits none. No prediction kernel is added.
#include <errno.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "myds4.h"

typedef struct {
  uint32_t magic, kind;
  int32_t pos, layer, rows, feature_dim, logit_dim, token_count;
  double timestamp;
} ProbeHeader;

// Native fields on the supported little-endian Apple silicon host, without
// padding. Payloads follow in this order: row-major features (FP32 except
// BF16 for kind 9), FP32 logits or mixes, then int32 values (token/expert IDs,
// or cache metadata for kind 7). A zero dimension omits that payload.
static_assert(sizeof(ProbeHeader) == 40, "prediction probe header must be 40 bytes");
static_assert(offsetof(ProbeHeader, timestamp) == 32, "prediction probe timestamp offset");
static_assert(sizeof(float) == 4 && sizeof(int) == 4, "prediction probe scalar sizes");

static int probe_fd = -2;

int predict_probe_enabled(void) {
  if (probe_fd == -2) {
    const char *value = getenv("MYDS4_PROBE_FD");
    probe_fd = -1;
    if (value) {
      char *end;
      errno = 0;
      long fd = strtol(value, &end, 10);
      if (errno || end == value || *end || fd < 0 || fd > INT_MAX)
        die("MYDS4_PROBE_FD must name an open nonnegative file descriptor");
      probe_fd = (int)fd;
    }
  }
  return probe_fd >= 0;
}

static void emit_bytes(const void *data, size_t size) {
  const uint8_t *p = data;
  while (size) {
    ssize_t n = write(probe_fd, p, size);
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) die("prediction probe write: %s", n ? strerror(errno) : "zero-byte write");
    p += n;
    size -= (size_t)n;
  }
}

static void emit_header_payload(uint32_t kind, int pos, int layer, int rows, int feature_dim, const float *features,
                                int logit_dim, const float *logits, int token_count) {
  ProbeHeader header = {0x50524231u, kind, pos, layer, rows, feature_dim, logit_dim, token_count, now()};
  emit_bytes(&header, sizeof header);
  emit_bytes(features, (size_t)rows * (size_t)feature_dim * sizeof(float));
  emit_bytes(logits, (size_t)rows * (size_t)logit_dim * sizeof(float));
}

void predict_probe_emit(uint32_t kind, int pos, int layer, int rows, int feature_dim, const float *features,
                        int logit_dim, const float *logits, int token_count, const int *tokens) {
  if (!predict_probe_enabled()) return;
  emit_header_payload(kind, pos, layer, rows, feature_dim, features, logit_dim, logits, token_count);
  emit_bytes(tokens, (size_t)token_count * sizeof(int));
}

// Target records append the native top-k IDs, before substitution, after the
// input token IDs. The payload is tokens[rows], selected[rows][TOPK].
void predict_probe_target(uint32_t kind, int pos, int layer, int rows, int feature_dim, const float *features,
                          int logit_dim, const float *logits, const int *tokens, const int *routes, int route_stride) {
  if (!predict_probe_enabled()) return;
  emit_header_payload(kind, pos, layer, rows, feature_dim, features, logit_dim, logits, rows * (1 + TOPK));
  emit_bytes(tokens, (size_t)rows * sizeof(int));
  for (int r = 0; r < rows; r++) emit_bytes(routes + (size_t)r * route_stride, TOPK * sizeof(int));
}

int predict_probe_streams_enabled(void) {
  static int enabled = -1;
  if (!predict_probe_enabled()) return 0;
  if (enabled < 0) {
    const char *value = getenv("MYDS4_PROBE_STREAMS");
    enabled = value && *value && strcmp(value, "0");
  }
  return enabled;
}

void predict_probe_streams(int pos, int layer, int rows, const float *streams, const float *mixes) {
  if (!predict_probe_streams_enabled() || rows < 1 || rows > 128) return;
  switch (layer) {
  case 0:
  case 1:
  case 4:
  case 8:
  case 12:
  case 20:
  case 28:
  case 35:
    break;
  default:
    if (layer < LAYERS || layer > CACHE_LAYERS) return;
  }
  size_t count = (size_t)rows * HC * DIM;
  for (size_t i = 0; i < count; i++) {
    uint32_t bits;
    memcpy(&bits, streams + i, sizeof bits);
    if (bits & 0xFFFFu) die("prediction probe stream is not BF16 at position %d, layer %d, element %zu", pos, layer, i);
  }
  ProbeHeader header = {0x50524231u, 9, pos, layer, HC * rows, DIM, MIX / HC, 0, now()};
  emit_bytes(&header, sizeof header);
  uint16_t packed[DIM];
  for (int row = 0; row < HC * rows; row++) {
    for (int i = 0; i < DIM; i++) {
      uint32_t bits;
      memcpy(&bits, streams + (size_t)row * DIM + i, sizeof bits);
      packed[i] = (uint16_t)(bits >> 16);
    }
    emit_bytes(packed, sizeof packed);
  }
  emit_bytes(mixes, (size_t)rows * MIX * sizeof(float));
}
