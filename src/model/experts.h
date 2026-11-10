// GPU addresses and scales for rows selecting the same expert.
#pragma once
#include "linear.h"

typedef struct {
  uint64_t weight, scales;
  float scale, input_scale;
} ExpertMatrix;
typedef struct {
  ExpertMatrix gate, up, down;
  uint32_t begin, count;
} ExpertGroup;
typedef struct { uint32_t group, first; } ExpertTile;
static_assert(sizeof(ExpertGroup) == 80);

static inline ExpertMatrix expert_matrix(const Linear *m) {
  return (ExpertMatrix){m->w.gpu, m->s.gpu, m->scale * m->input_scale, m->input_scale};
}

typedef struct {
  Linear gate, up, down;
  WeightSet *weights;
} Expert;
typedef struct {
  uint32_t row;
  float weight;
} ExpertRow;
typedef enum { EXPERT_SWIGLU, EXPERT_RELU2 } ExpertActivation;
typedef enum { EXPERT_SUM_BF16, EXPERT_SUM_FP32 } ExpertAccumulation;
typedef struct {
  int hidden, intermediate, experts, chosen, rows, active, group_count, group_rows, matrix_tiles;
  ExpertActivation activation;
  Buf plan, entries, order, groups, row_groups, tiles, input, second, hidden_values, quantized, output;
  uint32_t *active_ids;
  const Expert *held;
  int held_count;
  int input_kernel, up_kernel, hidden_kernel, down_kernel, sum_kernel;
  int up_single, down_single;
  int up_matrix, down_matrix;
  int plan_kernel;
} ExpertWork;

void experts_init(ExpertWork *work, int hidden, int intermediate, int experts, int chosen,
                  ExpertActivation activation, ExpertAccumulation accumulation);
uint64_t experts_workspace(ExpertWork *work, int rows, int allocate);
uint64_t experts_bytes(const ExpertWork *work);
// Routing and the preceding expert work have completed. Release the previous
// selection, then pin every new selection before loading any miss.
int experts_acquire(ExpertWork *work, const Expert *experts, const uint32_t *selected,
                     const float *weights, int rows, int prompt, CacheStats *stats);
// Call after the queued expert work completes. Resident planning retains this
// selection until a later acquisition or the end of the model pass.
void experts_release(ExpertWork *work);
// Encode NVFP4 groups together. Prompt groups use matrix tiles; decoding
// retains scalar reductions. Other formats write to experts_output separately.
void experts_encode(const ExpertWork *work, const Expert *experts, const Buf *input);
void experts_sum(const ExpertWork *work, const Buf *output);
// Build a table only when every expert is resident; unsupported formats use
// the ordinary path. Resident encoding takes prompt rows or at most eight rows.
int experts_table(const ExpertWork *work, const Expert *experts, const Buf *table, size_t offset);
void experts_resident(ExpertWork *work, const Buf *table, size_t offset, const Buf *selected,
                       const Buf *coefficients, const Buf *input, const Buf *unique, size_t unique_offset,
                       int rows, int prompt);
// A layer's selections in a buffer that holds every layer's for the pass, so
// that they outlast the layer: rows × chosen expert ids at the layer's place.
static inline Buf experts_selection(const Buf *selected, int layer, int rows, int chosen) {
  size_t offset = (size_t)layer * rows * chosen * 4;
  Buf b = *selected;
  b.ptr += offset, b.gpu += offset, b.off += offset, b.size -= offset;
  return b;
}
// After a pass that reached a layer's experts through its table, marks the
// experts its rows chose used, as an acquisition does: eviction takes the
// experts used longest ago first.
void experts_mark_used(const ExpertWork *work, const Expert *experts, const uint32_t *selected, int rows);
static inline int experts_count(const ExpertWork *work, int expert) {
  return ((const uint32_t *)work->plan.ptr)[expert];
}
static inline int experts_grouped(const ExpertWork *work, int expert) {
  return ((const uint32_t *)work->plan.ptr)[2 * work->experts + 1 + expert];
}
size_t experts_entry_offset(const ExpertWork *work, int expert);
Buf experts_output(const ExpertWork *work, int expert);
