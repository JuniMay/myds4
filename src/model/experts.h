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
  int hidden, intermediate, experts, chosen, rows, active, group_count, group_rows;
  ExpertActivation activation;
  Buf plan, entries, order, groups, input, second, hidden_values, quantized, output;
  uint32_t *active_ids;
  int input_kernel, up_kernel, hidden_kernel, down_kernel, sum_kernel;
} ExpertWork;

void experts_init(ExpertWork *work, int hidden, int intermediate, int experts, int chosen,
                  ExpertActivation activation, ExpertAccumulation accumulation);
uint64_t experts_workspace(ExpertWork *work, int rows, int allocate);
uint64_t experts_bytes(const ExpertWork *work);
// Routing has completed. Pin every selected expert before loading any miss.
int experts_acquire(ExpertWork *work, const Expert *experts, const uint32_t *selected,
                     const float *weights, int rows, CacheStats *stats);
void experts_release(const ExpertWork *work, const Expert *experts);
// Encode groups of up to eight rows. Larger or other-format groups use the
// model's matrix path and write to experts_output before experts_sum.
void experts_encode(const ExpertWork *work, const Expert *experts, const Buf *input);
void experts_sum(const ExpertWork *work, const Buf *output);
static inline int experts_count(const ExpertWork *work, int expert) {
  return ((const uint32_t *)work->plan.ptr)[expert];
}
static inline int experts_grouped(const ExpertWork *work, int expert) {
  return ((const uint32_t *)work->plan.ptr)[2 * work->experts + 1 + expert];
}
size_t experts_entry_offset(const ExpertWork *work, int expert);
Buf experts_output(const ExpertWork *work, int expert);
