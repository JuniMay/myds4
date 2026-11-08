#include "experts.h"
#include <string.h>

static uint32_t *counts(const ExpertWork *w) { return (uint32_t *)w->plan.ptr; }
static uint32_t *offsets(const ExpertWork *w) { return counts(w) + w->experts; }
static uint32_t *grouped(const ExpertWork *w) { return offsets(w) + w->experts + 1; }

void experts_init(ExpertWork *w, int hidden, int intermediate, int experts, int chosen,
                  ExpertActivation activation, ExpertAccumulation accumulation) {
  w->hidden = hidden;
  w->intermediate = intermediate;
  w->experts = experts;
  w->chosen = chosen;
  w->activation = activation;
  w->input_kernel = gpu_kernel(activation == EXPERT_SWIGLU ? "expert_input" : "expert_input_relu2");
  w->up_kernel = gpu_kernel(activation == EXPERT_SWIGLU ? "expert_up" : "expert_up_relu2");
  w->hidden_kernel = gpu_kernel("expert_hidden");
  w->down_kernel = gpu_kernel("expert_down");
  w->sum_kernel = gpu_kernel(accumulation == EXPERT_SUM_BF16 ? "expert_sum" : "expert_sum_fp32");
}
uint64_t experts_workspace(ExpertWork *w, int rows, int allocate) {
  uint64_t bytes = 0;
  size_t pairs = (size_t)rows * w->chosen;
#define B(field, amount) \
  do { \
    size_t size = (amount); \
    bytes += gpu_allocation_size(size); \
    if (allocate && w->field.size != size) { \
      gpu_free(&w->field); \
      w->field = gpu_alloc(size); \
    } \
  } while (0)
  B(plan, (4 * w->experts + 1) * 4);
  B(entries, pairs * sizeof(ExpertRow));
  B(order, pairs * 4);
  B(groups, (pairs < (size_t)w->experts ? pairs : (size_t)w->experts) * sizeof(ExpertGroup));
  B(input, pairs * w->hidden * 4);
  B(second, w->activation == EXPERT_SWIGLU ? pairs * w->hidden * 4 : 0);
  B(hidden_values, pairs * w->intermediate * 4);
  B(quantized, pairs * w->intermediate * 4);
  B(output, pairs * w->hidden * 4);
#undef B
  if (allocate) w->active_ids = grouped(w) + w->experts;
  return bytes;
}
uint64_t experts_bytes(const ExpertWork *w) {
  const Buf *buffers[] = {&w->plan, &w->entries, &w->order, &w->groups, &w->input, &w->second,
                          &w->hidden_values, &w->quantized, &w->output};
  uint64_t bytes = 0;
  for (size_t i = 0; i < sizeof buffers / sizeof *buffers; i++) bytes += buffers[i]->allocated;
  return bytes;
}
int experts_acquire(ExpertWork *w, const Expert *experts, const uint32_t *selected,
                     const float *weights, int rows, CacheStats *stats) {
  uint32_t *count = counts(w), *start = offsets(w), *small = grouped(w);
  uint32_t next[w->experts], row_count[rows];
  memset(w->plan.ptr, 0, w->plan.size);
  memset(row_count, 0, sizeof row_count);
  w->rows = rows;
  w->active = w->group_count = w->group_rows = 0;
  for (int i = 0; i < rows * w->chosen; i++) count[selected[i]]++;
  for (int e = 0; e < w->experts; e++) {
    start[e + 1] = start[e] + count[e];
    next[e] = start[e];
    if (count[e]) w->active_ids[w->active++] = e;
  }
  ExpertRow *entries = (ExpertRow *)w->entries.ptr;
  uint32_t *order = (uint32_t *)w->order.ptr;
  for (int r = 0; r < rows; r++) for (int k = 0; k < w->chosen; k++) {
    int at = r * w->chosen + k;
    entries[next[selected[at]]++] = (ExpertRow){r, weights[at]};
  }
  for (int i = 0; i < rows * w->chosen; i++) {
    uint32_t r = entries[i].row;
    order[r * w->chosen + row_count[r]++] = i;
  }
  for (int i = 0; i < w->active; i++)
    memory_pin(weights_slot(experts[w->active_ids[i]].weights), 1);
  ExpertGroup *groups = (ExpertGroup *)w->groups.ptr;
  for (int i = 0; i < w->active; i++) {
    int e = w->active_ids[i];
    const Expert *x = experts + e;
    if (memory_ready(weights_slot(x->weights))) stats->row_own += count[e];
    else stats->row_ssd += count[e];
    if (!weights_load(x->weights)) { experts_release(w, experts); return 0; }
    stats->row_unique++;
    small[e] = count[e] <= 8 && x->up.kind == 'n' && x->down.kind == 'n' &&
                (w->activation == EXPERT_RELU2 || x->gate.kind == 'n');
    if (small[e]) {
      groups[w->group_count++] = (ExpertGroup){expert_matrix(&x->gate), expert_matrix(&x->up),
                                               expert_matrix(&x->down), start[e], count[e]};
      if ((int)count[e] > w->group_rows) w->group_rows = count[e];
    }
  }
  return 1;
}
void experts_release(const ExpertWork *w, const Expert *experts) {
  for (int i = 0; i < w->active; i++)
    memory_pin(weights_slot(experts[w->active_ids[i]].weights), 0);
}
void experts_encode(const ExpertWork *w, const Expert *experts, const Buf *input) {
  if (!w->group_count) return;
  for (int i = 0; i < w->active; i++) {
    int e = w->active_ids[i];
    if (!grouped(w)[e]) continue;
    weights_use(experts[e].weights);
  }
  uint32_t dims[2] = {w->hidden, w->intermediate};
  gpu_kern(w->input_kernel);
  gpu_buf(0, &w->groups, 0); gpu_buf(1, &w->entries, 0); gpu_buf(2, input, 0);
  gpu_buf(3, &w->input, 0); gpu_buf(4, w->second.obj ? &w->second : &w->input, 0);
  gpu_bytes(5, dims, sizeof dims);
  gpu_go((w->hidden + 255) / 256, w->group_count, w->group_rows, 256, 1, 1);
  gpu_kern(w->up_kernel);
  gpu_buf(0, &w->groups, 0); gpu_buf(1, &w->input, 0);
  gpu_buf(2, w->second.obj ? &w->second : &w->input, 0); gpu_buf(3, &w->hidden_values, 0);
  gpu_bytes(4, dims, sizeof dims);
  gpu_go((w->intermediate + 7) / 8, w->group_count, 1, 256, 1, 1);
  gpu_kern(w->hidden_kernel);
  gpu_buf(0, &w->groups, 0); gpu_buf(1, &w->hidden_values, 0); gpu_buf(2, &w->quantized, 0);
  gpu_bytes(3, dims, sizeof dims);
  gpu_go((w->intermediate + 255) / 256, w->group_count, w->group_rows, 256, 1, 1);
  gpu_kern(w->down_kernel);
  gpu_buf(0, &w->groups, 0); gpu_buf(1, &w->quantized, 0); gpu_buf(2, &w->output, 0);
  gpu_bytes(3, dims, sizeof dims);
  gpu_go((w->hidden + 7) / 8, w->group_count, 1, 256, 1, 1);
}
void experts_sum(const ExpertWork *w, const Buf *output) {
  uint32_t dims[3] = {w->hidden, w->rows, w->chosen};
  gpu_kern(w->sum_kernel);
  gpu_buf(0, &w->output, 0); gpu_buf(1, &w->entries, 0); gpu_buf(2, &w->order, 0);
  gpu_buf(3, output, 0); gpu_bytes(4, dims, sizeof dims);
  gpu_go((w->rows * w->hidden + 255) / 256, 1, 1, 256, 1, 1);
}
size_t experts_entry_offset(const ExpertWork *w, int expert) {
  return offsets(w)[expert] * sizeof(ExpertRow);
}
Buf experts_output(const ExpertWork *w, int expert) {
  size_t offset = (size_t)offsets(w)[expert] * w->hidden * 4;
  Buf out = w->output;
  out.ptr += offset; out.gpu += offset; out.off += offset; out.size -= offset;
  return out;
}
