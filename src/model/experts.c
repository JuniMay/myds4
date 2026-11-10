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
  w->up_single = gpu_kernel(activation == EXPERT_SWIGLU ? "expert_up_single" : "expert_up_relu2_single");
  w->hidden_kernel = gpu_kernel("expert_hidden");
  w->down_kernel = gpu_kernel("expert_down");
  w->down_single = gpu_kernel("expert_down_single");
  w->up_matrix = gpu_kernel(activation == EXPERT_SWIGLU ? "expert_up_matrix" : "expert_up_relu2_matrix");
  w->down_matrix = gpu_kernel("expert_down_matrix");
  w->plan_kernel = gpu_kernel("expert_plan");
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
  B(row_groups, pairs * 4);
  B(tiles, ((pairs + 15) / 16 + w->experts) * sizeof(ExpertTile));
  B(input, (pairs + 15) * w->hidden * 2);
  B(second, w->activation == EXPERT_SWIGLU ? (pairs + 15) * w->hidden * 2 : 0);
  B(hidden_values, pairs * w->intermediate * 4);
  B(quantized, (pairs + 15) * w->intermediate * 2);
  B(output, pairs * w->hidden * 4);
#undef B
  if (allocate) w->active_ids = grouped(w) + w->experts;
  return bytes;
}
uint64_t experts_bytes(const ExpertWork *w) {
  const Buf *buffers[] = {&w->plan, &w->entries, &w->order, &w->groups, &w->row_groups, &w->tiles, &w->input, &w->second,
                          &w->hidden_values, &w->quantized, &w->output};
  uint64_t bytes = 0;
  for (size_t i = 0; i < sizeof buffers / sizeof *buffers; i++) bytes += buffers[i]->allocated;
  return bytes;
}
int experts_acquire(ExpertWork *w, const Expert *experts, const uint32_t *selected,
                     const float *weights, int rows, int prompt, CacheStats *stats) {
  experts_release(w);
  uint32_t *count = counts(w), *start = offsets(w), *small = grouped(w);
  uint32_t next[w->experts], row_count[rows];
  memset(w->plan.ptr, 0, w->plan.size);
  memset(row_count, 0, sizeof row_count);
  memset(w->row_groups.ptr, 0xff, w->row_groups.size);
  w->rows = rows;
  w->active = w->group_count = w->group_rows = w->matrix_tiles = 0;
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
  w->held = experts;
  w->held_count = w->active;
  WeightSet *selected_sets[w->active];
  for (int i = 0; i < w->active; i++) {
    int e = w->active_ids[i];
    selected_sets[i] = experts[e].weights;
    if (memory_ready(weights_slot(selected_sets[i]))) stats->row_own += count[e];
    else stats->row_ssd += count[e];
  }
  int loaded = weights_load_many(selected_sets, w->active);
  if (!loaded) { experts_release(w); return 0; }
  ExpertGroup *groups = (ExpertGroup *)w->groups.ptr;
  uint32_t *row_groups = (uint32_t *)w->row_groups.ptr;
  int capacity = w->groups.size / sizeof(ExpertGroup), matrix_groups = 0;
  for (int i = 0; i < w->active; i++) {
    int e = w->active_ids[i];
    const Expert *x = experts + e;
    stats->row_unique++;
    small[e] = (count[e] <= 8 || prompt) && x->up.kind == 'n' && x->down.kind == 'n' &&
                (w->activation == EXPERT_RELU2 || x->gate.kind == 'n');
    if (small[e]) {
      int group = count[e] <= 8 ? w->group_count++ : capacity - 1 - matrix_groups++;
      groups[group] = (ExpertGroup){expert_matrix(&x->gate), expert_matrix(&x->up),
                                    expert_matrix(&x->down), start[e], count[e]};
      if (count[e] <= 8 && (int)count[e] > w->group_rows) w->group_rows = count[e];
      for (uint32_t j = start[e]; j < start[e + 1]; j++) row_groups[j] = group;
    }
  }
  ExpertTile *tiles = (ExpertTile *)w->tiles.ptr;
  for (int group = capacity - matrix_groups; group < capacity; group++)
    for (uint32_t row = 0; row < groups[group].count; row += 16)
      tiles[w->matrix_tiles++] = (ExpertTile){group, row};
  return 1;
}
void experts_mark_used(const ExpertWork *w, const Expert *experts, const uint32_t *selected, int rows) {
  uint8_t marked[w->experts];
  memset(marked, 0, sizeof marked);
  for (int i = 0; i < rows * w->chosen; i++)
    if (!marked[selected[i]]) {
      marked[selected[i]] = 1;
      memory_touch(weights_slot(experts[selected[i]].weights));
    }
}
void experts_release(ExpertWork *w) {
  for (int i = 0; i < w->held_count; i++)
    memory_pin(weights_slot(w->held[w->active_ids[i]].weights), 0);
  w->held = NULL;
  w->held_count = 0;
}
static void encode_groups(const ExpertWork *w, const Buf *input) {
  uint32_t dims[2] = {w->hidden, w->intermediate};
  gpu_kern(w->input_kernel);
  gpu_buf(0, &w->groups, 0); gpu_buf(1, &w->entries, 0); gpu_buf(2, input, 0);
  gpu_buf(3, &w->input, 0); gpu_buf(4, w->second.obj ? &w->second : &w->input, 0);
  gpu_bytes(5, dims, sizeof dims);
  gpu_buf(6, &w->row_groups, 0);
  gpu_go((w->hidden + 255) / 256, w->rows * w->chosen, 1, 256, 1, 1);
  if (w->group_count) {
    gpu_kern(w->group_rows == 1 ? w->up_single : w->up_kernel);
    gpu_buf(0, &w->groups, 0); gpu_buf(1, &w->input, 0);
    gpu_buf(2, w->second.obj ? &w->second : &w->input, 0); gpu_buf(3, &w->hidden_values, 0);
    gpu_bytes(4, dims, sizeof dims);
    gpu_go((w->intermediate + 7) / 8, w->group_count, 1, 256, 1, 1);
  }
  if (w->matrix_tiles) {
    gpu_kern(w->up_matrix);
    gpu_buf(0, &w->groups, 0); gpu_buf(1, &w->input, 0);
    gpu_buf(2, w->second.obj ? &w->second : &w->input, 0); gpu_buf(3, &w->hidden_values, 0);
    gpu_bytes(4, dims, sizeof dims); gpu_buf(5, &w->tiles, 0);
    gpu_go((w->intermediate + 63) / 64, w->matrix_tiles, 1, 128, 1, 1);
  }
  gpu_kern(w->hidden_kernel);
  gpu_buf(0, &w->groups, 0); gpu_buf(1, &w->hidden_values, 0); gpu_buf(2, &w->quantized, 0);
  gpu_bytes(3, dims, sizeof dims);
  gpu_buf(4, &w->row_groups, 0);
  gpu_go((w->intermediate + 255) / 256, w->rows * w->chosen, 1, 256, 1, 1);
  if (w->group_count) {
    gpu_kern(w->group_rows == 1 ? w->down_single : w->down_kernel);
    gpu_buf(0, &w->groups, 0); gpu_buf(1, &w->quantized, 0); gpu_buf(2, &w->output, 0);
    gpu_bytes(3, dims, sizeof dims);
    gpu_go((w->hidden + 7) / 8, w->group_count, 1, 256, 1, 1);
  }
  if (w->matrix_tiles) {
    gpu_kern(w->down_matrix);
    gpu_buf(0, &w->groups, 0); gpu_buf(1, &w->quantized, 0); gpu_buf(2, &w->quantized, 0);
    gpu_buf(3, &w->output, 0); gpu_bytes(4, dims, sizeof dims); gpu_buf(5, &w->tiles, 0);
    gpu_go((w->hidden + 63) / 64, w->matrix_tiles, 1, 128, 1, 1);
  }
}
void experts_encode(const ExpertWork *w, const Expert *experts, const Buf *input) {
  if (!w->group_count && !w->matrix_tiles) return;
  for (int i = 0; i < w->active; i++) {
    int e = w->active_ids[i];
    if (!grouped(w)[e]) continue;
    weights_use(experts[e].weights);
  }
  encode_groups(w, input);
}

void experts_sum(const ExpertWork *w, const Buf *output) {
  uint32_t dims[3] = {w->hidden, w->rows, w->chosen};
  gpu_kern(w->sum_kernel);
  gpu_buf(0, &w->output, 0); gpu_buf(1, &w->entries, 0); gpu_buf(2, &w->order, 0);
  gpu_buf(3, output, 0); gpu_bytes(4, dims, sizeof dims);
  gpu_go((w->rows * w->hidden + 255) / 256, 1, 1, 256, 1, 1);
}
int experts_table(const ExpertWork *w, const Expert *experts, const Buf *table, size_t offset) {
  ExpertGroup *entries = (ExpertGroup *)(table->ptr + offset);
  for (int e = 0; e < w->experts; e++) {
    const Expert *x = experts + e;
    if (x->up.kind != 'n' || x->down.kind != 'n' ||
        (w->activation == EXPERT_SWIGLU && x->gate.kind != 'n')) return 0;
    entries[e] = (ExpertGroup){expert_matrix(&x->gate), expert_matrix(&x->up), expert_matrix(&x->down), 0, 0};
  }
  return 1;
}
void experts_resident(ExpertWork *w, const Buf *table, size_t offset, const Buf *selected,
                       const Buf *coefficients, const Buf *input, const Buf *unique, size_t unique_offset,
                       int rows, int prompt) {
  int pairs = rows * w->chosen, groups = pairs < w->experts ? pairs : w->experts;
  w->rows = rows;
  w->active = 0;
  w->group_count = groups;
  w->group_rows = rows == 1 ? 1 : 8;
  w->matrix_tiles = prompt && rows > 8 ? (pairs + 15) / 16 + groups : 0;
  uint32_t p[5] = {w->experts, w->chosen, rows, groups, w->matrix_tiles};
  gpu_kern(w->plan_kernel);
  gpu_buf(0, table, offset); gpu_buf(1, selected, 0); gpu_buf(2, coefficients, 0);
  gpu_buf(3, &w->groups, 0); gpu_buf(4, &w->entries, 0); gpu_buf(5, &w->order, 0);
  gpu_buf(6, &w->row_groups, 0); gpu_buf(7, &w->tiles, 0); gpu_buf(8, unique, unique_offset);
  gpu_bytes(9, p, sizeof p);
  gpu_go(1, 1, 1, rows == 1 ? 32 : w->experts, 1, 1);
  encode_groups(w, input);
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
