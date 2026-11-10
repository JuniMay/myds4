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
void experts_store(ExpertWork *w, ExpertStore *store, const ExpertFiles *files) {
  w->store = store;
  w->files = files;
  w->slot = xmalloc((size_t)w->experts * sizeof *w->slot);
  w->fresh = xmalloc((size_t)w->experts);
  memset(w->fresh, 0, (size_t)w->experts);
}
// An expert's nonempty tensors as spans: consecutive tensors of one file
// share a span. Writes each tensor's span and offset; returns the spans.
static int spans_of(const Tensor *tensors, int count, FileSpan *spans, ExpertPart *parts) {
  int order[count], n = 0, used = 0;
  for (int i = 0; i < count; i++) {  // by file and offset
    if (!tensors[i].size) continue;
    int k = used++;
    for (; k > 0 && (tensors[order[k - 1]].fd > tensors[i].fd ||
                     (tensors[order[k - 1]].fd == tensors[i].fd && tensors[order[k - 1]].off > tensors[i].off));
         k--)
      order[k] = order[k - 1];
    order[k] = i;
  }
  for (int j = 0; j < used; j++) {
    const Tensor *t = tensors + order[j];
    FileSpan *last = n ? spans + n - 1 : NULL;
    if (last && last->fd == t->fd && last->off + last->len == t->off) {
      parts[order[j]] = (ExpertPart){(uint8_t)(n - 1), (uint32_t)last->len};
      last->len += t->size;
    } else {
      if (n == STORE_PARTS) die("an expert's tensors take more than %d spans", STORE_PARTS);
      spans[n] = (FileSpan){t->fd, t->off, t->size};
      parts[order[j]] = (ExpertPart){(uint8_t)n++, 0};
    }
  }
  return n;
}
void experts_files(ExpertFiles *f, const Tensor *weight, const Tensor *scales, const Tensor *exceptions,
                   const Expert *const *experts, int layers, int count) {
  *f = (ExpertFiles){.layers = layers, .experts = count, .parts = STORE_PARTS};
  f->span = xmalloc((size_t)layers * count * STORE_PARTS * sizeof *f->span);
  memset(f->span, 0, (size_t)layers * count * STORE_PARTS * sizeof *f->span);
  f->spans = xmalloc((size_t)layers * sizeof *f->spans);
  f->places = xmalloc((size_t)layers * count * sizeof *f->places);
  const Expert *first = experts[0];
  const Linear *m[3] = {&first->gate, &first->up, &first->down};
  uint64_t decoded[3];
  for (int k = 0; k < 3; k++) {  // a compressed matrix's scales decode to rows x cols bytes, one after another
    f->rows[k] = m[k]->rows, f->cols[k] = m[k]->cols / 16;
    decoded[k] = f->extra;
    if (exceptions[k].dtype) f->extra += (uint64_t)f->rows[k] * f->cols[k];
  }
  for (int l = 0; l < layers; l++) {
    f->spans[l] = 0;
    for (int e = 0; e < count; e++) {
      size_t at = (size_t)l * count + e;
      Tensor t[9];
      ExpertPart parts[9] = {0};
      for (int k = 0; k < 3; k++)
        t[3 * k] = weight[at * 3 + k], t[3 * k + 1] = scales[at * 3 + k], t[3 * k + 2] = exceptions[at * 3 + k];
      int n = spans_of(t, 9, f->span + at * STORE_PARTS, parts);
      if (n > f->spans[l]) f->spans[l] = n;  // fewer spans leave empty ones, which read nothing
      ExpertPlaces *p = f->places + at;
      *p = (ExpertPlaces){0};
      for (int k = 0; k < 3; k++) {
        if (!t[3 * k].size) continue;
        if (t[3 * k].size != (size_t)f->rows[k] * f->cols[k] * 8 ||
            (t[3 * k + 2].dtype != 0) != (exceptions[k].dtype != 0))
          die("routed expert %d of layer %d differs in shape from the first", e, l);
        p->weight[k] = parts[3 * k];
        if (t[3 * k + 2].dtype) {  // compressed: its exceptions may be empty
          p->fixed[k] = parts[3 * k + 1], p->fixed_bytes[k] = (uint32_t)t[3 * k + 1].size;
          p->exceptions[k] = parts[3 * k + 2], p->exception_count[k] = (uint32_t)(t[3 * k + 2].size / 4);
          p->scales[k] = (ExpertPart){EXPERT_DECODED, (uint32_t)decoded[k]};
        } else
          p->scales[k] = parts[3 * k + 1];
      }
    }
  }
}
void experts_decode(void *context, int layer, int e, uint8_t *slot, const uint32_t *at, uint64_t slot_bytes) {
  const ExpertFiles *f = context;
  const ExpertPlaces *p = f->places + (size_t)layer * f->experts + e;
  for (int k = 0; k < 3; k++)
    if (p->fixed_bytes[k]) {
      const uint8_t *fixed = slot + at[p->fixed[k].span] + p->fixed[k].offset;
      const uint32_t *exceptions = (const uint32_t *)(slot + at[p->exceptions[k].span] + p->exceptions[k].offset);
      uint8_t *out = slot + slot_bytes - f->extra + p->scales[k].offset;
      if (!weights_csf_decode(out, fixed, p->fixed_bytes[k], exceptions, p->exception_count[k], f->rows[k], f->cols[k]))
        die("invalid compressed NVFP4 scales of routed expert %d of layer %d", e, layer);
    }
}
// The views of a held expert's matrices in its slot; at: the spans' offsets.
static Buf part(const Buf *slot, const uint32_t *at, const ExpertFiles *f, ExpertPart p, size_t bytes) {
  size_t offset = p.span == EXPERT_DECODED ? slot->size - f->extra + p.offset : at[p.span] + p.offset;
  return (Buf){.obj = slot->obj, .ptr = slot->ptr + offset, .gpu = slot->gpu + offset, .off = slot->off + offset,
               .size = bytes};
}
static void views(Expert *x, const Buf *slot, const uint32_t *at, const ExpertFiles *f, int layer, int e) {
  const ExpertPlaces *p = f->places + (size_t)layer * f->experts + e;
  Linear *m[3] = {&x->gate, &x->up, &x->down};
  for (int k = 0; k < 3; k++)
    if (m[k]->rows) {
      m[k]->w = part(slot, at, f, p->weight[k], (size_t)m[k]->rows * m[k]->cols / 2);
      m[k]->s = part(slot, at, f, p->scales[k], (size_t)m[k]->rows * m[k]->cols / 16);
    }
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
void experts_acquire(ExpertWork *w, Expert *experts, int layer, const uint32_t *selected, const float *weights,
                     int rows, int prompt, CacheStats *stats) {
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
  int ids[w->active];
  for (int i = 0; i < w->active; i++) {
    ids[i] = (int)w->active_ids[i];
    if (cache_ready(w->store, layer, ids[i])) stats->row_own += count[ids[i]];
    else stats->row_ssd += count[ids[i]];
  }
  // Held, the experts stay in their slots until experts_release.
  cache_hold(w->store, layer, ids, w->active);
  w->held = experts;
  w->held_count = w->active;
  w->held_layer = layer;
  cache_wait(w->store, layer, ids, w->active);
  for (int i = 0; i < w->active; i++) {
    uint32_t at[STORE_PARTS];
    int fresh;
    w->slot[ids[i]] = cache_get(w->store, layer, ids[i], at, &fresh);
    w->fresh[ids[i]] = (uint8_t)fresh;
    views(experts + ids[i], w->slot[ids[i]], at, w->files, layer, ids[i]);
  }
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
}
int experts_mark_used(const ExpertWork *w, const Expert *experts, int layer, const uint32_t *selected, int rows) {
  uint8_t marked[w->experts];
  int ids[w->experts], n = 0;
  memset(marked, 0, sizeof marked);
  for (int i = 0; i < rows * w->chosen; i++)
    if (!marked[selected[i]]) {
      marked[selected[i]] = 1;
      ids[n++] = (int)selected[i];
    }
  cache_use(w->store, layer, ids, n);
  return n;
}
void experts_routes_init(ExpertRoutes *r, int rows, int layers, int chosen) {
  *r = (ExpertRoutes){.rows = rows, .layers = layers, .chosen = chosen};
  r->id = xmalloc((size_t)rows * layers * chosen * sizeof *r->id);
  r->from = xmalloc((size_t)rows * layers * chosen);
  r->step = xmalloc((size_t)rows * layers * sizeof *r->step);
}
void experts_routes_record(ExpertRoutes *r, ExpertStore *store, int layer, const uint32_t *selected, int n,
                           int table) {
  static const uint8_t from[4] = {4, 0, 1, 2};  // by cache_where: the SSD, memory, read ahead, still being read
  for (int row = 0; row < n && row < r->rows; row++) {
    size_t at = ((size_t)row * r->layers + layer) * r->chosen;
    for (int k = 0; k < r->chosen; k++) {
      uint32_t e = selected[row * r->chosen + k];
      r->id[at + k] = (uint16_t)e;
      r->from[at + k] = table ? 0 : from[cache_where(store, layer, (int)e)];
    }
    r->step[(size_t)row * r->layers + layer] = r->steps;
  }
}
int experts_routes_get(const ExpertRoutes *r, int row, uint16_t *id, uint8_t *from) {
  if (row < 0 || row >= r->rows || !r->steps) return 0;
  for (int l = 0; l < r->layers; l++)
    if (r->step[(size_t)row * r->layers + l] != r->steps) return 0;  // the row did not reach every layer then
  size_t first = (size_t)row * r->layers * r->chosen, count = (size_t)r->layers * r->chosen;
  memcpy(id, r->id + first, count * sizeof *id);
  memcpy(from, r->from + first, count);
  return 1;
}
void experts_release(ExpertWork *w) {
  if (w->held_count) {
    int ids[w->held_count];
    for (int i = 0; i < w->held_count; i++) ids[i] = (int)w->active_ids[i];
    cache_unhold(w->store, w->held_layer, ids, w->held_count);
  }
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
    if (grouped(w)[e] && w->fresh[e]) gpu_use(w->slot[e]);
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
int experts_table(const ExpertWork *w, Expert *experts, int layer, const Buf *table, size_t offset) {
  ExpertGroup *entries = (ExpertGroup *)(table->ptr + offset);
  for (int e = 0; e < w->experts; e++) {
    Expert *x = experts + e;
    if (x->up.kind != 'n' || x->down.kind != 'n' ||
        (w->activation == EXPERT_SWIGLU && x->gate.kind != 'n')) return 0;
    // Every expert is mapped (cache_layer_ready): cache_get changes nothing.
    uint32_t at[STORE_PARTS];
    int fresh;
    const Buf *slot = cache_get(w->store, layer, e, at, &fresh);
    if (!slot || fresh) return 0;
    views(x, slot, at, w->files, layer, e);
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
