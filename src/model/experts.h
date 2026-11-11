// GPU addresses and scales for rows selecting the same expert.
#pragma once
#include "linear.h"
#include "../engine/expert_store.h"

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

// An expert's matrices. A routed expert's buffers are views of its slot in
// the expert store (experts_store), set while the expert is held.
typedef struct {
  Linear gate, up, down;
} Expert;
// Where each matrix of an expert lies in its slot (gate, up, down): its
// weights and scales, each a span and an offset in it. Compressed scales
// (CSF) are read as their fixed rows and exceptions; the store's readers
// decode them (experts_decode) into the region at the slot's end
// (EXPERT_DECODED), where the scales' offset then points.
enum { EXPERT_DECODED = 0xff };
typedef struct { uint8_t span; uint32_t offset; } ExpertPart;
typedef struct {
  ExpertPart weight[3], scales[3], fixed[3], exceptions[3];
  uint32_t fixed_bytes[3], exception_count[3];  // 0 for stored scales
} ExpertPlaces;
// A family's routed experts in the store: per expert, the files' bytes
// (spans), where its matrices lie (places), and the bytes its compressed
// scales decode to (extra, the same for every expert).
typedef struct {
  int layers, experts, parts;  // parts: spans per expert in `span`, STORE_PARTS
  FileSpan *span;              // [layers][experts][STORE_PARTS]
  int *spans;                  // [layers]: the most spans an expert of the layer takes
  ExpertPlaces *places;        // [layers][experts]
  uint64_t extra;
  int rows[3], cols[3];        // each matrix's scale rows and columns (cols / 16)
} ExpertFiles;
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
  ExpertStore *store;          // the routed experts' store (experts_store)
  const ExpertFiles *files;    // the store's layout of them
  int held_layer;              // the store layer of the held experts
  int *held_tables, held_table_count;  // [layers]: the layers held through their tables (experts_hold_table)
  const Buf **slot;            // [experts]: a held expert's slot
  uint8_t *fresh;              // [experts]: its slot needs gpu_use until the residency set is committed
  int input_kernel, up_kernel, hidden_kernel, down_kernel, sum_kernel;
  int up_single, down_single;
  int up_matrix, down_matrix;
  int plan_kernel;
} ExpertWork;

void experts_init(ExpertWork *work, int hidden, int intermediate, int experts, int chosen,
                  ExpertActivation activation, ExpertAccumulation accumulation);
// The routed experts' weights are in the store's slots, laid out by files.
void experts_store(ExpertWork *work, ExpertStore *store, const ExpertFiles *files);
// A planner of its own for a pass (src/engine/pass.h): the shape and kernels
// of `from`, without buffers until experts_workspace allocates them.
void experts_pass_begin(ExpertWork *work, const ExpertWork *from);
// Frees a pass's planner, after the pass's GPU work has finished.
void experts_pass_end(ExpertWork *work);
// The layer's experts stay in their slots until experts_release, for the GPU
// work that reaches them through the layer's table.
void experts_hold_table(ExpertWork *work, int layer);
// The store's layout from the routed experts' tensors and matrices: for each
// expert ([layers][experts][3]), its matrices' weights, their stored scales
// or compressed rows, and their exceptions (a zero Tensor, without a dtype,
// for stored scales; all three zero for an absent matrix, such as squared
// ReLU's gate). Consecutive tensors of one file share a span. Allocates the
// layout's arrays.
void experts_files(ExpertFiles *files, const Tensor *weight, const Tensor *scales, const Tensor *exceptions,
                   const Expert *const *experts, int layers, int experts_per_layer);
// The store's step after an expert's reads (ExpertLayout.transform; context:
// its ExpertFiles): the compressed scales decoded into the slot's end.
void experts_decode(void *files, int layer, int expert, uint8_t *slot, const uint32_t *at, uint64_t slot_bytes);
uint64_t experts_workspace(ExpertWork *work, int rows, int allocate);
uint64_t experts_bytes(const ExpertWork *work);
// Routing and the preceding expert work have completed. Release the previous
// selection, then hold every new selection in the store before reading any
// miss, and wait for the reads. `layer`: the store layer.
void experts_acquire(ExpertWork *work, Expert *experts, int layer, const uint32_t *selected, const float *weights,
                     int rows, int prompt, CacheStats *stats);
// Call after the queued expert work completes. Resident planning retains this
// selection until a later acquisition or the end of the model pass.
void experts_release(ExpertWork *work);
// Encode NVFP4 groups together. Prompt groups use matrix tiles; decoding
// retains scalar reductions. Other formats write to experts_output separately.
void experts_encode(const ExpertWork *work, const Expert *experts, const Buf *input);
void experts_sum(const ExpertWork *work, const Buf *output);
// Build a table only when every expert is resident (with a store:
// cache_layer_ready); unsupported formats use the ordinary path. Resident
// encoding takes prompt rows or at most eight rows.
int experts_table(const ExpertWork *work, Expert *experts, int layer, const Buf *table, size_t offset);
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
// experts used longest ago first. With a store, the store counts them as
// hits. Returns how many different experts the rows chose.
int experts_mark_used(const ExpertWork *work, const Expert *experts, int layer, const uint32_t *selected, int rows);
// The experts that each row of a model's latest decoding pass chose in each
// store layer, and where each one was when chosen: 0 in memory, 1 read
// ahead, 2 late (read ahead and still being read), 4 from the SSD
// (myds4_routes). Each decoding pass begins a step; a row's routes are
// complete when every layer recorded it in the latest step.
typedef struct {
  int rows, layers, chosen;  // capacity
  uint16_t *id;              // [rows][layers][chosen]
  uint8_t *from;             // [rows][layers][chosen]
  uint64_t *step;            // [rows][layers]: the step that recorded the row's layer
  uint64_t steps;            // decoding passes begun
} ExpertRoutes;
void experts_routes_init(ExpertRoutes *routes, int rows, int layers, int chosen);
static inline void experts_routes_begin(ExpertRoutes *routes) { routes->steps++; }
// Records a store layer's selections of n rows ([n][chosen]) before the
// layer's experts are acquired; a layer reached through its table (table)
// had every expert in memory.
void experts_routes_record(ExpertRoutes *routes, ExpertStore *store, int layer, const uint32_t *selected, int n,
                           int table);
// Copies row `row`'s routes of the latest step; 0 when they are incomplete.
int experts_routes_get(const ExpertRoutes *routes, int row, uint16_t *id, uint8_t *from);
static inline int experts_count(const ExpertWork *work, int expert) {
  return ((const uint32_t *)work->plan.ptr)[expert];
}
static inline int experts_grouped(const ExpertWork *work, int expert) {
  return ((const uint32_t *)work->plan.ptr)[2 * work->experts + 1 + expert];
}
size_t experts_entry_offset(const ExpertWork *work, int expert);
Buf experts_output(const ExpertWork *work, int expert);
