// Reloadable weight groups. A group is one indivisible memory-cache entry.
#pragma once
#include "../engine/myds4.h"
#include "../engine/memory.h"

typedef struct WeightSet WeightSet;
typedef struct { unsigned count; uint64_t generation; } WeightPresence;
typedef struct {
  CacheStats stats;
  uint64_t bytes;
  int slots, capacity;
} WeightCache;
typedef void (*WeightFill)(Buf *buffer, const void *parameters);
enum { WEIGHT_ADD = 1, WEIGHT_EXP = 2 };

WeightSet *weights_new(MemoryAccount *account, const char *name);
WeightSet *weights_cached(WeightSet *backbone, WeightCache *cache, const char *name);
WeightSet *weights_dependent(WeightSet *backbone, const char *name);
// Attach the layer's residency counter before the group's first load.
void weights_track(WeightSet *set, WeightPresence *presence);
MemorySlot *weights_slot(WeightSet *set);
uint64_t weights_required(const WeightSet *set);
uint64_t weights_resident_bytes(const WeightSet *set);
void weights_tensor(WeightSet *set, Buf *target, Tensor tensor, int aligned);
// Compressed Scale Format: 16 row bases, row-major nibble offsets, and byte exceptions.
int weights_csf_decode(uint8_t *out, const uint8_t *fixed, size_t fixed_bytes,
                       const uint32_t *exceptions, size_t exception_count, size_t rows, size_t cols);
void weights_csf(WeightSet *set, Buf *target, Tensor fixed, Tensor exceptions, int rows, int cols);
void weights_vector(WeightSet *set, Buf *target, Tensor tensor, float add, int operations);
void weights_product(WeightSet *set, Buf *target, Tensor first, Tensor second);
void weights_padded(WeightSet *set, Buf *target, Tensor tensor, int stride);
// Reconstruct quantized, pre-global-scale values as a padded BF16 matrix.
void weights_expanded(WeightSet *set, Buf *target, Tensor weight, Tensor scales, int kind, int cols, int stride);
void weights_generated(WeightSet *set, Buf *target, size_t bytes, WeightFill fill,
                       const void *parameters, size_t parameter_bytes);
void weights_alias(WeightSet *set, Buf *target, const Buf *source);
int weights_load(WeightSet *set);
// The caller pins a nonempty set of unique selections before loading. Reads run in a bounded worker
// group; Metal allocation, accounting and error reporting stay on the caller.
int weights_load_many(WeightSet *const *sets, int count);
void weights_use(const WeightSet *set);
void weights_reside(WeightSet *set, int resident);
void weights_free(WeightSet *set);
void weights_remove(WeightSet *set, Buf *target);
