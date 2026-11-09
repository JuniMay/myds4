#include "weights.h"
#include <errno.h>
#include <dispatch/dispatch.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef struct {
  Buf *target;
  const Buf *alias;
  size_t bytes;
  WeightFill fill;
  void *parameters;
} Weight;

struct WeightSet {
  MemorySlot slot;
  Weight *items;
  size_t count, capacity;
  uint64_t required, scratch;
  uint64_t read_bytes;
  WeightCache *cache;
  Buf storage;
  int separate;
  void *scratch_data;
  int read_error;
  uint64_t failed_offset;
  WeightPresence *presence;
};
static _Thread_local WeightSet *reading_set;
static _Thread_local int reading_worker;

// While a model loads, quitting stops its reads between weights and between
// 1 GiB parts of one; the load then frees what it took and fails. Weights
// that a pass loads (routed experts, vision weights) complete with the pass.
static int stop_on_quit;
void weights_stop_on_quit(int on) { stop_on_quit = on; }
static int quitting(void) { return stop_on_quit && shutdown_requested(); }

static void unload(void *context) {
  WeightSet *set = context;
  if (set->presence && memory_ready(&set->slot)) {
    set->presence->count--;
    set->presence->generation++;
  }
  if (set->cache && memory_ready(&set->slot)) {
    set->cache->bytes -= set->slot.payload;
    set->cache->slots--;
    set->cache->stats.evictions++;
  }
  if (set->storage.obj) {
    gpu_free(&set->storage);
    for (size_t i = 0; i < set->count; i++) *set->items[i].target = (Buf){0};
  } else {
    for (size_t i = 0; i < set->count; i++) {
      Weight *weight = &set->items[i];
      if (weight->alias) *weight->target = (Buf){0};
      else gpu_free(weight->target);
    }
  }
  gpu_residency_commit();
}

WeightSet *weights_new(MemoryAccount *account, const char *name) {
  WeightSet *set = xmalloc(sizeof *set);
  memory_slot_init(&set->slot, account, NULL, name, unload, set);
  return set;
}
WeightSet *weights_cached(WeightSet *backbone, WeightCache *cache, const char *name) {
  WeightSet *set = xmalloc(sizeof *set);
  memory_slot_init(&set->slot, backbone->slot.account, &backbone->slot, name, unload, set);
  set->cache = cache;
  cache->capacity++;
  return set;
}
WeightSet *weights_dependent(WeightSet *backbone, const char *name) {
  WeightSet *set = xmalloc(sizeof *set);
  memory_slot_init(&set->slot, backbone->slot.account, &backbone->slot, name, unload, set);
  return set;
}
MemorySlot *weights_slot(WeightSet *set) { return &set->slot; }
void weights_track(WeightSet *set, WeightPresence *presence) { set->presence = presence; }
static uint64_t packed_bytes(const WeightSet *set) {
  if (!set->cache || set->separate) return 0;
  uint64_t bytes = 0;
  for (size_t i = 0; i < set->count; i++)
    if (!set->items[i].alias) bytes = ((bytes + 255) & ~255ull) + set->items[i].bytes;
  return bytes <= UINT32_MAX ? bytes : 0;
}
uint64_t weights_required(const WeightSet *set) {
  if (!set) return 0;
  uint64_t packed = packed_bytes(set);
  return (packed ? gpu_allocation_size(packed) : set->required) + set->scratch;
}
uint64_t weights_resident_bytes(const WeightSet *set) { return set && memory_ready(&set->slot) ? set->slot.payload : 0; }

static Weight *append(WeightSet *set, Buf *target) {
  if (set->count == set->capacity) {
    set->capacity = set->capacity ? set->capacity * 2 : 64;
    Weight *items = realloc(set->items, set->capacity * sizeof *items);
    if (!items) die("out of memory for weight metadata");
    set->items = items;
  }
  Weight *weight = &set->items[set->count++];
  *weight = (Weight){.target = target};
  return weight;
}

void weights_generated(WeightSet *set, Buf *target, size_t bytes, WeightFill fill,
                       const void *parameters, size_t parameter_bytes) {
  Weight *weight = append(set, target);
  weight->bytes = bytes;
  weight->fill = fill;
  if (parameter_bytes) {
    weight->parameters = xmalloc(parameter_bytes);
    memcpy(weight->parameters, parameters, parameter_bytes);
  }
  set->required += gpu_allocation_size(bytes);
  target->size = bytes;
}

void weights_alias(WeightSet *set, Buf *target, const Buf *source) {
  Weight *weight = append(set, target);
  weight->alias = source;
}

// Checkpoint files are opened with F_NOCACHE, but uncached reads bypass the
// page cache only in whole 16 KiB-aligned spans into page-aligned memory
// (io.c); unaligned reads still fill it, and the kernel then keeps that file
// cache (avail_of, io.c). A tensor at its own offset is read in aligned
// spans of up to READ_PIECE into a page-aligned buffer, and its bytes are
// copied out.
enum { READ_ALIGN = 16384, READ_PIECE = 16 << 20 };
static void read_bytes(Tensor tensor, size_t offset, void *target, size_t bytes) {
  reading_set->read_bytes += bytes;
  uint64_t from = tensor.off + offset, end = from + bytes, at = from & ~(uint64_t)(READ_ALIGN - 1);
  uint64_t span = (end - at + READ_ALIGN - 1) & ~(uint64_t)(READ_ALIGN - 1);
  size_t room = span < READ_PIECE ? (size_t)span : READ_PIECE;
  uint8_t *buffer = NULL;
  if (posix_memalign((void **)&buffer, READ_ALIGN, room)) die("out of memory for a %zu-byte read buffer", room);
  while (at < end) {
    if (quitting()) {
      reading_set->read_error = ECANCELED;
      break;
    }
    uint64_t left = (end - at + READ_ALIGN - 1) & ~(uint64_t)(READ_ALIGN - 1);
    size_t part = left < room ? (size_t)left : room;
    uint64_t first = at > from ? at : from, last = at + part < end ? at + part : end;  // the bytes wanted
    ssize_t n = pread(tensor.fd, buffer, part, (off_t)at);
    if (n < 0 && errno == EINTR) continue;
    if (n < (ssize_t)(last - at)) {
      if (!reading_worker) die("cannot reload weight data at offset %llu", (unsigned long long)at);
      reading_set->read_error = n < 0 ? errno : EIO;
      reading_set->failed_offset = at;
      break;
    }
    memcpy((uint8_t *)target + (first - from), buffer + (first - at), last - first);
    at += part;
  }
  free(buffer);
}

typedef struct { Tensor tensor; int aligned; } Raw;
static void raw_fill(Buf *buffer, const void *context) {
  const Raw *raw = context;
  if (raw->aligned) {
    reading_set->read_bytes += raw->tensor.size;
    io_read(raw->tensor.fd, raw->tensor.off, raw->tensor.size, buffer);
  }
  else read_bytes(raw->tensor, 0, buffer->ptr, raw->tensor.size);
}
void weights_tensor(WeightSet *set, Buf *target, Tensor tensor, int aligned) {
  Raw raw = {tensor, aligned};
  if (aligned) set->separate = 1;
  weights_generated(set, target, tensor.size + (aligned ? 2 * 16384 : 0), raw_fill, &raw, sizeof raw);
}

int weights_csf_decode(uint8_t *out, const uint8_t *fixed, size_t fixed_bytes,
                       const uint32_t *exceptions, size_t exception_count, size_t rows, size_t cols) {
  if (!rows || rows % 16 || !cols || cols % 2 || rows > (1u << 24) / cols) return 0;
  size_t stride = 16 + cols * 8;
  if (fixed_bytes != rows / 16 * stride) return 0;
  for (size_t row = 0; row < rows; row++) {
    const uint8_t *slab = fixed + row / 16 * stride;
    unsigned base = slab[row % 16];
    if (base > 240) return 0;
    const uint8_t *codes = slab + 16 + row % 16 * (cols / 2);
    for (size_t col = 0; col < cols; col++)
      out[row * cols + col] = (uint8_t)(base + ((codes[col / 2] >> (4 * (col % 2))) & 15));
  }
  uint32_t previous = 0;
  for (size_t i = 0; i < exception_count; i++) {
    uint32_t entry = exceptions[i], position = entry & 0xffffffu;
    if (position >= rows * cols || (i && position <= previous)) return 0;
    out[position] = (uint8_t)(entry >> 24);
    previous = position;
  }
  return 1;
}

typedef struct { Tensor fixed, exceptions; int rows, cols; } CompressedScales;
static void csf_fill(Buf *buffer, const void *context) {
  const CompressedScales *scales = context;
  int borrowed = reading_set->scratch_data != NULL;
  void *fixed = borrowed ? reading_set->scratch_data : memory_host_required(scales->fixed.size, 0);
  void *exceptions = borrowed ? (char *)fixed + scales->fixed.size : memory_host_required(scales->exceptions.size, 0);
  read_bytes(scales->fixed, 0, fixed, scales->fixed.size);
  read_bytes(scales->exceptions, 0, exceptions, scales->exceptions.size);
  int decoded = !reading_set->read_error && weights_csf_decode(buffer->ptr, fixed, scales->fixed.size, exceptions,
                                                               scales->exceptions.size / 4, scales->rows, scales->cols);
  if (!borrowed) { memory_host_free(fixed); memory_host_free(exceptions); }
  if (reading_set->read_error) return;
  if (!decoded) {
    if (!reading_worker) die("invalid compressed NVFP4 scale stream");
    reading_set->read_error = EBADMSG;
    reading_set->failed_offset = scales->fixed.off;
  }
}
void weights_csf(WeightSet *set, Buf *target, Tensor fixed, Tensor exceptions, int rows, int cols) {
  if (fixed.dtype != 'u' || exceptions.dtype != 'U' || exceptions.size % 4)
    die("compressed NVFP4 scales require U8 codes and U32 exceptions");
  uint64_t scratch = fixed.size + exceptions.size;
  if (scratch > set->scratch) set->scratch = scratch;
  CompressedScales scales = {fixed, exceptions, rows, cols};
  weights_generated(set, target, (size_t)rows * cols, csf_fill, &scales, sizeof scales);
}

typedef struct { Tensor first, second; float add; int operations, product; } Vector;
static float value(const uint8_t *bytes, int index, int dtype) {
  if (dtype == 'b') {
    uint16_t short_bits;
    memcpy(&short_bits, bytes + (size_t)index * 2, 2);
    uint32_t bits = (uint32_t)short_bits << 16;
    float result; memcpy(&result, &bits, 4); return result;
  }
  float result; memcpy(&result, bytes + (size_t)index * 4, 4); return result;
}
static void vector_fill(Buf *buffer, const void *context) {
  const Vector *vector = context;
  uint8_t first[16384], second[16384];
  size_t count = vector->first.size / (vector->first.dtype == 'b' ? 2 : 4);
  float *output = (float *)buffer->ptr;
  for (size_t at = 0; at < count; at += 4096) {
    size_t n = count - at < 4096 ? count - at : 4096;
    size_t width = vector->first.dtype == 'b' ? 2 : 4;
    read_bytes(vector->first, at * width, first, n * width);
    if (vector->product) {
      size_t other_width = vector->second.dtype == 'b' ? 2 : 4;
      read_bytes(vector->second, at * other_width, second, n * other_width);
    }
    for (size_t i = 0; i < n; i++) {
      float v = value(first, (int)i, vector->first.dtype);
      if (vector->operations & WEIGHT_ADD) v += vector->add;
      if (vector->product) v *= value(second, (int)i, vector->second.dtype);
      if (vector->operations & WEIGHT_EXP) v = expf(v);
      output[at + i] = v;
    }
  }
}
void weights_vector(WeightSet *set, Buf *target, Tensor tensor, float add, int operations) {
  if (tensor.dtype != 'b' && tensor.dtype != 'f') die("weight vector must use BF16 or FP32");
  Vector vector = {.first = tensor, .add = add, .operations = operations};
  size_t bytes = tensor.size / (tensor.dtype == 'b' ? 2 : 4) * 4;
  weights_generated(set, target, bytes, vector_fill, &vector, sizeof vector);
}
void weights_product(WeightSet *set, Buf *target, Tensor first, Tensor second) {
  Vector vector = {.first = first, .second = second, .product = 1};
  size_t n = first.size / (first.dtype == 'b' ? 2 : 4);
  if (n != second.size / (second.dtype == 'b' ? 2 : 4)) die("weight product sizes differ");
  weights_generated(set, target, n * 4, vector_fill, &vector, sizeof vector);
}

typedef struct { Tensor tensor; int stride; } Padded;
static void padded_fill(Buf *buffer, const void *context) {
  const Padded *padded = context;
  size_t rows = padded->tensor.shape[0], row_bytes = padded->tensor.size / rows;
  memset(buffer->ptr, 0, buffer->size);
  for (size_t row = 0; row < rows; row++)
    read_bytes(padded->tensor, row * row_bytes, buffer->ptr + row * padded->stride * 2, row_bytes);
}
void weights_padded(WeightSet *set, Buf *target, Tensor tensor, int stride) {
  if (tensor.dtype != 'b' || stride * 2u < tensor.size / tensor.shape[0]) die("invalid padded BF16 weight");
  Padded padded = {tensor, stride};
  weights_generated(set, target, (size_t)tensor.shape[0] * stride * 2, padded_fill, &padded, sizeof padded);
}

typedef struct { Tensor weight, scales; int kind, cols, stride; } Expanded;
static float e4m3(uint8_t code) {
  unsigned exponent = (code >> 3) & 15, mantissa = code & 7;
  float value = exponent ? ldexpf(1 + mantissa * .125f, (int)exponent - 7) : ldexpf((float)mantissa, -9);
  return code & 128 ? -value : value;
}
static void expanded_fill(Buf *buffer, const void *context) {
  const Expanded *p = context;
  uint8_t *weight = memory_host_required(p->weight.size, 0);
  uint8_t *scales = memory_host_required(p->scales.size, 0);
  read_bytes(p->weight, 0, weight, p->weight.size);
  if (p->scales.size) read_bytes(p->scales, 0, scales, p->scales.size);
  memset(buffer->ptr, 0, buffer->size);
  static const float values[] = {0, .5f, 1, 1.5f, 2, 3, 4, 6};
  for (int row = 0; row < p->weight.shape[0]; row++) for (int col = 0; col < p->cols; col++) {
    float value;
    if (p->kind == 'n') {
      uint8_t code = weight[(size_t)row * (p->cols / 2) + col / 2] >> ((col % 2) * 4);
      value = values[code & 7]; if (code & 8) value = -value;
      value *= e4m3(scales[(size_t)row * (p->cols / 16) + col / 16]);
    } else {
      value = e4m3(weight[(size_t)row * p->cols + col]);
      if (p->kind == 'm') value *= ldexpf(1, (int)scales[(size_t)row * (p->cols / 32) + col / 32] - 127);
    }
    uint32_t bits; memcpy(&bits, &value, 4);
    ((uint16_t *)buffer->ptr)[(size_t)row * p->stride + col] = (uint16_t)((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
  }
  memory_host_free(weight); memory_host_free(scales);
}
void weights_expanded(WeightSet *set, Buf *target, Tensor weight, Tensor scales, int kind, int cols, int stride) {
  Expanded p = {weight, scales, kind, cols, stride};
  uint64_t scratch = weight.size + scales.size;
  if (scratch > set->scratch) set->scratch = scratch;
  weights_generated(set, target, (size_t)weight.shape[0] * stride * 2, expanded_fill, &p, sizeof p);
}

static int allocate_set(WeightSet *set, size_t scratch) {
  MemorySlot *slots[] = {&set->slot};
  uint64_t required = weights_required(set);
  if (!memory_admit(slots, &required, 1, MEMORY_REQUIRED)) return 0;
  memory_pin(&set->slot, 1);
  MemorySlot *before = memory_scope(&set->slot);
  set->read_bytes = 0;
  set->read_error = 0;
  uint64_t packed = packed_bytes(set), offset = 0;
  if (packed) {
    set->storage = gpu_new(packed);
    gpu_resident(&set->storage, 1);
  }
  for (size_t i = 0; i < set->count; i++) {
    Weight *weight = &set->items[i];
    if (weight->alias) continue;
    if (packed) {
      offset = (offset + 255) & ~255ull;
      *weight->target = (Buf){.obj = set->storage.obj, .ptr = set->storage.ptr + offset,
                              .gpu = set->storage.gpu + offset, .size = weight->bytes, .off = offset};
      offset += weight->bytes;
    } else {
      *weight->target = gpu_new(weight->bytes);
      gpu_resident(weight->target, 1);
    }
  }
  if (scratch) set->scratch_data = memory_host_required(scratch, 0);
  memory_scope(before);
  return 1;
}
static void fill_set(WeightSet *set) {
  WeightSet *previous = reading_set;
  reading_set = set;
  for (size_t i = 0; i < set->count && !set->read_error; i++) {
    Weight *weight = &set->items[i];
    if (quitting()) set->read_error = ECANCELED;
    else if (!weight->alias) weight->fill(weight->target, weight->parameters);
  }
  reading_set = previous;
}
static void finish_set(WeightSet *set, unsigned pinned) {
  for (size_t i = 0; i < set->count; i++)
    if (set->items[i].alias) *set->items[i].target = *set->items[i].alias;
  memory_host_free(set->scratch_data);
  set->scratch_data = NULL;
  memory_trim(&set->slot);
  memory_publish(&set->slot, 1);
  if (set->presence) {
    set->presence->count++;
    set->presence->generation++;
  }
  memory_pin(&set->slot, pinned);
  if (set->cache) {
    set->cache->slots++;
    set->cache->bytes += set->slot.payload;
    set->cache->stats.bytes += set->read_bytes;
    set->cache->stats.misses++;
  }
}
// A load that did not complete: its buffers and charge go back, and the
// set keeps the pin it had before the load.
static void discard(WeightSet *set, unsigned pinned) {
  memory_host_free(set->scratch_data);
  set->scratch_data = NULL;
  unload(set);
  memory_trim(&set->slot);
  memory_pin(&set->slot, pinned);
}
int weights_load(WeightSet *set) {
  if (memory_ready(&set->slot)) {
    memory_touch(&set->slot);
    if (set->cache) set->cache->stats.hits++;
    return 1;
  }
  double began = now();
  unsigned pinned = set->slot.pins;
  if (!allocate_set(set, 0)) return 0;
  MemorySlot *before = memory_scope(&set->slot);
  fill_set(set);
  memory_scope(before);
  if (set->read_error) {  // quitting stopped it; other read errors stop the process in fill_set
    discard(set, pinned);
    return 0;
  }
  finish_set(set, pinned);
  gpu_residency_commit();
  if (set->cache) set->cache->stats.read_s += now() - began;
  return 1;
}

typedef struct {
  WeightSet **sets;
  int count, workers;
} WeightReads;
static void read_sets(void *context, size_t worker) {
  WeightReads *job = context;
  reading_worker = 1;
  for (int i = worker; i < job->count; i += job->workers) {
    fill_set(job->sets[i]);
  }
  reading_worker = 0;
}
static int parallel_readable(const WeightSet *set) {
  for (size_t i = 0; i < set->count; i++) {
    const Weight *w = set->items + i;
    if (w->alias || w->fill == csf_fill) continue;
    if (w->fill != raw_fill || ((const Raw *)w->parameters)->aligned) return 0;
  }
  return 1;
}

int weights_load_many(WeightSet *const *sets, int count) {
  WeightSet *missing[count];
  unsigned pinned[count];
  int n = 0;
  WeightCache *cache = sets[0]->cache;
  for (int i = 0; i < count; i++)
    if (!memory_ready(&sets[i]->slot) &&
        (!cache || sets[i]->cache != cache || !parallel_readable(sets[i]))) {
      for (int k = 0; k < count; k++) if (!weights_load(sets[k])) return 0;
      return 1;
    }
  for (int i = 0; i < count; i++) {
    if (memory_ready(&sets[i]->slot)) {
      memory_touch(&sets[i]->slot);
      if (sets[i]->cache) sets[i]->cache->stats.hits++;
    } else missing[n++] = sets[i];
  }
  if (!n) return 1;
  if (n == 1) return weights_load(missing[0]);
  double began = now();
  for (int i = 0; i < n; i++) {
    pinned[i] = missing[i]->slot.pins;
    if (!allocate_set(missing[i], missing[i]->scratch)) {
      for (int k = 0; k < i; k++) discard(missing[k], pinned[k]);
      return 0;
    }
  }
  WeightReads job = {.sets = missing, .count = n, .workers = n < 8 ? n : 8};
  dispatch_apply_f(job.workers, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), &job, read_sets);
  for (int i = 0; i < n; i++) if (missing[i]->read_error) {
    int error = missing[i]->read_error;
    uint64_t offset = missing[i]->failed_offset;
    for (int k = 0; k < n; k++) discard(missing[k], pinned[k]);
    if (error == ECANCELED) return 0;  // quitting stopped the load
    die("cannot read expert at offset %llu: %s", (unsigned long long)offset, strerror(error));
  }
  for (int i = 0; i < n; i++) finish_set(missing[i], pinned[i]);
  gpu_residency_commit();
  cache->stats.read_s += now() - began;
  return 1;
}

void weights_use(const WeightSet *set) {
  if (set->storage.obj) {
    gpu_use(&set->storage);
    return;
  }
  for (size_t i = 0; i < set->count; i++)
    if (!set->items[i].alias) gpu_use(set->items[i].target);
}

void weights_reside(WeightSet *set, int resident) {
  if (!set || !memory_ready(&set->slot)) return;
  if (set->storage.obj) {
    gpu_resident(&set->storage, resident);
    return;
  }
  for (size_t i = 0; i < set->count; i++)
    if (!set->items[i].alias) gpu_resident(set->items[i].target, resident);
}

void weights_free(WeightSet *set) {
  if (!set) return;
  unload(set);
  memory_trim(&set->slot);
  memory_slot_remove(&set->slot);
  if (set->cache) set->cache->capacity--;
  for (size_t i = 0; i < set->count; i++) free(set->items[i].parameters);
  free(set->items);
  free(set);
}

void weights_remove(WeightSet *set, Buf *target) {
  if (memory_ready(&set->slot)) die("cannot change a resident weight group");
  for (size_t i = 0; i < set->count; i++) if (set->items[i].target == target) {
    Weight *weight = &set->items[i];
    if (!weight->alias) set->required -= gpu_allocation_size(weight->bytes);
    free(weight->parameters);
    memmove(weight, weight + 1, (set->count - i - 1) * sizeof *weight);
    set->count--;
    *target = (Buf){0};
    return;
  }
}
