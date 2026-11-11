// Metal through the Objective-C runtime, so that the host stays in C. Each
// message is sent through objc_msgSend cast to the method's exact type.
#include <objc/message.h>
#include <objc/runtime.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

#include "myds4.h"
#include "memory.h"
#include "pass.h"

typedef struct { unsigned long w, h, d; } Size3;
extern id MTLCreateSystemDefaultDevice(void);
extern void *objc_autoreleasePoolPush(void);
extern void objc_autoreleasePoolPop(void *);

#define S(name) sel_registerName(name)
#define MSG(type, ...) ((type)objc_msgSend)(__VA_ARGS__)

static id dev, queue, event, cb, enc, last, pipes[512];
// Command buffers committed and not yet accounted for (gpu_reap, gpu_sync),
// oldest first; each is retained until then, so that the autorelease pool
// can be drained while they run.
static id *done;
static int done_cap;
static uint64_t ncommitted, naccounted;  // command buffers committed, and accounted for
enum { RESIDENT_SETS = 32 };
static struct { id object; uint64_t bytes; } resident[RESIDENT_SETS];
static uint32_t resident_count, resident_dirty;
static uint64_t resident_target;
typedef struct { id object; uint64_t bytes; uint32_t set; } ResidentAllocation;
static ResidentAllocation *resident_allocations;
static size_t resident_capacity, resident_used, resident_occupied;
static int ndone, current, profile;
static double prof_s[512];
static long prof_n[512];
static id used[1024];  // resources made resident in the current encoder
static unsigned long used_usage[1024];
static int nused;
static void report(void);
static void collect(void);
static id compute_encoder(id command);
static void stage_collect(void);
extern id MTLCommonCounterSetTimestamp;
static id stage_samples;
enum { STAGE_SAMPLES = 4096, STAGE_GROUPS = 2048 };
static int stage_count, stage_groups;
static struct { int kernel; uint32_t rows, cols; } stages[STAGE_SAMPLES / 2];
static struct { int kernel; uint32_t rows, cols; long count; double seconds; } stage_group[STAGE_GROUPS];
static uint32_t stage_rows, stage_cols;
static struct { int kind; id object; unsigned long offset, size; unsigned char bytes[4096]; } stage_args[32];
double gpu_busy, gpu_gap;
long gpu_gaps;

static int npipes;
static const char *names[512];
static uint64_t signaled;
static void *pool;
static SEL s_pipe, s_buf, s_bytes, s_use, s_dispatch;

static id str(const char *s) {
  return MSG(id (*)(id, SEL, const char *), (id)objc_getClass("NSString"),
             S("stringWithUTF8String:"), s);
}

static const char *describe(id err) {
  id d = MSG(id (*)(id, SEL), err, S("localizedDescription"));
  return MSG(const char *(*)(id, SEL), d, S("UTF8String"));
}

static id library, clock_queue;
static int spin_kernel = -1;
static double spin_s;  // seconds per iteration
static Buf spin_out;

// The GPU lowers its clock within milliseconds of idling, and the work after
// a wait runs up to 3 times slower. While decoding waits for expert reads,
// one simdgroup spins on a second queue for about `seconds`: it occupies a
// sliver of one core, and the clock stays up.
static id spin(uint32_t iters) {
  id cb = MSG(id (*)(id, SEL), clock_queue, S("commandBuffer"));
  id e = MSG(id (*)(id, SEL), cb, S("computeCommandEncoder"));
  MSG(void (*)(id, SEL, id), e, s_pipe, pipes[spin_kernel]);
  MSG(void (*)(id, SEL, const void *, unsigned long, unsigned long), e, s_bytes, &iters, 4ul, 0ul);
  MSG(void (*)(id, SEL, id, unsigned long, unsigned long), e, s_buf, (id)spin_out.obj, 0ul, 1ul);
  MSG(void (*)(id, SEL, Size3, Size3), e, s_dispatch, (Size3){1, 1, 1}, (Size3){32, 1, 1});
  MSG(void (*)(id, SEL), e, S("endEncoding"));
  MSG(void (*)(id, SEL), cb, S("commit"));
  return cb;
}

void gpu_keep_clock(double seconds) {
  if (spin_kernel < 0) {  // first use: the time per iteration
    spin_kernel = gpu_kernel("spin");
    spin_out = gpu_alloc(16);
    // Warm the dispatch first. Host submission/wait delays do not measure
    // the GPU loop, and including them makes later keep-alives too short.
    MSG(void (*)(id, SEL), spin(200000), S("waitUntilCompleted"));
    double fastest = 1e30;
    for (int i = 0; i < 3; i++) {
      id cb = spin(200000);
      MSG(void (*)(id, SEL), cb, S("waitUntilCompleted"));
      double t = MSG(double (*)(id, SEL), cb, S("GPUEndTime")) - MSG(double (*)(id, SEL), cb, S("GPUStartTime"));
      if (t > 0 && t < fastest) fastest = t;
    }
    if (fastest == 1e30) die("cannot measure the GPU clock-keeping kernel");
    spin_s = fastest / 200000;
  }
  spin((uint32_t)(seconds / spin_s));
}

// A pass yields with no command buffer being encoded: others encode in turn.
static void closed_for_yield(void) {
  if (enc) die("a pass yields with a command buffer being encoded");
}

void gpu_init(const char *source, size_t len) {
  memory_collect_with(collect);
  pass_before_yield = closed_for_yield;
  pool = objc_autoreleasePoolPush();
  dev = MTLCreateSystemDefaultDevice();
  if (!dev) die("no Metal device");
  resident_target = gpu_working_set() / 20;
  if (resident_target < (64ull << 20)) resident_target = 64ull << 20;
  const char *set_bytes = getenv("MYDS4_RESIDENCY_SET_BYTES");
  if (set_bytes) {
    char *end; errno = 0;
    resident_target = strtoull(set_bytes, &end, 10);
    if (!*set_bytes || *set_bytes == '-' || *end || errno)
      die("MYDS4_RESIDENCY_SET_BYTES must be an unsigned byte count");
  }
  queue = MSG(id (*)(id, SEL), dev, S("newCommandQueue"));
  clock_queue = MSG(id (*)(id, SEL), dev, S("newCommandQueue"));
  event = MSG(id (*)(id, SEL), dev, S("newSharedEvent"));
  id opts = MSG(id (*)(id, SEL), MSG(id (*)(id, SEL), (id)objc_getClass("MTLCompileOptions"),
                                     S("alloc")), S("init"));
  MSG(void (*)(id, SEL, long), opts, S("setMathMode:"), 0);  // IEEE: no fast math
  MSG(void (*)(id, SEL, unsigned long), opts, S("setLanguageVersion:"), (4ul << 16) | 1);  // Metal 4.1: tensors
  char *src = xmalloc(len + 1);
  memcpy(src, source, len);
  src[len] = 0;
  id err = 0;
  library = MSG(id (*)(id, SEL, id, id, id *), dev, S("newLibraryWithSource:options:error:"),
                str(src), opts, &err);
  if (!library) die("Metal compile: %s", describe(err));
  free(src);
  s_pipe = S("setComputePipelineState:");
  s_buf = S("setBuffer:offset:atIndex:");
  s_bytes = S("setBytes:length:atIndex:");
  s_use = S("useResource:usage:");
  s_dispatch = S("dispatchThreadgroups:threadsPerThreadgroup:");
  const char *setting = getenv("MYDS4_PROFILE");
  profile = setting ? (!strcmp(setting, "stages") ? 2 : 1) : 0;
  if (profile == 2) {
    if (!MSG(BOOL (*)(id, SEL, unsigned long), dev, S("supportsCounterSampling:"), 0ul))
      die("GPU stage counters are unavailable");
    id sets = MSG(id (*)(id, SEL), dev, S("counterSets")), selected = nil;
    unsigned long count = MSG(unsigned long (*)(id, SEL), sets, S("count"));
    for (unsigned long i = 0; i < count; i++) {
      id set = MSG(id (*)(id, SEL, unsigned long), sets, S("objectAtIndex:"), i);
      id name = MSG(id (*)(id, SEL), set, S("name"));
      if (MSG(BOOL (*)(id, SEL, id), name, S("isEqualToString:"), MTLCommonCounterSetTimestamp)) selected = set;
    }
    if (!selected) die("GPU timestamp counter set is unavailable");
    id descriptor = MSG(id (*)(id, SEL), MSG(id (*)(id, SEL),
        (id)objc_getClass("MTLCounterSampleBufferDescriptor"), S("alloc")), S("init"));
    MSG(void (*)(id, SEL, id), descriptor, S("setCounterSet:"), selected);
    MSG(void (*)(id, SEL, unsigned long), descriptor, S("setStorageMode:"), 0ul);
    MSG(void (*)(id, SEL, unsigned long), descriptor, S("setSampleCount:"), (unsigned long)STAGE_SAMPLES);
    id error = nil;
    stage_samples = MSG(id (*)(id, SEL, id, id *), dev, S("newCounterSampleBufferWithDescriptor:error:"), descriptor, &error);
    MSG(void (*)(id, SEL), descriptor, S("release"));
    if (!stage_samples) die("GPU timestamp buffer: %s", describe(error));
  }
  if (profile) atexit(report);
}

static id compute_encoder(id command) {
  if (profile != 2) return MSG(id (*)(id, SEL), command, S("computeCommandEncoder"));
  if (stage_count == STAGE_SAMPLES / 2) die("GPU stage sample buffer filled before synchronization");
  id pass = MSG(id (*)(id, SEL), (id)objc_getClass("MTLComputePassDescriptor"), S("computePassDescriptor"));
  id attachments = MSG(id (*)(id, SEL), pass, S("sampleBufferAttachments"));
  id attachment = MSG(id (*)(id, SEL, unsigned long), attachments, S("objectAtIndexedSubscript:"), 0ul);
  MSG(void (*)(id, SEL, id), attachment, S("setSampleBuffer:"), stage_samples);
  MSG(void (*)(id, SEL, unsigned long), attachment, S("setStartOfEncoderSampleIndex:"), (unsigned long)stage_count * 2);
  MSG(void (*)(id, SEL, unsigned long), attachment, S("setEndOfEncoderSampleIndex:"), (unsigned long)stage_count * 2 + 1);
  stages[stage_count++].kernel = -1;
  return MSG(id (*)(id, SEL, id), command, S("computeCommandEncoderWithDescriptor:"), pass);
}

static void stage_collect(void) {
  if (!stage_count) return;
  struct { unsigned long location, length; } range = {0, (unsigned long)stage_count * 2};
  id data = MSG(id (*)(id, SEL, typeof(range)), stage_samples, S("resolveCounterRange:"), range);
  const uint64_t *ticks = MSG(const void *(*)(id, SEL), data, S("bytes"));
  for (int i = 0; i < stage_count; i++) {
    if (stages[i].kernel < 0) continue;
    uint64_t a = ticks[i * 2], b = ticks[i * 2 + 1];
    if (a == UINT64_MAX || b == UINT64_MAX || b < a) die("invalid GPU stage timestamp");
    int at;
    for (at = 0; at < stage_groups; at++)
      if (stage_group[at].kernel == stages[i].kernel && stage_group[at].rows == stages[i].rows &&
          stage_group[at].cols == stages[i].cols) break;
    if (at == stage_groups) {
      if (at == STAGE_GROUPS) die("too many GPU profiling shapes");
      stage_group[at].kernel = stages[i].kernel;
      stage_group[at].rows = stages[i].rows; stage_group[at].cols = stages[i].cols;
      stage_groups++;
    }
    stage_group[at].count++;
    // Metal resolves Apple GPU stage timestamps into nanoseconds.
    stage_group[at].seconds += (b - a) * 1e-9;
  }
  stage_count = 0;
}

int gpu_kernel(const char *name) {
  for (int i = 0; i < npipes; i++)
    if (!strcmp(names[i], name)
    ) return i;
  id fn = MSG(id (*)(id, SEL, id), library, S("newFunctionWithName:"), str(name));
  if (!fn) die("no kernel %s", name);
  id err = 0;
  id p;
    p = MSG(id (*)(id, SEL, id, id *), dev, S("newComputePipelineStateWithFunction:error:"), fn, &err);
  if (!p) die("pipeline %s: %s", name, describe(err));
  names[npipes] = name;
  pipes[npipes] = p;
  return npipes++;
}


int gpu_threads(int k) {
  return (int)MSG(unsigned long (*)(id, SEL), pipes[k], S("maxTotalThreadsPerThreadgroup"));
}

size_t gpu_allocation_size(size_t size) {
  size_t page = (size_t)getpagesize();
  if (size > SIZE_MAX - page + 1) die("GPU allocation size overflow");
  return (size + page - 1) / page * page;
}

static Buf allocate(size_t size, int kv, MemoryAdmission admission) {
  Buf b = {0};
  b.allocated = gpu_allocation_size(size);
  b.kv = kv;
  if (kv) memory_kv_change(b.allocated, 1);
  else {
    b.memory = memory_current() ? memory_current() : memory_shared();
    if (!memory_claim(b.memory, b.allocated, admission)) return (Buf){0};
  }
  b.obj = MSG(id (*)(id, SEL, unsigned long, unsigned long), dev,
              S("newBufferWithLength:options:"), size, 0ul);
  if (!b.obj) die("cannot allocate %zu bytes on the GPU", size);
  b.ptr = MSG(void *(*)(id, SEL), (id)b.obj, S("contents"));
  b.gpu = MSG(uint64_t (*)(id, SEL), (id)b.obj, S("gpuAddress"));
  b.size = size;
  return b;
}

Buf gpu_try_new(size_t size, int admission) { return allocate(size, 0, (MemoryAdmission)admission); }
Buf gpu_new(size_t size) {
  Buf b = allocate(size, 0, MEMORY_REQUIRED);
  if (!b.obj) die("model-memory budget cannot accommodate %zu bytes", size);
  return b;
}

Buf gpu_alloc(size_t size) {
  Buf b = gpu_new(size);
  gpu_resident(&b, 1);
  gpu_residency_commit();
  return b;
}

Buf gpu_alloc_kv(size_t size) {
  Buf b = allocate(size, 1, MEMORY_REQUIRED);
  gpu_resident(&b, 1);
  gpu_residency_commit();
  return b;
}

void gpu_free(Buf *b) {
  if (b->obj) {
    gpu_resident(b, 0);
    MSG(void (*)(id, SEL), (id)b->obj, S("release"));
    if (b->kv) memory_kv_change(b->allocated, 0);
    else if (b->memory) memory_release(b->memory, b->allocated);
  }
  memset(b, 0, sizeof *b);
}

// The engine thread owns this table. Buffer views share an allocation, so
// membership follows the Metal object rather than a view's address or size.
static size_t resident_find(id object) {
  size_t at = (((uintptr_t)object >> 4) * 11400714819323198485ull) & (resident_capacity - 1);
  size_t empty = SIZE_MAX;
  while (resident_allocations[at].object) {
    if (resident_allocations[at].object == object) return at;
    if (resident_allocations[at].object == (id)1 && empty == SIZE_MAX) empty = at;
    at = (at + 1) & (resident_capacity - 1);
  }
  return empty != SIZE_MAX ? empty : at;
}

static void resident_rehash(size_t capacity) {
  ResidentAllocation *old = resident_allocations;
  size_t old_capacity = resident_capacity;
  resident_capacity = capacity;
  resident_allocations = xmalloc(capacity * sizeof *resident_allocations);
  for (size_t i = 0; i < old_capacity; i++)
    if ((uintptr_t)old[i].object > 1) resident_allocations[resident_find(old[i].object)] = old[i];
  free(old);
  resident_occupied = resident_used;
}

static uint32_t resident_set(uint64_t bytes) {
  uint32_t smallest = 0;
  for (uint32_t i = 0; i < resident_count; i++) {
    uint64_t used = resident[i].bytes;
    if (!resident_target || !used || (used <= resident_target && bytes <= resident_target - used)) return i;
    if (used < resident[smallest].bytes) smallest = i;
  }
  // Metal permits at most 32 sets per queue. The byte target is soft: an
  // oversized allocation or a full queue uses the least occupied set.
  if (resident_count == RESIDENT_SETS) return smallest;
  id descriptor = MSG(id (*)(id, SEL), MSG(id (*)(id, SEL),
      (id)objc_getClass("MTLResidencySetDescriptor"), S("alloc")), S("init"));
  MSG(void (*)(id, SEL, unsigned long), descriptor, S("setInitialCapacity:"), 1024ul);
  id error = nil;
  id object = MSG(id (*)(id, SEL, id, id *), dev, S("newResidencySetWithDescriptor:error:"), descriptor, &error);
  MSG(void (*)(id, SEL), descriptor, S("release"));
  if (!object) die("residency set: %s", describe(error));
  MSG(void (*)(id, SEL, id), queue, S("addResidencySet:"), object);
  resident[resident_count].object = object;
  return resident_count++;
}

void gpu_resident(const Buf *b, int on) {
  if (!resident_capacity) resident_rehash(64);
  id object = (id)b->obj;
  size_t at = resident_find(object);
  if (on) {
    if (resident_allocations[at].object == object) return;
    if ((resident_occupied + 1) * 4 >= resident_capacity * 3) {
      resident_rehash(resident_used * 2 >= resident_capacity ? resident_capacity * 2 : resident_capacity);
      at = resident_find(object);
    }
    uint64_t bytes = MSG(unsigned long (*)(id, SEL), object, S("allocatedSize"));
    uint32_t set = resident_set(bytes);
    MSG(void (*)(id, SEL, id), resident[set].object, S("addAllocation:"), object);
    if (!resident_allocations[at].object) resident_occupied++;
    resident_allocations[at] = (ResidentAllocation){object, bytes, set};
    resident_used++;
    resident[set].bytes += bytes;
    resident_dirty |= 1u << set;
  } else if (resident_allocations[at].object == object) {
    ResidentAllocation *allocation = &resident_allocations[at];
    uint32_t set = allocation->set;
    MSG(void (*)(id, SEL, id), resident[set].object, S("removeAllocation:"), object);
    resident[set].bytes -= allocation->bytes;
    allocation->object = (id)1;
    resident_used--;
    resident_dirty |= 1u << set;
  }
}

void gpu_residency_commit(void) {
  if (!resident_dirty) return;
  for (uint32_t i = 0; i < resident_count; i++)
    if (resident_dirty & (1u << i)) MSG(void (*)(id, SEL), resident[i].object, S("commit"));
  resident_dirty = 0;
}

void gpu_residency_info(GpuResidencyInfo *info) {
  *info = (GpuResidencyInfo){.sets = resident_count, .target_set = resident_target};
  for (uint32_t s = 0; s < resident_count; s++) {
    id allocations = MSG(id (*)(id, SEL), resident[s].object, S("allAllocations"));
    unsigned long count = MSG(unsigned long (*)(id, SEL), allocations, S("count"));
    info->allocations += count;
    uint64_t bytes = 0;
    for (unsigned long i = 0; i < count; i++) {
      id allocation = MSG(id (*)(id, SEL, unsigned long), allocations, S("objectAtIndex:"), i);
      bytes += MSG(unsigned long (*)(id, SEL), allocation, S("allocatedSize"));
    }
    info->bytes += bytes;
    if (bytes > info->largest_set) info->largest_set = bytes;
  }
}

uint64_t gpu_working_set(void) {
  return MSG(uint64_t (*)(id, SEL), dev, S("recommendedMaxWorkingSetSize"));
}

// Command buffers run in the order they are opened (enqueued), whatever the
// order in which they are encoded and committed. `reserved` counts those
// being encoded and not committed yet, detached ones included: a buffer
// committed after one in the queue runs only once that one is committed, so
// the thread that will commit it must not wait for the later buffer.
static int reserved;

void *gpu_open(void) {
  id c = MSG(id (*)(id, SEL), queue, S("commandBufferWithUnretainedReferences"));
  MSG(void (*)(id, SEL), c, S("enqueue"));
  reserved++;
  return c;
}

void gpu_encode(void *c) {
  if (enc) MSG(void (*)(id, SEL), enc, S("endEncoding"));
  cb = c;
  enc = compute_encoder(cb);
  nused = 0;
  if (profile == 2) memset(stage_args, 0, sizeof stage_args);
}

void gpu_begin(void) { gpu_encode(gpu_open()); }

// As gpu_encode, but the dispatches may run at the same time unless a barrier
// lies between them.
void gpu_encode_concurrent(void *c) {
  if (enc) MSG(void (*)(id, SEL), enc, S("endEncoding"));
  cb = c;
  enc = MSG(id (*)(id, SEL, unsigned long), cb, S("computeCommandEncoderWithDispatchType:"), 1ul);  // MTLDispatchTypeConcurrent
  nused = 0;
}

void gpu_barrier(void) {
  MSG(void (*)(id, SEL, unsigned long), enc, S("memoryBarrierWithScope:"), 1ul);  // MTLBarrierScopeBuffers
}

void gpu_barrier_on(const Buf *const *buffers, int count) {
  id resources[count];
  for (int i = 0; i < count; i++) resources[i] = (id)buffers[i]->obj;
  MSG(void (*)(id, SEL, const id *, unsigned long), enc, S("memoryBarrierWithResources:count:"), resources, (unsigned long)count);
}

// A command buffer that joins the queue only when gpu_submit_detached commits
// it, so that buffers committed after its opening may run before it. It stays
// retained until then, across any drain of the autorelease pool.
void *gpu_open_detached(void) {
  id c = MSG(id (*)(id, SEL), queue, S("commandBufferWithUnretainedReferences"));
  MSG(id (*)(id, SEL), c, S("retain"));
  reserved++;
  return c;
}

// Ends the encoding of the current command buffer without committing it.
void *gpu_close(void) {
  MSG(void (*)(id, SEL), enc, S("endEncoding"));
  enc = 0;
  return cb;
}

static void committed(id c) {
  if (ndone == done_cap) {
    done_cap = done_cap ? 2 * done_cap : 256;
    done = realloc(done, (size_t)done_cap * sizeof *done);
    if (!done) die("out of memory");
  }
  done[ndone++] = MSG(id (*)(id, SEL), c, S("retain"));
  ncommitted++;
  last = c;
  reserved--;
}

void gpu_submit(void *c) {
  MSG(void (*)(id, SEL), (id)c, S("commit"));
  committed(c);
}

void gpu_submit_detached(void *c) {
  gpu_submit(c);
  MSG(void (*)(id, SEL), (id)c, S("release"));
}

void gpu_kern(int k) {
  current = k;
  if (profile == 2) stage_rows = stage_cols = 0;
  MSG(void (*)(id, SEL, id), enc, s_pipe, pipes[k]);
}

void gpu_buf(int idx, const Buf *b, size_t off) {
  MSG(void (*)(id, SEL, id, unsigned long, unsigned long), enc, s_buf, (id)b->obj,
      b->off + off, (unsigned long)idx);
  if (profile == 2) {
    stage_args[idx].kind = 1; stage_args[idx].object = (id)b->obj; stage_args[idx].offset = b->off + off;
  }
}

void gpu_bytes(int idx, const void *p, size_t n) {
  MSG(void (*)(id, SEL, const void *, unsigned long, unsigned long), enc, s_bytes, p, n,
      (unsigned long)idx);
  if (profile == 2) {
    stage_args[idx].kind = 2; stage_args[idx].size = n; memcpy(stage_args[idx].bytes, p, n);
    if (idx == 4 && n == 16 && (!strncmp(names[current], "mv1_", 4) || !strncmp(names[current], "mvr", 3) ||
                               !strncmp(names[current], "mv_split_", 9))) {
      const uint32_t *dims = p; stage_rows = dims[0]; stage_cols = dims[1];
    }
  }
}

void gpu_use(const Buf *b) {
  MSG(void (*)(id, SEL, id, unsigned long), enc, s_use, (id)b->obj, 1ul);  // read
  if (nused < (int)(sizeof used / sizeof *used)) used_usage[nused] = 1, used[nused++] = b->obj;
}

void gpu_use_rw(const Buf *b) {
  MSG(void (*)(id, SEL, id, unsigned long), enc, s_use, (id)b->obj, 3ul);
  if (nused < (int)(sizeof used / sizeof *used)) used_usage[nused] = 3, used[nused++] = b->obj;
}

void gpu_copies(const GpuCopy *copies, int count) {
  if (!count) return;
  MSG(void (*)(id, SEL), enc, S("endEncoding"));
  id blit = MSG(id (*)(id, SEL), cb, S("blitCommandEncoder"));
  for (int i = 0; i < count; i++) {
    const GpuCopy *c = copies + i;
    if (c->from_offset > c->from->size || c->to_offset > c->to->size ||
        c->bytes > c->from->size - c->from_offset || c->bytes > c->to->size - c->to_offset)
      die("GPU copy exceeds a buffer");
    MSG(void (*)(id, SEL, id, unsigned long, id, unsigned long, unsigned long), blit,
        S("copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:"), (id)c->from->obj,
        c->from->off + c->from_offset, (id)c->to->obj, c->to->off + c->to_offset, c->bytes);
  }
  MSG(void (*)(id, SEL), blit, S("endEncoding"));
  enc = compute_encoder(cb);
  nused = 0;
  if (profile == 2) memset(stage_args, 0, sizeof stage_args);
}

void gpu_go(uint32_t gx, uint32_t gy, uint32_t gz, uint32_t tx, uint32_t ty, uint32_t tz) {
  Size3 g = {gx, gy, gz}, t = {tx, ty, tz};
  MSG(void (*)(id, SEL, Size3, Size3), enc, s_dispatch, g, t);
  if (profile == 2) {
    stages[stage_count - 1].kernel = current;
    stages[stage_count - 1].rows = stage_rows; stages[stage_count - 1].cols = stage_cols;
    MSG(void (*)(id, SEL), enc, S("endEncoding"));
    enc = compute_encoder(cb);
    for (int i = 0; i < nused; i++) MSG(void (*)(id, SEL, id, unsigned long), enc, s_use, used[i], used_usage[i]);
    for (unsigned long i = 0; i < 32; i++) {
      if (stage_args[i].kind == 1)
        MSG(void (*)(id, SEL, id, unsigned long, unsigned long), enc, s_buf, stage_args[i].object, stage_args[i].offset, i);
      else if (stage_args[i].kind == 2)
        MSG(void (*)(id, SEL, const void *, unsigned long, unsigned long), enc, s_bytes,
            stage_args[i].bytes, stage_args[i].size, i);
    }
    MSG(void (*)(id, SEL, id), enc, s_pipe, pipes[current]);
  } else if (profile) {  // each dispatch alone in a command buffer, timed
    MSG(void (*)(id, SEL), enc, S("endEncoding"));
    MSG(void (*)(id, SEL), cb, S("commit"));
    reserved--;
    MSG(void (*)(id, SEL), cb, S("waitUntilCompleted"));
    prof_s[current] += MSG(double (*)(id, SEL), cb, S("GPUEndTime")) - MSG(double (*)(id, SEL), cb, S("GPUStartTime"));
    prof_n[current]++;
    int n = nused;
    gpu_encode(gpu_open());
    for (int i = 0; i < n; i++) MSG(void (*)(id, SEL, id, unsigned long), enc, s_use, used[i], used_usage[i]);
    nused = n;
    gpu_kern(current);
  }
}

static void report(void) {
  if (profile == 2) {
    double total = 0;
    for (int i = 0; i < stage_groups; i++) {
      fprintf(stderr, "STAGE %s rows=%u cols=%u calls=%ld total_ms=%.4f mean_us=%.4f\n",
              names[stage_group[i].kernel], stage_group[i].rows, stage_group[i].cols, stage_group[i].count,
              stage_group[i].seconds * 1e3, stage_group[i].seconds / stage_group[i].count * 1e6);
      total += stage_group[i].seconds;
    }
    fprintf(stderr, "STAGE total_ms=%.4f\n", total * 1e3);
    return;
  }
  double total = 0;
  for (int i = 0; i < npipes; i++) total += prof_s[i];
  fprintf(stderr, "kernel           calls   total ms   mean us\n");
  for (int i = 0; i < npipes; i++)
    if (prof_n[i])
      fprintf(stderr, "%-15s %6ld %10.2f %9.1f\n", names[i], prof_n[i], prof_s[i] * 1e3, prof_s[i] / prof_n[i] * 1e6);
  fprintf(stderr, "total %.2f ms\n", total * 1e3);
}

int gpu_profiling(void) { return profile; }

void gpu_profile_reset(void) {
  memset(prof_s, 0, sizeof prof_s);
  memset(prof_n, 0, sizeof prof_n);
  memset(stage_group, 0, sizeof stage_group);
  stage_groups = 0;
}

uint64_t gpu_signal(void) {
  MSG(void (*)(id, SEL), enc, S("endEncoding"));
  MSG(void (*)(id, SEL, id, uint64_t), cb, S("encodeSignalEvent:value:"), event, ++signaled);
  enc = compute_encoder(cb);
  nused = 0;
  if (profile == 2) memset(stage_args, 0, sizeof stage_args);
  return signaled;
}

void gpu_commit(void) {
  MSG(void (*)(id, SEL), enc, S("endEncoding"));
  MSG(void (*)(id, SEL), cb, S("commit"));
  committed(cb);
  enc = 0;
}

// The GPU signals the shared event's values in order, each after the command
// buffers before it. A value that a minute does not bring means that a
// command buffer failed or the GPU stopped, and the results the caller waits
// for are not there: a failed buffer's error, or this, ends the program.
static void unsignaled(uint64_t v) {
  for (int i = 0; i < ndone; i++)
    if (MSG(long (*)(id, SEL), done[i], S("status")) == 5)  // MTLCommandBufferStatusError
      die("GPU: %s", describe(MSG(id (*)(id, SEL), done[i], S("error"))));
  die("GPU: value %llu not signaled in 60 s", (unsigned long long)v);
}

// What a pass waits for runs outside it, where `event` is not the pass's: the
// wait carries its event.
typedef struct { id event; uint64_t value; } SignalWait;

static int signal_ready(void *context) {
  SignalWait *w = context;
  return MSG(uint64_t (*)(id, SEL), w->event, S("signaledValue")) >= w->value;
}

static void signal_block(void *context) {
  SignalWait *w = context;
  if (!MSG(BOOL (*)(id, SEL, uint64_t, uint64_t), w->event, S("waitUntilSignaledValue:timeoutMS:"), w->value, 60000ull))
    unsignaled(w->value);
}

void gpu_wait_signal(uint64_t v) {
  SignalWait wait = {event, v};
  if (pass_current()) pass_wait(&(PassWait){signal_ready, signal_block, &wait});
  else signal_block(&wait);
}

uint64_t gpu_signal_value(void) {
  return MSG(uint64_t (*)(id, SEL), event, S("signaledValue"));
}

// Each pass numbers its signals on its own event: its command buffers may be
// committed after a buffer another pass encoded later, and one event's values
// fire in queue order.
static const PassVariable signal_variables[] = {{&event, sizeof event}, {&signaled, sizeof signaled}};

const PassVariable *gpu_pass_variables(int *count) {
  *count = (int)(sizeof signal_variables / sizeof *signal_variables);
  return signal_variables;
}

Pass *pass_new_model(void (*body)(void *), void *argument, const PassVariable *variables, int count, void (*resumed)(void)) {
  int gpu_count;
  const PassVariable *gpu = gpu_pass_variables(&gpu_count);
  PassVariable all[count + gpu_count];
  memcpy(all, variables, (size_t)count * sizeof *all);
  memcpy(all + count, gpu, (size_t)gpu_count * sizeof *all);
  Pass *pass = pass_new(body, argument, 64u << 20);
  pass_variables(pass, all, count + gpu_count);
  pass_on_resume(pass, resumed);
  return pass;
}

void gpu_pass_begin(void) {
  event = MSG(id (*)(id, SEL), dev, S("newSharedEvent"));
  signaled = 0;
}

void gpu_pass_end(void) { MSG(void (*)(id, SEL), event, S("release")); }

static int by_start(const void *a, const void *b) {
  double x = ((const double *)a)[0], y = ((const double *)b)[0];
  return (x > y) - (x < y);
}

typedef struct { Buf b; uint64_t after; } Deferred;  // freed once `after` command buffers are accounted for
static Deferred *deferred;
static int ndeferred, deferred_cap;

static void free_deferred(void) {
  int kept = 0;
  for (int i = 0; i < ndeferred; i++) {
    if (deferred[i].after > naccounted) deferred[kept++] = deferred[i];
    else gpu_free(&deferred[i].b);
  }
  ndeferred = kept;
}

// As gpu_free, for a buffer that work committed or still being encoded may
// use: it stays allocated and charged until that work has finished.
void gpu_free_later(Buf *b) {
  uint64_t after = ncommitted + (uint64_t)reserved;
  if (!b->obj || after <= naccounted) { gpu_free(b); return; }
  if (ndeferred == deferred_cap) {
    deferred_cap = deferred_cap ? 2 * deferred_cap : 64;
    deferred = realloc(deferred, (size_t)deferred_cap * sizeof *deferred);
    if (!deferred) die("out of memory");
  }
  deferred[ndeferred++] = (Deferred){*b, after};
  memset(b, 0, sizeof *b);
}

// The first `count` buffers of `done`, all finished: their errors and their
// time. gpu_busy grows by the union of their execution intervals; gpu_gap by
// the time between them, and gpu_gaps by the gaps longer than 100 us.
static double accounted_end;

static void account(int count) {
  if (!count) return;
  double (*iv)[2] = xmalloc((size_t)count * sizeof *iv);
  for (int i = 0; i < count; i++) {
    id c = done[i];
    if (MSG(long (*)(id, SEL), c, S("status")) == 5)  // MTLCommandBufferStatusError
      die("GPU: %s", describe(MSG(id (*)(id, SEL), c, S("error"))));
    iv[i][0] = MSG(double (*)(id, SEL), c, S("GPUStartTime"));
    iv[i][1] = MSG(double (*)(id, SEL), c, S("GPUEndTime"));
    if (c == last) last = 0;
    MSG(void (*)(id, SEL), c, S("release"));
  }
  qsort(iv, (size_t)count, sizeof iv[0], by_start);
  double end = accounted_end;
  for (int i = 0; i < count; i++) {
    double a = iv[i][0] > end ? iv[i][0] : end;
    if (end && iv[i][0] > end) gpu_gap += iv[i][0] - end, gpu_gaps += iv[i][0] - end > 100e-6;
    if (iv[i][1] > a) gpu_busy += iv[i][1] - a;
    if (iv[i][1] > end) end = iv[i][1];
  }
  accounted_end = end;
  free(iv);
  ndone -= count;
  memmove(done, done + count, (size_t)ndone * sizeof *done);
  naccounted += (uint64_t)count;
  free_deferred();
}

// Accounts for the command buffers that have finished, oldest first, without
// waiting for the others.
void gpu_reap(void) {
  int n = 0;
  while (n < ndone && MSG(long (*)(id, SEL), done[n], S("status")) >= 4) n++;  // completed, or error
  account(n);
}

static int sync_ready(void *context) { return !last || MSG(long (*)(id, SEL), last, S("status")) >= 4; }
static void sync_block(void *context) { gpu_sync(); }

// Waits for the last committed command buffer; with it, all opened before.
void gpu_sync(void) {
  if (pass_current()) {
    pass_wait(&(PassWait){sync_ready, sync_block, NULL});
    gpu_reap();
    return;
  }
  if (!last) { accounted_end = 0; return; }
  MSG(void (*)(id, SEL), last, S("waitUntilCompleted"));
  for (int i = 0; i < ndone; i++) MSG(void (*)(id, SEL), done[i], S("waitUntilCompleted"));
  account(ndone);
  accounted_end = 0;
  last = 0;
  if (profile == 2) stage_collect();
  objc_autoreleasePoolPop(pool);
  pool = objc_autoreleasePoolPush();
}

void gpu_drain(void) {
  if (!dev) return;
  gpu_sync();
  gpu_residency_commit();
  void *scope = objc_autoreleasePoolPush();
  id fence = MSG(id (*)(id, SEL), clock_queue, S("commandBuffer"));
  MSG(void (*)(id, SEL), fence, S("commit"));
  MSG(void (*)(id, SEL), fence, S("waitUntilCompleted"));
  if (MSG(long (*)(id, SEL), fence, S("status")) == 5)
    die("GPU clock queue: %s", describe(MSG(id (*)(id, SEL), fence, S("error"))));
  objc_autoreleasePoolPop(scope);
}

// Residency updates can retain released buffers in the autorelease pool.
// Drain it once after an eviction batch, before admitting replacement data.
// Command buffers that are still running are retained (see `done`), so this
// need not wait for them. While a command buffer is being encoded its objects
// are in the pool: the pass's own gpu_sync drains it then.
static void collect(void) {
  gpu_residency_commit();
  if (reserved) return;
  gpu_reap();
  objc_autoreleasePoolPop(pool);
  pool = objc_autoreleasePoolPush();
}
