// Metal through the Objective-C runtime, so that the host stays in C. Each
// message is sent through objc_msgSend cast to the method's exact type.
#include <objc/message.h>
#include <objc/runtime.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>

#include "myds4.h"
#include "memory.h"

typedef struct { unsigned long w, h, d; } Size3;
extern id MTLCreateSystemDefaultDevice(void);
extern void *objc_autoreleasePoolPush(void);
extern void objc_autoreleasePoolPop(void *);

#define S(name) sel_registerName(name)
#define MSG(type, ...) ((type)objc_msgSend)(__VA_ARGS__)

static id dev, queue, event, cb, enc, last, resident, pipes[128], done[4096];
static int resident_dirty;
static int ndone, current, profile;
static double prof_s[128];
static long prof_n[128];
static id used[1024];  // resources made resident in the current encoder
static unsigned long used_usage[1024];
static int nused;
static void report(void);
static void collect(void);
double gpu_busy, gpu_gap;
long gpu_gaps;

static int npipes;
static const char *names[128];
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

void gpu_init(const char *source, size_t len) {
  memory_collect_with(collect);
  pool = objc_autoreleasePoolPush();
  dev = MTLCreateSystemDefaultDevice();
  if (!dev) die("no Metal device");
  queue = MSG(id (*)(id, SEL), dev, S("newCommandQueue"));
  // Every buffer stays resident through one set attached to the queue
  // (gpu_resident takes buffers out), so command buffers carry no residency.
  id rd = MSG(id (*)(id, SEL), MSG(id (*)(id, SEL), (id)objc_getClass("MTLResidencySetDescriptor"), S("alloc")), S("init"));
  MSG(void (*)(id, SEL, unsigned long), rd, S("setInitialCapacity:"), 32768ul);
  id err0 = 0;
  resident = MSG(id (*)(id, SEL, id, id *), dev, S("newResidencySetWithDescriptor:error:"), rd, &err0);
  if (!resident) die("residency set: %s", describe(err0));
  MSG(void (*)(id, SEL, id), queue, S("addResidencySet:"), resident);
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
  profile = getenv("MYDS4_PROFILE") != 0;
  if (profile) atexit(report);
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

void gpu_resident(const Buf *b, int on) {
  MSG(void (*)(id, SEL, id), resident, on ? S("addAllocation:") : S("removeAllocation:"), (id)b->obj);
  resident_dirty = 1;
}

void gpu_residency_commit(void) {
  if (!resident_dirty) return;
  MSG(void (*)(id, SEL), resident, S("commit"));
  resident_dirty = 0;
}

uint64_t gpu_working_set(void) {
  return MSG(uint64_t (*)(id, SEL), dev, S("recommendedMaxWorkingSetSize"));
}

// Command buffers run in the order they are opened (enqueued), whatever the
// order in which they are encoded and committed.
void *gpu_open(void) {
  id c = MSG(id (*)(id, SEL), queue, S("commandBufferWithUnretainedReferences"));
  MSG(void (*)(id, SEL), c, S("enqueue"));
  return c;
}

void gpu_encode(void *c) {
  if (enc) MSG(void (*)(id, SEL), enc, S("endEncoding"));
  cb = c;
  enc = MSG(id (*)(id, SEL), cb, S("computeCommandEncoder"));
  nused = 0;
}

void gpu_begin(void) { gpu_encode(gpu_open()); }

// A command buffer that joins the queue only when gpu_submit commits it, so
// that buffers committed after its opening may run before it.

// Ends the encoding of the current command buffer without committing it.
void *gpu_close(void) {
  MSG(void (*)(id, SEL), enc, S("endEncoding"));
  enc = 0;
  return cb;
}

void gpu_submit(void *c) {
  MSG(void (*)(id, SEL), (id)c, S("commit"));
  if (ndone < 4096) done[ndone++] = c;
  last = c;
}

void gpu_kern(int k) {
  current = k;
  MSG(void (*)(id, SEL, id), enc, s_pipe, pipes[k]);
}

void gpu_buf(int idx, const Buf *b, size_t off) {
  MSG(void (*)(id, SEL, id, unsigned long, unsigned long), enc, s_buf, (id)b->obj,
      b->off + off, (unsigned long)idx);
}

void gpu_bytes(int idx, const void *p, size_t n) {
  MSG(void (*)(id, SEL, const void *, unsigned long, unsigned long), enc, s_bytes, p, n,
      (unsigned long)idx);
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
  enc = MSG(id (*)(id, SEL), cb, S("computeCommandEncoder"));
  nused = 0;
}

void gpu_go(uint32_t gx, uint32_t gy, uint32_t gz, uint32_t tx, uint32_t ty, uint32_t tz) {
  Size3 g = {gx, gy, gz}, t = {tx, ty, tz};
  MSG(void (*)(id, SEL, Size3, Size3), enc, s_dispatch, g, t);
  if (profile) {  // each dispatch alone in a command buffer, timed
    MSG(void (*)(id, SEL), enc, S("endEncoding"));
    MSG(void (*)(id, SEL), cb, S("commit"));
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
}

uint64_t gpu_signal(void) {
  MSG(void (*)(id, SEL), enc, S("endEncoding"));
  MSG(void (*)(id, SEL, id, uint64_t), cb, S("encodeSignalEvent:value:"), event, ++signaled);
  enc = MSG(id (*)(id, SEL), cb, S("computeCommandEncoder"));
  nused = 0;
  return signaled;
}

void gpu_commit(void) {
  MSG(void (*)(id, SEL), enc, S("endEncoding"));
  MSG(void (*)(id, SEL), cb, S("commit"));
  if (ndone < 4096) done[ndone++] = cb;
  last = cb;
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

void gpu_wait_signal(uint64_t v) {
  if (!MSG(BOOL (*)(id, SEL, uint64_t, uint64_t), event, S("waitUntilSignaledValue:timeoutMS:"), v, 60000ull))
    unsignaled(v);
}

uint64_t gpu_signal_value(void) {
  return MSG(uint64_t (*)(id, SEL), event, S("signaledValue"));
}

static int by_start(const void *a, const void *b) {
  double x = ((const double *)a)[0], y = ((const double *)b)[0];
  return (x > y) - (x < y);
}

// Waits for the last committed command buffer; with it, all opened before.
// gpu_busy grows by the union of their execution intervals; gpu_gap by the
// time between them, and gpu_gaps by the gaps longer than 100 us.
void gpu_sync(void) {
  if (!last) return;
  MSG(void (*)(id, SEL), last, S("waitUntilCompleted"));
  static double iv[4096][2];
  for (int i = 0; i < ndone; i++) {
    id c = done[i];
    MSG(void (*)(id, SEL), c, S("waitUntilCompleted"));
    if (MSG(long (*)(id, SEL), c, S("status")) == 5)  // MTLCommandBufferStatusError
      die("GPU: %s", describe(MSG(id (*)(id, SEL), c, S("error"))));
    iv[i][0] = MSG(double (*)(id, SEL), c, S("GPUStartTime"));
    iv[i][1] = MSG(double (*)(id, SEL), c, S("GPUEndTime"));
  }
  qsort(iv, ndone, sizeof iv[0], by_start);
  double end = 0;
  for (int i = 0; i < ndone; i++) {
    double a = iv[i][0] > end ? iv[i][0] : end;
    if (end && iv[i][0] > end) gpu_gap += iv[i][0] - end, gpu_gaps += iv[i][0] - end > 100e-6;
    if (iv[i][1] > a) gpu_busy += iv[i][1] - a;
    if (iv[i][1] > end) end = iv[i][1];
  }
  ndone = 0;
  last = 0;
  objc_autoreleasePoolPop(pool);
  pool = objc_autoreleasePoolPush();
}

// Residency updates can retain released buffers in the autorelease pool.
// Drain it once after an eviction batch, before admitting replacement data.
static void collect(void) {
  gpu_residency_commit();
  if (last) gpu_sync();
  else {
    objc_autoreleasePoolPop(pool);
    pool = objc_autoreleasePoolPush();
  }
}
