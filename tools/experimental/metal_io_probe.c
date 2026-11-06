// Compare aligned uncached pread with public Metal I/O on existing expert
// spans. Read results must match byte for byte. No model files are written.
#include <dispatch/dispatch.h>
#include <fcntl.h>
#include <limits.h>
#include <mach/mach.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include "../../src/engine/gpu.c"

enum { EXP = 6, SPANS = 2 * EXP, JOBS = 128 };
#define ALIGNMENT UINT64_C(16384)
#define READ_PIECE (UINT64_C(2) << 20)
#define REGION_BYTES(n) (((n) + 3 * ALIGNMENT - 1) / ALIGNMENT * ALIGNMENT)
typedef struct { int fd; uint64_t off, bytes, dst; int expert; id handle; } Span;
typedef struct { int fd; uint64_t off, bytes; uint8_t *dst; dispatch_group_t group; } ReadJob;
typedef struct { double ms, user_ms, system_ms; int64_t free_change, file_change; } Measure;
static Buf shared_buffers[EXP], private_buffers[EXP];
static Span spans[SPANS];
static ReadJob jobs[JOBS];
static int nspans, njobs;
static uint64_t read_bytes;
static id io_queue;
static struct { int fd; id handle; } handles[256];
static int nhandles;

static vm_statistics64_data_t memory_now(void) {
  vm_statistics64_data_t v = {0};
  mach_msg_type_number_t n = HOST_VM_INFO64_COUNT;
  host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t)&v, &n);
  return v;
}

static Buf private_buffer(size_t size) {
  Buf b = {.size = size};
  // MTLStorageModePrivate (2), shifted by MTLResourceStorageModeShift (4).
  b.obj = MSG(id (*)(id, SEL, unsigned long, unsigned long), dev, S("newBufferWithLength:options:"), size, 32ul);
  if (!b.obj) die("cannot allocate private I/O destination");
  b.gpu = MSG(uint64_t (*)(id, SEL), (id)b.obj, S("gpuAddress"));
  gpu_resident(&b, 1), gpu_residency_commit();
  return b;
}

static id file_handle(int fd) {
  for (int i = 0; i < nhandles; i++) if (handles[i].fd == fd) return handles[i].handle;
  if (nhandles == 256) die("too many probe file handles");
  char path[PATH_MAX];
  if (fcntl(fd, F_GETPATH, path)) die("cannot resolve source descriptor");
  id url = MSG(id (*)(id, SEL, id), (id)objc_getClass("NSURL"), S("fileURLWithPath:"), str(path));
  id error = 0;
  id h = MSG(id (*)(id, SEL, id, id *), dev, S("newIOHandleWithURL:error:"), url, &error);
  if (!h) die("Metal I/O file handle: %s", describe(error));
  handles[nhandles].fd = fd, handles[nhandles++].handle = h;
  return h;
}

static void add_span(Tensor t, uint64_t bytes, int expert, uint64_t dst) {
  struct stat st;
  if (fstat(t.fd, &st) || t.off + bytes > (uint64_t)st.st_size) die("probe span exceeds its source file");
  uint64_t first = t.off & ~(ALIGNMENT - 1), end = (t.off + bytes + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
  if (end > (uint64_t)st.st_size) end = (uint64_t)st.st_size;
  if (nspans == SPANS || dst + end - first > shared_buffers[expert].size) die("probe span exceeds its destination");
  spans[nspans++] = (Span){t.fd, first, end - first, dst, expert, file_handle(t.fd)};
  read_bytes += end - first;
}

static void prepare(int count, int trial) {
  nspans = njobs = 0;
  read_bytes = 0;
  for (int e = 0; e < count; e++) {
    int layer = (trial * 7 + 3) % LAYERS, expert = (trial * 31 + e * 17 + 5) % EXPERTS;
    add_span(st_get("layers.%d.ffn.experts.%d.w1.weight", layer, expert), EXPERT_W, e, 0);
    add_span(st_get("layers.%d.ffn.experts.%d.w1.scale", layer, expert), EXPERT_S, e, REGION_BYTES(EXPERT_W));
  }
}

static void read_one(void *context) {
  ReadJob *j = context;
  for (uint64_t done = 0; done < j->bytes;) {
    ssize_t n = pread(j->fd, j->dst + done, j->bytes - done, j->off + done);
    if (n <= 0) die("probe pread failed");
    done += (uint64_t)n;
  }
  dispatch_group_leave(j->group);
}

static void copy_buffers(int count, int to_private) {
  GpuCopy copies[EXP];
  for (int e = 0; e < count; e++)
    copies[e] = (GpuCopy){to_private ? &shared_buffers[e] : &private_buffers[e],
                         to_private ? &private_buffers[e] : &shared_buffers[e], 0, 0, shared_buffers[e].size};
  gpu_begin(), gpu_copies(copies, count), gpu_commit(), gpu_sync();
}

static Measure measure(int count, int mode) {
  for (int e = 0; e < count; e++) memset(shared_buffers[e].ptr, 0, shared_buffers[e].size);
  if (mode == 2) copy_buffers(count, 1); // materialize and clear private storage outside the I/O timing
  struct rusage c0, c1;
  vm_statistics64_data_t v0 = memory_now();
  getrusage(RUSAGE_SELF, &c0);
  double t = now();
  if (!mode) {
    dispatch_group_t group = dispatch_group_create();
    njobs = 0;
    for (int s = 0; s < nspans; s++) {
      Span *p = spans + s;
      for (uint64_t done = 0; done < p->bytes; done += READ_PIECE) {
        if (njobs == JOBS) die("too many probe read pieces");
        uint64_t length = p->bytes - done < READ_PIECE ? p->bytes - done : READ_PIECE;
        jobs[njobs++] = (ReadJob){p->fd, p->off + done, length, shared_buffers[p->expert].ptr + p->dst + done, group};
      }
    }
    for (int j = 0; j < njobs; j++) {
      dispatch_group_enter(group);
      dispatch_async_f(dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), jobs + j, read_one);
    }
    dispatch_group_wait(group, DISPATCH_TIME_FOREVER), dispatch_release(group);
  } else {
    id command = MSG(id (*)(id, SEL), io_queue, S("commandBuffer"));
    for (int s = 0; s < nspans; s++) {
      Span *p = spans + s;
      Buf *b = mode == 2 ? private_buffers + p->expert : shared_buffers + p->expert;
      MSG(void (*)(id, SEL, id, unsigned long, unsigned long, id, unsigned long), command,
          S("loadBuffer:offset:size:sourceHandle:sourceHandleOffset:"), (id)b->obj,
          p->dst, p->bytes, p->handle, p->off);
    }
    MSG(void (*)(id, SEL), command, S("commit"));
    MSG(void (*)(id, SEL), command, S("waitUntilCompleted"));
    if (MSG(long (*)(id, SEL), command, S("status")) != 3)
      die("Metal I/O read: %s", describe(MSG(id (*)(id, SEL), command, S("error"))));
  }
  double elapsed = now() - t;
  getrusage(RUSAGE_SELF, &c1);
  vm_statistics64_data_t v1 = memory_now();
  Measure result = {elapsed * 1000,
    (c1.ru_utime.tv_sec - c0.ru_utime.tv_sec) * 1000.0 + (c1.ru_utime.tv_usec - c0.ru_utime.tv_usec) / 1000.0,
    (c1.ru_stime.tv_sec - c0.ru_stime.tv_sec) * 1000.0 + (c1.ru_stime.tv_usec - c0.ru_stime.tv_usec) / 1000.0,
    ((int64_t)v1.free_count - v0.free_count) * vm_page_size,
    ((int64_t)v1.external_page_count - v0.external_page_count) * vm_page_size};
  if (mode == 2) copy_buffers(count, 0); // equality check only; the model would consume the private buffer directly
  return result;
}

int main(int argc, char **argv) {
  if (argc != 2) die("usage: metal-io-probe CHECKPOINT");
  alarm(180);
  lock_model();
  const char source[] = "#include <metal_stdlib>\nusing namespace metal; kernel void unused(device uint *x [[buffer(0)]]) { x[0]=0; }";
  gpu_init(source, sizeof source), st_open(argv[1]);
  id d = MSG(id (*)(id, SEL), MSG(id (*)(id, SEL), (id)objc_getClass("MTLIOCommandQueueDescriptor"), S("alloc")), S("init"));
  MSG(void (*)(id, SEL, long), d, S("setType:"), 0l); // concurrent
  MSG(void (*)(id, SEL, long), d, S("setPriority:"), 0l); // high
  MSG(void (*)(id, SEL, unsigned long), d, S("setMaxCommandBufferCount:"), 64ul);
  MSG(void (*)(id, SEL, unsigned long), d, S("setMaxCommandsInFlight:"), 32ul);
  id error = 0;
  io_queue = MSG(id (*)(id, SEL, id, id *), dev, S("newIOCommandQueueWithDescriptor:error:"), d, &error);
  if (!io_queue) die("Metal I/O queue: %s", describe(error));
  MSG(void (*)(id, SEL), d, S("release"));
  size_t bytes = REGION_BYTES(EXPERT_W) + REGION_BYTES(EXPERT_S);
  uint8_t *reference[EXP];
  for (int e = 0; e < EXP; e++) {
    shared_buffers[e] = gpu_alloc(bytes), private_buffers[e] = private_buffer(bytes);
    reference[e] = xmalloc(bytes);
  }
  const int sizes[] = {1, 3, 6};
  for (int k = 0; k < 3; k++) for (int trial = 0; trial < 6; trial++) {
    int count = sizes[k];
    prepare(count, k * 6 + trial);
    Measure values[3];
    for (int step = 0; step < 3; step++) {
      int mode = (step + trial) % 3;
      values[mode] = measure(count, mode);
      for (int e = 0; e < count; e++) {
        if (!step) memcpy(reference[e], shared_buffers[e].ptr, bytes);
        else if (memcmp(reference[e], shared_buffers[e].ptr, bytes)) die("Metal I/O bytes differ from pread");
      }
    }
    printf("{\"metal_io\":{\"experts\":%d,\"trial\":%d,\"first_backend\":%d,\"read_bytes\":%llu,\"equal\":true,\"backends\":[",
           count, trial, trial % 3, (unsigned long long)read_bytes);
    for (int mode = 0; mode < 3; mode++)
      printf("%s{\"mode\":%d,\"ms\":%.6f,\"user_ms\":%.6f,\"system_ms\":%.6f,\"free_change\":%lld,\"file_cache_change\":%lld}",
             mode ? "," : "", mode, values[mode].ms, values[mode].user_ms, values[mode].system_ms,
             (long long)values[mode].free_change, (long long)values[mode].file_change);
    printf("]}}\n");
    fflush(stdout);
  }
  for (int e = 0; e < EXP; e++) gpu_free(shared_buffers + e), gpu_free(private_buffers + e), free(reference[e]);
  for (int h = 0; h < nhandles; h++) MSG(void (*)(id, SEL), handles[h].handle, S("release"));
  MSG(void (*)(id, SEL), io_queue, S("release"));
  return 0;
}
