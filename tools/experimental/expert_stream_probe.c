// Probe overlapping MXFP4 gate/up work with down-weight reads in a favorable case.
// All six selected experts are missing and shared by every row. This is not model throughput.
#include "expert_batch.h"
#include "../../src/engine/io.c"

enum { PROBE_ROWS = 6, PROBE_EXPERTS = 6, PROBE_JOBS = 96 };
typedef struct { Piece piece; dispatch_group_t group; } ProbeJob;
typedef struct { ProbeJob jobs[PROBE_JOBS]; int count; uint64_t bytes; dispatch_group_t group; } ProbeReads;
typedef struct { double wall, ready, read, gpu; } ProbeTime;
static Buf probe_weights[PROBE_EXPERTS], probe_refs, probe_groups, probe_x, probe_h, probe_shared;
static Buf probe_mix, probe_res, probe_out, probe_parts;
static ProbeReads probe_all, probe_first, probe_last;
static int probe_up, probe_down, probe_up_rows, probe_down_rows, probe_sum, probe_ng;

static void probe_read(void *context) {
  ProbeJob *job = context;
  read_piece(&job->piece, 0);
  dispatch_group_leave(job->group);
}

static void probe_range(ProbeReads *r, int fd, uint64_t first, uint64_t end, uint8_t *dst) {
  if (first % SECTOR || end % SECTOR || end < first) die("unaligned probe read");
  for (uint64_t at = first; at < end; at += PIECE) {
    if (r->count == PROBE_JOBS) die("too many probe pieces");
    uint64_t length = end - at < PIECE ? end - at : PIECE;
    r->jobs[r->count++] = (ProbeJob){{fd, at, length, dst + at - first}, r->group};
    r->bytes += length;
  }
}

static void probe_submit(ProbeReads *r) {
  for (int i = 0; i < r->count; i++) {
    dispatch_group_enter(r->group);
    dispatch_async_f(dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), &r->jobs[i], probe_read);
  }
}

static void probe_prepare(int rows, int trial) {
  probe_all.count = probe_first.count = probe_last.count = 0;
  probe_all.bytes = probe_first.bytes = probe_last.bytes = 0;
  for (int e = 0; e < PROBE_EXPERTS; e++) {
    int layer = (trial * 7 + rows) % LAYERS, expert = (trial * 31 + e * 17) % EXPERTS;
    Tensor w = st_get("layers.%d.ffn.experts.%d.w1.weight", layer, expert);
    Tensor s = st_get("layers.%d.ffn.experts.%d.w1.scale", layer, expert);
    uint64_t a = w.off & ~(SECTOR - 1), z = (w.off + EXPERT_W + SECTOR - 1) & ~(SECTOR - 1);
    // Boundary pages are read first. The late read never writes a byte of
    // the gate/up matrices, including their shared alignment pages.
    uint64_t u = (w.off + EXPERT_W / 3 + SECTOR - 1) & ~(SECTOR - 1);
    uint64_t v = (w.off + 2 * (EXPERT_W / 3)) & ~(SECTOR - 1);
    uint64_t sa = s.off & ~(SECTOR - 1), sz = (s.off + EXPERT_S + SECTOR - 1) & ~(SECTOR - 1);
    uint8_t *base = probe_weights[e].ptr, *scales = base + REGION(EXPERT_W);
    if (!(a <= u && u <= v && v <= z) || z - a > REGION(EXPERT_W) || sz - sa > REGION(EXPERT_S))
      die("invalid probe partition");
    probe_range(&probe_all, w.fd, a, z, base);
    probe_range(&probe_all, s.fd, sa, sz, scales);
    probe_range(&probe_first, s.fd, sa, sz, scales);
    probe_range(&probe_first, w.fd, a, u, base);
    probe_range(&probe_first, w.fd, v, z, base + v - a);
    probe_range(&probe_last, w.fd, u, v, base + u - a);
    for (int r = 0; r < rows; r++)
      ((ExpertRef *)probe_refs.ptr)[r * PROBE_EXPERTS + e] = (ExpertRef){
          .p = {probe_weights[e].gpu + w.off - a,
                probe_weights[e].gpu + REGION(EXPERT_W) + s.off - sa, 0}, .weight = .1f};
  }
  if (probe_all.bytes != probe_first.bytes + probe_last.bytes) die("probe byte totals differ");
  probe_ng = expert_batch_plan((ExpertRows *)probe_groups.ptr, probe_refs.ptr, rows,
                               PROBE_EXPERTS, PROBE_EXPERTS * sizeof(ExpertRef));
}

static void probe_encode_up(int rows) {
  gpu_kern(rows == 1 ? probe_up : probe_up_rows);
  gpu_buf(0, rows == 1 ? &probe_refs : &probe_groups, 0);
  gpu_buf(1, &probe_x, 0), gpu_buf(2, &probe_h, 0);
  gpu_go(INTER / 32, rows == 1 ? PROBE_EXPERTS : probe_ng, 1, 256, 1, 1);
}

static void probe_encode_down(int rows) {
  uint32_t n = PROBE_EXPERTS, pm = 0;
  if (rows == 1) {
    gpu_kern(probe_down);
    gpu_buf(0, &probe_refs, 0), gpu_buf(1, &probe_h, 0), gpu_buf(2, &probe_shared, 0);
    gpu_buf(3, &probe_mix, 0), gpu_buf(4, &probe_res, 0), gpu_buf(5, &probe_out, 0);
    gpu_bytes(6, &n, 4), gpu_bytes(7, &pm, 4), gpu_buf(8, &probe_parts, 0);
    gpu_go(DIM / 4, 1, 1, 32 * PROBE_EXPERTS, 1, 1);
    return;
  }
  gpu_kern(probe_down_rows);
  gpu_buf(0, &probe_groups, 0), gpu_buf(1, &probe_h, 0), gpu_buf(2, &probe_parts, 0);
  gpu_go(DIM / 32, probe_ng, 1, 256, 1, 1);
  for (int r = 0; r < rows; r++) {
    gpu_kern(probe_sum);
    gpu_buf(0, &probe_parts, (size_t)r * DIM * 8 * 4), gpu_buf(1, &probe_shared, (size_t)r * DIM * 4);
    gpu_buf(2, &probe_mix, (size_t)r * MIX * 4), gpu_buf(3, &probe_res, (size_t)r * HC * DIM * STREAM_BYTES);
    gpu_buf(4, &probe_out, (size_t)r * HC * DIM * STREAM_BYTES), gpu_bytes(5, &n, 4);
    gpu_go(DIM / 256, 1, 1, 256, 1, 1);
  }
}

static ProbeTime probe_measure(int rows, int phased) {
  for (int e = 0; e < PROBE_EXPERTS; e++) memset(probe_weights[e].ptr, 0xcd, probe_weights[e].size);
  memset(probe_h.ptr, 0, probe_h.size), memset(probe_parts.ptr, 0, probe_parts.size);
  gpu_keep_clock(.02);
  double start = now(), g = gpu_busy, ready, read;
  ProbeReads *first = phased ? &probe_first : &probe_all;
  probe_submit(first);
  dispatch_group_wait(first->group, DISPATCH_TIME_FOREVER);
  ready = read = now();
  gpu_begin();
  probe_encode_up(rows);
  if (phased) {
    gpu_commit();
    probe_submit(&probe_last);
    dispatch_group_wait(probe_last.group, DISPATCH_TIME_FOREVER);
    read = now();
    gpu_begin();
  }
  probe_encode_down(rows);
  gpu_commit(), gpu_sync();
  return (ProbeTime){1000 * (now() - start), 1000 * (ready - start), 1000 * (read - start), 1000 * (gpu_busy - g)};
}

int main(int argc, char **argv) {
  if (argc != 2) die("usage: expert-stream-probe CHECKPOINT");
  alarm(120);
  lock_model();
  static const char source[] = {
#embed "../../src/kernels/kernels.metal"
  };
  gpu_init(source, sizeof source);
  st_open(argv[1]);
  probe_up = gpu_kernel("moe_up"), probe_down = gpu_kernel("moe_down");
  probe_up_rows = gpu_kernel("moe_up_rows"), probe_down_rows = gpu_kernel("moe_down_rows"), probe_sum = gpu_kernel("moe_parts");
  for (int e = 0; e < PROBE_EXPERTS; e++) probe_weights[e] = gpu_alloc(REGION(EXPERT_W) + REGION(EXPERT_S));
  probe_refs = gpu_alloc(PROBE_ROWS * PROBE_EXPERTS * sizeof(ExpertRef));
  probe_groups = gpu_alloc(PROBE_ROWS * PROBE_EXPERTS * sizeof(ExpertRows));
  probe_x = gpu_alloc(PROBE_ROWS * DIM * 4), probe_h = gpu_alloc(PROBE_ROWS * PROBE_EXPERTS * INTER * 4);
  probe_shared = gpu_alloc(PROBE_ROWS * DIM * 4), probe_mix = gpu_alloc(PROBE_ROWS * MIX * 4);
  probe_res = gpu_alloc(PROBE_ROWS * HC * DIM * STREAM_BYTES), probe_out = gpu_alloc(probe_res.size);
  probe_parts = gpu_alloc(PROBE_ROWS * DIM * 8 * 4);
  memset(probe_shared.ptr, 0, probe_shared.size), memset(probe_res.ptr, 0, probe_res.size);
  memset(probe_mix.ptr, 0, probe_mix.size);
  for (int r = 0; r < PROBE_ROWS; r++) {
    for (int d = 0; d < DIM; d++) ((float *)probe_x.ptr)[r * DIM + d] = (float)((d + r) % 9 - 4) / 16;
    for (int h = 0; h < HC; h++) ((float *)probe_mix.ptr)[r * MIX + HC + h] = 1;
  }
  probe_all.group = dispatch_group_create(), probe_first.group = dispatch_group_create(), probe_last.group = dispatch_group_create();
  uint8_t *reference = xmalloc(probe_out.size), *reference_h = xmalloc(probe_h.size);
  gpu_keep_clock(.001); // initialize clock calibration outside all samples
  const int sizes[] = {1, 3, 6};
  for (int k = 0; k < 3; k++) for (int trial = 0; trial < 8; trial++) {
    int rows = sizes[k];
    probe_prepare(rows, trial);
    ProbeTime result[2];
    for (int step = 0; step < 2; step++) {
      int mode = step ^ (trial & 1);
      result[mode] = probe_measure(rows, mode);
      int nonzero = 0;
      for (int i = 0; i < rows * HC * DIM; i++) {
        uint32_t bits = (uint32_t)((uint16_t *)probe_out.ptr)[i] << 16;
        float value;
        memcpy(&value, &bits, sizeof value);
        if (!isfinite(value)) die("nonfinite probe output");
        nonzero |= value != 0;
      }
      if (!nonzero) die("probe produced only zero output");
      if (!step) {
        memcpy(reference, probe_out.ptr, (size_t)rows * HC * DIM * STREAM_BYTES);
        memcpy(reference_h, probe_h.ptr, (size_t)rows * PROBE_EXPERTS * INTER * sizeof(float));
      } else if (memcmp(reference, probe_out.ptr, (size_t)rows * HC * DIM * STREAM_BYTES) ||
                 memcmp(reference_h, probe_h.ptr, (size_t)rows * PROBE_EXPERTS * INTER * sizeof(float)))
        die("phased expert output differs");
    }
    printf("{\"rows\":%d,\"trial\":%d,\"read_bytes\":%llu,\"late_bytes\":%llu,"
           "\"full_ms\":%.6f,\"phased_ms\":%.6f,\"first_ready_ms\":%.6f,"
           "\"phased_read_ms\":%.6f,\"full_gpu_ms\":%.6f,\"phased_gpu_ms\":%.6f,\"output_equal\":true}\n",
           rows, trial, (unsigned long long)probe_all.bytes, (unsigned long long)probe_last.bytes,
           result[0].wall, result[1].wall, result[1].ready, result[1].read, result[0].gpu, result[1].gpu);
    fflush(stdout);
  }
  free(reference), free(reference_h);
  return 0;
}
