// SSD reads, the routed-expert cache and its adaptive size.
//
// Experts are read with F_NOCACHE over 16 KiB-aligned spans: unaligned
// uncached reads still fill the page cache. A slot holds one expert's spans
// (expert_span) one after the other: its MXFP4 codes (w1, w2, w3), then its
// scales; or with --experts its GGUF rows of w1, w3 and w2.
//
// Draft stages have separate MXFP4-sized buffer pools whose bytes come off
// the same budget. By default every expert is preloaded. A bounded pool
// instead reuses its stage's buffers by LRU, loading native selections on
// demand. The buffers stay resident; backbone eviction cannot reuse them.
//
// Memory: the slots stay in memory within the cache's budget, and the GPU
// reads a slot only while it is mapped for it (in the residency set that the
// command queue holds). The mapped slots stay below the point where the GPU
// stalls (mem_gpu_room): a slot whose expert a layer needs is mapped before
// the GPU uses it, and between tokens (in a prompt, for the layers done) the
// mapped slots with the lowest scores leave the set until the rest fit. The
// cache counts free pages and reclaimable file pages as available, and
// leaves a configurable reserve for other programs.
// It gives slots back when the kernel reports pressure, when anything was
// swapped out or the available memory falls below the reserve, waiting a
// second between steps for the memory given back to show; it grows again
// only after a calm period. It checks after every token, after every layer
// of a prompt and every second while the server idles.
#include <CoreFoundation/CoreFoundation.h>
#include <IOKit/IOKitLib.h>
#include <dispatch/dispatch.h>
#include <mach/mach.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <string.h>
#include <sys/sysctl.h>
#include <unistd.h>

#include "myds4.h"
#ifdef MYDS4_PREDICT_LAB
#include "../experimental/predictor_lab.h"
#endif

#define SECTOR 16384ull
#define PIECE (2ull << 20)
#define GiB (1024.0 * 1024 * 1024)
#define REGION(len) (((len) + 3 * SECTOR - 1) / SECTOR * SECTOR)  // a span's aligned cover, at most
static uint64_t SLOT_SIZE;  // the largest expert in use

typedef struct { int fd; uint64_t off, len; uint8_t *dst; } Piece;

static void read_piece(void *ctx, size_t i) {
  Piece *p = (Piece *)ctx + i;
  for (uint64_t got = 0; got < p->len;) {
    ssize_t r = pread(p->fd, p->dst + got, p->len - got, p->off + got);
    if (r < 0) die("read failed at offset %llu", p->off + got);
    if (!r) break;  // the aligned cover may extend past the end of the file
    got += r;
  }
}

// Splits the aligned cover of [off, off + len) into pieces landing at dst;
// returns their count and the payload's offset from dst.
static int split(Piece *out, int fd, uint64_t off, uint64_t len, uint8_t *dst, uint32_t *at) {
  uint64_t a = off & ~(SECTOR - 1), e = (off + len + SECTOR - 1) & ~(SECTOR - 1);
  int n = 0;
  *at = off - a;
  for (uint64_t p = a; p < e; p += PIECE)
    out[n++] = (Piece){fd, p, e - p < PIECE ? e - p : PIECE, dst + (p - a)};
  return n;
}

static void run(Piece *p, int n) {
  dispatch_apply_f(n, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), p, read_piece);
}

void io_read(int fd, uint64_t off, uint64_t len, Buf *dst) {
  int n = (len + 2 * SECTOR) / PIECE + 2;
  Piece *p = xmalloc(n * sizeof *p);
  n = split(p, fd, off, len, dst->ptr, &dst->off);
  if (dst->off + len > dst->size) die("io_read: buffer too small");
  run(p, n);
  free(p);
}

typedef struct { int fd; uint64_t off; uint32_t len; uint8_t *dst; } Row;

// A row is read in whole 4 KiB blocks (the file system's block size), which
// uncached reads also take past the page cache.
#define BLOCK 4096ull

static void read_row(void *ctx, size_t i) {
  Row *r = (Row *)ctx + i;
  _Alignas(16384) uint8_t tmp[3 * SECTOR];
  uint64_t a = r->off & ~(BLOCK - 1), e = (r->off + r->len + BLOCK - 1) & ~(BLOCK - 1);
  if (e - a > sizeof tmp) die("io_rows: row too long");
  if (pread(r->fd, tmp, e - a, a) < (ssize_t)(r->off + r->len - a)) die("read failed at offset %llu", a);
  memcpy(r->dst, tmp + (r->off - a), r->len);
}

// Rows are small random reads: their speed is the number in flight, so 64
// lanes on the global queue, which adds threads while others wait in pread.
#define LANES 64
typedef struct { Row *r; int n, lane, lanes; } Lane;

static void read_lane(void *ctx) {
  Lane *l = ctx;
  for (int i = l->lane; i < l->n; i += l->lanes) read_row(l->r, i);
}

static int cmp_u64(const void *a, const void *b) {
  uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
  return (x > y) - (x < y);
}

// A row requested more than once is read once and copied: a prompt repeats
// many of its n-grams.
void io_rows(int fd, uint64_t base, uint32_t rowbytes, const int64_t *rows, int n, uint8_t *dst) {
  if (n >= 1 << 20) die("io_rows: too many rows");
  uint64_t *key = xmalloc((size_t)n * 8);  // row, then request index in the low 20 bits
  for (int i = 0; i < n; i++) key[i] = (uint64_t)rows[i] << 20 | i;
  qsort(key, n, 8, cmp_u64);
  Row *r = xmalloc((size_t)n * sizeof *r);
  int u = 0;
  for (int i = 0; i < n; i++)
    if (!i || key[i] >> 20 != key[i - 1] >> 20) {
      uint64_t at = key[i] & 0xfffff;
      r[u++] = (Row){fd, base + rows[at] * (uint64_t)rowbytes, rowbytes, dst + at * rowbytes};
    }
  Lane lane[LANES];
  int k = u < LANES ? u : LANES;
  dispatch_group_t g = dispatch_group_create();
  for (int j = 0; j < k; j++) {
    lane[j] = (Lane){r, u, j, k};
    dispatch_group_async_f(g, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), &lane[j], read_lane);
  }
  dispatch_group_wait(g, DISPATCH_TIME_FOREVER);
  dispatch_release(g);
  for (int i = 1, first = key[0] & 0xfffff; i < n; i++) {
    int at = key[i] & 0xfffff;
    if (key[i] >> 20 == key[i - 1] >> 20) memcpy(dst + (uint64_t)at * rowbytes, dst + (uint64_t)first * rowbytes, rowbytes);
    else first = at;
  }
  free(key);
  free(r);
}

// The kernel's page counts. Its free count includes the speculative pages
// (file pages read ahead), which it takes back first.
static vm_statistics64_data_t vm(void) {
  vm_statistics64_data_t s = {0};
  mach_msg_type_number_t n = HOST_VM_INFO64_COUNT;
  host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t)&s, &n);
  return s;
}

static uint64_t sysctl_u64(const char *name) {  // 0 if there is no such value
  uint64_t v = 0;
  size_t n = sizeof v;
  return sysctlbyname(name, &v, &n, 0, 0) ? 0 : v;
}

// Free pages already include speculative file pages. The remaining file
// pages are reclaimable too; the kernel's cache target is not used memory.
static uint64_t file_of(vm_statistics64_data_t s) {
  uint64_t file = s.external_page_count > s.speculative_count ? s.external_page_count - s.speculative_count : 0;
  return file * vm_page_size;
}

static uint64_t avail_of(vm_statistics64_data_t s) {
  return (uint64_t)s.free_count * vm_page_size + file_of(s);
}

uint64_t mem_available(void) { return avail_of(vm()); }
static uint64_t mem_free(void) { return (uint64_t)vm().free_count * vm_page_size; }
uint64_t mem_swapouts(void) { return vm().swapouts; }
uint64_t mem_compressions(void) { return vm().compressions; }

static task_vm_info_data_t task_vm(void) {
  task_vm_info_data_t v = {0};
  mach_msg_type_number_t n = TASK_VM_INFO_COUNT;
  task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&v, &n);
  return v;
}

uint64_t mem_footprint(void) { return task_vm().phys_footprint; }
uint64_t mem_compressed(int peak) {
  task_vm_info_data_t v = task_vm();
  return peak ? v.compressed_peak : v.compressed;
}

int mem_pressure(void) {
  int level = 1;
  size_t len = sizeof level;
  sysctlbyname("kern.memorystatus_vm_pressure_level", &level, &len, 0, 0);
  return level;
}

// The GPU driver's resident memory for all processes ("In use system
// memory" in the accelerator's statistics).
uint64_t mem_gpu(void) {
  static io_service_t acc;
  if (!acc) acc = IOServiceGetMatchingService(kIOMainPortDefault, IOServiceMatching("IOAccelerator"));
  int64_t v = 0;
  CFDictionaryRef d = acc ? IORegistryEntryCreateCFProperty(acc, CFSTR("PerformanceStatistics"), kCFAllocatorDefault, 0) : 0;
  if (d) {
    CFNumberRef n = CFGetTypeID(d) == CFDictionaryGetTypeID() ? CFDictionaryGetValue(d, CFSTR("In use system memory")) : 0;
    if (n) CFNumberGetValue(n, kCFNumberSInt64Type, &v);
    CFRelease(d);
  }
  return v > 0 ? (uint64_t)v : 0;
}

// How far the GPU driver's resident memory is below the point past which
// the GPU stalls (INT64_MAX: no such point). Measured on an M5 Max with 128
// GiB (macOS 27): with 54.4 GiB resident decoding ran at its usual speed,
// from 55.2 GiB on it took 8 to 15 times its GPU time per token, the GPU
// only 17 to 46% busy meanwhile, whatever the slot count or the free memory;
// allocated but not resident memory (68.9 GiB in all) did not matter.
// Presumably the IOGPU driver's wired-memory collector: unless it is off
// (iogpu.disable_wired_collector) or a low-water mark is set
// (iogpu.wired_lwm_mb, then the limit), the limit is 41% of the physical
// memory, about 2 GiB below the measured point.
int64_t mem_gpu_limit(void) {
  static int64_t limit = -1;
  if (limit < 0) {
    int off = 0, lwm = 0;
    uint64_t ram = 0;
    size_t n = sizeof off;
    sysctlbyname("iogpu.disable_wired_collector", &off, &n, 0, 0);
    n = sizeof lwm;
    sysctlbyname("iogpu.wired_lwm_mb", &lwm, &n, 0, 0);
    n = sizeof ram;
    sysctlbyname("hw.memsize", &ram, &n, 0, 0);
    limit = off ? INT64_MAX : lwm > 0 ? (int64_t)lwm << 20 : (int64_t)(ram / 100 * 41);
  }
  return limit;
}

int64_t mem_gpu_room(void) {
  int64_t limit = mem_gpu_limit();
  return limit == INT64_MAX ? INT64_MAX : limit - (int64_t)mem_gpu();
}

void mem_info(MemInfo *m) {
  vm_statistics64_data_t s = vm();
  uint64_t p = vm_page_size, keep = sysctl_u64("vm.vm_page_filecache_min");
  struct xsw_usage sw = {0};
  size_t n = sizeof sw;
  sysctlbyname("vm.swapusage", &sw, &n, 0, 0);
  *m = (MemInfo){.total = sysctl_u64("hw.memsize"), .free = s.free_count * p,
                 .file = file_of(s),
                 .file_kept = keep * p, .wired = s.wire_count * p, .compressor = s.compressor_page_count * p,
                 .compressed = s.total_uncompressed_pages_in_compressor * p, .swap_used = sw.xsu_used,
                 .swapouts = s.swapouts, .compressions = s.compressions, .pressure = mem_pressure()};
}

// ---- the cache: slots in the lists of an adaptive replacement cache ----

typedef struct {
  Buf b;
  uint32_t at[3];  // where the spans are in b
  int16_t layer, expert;
  int prev, next;  // links in the recency list, head = most recent
  int prefetched;  // loaded on a prediction, not yet requested
  int pinned;      // a DSpark pool buffer, exempt from backbone eviction/unmapping
  int fresh;       // joined the residency set this token: needs gpu_use until it is committed
  int mapped;      // in the residency set (or joining it: fresh)
  uint64_t used;   // token of the last use
  dispatch_group_t loading;  // owned until finish(), including after its reads complete
  Piece piece[16];
  int npiece;
  int background_load; // this load originally occupied a prefetch admission
  _Atomic int unfinished; // read or cancelled pieces that have not completed
  _Atomic char taken[16];  // piece k has been started (or dropped)
#ifdef MYDS4_PREDICT_LAB
  uint64_t action_id;
#endif
} Slot;

static uint64_t token;

FileSpan expert_span[CACHE_LAYERS][EXPERTS][3];
int expert_spans[CACHE_LAYERS];
CacheStats cache_stats;
static Slot *slots;
static int16_t where[CACHE_LAYERS][EXPERTS];
static int nslots, cap, max_slots, *free_list, nfree, npinned;
static int nmapped, mcap;  // slots mapped for the GPU, and how many may be
static uint64_t reserve, pinned_bytes;

void cache_reserve(double gib) { reserve = (uint64_t)((gib > 0 ? gib : CACHE_RESERVE_DEFAULT) * GiB); }
static int draft_capacity = DRAFT_EXPERTS;
static uint64_t draft_clock, draft_touch_clock;
static uint64_t draft_age[DRAFTS * DRAFT_EXPERTS];  // recency within the separate pools

// Observation metadata is separate from the eviction policy's prefetched flag.
// The model thread owns it until a read group finishes; readers update only
// atomic byte and timing counters, before leaving that group. No trace records are saved.
typedef struct {
  uint64_t target_pass, requested, queued_ns;
  _Atomic uint64_t completed, first_io_ns;
  uint8_t origin_prefetch, used, tracked, phase, issuer;
} PrefetchLoad;
typedef struct {
  uint64_t passes, rows, unsupported_passes, unsupported_rows, batches[2];
  uint64_t states[2][PF_STATES], payload[2][PF_STATES];
  uint64_t admissions[PF_ISSUERS], timely_first[PF_ISSUERS];
  uint64_t used_by_issuer[PF_ISSUERS], unstarted_at_use[PF_ISSUERS], age_at_use_ns[PF_ISSUERS];
  uint64_t target_uses[PF_ISSUERS], target_unstarted[PF_ISSUERS], target_age_ns[PF_ISSUERS];
  uint64_t first_uses, eventual_first_uses, first_completed_bytes, first_remaining_bytes, unobserved_first_uses;
  uint64_t unused_removed, unused_removed_bytes, unused_cancelled_bytes;
  uint64_t prefetch_evictions, rerequests[2], absent_after_eviction[2];
  _Atomic uint64_t prefetch_read_bytes, demand_read_bytes;
  _Atomic uint64_t io_pieces[3], queue_ns[3], service_ns[3]; // demand, background, promoted
} PrefetchStats;
static int pf_enabled, pf_active, pf_phase, pf_rows, pf_issuer = PF_OTHER;
static uint64_t pf_pass;
static PrefetchLoad *pf_loads;
static PrefetchStats pf_stats[PF_PHASES];
static uint8_t pf_ghost[LAYERS][EXPERTS];
static int slot_reading(int i);
static uint64_t metrics_ns(void) { return (uint64_t)(now() * 1e9); }

void cache_metrics_begin(int phase, int rows) {
  if (!pf_enabled) return;
  if (phase < 0 || phase >= PF_PHASES || rows < 1 || pf_pass == UINT64_MAX) die("invalid prefetch metric pass");
  pf_phase = phase, pf_rows = rows, pf_active = 1, pf_pass++;
  pf_stats[phase].passes++, pf_stats[phase].rows += rows;
}

void cache_metrics_end(void) { pf_active = 0; }

void cache_metrics_unsupported(void) {
  if (!pf_enabled || !pf_active) return;
  pf_stats[pf_phase].unsupported_passes++;
  pf_stats[pf_phase].unsupported_rows += pf_rows;
  pf_active = 0;
}

void cache_observe_snapshot(int layer, CacheObservation *o) {
  o->active = pf_enabled && pf_active && layer >= 0 && layer < LAYERS;
  if (!o->active) return;
  o->layer = layer, o->phase = pf_phase, o->pass = pf_pass;
  for (int e = 0; e < EXPERTS; e++) {
    int i = where[layer][e];
    o->state[e] = PF_ABSENT, o->first[e] = 0, o->completed[e] = 0;
    o->ghost[e] = pf_ghost[layer][e];
    if (i < 0) continue;
    PrefetchLoad *m = &pf_loads[i];
    int pending = slot_reading(i);
    o->first[e] = m->tracked && m->origin_prefetch && !m->used;
    o->completed[e] = atomic_load_explicit(&m->completed, memory_order_relaxed);
    o->state[e] = pending ? (m->origin_prefetch ? PF_PENDING_PREFETCH : PF_PENDING_DEMAND)
                         : o->first[e] && m->target_pass == pf_pass ? PF_FRESH_READY : PF_EXISTING_READY;
  }
}

void cache_observe_routes(const CacheObservation *o, const int *ids, int rows, int stride, int served) {
  if (!o->active) return;
  if (served < 0 || served > 1 || o->pass != pf_pass) die("invalid prefetch observation");
  uint8_t used[EXPERTS] = {0};
  for (int r = 0; r < rows; r++)
    for (int k = 0; k < TOPK; k++) {
      int e = ids[r * stride + k];
      if (e < 0 || e >= EXPERTS) die("invalid expert in prefetch observation");
      used[e] = 1;
    }
  PrefetchStats *s = &pf_stats[o->phase];
  s->batches[served]++;
  for (int e = 0; e < EXPERTS; e++) {
    if (!used[e]) continue;
    int state = o->state[e], i = where[o->layer][e];
    s->states[served][state]++;
    for (int k = 0; k < expert_spans[o->layer]; k++) s->payload[served][state] += expert_span[o->layer][e][k].len;
    if (o->ghost[e]) {
      s->rerequests[served]++;
      s->absent_after_eviction[served] += state == PF_ABSENT;
      if (served) pf_ghost[o->layer][e] = 0;
    }
    if (!served || !o->first[e] || i < 0 || pf_loads[i].used) continue;
    PrefetchLoad *m = &pf_loads[i];
    m->used = 1;
    s->first_uses++;
    s->used_by_issuer[m->issuer]++;
    int unstarted = !atomic_load_explicit(&m->first_io_ns, memory_order_acquire);
    s->unstarted_at_use[m->issuer] += unstarted;
    uint64_t observed_ns = metrics_ns();
    s->age_at_use_ns[m->issuer] += observed_ns >= m->queued_ns ? observed_ns - m->queued_ns : 0;
    if (m->target_pass == o->pass) {
      s->target_uses[m->issuer]++;
      s->target_unstarted[m->issuer] += unstarted;
      s->target_age_ns[m->issuer] += observed_ns >= m->queued_ns ? observed_ns - m->queued_ns : 0;
    }
    s->eventual_first_uses += m->target_pass != o->pass;
    s->first_completed_bytes += o->completed[e];
    s->first_remaining_bytes += m->requested > o->completed[e] ? m->requested - o->completed[e] : 0;
    if (state == PF_FRESH_READY) s->timely_first[m->issuer]++;
  }
}

static void metrics_start(int i, int urgent) {
  if (!pf_enabled) return;
  PrefetchLoad *m = &pf_loads[i];
  m->target_pass = pf_pass, m->used = 0, m->origin_prefetch = !urgent;
  m->phase = pf_phase, m->issuer = pf_issuer;
  m->tracked = pf_active && slots[i].layer < LAYERS;
  m->requested = 0;
  m->queued_ns = metrics_ns();
  atomic_store_explicit(&m->completed, 0, memory_order_relaxed);
  atomic_store_explicit(&m->first_io_ns, 0, memory_order_relaxed);
  for (int k = 0; k < slots[i].npiece; k++) m->requested += slots[i].piece[k].len;
  if (m->tracked && !urgent) pf_stats[m->phase].admissions[m->issuer]++;
}

static uint64_t metrics_io_begin(Slot *slot, int urgent_worker) {
  if (!pf_enabled) return 0;
  PrefetchLoad *m = &pf_loads[slot - slots];
  if (!m->tracked) return 0;
  uint64_t began = metrics_ns(), zero = 0;
  atomic_compare_exchange_strong_explicit(&m->first_io_ns, &zero, began, memory_order_release, memory_order_relaxed);
  int kind = !m->origin_prefetch ? 0 : urgent_worker ? 2 : 1;
  PrefetchStats *s = &pf_stats[m->phase];
  atomic_fetch_add_explicit(&s->io_pieces[kind], 1, memory_order_relaxed);
  atomic_fetch_add_explicit(&s->queue_ns[kind], began - m->queued_ns, memory_order_relaxed);
  return began;
}

static void metrics_io_end(Slot *slot, int urgent_worker, uint64_t began) {
  if (!began) return;
  PrefetchLoad *m = &pf_loads[slot - slots];
  int kind = !m->origin_prefetch ? 0 : urgent_worker ? 2 : 1;
  atomic_fetch_add_explicit(&pf_stats[m->phase].service_ns[kind], metrics_ns() - began, memory_order_relaxed);
}

static void metrics_piece_done(Slot *slot, uint64_t bytes) {
#ifdef MYDS4_PREDICT_LAB
  predictor_lab_read(slot->action_id, bytes);
#endif
  if (!pf_enabled) return;
  PrefetchLoad *m = &pf_loads[slot - slots];
  atomic_fetch_add_explicit(&m->completed, bytes, memory_order_relaxed);
  if (m->tracked) {
    PrefetchStats *s = &pf_stats[m->phase];
    atomic_fetch_add_explicit(m->origin_prefetch ? &s->prefetch_read_bytes : &s->demand_read_bytes,
                              bytes, memory_order_relaxed);
  }
}

// Called only after cancel/finish joins all old readers, before reusing or
// releasing the slot. Ghosts record observed rerequests, not baseline misses.
static void metrics_retire(int i, int by_prefetch) {
  if (!pf_enabled || slots[i].layer >= LAYERS || slots[i].expert < 0) return;
  PrefetchLoad *m = &pf_loads[i];
  if (by_prefetch && pf_active) {
    pf_stats[pf_phase].prefetch_evictions++;
    pf_ghost[slots[i].layer][slots[i].expert] = 1;
  }
  if (m->tracked && m->origin_prefetch && !m->used) {
    uint64_t done = atomic_load_explicit(&m->completed, memory_order_relaxed);
    PrefetchStats *s = &pf_stats[m->phase];
    s->unused_removed++;
    s->unused_removed_bytes += done;
    s->unused_cancelled_bytes += m->requested > done ? m->requested - done : 0;
  }
  m->tracked = 0;
}

void cache_draft_limit(int slots) {
  if (slots && (slots < DRAFT_CACHE_MIN || slots > DRAFT_EXPERTS))
    die("draft cache needs %d to %d slots per stage", DRAFT_CACHE_MIN, DRAFT_EXPERTS);
  draft_capacity = slots ? slots : DRAFT_EXPERTS;
}

int cache_draft_capacity(void) { return draft_capacity; }

static int draft_pool_count(int layers) {
  return layers > LAYERS ? (layers - LAYERS) * draft_capacity : 0;
}
// The driver counts a slot taken out of the residency set for about a second
// more; the mapped slots leave this much room for those.
#define MAP_SLACK (3ll << 30)

// The eviction order: every expert has a score, the sum of a weight for each
// request that selected it; the weight doubles over the configured number of
// forward-pass aging steps. A verification batch advances once, as does the
// end of prefill. The victim has the lowest priority, an empty slot first.
// Optional read-cost weighting affects that priority, not the score history.
// A recency list
// (head = most recently used) orders the slots for prompt passes and for giving
// slots back.
static int head = -1, tail = -1;
static double weight = 1;  // the current token's
static double escore[CACHE_LAYERS * EXPERTS];  // per expert
static double *score;  // per slot: its expert's score; INFINITY for a free slot, -INFINITY for an empty one
static int retention_half_life = CACHE_HALF_LIFE_DEFAULT, retention_read_cost;
static double retention_growth;
static double reload_cost[LAYERS * EXPERTS];

void cache_retention(int half_life, int read_cost) {
  if (slots) die("cache retention must be set before cache_init");
  if (half_life < 0 || half_life > CACHE_HALF_LIFE_MAX || (read_cost != 0 && read_cost != 1))
    die("invalid cache retention policy");
  retention_half_life = half_life ? half_life : CACHE_HALF_LIFE_DEFAULT;
  retention_read_cost = read_cost;
  // Keep the original constant expression for the default; cache other
  // ratios once instead of evaluating exp2 at every forward pass.
  retention_growth = retention_half_life == CACHE_HALF_LIFE_DEFAULT ? exp2(1.0 / CACHE_HALF_LIFE_DEFAULT)
                                                                  : exp2(1.0 / retention_half_life);
}

static void prepare_reload_cost(int layers) {
  if (!retention_read_cost) return;
  for (int l = 0; l < layers && l < LAYERS; l++)
    for (int e = 0; e < EXPERTS; e++) {
      uint64_t bytes = 0;
      for (int k = 0; k < expert_spans[l]; k++) {
        const FileSpan *p = &expert_span[l][e][k];
        uint64_t a = p->off & ~(SECTOR - 1), end = (p->off + p->len + SECTOR - 1) & ~(SECTOR - 1);
        bytes += end - a;
      }
      if (!bytes || bytes > SLOT_SIZE) die("expert read footprint does not fit its cache slot");
      reload_cost[l * EXPERTS + e] = (double)bytes / SLOT_SIZE;
    }
}

static double eviction_priority(int i) {
  if (!retention_read_cost || !isfinite(score[i]) || slots[i].expert < 0 ||
      slots[i].layer < 0 || slots[i].layer >= LAYERS) return score[i];
  return score[i] * reload_cost[slots[i].layer * EXPERTS + slots[i].expert];
}

static void advance_score_clock(void) {
  token++;
  if ((weight *= retention_growth) > 0x1p500) {  // rescale raw history and resident scores together
    for (int k = 0; k < CACHE_LAYERS * EXPERTS; k++) escore[k] /= weight;
    for (int i = 0; i < max_slots + npinned; i++) score[i] /= weight;
    weight = 1;
  }
}

static void unlink_slot(int i) {
  Slot *s = &slots[i];
  if (s->prev >= 0) slots[s->prev].next = s->next; else head = s->next;
  if (s->next >= 0) slots[s->next].prev = s->prev; else tail = s->prev;
}

static void push_back(int i) {
  slots[i].next = -1;
  slots[i].prev = tail;
  if (tail >= 0) slots[tail].next = i;
  tail = i;
  if (head < 0) head = i;
}

static void push_front(int i) {
  slots[i].prev = -1;
  slots[i].next = head;
  if (head >= 0) slots[head].prev = i;
  head = i;
  if (tail < 0) tail = i;
}

static void cancel(int i), start_readers(void), start(int i, int urgent), finish(int i);

static void release(int i) {
  cancel(i);
  metrics_retire(i, 0);
  unlink_slot(i);
  if (slots[i].expert >= 0) where[slots[i].layer][slots[i].expert] = -1;
  if (slots[i].mapped) nmapped--, slots[i].mapped = 0;
  gpu_free(&slots[i].b);
  score[i] = INFINITY;
  free_list[nfree++] = i;
  nslots--;
  cache_stats.evictions++;
}

void cache_init(uint64_t budget, int layers) {
  if (!retention_growth) cache_retention(0, 0);
  const char *observe = getenv("MYDS4_PREFETCH_METRICS");
  pf_enabled = observe && *observe && strcmp(observe, "0");
  memset(where, 0xff, sizeof where);
  for (int l = 0; l < CACHE_LAYERS; l++)
    for (int e = 0; !expert_spans[l] && e < (l < LAYERS ? EXPERTS : DRAFT_EXPERTS); e++) {
      Tensor w = l < LAYERS ? st_get("layers.%d.ffn.experts.%d.w1.weight", l, e)
                            : st_get("mtp.%d.ffn.experts.%d.w1.weight", l - LAYERS, e);
      Tensor s = l < LAYERS ? st_get("layers.%d.ffn.experts.%d.w1.scale", l, e)
                            : st_get("mtp.%d.ffn.experts.%d.w1.scale", l - LAYERS, e);
      expert_span[l][e][0] = (FileSpan){w.fd, w.off, EXPERT_W};
      expert_span[l][e][1] = (FileSpan){s.fd, s.off, EXPERT_S};
      if (e == (l < LAYERS ? EXPERTS : DRAFT_EXPERTS) - 1) expert_spans[l] = 2;
    }
  uint64_t pin_size = 0;  // the stages' slots
  for (int l = 0; l < layers; l++) {
    uint64_t size = 0, pieces = 0;
    for (int k = 0; k < expert_spans[l]; k++)
      size += REGION(expert_span[l][0][k].len), pieces += (REGION(expert_span[l][0][k].len) + PIECE - 1) / PIECE;
    if (pieces > 16) die("layer %d's experts take %llu reads of %llu MiB, more than a slot tracks", l, pieces, PIECE >> 20);
    uint64_t *at = l < LAYERS ? &SLOT_SIZE : &pin_size;
    if (size > *at) *at = size;
  }
  prepare_reload_cost(layers);
  npinned = draft_pool_count(layers);
  pinned_bytes = npinned * pin_size;
  draft_clock = draft_touch_clock = 0;
  memset(draft_age, 0, sizeof draft_age);
  uint64_t free = mem_free(), avail = mem_available(), limit = gpu_working_set();
  if (!reserve) cache_reserve(0);
  int fixed = budget > 0;
  if (!budget) budget = avail > reserve ? avail - reserve : 0;
  if (budget > limit) budget = limit;
  budget = budget > pinned_bytes ? budget - pinned_bytes : 0;
  // the cache may grow up to max_slots when memory becomes free; a given budget is a fixed bound
  max_slots = (int)((fixed ? budget : limit) / SLOT_SIZE);
  if (max_slots > LAYERS * EXPERTS) max_slots = LAYERS * EXPERTS;
  if (max_slots < 2 * EXPERTS) max_slots = 2 * EXPERTS;
  if (budget < 2 * EXPERTS * SLOT_SIZE)
    die("an expert cache of %.1f GiB is too small: a prompt needs %.1f GiB", budget / GiB, 2.0 * EXPERTS * SLOT_SIZE / GiB);
  // File-cache reclamation is handled by macOS as slots are populated.
  uint64_t within = avail > reserve + pinned_bytes ? avail - reserve - pinned_bytes : 0;
  cap = (int)((budget < within ? budget : within) / SLOT_SIZE);
  if (cap > max_slots) cap = max_slots;
  if (cap < 2 * EXPERTS) cap = 2 * EXPERTS;
  // The mapped slots stay within the GPU's limit, beside the weights, which
  // count as resident only once the GPU uses them.
  int64_t room = mem_gpu_room() - (int64_t)pinned_bytes - (int64_t)mem_footprint() - MAP_SLACK;
  mcap = room > 2 * EXPERTS * (int64_t)SLOT_SIZE ? (int)(room / (int64_t)SLOT_SIZE) : 2 * EXPERTS;
  if (mcap > max_slots) mcap = max_slots;
  slots = xmalloc((max_slots + npinned) * sizeof *slots);
  if (pf_enabled) {
    pf_loads = xmalloc((max_slots + npinned) * sizeof *pf_loads);
    memset(pf_loads, 0, (max_slots + npinned) * sizeof *pf_loads);
  }
  score = xmalloc((max_slots + npinned) * sizeof *score);
  for (int i = 0; i < max_slots + npinned; i++) score[i] = INFINITY;
  start_readers();
  free_list = xmalloc(max_slots * sizeof *free_list);
  for (int i = max_slots - 1; i >= 0; i--) free_list[nfree++] = i;
  for (int i = max_slots; i < max_slots + npinned; i++) {  // separate, MXFP4-sized draft pools
    Slot *s = &slots[i];
    memset(s, 0, sizeof *s);
    s->b = gpu_alloc(pin_size);
    s->layer = LAYERS + (i - max_slots) / draft_capacity;
    s->expert = draft_capacity == DRAFT_EXPERTS ? (i - max_slots) % draft_capacity : -1;
    s->pinned = 1;
    if (s->expert >= 0) {
      where[s->layer][s->expert] = i;
      start(i, 1);
    }
  }
  for (int i = max_slots; i < max_slots + npinned; i++) finish(i);
  int64_t gpu_room = mem_gpu_room();
  fprintf(stderr, "expert cache: %d slots (%.1f GiB) to start, up to %d mapped for the GPU, %.1f GiB free and %.1f GiB "
          "of reclaimable file cache, %.1f GiB kept for other programs, GPU resident memory %.1f GiB",
          cap, cap * (double)SLOT_SIZE / GiB, mcap, free / GiB, (avail > free ? avail - free : 0) / GiB, reserve / GiB,
          mem_gpu() / GiB);
  if (gpu_room < INT64_MAX) fprintf(stderr, " (%.1f GiB below the GPU's limit)", gpu_room / GiB);
  fprintf(stderr, ", GPU working set %.1f GiB\n", limit / GiB);
  fprintf(stderr, "cache policy: half-life %d forward passes, read-cost weighting %s\n",
          retention_half_life, retention_read_cost ? "on" : "off");
  if (npinned) {
    if (draft_capacity == DRAFT_EXPERTS)
      fprintf(stderr, "the DSpark stages' %d experts: %.1f GiB, resident\n", npinned, pinned_bytes / GiB);
    else
      fprintf(stderr, "DSpark cache: %d slots per stage, %.1f GiB reserved, native experts read on demand\n",
              draft_capacity, pinned_bytes / GiB);
  }
}

const Buf *cache_get(int layer, int expert, uint32_t at[3], int *fresh) {
  int i = where[layer][expert];
  if (i < 0) return 0;
  *fresh = slots[i].fresh;
  memcpy(at, slots[i].at, sizeof slots[i].at);
  return &slots[i].b;
}

// Prefetch reads: a prefetch's pieces wait in a queue for READERS threads,
// which start a piece only while no demand piece is in flight. A demand read
// so shares the SSD with at most READERS pieces, and a demand request for an
// expert still being prefetched claims the pieces no reader has started.
// The threads and the demand reads run above the utility QoS class, whose
// disk reads macOS throttles.
#define READERS 4
#define QUEUE (1u << 14)
static uint32_t queue[QUEUE], qhead, qtail;  // entries: slot << 4 | piece
static pthread_mutex_t qlock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t qcond = PTHREAD_COND_INITIALIZER;
static _Atomic int demand, prefetching;  // demand pieces, and prefetched slots, in flight

static void piece_finished(Slot *s) {
  // Release the admission when the last I/O/cancellation finishes. Deferring
  // this to a dispatch notification can block a newer forecast after data is ready.
  if (atomic_fetch_sub_explicit(&s->unfinished, 1, memory_order_acq_rel) == 1 && s->background_load)
    atomic_fetch_sub_explicit(&prefetching, 1, memory_order_relaxed);
  dispatch_group_leave(s->loading);
}

static void *reader(void *arg) {
  for (;;) {
    pthread_mutex_lock(&qlock);
    while (qhead == qtail || demand) pthread_cond_wait(&qcond, &qlock);
    if (qtail - qhead == QUEUE) pthread_cond_broadcast(&qcond);  // room for start()
    uint32_t q = queue[qhead++ % QUEUE];
    pthread_mutex_unlock(&qlock);
    Slot *s = &slots[q >> 4];
    if (!atomic_exchange(&s->taken[q & 15], 1)) {
      uint64_t began = metrics_io_begin(s, 0);
      read_piece(&s->piece[q & 15], 0);
      metrics_io_end(s, 0, began);
      __atomic_fetch_add(&cache_stats.bytes, s->piece[q & 15].len, __ATOMIC_RELAXED);
      metrics_piece_done(s, s->piece[q & 15].len);
      piece_finished(s);
    }
  }
}

static void start_readers(void) {
  pthread_attr_t a;
  pthread_attr_init(&a);
  pthread_attr_set_qos_class_np(&a, QOS_CLASS_USER_INITIATED, 0);
  for (int k = 0; k < READERS; k++) {
    pthread_t t;
    pthread_create(&t, &a, reader, 0);
  }
}

static void read_demand(void *ctx) {
  uint32_t q = (uint32_t)(uintptr_t)ctx;
  Slot *s = &slots[q >> 4];
  uint64_t began = metrics_io_begin(s, 1);
  read_piece(&s->piece[q & 15], 0);
  metrics_io_end(s, 1, began);
  __atomic_fetch_add(&cache_stats.bytes, s->piece[q & 15].len, __ATOMIC_RELAXED);
  metrics_piece_done(s, s->piece[q & 15].len);
  piece_finished(s);
  if (--demand == 0) pthread_mutex_lock(&qlock), pthread_cond_broadcast(&qcond), pthread_mutex_unlock(&qlock);
}

// Reads slot i's pieces that no reader has started, at once.
static void claim(int i) {
  for (int k = 0; k < slots[i].npiece; k++)
    if (!atomic_exchange(&slots[i].taken[k], 1))
      demand++, dispatch_async_f(dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0),
                                 (void *)(uintptr_t)(i << 4 | k), read_demand);
}

// The model thread owns each group until finish(). Parallel substitution
// helpers may query it while that thread waits for them; they must not
// release it. Reader threads only complete the group's pending work.
static int slot_reading(int i) {
  dispatch_group_t g = slots[i].loading;
  return g && dispatch_group_wait(g, DISPATCH_TIME_NOW) != 0;
}

static void finish(int i) {
  if (!slots[i].loading) return;
  dispatch_group_wait(slots[i].loading, DISPATCH_TIME_FOREVER);
  dispatch_release(slots[i].loading);
  slots[i].loading = 0;
}

// Slot i is reused: its pieces no reader has started are not read.
static void cancel(int i) {
  if (!slots[i].loading) return;
  for (int k = 0; k < slots[i].npiece; k++)
    if (!atomic_exchange(&slots[i].taken[k], 1)) piece_finished(&slots[i]);
  finish(i);
}

// During a prompt pass at layer pass_layer, the experts of later layers
// will be needed in this pass: the victim is the least recently used slot of
// a layer already done.
static int pass_layer = -1;
void cache_pass(int layer) { pass_layer = layer; }

// A slot for (layer, e), resident and non-volatile: a free one, or a victim.
// When decoding, the victim has the lowest score, except the experts that
// the layer being served selected this token and read-aheads whose layer has
// not been served yet. In a prompt pass the victim is of a layer already
// done, else of a layer after the next. A prefetch that finds none is skipped
// (-1) rather than evicting experts in use.
static int serving = -1;  // the layer cache_request makes resident: its experts used this token stay

static int published = -1;  // --gpu-route: the layer whose table entries a committed buffer may still read

static int victim(void) {
  if (pass_layer < 0) {
    int best = -1;
    double lowest = INFINITY;
    for (int j = 0; j < max_slots; j++)
      if (score[j] < INFINITY && !slots[j].prefetched && !(slots[j].layer == serving && slots[j].used == token) &&
          (slots[j].layer != published || slots[j].expert < 0)) {
        double priority = eviction_priority(j);
        if (priority < lowest) best = j, lowest = priority;
      }
    return best;
  }
  for (int j = tail; j >= 0; j = slots[j].prev)
    if (slots[j].layer < pass_layer) return j;
  for (int j = tail; j >= 0; j = slots[j].prev)
    if (slots[j].layer > pass_layer + 1) return j;
  return -1;
}

// Maps slot i for the GPU; until the set is committed (between tokens) the
// GPU reaches it through gpu_use (fresh).
static void map_slot(int i) {
  if (slots[i].mapped) return;
  gpu_resident(&slots[i].b, 1);
  slots[i].mapped = 1, slots[i].fresh = 1;
  nmapped++;
  cache_stats.remaps++;
}

// Experiments rerank a small eligible shortlist. Production victim selection
// is unchanged; the live action policy cannot evict published/current work.
#ifdef MYDS4_PREDICT_LAB
static uint64_t lab_payload(int layer, int e) {
  uint64_t bytes = 0;
  if (layer >= 0 && layer < LAYERS && e >= 0 && e < EXPERTS)
    for (int k = 0; k < expert_spans[layer]; k++) bytes += expert_span[layer][e][k].len;
  return bytes;
}

static int action_victims(LabVictim *list) {
  double priority[128];
  int limit = predictor_lab_victim_limit();
  if (limit < 1 || limit > 128) die("invalid action victim limit");
  int n = 0;
  for (int j = 0; j < max_slots; j++) {
    Slot *s = &slots[j];
    if (isnan(score[j]) || score[j] == INFINITY || s->prefetched || s->pinned || slot_reading(j) ||
        (s->layer == serving && s->used == token) || (s->layer == published && s->expert >= 0)) continue;
    double p = eviction_priority(j);
    if (n == limit && p >= priority[limit - 1]) continue;
    int at = n < limit ? n++ : limit - 1;
    while (at > 0 && p < priority[at - 1]) {
      priority[at] = priority[at - 1], list[at] = list[at - 1], at--;
    }
    priority[at] = p;
    list[at] = (LabVictim){j, s->layer, s->expert, s->expert < 0 ? 0 : score[j] / weight, (double)(token - s->used),
                           lab_payload(s->layer, s->expert)};
  }
  return n;
}
#endif

static int take(int layer, int e, int optional) {
  int i;
#ifdef MYDS4_PREDICT_LAB
  int lab_policy = predictor_lab_cache_policy();
  int policy = optional && pass_layer < 0 && lab_policy &&
      (lab_policy >= 3 ? pf_issuer == PF_PC || pf_issuer == PF_H1 || pf_issuer == PF_H2 ||
                        pf_issuer == PF_H4 || pf_issuer == PF_H8 : pf_issuer == PF_DRAFT);
  uint64_t action_id = 0;
#endif
  if (nslots >= cap || !nfree) {
#ifdef MYDS4_PREDICT_LAB
    i = policy ? -1 : victim();
#else
    i = victim();
#endif
#ifdef MYDS4_PREDICT_LAB
    if (policy) {
      LabVictim choices[128];
      int n = action_victims(choices);
      if (!n) return -1;
      i = predictor_lab_choose(layer, e, lab_payload(layer, e), atomic_load(&prefetching), choices, n, &action_id);
      if (i < 0) return -1;
      int found = 0;
      for (int k = 0; k < n; k++) found |= choices[k].slot == i;
      if (!found) die("action policy chose an ineligible victim");
    }
#endif
    if (i < 0) {
      if (optional) return -1;
      die("the expert cache (%d slots) cannot hold the experts of two layers", nslots);
    }
    // Routing can demote a prefetch while its read is still running. Do not
    // stall the model for optional work or evict a higher-scoring resident.
    if (optional && slot_reading(i)) return -1;
    cancel(i);
    metrics_retire(i, optional);
    unlink_slot(i);
    if (slots[i].expert >= 0) where[slots[i].layer][slots[i].expert] = -1;
    cache_stats.evictions++;
    if (!slots[i].mapped) map_slot(i);
  } else {
#ifdef MYDS4_PREDICT_LAB
    if (policy && predictor_lab_choose(layer, e, lab_payload(layer, e), atomic_load(&prefetching),
                                      NULL, 0, &action_id) == -2) return -1;
#endif
    i = free_list[--nfree];
    slots[i].b = gpu_new(SLOT_SIZE);
    gpu_resident(&slots[i].b, 1), slots[i].fresh = 1;  // committed with the token's other changes
    slots[i].mapped = 1, nmapped++;
    nslots++;
  }
#ifdef MYDS4_PREDICT_LAB
  slots[i].action_id = action_id;
#endif
  slots[i].prefetched = 0;
  slots[i].layer = layer, slots[i].expert = e;
  where[layer][e] = i;
  score[i] = escore[layer * EXPERTS + e];
  push_front(i);
  return i;
}

// Reads slot i's expert in the background: on demand (urgent) at once, else
// through the prefetch queue.
static void start(int i, int urgent) {
  Slot *s = &slots[i];
  int n = 0;
  uint64_t region = 0;
  for (int k = 0; k < expert_spans[s->layer]; k++) {
    const FileSpan *p = &expert_span[s->layer][s->expert][k];
    n += split(s->piece + n, p->fd, p->off, p->len, s->b.ptr + region, &s->at[k]);
    s->at[k] += region;
    region += REGION(p->len);
  }
  s->npiece = n;
  s->background_load = !urgent;
  atomic_store_explicit(&s->unfinished, n, memory_order_relaxed);
  metrics_start(i, urgent);
  s->loading = dispatch_group_create();
  for (int k = 0; k < n; k++) dispatch_group_enter(s->loading), s->taken[k] = 0;
  if (urgent) {
    claim(i);
    return;
  }
  prefetching++;
  pthread_mutex_lock(&qlock);
  for (int k = 0; k < n; k++) {
    while (qtail - qhead == QUEUE) pthread_cond_wait(&qcond, &qlock);
    queue[qtail++ % QUEUE] = i << 4 | k;
  }
  pthread_cond_broadcast(&qcond);
  pthread_mutex_unlock(&qlock);
}

// Makes a layer's selected experts resident: starts reading the missing
// ones (cache_request), then waits for them and for those a prefetch is
// still reading (cache_wait); cache_load does both.
// The draft has separate buffer pools because its MXFP4 slots may be larger
// than the backbone's GGUF slots. The caller must have finished prior GPU
// use of this stage's pool; draft_stage already synchronizes before loading.
static int draft_victim(int layer, uint64_t stamp) {
  int first = max_slots + (layer - LAYERS) * draft_capacity, best = -1;
  if (layer < LAYERS || layer >= CACHE_LAYERS || first + draft_capacity > max_slots + npinned)
    die("draft cache layer %d has no pool", layer);
  for (int i = first; i < first + draft_capacity; i++) {
    Slot *s = &slots[i];
    if ((s->expert >= 0 && s->used == stamp) || slot_reading(i)) continue;
    if (s->expert < 0) return i;
    if (best < 0 || draft_age[i - max_slots] < draft_age[best - max_slots]) best = i;
  }
  return best;
}

static int draft_request(int layer, const int *experts, int n) {
  if (layer < LAYERS || layer >= CACHE_LAYERS || (layer - LAYERS + 1) * draft_capacity > npinned || n < 0)
    die("invalid bounded draft cache request for layer %d", layer);
  char wanted[DRAFT_EXPERTS] = {0};
  int unique = 0, reading = 0;
  for (int k = 0; k < n; k++) {
    int e = experts[k];
    if (e < 0 || e >= DRAFT_EXPERTS) die("draft expert %d is outside its stage", e);
    if (!wanted[e]) wanted[e] = 1, unique++;
  }
  if (unique > draft_capacity)
    die("draft stage needs %d experts together, but its cache holds %d", unique, draft_capacity);
  if (draft_clock == UINT64_MAX || (uint64_t)n > UINT64_MAX - draft_touch_clock)
    die("draft cache request counter exhausted");
  uint64_t stamp = ++draft_clock;
  for (int e = 0; e < DRAFT_EXPERTS; e++)
    if (wanted[e] && where[layer][e] >= 0) slots[where[layer][e]].used = stamp;
  for (int k = 0; k < n; k++) {
    int e = experts[k], i = where[layer][e];
    if (i >= 0) {
      if (slot_reading(i)) claim(i);
      cache_stats.hits++;
    } else {
      i = draft_victim(layer, stamp);
      if (i < 0) die("draft stage %d has no reusable slot: its slots are needed or still being read", layer - LAYERS);
      finish(i);  // reap a completed handle before assigning a new expert
      Slot *s = &slots[i];
      if (s->expert >= 0) where[layer][s->expert] = -1, cache_stats.evictions++;
      s->expert = e;
      s->prefetched = 0;
      where[layer][e] = i;
      cache_stats.misses++;
      start(i, 1);  // draft misses always use the demand-priority path
    }
    slots[i].used = stamp;
    draft_age[i - max_slots] = ++draft_touch_clock;
    score[i] = escore[layer * EXPERTS + e] += weight;
    reading += slot_reading(i);
  }
  return reading;
}

int cache_request(int layer, const int *experts, int n) {
  if (layer >= LAYERS && draft_capacity < DRAFT_EXPERTS) return draft_request(layer, experts, n);
  int reading = 0;
  serving = layer;
  for (int k = 0; k < n; k++)  // the resident ones first, so that the others' slots are not theirs
    if (where[layer][experts[k]] >= 0) slots[where[layer][experts[k]]].used = token;
  for (int k = 0; k < n; k++) {
    int e = experts[k], i = where[layer][e], pending;
    if (i >= 0) {
      if (!slots[i].pinned) unlink_slot(i), push_front(i), map_slot(i);
      pending = slot_reading(i);
      // A forecast can outlive its predicted pass and lose its protection.
      // Any later demand must promote its queued pieces, protected or not.
      if (pending) claim(i);
      if (slots[i].prefetched) {
        cache_stats.prefetch_hits++;
        if (pending) cache_stats.late++;
      } else {
        cache_stats.hits++;
      }
    } else {
      i = take(layer, e, 0);
      cache_stats.misses++;
      start(i, 1);
      pending = slot_reading(i);
    }
    // This is an actual served request even when GPU-route timeliness is not
    // observed. Keep it out of first-use rewards and later unused-byte totals.
    if (pf_enabled && !pf_active && layer < LAYERS) {
      PrefetchLoad *m = &pf_loads[i];
      if (m->tracked && m->origin_prefetch && !m->used) {
        m->used = 1;
        pf_stats[m->phase].unobserved_first_uses++;
      }
    }
    slots[i].prefetched = 0;
    slots[i].used = token;
    double *history = &escore[layer * EXPERTS + e];
#ifdef MYDS4_DECODE_ONLY_SCORE
    if (pass_layer < 0 || layer >= LAYERS) *history += weight;
#else
    *history += weight;
#endif
    score[i] = *history;
    reading += pending;
  }
  return reading;
}

int cache_slot(int layer, int e) { return where[layer][e] >= 0; }

void cache_repeat_credit(int layer, int expert, int repeats) {
  if (layer < 0 || layer >= LAYERS || expert < 0 || expert >= EXPERTS || repeats < 0 || repeats > VERIFY_DRAFTS)
    die("invalid repeated expert usage");
  if (!repeats) return;
  int i = where[layer][expert];
  if (i < 0 || slots[i].used != token) die("repeated usage requires a current expert request");
  score[i] = escore[layer * EXPERTS + expert] += weight * repeats;
}

int cache_ready(int layer, int e) {
  int i = where[layer][e];
  return i >= 0 && !slot_reading(i);
}

#ifdef MYDS4_PREDICT_LAB
int cache_action_state(int layer, int e, uint64_t action_id) {
  int i = where[layer][e];
  return i < 0 || !action_id || slots[i].action_id != action_id ? 0 : slot_reading(i) ? 1 : 2;
}
#endif

int cache_where(int layer, int e) {  // as cache_request will count it
  int i = where[layer][e];
  return i < 0 ? 0 : !slots[i].prefetched ? 1 : slot_reading(i) ? 3 : 2;
}

void cache_wait(int layer, const int *experts, int n) {
  double t = now();
  for (int k = 0; k < n; k++) finish(where[layer][experts[k]]);
  cache_stats.read_s += now() - t;
}

void cache_load(int layer, const int *experts, int n) {
  cache_request(layer, experts, n);
  cache_wait(layer, experts, n);
}

// Starts reading the first `admit` of the predicted experts (in rank order)
// that are not resident, while fewer than `inflight` prefetches are reading.
void cache_prefetch(int layer, const int *experts, int n, int admit, int inflight) {
  if (layer >= LAYERS && draft_capacity < DRAFT_EXPERTS) return;  // bounded draft pools are demand-only
  for (int k = 0; k < n && admit > 0 && prefetching < inflight; k++) {
    int e = experts[k];
    if (where[layer][e] >= 0) continue;
    int i = take(layer, e, 1);
    if (i < 0) return;
    slots[i].prefetched = 1;
    slots[i].used = token;
    start(i, 0);
    cache_stats.prefetches++;
    admit--;
  }
}

void cache_prefetch_from(int layer, const int *experts, int n, int admit, int inflight, int issuer) {
#ifdef MYDS4_PREDICT_LAB
  if (predictor_lab_suppress_prefetch()) return;
#endif
  int before = pf_issuer;
  if (issuer < 0 || issuer >= PF_ISSUERS) die("invalid prefetch issuer");
  pf_issuer = issuer;
#ifdef MYDS4_PREDICT_LAB
  predictor_lab_native_candidates(layer, experts, n, issuer);
#endif
  cache_prefetch(layer, experts, n, admit, inflight);
  pf_issuer = before;
}

// --gpu-route: the table the GPU looks a decoded row's experts up in (the
// resolve kernel): the GPU addresses of each expert's three spans in its
// slot for the experts read in full, 1 for those still being read, 0 for
// those without a slot. A layer's entries are written before the command
// buffer that reads them is committed, and until the next layer's are, no
// slot of the layer is evicted or released. A slot that joined the residency
// set this token, which the GPU may otherwise use only through gpu_use, is
// committed into it first.
static Buf tab;

const Buf *cache_table(void) {
  if (!tab.obj) tab = gpu_alloc((size_t)CACHE_LAYERS * EXPERTS * 3 * 8);
  return &tab;
}

int cache_publish(int layer) {
  uint64_t *t = (uint64_t *)cache_table()->ptr + (size_t)layer * EXPERTS * 3;
  int fresh = 0, loading = 0;
  for (int e = 0; e < EXPERTS; e++) {
    int i = where[layer][e];
    if (i >= 0 && slots[i].loading && !dispatch_group_wait(slots[i].loading, DISPATCH_TIME_NOW)) finish(i);
    fresh |= i >= 0 && !slots[i].loading && slots[i].fresh;
  }
  if (fresh) {
    gpu_residency_commit();
    for (int i = head; i >= 0; i = slots[i].next) slots[i].fresh = 0;
  }
  for (int e = 0; e < EXPERTS; e++, t += 3) {
    int i = where[layer][e], ok = i >= 0 && !slots[i].loading && (slots[i].mapped || slots[i].pinned);
    loading += i >= 0 && !ok;
    for (int k = 0; k < 3; k++) t[k] = ok ? slots[i].b.gpu + slots[i].at[k] : 0;
    if (i >= 0 && !ok) t[0] = 1;  // being read or not mapped: the CPU's path takes it, mapped, as substitute() does
  }
  published = layer;
  return loading;
}

void cache_unpublish(void) { published = -1; }

// After a decoded layer's routing: its read-ahead experts that it did not
// select may be evicted again, by the score earlier selections gave them.
void cache_demote(int layer) {
  for (int e = 0; e < EXPERTS; e++) {
    int i = where[layer][e];
    if (i >= 0) slots[i].prefetched = 0;
  }
}

// Cancels the read-aheads for `layer` that no reader has started, unless
// their expert is among keep[0..n): a newer prediction for the layer rules
// them out. Their slots hold no expert (expert -1) and are taken first.
void cache_refine(int layer, const int *keep, int n) {
  for (int e = 0; e < EXPERTS; e++) {
    int i = where[layer][e], k = 0, started = 0;
    if (i < 0 || !slots[i].prefetched || !slots[i].loading) continue;
    while (k < n && keep[k] != e) k++;
    for (int p = 0; p < slots[i].npiece; p++) started |= slots[i].taken[p];
    if (k < n || started) continue;
    cancel(i);  // a reader that started a piece meanwhile is waited for
    metrics_retire(i, 0);
    cache_stats.cancelled++;
    where[layer][e] = -1;
    slots[i].expert = -1;
    slots[i].prefetched = 0;
    score[i] = -INFINITY;
    unlink_slot(i);
    push_back(i);
  }
}

// Resizes the cache. It gives slots back, the least recently used of the
// layers below `below` first, on memory pressure, when anything was swapped
// out or the swapper woke, when the available memory falls below the
// reserve. Compression in RAM has no separate limit. It grows again after
// 32 calm calls by a quarter of the headroom; after swapping, growth uses
// only free pages for five minutes.
static void resize(int below) {
  static uint64_t swapouts, wakeups;
  static int calm, init;
  static double gave;  // when slots were last given back: their memory shows as free only later
  static double bad = -1e9;  // last observed swap-out or swapper wakeup
  vm_statistics64_data_t s = vm();
  uint64_t free = (uint64_t)s.free_count * vm_page_size, avail = avail_of(s), slot = SLOT_SIZE, margin = 4 * (uint64_t)GiB;
  uint64_t woke = sysctl_u64("vm.compressor.swapper.wakeups.total");
  if (!init) swapouts = s.swapouts, wakeups = woke, init = 1;
  int64_t room = mem_gpu_room();
  int level = mem_pressure(), swapped = s.swapouts > swapouts || woke > wakeups;
  int low = level >= 2 || swapped || avail < reserve;
  static int calm_map;
  static double unmapped;  // when the GPU's excess was last unmapped: the driver shows it about a second later
  if (swapped) bad = now();
  swapouts = s.swapouts, wakeups = woke;
  uint64_t headroom = now() - bad > 300 ? avail : free;
  if (low && (level >= 4 || swapped || now() - gave > 1)) {
    uint64_t to = avail;
    int64_t give = level >= 4 ? nslots / 4 : nslots / 16 + 1;
    if (to < reserve && (int64_t)((reserve - to) / slot) + 1 > give) give = (reserve - to) / slot + 1;
    cap = nslots - (int)give > 2 * EXPERTS ? nslots - (int)give : 2 * EXPERTS;
    calm = 0, gave = now();
  } else if (!low && ++calm >= 32 && cap < max_slots && headroom > reserve + margin) {
    cap += (int)((headroom - reserve - margin) / slot / 4) + 1;
    if (cap > max_slots) cap = max_slots;
    calm = 0;
  }
  if (room < 0 && now() - unmapped > 1) {  // past the GPU's limit: map fewer, exactly the excess fewer
    mcap = nmapped - (int)(-room / (int64_t)slot) - 1 > 2 * EXPERTS ? nmapped - (int)(-room / (int64_t)slot) - 1 : 2 * EXPERTS;
    calm_map = 0, unmapped = now();
  } else if (room >= 0 && ++calm_map >= 32 && mcap < max_slots && room > MAP_SLACK + (int64_t)GiB) {
    mcap += (int)((room - MAP_SLACK - (int64_t)GiB) / (int64_t)slot / 4) + 1;
    if (mcap > max_slots) mcap = max_slots;
    calm_map = 0;
  }
  for (int i = tail; nslots > cap && i >= 0;) {
    int prev = slots[i].prev;
    if (slots[i].layer < below && slots[i].layer != published) release(i), cache_stats.given_back++;
    i = prev;
  }
  while (nmapped > mcap) {  // the mapped slots with the lowest scores that nothing of this pass still needs
    int best = -1;
    for (int j = 0; j < max_slots; j++)
      if (slots[j].b.obj && slots[j].mapped && slots[j].layer < below && slots[j].layer != published &&
          !slots[j].loading && !slots[j].prefetched && (best < 0 || score[j] < score[best]))
        best = j;
    if (best < 0) break;
    gpu_resident(&slots[best].b, 0);
    slots[best].mapped = 0, slots[best].fresh = 0;
    nmapped--;
  }
  gpu_residency_commit();
}

// Between tokens (pass < 0) no GPU work references a slot, so any slot can be
// released. Within a prompt pass at layer `pass` the GPU may still use the
// slots of the layer before, so only those of earlier layers are released.
void cache_adapt_pass(int pass) {
  if (pass < 0) {
    advance_score_clock();
  }
  resize(pass < 0 ? CACHE_LAYERS : pass - 1);
  if (pass < 0)
    for (int i = head; i >= 0; i = slots[i].next) slots[i].fresh = 0;
}

void cache_adapt(void) { cache_adapt_pass(-1); }

// While the server idles no GPU work references a slot: the cache resizes
// as between tokens, so it gives slots back when other programs need the
// memory and leaves them as they are otherwise.
void cache_idle(void) { resize(CACHE_LAYERS); }

uint64_t cache_bytes(void) { return (uint64_t)nslots * SLOT_SIZE + pinned_bytes; }
int cache_slots(void) { return nslots; }
int cache_mapped(void) { return nmapped; }
int cache_cap(void) { return cap; }
int cache_max(void) { return max_slots; }
uint64_t cache_slot_bytes(void) { return SLOT_SIZE; }

static void metric_array(FILE *out, const char *const *names, const uint64_t *values, int n) {
  fputc('{', out);
  for (int i = 0; i < n; i++) fprintf(out, "%s\"%s\":%llu", i ? "," : "", names[i], (unsigned long long)values[i]);
  fputc('}', out);
}

void cache_metrics_report(FILE *out) {
  if (!pf_enabled) return;
  const char *const phase[] = {"prefill", "decode"};
  const char *const state[] = {"existing_ready", "fresh_timely_prefetch", "pending_prefetch", "pending_demand", "absent"};
  const char *const issuer[] = {"pc", "h1", "h2", "prefill_bulk", "other", "draft", "h4", "h8"};
  uint64_t unused_live[PF_PHASES] = {0}, unread_live[PF_PHASES] = {0};
  for (int l = 0; l < LAYERS; l++)
    for (int e = 0; e < EXPERTS; e++) {
      int i = where[l][e];
      if (i < 0) continue;
      PrefetchLoad *m = &pf_loads[i];
      if (!m->tracked || !m->origin_prefetch) continue;
      uint64_t done = atomic_load_explicit(&m->completed, memory_order_relaxed);
      if (!m->used) unused_live[m->phase] += done;
      unread_live[m->phase] += m->requested > done ? m->requested - done : 0;
    }
  for (int p = 0; p < PF_PHASES; p++) {
    PrefetchStats *s = &pf_stats[p];
    fprintf(out, "{\"prefetch_metrics\":{\"version\":1,\"phase\":\"%s\",\"passes\":%llu,\"rows\":%llu,"
                 "\"unsupported_gpu_route_passes\":%llu,\"unsupported_gpu_route_rows\":%llu,",
            phase[p], (unsigned long long)s->passes, (unsigned long long)s->rows,
            (unsigned long long)s->unsupported_passes, (unsigned long long)s->unsupported_rows);
    for (int v = 0; v < 2; v++) {
      fprintf(out, "\"%s\":{\"layer_batches\":%llu,\"unique_demands\":", v ? "served" : "exact_router",
              (unsigned long long)s->batches[v]);
      metric_array(out, state, s->states[v], PF_STATES);
      fputs(",\"requested_payload_bytes\":", out);
      metric_array(out, state, s->payload[v], PF_STATES);
      fprintf(out, ",\"rerequests_after_prefetch_eviction\":%llu,\"absent_after_prefetch_eviction\":%llu},",
              (unsigned long long)s->rerequests[v], (unsigned long long)s->absent_after_eviction[v]);
    }
    fputs("\"admissions_by_issuer\":", out);
    metric_array(out, issuer, s->admissions, PF_ISSUERS);
    fputs(",\"fresh_timely_first_uses_by_issuer\":", out);
    metric_array(out, issuer, s->timely_first, PF_ISSUERS);
    fputs(",\"first_uses_by_issuer\":", out);
    metric_array(out, issuer, s->used_by_issuer, PF_ISSUERS);
    fputs(",\"first_uses_before_io_started_by_issuer\":", out);
    metric_array(out, issuer, s->unstarted_at_use, PF_ISSUERS);
    fputs(",\"admission_to_first_use_ns_by_issuer\":", out);
    metric_array(out, issuer, s->age_at_use_ns, PF_ISSUERS);
    fputs(",\"target_pass_first_uses_by_issuer\":", out);
    metric_array(out, issuer, s->target_uses, PF_ISSUERS);
    fputs(",\"target_pass_first_uses_before_io_started_by_issuer\":", out);
    metric_array(out, issuer, s->target_unstarted, PF_ISSUERS);
    fputs(",\"target_pass_admission_to_first_use_ns_by_issuer\":", out);
    metric_array(out, issuer, s->target_age_ns, PF_ISSUERS);
    static const char *const io_kind[] = {"demand", "background", "promoted_prefetch"};
    fputs(",\"io_timing\":{", out);
    for (int k = 0; k < 3; k++)
      fprintf(out, "%s\"%s\":{\"pieces\":%llu,\"queue_ns\":%llu,\"service_ns\":%llu}", k ? "," : "", io_kind[k],
              (unsigned long long)atomic_load(&s->io_pieces[k]), (unsigned long long)atomic_load(&s->queue_ns[k]),
              (unsigned long long)atomic_load(&s->service_ns[k]));
    fputc('}', out);
#define METRIC(name, value) fprintf(out, ",\"%s\":%llu", name, (unsigned long long)(value))
    METRIC("prefetch_first_uses", s->first_uses);
    METRIC("first_uses_outside_observed_routes", s->unobserved_first_uses);
    METRIC("eventual_first_uses_after_target_pass", s->eventual_first_uses);
    METRIC("bytes_completed_at_first_use", s->first_completed_bytes);
    METRIC("bytes_remaining_at_first_use", s->first_remaining_bytes);
    METRIC("prefetch_origin_read_bytes", atomic_load_explicit(&s->prefetch_read_bytes, memory_order_relaxed));
    METRIC("demand_origin_read_bytes", atomic_load_explicit(&s->demand_read_bytes, memory_order_relaxed));
    METRIC("unused_prefetches_removed", s->unused_removed);
    METRIC("unused_prefetch_read_bytes_removed", s->unused_removed_bytes);
    METRIC("unread_bytes_cancelled_on_removal", s->unused_cancelled_bytes);
    METRIC("unconsumed_prefetch_read_bytes_live", unused_live[p]);
    METRIC("prefetch_bytes_outstanding_live", unread_live[p]);
    METRIC("prefetch_caused_evictions", s->prefetch_evictions);
#undef METRIC
    fputs(",\"scope\":\"backbone only; unions per layer batch; pass IDs are not emitted-token IDs; "
          "decode rows include rejected verification rows; read bytes follow admission phase and requested aligned "
          "piece lengths; first-use counts follow consumption phase; live byte counters are asynchronous snapshots\","
          "\"pollution_scope\":\"observed eviction and rerequest counts, no counterfactual baseline\"}}\n", out);
  }
}

// Read while the cache changes, for display: a count may be off by the
// experts that change meanwhile.
uint64_t cache_uncached_bytes(int *experts) {
  uint64_t b = 0;
  int n = 0;
  for (int l = 0; l < LAYERS; l++)
    for (int e = 0; e < EXPERTS; e++)
      if (where[l][e] < 0) {
        n++;
        for (int k = 0; k < expert_spans[l]; k++) b += expert_span[l][e][k].len;
      }
  *experts = n;
  return b;
}
