// SSD reads, the routed-expert cache and its adaptive size.
//
// Experts are read with F_NOCACHE over 16 KiB-aligned spans: unaligned
// uncached reads still fill the page cache. A slot holds one expert's spans
// (expert_span) one after the other: its MXFP4 codes (w1, w2, w3), then its
// scales; or with --experts its GGUF rows of w1, w3 and w2. Every slot's
// buffer has the size of the largest expert, so that any expert fits any
// slot; when a layer's experts are more than a sixteenth smaller than the
// largest (a GGUF set quantizes its layers differently), a buffer has the
// size of its layer's experts instead, and the cache counts bytes (`sized`).
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
#include "expert_store.h"
#include <CoreFoundation/CoreFoundation.h>
#include <IOKit/IOKitLib.h>
#include <dispatch/dispatch.h>
#include <mach/mach.h>
#include <math.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <string.h>
#include <sys/sysctl.h>
#include <unistd.h>
#include <time.h>

#include "myds4.h"
#include "pass.h"

#define SECTOR 16384ull
#define PIECE (2ull << 20)
#define GiB (1024.0 * 1024 * 1024)
#define REGION(len) (((len) + 3 * SECTOR - 1) / SECTOR * SECTOR)  // a span's aligned cover, at most

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
#define FILE_BLOCK 4096ull

static void read_row(void *ctx, size_t i) {
  Row *r = (Row *)ctx + i;
  _Alignas(16384) uint8_t tmp[3 * SECTOR];
  uint64_t a = r->off & ~(FILE_BLOCK - 1), e = (r->off + r->len + FILE_BLOCK - 1) & ~(FILE_BLOCK - 1);
  if (e - a > sizeof tmp) die("io_rows: row too long");
  if (pread(r->fd, tmp, e - a, a) < (ssize_t)(r->off + r->len - a)) die("read failed at offset %llu", a);
  memcpy(r->dst, tmp + (r->off - a), r->len);
}

// Rows are small random reads: their speed is the number in flight, so 64
// lanes on the global queue, which adds threads while others wait in pread.
#define LANES 64
typedef struct { Row *r; int n, lane, lanes; } Lane;
uint64_t io_rows_workspace(int rows) { return (uint64_t)rows * (sizeof(uint64_t) + sizeof(Row)); }

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
  uint64_t *key = memory_host_required((size_t)n * 8, 0);  // row, then request index in the low 20 bits
  for (int i = 0; i < n; i++) key[i] = (uint64_t)rows[i] << 20 | i;
  qsort(key, n, 8, cmp_u64);
  Row *r = memory_host_required((size_t)n * sizeof *r, 0);
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
  memory_host_free(key);
  memory_host_free(r);
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

// Free pages already include speculative file pages; these are the others.
static uint64_t file_of(vm_statistics64_data_t s) {
  uint64_t file = s.external_page_count > s.speculative_count ? s.external_page_count - s.speculative_count : 0;
  return file * vm_page_size;
}

// Memory available to this program: the free pages and the file cache
// beyond the share the kernel keeps (vm.vm_page_filecache_min, 23.9 GiB with
// 128 GiB). While the file cache is below that share, the kernel compresses
// and swaps other programs' memory rather than drop cached files: on the
// machine measured it compressed 8.6 pages for every file page it dropped,
// and on 2026-11-09, with that share counted as available, free pages fell
// to 2.2 GiB with 23 GiB of file cache, and macOS swapped. When the share
// cannot be read, only the free pages count.
static uint64_t avail_of(vm_statistics64_data_t s) {
  uint64_t keep = sysctl_u64("vm.vm_page_filecache_min"), file = file_of(s) / vm_page_size;
  return ((uint64_t)s.free_count + (keep && file > keep ? file - keep : 0)) * vm_page_size;
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

// Metal's working-set recommendation guides residency, not RAM allocation.
// Registration can query memory before Metal starts; do not cache that zero.
int64_t mem_gpu_limit(void) {
  uint64_t target = gpu_working_set();
  return target ? (int64_t)target : INT64_MAX;
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
  uint32_t at[STORE_PARTS];  // where the spans are in b
  int16_t layer, expert;
  int prev, next;  // links in the recency list, head = most recent
  int prefetched;  // loaded on a prediction, not yet requested
  const Pass *issuer;  // the pass that predicted it: only that pass cancels or demotes it
  int pinned;      // a DSpark pool buffer, exempt from backbone eviction/unmapping
  int fresh;       // joined the residency set this token: needs gpu_use until it is committed
  int mapped;      // in the residency set (or joining it: fresh)
  int mapped_once; // this buffer has been mapped; the next mapping is a remap
  uint64_t used;   // token of the last use
  unsigned holds; // jobs awaiting or using this expert; released after GPU completion
  dispatch_group_t loading;  // owned until finish(), including after its reads complete
  Piece piece[16];
  int npiece;
  int background_load; // this load originally occupied a prefetch admission
  _Atomic int unfinished; // read or cancelled pieces that have not completed
  _Atomic char taken[16];  // piece k has been started (or dropped)
  _Atomic int read;        // pieces of this load that a reader completed
  uint64_t extent;         // bytes of b this load writes
  uint64_t written;        // bytes of b that completed loads have written
  MemorySlot memory;
  ExpertStore *store;      // the store the slot belongs to, for the memory pool's callbacks
} Slot;


static uint64_t reserve;  // every store keeps it available to other programs

// This program's memory is what other programs leave, less the reserve kept
// available to them: model memory and KV follow the available memory
// (memory_follow), within `model_gib` when it is set. Expert slots whose
// reads have not written them yet count as in use (memory_unwritten).
static uint64_t cache_unwritten(void);
void memory_configure(double model_gib, double reserve_gib) {
  memory_init(model_gib > 0 ? (uint64_t)(model_gib * GiB) : 0);
  cache_reserve(reserve_gib);
  memory_follow(mem_available, reserve);
  memory_unwritten(cache_unwritten);
}

void cache_reserve(double gib) { reserve = (uint64_t)((gib > 0 ? gib : CACHE_RESERVE_DEFAULT) * GiB); }

// Observation metadata is separate from the eviction policy's prefetched flag.
// The model thread owns it until a read group finishes; readers update only
// atomic byte and timing counters, before leaving that group. No trace records are saved.
typedef struct {
  uint64_t target_pass, requested, queued_ns;
  _Atomic uint64_t completed, first_io_ns;
  uint8_t origin_prefetch, used, tracked, phase, issuer;
} PrefetchLoad;
typedef struct {
  uint64_t passes, rows, batches[2];
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
// An expert store's state: its slots, eviction order, readers, statistics
// and memory policy. Each model with routed experts creates its own
// (cache_create).
#define READERS 4
#define QUEUE (1u << 14)
struct ExpertStore {
  int index;  // in stores[]
  CacheStats stats;
  // the layout (cache_create)
  int layers, scored_layers, experts, chosen, stage_experts, stage_min, parts;
  const FileSpan *span;
  const int *spans;
  uint64_t extra;  // ExpertLayout.extra, transform and context
  void (*transform)(void *context, int layer, int expert, uint8_t *slot, const uint32_t *at, uint64_t slot_bytes);
  void *context;
  uint64_t SLOT_SIZE;  // the largest expert in use
  // `sized`: the scored layers' experts differ in size by more than a
  // sixteenth (cache_required). Then a slot's buffer has the size of its
  // layer's experts (layer_bytes), or the largest when a prompt pass
  // allocates it, and the cache counts bytes: cap, mcap and max_cap are bytes
  // in units of SLOT_SIZE, and more slots than cap fit in them. Otherwise
  // every slot has SLOT_SIZE and the limits count slots.
  uint64_t *layer_bytes;  // [scored_layers]; 0: the layer's spans are not known
  int sized;
  uint64_t bytes, mapped_bytes;  // the scored slots' buffers, and those mapped for the GPU
  int max_cap;                   // the most cap and mcap may be (max_slots when not sized)
  uint64_t conversions;          // buffers given back for one of another size
  uint64_t token;
  Slot *slots;
  int pass_layer;
  MemorySlot *cache_backbone;
  int cache_busy;
  int minimum_slots;
  double last_pool_growth;
  int16_t *where;  // [layers][experts]: an expert's slot (WHERE), -1: none
  uint64_t *generation;  // [layers]: counts the changes of the layer's `where` (place, cache_layer_ready)
  // cache_layer_ready's last finding of each layer: its generation then, and
  // the unmappings (resize) by then. Until either changes the layer stays ready.
  uint64_t *ready_generation, *ready_unmaps, unmaps;
  int any_fresh;  // a slot may be fresh: cache_adapt clears them
  int nslots, cap, max_slots, *free_list, nfree, npinned;
  int nmapped, mcap;  // slots mapped for the GPU, and how many may be
  uint64_t pinned_bytes;
  int draft_capacity;
  uint64_t draft_clock, draft_touch_clock;
  uint64_t *draft_age;  // [stage layers * stage_experts]: recency within the separate pools
  int pf_enabled, pf_active, pf_phase, pf_rows, pf_issuer;
  uint64_t pf_pass;
  PrefetchLoad *pf_loads;
  PrefetchStats pf_stats[PF_PHASES];
  uint8_t *pf_ghost;  // [scored_layers][experts] (GHOST)
  // the eviction order (see the scores below)
  int head, tail;
  double weight;  // the current token's
  double *escore;  // [layers][experts]: per expert
  double *dscore;  // [layers][experts]: the part of escore that decoding's requests gave
  double near_weight;  // the current token's weight of a near miss (see cache_near)
  double *nscore;  // [layers][experts]: near misses, each of the near weight then
  double *score;  // per slot: its expert's priority; INFINITY for a free slot, -INFINITY for an empty one
  int retention_half_life, retention_read_cost;
  double retention_growth;
  double *reload_cost;  // [scored_layers][experts]
  uint64_t pinned_slot_size;
  int layout_ready;
  // the readers' queue (see the prefetch reads below)
  uint32_t queue[QUEUE], qhead, qtail;  // entries: slot << 4 | piece
  uint16_t qlayer[QUEUE];               // each entry's layer, as it was queued
  pthread_mutex_t qlock;
  pthread_cond_t qcond;
  _Atomic int demand, prefetching;  // demand pieces, and prefetched slots, in flight
  pthread_mutex_t progress_lock;
  pthread_cond_t progress_cond;
  _Atomic int progress_enabled;
  uint64_t progress_epoch;
  int serving;  // the layer cache_request makes resident: its experts used this token stay
  unsigned held_references;
  unsigned *layer_holds;  // [layers]: passes whose GPU work reaches the layer through its table
  // resize's readings and clocks (the clock of the last give-back is shared)
  uint64_t swapouts;
  int calm, init;
  double bad;   // last swap-out while the cache could release memory
  int calm_map;
  double unmapped;  // when the GPU's excess was last unmapped: the driver shows it about a second later
};
// Every store, for what concerns them all: the slot memory that reads have
// not written yet, draining their reads, and the time of the last give-back.
// That clock is shared so that one shortfall goes back once: after any
// store gives slots back, none gives more until the memory readings can show
// it (resize).
enum { STORES = 8 };
static ExpertStore *stores[STORES];
static int nstores;
static double gave;  // when a store last gave slots back: their memory shows as free only later
#define WHERE(l, e) st->where[(size_t)(l) * st->experts + (e)]
#define GHOST(l, e) st->pf_ghost[(size_t)(l) * st->experts + (e)]
#define SPAN(l, e, k) st->span[((size_t)(l) * st->experts + (e)) * st->parts + (k)]
// The arrays' bytes: a slot index and two scores for every expert of every
// layer, a recency for every stage expert (and one more, so it is never empty).
#define WHERE_BYTES ((size_t)st->layers * st->experts * sizeof *st->where)
#define SCORE_BYTES ((size_t)st->layers * st->experts * sizeof *st->escore)
#define DRAFT_AGE_BYTES (((size_t)(st->layers - st->scored_layers) * st->stage_experts + 1) * sizeof *st->draft_age)

// Whether the expert in slot s must stay: a pass holds it, or GPU work reaches
// its layer through the layer's table.
static int slot_held(const ExpertStore *st, const Slot *s) {
  return s->holds || (s->expert >= 0 && st->layer_holds[s->layer]);
}

// Puts expert e of the layer in slot i (-1: in none). A table of the layer's
// addresses is then out of date (cache_layer_ready).
static void place(ExpertStore *st, int layer, int e, int i) {
  WHERE(layer, e) = i;
  st->generation[layer]++;
}

ExpertStore *cache_create(const ExpertLayout *l) {
  if (l->experts < 1 || l->experts > STORE_EXPERTS_MAX || l->scored_layers < 1 || l->scored_layers > l->layers ||
      l->chosen < 1 || l->chosen > l->experts || l->stage_experts > l->experts || l->stage_min > l->stage_experts ||
      l->parts < 1 || l->parts > STORE_PARTS || !l->span || !l->spans)
    die("invalid expert layout");
  if (nstores == STORES) die("more than %d expert stores", STORES);
  ExpertStore *st = xmalloc(sizeof *st);
  memset(st, 0, sizeof *st);
  st->pass_layer = -1, st->cache_busy = 1, st->pf_issuer = PF_OTHER, st->head = -1, st->tail = -1, st->weight = 1;
  st->retention_half_life = CACHE_HALF_LIFE_DEFAULT, st->serving = -1, st->bad = -1e9;
  pthread_mutex_init(&st->qlock, NULL), pthread_cond_init(&st->qcond, NULL);
  pthread_mutex_init(&st->progress_lock, NULL), pthread_cond_init(&st->progress_cond, NULL);
  st->index = nstores, stores[nstores++] = st;
  st->layers = l->layers, st->scored_layers = l->scored_layers, st->experts = l->experts, st->chosen = l->chosen;
  st->stage_experts = l->stage_experts, st->stage_min = l->stage_min, st->parts = l->parts;
  st->span = l->span, st->spans = l->spans;
  st->extra = l->transform ? l->extra : 0, st->transform = l->transform, st->context = l->context;
  size_t scored = (size_t)l->scored_layers * l->experts;
  st->where = xmalloc(WHERE_BYTES);
  memset(st->where, 0, WHERE_BYTES);
  st->generation = xmalloc((size_t)l->layers * sizeof *st->generation);
  memset(st->generation, 0, (size_t)l->layers * sizeof *st->generation);
  st->ready_generation = xmalloc((size_t)l->layers * sizeof *st->ready_generation);
  memset(st->ready_generation, 0xff, (size_t)l->layers * sizeof *st->ready_generation);  // never found ready
  st->layer_holds = calloc((size_t)l->layers, sizeof *st->layer_holds);
  if (!st->layer_holds) die("out of memory");
  st->ready_unmaps = xmalloc((size_t)l->layers * sizeof *st->ready_unmaps);
  memset(st->ready_unmaps, 0, (size_t)l->layers * sizeof *st->ready_unmaps);
  st->escore = xmalloc(SCORE_BYTES), st->dscore = xmalloc(SCORE_BYTES), st->nscore = xmalloc(SCORE_BYTES);
  memset(st->escore, 0, SCORE_BYTES), memset(st->dscore, 0, SCORE_BYTES), memset(st->nscore, 0, SCORE_BYTES);
  st->near_weight = 1;
  st->reload_cost = xmalloc(scored * sizeof *st->reload_cost);
  memset(st->reload_cost, 0, scored * sizeof *st->reload_cost);
  st->pf_ghost = xmalloc(scored);
  memset(st->pf_ghost, 0, scored);
  st->draft_age = xmalloc(DRAFT_AGE_BYTES);
  memset(st->draft_age, 0, DRAFT_AGE_BYTES);
  st->layer_bytes = xmalloc((size_t)l->scored_layers * sizeof *st->layer_bytes);
  memset(st->layer_bytes, 0, (size_t)l->scored_layers * sizeof *st->layer_bytes);
  st->minimum_slots = 2 * l->experts;
  st->draft_capacity = l->stage_experts;
  return st;
}

// The bytes a slot for an expert of the layer takes.
static uint64_t slot_need(const ExpertStore *st, int layer) {
  return st->sized && layer < st->scored_layers && st->layer_bytes[layer] ? st->layer_bytes[layer] : st->SLOT_SIZE;
}

// Whether a buffer of `size` bytes may hold an expert that takes `need`: it
// is large enough and wastes at most a sixteenth of the expert's bytes.
static int slot_fits(uint64_t size, uint64_t need) { return size >= need && size - need <= need / 16; }

// The cache's bytes and its mapped bytes in the unit of its limits: slots
// of the largest size when the layers differ, else slots.
static int used_units(const ExpertStore *st) {
  return st->sized ? (int)((st->bytes + st->SLOT_SIZE - 1) / st->SLOT_SIZE) : st->nslots;
}
static int mapped_units(const ExpertStore *st) {
  return st->sized ? (int)((st->mapped_bytes + st->SLOT_SIZE - 1) / st->SLOT_SIZE) : st->nmapped;
}
// Whether the cache holds more than its limit, or maps more than may be mapped.
static int over_cap(const ExpertStore *st) {
  return st->sized ? st->bytes > (uint64_t)st->cap * st->SLOT_SIZE : st->nslots > st->cap;
}
static int over_mcap(const ExpertStore *st) {
  return st->sized ? st->mapped_bytes > (uint64_t)st->mcap * st->SLOT_SIZE : st->nmapped > st->mcap;
}
// The most the limits may be.
static int cap_ceiling(const ExpertStore *st) { return st->sized ? st->max_cap : st->max_slots; }

CacheStats *cache_counters(ExpertStore *st) { return &st->stats; }
static int slot_reading(ExpertStore *st, int i);
static uint64_t metrics_ns(void) { return (uint64_t)(now() * 1e9); }

void cache_metrics_begin(ExpertStore *st, int phase, int rows) {
  if (!st->pf_enabled) return;
  if (phase < 0 || phase >= PF_PHASES || rows < 1 || st->pf_pass == UINT64_MAX) die("invalid prefetch metric pass");
  st->pf_phase = phase, st->pf_rows = rows, st->pf_active = 1, st->pf_pass++;
  st->pf_stats[phase].passes++, st->pf_stats[phase].rows += rows;
}

void cache_metrics_end(ExpertStore *st) { st->pf_active = 0; }

void cache_observe_snapshot(ExpertStore *st, int layer, CacheObservation *o) {
  o->active = st->pf_enabled && st->pf_active && layer >= 0 && layer < st->scored_layers;
  if (!o->active) return;
  o->layer = layer, o->phase = st->pf_phase, o->pass = st->pf_pass;
  for (int e = 0; e < st->experts; e++) {
    int i = WHERE(layer, e);
    o->state[e] = PF_ABSENT, o->first[e] = 0, o->completed[e] = 0;
    o->ghost[e] = GHOST(layer, e);
    if (i < 0) continue;
    PrefetchLoad *m = &st->pf_loads[i];
    int pending = slot_reading(st, i);
    o->first[e] = m->tracked && m->origin_prefetch && !m->used;
    o->completed[e] = atomic_load_explicit(&m->completed, memory_order_relaxed);
    o->state[e] = pending ? (m->origin_prefetch ? PF_PENDING_PREFETCH : PF_PENDING_DEMAND)
                         : o->first[e] && m->target_pass == st->pf_pass ? PF_FRESH_READY : PF_EXISTING_READY;
  }
}

void cache_observe_routes(ExpertStore *st, const CacheObservation *o, const int *ids, int rows, int stride, int served) {
  if (!o->active) return;
  if (served < 0 || served > 1 || o->pass != st->pf_pass) die("invalid prefetch observation");
  uint8_t used[st->experts];
  memset(used, 0, sizeof used);
  for (int r = 0; r < rows; r++)
    for (int k = 0; k < st->chosen; k++) {
      int e = ids[r * stride + k];
      if (e < 0 || e >= st->experts) die("invalid expert in prefetch observation");
      used[e] = 1;
    }
  PrefetchStats *s = &st->pf_stats[o->phase];
  s->batches[served]++;
  for (int e = 0; e < st->experts; e++) {
    if (!used[e]) continue;
    int state = o->state[e], i = WHERE(o->layer, e);
    s->states[served][state]++;
    for (int k = 0; k < st->spans[o->layer]; k++) s->payload[served][state] += SPAN(o->layer, e, k).len;
    if (o->ghost[e]) {
      s->rerequests[served]++;
      s->absent_after_eviction[served] += state == PF_ABSENT;
      if (served) GHOST(o->layer, e) = 0;
    }
    if (!served || !o->first[e] || i < 0 || st->pf_loads[i].used) continue;
    PrefetchLoad *m = &st->pf_loads[i];
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

static void metrics_start(ExpertStore *st, int i, int urgent) {
  if (!st->pf_enabled) return;
  PrefetchLoad *m = &st->pf_loads[i];
  m->target_pass = st->pf_pass, m->used = 0, m->origin_prefetch = !urgent;
  m->phase = st->pf_phase, m->issuer = st->pf_issuer;
  m->tracked = st->pf_active && st->slots[i].layer < st->scored_layers;
  m->requested = 0;
  m->queued_ns = metrics_ns();
  atomic_store_explicit(&m->completed, 0, memory_order_relaxed);
  atomic_store_explicit(&m->first_io_ns, 0, memory_order_relaxed);
  for (int k = 0; k < st->slots[i].npiece; k++) m->requested += st->slots[i].piece[k].len;
  if (m->tracked && !urgent) st->pf_stats[m->phase].admissions[m->issuer]++;
}

static uint64_t metrics_io_begin(ExpertStore *st, Slot *slot, int urgent_worker) {
  if (!st->pf_enabled) return 0;
  PrefetchLoad *m = &st->pf_loads[slot - st->slots];
  if (!m->tracked) return 0;
  uint64_t began = metrics_ns(), zero = 0;
  atomic_compare_exchange_strong_explicit(&m->first_io_ns, &zero, began, memory_order_release, memory_order_relaxed);
  int kind = !m->origin_prefetch ? 0 : urgent_worker ? 2 : 1;
  PrefetchStats *s = &st->pf_stats[m->phase];
  atomic_fetch_add_explicit(&s->io_pieces[kind], 1, memory_order_relaxed);
  atomic_fetch_add_explicit(&s->queue_ns[kind], began - m->queued_ns, memory_order_relaxed);
  return began;
}

static void metrics_io_end(ExpertStore *st, Slot *slot, int urgent_worker, uint64_t began) {
  if (!began) return;
  PrefetchLoad *m = &st->pf_loads[slot - st->slots];
  int kind = !m->origin_prefetch ? 0 : urgent_worker ? 2 : 1;
  atomic_fetch_add_explicit(&st->pf_stats[m->phase].service_ns[kind], metrics_ns() - began, memory_order_relaxed);
}

static void metrics_piece_done(ExpertStore *st, Slot *slot, uint64_t bytes) {
  if (!st->pf_enabled) return;
  PrefetchLoad *m = &st->pf_loads[slot - st->slots];
  atomic_fetch_add_explicit(&m->completed, bytes, memory_order_relaxed);
  if (m->tracked) {
    PrefetchStats *s = &st->pf_stats[m->phase];
    atomic_fetch_add_explicit(m->origin_prefetch ? &s->prefetch_read_bytes : &s->demand_read_bytes,
                              bytes, memory_order_relaxed);
  }
}

// Called only after cancel/finish joins all old readers, before reusing or
// releasing the slot. Ghosts record observed rerequests, not baseline misses.
static void metrics_retire(ExpertStore *st, int i, int by_prefetch) {
  if (!st->pf_enabled || st->slots[i].layer >= st->scored_layers || st->slots[i].expert < 0) return;
  PrefetchLoad *m = &st->pf_loads[i];
  if (by_prefetch && st->pf_active) {
    st->pf_stats[st->pf_phase].prefetch_evictions++;
    GHOST(st->slots[i].layer, st->slots[i].expert) = 1;
  }
  if (m->tracked && m->origin_prefetch && !m->used) {
    uint64_t done = atomic_load_explicit(&m->completed, memory_order_relaxed);
    PrefetchStats *s = &st->pf_stats[m->phase];
    s->unused_removed++;
    s->unused_removed_bytes += done;
    s->unused_cancelled_bytes += m->requested > done ? m->requested - done : 0;
  }
  m->tracked = 0;
}

void cache_draft_limit(ExpertStore *st, int per_stage) {
  if (per_stage && (per_stage < st->stage_min || per_stage > st->stage_experts))
    die("draft cache needs %d to %d slots per stage", st->stage_min, st->stage_experts);
  st->draft_capacity = per_stage ? per_stage : st->stage_experts;
}

int cache_draft_capacity(ExpertStore *st) { return st->draft_capacity; }

static int draft_pool_count(ExpertStore *st, int layers) {
  return layers > st->scored_layers ? (layers - st->scored_layers) * st->draft_capacity : 0;
}
// The driver counts a slot taken out of the residency set for about a second
// more; the mapped slots leave this much room for those.
#define MAP_SLACK (3ll << 30)

// The eviction order: every expert has a score, the sum of a weight for each
// request that selected it; the weight doubles over the configured number of
// forward-pass aging steps. A verification batch advances once, as does the
// end of prefill. A decoded layer also credits its near misses, the experts
// its router ranked just below the ones it selected (cache_near): each counts
// NEAR_CREDIT of a selection, with a weight that doubles every NEAR_HALF_LIFE
// steps, because such an expert is often selected within a few tokens and
// seldom later. A slot's priority is its expert's two sums, each over its
// current weight: the selections decayed by the configured half-life and the
// near misses by theirs. The victim has the lowest priority, an empty slot
// first. Optional read-cost weighting affects that priority, not the score
// history. A recency list (head = most recently used) orders the slots for
// giving slots back.
enum { NEAR_HALF_LIFE = 12 };
#define NEAR_CREDIT 0.5

void cache_retention(ExpertStore *st, int half_life, int read_cost) {
  if (st->slots) die("cache retention must be set before cache_init");
  if (half_life < 0 || half_life > CACHE_HALF_LIFE_MAX || (read_cost != 0 && read_cost != 1))
    die("invalid cache retention policy");
  st->retention_half_life = half_life ? half_life : CACHE_HALF_LIFE_DEFAULT;
  st->retention_read_cost = read_cost;
  // Keep the original constant expression for the default; cache other
  // ratios once instead of evaluating exp2 at every forward pass.
  st->retention_growth = st->retention_half_life == CACHE_HALF_LIFE_DEFAULT ? exp2(1.0 / CACHE_HALF_LIFE_DEFAULT)
                                                                  : exp2(1.0 / st->retention_half_life);
}

static void prepare_reload_cost(ExpertStore *st, int layers) {
  if (!st->retention_read_cost) return;
  for (int l = 0; l < layers && l < st->scored_layers; l++)
    for (int e = 0; e < st->experts; e++) {
      uint64_t bytes = 0;
      for (int k = 0; k < st->spans[l]; k++) {
        const FileSpan *p = &SPAN(l, e, k);
        uint64_t a = p->off & ~(SECTOR - 1), end = (p->off + p->len + SECTOR - 1) & ~(SECTOR - 1);
        bytes += end - a;
      }
      if (!bytes || bytes > st->SLOT_SIZE) die("expert read footprint does not fit its cache slot");
      st->reload_cost[l * st->experts + e] = (double)bytes / st->SLOT_SIZE;
    }
}

static double eviction_priority(ExpertStore *st, int i) {
  if (!st->retention_read_cost || !isfinite(st->score[i]) || st->slots[i].expert < 0 ||
      st->slots[i].layer < 0 || st->slots[i].layer >= st->scored_layers) return st->score[i];
  return st->score[i] * st->reload_cost[st->slots[i].layer * st->experts + st->slots[i].expert];
}

// The priority of expert key = layer * experts + expert, of a scored layer.
static double expert_priority(const ExpertStore *st, size_t key) {
  return st->escore[key] / st->weight + NEAR_CREDIT * st->nscore[key] / st->near_weight;
}

// Ages every history by one step. The two sums of a scored layer's expert
// decay at different rates, so its slot's priority is computed again; the
// draft stages' pinned slots keep their raw score, rescaled with the history.
static void advance_score_clock(ExpertStore *st) {
  st->token++;
  if ((st->weight *= st->retention_growth) > 0x1p500) {  // rescale raw history and pinned scores together
    for (int k = 0; k < st->layers * st->experts; k++) st->escore[k] /= st->weight, st->dscore[k] /= st->weight;
    for (int i = st->max_slots; i < st->max_slots + st->npinned; i++) st->score[i] /= st->weight;
    st->weight = 1;
  }
  if ((st->near_weight *= exp2(1.0 / NEAR_HALF_LIFE)) > 0x1p500) {
    for (int k = 0; k < st->layers * st->experts; k++) st->nscore[k] /= st->near_weight;
    st->near_weight = 1;
  }
  for (int i = 0; i < st->max_slots; i++) {
    const Slot *s = &st->slots[i];
    if (isfinite(st->score[i]) && s->expert >= 0 && s->layer >= 0 && s->layer < st->scored_layers)
      st->score[i] = expert_priority(st, (size_t)s->layer * st->experts + s->expert);
  }
}

void cache_near(ExpertStore *st, int layer, const int *experts, int n) {
  if (layer < 0 || layer >= st->scored_layers) return;
  for (int k = 0; k < n; k++) {
    size_t key = (size_t)layer * st->experts + experts[k];
    st->nscore[key] += st->near_weight;
    int i = WHERE(layer, experts[k]);
    if (i >= 0 && i < st->max_slots && isfinite(st->score[i])) st->score[i] = expert_priority(st, key);
  }
}

static void unlink_slot(ExpertStore *st, int i) {
  Slot *s = &st->slots[i];
  if (s->prev >= 0) st->slots[s->prev].next = s->next; else st->head = s->next;
  if (s->next >= 0) st->slots[s->next].prev = s->prev; else st->tail = s->prev;
}

static void push_back(ExpertStore *st, int i) {
  st->slots[i].next = -1;
  st->slots[i].prev = st->tail;
  if (st->tail >= 0) st->slots[st->tail].next = i;
  st->tail = i;
  if (st->head < 0) st->head = i;
}

static void push_front(ExpertStore *st, int i) {
  st->slots[i].prev = -1;
  st->slots[i].next = st->head;
  if (st->head >= 0) st->slots[st->head].prev = i;
  st->head = i;
  if (st->tail < 0) st->tail = i;
}

static void cancel(ExpertStore *st, int i), start_readers(ExpertStore *st), start(ExpertStore *st, int i, int urgent),
    finish(ExpertStore *st, int i), map_slot(ExpertStore *st, int i);

// Gives slot i's buffer back: its expert leaves, and the slot is free again.
// by_prefetch: a read-ahead takes its room (for the prefetch metrics).
static void discard(ExpertStore *st, int i, int by_prefetch) {
  if (slot_held(st, &st->slots[i])) die("cannot release an expert held by a runahead job");
  cancel(st, i);
  metrics_retire(st, i, by_prefetch);
  unlink_slot(st, i);
  if (st->slots[i].expert >= 0) place(st, st->slots[i].layer, st->slots[i].expert, -1);
  if (st->slots[i].mapped) st->nmapped--, st->mapped_bytes -= st->slots[i].b.size, st->slots[i].mapped = 0;
  st->bytes -= st->slots[i].b.size;
  gpu_free(&st->slots[i].b);
  st->score[i] = INFINITY;
  st->free_list[st->nfree++] = i;
  st->nslots--;
  st->stats.evictions++;
}

static void release(ExpertStore *st, int i) { discard(st, i, 0); }

void cache_memory(ExpertStore *st, MemorySlot *backbone) { st->cache_backbone = backbone; }
void cache_active(ExpertStore *st, int active) { st->cache_busy = !!active; }
void cache_working_rows(ExpertStore *st, int rows) {
  int experts = rows * st->chosen < st->experts ? rows * st->chosen : st->experts;
  int slots = 2 * experts;
  // Passes that run together each need their slots: while others run, the
  // minimum only rises, and a pass that needs fewer does not take it away.
  if (__atomic_load_n(&gpu_other_passes, __ATOMIC_RELAXED) > 0 && slots < st->minimum_slots) slots = st->minimum_slots;
  st->minimum_slots = slots;
  // The model reserves these slots' memory before such a pass (cache_required),
  // so limits that cache_init or a resize set for fewer rows rise to them.
  if (st->max_slots) {
    int fewest = st->minimum_slots < cap_ceiling(st) ? st->minimum_slots : cap_ceiling(st);
    if (st->cap < fewest) st->cap = fewest;
    if (st->mcap < fewest) st->mcap = fewest;
  }
}

static int pool_can_evict(void *context) {
  Slot *slot = context;
  ExpertStore *st = slot->store;
  if (!slot->b.obj || slot_held(st, slot)) return 0;
  if (!st->cache_busy) return 1;
  if (slot->pinned || slot->used == st->token) return 0;
  return st->pass_layer < 0 || slot->layer < st->pass_layer - 1;
}

static void pool_evict(void *context) {
  Slot *slot = context;
  ExpertStore *st = slot->store;
  int index = (int)(slot - st->slots);
  if (slot->pinned) {
    cancel(st, index);
    if (slot->expert >= 0) place(st, slot->layer, slot->expert, -1);
    slot->expert = -1;
    gpu_free(&slot->b);
  } else release(st, index);
  st->stats.given_back++;
  gpu_residency_commit();
}

static Buf pool_buffer(ExpertStore *st, int index, size_t bytes, int mapped) {
  if (!st->cache_backbone) return mapped ? gpu_alloc(bytes) : gpu_new(bytes);
  MemorySlot *previous = memory_scope(&st->slots[index].memory);
  // Slot growth borrows free budget. Reuse the native victim if it cannot
  // grow; a routing prediction never evicts another model's weight group.
  Buf buffer = gpu_try_new(bytes, MEMORY_SPECULATIVE);
  memory_scope(previous);
  if (buffer.obj && mapped) gpu_resident(&buffer, 1), gpu_residency_commit();
  return buffer;
}

// A slot's buffer that a demand cannot do without: required memory, for which
// the pool takes idle model memory back. Without an object when the pool
// cannot make the room.
static Buf pool_buffer_required(ExpertStore *st, int index, size_t bytes) {
  if (!st->cache_backbone) return gpu_new(bytes);
  MemorySlot *previous = memory_scope(&st->slots[index].memory);
  Buf buffer = gpu_try_new(bytes, MEMORY_REQUIRED);
  memory_scope(previous);
  return buffer;
}

uint64_t cache_required(ExpertStore *st, int layers) {
  if (!st->layout_ready) {
  uint64_t pin_size = 0;  // the stages' slots
  for (int l = 0; l < layers; l++)  // a slot holds the layer's largest expert
    for (int e = 0; e < st->experts; e++) {
      uint64_t size = st->extra, pieces = 0;
      for (int k = 0; k < st->spans[l]; k++)  // an empty span (an expert with fewer spans) takes no room
        if (SPAN(l, e, k).len)
          size += REGION(SPAN(l, e, k).len), pieces += (REGION(SPAN(l, e, k).len) + PIECE - 1) / PIECE;
      if (pieces > 16) die("layer %d's experts take %llu reads of %llu MiB, more than a slot tracks", l, pieces, PIECE >> 20);
      uint64_t *at = l < st->scored_layers ? &st->SLOT_SIZE : &pin_size;
      if (size > *at) *at = size;
      if (l < st->scored_layers && st->spans[l] && size > st->layer_bytes[l]) st->layer_bytes[l] = size;
    }
  // The layers' own sizes are used when one of them is more than a sixteenth
  // below the largest. A store whose layers are closer keeps slots of one
  // size, which any expert fits, and counts slots: buffers of nearly one
  // size would be given back for one another without holding more experts.
  for (int l = 0; l < layers && l < st->scored_layers; l++)
    if (st->layer_bytes[l] && st->layer_bytes[l] < st->SLOT_SIZE - st->SLOT_SIZE / 16) st->sized = 1;
  prepare_reload_cost(st, layers);
  st->npinned = draft_pool_count(st, layers);
  st->pinned_bytes = st->npinned * pin_size;
    st->pinned_slot_size = pin_size;
    st->layout_ready = 1;
  }
  return st->minimum_slots * gpu_allocation_size(st->SLOT_SIZE) + st->npinned * gpu_allocation_size(st->pinned_slot_size);
}

void cache_init(ExpertStore *st, uint64_t budget, int layers) {
  if (!st->retention_growth) cache_retention(st, 0, 0);
  const char *observe = getenv("MYDS4_PREFETCH_METRICS");
  st->pf_enabled = observe && *observe && strcmp(observe, "0");
  cache_required(st, layers);
  memset(st->where, 0xff, WHERE_BYTES);
  st->draft_clock = st->draft_touch_clock = 0;
  memset(st->draft_age, 0, DRAFT_AGE_BYTES);
  uint64_t pin_size = st->pinned_slot_size;
  uint64_t free = mem_free(), avail = mem_available(), gpu_limit = gpu_working_set(), limit = gpu_limit;
  if (!reserve) cache_reserve(0);
  int fixed = budget > 0;
  if (!budget) budget = avail > reserve ? avail - reserve : 0;
  if (st->cache_backbone && memory_usage().limit < limit) limit = memory_usage().limit;
  if (budget > limit) budget = limit;
  budget = budget > st->pinned_bytes ? budget - st->pinned_bytes : 0;
  // the cache may grow up to max_slots when memory becomes free; a given budget is a fixed bound
  st->max_slots = (int)((fixed ? budget : limit) / st->SLOT_SIZE);
  if (st->max_slots > st->scored_layers * st->experts) st->max_slots = st->scored_layers * st->experts;
  if (st->max_slots < st->minimum_slots) st->max_slots = st->minimum_slots;
  st->max_cap = st->max_slots;
  if (st->sized) {  // the same bytes hold more slots when they are the smallest layer's
    uint64_t smallest = st->SLOT_SIZE;
    for (int l = 0; l < layers && l < st->scored_layers; l++)
      if (st->layer_bytes[l] && st->layer_bytes[l] < smallest) smallest = st->layer_bytes[l];
    uint64_t most = (uint64_t)st->max_cap * st->SLOT_SIZE / smallest;
    st->max_slots = most < (uint64_t)st->scored_layers * st->experts ? (int)most : st->scored_layers * st->experts;
  }
  if (budget < st->minimum_slots * st->SLOT_SIZE)
    die("an expert cache of %.1f GiB is too small: a prompt needs %.1f GiB", budget / GiB, (double)st->minimum_slots * st->SLOT_SIZE / GiB);
  // File-cache reclamation is handled by macOS as slots are populated.
  uint64_t within = avail > reserve + st->pinned_bytes ? avail - reserve - st->pinned_bytes : 0;
  st->cap = (int)((budget < within ? budget : within) / st->SLOT_SIZE);
  if (st->cap > cap_ceiling(st)) st->cap = cap_ceiling(st);
  if (st->cap < st->minimum_slots) st->cap = st->minimum_slots;
  // The mapped slots stay within Metal's recommendation, beside weights that
  // count as resident only once the GPU uses them.
  int64_t room = mem_gpu_room() - (int64_t)st->pinned_bytes - (int64_t)mem_footprint() - MAP_SLACK;
  st->mcap = room > st->minimum_slots * (int64_t)st->SLOT_SIZE ? (int)(room / (int64_t)st->SLOT_SIZE) : st->minimum_slots;
  if (st->mcap > cap_ceiling(st)) st->mcap = cap_ceiling(st);
  st->slots = xmalloc((st->max_slots + st->npinned) * sizeof *st->slots);
  for (int i = 0; i < st->max_slots + st->npinned; i++) st->slots[i].store = st;
  if (st->cache_backbone) for (int i = 0; i < st->max_slots + st->npinned; i++) {
    memory_slot_init(&st->slots[i].memory, st->cache_backbone->account, st->cache_backbone,
                     "routed expert", pool_evict, &st->slots[i]);
    st->slots[i].memory.can_evict = pool_can_evict;
    st->slots[i].memory.automatic = 1;
  }
  if (st->pf_enabled) {
    st->pf_loads = xmalloc((st->max_slots + st->npinned) * sizeof *st->pf_loads);
    memset(st->pf_loads, 0, (st->max_slots + st->npinned) * sizeof *st->pf_loads);
  }
  st->score = xmalloc((st->max_slots + st->npinned) * sizeof *st->score);
  for (int i = 0; i < st->max_slots + st->npinned; i++) st->score[i] = INFINITY;
  start_readers(st);
  st->free_list = xmalloc(st->max_slots * sizeof *st->free_list);
  for (int i = st->max_slots - 1; i >= 0; i--) st->free_list[st->nfree++] = i;
  for (int i = st->max_slots; i < st->max_slots + st->npinned; i++) {  // separate, MXFP4-sized draft pools
    Slot *s = &st->slots[i];
    s->b = pool_buffer(st, i, pin_size, 1), s->written = 0;
    if (!s->b.obj) die("reserved draft expert memory is unavailable");
    s->layer = st->scored_layers + (i - st->max_slots) / st->draft_capacity;
    s->expert = st->draft_capacity == st->stage_experts ? (i - st->max_slots) % st->draft_capacity : -1;
    s->pinned = 1;
    if (s->expert >= 0) {
      place(st, s->layer, s->expert, i);
      start(st, i, 1);
    }
  }
  for (int i = st->max_slots; i < st->max_slots + st->npinned; i++) finish(st, i);
  int64_t gpu_room = mem_gpu_room();
  fprintf(stderr, "expert cache: slot ceiling %d (%.1f GiB) within the shared model budget, up to %d mapped for the GPU, %.1f GiB free and %.1f GiB "
          "of reclaimable file cache, %.1f GiB kept for other programs, GPU resident memory %.1f GiB",
          st->cap, st->cap * (double)st->SLOT_SIZE / GiB, st->mcap, free / GiB, (avail > free ? avail - free : 0) / GiB, reserve / GiB,
          mem_gpu() / GiB);
  if (gpu_room < INT64_MAX) fprintf(stderr, " (%.1f GiB below the residency target)", gpu_room / GiB);
  fprintf(stderr, ", GPU working set %.1f GiB\n", gpu_limit / GiB);
  fprintf(stderr, "cache policy: half-life %d forward passes, read-cost weighting %s\n",
          st->retention_half_life, st->retention_read_cost ? "on" : "off");
  if (st->sized) {
    uint64_t smallest = st->SLOT_SIZE;
    for (int l = 0; l < layers && l < st->scored_layers; l++)
      if (st->layer_bytes[l] && st->layer_bytes[l] < smallest) smallest = st->layer_bytes[l];
    fprintf(stderr, "expert slots have their layer's size, %.2f to %.2f MiB: the slot ceiling counts the largest, "
            "and up to %d slots fit\n", smallest / 1048576.0, st->SLOT_SIZE / 1048576.0, st->max_slots);
  }
  if (st->npinned) {
    if (st->draft_capacity == st->stage_experts)
      fprintf(stderr, "the DSpark stages' %d experts: %.1f GiB, resident\n", st->npinned, st->pinned_bytes / GiB);
    else
      fprintf(stderr, "DSpark cache: %d slots per stage, %.1f GiB reserved, native experts read on demand\n",
              st->draft_capacity, st->pinned_bytes / GiB);
  }
}

const Buf *cache_get(ExpertStore *st, int layer, int expert, uint32_t at[STORE_PARTS], int *fresh) {
  int i = WHERE(layer, expert);
  if (i < 0) return 0;
  if (!st->slots[i].pinned) map_slot(st, i);
  *fresh = st->slots[i].fresh;
  memcpy(at, st->slots[i].at, (size_t)st->parts * sizeof *at);  // the layout's spans
  return &st->slots[i].b;
}

// Prefetch reads: a prefetch's pieces wait in a queue for READERS threads,
// which start a piece only while no demand piece is queued or running. A demand read
// so shares the SSD with at most READERS pieces, and a demand request for an
// expert still being prefetched claims the pieces no reader has started.
// The threads and the demand reads run above the utility QoS class, whose
// disk reads macOS throttles.

uint64_t cache_read_epoch(ExpertStore *st) {
  atomic_store(&st->progress_enabled, 1);
  pthread_mutex_lock(&st->progress_lock);
  uint64_t value = st->progress_epoch;
  pthread_mutex_unlock(&st->progress_lock);
  return value;
}

void cache_wait_progress(ExpertStore *st, uint64_t epoch) {
  struct timespec until;
  clock_gettime(CLOCK_REALTIME, &until);
  until.tv_nsec += 20000000;
  if (until.tv_nsec >= 1000000000) until.tv_sec++, until.tv_nsec -= 1000000000;
  pthread_mutex_lock(&st->progress_lock);
  if (st->progress_epoch == epoch) pthread_cond_timedwait(&st->progress_cond, &st->progress_lock, &until);
  pthread_mutex_unlock(&st->progress_lock);
}

// `read`: a reader wrote the piece; 0 when it is dropped unread.
static void piece_finished(ExpertStore *st, Slot *s, int piece, int read) {
  if (read) atomic_fetch_add_explicit(&s->read, 1, memory_order_release);
  // Release the admission when the last I/O/cancellation finishes. Deferring
  // this to a dispatch notification can block a newer forecast after data is ready.
  int last_piece = atomic_fetch_sub_explicit(&s->unfinished, 1, memory_order_acq_rel) == 1;
  if (last_piece) {
    if (s->background_load) atomic_fetch_sub_explicit(&st->prefetching, 1, memory_order_relaxed);
    // The layout's step after a complete load, before its group completes:
    // the expert counts as read only after it.
    // The transform's bytes end where the slot's own buffer ends (experts.c
    // finds them from the buffer's size).
    if (st->transform && atomic_load_explicit(&s->read, memory_order_acquire) == s->npiece)
      st->transform(st->context, s->layer, s->expert, s->b.ptr, s->at, s->pinned ? st->SLOT_SIZE : s->b.size);
  }
  dispatch_group_leave(s->loading);
  // Publish after leaving the group. Every piece signals: the reader that
  // decrements unfinished last need not leave the dispatch group last.
  if (atomic_load(&st->progress_enabled)) {
    pthread_mutex_lock(&st->progress_lock);
    st->progress_epoch++;
    pthread_cond_broadcast(&st->progress_cond);
    pthread_mutex_unlock(&st->progress_lock);
  }
}

// Queues a read-ahead piece of `layer`, under qlock and with room in the
// queue, after the queued pieces of its layer and the layers before it: the
// background readers take the lowest layer's pieces first, because a pass
// runs its layers in order and the nearest layer needs its experts soonest.
// Pieces of one layer keep the order they were queued in.
static void queue_push(ExpertStore *st, uint32_t entry, uint16_t layer) {
  uint32_t at = st->qtail++;
  for (; at != st->qhead && st->qlayer[(at - 1) % QUEUE] > layer; at--)
    st->queue[at % QUEUE] = st->queue[(at - 1) % QUEUE], st->qlayer[at % QUEUE] = st->qlayer[(at - 1) % QUEUE];
  st->queue[at % QUEUE] = entry, st->qlayer[at % QUEUE] = layer;
}

// The SSD's time for expert reads, for measuring where the time goes: the
// seconds with at least one read in flight (their union), and the demand and
// background reads' summed seconds. Every store's reads count.
static pthread_mutex_t ssd_lock = PTHREAD_MUTEX_INITIALIZER;
static int ssd_reads;
static double ssd_since, ssd_busy, ssd_read[2];  // ssd_read: demand, background

static double ssd_begin(void) {
  double t = now();
  pthread_mutex_lock(&ssd_lock);
  if (!ssd_reads++) ssd_since = t;
  pthread_mutex_unlock(&ssd_lock);
  return t;
}

static void ssd_end(int background, double began) {
  double t = now();
  pthread_mutex_lock(&ssd_lock);
  ssd_read[background] += t - began;
  if (!--ssd_reads) ssd_busy += t - ssd_since;
  pthread_mutex_unlock(&ssd_lock);
}

void io_ssd_times(double *busy, double *demand, double *background) {
  pthread_mutex_lock(&ssd_lock);
  *busy = ssd_busy + (ssd_reads ? now() - ssd_since : 0);
  *demand = ssd_read[0], *background = ssd_read[1];
  pthread_mutex_unlock(&ssd_lock);
}

static void *reader(void *arg) {
  ExpertStore *st = arg;
  for (;;) {
    pthread_mutex_lock(&st->qlock);
    // Demand reads go first, as long as they are this pass's own: while other
    // passes run, read-aheads go on beside another pass's demand reads, which
    // would otherwise hold them back almost all the time.
    while (st->qhead == st->qtail || (st->demand && __atomic_load_n(&gpu_other_passes, __ATOMIC_RELAXED) <= 0))
      pthread_cond_wait(&st->qcond, &st->qlock);
    if (st->qtail - st->qhead == QUEUE) pthread_cond_broadcast(&st->qcond);  // room for start()
    uint32_t q = st->queue[st->qhead++ % QUEUE];
    // Cancellation examines taken under this lock. Announce ownership
    // before a popped piece can become invisible to the queue scan.
    int owned = !atomic_exchange(&st->slots[q >> 4].taken[q & 15], 1);
    pthread_mutex_unlock(&st->qlock);
    Slot *s = &st->slots[q >> 4];
    if (owned) {
      uint64_t began = metrics_io_begin(st, s, 0);
      double reading = ssd_begin();
      read_piece(&s->piece[q & 15], 0);
      ssd_end(1, reading);
      metrics_io_end(st, s, 0, began);
      __atomic_fetch_add(&st->stats.bytes, s->piece[q & 15].len, __ATOMIC_RELAXED);
      metrics_piece_done(st, s, s->piece[q & 15].len);
      piece_finished(st, s, q & 15, 1);
    }
  }
}

static void start_readers(ExpertStore *st) {
  pthread_attr_t a;
  pthread_attr_init(&a);
  pthread_attr_set_qos_class_np(&a, QOS_CLASS_USER_INITIATED, 0);
  for (int k = 0; k < READERS; k++) {
    pthread_t t;
    pthread_create(&t, &a, reader, st);
  }
}

// A demand read's piece: its store's index in the high 32 bits, then its
// slot << 4 | piece.
static void read_demand(void *ctx) {
  uint64_t entry = (uintptr_t)ctx;
  ExpertStore *st = stores[entry >> 32];
  uint32_t q = (uint32_t)entry;
  Slot *s = &st->slots[q >> 4];
  uint64_t began = metrics_io_begin(st, s, 1);
  double reading = ssd_begin();
  read_piece(&s->piece[q & 15], 0);
  ssd_end(0, reading);
  metrics_io_end(st, s, 1, began);
  __atomic_fetch_add(&st->stats.bytes, s->piece[q & 15].len, __ATOMIC_RELAXED);
  metrics_piece_done(st, s, s->piece[q & 15].len);
  piece_finished(st, s, q & 15, 1);
  if (--st->demand == 0) pthread_mutex_lock(&st->qlock), pthread_cond_broadcast(&st->qcond), pthread_mutex_unlock(&st->qlock);
}


static void submit_demand(ExpertStore *st, uint32_t q) {
  uint64_t entry = (uint64_t)st->index << 32 | q;
  dispatch_async_f(dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), (void *)(uintptr_t)entry, read_demand);
}

// Promotes slot i's pieces that no reader has started to the urgent path.
static void claim(ExpertStore *st, int i) {
  for (int k = 0; k < st->slots[i].npiece; k++)
    if (!atomic_exchange(&st->slots[i].taken[k], 1)) {
      // demand counts both queued and active pieces. Background readers
      // therefore stay suppressed even before a bounded worker starts.
      st->demand++, submit_demand(st, (uint32_t)(i << 4 | k));
    }
}

// The model thread owns each group until finish(). Parallel substitution
// helpers may query it while that thread waits for them; they must not
// release it. Reader threads only complete the group's pending work.
static int slot_reading(ExpertStore *st, int i) {
  dispatch_group_t g = st->slots[i].loading;
  return g && dispatch_group_wait(g, DISPATCH_TIME_NOW) != 0;
}

static void finish(ExpertStore *st, int i) {
  if (!st->slots[i].loading) return;
  dispatch_group_wait(st->slots[i].loading, DISPATCH_TIME_FOREVER);
  dispatch_release(st->slots[i].loading);
  st->slots[i].loading = 0;
  // A load that dropped a piece adds nothing: its pages may be partly written.
  if (atomic_load_explicit(&st->slots[i].read, memory_order_acquire) == st->slots[i].npiece &&
      st->slots[i].extent > st->slots[i].written)
    st->slots[i].written = st->slots[i].extent;
}

void cache_drain(void) {
  for (int k = 0; k < nstores; k++) {
    ExpertStore *st = stores[k];
    for (int i = 0; st->slots && i < st->max_slots + st->npinned; i++) finish(st, i);
  }
}

// Slot i is reused: its pieces no reader has started are not read.
static void cancel(ExpertStore *st, int i) {
  if (!st->slots[i].loading) return;
  for (int k = 0; k < st->slots[i].npiece; k++)
    if (!atomic_exchange(&st->slots[i].taken[k], 1)) piece_finished(st, &st->slots[i], k, 0);
  finish(st, i);
}

// The charged slot bytes that no completed load has written (memory_unwritten):
// a new buffer's pages are in use only once a read writes them, and a slot
// beyond its expert's bytes stays unwritten until a larger expert fills it.
// A load whose pieces were all read counts before the model thread reaps it.
static uint64_t cache_unwritten(void) {
  uint64_t bytes = 0;
  for (int k = 0; k < nstores; k++) {
    ExpertStore *st = stores[k];
    for (int i = 0; st->slots && i < st->max_slots + st->npinned; i++) {
      const Slot *s = &st->slots[i];
      if (!s->b.obj) continue;
      uint64_t written = s->written;
      if (s->loading && atomic_load_explicit(&s->read, memory_order_acquire) == s->npiece && s->extent > written)
        written = s->extent;
      if (s->b.allocated > written) bytes += s->b.allocated - written;
    }
  }
  return bytes;
}

// The layer a prompt pass is at (-1: decoding). The pass still needs the
// experts of this layer and the next; it is done with earlier layers.
void cache_pass(ExpertStore *st, int layer) { st->pass_layer = layer; }

void cache_pass_variables(ExpertStore *st, PassVariable out[CACHE_PASS_VARIABLES]) {
  out[0] = (PassVariable){&st->pass_layer, sizeof st->pass_layer};
  out[1] = (PassVariable){&st->serving, sizeof st->serving};
}

// A slot for (layer, e), resident and non-volatile: a free one, or a victim.
// When decoding, the victim has the lowest priority (eviction_priority), so
// the experts used most stay; the experts that the layer being served
// selected this token and read-aheads whose layer has not been served yet
// are not victims. In a prompt pass the victim is of a layer the pass is done
// with, else of a layer after the next, and is the one whose expert decoding
// used least (decoding_priority): the experts the prompt alone needed go
// first, whatever their layer, and decoding after the prompt finds the
// experts it used before. Among equals, as in a cache that has not decoded
// yet, the least recently used goes, so that the experts of the prompt's
// last layers and rows, which decoding needs first, stay longest. A
// prefetch that finds none is skipped (-1) rather than evicting experts in
// use.

// What a prompt pass evicts by: how much decoding used slot i's expert (its
// part of the score, read-cost weighted as eviction_priority weights it); an
// empty slot first.
static double decoding_priority(ExpertStore *st, int i) {
  if (!isfinite(st->score[i]) || st->slots[i].expert < 0 || st->slots[i].layer < 0 || st->slots[i].layer >= st->layers)
    return st->score[i];
  double d = st->dscore[st->slots[i].layer * st->experts + st->slots[i].expert];
  if (!st->retention_read_cost || st->slots[i].layer >= st->scored_layers) return d;
  return d * st->reload_cost[st->slots[i].layer * st->experts + st->slots[i].expert];
}

static int victim(ExpertStore *st) {
  int best = -1;
  double lowest = INFINITY;
  if (st->pass_layer < 0) {
    for (int j = 0; j < st->max_slots; j++)
      if (st->score[j] < INFINITY && !st->slots[j].prefetched &&
          !slot_held(st, &st->slots[j]) &&
          !(st->slots[j].expert >= 0 && st->slots[j].layer == st->serving && st->slots[j].used == st->token)) {
        double priority = eviction_priority(st, j);
        if (priority < lowest) best = j, lowest = priority;
      }
    return best;
  }
  // From the least recently used on, so that the oldest of equals goes.
  for (int later = 0; later < 2 && best < 0; later++)
    for (int j = st->tail; j >= 0; j = st->slots[j].prev) {
      int l = st->slots[j].layer;
      if (later ? l <= st->pass_layer + 1 : l >= st->pass_layer) continue;
      if (st->score[j] == INFINITY || slot_held(st, &st->slots[j])) continue;
      double priority = decoding_priority(st, j);
      if (best < 0 || priority < lowest) best = j, lowest = priority;
    }
  return best;
}

// The victim when the layers' experts differ in size, for an expert that
// takes `need` bytes. It is victim()'s slot if that slot's buffer fits the
// expert. Otherwise a slot whose buffer fits is reused if it is nearly as
// low: when decoding, the lowest-priority fitting slot among the FIT_RANK
// lowest priorities, which are the slots the next misses would take anyway;
// in a prompt pass, the least recently used slot of the same priority whose
// buffer is large enough (a pass's buffers are of the largest size: see
// take). When there is none, *convert is set and victim()'s slot is
// returned: the caller gives buffers back until one of the wanted size fits
// in the cache's bytes (take).
enum { FIT_RANK = 64 };
static int victim_sized(ExpertStore *st, uint64_t need, int *convert) {
  int best = -1, fit = -1;
  double lowest = INFINITY, fit_lowest = INFINITY;
  *convert = 0;
  if (st->pass_layer < 0) {
    for (int j = 0; j < st->max_slots; j++)
      if (st->score[j] < INFINITY && !st->slots[j].prefetched &&
          !slot_held(st, &st->slots[j]) &&
          !(st->slots[j].expert >= 0 && st->slots[j].layer == st->serving && st->slots[j].used == st->token)) {
        double priority = eviction_priority(st, j);
        if (priority < lowest) best = j, lowest = priority;
        if (priority < fit_lowest && slot_fits(st->slots[j].b.size, need)) fit = j, fit_lowest = priority;
      }
    if (best < 0 || fit == best) return best;
    if (fit >= 0) {
      int below = 0;
      for (int j = 0; j < st->max_slots && below < FIT_RANK; j++)
        below += st->score[j] < INFINITY && !st->slots[j].prefetched && !slot_held(st, &st->slots[j]) &&
                 !(st->slots[j].expert >= 0 && st->slots[j].layer == st->serving && st->slots[j].used == st->token) &&
                 eviction_priority(st, j) < fit_lowest;
      if (below < FIT_RANK) return fit;
    }
    *convert = 1;
    return best;
  }
  for (int later = 0; later < 2 && best < 0; later++)
    for (int j = st->tail; j >= 0; j = st->slots[j].prev) {
      int l = st->slots[j].layer;
      if (later ? l <= st->pass_layer + 1 : l >= st->pass_layer) continue;
      if (st->score[j] == INFINITY || slot_held(st, &st->slots[j])) continue;
      double priority = decoding_priority(st, j);
      if (best < 0 || priority < lowest) best = j, lowest = priority;
      if (st->slots[j].b.size >= need && (fit < 0 || priority < fit_lowest)) fit = j, fit_lowest = priority;
    }
  if (best < 0 || st->slots[best].b.size >= need) return best;
  if (fit >= 0 && fit_lowest == lowest) return fit;
  *convert = 1;
  return best;
}


// Maps slot i for the GPU; until the set is committed (between tokens) the
// GPU reaches it through gpu_use (fresh).
static void map_slot(ExpertStore *st, int i) {
  if (st->slots[i].mapped) return;
  gpu_resident(&st->slots[i].b, 1);
  st->slots[i].mapped = 1, st->slots[i].fresh = 1, st->any_fresh = 1;
  st->nmapped++, st->mapped_bytes += st->slots[i].b.size;
  if (st->slots[i].mapped_once) st->stats.remaps++;
  st->slots[i].mapped_once = 1;
}

// A pass that needs a slot while every slot is held waits for the GPU work of
// other passes to release one; after this long it gives up (a program error).
enum { TAKE_RESIDENT = -2 };
typedef struct {
  ExpertStore *st;
  double deadline;
} SlotWait;

static int slot_unheld(ExpertStore *st) {
  for (int j = 0; j < st->max_slots; j++)
    if (st->score[j] < INFINITY && !slot_held(st, &st->slots[j])) return j;
  return -1;
}
static int slot_released(void *context) {
  SlotWait *w = context;
  return slot_unheld(w->st) >= 0 || now() > w->deadline;
}
static void slot_released_block(void *context) {
  while (!slot_released(context)) sched_yield();
}

// The slot whose buffer a conversion gives back next, -1 if none. Nothing
// holds it, and it is no read-ahead that is still protected. When decoding,
// it is the lowest priority among the slots not used or read ahead in this
// token: the memory pool does not free those either (pool_can_evict). In a
// prompt pass it follows victim()'s order, the layers the pass is done with
// before the layers after the next, but leaves out the layer just before the
// pass's own, whose buffers the pool does not free either.
static int victim_freeable(ExpertStore *st) {
  int best = -1;
  double lowest = INFINITY;
  if (st->pass_layer < 0) {
    for (int j = 0; j < st->max_slots; j++)
      if (st->score[j] < INFINITY && !st->slots[j].prefetched && !slot_held(st, &st->slots[j]) &&
          st->slots[j].used != st->token) {
        double priority = eviction_priority(st, j);
        if (priority < lowest) best = j, lowest = priority;
      }
    return best;
  }
  for (int later = 0; later < 2 && best < 0; later++)
    for (int j = st->tail; j >= 0; j = st->slots[j].prev) {
      int l = st->slots[j].layer;
      if (later ? l <= st->pass_layer + 1 : l >= st->pass_layer - 1) continue;
      if (st->score[j] == INFINITY || slot_held(st, &st->slots[j])) continue;
      double priority = decoding_priority(st, j);
      if (best < 0 || priority < lowest) best = j, lowest = priority;
    }
  return best;
}

// The slot a demand reuses when no buffer can be given back or allocated for
// it, -1 if none: its buffer is at least `need` bytes, whatever it wastes, and
// nothing holds it. Among such slots victim()'s preferences decide. First
// come the slots victim() takes from: when decoding, those that are no
// protected read-ahead and that the layer being served did not select this
// token; in a prompt pass, those of the layers the pass is done with, then
// those of the layers after the next. The other slots come last. Within
// each group the lowest priority goes, and among equals the least recently
// used.
static int victim_large_enough(ExpertStore *st, uint64_t need) {
  int best[3] = {-1, -1, -1};
  double lowest[3] = {INFINITY, INFINITY, INFINITY};
  for (int j = st->tail; j >= 0; j = st->slots[j].prev) {
    const Slot *s = &st->slots[j];
    if (st->score[j] == INFINITY || slot_held(st, s) || s->b.size < need) continue;
    int group;
    double priority;
    if (st->pass_layer < 0) {
      group = s->prefetched || (s->expert >= 0 && s->layer == st->serving && s->used == st->token) ? 2 : 0;
      priority = eviction_priority(st, j);
    } else {
      group = s->layer < st->pass_layer ? 0 : s->layer > st->pass_layer + 1 ? 1 : 2;
      priority = decoding_priority(st, j);
    }
    if (best[group] < 0 || priority < lowest[group]) best[group] = j, lowest[group] = priority;
  }
  return best[0] >= 0 ? best[0] : best[1] >= 0 ? best[1] : best[2];
}

// A conversion: no slot with a fitting buffer is low enough, so the cache
// gives buffers back, the lowest priorities first, until one of `need` bytes
// fits in its bytes and the memory pool admits it. Returns the free slot the
// new buffer (*buffer) is for, or -1. A read-ahead gives at most two buffers
// back and a demand four: two of the smallest make room for one of the
// largest, and the pool admits a buffer only while a second one would fit
// above the reserve (memory_admit). A pool that still refuses is short of
// memory, and giving more back would empty the cache into it; the demand
// then reuses a buffer that is large enough (victim_large_enough).
enum { CONVERT_OPTIONAL = 2, CONVERT_DEMAND = 4 };
static int convert(ExpertStore *st, uint64_t need, int optional, Buf *buffer) {
  for (int given = 0;; given++) {
    if (st->nfree && st->bytes + need <= (uint64_t)st->cap * st->SLOT_SIZE) {
      int j = st->free_list[st->nfree - 1];
      *buffer = pool_buffer(st, j, need, 0);
      if (buffer->obj) {
        st->conversions += given > 0;
        return j;
      }
    }
    if (given >= (optional ? CONVERT_OPTIONAL : CONVERT_DEMAND)) return -1;
    int i = victim_freeable(st);
    if (i < 0 || (optional && slot_reading(st, i))) return -1;
    discard(st, i, optional);
  }
}

// A slot for expert e of the layer, or -1 for an optional request when none
// is free, or TAKE_RESIDENT when, during the wait for a slot that passes
// hold, another pass made the expert resident.
static int take(ExpertStore *st, int layer, int e, int optional) {
  int i = -1;
  // What the expert takes, and the size of a new buffer for it: the expert's
  // when decoding. A prompt pass goes through every layer and needs each
  // one's experts only until the next, so its new buffers have the largest
  // size and serve all its layers; decoding gives them back for buffers of
  // its experts' sizes as it replaces their experts.
  uint64_t need = slot_need(st, layer), size = st->pass_layer >= 0 ? st->SLOT_SIZE : need;
  int reuse;
  Buf buffer;
  for (;;) {  // once; again after a demand that found no buffer at all gave a slot back
    i = -1, buffer = (Buf){0};
    reuse = (st->sized ? st->bytes + size > (uint64_t)st->cap * st->SLOT_SIZE : st->nslots >= st->cap) || !st->nfree;
    if (!reuse) {
      i = st->free_list[st->nfree - 1];
      buffer = pool_buffer(st, i, size, 0);
      if (!buffer.obj) reuse = 1;
    }
    if (reuse) {
      int converting = 0;
      i = st->sized ? victim_sized(st, need, &converting) : victim(st);
      if (i < 0) {
        if (optional) return -1;
        // The preferences leave only slots that another pass may read soon: a
        // demand may take any slot that nothing holds.
        i = slot_unheld(st);
        if (i < 0 && pass_current()) {
          SlotWait wait = {st, now() + 60};
          pass_wait(&(PassWait){slot_released, slot_released_block, &wait, PASS_WAIT_OTHER});
          if (WHERE(layer, e) >= 0) return TAKE_RESIDENT;
          i = slot_unheld(st);
        }
        if (i < 0) die("the expert cache (%d slots) cannot hold the experts of two layers", st->nslots);
        converting = st->sized && st->slots[i].b.size < need;
      }
      // Routing can demote a prefetch while its read is still running. Do not
      // stall the model for optional work or evict a higher-scoring resident.
      if (optional && slot_reading(st, i)) return -1;
      if (converting) {
        // The buffer cannot hold the expert, or wastes too much of the cache
        // on it: buffers go back to make room for one of the wanted size.
        i = convert(st, size, optional, &buffer);
        if (i >= 0) reuse = 0;
        else if (optional) return -1;
        else if ((i = victim_large_enough(st, need)) < 0) {
          // No buffer could be converted and none is large enough: the demand
          // allocates one as required memory, for which the pool makes room.
          // The pool may take other slots of this cache back for it, and they
          // join the free list: the slot the buffer is charged to leaves the
          // list meanwhile and returns to its top.
          if (st->nfree) {
            i = st->free_list[--st->nfree];
            buffer = pool_buffer_required(st, i, size);
            st->free_list[st->nfree++] = i;
          }
          if (!buffer.obj) {
            // The pool cannot make the room: passes hold what it would take
            // back. The lowest priority that nothing holds goes back, and the
            // demand starts over. It then finds room in the cache's own
            // bytes; or, with nothing left that is not held, it waits above
            // for another pass to release a slot, as with slots of one size.
            if ((i = victim_large_enough(st, 0)) >= 0) discard(st, i, 0);
            continue;
          }
          reuse = 0;
        }
      }
    }
    break;
  }
  if (reuse) {
    cancel(st, i);
    metrics_retire(st, i, optional);
    unlink_slot(st, i);
    if (st->slots[i].expert >= 0) place(st, st->slots[i].layer, st->slots[i].expert, -1);
    st->stats.evictions++;
    if (st->slots[i].mapped) {
      gpu_resident(&st->slots[i].b, 0);
      st->slots[i].mapped = st->slots[i].fresh = 0;
      st->nmapped--, st->mapped_bytes -= st->slots[i].b.size;
    }
  } else {
    i = st->free_list[--st->nfree];
    st->slots[i].b = buffer, st->slots[i].written = 0;
    // CPU reads fill the buffer. cache_get maps it when GPU work uses it;
    // a forecast that is never selected does not occupy GPU residency.
    st->slots[i].mapped = st->slots[i].fresh = st->slots[i].mapped_once = 0;
    st->nslots++, st->bytes += buffer.size;
  }
  if (st->cache_backbone) memory_touch(&st->slots[i].memory);
  st->slots[i].prefetched = 0;
  st->slots[i].layer = layer, st->slots[i].expert = e;
  place(st, layer, e, i);
  size_t key = (size_t)layer * st->experts + e;
  st->score[i] = layer < st->scored_layers ? expert_priority(st, key) : st->escore[key];
  push_front(st, i);
  return i;
}


// A forecast (urgent 0) goes to the background queue and counts as a
// prefetch in the metrics; a demand (urgent 1) is claimed at once.
static void start(ExpertStore *st, int i, int urgent) {
  Slot *s = &st->slots[i];
  int n = 0;
  uint64_t region = 0;
  for (int k = 0; k < st->spans[s->layer]; k++) {
    const FileSpan *p = &SPAN(s->layer, s->expert, k);
    if (!p->len) continue;  // an expert with fewer spans than the layer's most
    n += split(s->piece + n, p->fd, p->off, p->len, s->b.ptr + region, &s->at[k]);
    s->at[k] += region;
    region += REGION(p->len);
  }
  s->npiece = n;
  s->extent = st->extra;  // the transform's bytes at the slot's end
  for (int k = 0; k < n; k++) s->extent += s->piece[k].len;
  atomic_store_explicit(&s->read, 0, memory_order_relaxed);
  s->background_load = !urgent;
  atomic_store_explicit(&s->unfinished, n, memory_order_relaxed);
  metrics_start(st, i, urgent);
  s->loading = dispatch_group_create();
  // A stale queue entry may claim a newly reused slot as soon as taken is
  // reset. Publish all group entries and the admission before that reset.
  for (int k = 0; k < n; k++) dispatch_group_enter(s->loading);
  if (!urgent) atomic_fetch_add_explicit(&st->prefetching, 1, memory_order_relaxed);
  for (int k = 0; k < n; k++) atomic_store_explicit(&s->taken[k], 0, memory_order_release);
  if (urgent) {
    claim(st, i);
    return;
  }
  pthread_mutex_lock(&st->qlock);
  for (int k = 0; k < n; k++) {
    while (st->qtail - st->qhead == QUEUE) pthread_cond_wait(&st->qcond, &st->qlock);
    queue_push(st, (uint32_t)(i << 4 | k), (uint16_t)s->layer);
  }
  pthread_cond_broadcast(&st->qcond);
  pthread_mutex_unlock(&st->qlock);
}

// Makes a layer's selected experts resident: starts reading the missing
// ones (cache_request), then waits for them and for those a prefetch is
// still reading (cache_wait); cache_load does both.
// The draft has separate buffer pools because its MXFP4 slots may be larger
// than the backbone's GGUF slots. The caller must have finished prior GPU
// use of this stage's pool; draft_stage already synchronizes before loading.
static int draft_victim(ExpertStore *st, int layer, uint64_t stamp) {
  int first = st->max_slots + (layer - st->scored_layers) * st->draft_capacity, best = -1;
  if (layer < st->scored_layers || layer >= st->layers || first + st->draft_capacity > st->max_slots + st->npinned)
    die("draft cache layer %d has no pool", layer);
  int end = first + st->draft_capacity;
  for (int i = first; i < end; i++) {
    Slot *s = &st->slots[i];
    if ((s->expert >= 0 && s->used == stamp) || slot_reading(st, i)) continue;
    if (s->expert < 0) return i;
    if (best < 0 || st->draft_age[i - st->max_slots] < st->draft_age[best - st->max_slots]) best = i;
  }
  return best;
}

static int draft_request(ExpertStore *st, int layer, const int *experts, int n) {
  if (layer < st->scored_layers || layer >= st->layers || (layer - st->scored_layers + 1) * st->draft_capacity > st->npinned || n < 0)
    die("invalid bounded draft cache request for layer %d", layer);
  char wanted[st->stage_experts];
  memset(wanted, 0, sizeof wanted);
  int unique = 0, reading = 0;
  for (int k = 0; k < n; k++) {
    int e = experts[k];
    if (e < 0 || e >= st->stage_experts) die("draft expert %d is outside its stage", e);
    if (!wanted[e]) wanted[e] = 1, unique++;
  }
  if (unique > st->draft_capacity)
    die("draft stage needs %d experts together, but its cache holds %d", unique, st->draft_capacity);
  if (st->draft_clock == UINT64_MAX || (uint64_t)n > UINT64_MAX - st->draft_touch_clock)
    die("draft cache request counter exhausted");
  uint64_t stamp = ++st->draft_clock;
  for (int e = 0; e < st->stage_experts; e++)
    if (wanted[e] && WHERE(layer, e) >= 0) st->slots[WHERE(layer, e)].used = stamp;
  for (int k = 0; k < n; k++) {
    int e = experts[k], i = WHERE(layer, e);
    if (i >= 0) {
      if (slot_reading(st, i)) claim(st, i);
      st->stats.hits++;
    } else {
      i = draft_victim(st, layer, stamp);
      if (i < 0) die("draft stage %d has no reusable slot: its slots are needed or still being read", layer - st->scored_layers);
      finish(st, i);  // reap a completed handle before assigning a new expert
      Slot *s = &st->slots[i];
      if (s->expert >= 0) place(st, s->layer, s->expert, -1), st->stats.evictions++;
      if (!s->b.obj) {
        s->b = pool_buffer(st, i, st->pinned_slot_size, 1), s->written = 0;
        if (!s->b.obj) die("reserved draft expert memory is unavailable");
      }
      s->layer = layer, s->expert = e;
      s->prefetched = 0;
      place(st, layer, e, i);
      st->stats.misses++;
      start(st, i, 1);  // draft misses always use the demand-priority path
    }
    st->slots[i].used = stamp;
    if (st->cache_backbone) memory_touch(&st->slots[i].memory);
    st->draft_age[i - st->max_slots] = ++st->draft_touch_clock;
    st->score[i] = st->escore[layer * st->experts + e] += st->weight;
    st->dscore[layer * st->experts + e] += st->weight;  // drafts run only while decoding
    st->stats.decode_uses++;
    reading += slot_reading(st, i);
  }
  return reading;
}

// Slot i's expert e of the layer is in use this token: eviction keeps it
// longest, and its score grows.
static void mark_used(ExpertStore *st, int layer, int e, int i) {
  st->slots[i].prefetched = 0;
  st->slots[i].used = st->token;
  if (st->cache_backbone) memory_touch(&st->slots[i].memory);
  size_t key = (size_t)layer * st->experts + e;
  st->escore[key] += st->weight;
  st->score[i] = i < st->max_slots && layer < st->scored_layers ? expert_priority(st, key) : st->escore[key];
  if (st->pass_layer < 0) st->dscore[key] += st->weight, st->stats.decode_uses++;
}

int cache_request(ExpertStore *st, int layer, const int *experts, int n) {
  if (layer >= st->scored_layers && st->draft_capacity < st->stage_experts) return draft_request(st, layer, experts, n);
  int reading = 0;
  st->serving = layer;
  for (int k = 0; k < n; k++)  // the resident ones first, so that the others' slots are not theirs
    if (WHERE(layer, experts[k]) >= 0) st->slots[WHERE(layer, experts[k])].used = st->token;
  for (int k = 0; k < n; k++) {
    int e = experts[k], i = WHERE(layer, e), pending, fresh = 0;
    if (i < 0) {
      i = take(st, layer, e, 0);
      if (i == TAKE_RESIDENT) i = WHERE(layer, e);  // another pass read it while this one waited for a slot
      else fresh = 1;
    }
    if (!fresh) {
      if (!st->slots[i].pinned) unlink_slot(st, i), push_front(st, i);
      pending = slot_reading(st, i);
      // A forecast can outlive its predicted pass and lose its protection.
      // Any later demand must promote its queued pieces, protected or not.
      if (pending) claim(st, i);
      if (st->slots[i].prefetched) {
        st->stats.prefetch_hits++;
        if (pending) st->stats.late++;
      } else {
        st->stats.hits++;
      }
    } else {
      st->stats.misses++;
      start(st, i, 1);
      pending = slot_reading(st, i);
    }
    // A request outside an observed pass is still a use: keep it out of
    // first-use rewards and later unused-byte totals.
    if (st->pf_enabled && !st->pf_active && layer < st->scored_layers) {
      PrefetchLoad *m = &st->pf_loads[i];
      if (m->tracked && m->origin_prefetch && !m->used) {
        m->used = 1;
        st->pf_stats[m->phase].unobserved_first_uses++;
      }
    }
    mark_used(st, layer, e, i);
    reading += pending;
  }
  return reading;
}

void cache_use(ExpertStore *st, int layer, const int *experts, int n) {
  for (int k = 0; k < n; k++) {
    int e = experts[k], i = WHERE(layer, e);
    if (i < 0) continue;
    if (!st->slots[i].pinned) unlink_slot(st, i), push_front(st, i);
    if (st->slots[i].prefetched) st->stats.prefetch_hits++;
    else st->stats.hits++;
    mark_used(st, layer, e, i);
  }
}

int cache_slot(ExpertStore *st, int layer, int e) { return WHERE(layer, e) >= 0; }

int cache_hold(ExpertStore *st, int layer, const int *experts, int n) {
  if (layer < 0 || layer >= st->scored_layers || n < 1 || n > st->experts) die("invalid expert set to hold");
  unsigned char resident[n];
  // Protect resident entries before allocating missing ones. Jobs share
  // the reservation and pending fetch when they need the same expert.
  for (int k = 0; k < n; k++) {
    int e = experts[k];
    if (e < 0 || e >= st->experts) die("invalid runahead expert");
    int i = WHERE(layer, e);
    resident[k] = i >= 0;
    if (i >= 0) st->slots[i].holds++;
  }
  // Each missing one is held as soon as it has a slot: the slot of the next
  // one must not be taken from it.
  int pending = 0;
  for (int k = 0; k < n; k++) {
    pending += cache_request(st, layer, &experts[k], 1);
    if (!resident[k]) st->slots[WHERE(layer, experts[k])].holds++;
  }
  st->held_references += n;
  return pending;
}

void cache_hold_layer(ExpertStore *st, int layer) { st->layer_holds[layer]++; }

void cache_unhold_layer(ExpertStore *st, int layer) {
  if (!st->layer_holds[layer]) die("a layer's hold was released twice");
  st->layer_holds[layer]--;
}

void cache_unhold(ExpertStore *st, int layer, const int *experts, int n) {
  for (int k = 0; k < n; k++) {
    int i = WHERE(layer, experts[k]);
    if (i < 0 || !st->slots[i].holds) die("runahead expert lost its reservation");
    st->slots[i].holds--;
  }
  if (st->held_references < (unsigned)n) die("invalid runahead reference count");
  st->held_references -= n;
}


int cache_ready(ExpertStore *st, int layer, int e) {
  int i = WHERE(layer, e);
  return i >= 0 && !slot_reading(st, i);
}

// A layer found ready stays ready until one of its experts moves (place) or
// a slot is unmapped (resize): a slot is fresh or reading only after either.
int cache_layer_ready(ExpertStore *st, int layer, uint64_t *generation) {
  *generation = st->generation[layer];
  if (st->ready_generation[layer] == *generation && st->ready_unmaps[layer] == st->unmaps) return 1;
  for (int e = 0; e < st->experts; e++) {
    int i = WHERE(layer, e);
    if (i < 0 || st->slots[i].loading || !st->slots[i].mapped || st->slots[i].fresh) return 0;
  }
  st->ready_generation[layer] = *generation, st->ready_unmaps[layer] = st->unmaps;
  return 1;
}


int cache_where(ExpertStore *st, int layer, int e) {  // as cache_request will count it
  int i = WHERE(layer, e);
  return i < 0 ? 0 : !st->slots[i].prefetched ? 1 : slot_reading(st, i) ? 3 : 2;
}

typedef struct { ExpertStore *st; int layer; const int *experts; int n; } ReadsWait;

static int reads_ready(void *context) {
  ReadsWait *w = context;
  ExpertStore *st = w->st;
  for (int k = 0; k < w->n; k++)
    if (slot_reading(st, WHERE(w->layer, w->experts[k]))) return 0;
  return 1;
}

static void reads_block(void *context) {
  ReadsWait *w = context;
  ExpertStore *st = w->st;
  for (int k = 0; k < w->n; k++) {
    dispatch_group_t g = st->slots[WHERE(w->layer, w->experts[k])].loading;
    if (g) dispatch_group_wait(g, DISPATCH_TIME_FOREVER);
  }
}

void cache_wait(ExpertStore *st, int layer, const int *experts, int n) {
  double t = now();
  if (pass_current()) {
    ReadsWait wait = {st, layer, experts, n};
    pass_wait(&(PassWait){reads_ready, reads_block, &wait, PASS_WAIT_READ});
  }
  for (int k = 0; k < n; k++) finish(st, WHERE(layer, experts[k]));
  st->stats.read_s += now() - t;
}



void cache_load(ExpertStore *st, int layer, const int *experts, int n) {
  cache_request(st, layer, experts, n);
  cache_wait(st, layer, experts, n);
}

// Starts reading the predicted experts (in rank order)
// that are not resident, while fewer than `inflight` prefetches are reading.
void cache_prefetch(ExpertStore *st, int layer, const int *experts, int n, int inflight) {
  if (layer >= st->scored_layers && st->draft_capacity < st->stage_experts) return;  // bounded draft pools are demand-only
  // The limit is per pass: passes running together each read ahead as many.
  int others = __atomic_load_n(&gpu_other_passes, __ATOMIC_RELAXED);
  int limit = inflight * (1 + (others > 0 ? others : 0));
  for (int k = 0; k < n && st->prefetching < limit; k++) {
    int e = experts[k];
    if (WHERE(layer, e) >= 0) continue;
    int i = take(st, layer, e, 1);
    if (i < 0) return;
    st->slots[i].prefetched = 1;
    st->slots[i].issuer = pass_current();
    st->slots[i].used = st->token;
    if (st->cache_backbone) memory_touch(&st->slots[i].memory);
    start(st, i, 0);
    st->stats.prefetches++;
  }
}

void cache_prefetch_from(ExpertStore *st, int layer, const int *experts, int n, int inflight, int issuer) {
  int before = st->pf_issuer;
  if (issuer < 0 || issuer >= PF_ISSUERS) die("invalid prefetch issuer");
  st->pf_issuer = issuer;
  cache_prefetch(st, layer, experts, n, inflight);
  st->pf_issuer = before;
}


// The model thread cancels only pieces that no reader owns. Check and
// claim under the reader's lock so optional cancellation never waits for I/O.
static int discard_unstarted(ExpertStore *st, int layer, const int *keep, int n) {
  if (!atomic_load(&st->prefetching)) return 0;
  int retired[st->experts], nc = 0;
  // Only the model thread changes slot metadata. Readers can change taken,
  // so this scan is a filter; ownership is rechecked under qlock below.
  for (int e = 0; e < st->experts; e++) {
    int i = WHERE(layer, e);
    if (i < 0 || !st->slots[i].prefetched || !st->slots[i].loading || !st->slots[i].npiece) continue;
    if (st->slots[i].issuer != pass_current()) continue;  // another pass's prediction, for its own layer
    int k = 0;
    while (k < n && keep[k] != e) k++;
    if (k < n) continue;
    if (slot_held(st, &st->slots[i])) continue;
    int started = 0;
    for (int p = 0; p < st->slots[i].npiece; p++) started |= atomic_load(&st->slots[i].taken[p]);
    if (!started) retired[nc++] = i;
  }
  if (!nc) return 0;
  unsigned char discard[st->experts];
  memset(discard, 0, sizeof discard);
  int nr = 0;
  pthread_mutex_lock(&st->qlock);
  for (int k = 0; k < nc; k++) {
    int i = retired[k], started = 0;
    for (int p = 0; p < st->slots[i].npiece; p++) started |= atomic_load(&st->slots[i].taken[p]);
    if (started) continue;
    // Reader claims use qlock too. The model thread is the only demand
    // claimant, so every piece is now owned by this cancellation.
    for (int p = 0; p < st->slots[i].npiece; p++) atomic_store(&st->slots[i].taken[p], 1);
    discard[st->slots[i].expert] = 1, retired[nr++] = i;
  }
  if (nr) {
    uint32_t end = st->qtail, write = st->qhead;
    for (uint32_t read = st->qhead; read != end; read++) {
      uint32_t q = st->queue[read % QUEUE];
      uint16_t ql = st->qlayer[read % QUEUE];
      Slot *s = &st->slots[q >> 4];
      if (s->layer == layer && s->expert >= 0 && discard[s->expert]) continue;
      st->qlayer[write % QUEUE] = ql;
      st->queue[write++ % QUEUE] = q;
    }
    st->qtail = write;
    pthread_cond_broadcast(&st->qcond);
  }
  pthread_mutex_unlock(&st->qlock);
  // All old queue references are gone, and no reader owns a retired piece.
  // Complete the groups before making these buffers available for reuse.
  for (int k = 0; k < nr; k++) {
    int i = retired[k], expert = st->slots[i].expert;
    for (int p = 0; p < st->slots[i].npiece; p++) piece_finished(st, &st->slots[i], p, 0);
    finish(st, i);
    metrics_retire(st, i, 0);
    place(st, layer, expert, -1);
    st->slots[i].expert = -1, st->slots[i].prefetched = 0;
    st->slots[i].used = st->token - 1; // an empty slot is not part of the selected union
    st->score[i] = -INFINITY;
    unlink_slot(st, i), push_back(st, i);
    st->stats.cancelled++;
  }
  return nr;
}

// After a decoded layer's routing: its read-ahead experts that it did not
// select may be evicted again, by the score earlier selections gave them.
void cache_demote(ExpertStore *st, int layer) {
  for (int e = 0; e < st->experts; e++) {
    int i = WHERE(layer, e);
    if (i >= 0 && st->slots[i].issuer == pass_current()) st->slots[i].prefetched = 0;
  }
}

// Cancels the read-aheads for `layer` that no reader has started, unless
// their expert is among keep[0..n): a newer prediction for the layer rules
// them out. Their slots hold no expert (expert -1) and are taken first.
void cache_refine(ExpertStore *st, int layer, const int *keep, int n) {
  discard_unstarted(st, layer, keep, n);
}

// Resizes the cache. It gives slots back, the least recently used of the
// layers below `below` first, on memory pressure or when the available
// memory falls below the reserve: a sixteenth of the cache (a quarter at
// critical pressure) under pressure or swapping, and otherwise the slots
// that restore the reserve. Actual swap-outs also shrink a cache above its
// minimum. A global swap-out cannot reclaim an empty cache.
// Compression in RAM has no separate limit. Growth uses a quarter of the
// headroom after 32 calm calls; for five minutes after swap-driven
// reclamation it uses only free pages.
static void resize(ExpertStore *st, int below) {
  vm_statistics64_data_t s = vm();
  uint64_t free = (uint64_t)s.free_count * vm_page_size, avail = avail_of(s), slot = st->SLOT_SIZE, margin = 4 * (uint64_t)GiB;
  if (!st->init) st->swapouts = s.swapouts, st->init = 1;
  int64_t room = mem_gpu_room();
  uint64_t swap_delta = s.swapouts > st->swapouts ? s.swapouts - st->swapouts : 0;
  // What the cache holds and maps, in the unit of its limits (slots, or the
  // largest slot's bytes when the layers differ), and the most they may be.
  int used = used_units(st), mapped = mapped_units(st), ceiling = cap_ceiling(st);
  int level = mem_pressure(), swapped = swap_delta && used > st->minimum_slots;
  int low = level >= 2 || swapped || avail < reserve;
  if (swapped) st->bad = now();
  st->swapouts = s.swapouts;
  int old_cap = st->cap, old_mcap = st->mcap;
  uint64_t headroom = now() - st->bad > 300 ? avail : free;
  if (low && (level >= 4 || swapped || now() - gave > 1)) {
    uint64_t to = avail;
    int64_t give = level >= 4 ? used / 4 : level >= 2 || swapped ? used / 16 + 1 : 0;
    if (to < reserve && (int64_t)((reserve - to) / slot) + 1 > give) give = (reserve - to) / slot + 1;
    st->cap = used - (int)give > st->minimum_slots ? used - (int)give : st->minimum_slots;
    st->calm = 0, gave = now();
  } else if (!low && ++st->calm >= 32 && st->cap < ceiling && headroom > reserve + margin) {
    st->cap += (int)((headroom - reserve - margin) / slot / 4) + 1;
    if (st->cap > ceiling) st->cap = ceiling;
    st->calm = 0;
  }
  if (room < 0 && now() - st->unmapped > 1) {  // past the GPU's limit: map fewer, exactly the excess fewer
    st->mcap = mapped - (int)(-room / (int64_t)slot) - 1 > st->minimum_slots ? mapped - (int)(-room / (int64_t)slot) - 1 : st->minimum_slots;
    st->calm_map = 0, st->unmapped = now();
  } else if (room >= 0 && ++st->calm_map >= 32 && st->mcap < ceiling && room > MAP_SLACK + (int64_t)GiB) {
    st->mcap += (int)((room - MAP_SLACK - (int64_t)GiB) / (int64_t)slot / 4) + 1;
    if (st->mcap > ceiling) st->mcap = ceiling;
    st->calm_map = 0;
  }
  for (int i = st->tail; over_cap(st) && i >= 0;) {
    int prev = st->slots[i].prev;
    if (st->slots[i].layer < below && !slot_held(st, &st->slots[i])) release(st, i), st->stats.given_back++;
    i = prev;
  }
  while (over_mcap(st)) {  // the mapped slots with the lowest scores that nothing of this pass still needs
    int best = -1;
    for (int j = 0; j < st->max_slots; j++)
      if (st->slots[j].b.obj && st->slots[j].mapped && st->slots[j].layer < below && !slot_held(st, &st->slots[j]) &&
          !st->slots[j].loading && !st->slots[j].prefetched && (best < 0 || st->score[j] < st->score[best]))
        best = j;
    if (best < 0) break;
    gpu_resident(&st->slots[best].b, 0);
    st->slots[best].mapped = 0, st->slots[best].fresh = 0;
    st->nmapped--, st->mapped_bytes -= st->slots[best].b.size, st->unmaps++;
  }
  gpu_residency_commit();
  static int trace = -1;
  if (trace < 0) trace = getenv("MYDS4_CACHE_TRACE") != NULL;
  if (trace && (st->cap != old_cap || st->mcap != old_mcap))
    fprintf(stderr, "cache resize: limit %d -> %d, allocated %d, map limit %d -> %d, mapped %d, "
                    "maximum %d, pressure %d, swap-outs +%llu, available %.2f GiB, free %.2f GiB, "
                    "slots not yet written %.2f GiB, GPU room %.2f GiB, growth uses %s\n",
            old_cap, st->cap, st->nslots, old_mcap, st->mcap, st->nmapped, st->max_slots, level, (unsigned long long)swap_delta,
            avail / GiB, free / GiB, cache_unwritten() / GiB, room / GiB,
            now() - st->bad > 300 ? "available memory" : "free pages");
}

// Between tokens (pass < 0) no GPU work of this pass references a slot, and
// within a prompt pass at layer `pass` it may still use the slots of the
// layer before, so only those of earlier layers are released; held slots
// stay, for the work of any pass that holds them.
void cache_adapt_pass(ExpertStore *st, int pass) {
  if (pass < 0) {
    advance_score_clock(st);
  }
  if (st->cache_backbone && pass < 0 && now() - st->last_pool_growth >= 1.0 &&
      memory_room(st->cache_backbone->account) < 4 * st->SLOT_SIZE) {
    MemorySlot growth;
    memory_slot_init(&growth, st->cache_backbone->account, NULL, "cache growth", NULL, NULL);
    MemorySlot *wanted[] = {&growth};
    uint64_t bytes = 4 * st->SLOT_SIZE;
    memory_admit(wanted, &bytes, 1, MEMORY_CACHE);
    memory_slot_remove(&growth);
    st->last_pool_growth = now();
  }
  resize(st, pass < 0 ? st->layers : pass - 1);
  if (pass < 0 && st->any_fresh) {  // resize committed the residency set
    for (int i = st->head; i >= 0; i = st->slots[i].next) st->slots[i].fresh = 0;
    st->any_fresh = 0;
  }
}

void cache_adapt(ExpertStore *st) { cache_adapt_pass(st, -1); }

// While the server idles no GPU work references a slot: the cache resizes
// as between tokens, so it gives slots back when other programs need the
// memory and leaves them as they are otherwise.
void cache_idle(ExpertStore *st) { resize(st, st->layers); }

// With slots of the layers' own sizes, the limits are bytes: the counts
// below say how many slots fit in them, the allocated ones and as many of
// the largest size as the remaining bytes hold; a slot's bytes are the
// allocated slots' average.
static int units_as_slots(const ExpertStore *st, int units) {
  uint64_t limit = (uint64_t)units * st->SLOT_SIZE;
  return st->sized ? st->nslots + (int)(limit > st->bytes ? (limit - st->bytes) / st->SLOT_SIZE : 0) : units;
}
uint64_t cache_bytes(ExpertStore *st) { return (st->sized ? st->bytes : (uint64_t)st->nslots * st->SLOT_SIZE) + st->pinned_bytes; }
int cache_slots(ExpertStore *st) { return st->nslots; }
int cache_mapped(ExpertStore *st) { return st->nmapped; }
int cache_cap(ExpertStore *st) { return units_as_slots(st, st->cap); }
int cache_max(ExpertStore *st) { return units_as_slots(st, cap_ceiling(st)); }
uint64_t cache_slot_bytes(ExpertStore *st) { return st->sized && st->nslots ? st->bytes / st->nslots : st->SLOT_SIZE; }
uint64_t cache_conversions(ExpertStore *st) { return st->conversions; }

static void metric_array(FILE *out, const char *const *names, const uint64_t *values, int n) {
  fputc('{', out);
  for (int i = 0; i < n; i++) fprintf(out, "%s\"%s\":%llu", i ? "," : "", names[i], (unsigned long long)values[i]);
  fputc('}', out);
}

void cache_metrics_report(ExpertStore *st, FILE *out) {
  if (!st->pf_enabled) return;
  const char *const phase[] = {"prefill", "decode"};
  const char *const state[] = {"existing_ready", "fresh_timely_prefetch", "pending_prefetch", "pending_demand", "absent"};
  const char *const issuer[] = {"pc", "h1", "h2", "prefill_bulk", "other"};
  uint64_t unused_live[PF_PHASES] = {0}, unread_live[PF_PHASES] = {0};
  for (int l = 0; l < st->scored_layers; l++)
    for (int e = 0; e < st->experts; e++) {
      int i = WHERE(l, e);
      if (i < 0) continue;
      PrefetchLoad *m = &st->pf_loads[i];
      if (!m->tracked || !m->origin_prefetch) continue;
      uint64_t done = atomic_load_explicit(&m->completed, memory_order_relaxed);
      if (!m->used) unused_live[m->phase] += done;
      unread_live[m->phase] += m->requested > done ? m->requested - done : 0;
    }
  for (int p = 0; p < PF_PHASES; p++) {
    PrefetchStats *s = &st->pf_stats[p];
    fprintf(out, "{\"prefetch_metrics\":{\"version\":1,\"phase\":\"%s\",\"passes\":%llu,\"rows\":%llu,", phase[p],
            (unsigned long long)s->passes, (unsigned long long)s->rows);
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
uint64_t cache_uncached_bytes(ExpertStore *st, int *experts) {
  uint64_t b = 0;
  int n = 0;
  for (int l = 0; l < st->scored_layers; l++)
    for (int e = 0; e < st->experts; e++)
      if (WHERE(l, e) < 0) {
        n++;
        for (int k = 0; k < st->spans[l]; k++) b += SPAN(l, e, k).len;
      }
  *experts = n;
  return b;
}
