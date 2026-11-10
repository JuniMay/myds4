// The routed-expert store (src/engine/io.c): experts read from the SSD into
// slots in memory, read ahead on predictions, evicted by score, within the
// memory other programs leave. The model gives its layout: its layers, their
// experts and each expert's bytes in the files.
#pragma once
#include <stdio.h>
#include "memory.h"
#include "myds4.h"

// One model's routed experts in memory (cache_create); every call takes it.
typedef struct ExpertStore ExpertStore;

// An expert's bytes in the files, one span per part: a slot holds the parts
// one after the other.
typedef struct { int fd; uint64_t off, len; } FileSpan;

enum { STORE_EXPERTS_MAX = 512, STORE_PARTS = 3 };

// The layout, given before cache_init. Layers [0, scored_layers) share one
// pool of slots, evicted by score; each later layer (a stage) has a pool of
// its own for its stage_experts experts, preloaded or bounded
// (cache_draft_limit).
typedef struct {
  int layers;            // every layer the store may hold
  int scored_layers;     // the shared pool's layers
  int experts;           // experts per layer, at most STORE_EXPERTS_MAX
  int chosen;            // experts a row chooses in a scored layer
  int stage_experts;     // experts per stage layer, at most experts
  int stage_min;         // a bounded stage pool's fewest slots
  int parts;             // spans per expert at most, at most STORE_PARTS
  const FileSpan *span;  // [layers][experts][parts]
  const int *spans;      // [layers]: spans per expert of the layer, 0: none yet
} ExpertLayout;
ExpertStore *cache_create(const ExpertLayout *layout);
CacheStats *cache_counters(ExpertStore *st);  // the store's statistics

// layers: the layers in use, the scored ones or every layer with the stages.
// Before cache_init: 0 or 128 preloads every draft expert; 15..127 reserves
// that many MXFP4 slots per stage and reads its native selections on demand.
void cache_draft_limit(ExpertStore *st, int slots);
int cache_draft_capacity(ExpertStore *st);  // per stage; 128 is the preloaded default
enum { CACHE_HALF_LIFE_DEFAULT = 50, CACHE_HALF_LIFE_MAX = 1000000 };
// Before cache_init: 0 selects the default half-life. Aging advances once per
// decode/verify pass and after prefill, not once per emitted token. Read-cost
// weighting changes backbone decode eviction priority only; raw history stays.
void cache_retention(ExpertStore *st, int half_life, int read_cost);
// Before cache_init: the memory in GiB that the cache leaves available to
// other programs, 0 for the default. The cache starts within available memory
// less the reserve, gives slots back when the available memory falls below
// it and grows only while 4 GiB more than it is available.
void cache_memory(ExpertStore *st, MemorySlot *backbone);
void cache_active(ExpertStore *st, int active);
void cache_working_rows(ExpertStore *st, int rows);
uint64_t cache_required(ExpertStore *st, int layers);
void cache_init(ExpertStore *st, uint64_t budget, int layers);
// Bounded draft pools are demand-only. Finish prior GPU use of a stage's
// buffers before cache_load/cache_request may replace their experts.
// The model thread owns cache mutations and slot lifetimes. Read-only
// queries (cache_slot, cache_ready, cache_where) may run in synchronous
// helpers while the model thread waits. Slots cannot change until they
// return. cache_get runs on the model thread and maps the expert for GPU
// use; at[k] is span k's offset, and *fresh requires gpu_use until the
// residency set is next committed.
const Buf *cache_get(ExpertStore *st, int layer, int expert, uint32_t at[STORE_PARTS], int *fresh);
void cache_load(ExpertStore *st, int layer, const int *experts, int n);  // finishes their reads into memory
// Starts reading the experts into memory. Returns how many had pending reads
// when each was checked; prefetch lateness uses that same completion check.
int cache_request(ExpertStore *st, int layer, const int *experts, int n);
void cache_wait(ExpertStore *st, int layer, const int *experts, int n);     // waits for their reads
// Read the epoch before scanning pending requests, then wait only if no
// piece completed since that scan. A completion for any layer wakes it.
uint64_t cache_read_epoch(ExpertStore *st);
void cache_wait_progress(ExpertStore *st, uint64_t epoch);
int cache_slot(ExpertStore *st, int layer, int expert);   // it has a slot: resident, or being read
int cache_hold(ExpertStore *st, int layer, const int *experts, int n);
void cache_unhold(ExpertStore *st, int layer, const int *experts, int n);
int cache_ready(ExpertStore *st, int layer, int expert);  // resident and read
int cache_where(ExpertStore *st, int layer, int expert);  // 0 only on the SSD, 1 in memory, 2 read ahead, 3 read ahead and still being read
// Starts reading the n predicted experts that are not resident, in order,
// while fewer than `inflight` prefetches are reading.
void cache_prefetch(ExpertStore *st, int layer, const int *experts, int n, int inflight);
// Optional MYDS4_PREFETCH_METRICS=1 observation. Pass IDs identify forward
// batches, not emitted tokens. Snapshot once before substitution, then observe
// the exact and served row unions from that same snapshot on the model thread.
// Only served routes consume first-use attribution. These hooks never change
// routing or cache admission.
enum { PF_PREFILL, PF_DECODE, PF_PHASES };
enum { PF_PC, PF_H1, PF_H2, PF_BULK, PF_OTHER, PF_ISSUERS };
enum { PF_EXISTING_READY, PF_FRESH_READY, PF_PENDING_PREFETCH, PF_PENDING_DEMAND, PF_ABSENT, PF_STATES };
typedef struct {
  int active, layer, phase;
  uint64_t pass;
  uint8_t state[STORE_EXPERTS_MAX], first[STORE_EXPERTS_MAX], ghost[STORE_EXPERTS_MAX];
  uint64_t completed[STORE_EXPERTS_MAX];
} CacheObservation;
void cache_metrics_begin(ExpertStore *st, int phase, int rows);
void cache_metrics_end(ExpertStore *st);
void cache_observe_snapshot(ExpertStore *st, int layer, CacheObservation *snapshot);
void cache_observe_routes(ExpertStore *st, const CacheObservation *snapshot, const int *ids, int rows, int stride, int served);
void cache_prefetch_from(ExpertStore *st, int layer, const int *experts, int n, int inflight, int issuer);
void cache_metrics_report(ExpertStore *st, FILE *out);
// Cancels the read-aheads for the layer that no reader has started and that keep[0..n) does not name.
void cache_refine(ExpertStore *st, int layer, const int *keep, int n);
void cache_demote(ExpertStore *st, int layer);  // after its routing: the layer's unselected read-aheads are evicted first
void cache_pass(ExpertStore *st, int layer);  // a prompt pass is at this layer (-1: none): keep later layers' experts
void cache_adapt(ExpertStore *st);             // between tokens
void cache_adapt_pass(ExpertStore *st, int layer);   // within a prompt pass at this layer
void cache_idle(ExpertStore *st);              // while no request runs: resizes as between tokens
void cache_drain(void);             // complete queued and active expert reads
uint64_t cache_bytes(ExpertStore *st);
int cache_slots(ExpertStore *st);
int cache_mapped(ExpertStore *st);     // slots mapped for the GPU
int cache_cap(ExpertStore *st);        // slots the cache may hold now
int cache_max(ExpertStore *st);        // and at most
uint64_t cache_slot_bytes(ExpertStore *st);
uint64_t cache_uncached_bytes(ExpertStore *st, int *experts);  // the backbone's experts without a slot: bytes and count
