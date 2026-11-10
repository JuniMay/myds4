#include "memory.h"
#include "myds4.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

static MemoryAccount *accounts;
static MemorySlot *entries;
static MemoryUsage usage;
static int pool_initialized;
static void (*collect_released)(void);
static _Thread_local MemorySlot *current;

// The available memory decides with the limit (memory_follow). The kernel's
// counters show what this program uses (allocated model memory and KV) only
// as its pages are written, and released memory a little late. So a reading
// is adjusted by this program's own changes since it was taken, charges not
// yet allocated count as used, the charged bytes not written at the reading
// (memory_unwritten) count as used, and no reading is taken within 1.5 s
// after a release unless the last one is 5 s old. An addition of 1 GiB or
// more (a model load) reads the counters; others use a reading up to a
// second old.
static uint64_t (*read_available)(void), (*read_unwritten)(void), reserve;
// The last reading, what this program used then, and how much of that was
// not written yet.
static uint64_t read_bytes, read_used, read_pending;
static double read_at = -1e9, released_at = -1e9;

static uint64_t in_use(void) { return usage.allocated + usage.kv; }
#define GIB_F 1073741824.0
// The log names take-backs of this much or more (a model's load, a long
// prompt's workspace); smaller ones happen in passes and only count
// (usage.evictions).
#define LOGGED_TAKE_BACK (1ull << 30)

static uint64_t available(uint64_t adding) {
  double t = now();
  if ((t - released_at >= 1.5 || t - read_at >= 5) && (adding >= 1ull << 30 || t - read_at >= 1)) {
    read_bytes = read_available(), read_used = in_use(), read_at = t;
    read_pending = read_unwritten ? read_unwritten() : 0;
  }
  uint64_t used = in_use(), promised = usage.committed > usage.allocated ? usage.committed - usage.allocated : 0;
  uint64_t a = used >= read_used ? (read_bytes > used - read_used ? read_bytes - (used - read_used) : 0)
                                 : read_bytes + (read_used - used);
  a = a > read_pending ? a - read_pending : 0;
  return a > promised ? a - promised : 0;
}

// How far adding `bytes` would take the available memory below the reserve.
static uint64_t shortage(uint64_t bytes) {
  if (!read_available || !bytes) return 0;
  uint64_t a = available(bytes), room = a > reserve ? a - reserve : 0;
  return bytes > room ? bytes - room : 0;
}

static uint64_t charged(const MemoryAccount *account) {
  return account->bytes > account->minimum ? account->bytes : account->minimum;
}

static void change(MemorySlot *slot, uint64_t bytes) {
  uint64_t before = charged(slot->account);
  slot->account->bytes -= slot->bytes;
  slot->account->bytes += bytes;
  slot->bytes = bytes;
  usage.committed = usage.committed - before + charged(slot->account);
  if (usage.committed > usage.peak) usage.peak = usage.committed;
}

void memory_init(uint64_t limit) {
  usage = (MemoryUsage){.limit = limit ? limit : UINT64_MAX};
  pool_initialized = 1;
}

void memory_set_limit(uint64_t bytes) { usage.limit = bytes; pool_initialized = 1; }

void memory_follow(uint64_t (*reader)(void), uint64_t reserve_bytes) {
  read_available = reader, reserve = reserve_bytes, read_at = -1e9;
}

void memory_unwritten(uint64_t (*source)(void)) { read_unwritten = source; }

uint64_t memory_current_limit(void) {
  if (!read_available) return usage.limit;
  uint64_t a = available(0), room = a > reserve ? a - reserve : 0;
  uint64_t limit = UINT64_MAX - usage.committed < room ? UINT64_MAX : usage.committed + room;
  return limit < usage.limit ? limit : usage.limit;
}

void memory_account_init(MemoryAccount *account, const char *name) {
  if (!pool_initialized) memory_init(0);
  *account = (MemoryAccount){.next = accounts, .name = name};
  accounts = account;
}

void memory_slot_init(MemorySlot *slot, MemoryAccount *account, MemorySlot *dependency,
                      const char *name, void (*evict)(void *), void *context) {
  *slot = (MemorySlot){.next = entries, .account = account, .dependency = dependency,
    .name = name, .evict = evict, .context = context, .touched = now()};
  atomic_init(&slot->ready, 0);
  if (dependency) {
    slot->sibling = dependency->children;
    dependency->children = slot;
  }
  entries = slot;
}

void memory_slot_remove(MemorySlot *slot) {
  change(slot, 0);
  MemorySlot **at = &entries;
  while (*at && *at != slot) at = &(*at)->next;
  if (*at) *at = slot->next;
  if (slot->dependency) {
    at = &slot->dependency->children;
    while (*at && *at != slot) at = &(*at)->sibling;
    if (*at) *at = slot->sibling;
  }
}

void memory_touch(MemorySlot *slot) { slot->touched = now(); }
void memory_pin(MemorySlot *slot, int pinned) { slot->pins = !!pinned; if (pinned) memory_touch(slot); }
int memory_ready(const MemorySlot *slot) { return atomic_load_explicit(&slot->ready, memory_order_acquire); }
void memory_publish(MemorySlot *slot, int ready) { atomic_store_explicit(&slot->ready, !!ready, memory_order_release); }

static int requested(MemorySlot *slot, MemorySlot *const *wanted, size_t n) {
  for (size_t i = 0; i < n; i++) if (wanted[i] == slot) return 1;
  return 0;
}

static int evictable(MemorySlot *slot, MemorySlot *const *wanted, size_t n, MemoryAdmission mode) {
  if (!slot->bytes || !slot->evict || slot->pins || slot->account->admitting || requested(slot, wanted, n) ||
      (slot->can_evict && !slot->can_evict(slot->context))) return 0;
  if (mode == MEMORY_SPECULATIVE) return 0;
  if (mode == MEMORY_WORKSPACE && !slot->dependency) return 0;
  if (mode == MEMORY_CACHE) {
    if (slot->account->minimum) return 0;
    for (size_t i = 0; i < n; i++) if (wanted[i]->account == slot->account) return 0;
  }
  // Optional cache growth does not displace a recently used weight group.
  if (mode == MEMORY_CACHE && !slot->dependency && now() - slot->touched < 30) return 0;
  for (MemorySlot *child = slot->children; child; child = child->sibling)
    if (child->bytes && !evictable(child, wanted, n, mode)) return 0;
  return 1;
}

static int has_children(MemorySlot *slot) {
  for (MemorySlot *child = slot->children; child; child = child->sibling)
    if (child->bytes) return 1;
  return 0;
}

static uint64_t proposed_bytes(const MemoryAccount *account, MemorySlot *const *wanted, const uint64_t *sizes, size_t n) {
  uint64_t proposed = account->bytes;
  for (size_t i = 0; i < n; i++) if (wanted[i]->account == account) {
    uint64_t bytes = sizes[i] > wanted[i]->payload ? sizes[i] : wanted[i]->payload;
    if (bytes > wanted[i]->bytes && UINT64_MAX - proposed < bytes - wanted[i]->bytes) return UINT64_MAX;
    proposed = proposed - wanted[i]->bytes + bytes;
  }
  return proposed;
}

static uint64_t growth(MemorySlot *const *wanted, const uint64_t *sizes, size_t n) {
  uint64_t extra = 0;
  for (MemoryAccount *a = accounts; a; a = a->next) {
    uint64_t proposed = proposed_bytes(a, wanted, sizes, n);
    uint64_t after = proposed > a->minimum ? proposed : a->minimum;
    if (after > charged(a)) {
      if (UINT64_MAX - extra < after - charged(a)) return UINT64_MAX;
      extra += after - charged(a);
    }
  }
  return extra;
}

static uint64_t reclaim_room(const MemoryAccount *account, MemorySlot *const *wanted, const uint64_t *sizes, size_t n) {
  uint64_t bytes = proposed_bytes(account, wanted, sizes, n);
  if (bytes < account->bytes) bytes = account->bytes;
  return bytes > account->minimum ? bytes - account->minimum : 0;
}

static uint64_t reclaimable(MemorySlot *const *wanted, const uint64_t *sizes, size_t n, MemoryAdmission mode) {
  uint64_t total = 0;
  for (MemoryAccount *a = accounts; a; a = a->next) {
    uint64_t possible = 0;
    for (MemorySlot *s = entries; s; s = s->next)
      if (s->account == a && evictable(s, wanted, n, mode)) possible += s->bytes;
    uint64_t room = reclaim_room(a, wanted, sizes, n);
    total += possible < room ? possible : room;
  }
  return total;
}

static int fits(uint64_t extra) { return usage.committed <= usage.limit && extra <= usage.limit - usage.committed; }

static MemorySlot *victim(MemorySlot *const *wanted, const uint64_t *sizes, size_t n,
                          MemoryAdmission mode, uint64_t need) {
  MemorySlot *best = NULL;
  double rank = INFINITY;
  for (MemorySlot *s = entries; s; s = s->next) {
    if (!evictable(s, wanted, n, mode) || has_children(s)) continue;
    uint64_t useful = reclaim_room(s->account, wanted, sizes, n);
    if (!useful) continue;
    if (useful > s->bytes) useful = s->bytes;
    if (useful > need) useful = need;
    double age = fmax(0, now() - s->touched), score = exp(-age / 30) * (double)s->bytes / useful;
    if (!best || score < rank) best = s, rank = score;
  }
  return best;
}

// Returns the charge it took back.
static uint64_t release_entry(MemorySlot *slot) {
  uint64_t before = usage.committed;
  slot->evict(slot->context);
  if (slot->payload) die("memory eviction retained its payload");
  change(slot, 0);
  memory_publish(slot, 0);
  usage.evictions++;
  released_at = now();
  return before - usage.committed;
}

uint64_t memory_reclaim(uint64_t bytes) {
  uint64_t before = usage.allocated;
  while (before - usage.allocated < bytes) {
    MemorySlot *slot = victim(NULL, NULL, 0, MEMORY_REQUIRED, bytes - (before - usage.allocated));
    if (!slot) break;
    release_entry(slot);
  }
  uint64_t freed = before - usage.allocated;
  if (freed && collect_released) collect_released();
  return freed;
}

uint64_t memory_unload(MemoryAccount *account) {
  uint64_t before = usage.allocated;
  // Each pass releases the account's idle leaves; an entry whose dependents
  // went in one pass is a leaf in the next.
  for (int again = 1; again;) {
    again = 0;
    for (MemorySlot *s = entries; s; s = s->next)
      if (s->account == account && !has_children(s) && evictable(s, NULL, 0, MEMORY_REQUIRED)) {
        release_entry(s);
        again = 1;
      }
  }
  uint64_t freed = before - usage.allocated;
  if (freed && collect_released) collect_released();
  return freed;
}

// What an addition needs back past the limit (UINT64_MAX: more than can come back).
static uint64_t over_limit(uint64_t extra) {
  if (fits(extra)) return 0;
  return extra == UINT64_MAX || usage.committed > usage.limit ? UINT64_MAX : extra - (usage.limit - usage.committed);
}

int memory_admit(MemorySlot *const *wanted, const uint64_t *sizes, size_t n, MemoryAdmission mode) {
  int evicted = 0;
  uint64_t extra = growth(wanted, sizes, n);
  // What the addition needs past the reserve comes back from idle model
  // memory in the same order as what it needs past the limit. Speculative
  // growth (the expert cache's) stops one more of itself short of the
  // reserve: a pass's small buffers then fit without taking an expert back,
  // which the next miss would read again.
  uint64_t probe = mode == MEMORY_SPECULATIVE && extra <= UINT64_MAX / 2 ? 2 * extra : extra;
  uint64_t short_by = extra == UINT64_MAX ? 0 : shortage(probe), over = over_limit(extra);
  uint64_t need = over > short_by ? over : short_by, freed = 0;
  if (mode == MEMORY_SPECULATIVE && need) { usage.refusals++; return 0; }
  if (need && (need == UINT64_MAX || need > reclaimable(wanted, sizes, n, mode))) {
    if (short_by)
      say("memory: %s: %.2f GiB to add, %.2f GiB short of the %.1f GiB reserve, too little idle: refused",
          wanted[0]->name, extra / GIB_F, short_by / GIB_F, reserve / GIB_F);
    usage.refusals++;
    return 0;
  }
  int taken = 0;
  while ((over = over_limit(extra)) || freed < short_by) {
    uint64_t left = freed < short_by ? short_by - freed : 0, want = over > left ? over : left;
    MemorySlot *best = victim(wanted, sizes, n, mode, want);
    if (!best) die("memory reservation changed during eviction");
    freed += release_entry(best);
    evicted = 1, taken++;
    extra = growth(wanted, sizes, n);
  }
  if (short_by && freed >= LOGGED_TAKE_BACK)
    say("memory: %s: %.2f GiB to add, %.2f GiB short of the %.1f GiB reserve (reading %.1f s old): "
        "took back %.2f GiB from %d idle entries", wanted[0]->name, extra / GIB_F, short_by / GIB_F,
        reserve / GIB_F, now() - read_at, freed / GIB_F, taken);
  for (size_t i = 0; i < n; i++) change(wanted[i], sizes[i] > wanted[i]->payload ? sizes[i] : wanted[i]->payload);
  if (evicted && collect_released) collect_released();
  return 1;
}

int memory_minimum_mode(MemoryAccount *account, uint64_t bytes, MemoryAdmission mode) {
  uint64_t old = account->minimum, before = charged(account);
  account->minimum = bytes;
  uint64_t after = charged(account);
  account->minimum = old;
  if (after > before) {
    MemorySlot promise;
    memory_slot_init(&promise, account, NULL, "admission", NULL, NULL);
    uint64_t extra = after - account->bytes;
    MemorySlot *wanted[] = {&promise};
    account->admitting = 1;
    int admitted = memory_admit(wanted, &extra, 1, mode);
    account->admitting = 0;
    if (!admitted) { memory_slot_remove(&promise); return 0; }
    memory_slot_remove(&promise);
  }
  before = charged(account);
  account->minimum = bytes;
  usage.committed = usage.committed - before + charged(account);
  if (usage.committed > usage.peak) usage.peak = usage.committed;
  return 1;
}

int memory_minimum(MemoryAccount *account, uint64_t bytes) { return memory_minimum_mode(account, bytes, MEMORY_REQUIRED); }

int memory_claim(MemorySlot *slot, uint64_t bytes, MemoryAdmission mode) {
  if (UINT64_MAX - slot->payload < bytes) return 0;
  uint64_t wanted_size = slot->payload + bytes;
  MemorySlot *wanted[] = {slot};
  if (wanted_size > slot->bytes && !memory_admit(wanted, &wanted_size, 1, mode)) return 0;
  slot->payload += bytes;
  usage.allocated += bytes;
  return 1;
}

void memory_release(MemorySlot *slot, uint64_t bytes) {
  slot->payload -= bytes;
  usage.allocated -= bytes;
  if (slot->automatic) memory_trim(slot);
}
void memory_trim(MemorySlot *slot) { change(slot, slot->payload); }
void memory_kv_change(uint64_t bytes, int allocated) {
  if (!allocated) {
    usage.kv -= bytes;
    return;
  }
  uint64_t short_by = shortage(bytes), freed = 0;
  int released = 0;
  while (freed < short_by) {
    MemorySlot *slot = victim(NULL, NULL, 0, MEMORY_REQUIRED, short_by - freed);
    if (!slot) break;  // nothing idle is left: the once-a-second check frees idle KV
    freed += release_entry(slot);
    released++;
  }
  if (released && collect_released) collect_released();
  if (short_by && freed >= LOGGED_TAKE_BACK)
    say("memory: KV grows %.2f GiB, %.2f GiB short of the %.1f GiB reserve: took back %.2f GiB from %d idle entries",
        bytes / GIB_F, short_by / GIB_F, reserve / GIB_F, freed / GIB_F, released);
  usage.kv += bytes;
}
MemoryUsage memory_usage(void) { return usage; }
void memory_collect_with(void (*collect)(void)) { collect_released = collect; }
uint64_t memory_room(const MemoryAccount *account) {
  uint64_t free = usage.limit > usage.committed ? usage.limit - usage.committed : 0;
  uint64_t own = account->minimum > account->bytes ? account->minimum - account->bytes : 0;
  return UINT64_MAX - free < own ? UINT64_MAX : free + own;
}
MemorySlot *memory_scope(MemorySlot *slot) { MemorySlot *before = current; current = slot; return before; }
MemorySlot *memory_current(void) { return current; }
MemorySlot *memory_shared(void) {
  static MemoryAccount shared;
  static MemorySlot slot;
  static int initialized;
  if (!initialized) {
    memory_account_init(&shared, "shared");
    memory_slot_init(&slot, &shared, NULL, "shared runtime", NULL, NULL);
    slot.automatic = 1;
    initialized = 1;
  }
  return &slot;
}

typedef struct HostAllocation {
  struct HostAllocation *next;
  void *pointer;
  MemorySlot *slot;
  size_t bytes;
  int kv;
} HostAllocation;
static HostAllocation *host_allocations;
static pthread_mutex_t host_lock = PTHREAD_MUTEX_INITIALIZER;

void *memory_host_alloc(size_t bytes, int kv) {
  if (!current && !kv) return calloc(1, bytes ? bytes : 1);
  MemorySlot *slot = current;
  if (!kv && !memory_claim(slot, bytes, MEMORY_REQUIRED)) return NULL;
  void *pointer = calloc(1, bytes ? bytes : 1);
  HostAllocation *record = pointer ? malloc(sizeof *record) : NULL;
  if (!pointer || !record) {
    free(pointer);
    if (!kv) memory_release(slot, bytes);
    return NULL;
  }
  if (kv) memory_kv_change(bytes, 1);
  pthread_mutex_lock(&host_lock);
  *record = (HostAllocation){host_allocations, pointer, slot, bytes, kv};
  host_allocations = record;
  pthread_mutex_unlock(&host_lock);
  return pointer;
}

void *memory_host_required(size_t bytes, int kv) {
  void *pointer = memory_host_alloc(bytes, kv);
  if (!pointer) die("cannot allocate %zu bytes of %s memory", bytes, kv ? "KV" : "model");
  return pointer;
}

void memory_host_free(void *pointer) {
  pthread_mutex_lock(&host_lock);
  HostAllocation **at = &host_allocations;
  while (*at && (*at)->pointer != pointer) at = &(*at)->next;
  HostAllocation *record = *at;
  if (record) *at = record->next;
  pthread_mutex_unlock(&host_lock);
  if (record) {
    if (record->kv) memory_kv_change(record->bytes, 0);
    else memory_release(record->slot, record->bytes);
    free(record);
  }
  free(pointer);
}

void *memory_host_resize(void *pointer, size_t bytes) {
  if (!pointer) return memory_host_alloc(bytes, 0);
  pthread_mutex_lock(&host_lock);
  HostAllocation *record = host_allocations;
  while (record && record->pointer != pointer) record = record->next;
  pthread_mutex_unlock(&host_lock);
  if (!record) return realloc(pointer, bytes);
  size_t added = bytes > record->bytes ? bytes - record->bytes : 0;
  if (added && !record->kv && !memory_claim(record->slot, added, MEMORY_REQUIRED)) return NULL;
  void *next = realloc(pointer, bytes ? bytes : 1);
  if (!next) {
    if (added && !record->kv) memory_release(record->slot, added);
    return NULL;
  }
  if (record->kv) {
    if (added) memory_kv_change(added, 1);
    else memory_kv_change(record->bytes - bytes, 0);
  } else if (!added) memory_release(record->slot, record->bytes - bytes);
  pthread_mutex_lock(&host_lock);
  record->pointer = next;
  record->bytes = bytes;
  pthread_mutex_unlock(&host_lock);
  return next;
}
