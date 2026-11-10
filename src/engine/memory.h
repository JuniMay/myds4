// Model allocations under an optional limit, and within the memory that
// other programs leave: memory_follow keeps the available memory at or
// above a reserve. KV storage is counted separately and comes first.
#pragma once
#include <stddef.h>
#include <stdint.h>
#include <stdatomic.h>

typedef struct MemoryAccount MemoryAccount;
typedef struct MemorySlot MemorySlot;
typedef enum { MEMORY_REQUIRED, MEMORY_CACHE, MEMORY_SPECULATIVE, MEMORY_WORKSPACE } MemoryAdmission;

struct MemoryAccount {
  MemoryAccount *next;
  const char *name;
  uint64_t bytes, minimum;
  int admitting;
};

struct MemorySlot {
  MemorySlot *next;
  MemoryAccount *account;
  MemorySlot *dependency; // this slot is discarded before its dependency
  MemorySlot *children, *sibling;
  const char *name;
  uint64_t bytes, payload;
  unsigned pins;
  double touched;
  int automatic; // ordinary allocations release their charge with their payload
  atomic_int ready;
  int (*can_evict)(void *context);
  void (*evict)(void *context);
  void *context;
};

typedef struct {
  uint64_t limit, allocated, committed, peak, kv, evictions, refusals;
} MemoryUsage;

void memory_init(uint64_t limit); // 0: no limit
void memory_set_limit(uint64_t bytes); // absolute temporary limit, including zero
// Additions keep available() (free pages and file pages the kernel can drop)
// at or above `reserve`: idle model memory goes back first, as for the limit.
void memory_follow(uint64_t (*available)(void), uint64_t reserve);
// macOS counts a new buffer's pages as in use only once they are written,
// while the engine charges the buffer when it allocates it. `unwritten`
// returns the charged bytes not written yet; readings count them as in use.
void memory_unwritten(uint64_t (*unwritten)(void));
// The most model memory this program may hold now: what it holds and the
// available memory above the reserve, within the limit.
uint64_t memory_current_limit(void);
void memory_account_init(MemoryAccount *account, const char *name);
void memory_slot_init(MemorySlot *slot, MemoryAccount *account, MemorySlot *dependency,
                      const char *name, void (*evict)(void *), void *context);
void memory_slot_remove(MemorySlot *slot);
void memory_touch(MemorySlot *slot);
void memory_pin(MemorySlot *slot, int pinned);
int memory_ready(const MemorySlot *slot);
void memory_publish(MemorySlot *slot, int ready);

// Reserve all requested sizes together. Refusal leaves existing entries intact.
int memory_admit(MemorySlot *const *slots, const uint64_t *bytes, size_t count, MemoryAdmission mode);
int memory_minimum(MemoryAccount *account, uint64_t bytes);
int memory_minimum_mode(MemoryAccount *account, uint64_t bytes, MemoryAdmission mode);
int memory_claim(MemorySlot *slot, uint64_t bytes, MemoryAdmission mode);
void memory_release(MemorySlot *slot, uint64_t bytes);
void memory_trim(MemorySlot *slot);
// Before KV grows, idle model memory gives back the space it needs above the
// reserve: KV comes first, and the experts use only the space it leaves.
void memory_kv_change(uint64_t bytes, int allocated);
MemoryUsage memory_usage(void);
void memory_collect_with(void (*collect)(void));

// The engine thread supplies the current operation's allocation owner.
MemorySlot *memory_scope(MemorySlot *slot);
MemorySlot *memory_current(void);
MemorySlot *memory_shared(void);
uint64_t memory_room(const MemoryAccount *account);
uint64_t memory_reclaim(uint64_t bytes); // release idle entries, respecting active allocations and dependencies
// Releases every idle entry of one account, each before the entries it
// depends on; pinned entries and other accounts' stay. Returns the bytes.
uint64_t memory_unload(MemoryAccount *account);
void *memory_host_alloc(size_t bytes, int kv); // zeroed; null when the budget refuses it
void *memory_host_required(size_t bytes, int kv);
void *memory_host_resize(void *pointer, size_t bytes);
void memory_host_free(void *pointer); // also accepts ordinary malloc allocations
