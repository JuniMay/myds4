// Variable-sized model entries, dependency order and separate KV accounting.
#include "../../src/engine/memory.h"
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>

static double clock_now = 100;
double now(void) { return clock_now; }
_Noreturn void die(const char *format, ...) {
  va_list args; va_start(args, format); vfprintf(stderr, format, args); va_end(args);
  fputc('\n', stderr); exit(2);
}
#define CHECK(test) do { if (!(test)) die("%s:%d: %s", __func__, __LINE__, #test); } while (0)

typedef struct { MemorySlot slot; int id, evicted; } Entry;
static int order[16], order_count;
static void evict(void *context) {
  Entry *entry = context;
  order[order_count++] = entry->id;
  entry->evicted++;
  memory_release(&entry->slot, entry->slot.payload);
}
static void entry(Entry *entry, MemoryAccount *account, Entry *parent, uint64_t bytes, int id) {
  *entry = (Entry){.id = id};
  memory_slot_init(&entry->slot, account, parent ? &parent->slot : NULL, "fixture", evict, entry);
  CHECK(memory_claim(&entry->slot, bytes, MEMORY_REQUIRED));
  memory_publish(&entry->slot, 1);
}
static int claim(Entry *entry, uint64_t bytes, MemoryAdmission mode) {
  return memory_claim(&entry->slot, bytes, mode);
}

static void different_sizes_and_dependency_order(void) {
  memory_init(120);
  MemoryAccount moe, dense, other;
  memory_account_init(&moe, "MoE"), memory_account_init(&dense, "dense"), memory_account_init(&other, "other");
  Entry core, a, b, fixed, wanted;
  entry(&core, &moe, NULL, 20, 1); core.slot.touched = -1000;
  entry(&a, &moe, &core, 8, 2);
  entry(&b, &moe, &core, 12, 3);
  entry(&fixed, &dense, NULL, 60, 4); memory_pin(&fixed.slot, 1);
  entry(&wanted, &other, NULL, 0, 5);
  CHECK(claim(&wanted, 40, MEMORY_REQUIRED));
  CHECK(a.evicted == 1 && b.evicted == 1 && core.evicted == 0);
  CHECK(memory_usage().committed == 120);
  CHECK(claim(&wanted, 20, MEMORY_REQUIRED));
  CHECK(core.evicted == 1 && fixed.evicted == 0 && order_count == 3 && order[2] == 1);
  CHECK(memory_usage().allocated == 120);
}

static void dense_weights_are_one_entry(void) {
  memory_init(100);
  MemoryAccount a, b; memory_account_init(&a, "A"), memory_account_init(&b, "B");
  Entry dense, wanted; entry(&dense, &a, NULL, 80, 1); entry(&wanted, &b, NULL, 10, 2);
  CHECK(claim(&wanted, 25, MEMORY_REQUIRED));
  CHECK(dense.evicted == 1 && !memory_ready(&dense.slot));
  CHECK(dense.slot.bytes == 0 && wanted.slot.payload == 35);
}

static void pinned_entries_and_kv_are_separate(void) {
  memory_init(100);
  MemoryAccount a, b; memory_account_init(&a, "A"), memory_account_init(&b, "B");
  Entry dense, wanted; entry(&dense, &a, NULL, 90, 1); entry(&wanted, &b, NULL, 0, 2);
  memory_pin(&dense.slot, 1);
  memory_kv_change(250, 1);
  CHECK(claim(&wanted, 10, MEMORY_REQUIRED));
  CHECK(!claim(&wanted, 1, MEMORY_REQUIRED));
  CHECK(dense.evicted == 0 && wanted.slot.payload == 10);
  CHECK(memory_usage().committed == 100 && memory_usage().kv == 250);
  memory_kv_change(250, 0);
  CHECK(memory_usage().allocated == 100 && memory_usage().kv == 0);
}

static void active_minimum_cannot_overcommit(void) {
  memory_init(100);
  MemoryAccount a, b; memory_account_init(&a, "A"), memory_account_init(&b, "B");
  CHECK(memory_minimum(&a, 60));
  Entry fixed; entry(&fixed, &b, NULL, 40, 1); memory_pin(&fixed.slot, 1);
  CHECK(!memory_minimum(&a, 80));
  CHECK(a.minimum == 60 && memory_usage().committed == 100 && fixed.evicted == 0);
}

static void speculative_reads_do_not_evict_weights(void) {
  memory_init(100);
  MemoryAccount a, b; memory_account_init(&a, "A"), memory_account_init(&b, "B");
  Entry dense, wanted; entry(&dense, &a, NULL, 80, 1); entry(&wanted, &b, NULL, 0, 2);
  clock_now += 20;
  CHECK(!claim(&wanted, 40, MEMORY_CACHE));
  clock_now += 40;
  CHECK(!claim(&wanted, 40, MEMORY_SPECULATIVE));
  CHECK(dense.evicted == 0);
  CHECK(claim(&wanted, 40, MEMORY_CACHE));
  CHECK(dense.evicted == 1 && memory_usage().committed == 40);
}

static void combined_admission_refuses_without_partial_eviction(void) {
  memory_init(100);
  MemoryAccount a, b; memory_account_init(&a, "A"), memory_account_init(&b, "B");
  Entry idle, busy, one, two;
  entry(&idle, &a, NULL, 30, 1); entry(&busy, &a, NULL, 60, 2); memory_pin(&busy.slot, 1);
  entry(&one, &b, NULL, 0, 3); entry(&two, &b, NULL, 0, 4);
  MemorySlot *slots[] = {&one.slot, &two.slot};
  uint64_t sizes[] = {25, 25};
  CHECK(!memory_admit(slots, sizes, 2, MEMORY_REQUIRED));
  CHECK(idle.evicted == 0 && memory_usage().committed == 90);
  sizes[1] = 15;
  CHECK(memory_admit(slots, sizes, 2, MEMORY_REQUIRED));
  CHECK(idle.evicted == 1 && memory_usage().committed == 100);
  CHECK(claim(&one, 25, MEMORY_REQUIRED) && claim(&two, 15, MEMORY_REQUIRED));
  CHECK(memory_usage().allocated == 100);
}

static void cached_bytes_can_fund_a_promised_workspace(void) {
  memory_init(100);
  MemoryAccount account; memory_account_init(&account, "model");
  Entry work, cache;
  entry(&work, &account, NULL, 40, 1); memory_pin(&work.slot, 1);
  entry(&cache, &account, NULL, 50, 2);
  CHECK(memory_minimum(&account, 100));
  CHECK(claim(&work, 20, MEMORY_REQUIRED));
  CHECK(cache.evicted == 1 && work.slot.payload == 60);
  CHECK(memory_usage().committed == 100 && memory_usage().allocated == 60);
}

static void host_and_gpu_payloads_share_the_budget(void) {
  memory_init(100);
  MemoryAccount account; memory_account_init(&account, "model");
  MemorySlot work; memory_slot_init(&work, &account, NULL, "workspace", NULL, NULL);
  work.automatic = 1;
  memory_scope(&work);
  void *host = memory_host_alloc(60, 0), *kv = memory_host_alloc(250, 1);
  CHECK(host && kv && memory_usage().allocated == 60 && memory_usage().kv == 250);
  CHECK(memory_host_resize(host, 110) == NULL);
  CHECK(memory_usage().allocated == 60);
  host = memory_host_resize(host, 90);
  CHECK(host && memory_usage().committed == 90);
  memory_host_free(kv); memory_host_free(host);
  CHECK(memory_usage().allocated == 0 && memory_usage().kv == 0 && memory_usage().committed == 0);
}

static void ram_reserve_reclaims_idle_entries(void) {
  memory_init(120);
  MemoryAccount moe, dense;
  memory_account_init(&moe, "MoE"); memory_account_init(&dense, "dense");
  Entry core, a, b, fixed;
  entry(&core, &moe, NULL, 20, 1); core.slot.touched = -1000;
  entry(&a, &moe, &core, 8, 2); entry(&b, &moe, &core, 12, 3);
  entry(&fixed, &dense, NULL, 60, 4); memory_pin(&fixed.slot, 1);
  uint64_t peak = memory_usage().peak;
  CHECK(memory_reclaim(19) == 20);
  CHECK(a.evicted == 1 && b.evicted == 1 && core.evicted == 0);
  CHECK(memory_reclaim(100) == 20);
  CHECK(core.evicted == 1 && fixed.evicted == 0 && order[2] == 1);
  CHECK(memory_usage().allocated == 60 && memory_usage().peak == peak);
  CHECK(memory_reclaim(1) == 0);
}

static void temporary_limits_survive_new_accounts(void) {
  memory_init(100);
  MemoryAccount first, second;
  memory_account_init(&first, "first");
  Entry cached;
  entry(&cached, &first, NULL, 40, 1);
  CHECK(memory_reclaim(40) == 40);
  memory_set_limit(0);
  memory_account_init(&second, "second");
  CHECK(memory_usage().limit == 0 && memory_usage().allocated == 0);
  CHECK(!claim(&cached, 1, MEMORY_CACHE));
  memory_set_limit(100);
  CHECK(claim(&cached, 40, MEMORY_REQUIRED));
  CHECK(memory_usage().limit == 100 && memory_usage().allocated == 40);
}

int main(void) {
  void (*cases[])(void) = {different_sizes_and_dependency_order, dense_weights_are_one_entry,
    pinned_entries_and_kv_are_separate, active_minimum_cannot_overcommit,
    speculative_reads_do_not_evict_weights, combined_admission_refuses_without_partial_eviction,
    cached_bytes_can_fund_a_promised_workspace, host_and_gpu_payloads_share_the_budget,
    ram_reserve_reclaims_idle_entries, temporary_limits_survive_new_accounts};
  int failed = 0;
  for (size_t i = 0; i < sizeof cases / sizeof *cases; i++) {
    pid_t child = fork();
    if (!child) { cases[i](); _exit(0); }
    int status;
    CHECK(child > 0 && waitpid(child, &status, 0) == child);
    failed += !WIFEXITED(status) || WEXITSTATUS(status) != 0;
  }
  printf("shared model-memory pool: %zu/%zu cases passed\n", sizeof cases / sizeof *cases - failed, sizeof cases / sizeof *cases);
  return failed != 0;
}
