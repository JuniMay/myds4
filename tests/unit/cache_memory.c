// Exercise real cache initialization and resizing with controlled VM counters.
// No model weights, readers or GPU commands are created.
#include "../../src/model/deepseek/model.h"
#include <mach/mach.h>
#include <pthread.h>
#include <sys/sysctl.h>
#include <sys/wait.h>
#include "myds4.h"
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
static vm_statistics64_data_t vm_fixture;
static int pressure_fixture = 1;
static uint64_t compressed_fixture, wakeups_fixture, working_set_fixture = 128ull << 30;
static uint64_t filecache_min_fixture;  // pages; set in main
static kern_return_t fixture_host(host_t host, host_flavor_t flavor, host_info64_t out, mach_msg_type_number_t *n) {
  memcpy(out, &vm_fixture, sizeof vm_fixture);
  return KERN_SUCCESS;
}
static kern_return_t fixture_task(task_name_t task, task_flavor_t flavor, task_info_t out, mach_msg_type_number_t *n) {
  task_vm_info_data_t value = {.phys_footprint = 8ull << 30, .compressed = compressed_fixture};
  memcpy(out, &value, sizeof value);
  return KERN_SUCCESS;
}
static int fixture_sysctl(const char *name, void *out, size_t *n, void *in, size_t len) {
  uint64_t value = !strcmp(name, "hw.memsize") ? 128ull << 30 :
                   !strcmp(name, "kern.memorystatus_vm_pressure_level") ? pressure_fixture :
                   !strcmp(name, "vm.compressor.swapper.wakeups.total") ? wakeups_fixture :
                   !strcmp(name, "vm.vm_page_filecache_min") ? filecache_min_fixture : 0;
  if (!out || !n) return -1;
  size_t size = *n < sizeof value ? *n : sizeof value;
  memcpy(out, &value, size), *n = size;
  return 0;
}
static uint64_t fixture_working_set(void) { return working_set_fixture; }
static void fixture_residency(void) {}
static int fixture_thread(pthread_t *thread, const pthread_attr_t *a, void *(*f)(void *), void *arg) { return 0; }
#define host_statistics64 fixture_host
#define task_info fixture_task
#define sysctlbyname fixture_sysctl
#define gpu_working_set fixture_working_set
#define gpu_residency_commit fixture_residency
#define pthread_create fixture_thread
#include "../../src/engine/io.c"
#include "cache_layout.h"
#undef host_statistics64
#undef task_info
#undef sysctlbyname
#undef gpu_working_set
#undef gpu_residency_commit
#undef pthread_create

static int checks, failures;
static void expect(const char *name, int ok) {
  checks++;
  if (!ok) failures++, fprintf(stderr, "FAIL: %s\n", name);
}
static void child_case(int kind) {
  vm_fixture.free_count = (1ull << 30) / vm_page_size;
  vm_fixture.external_page_count = (20ull << 30) / vm_page_size;
  vm_fixture.speculative_count = 0;
  st->nslots = st->cap = st->max_slots - 256;
  if (kind == 7) st->nslots = 0;
  if (kind == 8) st->nslots = st->cap = 2 * EXPERTS;
  if (kind >= 9) vm_fixture.external_page_count = (8ull << 30) / vm_page_size;  // within the kept share
  if (kind >= 10) vm_fixture.free_count = (15ull << 28) / vm_page_size;  // 3.75 GiB: 0.25 GiB short of the reserve
  if (kind == 11) {  // a second model's store beside the first, the same shortfall for both
    ExpertStore *other = deepseek_cache_create();
    cache_init(other, 56ull << 30, LAYERS);
    other->nslots = other->cap = other->max_slots - 256;
    uint64_t alone = cache_unwritten();
    other->slots[0].b = (Buf){.obj = &other->slots[0], .allocated = 1ull << 20};  // charged, never written
    int counted = cache_unwritten() == alone + (1ull << 20);
    other->slots[0].b = (Buf){0};
    int first = st->cap, second = other->cap;
    resize(st, -1);
    resize(other, -1);  // at once: the memory the first gave back cannot show yet
    int expected = first - (int)(((4ull << 30) - (15ull << 28)) / st->SLOT_SIZE) - 1;
    fprintf(stderr, "cache reproduction %d: two stores 0.25 GiB short of the reserve: limits %d -> %d and %d -> %d "
            "(expected %d and %d); the second store's unwritten slot %s\n", kind, first, st->cap, second, other->cap,
            expected, second, counted ? "counts" : "does not count");
    _exit(counted && st->cap == expected && other->cap == second ? 0 : 1);
  }
  int before = st->cap;
  if (kind == 2) pressure_fixture = 2;
  resize(st, -1);
  if (kind == 10) {  // the slots that restore the reserve, not a sixteenth of the cache
    int expected = before - (int)(((4ull << 30) - (15ull << 28)) / st->SLOT_SIZE) - 1;
    fprintf(stderr, "cache reproduction %d: 0.25 GiB short of the reserve, no pressure: limit %d -> %d (expected %d)\n",
            kind, before, st->cap, expected);
    _exit(st->cap == expected ? 0 : 1);
  }
  if (kind == 0) _exit(st->cap == before ? 0 : 1);
  if (kind == 1) {
    for (int i = 0; i < 32; i++) resize(st, -1);
    _exit(st->cap > before && st->cap <= st->max_slots ? 0 : 1);
  }
  if (kind == 3) vm_fixture.compressions++, compressed_fixture = 128ull << 20;
  if (kind == 4) vm_fixture.swapouts++;
  if (kind == 6) wakeups_fixture++;
  if (kind == 7 || kind == 8) vm_fixture.swapouts++;
  if (kind == 5) {
    for (int i = 0; i < 32; i++) {
      vm_fixture.compressions++, compressed_fixture += 128ull << 20;
      resize(st, -1);
    }
    _exit(st->cap > before && st->cap <= st->max_slots ? 0 : 1);
  }
  if (kind >= 3) resize(st, -1);
  if (kind == 6 || kind == 7) {
    fprintf(stderr, "cache reproduction %d: limit %d -> %d, allocated %d, swapouts %llu, wakeups %llu\n",
            kind, before, st->cap, st->nslots, (unsigned long long)vm_fixture.swapouts, (unsigned long long)wakeups_fixture);
    _exit(st->cap == before ? 0 : 1);
  }
  if (kind == 8) {
    for (int i = 0; i < 32; i++) resize(st, -1);
    fprintf(stderr, "cache reproduction %d: limit %d -> %d after 32 calm calls, allocated %d\n", kind, before, st->cap, st->nslots);
    _exit(st->cap > before && st->cap <= st->max_slots ? 0 : 1);
  }
  if (kind == 3) _exit(st->cap == before ? 0 : 1);
  if (kind == 9) {
    fprintf(stderr, "cache reproduction %d: 1 GiB free and 8 GiB of file cache within a 10 GiB kept share: limit %d -> %d\n",
            kind, before, st->cap);
    _exit(st->cap < before ? 0 : 1);
  }
  _exit(st->cap < before ? 0 : 1);
}
// macOS counts a slot's pages as in use only once a read writes them, so the
// cache reports the charged bytes no completed load has written
// (memory_unwritten). Layer 1's experts here are smaller than the slot.
static void reads(Slot *s, int from, int to) {  // a reader completes pieces from..to-1
  for (int k = from; k < to; k++) atomic_store(&s->taken[k], 1), piece_finished(st, s, k, 1);
}
static void unwritten_checks(void) {
  for (int e = 0; e < EXPERTS; e++) expert_span[1][e][0].len = EXPERT_W / 2;
  Slot *s = &st->slots[0];
  uint64_t whole = gpu_allocation_size(st->SLOT_SIZE);
  s->b = (Buf){.obj = s, .ptr = malloc(whole), .allocated = whole};  // a new buffer, never touched
  s->layer = 1, s->expert = 0;
  start(st, 0, 0);
  expect("a new slot counts as unwritten while its read is under way", cache_unwritten() == whole);
  reads(s, 0, s->npiece - 1);
  expect("a partly read slot still counts as unwritten", cache_unwritten() == whole);
  reads(s, s->npiece - 1, s->npiece);
  uint64_t small = s->extent;
  expect("a completed read counts as written before the model thread reaps it",
         small && small < whole && cache_unwritten() == whole - small);
  finish(st, 0);
  expect("a reaped read leaves unwritten only the slot beyond its smaller expert",
         s->written == small && cache_unwritten() == whole - small);
  s->layer = 0;
  start(st, 0, 0);
  reads(s, 0, 1);
  cancel(st, 0);
  expect("a load that dropped a piece adds nothing written", s->written == small && cache_unwritten() == whole - small);
  start(st, 0, 0);
  reads(s, 0, s->npiece);
  finish(st, 0);
  expect("a larger expert's completed read writes the rest of the slot",
         s->written == s->extent && cache_unwritten() == whole - s->extent && s->extent > small);
  memset(s, 0, sizeof *s);
}

int main(void) {
  st = deepseek_cache_create();
  alarm(10);
  working_set_fixture = 0;
  expect("an unopened Metal device supplies no residency target", mem_gpu_limit() == INT64_MAX);
  working_set_fixture = 100ull << 30;
  expect("residency follows Metal's working-set recommendation", mem_gpu_limit() == (int64_t)working_set_fixture);
  working_set_fixture = 128ull << 30;
  expect("an early query does not freeze the residency target", mem_gpu_limit() == (int64_t)working_set_fixture);
  filecache_min_fixture = (10ull << 30) / vm_page_size;
  vm_fixture.free_count = (48ull << 30) / vm_page_size;
  vm_fixture.external_page_count = (30ull << 30) / vm_page_size;
  vm_fixture.speculative_count = (2ull << 30) / vm_page_size;
  for (int l = 0; l < CACHE_LAYERS; l++) {
    expert_spans[l] = 2;
    for (int e = 0; e < EXPERTS; e++) {
      expert_span[l][e][0].len = EXPERT_W;
      expert_span[l][e][1].len = EXPERT_S;
    }
  }
  cache_reserve(4);
  cache_init(st, 56ull << 30, LAYERS);
  expect("initial capacity uses file cache beyond the kept share to fit the requested 56 GiB",
         st->cap == (int)((56ull << 30) / st->SLOT_SIZE));
  expect("initial capacity respects the configured upper bound", st->cap <= st->max_slots);
  const char *names[] = {"low free pages with file cache beyond the kept share do not shrink",
                         "calm growth uses file cache beyond the kept share",
                         "actual memory pressure still shrinks", "compression alone does not shrink", "new swapping still shrinks",
                         "compression alone does not prevent growth", "a swapper wakeup without swap-outs does not shrink",
                         "a global swap-out cannot reclaim an empty expert cache", "a minimum-sized cache can recover using available memory",
                         "file cache within the kept share does not count: low free pages shrink",
                         "a shortfall without pressure gives back only the slots that restore the reserve",
                         "two stores give one shortfall back once, and both stores' unwritten slots count"};
  for (int kind = 0; kind < 12; kind++) {
    pid_t pid = fork();
    if (!pid) child_case(kind);
    int status;
    if (pid < 0 || waitpid(pid, &status, 0) != pid) return 1;
    expect(names[kind], WIFEXITED(status) && WEXITSTATUS(status) == 0);
  }
  unwritten_checks();
  free(st->slots), free(st->score), free(st->free_list);
  printf("cache memory policy: %d/%d CPU checks passed\n", checks - failures, checks);
  return failures != 0;
}
