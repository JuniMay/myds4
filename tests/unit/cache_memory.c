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
                   !strcmp(name, "vm.compressor.swapper.wakeups.total") ? wakeups_fixture : 0;
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
  nslots = cap = max_slots - 256;
  if (kind == 7) nslots = 0;
  if (kind == 8) nslots = cap = 2 * EXPERTS;
  int before = cap;
  if (kind == 2) pressure_fixture = 2;
  resize(-1);
  if (kind == 0) _exit(cap == before ? 0 : 1);
  if (kind == 1) {
    for (int i = 0; i < 32; i++) resize(-1);
    _exit(cap > before && cap <= max_slots ? 0 : 1);
  }
  if (kind == 3) vm_fixture.compressions++, compressed_fixture = 128ull << 20;
  if (kind == 4) vm_fixture.swapouts++;
  if (kind == 6) wakeups_fixture++;
  if (kind == 7 || kind == 8) vm_fixture.swapouts++;
  if (kind == 5) {
    for (int i = 0; i < 32; i++) {
      vm_fixture.compressions++, compressed_fixture += 128ull << 20;
      resize(-1);
    }
    _exit(cap > before && cap <= max_slots ? 0 : 1);
  }
  if (kind >= 3) resize(-1);
  if (kind == 6 || kind == 7) {
    fprintf(stderr, "cache reproduction %d: limit %d -> %d, allocated %d, swapouts %llu, wakeups %llu\n",
            kind, before, cap, nslots, (unsigned long long)vm_fixture.swapouts, (unsigned long long)wakeups_fixture);
    _exit(cap == before ? 0 : 1);
  }
  if (kind == 8) {
    for (int i = 0; i < 32; i++) resize(-1);
    fprintf(stderr, "cache reproduction %d: limit %d -> %d after 32 calm calls, allocated %d\n", kind, before, cap, nslots);
    _exit(cap > before && cap <= max_slots ? 0 : 1);
  }
  if (kind == 3) _exit(cap == before ? 0 : 1);
  _exit(cap < before ? 0 : 1);
}
int main(void) {
  alarm(10);
  working_set_fixture = 0;
  expect("an unopened Metal device supplies no residency target", mem_gpu_limit() == INT64_MAX);
  working_set_fixture = 100ull << 30;
  expect("residency follows Metal's working-set recommendation", mem_gpu_limit() == (int64_t)working_set_fixture);
  working_set_fixture = 128ull << 30;
  expect("an early query does not freeze the residency target", mem_gpu_limit() == (int64_t)working_set_fixture);
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
  cache_init(NULL, 56ull << 30, LAYERS);
  expect("initial capacity uses file cache to fit the requested 56 GiB", cap == (int)((56ull << 30) / SLOT_SIZE));
  expect("initial capacity respects the configured upper bound", cap <= max_slots);
  const char *names[] = {"low free pages with ample file cache do not shrink", "calm growth uses available file cache",
                         "actual memory pressure still shrinks", "compression alone does not shrink", "new swapping still shrinks",
                         "compression alone does not prevent growth", "a swapper wakeup without swap-outs does not shrink",
                         "a global swap-out cannot reclaim an empty expert cache", "a minimum-sized cache can recover using available memory"};
  for (int kind = 0; kind < 9; kind++) {
    pid_t pid = fork();
    if (!pid) child_case(kind);
    int status;
    if (pid < 0 || waitpid(pid, &status, 0) != pid) return 1;
    expect(names[kind], WIFEXITED(status) && WEXITSTATUS(status) == 0);
  }
  free(slots), free(score), free(free_list);
  printf("cache memory policy: %d/%d CPU checks passed\n", checks - failures, checks);
  return failures != 0;
}
