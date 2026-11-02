// CPU-only cache-budget regression: kernel file-cache limits may change.
#include <sys/sysctl.h>
#include <errno.h>
#include <stdint.h>
#include <string.h>

static uint64_t reported_floor;
static int query_failed, queries;
static int fixture_sysctl(const char *name, void *old, size_t *length, void *value, size_t value_length) {
  queries++;
  if (query_failed || strcmp(name, "vm.vm_page_filecache_min")) { errno = ENOENT; return -1; }
  if (!old || !length || *length < sizeof reported_floor || value || value_length) return -1;
  memcpy(old, &reported_floor, sizeof reported_floor);
  *length = sizeof reported_floor;
  return 0;
}

#define sysctlbyname fixture_sysctl
#include "../src/io.c"
#undef sysctlbyname

static int checks, failures;
static void expect(const char *name, uint64_t actual, uint64_t wanted) {
  checks++;
  if (actual == wanted) return;
  failures++;
  fprintf(stderr, "FAIL: %s (%llu != %llu)\n", name, (unsigned long long)actual, (unsigned long long)wanted);
}

int main(void) {
  vm_statistics64_data_t pages = {.free_count = 100, .external_page_count = 500, .speculative_count = 20};
  reported_floor = 0;
  expect("zero floor remains conservative", avail_of(pages), 100 * vm_page_size);
  reported_floor = 200;
  expect("later published floor is observed", avail_of(pages), 380 * vm_page_size);
  reported_floor = 450;
  expect("increased floor reduces reclaimable cache", avail_of(pages), 130 * vm_page_size);
  reported_floor = 600;
  expect("file cache below its floor contributes nothing", avail_of(pages), 100 * vm_page_size);
  query_failed = 1;
  expect("query failure falls back to free pages", avail_of(pages), 100 * vm_page_size);
  query_failed = 0, reported_floor = 200;
  expect("query failure is retried", avail_of(pages), 380 * vm_page_size);
  expect("each budget calculation observes current kernel policy", queries, 6);
  fprintf(stderr, "available memory: %d/%d CPU checks passed\n", checks - failures, checks);
  return failures != 0;
}
