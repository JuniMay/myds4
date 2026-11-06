// Available memory includes reclaimable file pages without counting them twice.
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
#include "../../src/engine/io.c"
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
  expect("zero floor includes reclaimable files", avail_of(pages), 580 * vm_page_size);
  reported_floor = 200;
  expect("kernel floor does not mark cached files as used", avail_of(pages), 580 * vm_page_size);
  reported_floor = 450;
  expect("changing floor does not change available bytes", avail_of(pages), 580 * vm_page_size);
  reported_floor = 600;
  expect("all cached files remain reclaimable", avail_of(pages), 580 * vm_page_size);
  query_failed = 1;
  expect("unavailable floor does not discard cached files", avail_of(pages), 580 * vm_page_size);
  query_failed = 0, reported_floor = 200;
  expect("query recovery preserves available bytes", avail_of(pages), 580 * vm_page_size);
  pages.external_page_count = 10;
  expect("speculative pages cannot underflow file count", avail_of(pages), 100 * vm_page_size);
  pages.external_page_count = 20;
  expect("speculative pages are counted only in free pages", avail_of(pages), 100 * vm_page_size);
  pages.external_page_count = 0, pages.speculative_count = 0;
  expect("no file cache leaves only free pages", avail_of(pages), 100 * vm_page_size);
  pages.free_count = 0, pages.external_page_count = 500;
  expect("cached pages remain available with no free pages", avail_of(pages), 500 * vm_page_size);
  expect("available memory does not query the cache floor", queries, 0);
  fprintf(stderr, "available memory: %d/%d CPU checks passed\n", checks - failures, checks);
  return failures != 0;
}
