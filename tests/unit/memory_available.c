// Available memory is the free pages (with the speculative ones) and the file
// cache beyond the share the kernel keeps (vm.vm_page_filecache_min): while
// the file cache is below that share the kernel compresses and swaps other
// programs' memory rather than drop cached files. When the share cannot be
// read, only the free pages count.
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
#include "cache_layout.h"
#undef sysctlbyname

static int checks, failures;
static void expect(const char *name, uint64_t actual, uint64_t wanted) {
  checks++;
  if (actual == wanted) return;
  failures++;
  fprintf(stderr, "FAIL: %s (%llu != %llu)\n", name, (unsigned long long)actual, (unsigned long long)wanted);
}

int main(void) {
  st = deepseek_cache_create();
  vm_statistics64_data_t pages = {.free_count = 100, .external_page_count = 500, .speculative_count = 20};
  query_failed = 1;
  expect("an unreadable share counts only the free pages", avail_of(pages), 100 * vm_page_size);
  query_failed = 0, reported_floor = 200;
  expect("the file cache beyond the kept share counts", avail_of(pages), (100 + 480 - 200) * vm_page_size);
  reported_floor = 450;
  expect("a larger kept share leaves less", avail_of(pages), (100 + 30) * vm_page_size);
  reported_floor = 600;
  expect("file cache within the kept share does not count", avail_of(pages), 100 * vm_page_size);
  reported_floor = 0;
  expect("a share read as zero counts only the free pages", avail_of(pages), 100 * vm_page_size);
  reported_floor = 200, pages.external_page_count = 10;
  expect("speculative pages cannot underflow the file count", avail_of(pages), 100 * vm_page_size);
  pages.external_page_count = 220;
  expect("speculative pages are counted only in the free pages", avail_of(pages), 100 * vm_page_size);
  pages.free_count = 0, pages.external_page_count = 500;
  expect("file cache beyond the share remains available with no free pages", avail_of(pages), 280 * vm_page_size);
  expect("available memory reads the kept share", queries > 0, 1);
  fprintf(stderr, "available memory: %d/%d CPU checks passed\n", checks - failures, checks);
  return failures != 0;
}
