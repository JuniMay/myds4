// A model's weights load without filling the file cache. Checkpoint files
// are opened with F_NOCACHE, but uncached reads bypass the page cache only in
// whole 16 KiB-aligned spans into page-aligned memory (io.c); unaligned reads
// still fill it, and the kernel keeps that file cache below its kept share
// (avail_of, io.c). So every read of a tensor at its own offset is aligned,
// and the tensor's bytes arrive intact.
//
// weights.c runs with fixture buffers and a fixture file: no Metal device.
#include "myds4.h"
#include "../../src/engine/memory.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

enum { FILE_BYTES = 1 << 20, ALIGN = 16384 };
static uint8_t file[FILE_BYTES];
static int reads, unaligned;

static ssize_t fixture_pread(int fd, void *buffer, size_t bytes, off_t offset) {
  reads++;
  if (offset % ALIGN || bytes % ALIGN || (uintptr_t)buffer % ALIGN) unaligned++;
  if (offset >= FILE_BYTES) return 0;
  size_t n = (size_t)offset + bytes > FILE_BYTES ? FILE_BYTES - (size_t)offset : bytes;
  memcpy(buffer, file + offset, n);
  return (ssize_t)n;
}
static Buf fixture_new(size_t size) {
  Buf b = {.allocated = size, .size = size};
  b.memory = memory_current() ? memory_current() : memory_shared();
  if (!memory_claim(b.memory, b.allocated, MEMORY_REQUIRED)) die("the fixture budget refused %zu bytes", size);
  b.ptr = calloc(1, size ? size : 1);
  b.obj = b.ptr;
  return b;
}
static void fixture_free(Buf *b) {
  if (b->obj) {
    memory_release(b->memory, b->allocated);
    free(b->ptr);
  }
  *b = (Buf){0};
}
static size_t fixture_allocation_size(size_t size) { return size; }
static void fixture_resident(const Buf *b, int on) {}
static void fixture_commit(void) {}
static void fixture_read(int fd, uint64_t off, uint64_t len, Buf *dst) {}
#define pread fixture_pread
#define gpu_new fixture_new
#define gpu_free fixture_free
#define gpu_allocation_size fixture_allocation_size
#define gpu_resident fixture_resident
#define gpu_residency_commit fixture_commit
#define io_read fixture_read
#include "../../src/model/weights.c"
#undef pread
#undef gpu_new
#undef gpu_free
#undef gpu_allocation_size
#undef gpu_resident
#undef gpu_residency_commit
#undef io_read

int main(void) {
  for (size_t i = 0; i < FILE_BYTES; i++) file[i] = (uint8_t)(i * 2654435761u >> 24);
  memory_init(1ull << 30);
  MemoryAccount account;
  memory_account_init(&account, "fixture model");
  WeightSet *set = weights_new(&account, "fixture weights");
  // tensors at offsets a safetensors file gives: unaligned, of any size, one
  // reaching the end of the file
  const struct { uint64_t off; size_t size; } at[] = {{12345, 100000}, {200001, 3}, {300000, 70000},
                                                      {FILE_BYTES - 5000, 5000}};
  enum { TENSORS = sizeof at / sizeof *at };
  Buf targets[TENSORS] = {0};
  for (int i = 0; i < TENSORS; i++)
    weights_tensor(set, &targets[i], (Tensor){.fd = 3, .off = at[i].off, .size = at[i].size}, 0);
  int loaded = weights_load(set), intact = 1;
  for (int i = 0; i < TENSORS; i++) intact &= !memcmp(targets[i].ptr, file + at[i].off, at[i].size);
  int failures = 0;
  if (!loaded) failures++, fprintf(stderr, "FAIL: the weights load\n");
  if (unaligned) failures++, fprintf(stderr, "FAIL: %d of %d reads are not 16 KiB-aligned spans into aligned memory\n", unaligned, reads);
  if (!intact) failures++, fprintf(stderr, "FAIL: a tensor's bytes differ from the file's\n");
  fprintf(stderr, "weight reads: %d/3 CPU checks passed (%d reads, %d unaligned)\n", 3 - failures, reads, unaligned);
  return failures != 0;
}
