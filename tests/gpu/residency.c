// Residency partitioning must preserve bytes, aliases, removal and GPU data.
#include "myds4.h"
#include "memory.h"
#include <stdlib.h>
#include <string.h>

static int failures, checks;
#define CHECK(x) do { checks++; if (!(x)) { failures++; fprintf(stderr, "FAIL: %s\n", #x); } } while (0)

int main(void) {
  setenv("MYDS4_RESIDENCY_SET_BYTES", "1048576", 1);
  memory_init(1ull << 30);
  static const char source[] =
      "#include <metal_stdlib>\nusing namespace metal;\n"
      "kernel void read_value(device const uint *x [[buffer(0)]], device uint *y [[buffer(1)]]) { y[0]=x[0]; }\n";
  gpu_init(source, sizeof source - 1);
  GpuResidencyInfo info;
  Buf a = gpu_alloc(768 << 10), b = gpu_alloc(768 << 10);
  gpu_residency_info(&info);
  CHECK(info.sets == 2);
  CHECK(info.bytes == a.allocated + b.allocated);
  CHECK(info.largest_set <= info.target_set);
  CHECK(info.allocations == 2);
  Buf view = a; view.off = 16; view.size -= 16; view.allocated = 0;
  gpu_resident(&view, 1); gpu_residency_commit();
  gpu_residency_info(&info);
  CHECK(info.allocations == 2);
  CHECK(info.bytes == a.allocated + b.allocated);
  gpu_resident(&a, 0); gpu_residency_commit();
  gpu_residency_info(&info);
  CHECK(info.allocations == 1 && info.bytes == b.allocated);
  gpu_resident(&a, 1); gpu_residency_commit();
  gpu_residency_info(&info);
  CHECK(info.sets == 2 && info.allocations == 2);
  ((uint32_t *)a.ptr)[0] = 123456789;
  int read = gpu_kernel("read_value");
  gpu_begin(); gpu_kern(read); gpu_buf(0, &a, 0); gpu_buf(1, &b, 0);
  gpu_go(1, 1, 1, 1, 1, 1); gpu_commit(); gpu_sync();
  CHECK(((uint32_t *)b.ptr)[0] == 123456789);
  gpu_free(&a); gpu_free(&b); gpu_residency_commit();
  gpu_residency_info(&info);
  CHECK(info.allocations == 0 && info.bytes == 0);
  Buf large = gpu_alloc(2 << 20), small = gpu_alloc(128 << 10);
  gpu_residency_info(&info);
  CHECK(info.sets == 2);
  CHECK(info.largest_set == large.allocated);
  gpu_free(&large); gpu_free(&small); gpu_residency_commit();
  enum { COUNT = 80 };
  Buf many[COUNT];
  for (int i = 0; i < COUNT; i++) many[i] = gpu_alloc(768 << 10);
  gpu_residency_info(&info);
  CHECK(info.sets == 32 && info.allocations == COUNT);
  CHECK(info.bytes == (uint64_t)COUNT * many[0].allocated);
  for (int round = 0; round < 4; round++) {
    for (int i = 0; i < COUNT; i += 2) gpu_resident(&many[i], 0);
    gpu_residency_commit(); gpu_residency_info(&info);
    CHECK(info.allocations == COUNT / 2 && info.bytes == (uint64_t)(COUNT / 2) * many[0].allocated);
    for (int i = 0; i < COUNT; i += 2) gpu_resident(&many[i], 1);
    gpu_residency_commit(); gpu_residency_info(&info);
    CHECK(info.allocations == COUNT && info.bytes == (uint64_t)COUNT * many[0].allocated);
  }
  for (int i = 0; i < COUNT; i++) gpu_free(&many[i]);
  gpu_residency_commit(); gpu_residency_info(&info);
  CHECK(info.allocations == 0 && info.bytes == 0);
  printf("residency sets: %d/%d checks passed\n", checks - failures, checks);
  return failures != 0;
}
