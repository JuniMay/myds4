// Completion without a full synchronization (gpu_reap) and deferred frees
// (gpu_free_later). A buffer that queued work uses stays allocated and charged
// until that work has finished; one that no work uses is freed at once. Runs
// a spin kernel for about 50 ms; needs no model.
#include "myds4.h"
#include "memory.h"
#include <stdio.h>
#include <unistd.h>

static int failed;
#define CHECK(x) do { if (!(x)) printf("FAIL: %s\n", #x), failed = 1; } while (0)

static int spin;

static void spinning(const Buf *out, uint32_t iterations) {
  gpu_kern(spin);
  gpu_bytes(0, &iterations, 4);
  gpu_buf(1, out, 0);
  gpu_go(1, 1, 1, 32, 1, 1);
}

int main(void) {
  memory_configure(8, 10);
  gpu_start();
  spin = gpu_kernel("spin");
  enum { SIZE = 1 << 20 };
  Buf out = gpu_alloc(16);

  // The time per iteration, measured with a full synchronization; the first
  // run warms up the GPU clock.
  double busy = 0, per = 0;
  for (int i = 0; i < 2; i++) {
    busy = gpu_busy;
    gpu_begin();
    spinning(&out, 200000);
    gpu_commit();
    gpu_sync();
    per = (gpu_busy - busy) / 200000;
  }
  CHECK(per > 0);
  uint32_t iterations = (uint32_t)(0.05 / per);

  // With nothing outstanding, a deferred free is immediate.
  Buf idle = gpu_alloc(SIZE);
  uint64_t held = memory_usage().allocated;
  gpu_free_later(&idle);
  CHECK(!idle.obj);
  CHECK(memory_usage().allocated == held - gpu_allocation_size(SIZE));

  // A buffer that a running command buffer uses stays charged until it ends.
  Buf used = gpu_alloc(SIZE);
  held = memory_usage().allocated;
  busy = gpu_busy;
  gpu_begin();
  spinning(&used, iterations);
  uint64_t signal = gpu_signal();
  gpu_commit();
  gpu_free_later(&used);
  CHECK(!used.obj);
  CHECK(memory_usage().allocated == held);
  gpu_reap();  // about 50 ms of work: it has not finished
  CHECK(memory_usage().allocated == held);
  CHECK(gpu_busy == busy);
  gpu_wait_signal(signal);
  for (int i = 0; i < 2000 && memory_usage().allocated == held; i++) {
    gpu_reap();
    usleep(1000);
  }
  CHECK(memory_usage().allocated == held - gpu_allocation_size(SIZE));
  CHECK(gpu_busy > busy);

  // A buffer used by a command buffer still being encoded waits for it too.
  Buf open = gpu_alloc(SIZE);
  held = memory_usage().allocated;
  gpu_begin();
  spinning(&open, 1000);
  gpu_free_later(&open);
  CHECK(!open.obj);
  CHECK(memory_usage().allocated == held);
  gpu_commit();
  CHECK(memory_usage().allocated == held);
  gpu_sync();
  CHECK(memory_usage().allocated == held - gpu_allocation_size(SIZE));

  // Reaped buffers need no further synchronization.
  gpu_sync();
  printf("%s\n", failed ? "gpu steps: FAILED" : "gpu steps: reap and deferred frees passed");
  return failed;
}
