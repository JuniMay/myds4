// An eviction in the middle of a pass must not wait for the GPU. Command
// buffers run in the order they are opened. With buffers open in the queue
// and another committed behind them, an allocation that takes back an idle
// expert must not wait for the committed one: the GPU runs it only after the
// open buffers, which the waiting thread commits later, so the wait would
// never end.
#include "myds4.h"
#include "memory.h"
#include <stdlib.h>
#include <unistd.h>

typedef struct { MemorySlot slot; Buf b; int evicted; } Entry;
static void evict(void *context) {
  Entry *e = context;
  gpu_free(&e->b);
  e->evicted++;
}

int main(void) {
  alarm(20);  // a wait that never ends fails the check by this alarm
  memory_init(64ull << 20);
  static const char source[] =
      "#include <metal_stdlib>\nusing namespace metal;\n"
      "kernel void add_one(device uint *y [[buffer(0)]]) { y[0] += 1; }\n";
  gpu_init(source, sizeof source - 1);
  int add_one = gpu_kernel("add_one");
  MemoryAccount cache;
  memory_account_init(&cache, "expert cache");
  static Entry expert;
  memory_slot_init(&expert.slot, &cache, NULL, "routed expert", evict, &expert);
  MemorySlot *before = memory_scope(&expert.slot);
  expert.b = gpu_alloc(56ull << 20);  // an idle expert holds most of the 64 MiB limit
  memory_scope(before);
  Buf out = gpu_alloc(16384);
  // Two buffers open in the queue, then one committed behind them.
  void *experts = gpu_open(), *rest = gpu_open();
  gpu_begin(), gpu_kern(add_one), gpu_buf(0, &out, 0), gpu_go(1, 1, 1, 1, 1, 1), gpu_commit();
  double t0 = now();
  Buf more = gpu_alloc(16ull << 20);  // past the limit: the idle expert goes back
  double waited = now() - t0;
  gpu_encode(experts), gpu_kern(add_one), gpu_buf(0, &out, 0), gpu_go(1, 1, 1, 1, 1, 1), gpu_commit();
  gpu_submit(rest);  // an open buffer with nothing encoded in it
  gpu_sync();
  uint32_t added = ((uint32_t *)out.ptr)[0];
  int ok = expert.evicted == 1 && more.obj && added == 2 && waited < 1.0;
  fprintf(stderr, "an eviction in the middle of a pass: the expert %s, the allocation waited %.3f s, "
                  "the buffers added %u (2): %s\n",
          expert.evicted ? "went back" : "stayed", waited, added, ok ? "passed" : "FAILED");
  gpu_free(&more), gpu_free(&out);
  return !ok;
}
