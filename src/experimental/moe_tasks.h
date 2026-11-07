// Host layout of MoeTask in moe_tasks.metal. Addresses name resident Metal
// buffers; a descriptor remains unchanged until its GPU batch completes.
#pragma once
#include <stdint.h>
typedef struct {
  uint64_t ex, x, h, shared_out, mix, res, out;
  uint32_t first, groups, kind, experts;
} MoeTask;
static_assert(sizeof(MoeTask) == 72);
