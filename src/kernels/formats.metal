// Number formats and small functions shared by every kernel source; the
// first source of the library (src/kernels/library.c).
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

// An FP8 E4M3 code as a float (no NaN codes in the weights).
inline float fp8(uchar b) {
  return float(as_type<half>(ushort(((b & 0x7Fu) << 7) | ((b & 0x80u) << 8)))) * 256.0f;
}
// An FP4 E2M1 code (the low 4 bits) as a float.
inline float fp4(uint nib) { return float(as_type<half>(ushort(((nib & 7u) << 9) | ((nib & 8u) << 12)))) * 16384.0f; }
// A BF16 value as a float.
inline float bf16f(ushort u) { return as_type<float>(uint(u) << 16); }
// Four BF16 values, the low half of each word first, as floats.
inline float4 bf16x4(uint2 q) {
  return float4(as_type<float>(q.x << 16), as_type<float>(q.x & 0xFFFF0000u),
                as_type<float>(q.y << 16), as_type<float>(q.y & 0xFFFF0000u));
}
inline float sigm(float x) { return 1.0f / (1.0f + exp(-x)); }

// The global top k is contained in the union of each SIMD group's top k.
// Equal scores keep the smaller original index. Scratch holds k entries per
// SIMD group; k is at most 16 for the routed-expert callers.
inline void group_topk(float key, uint index, uint k, threadgroup float *values,
                       threadgroup uint *indexes, device uint *out, threadgroup float *selected,
                       uint t, uint threads) {
  uint lane = t % 32, group = t / 32;
  for (uint rank = 0; rank < k; rank++) {
    float score = simd_max(key);
    uint winner = simd_min(key == score ? index : UINT_MAX);
    if (!lane) {
      values[group * k + rank] = score;
      indexes[group * k + rank] = winner;
    }
    if (index == winner) { key = -INFINITY; index = UINT_MAX; }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (!group) {
    uint at = 0;
    float score = lane < threads / 32 ? values[lane * k] : -INFINITY;
    uint candidate = lane < threads / 32 ? indexes[lane * k] : UINT_MAX;
    for (uint rank = 0; rank < k; rank++) {
      float maximum = simd_max(score);
      uint winner = simd_min(score == maximum ? candidate : UINT_MAX);
      if (!lane) { out[rank] = winner; selected[rank] = maximum; }
      if (candidate == winner) {
        at++;
        score = at < k ? values[lane * k + at] : -INFINITY;
        candidate = at < k ? indexes[lane * k + at] : UINT_MAX;
      }
    }
  }
}
// log(1 + e) without losing a small e (Metal has no log1p): Goldberg's
// correction of log(u) by e / (u - 1), u = 1 + e rounded.
inline float log1p_(float e) {
  float u = 1.0f + e;
  return u == 1.0f ? e : log(u) * e / (u - 1.0f);
}
