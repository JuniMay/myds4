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
// log(1 + e) without losing a small e (Metal has no log1p): Goldberg's
// correction of log(u) by e / (u - 1), u = 1 + e rounded.
inline float log1p_(float e) {
  float u = 1.0f + e;
  return u == 1.0f ? e : log(u) * e / (u - 1.0f);
}
