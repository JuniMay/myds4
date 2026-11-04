// CPU reference codecs for compact activation buffers. These do not select
// a runtime kernel or change the engine's current activation storage.
#ifndef MYDS4_REFERENCE_ACTIVATION_FORMAT_H
#define MYDS4_REFERENCE_ACTIVATION_FORMAT_H

#include <math.h>
#include <stdint.h>
#include <string.h>

enum { REF_ACT_BLOCK = 32 };
typedef struct {
  uint8_t values[REF_ACT_BLOCK];
  uint8_t scale;
} RefFp8Block;
static_assert(sizeof(RefFp8Block) == 33, "32 E4M3 values and one E8M0 scale");

static inline uint32_t ref_float_bits(float x) {
  uint32_t u;
  memcpy(&u, &x, sizeof u);
  return u;
}
static inline float ref_bits_float(uint32_t u) {
  float x;
  memcpy(&x, &u, sizeof x);
  return x;
}
static inline uint16_t ref_bf16(float x) {
  uint32_t u = ref_float_bits(x);
  if ((u & 0x7FFFFFFFu) > 0x7F800000u) return (uint16_t)(u >> 16) | 0x40u;
  return (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}
static inline float ref_from_bf16(uint16_t u) { return ref_bits_float((uint32_t)u << 16); }

static inline float ref_from_e4m3(uint8_t u) {
  unsigned magnitude = u & 0x7Fu, exponent = magnitude >> 3, mantissa = magnitude & 7u;
  float x = magnitude == 127 ? NAN : ldexpf((float)(mantissa + (exponent ? 8 : 0)), exponent ? (int)exponent - 10 : -9);
  return u & 0x80u ? -x : x;
}

// Search the ordered finite codebook. This deliberately uses a simple
// nearest-value reference rather than a runtime bit-rounding shortcut.
static inline uint8_t ref_e4m3(float x) {
  uint8_t sign = signbit(x) ? 0x80 : 0;
  if (isnan(x)) return sign | 0x7F;
  x = fabsf(x);
  unsigned lo = 0, hi = 126;
  while (lo < hi) {
    unsigned mid = (lo + hi + 1) / 2;
    if (ref_from_e4m3((uint8_t)mid) <= x) lo = mid;
    else hi = mid - 1;
  }
  if (lo == 126) return sign | (uint8_t)lo;
  double below = (double)x - ref_from_e4m3((uint8_t)lo);
  double above = ref_from_e4m3((uint8_t)(lo + 1)) - (double)x;
  if (above < below || (above == below && (lo & 1))) lo++;
  return sign | (uint8_t)lo;
}

// Matches the reference's per-32 E4M3 activation quantization: amax floor
// 1e-4, power-of-two scale rounded upward, and nearest-even value rounding.
// Reject nonfinite inputs without modifying the destination.
static inline int ref_fp8_pack(const float x[REF_ACT_BLOCK], RefFp8Block *out) {
  float amax = 1e-4f;
  for (int i = 0; i < REF_ACT_BLOCK; i++) {
    if (!isfinite(x[i])) return 0;
    if (fabsf(x[i]) > amax) amax = fabsf(x[i]);
  }
  uint32_t bits = ref_float_bits(amax * (1.0f / 448.0f));
  unsigned scale = (bits >> 23) + ((bits & 0x7FFFFFu) != 0);
  if (scale == 0 || scale >= 255) return 0;
  float s = ref_bits_float(scale << 23);
  RefFp8Block result = {.scale = (uint8_t)scale};
  for (int i = 0; i < REF_ACT_BLOCK; i++) result.values[i] = ref_e4m3(x[i] / s);
  *out = result;
  return 1;
}

static inline void ref_fp8_unpack(const RefFp8Block *in, float x[REF_ACT_BLOCK]) {
  float scale = ref_bits_float((uint32_t)in->scale << 23);
  for (int i = 0; i < REF_ACT_BLOCK; i++) x[i] = ref_from_e4m3(in->values[i]) * scale;
}

#endif
