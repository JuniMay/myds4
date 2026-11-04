// CPU properties of the proposed BF16/FP8 activation storage formats.
// No model, GPU, files or large arrays are needed.
#include "../reference/activation_format.h"
#include <stdio.h>
#include <unistd.h>

static int checks, failures;
static void expect(const char *name, int ok) {
  checks++;
  if (!ok) failures++, fprintf(stderr, "FAIL: %s\n", name);
}

int main(void) {
  alarm(10);
  int roundtrip = 1, nan = 1, midpoint = 1, neighbors = 1;
  for (unsigned u = 0; u <= UINT16_MAX; u++) {
    float x = ref_from_bf16((uint16_t)u);
    if (isnan(x)) nan &= isnan(ref_from_bf16(ref_bf16(x)));
    else roundtrip &= ref_bf16(x) == u;
  }
  for (unsigned u = 0; u < 0x7F7Fu; u++) {
    float a = ref_from_bf16((uint16_t)u), b = ref_from_bf16((uint16_t)(u + 1));
    float m = (float)(((double)a + b) / 2);
    unsigned even = u + (u & 1u);
    midpoint &= ref_bf16(m) == even && ref_bf16(-m) == (even | 0x8000u);
    float low = nextafterf(m, -INFINITY), high = nextafterf(m, INFINITY);
    neighbors &= ref_bf16(low) == u && ref_bf16(high) == u + 1;
    neighbors &= ref_bf16(-low) == (u | 0x8000u) && ref_bf16(-high) == ((u + 1) | 0x8000u);
  }
  expect("all finite BF16 values and infinities round-trip", roundtrip);
  expect("BF16 NaNs remain NaNs", nan);
  expect("BF16 halfway values use nearest-even rounding", midpoint);
  expect("both neighbors of every finite BF16 midpoint round correctly", neighbors);
  expect("FP32 NaNs with only low payload bits remain NaNs", isnan(ref_from_bf16(ref_bf16(ref_bits_float(0x7F800001u)))));

  roundtrip = midpoint = neighbors = 1;
  for (unsigned u = 0; u < 256; u++) {
    float x = ref_from_e4m3((uint8_t)u);
    if (!isnan(x)) roundtrip &= ref_e4m3(x) == u;
  }
  for (unsigned u = 0; u < 126; u++) {
    float a = ref_from_e4m3((uint8_t)u), b = ref_from_e4m3((uint8_t)(u + 1)), m = (a + b) / 2;
    unsigned even = u + (u & 1u);
    midpoint &= ref_e4m3(m) == even && ref_e4m3(-m) == (even | 0x80u);
    neighbors &= ref_e4m3(nextafterf(m, -INFINITY)) == u && ref_e4m3(nextafterf(m, INFINITY)) == u + 1;
  }
  expect("all finite E4M3 codes round-trip including signed zero", roundtrip);
  expect("E4M3 halfway values use nearest-even rounding", midpoint);
  expect("both neighbors of E4M3 midpoints round correctly", neighbors);
  expect("E4M3 known values match the format", ref_from_e4m3(0x01) == 0x1p-9f && ref_from_e4m3(0x38) == 1 && ref_from_e4m3(0x7E) == 448);
  expect("E4M3 finite overflow saturates", ref_e4m3(500) == 0x7E && ref_e4m3(-500) == 0xFE);

  float x[REF_ACT_BLOCK] = {0}, restored[REF_ACT_BLOCK];
  RefFp8Block packed;
  expect("FP8 block has no implicit padding", sizeof packed == 33);
  expect("zero block uses the reference scale floor", ref_fp8_pack(x, &packed) && packed.scale == 105);
  x[0] = -0.0f;
  expect("packed activations preserve signed zero", ref_fp8_pack(x, &packed) && packed.values[0] == 0x80);
  x[31] = 448;
  expect("scale one includes the exact maximum", ref_fp8_pack(x, &packed) && packed.scale == 127 && packed.values[31] == 126);
  x[31] = nextafterf(448, INFINITY);
  expect("the next scale is selected above its boundary", ref_fp8_pack(x, &packed) && packed.scale == 128);
  RefFp8Block before = packed;
  x[0] = NAN;
  expect("invalid blocks leave the destination unchanged", !ref_fp8_pack(x, &packed) && !memcmp(&before, &packed, sizeof packed));

  int compact = 1, bf16_exact = 1, scaled_half_exact = 1;
  for (int scale = 105; scale <= 247; scale++) {
    for (unsigned u = 0; u < 127; u++) {
      for (unsigned sign = 0; sign < 2; sign++) {
        float value = ref_from_e4m3((uint8_t)(u | sign * 128)) * ref_bits_float((uint32_t)scale << 23);
        if (!isfinite(value)) continue;
        bf16_exact &= ref_float_bits(ref_from_bf16(ref_bf16(value))) == ref_float_bits(value);
        if (fabsf(value) <= 511.75f)
          scaled_half_exact &= ref_float_bits((float)(_Float16)(value * 128.0f) * 0x1p-7f) == ref_float_bits(value);
      }
    }
  }
  for (unsigned u = 0; u < 127; u++) {
    memset(x, 0, sizeof x);
    x[0] = ref_from_e4m3((uint8_t)u), x[1] = -x[0], x[31] = 448;
    compact &= ref_fp8_pack(x, &packed);
    ref_fp8_unpack(&packed, restored);
    compact &= !memcmp(x, restored, sizeof x);
  }
  expect("packing preserves all scaled code values at scale one", compact);
  expect("BF16 holds every finite scaled E4M3 value in the activation scale range", bf16_exact);
  expect("scaled half preserves E4M3 values within the checked input bound", scaled_half_exact);
  float tiny = 0x1p-26f;
  expect("BF16 preserves the small activation fixture", ref_from_bf16(ref_bf16(tiny)) == tiny);
  expect("FP16 loses the same small activation", (float)(_Float16)tiny != tiny);
  fprintf(stderr, "activation formats: %d/%d CPU checks passed\n", checks - failures, checks);
  return failures != 0;
}
