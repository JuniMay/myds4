// Kernels for DeepSeek-V4.1-Flash. Residual streams are BF16 in memory;
// other values the reference keeps in BF16 are rounded where it rounds them, and
// its activation quantization (FP8 E4M3, FP4 E2M1 with block scales) is
// applied exactly, so the GPU computes what inference/model.py computes.
// The shared number formats (fp8, fp4, bf16x4) are in formats.metal.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

// ---- number formats ----

inline float bf(float x) {  // round to BF16, ties to even
  uint u = as_type<uint>(x);
  return as_type<float>((u + 0x7FFFu + ((u >> 16) & 1u)) & 0xFFFF0000u);
}
inline float pow2(int e) { return as_type<float>(uint(e + 127) << 23); }
inline int exponent(float a) { return int((as_type<uint>(a) >> 23) & 0xFFu) - 127; }
inline float e8m0(uchar s) { return as_type<float>(uint(s) << 23); }
inline float e4m3(float v) {  // round |v| <= 448 to E4M3
  int e = max(exponent(fabs(v)), -6);
  return copysign(rint(fabs(v) * pow2(3 - e)) * pow2(e - 3), v);
}
inline float e2m1(float v) {  // round |v| <= 6 to E2M1
  int e = max(exponent(fabs(v)), 0);
  return copysign(rint(fabs(v) * pow2(1 - e)) * pow2(e - 1), v);
}
inline float pow2_scale(float amax, float inv_max) {  // kernel.py's fast_round_scale
  uint b = as_type<uint>(amax * inv_max);
  return pow2(int((b >> 23) & 0xFFu) - 127 + ((b & 0x7FFFFFu) ? 1 : 0));
}
// FP8 activation quantization of one value; its block of 32 is the simdgroup.
inline float fp8_act(float v) {
  float s = pow2_scale(max(simd_max(fabs(v)), 1e-4f), 1.0f / 448.0f);
  return e4m3(clamp(v / s, -448.0f, 448.0f)) * s;
}
// FP4 quantization with a power-of-two scale per 32 (the indexer's).
inline float fp4_act(float v) {
  float s = pow2_scale(max(simd_max(fabs(v)), 6.0f * 0x1p-126f), 1.0f / 6.0f);
  return e2m1(clamp(v / s, -6.0f, 6.0f)) * s;
}
// Four E4M3 codes to floats.
inline float4 fp8x4(uint u) {
  uint lo = ((u & 0x7Fu) << 7) | ((u & 0x80u) << 8) | ((u & 0x7F00u) << 15) | ((u & 0x8000u) << 16);
  uint hi = ((u & 0x7F0000u) >> 9) | ((u & 0x800000u) >> 8) | ((u & 0x7F000000u) >> 1) | (u & 0x80000000u);
  return float4(float2(as_type<half2>(lo)), float2(as_type<half2>(hi)));  // times 2^-8
}
// Eight E2M1 codes, low nibble first, to floats.
inline half2 e2m1x2(uint p) {
  return as_type<half2>(((p & 0x00070007u) << 9) | ((p & 0x00080008u) << 12));
}
inline void fp4x8(uint u, thread float4 &a, thread float4 &b) {
  uint lo = u & 0x0F0F0F0Fu, hi = (u >> 4) & 0x0F0F0F0Fu;
  half2 h0 = e2m1x2((lo & 0xFFu) | ((hi & 0xFFu) << 16));
  half2 h1 = e2m1x2(((lo >> 8) & 0xFFu) | (((hi >> 8) & 0xFFu) << 16));
  half2 h2 = e2m1x2(((lo >> 16) & 0xFFu) | (((hi >> 16) & 0xFFu) << 16));
  half2 h3 = e2m1x2((lo >> 24) | ((hi >> 24) << 16));
  a = float4(float2(h0), float2(h1)) * 16384.0f;
  b = float4(float2(h2), float2(h3)) * 16384.0f;
}

// Sum over a threadgroup of up to 1024 threads; every thread gets the result.
inline float tg_sum(float v, threadgroup float *tmp, uint t, uint n) {
  v = simd_sum(v);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t % 32 == 0) tmp[t / 32] = v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float s = 0;
  for (uint i = 0; i < (n + 31) / 32; i++) s += tmp[i];
  return s;
}
inline float tg_max(float v, threadgroup float *tmp, uint t, uint n) {
  v = simd_max(v);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t % 32 == 0) tmp[t / 32] = v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float s = -INFINITY;
  for (uint i = 0; i < (n + 31) / 32; i++) s = max(s, tmp[i]);
  return s;
}

// ---- matrix-vector products: one simdgroup per output row ----

struct MV {
  uint n, k;            // rows, columns
  uint group_rows;      // rows that read the same input slice (wo_a's groups)
  uint group_k;         // input offset between groups
  uint round;           // round results to BF16
  uint split;           // simdgroups sharing a row quad, each taking every split-th chunk
};

// A simdgroup computes 4 consecutive rows over every split-th 16-byte chunk
// (lane-interleaved), so that each input chunk a lane loads serves four
// weight rows; the split partial sums meet in threadgroup memory.
// 8 simdgroups per threadgroup, i.e. 32 / split rows. The forms of N = 3
// (*_3) run up to 3 products of one k and split on the same input in one
// dispatch (grid y), each with its rows and output stride (buffer 11), the
// second's weights, scales and outputs at buffers 5 to 7, the third's at 8 to
// 10; a product with fewer threadgroups than the grid's width skips the rest.
inline void mv_store(float4 acc, threadgroup float4 *part, device float *y, constant MV &p, uint n,
                     uint row, uint sg, uint lane) {
  acc = float4(simd_sum(acc.x), simd_sum(acc.y), simd_sum(acc.z), simd_sum(acc.w));
  if (p.split > 1) {
    if (lane == 0) part[sg] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg % p.split) return;
    for (uint i = 1; i < p.split; i++) acc += part[sg + i];
  }
  if (lane < 4 && row < n) y[row + lane] = p.round ? bf(acc[lane]) : acc[lane];
}

// FP8 E4M3 weights with one UE8M0 scale per 32 x 32 block.
template <uint N>
kernel void mv_fp8(device const uchar *w [[buffer(0)]], device const uchar *s [[buffer(1)]],
                   device const float *x [[buffer(2)]], device float *y [[buffer(3)]],
                   constant MV &p [[buffer(4)]], device const uchar *w1 [[buffer(5)]],
                   device const uchar *s1 [[buffer(6)]], device float *y1 [[buffer(7)]],
                   device const uchar *w2 [[buffer(8)]], device const uchar *s2 [[buffer(9)]],
                   device float *y2 [[buffer(10)]], constant uint2 *rows [[buffer(11)]],
                   uint2 g [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                   uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float4 part[8];
  if (N > 1 && g.y) w = g.y == 1 ? w1 : w2, s = g.y == 1 ? s1 : s2, y = g.y == 1 ? y1 : y2;
  uint tg = g.x, n = N > 1 ? rows[g.y].x : p.n;
  uint row = (tg * (8 / p.split) + sg / p.split) * 4, chunks = p.k / 16;
  float4 acc = 0;
  if (row < n) {
    device const uint4 *wr = (device const uint4 *)(w + (ulong)row * p.k);
    device const uchar *sr = s + (row / 32) * (p.k / 32);
    device const float4 *xr = (device const float4 *)(x + (row / p.group_rows) * p.group_k);
    for (uint c = (sg % p.split) * 32 + lane; c < chunks; c += 32 * p.split) {
      uint4 q[4];  // the weights first: their loads go out before the inputs' and the scale's
      for (uint r = 0; r < 4; r++) q[r] = wr[r * chunks + c];
      float sc = e8m0(sr[c / 2]) * 256.0f;
      float4 x0 = xr[4 * c], x1 = xr[4 * c + 1], x2 = xr[4 * c + 2], x3 = xr[4 * c + 3];
      for (uint r = 0; r < 4; r++) {
        float4 d = fp8x4(q[r].x) * x0 + fp8x4(q[r].y) * x1 + fp8x4(q[r].z) * x2 + fp8x4(q[r].w) * x3;
        acc[r] += (d.x + d.y + d.z + d.w) * sc;
      }
    }
  }
  mv_store(acc, part, y, p, n, row, sg, lane);
}

// BF16 weights.
template <uint N>
kernel void mv_bf16(device const ushort *w [[buffer(0)]], device const float *x [[buffer(2)]],
                    device float *y [[buffer(3)]], constant MV &p [[buffer(4)]],
                    device const ushort *w1 [[buffer(5)]], device float *y1 [[buffer(7)]],
                    device const ushort *w2 [[buffer(8)]], device float *y2 [[buffer(10)]],
                    constant uint2 *rows [[buffer(11)]], uint2 g [[threadgroup_position_in_grid]],
                    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float4 part[8];
  if (N > 1 && g.y) w = g.y == 1 ? w1 : w2, y = g.y == 1 ? y1 : y2;
  uint tg = g.x, n = N > 1 ? rows[g.y].x : p.n;
  uint row = (tg * (8 / p.split) + sg / p.split) * 4, chunks = p.k / 8;
  float4 acc = 0;
  if (row < n) {
    device const uint4 *wr = (device const uint4 *)(w + (ulong)row * p.k);
    device const float4 *xr = (device const float4 *)x;
    for (uint c = (sg % p.split) * 32 + lane; c < chunks; c += 32 * p.split) {
      float4 x0 = xr[2 * c], x1 = xr[2 * c + 1];
      for (uint r = 0; r < 4; r++) {
        uint4 q = wr[r * chunks + c];
        float4 d = bf16x4(q.xy) * x0 + bf16x4(q.zw) * x1;
        acc[r] += d.x + d.y + d.z + d.w;
      }
    }
  }
  mv_store(acc, part, y, p, n, row, sg, lane);
}

// The same products for m of up to 8 input rows (stride lda floats; outputs
// at stride ldy): each weight chunk a lane loads serves every row, and each
// row's sums are those mv_fp8 and mv_bf16 make for it alone.
struct MVM { MV mv; uint m, lda, ldy; };

inline void mvm_store(thread float4 *acc, threadgroup float4 (*part)[8], device float *y, constant MVM &p,
                      uint n, uint ldy, uint row, uint sg, uint lane) {
  for (uint j = 0; j < p.m; j++) {
    acc[j] = float4(simd_sum(acc[j].x), simd_sum(acc[j].y), simd_sum(acc[j].z), simd_sum(acc[j].w));
    if (p.mv.split > 1 && lane == 0) part[j][sg] = acc[j];
  }
  if (p.mv.split > 1) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg % p.mv.split) return;
    for (uint j = 0; j < p.m; j++)
      for (uint i = 1; i < p.mv.split; i++) acc[j] += part[j][sg + i];
  }
  if (lane < 4 && row < n)
    for (uint j = 0; j < p.m; j++) y[j * ldy + row + lane] = p.mv.round ? bf(acc[j][lane]) : acc[j][lane];
}

template <uint N>
kernel void mvm_fp8(device const uchar *w [[buffer(0)]], device const uchar *s [[buffer(1)]],
                    device const float *x [[buffer(2)]], device float *y [[buffer(3)]],
                    constant MVM &p [[buffer(4)]], device const uchar *w1 [[buffer(5)]],
                    device const uchar *s1 [[buffer(6)]], device float *y1 [[buffer(7)]],
                    device const uchar *w2 [[buffer(8)]], device const uchar *s2 [[buffer(9)]],
                    device float *y2 [[buffer(10)]], constant uint2 *rows [[buffer(11)]],
                    uint2 g [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float4 part[8][8];
  if (N > 1 && g.y) w = g.y == 1 ? w1 : w2, s = g.y == 1 ? s1 : s2, y = g.y == 1 ? y1 : y2;
  uint tg = g.x, n = N > 1 ? rows[g.y].x : p.mv.n, ldy = N > 1 ? rows[g.y].y : p.ldy;
  uint row = (tg * (8 / p.mv.split) + sg / p.mv.split) * 4, chunks = p.mv.k / 16;
  float4 acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  if (row < n) {
    device const uint4 *wr = (device const uint4 *)(w + (ulong)row * p.mv.k);
    device const uchar *sr = s + (row / 32) * (p.mv.k / 32);
    x += (row / p.mv.group_rows) * p.mv.group_k;
    for (uint c = (sg % p.mv.split) * 32 + lane; c < chunks; c += 32 * p.mv.split) {
      float sc = e8m0(sr[c / 2]) * 256.0f;
      float4 a[4][4];
      for (uint r = 0; r < 4; r++) {
        uint4 q = wr[r * chunks + c];
        a[r][0] = fp8x4(q.x), a[r][1] = fp8x4(q.y), a[r][2] = fp8x4(q.z), a[r][3] = fp8x4(q.w);
      }
      for (uint j = 0; j < p.m; j++) {
        device const float4 *xr = (device const float4 *)(x + j * p.lda);
        float4 x0 = xr[4 * c], x1 = xr[4 * c + 1], x2 = xr[4 * c + 2], x3 = xr[4 * c + 3];
        for (uint r = 0; r < 4; r++) {
          float4 d = a[r][0] * x0 + a[r][1] * x1 + a[r][2] * x2 + a[r][3] * x3;
          acc[j][r] += (d.x + d.y + d.z + d.w) * sc;
        }
      }
    }
  }
  mvm_store(acc, part, y, p, n, ldy, row, sg, lane);
}

template <uint N>
kernel void mvm_bf16(device const ushort *w [[buffer(0)]], device const float *x [[buffer(2)]],
                     device float *y [[buffer(3)]], constant MVM &p [[buffer(4)]],
                     device const ushort *w1 [[buffer(5)]], device float *y1 [[buffer(7)]],
                     device const ushort *w2 [[buffer(8)]], device float *y2 [[buffer(10)]],
                     constant uint2 *rows [[buffer(11)]], uint2 g [[threadgroup_position_in_grid]],
                     uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float4 part[8][8];
  if (N > 1 && g.y) w = g.y == 1 ? w1 : w2, y = g.y == 1 ? y1 : y2;
  uint tg = g.x, n = N > 1 ? rows[g.y].x : p.mv.n, ldy = N > 1 ? rows[g.y].y : p.ldy;
  uint row = (tg * (8 / p.mv.split) + sg / p.mv.split) * 4, chunks = p.mv.k / 8;
  float4 acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  if (row < n) {
    device const uint4 *wr = (device const uint4 *)(w + (ulong)row * p.mv.k);
    for (uint c = (sg % p.mv.split) * 32 + lane; c < chunks; c += 32 * p.mv.split) {
      float4 a[4][2];
      for (uint r = 0; r < 4; r++) {
        uint4 q = wr[r * chunks + c];
        a[r][0] = bf16x4(q.xy), a[r][1] = bf16x4(q.zw);
      }
      for (uint j = 0; j < p.m; j++) {
        device const float4 *xr = (device const float4 *)(x + j * p.lda);
        float4 x0 = xr[2 * c], x1 = xr[2 * c + 1];
        for (uint r = 0; r < 4; r++) {
          float4 d = a[r][0] * x0 + a[r][1] * x1;
          acc[j][r] += d.x + d.y + d.z + d.w;
        }
      }
    }
  }
  mvm_store(acc, part, y, p, n, ldy, row, sg, lane);
}

#define MV_KERNELS(k)                                              \
  typedef decltype(k<1>) k##_t;                                    \
  template [[host_name(#k)]] kernel k##_t k<1>;                    \
  template [[host_name(#k "_3")]] kernel k##_t k<3>;
MV_KERNELS(mv_fp8)
MV_KERNELS(mv_bf16)
MV_KERNELS(mvm_fp8)
MV_KERNELS(mvm_bf16)

// ---- routed experts: MXFP4 codes and scales in cache slots ----

// p: an MXFP4 expert's codes (w1, w2, w3 back to back) and scales; or the
// GGUF rows of w1, w3 and w2 (see gguf.metal).
struct Expert { device const uchar *p[3]; float weight; float pad; };

// One 32-value chunk c of 4 FP4 rows (row stride `stride` bytes, scale
// stride stride / 16) against the chunk's inputs xv, added to acc.
inline void fp4_chunk4(device const uchar *w, device const uchar *s, uint stride, thread const float4 *xv,
                       uint c, thread float4 &acc) {
  for (uint r = 0; r < 4; r++) {
    uint4 q = ((device const uint4 *)(w + r * stride))[c];
    float4 a, b, d = 0;
    fp4x8(q.x, a, b); d += a * xv[0] + b * xv[1];
    fp4x8(q.y, a, b); d += a * xv[2] + b * xv[3];
    fp4x8(q.z, a, b); d += a * xv[4] + b * xv[5];
    fp4x8(q.w, a, b); d += a * xv[6] + b * xv[7];
    acc[r] += (d.x + d.y + d.z + d.w) * e8m0(s[r * (stride / 16) + c]);
  }
}

// Dot products of 4 FP4 rows with one input; lane-partial sums.
inline float4 fp4_dot4(device const uchar *w, device const uchar *s, uint stride,
                       device const float4 *x, uint chunks, uint lane) {
  float4 acc = 0;
  for (uint c = lane; c < chunks; c += 32) {
    float4 xv[8];
    for (uint i = 0; i < 8; i++) xv[i] = x[8 * c + i];
    fp4_chunk4(w, s, stride, xv, c, acc);
  }
  return acc;
}

// h[e][r] = weight_e * silu(min(w1 x, 10)) * clamp(w3 x, -10, 10) for expert
// e, quantized to FP8 per 32 rows: a threadgroup's rows. A simdgroup
// computes 4 rows of both w1 and w3, loading each input chunk once.
kernel void moe_up(constant Expert *ex [[buffer(0)]], device const float *x [[buffer(1)]],
                   device float *h [[buffer(2)]], uint2 tg [[threadgroup_position_in_grid]],
                   uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float block[32];
  if (!ex[tg.y].p[0]) return;  // a null entry: no expert
  uint row = (tg.x * 8 + sg) * 4, e = tg.y;
  constexpr uint K = 5120, W = K / 2 * 2304, S = K / 32 * 2304;
  device const uchar *w = ex[e].p[0] + row * (K / 2), *s = ex[e].p[1] + row * (K / 32);
  device const float4 *xv = (device const float4 *)x;
  float4 a = 0, b = 0;
  for (uint c = lane; c < K / 32; c += 32) {
    float4 xc[8];
    for (uint i = 0; i < 8; i++) xc[i] = xv[8 * c + i];
    fp4_chunk4(w, s, K / 2, xc, c, a);
    fp4_chunk4(w + 2 * W, s + 2 * S, K / 2, xc, c, b);
  }
  a = float4(simd_sum(a.x), simd_sum(a.y), simd_sum(a.z), simd_sum(a.w));
  b = float4(simd_sum(b.x), simd_sum(b.y), simd_sum(b.z), simd_sum(b.w));
  if (lane < 4) {
    float g = min(bf(a[lane]), 10.0f), u = clamp(bf(b[lane]), -10.0f, 10.0f);
    block[sg * 4 + lane] = bf(g / (1.0f + exp(-g)) * u * ex[e].weight);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) h[e * 2304 + tg.x * 32 + lane] = fp8_act(block[lane]);
}

// y[d] = bf16(sum_e bf16(w2_e h_e) + shared[d]), then the hyper-connection
// expansion of y into the four streams (hc_post). A threadgroup takes 4 rows,
// simdgroup e expert e; the experts' parts are summed in expert order. With
// pm, each expert with an entry writes its part to parts instead.
kernel void moe_down(constant Expert *ex [[buffer(0)]], device const float *h [[buffer(1)]],
                     device const float *shared_out [[buffer(2)]], device const float *mix [[buffer(3)]],
                     device const bfloat *res [[buffer(4)]], device bfloat *out [[buffer(5)]],
                     constant uint &n [[buffer(6)]], constant uint &pm [[buffer(7)]], device float *parts [[buffer(8)]],
                     uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float4 part[8];
  if (!pm && !ex[0].p[0]) return;  // null entries: no expert
  if (pm && !ex[sg].p[0]) return;
  uint d = tg * 4;
  constexpr uint K = 2304, W = 5120 * 2304 / 2, S = 5120 * 2304 / 32;
  float4 a = fp4_dot4(ex[sg].p[0] + W + d * (K / 2), ex[sg].p[1] + S + d * (K / 32), K / 2,
                      (device const float4 *)(h + sg * K), K / 32, lane);
  a = float4(bf(simd_sum(a.x)), bf(simd_sum(a.y)), bf(simd_sum(a.z)), bf(simd_sum(a.w)));
  if (pm) {
    if (lane == 0)
      for (uint r = 0; r < 4; r++) parts[(d + r) * 8 + sg] = a[r];
    return;
  }
  if (lane == 0) part[sg] = a;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0 && lane < 4) {
    uint i = d + lane;
    float y = 0;
    for (uint e = 0; e < n; e++) y += part[e][lane];
    float v0 = bf(y + shared_out[i]);
    for (uint k = 0; k < 4; k++) {
      float v = mix[4 + k] * v0;
      for (uint j = 0; j < 4; j++) v += mix[8 + 4 * j + k] * res[j * 5120 + i];
      out[k * 5120 + i] = bfloat(bf(v));
    }
  }
}

// A decoded row's routed experts from their parts (moe_down and gg_down with
// pm: row i's part of expert e at parts[8 i + e]), summed and expanded into
// the four streams exactly as those kernels do it.
kernel void moe_parts(device const float *parts [[buffer(0)]], device const float *shared_out [[buffer(1)]],
                      device const float *mix [[buffer(2)]], device const bfloat *res [[buffer(3)]],
                      device bfloat *out [[buffer(4)]], constant uint &n [[buffer(5)]],
                      uint i [[thread_position_in_grid]]) {
  float y = 0;
  for (uint e = 0; e < n; e++) y += parts[i * 8 + e];
  float v0 = bf(y + shared_out[i]);
  for (uint k = 0; k < 4; k++) {
    float v = mix[4 + k] * v0;
    for (uint j = 0; j < 4; j++) v += mix[8 + 4 * j + k] * res[j * 5120 + i];
    out[k * 5120 + i] = bfloat(bf(v));
  }
}

// ---- elementwise and normalization ----

// Rows of 512 values of positions [first, first + count) between buffers whose
// row for position p is (p - base) % mod: FP32 rows (the compressor's,
// copy_rows) or BF16 rows (keys, copy_keys).
struct Copy { uint first, count, src_base, src_mod, dst_base, dst_mod; };
template <typename T>
kernel void copy_rows(device const T *src [[buffer(0)]], device T *dst [[buffer(1)]],
                      constant Copy &c [[buffer(2)]], uint2 g [[threadgroup_position_in_grid]],
                      uint t [[thread_index_in_threadgroup]]) {
  uint p = c.first + g.y, i = g.x * 256 + t;
  dst[((p - c.dst_base) % c.dst_mod) * 512 + i] = src[((p - c.src_base) % c.src_mod) * 512 + i];
}
template [[host_name("copy_rows")]] kernel void copy_rows<float>(device const float *, device float *, constant Copy &, uint2, uint);
template [[host_name("copy_keys")]] kernel void copy_rows<ushort>(device const ushort *, device ushort *, constant Copy &, uint2, uint);

kernel void act_quant(device const float *x [[buffer(0)]], device float *y [[buffer(1)]],
                      uint i [[thread_position_in_grid]]) {
  y[i] = fp8_act(x[i]);
}

// Rows of the embedding table (BF16, staged per token) into the four streams.
kernel void embed(device const ushort *row [[buffer(0)]], device bfloat *h [[buffer(1)]],
                  uint2 i [[thread_position_in_grid]]) {
  float v = as_type<float>(uint(row[i.y * 5120 + i.x]) << 16);
  for (uint c = 0; c < 4; c++) h[i.y * 20480 + c * 5120 + i.x] = bfloat(v);
}

// Kernels that work on token rows take the row from the last grid dimension.

// RMSNorm of n values with weights, rounded to BF16; q gets its FP8
// activation quantization. One threadgroup of 1024 threads per row; n % 1024
// == 0 or n <= 1024 with n % 32 == 0.
kernel void rmsnorm(device const float *x [[buffer(0)]], device const float *w [[buffer(1)]],
                    device float *y [[buffer(2)]], device float *q [[buffer(3)]],
                    constant uint &n [[buffer(4)]], uint row [[threadgroup_position_in_grid]],
                    uint t [[thread_index_in_threadgroup]], uint nt [[threads_per_threadgroup]]) {
  threadgroup float tmp[32];
  x += row * n, y += row * n, q += row * n;
  float ss = 0;
  for (uint i = t; i < n; i += nt) ss += x[i] * x[i];
  float r = rsqrt(tg_sum(ss, tmp, t, nt) / n + 1e-20f);
  for (uint i = t; i < n; i += nt) {
    float v = bf(w[i] * (x[i] * r));
    y[i] = v;
    q[i] = fp8_act(v);
  }
}

// Hyper-connection mixes, part 1: 24 dot products with the flattened four
// streams and their sum of squares, split over 16 threadgroups of 256.
kernel void hc_mix(device const float *fn [[buffer(0)]], device const bfloat *x [[buffer(1)]],
                   device float *part [[buffer(2)]], uint2 g [[threadgroup_position_in_grid]],
                   uint t [[thread_index_in_threadgroup]]) {
  threadgroup float tmp[25][8];
  uint tg = g.x;
  x += g.y * 20480, part += g.y * 400;
  float acc[25] = {0};
  for (uint i = tg * 1280 + t; i < tg * 1280 + 1280; i += 256) {
    float v = x[i];
    acc[24] += v * v;
    for (uint j = 0; j < 24; j++) acc[j] += fn[j * 20480 + i] * v;
  }
  for (uint j = 0; j < 25; j++) {
    float v = simd_sum(acc[j]);
    if (t % 32 == 0) tmp[j][t / 32] = v;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t < 25) {
    float v = 0;
    for (uint i = 0; i < 8; i++) v += tmp[t][i];
    part[tg * 25 + t] = v;
  }
}

// Part 2: pre (4), post (4) and the Sinkhorn-normalized comb (4 x 4), in
// one simdgroup: lane j * 4 + k holds comb[j][k]. Row and column sums add in
// index order, as the reference does.
inline float row_sum(float c, uint t) {
  uint b = t & 12u;
  return simd_shuffle(c, b) + simd_shuffle(c, b + 1) + simd_shuffle(c, b + 2) + simd_shuffle(c, b + 3);
}
inline float col_sum(float c, uint t) {
  uint b = t & 3u;
  return simd_shuffle(c, b) + simd_shuffle(c, b + 4) + simd_shuffle(c, b + 8) + simd_shuffle(c, b + 12);
}
inline void hc_split(device const float *part, device const float *scale, device const float *base,
                     device float *mix, uint t) {
  float m = 0;
  if (t < 25)
    for (uint i = 0; i < 16; i++) m += part[i * 25 + t];
  float r = rsqrt(simd_shuffle(m, 24) / 20480 + 1e-20f), v = m * r;
  if (t < 4) mix[t] = 1.0f / (1.0f + exp(-(v * scale[0] + base[t]))) + 1e-6f;
  else if (t < 8) mix[t] = 2.0f / (1.0f + exp(-(v * scale[1] + base[t])));
  uint i = t & 15u;
  float c = simd_shuffle(v, 8 + i) * scale[2] + base[8 + i];
  uint b = t & 12u;
  float mx = max(max(simd_shuffle(c, b), simd_shuffle(c, b + 1)), max(simd_shuffle(c, b + 2), simd_shuffle(c, b + 3)));
  c = exp(c - mx);
  c = c / row_sum(c, t) + 1e-6f;
  c = c / (col_sum(c, t) + 1e-6f);
  for (uint it = 1; it < 20; it++) {
    c = c / (row_sum(c, t) + 1e-6f);
    c = c / (col_sum(c, t) + 1e-6f);
  }
  if (t < 16) mix[8 + t] = c;
}

// The residual part of a decoded row's FFN input, known before attention:
// with the attention's mix (pre a, post p, comb C), the FFN input is
// sum_k a_k (p_k A + sum_j C_jk x_j) for attention output A, so its residual
// part is sum_j (sum_k a_k C_jk) x_j. Writes it RMS-normalized times the
// FFN norm weight w. 1024 threads.
inline void residual(device const bfloat *x, device const float *mix, device const float *w, device float *y,
                     threadgroup float *tmp, uint t) {
  float c[4], v[5], ss = 0;
  for (uint j = 0; j < 4; j++)
    c[j] = mix[0] * mix[8 + 4 * j] + mix[1] * mix[9 + 4 * j] + mix[2] * mix[10 + 4 * j] + mix[3] * mix[11 + 4 * j];
  for (uint i = 0; i < 5; i++) {
    uint d = t + 1024 * i;
    v[i] = c[0] * x[d] + c[1] * x[5120 + d] + c[2] * x[10240 + d] + c[3] * x[15360 + d];
    ss += v[i] * v[i];
  }
  float r = rsqrt(tg_sum(ss, tmp, t, 1024) / 5120 + 1e-20f);
  for (uint i = 0; i < 5; i++) y[t + 1024 * i] = w[t + 1024 * i] * v[i] * r;
}

// hc_pre then RMSNorm: y = norm(bf16(sum_c pre[c] x[c])), q = its FP8 quantization.
// With a grid of (rows, 2), threadgroup (r, 1) runs part 2 for row r beside
// it: from part 1's sums at buffer 5, scale and base into the mixes at 8;
// then, if res, the residual part of the row's FFN input from x and these
// mixes, with the norm weights at 9, into buffer 10.
kernel void hc_pre_norm(device const bfloat *x [[buffer(0)]], device const float *mix [[buffer(1)]],
                        device const float *w [[buffer(2)]], device float *y [[buffer(3)]],
                        device float *q [[buffer(4)]], device const float *part [[buffer(5)]],
                        device const float *scale [[buffer(6)]], device const float *base [[buffer(7)]],
                        device float *mixes [[buffer(8)]], device const float *rw [[buffer(9)]],
                        device float *ry [[buffer(10)]], constant uint &res [[buffer(11)]],
                        uint2 g [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]]) {
  threadgroup float tmp[32];
  uint row = g.x;
  x += row * 20480;
  if (g.y) {
    if (t < 32) hc_split(part + row * 400, scale, base, mixes + row * 24, t);
    if (!res) return;
    threadgroup_barrier(mem_flags::mem_device);
    residual(x, mixes + row * 24, rw, ry + row * 5120, tmp, t);
    return;
  }
  mix += row * 24, y += row * 5120, q += row * 5120;
  float v[5], ss = 0;
  for (uint i = 0; i < 5; i++) {
    uint d = t + 1024 * i;
    v[i] = bf(mix[0] * x[d] + mix[1] * x[5120 + d] + mix[2] * x[10240 + d] + mix[3] * x[15360 + d]);
    ss += v[i] * v[i];
  }
  float r = rsqrt(tg_sum(ss, tmp, t, 1024) / 5120 + 1e-20f);
  for (uint i = 0; i < 5; i++) {
    uint d = t + 1024 * i;
    float o = bf(w[d] * (v[i] * r));
    y[d] = o;
    q[d] = fp8_act(o);
  }
}

// out[k] = bf16(post[k] y + sum_j comb[j][k] res[j]).
kernel void hc_post(device const float *y [[buffer(0)]], device const float *mix [[buffer(1)]],
                    device const bfloat *res [[buffer(2)]], device bfloat *out [[buffer(3)]],
                    uint2 i [[thread_position_in_grid]]) {
  uint d = i.x;
  y += i.y * 5120, mix += i.y * 24, res += i.y * 20480, out += i.y * 20480;
  for (uint k = 0; k < 4; k++) {
    float v = mix[4 + k] * y[d];
    for (uint j = 0; j < 4; j++) v += mix[8 + 4 * j + k] * res[j * 5120 + d];
    out[k * 5120 + d] = bfloat(bf(v));
  }
}

// Shared expert activation: h = bf16(silu(min(g, 10)) clamp(u, -10, 10)), FP8-quantized.
kernel void swiglu(device const float *g [[buffer(0)]], device const float *u [[buffer(1)]],
                   device float *h [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  float a = min(g[i], 10.0f), b = clamp(u[i], -10.0f, 10.0f);
  h[i] = fp8_act(bf(a / (1.0f + exp(-a)) * b));
}

// ---- attention ----

// Rotates the last 64 of n values (adjacent pairs as complex numbers) by the
// angles of one position; cs holds (cos, sin) pairs, 32 per position.
inline float rope1(float v, uint i, uint n, device const float2 *cs, bool inverse) {
  uint j = i - (n - 64);
  float other = simd_shuffle_xor(v, 1);
  if (i < n - 64) return v;
  float2 c = cs[j / 2];
  float s = inverse ? -c.y : c.y;
  return bf((j & 1) ? other * s + v * c.x : v * c.x - other * s);
}

// Window keys live in a buffer whose row for position p is (p - base) % mod:
// the 128-row ring in decoding, a row per position in prefill.
struct Win { uint pos, base, mod; };
inline uint win_row(uint p, constant Win &w) { return (p - w.base) % w.mod; }

// Window key: norm, RoPE and FP8 quantization, stored at its row as BF16,
// which holds the quantized value exactly. One threadgroup of 512 per token
// row.
kernel void kv_window(device const float *kv [[buffer(0)]], device const float *w [[buffer(1)]],
                      device const float2 *cs [[buffer(2)]], device bfloat *ring [[buffer(3)]],
                      constant Win &p [[buffer(4)]], uint row [[threadgroup_position_in_grid]],
                      uint t [[thread_index_in_threadgroup]]) {
  threadgroup float tmp[32];
  uint pos = p.pos + row;
  float v = kv[row * 512 + t];
  float r = rsqrt(tg_sum(v * v, tmp, t, 512) / 512 + 1e-20f);
  v = rope1(bf(w[t] * (v * r)), t, 512, cs + pos * 32, false);
  ring[win_row(pos, p) * 512 + t] = bfloat(fp8_act(v));
}

// Compressor: pools a group (ratio 2: softmax-gated over its two rows),
// normalizes, and writes the latent before RoPE (for the index keys) and the
// compressed key after RoPE and FP4 quantization (16 per E4M3 scale), as
// BF16: an E2M1 code times an E4M3 scale has at most 6 significant bits.
// Group g starts at position start + g ratio; its rows of kv and score are
// row0 + g ratio (and the next, for ratio 2). One threadgroup per group.
struct Groups { uint ratio, start, row0; };
kernel void compress(device const float *kv [[buffer(0)]], device const float *score [[buffer(1)]],
                     device const float *w [[buffer(2)]], device const float2 *cs [[buffer(3)]],
                     device float *latent [[buffer(4)]], device bfloat *rows [[buffer(5)]],
                     constant Groups &g [[buffer(6)]], uint grp [[threadgroup_position_in_grid]],
                     uint t [[thread_index_in_threadgroup]]) {
  threadgroup float tmp[32];
  uint2 p = uint2(g.ratio, g.start + grp * g.ratio);
  kv += (g.row0 + grp * g.ratio) * 512, score += (g.row0 + grp * g.ratio) * 512, latent += grp * 512;
  float v = kv[t];
  if (p.x == 2) {
    float a = score[t], b = score[512 + t], m = max(a, b);
    float ea = exp(a - m), eb = exp(b - m);
    v = bf((kv[t] * ea + kv[512 + t] * eb) / (ea + eb));
  }
  float r = rsqrt(tg_sum(v * v, tmp, t, 512) / 512 + 1e-20f);
  v = bf(w[t] * (v * r));
  latent[t] = v;
  v = rope1(v, t, 512, cs + p.y * 32, false);
  float amax = fabs(v);  // blocks of 16: half a simdgroup
  for (uint o = 1; o < 16; o <<= 1) amax = max(amax, simd_shuffle_xor(amax, o));
  float s = e4m3(max(amax, 6.0f * 0x1p-9f) / 6.0f);
  rows[(p.y / p.x) * 512 + t] = bfloat(e2m1(clamp(v / s, -6.0f, 6.0f)) * s);
}

// Index key: k_norm, RoPE and FP4 quantization (32 per power of two), as
// BF16. 128 threads.
kernel void index_key(device const float *k [[buffer(0)]], device const float *w [[buffer(1)]],
                      device const float2 *cs [[buffer(2)]], device bfloat *keys [[buffer(3)]],
                      constant Groups &g [[buffer(4)]], uint grp [[threadgroup_position_in_grid]],
                      uint t [[thread_index_in_threadgroup]]) {
  threadgroup float tmp[32];
  uint ratio = g.ratio, pos = g.start + grp * ratio;
  float v = k[grp * 128 + t];
  float r = rsqrt(tg_sum(v * v, tmp, t, 128) / 128 + 1e-20f);
  v = rope1(bf(w[t] * (v * r)), t, 128, cs + pos * 32, false);
  keys[(pos / ratio) * 128 + t] = bfloat(fp4_act(v));
}

// Index scores of one query over n compressed positions:
// score[t] = sum_h relu(q_h . k_t) w_h, with q rotated and FP4-quantized here.
// Query row r is at position pos + r and sees (pos + r + 1) / ratio rows.
struct Index { uint pos, ratio, stride, cand; };  // cand: words of a row's candidate bits, 0: none
kernel void index_score(device const float *q [[buffer(0)]], device const float *wts [[buffer(1)]],
                        device const bfloat *keys [[buffer(2)]], device const float2 *cs [[buffer(3)]],
                        device float *score [[buffer(4)]], constant Index &ix [[buffer(5)]],
                        uint2 g [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]]) {
  threadgroup float4 qs[32 * 32];
  uint row = g.y, gid = g.x * 256 + t;
  uint2 p = uint2((ix.pos + row + 1) / ix.ratio, ix.pos + row);
  q += row * 4096, wts += row * 32, score += row * ix.stride;
  threadgroup float ws[32];
  threadgroup float *qf = (threadgroup float *)qs;
  for (uint i = t; i < 4096; i += 256)  // each run of 32 is a simdgroup: one quantization block
    qf[i] = fp4_act(rope1(q[i], i % 128, 128, cs + p.y * 32, false));
  if (t < 32) ws[t] = bf(wts[t] * 0.015625f);  // softmax scale 128^-0.5 times 32^-0.5
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (gid >= p.x) return;
  device const bfloat4 *kr = (device const bfloat4 *)(keys + gid * 128);
  float acc[32];
  for (uint h = 0; h < 32; h++) acc[h] = 0;
  for (uint d = 0; d < 32; d++) {
    float4 k = float4(kr[d]);
    for (uint h = 0; h < 32; h++) acc[h] += dot(qs[h * 32 + d], k);
  }
  float total = 0;
  for (uint h = 0; h < 32; h++) total += max(acc[h], 0.0f) * ws[h];
  score[gid] = total;
}

// The top 512 of n scores in position order (radix select on order keys);
// returns the count in sel[-1]. One threadgroup of 1024.
inline uint order_key(float f) {
  uint u = as_type<uint>(f);
  return (u & 0x80000000u) ? ~u : u | 0x80000000u;
}

// The order key of the k-th largest of n keys, 8 bits at a time; `want`
// ends as the number of keys equal to it among the k largest.
template <typename Keys>
inline uint kth_key(Keys keys, uint n, uint k, uint t, threadgroup atomic_uint *hist, threadgroup uint &prefix,
                    threadgroup uint &want) {
  uint thr = 0, mask = 0;
  if (t == 0) want = k;
  for (int shift = 24; shift >= 0; shift -= 8) {
    if (t < 256) atomic_store_explicit(&hist[t], 0, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = t; i < n; i += 1024) {
      uint key = keys(i);
      if ((key & mask) == thr) atomic_fetch_add_explicit(&hist[(key >> shift) & 255], 1, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0) {
      uint acc = 0, b = 255;
      for (;; b--) {
        uint c = atomic_load_explicit(&hist[b], memory_order_relaxed);
        if (acc + c >= want || b == 0) break;
        acc += c;
      }
      want -= acc;
      prefix = b;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    thr |= prefix << shift;
    mask |= 255u << shift;
  }
  return thr;
}

// A row's scores as order keys; with candidate blocks (layers after 20 beyond
// 2048 blocks), the positions outside them at -inf. Bit b of `cand` stands
// for positions 8b .. 8b + 7.
struct ScoreKeys {
  device const float *score;
  device const uint *cand;
  uint operator()(uint i) const {
    return order_key(!cand || ((cand[i >> 8] >> ((i >> 3) & 31)) & 1) ? score[i] : -INFINITY);
  }
};

// sel holds rows of 513: the count, then the indices.
kernel void index_topk(device const float *score [[buffer(0)]], device int *sel [[buffer(1)]],
                       constant Index &ix [[buffer(2)]], device const uint *cand [[buffer(3)]],
                       uint row [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]]) {
  threadgroup atomic_uint hist[256];
  threadgroup uint prefix, want, pa[1024], pe[1024];
  ScoreKeys keys = {score + row * ix.stride, ix.cand ? cand + row * ix.cand : nullptr};
  sel += row * 513 + 1;
  uint n = (ix.pos + row + 1) / ix.ratio, k = min(n, 512u);
  uint thr = n > 512 ? kth_key(keys, n, 512, t, hist, prefix, want) : 0;  // keys >= thr are selected
  // Keys above thr are all taken; of the keys equal to thr, the first `want`.
  uint per = (n + 1023) / 1024, begin = t * per, end = min(begin + per, n), above = 0, equal = 0;
  for (uint i = begin; i < end; i++) {
    uint key = keys(i);
    above += key > thr || n <= 512;
    equal += key == thr && n > 512;
  }
  // exclusive prefix sums over threads, in thread (= position) order
  pa[t] = above;
  pe[t] = equal;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint o = 1; o < 1024; o <<= 1) {
    uint a = t >= o ? pa[t - o] : 0, e = t >= o ? pe[t - o] : 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    pa[t] += a;
    pe[t] += e;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  uint total_above = pa[1023], ties = n > 512 ? k - total_above : 0;
  uint ra = pa[t] - above, re = pe[t] - equal;
  // output order by position: interleave above/equal keys as they occur
  uint out = ra + min(re, ties);
  for (uint i = begin; i < end; i++) {
    uint key = keys(i);
    bool take = n <= 512 || key > thr || (key == thr && re++ < ties);
    if (take) sel[out++] = int(i);
  }
  if (t == 0) sel[-1] = int(k);
}

// Layer 20's candidate blocks of a row beyond 2048 blocks (its indexer's
// first level, inference/model.py select_candidate_blocks): blocks of 8
// positions scored by their best one, the block of the newest position
// always in, the 2048 best kept (with any tied with the 2048th), as the bits
// of `cand`. One threadgroup of 1024.
struct BlockKeys {
  device const float *score;
  uint n, last;
  uint operator()(uint b) const {
    if (b == last) return order_key(INFINITY);
    float m = score[b * 8];
    for (uint i = b * 8 + 1; i < min(b * 8 + 8, n); i++) m = max(m, score[i]);
    return order_key(m);
  }
};

kernel void cand_topk(device const float *score [[buffer(0)]], device uint *cand [[buffer(1)]],
                      constant Index &ix [[buffer(2)]], uint row [[threadgroup_position_in_grid]],
                      uint t [[thread_index_in_threadgroup]]) {
  threadgroup atomic_uint hist[256];
  threadgroup uint prefix, want;
  uint n = (ix.pos + row + 1) / ix.ratio, nb = (n + 7) / 8;
  BlockKeys keys = {score + row * ix.stride, n, nb - 1};
  uint thr = nb > 2048 ? kth_key(keys, nb, 2048, t, hist, prefix, want) : 0;
  cand += row * ix.cand;
  for (uint w = t; w < ix.cand; w += 1024) {
    uint bits = 0;
    for (uint j = 0; j < 32; j++) bits |= uint(w * 32 + j < nb && keys(w * 32 + j) >= thr) << j;
    cand[w] = bits;
  }
}

// Sparse attention of 64 heads over the window ring and selected compressed
// rows, with each head's sink in the denominator. The heads share their keys
// (which are also the values), so a threadgroup takes several heads and
// reads each row once for them.
// Query row r at position pos + r attends to the window rows of positions
// pos + r - 127 .. pos + r (from position `first` on), and to its selected
// compressed rows if `comp_rows` is set.
struct Att { Win win; uint first, comp_rows; };

// Decoding attention of one row over the rows it sees: the window, then the
// selected compressed rows, up to 640, in chunks of 64 and three passes: the
// scores and each chunk's maximum per head; the probabilities against the
// row's maximum (summed as they are, BF16 for the values, as the reference
// rounds them) times the values, per chunk; and the chunks' sums. The keys
// are also the values. Loads go to registers ahead of their use (the values'
// rows 8 at a time, the maxima and parts all at once): a load used as soon as
// it is issued costs a whole memory latency.
#define ATT_CHUNK 64
#define ATT_ROWS 640  // the window and the selected compressed rows, at most
inline uint att_rows(constant Att &at, device const int *sel, thread uint &lo, thread uint &nwin) {
  uint pos = at.win.pos;
  lo = max(pos >= 127 ? pos - 127 : 0u, at.first);  // the first window position
  nwin = pos + 1 - lo;
  return nwin + (at.comp_rows ? uint(sel[0]) : 0);
}
// Row r's index in its buffer: the ring for the window, comp for the rest.
inline uint att_index(uint r, uint lo, uint nwin, constant Att &at, device const int *sel) {
  return r < nwin ? win_row(lo + r, at.win) : uint(sel[1 + r - nwin]);
}

// Scores: a threadgroup takes 4 heads and two chunks; each of its 16
// simdgroups takes two rows at a time, which share the loads of the queries.
kernel void attend_scores(device const float *q [[buffer(0)]], device const bfloat *ring [[buffer(1)]],
                          device const bfloat *comp [[buffer(2)]], device const int *sel [[buffer(3)]],
                          device const float2 *cs [[buffer(4)]], device float *sc [[buffer(5)]],
                          device float *cmax [[buffer(6)]], constant Att &at [[buffer(7)]],
                          uint2 gr [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float4 qs[4][128];
  threadgroup float s[4][2 * ATT_CHUNK], tmp[16];
  threadgroup uint rows[2 * ATT_CHUNK];
  uint g = gr.x, pos = at.win.pos, lo, nwin;
  uint n = att_rows(at, sel, lo, nwin), r0 = gr.y * 2 * ATT_CHUNK, r1 = min(n, r0 + 2 * ATT_CHUNK);
  if (r0 >= n) return;
  if (t < r1 - r0) rows[t] = att_index(r0 + t, lo, nwin, at, sel);
  for (uint i = t; i < 2048; i += 512) {
    uint h = i / 512, d = i % 512;
    ((threadgroup float *)qs)[i] = rope1(q[(g * 4 + h) * 512 + d], d, 512, cs + pos * 32, false);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint r = r0 + 2 * sg; r < r1; r += 32) {  // rows r and r + 1 (r again, not stored, past r1)
    uint r2 = min(r + 1, r1 - 1);
    device const bfloat4 *k0 = (device const bfloat4 *)((r < nwin ? ring : comp) + rows[r - r0] * 512);
    device const bfloat4 *k1 = (device const bfloat4 *)((r2 < nwin ? ring : comp) + rows[r2 - r0] * 512);
    float4 d0 = 0, d1 = 0;
    for (uint i = lane; i < 128; i += 32) {
      float4 q0 = qs[0][i], q1 = qs[1][i], q2 = qs[2][i], q3 = qs[3][i], a = float4(k0[i]), b = float4(k1[i]);
      d0 += float4(dot(q0, a), dot(q1, a), dot(q2, a), dot(q3, a));
      d1 += float4(dot(q0, b), dot(q1, b), dot(q2, b), dot(q3, b));
    }
    d0 = float4(simd_sum(d0.x), simd_sum(d0.y), simd_sum(d0.z), simd_sum(d0.w)) * 0.044194173824159216f;
    d1 = float4(simd_sum(d1.x), simd_sum(d1.y), simd_sum(d1.z), simd_sum(d1.w)) * 0.044194173824159216f;
    if (lane < 4) s[lane][r - r0] = d0[lane], sc[(g * 4 + lane) * ATT_ROWS + r] = d0[lane];
    if (lane < 4 && r + 1 < r1) s[lane][r + 1 - r0] = d1[lane], sc[(g * 4 + lane) * ATT_ROWS + r + 1] = d1[lane];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint k = t / 256, h = t / 64 % 4, r = k * ATT_CHUNK + t % 64;  // chunk k, head h: simdgroups 8k + 2h, 8k + 2h + 1
  float v = simd_max(r0 + r < r1 ? s[h][r] : -INFINITY);
  if (lane == 0) tmp[sg] = v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t < 8 && r0 + t / 4 * ATT_CHUNK < r1)
    cmax[(gr.y * 2 + t / 4) * 64 + g * 4 + t % 4] = max(tmp[t / 4 * 8 + t % 4 * 2], tmp[t / 4 * 8 + t % 4 * 2 + 1]);
}

// Adds rows j .. j1 - 1, of one buffer, of the chunk from row r0, times their
// probabilities p (4 heads per float4), to the sums acc of 4 columns of the
// heads of hq, row after row, loading 8 rows at a time.
inline void att_values(device const bfloat *base, threadgroup const uint *rows, uint j, uint j1, uint r0,
                       threadgroup const float4 (*p)[4], uint hq, uint col, thread float4 *acc) {
  for (; j + 8 <= j1; j += 8) {
    float4 x[8];
    for (uint u = 0; u < 8; u++) x[u] = float4(*(device const bfloat4 *)(base + rows[j + u - r0] * 512 + col));
    for (uint u = 0; u < 8; u++) {
      float4 pv = p[j + u - r0][hq];
      acc[0] += pv.x * x[u], acc[1] += pv.y * x[u], acc[2] += pv.z * x[u], acc[3] += pv.w * x[u];
    }
  }
  for (; j < j1; j++) {
    float4 x = float4(*(device const bfloat4 *)(base + rows[j - r0] * 512 + col)), pv = p[j - r0][hq];
    acc[0] += pv.x * x, acc[1] += pv.y * x, acc[2] += pv.z * x, acc[3] += pv.w * x;
  }
}

// A chunk's part per head: 512 sums of probability times value, then the sum
// of its probabilities. A threadgroup takes 16 heads, two chunks and half of
// the columns; thread t takes chunk t / 256 and 4 columns of 4 heads.
kernel void attend_values(device const bfloat *ring [[buffer(0)]], device const bfloat *comp [[buffer(1)]],
                          device const int *sel [[buffer(2)]], device const float *sc [[buffer(3)]],
                          device const float *cmax [[buffer(4)]], device float *part [[buffer(5)]],
                          constant Att &at [[buffer(6)]],
                          uint2 gr [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float4 p[2][ATT_CHUNK][4];  // [chunk][row][head / 4]
  threadgroup float tmp[2][32], mxs[16];
  threadgroup uint rows[2 * ATT_CHUNK];
  uint g = gr.x / 2, lo, nwin, n = att_rows(at, sel, lo, nwin), nc = (n + ATT_CHUNK - 1) / ATT_CHUNK;
  uint rc = gr.y * 2 * ATT_CHUNK;  // the first row of the two chunks
  if (rc >= n) return;
  if (t < min(n - rc, 2u * ATT_CHUNK)) rows[t] = att_index(rc + t, lo, nwin, at, sel);
  if (t < 16) {  // head g 16 + t's maximum over the chunks
    float m[ATT_ROWS / ATT_CHUNK], mx = -INFINITY;
    for (uint k = 0; k < ATT_ROWS / ATT_CHUNK; k++) m[k] = cmax[min(k, nc - 1) * 64 + g * 16 + t];
    for (uint k = 0; k < nc; k++) mx = max(mx, m[k]);
    mxs[t] = mx;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = sg; i < 64; i += 16) {  // chunk i / 32, head i % 32 / 2, rows i % 2 * 32 + lane
    uint c = i / 32, h = i % 32 / 2, r = i % 2 * 32 + lane, row = rc + c * ATT_CHUNK + r;
    float e = 0;
    if (row < n) e = exp(sc[(g * 16 + h) * ATT_ROWS + row] - mxs[h]);
    ((threadgroup float *)p[c])[r * 16 + h] = bf(e);
    float v = simd_sum(e);
    if (lane == 0) tmp[c][i % 32] = v;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint c = t / 256, hq = t % 256 / 64, col = gr.x % 2 * 256 + t % 64 * 4;
  uint r0 = rc + c * ATT_CHUNK, r1 = min(n, r0 + ATT_CHUNK);
  if (r0 >= n) return;
  float4 acc[4] = {0, 0, 0, 0};
  att_values(ring, rows + c * ATT_CHUNK, r0, min(r1, nwin), r0, p[c], hq, col, acc);
  att_values(comp, rows + c * ATT_CHUNK, max(r0, nwin), r1, r0, p[c], hq, col, acc);
  device float *out = part + ((gr.y * 2 + c) * 64 + g * 16 + hq * 4) * 514 + col;
  for (uint k = 0; k < 4; k++)
    *(device float2 *)(out + k * 514) = acc[k].xy, *(device float2 *)(out + k * 514 + 2) = acc[k].zw;
  if (gr.x % 2 == 0 && t % 256 < 16)
    part[((gr.y * 2 + c) * 64 + g * 16 + t % 256) * 514 + 512] = tmp[c][t % 256 * 2] + tmp[c][t % 256 * 2 + 1];
}

// The chunks' sums: a threadgroup takes a head and half of the columns,
// thread t two of them.
kernel void attend_combine(device const float *part [[buffer(0)]], device const float *cmax [[buffer(1)]],
                           device const float *sink [[buffer(2)]], device float *o [[buffer(3)]],
                           device const int *sel [[buffer(4)]], constant Att &at [[buffer(5)]],
                           uint2 gr [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]]) {
  threadgroup float den;
  uint h = gr.x, col = gr.y * 256 + 2 * t, lo, nwin, n = att_rows(at, sel, lo, nwin);
  uint nc = (n + ATT_CHUNK - 1) / ATT_CHUNK;
  float2 v[ATT_ROWS / ATT_CHUNK], acc = 0;
  for (uint k = 0; k < ATT_ROWS / ATT_CHUNK; k++)
    v[k] = *(device const float2 *)(part + (min(k, nc - 1) * 64 + h) * 514 + col);
  if (t == 0) {
    float m[ATT_ROWS / ATT_CHUNK], s[ATT_ROWS / ATT_CHUNK], mx = -INFINITY, sum = 0;
    for (uint k = 0; k < ATT_ROWS / ATT_CHUNK; k++) {
      uint i = min(k, nc - 1) * 64 + h;
      m[k] = cmax[i], s[k] = part[i * 514 + 512];
    }
    for (uint k = 0; k < nc; k++) mx = max(mx, m[k]), sum += s[k];
    den = sum + exp(sink[h] - mx);
  }
  for (uint k = 0; k < nc; k++) acc += v[k];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  o[h * 512 + col] = bf(acc.x / den), o[h * 512 + col + 1] = bf(acc.y / den);
}

// Attention of many query rows (prefill) on the matrix units: a threadgroup
// of 256 threads takes one row and 32 heads and runs over the row's rows of
// keys in chunks of 32 with an online softmax. The matrix units read the
// queries (with RoPE, as BF16, as the reference holds them: q_rope) from
// device memory; the 512 columns of a chunk's keys go through threadgroup
// memory in slices of 128, each loaded into registers while the products
// of the slice before run. The keys are also the values.
kernel void q_rope(device const float *q [[buffer(0)]], device const float2 *cs [[buffer(1)]],
                   device bfloat *qb [[buffer(2)]], constant uint &pos [[buffer(3)]],
                   uint2 i [[thread_position_in_grid]]) {
  qb[i.y * 32768 + i.x] = bfloat(rope1(q[i.y * 32768 + i.x], i.x % 512, 512, cs + (pos + i.y) * 32, false));
}
struct Keys { bfloat4 v[4]; };  // a thread's share of a chunk's slice: rows t / 32 + 8 i, columns 4 (t % 32)..
inline Keys keys_load(device const bfloat *ring, device const bfloat *comp, threadgroup const uint *rows, uint r0,
                      uint n, uint ds, uint t) {
  Keys c;
  for (uint i = 0; i < 4; i++) {
    uint r = r0 + t / 32 + 8 * i, row = r < n ? rows[r] : 0;
    c.v[i] = r < n ? *(device const bfloat4 *)((row < 0x80000000u ? ring + row * 512 : comp + (row & 0x7FFFFFFFu) * 512) + ds * 128 + (t % 32) * 4) : bfloat4(0);
  }
  return c;
}
inline void keys_store(threadgroup bfloat *kv, Keys c, uint t) {
  for (uint i = 0; i < 4; i++) *(threadgroup bfloat4 *)(kv + (t / 32 + 8 * i) * 128 + (t % 32) * 4) = c.v[i];
}
kernel void attend_m(device const bfloat *qb [[buffer(0)]], device const bfloat *ring [[buffer(1)]],
                     device const bfloat *comp [[buffer(2)]], device const int *sel [[buffer(3)]],
                     device const float *sink [[buffer(4)]], device float *o [[buffer(5)]],
                     constant Att &at [[buffer(6)]], uint2 gr [[threadgroup_position_in_grid]],
                     uint t [[thread_index_in_threadgroup]]) {
  threadgroup bfloat kv[32 * 128], ps[32 * 32];
  threadgroup float ss[32 * 32], mx[32], sum[32], rescale[32];
  threadgroup uint rows[640];
  uint h0 = gr.x * 32, pos = at.win.pos + gr.y;
  o += gr.y * 32768, sel += gr.y * 513 + 1;
  uint lo = max(pos >= 127 ? pos - 127 : 0u, at.first);
  uint nwin = pos + 1 - lo, ncomp = at.comp_rows ? uint(sel[-1]) : 0, n = nwin + ncomp;
  for (uint r = t; r < n; r += 256) rows[r] = r < nwin ? win_row(lo + r, at.win) : 0x80000000u | uint(sel[r - nwin]);
  if (t < 32) mx[t] = -1e30f, sum[t] = 0;
  constexpr auto sdesc = matmul2d_descriptor(32, 32, 128, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  constexpr auto odesc = matmul2d_descriptor(32, 128, 32, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<sdesc, execution_simdgroups<8>> sop;
  matmul2d<odesc, execution_simdgroups<8>> oop;
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tq((device bfloat *)qb + gr.y * 32768 + h0 * 512, dextents<int32_t, 2>(512, 32));
  tensor<threadgroup bfloat, extents<int32_t, 128, 32>, tensor_inline> tk(kv, extents<int32_t, 128, 32>());
  tensor<threadgroup bfloat, extents<int32_t, 32, 32>, tensor_inline> tp(ps, extents<int32_t, 32, 32>());
  auto acc0 = oop.get_destination_cooperative_tensor<decltype(tp), decltype(tk), float>();
  auto acc1 = oop.get_destination_cooperative_tensor<decltype(tp), decltype(tk), float>();
  auto acc2 = oop.get_destination_cooperative_tensor<decltype(tp), decltype(tk), float>();
  auto acc3 = oop.get_destination_cooperative_tensor<decltype(tp), decltype(tk), float>();
  for (uint16_t i = 0; i < acc0.get_capacity(); i++)
    if (acc0.is_valid_element(i)) acc0[i] = acc1[i] = acc2[i] = acc3[i] = 0;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  Keys kc = keys_load(ring, comp, rows, 0, n, 0, t);
  for (uint r0 = 0; r0 < n; r0 += 32) {
    auto sc = sop.get_destination_cooperative_tensor<decltype(tq.slice(0, 0)), decltype(tk), float>();
    for (uint16_t i = 0; i < sc.get_capacity(); i++)
      if (sc.is_valid_element(i)) sc[i] = 0;
    for (uint ds = 0; ds < 4; ds++) {  // scores: the chunk's keys, a slice of 128 columns at a time
      keys_store(kv, kc, t);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      kc = keys_load(ring, comp, rows, r0, n, (ds + 1) % 4, t);  // after the last, the values' first
      auto qd = tq.slice(ds * 128, 0);
      sop.run(qd, tk, sc);
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint16_t i = 0; i < sc.get_capacity(); i++)
      if (sc.is_valid_element(i)) {
        auto a = sc.get_multidimensional_index(i);
        ss[a[1] * 32 + a[0]] = r0 + a[0] < n ? sc[i] * 0.044194173824159216f : -INFINITY;
      }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t < 32) {  // the online softmax of head t over the chunk
      float m = mx[t];
      for (uint j = 0; j < 32; j++) m = max(m, ss[t * 32 + j]);
      float f = exp(mx[t] - m), s2 = sum[t] * f;
      for (uint j = 0; j < 32; j++) {
        float e = exp(ss[t * 32 + j] - m);
        s2 += e;
        ps[t * 32 + j] = bfloat(e);
      }
      mx[t] = m, sum[t] = s2, rescale[t] = f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint16_t i = 0; i < acc0.get_capacity(); i++)
      if (acc0.is_valid_element(i)) {
        float f = rescale[acc0.get_multidimensional_index(i)[1]];
        acc0[i] *= f, acc1[i] *= f, acc2[i] *= f, acc3[i] *= f;
      }
    for (uint ds = 0; ds < 4; ds++) {  // values: the same rows, a slice of 128 columns at a time
      keys_store(kv, kc, t);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (ds < 3) kc = keys_load(ring, comp, rows, r0, n, ds + 1, t);
      else if (r0 + 32 < n) kc = keys_load(ring, comp, rows, r0 + 32, n, 0, t);
      if (ds == 0) oop.run(tp, tk, acc0);
      else if (ds == 1) oop.run(tp, tk, acc1);
      else if (ds == 2) oop.run(tp, tk, acc2);
      else oop.run(tp, tk, acc3);
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
  for (uint16_t i = 0; i < acc0.get_capacity(); i++)
    if (acc0.is_valid_element(i)) {
      auto a = acc0.get_multidimensional_index(i);
      uint h = a[1], d = a[0];
      float den = sum[h] + exp(sink[h0 + h] - mx[h]);
      device float *out = o + (h0 + h) * 512 + d;
      out[0] = bf(acc0[i] / den), out[128] = bf(acc1[i] / den), out[256] = bf(acc2[i] / den), out[384] = bf(acc3[i] / den);
    }
}

// Attention output: inverse RoPE per head, then FP8 quantization.
kernel void attn_out(device const float *o [[buffer(0)]], device const float2 *cs [[buffer(1)]],
                     device float *q [[buffer(2)]], constant uint &pos [[buffer(3)]],
                     uint i [[thread_position_in_grid]]) {
  float v = rope1(o[i], i % 512, 512, cs + (pos + i / 32768) * 32, true);
  q[i] = fp8_act(v);
}

// ---- routing ----

// sqrt(softplus) scores over n experts; the bias selects the top k, and if
// they are weighted, the unbiased scores weight them (normalized, times 1.5).
// One threadgroup of n threads (n % 32 == 0, n <= 384; k <= 32, weighted
// k <= 8) per row of a routing, up to 3 routings of the same rows in one
// dispatch (grid y): routing j's logits, bias and indices at buffers 3 j to
// 3 j + 2. Row r reads its n logits at logits + r n and writes its indices
// at idx + r stride and weights at idx + r stride + 8. A prediction writes
// the k best indices only. Simdgroup 0 selects: lane i holds the keys of
// experts i, i + 32, ..., and each step takes the largest key left, the
// lowest expert on ties.
struct Route { uint k, stride, n, weighted; };
kernel void route(device const float *lg0 [[buffer(0)]], device const float *bias0 [[buffer(1)]],
                  device int *idx0 [[buffer(2)]], device const float *lg1 [[buffer(3)]],
                  device const float *bias1 [[buffer(4)]], device int *idx1 [[buffer(5)]],
                  device const float *lg2 [[buffer(6)]], device const float *bias2 [[buffer(7)]],
                  device int *idx2 [[buffer(8)]], constant Route *rps [[buffer(9)]],
                  device const float *vl0 [[buffer(10)]], device const float *vl1 [[buffer(11)]],
                  device const float *vl2 [[buffer(12)]], device const uchar *img [[buffer(13)]],
                  uint2 g [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]]) {
  threadgroup float key[384], score[384];
  constant Route &rp = rps[g.y];
  device const float *logits = g.y == 0 ? lg0 : g.y == 1 ? lg1 : lg2, *bias = g.y == 0 ? bias0 : g.y == 1 ? bias1 : bias2;
  if (img[g.x]) bias = g.y == 0 ? vl0 : g.y == 1 ? vl1 : vl2;  // a row inside an image span: bias_vl
  device int *idx = (g.y == 0 ? idx0 : g.y == 1 ? idx1 : idx2) + g.x * rp.stride;
  logits += g.x * rp.n;
  float l = logits[t], u = exp(l), v = 1.0f + u;  // softplus; log1p(u) by Goldberg's rearrangement
  float sc = sqrt(l > 20.0f ? l : v == 1.0f ? u : log(v) * u / (v - 1.0f));
  key[t] = sc + bias[t], score[t] = sc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t >= 32) return;
  float kv[12];
  for (uint i = 0; i < 12; i++) kv[i] = t + 32 * i < rp.n ? key[t + 32 * i] : -INFINITY;
  uint pick = 0;  // lane k: the k-th expert picked
  for (uint k = 0; k < rp.k; k++) {
    float v = kv[0];
    uint a = t;
    for (uint i = 1; i < 12; i++)
      if (kv[i] > v) v = kv[i], a = t + 32 * i;
    for (uint o = 16; o; o >>= 1) {
      float v2 = simd_shuffle_xor(v, o);
      uint a2 = simd_shuffle_xor(a, o);
      if (v2 > v || (v2 == v && a2 < a)) v = v2, a = a2;
    }
    if (t == k) pick = a;
    for (uint i = 0; i < 12; i++)
      if (t + 32 * i == a) kv[i] = -INFINITY;
  }
  if (!rp.weighted) {
    if (t < rp.k) idx[t] = int(pick);
    return;
  }
  float w = t < rp.k ? score[pick] : 0, sum = 0;
  for (uint k = 0; k < rp.k; k++) sum += simd_shuffle(w, k);
  if (t < rp.k) idx[t] = int(pick), ((device float *)idx)[8 + t] = w / (sum + 1e-20f) * 1.5f;
}

// The thread with the largest v, the lowest one on ties; -1 if every v is -inf.
inline int tg_argmax(float v, uint t, uint n, threadgroup float *val, threadgroup uint *arg) {
  uint a = t;
  for (uint o = 16; o; o >>= 1) {
    float v2 = simd_shuffle_xor(v, o);
    uint a2 = simd_shuffle_xor(a, o);
    if (v2 > v || (v2 == v && a2 < a)) v = v2, a = a2;
  }
  if (t % 32 == 0) val[t / 32] = v, arg[t / 32] = a;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint b = 0;
  for (uint i = 1; i < n / 32; i++)
    if (val[i] > val[b] || (val[i] == val[b] && arg[i] < arg[b])) b = i;
  int r = val[b] == -INFINITY ? -1 : int(arg[b]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return r;
}

// Keeps the GPU's clock up while decoding waits for the SSD: one simdgroup
// counting for a bounded number of iterations (gpu_keep_clock).
kernel void spin(constant uint &iters [[buffer(0)]], device float *out [[buffer(1)]], uint t [[thread_position_in_grid]]) {
  float x = float(t);
  for (uint i = 0; i < iters; i++) x = fma(x, 1.0000001f, 0.5f);
  if (x == 1.2345f) out[0] = x;  // never true: keeps the loop
}

// ---- the DSpark draft ----

// The draft's input from a target layer (part 0 to 2): the mean of the row's
// four streams, as BF16, into its third of a row of 3 x 5120.
kernel void hc_mean(device const bfloat *h [[buffer(0)]], device float *out [[buffer(1)]],
                    constant uint &part [[buffer(2)]], uint2 i [[thread_position_in_grid]]) {
  h += i.y * 20480, out += i.y * 15360 + part * 5120;
  out[i.x] = bf((float(h[i.x]) + float(h[5120 + i.x]) + float(h[10240 + i.x]) + float(h[15360 + i.x])) * 0.25f);
}

// The Markov head's bias for the previous drafted token: logits[v] +=
// head[v] . embed[token] over 256 BF16 values, in FP32. A simdgroup per
// vocabulary row, eight values per lane.
kernel void markov(device const ushort *embed [[buffer(0)]], device const ushort *head [[buffer(1)]],
                   device const int *tok [[buffer(2)]], device float *logits [[buffer(3)]],
                   uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                   uint lane [[thread_index_in_simdgroup]]) {
  uint v = tg * 8 + sg;
  uint4 a = ((device const uint4 *)(embed + uint(tok[0]) * 256))[lane], b = ((device const uint4 *)(head + v * 256))[lane];
  float s = 0;
  for (uint i = 0; i < 4; i++)
    s += as_type<float>(a[i] << 16) * as_type<float>(b[i] << 16) + as_type<float>(a[i] & 0xFFFF0000u) * as_type<float>(b[i] & 0xFFFF0000u);
  s = simd_sum(s);
  if (lane == 0) logits[v] += s;
}

// The first index of the largest of n values, into out[0]. One threadgroup
// of 1024.
kernel void argmax(device const float *x [[buffer(0)]], device int *out [[buffer(1)]],
                   constant uint &n [[buffer(2)]], uint t [[thread_index_in_threadgroup]],
                   uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float bv[32];
  threadgroup uint bi[32];
  float v = -INFINITY;
  uint a = 0;
  for (uint i = t; i < n; i += 1024)
    if (x[i] > v) v = x[i], a = i;
  for (uint o = 16; o; o >>= 1) {
    float v2 = simd_shuffle_xor(v, o);
    uint a2 = simd_shuffle_xor(a, o);
    if (v2 > v || (v2 == v && a2 < a)) v = v2, a = a2;
  }
  if (lane == 0) bv[sg] = v, bi[sg] = a;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t == 0) {
    uint b = 0;
    for (uint i = 1; i < 32; i++)
      if (bv[i] > bv[b] || (bv[i] == bv[b] && bi[i] < bi[b])) b = i;
    out[0] = int(bi[b]);
  }
}

// ---- Engram ----

// Dequantizes the 24 fetched rows (E4M3 codes, UE8M0 per 32) to BF16, then
// quantizes them for the FP8 product.
kernel void engram_rows(device const uchar *codes [[buffer(0)]], device const uchar *scales [[buffer(1)]],
                        device float *x [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  uint r = i / 256, c = i % 256;
  uint b = codes[r * 256 + c];
  float v = fp8x4(b).x * 256.0f * e8m0(scales[r * 8 + c / 32]);
  x[i] = fp8_act(bf(v));
}

// h_c += sigmoid(signed sqrt(normalized h_c . (qk_c key_c))) value. One
// threadgroup of 1024 per stream.
kernel void engram_gate(device bfloat *h [[buffer(0)]], device const float *kv [[buffer(1)]],
                        device const float *qk [[buffer(2)]], device const uchar *img [[buffer(3)]],
                        uint2 gp [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]]) {
  threadgroup float tmp[32];
  if (img[gp.y]) return;  // a row inside an image span: the gate is shut, the row passes through
  uint c = gp.x;
  h += gp.y * 20480, kv += gp.y * 25600;
  float hh = 0, kk = 0, dot = 0;
  for (uint d = t; d < 5120; d += 1024) {
    float x = h[c * 5120 + d], k = kv[c * 5120 + d];
    hh += x * x;
    kk += k * k;
    dot += x * qk[c * 5120 + d] * k;
  }
  hh = tg_sum(hh, tmp, t, 1024);
  kk = tg_sum(kk, tmp, t, 1024);
  dot = tg_sum(dot, tmp, t, 1024);
  float g = dot * rsqrt(hh / 5120 + 1e-20f) * rsqrt(kk / 5120 + 1e-20f) * 0.013975424859373685f;
  g = 1.0f / (1.0f + exp(-copysign(sqrt(max(fabs(g), 1e-6f)), g)));
  for (uint d = t; d < 5120; d += 1024) h[c * 5120 + d] = bfloat(bf(float(h[c * 5120 + d]) + g * kv[20480 + d]));
}

// ---- matrix products for many rows (prefill) ----
//
// A threadgroup of 4 simdgroups computes a tile of the product with the GPU's
// matrix units (Metal tensor operations), from BF16 tiles of the activations
// and of the dequantized weights, both exact in BF16, accumulating in FP32.

struct MM { uint m, n, k, round, lda, ldy; };  // lda, ldy: row strides of x and y

#define MM_STORE(expr)                                          \
  for (uint16_t i = 0; i < c.get_capacity(); i++)               \
    if (c.is_valid_element(i)) {                                \
      auto at = c.get_multidimensional_index(i);                \
      uint col = n0 + at[0], row = m0 + at[1];                  \
      if (row < p.m && col < p.n) expr;                         \
    }

#define MM_SETUP                                                                                                       \
  threadgroup bfloat xs[64 * 64], ws[64 * 64];                                                                         \
  constexpr auto desc = matmul2d_descriptor(64, 64, 64, false, true, false, matmul2d_descriptor::mode::multiply_accumulate); \
  matmul2d<desc, execution_simdgroups<4>> op;                                                                          \
  tensor<threadgroup bfloat, extents<int32_t, 64, 64>, tensor_inline> tx(xs, extents<int32_t, 64, 64>());              \
  tensor<threadgroup bfloat, extents<int32_t, 64, 64>, tensor_inline> tw(ws, extents<int32_t, 64, 64>());              \
  auto c = op.get_destination_cooperative_tensor<decltype(tx), decltype(tw), float>();                                 \
  for (uint16_t i = 0; i < c.get_capacity(); i++)                                                                      \
    if (c.is_valid_element(i)) c[i] = 0;                                                                               \
  uint m0 = tg.y * 64, n0 = tg.x * 64;

// 64 rows x 64 columns of x (float) as BF16, eight four-float chunks per
// thread. As in the experts' kernels, a K step's loads are issued before the
// previous step's products and stored after them.
struct X8 { float4 v[8]; };
inline X8 x_load(device const float *x, uint m0, uint m, uint lda, uint k0, uint t) {
  X8 c;
  for (uint i = 0; i < 8; i++) {
    uint r = m0 + t / 16 + 8 * i;
    c.v[i] = r < m ? *(device const float4 *)(x + r * lda + k0 + (t % 16) * 4) : float4(0);
  }
  return c;
}
inline void x_store(threadgroup bfloat *xs, X8 c, uint t) {
  for (uint i = 0; i < 8; i++) *(threadgroup bfloat4 *)(xs + (t / 16 + 8 * i) * 64 + (t % 16) * 4) = bfloat4(c.v[i]);
}

// FP8 E4M3 weights with 32 x 32 block scales: two sixteen-byte chunks of W
// per thread and K step.
struct W8 { uint4 q[2]; float sc[2]; };
inline W8 fp8_load(device const uchar *w, device const uchar *s, uint n0, uint n, uint k, uint k0, uint t) {
  W8 c;
  for (uint i = 0; i < 2; i++) {
    uint r = n0 + t / 4 + 32 * i, j = k0 + (t % 4) * 16;
    c.q[i] = r < n ? *(device const uint4 *)(w + r * k + j) : uint4(0);
    c.sc[i] = r < n ? e8m0(s[(r / 32) * (k / 32) + j / 32]) * 256.0f : 0.0f;
  }
  return c;
}
inline void fp8_store(threadgroup bfloat *ws, W8 c, uint t) {
  for (uint i = 0; i < 2; i++) {
    threadgroup bfloat4 *d = (threadgroup bfloat4 *)(ws + (t / 4 + 32 * i) * 64 + (t % 4) * 16);
    uint u[4] = {c.q[i].x, c.q[i].y, c.q[i].z, c.q[i].w};
    for (uint b = 0; b < 4; b++) d[b] = bfloat4(fp8x4(u[b]) * c.sc[i]);
  }
}

// The matrix units read x as BF16 from device memory, rows of k (x_bf16,
// rows padded to 128), and the dequantized weights from threadgroup memory:
// a threadgroup takes 128 rows (two products of 64) and 64 columns, so that
// each K step's weights serve 128 rows. The products and the staging take
// turns on the GPU's cores rather than overlap, so staging less is faster.
kernel void mm_fp8(device const uchar *w [[buffer(0)]], device const uchar *s [[buffer(1)]],
                   device const bfloat *x [[buffer(2)]], device float *y [[buffer(3)]],
                   constant MM &p [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
                   uint t [[thread_index_in_threadgroup]]) {
  threadgroup bfloat ws[64 * 64];
  constexpr auto desc = matmul2d_descriptor(64, 64, 64, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<4>> op;
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tx((device bfloat *)x, dextents<int32_t, 2>(p.k, (p.m + 127) / 128 * 128));
  tensor<threadgroup bfloat, extents<int32_t, 64, 64>, tensor_inline> tw(ws, extents<int32_t, 64, 64>());
  auto c0 = op.get_destination_cooperative_tensor<decltype(tx.slice(0, 0)), decltype(tw), float>();
  auto c1 = op.get_destination_cooperative_tensor<decltype(tx.slice(0, 0)), decltype(tw), float>();
  for (uint16_t i = 0; i < c0.get_capacity(); i++)
    if (c0.is_valid_element(i)) c0[i] = 0, c1[i] = 0;
  uint m0 = tg.y * 128, n0 = tg.x * 64;
  W8 wc = fp8_load(w, s, n0, p.n, p.k, 0, t);
  for (uint k0 = 0; k0 < p.k; k0 += 64) {
    fp8_store(ws, wc, t);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (k0 + 64 < p.k) wc = fp8_load(w, s, n0, p.n, p.k, k0 + 64, t);
    auto a = tx.slice(k0, m0);
    op.run(a, tw, c0);
    if (m0 + 64 < p.m) {
      auto b = tx.slice(k0, m0 + 64);
      op.run(b, tw, c1);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (uint16_t i = 0; i < c0.get_capacity(); i++)
    if (c0.is_valid_element(i)) {
      auto at = c0.get_multidimensional_index(i);
      uint col = n0 + at[0];
      for (uint h = 0; h < 2; h++) {
        uint row = m0 + 64 * h + at[1];
        float v = h ? c1[i] : c0[i];
        if (row < p.m && col < p.n) y[row * p.ldy + col] = p.round ? bf(v) : v;
      }
    }
}
// m rows of k values (row stride lda) as BF16, in rows of k: the quantized
// activations that the FP8 products take are exact in BF16.
kernel void x_bf16(device const float4 *x [[buffer(0)]], device bfloat4 *y [[buffer(1)]],
                   constant uint2 &kl [[buffer(2)]], uint2 i [[thread_position_in_grid]]) {
  y[i.y * (kl.x / 4) + i.x] = bfloat4(x[i.y * (kl.y / 4) + i.x]);
}

// BF16 weights: four sixteen-byte chunks per thread and K step, stored as
// they are.
struct W16 { uint4 q[4]; };
inline W16 bf16_load(device const ushort *w, uint n0, uint n, uint k, uint k0, uint t) {
  W16 c;
  for (uint i = 0; i < 4; i++) {
    uint r = n0 + t / 8 + 16 * i;
    c.q[i] = r < n ? *(device const uint4 *)(w + r * k + k0 + (t % 8) * 8) : uint4(0);
  }
  return c;
}

kernel void mm_bf16(device const ushort *w [[buffer(0)]], device const float *x [[buffer(2)]],
                    device float *y [[buffer(3)]], constant MM &p [[buffer(4)]],
                    uint2 tg [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]]) {
  MM_SETUP
  X8 xc = x_load(x, m0, p.m, p.lda, 0, t);
  W16 wc = bf16_load(w, n0, p.n, p.k, 0, t);
  for (uint k0 = 0; k0 < p.k; k0 += 64) {
    x_store(xs, xc, t);
    for (uint i = 0; i < 4; i++) *(threadgroup uint4 *)(ws + (t / 8 + 16 * i) * 64 + (t % 8) * 8) = wc.q[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (k0 + 64 < p.k) xc = x_load(x, m0, p.m, p.lda, k0 + 64, t), wc = bf16_load(w, n0, p.n, p.k, k0 + 64, t);
    op.run(tx, tw, c);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  MM_STORE(y[row * p.ldy + col] = p.round ? bf(c[i]) : c[i])
}

// Routed experts over many tokens. The (token, expert) pairs are grouped by
// expert; a tile is up to 32 consecutive pairs of one expert. Pair q
// belongs to token row ptok[q] with routing weight pw[q].
struct MoeTile { device const uchar *p[3]; uint start, count; };  // p as in Expert

// An expert's MXFP4 matrix, n rows of k (E2M1 codes, two per byte, low
// nibble first; one UE8M0 scale per 32), as the matrix units read it from
// the slot.
using Mxfp4 = tensor<device metal_fp4_e2m1_format, dextents<int32_t, 2>, tensor_inline,
                     tensor_blockwise<tensor_plane_scales, device metal_fp8_ue8m0_format, 32, 1>>;
inline Mxfp4 mxfp4(device const uchar *w, device const uchar *s, int k, int n) {
  return Mxfp4((device uchar *)w, dextents<int32_t, 2>(k, n),
               tensor_blockwise<tensor_plane_scales, device metal_fp8_ue8m0_format, 32, 1>((device uchar *)s));
}

// E4M3's activation-scale floor gives a smallest step of 2^-31. Scaling by
// 2^7 makes that step an FP16 subnormal exactly; values up to 511.75 fit.
// The model checks its FFN norm bound at load, and SwiGLU is bounded by 150.
// Matrix outputs are scaled back before the reference's BF16 rounding.
kernel void gather_mxfp4(device const float4 *x [[buffer(0)]], device const int *ptok [[buffer(1)]],
                          device half4 *out [[buffer(2)]],
                          uint2 i [[thread_position_in_grid]]) {
  out[i.y * 1280 + i.x] = half4(x[ptok[i.y] * 1280 + i.x] * 128.0f);
}
kernel void act_quant_mxfp4(device const float *x [[buffer(0)]], device half *out [[buffer(1)]],
                             uint i [[thread_position_in_grid]]) {
  out[i] = half(fp8_act(x[i]) * 128.0f);
}

// The matrix units read the MXFP4 weights from the slot and the pairs' rows
// of x or h as half values multiplied by 128, in pair order.
// Each simdgroup runs its own products of the tile's pairs (32 rows, padded
// by 64 for the last tile) with 32 columns, in K steps of 128.
#define MOE_K 128
constexpr constant auto moe_desc = matmul2d_descriptor(32, 32, MOE_K, false, true, false,
                                                       matmul2d_descriptor::mode::multiply_accumulate);
#define MOE_SETUP(K)                                                                                       \
  matmul2d<moe_desc, execution_simdgroups<1>> op;                                                          \
  tensor<device half, dextents<int32_t, 2>, tensor_inline> tx((device half *)xh, dextents<int32_t, 2>(K, npairs + 64)); \
  MoeTile tl = tiles[tg.x];

// h[q] = weight_q silu(min(w1 x, 10)) clamp(w3 x, -10, 10). A threadgroup
// takes 64 columns: simdgroups 0 and 1 those of w1, 2 and 3 those of w3,
// whose sums meet w1's through threadgroup memory.
kernel void moe_up_m(device const MoeTile *tiles [[buffer(0)]], device const half *xh [[buffer(1)]],
                     device const float *pw [[buffer(2)]], device float *h [[buffer(3)]],
                     constant uint &npairs [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
                     uint sg [[simdgroup_index_in_threadgroup]]) {
  constexpr uint K = 5120, W = K / 2 * 2304, S = K / 32 * 2304;
  MOE_SETUP(K)
  bool three = sg >= 2;
  uint j = sg & 1, n0 = tg.y * 64 + 32 * j;
  Mxfp4 w = mxfp4(tl.p[0] + (three ? 2 * W : 0), tl.p[1] + (three ? 2 * S : 0), K, 2304);
  auto c = op.get_destination_cooperative_tensor<decltype(tx.slice(0, 0)), decltype(w.slice(0, 0)), float>();
  for (uint16_t i = 0; i < c.get_capacity(); i++)
    if (c.is_valid_element(i)) c[i] = 0;
  for (uint k0 = 0; k0 < K; k0 += MOE_K) {
    auto x = tx.slice(k0, tl.start);
    auto b = w.slice(k0, n0);
    op.run(x, b, c);
  }
  threadgroup float acc[2][32 * 32];  // w3's sums: pair r's column c at 32 r + c
  tensor<threadgroup float, extents<int32_t, 32, 32>, tensor_inline> ta(acc[j], extents<int32_t, 32, 32>());
  if (three) c.store(ta);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (!three)
    for (uint16_t i = 0; i < c.get_capacity(); i++)
      if (c.is_valid_element(i)) {
        auto at = c.get_multidimensional_index(i);
        uint col = at[0], r = at[1];
        if (r < tl.count) {
          uint q = tl.start + r;
          float g = min(bf(c[i] * 0x1p-7f), 10.0f), u = clamp(bf(acc[j][32 * r + col] * 0x1p-7f), -10.0f, 10.0f);
          h[q * 2304 + n0 + col] = bf(g / (1.0f + exp(-g)) * u * pw[q]);
        }
      }
}

// y[q] = bf16(w2 h[q]) for the quantized h. A threadgroup takes 128
// columns, 32 per simdgroup.
kernel void moe_down_m(device const MoeTile *tiles [[buffer(0)]], device const half *xh [[buffer(1)]],
                       device float *y [[buffer(2)]], constant uint &npairs [[buffer(3)]],
                       uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]]) {
  constexpr uint K = 2304, W = 5120 * 2304 / 2, S = 5120 * 2304 / 32;
  MOE_SETUP(K)
  uint n0 = tg.y * 128 + 32 * sg;
  Mxfp4 w2 = mxfp4(tl.p[0] + W, tl.p[1] + S, K, 5120);
  auto c = op.get_destination_cooperative_tensor<decltype(tx.slice(0, 0)), decltype(w2.slice(0, 0)), float>();
  for (uint16_t i = 0; i < c.get_capacity(); i++)
    if (c.is_valid_element(i)) c[i] = 0;
  for (uint k0 = 0; k0 < K; k0 += MOE_K) {
    auto x = tx.slice(k0, tl.start);
    auto b = w2.slice(k0, n0);
    op.run(x, b, c);
  }
  for (uint16_t i = 0; i < c.get_capacity(); i++)
    if (c.is_valid_element(i)) {
      auto at = c.get_multidimensional_index(i);
      uint r = at[1];
      if (r < tl.count) y[(tl.start + r) * 5120 + n0 + at[0]] = bf(c[i] * 0x1p-7f);
    }
}

// Token row r: y = bf16(sum of its experts' outputs in expert order + shared),
// then hc_post into the four streams. tp[r] lists its pairs in expert order.
kernel void moe_combine(device const float *y [[buffer(0)]], device const int *tp [[buffer(1)]],
                        device const float *shared_out [[buffer(2)]], device const float *mix [[buffer(3)]],
                        device const bfloat *res [[buffer(4)]], device bfloat *out [[buffer(5)]],
                        uint2 i [[thread_position_in_grid]]) {
  uint d = i.x, r = i.y;
  float v = 0;
  for (uint k = 0; k < 6; k++) v += y[tp[r * 6 + k] * 5120 + d];
  v = bf(v + shared_out[r * 5120 + d]);
  mix += r * 24, res += r * 20480, out += r * 20480;
  for (uint k = 0; k < 4; k++) {
    float o = mix[4 + k] * v;
    for (uint j = 0; j < 4; j++) o += mix[8 + 4 * j + k] * res[j * 5120 + d];
    out[k * 5120 + d] = bfloat(bf(o));
  }
}
