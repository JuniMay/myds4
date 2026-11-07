// The Qwen3.8 (qwen3_5) kernels of src/qwen/qwen.c. A forward pass runs up
// to TM token rows at once; the rows may belong to different sequences
// (struct Seq), each sequence's rows consecutive and at consecutive
// positions. Every row's arithmetic is the same whatever the number of rows
// and their place in the pass, so a row's result does not depend on how a
// prompt is split into passes or batched with other sequences. Activations
// are FP32; keys and values are stored as FP16. The math follows the
// reference implementation (MLX's qwen3_5 forward): see docs/qwen.md.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

#define TM 8      // most token rows per pass (qwen.c's ROWS)
#define BLK 256   // positions per attention block

// An FP8 E4M3 code as a float (no NaN codes in the weights).
inline float fp8(uchar b) { return float(as_type<half>(ushort(((b & 0x7Fu) << 7) | ((b & 0x80u) << 8)))) * 256.0f; }
// An FP4 E2M1 code (low 4 bits) as a float.
inline float fp4(uint nib) { return float(as_type<half>(ushort(((nib & 7u) << 9) | ((nib & 8u) << 12)))) * 16384.0f; }
inline float bf16f(ushort u) { return as_type<float>(uint(u) << 16); }
inline float sigm(float x) { return 1.0f / (1.0f + exp(-x)); }
// log(1 + e) without losing a small e (Metal has no log1p): Goldberg's
// correction of log(u) by e / (u - 1), u = 1 + e rounded.
inline float log1p_(float e) {
  float u = 1.0f + e;
  return u == 1.0f ? e : log(u) * e / (u - 1.0f);
}

// A row's sequence (GPU addresses; qwen.c's Seq).
struct Seq {
  device float *gdn, *gdn_save;    // DeltaNet states [linear layer][value head][value dim][key dim]
  device float *conv, *conv_save;  // convolution inputs [linear layer][3][channels], oldest first
  device half *k, *v;              // [attention layer][cap][kv head][head dim]
  uint pos, cap;                   // the row's position; the keys' capacity
};

// ---- matrices times rows ----

struct Mv { uint rows, cols, n; float scale; };  // n: token rows; y = scale * W x per row

// Sixteen values of a 16-value chunk as four float4.
inline void nvfp4_chunk(uint2 c, float s, thread float4 *w) {
  w[0] = float4(fp4(c.x & 15), fp4((c.x >> 4) & 15), fp4((c.x >> 8) & 15), fp4((c.x >> 12) & 15)) * s;
  w[1] = float4(fp4((c.x >> 16) & 15), fp4((c.x >> 20) & 15), fp4((c.x >> 24) & 15), fp4(c.x >> 28)) * s;
  w[2] = float4(fp4(c.y & 15), fp4((c.y >> 4) & 15), fp4((c.y >> 8) & 15), fp4((c.y >> 12) & 15)) * s;
  w[3] = float4(fp4((c.y >> 16) & 15), fp4((c.y >> 20) & 15), fp4((c.y >> 24) & 15), fp4(c.y >> 28)) * s;
}
inline float4 fp8x4(uint u) { return float4(fp8(u & 255), fp8((u >> 8) & 255), fp8((u >> 16) & 255), fp8(u >> 24)); }
inline float4 bf16x4(uint2 u) {
  return float4(bf16f(u.x & 0xFFFF), bf16f(u.x >> 16), bf16f(u.y & 0xFFFF), bf16f(u.y >> 16));
}

// A 16-value chunk c of weight row `row` as four float4 (NVFP4: times its
// E4M3 scale, exact in FP32; FP8 and BF16: as stored).
template <int F> inline void chunk(device const uchar *W, device const uchar *S, uint row, uint cols, uint c,
                                   thread float4 *w);
template <> inline void chunk<0>(device const uchar *W, device const uchar *S, uint row, uint cols, uint c,
                                 thread float4 *w) {
  uint2 q = ((device const uint2 *)(W + (size_t)row * (cols / 2)))[c];
  nvfp4_chunk(q, fp8(S[(size_t)row * (cols / 16) + c]), w);
}
template <> inline void chunk<1>(device const uchar *W, device const uchar *S, uint row, uint cols, uint c,
                                 thread float4 *w) {
  uint4 q = ((device const uint4 *)(W + (size_t)row * cols))[c];
  w[0] = fp8x4(q.x), w[1] = fp8x4(q.y), w[2] = fp8x4(q.z), w[3] = fp8x4(q.w);
}
template <> inline void chunk<2>(device const uchar *W, device const uchar *S, uint row, uint cols, uint c,
                                 thread float4 *w) {
  device const uint4 *p = (device const uint4 *)(W + (size_t)row * cols * 2) + 2 * c;
  uint4 a = p[0], b = p[1];
  w[0] = bf16x4(a.xy), w[1] = bf16x4(a.zw), w[2] = bf16x4(b.xy), w[3] = bf16x4(b.zw);
}

// y[t][r] = scale * sum_c W[r][c] x[t][c] for the n rows t of x: a
// simdgroup a weight row, 8 a threadgroup. Lane l takes the 16-value chunks
// l, l + 32, ... of the row, each chunk's products summed as four float4
// then across, the lanes' sums at the end: the same arithmetic per (t, r)
// for any n, in mv1 (one row of x) and mvr (up to TM rows, each chunk of
// weights it decodes serving all of them). F: 0 NVFP4 (S: E4M3 scales per
// 16), 1 FP8, 2 BF16.
template <int F>
kernel void mv1(device const uchar *W [[buffer(0)]], device const uchar *S [[buffer(1)]],
                device const float *x [[buffer(2)]], device float *y [[buffer(3)]], constant Mv &p [[buffer(4)]],
                uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                uint lane [[thread_index_in_simdgroup]]) {
  uint row = tg * 8 + sg, chunks = p.cols / 16;
  if (row >= p.rows) return;
  float acc = 0;
  for (uint c = lane; c < chunks; c += 32) {
    float4 w[4];
    chunk<F>(W, S, row, p.cols, c, w);
    device const float4 *xv = (device const float4 *)x + c * 4;
    float4 d = w[0] * xv[0] + w[1] * xv[1] + w[2] * xv[2] + w[3] * xv[3];
    acc += d.x + d.y + d.z + d.w;
  }
  acc = simd_sum(acc);
  if (lane == 0) y[row] = acc * p.scale;
}

template <int F>
kernel void mvr(device const uchar *W [[buffer(0)]], device const uchar *S [[buffer(1)]],
                device const float *x [[buffer(2)]], device float *y [[buffer(3)]], constant Mv &p [[buffer(4)]],
                uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                uint lane [[thread_index_in_simdgroup]]) {
  uint row = tg * 8 + sg, chunks = p.cols / 16;
  if (row >= p.rows) return;
  float acc[TM];
  for (uint t = 0; t < TM; t++) acc[t] = 0;
  for (uint c = lane; c < chunks; c += 32) {
    float4 w[4];
    chunk<F>(W, S, row, p.cols, c, w);
    for (uint t = 0; t < TM; t++) {
      if (t >= p.n) break;
      device const float4 *xv = (device const float4 *)(x + (size_t)t * p.cols) + c * 4;
      float4 d = w[0] * xv[0] + w[1] * xv[1] + w[2] * xv[2] + w[3] * xv[3];
      acc[t] += d.x + d.y + d.z + d.w;
    }
  }
  for (uint t = 0; t < TM; t++) {
    if (t >= p.n) break;
    float v = simd_sum(acc[t]);
    if (lane == 0) y[(size_t)t * p.rows + row] = v * p.scale;
  }
}

// h[t][r] = silu(G x)[t][r] * (U x)[t][r] for the n rows of x (1 to TM): a
// simdgroup a row of both matrices, each as mv1/mvr computes it.
template <int F>
kernel void gate_up(device const uchar *Wg [[buffer(0)]], device const uchar *Sg [[buffer(1)]],
                    device const uchar *Wu [[buffer(5)]], device const uchar *Su [[buffer(6)]],
                    device const float *x [[buffer(2)]], device float *h [[buffer(3)]], constant Mv &p [[buffer(4)]],
                    constant float &up_scale [[buffer(7)]], uint tg [[threadgroup_position_in_grid]],
                    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  uint row = tg * 8 + sg, chunks = p.cols / 16;
  if (row >= p.rows) return;
  float ag[TM], au[TM];
  for (uint t = 0; t < TM; t++) ag[t] = 0, au[t] = 0;
  for (uint c = lane; c < chunks; c += 32) {
    float4 wg[4], wu[4];
    chunk<F>(Wg, Sg, row, p.cols, c, wg);
    chunk<F>(Wu, Su, row, p.cols, c, wu);
    for (uint t = 0; t < TM; t++) {
      if (t >= p.n) break;
      device const float4 *xv = (device const float4 *)(x + (size_t)t * p.cols) + c * 4;
      float4 dg = wg[0] * xv[0] + wg[1] * xv[1] + wg[2] * xv[2] + wg[3] * xv[3];
      float4 du = wu[0] * xv[0] + wu[1] * xv[1] + wu[2] * xv[2] + wu[3] * xv[3];
      ag[t] += dg.x + dg.y + dg.z + dg.w;
      au[t] += du.x + du.y + du.z + du.w;
    }
  }
  for (uint t = 0; t < TM; t++) {
    if (t >= p.n) break;
    float g = simd_sum(ag[t]) * p.scale, u = simd_sum(au[t]) * up_scale;
    if (lane == 0) h[(size_t)t * p.rows + row] = g * sigm(g) * u;
  }
}
typedef decltype(gate_up<0>) gate_up_t;
template [[host_name("gate_up_nvfp4")]] kernel gate_up_t gate_up<0>;
template [[host_name("gate_up_fp8")]] kernel gate_up_t gate_up<1>;
template [[host_name("gate_up_bf16")]] kernel gate_up_t gate_up<2>;

typedef decltype(mv1<0>) mv1_t;
typedef decltype(mvr<0>) mvr_t;
template [[host_name("mvr_nvfp4")]] kernel mvr_t mvr<0>;
template [[host_name("mvr_fp8")]] kernel mvr_t mvr<1>;
template [[host_name("mvr_bf16")]] kernel mvr_t mvr<2>;
template [[host_name("mv1_nvfp4")]] kernel mv1_t mv1<0>;
template [[host_name("mv1_fp8")]] kernel mv1_t mv1<1>;
template [[host_name("mv1_bf16")]] kernel mv1_t mv1<2>;

// ---- matrices times a prompt's rows, on the matrix units ----
//
// A prompt pass (qwen_prefill) takes the matrices through the GPU's matrix
// units (Metal tensor operations) with BF16 operands and FP32 sums: x rounded
// to BF16 (x_bf16), the weights dequantized into BF16 tiles, where they are
// exact (E2M1 times an E4M3 scale has at most 6 significant bits, E4M3 4,
// BF16 is BF16), the matrix's own scale applied at the end. A threadgroup of
// 4 simdgroups takes 128 rows of x (two 64-row products) and 64 weight rows;
// a K step takes 64 columns, the next step's weights loaded while this step's
// products run (as the DeepSeek engine's mm_fp8). The units add in an order
// that does not depend on the tile or on the number of rows, so a row's
// result is the same however a prompt is split into passes.

struct MM { uint m, n, k, pad; float scale; uint pad0, pad1, pad2; };  // m rows of x (k each), n weight rows

// Rows of x (FP32, k each) as BF16, in rows of k padded to 128.
kernel void x_bf16(device const float4 *x [[buffer(0)]], device bfloat4 *y [[buffer(1)]],
                   constant uint &k [[buffer(2)]], uint2 i [[thread_position_in_grid]]) {
  if (i.x < k / 4) y[(size_t)i.y * (k / 4) + i.x] = bfloat4(x[(size_t)i.y * (k / 4) + i.x]);
}

// A thread's share of a K step's weight tile: 32 values of weight row
// n0 + t / 2, columns k0 + 32 (t % 2) on, as BF16 into ws.
template <int F>
inline void tile_load(device const uchar *W, device const uchar *S, uint n0, uint n, uint k, uint k0, uint t,
                      thread uint4 *q, thread float *sc) {
  uint r = n0 + t / 2, c = k0 + (t % 2) * 32;
  bool in = r < n;
  if (F == 0) {  // NVFP4: 16 bytes of codes, two E4M3 scales
    q[0] = in ? *(device const uint4 *)(W + (size_t)r * (k / 2) + c / 2) : uint4(0);
    sc[0] = in ? fp8(S[(size_t)r * (k / 16) + c / 16]) : 0.0f;
    sc[1] = in ? fp8(S[(size_t)r * (k / 16) + c / 16 + 1]) : 0.0f;
  } else if (F == 1) {  // FP8: 32 bytes
    device const uint4 *p = (device const uint4 *)(W + (size_t)r * k + c);
    q[0] = in ? p[0] : uint4(0), q[1] = in ? p[1] : uint4(0);
  } else {  // BF16: 64 bytes
    device const uint4 *p = (device const uint4 *)(W + (size_t)r * k * 2 + c * 2);
    for (uint i = 0; i < 4; i++) q[i] = in ? p[i] : uint4(0);
  }
}

template <int F>
inline void tile_store(threadgroup bfloat *ws, thread const uint4 *q, thread const float *sc, uint t) {
  threadgroup bfloat4 *d = (threadgroup bfloat4 *)(ws + (t / 2) * 64 + (t % 2) * 32);
  if (F == 0) {
    uint u[4] = {q[0].x, q[0].y, q[0].z, q[0].w};
    for (uint b = 0; b < 4; b++) {  // 8 codes a word, a scale per 16 codes
      float s = sc[b / 2];
      uint w = u[b];
      d[2 * b] = bfloat4(float4(fp4(w & 15), fp4((w >> 4) & 15), fp4((w >> 8) & 15), fp4((w >> 12) & 15)) * s);
      d[2 * b + 1] = bfloat4(float4(fp4((w >> 16) & 15), fp4((w >> 20) & 15), fp4((w >> 24) & 15), fp4(w >> 28)) * s);
    }
  } else if (F == 1) {
    uint u[8] = {q[0].x, q[0].y, q[0].z, q[0].w, q[1].x, q[1].y, q[1].z, q[1].w};
    for (uint b = 0; b < 8; b++) d[b] = bfloat4(fp8x4(u[b]));
  } else {
    threadgroup uint4 *e = (threadgroup uint4 *)d;
    for (uint i = 0; i < 4; i++) e[i] = q[i];
  }
}

// y[t][r] = scale * sum_c W[r][c] x[t][c] for the m rows of x (BF16, rows
// of k padded to 128), y's rows of n.
template <int F>
kernel void mm(device const uchar *W [[buffer(0)]], device const uchar *S [[buffer(1)]],
               device const bfloat *x [[buffer(2)]], device float *y [[buffer(3)]], constant MM &p [[buffer(4)]],
               uint2 tg [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]]) {
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
  uint4 q[4];
  float sc[2];
  tile_load<F>(W, S, n0, p.n, p.k, 0, t, q, sc);
  for (uint k0 = 0; k0 < p.k; k0 += 64) {
    tile_store<F>(ws, q, sc, t);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (k0 + 64 < p.k) tile_load<F>(W, S, n0, p.n, p.k, k0 + 64, t, q, sc);
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
        if (row < p.m && col < p.n) y[(size_t)row * p.n + col] = (h ? c1[i] : c0[i]) * p.scale;
      }
    }
}
typedef decltype(mm<0>) mm_t;
template [[host_name("mm_nvfp4")]] kernel mm_t mm<0>;
template [[host_name("mm_fp8")]] kernel mm_t mm<1>;
template [[host_name("mm_bf16")]] kernel mm_t mm<2>;

// ---- per-row operations ----

// Row t of x: the embedding of token tok[t] (BF16) as floats.
kernel void embed(device const ushort *E [[buffer(0)]], device const int *tok [[buffer(1)]],
                  device float *x [[buffer(2)]], constant uint &n [[buffer(3)]], uint2 g [[thread_position_in_grid]]) {
  if (g.x < n) x[(size_t)g.y * n + g.x] = bf16f(E[(size_t)tok[g.y] * n + g.x]);
}

struct Norm { uint n; float eps; float scale; uint gate; };

// Row r of n values: y = x / sqrt(mean(x^2) + eps) * w * scale, and with
// gate (the gated DeltaNet's output) times silu(z). One threadgroup a row.
kernel void rms_norm(device const float *x [[buffer(0)]], device const float *w [[buffer(1)]],
                     device float *y [[buffer(2)]], constant Norm &p [[buffer(3)]],
                     device const float *z [[buffer(4)]], uint row [[threadgroup_position_in_grid]],
                     uint t [[thread_position_in_threadgroup]], uint tn [[threads_per_threadgroup]],
                     uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float part[32];
  device const float *xr = x + (size_t)row * p.n;
  float s = 0;
  for (uint i = t; i < p.n; i += tn) s += xr[i] * xr[i];
  s = simd_sum(s);
  if (lane == 0) part[sg] = s;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    float v = lane < (tn + 31) / 32 ? part[lane] : 0.0f;
    v = simd_sum(v);
    if (lane == 0) part[0] = v;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float r = rsqrt(part[0] / p.n + p.eps) * p.scale;
  device float *yr = y + (size_t)row * p.n;
  for (uint i = t; i < p.n; i += tn) {
    float v = xr[i] * r * w[i];
    if (p.gate) {
      float g = z[(size_t)row * p.n + i];
      v *= g * sigm(g);
    }
    yr[i] = v;
  }
}

// x += r, then y = RMSNorm(x) * w: add then rms_norm in one kernel, the same
// arithmetic (each thread adds the values it then squares and scales).
kernel void add_norm(device float *x [[buffer(0)]], device const float *r [[buffer(1)]],
                     device const float *w [[buffer(2)]], device float *y [[buffer(3)]],
                     constant Norm &p [[buffer(4)]], uint row [[threadgroup_position_in_grid]],
                     uint t [[thread_position_in_threadgroup]], uint tn [[threads_per_threadgroup]],
                     uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float part[32];
  device float *xr = x + (size_t)row * p.n;
  device const float *rr = r + (size_t)row * p.n;
  float s = 0;
  for (uint i = t; i < p.n; i += tn) {
    float v = xr[i] + rr[i];
    xr[i] = v;
    s += v * v;
  }
  s = simd_sum(s);
  if (lane == 0) part[sg] = s;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    float v = lane < (tn + 31) / 32 ? part[lane] : 0.0f;
    v = simd_sum(v);
    if (lane == 0) part[0] = v;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float q = rsqrt(part[0] / p.n + p.eps) * p.scale;
  device float *yr = y + (size_t)row * p.n;
  for (uint i = t; i < p.n; i += tn) yr[i] = xr[i] * q * w[i];
}

struct L2 { uint n, stride; float eps, scale; };

// Head g.x of row g.y (n values at x + g.y * stride + g.x * n): divided by
// sqrt(sum(x^2) + eps), times scale, in place (the DeltaNet's queries and
// keys). One threadgroup a head.
kernel void l2_norm(device float *x [[buffer(0)]], constant L2 &p [[buffer(1)]],
                    uint2 g [[threadgroup_position_in_grid]], uint2 t2 [[thread_position_in_threadgroup]],
                    uint2 tn2 [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float part[32];
  uint t = t2.x, tn = tn2.x;
  device float *xr = x + (size_t)g.y * p.stride + g.x * p.n;
  float s = 0;
  for (uint i = t; i < p.n; i += tn) s += xr[i] * xr[i];
  s = simd_sum(s);
  if (lane == 0) part[sg] = s;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    float v = lane < (tn + 31) / 32 ? part[lane] : 0.0f;
    v = simd_sum(v);
    if (lane == 0) part[0] = v;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float r = rsqrt(part[0] + p.eps) * p.scale;
  for (uint i = t; i < p.n; i += tn) xr[i] *= r;
}

struct Conv { uint channels, n, layer, pad; };

// The DeltaNet's causal depthwise convolution (kernel 4) of channel c over
// the rows' projected q, k and v, then silu. A sequence's 3 previous inputs
// move on by one a row; a row with conv_save also leaves them there.
kernel void conv(device const float *in [[buffer(0)]], device float *out [[buffer(1)]],
                 device const float *w [[buffer(2)]], constant Seq *rows [[buffer(3)]],
                 constant Conv &p [[buffer(4)]], uint c [[thread_position_in_grid]]) {
  uint C = p.channels;
  if (c >= C) return;
  size_t lo = (size_t)p.layer * 3 * C;
  float w0 = w[c * 4], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3];
  device float *st = 0;
  float s0 = 0, s1 = 0, s2 = 0;
  for (uint t = 0; t < p.n; t++) {
    device float *rs = rows[t].conv + lo;
    if (rs != st) {
      if (st) st[c] = s0, st[C + c] = s1, st[2 * C + c] = s2;
      st = rs, s0 = st[c], s1 = st[C + c], s2 = st[2 * C + c];
    }
    float x = in[(size_t)t * C + c];
    float v = w0 * s0 + w1 * s1 + w2 * s2 + w3 * x;
    out[(size_t)t * C + c] = v * sigm(v);
    s0 = s1, s1 = s2, s2 = x;
    if (rows[t].conv_save) {
      device float *sv = rows[t].conv_save + lo;
      sv[c] = s0, sv[C + c] = s1, sv[2 * C + c] = s2;
    }
  }
  if (st) st[c] = s0, st[C + c] = s1, st[2 * C + c] = s2;
}

// The DeltaNet's gates of value head i % heads of row i / heads:
// beta = sigmoid(b), the decay g = exp(-A * softplus(a + dt_bias)), A = exp(A_log).
kernel void gdn_gates(device const float *a [[buffer(0)]], device const float *b [[buffer(1)]],
                      device const float *A [[buffer(2)]], device const float *dtb [[buffer(3)]],
                      device float *g [[buffer(4)]], device float *beta [[buffer(5)]],
                      constant uint2 &p [[buffer(6)]], uint i [[thread_position_in_grid]]) {
  if (i >= p.x * p.y) return;
  uint h = i % p.x;
  float x = a[i] + dtb[h];
  float sp = max(x, 0.0f) + log1p_(exp(-fabs(x)));
  g[i] = exp(-A[h] * sp);
  beta[i] = sigm(b[i]);
}

struct Gdn { uint n, heads, ratio, layer, channels, keys, pad0, pad1; };  // keys: key heads * 128

// The gated delta rule of value head hv and value dimension dv over the
// rows: the state row S[dv][0..128) (32 lanes, 4 each) decays by g, learns
// beta * (v - S.k) along k, and gives S.q. qkv rows: queries and keys (key
// heads of 128, each serving `ratio` consecutive value heads), then values.
// A row with gdn_save also leaves the state there.
kernel void gdn(device const float *qkv [[buffer(0)]], device const float *g [[buffer(1)]],
                device const float *beta [[buffer(2)]], constant Seq *rows [[buffer(3)]],
                device float *out [[buffer(4)]], constant Gdn &p [[buffer(5)]],
                uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                uint lane [[thread_index_in_simdgroup]]) {
  uint hv = tg.x, dv = tg.y * 4 + sg, hk = hv / p.ratio;
  size_t off = (((size_t)p.layer * p.heads + hv) * 128 + dv) * 128 + lane * 4;
  device float *st = 0;
  float4 s = 0;
  for (uint t = 0; t < p.n; t++) {
    device float *rs = rows[t].gdn + off;
    if (rs != st) {
      if (st) *(device float4 *)st = s;
      st = rs, s = *(device float4 *)st;
    }
    device const float *r = qkv + (size_t)t * p.channels;
    float4 q4 = *(device const float4 *)(r + hk * 128 + lane * 4);
    float4 k4 = *(device const float4 *)(r + p.keys + hk * 128 + lane * 4);
    float v = r[2 * p.keys + hv * 128 + dv];
    s *= g[t * p.heads + hv];
    float mem = simd_sum(dot(s, k4));
    float delta = (v - mem) * beta[t * p.heads + hv];
    s += k4 * delta;
    float o = simd_sum(dot(s, q4));
    if (lane == 0) out[((size_t)t * p.heads + hv) * 128 + dv] = o;
    if (rows[t].gdn_save) *(device float4 *)(rows[t].gdn_save + off) = s;
  }
  if (st) *(device float4 *)st = s;
}

struct Rope { uint heads, dim, rot, stride; float theta, eps; uint pad0, pad1; };

// Full attention's query or key head g.x of row g.y (dim values at
// x + (g.y * heads + g.x) * stride): RMSNorm (weights w), then the rotary
// embedding of the first rot dimensions at the row's position (pairs i,
// i + rot/2; frequency theta^(-2i/rot)). One threadgroup a head.
kernel void qk_rope(device const float *x [[buffer(0)]], device const float *w [[buffer(1)]],
                    device float *y [[buffer(2)]], constant Rope &p [[buffer(3)]],
                    constant Seq *rows [[buffer(4)]], uint2 g [[threadgroup_position_in_grid]],
                    uint2 t2 [[thread_position_in_threadgroup]], uint2 tn2 [[threads_per_threadgroup]],
                    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float part[32];
  threadgroup float row[512];
  uint t = t2.x, tn = tn2.x;
  device const float *xr = x + ((size_t)g.y * p.heads + g.x) * p.stride;
  float s = 0;
  for (uint i = t; i < p.dim; i += tn) s += xr[i] * xr[i];
  s = simd_sum(s);
  if (lane == 0) part[sg] = s;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    float v = lane < (tn + 31) / 32 ? part[lane] : 0.0f;
    v = simd_sum(v);
    if (lane == 0) part[0] = v;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float r = rsqrt(part[0] / p.dim + p.eps);
  for (uint i = t; i < p.dim; i += tn) row[i] = xr[i] * r * w[i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  device float *yr = y + ((size_t)g.y * p.heads + g.x) * p.dim;
  uint half_rot = p.rot / 2;
  float pos = float(rows[g.y].pos);
  for (uint i = t; i < p.dim; i += tn) {
    float v = row[i];
    if (i < p.rot) {
      uint j = i < half_rot ? i : i - half_rot;
      float f = pos * pow(p.theta, -2.0f * float(j) / float(p.rot));
      float c = cos(f), sn = sin(f);
      v = i < half_rot ? row[i] * c - row[i + half_rot] * sn : row[i - half_rot] * sn + row[i] * c;
    }
    yr[i] = v;
  }
}

struct Kv { uint n, width, layer, pad; };  // width: kv heads * head dim

// Row g.y's key and value (FP32, `width` values each) into its sequence's
// cache at its position, as FP16.
kernel void kv_store(device const float *k [[buffer(0)]], device const float *v [[buffer(1)]],
                     constant Seq *rows [[buffer(2)]], constant Kv &p [[buffer(3)]],
                     uint2 g [[thread_position_in_grid]]) {
  if (g.x >= p.width) return;
  size_t at = ((size_t)p.layer * rows[g.y].cap + rows[g.y].pos) * p.width + g.x;
  rows[g.y].k[at] = half(k[(size_t)g.y * p.width + g.x]);
  rows[g.y].v[at] = half(v[(size_t)g.y * p.width + g.x]);
}

struct Att { uint heads, kv, dim, layer; float scale; uint blocks, pad0, pad1; };  // blocks: partials per head

// Attention of row g.z's query heads of key head g.x (heads / kv of them,
// at most 8) over the positions of block g.y (BLK of them, up to the row's
// position): per head the block's highest score m, the sum l of exp(score -
// m) and the sum of exp(score - m) * value, for attend_combine.
kernel void attend_block(device const float *q [[buffer(0)]], constant Seq *rows [[buffer(1)]],
                         device float *part [[buffer(2)]], device float2 *ml [[buffer(3)]],
                         constant Att &p [[buffer(4)]], uint3 g [[threadgroup_position_in_grid]],
                         uint j [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
  uint kvh = g.x, b = g.y, t = g.z, pos = rows[t].pos, G = p.heads / p.kv;
  if (b * BLK > pos) return;
  // the queries (8 heads x 256) and the scores (8 x BLK), and after them the
  // simdgroups' value sums (8 simdgroups x 2 heads x 256)
  threadgroup float stage[8 * 256 + 8 * BLK];
  threadgroup float(*qs)[256] = (threadgroup float(*)[256])stage;
  threadgroup float(*ps)[BLK] = (threadgroup float(*)[BLK])(stage + 8 * 256);
  threadgroup float red[8][8];
  for (uint i = j; i < G * p.dim; i += BLK) qs[i / p.dim][i % p.dim] = q[((size_t)t * p.heads + kvh * G) * p.dim + i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint at = b * BLK + j, width = p.kv * p.dim;
  size_t layer = (size_t)p.layer * rows[t].cap * width;
  // the scores: a simdgroup reads one key row at a time, its lanes 8
  // dimensions each (a row of 256 values: coalesced), each head's partial
  // dot products summed across the lanes
  {
    float4 qa[8][2];
    for (uint h = 0; h < G; h++)
      qa[h][0] = *(threadgroup const float4 *)&qs[h][lane * 8], qa[h][1] = *(threadgroup const float4 *)&qs[h][lane * 8 + 4];
    for (uint i = sg; i < BLK; i += BLK / 32) {
      uint ati = b * BLK + i;
      if (ati > pos) break;
      device const half4 *kr = (device const half4 *)(rows[t].k + layer + (size_t)ati * width + kvh * p.dim + lane * 8);
      float4 k0 = float4(kr[0]), k1 = float4(kr[1]);
      for (uint h = 0; h < G; h++) {
        float d = simd_sum(dot(k0, qa[h][0]) + dot(k1, qa[h][1]));
        if (lane == 0) ps[h][i] = d * p.scale;
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float sc[8];
  for (uint h = 0; h < G; h++) sc[h] = at <= pos ? ps[h][j] : -INFINITY;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint h = 0; h < G; h++) {
    float m = simd_max(sc[h]);
    if (lane == 0) red[h][sg] = m;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float mh[8];
  for (uint h = 0; h < G; h++) {
    float m = red[h][0];
    for (uint s = 1; s < BLK / 32; s++) m = max(m, red[h][s]);
    mh[h] = m;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint h = 0; h < G; h++) {
    float e = at <= pos ? exp(sc[h] - mh[h]) : 0.0f;
    ps[h][j] = e;
    float s = simd_sum(e);
    if (lane == 0) red[h][sg] = s;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint valid = min((uint)BLK, pos + 1 - b * BLK);
  size_t first = ((size_t)t * p.heads + kvh * G) * p.blocks + b;
  // the values: simdgroup sg takes the block's positions 32 sg .. 32 sg + 31,
  // lane l the dimensions 8 l .. 8 l + 7 (a value row coalesced); the
  // simdgroups' sums meet in threadgroup memory two heads at a time, added in
  // simdgroup order (in the queries' and scores' place, done with)
  {
    float4 acc[8][2];
    for (uint h = 0; h < G; h++) acc[h][0] = acc[h][1] = 0;
    device const half4 *vr = (device const half4 *)(rows[t].v + layer + (size_t)(b * BLK) * width + kvh * p.dim + lane * 8);
    uint lo = sg * 32, hi = min(lo + 32, valid);
    for (uint i = lo; i < hi; i++) {
      device const half4 *v = vr + (size_t)i * (width / 4);
      float4 v0 = float4(v[0]), v1 = float4(v[1]);
      for (uint h = 0; h < G; h++) {
        float w = ps[h][i];
        acc[h][0] += w * v0, acc[h][1] += w * v1;
      }
    }
    threadgroup float4(*vs)[2][64] = (threadgroup float4(*)[2][64])stage;  // [8 simdgroups][2 heads][256 values]
    for (uint h0 = 0; h0 < G; h0 += 2) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint k = 0; k < 2 && h0 + k < G; k++) vs[sg][k][lane * 2] = acc[h0 + k][0], vs[sg][k][lane * 2 + 1] = acc[h0 + k][1];
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint k = 0; k < 2 && h0 + k < G; k++) {
        float sum = 0;
        for (uint s2 = 0; s2 < BLK / 32; s2++) sum += ((threadgroup float *)vs[s2][k])[j];
        part[(first + (h0 + k) * p.blocks) * p.dim + j] = sum;
      }
    }
  }
  if (j < G) {
    float l = 0;
    for (uint s = 0; s < BLK / 32; s++) l += red[j][s];
    ml[first + j * p.blocks] = float2(mh[j], l);
  }
}

// Row g.y's head g.x: the blocks' partial sums combined, times sigmoid of the
// head's gate (qg: rows of heads * 2 * dim, the gate after each head's query).
kernel void attend_combine(device const float *part [[buffer(0)]], device const float2 *ml [[buffer(1)]],
                           device const float *qg [[buffer(2)]], device float *out [[buffer(3)]],
                           constant Seq *rows [[buffer(4)]], constant Att &p [[buffer(5)]],
                           uint2 g [[threadgroup_position_in_grid]], uint d [[thread_index_in_threadgroup]]) {
  uint nb = rows[g.y].pos / BLK + 1;
  size_t r = (size_t)g.y * p.heads + g.x;
  float m = -INFINITY;
  for (uint b = 0; b < nb; b++) m = max(m, ml[r * p.blocks + b].x);
  float l = 0, acc = 0;
  for (uint b = 0; b < nb; b++) {
    float2 v = ml[r * p.blocks + b];
    float w = exp(v.x - m);
    l += v.y * w;
    acc += part[(r * p.blocks + b) * p.dim + d] * w;
  }
  out[r * p.dim + d] = acc / l * sigm(qg[r * 2 * p.dim + p.dim + d]);
}

// ---- attention of a prompt pass, on the matrix units ----

struct AttP { uint heads, kv, n, p0, layer, cap, rows, pad; float scale; uint pad0, pad1, pad2; };  // rows: n padded to 32

// The pass's queries (FP32 rows of heads * 256) as FP16, by head: qh[head][row][256], rows n.. zero.
kernel void q_half(device const float *q [[buffer(0)]], device half *qh [[buffer(1)]], constant AttP &p [[buffer(2)]],
                   uint3 g [[thread_position_in_grid]]) {
  uint d = g.x, r = g.y, h = g.z;
  if (d >= 256 || r >= p.rows) return;
  qh[((size_t)h * p.rows + r) * 256 + d] = r < p.n ? half(q[((size_t)r * p.heads + h) * 256 + d]) : half(0);
}

// Attention of query head g.x over 32 consecutive rows of one sequence,
// rows 32 g.y on (at positions p0 on), with the matrix units: per chunk of 32
// keys counted from position 0, the scores (FP16 queries and keys, FP32
// sums), an online softmax per row, and the probabilities (FP16) times the
// values; the output times sigmoid of the head's gate (qg: rows of heads *
// 512, the gate after each head's query). A row's result depends only on its
// position and the keys and values up to it.
kernel void attend_prompt(device const half *qh [[buffer(0)]], constant Seq *rows [[buffer(1)]],
                          device const float *qg [[buffer(3)]], device float *out [[buffer(4)]],
                          constant AttP &p [[buffer(5)]], uint2 g [[threadgroup_position_in_grid]],
                          uint t [[thread_index_in_threadgroup]]) {
  device const half *K = rows[0].k, *V = rows[0].v;
  threadgroup half kv[32 * 128], ps[32 * 32];
  threadgroup float ss[32 * 32], mx[32], sum[32], rescale[32];
  uint h = g.x, r0 = g.y * 32, kvh = h / (p.heads / p.kv), width = p.kv * 256;
  uint plast = p.p0 + min(r0 + 32, p.n) - 1;  // the tile's last row's position
  size_t lay = (size_t)p.layer * p.cap * width + kvh * 256;
  if (t < 32) mx[t] = -1e30f, sum[t] = 0;
  constexpr auto sdesc = matmul2d_descriptor(32, 32, 128, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  constexpr auto odesc = matmul2d_descriptor(32, 128, 32, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<sdesc, execution_simdgroups<8>> sop;
  matmul2d<odesc, execution_simdgroups<8>> oop;
  tensor<device half, dextents<int32_t, 2>, tensor_inline> tq((device half *)qh + ((size_t)h * p.rows + r0) * 256, dextents<int32_t, 2>(256, 32));
  tensor<threadgroup half, extents<int32_t, 128, 32>, tensor_inline> tk(kv, extents<int32_t, 128, 32>());
  tensor<threadgroup half, extents<int32_t, 32, 32>, tensor_inline> tp(ps, extents<int32_t, 32, 32>());
  auto acc0 = oop.get_destination_cooperative_tensor<decltype(tp), decltype(tk), float>();
  auto acc1 = oop.get_destination_cooperative_tensor<decltype(tp), decltype(tk), float>();
  for (uint16_t i = 0; i < acc0.get_capacity(); i++)
    if (acc0.is_valid_element(i)) acc0[i] = acc1[i] = 0;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint c0 = 0; c0 <= plast; c0 += 32) {
    auto sc = sop.get_destination_cooperative_tensor<decltype(tq.slice(0, 0)), decltype(tk), float>();
    for (uint16_t i = 0; i < sc.get_capacity(); i++)
      if (sc.is_valid_element(i)) sc[i] = 0;
    for (uint ds = 0; ds < 2; ds++) {  // scores: the chunk's keys, 128 columns at a time
      for (uint i = t; i < 32 * 32; i += 256) {  // 32 keys x 32 half4
        uint j = i / 32, c = i % 32, at = c0 + j;
        *(threadgroup half4 *)(kv + j * 128 + c * 4) =
            at <= plast ? *(device const half4 *)(K + lay + (size_t)at * width + ds * 128 + c * 4) : half4(0);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      auto qd = tq.slice(ds * 128, 0);
      sop.run(qd, tk, sc);
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint16_t i = 0; i < sc.get_capacity(); i++)
      if (sc.is_valid_element(i)) {
        auto a = sc.get_multidimensional_index(i);  // a[0]: the key, a[1]: the row
        uint row = r0 + a[1];
        bool valid = row < p.n && c0 + a[0] <= p.p0 + row;
        ss[a[1] * 32 + a[0]] = valid ? sc[i] * p.scale : -INFINITY;
      }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t < 32) {  // row t's online softmax over the chunk
      float m = mx[t];
      for (uint j = 0; j < 32; j++) m = max(m, ss[t * 32 + j]);
      float f = exp(mx[t] - m), s2 = sum[t] * f;
      for (uint j = 0; j < 32; j++) {
        float e = exp(ss[t * 32 + j] - m);
        s2 += e;
        ps[t * 32 + j] = half(e);
      }
      mx[t] = m, sum[t] = s2, rescale[t] = f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint16_t i = 0; i < acc0.get_capacity(); i++)
      if (acc0.is_valid_element(i)) {
        float f = rescale[acc0.get_multidimensional_index(i)[1]];
        acc0[i] *= f, acc1[i] *= f;
      }
    for (uint ds = 0; ds < 2; ds++) {  // values: the same keys' rows, 128 columns at a time
      for (uint i = t; i < 32 * 32; i += 256) {
        uint j = i / 32, c = i % 32, at = c0 + j;
        *(threadgroup half4 *)(kv + j * 128 + c * 4) =
            at <= plast ? *(device const half4 *)(V + lay + (size_t)at * width + ds * 128 + c * 4) : half4(0);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (ds == 0) oop.run(tp, tk, acc0);
      else oop.run(tp, tk, acc1);
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
  for (uint16_t i = 0; i < acc0.get_capacity(); i++)
    if (acc0.is_valid_element(i)) {
      auto a = acc0.get_multidimensional_index(i);  // a[0]: the column, a[1]: the row
      uint row = r0 + a[1];
      if (row >= p.n) continue;
      size_t r = (size_t)row * p.heads + h;
      for (uint ds = 0; ds < 2; ds++) {
        uint d = ds * 128 + a[0];
        float v = (ds ? acc1[i] : acc0[i]) / sum[a[1]];
        out[r * 256 + d] = v * sigm(qg[r * 512 + 256 + d]);
      }
    }
}

// Row g.y of y: a's row then b's (n values each).
kernel void concat(device const float *a [[buffer(0)]], device const float *b [[buffer(1)]], device float *y [[buffer(2)]],
                   constant uint &n [[buffer(3)]], uint2 g [[thread_position_in_grid]]) {
  if (g.x < 2 * n) y[(size_t)g.y * 2 * n + g.x] = g.x < n ? a[(size_t)g.y * n + g.x] : b[(size_t)g.y * n + g.x - n];
}

// h = silu(g) * u
kernel void swiglu(device const float *g [[buffer(0)]], device const float *u [[buffer(1)]],
                   device float *h [[buffer(2)]], constant uint &n [[buffer(3)]], uint i [[thread_position_in_grid]]) {
  if (i < n) h[i] = g[i] * sigm(g[i]) * u[i];
}

// x += r
kernel void add(device float *x [[buffer(0)]], device const float *r [[buffer(1)]], constant uint &n [[buffer(2)]],
                uint i [[thread_position_in_grid]]) {
  if (i < n) x[i] += r[i];
}
