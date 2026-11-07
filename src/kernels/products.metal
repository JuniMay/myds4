// Matrix products for weights with one scale per matrix, stored as NVFP4
// (E2M1 codes, two a byte with the low nibble first, an E4M3 scale per 16
// values), FP8 E4M3 or BF16 (the template parameter F: 0, 1, 2). Activations
// are FP32 rows. A decoding pass runs up to TM rows at once; a row's
// arithmetic is the same whatever the number of rows, so its result does not
// depend on the pass.

#define TM 8  // most token rows of a decoding pass

struct Mv { uint rows, cols, n; float scale; };  // n: token rows; y = scale * W x per row

// Sixteen values of a 16-value NVFP4 chunk, times its scale s, as four float4.
inline void nvfp4_chunk(uint2 c, float s, thread float4 *w) {
  w[0] = float4(fp4(c.x & 15), fp4((c.x >> 4) & 15), fp4((c.x >> 8) & 15), fp4((c.x >> 12) & 15)) * s;
  w[1] = float4(fp4((c.x >> 16) & 15), fp4((c.x >> 20) & 15), fp4((c.x >> 24) & 15), fp4(c.x >> 28)) * s;
  w[2] = float4(fp4(c.y & 15), fp4((c.y >> 4) & 15), fp4((c.y >> 8) & 15), fp4((c.y >> 12) & 15)) * s;
  w[3] = float4(fp4((c.y >> 16) & 15), fp4((c.y >> 20) & 15), fp4((c.y >> 24) & 15), fp4(c.y >> 28)) * s;
}
// Four E4M3 codes as their values.
inline float4 e4m3x4(uint u) { return float4(fp8(u & 255), fp8((u >> 8) & 255), fp8((u >> 16) & 255), fp8(u >> 24)); }

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
  w[0] = e4m3x4(q.x), w[1] = e4m3x4(q.y), w[2] = e4m3x4(q.z), w[3] = e4m3x4(q.w);
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
// weights it decodes serving all of them). S: NVFP4's E4M3 scales per 16.
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
// A prompt pass takes the matrices through the GPU's matrix units (Metal
// tensor operations) with BF16 operands and FP32 sums: x rounded to BF16
// (x_bf16), the weights dequantized into BF16 tiles, where they are exact
// (E2M1 times an E4M3 scale has at most 6 significant bits, E4M3 4, BF16 is
// BF16), the matrix's own scale applied at the end. A threadgroup of 4
// simdgroups takes 128 rows of x (two 64-row products) and 64 weight rows; a
// K step takes 64 columns, the next step's weights loaded while this step's
// products run (as mm_fp8 does for block-scaled FP8). The units add in an order that does not
// depend on the tile or on the number of rows, so a row's result is the same
// however a prompt is split into passes.

struct Gemm { uint m, n, k, pad; float scale; uint pad0, pad1, pad2; };  // m rows of x (k each), n weight rows

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
    for (uint b = 0; b < 8; b++) d[b] = bfloat4(e4m3x4(u[b]));
  } else {
    threadgroup uint4 *e = (threadgroup uint4 *)d;
    for (uint i = 0; i < 4; i++) e[i] = q[i];
  }
}

// y[t][r] = scale * sum_c W[r][c] x[t][c] for the m rows of x (BF16, rows
// of k), y's rows of n.
template <int F>
kernel void gemm(device const uchar *W [[buffer(0)]], device const uchar *S [[buffer(1)]],
               device const bfloat *x [[buffer(2)]], device float *y [[buffer(3)]], constant Gemm &p [[buffer(4)]],
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
typedef decltype(gemm<0>) gemm_t;
template [[host_name("gemm_nvfp4")]] kernel gemm_t gemm<0>;
template [[host_name("gemm_fp8")]] kernel gemm_t gemm<1>;
template [[host_name("gemm_bf16")]] kernel gemm_t gemm<2>;
