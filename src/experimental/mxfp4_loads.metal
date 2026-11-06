// Alternative load schedules for the scalar MXFP4 experts. The arithmetic,
// lane assignment and reduction order match kernels.metal.
struct Fp4Load4 { uint4 q[4]; float scale[4]; };

inline Fp4Load4 fp4_load4(device const uchar *w, device const uchar *s, uint stride, uint c) {
  Fp4Load4 b;
  for (uint r = 0; r < 4; r++) {
    b.q[r] = ((device const uint4 *)(w + r * stride))[c];
    b.scale[r] = e8m0(s[r * (stride / 16) + c]);
  }
  return b;
}

inline void fp4_acc4(thread const Fp4Load4 &q, thread const float4 *x, thread float4 &acc) {
  for (uint r = 0; r < 4; r++) {
    float4 a, b, d = 0;
    fp4x8(q.q[r].x, a, b); d += a * x[0] + b * x[1];
    fp4x8(q.q[r].y, a, b); d += a * x[2] + b * x[3];
    fp4x8(q.q[r].z, a, b); d += a * x[4] + b * x[5];
    fp4x8(q.q[r].w, a, b); d += a * x[6] + b * x[7];
    acc[r] += (d.x + d.y + d.z + d.w) * q.scale[r];
  }
}

template <bool Ahead>
inline float4 fp4_dot4_loads(device const uchar *w, device const uchar *s, uint stride,
                            device const float4 *x, uint chunks, uint lane) {
  float4 acc = 0;
  Fp4Load4 pending = fp4_load4(w, s, stride, lane);
  for (uint c = lane; c < chunks; c += 32) {
    Fp4Load4 current = pending;
    if (Ahead && c + 32 < chunks) pending = fp4_load4(w, s, stride, c + 32);
    float4 xv[8];
    for (uint i = 0; i < 8; i++) xv[i] = x[8 * c + i];
    fp4_acc4(current, xv, acc);
    if (!Ahead && c + 32 < chunks) pending = fp4_load4(w, s, stride, c + 32);
  }
  return acc;
}

template <bool Ahead>
inline void moe_up_loads(constant Expert *ex, device const float *x, device float *h,
                         uint2 tg, uint sg, uint lane, threadgroup float *block) {
  if (!ex[tg.y].p[0]) return;
  uint row = (tg.x * 8 + sg) * 4, e = tg.y;
  constexpr uint K = 5120, W = K / 2 * 2304, S = K / 32 * 2304;
  device const uchar *w = ex[e].p[0] + row * (K / 2), *s = ex[e].p[1] + row * (K / 32);
  device const float4 *xv = (device const float4 *)x;
  float4 a = 0, b = 0;
  Fp4Load4 gate = fp4_load4(w, s, K / 2, lane);
  Fp4Load4 up = fp4_load4(w + 2 * W, s + 2 * S, K / 2, lane);
  for (uint c = lane; c < K / 32; c += 32) {
    Fp4Load4 g = gate, u = up;
    if (Ahead && c + 32 < K / 32) {
      gate = fp4_load4(w, s, K / 2, c + 32);
      up = fp4_load4(w + 2 * W, s + 2 * S, K / 2, c + 32);
    }
    float4 xc[8];
    for (uint i = 0; i < 8; i++) xc[i] = xv[8 * c + i];
    fp4_acc4(g, xc, a);
    fp4_acc4(u, xc, b);
    if (!Ahead && c + 32 < K / 32) {
      gate = fp4_load4(w, s, K / 2, c + 32);
      up = fp4_load4(w + 2 * W, s + 2 * S, K / 2, c + 32);
    }
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

template <bool Ahead>
inline void moe_down_loads(constant Expert *ex, device const float *h, device const float *shared_out,
                           device const float *mix, device const bfloat *res, device bfloat *out,
                           uint n, uint pm, device float *parts, uint tg, uint sg, uint lane,
                           threadgroup float4 *part) {
  if (!pm && !ex[0].p[0]) return;
  if (pm && !ex[sg].p[0]) return;
  uint d = tg * 4;
  constexpr uint K = 2304, W = 5120 * K / 2, S = 5120 * K / 32;
  float4 a = fp4_dot4_loads<Ahead>(ex[sg].p[0] + W + d * (K / 2), ex[sg].p[1] + S + d * (K / 32), K / 2,
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

#define FP4_LOAD_KERNELS(NAME, AHEAD) \
kernel void moe_up_##NAME(constant Expert *ex [[buffer(0)]], device const float *x [[buffer(1)]], \
                          device float *h [[buffer(2)]], uint2 tg [[threadgroup_position_in_grid]], \
                          uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
  threadgroup float block[32]; \
  moe_up_loads<AHEAD>(ex, x, h, tg, sg, lane, block); \
} \
kernel void moe_down_##NAME(constant Expert *ex [[buffer(0)]], device const float *h [[buffer(1)]], \
                            device const float *shared_out [[buffer(2)]], device const float *mix [[buffer(3)]], \
                            device const bfloat *res [[buffer(4)]], device bfloat *out [[buffer(5)]], \
                            constant uint &n [[buffer(6)]], constant uint &pm [[buffer(7)]], device float *parts [[buffer(8)]], \
                            uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]], \
                            uint lane [[thread_index_in_simdgroup]]) { \
  threadgroup float4 part[8]; \
  moe_down_loads<AHEAD>(ex, h, shared_out, mix, res, out, n, pm, parts, tg, sg, lane, part); \
}
FP4_LOAD_KERNELS(early, false)
FP4_LOAD_KERNELS(pipeline, true)
#undef FP4_LOAD_KERNELS

// One token/expert pair per tile. Store its result in the scalar path's
// expert-sorted partials so the existing final reduction remains unchanged.
kernel void moe_down_single_m(device const MoeTile *tiles [[buffer(0)]], device const half *xh [[buffer(1)]],
                               device float *parts [[buffer(2)]], constant uint &npairs [[buffer(3)]],
                               uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]]) {
  constexpr uint K = 2304, W = 5120 * K / 2, S = 5120 * K / 32;
  MOE_SETUP(K)
  uint n0 = tg.y * 128 + 32 * sg;
  Mxfp4 w2 = mxfp4(tl.p[0] + W, tl.p[1] + S, K, 5120);
  auto c = op.get_destination_cooperative_tensor<decltype(tx.slice(0, 0)), decltype(w2.slice(0, 0)), float>();
  for (uint16_t i = 0; i < c.get_capacity(); i++)
    if (c.is_valid_element(i)) c[i] = 0;
  for (uint k0 = 0; k0 < K; k0 += MOE_K) {
    auto x = tx.slice(k0, tl.start);
    auto w = w2.slice(k0, n0);
    op.run(x, w, c);
  }
  for (uint16_t i = 0; i < c.get_capacity(); i++)
    if (c.is_valid_element(i)) {
      auto at = c.get_multidimensional_index(i);
      if (uint(at[1]) < tl.count) {
        uint pair = tl.start + at[1];
        parts[((pair / 6) * 5120 + n0 + at[0]) * 8 + pair % 6] = bf(c[i] * 0x1p-7f);
      }
    }
}

// Moving an exact power-of-two factor from each decoded code to its block
// scale removes repeated vector multiplies. Very large scale codes retain
// the original path so the adjusted scale never overflows.
inline void fp4x8_unscaled(uint u, thread float4 &a, thread float4 &b) {
  uint lo = u & 0x0F0F0F0Fu, hi = (u >> 4) & 0x0F0F0F0Fu;
  half2 h0 = e2m1x2((lo & 0xFFu) | ((hi & 0xFFu) << 16));
  half2 h1 = e2m1x2(((lo >> 8) & 0xFFu) | (((hi >> 8) & 0xFFu) << 16));
  half2 h2 = e2m1x2(((lo >> 16) & 0xFFu) | (((hi >> 16) & 0xFFu) << 16));
  half2 h3 = e2m1x2((lo >> 24) | ((hi >> 24) << 16));
  a = float4(float2(h0), float2(h1));
  b = float4(float2(h2), float2(h3));
}
inline void fp4_chunk4_factor(device const uchar *w, device const uchar *s, uint stride,
                              thread const float4 *xv, uint c, thread float4 &acc) {
  for (uint r = 0; r < 4; r++) {
    uint4 q = ((device const uint4 *)(w + r * stride))[c];
    uchar code = s[r * (stride / 16) + c];
    float4 a, b, d = 0;
    if (code <= 240) {
      fp4x8_unscaled(q.x, a, b); d += a * xv[0] + b * xv[1];
      fp4x8_unscaled(q.y, a, b); d += a * xv[2] + b * xv[3];
      fp4x8_unscaled(q.z, a, b); d += a * xv[4] + b * xv[5];
      fp4x8_unscaled(q.w, a, b); d += a * xv[6] + b * xv[7];
      acc[r] += (d.x + d.y + d.z + d.w) * (code ? e8m0(code + 14) : 0);
    } else {
      fp4x8(q.x, a, b); d += a * xv[0] + b * xv[1];
      fp4x8(q.y, a, b); d += a * xv[2] + b * xv[3];
      fp4x8(q.z, a, b); d += a * xv[4] + b * xv[5];
      fp4x8(q.w, a, b); d += a * xv[6] + b * xv[7];
      acc[r] += (d.x + d.y + d.z + d.w) * e8m0(code);
    }
  }
}
inline void fp4_chunk4_row(device const uchar *w, device const uchar *s, uint stride,
                           thread const float4 *xv, uint c, thread float4 &acc) {
  uint4 q = ((device const uint4 *)w)[c];
  float scale = e8m0(s[c]);
  for (uint r = 0; r < 4; r++) {
    uint4 next = 0;
    float next_scale = 0;
    if (r + 1 < 4) {
      next = ((device const uint4 *)(w + (r + 1) * stride))[c];
      next_scale = e8m0(s[(r + 1) * (stride / 16) + c]);
    }
    float4 a, b, d = 0;
    fp4x8(q.x, a, b); d += a * xv[0] + b * xv[1];
    fp4x8(q.y, a, b); d += a * xv[2] + b * xv[3];
    fp4x8(q.z, a, b); d += a * xv[4] + b * xv[5];
    fp4x8(q.w, a, b); d += a * xv[6] + b * xv[7];
    acc[r] += (d.x + d.y + d.z + d.w) * scale;
    q = next, scale = next_scale;
  }
}

inline float4 fp4_dot4_row(device const uchar *w, device const uchar *s, uint stride,
                       device const float4 *x, uint chunks, uint lane) {
  float4 acc = 0;
  for (uint c = lane; c < chunks; c += 32) {
    float4 xv[8];
    for (uint i = 0; i < 8; i++) xv[i] = x[8 * c + i];
    fp4_chunk4_row(w, s, stride, xv, c, acc);
  }
  return acc;
}

kernel void moe_up_row(constant Expert *ex [[buffer(0)]], device const float *x [[buffer(1)]],
                   device float *h [[buffer(2)]], uint2 tg [[threadgroup_position_in_grid]],
                   uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float block[32];
  if (!ex[tg.y].p[0]) return;  // a null entry (see resolve)
  uint row = (tg.x * 8 + sg) * 4, e = tg.y;
  constexpr uint K = 5120, W = K / 2 * 2304, S = K / 32 * 2304;
  device const uchar *w = ex[e].p[0] + row * (K / 2), *s = ex[e].p[1] + row * (K / 32);
  device const float4 *xv = (device const float4 *)x;
  float4 a = 0, b = 0;
  for (uint c = lane; c < K / 32; c += 32) {
    float4 xc[8];
    for (uint i = 0; i < 8; i++) xc[i] = xv[8 * c + i];
    fp4_chunk4_row(w, s, K / 2, xc, c, a);
    fp4_chunk4_row(w + 2 * W, s + 2 * S, K / 2, xc, c, b);
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

kernel void moe_down_row(constant Expert *ex [[buffer(0)]], device const float *h [[buffer(1)]],
                     device const float *shared_out [[buffer(2)]], device const float *mix [[buffer(3)]],
                     device const bfloat *res [[buffer(4)]], device bfloat *out [[buffer(5)]],
                     constant uint &n [[buffer(6)]], constant uint &pm [[buffer(7)]], device float *parts [[buffer(8)]],
                     uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float4 part[8];
  if (!pm && !ex[0].p[0]) return;  // null entries (see resolve)
  if (pm && !ex[sg].p[0]) return;
  uint d = tg * 4;
  constexpr uint K = 2304, W = 5120 * 2304 / 2, S = 5120 * 2304 / 32;
  float4 a = fp4_dot4_row(ex[sg].p[0] + W + d * (K / 2), ex[sg].p[1] + S + d * (K / 32), K / 2,
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

inline float4 fp4_dot4_factor(device const uchar *w, device const uchar *s, uint stride,
                       device const float4 *x, uint chunks, uint lane) {
  float4 acc = 0;
  for (uint c = lane; c < chunks; c += 32) {
    float4 xv[8];
    for (uint i = 0; i < 8; i++) xv[i] = x[8 * c + i];
    fp4_chunk4_factor(w, s, stride, xv, c, acc);
  }
  return acc;
}

kernel void moe_up_factor(constant Expert *ex [[buffer(0)]], device const float *x [[buffer(1)]],
                   device float *h [[buffer(2)]], uint2 tg [[threadgroup_position_in_grid]],
                   uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float block[32];
  if (!ex[tg.y].p[0]) return;  // a null entry (see resolve)
  uint row = (tg.x * 8 + sg) * 4, e = tg.y;
  constexpr uint K = 5120, W = K / 2 * 2304, S = K / 32 * 2304;
  device const uchar *w = ex[e].p[0] + row * (K / 2), *s = ex[e].p[1] + row * (K / 32);
  device const float4 *xv = (device const float4 *)x;
  float4 a = 0, b = 0;
  for (uint c = lane; c < K / 32; c += 32) {
    float4 xc[8];
    for (uint i = 0; i < 8; i++) xc[i] = xv[8 * c + i];
    fp4_chunk4_factor(w, s, K / 2, xc, c, a);
    fp4_chunk4_factor(w + 2 * W, s + 2 * S, K / 2, xc, c, b);
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

kernel void moe_down_factor(constant Expert *ex [[buffer(0)]], device const float *h [[buffer(1)]],
                     device const float *shared_out [[buffer(2)]], device const float *mix [[buffer(3)]],
                     device const bfloat *res [[buffer(4)]], device bfloat *out [[buffer(5)]],
                     constant uint &n [[buffer(6)]], constant uint &pm [[buffer(7)]], device float *parts [[buffer(8)]],
                     uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float4 part[8];
  if (!pm && !ex[0].p[0]) return;  // null entries (see resolve)
  if (pm && !ex[sg].p[0]) return;
  uint d = tg * 4;
  constexpr uint K = 2304, W = 5120 * 2304 / 2, S = 5120 * 2304 / 32;
  float4 a = fp4_dot4_factor(ex[sg].p[0] + W + d * (K / 2), ex[sg].p[1] + S + d * (K / 32), K / 2,
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

// Eight-row native tensor tiles reduce padding in single-pair probes.
constexpr constant auto moe_small8_desc = matmul2d_descriptor(8, 32, MOE_K, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
kernel void moe_up_small8(device const MoeTile *tiles [[buffer(0)]], device const half *xh [[buffer(1)]],
                     device const float *pw [[buffer(2)]], device float *h [[buffer(3)]],
                     constant uint &npairs [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
                     uint sg [[simdgroup_index_in_threadgroup]]) {
  constexpr uint K = 5120, W = K / 2 * 2304, S = K / 32 * 2304;
  matmul2d<moe_small8_desc, execution_simdgroups<1>> op;
  tensor<device half, dextents<int32_t, 2>, tensor_inline> tx((device half *)xh, dextents<int32_t, 2>(K, npairs + 64));
  MoeTile tl = tiles[tg.x];
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
  threadgroup float acc[2][8 * 32];  // w3's sums: pair r's column c at 32 r + c
  tensor<threadgroup float, extents<int32_t, 32, 8>, tensor_inline> ta(acc[j], extents<int32_t, 32, 8>());
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


kernel void moe_down_small8(device const MoeTile *tiles [[buffer(0)]], device const half *xh [[buffer(1)]],
                               device float *parts [[buffer(2)]], constant uint &npairs [[buffer(3)]],
                               uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]]) {
  constexpr uint K = 2304, W = 5120 * K / 2, S = 5120 * K / 32;
  matmul2d<moe_small8_desc, execution_simdgroups<1>> op;
  tensor<device half, dextents<int32_t, 2>, tensor_inline> tx((device half *)xh, dextents<int32_t, 2>(K, npairs + 64));
  MoeTile tl = tiles[tg.x];
  uint n0 = tg.y * 128 + 32 * sg;
  Mxfp4 w2 = mxfp4(tl.p[0] + W, tl.p[1] + S, K, 5120);
  auto c = op.get_destination_cooperative_tensor<decltype(tx.slice(0, 0)), decltype(w2.slice(0, 0)), float>();
  for (uint16_t i = 0; i < c.get_capacity(); i++)
    if (c.is_valid_element(i)) c[i] = 0;
  for (uint k0 = 0; k0 < K; k0 += MOE_K) {
    auto x = tx.slice(k0, tl.start);
    auto w = w2.slice(k0, n0);
    op.run(x, w, c);
  }
  for (uint16_t i = 0; i < c.get_capacity(); i++)
    if (c.is_valid_element(i)) {
      auto at = c.get_multidimensional_index(i);
      if (uint(at[1]) < tl.count) {
        uint pair = tl.start + at[1];
        parts[((pair / 6) * 5120 + n0 + at[0]) * 8 + pair % 6] = bf(c[i] * 0x1p-7f);
      }
    }
}
