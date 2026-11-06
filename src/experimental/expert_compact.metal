// Partial batches keep the ordinary arithmetic and original expert slots.
kernel void moe_up_active(constant Expert *ex [[buffer(0)]], device const float *x [[buffer(1)]],
                          device float *h [[buffer(2)]], uint2 tg [[threadgroup_position_in_grid]],
                          uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float block[32];
  constant uint *active = (constant uint *)(ex + 6);
  uint row = (tg.x * 8 + sg) * 4, e = active[tg.y];
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

kernel void moe_down_active(constant Expert *ex [[buffer(0)]], device const float *h [[buffer(1)]],
                            device float *parts [[buffer(2)]], uint tg [[threadgroup_position_in_grid]],
                            uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  constant uint *active = (constant uint *)(ex + 6);
  uint e = active[sg], d = tg * 4;
  constexpr uint K = 2304, W = 5120 * 2304 / 2, S = 5120 * 2304 / 32;
  float4 a = fp4_dot4(ex[e].p[0] + W + d * (K / 2), ex[e].p[1] + S + d * (K / 32), K / 2,
                      (device const float4 *)(h + e * K), K / 32, lane);
  a = float4(bf(simd_sum(a.x)), bf(simd_sum(a.y)), bf(simd_sum(a.z)), bf(simd_sum(a.w)));
  if (lane == 0)
    for (uint r = 0; r < 4; r++) parts[(d + r) * 8 + e] = a[r];
}
