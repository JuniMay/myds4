// Same scalar MXFP4 expansion, reading FP8 codes and per-32 scales.
kernel void moe_up_packed(constant Expert *ex [[buffer(0)]], device const uchar *x [[buffer(1)]],
                          device const uchar *scale [[buffer(2)]], device float *h [[buffer(3)]],
                          uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float block[32];
  uint row = (tg.x * 8 + sg) * 4, e = tg.y;
  constexpr uint K = 5120, W = K / 2 * 2304, S = K / 32 * 2304;
  device const uchar *w = ex[e].p[0] + row * (K / 2), *s = ex[e].p[1] + row * (K / 32);
  float4 a = 0, b = 0;
  for (uint c = lane; c < K / 32; c += 32) {
    float4 xc[8];
    float xs = e8m0(scale[c]) * 256.0f;
    for (uint i = 0; i < 8; i++) xc[i] = fp8x4(((device const uint *)x)[8 * c + i]) * xs;
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
