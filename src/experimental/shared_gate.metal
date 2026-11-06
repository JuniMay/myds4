// Shared-expert gate/up fusion for one decoded row. Keep mv_fp8's four
// split sums, BF16 rounding, SwiGLU, and per-32 FP8 activation rounding.
// The schedules vary the partial sums' placement and the rounding dispatch.

inline void shared_chunk(device const uchar *w, device const uchar *s,
                         device const float4 *x, uint row, uint c,
                         thread float4 &acc) {
  constexpr uint K = 5120, chunks = K / 16;
  device const uint4 *wr = (device const uint4 *)(w + ulong(row) * K);
  uint4 q[4];
  for (uint r = 0; r < 4; r++) q[r] = wr[r * chunks + c];
  float sc = e8m0(s[(row / 32) * (K / 32) + c / 2]) * 256.0f;
  float4 x0 = x[4 * c], x1 = x[4 * c + 1], x2 = x[4 * c + 2], x3 = x[4 * c + 3];
  for (uint r = 0; r < 4; r++) {
    float4 d = fp8x4(q[r].x) * x0 + fp8x4(q[r].y) * x1 + fp8x4(q[r].z) * x2 + fp8x4(q[r].w) * x3;
    acc[r] += (d.x + d.y + d.z + d.w) * sc;
  }
}

inline float4 shared_sum(float4 x) {
  return float4(simd_sum(x.x), simd_sum(x.y), simd_sum(x.z), simd_sum(x.w));
}

inline float shared_activation(float g, float u) {
  float a = min(bf(g), 10.0f), b = clamp(bf(u), -10.0f, 10.0f);
  return bf(a / (1.0f + exp(-a)) * b);
}

kernel void shared_gate_fp8(device const uchar *wg [[buffer(0)]], device const uchar *sgate [[buffer(1)]],
                            device const uchar *wu [[buffer(2)]], device const uchar *sup [[buffer(3)]],
                            device const float4 *x [[buffer(4)]], device float *h [[buffer(5)]],
                            uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float block[32];
  uint row = (tg * 8 + sg) * 4;
  float4 gate[4] = {0, 0, 0, 0}, up[4] = {0, 0, 0, 0};
  for (uint split = 0; split < 4; split++) {
    for (uint c = split * 32 + lane; c < 320; c += 128) {
      shared_chunk(wg, sgate, x, row, c, gate[split]);
      shared_chunk(wu, sup, x, row, c, up[split]);
    }
    gate[split] = shared_sum(gate[split]);
    up[split] = shared_sum(up[split]);
  }
  for (uint split = 1; split < 4; split++) gate[0] += gate[split], up[0] += up[split];
  if (lane < 4) block[sg * 4 + lane] = shared_activation(gate[0][lane], up[0][lane]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) h[tg * 32 + lane] = fp8_act(block[lane]);
}

kernel void shared_gate_fp8_wide(device const uchar *wg [[buffer(0)]], device const uchar *sgate [[buffer(1)]],
                                 device const uchar *wu [[buffer(2)]], device const uchar *sup [[buffer(3)]],
                                 device const float4 *x [[buffer(4)]], device float *h [[buffer(5)]],
                                 uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float4 gate[32], up[32];
  threadgroup float block[32];
  uint row = (tg * 8 + sg / 4) * 4;
  float4 a = 0, b = 0;
  for (uint c = (sg % 4) * 32 + lane; c < 320; c += 128) {
    shared_chunk(wg, sgate, x, row, c, a);
    shared_chunk(wu, sup, x, row, c, b);
  }
  a = shared_sum(a), b = shared_sum(b);
  if (lane == 0) gate[sg] = a, up[sg] = b;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg % 4 == 0) {
    for (uint split = 1; split < 4; split++) a += gate[sg + split], b += up[sg + split];
    if (lane < 4) block[sg + lane] = shared_activation(a[lane], b[lane]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) h[tg * 32 + lane] = fp8_act(block[lane]);
}

// Four output values per threadgroup retain more independent work groups.
// FP8 rounding follows in act_quant, after all 32 values of each block exist.
kernel void shared_gate_fp8_quad(device const uchar *wg [[buffer(0)]], device const uchar *sgate [[buffer(1)]],
                                 device const uchar *wu [[buffer(2)]], device const uchar *sup [[buffer(3)]],
                                 device const float4 *x [[buffer(4)]], device float *h [[buffer(5)]],
                                 uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float4 gate[4], up[4];
  uint row = tg * 4;
  float4 a = 0, b = 0;
  for (uint c = sg * 32 + lane; c < 320; c += 128) {
    shared_chunk(wg, sgate, x, row, c, a);
    shared_chunk(wu, sup, x, row, c, b);
  }
  a = shared_sum(a), b = shared_sum(b);
  if (lane == 0) gate[sg] = a, up[sg] = b;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    for (uint split = 1; split < 4; split++) a += gate[split], b += up[split];
    if (lane < 4) h[row + lane] = shared_activation(a[lane], b[lane]);
  }
}
