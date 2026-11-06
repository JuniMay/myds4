// Preserve the scalar decoder's arithmetic while varying the physical
// distance between dense weight rows. stride is measured in weight elements.
struct PaddedMV { MV mv; uint stride; };

kernel void mv_fp8_padded(device const uchar *w [[buffer(0)]], device const uchar *s [[buffer(1)]],
                          device const float *x [[buffer(2)]], device float *y [[buffer(3)]],
                          constant PaddedMV &p [[buffer(4)]], uint tg [[threadgroup_position_in_grid]],
                          uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float4 part[8];
  constant MV &v = p.mv;
  uint row = (tg * (8 / v.split) + sg / v.split) * 4, chunks = v.k / 16;
  float4 acc = 0;
  if (row < v.n) {
    device const uint4 *wr = (device const uint4 *)(w + (ulong)row * p.stride);
    device const uchar *sr = s + (row / 32) * (v.k / 32);
    device const float4 *xr = (device const float4 *)(x + (row / v.group_rows) * v.group_k);
    for (uint c = (sg % v.split) * 32 + lane; c < chunks; c += 32 * v.split) {
      uint4 q[4];
      for (uint r = 0; r < 4; r++) q[r] = wr[r * (p.stride / 16) + c];
      float sc = e8m0(sr[c / 2]) * 256.0f;
      float4 x0 = xr[4 * c], x1 = xr[4 * c + 1], x2 = xr[4 * c + 2], x3 = xr[4 * c + 3];
      for (uint r = 0; r < 4; r++) {
        float4 d = fp8x4(q[r].x) * x0 + fp8x4(q[r].y) * x1 + fp8x4(q[r].z) * x2 + fp8x4(q[r].w) * x3;
        acc[r] += (d.x + d.y + d.z + d.w) * sc;
      }
    }
  }
  mv_store(acc, part, y, v, v.n, row, sg, lane);
}

kernel void mv_bf16_padded(device const ushort *w [[buffer(0)]], device const float *x [[buffer(2)]],
                           device float *y [[buffer(3)]], constant PaddedMV &p [[buffer(4)]],
                           uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float4 part[8];
  constant MV &v = p.mv;
  uint row = (tg * (8 / v.split) + sg / v.split) * 4, chunks = v.k / 8;
  float4 acc = 0;
  if (row < v.n) {
    device const uint4 *wr = (device const uint4 *)(w + (ulong)row * p.stride);
    device const float4 *xr = (device const float4 *)x;
    for (uint c = (sg % v.split) * 32 + lane; c < chunks; c += 32 * v.split) {
      float4 x0 = xr[2 * c], x1 = xr[2 * c + 1];
      for (uint r = 0; r < 4; r++) {
        uint4 q = wr[r * (p.stride / 8) + c];
        float4 d = bf16x4(q.xy) * x0 + bf16x4(q.zw) * x1;
        acc[r] += d.x + d.y + d.z + d.w;
      }
    }
  }
  mv_store(acc, part, y, v, v.n, row, sg, lane);
}
