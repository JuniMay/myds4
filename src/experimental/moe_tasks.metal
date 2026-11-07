// Experimental work dispatch for ready MXFP4 expert stages from independent
// requests, including requests at different layers. A task is up or down;
// the host submits down only after that request's up has completed. There
// is no device-wide spin barrier and no wait for SSD reads in this kernel.
struct MoeTask {
  device const Expert *ex;
  device const float *x;
  device float *h;
  device const float *shared_out;
  device const float *mix;
  device const bfloat *res;
  device bfloat *out;
  uint first, groups, kind, experts;
};

kernel void moe_tasks(device const MoeTask *tasks [[buffer(0)]], constant uint &count [[buffer(1)]],
                      uint group [[threadgroup_position_in_grid]],
                      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float block[32];
  threadgroup float4 part[8];
  uint index = 0;
  while (index < count && group >= tasks[index].first + tasks[index].groups) index++;
  if (index == count) return;
  device const MoeTask &job = tasks[index];
  uint tile = group - job.first;
  if (!job.kind) {
    uint e = tile / 72, row_group = tile % 72, row = (row_group * 8 + sg) * 4;
    constexpr uint K = 5120, W = K / 2 * 2304, S = K / 32 * 2304;
    device const uchar *w = job.ex[e].p[0] + row * (K / 2), *s = job.ex[e].p[1] + row * (K / 32);
    device const float4 *xv = (device const float4 *)job.x;
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
      block[sg * 4 + lane] = bf(g / (1.0f + exp(-g)) * u * job.ex[e].weight);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) job.h[e * 2304 + row_group * 32 + lane] = fp8_act(block[lane]);
  } else {
    uint d = tile * 4;
    constexpr uint K = 2304, W = 5120 * 2304 / 2, S = 5120 * 2304 / 32;
    float4 a = 0;
    if (sg < job.experts) {
      a = fp4_dot4(job.ex[sg].p[0] + W + d * (K / 2), job.ex[sg].p[1] + S + d * (K / 32), K / 2,
                    (device const float4 *)(job.h + sg * K), K / 32, lane);
      a = float4(bf(simd_sum(a.x)), bf(simd_sum(a.y)), bf(simd_sum(a.z)), bf(simd_sum(a.w)));
    }
    if (lane == 0) part[sg] = a;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0 && lane < 4) {
      uint i = d + lane;
      float y = 0;
      for (uint e = 0; e < job.experts; e++) y += part[e][lane];
      float v0 = bf(y + job.shared_out[i]);
      for (uint k = 0; k < 4; k++) {
        float v = job.mix[4 + k] * v0;
        for (uint j = 0; j < 4; j++) v += job.mix[8 + 4 * j + k] * job.res[j * 5120 + i];
        job.out[k * 5120 + i] = bfloat(bf(v));
      }
    }
  }
}
