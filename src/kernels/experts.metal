// NVFP4 expert groups retain the scalar products' reduction and BF16 boundaries.
struct ExpertMatrix {
  ulong weight, scales;
  float scale, input_scale;
};
struct ExpertGroup {
  ExpertMatrix gate, up, down;
  uint begin, count;
};
struct ExpertDims { uint hidden, intermediate; };

inline float expert_quantize(float v, float global_scale, uint lane) {
  float value = global_scale > 0 ? v / global_scale : 0;
  float left = simd_max(lane < 16 ? fabs(value) : 0.0f);
  float right = simd_max(lane >= 16 ? fabs(value) : 0.0f);
  float scale = e4m3(min((lane < 16 ? left : right) / 6, 448.0f));
  return scale > 0 ? e2m1(clamp(value / scale, -6.0f, 6.0f)) * scale : 0;
}
template <bool Gated>
kernel void expert_input_impl(device const ExpertGroup *groups [[buffer(0)]],
                         device const WeightedRow *entries [[buffer(1)]],
                         device const float *x [[buffer(2)]], device float *gate [[buffer(3)]],
                         device float *up [[buffer(4)]], constant ExpertDims &p [[buffer(5)]],
                         uint3 i [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  ExpertGroup e = groups[i.y];
  if (i.x >= p.hidden || i.z >= e.count) return;
  uint row = e.begin + i.z;
  float v = x[(size_t)entries[row].row * p.hidden + i.x];
  gate[(size_t)row * p.hidden + i.x] = expert_quantize(v, Gated ? e.gate.input_scale : e.up.input_scale, lane);
  if (Gated) up[(size_t)row * p.hidden + i.x] = expert_quantize(v, e.up.input_scale, lane);
}
template <bool Gated>
kernel void expert_up_impl(device const ExpertGroup *groups [[buffer(0)]],
                      device const float *gate [[buffer(1)]], device const float *up [[buffer(2)]],
                      device float *h [[buffer(3)]], constant ExpertDims &p [[buffer(4)]],
                      uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
  ExpertGroup e = groups[tg.y];
  uint row = tg.x * 8 + sg;
  if (row >= p.intermediate) return;
  float ag[8], au[8];
  for (uint t = 0; t < e.count; t++) ag[t] = au[t] = 0;
  for (uint c = lane; c < p.hidden / 16; c += 32) {
    float4 wg[4], wu[4];
    if (Gated) chunk<0>((device const uchar *)e.gate.weight, (device const uchar *)e.gate.scales, row, p.hidden, c, wg);
    chunk<0>((device const uchar *)e.up.weight, (device const uchar *)e.up.scales, row, p.hidden, c, wu);
    for (uint t = 0; t < e.count; t++) {
      device const float4 *g = (device const float4 *)(gate + (size_t)(e.begin + t) * p.hidden) + c * 4;
      device const float4 *u = Gated ? (device const float4 *)(up + (size_t)(e.begin + t) * p.hidden) + c * 4 : g;
      if (Gated) {
        float4 dg = wg[0] * g[0] + wg[1] * g[1] + wg[2] * g[2] + wg[3] * g[3];
        ag[t] += dg.x + dg.y + dg.z + dg.w;
      }
      float4 du = wu[0] * u[0] + wu[1] * u[1] + wu[2] * u[2] + wu[3] * u[3];
      au[t] += du.x + du.y + du.z + du.w;
    }
  }
  for (uint t = 0; t < e.count; t++) {
    float u = bf(simd_sum(au[t]) * e.up.scale), value;
    if (Gated) {
      float g = bf(simd_sum(ag[t]) * e.gate.scale);
      value = bf(bf(g * sigm(g)) * u);
    } else {
      float v = max(u, 0.0f);
      value = bf(v * v);
    }
    if (lane == 0) h[(size_t)(e.begin + t) * p.intermediate + row] = value;
  }
}
typedef decltype(expert_input_impl<true>) expert_input_t;
template [[host_name("expert_input")]] kernel expert_input_t expert_input_impl<true>;
template [[host_name("expert_input_relu2")]] kernel expert_input_t expert_input_impl<false>;
typedef decltype(expert_up_impl<true>) expert_up_t;
template [[host_name("expert_up")]] kernel expert_up_t expert_up_impl<true>;
template [[host_name("expert_up_relu2")]] kernel expert_up_t expert_up_impl<false>;
kernel void expert_hidden(device const ExpertGroup *groups [[buffer(0)]], device const float *h [[buffer(1)]],
                           device float *x [[buffer(2)]], constant ExpertDims &p [[buffer(3)]],
                           uint3 i [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  ExpertGroup e = groups[i.y];
  if (i.x >= p.intermediate || i.z >= e.count) return;
  size_t at = (size_t)(e.begin + i.z) * p.intermediate + i.x;
  x[at] = expert_quantize(h[at], e.down.input_scale, lane);
}
kernel void expert_down(device const ExpertGroup *groups [[buffer(0)]], device const float *x [[buffer(1)]],
                        device float *out [[buffer(2)]], constant ExpertDims &p [[buffer(3)]],
                        uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
  ExpertGroup e = groups[tg.y];
  uint row = tg.x * 8 + sg;
  if (row >= p.hidden) return;
  float acc[8];
  for (uint t = 0; t < e.count; t++) acc[t] = 0;
  for (uint c = lane; c < p.intermediate / 16; c += 32) {
    float4 w[4];
    chunk<0>((device const uchar *)e.down.weight, (device const uchar *)e.down.scales, row, p.intermediate, c, w);
    for (uint t = 0; t < e.count; t++) {
      device const float4 *v = (device const float4 *)(x + (size_t)(e.begin + t) * p.intermediate) + c * 4;
      float4 d = w[0] * v[0] + w[1] * v[1] + w[2] * v[2] + w[3] * v[3];
      acc[t] += d.x + d.y + d.z + d.w;
    }
  }
  for (uint t = 0; t < e.count; t++) {
    float v = bf(simd_sum(acc[t]) * e.down.scale);
    if (lane == 0) out[(size_t)(e.begin + t) * p.hidden + row] = v;
  }
}
template <bool Rounded>
kernel void expert_sum_impl(device const float *parts [[buffer(0)]], device const WeightedRow *entries [[buffer(1)]],
                       device const uint *order [[buffer(2)]], device float *out [[buffer(3)]],
                       constant uint3 &p [[buffer(4)]], uint i [[thread_position_in_grid]]) {
  if (i >= p.x * p.y) return;
  uint row = i / p.x, column = i % p.x;
  float value = 0;
  for (uint k = 0; k < p.z; k++) {
    uint at = order[row * p.z + k];
    if (Rounded) value = bf(value + bf(parts[(size_t)at * p.x + column] * entries[at].weight));
    else value += parts[(size_t)at * p.x + column] * entries[at].weight;
  }
  out[i] = Rounded ? value : bf(value);
}
typedef decltype(expert_sum_impl<true>) expert_sum_t;
template [[host_name("expert_sum")]] kernel expert_sum_t expert_sum_impl<true>;
template [[host_name("expert_sum_fp32")]] kernel expert_sum_t expert_sum_impl<false>;
