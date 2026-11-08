// DFlash2 query convolutions and context keys; target verification decides acceptance.
kernel void df_capture(device const float *x [[buffer(0)]], device float *out [[buffer(1)]],
                       constant uint3 &p [[buffer(2)]], uint2 i [[thread_position_in_grid]]) {
  if (i.x < p.x)
    out[((size_t)i.y * p.y + p.z) * p.x + i.x] = x[(size_t)i.y * p.x + i.x];
}
struct DfConv {
  uint width, rows, side, pad;
};
kernel void df_conv(device const float *x [[buffer(0)]], device const float *coefficients [[buffer(1)]],
                    device const float *base [[buffer(2)]], device float *out [[buffer(3)]],
                    constant DfConv &p [[buffer(4)]], uint2 i [[thread_position_in_grid]]) {
  if (i.x >= p.width || i.y >= p.rows)
    return;
  uint group = i.x / 16, groups = p.width / 16;
  size_t at = (size_t)i.y * 4 * groups + p.side * 2 * groups;
  float value = (base[p.side * 2 * p.width + i.x] + coefficients[at + group]) * x[(size_t)i.y * p.width + i.x];
  if (i.y)
    value += (base[(p.side * 2 + 1) * p.width + i.x] + coefficients[at + groups + group]) *
             x[(size_t)(i.y - 1) * p.width + i.x];
  out[(size_t)i.y * p.width + i.x] = bf(value);
}
struct DfRope {
  uint heads, dim, layer, base;
  float eps, theta;
  uint pad0, pad1;
};
kernel void df_rope(device const float *x [[buffer(0)]], device const float *weight [[buffer(1)]],
                    device float *out [[buffer(2)]], constant NmSeq *seq [[buffer(3)]],
                    constant DfRope &p [[buffer(4)]], uint2 group [[threadgroup_position_in_grid]],
                    uint t [[thread_index_in_threadgroup]]) {
  threadgroup float values[128], partial[32];
  size_t at = ((size_t)group.y * p.heads + group.x) * 128;
  float value = x[at + t];
  float inverse = rsqrt(tg_sum(value * value, partial, t, 128) / 128 + p.eps);
  values[t] = bf(value * inverse * weight[t]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint pair = t % 64;
  float angle = seq[group.y].pos * pow(p.theta, -2.0f * pair / 128);
  float c = cos(angle), s = sin(angle);
  out[at + t] = bf(t < 64 ? values[t] * c - values[t + 64] * s : values[t] * c + values[t - 64] * s);
}
kernel void df_swiglu(device const float *gate [[buffer(0)]], device const float *up [[buffer(1)]],
                      device float *out [[buffer(2)]], constant uint &count [[buffer(3)]],
                      uint i [[thread_position_in_grid]]) {
  if (i < count) {
    float g = gate[i];
    out[i] = bf(g * sigm(g) * up[i]);
  }
}
