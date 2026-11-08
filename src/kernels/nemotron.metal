// Nemotron-H: Mamba2 recurrence, BF16 residuals and full attention.
struct NmSeq {
  device float *state, *saved_state, *conv, *saved_conv;
  device bfloat *k, *v;
  uint pos, cap;
  int rope_t, rope_h, rope_w, pad;
};
struct NmConv {
  uint rows, channels, intermediate, layer;
};
kernel void nm_conv(device const float *x [[buffer(0)]], device const float *weight [[buffer(1)]],
                    device const float *bias [[buffer(2)]], constant NmSeq *seq [[buffer(3)]],
                    device float *out [[buffer(4)]], constant NmConv &p [[buffer(5)]],
                    uint channel [[thread_position_in_grid]]) {
  if (channel >= p.channels)
    return;
  for (uint row = 0; row < p.rows; row++) {
    device float *history = seq[row].conv + ((size_t)p.layer * p.channels + channel) * 4;
    float value = x[(size_t)row * (p.intermediate + p.channels + 64) + p.intermediate + channel];
    float h[4] = {history[1], history[2], history[3], value};
    float sum = bias[channel];
    for (uint j = 0; j < 4; j++)
      sum += h[j] * weight[channel * 4 + j];
    sum = bf(sum);
    out[(size_t)row * p.channels + channel] = bf(sum * sigm(sum));
    for (uint j = 0; j < 4; j++)
      history[j] = h[j];
    if (seq[row].saved_conv)
      for (uint j = 0; j < 4; j++)
        seq[row].saved_conv[((size_t)p.layer * p.channels + channel) * 4 + j] = h[j];
  }
}
struct NmScan {
  uint rows, heads, groups, dim, state, layer, channels, projected;
  float minimum;
};
kernel void nm_scan(device const float *x [[buffer(0)]], device const float *projected [[buffer(1)]],
                    device const float *a [[buffer(2)]], device const float *d [[buffer(3)]],
                    device const float *bias [[buffer(4)]], constant NmSeq *seq [[buffer(5)]],
                    device float *out [[buffer(6)]], constant NmScan &p [[buffer(7)]],
                    uint2 group [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
                    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  uint head = group.x, column = group.y * 4, intermediate = p.heads * p.dim;
  uint bc = head / (p.heads / p.groups);
  threadgroup float reduce[4][4];
  device float *previous = nullptr;
  float4 state = 0;
  for (uint row = 0; row < p.rows; row++) {
    device float *base = seq[row].state + ((size_t)p.layer * p.heads + head) * p.dim * p.state;
    if (base != previous) {
      if (previous)
        for (uint i = 0; i < 4; i++)
          previous[(column + i) * p.state + t] = state[i];
      for (uint i = 0; i < 4; i++)
        state[i] = base[(column + i) * p.state + t];
      previous = base;
    }
    float dt = projected[(size_t)row * p.projected + intermediate + p.channels + head] + bias[head];
    dt = max(p.minimum, dt > 20 ? dt : log(1 + exp(dt)));
    float decay = exp(-dt * a[head]);
    float bv = x[(size_t)row * p.channels + intermediate + bc * p.state + t];
    float cv = x[(size_t)row * p.channels + intermediate + p.groups * p.state + bc * p.state + t];
    float4 value = *(device const float4 *)(x + (size_t)row * p.channels + head * p.dim + column);
    state = decay * state + (dt * bv) * value;
    float4 sum = state * cv;
    for (uint i = 0; i < 4; i++) {
      sum[i] = simd_sum(sum[i]);
      if (!lane)
        reduce[i][sg] = sum[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t < 4) {
      float total = reduce[t][0] + reduce[t][1] + reduce[t][2] + reduce[t][3];
      out[(size_t)row * intermediate + head * p.dim + column + t] = bf(total + d[head] * value[t]);
    }
    if (seq[row].saved_state)
      for (uint i = 0; i < 4; i++)
        seq[row].saved_state[(((size_t)p.layer * p.heads + head) * p.dim + column + i) * p.state + t] = state[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (previous)
    for (uint i = 0; i < 4; i++)
      previous[(column + i) * p.state + t] = state[i];
}
struct NmGate {
  uint width, groups, stride, pad;
  float eps;
};
kernel void nm_gate_norm(device const float *x [[buffer(0)]], device const float *gate [[buffer(1)]],
                         device const float *weight [[buffer(2)]], device float *out [[buffer(3)]],
                         constant NmGate &p [[buffer(4)]], uint group [[threadgroup_position_in_grid]],
                         uint t [[thread_index_in_threadgroup]]) {
  threadgroup float partial[32];
  uint row = group / p.groups, g = group % p.groups;
  float sum = 0;
  for (uint i = t; i < p.width; i += 256) {
    uint col = g * p.width + i;
    float z = gate[(size_t)row * p.stride + col];
    float value = x[(size_t)row * p.groups * p.width + col] * (z * sigm(z));
    sum += value * value;
  }
  float inverse = rsqrt(tg_sum(sum, partial, t, 256) / p.width + p.eps);
  for (uint i = t; i < p.width; i += 256) {
    uint col = g * p.width + i;
    float z = gate[(size_t)row * p.stride + col];
    float value = x[(size_t)row * p.groups * p.width + col] * (z * sigm(z));
    out[(size_t)row * p.groups * p.width + col] = bf(bf(value * inverse) * weight[col]);
  }
}
kernel void nm_relu2(device float *x [[buffer(0)]], constant uint &count [[buffer(1)]],
                     uint i [[thread_position_in_grid]]) {
  if (i < count) {
    float v = max(x[i], 0.0f);
    x[i] = bf(v * v);
  }
}
kernel void nm_add(device float *x [[buffer(0)]], device const float *y [[buffer(1)]],
                   constant uint &count [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i < count)
    x[i] = bf(x[i] + y[i]);
}
struct NmRoute {
  uint experts, chosen;
  float factor;
  uint normalize;
};
kernel void nm_route(device const float *logits [[buffer(0)]], device const float *bias [[buffer(1)]],
                     device uint *selected [[buffer(2)]], device float *weights [[buffer(3)]],
                     constant NmRoute &p [[buffer(4)]], uint row [[threadgroup_position_in_grid]],
                     uint t [[thread_index_in_threadgroup]]) {
  threadgroup float score[128], choice[128], selected_score[16], total;
  if (t < p.experts)
    score[t] = sigm(logits[row * p.experts + t]), choice[t] = score[t] + bias[t];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (!t) {
    total = 0;
    for (uint k = 0; k < p.chosen; k++) {
      uint best = 0;
      for (uint e = 1; e < p.experts; e++)
        if (choice[e] > choice[best])
          best = e;
      selected[row * p.chosen + k] = best;
      selected_score[k] = score[best];
      total += score[best];
      choice[best] = -INFINITY;
    }
    for (uint k = 0; k < p.chosen; k++)
      weights[row * p.chosen + k] = selected_score[k] * p.factor / (p.normalize ? total + 1e-20f : 1);
  }
}
kernel void nm_scatter(device const float *values [[buffer(0)]], device const uint2 *entries [[buffer(1)]],
                       device float *out [[buffer(2)]], constant uint2 &p [[buffer(3)]],
                       uint i [[thread_position_in_grid]]) {
  if (i >= p.x * p.y)
    return;
  uint2 entry = entries[i / p.x];
  out[(size_t)entry.x * p.x + i % p.x] += values[i] * as_type<float>(entry.y);
}
kernel void nm_kv_store(device const float *k [[buffer(0)]], device const float *v [[buffer(1)]],
                        constant NmSeq *seq [[buffer(2)]], constant uint2 &p [[buffer(3)]],
                        uint2 i [[thread_position_in_grid]]) {
  if (i.x >= p.x)
    return;
  size_t at = ((size_t)p.y * seq[i.y].cap + seq[i.y].pos) * p.x + i.x;
  seq[i.y].k[at] = bfloat(k[(size_t)i.y * p.x + i.x]);
  seq[i.y].v[at] = bfloat(v[(size_t)i.y * p.x + i.x]);
}
struct NmAttention {
  uint heads, kv, dim, layer, blocks, causal, length, pad;
  float scale;
};
kernel void nm_attention(device const float *queries [[buffer(0)]], constant NmSeq *seq [[buffer(1)]],
                         device float *part [[buffer(2)]], device float2 *ml [[buffer(3)]],
                         constant NmAttention &p [[buffer(4)]], uint3 group [[threadgroup_position_in_grid]],
                         uint t [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]]) {
  uint head = group.x, block = group.y, row = group.z, kv = head / (p.heads / p.kv);
  uint end = p.causal ? seq[row].pos + 1 : p.length;
  if (block * 256 >= end)
    return;
  threadgroup float q[128], scores[256], partial[256], red[32];
  if (t < p.dim)
    q[t] = queries[((size_t)row * p.heads + head) * p.dim + t];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  size_t base = (size_t)p.layer * seq[row].cap * p.kv * p.dim + kv * p.dim;
  for (uint k = sg; k < 256; k += 8) {
    uint pos = block * 256 + k;
    float sum = 0;
    if (pos < end)
      for (uint d = lane; d < p.dim; d += 32)
        sum += q[d] * float(seq[row].k[base + (size_t)pos * p.kv * p.dim + d]);
    sum = simd_sum(sum);
    if (!lane)
      scores[k] = pos < end ? sum * p.scale : -INFINITY;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float mx = simd_max(scores[t]);
  if (!lane)
    red[sg] = mx;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  mx = -INFINITY;
  for (uint i = 0; i < 8; i++)
    mx = max(mx, red[i]);
  scores[t] = exp(scores[t] - mx);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float sum = tg_sum(scores[t], red, t, 256);
  size_t index = ((size_t)row * p.heads + head) * p.blocks + block;
  if (!t)
    ml[index] = float2(mx, sum);
  uint dimension = t % p.dim, lane_group = t / p.dim;
  float acc = 0;
  for (uint k = lane_group; k < 256 && block * 256 + k < end; k += 256 / p.dim)
    acc += scores[k] * float(seq[row].v[base + (size_t)(block * 256 + k) * p.kv * p.dim + dimension]);
  partial[t] = acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t < p.dim) {
    for (uint other = t + p.dim; other < 256; other += p.dim)
      acc += partial[other];
    part[index * p.dim + t] = acc;
  }
}
kernel void nm_attention_sum(device const float *part [[buffer(0)]], device const float2 *ml [[buffer(1)]],
                             constant NmSeq *seq [[buffer(2)]], device float *out [[buffer(3)]],
                             constant NmAttention &p [[buffer(4)]], uint2 group [[threadgroup_position_in_grid]],
                             uint t [[thread_index_in_threadgroup]]) {
  uint blocks = ((p.causal ? seq[group.y].pos + 1 : p.length) + 255) / 256;
  size_t base = ((size_t)group.y * p.heads + group.x) * p.blocks;
  float maximum = -INFINITY;
  for (uint b = 0; b < blocks; b++)
    maximum = max(maximum, ml[base + b].x);
  float total = 0, value = 0;
  for (uint b = 0; b < blocks; b++) {
    float scale = exp(ml[base + b].x - maximum);
    total += ml[base + b].y * scale;
    if (t < p.dim)
      value += part[(base + b) * p.dim + t] * scale;
  }
  if (t < p.dim)
    out[((size_t)group.y * p.heads + group.x) * p.dim + t] = bf(value / total);
}
