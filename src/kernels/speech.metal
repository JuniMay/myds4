// Speech transformer normalization, rotary positions and speaker features.
struct SpeechNorm {
  uint width, rows, offset, stride;
  float eps;
  uint round;
};
kernel void speech_norm(device const float *x [[buffer(0)]], device const float *w [[buffer(1)]],
                        device const float *bias [[buffer(2)]], device float *out [[buffer(3)]],
                        constant SpeechNorm &p [[buffer(4)]], uint row [[threadgroup_position_in_grid]],
                        uint t [[thread_index_in_threadgroup]]) {
  threadgroup float scratch[32];
  x += (size_t)row * p.stride + p.offset;
  float sum = 0;
  for (uint i = t; i < p.width; i += 256)
    sum += x[i];
  float mean = tg_sum(sum, scratch, t, 256) / p.width;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  sum = 0;
  for (uint i = t; i < p.width; i += 256)
    sum += (x[i] - mean) * (x[i] - mean);
  float inv = rsqrt(tg_sum(sum, scratch, t, 256) / p.width + p.eps);
  for (uint i = t; i < p.width; i += 256) {
    float value = (x[i] - mean) * inv * w[i] + bias[i];
    out[(size_t)row * p.width + i] = p.round ? bf(value) : value;
  }
}
kernel void speech_bias(device float *x [[buffer(0)]], device const float *bias [[buffer(1)]],
                        constant uint2 &p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i < p.x * p.y)
    x[i] = bf(x[i] + bias[i % p.x]);
}
struct SpeechHeads {
  uint width, dim, rot, norm;
  float theta;
  uint rows;
};
kernel void speech_heads(device const float *x [[buffer(0)]], device const float *qnorm [[buffer(1)]],
                         device const float *qbias [[buffer(2)]], device const float *knorm [[buffer(3)]],
                         device const float *kbias [[buffer(4)]], device float *q [[buffer(5)]],
                         device bfloat *k [[buffer(6)]], device bfloat *v [[buffer(7)]],
                         constant SpeechHeads &p [[buffer(8)]], uint2 group [[threadgroup_position_in_grid]],
                         uint t [[thread_index_in_threadgroup]]) {
  threadgroup float values[128], scratch[32];
  uint part = group.x / (p.width / p.dim), head = group.x % (p.width / p.dim), row = group.y;
  float value = t < p.dim ? x[(size_t)row * p.width * 3 + part * p.width + head * p.dim + t] : 0;
  if (part < 2 && p.norm) {
    float mean = tg_sum(value, scratch, t, 128) / p.dim;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inverse = rsqrt(tg_sum(t < p.dim ? (value - mean) * (value - mean) : 0, scratch, t, 128) / p.dim + 1e-5f);
    if (t < p.dim)
      value = bf((value - mean) * inverse * (part ? knorm[t] : qnorm[t]) + (part ? kbias[t] : qbias[t]));
  }
  if (t < p.dim)
    values[t] = value;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t >= p.dim)
    return;
  if (part < 2 && t < p.rot) {
    uint pair = t % (p.rot / 2);
    float angle = row * pow(p.theta, -2.0f * pair / p.rot), c = cos(angle), s = sin(angle);
    value = t < p.rot / 2 ? value * c - values[t + p.rot / 2] * s : value * c + values[t - p.rot / 2] * s;
  }
  size_t at = (size_t)row * p.width + head * p.dim + t;
  if (!part)
    q[at] = bf(value);
  else if (part == 1)
    k[at] = bfloat(value);
  else
    v[at] = bfloat(value);
}
kernel void speech_relu(device float *x [[buffer(0)]], constant uint &count [[buffer(1)]],
                        uint i [[thread_position_in_grid]]) {
  if (i < count)
    x[i] = max(x[i], 0.0f);
}
kernel void speech_conv_rows(device const float *x [[buffer(0)]], device float *out [[buffer(1)]],
                             constant uint2 &p [[buffer(2)]], uint2 i [[thread_position_in_grid]]) {
  if (i.x >= p.x * 3 || i.y >= p.y)
    return;
  int row = (int)i.y + (int)(i.x % 3) - 1;
  out[(size_t)i.y * p.x * 3 + i.x] = row >= 0 && row < (int)p.y ? x[(size_t)row * p.x + i.x / 3] : 0;
}
kernel void speech_activity(device const float *logits [[buffer(0)]], device float *out [[buffer(1)]],
                            constant uint2 &p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i >= p.x * 8)
    return;
  uint row = i / 8, speaker = i % 8;
  float sum = 0;
  uint count = row + 1 == p.x ? p.y : 8;
  for (uint sub = 0; sub < count; sub++)
    sum += bf(sigm(logits[(size_t)(row * 8 + sub) * 8 + speaker]));
  out[i] = bf(sum / count);
}
kernel void speech_fuse(device float *x [[buffer(0)]], device const float *activity [[buffer(1)]],
                        device const float *speaker_kernel [[buffer(2)]], constant uint2 &p [[buffer(3)]],
                        uint i [[thread_position_in_grid]]) {
  if (i >= p.x * p.y)
    return;
  uint row = i / p.x, col = i % p.x;
  float sum = 0;
  for (uint speaker = 0; speaker < 8; speaker++)
    sum += activity[row * 8 + speaker] * speaker_kernel[speaker * p.x + col];
  x[i] = bf(x[i] + bf(sum));
}
