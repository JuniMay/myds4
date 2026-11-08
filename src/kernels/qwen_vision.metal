// Qwen vision: BF16 boundaries, 72-dimensional attention heads and a
// learned 2D position table. Host code supplies the reordered patch grid.
kernel void qwen_embed_rows(device const ushort *embedding [[buffer(0)]], device const int *tokens [[buffer(1)]],
                            device float *out [[buffer(2)]], constant uint &width [[buffer(3)]],
                            device const ulong *images [[buffer(4)]], uint2 i [[thread_position_in_grid]]) {
  if (i.x >= width) return;
  device const ushort *row = images[i.y] ? (device const ushort *)images[i.y] : embedding + (size_t)tokens[i.y] * width;
  out[(size_t)i.y * width + i.x] = bf16f(row[i.x]);
}

kernel void qv_pack(device const float *x [[buffer(0)]], device bfloat *y [[buffer(1)]],
                    constant uint3 &p [[buffer(2)]], uint2 i [[thread_position_in_grid]]) {
  if (i.x < p.y) y[(size_t)i.y * p.y + i.x] = i.y < p.z && i.x < p.x ? bfloat(x[(size_t)i.y * p.x + i.x]) : bfloat(0);
}

kernel void qv_bias(device float *x [[buffer(0)]], device const bfloat *bias [[buffer(1)]],
                    constant uint2 &p [[buffer(2)]], uint2 i [[thread_position_in_grid]]) {
  if (i.x < p.x && i.y < p.y) x[(size_t)i.y * p.x + i.x] = bf(x[(size_t)i.y * p.x + i.x] + float(bias[i.x]));
}

kernel void qv_norm(device const float *x [[buffer(0)]], device const bfloat *w [[buffer(1)]],
                    device const bfloat *b [[buffer(2)]], device float *out [[buffer(3)]],
                    constant uint &width [[buffer(4)]], uint row [[threadgroup_position_in_grid]],
                    uint t [[thread_index_in_threadgroup]]) {
  threadgroup float work[32];
  x += (size_t)row * width; out += (size_t)row * width;
  float sum = 0;
  for (uint i = t; i < width; i += 256) sum += x[i];
  float mean = tg_sum(sum, work, t, 256) / width;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  sum = 0;
  for (uint i = t; i < width; i += 256) { float d = x[i] - mean; sum += d * d; }
  float inv = rsqrt(tg_sum(sum, work, t, 256) / width + 1e-6f);
  for (uint i = t; i < width; i += 256) out[i] = bf((x[i] - mean) * inv * float(w[i]) + float(b[i]));
}

kernel void qv_positions(device float *x [[buffer(0)]], device const float *position [[buffer(1)]],
                         constant uint &count [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i < count) x[i] = bf(x[i] + position[i]);
}

// Metal has no erf intrinsic. This approximation is below BF16 rounding
// precision; the vision kernel checks compare it with the host's erff.
inline float qv_erf(float x) {
  float a = abs(x), t = 1.0f / (1.0f + 0.3275911f * a);
  float p = (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t;
  return copysign(1.0f - p * exp(-a * a), x);
}

kernel void qv_gelu(device float *x [[buffer(0)]], constant uint2 &p [[buffer(1)]], uint i [[thread_position_in_grid]]) {
  if (i >= p.x) return;
  float v = x[i];
  // tanh already rounds to +/-1 at these bounds. Limiting its argument
  // avoids overflow in the device implementation for large activations.
  float argument = 0.7978845608028654f * (v + 0.044715f * v * v * v);
  float g = p.y ? qv_erf(v * 0.7071067811865475f) : tanh(clamp(argument, -10.0f, 10.0f));
  x[i] = bf(0.5f * v * (1 + g));
}

kernel void qv_heads(device const float *qkv [[buffer(0)]], device const float *rope [[buffer(1)]],
                     device bfloat *hv [[buffer(2)]], constant uint2 &p [[buffer(3)]],
                     uint2 i [[thread_position_in_grid]]) {
  uint part = i.x / 1536, head = i.x % 1536 / 96, d = i.x % 96;
  if (part >= 3 || i.y >= p.y) return;
  float v = 0;
  if (i.y < p.x && d < 72) {
    device const float *row = qkv + (size_t)i.y * 3456 + part * 1152 + head * 72;
    v = row[d];
    if (part < 2) {
      uint pair = d % 36;
      float c = rope[(size_t)i.y * 72 + pair], s = rope[(size_t)i.y * 72 + 36 + pair];
      v = d < 36 ? v * c - row[d + 36] * s : v * c + row[d - 36] * s;
    }
  }
  hv[(((size_t)part * 16 + head) * p.y + i.y) * 96 + d] = bfloat(v);
}

kernel void qv_attention(device const bfloat *hv [[buffer(0)]], device float *o [[buffer(1)]],
                     constant uint2 &dims [[buffer(2)]], uint2 tg [[threadgroup_position_in_grid]],
                     uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float state[4][4][16];  // a simdgroup's rows: running maximum, sum, rescale, reduction
  threadgroup float *mx = state[sg][0], *sum = state[sg][1], *rescale = state[sg][2], *red_rows = state[sg][3];
  uint n = dims.x, np = dims.y, h = tg.y, q0 = tg.x * 64 + sg * 16;
  constexpr auto sdesc = matmul2d_descriptor(16, 64, 96, false, true, false, matmul2d_descriptor::mode::multiply);
  constexpr auto odesc = matmul2d_descriptor(16, 96, 64, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<sdesc, execution_simdgroup> sop;
  matmul2d<odesc, execution_simdgroup> oop;
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tq((device bfloat *)hv + h * np * 96, dextents<int32_t, 2>(96, np));
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tk((device bfloat *)hv + (16 + h) * np * 96, dextents<int32_t, 2>(96, np));
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tv((device bfloat *)hv + (32 + h) * np * 96, dextents<int32_t, 2>(96, np));
  auto qt = tq.slice(0, q0), kt = tk.slice(0, 0), vt = tv.slice(0, 0);
  auto acc = oop.get_destination_cooperative_tensor<decltype(oop.get_left_input_cooperative_tensor<bfloat, bfloat, float>()), decltype(vt), float>();
  for (uint16_t i = 0; i < acc.get_capacity(); i++)
    if (acc.is_valid_element(i)) acc[i] = 0;
  if (lane < 16) mx[lane] = -INFINITY, sum[lane] = 0;
  simdgroup_barrier(mem_flags::mem_threadgroup);
  for (uint k0 = 0; k0 < n; k0 += 64) {
    kt = tk.slice(0, k0), vt = tv.slice(0, k0);
    auto sc = sop.get_destination_cooperative_tensor<decltype(qt), decltype(kt), float>();
    sop.run(qt, kt, sc);
    for (uint16_t i = 0; i < sc.get_capacity(); i++)
      if (sc.is_valid_element(i)) sc[i] = k0 + sc.get_multidimensional_index(i)[0] < n ? sc[i] * 0.1178511301977579f : -INFINITY;
    auto red = sop.get_row_reduction_destination_cooperative_tensor<decltype(qt), decltype(kt), float>();
    reduce_rows(sc, red, reduction_operation::max, -INFINITY);
    for (uint16_t i = 0; i < red.get_capacity(); i++)  // lanes may hold a row's result alike
      if (red.is_valid_element(i)) red_rows[red.get_multidimensional_index(i)[0]] = red[i];
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (lane < 16) {
      float m = max(mx[lane], red_rows[lane]);
      rescale[lane] = exp(mx[lane] - m), mx[lane] = m;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    auto pb = sop.get_destination_cooperative_tensor<decltype(qt), decltype(kt), bfloat>();
    for (uint16_t i = 0; i < sc.get_capacity(); i++)
      if (sc.is_valid_element(i)) sc[i] = exp(sc[i] - mx[sc.get_multidimensional_index(i)[1]]), pb[i] = bfloat(sc[i]);
    reduce_rows(sc, red, reduction_operation::sum, 0.0f);
    for (uint16_t i = 0; i < red.get_capacity(); i++)
      if (red.is_valid_element(i)) red_rows[red.get_multidimensional_index(i)[0]] = red[i];
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (lane < 16) sum[lane] = sum[lane] * rescale[lane] + red_rows[lane];
    for (uint16_t i = 0; i < acc.get_capacity(); i++)
      if (acc.is_valid_element(i)) acc[i] *= rescale[acc.get_multidimensional_index(i)[1]];
    auto p = oop.get_left_input_cooperative_tensor<bfloat, bfloat, float>(pb);
    oop.run(p, vt, acc);
    simdgroup_barrier(mem_flags::mem_threadgroup);  // the next block rewrites the rows' values
  }
  for (uint16_t i = 0; i < acc.get_capacity(); i++)
    if (acc.is_valid_element(i)) {
      auto a = acc.get_multidimensional_index(i);
      if (q0 + a[1] < n && a[0] < 72) o[(q0 + a[1]) * 1152 + h * 72 + a[0]] = bf(acc[i] / sum[a[1]]);
    }
}
