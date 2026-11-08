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
struct ExpertTile { uint group, first; };
struct ExpertPlan { uint experts, chosen, rows, groups, tiles; };

// Resident weights let the router feed expert work without a CPU round trip.
kernel void expert_plan(device const ExpertGroup *available [[buffer(0)]],
                         device const uint *selected [[buffer(1)]], device const float *weights [[buffer(2)]],
                         device ExpertGroup *groups [[buffer(3)]], device WeightedRow *entries [[buffer(4)]],
                         device uint *order [[buffer(5)]], device uint *row_groups [[buffer(6)]],
                         device ExpertTile *tiles [[buffer(7)]], device uint *unique [[buffer(8)]],
                         constant ExpertPlan &p [[buffer(9)]], uint t [[thread_index_in_threadgroup]]) {
  if (p.rows == 1) {
    if (t >= p.chosen) return;
    uint e = selected[t], rank = 0;
    for (uint k = 0; k < p.chosen; k++) rank += selected[k] < e;
    ExpertGroup g = available[e];
    g.begin = t; g.count = 1;
    groups[t] = g;
    entries[t] = WeightedRow{0, weights[t]};
    order[rank] = t;
    row_groups[t] = t;
    if (!t) *unique = p.chosen;
    return;
  }
  threadgroup uint counts[512], starts[512], indexes[512], tile_starts[512];
  uint pairs = p.rows * p.chosen;
  if (t < p.groups) groups[t].count = 0;
  for (uint i = t; i < p.tiles; i += p.experts) tiles[i].group = UINT_MAX;
  uint count = 0;
  for (uint i = 0; i < pairs; i++) count += selected[i] == t;
  counts[t] = count;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (!t) {
    uint at = 0, small = 0, large = 0, jobs = 0, used = 0;
    for (uint e = 0; e < p.experts; e++) {
      starts[e] = at;
      at += counts[e];
      if (!counts[e]) continue;
      used++;
      indexes[e] = counts[e] <= 8 ? small++ : p.groups - 1 - large++;
      tile_starts[e] = jobs;
      if (counts[e] > 8) jobs += (counts[e] + 15) / 16;
    }
    *unique = used;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (!count) return;
  uint group = indexes[t], at = starts[t];
  ExpertGroup g = available[t];
  g.begin = at; g.count = count;
  groups[group] = g;
  for (uint i = 0; i < pairs; i++) if (selected[i] == t) {
    uint row = i / p.chosen, rank = 0;
    for (uint k = 0; k < p.chosen; k++) rank += selected[row * p.chosen + k] < t;
    entries[at] = WeightedRow{row, weights[i]};
    order[row * p.chosen + rank] = at;
    row_groups[at++] = group;
  }
  if (count > 8) for (uint r = 0; r < count; r += 16)
    tiles[tile_starts[t] + r / 16] = ExpertTile{group, r};
}

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
                         device const float *x [[buffer(2)]], device bfloat *gate [[buffer(3)]],
                         device bfloat *up [[buffer(4)]], constant ExpertDims &p [[buffer(5)]],
                         device const uint *row_groups [[buffer(6)]],
                         uint2 i [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  uint group = row_groups[i.y];
  if (i.x >= p.hidden || group == UINT_MAX) return;
  ExpertGroup e = groups[group];
  uint row = i.y;
  float v = x[(size_t)entries[row].row * p.hidden + i.x];
  gate[(size_t)row * p.hidden + i.x] = bfloat(expert_quantize(v, Gated ? e.gate.input_scale : e.up.input_scale, lane));
  if (Gated) up[(size_t)row * p.hidden + i.x] = bfloat(expert_quantize(v, e.up.input_scale, lane));
}
template <bool Gated, int R = 8>
kernel void expert_up_impl(device const ExpertGroup *groups [[buffer(0)]],
                      device const bfloat *gate [[buffer(1)]], device const bfloat *up [[buffer(2)]],
                      device float *h [[buffer(3)]], constant ExpertDims &p [[buffer(4)]],
                      uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
  ExpertGroup e = groups[tg.y];
  if (!e.count || e.count > 8) return;
  uint row = tg.x * 8 + sg;
  if (row >= p.intermediate) return;
  uint count = R == 1 ? 1 : e.count;
  float ag[R], au[R];
  for (uint t = 0; t < count; t++) ag[t] = au[t] = 0;
  for (uint c = lane; c < p.hidden / 16; c += 32) {
    float4 wg[4], wu[4];
    if (Gated) chunk<0>((device const uchar *)e.gate.weight, (device const uchar *)e.gate.scales, row, p.hidden, c, wg);
    chunk<0>((device const uchar *)e.up.weight, (device const uchar *)e.up.scales, row, p.hidden, c, wu);
    for (uint t = 0; t < count; t++) {
      device const bfloat4 *g = (device const bfloat4 *)(gate + (size_t)(e.begin + t) * p.hidden) + c * 4;
      device const bfloat4 *u = Gated ? (device const bfloat4 *)(up + (size_t)(e.begin + t) * p.hidden) + c * 4 : g;
      if (Gated) {
        float4 dg = wg[0] * float4(g[0]) + wg[1] * float4(g[1]) + wg[2] * float4(g[2]) + wg[3] * float4(g[3]);
        ag[t] += dg.x + dg.y + dg.z + dg.w;
      }
      float4 du = wu[0] * float4(u[0]) + wu[1] * float4(u[1]) + wu[2] * float4(u[2]) + wu[3] * float4(u[3]);
      au[t] += du.x + du.y + du.z + du.w;
    }
  }
  for (uint t = 0; t < count; t++) {
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
template [[host_name("expert_up_single")]] kernel expert_up_t expert_up_impl<true, 1>;
template [[host_name("expert_up_relu2_single")]] kernel expert_up_t expert_up_impl<false, 1>;
kernel void expert_hidden(device const ExpertGroup *groups [[buffer(0)]], device const float *h [[buffer(1)]],
                           device bfloat *x [[buffer(2)]], constant ExpertDims &p [[buffer(3)]],
                           device const uint *row_groups [[buffer(4)]],
                           uint2 i [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  uint group = row_groups[i.y];
  if (i.x >= p.intermediate || group == UINT_MAX) return;
  ExpertGroup e = groups[group];
  size_t at = (size_t)i.y * p.intermediate + i.x;
  x[at] = bfloat(expert_quantize(h[at], e.down.input_scale, lane));
}
template <int R = 8>
kernel void expert_down_impl(device const ExpertGroup *groups [[buffer(0)]], device const bfloat *x [[buffer(1)]],
                        device float *out [[buffer(2)]], constant ExpertDims &p [[buffer(3)]],
                        uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
  ExpertGroup e = groups[tg.y];
  if (!e.count || e.count > 8) return;
  uint row = tg.x * 8 + sg;
  if (row >= p.hidden) return;
  uint count = R == 1 ? 1 : e.count;
  float acc[R];
  for (uint t = 0; t < count; t++) acc[t] = 0;
  for (uint c = lane; c < p.intermediate / 16; c += 32) {
    float4 w[4];
    chunk<0>((device const uchar *)e.down.weight, (device const uchar *)e.down.scales, row, p.intermediate, c, w);
    for (uint t = 0; t < count; t++) {
      device const bfloat4 *v = (device const bfloat4 *)(x + (size_t)(e.begin + t) * p.intermediate) + c * 4;
      float4 d = w[0] * float4(v[0]) + w[1] * float4(v[1]) + w[2] * float4(v[2]) + w[3] * float4(v[3]);
      acc[t] += d.x + d.y + d.z + d.w;
    }
  }
  for (uint t = 0; t < count; t++) {
    float v = bf(simd_sum(acc[t]) * e.down.scale);
    if (lane == 0) out[(size_t)(e.begin + t) * p.hidden + row] = v;
  }
}
typedef decltype(expert_down_impl<8>) expert_down_t;
template [[host_name("expert_down")]] kernel expert_down_t expert_down_impl<8>;
template [[host_name("expert_down_single")]] kernel expert_down_t expert_down_impl<1>;

// Tiles from different experts share a dispatch. Padding affects only discarded
// output rows; each retained row reads its own quantized activations.
template <bool Up, bool Gated>
kernel void expert_matrix_impl(device const ExpertGroup *groups [[buffer(0)]],
                                device const bfloat *x [[buffer(1)]], device const bfloat *second [[buffer(2)]],
                                device float *out [[buffer(3)]], constant ExpertDims &p [[buffer(4)]],
                                device const ExpertTile *tiles [[buffer(5)]],
                                uint2 tg [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]]) {
  ExpertTile tile = tiles[tg.y];
  if (tile.group == UINT_MAX) return;
  ExpertGroup e = groups[tile.group];
  ExpertMatrix main = Up ? e.up : e.down;
  uint cols = Up ? p.hidden : p.intermediate, width = Up ? p.intermediate : p.hidden;
  device const bfloat *input = Up && Gated ? second : x;
  threadgroup bfloat ws[64 * 64], gs[Up && Gated ? 64 * 64 : 1];
  constexpr auto desc = matmul2d_descriptor(16, 64, 64, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<4>> op;
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tx((device bfloat *)input + (size_t)e.begin * cols,
      dextents<int32_t, 2>(cols, (e.count + 15) / 16 * 16));
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> gx((device bfloat *)x + (size_t)e.begin * cols,
      dextents<int32_t, 2>(cols, (e.count + 15) / 16 * 16));
  tensor<threadgroup bfloat, extents<int32_t, 64, 64>, tensor_inline> tw(ws, extents<int32_t, 64, 64>());
  tensor<threadgroup bfloat, extents<int32_t, 64, 64>, tensor_inline> gw(gs, extents<int32_t, 64, 64>());
  auto acc = op.get_destination_cooperative_tensor<decltype(tx.slice(0, 0)), decltype(tw), float>();
  auto gate = op.get_destination_cooperative_tensor<decltype(gx.slice(0, 0)), decltype(gw), float>();
  for (uint16_t i = 0; i < acc.get_capacity(); i++) if (acc.is_valid_element(i)) acc[i] = gate[i] = 0;
  uint n0 = tg.x * 64;
  uint4 q[4], qg[4]; float scale[2], gate_scale[2];
  tile_load<0>((device const uchar *)main.weight, (device const uchar *)main.scales, n0, width, cols, 0, t, q, scale);
  if (Up && Gated)
    tile_load<0>((device const uchar *)e.gate.weight, (device const uchar *)e.gate.scales, n0, width, cols, 0, t, qg, gate_scale);
  for (uint k0 = 0; k0 < cols; k0 += 64) {
    tile_store<0>(ws, q, scale, t);
    if (Up && Gated) tile_store<0>(gs, qg, gate_scale, t);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (k0 + 64 < cols) {
      tile_load<0>((device const uchar *)main.weight, (device const uchar *)main.scales, n0, width, cols, k0 + 64, t, q, scale);
      if (Up && Gated)
        tile_load<0>((device const uchar *)e.gate.weight, (device const uchar *)e.gate.scales, n0, width, cols, k0 + 64, t, qg, gate_scale);
    }
    auto a = tx.slice(k0, tile.first);
    op.run(a, tw, acc);
    if (Up && Gated) {
      auto b = gx.slice(k0, tile.first);
      op.run(b, gw, gate);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (uint16_t i = 0; i < acc.get_capacity(); i++) if (acc.is_valid_element(i)) {
    auto at = acc.get_multidimensional_index(i);
    uint row = tile.first + at[1], col = n0 + at[0];
    if (row >= e.count || col >= width) continue;
    float value = bf(acc[i] * main.scale);
    if (Up && Gated) {
      float g = bf(gate[i] * e.gate.scale);
      value = bf(bf(g * sigm(g)) * value);
    } else if (Up) {
      value = max(value, 0.0f);
      value = bf(value * value);
    }
    out[(size_t)(e.begin + row) * width + col] = value;
  }
}
typedef decltype(expert_matrix_impl<true, true>) expert_matrix_t;
template [[host_name("expert_up_matrix")]] kernel expert_matrix_t expert_matrix_impl<true, true>;
template [[host_name("expert_up_relu2_matrix")]] kernel expert_matrix_t expert_matrix_impl<true, false>;
template [[host_name("expert_down_matrix")]] kernel expert_matrix_t expert_matrix_impl<false, false>;
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
