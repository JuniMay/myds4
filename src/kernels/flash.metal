// Qwen Flash-Next: BF16 layer boundaries, gated residual streams and routed experts.
kernel void flash_embed(device const ushort *w [[buffer(0)]], device const int *tokens [[buffer(1)]],
                        device float *out [[buffer(2)]], constant uint2 &p [[buffer(3)]],
                        device const ulong *images [[buffer(4)]], uint2 i [[thread_position_in_grid]]) {
  if (i.x >= p.x * p.y)
    return;
  device const ushort *source = images[i.y] ? (device const ushort *)images[i.y] : w + (size_t)tokens[i.y] * p.x;
  out[(size_t)i.y * p.x * p.y + i.x] = bf16f(source[i.x % p.x]);
}
kernel void flash_hc_silu(device float *x [[buffer(0)]], constant uint2 &p [[buffer(1)]],
                          uint i [[thread_position_in_grid]]) {
  if (i < p.x) {
    float v = bf(x[i] / p.y);
    x[i] = bf(v * sigm(v));
  }
}
kernel void flash_hc_mix(device const float *x [[buffer(0)]], device const float *gate [[buffer(1)]],
                         device float *out [[buffer(2)]], constant uint3 &p [[buffer(3)]],
                         uint i [[thread_position_in_grid]]) {
  if (i >= p.x * p.z)
    return;
  uint row = i / p.x, column = i % p.x;
  float sum = 0;
  for (uint stream = 0; stream < p.y; stream++) {
    size_t at = ((size_t)row * p.y + stream) * p.x + column;
    sum += bf(x[at] * bf(sigm(gate[at])));
  }
  out[i] = bf(sum / p.y);
}
kernel void flash_hc_inject(device float *x [[buffer(0)]], device const float *value [[buffer(1)]],
                            device const float *gate [[buffer(2)]], constant uint3 &p [[buffer(3)]],
                            uint i [[thread_position_in_grid]]) {
  if (i >= p.x * p.y * p.z)
    return;
  uint row = i / (p.x * p.y), stream = i / p.x % p.y, column = i % p.x;
  float factor = bf(2 * bf(sigm(bf(gate[row * p.y + stream] / p.y))));
  x[i] = bf(x[i] + bf(value[(size_t)row * p.x + column] * factor));
}
kernel void flash_swiglu(device const float *gate [[buffer(0)]], device const float *up [[buffer(1)]],
                         device float *out [[buffer(2)]], constant uint &n [[buffer(3)]],
                         uint i [[thread_position_in_grid]]) {
  if (i < n)
    out[i] = bf(bf(gate[i] * sigm(gate[i])) * up[i]);
}
kernel void flash_shared(device float *out [[buffer(0)]], device const float *shared [[buffer(1)]],
                         device const float *gate [[buffer(2)]], constant uint2 &p [[buffer(3)]],
                         uint i [[thread_position_in_grid]]) {
  if (i < p.x * p.y)
    out[i] = bf(out[i] + bf(shared[i] * bf(sigm(gate[i / p.x]))));
}
struct FlashRoute {
  uint experts, chosen, normalized, pad;
};
kernel void flash_route(device const float *logits [[buffer(0)]], device uint *ids [[buffer(1)]],
                        device float *weights [[buffer(2)]], constant FlashRoute &p [[buffer(3)]],
                        uint row [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
                        uint threads [[threads_per_threadgroup]]) {
  threadgroup float partial[32], selected[16], local_scores[32 * 16];
  threadgroup uint local_ids[32 * 16];
  float key = t < p.experts ? logits[(size_t)row * p.experts + t] : -INFINITY;
  float maximum = tg_max(key, partial, t, threads);
  float base = t < p.experts ? exp(key - maximum) : 0;
  float sum = tg_sum(base, partial, t, threads);
  base /= sum;
  key = t < p.experts ? base : -INFINITY;
  group_topk(key, t < p.experts ? t : UINT_MAX, p.chosen, local_scores, local_ids,
             ids + row * p.chosen, selected, t, threads);
  if (t == 0) {
    float total = 0;
    for (uint k = 0; k < p.chosen; k++)
      total += selected[k];
    for (uint k = 0; k < p.chosen; k++)
      weights[row * p.chosen + k] = bf(selected[k] / (p.normalized ? total : 1.0f));
  }
}
kernel void flash_scatter(device const float *x [[buffer(0)]], device const WeightedRow *entries [[buffer(1)]],
                          device float *out [[buffer(2)]], constant uint2 &p [[buffer(3)]],
                          uint i [[thread_position_in_grid]]) {
  if (i >= p.x * p.y)
    return;
  WeightedRow e = entries[i / p.x];
  size_t at = (size_t)e.row * p.x + i % p.x;
  out[at] = bf(out[at] + bf(x[i] * e.weight));
}
kernel void flash_kv_store(device const float *k [[buffer(0)]], device const float *v [[buffer(1)]],
                           constant Seq *rows [[buffer(2)]], constant Kv &p [[buffer(3)]],
                           uint2 i [[thread_position_in_grid]]) {
  if (i.x >= p.width)
    return;
  size_t at = ((size_t)p.layer * rows[i.y].cap + rows[i.y].pos) * p.width + i.x;
  ((device bfloat *)rows[i.y].k)[at] = bfloat(k[(size_t)i.y * p.width + i.x]);
  ((device bfloat *)rows[i.y].v)[at] = bfloat(v[(size_t)i.y * p.width + i.x]);
}


kernel void flash_ple_gate(device const float *key [[buffer(0)]], device const float *query [[buffer(1)]],
                           device const float *value [[buffer(2)]], device float *out [[buffer(3)]],
                           constant uint2 &p [[buffer(4)]], uint row [[threadgroup_position_in_grid]],
                           uint t [[thread_index_in_threadgroup]], uint threads [[threads_per_threadgroup]]) {
  threadgroup float partial[32];
  float sum = 0;
  for (uint i = t; i < p.x; i += threads) {
    size_t at = (size_t)row * p.x + i;
    sum += bf(key[at] * query[at]);
  }
  float gate = bf(bf(tg_sum(sum, partial, t, threads)) / sqrt(float(p.x)));
  gate = bf(sqrt(max(fabs(gate), 1e-6f))) * sign(gate);
  gate = bf(sigm(gate));
  for (uint i = t; i < p.x; i += threads)
    out[(size_t)row * p.x + i] = bf(gate * value[(size_t)(row / p.y) * p.x + i]);
}
struct FlashPleConv {
  uint width, n, layer, pad;
};
kernel void flash_ple_conv(device const float *normalized [[buffer(0)]], device const float *gated [[buffer(1)]],
                           device const float *weight [[buffer(2)]], device const ulong *states [[buffer(3)]],
                           device float *x [[buffer(4)]], constant FlashPleConv &p [[buffer(5)]],
                           uint column [[thread_position_in_grid]]) {
  if (column >= p.width)
    return;
  float history[9];
  device float *state = nullptr;
  for (uint row = 0; row < p.n; row++) {
    device float *next = (device float *)states[row] + (size_t)p.layer * 9 * p.width;
    if (next != state) {
      if (state)
        for (uint i = 0; i < 9; i++)
          state[i * p.width + column] = history[i];
      state = next;
      for (uint i = 0; i < 9; i++)
        history[i] = state[i * p.width + column];
    }
    size_t at = (size_t)row * p.width + column;
    float v = normalized[at];
    float convolution = bf(weight[column * 4] * history[0] + weight[column * 4 + 1] * history[3] +
                           weight[column * 4 + 2] * history[6] + weight[column * 4 + 3] * v);
    x[at] = bf(x[at] + bf(gated[at] + bf(convolution * sigm(convolution))));
    for (uint i = 0; i < 8; i++)
      history[i] = history[i + 1];
    history[8] = v;
  }
  if (state)
    for (uint i = 0; i < 9; i++)
      state[i * p.width + column] = history[i];
}

struct FlashAttention {
  uint heads, kv, dim, layer;
  float scale;
  uint blocks, stride, pad;
};
kernel void flash_attention_block(device const float *q [[buffer(0)]], constant Seq *rows [[buffer(1)]],
                                  device float *part [[buffer(2)]], device float2 *ml [[buffer(3)]],
                                  constant FlashAttention &p [[buffer(4)]], device const int *selected [[buffer(5)]],
                                  uint3 g [[threadgroup_position_in_grid]], uint j [[thread_index_in_threadgroup]],
                                  uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  uint kvh = g.x, b = g.y, t = g.z, pos = uint(selected[t * p.stride]) - 1, G = p.heads / p.kv;
  if (b * BLK > pos)
    return;
  // the queries (8 heads x 256) and the scores (8 x BLK), and after them the
  // simdgroups' value sums (8 simdgroups x 2 heads x 256)
  threadgroup float stage[12 * 256 + 12 * BLK];
  threadgroup float (*qs)[256] = (threadgroup float (*)[256])stage;
  threadgroup float (*ps)[BLK] = (threadgroup float (*)[BLK])(stage + 12 * 256);
  threadgroup float red[12][8];
  for (uint i = j; i < G * p.dim; i += BLK)
    qs[i / p.dim][i % p.dim] = q[((size_t)t * p.heads + kvh * G) * p.dim + i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint at = b * BLK + j, width = p.kv * p.dim;
  size_t layer = (size_t)p.layer * rows[t].cap * width;
  // the scores: a simdgroup reads one key row at a time, its lanes 8
  // dimensions each (a row of 256 values: coalesced), each head's partial
  // dot products summed across the lanes
  {
    float4 qa[12][2];
    for (uint h = 0; h < G; h++)
      qa[h][0] = *(threadgroup const float4 *)&qs[h][lane * 8],
      qa[h][1] = *(threadgroup const float4 *)&qs[h][lane * 8 + 4];
    for (uint i = sg; i < BLK; i += BLK / 32) {
      uint ati = b * BLK + i;
      if (ati > pos)
        break;
      uint token = uint(selected[t * p.stride + 1 + ati]);
      device const bfloat4 *kr =
          (device const bfloat4 *)((device bfloat *)rows[t].k + layer + (size_t)token * width + kvh * p.dim + lane * 8);
      float4 k0 = float4(kr[0]), k1 = float4(kr[1]);
      for (uint h = 0; h < G; h++) {
        float d = simd_sum(dot(k0, qa[h][0]) + dot(k1, qa[h][1]));
        if (lane == 0)
          ps[h][i] = d * p.scale;
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float sc[12];
  for (uint h = 0; h < G; h++)
    sc[h] = at <= pos ? ps[h][j] : -INFINITY;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint h = 0; h < G; h++) {
    float m = simd_max(sc[h]);
    if (lane == 0)
      red[h][sg] = m;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float mh[12];
  for (uint h = 0; h < G; h++) {
    float m = red[h][0];
    for (uint s = 1; s < BLK / 32; s++)
      m = max(m, red[h][s]);
    mh[h] = m;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint h = 0; h < G; h++) {
    float e = at <= pos ? exp(sc[h] - mh[h]) : 0.0f;
    ps[h][j] = e;
    float s = simd_sum(e);
    if (lane == 0)
      red[h][sg] = s;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint valid = min((uint)BLK, pos + 1 - b * BLK);
  size_t first = ((size_t)t * p.heads + kvh * G) * p.blocks + b;
  // the values: simdgroup sg takes the block's positions 32 sg .. 32 sg + 31,
  // lane l the dimensions 8 l .. 8 l + 7 (a value row coalesced); the
  // simdgroups' sums meet in threadgroup memory two heads at a time, added in
  // simdgroup order (in the queries' and scores' place, done with)
  {
    float4 acc[12][2];
    for (uint h = 0; h < G; h++)
      acc[h][0] = acc[h][1] = 0;
    device const bfloat *vr = (device const bfloat *)rows[t].v + layer + kvh * p.dim + lane * 8;
    uint lo = sg * 32, hi = min(lo + 32, valid);
    for (uint i = lo; i < hi; i++) {
      uint token = uint(selected[t * p.stride + 1 + b * BLK + i]);
      device const bfloat4 *v = (device const bfloat4 *)(vr + (size_t)token * width);
      float4 v0 = float4(v[0]), v1 = float4(v[1]);
      for (uint h = 0; h < G; h++) {
        float w = ps[h][i];
        acc[h][0] += w * v0, acc[h][1] += w * v1;
      }
    }
    threadgroup float4(*vs)[2][64] = (threadgroup float4(*)[2][64])stage; // [8 simdgroups][2 heads][256 values]
    for (uint h0 = 0; h0 < G; h0 += 2) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint k = 0; k < 2 && h0 + k < G; k++)
        vs[sg][k][lane * 2] = acc[h0 + k][0], vs[sg][k][lane * 2 + 1] = acc[h0 + k][1];
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint k = 0; k < 2 && h0 + k < G; k++) {
        float sum = 0;
        for (uint s2 = 0; s2 < BLK / 32; s2++)
          sum += ((threadgroup float *)vs[s2][k])[j];
        part[(first + (h0 + k) * p.blocks) * p.dim + j] = sum;
      }
    }
  }
  if (j < G) {
    float l = 0;
    for (uint s = 0; s < BLK / 32; s++)
      l += red[j][s];
    ml[first + j * p.blocks] = float2(mh[j], l);
  }
}

kernel void flash_attention_combine(device const float *part [[buffer(0)]], device const float2 *ml [[buffer(1)]],
                                    device const float *qg [[buffer(2)]], device float *out [[buffer(3)]],
                                    device const int *selected [[buffer(4)]], constant FlashAttention &p [[buffer(5)]],
                                    uint2 g [[threadgroup_position_in_grid]], uint i [[thread_index_in_threadgroup]]) {
  uint blocks = (uint(selected[g.y * p.stride]) + BLK - 1) / BLK;
  size_t start = ((size_t)g.y * p.heads + g.x) * p.blocks;
  float maximum = -INFINITY;
  for (uint b = 0; b < blocks; b++)
    maximum = max(maximum, ml[start + b].x);
  float sum = 0, value = 0;
  for (uint b = 0; b < blocks; b++) {
    float factor = exp(ml[start + b].x - maximum);
    sum += ml[start + b].y * factor;
    value += part[(start + b) * p.dim + i] * factor;
  }
  float gate = bf(sigm(qg[((size_t)g.y * p.heads + g.x) * 2 * p.dim + p.dim + i]));
  out[((size_t)g.y * p.heads + g.x) * p.dim + i] = bf(bf(value / sum) * gate);
}

kernel void flash_rope(device const float *x [[buffer(0)]], device const float *w [[buffer(1)]],
                       device float *y [[buffer(2)]], constant Rope &p [[buffer(3)]], constant Seq *rows [[buffer(4)]],
                       uint2 g [[threadgroup_position_in_grid]], uint2 t2 [[thread_position_in_threadgroup]],
                       uint2 tn2 [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float part[32];
  threadgroup float row[512];
  uint t = t2.x, tn = tn2.x;
  device const float *xr = x + ((size_t)g.y * p.heads + g.x) * p.stride;
  float s = 0;
  for (uint i = t; i < p.dim; i += tn)
    s += xr[i] * xr[i];
  s = simd_sum(s);
  if (lane == 0)
    part[sg] = s;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    float v = lane < (tn + 31) / 32 ? part[lane] : 0.0f;
    v = simd_sum(v);
    if (lane == 0)
      part[0] = v;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float r = rsqrt(part[0] / p.dim + p.eps);
  for (uint i = t; i < p.dim; i += tn)
    row[i] = bf(xr[i] * r * w[i]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  device float *yr = y + ((size_t)g.y * p.heads + g.x) * p.dim;
  uint half_rot = p.rot / 2;
  for (uint i = t; i < p.dim; i += tn) {
    float v = row[i];
    if (i < p.rot) {
      uint j = i < half_rot ? i : i - half_rot;
      float pos = float(j % 3 == 1 && j < p.section_h * 3   ? rows[g.y].rope_h
                        : j % 3 == 2 && j < p.section_w * 3 ? rows[g.y].rope_w
                                                            : rows[g.y].rope_t);
      float f = pos * pow(p.theta, -2.0f * float(j) / float(p.rot));
      float c = bf(cos(f)), sn = bf(sin(f));
      v = i < half_rot ? bf(row[i] * c) - bf(row[i + half_rot] * sn) : bf(row[i - half_rot] * sn) + bf(row[i] * c);
    }
    yr[i] = bf(v);
  }
}
