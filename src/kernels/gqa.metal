// Grouped-query attention over FP16 key and value rows: the queries' and
// keys' RMSNorm and rotary embedding, the keys' and values' store, and the
// attention of decoding passes (in blocks of BLK positions, then combined)
// and of prompt passes (on the matrix units), each output times sigmoid of
// its head's gate. Rows carry their sequence (struct Seq, deltanet.metal).

#define BLK 256  // positions per attention block

struct Rope { uint heads, dim, rot, stride; float theta, eps; uint pad0, pad1; };

// Full attention's query or key head g.x of row g.y (dim values at
// x + (g.y * heads + g.x) * stride): RMSNorm (weights w), then the rotary
// embedding of the first rot dimensions at the row's position (pairs i,
// i + rot/2; frequency theta^(-2i/rot)). One threadgroup a head.
kernel void qk_rope(device const float *x [[buffer(0)]], device const float *w [[buffer(1)]],
                    device float *y [[buffer(2)]], constant Rope &p [[buffer(3)]],
                    constant Seq *rows [[buffer(4)]], uint2 g [[threadgroup_position_in_grid]],
                    uint2 t2 [[thread_position_in_threadgroup]], uint2 tn2 [[threads_per_threadgroup]],
                    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float part[32];
  threadgroup float row[512];
  uint t = t2.x, tn = tn2.x;
  device const float *xr = x + ((size_t)g.y * p.heads + g.x) * p.stride;
  float s = 0;
  for (uint i = t; i < p.dim; i += tn) s += xr[i] * xr[i];
  s = simd_sum(s);
  if (lane == 0) part[sg] = s;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    float v = lane < (tn + 31) / 32 ? part[lane] : 0.0f;
    v = simd_sum(v);
    if (lane == 0) part[0] = v;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float r = rsqrt(part[0] / p.dim + p.eps);
  for (uint i = t; i < p.dim; i += tn) row[i] = xr[i] * r * w[i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  device float *yr = y + ((size_t)g.y * p.heads + g.x) * p.dim;
  uint half_rot = p.rot / 2;
  float pos = float(rows[g.y].pos);
  for (uint i = t; i < p.dim; i += tn) {
    float v = row[i];
    if (i < p.rot) {
      uint j = i < half_rot ? i : i - half_rot;
      float f = pos * pow(p.theta, -2.0f * float(j) / float(p.rot));
      float c = cos(f), sn = sin(f);
      v = i < half_rot ? row[i] * c - row[i + half_rot] * sn : row[i - half_rot] * sn + row[i] * c;
    }
    yr[i] = v;
  }
}

struct Kv { uint n, width, layer, pad; };  // width: kv heads * head dim

// Row g.y's key and value (FP32, `width` values each) into its sequence's
// cache at its position, as FP16.
kernel void kv_store(device const float *k [[buffer(0)]], device const float *v [[buffer(1)]],
                     constant Seq *rows [[buffer(2)]], constant Kv &p [[buffer(3)]],
                     uint2 g [[thread_position_in_grid]]) {
  if (g.x >= p.width) return;
  size_t at = ((size_t)p.layer * rows[g.y].cap + rows[g.y].pos) * p.width + g.x;
  rows[g.y].k[at] = half(k[(size_t)g.y * p.width + g.x]);
  rows[g.y].v[at] = half(v[(size_t)g.y * p.width + g.x]);
}

struct Gqa { uint heads, kv, dim, layer; float scale; uint blocks, pad0, pad1; };  // blocks: partials per head

// Attention of row g.z's query heads of key head g.x (heads / kv of them,
// at most 8) over the positions of block g.y (BLK of them, up to the row's
// position): per head the block's highest score m, the sum l of exp(score -
// m) and the sum of exp(score - m) * value, for gqa_combine.
kernel void gqa_block(device const float *q [[buffer(0)]], constant Seq *rows [[buffer(1)]],
                         device float *part [[buffer(2)]], device float2 *ml [[buffer(3)]],
                         constant Gqa &p [[buffer(4)]], uint3 g [[threadgroup_position_in_grid]],
                         uint j [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
  uint kvh = g.x, b = g.y, t = g.z, pos = rows[t].pos, G = p.heads / p.kv;
  if (b * BLK > pos) return;
  // the queries (8 heads x 256) and the scores (8 x BLK), and after them the
  // simdgroups' value sums (8 simdgroups x 2 heads x 256)
  threadgroup float stage[8 * 256 + 8 * BLK];
  threadgroup float(*qs)[256] = (threadgroup float(*)[256])stage;
  threadgroup float(*ps)[BLK] = (threadgroup float(*)[BLK])(stage + 8 * 256);
  threadgroup float red[8][8];
  for (uint i = j; i < G * p.dim; i += BLK) qs[i / p.dim][i % p.dim] = q[((size_t)t * p.heads + kvh * G) * p.dim + i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint at = b * BLK + j, width = p.kv * p.dim;
  size_t layer = (size_t)p.layer * rows[t].cap * width;
  // the scores: a simdgroup reads one key row at a time, its lanes 8
  // dimensions each (a row of 256 values: coalesced), each head's partial
  // dot products summed across the lanes
  {
    float4 qa[8][2];
    for (uint h = 0; h < G; h++)
      qa[h][0] = *(threadgroup const float4 *)&qs[h][lane * 8], qa[h][1] = *(threadgroup const float4 *)&qs[h][lane * 8 + 4];
    for (uint i = sg; i < BLK; i += BLK / 32) {
      uint ati = b * BLK + i;
      if (ati > pos) break;
      device const half4 *kr = (device const half4 *)(rows[t].k + layer + (size_t)ati * width + kvh * p.dim + lane * 8);
      float4 k0 = float4(kr[0]), k1 = float4(kr[1]);
      for (uint h = 0; h < G; h++) {
        float d = simd_sum(dot(k0, qa[h][0]) + dot(k1, qa[h][1]));
        if (lane == 0) ps[h][i] = d * p.scale;
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float sc[8];
  for (uint h = 0; h < G; h++) sc[h] = at <= pos ? ps[h][j] : -INFINITY;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint h = 0; h < G; h++) {
    float m = simd_max(sc[h]);
    if (lane == 0) red[h][sg] = m;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float mh[8];
  for (uint h = 0; h < G; h++) {
    float m = red[h][0];
    for (uint s = 1; s < BLK / 32; s++) m = max(m, red[h][s]);
    mh[h] = m;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint h = 0; h < G; h++) {
    float e = at <= pos ? exp(sc[h] - mh[h]) : 0.0f;
    ps[h][j] = e;
    float s = simd_sum(e);
    if (lane == 0) red[h][sg] = s;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint valid = min((uint)BLK, pos + 1 - b * BLK);
  size_t first = ((size_t)t * p.heads + kvh * G) * p.blocks + b;
  // the values: simdgroup sg takes the block's positions 32 sg .. 32 sg + 31,
  // lane l the dimensions 8 l .. 8 l + 7 (a value row coalesced); the
  // simdgroups' sums meet in threadgroup memory two heads at a time, added in
  // simdgroup order (in the queries' and scores' place, done with)
  {
    float4 acc[8][2];
    for (uint h = 0; h < G; h++) acc[h][0] = acc[h][1] = 0;
    device const half4 *vr = (device const half4 *)(rows[t].v + layer + (size_t)(b * BLK) * width + kvh * p.dim + lane * 8);
    uint lo = sg * 32, hi = min(lo + 32, valid);
    for (uint i = lo; i < hi; i++) {
      device const half4 *v = vr + (size_t)i * (width / 4);
      float4 v0 = float4(v[0]), v1 = float4(v[1]);
      for (uint h = 0; h < G; h++) {
        float w = ps[h][i];
        acc[h][0] += w * v0, acc[h][1] += w * v1;
      }
    }
    threadgroup float4(*vs)[2][64] = (threadgroup float4(*)[2][64])stage;  // [8 simdgroups][2 heads][256 values]
    for (uint h0 = 0; h0 < G; h0 += 2) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint k = 0; k < 2 && h0 + k < G; k++) vs[sg][k][lane * 2] = acc[h0 + k][0], vs[sg][k][lane * 2 + 1] = acc[h0 + k][1];
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint k = 0; k < 2 && h0 + k < G; k++) {
        float sum = 0;
        for (uint s2 = 0; s2 < BLK / 32; s2++) sum += ((threadgroup float *)vs[s2][k])[j];
        part[(first + (h0 + k) * p.blocks) * p.dim + j] = sum;
      }
    }
  }
  if (j < G) {
    float l = 0;
    for (uint s = 0; s < BLK / 32; s++) l += red[j][s];
    ml[first + j * p.blocks] = float2(mh[j], l);
  }
}

// Row g.y's head g.x: the blocks' partial sums combined, times sigmoid of the
// head's gate (qg: rows of heads * 2 * dim, the gate after each head's query).
kernel void gqa_combine(device const float *part [[buffer(0)]], device const float2 *ml [[buffer(1)]],
                           device const float *qg [[buffer(2)]], device float *out [[buffer(3)]],
                           constant Seq *rows [[buffer(4)]], constant Gqa &p [[buffer(5)]],
                           uint2 g [[threadgroup_position_in_grid]], uint d [[thread_index_in_threadgroup]]) {
  uint nb = rows[g.y].pos / BLK + 1;
  size_t r = (size_t)g.y * p.heads + g.x;
  float m = -INFINITY;
  for (uint b = 0; b < nb; b++) m = max(m, ml[r * p.blocks + b].x);
  float l = 0, acc = 0;
  for (uint b = 0; b < nb; b++) {
    float2 v = ml[r * p.blocks + b];
    float w = exp(v.x - m);
    l += v.y * w;
    acc += part[(r * p.blocks + b) * p.dim + d] * w;
  }
  out[r * p.dim + d] = acc / l * sigm(qg[r * 2 * p.dim + p.dim + d]);
}

// ---- attention of a prompt pass, on the matrix units ----

struct GqaPrompt { uint heads, kv, n, p0, layer, cap, rows, pad; float scale; uint pad0, pad1, pad2; };  // rows: n padded to 32

// The pass's queries (FP32 rows of heads * 256) as FP16, by head: qh[head][row][256], rows n.. zero.
kernel void gqa_q_half(device const float *q [[buffer(0)]], device half *qh [[buffer(1)]], constant GqaPrompt &p [[buffer(2)]],
                   uint3 g [[thread_position_in_grid]]) {
  uint d = g.x, r = g.y, h = g.z;
  if (d >= 256 || r >= p.rows) return;
  qh[((size_t)h * p.rows + r) * 256 + d] = r < p.n ? half(q[((size_t)r * p.heads + h) * 256 + d]) : half(0);
}

// Attention of query head g.x over 32 consecutive rows of one sequence,
// rows 32 g.y on (at positions p0 on), with the matrix units: per chunk of 32
// keys counted from position 0, the scores (FP16 queries and keys, FP32
// sums), an online softmax per row, and the probabilities (FP16) times the
// values; the output times sigmoid of the head's gate (qg: rows of heads *
// 512, the gate after each head's query). A row's result depends only on its
// position and the keys and values up to it.
kernel void gqa_prompt(device const half *qh [[buffer(0)]], constant Seq *rows [[buffer(1)]],
                          device const float *qg [[buffer(3)]], device float *out [[buffer(4)]],
                          constant GqaPrompt &p [[buffer(5)]], uint2 g [[threadgroup_position_in_grid]],
                          uint t [[thread_index_in_threadgroup]]) {
  device const half *K = rows[0].k, *V = rows[0].v;
  threadgroup half kv[32 * 128], ps[32 * 32];
  threadgroup float ss[32 * 32], mx[32], sum[32], rescale[32];
  uint h = g.x, r0 = g.y * 32, kvh = h / (p.heads / p.kv), width = p.kv * 256;
  uint plast = p.p0 + min(r0 + 32, p.n) - 1;  // the tile's last row's position
  size_t lay = (size_t)p.layer * p.cap * width + kvh * 256;
  if (t < 32) mx[t] = -1e30f, sum[t] = 0;
  constexpr auto sdesc = matmul2d_descriptor(32, 32, 128, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  constexpr auto odesc = matmul2d_descriptor(32, 128, 32, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<sdesc, execution_simdgroups<8>> sop;
  matmul2d<odesc, execution_simdgroups<8>> oop;
  tensor<device half, dextents<int32_t, 2>, tensor_inline> tq((device half *)qh + ((size_t)h * p.rows + r0) * 256, dextents<int32_t, 2>(256, 32));
  tensor<threadgroup half, extents<int32_t, 128, 32>, tensor_inline> tk(kv, extents<int32_t, 128, 32>());
  tensor<threadgroup half, extents<int32_t, 32, 32>, tensor_inline> tp(ps, extents<int32_t, 32, 32>());
  auto acc0 = oop.get_destination_cooperative_tensor<decltype(tp), decltype(tk), float>();
  auto acc1 = oop.get_destination_cooperative_tensor<decltype(tp), decltype(tk), float>();
  for (uint16_t i = 0; i < acc0.get_capacity(); i++)
    if (acc0.is_valid_element(i)) acc0[i] = acc1[i] = 0;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint c0 = 0; c0 <= plast; c0 += 32) {
    auto sc = sop.get_destination_cooperative_tensor<decltype(tq.slice(0, 0)), decltype(tk), float>();
    for (uint16_t i = 0; i < sc.get_capacity(); i++)
      if (sc.is_valid_element(i)) sc[i] = 0;
    for (uint ds = 0; ds < 2; ds++) {  // scores: the chunk's keys, 128 columns at a time
      for (uint i = t; i < 32 * 32; i += 256) {  // 32 keys x 32 half4
        uint j = i / 32, c = i % 32, at = c0 + j;
        *(threadgroup half4 *)(kv + j * 128 + c * 4) =
            at <= plast ? *(device const half4 *)(K + lay + (size_t)at * width + ds * 128 + c * 4) : half4(0);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      auto qd = tq.slice(ds * 128, 0);
      sop.run(qd, tk, sc);
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint16_t i = 0; i < sc.get_capacity(); i++)
      if (sc.is_valid_element(i)) {
        auto a = sc.get_multidimensional_index(i);  // a[0]: the key, a[1]: the row
        uint row = r0 + a[1];
        bool valid = row < p.n && c0 + a[0] <= p.p0 + row;
        ss[a[1] * 32 + a[0]] = valid ? sc[i] * p.scale : -INFINITY;
      }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t < 32) {  // row t's online softmax over the chunk
      float m = mx[t];
      for (uint j = 0; j < 32; j++) m = max(m, ss[t * 32 + j]);
      float f = exp(mx[t] - m), s2 = sum[t] * f;
      for (uint j = 0; j < 32; j++) {
        float e = exp(ss[t * 32 + j] - m);
        s2 += e;
        ps[t * 32 + j] = half(e);
      }
      mx[t] = m, sum[t] = s2, rescale[t] = f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint16_t i = 0; i < acc0.get_capacity(); i++)
      if (acc0.is_valid_element(i)) {
        float f = rescale[acc0.get_multidimensional_index(i)[1]];
        acc0[i] *= f, acc1[i] *= f;
      }
    for (uint ds = 0; ds < 2; ds++) {  // values: the same keys' rows, 128 columns at a time
      for (uint i = t; i < 32 * 32; i += 256) {
        uint j = i / 32, c = i % 32, at = c0 + j;
        *(threadgroup half4 *)(kv + j * 128 + c * 4) =
            at <= plast ? *(device const half4 *)(V + lay + (size_t)at * width + ds * 128 + c * 4) : half4(0);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (ds == 0) oop.run(tp, tk, acc0);
      else oop.run(tp, tk, acc1);
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
  for (uint16_t i = 0; i < acc0.get_capacity(); i++)
    if (acc0.is_valid_element(i)) {
      auto a = acc0.get_multidimensional_index(i);  // a[0]: the column, a[1]: the row
      uint row = r0 + a[1];
      if (row >= p.n) continue;
      size_t r = (size_t)row * p.heads + h;
      for (uint ds = 0; ds < 2; ds++) {
        uint d = ds * 128 + a[0];
        float v = (ds ? acc1[i] : acc0[i]) / sum[a[1]];
        out[r * 256 + d] = v * sigm(qg[r * 512 + 256 + d]);
      }
    }
}
