// Operations on FP32 activation rows: embedding lookup, RMSNorm (alone,
// gated, or after a residual add), L2 normalization of heads, concatenation,
// SiLU times a second input, and the residual add.

struct Norm { uint n; float eps; float scale; uint gate; };

// Row r of n values: y = x / sqrt(mean(x^2) + eps) * w * scale, and with
// gate (the gated DeltaNet's output) times silu(z). One threadgroup a row.
kernel void rms_norm(device const float *x [[buffer(0)]], device const float *w [[buffer(1)]],
                     device float *y [[buffer(2)]], constant Norm &p [[buffer(3)]],
                     device const float *z [[buffer(4)]], uint row [[threadgroup_position_in_grid]],
                     uint t [[thread_position_in_threadgroup]], uint tn [[threads_per_threadgroup]],
                     uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float part[32];
  device const float *xr = x + (size_t)row * p.n;
  float s = 0;
  for (uint i = t; i < p.n; i += tn) s += xr[i] * xr[i];
  s = simd_sum(s);
  if (lane == 0) part[sg] = s;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    float v = lane < (tn + 31) / 32 ? part[lane] : 0.0f;
    v = simd_sum(v);
    if (lane == 0) part[0] = v;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float r = rsqrt(part[0] / p.n + p.eps) * p.scale;
  device float *yr = y + (size_t)row * p.n;
  for (uint i = t; i < p.n; i += tn) {
    float v = xr[i] * r * w[i];
    if (p.gate) {
      float g = z[(size_t)row * p.n + i];
      if (p.gate == 2) v = bf(bf(xr[i] * r) * w[i]) * sigm(g);
      else v *= g * sigm(g);
    }
    yr[i] = v;
  }
}

// x += r, then y = RMSNorm(x) * w: add then rms_norm in one kernel, the same
// arithmetic (each thread adds the values it then squares and scales).
kernel void add_norm(device float *x [[buffer(0)]], device const float *r [[buffer(1)]],
                     device const float *w [[buffer(2)]], device float *y [[buffer(3)]],
                     constant Norm &p [[buffer(4)]], uint row [[threadgroup_position_in_grid]],
                     uint t [[thread_position_in_threadgroup]], uint tn [[threads_per_threadgroup]],
                     uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float part[32];
  device float *xr = x + (size_t)row * p.n;
  device const float *rr = r + (size_t)row * p.n;
  float s = 0;
  for (uint i = t; i < p.n; i += tn) {
    float v = xr[i] + rr[i];
    xr[i] = v;
    s += v * v;
  }
  s = simd_sum(s);
  if (lane == 0) part[sg] = s;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    float v = lane < (tn + 31) / 32 ? part[lane] : 0.0f;
    v = simd_sum(v);
    if (lane == 0) part[0] = v;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float q = rsqrt(part[0] / p.n + p.eps) * p.scale;
  device float *yr = y + (size_t)row * p.n;
  for (uint i = t; i < p.n; i += tn) yr[i] = xr[i] * q * w[i];
}

struct L2 { uint n, stride; float eps, scale; };

// Head g.x of row g.y (n values at x + g.y * stride + g.x * n): divided by
// sqrt(sum(x^2) + eps), times scale, in place. One threadgroup a head.
kernel void l2_norm(device float *x [[buffer(0)]], constant L2 &p [[buffer(1)]],
                    uint2 g [[threadgroup_position_in_grid]], uint2 t2 [[thread_position_in_threadgroup]],
                    uint2 tn2 [[threads_per_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                    uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float part[32];
  uint t = t2.x, tn = tn2.x;
  device float *xr = x + (size_t)g.y * p.stride + g.x * p.n;
  float s = 0;
  for (uint i = t; i < p.n; i += tn) s += xr[i] * xr[i];
  s = simd_sum(s);
  if (lane == 0) part[sg] = s;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    float v = lane < (tn + 31) / 32 ? part[lane] : 0.0f;
    v = simd_sum(v);
    if (lane == 0) part[0] = v;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float r = rsqrt(part[0] + p.eps) * p.scale;
  for (uint i = t; i < p.n; i += tn) xr[i] *= r;
}

// Row g.y of y: a's row then b's (n values each).
kernel void concat(device const float *a [[buffer(0)]], device const float *b [[buffer(1)]], device float *y [[buffer(2)]],
                   constant uint &n [[buffer(3)]], uint2 g [[thread_position_in_grid]]) {
  if (g.x < 2 * n) y[(size_t)g.y * 2 * n + g.x] = g.x < n ? a[(size_t)g.y * n + g.x] : b[(size_t)g.y * n + g.x - n];
}

// h = silu(g) * u
kernel void silu_mul(device const float *g [[buffer(0)]], device const float *u [[buffer(1)]],
                     device float *h [[buffer(2)]], constant uint &n [[buffer(3)]], uint i [[thread_position_in_grid]]) {
  if (i < n) h[i] = g[i] * sigm(g[i]) * u[i];
}

// x += r
kernel void add(device float *x [[buffer(0)]], device const float *r [[buffer(1)]], constant uint &n [[buffer(2)]],
                uint i [[thread_position_in_grid]]) {
  if (i < n) x[i] += r[i];
}

struct GroupRms {
  uint width, groups;
  float eps;
  uint pad;
};
kernel void group_rms_bf16(device const float *x [[buffer(0)]], device const float *w [[buffer(1)]],
                       device float *out [[buffer(2)]], constant GroupRms &p [[buffer(3)]],
                       uint row [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
                       uint threads [[threads_per_threadgroup]]) {
  threadgroup float partial[32];
  float sum = 0;
  for (uint i = t; i < p.width; i += threads) {
    float v = x[(size_t)row * p.width + i];
    sum += v * v;
  }
  float inv = rsqrt(tg_sum(sum, partial, t, threads) / p.width + p.eps);
  for (uint i = t; i < p.width; i += threads)
    out[(size_t)row * p.width + i] = bf(x[(size_t)row * p.width + i] * inv * w[(row % p.groups) * p.width + i]);
}
kernel void round_bf16(device float *x [[buffer(0)]], constant uint &n [[buffer(1)]],
                        uint i [[thread_position_in_grid]]) {
  if (i < n)
    x[i] = bf(x[i]);
}
kernel void clear_f32(device float *x [[buffer(0)]], constant uint &n [[buffer(1)]],
                       uint i [[thread_position_in_grid]]) {
  if (i < n)
    x[i] = 0;
}

struct WeightedRow {
  uint row;
  float weight;
};
kernel void gather_rows(device const float *x [[buffer(0)]], device const WeightedRow *entries [[buffer(1)]],
                         device float *out [[buffer(2)]], constant uint2 &p [[buffer(3)]],
                         uint i [[thread_position_in_grid]]) {
  if (i < p.x * p.y)
    out[i] = x[(size_t)entries[i / p.x].row * p.x + i % p.x];
}

kernel void embed_rows(device const ushort *embedding [[buffer(0)]], device const int *tokens [[buffer(1)]],
                            device float *out [[buffer(2)]], constant uint &width [[buffer(3)]],
                            device const ulong *images [[buffer(4)]], uint2 i [[thread_position_in_grid]]) {
  if (i.x >= width) return;
  device const ushort *row = images[i.y] ? (device const ushort *)images[i.y] : embedding + (size_t)tokens[i.y] * width;
  out[(size_t)i.y * width + i.x] = bf16f(row[i.x]);
}
