// ---- the vision encoder (inference/vision.py: ViT, Aligner) ----
//
// The reference runs in BF16: every product's output (after its bias), the
// norms, RoPE, attention and the activations are rounded to BF16 (bf), on
// rows of FP32 in memory. The products themselves are linear() with mm_bf16.

// y = bf(y + b) over rows of n.
kernel void vit_bias(device float *y [[buffer(0)]], device const float *b [[buffer(1)]],
                     constant uint &n [[buffer(2)]], uint2 i [[thread_position_in_grid]]) {
  y[i.y * n + i.x] = bf(y[i.y * n + i.x] + b[i.x]);
}

// RMSNorm over 1024 (eps 1e-6, FP32 weight): y = bf(w (x / rms(x))).
kernel void vit_norm(device const float *x [[buffer(0)]], device const float *w [[buffer(1)]],
                     device float *y [[buffer(2)]], uint row [[threadgroup_position_in_grid]],
                     uint t [[thread_index_in_threadgroup]]) {
  threadgroup float tmp[32];
  x += row * 1024, y += row * 1024;
  float ss = 0;
  for (uint i = t; i < 1024; i += 256) ss += x[i] * x[i];
  float r = rsqrt(tg_sum(ss, tmp, t, 256) / 1024 + 1e-6f);
  for (uint i = t; i < 1024; i += 256) y[i] = bf(w[i] * (x[i] * r));
}

// 2D RoPE of the queries and keys (apply_rotary): in each head of 64, value
// i < 32 pairs with i + 32 and turns by angle i of the patch (cs: its 32
// cosines, then its 32 sines). Thread x: queries for x < 512, keys after.
kernel void vit_rope(device float *qkv [[buffer(0)]], device const float *cs [[buffer(1)]],
                     uint2 g [[thread_position_in_grid]]) {
  uint i = g.x % 32;
  device float *v = qkv + g.y * 3072 + (g.x / 512) * 1024 + (g.x % 512) / 32 * 64;
  float c = cs[g.y * 64 + i], s = cs[g.y * 64 + 32 + i], x1 = v[i], x2 = v[i + 32];
  v[i] = bf(x1 * c - x2 * s);
  v[i + 32] = bf(x2 * c + x1 * s);
}

// Attention over all n patches of an image (scaled_dot_product_attention
// without a mask): 16 heads of 64, scale 1/8, an online softmax in FP32, the
// output rounded to BF16. A threadgroup of 256 threads takes 64 queries of
// one head, four threads a query, over blocks of 32 keys: thread part p of a
// query scores keys p, p + 4, ... of the block and sums the query's outputs
// p, p + 4, ... (strides that keep the threads on distinct memory banks).
kernel void vit_attn(device const float *qkv [[buffer(0)]], device float *o [[buffer(1)]],
                     constant uint &n [[buffer(2)]], uint2 tg [[threadgroup_position_in_grid]],
                     uint t [[thread_index_in_threadgroup]]) {
  threadgroup float ks[32 * 65], vs[32 * 64], ps[64 * 33];
  uint h = tg.y, qi = t / 4, p = t % 4, q = tg.x * 64 + qi;
  float qv[64], acc[16], m = -INFINITY, l = 0;
  for (uint d = 0; d < 64; d++) qv[d] = q < n ? qkv[q * 3072 + h * 64 + d] * 0.125f : 0;
  for (uint d = 0; d < 16; d++) acc[d] = 0;
  for (uint k0 = 0; k0 < n; k0 += 32) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = t; i < 32 * 64; i += 256) {
      uint r = k0 + i / 64, d = i % 64;
      ks[(i / 64) * 65 + d] = r < n ? qkv[r * 3072 + 1024 + h * 64 + d] : 0;
      vs[i] = r < n ? qkv[r * 3072 + 2048 + h * 64 + d] : 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float s[8], mx = -INFINITY;
    for (uint j = 0; j < 8; j++) {
      uint kj = 4 * j + p;
      float a = 0;
      for (uint d = 0; d < 64; d++) a += qv[d] * ks[kj * 65 + d];
      s[j] = k0 + kj < n ? a : -INFINITY;
      mx = max(mx, s[j]);
    }
    mx = max(mx, simd_shuffle_xor(mx, 1));
    mx = max(mx, simd_shuffle_xor(mx, 2));
    float mn = max(m, mx), c = exp(m - mn), sum = 0;
    for (uint j = 0; j < 8; j++) {
      float e = exp(s[j] - mn);
      sum += e;
      ps[qi * 33 + 4 * j + p] = e;
    }
    sum += simd_shuffle_xor(sum, 1);
    sum += simd_shuffle_xor(sum, 2);
    l = l * c + sum, m = mn;
    for (uint d = 0; d < 16; d++) acc[d] *= c;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j = 0; j < 32; j++) {
      float e = ps[qi * 33 + j];
      for (uint d = 0; d < 16; d++) acc[d] += e * vs[j * 64 + 4 * d + p];
    }
  }
  if (q < n)
    for (uint d = 0; d < 16; d++) o[q * 1024 + h * 64 + 4 * d + p] = bf(acc[d] / l);
}

// The MLP's activation: a = bf(bf(silu(gate)) up), gate and up the two
// halves of w1's 5632 outputs.
kernel void vit_swiglu(device const float *h [[buffer(0)]], device float *a [[buffer(1)]],
                       uint2 i [[thread_position_in_grid]]) {
  float g = h[i.y * 5632 + i.x], u = h[i.y * 5632 + 2816 + i.x];
  a[i.y * 2816 + i.x] = bf(bf(g / (1.0f + exp(-g))) * u);
}

// The residual: x = bf(x + y).
kernel void vit_add(device float *x [[buffer(0)]], device const float *y [[buffer(1)]], uint i [[thread_position_in_grid]]) {
  x[i] = bf(x[i] + y[i]);
}

// The Aligner's 3 x 3 merge (F.pad, then F.unfold with kernel and stride
// 3): token r = (oh, ow) of the th x tw grid takes value c of patch
// (3 oh + kh, 3 ow + kw) at 9 c + 3 kh + kw, zero beyond the patch grid.
// g: vh, vw, tw.
kernel void vit_unfold(device const float *x [[buffer(0)]], device float *u [[buffer(1)]],
                       constant uint3 &g [[buffer(2)]], uint2 i [[thread_position_in_grid]]) {
  uint c = i.x / 9, kh = i.x % 9 / 3, kw = i.x % 3, ph = i.y / g.z * 3 + kh, pw = i.y % g.z * 3 + kw;
  u[i.y * 9216 + i.x] = ph < g.x && pw < g.y ? x[(ph * g.y + pw) * 1024 + c] : 0.0f;
}
