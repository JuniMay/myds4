// The gated DeltaNet's recurrent layers: the causal depthwise convolution
// over q, k and v, the gates, and the gated delta rule. A pass's rows may
// belong to different sequences (struct Seq), each sequence's rows
// consecutive and at consecutive positions; a row's arithmetic does not
// depend on the other rows of the pass.

// A row's sequence (GPU addresses): the DeltaNet and convolution states, and
// the keys and values of the attention layers (gqa.metal).
struct Seq {
  device float *gdn, *gdn_save;    // DeltaNet states [linear layer][value head][value dim][key dim]
  device float *conv, *conv_save;  // convolution inputs [linear layer][3][channels], oldest first
  device half *k, *v;              // [attention layer][cap][kv head][head dim]
  uint pos, cap;                   // the row's position; the keys' capacity
  int rope_t, rope_h, rope_w, pad;
};

struct Conv { uint channels, n, layer, pad; };

// The causal depthwise convolution (kernel 4) of channel c over the rows'
// projected q, k and v, then silu. A sequence's 3 previous inputs move on by
// one a row; a row with conv_save also leaves them there.
kernel void conv(device const float *in [[buffer(0)]], device float *out [[buffer(1)]],
                 device const float *w [[buffer(2)]], constant Seq *rows [[buffer(3)]],
                 constant Conv &p [[buffer(4)]], uint c [[thread_position_in_grid]]) {
  uint C = p.channels;
  if (c >= C) return;
  size_t lo = (size_t)p.layer * 3 * C;
  float w0 = w[c * 4], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3];
  device float *st = 0;
  float s0 = 0, s1 = 0, s2 = 0;
  for (uint t = 0; t < p.n; t++) {
    device float *rs = rows[t].conv + lo;
    if (rs != st) {
      if (st) st[c] = s0, st[C + c] = s1, st[2 * C + c] = s2;
      st = rs, s0 = st[c], s1 = st[C + c], s2 = st[2 * C + c];
    }
    float x = in[(size_t)t * C + c];
    float v = w0 * s0 + w1 * s1 + w2 * s2 + w3 * x;
    out[(size_t)t * C + c] = v * sigm(v);
    s0 = s1, s1 = s2, s2 = x;
    if (rows[t].conv_save) {
      device float *sv = rows[t].conv_save + lo;
      sv[c] = s0, sv[C + c] = s1, sv[2 * C + c] = s2;
    }
  }
  if (st) st[c] = s0, st[C + c] = s1, st[2 * C + c] = s2;
}

// The gates of value head i % heads of row i / heads: beta = sigmoid(b), the
// decay g = exp(-A * softplus(a + dt_bias)), A = exp(A_log).
kernel void gdn_gates(device const float *a [[buffer(0)]], device const float *b [[buffer(1)]],
                      device const float *A [[buffer(2)]], device const float *dtb [[buffer(3)]],
                      device float *g [[buffer(4)]], device float *beta [[buffer(5)]],
                      constant uint2 &p [[buffer(6)]], uint i [[thread_position_in_grid]]) {
  if (i >= p.x * p.y) return;
  uint h = i % p.x;
  float x = a[i] + dtb[h];
  float sp = max(x, 0.0f) + log1p_(exp(-fabs(x)));
  g[i] = exp(-A[h] * sp);
  beta[i] = sigm(b[i]);
}

struct Gdn { uint n, heads, ratio, layer, channels, keys, pad0, pad1; };  // keys: key heads * 128

// The gated delta rule of value head hv and value dimension dv over the
// rows: the state row S[dv][0..128) (32 lanes, 4 each) decays by g, learns
// beta * (v - S.k) along k, and gives S.q. qkv rows: queries and keys (key
// heads of 128, each serving `ratio` consecutive value heads), then values.
// A row with gdn_save also leaves the state there.
kernel void gdn(device const float *qkv [[buffer(0)]], device const float *g [[buffer(1)]],
                device const float *beta [[buffer(2)]], constant Seq *rows [[buffer(3)]],
                device float *out [[buffer(4)]], constant Gdn &p [[buffer(5)]],
                uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                uint lane [[thread_index_in_simdgroup]]) {
  uint hv = tg.x, dv = tg.y * 4 + sg, hk = hv / p.ratio;
  size_t off = (((size_t)p.layer * p.heads + hv) * 128 + dv) * 128 + lane * 4;
  device float *st = 0;
  float4 s = 0;
  for (uint t = 0; t < p.n; t++) {
    device float *rs = rows[t].gdn + off;
    if (rs != st) {
      if (st) *(device float4 *)st = s;
      st = rs, s = *(device float4 *)st;
    }
    device const float *r = qkv + (size_t)t * p.channels;
    float4 q4 = *(device const float4 *)(r + hk * 128 + lane * 4);
    float4 k4 = *(device const float4 *)(r + p.keys + hk * 128 + lane * 4);
    float v = r[2 * p.keys + hv * 128 + dv];
    s *= g[t * p.heads + hv];
    float mem = simd_sum(dot(s, k4));
    float delta = (v - mem) * beta[t * p.heads + hv];
    s += k4 * delta;
    float o = simd_sum(dot(s, q4));
    if (lane == 0) out[((size_t)t * p.heads + hv) * 128 + dv] = o;
    if (rows[t].gdn_save) *(device float4 *)(rows[t].gdn_save + off) = s;
  }
  if (st) *(device float4 *)st = s;
}
