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

// The queries, keys and values as BF16 for the matrix units, which vit_bias
// and vit_rope already rounded them to: of head h, rows of 64 in hv at
// (s 16 + h) np 64 for s = 0, 1, 2 (queries, keys, values), the np - n rows
// after the patches' zero. Thread x: four of a patch's 3072 values.
kernel void vit_heads(device const float4 *qkv [[buffer(0)]], device bfloat4 *hv [[buffer(1)]],
                      constant uint2 &dims [[buffer(2)]], uint2 i [[thread_position_in_grid]]) {
  uint n = dims.x, np = dims.y, c = i.x * 4, s = c / 1024, h = c % 1024 / 64;
  hv[(((s * 16 + h) * np + i.y) * 64 + c % 64) / 4] = i.y < n ? bfloat4(qkv[i.y * 768 + i.x]) : bfloat4(0);
}

// Attention over all n patches of an image (scaled_dot_product_attention
// without a mask): 16 heads of 64, scale 1/8, on the matrix units. A
// threadgroup takes 64 queries of one head, each of its 4 simdgroups 16 of
// them on its own, over blocks of 64 keys: the scores accumulate in FP32, an
// online softmax in FP32 over the simdgroup's rows of scores, the
// probabilities rounded to BF16 for their product with the values (PyTorch's
// CPU attention rounds them too for the reference's BF16 inputs, over blocks
// of 512 keys, as an emulation of it showed), and the outputs accumulate in
// FP32, rounded to BF16. The
// probabilities pass from the scores' layout to the second product's through
// get_left_input_cooperative_tensor, which needs a source of its element type
// (a float source passed this way gave wrong rows): a change of the products'
// shapes or types needs tools/reference/check.py vision again.
kernel void vit_attn(device const bfloat *hv [[buffer(0)]], device float *o [[buffer(1)]],
                     constant uint2 &dims [[buffer(2)]], uint2 tg [[threadgroup_position_in_grid]],
                     uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float state[4][4][16];  // a simdgroup's rows: running maximum, sum, rescale, reduction
  threadgroup float *mx = state[sg][0], *sum = state[sg][1], *rescale = state[sg][2], *red_rows = state[sg][3];
  uint n = dims.x, np = dims.y, h = tg.y, q0 = tg.x * 64 + sg * 16;
  constexpr auto sdesc = matmul2d_descriptor(16, 64, 64, false, true, false, matmul2d_descriptor::mode::multiply);
  constexpr auto odesc = matmul2d_descriptor(16, 64, 64, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<sdesc, execution_simdgroup> sop;
  matmul2d<odesc, execution_simdgroup> oop;
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tq((device bfloat *)hv + h * np * 64, dextents<int32_t, 2>(64, np));
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tk((device bfloat *)hv + (16 + h) * np * 64, dextents<int32_t, 2>(64, np));
  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tv((device bfloat *)hv + (32 + h) * np * 64, dextents<int32_t, 2>(64, np));
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
      if (sc.is_valid_element(i)) sc[i] = k0 + sc.get_multidimensional_index(i)[0] < n ? sc[i] * 0.125f : -INFINITY;
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
      if (q0 + a[1] < n) o[(q0 + a[1]) * 1024 + h * 64 + a[0]] = bf(acc[i] / sum[a[1]]);
    }
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
