// Direct FP8 tensor operands, compared with the BF16 staging path.
kernel void fp8_pack_native(device const float *x [[buffer(0)]], device uchar *codes [[buffer(1)]],
                             device uchar *scales [[buffer(2)]], constant uint2 &kl [[buffer(3)]],
                             uint2 i [[thread_position_in_grid]]) {
  if (i.x >= kl.x) return;
  float value = x[i.y * kl.y + i.x];
  float scale = pow2_scale(max(simd_max(fabs(value)), 1e-4f), 1.0f / 448.0f);
  float q = e4m3(clamp(value / scale, -448.0f, 448.0f));
  ushort bits = as_type<ushort>(half(q * 0x1p-8f));
  codes[i.y * kl.x + i.x] = uchar(((bits >> 7) & 0x7f) | ((bits >> 8) & 0x80));
  if (!(i.x & 31)) scales[i.y * (kl.x / 32) + i.x / 32] = uchar(as_type<uint>(scale) >> 23);
}

using Fp8Rows = tensor<device metal_fp8_e4m3_format, dextents<int32_t, 2>, tensor_inline,
                       tensor_blockwise<tensor_plane_scales, device metal_fp8_ue8m0_format, 32, 1>>;

kernel void mm_fp8_native(device const uchar *w [[buffer(0)]], device const uchar *ws [[buffer(1)]],
                          device const uchar *x [[buffer(2)]], device const uchar *xs [[buffer(3)]],
                          device float *y [[buffer(4)]], constant MM &p [[buffer(5)]],
                          uint2 tg [[threadgroup_position_in_grid]]) {
  constexpr auto desc = matmul2d_descriptor(64, 64, 64, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<4>> op;
  Fp8Rows a((device uchar *)x, dextents<int32_t, 2>(p.k, (p.m + 63) / 64 * 64),
            tensor_blockwise<tensor_plane_scales, device metal_fp8_ue8m0_format, 32, 1>((device uchar *)xs));
  // The API requires one scale row per weight row. The checkpoint's shared
  // 32-row scales are duplicated once at load, without changing values.
  Fp8Rows b((device uchar *)w, dextents<int32_t, 2>(p.k, p.n),
             tensor_blockwise<tensor_plane_scales, device metal_fp8_ue8m0_format, 32, 1>((device uchar *)ws));
  uint m0 = tg.y * 64, n0 = tg.x * 64;
  auto c = op.get_destination_cooperative_tensor<decltype(a.slice(0, 0)), decltype(b.slice(0, 0)), float>();
  for (uint16_t i = 0; i < c.get_capacity(); i++)
    if (c.is_valid_element(i)) c[i] = 0;
  for (uint k0 = 0; k0 < p.k; k0 += 64) {
    auto at = a.slice(k0, m0);
    auto bt = b.slice(k0, n0);
    op.run(at, bt, c);
  }
  for (uint16_t i = 0; i < c.get_capacity(); i++)
    if (c.is_valid_element(i)) {
      auto at = c.get_multidimensional_index(i);
      uint row = m0 + at[1], col = n0 + at[0];
      if (row < p.m && col < p.n) y[row * p.ldy + col] = p.round ? bf(c[i]) : c[i];
    }
}

kernel void mm_fp8_native128(device const uchar *w [[buffer(0)]], device const uchar *ws [[buffer(1)]],
                          device const uchar *x [[buffer(2)]], device const uchar *xs [[buffer(3)]],
                          device float *y [[buffer(4)]], constant MM &p [[buffer(5)]],
                          uint2 tg [[threadgroup_position_in_grid]]) {
  constexpr auto desc = matmul2d_descriptor(64, 64, 64, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<4>> op;
  Fp8Rows a((device uchar *)x, dextents<int32_t, 2>(p.k, (p.m + 127) / 128 * 128),
            tensor_blockwise<tensor_plane_scales, device metal_fp8_ue8m0_format, 32, 1>((device uchar *)xs));
  // The API requires one scale row per weight row. The checkpoint's shared
  // 32-row scales are duplicated once at load, without changing values.
  Fp8Rows b((device uchar *)w, dextents<int32_t, 2>(p.k, p.n),
             tensor_blockwise<tensor_plane_scales, device metal_fp8_ue8m0_format, 32, 1>((device uchar *)ws));
  uint m0 = tg.y * 128, n0 = tg.x * 64;
  auto c = op.get_destination_cooperative_tensor<decltype(a.slice(0, 0)), decltype(b.slice(0, 0)), float>();
  auto c1 = op.get_destination_cooperative_tensor<decltype(a.slice(0, 0)), decltype(b.slice(0, 0)), float>();
  for (uint16_t i = 0; i < c.get_capacity(); i++)
    if (c.is_valid_element(i)) c[i] = c1[i] = 0;
  for (uint k0 = 0; k0 < p.k; k0 += 64) {
    auto at = a.slice(k0, m0);
    auto bt = b.slice(k0, n0);
    op.run(at, bt, c);
    if (m0 + 64 < p.m) {
      auto at1 = a.slice(k0, m0 + 64);
      op.run(at1, bt, c1);
    }
  }
  for (uint16_t i = 0; i < c.get_capacity(); i++)
    if (c.is_valid_element(i)) {
      auto at = c.get_multidimensional_index(i);
      uint row = m0 + at[1], col = n0 + at[0];
      if (row < p.m && col < p.n) y[row * p.ldy + col] = p.round ? bf(c[i]) : c[i];
      row += 64;
      if (row < p.m && col < p.n) y[row * p.ldy + col] = p.round ? bf(c1[i]) : c1[i];
    }
}


// This probe checks exact representability before accepting its results.
kernel void fp8_pack_half(device const float4 *x [[buffer(0)]], device half4 *y [[buffer(1)]],
                          constant uint2 &kl [[buffer(2)]], uint2 i [[thread_position_in_grid]]) {
  y[i.y * (kl.x / 4) + i.x] = half4(x[i.y * (kl.y / 4) + i.x] * 128.0f);
}

kernel void mm_fp8_half128(device const uchar *w [[buffer(0)]], device const uchar *ws [[buffer(1)]],
                          device const half *x [[buffer(2)]],
                          device float *y [[buffer(3)]], constant MM &p [[buffer(4)]],
                          uint2 tg [[threadgroup_position_in_grid]]) {
  constexpr auto desc = matmul2d_descriptor(64, 64, 64, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<4>> op;
  tensor<device half, dextents<int32_t, 2>, tensor_inline> a((device half *)x, dextents<int32_t, 2>(p.k, (p.m + 127) / 128 * 128));
  // The API requires one scale row per weight row. The checkpoint's shared
  // 32-row scales are duplicated once at load, without changing values.
  Fp8Rows b((device uchar *)w, dextents<int32_t, 2>(p.k, p.n),
             tensor_blockwise<tensor_plane_scales, device metal_fp8_ue8m0_format, 32, 1>((device uchar *)ws));
  uint m0 = tg.y * 128, n0 = tg.x * 64;
  auto c = op.get_destination_cooperative_tensor<decltype(a.slice(0, 0)), decltype(b.slice(0, 0)), float>();
  auto c1 = op.get_destination_cooperative_tensor<decltype(a.slice(0, 0)), decltype(b.slice(0, 0)), float>();
  for (uint16_t i = 0; i < c.get_capacity(); i++)
    if (c.is_valid_element(i)) c[i] = c1[i] = 0;
  for (uint k0 = 0; k0 < p.k; k0 += 64) {
    auto at = a.slice(k0, m0);
    auto bt = b.slice(k0, n0);
    op.run(at, bt, c);
    if (m0 + 64 < p.m) {
      auto at1 = a.slice(k0, m0 + 64);
      op.run(at1, bt, c1);
    }
  }
  for (uint16_t i = 0; i < c.get_capacity(); i++)
    if (c.is_valid_element(i)) {
      auto at = c.get_multidimensional_index(i);
      uint row = m0 + at[1], col = n0 + at[0];
      if (row < p.m && col < p.n) y[row * p.ldy + col] = p.round ? bf(c[i] * 0x1p-7f) : c[i] * 0x1p-7f;
      row += 64;
      if (row < p.m && col < p.n) y[row * p.ldy + col] = p.round ? bf(c1[i] * 0x1p-7f) : c1[i] * 0x1p-7f;
    }
}
