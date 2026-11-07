// ---- routed experts in GGUF codes (--experts) ----
//
// The rows of an expert's w1, w3 and w2 in one of llama.cpp's codes, blocks
// of 256 values in sub-blocks of 32, decoded as llama.cpp's dequantize_row_*
// do (ggml-quants.c; MIT License, see gguf_grids.metal). A row of 5120 is 20
// blocks, one of 2304 is 9. An Expert's p[0], p[1], p[2] are its w1, w3 and
// w2. A kernel builds its code's tables in threadgroup memory first, as
// llama.cpp's Metal kernels copy theirs: the lanes look up different entries.

struct block_q2_K { uchar scales[16]; uchar qs[64]; half d, dmin; };
struct block_iq2_xxs { half d; ushort qs[32]; };
struct block_iq2_xs { half d; ushort qs[32]; uchar scales[8]; };
struct block_iq2_s { half d; uchar qs[64]; uchar qh[8]; uchar scales[8]; };
struct block_iq3_xxs { half d; uchar qs[96]; };
struct block_iq1_s { half d; uchar qs[32]; ushort qh[8]; };
struct block_iq1_m { uchar qs[32]; uchar qh[16]; uchar scales[8]; };

// The tables. The IQ2 and IQ3 grids hold each value as half, and the eight
// signs of a sign index are half sign bits, to XOR onto eight values. IQ1_S
// and IQ1_M keep llama.cpp's grid of nibbles; Q2_K has no table.
template <typename B> struct Code { enum { tab = 1 }; };  // uint4 entries
template <> struct Code<block_iq2_xxs> { enum { tab = 256 + 128 }; };
template <> struct Code<block_iq2_xs> { enum { tab = 512 + 128 }; };
template <> struct Code<block_iq2_s> { enum { tab = 1024 + 256 }; };
template <> struct Code<block_iq3_xxs> { enum { tab = 128 + 128 }; };
template <> struct Code<block_iq1_s> { enum { tab = 512 }; };
template <> struct Code<block_iq1_m> { enum { tab = 512 }; };
typedef threadgroup const uint4 *Tab;
inline uint4 half_bytes(ulong g) {  // eight bytes as halves
  uint4 h;
  for (uint c = 0; c < 4; c++) h[c] = as_type<uint>(half2(float((g >> 16 * c) & 255), float((g >> (16 * c + 8)) & 255)));
  return h;
}
inline uint4 sign_bits(uint s) {  // eight sign bits as half sign bits
  uint4 m;
  for (uint c = 0; c < 4; c++) m[c] = ((s >> 2 * c) & 1) << 15 | ((s >> (2 * c + 1)) & 1) << 31;
  return m;
}
template <typename B> void tables(threadgroup uint4 *tab, uint t, uint nt);
template <> void tables<block_q2_K>(threadgroup uint4 *tab, uint t, uint nt) {}
template <> void tables<block_iq2_xxs>(threadgroup uint4 *tab, uint t, uint nt) {
  for (uint i = t; i < 256; i += nt) tab[i] = half_bytes(iq2xxs_grid[i]);
  for (uint i = t; i < 128; i += nt) tab[256 + i] = sign_bits(ksigns_iq2xs[i]);
}
template <> void tables<block_iq2_xs>(threadgroup uint4 *tab, uint t, uint nt) {
  for (uint i = t; i < 512; i += nt) tab[i] = half_bytes(iq2xs_grid[i]);
  for (uint i = t; i < 128; i += nt) tab[512 + i] = sign_bits(ksigns_iq2xs[i]);
}
template <> void tables<block_iq2_s>(threadgroup uint4 *tab, uint t, uint nt) {
  for (uint i = t; i < 1024; i += nt) tab[i] = half_bytes(iq2s_grid[i]);
  for (uint i = t; i < 256; i += nt) tab[1024 + i] = sign_bits(i);
}
template <> void tables<block_iq3_xxs>(threadgroup uint4 *tab, uint t, uint nt) {
  for (uint i = t; i < 256; i += nt) ((threadgroup uint2 *)tab)[i] = half_bytes(iq3xxs_grid[i]).xy;
  for (uint i = t; i < 128; i += nt) tab[128 + i] = sign_bits(ksigns_iq2xs[i]);
}
template <> void tables<block_iq1_s>(threadgroup uint4 *tab, uint t, uint nt) {
  for (uint i = t; i < 2048; i += nt) ((threadgroup uint *)tab)[i] = iq1s_grid_gpu[i];
}
template <> void tables<block_iq1_m>(threadgroup uint4 *tab, uint t, uint nt) { tables<block_iq1_s>(tab, t, nt); }

// Sub-block ib (32 values) of block b: v[0..7] receive its values, four per
// float4, so that the dot product and the threadgroup tiles share them. A
// value is a table's half times the scale, or a small code times its scale
// and offset; both products are exact in FP32, as are llama.cpp's.
inline float4 half4x(uint2 w) { return float4(as_type<half4>(w)); }
// Four codes of up to 4 bits in the bytes of g from bit `at`, as half
// subnormals: the codes times 2^-24, exactly.
inline float4 sub4(uint g, uint at, uint mask) {
  half2 a = as_type<half2>((g >> at) & mask), b = as_type<half2>((g >> (at + 8)) & mask);
  return float4(a.x, b.x, a.y, b.y);
}
inline void dq32(device const block_iq2_xxs *b, uint ib, Tab tab, thread float4 *v) {
  device const ushort *q = b->qs + 4 * ib;
  uint g = q[0] | uint(q[1]) << 16, s = q[2] | uint(q[3]) << 16;
  float db = float(b->d) * (0.5f + (s >> 28)) * 0.25f;
  for (uint l = 0; l < 4; l++) {
    uint4 w = tab[(g >> 8 * l) & 255] ^ tab[256 + ((s >> 7 * l) & 127)];
    v[2 * l] = half4x(w.xy) * db, v[2 * l + 1] = half4x(w.zw) * db;
  }
}
inline void dq32(device const block_iq2_xs *b, uint ib, Tab tab, thread float4 *v) {
  uint sc = b->scales[ib];
  float d = float(b->d) * 0.25f, db[2] = {d * (0.5f + (sc & 15)), d * (0.5f + (sc >> 4))};
  for (uint l = 0; l < 4; l++) {
    uint q = b->qs[4 * ib + l];
    uint4 w = tab[q & 511] ^ tab[512 + (q >> 9)];
    v[2 * l] = half4x(w.xy) * db[l / 2], v[2 * l + 1] = half4x(w.zw) * db[l / 2];
  }
}
inline void dq32(device const block_iq2_s *b, uint ib, Tab tab, thread float4 *v) {
  device const ushort *q = (device const ushort *)(b->qs + 4 * ib), *p = (device const ushort *)(b->qs + 32 + 4 * ib);
  uint g = q[0] | uint(q[1]) << 16, sg = p[0] | uint(p[1]) << 16, sc = b->scales[ib], qh = b->qh[ib];
  float d = float(b->d) * 0.25f, db[2] = {d * (0.5f + (sc & 15)), d * (0.5f + (sc >> 4))};
  for (uint l = 0; l < 4; l++) {
    uint4 w = tab[((g >> 8 * l) & 255) | ((qh << (8 - 2 * l)) & 0x300)] ^ tab[1024 + ((sg >> 8 * l) & 255)];
    v[2 * l] = half4x(w.xy) * db[l / 2], v[2 * l + 1] = half4x(w.zw) * db[l / 2];
  }
}
inline void dq32(device const block_iq3_xxs *b, uint ib, Tab tab, thread float4 *v) {
  device const ushort *q = (device const ushort *)(b->qs + 8 * ib), *p = (device const ushort *)(b->qs + 64 + 4 * ib);
  uint g[2] = {q[0] | uint(q[1]) << 16, q[2] | uint(q[3]) << 16}, s = p[0] | uint(p[1]) << 16;
  float db = float(b->d) * (0.5f + (s >> 28)) * 0.5f;
  threadgroup const uint2 *grid = (threadgroup const uint2 *)tab;
  for (uint l = 0; l < 4; l++) {
    uint k = g[l / 2] >> 16 * (l & 1);
    uint4 w = uint4(grid[k & 255], grid[(k >> 8) & 255]) ^ tab[128 + ((s >> 7 * l) & 127)];
    v[2 * l] = half4x(w.xy) * db, v[2 * l + 1] = half4x(w.zw) * db;
  }
}
inline void dq32(device const block_iq1_s *b, uint ib, Tab tab, thread float4 *v) {
  device const ushort *q = (device const ushort *)(b->qs + 4 * ib);
  uint qs = q[0] | uint(q[1]) << 16, qh = b->qh[ib];
  float dl = float(b->d) * (2 * ((qh >> 12) & 7) + 1), ml = dl * (qh & 0x8000 ? -1.125f : -0.875f);
  threadgroup const uint *grid = (threadgroup const uint *)tab;
  for (uint l = 0; l < 4; l++) {  // grid nibbles: values 0..3 low, 4..7 high, each value + 1
    uint g = grid[((qs >> 8 * l) & 255) | (((qh >> 3 * l) & 7) << 8)];
    v[2 * l] = sub4(g, 0, 0x000f000fu) * (dl * 0x1p24f) + ml;
    v[2 * l + 1] = sub4(g, 4, 0x000f000fu) * (dl * 0x1p24f) + ml;
  }
}
// IQ1_M has no scale field: the FP16 scale is the top nibbles of the four
// scale words, whose low 12 bits hold the 3-bit scales of each 16 values.
// Each 8 values take a nibble of qh: index bits 8 to 10, then the offset's sign.
inline void dq32(device const block_iq1_m *b, uint ib, Tab tab, thread float4 *v) {
  device const ushort *sc = (device const ushort *)b->scales, *q = (device const ushort *)(b->qs + 4 * ib);
  half d = as_type<half>(ushort((sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000)));
  uint s = sc[ib / 2] >> 6 * (ib % 2), qs = q[0] | uint(q[1]) << 16, qh = ((device const ushort *)b->qh)[ib];
  float dl[2] = {float(d) * (2 * (s & 7) + 1), float(d) * (2 * ((s >> 3) & 7) + 1)};
  threadgroup const uint *grid = (threadgroup const uint *)tab;
  for (uint l = 0; l < 4; l++) {
    uint h = qh >> 4 * l, g = grid[((qs >> 8 * l) & 255) | ((h & 7) << 8)];
    float ml = dl[l / 2] * (h & 8 ? -1.125f : -0.875f);
    v[2 * l] = sub4(g, 0, 0x000f000fu) * (dl[l / 2] * 0x1p24f) + ml;
    v[2 * l + 1] = sub4(g, 4, 0x000f000fu) * (dl[l / 2] * 0x1p24f) + ml;
  }
}
inline void dq32(device const block_q2_K *b, uint ib, Tab tab, thread float4 *v) {
  uint n = ib / 4, j = ib % 4;
  float d = float(b->d), m = float(b->dmin);
  device const uint *qs = (device const uint *)(b->qs + 32 * n);
  for (uint h = 0; h < 2; h++) {
    uint sc = b->scales[8 * n + 2 * j + h];
    float dl = d * (sc & 15), ml = m * (sc >> 4);
    for (uint i = 0; i < 4; i++) v[4 * h + i] = sub4(qs[4 * h + i], 2 * j, 0x00030003u) * (dl * 0x1p24f) - ml;
  }
}
// The prefill tiles leave a code 12 KB of tables: there, IQ2_S keeps its
// grid in bytes (block_iq2_s_b) and applies its explicit signs to them.
struct block_iq2_s_b { half d; uchar qs[64]; uchar qh[8]; uchar scales[8]; };
template <> struct Code<block_iq2_s_b> { enum { tab = 512 }; };
template <> void tables<block_iq2_s_b>(threadgroup uint4 *tab, uint t, uint nt) {
  for (uint i = t; i < 2048; i += nt) ((threadgroup uint *)tab)[i] = ((constant uint *)iq2s_grid)[i];
}
inline void dq32(device const block_iq2_s_b *b, uint ib, Tab tab, thread float4 *v) {
  uint sc = b->scales[ib], qh = b->qh[ib];
  float d = float(b->d) * 0.25f, db[2] = {d * (0.5f + (sc & 15)), d * (0.5f + (sc >> 4))};
  threadgroup const uint *grid = (threadgroup const uint *)tab;
  for (uint l = 0; l < 4; l++) {
    uint k = b->qs[4 * ib + l] | ((qh << (8 - 2 * l)) & 0x300), sg = b->qs[32 + 4 * ib + l];
    float4 lo = float4(as_type<uchar4>(grid[2 * k])), hi = float4(as_type<uchar4>(grid[2 * k + 1]));
    v[2 * l] = select(lo, -lo, bool4((uint4(sg) >> uint4(0, 1, 2, 3)) & 1u)) * db[l / 2];
    v[2 * l + 1] = select(hi, -hi, bool4((uint4(sg) >> uint4(4, 5, 6, 7)) & 1u)) * db[l / 2];
  }
}
template <typename B> struct Tiled { typedef B type; };
template <> struct Tiled<block_iq2_s> { typedef block_iq2_s_b type; };

// Decoding, as moe_up: a threadgroup takes 32 rows of an expert's w1 and w3,
// a simdgroup 4 of them, its lanes every 32nd sub-block.
template <typename B>
kernel void gg_up(constant Expert *ex [[buffer(0)]], device const float *x [[buffer(1)]], device float *h [[buffer(2)]],
                  uint2 tg [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
                  uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup uint4 tab[Code<B>::tab];
  threadgroup float block[32];
  if (!ex[tg.y].p[0]) return;  // a null entry (see resolve)
  tables<B>(tab, t, 256);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint row = (tg.x * 8 + sg) * 4, e = tg.y;
  constexpr uint NB = 5120 / 256;
  device const B *w1 = (device const B *)ex[e].p[0] + row * NB, *w3 = (device const B *)ex[e].p[1] + row * NB;
  device const float4 *xv = (device const float4 *)x;
  float4 a = 0, b = 0;
  for (uint c = lane; c < NB * 8; c += 32) {
    float4 xc[8], v[8];
    for (uint i = 0; i < 8; i++) xc[i] = xv[8 * c + i];
    for (uint r = 0; r < 4; r++) {
      float s1 = 0, s3 = 0;
      dq32(w1 + r * NB + c / 8, c % 8, tab, v);
      for (uint i = 0; i < 8; i++) s1 += dot(v[i], xc[i]);
      dq32(w3 + r * NB + c / 8, c % 8, tab, v);
      for (uint i = 0; i < 8; i++) s3 += dot(v[i], xc[i]);
      a[r] += s1, b[r] += s3;
    }
  }
  a = float4(simd_sum(a.x), simd_sum(a.y), simd_sum(a.z), simd_sum(a.w));
  b = float4(simd_sum(b.x), simd_sum(b.y), simd_sum(b.z), simd_sum(b.w));
  if (lane < 4) {
    float g = min(bf(a[lane]), 10.0f), u = clamp(bf(b[lane]), -10.0f, 10.0f);
    block[sg * 4 + lane] = bf(g / (1.0f + exp(-g)) * u * ex[e].weight);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) h[e * 2304 + tg.x * 32 + lane] = fp8_act(block[lane]);
}

// Decoding, as moe_down: 8 rows of w2 per threadgroup, simdgroup q n + e
// rows 4 q to 4 q + 3 (q = 0, 1) of expert e. Lane l sums sub-blocks l and
// l + 32 of each row, then sub-block 64 + l: those last 8 of the 4 rows are
// decoded one per lane, row r's by lane 8 r + l. With pm, each expert with an
// entry writes its part to parts instead (see moe_parts).
template <typename B>
kernel void gg_down(constant Expert *ex [[buffer(0)]], device const float *h [[buffer(1)]],
                    device const float *shared_out [[buffer(2)]], device const float *mix [[buffer(3)]],
                    device const bfloat *res [[buffer(4)]], device bfloat *out [[buffer(5)]],
                    constant uint &n [[buffer(6)]], constant uint &pm [[buffer(7)]], device float *parts [[buffer(8)]],
                    uint tg [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
                    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup uint4 tab[Code<B>::tab];
  threadgroup float4 part[2][8];
  if (!pm && !ex[0].p[0]) return;  // null entries (see resolve)
  tables<B>(tab, t, 64 * n);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint q = sg / n, e = sg % n;
  if (pm && !ex[e].p[0]) return;
  constexpr uint NB = 2304 / 256;
  device const B *w2 = (device const B *)ex[e].p[2] + (tg * 8 + q * 4) * NB;
  device const float4 *hv = (device const float4 *)(h + e * 2304);
  float4 a = 0, hc[8], v[8];
  for (uint c = lane; c < 64; c += 32) {
    for (uint i = 0; i < 8; i++) hc[i] = hv[8 * c + i];
    for (uint r = 0; r < 4; r++) {
      float s = 0;
      dq32(w2 + r * NB + c / 8, c % 8, tab, v);
      for (uint i = 0; i < 8; i++) s += dot(v[i], hc[i]);
      a[r] += s;
    }
  }
  uint c = 64 + lane % 8;
  for (uint i = 0; i < 8; i++) hc[i] = hv[8 * c + i];
  float s = 0;
  dq32(w2 + lane / 8 * NB + c / 8, c % 8, tab, v);
  for (uint i = 0; i < 8; i++) s += dot(v[i], hc[i]);
  for (uint r = 0; r < 4; r++) {
    float o = simd_shuffle(s, 8 * r + lane % 8);
    if (lane < 8) a[r] += o;
  }
  a = float4(bf(simd_sum(a.x)), bf(simd_sum(a.y)), bf(simd_sum(a.z)), bf(simd_sum(a.w)));
  if (pm) {
    if (lane == 0)
      for (uint r = 0; r < 4; r++) parts[(tg * 8 + q * 4 + r) * 8 + e] = a[r];
    return;
  }
  if (lane == 0) part[q][e] = a;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0 && lane < 8) {
    uint i = tg * 8 + lane;
    float y = 0;
    for (uint e = 0; e < n; e++) y += part[lane / 4][e][lane % 4];
    float v0 = bf(y + shared_out[i]);
    for (uint k = 0; k < 4; k++) {
      float v = mix[4 + k] * v0;
      for (uint j = 0; j < 4; j++) v += mix[8 + 4 * j + k] * res[j * 5120 + i];
      out[k * 5120 + i] = bfloat(bf(v));
    }
  }
}

// Many tokens, as moe_up_m and moe_down_m, on the matrix units (128 threads).
// A threadgroup takes up to 64 pairs of one expert (a MoeTile), so that the
// weights it decodes into threadgroup memory as half, a K step of 64 at a
// time, serve all of them; the matrix units read the pairs' rows of x or h
// as half from device memory, in pair order (gg_gather, act_quant_h). On
// this GPU the products and the decoding take turns rather than overlap, so
// the kernels save work: no staging of x, one decoding per 64 pairs, and the
// next K step's codes loaded and decoded while the products of this one are
// issued. The rows of x or h are padded by 64 for the last group's products.
// The up projection multiplies 16 pairs at a time with 64 columns of w1 and
// the same 64 of w3 (16 x 128); the down projection 32 pairs with 128
// columns of w2.

// A thread's share of a K step's 128 x 64 weights, for j = t and t + 128:
// the first or second 32 values (j & 1) of row j / 2 of w2, or of row j / 4
// of w1 (j & 2 clear) or of w3, which takes the tile's rows 64 to 127.
struct GgW { half4 v[16]; };
template <typename B>
inline GgW gg_load(device const B *w, device const B *w3, bool up, uint nb, uint n0, uint k0, uint t, Tab tab) {
  GgW o;
  for (uint s = 0; s < 2; s++) {
    uint j = t + 128 * s, c = k0 / 32 + (j & 1);
    float4 v[8];
    dq32((up && (j & 2) ? w3 : w) + (n0 + (up ? j / 4 : j / 2)) * nb + c / 8, c % 8, tab, v);
    for (uint i = 0; i < 8; i++) o.v[8 * s + i] = half4(v[i]);
  }
  return o;
}
inline void gg_store(threadgroup half *ws, GgW o, bool up, uint t) {
  for (uint s = 0; s < 2; s++) {
    uint j = t + 128 * s, row = up ? (j & 2) * 32 + j / 4 : j / 2;
    for (uint i = 0; i < 8; i++) *(threadgroup half4 *)(ws + row * 64 + 32 * (j & 1) + 4 * i) = o.v[8 * s + i];
  }
}
#define GG_MM(M, K)                                                                                                \
  threadgroup uint4 tab[Code<B>::tab];                                                                             \
  threadgroup half ws[128 * 64];                                                                                   \
  constexpr auto desc = matmul2d_descriptor(M, 128, 64, false, true, false, matmul2d_descriptor::mode::multiply_accumulate); \
  matmul2d<desc, execution_simdgroups<4>> op;                                                                      \
  tensor<device half, dextents<int32_t, 2>, tensor_inline> tx((device half *)xh, dextents<int32_t, 2>(K, npairs + 64)); \
  tensor<threadgroup half, extents<int32_t, 64, 128>, tensor_inline> tw(ws, extents<int32_t, 64, 128>());           \
  MoeTile tl = tiles[tg.x];                                                                                        \
  tables<B>(tab, t, 128);
#define GG_ACC(a) auto a = op.get_destination_cooperative_tensor<decltype(tx.slice(0, 0)), decltype(tw), float>();
#define GG_RUN(a, r)                       \
  if (r < tl.count) {                      \
    auto x = tx.slice(k0, tl.start + r);   \
    op.run(x, tw, a);                      \
  }

template <typename B>
kernel void gg_up_m(device const MoeTile *tiles [[buffer(0)]], device const half *xh [[buffer(1)]],
                    device const float *pw [[buffer(2)]], device float *h [[buffer(3)]],
                    constant uint &npairs [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
                    uint t [[thread_index_in_threadgroup]]) {
  constexpr uint K = 5120;
  GG_MM(16, K)
  GG_ACC(a0) GG_ACC(a1) GG_ACC(a2) GG_ACC(a3)
  for (uint16_t i = 0; i < a0.get_capacity(); i++)
    if (a0.is_valid_element(i)) a0[i] = 0, a1[i] = 0, a2[i] = 0, a3[i] = 0;
  uint n0 = tg.y * 64;
  device const B *w1 = (device const B *)tl.p[0], *w3 = (device const B *)tl.p[1];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  GgW wr = gg_load(w1, w3, true, K / 256, n0, 0, t, tab);
  for (uint k0 = 0; k0 < K; k0 += 64) {
    gg_store(ws, wr, true, t);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    GG_RUN(a0, 0) GG_RUN(a1, 16) GG_RUN(a2, 32) GG_RUN(a3, 48)
    if (k0 + 64 < K) wr = gg_load(w1, w3, true, K / 256, n0, k0 + 64, t, tab);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  // w1's and w3's column c of 16 pairs meet through threadgroup memory
  threadgroup float *acc = (threadgroup float *)ws;
  tensor<threadgroup float, extents<int32_t, 128, 16>, tensor_inline> ta(acc, extents<int32_t, 128, 16>());
  for (uint s = 0; s < 4 && 16 * s < tl.count; s++) {
    if (s == 0) a0.store(ta);
    else if (s == 1) a1.store(ta);
    else if (s == 2) a2.store(ta);
    else a3.store(ta);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = t; i < 16 * 64; i += 128) {
      uint r = 16 * s + i / 64, col = i % 64;
      if (r < tl.count) {
        uint q = tl.start + r;
        float g = min(bf(acc[i / 64 * 128 + col]), 10.0f), u = clamp(bf(acc[i / 64 * 128 + 64 + col]), -10.0f, 10.0f);
        h[q * 2304 + n0 + col] = bf(g / (1.0f + exp(-g)) * u * pw[q]);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

template <typename B>
kernel void gg_down_m(device const MoeTile *tiles [[buffer(0)]], device const half *xh [[buffer(1)]],
                      device float *y [[buffer(2)]], constant uint &npairs [[buffer(3)]],
                      uint2 tg [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]]) {
  constexpr uint K = 2304;
  GG_MM(32, K)
  GG_ACC(a0) GG_ACC(a1)
  for (uint16_t i = 0; i < a0.get_capacity(); i++)
    if (a0.is_valid_element(i)) a0[i] = 0, a1[i] = 0;
  uint n0 = tg.y * 128;
  device const B *w2 = (device const B *)tl.p[2];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  GgW wr = gg_load(w2, w2, false, K / 256, n0, 0, t, tab);
  for (uint k0 = 0; k0 < K; k0 += 64) {
    gg_store(ws, wr, false, t);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    GG_RUN(a0, 0) GG_RUN(a1, 32)
    if (k0 + 64 < K) wr = gg_load(w2, w2, false, K / 256, n0, k0 + 64, t, tab);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (uint16_t i = 0; i < a0.get_capacity(); i++)
    if (a0.is_valid_element(i)) {
      auto at = a0.get_multidimensional_index(i);
      for (uint s = 0; s < 2; s++) {
        uint r = 32 * s + at[1];
        if (r < tl.count) y[(tl.start + r) * 5120 + n0 + at[0]] = bf(s ? a1[i] : a0[i]);
      }
    }
}

// The pairs' rows of x (5120) as half, pair q's from token row ptok[q].
kernel void gg_gather(device const float4 *x [[buffer(0)]], device const int *ptok [[buffer(1)]],
                      device half4 *xh [[buffer(2)]], uint2 i [[thread_position_in_grid]]) {
  xh[i.y * 1280 + i.x] = half4(x[ptok[i.y] * 1280 + i.x]);
}
// act_quant into half.
kernel void act_quant_h(device const float *x [[buffer(0)]], device half *y [[buffer(1)]],
                        uint i [[thread_position_in_grid]]) {
  y[i] = half(fp8_act(x[i]));
}

typedef decltype(gg_up<block_q2_K>) gg_up_t;
typedef decltype(gg_down<block_q2_K>) gg_down_t;
typedef decltype(gg_up_m<block_q2_K>) gg_up_m_t;
typedef decltype(gg_down_m<block_q2_K>) gg_down_m_t;
// The output head in Q6_K (--experts, when the GGUF file has one): the
// logits of m rows of x (up to MAXM: a verify pass's or the draft's), each
// summed as for one row. A simdgroup takes 4 rows of the head.
struct block_q6_K { uchar ql[128]; uchar qh[64]; char scales[16]; half d; };
inline void dq32(device const block_q6_K *b, uint ib, Tab tab, thread float4 *v) {
  uint n = ib / 4, j = ib % 4, shl = 4 * (j / 2), shh = 2 * j;
  device const uchar *ql = b->ql + 64 * n + 32 * (j & 1), *qh = b->qh + 32 * n;
  float d = float(b->d);
  for (uint i = 0; i < 8; i++) {
    uint4 lo = uint4(*(device const packed_uchar4 *)(ql + 4 * i)), hi = uint4(*(device const packed_uchar4 *)(qh + 4 * i));
    uint4 q = ((lo >> shl) & 15u) | (((hi >> shh) & 3u) << 4);
    v[i] = (float4(q) - 32.0f) * (d * float(b->scales[8 * n + i / 4 + 2 * j]));
  }
}
template <uint MAXM>
kernel void gg_head(device const block_q6_K *w [[buffer(0)]], device const float *x [[buffer(1)]], device float *y [[buffer(2)]],
                    constant uint &m [[buffer(3)]], uint tg [[threadgroup_position_in_grid]],
                    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  constexpr uint NB = 5120 / 256, V = 129280;
  uint row = (tg * 8 + sg) * 4;
  float acc[MAXM][4] = {};
  for (uint c = lane; c < NB * 8; c += 32)
    for (uint r = 0; r < 4; r++) {
      float4 v[8];
      dq32(w + (row + r) * NB + c / 8, c % 8, nullptr, v);
      for (uint j = 0; j < MAXM; j++)
        if (j < m) {
          device const float4 *xv = (device const float4 *)(x + j * 5120) + 8 * c;
          float s = 0;
          for (uint i = 0; i < 8; i++) s += dot(v[i], xv[i]);
          acc[j][r] += s;
        }
    }
  for (uint j = 0; j < MAXM; j++)
    for (uint r = 0; r < 4; r++) {
      float t = simd_sum(acc[j][r]);
      if (j < m && lane == 0) y[j * V + row + r] = t;
    }
}
typedef decltype(gg_head<1>) gg_head_t;
template [[host_name("gg_head1")]] kernel gg_head_t gg_head<1>;
template [[host_name("gg_head6")]] kernel gg_head_t gg_head<6>;

#define GG_KERNELS(B, name)                                                                \
  template [[host_name("gg_up_" name)]] kernel gg_up_t gg_up<B>;                           \
  template [[host_name("gg_down_" name)]] kernel gg_down_t gg_down<B>;                     \
  template [[host_name("gg_up_m_" name)]] kernel gg_up_m_t gg_up_m<Tiled<B>::type>;         \
  template [[host_name("gg_down_m_" name)]] kernel gg_down_m_t gg_down_m<Tiled<B>::type>;
GG_KERNELS(block_q2_K, "q2_k")
GG_KERNELS(block_iq2_xxs, "iq2_xxs")
GG_KERNELS(block_iq2_xs, "iq2_xs")
GG_KERNELS(block_iq2_s, "iq2_s")
GG_KERNELS(block_iq3_xxs, "iq3_xxs")
GG_KERNELS(block_iq1_s, "iq1_s")
GG_KERNELS(block_iq1_m, "iq1_m")
