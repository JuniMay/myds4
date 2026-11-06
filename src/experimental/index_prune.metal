// Skip dot products that the existing candidate mask excludes from top-k.
kernel void index_score_pruned(device const float *q [[buffer(0)]], device const float *wts [[buffer(1)]],
                        device const bfloat *keys [[buffer(2)]], device const float2 *cs [[buffer(3)]],
                        device float *score [[buffer(4)]], constant Index &ix [[buffer(5)]], device const uint *cand [[buffer(6)]],
                        uint2 g [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]]) {
  threadgroup float4 qs[32 * 32];
  uint row = g.y, gid = g.x * 256 + t;
  uint2 p = uint2((ix.pos + row + 1) / ix.ratio, ix.pos + row);
  q += row * 4096, wts += row * 32, score += row * ix.stride;
  // One 256-position threadgroup corresponds to one 32-block mask word.
  // A fully excluded group has no shared query work to perform.
  uint mask = ix.cand ? cand[row * ix.cand + g.x] : ~0u;
  if (!mask) {
    if (gid < p.x) score[gid] = -INFINITY;
    return;
  }
  threadgroup float ws[32];
  threadgroup float *qf = (threadgroup float *)qs;
  for (uint i = t; i < 4096; i += 256)  // each run of 32 is a simdgroup: one quantization block
    qf[i] = fp4_act(rope1(q[i], i % 128, 128, cs + p.y * 32, false));
  if (t < 32) ws[t] = bf(wts[t] * 0.015625f);  // softmax scale 128^-0.5 times 32^-0.5
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (gid >= p.x) return;
  // The later top-k operation assigns these positions -infinity already.
  // All threads completed query quantization before this early return.
  if (!(mask & (1u << (t / 8)))) {
    score[gid] = -INFINITY;
    return;
  }
  device const bfloat4 *kr = (device const bfloat4 *)(keys + gid * 128);
  float acc[32];
  for (uint h = 0; h < 32; h++) acc[h] = 0;
  for (uint d = 0; d < 32; d++) {
    float4 k = float4(kr[d]);
    for (uint h = 0; h < 32; h++) acc[h] += dot(qs[h * 32 + d], k);
  }
  float total = 0;
  for (uint h = 0; h < 32; h++) total += max(acc[h], 0.0f) * ws[h];
  score[gid] = total;
}

// Every excluded position contributes the same -infinity order key. Add
// that bucket's entire count once, preserving even exceptional-value ties.
inline uint kth_score_key_masked(ScoreKeys keys, uint n, uint k, uint t, threadgroup atomic_uint *hist, threadgroup uint &prefix,
                    threadgroup uint &want, uint omitted) {
  uint thr = 0, mask = 0;
  if (t == 0) want = k;
  for (int shift = 24; shift >= 0; shift -= 8) {
    if (t < 256) atomic_store_explicit(&hist[t], 0, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = t; i < n; i += 1024) {
      if (!((keys.cand[i >> 8] >> ((i >> 3) & 31)) & 1)) continue;
      uint key = order_key(keys.score[i]);
      if ((key & mask) == thr) atomic_fetch_add_explicit(&hist[(key >> shift) & 255], 1, memory_order_relaxed);
    }
    uint absent = order_key(-INFINITY);
    if (!t && (absent & mask) == thr)
      atomic_fetch_add_explicit(&hist[(absent >> shift) & 255], omitted, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0) {
      uint acc = 0, b = 255;
      for (;; b--) {
        uint c = atomic_load_explicit(&hist[b], memory_order_relaxed);
        if (acc + c >= want || b == 0) break;
        acc += c;
      }
      want -= acc;
      prefix = b;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    thr |= prefix << shift;
    mask |= 255u << shift;
  }
  return thr;
}

kernel void index_topk_masked(device const float *score [[buffer(0)]], device int *sel [[buffer(1)]],
                       constant Index &ix [[buffer(2)]], device const uint *cand [[buffer(3)]],
                       uint row [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]]) {
  threadgroup atomic_uint hist[256];
  threadgroup uint prefix, want, pa[1024], pe[1024];
  ScoreKeys keys = {score + row * ix.stride, ix.cand ? cand + row * ix.cand : nullptr};
  sel += row * 513 + 1;
  uint n = (ix.pos + row + 1) / ix.ratio, k = min(n, 512u);
  bool filtered = false;
  uint omitted = 0;
  if (keys.cand && n > 512) {
    uint allowed = 0;
    for (uint word = t; word < (n + 255) / 256; word += 1024) {
      uint valid = min(256u, n - word * 256), blocks = (valid + 7) / 8;
      uint bits = keys.cand[word] & (~0u >> (32 - blocks));
      allowed += popcount(bits) * 8;
      if ((valid & 7) && (bits & (1u << (blocks - 1)))) allowed -= 8 - (valid & 7);
    }
    uint group_sum = simd_sum(allowed);
    if (!(t & 31)) pa[t >> 5] = group_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t < 32) {
      uint total = simd_sum(pa[t]);
      if (!t) prefix = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    omitted = n - prefix;
    filtered = omitted != 0;
  }
  uint thr = n > 512 ? (filtered ? kth_score_key_masked(keys, n, 512, t, hist, prefix, want, omitted)
                                : kth_key(keys, n, 512, t, hist, prefix, want)) : 0;
  // Keys above thr are all taken; of the keys equal to thr, the first `want`.
  uint per = (n + 1023) / 1024, begin = t * per, end = min(begin + per, n), above = 0, equal = 0;
  for (uint i = begin; i < end; i++) {
    uint key = keys(i);
    above += key > thr || n <= 512;
    equal += key == thr && n > 512;
  }
  // exclusive prefix sums over threads, in thread (= position) order
  pa[t] = above;
  pe[t] = equal;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint o = 1; o < 1024; o <<= 1) {
    uint a = t >= o ? pa[t - o] : 0, e = t >= o ? pe[t - o] : 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    pa[t] += a;
    pe[t] += e;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  uint total_above = pa[1023], ties = n > 512 ? k - total_above : 0;
  uint ra = pa[t] - above, re = pe[t] - equal;
  // output order by position: interleave above/equal keys as they occur
  uint out = ra + min(re, ties);
  for (uint i = begin; i < end; i++) {
    uint key = keys(i);
    bool take = n <= 512 || key > thr || (key == thr && re++ < ties);
    if (take) sel[out++] = int(i);
  }
  if (t == 0) sel[-1] = int(k);
}
