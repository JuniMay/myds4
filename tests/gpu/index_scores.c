// The index scores with candidate blocks against the scores of every position:
// a candidate's score must be the same bits, and no other position written.
#include "../../src/engine/myds4.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#define METAL(...) #__VA_ARGS__
// Every position's score, a thread per position, without candidate blocks.
static const char reference[] = METAL(
kernel void index_score_reference(device const float *q [[buffer(0)]], device const float *wts [[buffer(1)]],
                                  device const bfloat *keys [[buffer(2)]], device const float2 *cs [[buffer(3)]],
                                  device float *score [[buffer(4)]], constant Index &ix [[buffer(5)]],
                                  uint2 g [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]]) {
  threadgroup float4 qs[32 * 32];
  uint row = g.y, gid = g.x * 256 + t;
  uint2 p = uint2((ix.pos + row + 1) / ix.ratio, ix.pos + row);
  q += row * 4096, wts += row * 32, score += row * ix.stride;
  threadgroup float ws[32];
  threadgroup float *qf = (threadgroup float *)qs;
  for (uint i = t; i < 4096; i += 256) qf[i] = fp4_act(rope1(q[i], i % 128, 128, cs + p.y * 32, false));
  if (t < 32) ws[t] = bf(wts[t] * 0.015625f);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (gid >= p.x) return;
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
});
#undef METAL

typedef struct { uint32_t pos, ratio, stride, cand; } Index;
enum { ROWS = 256, POSITIONS = 65536, WORDS = POSITIONS / 256 + 1 };
static const uint32_t untouched = 0x7fc0beefu;  // a NaN no score takes
static Buf Q, W, KEYS, CS, CAND, SCORE[2];
static uint32_t seed = 12345;
static uint32_t next(void) { return seed = seed * 1664525u + 1013904223u; }

// Rows `rows` of queries from position pos; returns the GPU's microseconds.
static double run(int kernel, Index ix, int rows, const Buf *score, int repeat) {
  uint32_t seen = (ix.pos + rows) / ix.ratio;
  double before = gpu_busy;
  gpu_begin();
  gpu_kern(kernel);
  gpu_buf(0, &Q, 0), gpu_buf(1, &W, 0), gpu_buf(2, &KEYS, 0), gpu_buf(3, &CS, 0), gpu_buf(4, score, 0);
  gpu_bytes(5, &ix, sizeof ix), gpu_buf(6, &CAND, 0);
  for (int i = 0; i < repeat; i++) gpu_go((seen + 255) / 256, rows, 1, 256, 1, 1);
  gpu_commit();
  gpu_sync();
  return (gpu_busy - before) * 1e6 / repeat;
}

static void clear(const Buf *score) {
  uint32_t *s = (uint32_t *)score->ptr;
  for (size_t i = 0; i < (size_t)ROWS * POSITIONS; i++) s[i] = untouched;
}

// A row's candidate bits of the given kind over its nb blocks; none beyond them.
static void candidates(uint32_t *bits, int nb, int kind) {
  memset(bits, 0, WORDS * 4);
  for (int b = 0; b < nb; b++) {
    int word = b / 32, in;
    if (kind == 0) in = 1;                        // every block, as layer 20 marks a row of few blocks
    else if (kind == 1) in = next() >> 31;        // half of them
    else if (kind == 2) in = next() % 8 == 0;     // an eighth
    else if (kind == 3) in = 0;                   // the newest block alone
    else if (kind == 4) in = word % 3 == 0;       // whole words
    else in = b % 32 < word % 33;                 // 0 to 32 blocks a word, in turn
    if (in || b == nb - 1) bits[word] |= 1u << (b % 32);
  }
}

int main(void) {
  memory_configure(1, 10);
  size_t source_bytes;
  const char *source = gpu_source(&source_bytes);
  char *code = xmalloc(source_bytes + sizeof reference + 1);
  memcpy(code, source, source_bytes), code[source_bytes] = '\n';
  memcpy(code + source_bytes + 1, reference, sizeof reference);
  gpu_init(code, source_bytes + sizeof reference), free(code);
  int every = gpu_kernel("index_score_reference"), scored = gpu_kernel("index_score");
  Q = gpu_alloc((size_t)ROWS * 4096 * 4), W = gpu_alloc(ROWS * 32 * 4);
  KEYS = gpu_alloc((size_t)POSITIONS * 128 * 2), CS = gpu_alloc((size_t)(POSITIONS + ROWS) * 32 * 8);
  CAND = gpu_alloc((size_t)ROWS * WORDS * 4);
  for (int i = 0; i < 2; i++) SCORE[i] = gpu_alloc((size_t)ROWS * POSITIONS * 4);
  for (size_t i = 0; i < (size_t)ROWS * 4096; i++) ((float *)Q.ptr)[i] = (float)(next() >> 8) / (1 << 23) - 1;
  for (int i = 0; i < ROWS * 32; i++) ((float *)W.ptr)[i] = (float)(next() >> 8) / (1 << 24) - 0.25f;
  for (size_t i = 0; i < (size_t)POSITIONS * 128; i++) {  // keys of few bits, exact in BF16
    float v = (float)((int)(next() >> 26) - 32) / 8;
    uint32_t u;
    memcpy(&u, &v, 4);
    ((uint16_t *)KEYS.ptr)[i] = (uint16_t)(u >> 16);
  }
  for (size_t p = 0; p < POSITIONS + ROWS; p++)
    for (int i = 0; i < 32; i++) {
      float a = (float)p * powf(10000, -(float)i / 32);
      ((float *)CS.ptr)[(p * 32 + i) * 2] = cosf(a), ((float *)CS.ptr)[(p * 32 + i) * 2 + 1] = sinf(a);
    }
  const uint32_t *a = (const uint32_t *)SCORE[0].ptr, *b = (const uint32_t *)SCORE[1].ptr;
  int checks = 0;
  // Without candidate blocks: every position, as before.
  int starts[][2] = {{0, 1}, {300, 1}, {5000, 2}, {20000, 1}, {40001, 2}};
  for (size_t c = 0; c < sizeof starts / sizeof *starts; c++) {
    Index ix = {(uint32_t)starts[c][0], (uint32_t)starts[c][1], POSITIONS, 0};
    int rows = 48;
    clear(&SCORE[0]), clear(&SCORE[1]);
    run(every, ix, rows, &SCORE[0], 1), run(scored, ix, rows, &SCORE[1], 1);
    if (memcmp(a, b, (size_t)ROWS * POSITIONS * 4)) die("index scores differ without candidates: position %u, ratio %u", ix.pos, ix.ratio);
    uint64_t written = 0;
    for (int r = 0; r < rows; r++)
      for (uint32_t i = 0; i < (ix.pos + r + 1) / ix.ratio; i++) written += b[(size_t)r * POSITIONS + i] != untouched;
    if (!written) die("no index score was written");
    checks++;
  }
  // With candidate blocks: the candidates' scores, bit for bit, and nothing else.
  for (int kind = 0; kind < 6; kind++) {
    Index ix = {20000 - 5 * kind, 1, POSITIONS, WORDS};
    int rows = 48;
    uint64_t kept = 0, skipped = 0;
    for (int r = 0; r < rows; r++) candidates((uint32_t *)CAND.ptr + (size_t)r * WORDS, (int)(ix.pos + r + 1 + 7) / 8, kind);
    clear(&SCORE[0]), clear(&SCORE[1]);
    Index all = ix;
    all.cand = 0;
    run(every, all, rows, &SCORE[0], 1), run(scored, ix, rows, &SCORE[1], 1);
    for (int r = 0; r < rows; r++) {
      const uint32_t *bits = (const uint32_t *)CAND.ptr + (size_t)r * WORDS;
      uint32_t n = ix.pos + r + 1;
      for (uint32_t i = 0; i < POSITIONS; i++) {
        int in = i < n && (bits[i >> 8] >> ((i >> 3) & 31) & 1);
        uint32_t want = in ? a[(size_t)r * POSITIONS + i] : untouched, got = b[(size_t)r * POSITIONS + i];
        if (in && want == untouched) die("the reference left a candidate unscored");
        if (got != want) die("index scores differ with candidates of kind %d: row %d, position %u, %s", kind, r, i,
                             in ? "a candidate's score" : "a score outside the candidates was written");
        kept += in, skipped += !in && i < n;
      }
    }
    printf("index scores, candidates of kind %d: %llu equal, %llu positions left out\n", kind, (unsigned long long)kept,
           (unsigned long long)skipped);
    checks++;
  }
  printf("index scores: %d cases match bit for bit\n", checks);
  // Time: every position, then 2,048 candidate blocks of 8,190, spread and in runs.
  Index ix = {POSITIONS - ROWS, 1, POSITIONS, 0};
  for (int spread = 0; spread < 3; spread++) {
    for (int r = 0; r < ROWS && spread; r++) {
      uint32_t *bits = (uint32_t *)CAND.ptr + (size_t)r * WORDS;
      int nb = (int)(ix.pos + r + 1 + 7) / 8;
      memset(bits, 0, WORDS * 4);
      for (int b = 0; b < nb; b++)
        if (spread == 1 ? b % 4 == r % 4 : (b / 64) % 4 == r % 4) bits[b / 32] |= 1u << (b % 32);
    }
    ix.cand = spread ? WORDS : 0;
    run(scored, ix, ROWS, &SCORE[1], 1);  // raise the GPU's clock
    double before = run(every, (Index){ix.pos, 1, POSITIONS, 0}, ROWS, &SCORE[0], 4);
    double middle = run(scored, ix, ROWS, &SCORE[1], 4);
    double after = run(every, (Index){ix.pos, 1, POSITIONS, 0}, ROWS, &SCORE[0], 4);
    printf("index scores of %d rows over %d positions, %s: every position %.0f / %.0f us, scored %.0f us\n", ROWS,
           POSITIONS, spread == 0 ? "no candidates" : spread == 1 ? "a quarter of the blocks, spread" : "a quarter of the blocks, in runs of 64",
           before, after, middle);
  }
  return 0;
}
