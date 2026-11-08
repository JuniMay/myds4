// IQ1_M expert decoding (gguf.metal's dq32) against llama.cpp's
// dequantize_row_iq1_m (ggml-quants.c, MIT License, ported below), value by
// value: on random blocks, and, given a GGUF file (any shard of a split set),
// on blocks from the start, middle and end of each of its IQ1_M expert
// tensors. Run with `make gguf-decode-check [GGUF=FILE]`. Values compare with
// ==, since a zero scale gives llama.cpp -0 where the GPU gives +0. The grid
// (gguf_grids.metal's iq1s_grid_gpu, shared with IQ1_S) matched llama.cpp's
// iq1s_grid entry for entry when IQ1_M was added (2026-11-01).
#include "../../src/engine/myds4.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-const-variable"
#define constant static const
#define uchar uint8_t
#define uint uint32_t
#define ulong uint64_t
#include "../../src/kernels/gguf_grids.metal"
#undef constant
#undef uchar
#undef uint
#undef ulong
#pragma clang diagnostic pop

enum { QB = 56, PER = 256 };  // bytes per block of 256 values; blocks per real sample (a multiple of 32)

// Value j (-1, 0 or 1) of grid entry i: the GPU table keeps each value + 1 as
// a nibble, values 0 to 3 in the low nibbles of its four bytes, 4 to 7 high.
static int grid8(int i, int j) { return (int)((iq1s_grid_gpu[i] >> (8 * (j & 3) + 4 * (j >> 2))) & 15) - 1; }

static void reference(const uint8_t *x, float *y, int nb) {  // dequantize_row_iq1_m
  for (int i = 0; i < nb; i++, x += QB) {
    const uint8_t *qs = x, *qh = x + 32;
    uint16_t sc[4];
    memcpy(sc, x + 48, sizeof sc);
    uint16_t u = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
    _Float16 h;
    memcpy(&h, &u, 2);
    const float d = (float)h;
    for (int ib = 0; ib < 8; ++ib) {
      const float dl1 = d * (2 * ((sc[ib / 2] >> (6 * (ib % 2) + 0)) & 0x7) + 1);
      const float dl2 = d * (2 * ((sc[ib / 2] >> (6 * (ib % 2) + 3)) & 0x7) + 1);
      int idx[4] = {qs[0] | ((qh[0] << 8) & 0x700), qs[1] | ((qh[0] << 4) & 0x700), qs[2] | ((qh[1] << 8) & 0x700),
                    qs[3] | ((qh[1] << 4) & 0x700)};
      float delta[4] = {qh[0] & 0x08 ? -0.125f : 0.125f, qh[0] & 0x80 ? -0.125f : 0.125f, qh[1] & 0x08 ? -0.125f : 0.125f,
                        qh[1] & 0x80 ? -0.125f : 0.125f};
      for (int l = 0; l < 4; ++l)
        for (int j = 0; j < 8; ++j) *y++ = (l < 2 ? dl1 : dl2) * (grid8(idx[l], j) + delta[l]);
      qs += 4, qh += 2;
    }
  }
}

static int kernel;
static long values, failures;

static void check(const char *what, const uint8_t *blocks, int nb) {
  Buf in = gpu_alloc((size_t)nb * QB), out = gpu_alloc((size_t)nb * 256 * 4);
  memcpy(in.ptr, blocks, (size_t)nb * QB);
  gpu_begin();
  gpu_kern(kernel);
  gpu_buf(0, &in, 0), gpu_buf(1, &out, 0);
  gpu_go(nb * 8 / 256, 1, 1, 256, 1, 1);
  gpu_commit();
  gpu_sync();
  float *want = xmalloc((size_t)nb * 256 * 4);
  const float *got = (const float *)out.ptr;
  reference(blocks, want, nb);
  long bad = 0;
  for (long i = 0; i < (long)nb * 256; i++)
    if (!(got[i] == want[i]) && bad++ < 3)
      fprintf(stderr, "%s: block %ld value %ld: GPU %a, llama.cpp %a\n", what, i / 256, i % 256, got[i], want[i]);
  values += (long)nb * 256, failures += bad;
  free(want), gpu_free(&in), gpu_free(&out);
}

int main(int argc, char **argv) {
  static const char source[] = {
#embed "../../src/kernels/formats.metal"
      , '\n',
#embed "../../src/kernels/kernels.metal"
      , '\n',
#embed "../../src/kernels/gguf_grids.metal"
      , '\n',
#embed "../../src/kernels/gguf.metal"
      , '\n'};
  Str s = {0};
  str_put(&s, source, sizeof source);
  str_cat(&s, "kernel void iq1m_decode(device const block_iq1_m *b [[buffer(0)]], device float4 *y [[buffer(1)]],\n"
              "                        uint i [[thread_position_in_grid]], uint t [[thread_index_in_threadgroup]]) {\n"
              "  threadgroup uint4 tab[Code<block_iq1_m>::tab];\n"
              "  tables<block_iq1_m>(tab, t, 256);\n"
              "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
              "  float4 v[8];\n"
              "  dq32(b + i / 8, i % 8, tab, v);\n"
              "  for (uint k = 0; k < 8; k++) y[8 * i + k] = v[k];\n"
              "}\n");
  gpu_init(s.p, s.n);
  free(s.p);
  kernel = gpu_kernel("iq1m_decode");

  // Random blocks; a scale whose FP16 exponent is all ones (inf or NaN) loses its top exponent bit.
  enum { NR = 8192 };
  uint8_t *r = xmalloc((size_t)NR * QB);
  srandom(1);
  for (size_t i = 0; i < (size_t)NR * QB; i++) r[i] = (uint8_t)random();
  for (int i = 0; i < NR; i++) {
    uint8_t *sc = r + (size_t)i * QB + 48;
    if ((sc[5] >> 6) == 3 && (sc[7] & 0x70) == 0x70) sc[7] &= 0xbf;
  }
  check("random", r, NR);
  free(r);
  long random_values = values;

  int tensors = 0;
  if (argc > 1) {
    gguf_open(argv[1]);
    static const char *mat[3] = {"gate", "up", "down"};
    uint8_t *b = xmalloc((size_t)PER * QB);
    for (int l = 0; l < 64; l++)
      for (int k = 0; k < 3; k++) {
        char name[64];
        snprintf(name, sizeof name, "blk.%d.ffn_%s_exps.weight", l, mat[k]);
        Gguf t;
        if (!gguf_get(name, &t) || t.type != GG_IQ1_M) continue;
        uint64_t nb = t.ne[0] / 256 * t.ne[1] * t.ne[2];
        for (int p = 0; p < 3; p++) {
          uint64_t first = p == 0 ? 0 : p == 1 ? nb / 2 : nb - PER;
          if (pread(t.fd, b, (size_t)PER * QB, (off_t)(t.off + first * QB)) != PER * QB) die("%s: short read", name);
          check(name, b, PER);
        }
        tensors++;
      }
    free(b);
    if (!tensors) die("%s: no IQ1_M expert tensor", argv[1]);
  }
  fprintf(stderr, "IQ1_M decoding: %ld random values", random_values);
  if (tensors) fprintf(stderr, ", %ld values from %d tensors of %s", values - random_values, tensors, argv[1]);
  fprintf(stderr, "; %s (%ld differ)\n", failures ? "FAILED" : "all equal llama.cpp's", failures);
  return failures != 0;
}
