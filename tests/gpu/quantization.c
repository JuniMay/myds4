// GGUF expert decoding against llama.cpp's scalar dequantize_row functions
// (ggml/src/ggml-quants.c, MIT License; see gguf_grids.metal). Check random
// blocks and the start, middle and end of each tensor in a supplied split
// checkpoint. A checkpoint's name need not name its individual tensor types.
// Run with `make gguf-decode-check [GGUF=FILE]`. Compare values with == so
// that the reference's -0 and the GPU's +0 are equal.
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

enum { PER = 256, TYPE_Q6_K = 14 };  // blocks per real sample; GGUF output-head type
typedef struct {
  int type;
  const char *name;
  int bytes, kernel;
  char entry[32];
  long values, failures;
  int tensors;
} Format;
static Format formats[] = {
    {.type = GG_Q2_K, .name = "q2_K", .bytes = 84},
    {.type = GG_IQ2_XXS, .name = "iq2_xxs", .bytes = 66},
    {.type = GG_IQ2_XS, .name = "iq2_xs", .bytes = 74},
    {.type = GG_IQ2_S, .name = "iq2_s", .bytes = 82},
    {.type = GG_IQ2_S, .name = "iq2_s_b", .bytes = 82},
    {.type = GG_IQ3_XXS, .name = "iq3_xxs", .bytes = 98},
    {.type = GG_IQ1_S, .name = "iq1_s", .bytes = 50},
    {.type = GG_IQ1_M, .name = "iq1_m", .bytes = 56},
    {.type = TYPE_Q6_K, .name = "q6_K", .bytes = 210},
};

// Value j (-1, 0 or 1) of grid entry i: the GPU table keeps each value + 1 as
// a nibble, values 0 to 3 in the low nibbles of its four bytes, 4 to 7 high.
static int grid8(int i, int j) { return (int)((iq1s_grid_gpu[i] >> (8 * (j & 3) + 4 * (j >> 2))) & 15) - 1; }

static void reference_iq1_m(const uint8_t *x, float *y, int nb) {
  for (int i = 0; i < nb; i++, x += 56) {
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

static uint16_t u16(const uint8_t *p) { uint16_t v; memcpy(&v, p, 2); return v; }
static uint32_t u32(const uint8_t *p) { uint32_t v; memcpy(&v, p, 4); return v; }
static float half_value(const uint8_t *p) { _Float16 v; memcpy(&v, p, 2); return v; }

static void reference(const Format *format, const uint8_t *x, float *y, int nb) {
  if (format->type == GG_IQ1_M) { reference_iq1_m(x, y, nb); return; }
  for (int block = 0; block < nb; block++, x += format->bytes) {
    if (format->type == TYPE_Q6_K) {
      float d = half_value(x + 208);
      const int8_t *scales = (const int8_t *)x + 192;
      for (int n = 0; n < 2; n++) {
        const uint8_t *low = x + 64 * n, *high = x + 128 + 32 * n;
        for (int l = 0; l < 32; l++) {
          int q1 = ((low[l] & 15) | (((high[l] >> 0) & 3) << 4)) - 32;
          int q2 = ((low[l + 32] & 15) | (((high[l] >> 2) & 3) << 4)) - 32;
          int q3 = ((low[l] >> 4) | (((high[l] >> 4) & 3) << 4)) - 32;
          int q4 = ((low[l + 32] >> 4) | (((high[l] >> 6) & 3) << 4)) - 32;
          int at = 8 * n + l / 16;
          y[128 * n + l] = d * scales[at] * q1;
          y[128 * n + l + 32] = d * scales[at + 2] * q2;
          y[128 * n + l + 64] = d * scales[at + 4] * q3;
          y[128 * n + l + 96] = d * scales[at + 6] * q4;
        }
      }
      y += 256;
      continue;
    }
    float d = half_value(x);
    for (int ib = 0; ib < 8; ib++) {
      if (format->type == GG_Q2_K) {
        for (int h = 0; h < 2; h++) {
          int scale = x[2 * ib + h];
          float dl = half_value(x + 80) * (scale & 15), ml = half_value(x + 82) * (scale >> 4);
          for (int j = 0; j < 16; j++)
            *y++ = dl * ((x[16 + 32 * (ib / 4) + 16 * h + j] >> (2 * (ib % 4))) & 3) - ml;
        }
        continue;
      }
      for (int l = 0; l < 4; l++) {
        if (format->type == GG_IQ1_S) {
          uint16_t high = u16(x + 34 + 2 * ib);
          float dl = d * (2 * ((high >> 12) & 7) + 1);
          float delta = high & 0x8000 ? -.125f : .125f;
          int index = x[2 + 4 * ib + l] | (((high >> (3 * l)) & 7) << 8);
          for (int j = 0; j < 8; j++) *y++ = dl * (grid8(index, j) + delta);
          continue;
        }
        float scale;
        uint8_t signs;
        uint64_t grid;
        if (format->type == GG_IQ2_XXS) {
          uint32_t high = u32(x + 2 + 8 * ib + 4);
          scale = d * (.5f + (high >> 28)) * .25f;
          signs = ksigns_iq2xs[(high >> (7 * l)) & 127];
          grid = iq2xxs_grid[x[2 + 8 * ib + l]];
        } else if (format->type == GG_IQ2_XS) {
          uint16_t q = u16(x + 2 + 2 * (4 * ib + l));
          scale = d * (.5f + ((x[66 + ib] >> (4 * (l / 2))) & 15)) * .25f;
          signs = ksigns_iq2xs[q >> 9];
          grid = iq2xs_grid[q & 511];
        } else if (format->type == GG_IQ2_S) {
          scale = d * (.5f + ((x[74 + ib] >> (4 * (l / 2))) & 15)) * .25f;
          signs = x[34 + 4 * ib + l];
          grid = iq2s_grid[x[2 + 4 * ib + l] | ((x[66 + ib] << (8 - 2 * l)) & 0x300)];
        } else {
          uint32_t high = u32(x + 66 + 4 * ib);
          scale = d * (.5f + (high >> 28)) * .5f;
          signs = ksigns_iq2xs[(high >> (7 * l)) & 127];
          grid = (uint64_t)iq3xxs_grid[x[2 + 8 * ib + 2 * l]] |
                 (uint64_t)iq3xxs_grid[x[3 + 8 * ib + 2 * l]] << 32;
        }
        for (int j = 0; j < 8; j++)
          *y++ = scale * ((grid >> (8 * j)) & 255) * (signs & (1 << j) ? -1.f : 1.f);
      }
    }
  }
}

static void check(Format *format, const char *what, const uint8_t *blocks, int nb) {
  Buf in = gpu_alloc((size_t)nb * format->bytes), out = gpu_alloc((size_t)nb * 256 * 4);
  memcpy(in.ptr, blocks, (size_t)nb * format->bytes);
  gpu_begin();
  gpu_kern(format->kernel);
  gpu_buf(0, &in, 0), gpu_buf(1, &out, 0);
  gpu_go(nb * 8 / 256, 1, 1, 256, 1, 1);
  gpu_commit();
  gpu_sync();
  float *want = xmalloc((size_t)nb * 256 * 4);
  const float *got = (const float *)out.ptr;
  reference(format, blocks, want, nb);
  long bad = 0;
  for (long i = 0; i < (long)nb * 256; i++)
    if (!(got[i] == want[i]) && bad++ < 3)
      fprintf(stderr, "%s %s: block %ld value %ld: GPU %a, llama.cpp %a\n", format->name, what, i / 256, i % 256, got[i], want[i]);
  format->values += (long)nb * 256, format->failures += bad;
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
  str_cat(&s, "template <> void tables<block_q6_K>(threadgroup uint4 *tab, uint t, uint nt) {}\n");
  for (size_t i = 0; i < sizeof formats / sizeof *formats; i++) {
    Format *f = &formats[i];
    snprintf(f->entry, sizeof f->entry, "decode_%s", f->name);
    char kernel[1024];
    snprintf(kernel, sizeof kernel,
             "kernel void %s(device const block_%s *b [[buffer(0)]], device float4 *y [[buffer(1)]],\n"
             "  uint i [[thread_position_in_grid]], uint t [[thread_index_in_threadgroup]]) {\n"
             "  threadgroup uint4 tab[Code<block_%s>::tab];\n"
             "  tables<block_%s>(tab, t, 256);\n"
             "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
             "  float4 v[8]; dq32(b + i / 8, i %% 8, tab, v);\n"
             "  for (uint k = 0; k < 8; k++) y[8 * i + k] = v[k];\n}\n",
             f->entry, f->name, f->name, f->name);
    str_cat(&s, kernel);
  }
  gpu_init(s.p, s.n);
  free(s.p);

  // Random finite scale fields exercise all code and sign combinations.
  enum { NR = 8192 };
  srandom(1);
  for (size_t k = 0; k < sizeof formats / sizeof *formats; k++) {
    Format *f = &formats[k];
    f->kernel = gpu_kernel(f->entry);
    uint8_t *r = xmalloc((size_t)NR * f->bytes);
    for (size_t i = 0; i < (size_t)NR * f->bytes; i++) r[i] = (uint8_t)random();
    for (int i = 0; i < NR; i++) {
      uint8_t *block = r + (size_t)i * f->bytes;
      if (f->type == GG_IQ1_M) {
        uint8_t *sc = block + 48;
        if ((sc[5] >> 6) == 3 && (sc[7] & 0x70) == 0x70) sc[7] &= 0xbf;
      } else {
        for (int q = 0; q < (f->type == GG_Q2_K ? 2 : 1); q++) {
          uint8_t *p = block + (f->type == GG_Q2_K ? 80 + 2 * q : f->type == TYPE_Q6_K ? 208 : 0);
          uint16_t bits = u16(p);
          if ((bits & 0x7c00) == 0x7c00) bits &= ~0x0400;
          memcpy(p, &bits, 2);
        }
      }
    }
    check(f, "random", r, NR);
    free(r);
  }

  int tensors = 0;
  if (argc > 1) {
    gguf_open(argv[1]);
    static const char *mat[3] = {"gate", "up", "down"};
    uint8_t *b = xmalloc((size_t)PER * 210);
    for (int l = -1; l < 64; l++)
      for (int k = 0; k < (l < 0 ? 1 : 3); k++) {
        char name[64];
        if (l < 0) snprintf(name, sizeof name, "output.weight");
        else snprintf(name, sizeof name, "blk.%d.ffn_%s_exps.weight", l, mat[k]);
        Gguf t;
        if (!gguf_get(name, &t)) continue;
        if (l < 0 && t.type != TYPE_Q6_K) continue;  // other head types use native weights
        int matched = 0;
        for (size_t i = 0; i < sizeof formats / sizeof *formats; i++) {
          Format *f = &formats[i];
          if (f->type != t.type) continue;
          uint64_t nb = t.ne[0] / 256 * t.ne[1] * (t.ne[2] ? t.ne[2] : 1);
          for (int p = 0; p < 3; p++) {
            uint64_t first = p == 0 ? 0 : p == 1 ? nb / 2 : nb - PER;
            if (pread(t.fd, b, (size_t)PER * f->bytes, (off_t)(t.off + first * f->bytes)) != PER * f->bytes)
              die("%s: short read", name);
            check(f, name, b, PER);
          }
          f->tensors++;
          matched = 1;
        }
        if (!matched) die("%s: unchecked expert format %d", name, t.type);
        tensors++;
      }
    free(b);
    if (!tensors) die("%s: no routed expert tensor", argv[1]);
  }
  long failures = 0;
  for (size_t i = 0; i < sizeof formats / sizeof *formats; i++) {
    Format *f = &formats[i];
    printf("%s: %ld random values, %ld values from %d tensors; %ld differ from llama.cpp\n",
           f->name, (long)NR * 256, f->values - (long)NR * 256, f->tensors, f->failures);
    failures += f->failures;
  }
  if (tensors) printf("checked %d real expert and output-head tensors\n", tensors);
  return failures != 0;
}
