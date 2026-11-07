// Compare MXFP4 prefill activation staging with the scalar expert kernel.
// One synthetic expert; no checkpoint or SSD reads.
#include "myds4.h"
#include "../reference/activation_format.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

enum { M = 32, K = 5120, H = 2304 };
typedef struct { uint64_t p[3]; float weight, pad; } ExpertEntry;
typedef struct { uint64_t p[3]; uint32_t start, count; } Tile;

int main(void) {
  alarm(90);
  lock_model();
  static const char source[] = {
#embed "../../src/kernels/formats.metal"
      , '\n',
#embed "../../src/kernels/kernels.metal"
      , '\n',
#embed "../../src/kernels/gguf_grids.metal"
      , '\n',
#embed "../../src/kernels/gguf.metal"
  };
  gpu_init(source, sizeof source);
  Buf weights = gpu_alloc(EXPERT_W + EXPERT_S), entry = gpu_alloc(sizeof(ExpertEntry));
  Buf tiles = gpu_alloc(sizeof(Tile)), x = gpu_alloc(M * K * 4), ids = gpu_alloc(M * 4), pw = gpu_alloc(M * 4);
  Buf xh = gpu_alloc((M + 64) * K * 2), h = gpu_alloc(M * H * 4), quantized = gpu_alloc(M * H * 4);
  Buf reference = gpu_alloc(M * H * 4);
  Buf hh = gpu_alloc((M + 64) * H * 2), y = gpu_alloc(M * K * 4);
  memset(weights.ptr, 0x22, EXPERT_W); // E2M1 value 1 in every weight
  memset(weights.ptr + EXPERT_W, 127, EXPERT_S); // E8M0 scale 1
  *(ExpertEntry *)entry.ptr = (ExpertEntry){{weights.gpu, weights.gpu + EXPERT_W, 0}, 1, 0};
  *(Tile *)tiles.ptr = (Tile){{weights.gpu, weights.gpu + EXPERT_W, 0}, 0, M};
  memset(xh.ptr, 0, xh.size);
  memset(hh.ptr, 0, hh.size);
  for (int r = 0; r < M; r++) {
    ((int *)ids.ptr)[r] = r;
    ((float *)pw.ptr)[r] = 1;
    float value = r == M - 1 ? 0x1p-31f : r % 3 == 0 ? 0x1p-26f : r % 3 == 1 ? 0x1p-12f : -0x1p-12f;
    for (int d = 0; d < K; d++) ((float *)x.ptr)[r * K + d] = value;
  }
  int scalar = gpu_kernel("moe_up"), gather = gpu_kernel("gather_mxfp4");
  int batch = gpu_kernel("moe_up_m"), quant = gpu_kernel("act_quant");
  int quant_scaled = gpu_kernel("act_quant_mxfp4"), down = gpu_kernel("moe_down_m");
  uint32_t count = M;
  gpu_begin();
  for (int r = 0; r < M; r++) {
    gpu_kern(scalar);
    gpu_buf(0, &entry, 0), gpu_buf(1, &x, (size_t)r * K * 4), gpu_buf(2, &reference, (size_t)r * H * 4);
    gpu_go(H / 32, 1, 1, 256, 1, 1);
  }
  gpu_kern(gather);
  gpu_buf(0, &x, 0), gpu_buf(1, &ids, 0), gpu_buf(2, &xh, 0);
  gpu_go(K / 4 / 256, M, 1, 256, 1, 1);
  gpu_kern(batch);
  gpu_buf(0, &tiles, 0), gpu_buf(1, &xh, 0), gpu_buf(2, &pw, 0), gpu_buf(3, &h, 0), gpu_bytes(4, &count, 4);
  gpu_go(1, H / 64, 1, 128, 1, 1);
  gpu_kern(quant);
  gpu_buf(0, &h, 0), gpu_buf(1, &quantized, 0);
  gpu_go(M * H / 256, 1, 1, 256, 1, 1);
  gpu_kern(quant_scaled);
  gpu_buf(0, &h, 0), gpu_buf(1, &hh, 0);
  gpu_go(M * H / 256, 1, 1, 256, 1, 1);
  gpu_kern(down);
  gpu_buf(0, &tiles, 0), gpu_buf(1, &hh, 0), gpu_buf(2, &y, 0), gpu_bytes(3, &count, 4);
  gpu_go(1, K / 128, 1, 128, 1, 1);
  gpu_commit(), gpu_sync();
  int small = 0, ordinary = 0, down_errors = 0;
  for (int r = 0; r < M; r++)
    for (int d = 0; d < H; d++) {
      float a = ((float *)reference.ptr)[r * H + d], b = ((float *)quantized.ptr)[r * H + d];
      if (!isfinite(a) || !isfinite(b) || memcmp(&a, &b, sizeof a)) {
        if (r % 3 == 0) small++;
        else ordinary++;
      }
    }
  for (int r = 0; r < M; r++) {
    float expected = ref_from_bf16(ref_bf16(((float *)reference.ptr)[r * H] * H));
    for (int d = 0; d < K; d++) {
      float actual = ((float *)y.ptr)[r * K + d];
      down_errors += !isfinite(actual) || memcmp(&expected, &actual, sizeof actual) != 0;
    }
  }
  int gather_edge = ((uint16_t *)xh.ptr)[(M - 1) * K] != 1; // 2^-31 * 128 is half's smallest subnormal.
  const int exponents[] = {-31, -30, -26, -17, -12, -8, 0, 8};
  for (int r = 0; r < M; r++) {
    float value = ldexpf(r & 8 ? -1 : 1, exponents[r % 8]);
    for (int d = 0; d < H; d++) ((float *)h.ptr)[r * H + d] = value;
  }
  gpu_begin();
  gpu_kern(quant_scaled);
  gpu_buf(0, &h, 0), gpu_buf(1, &hh, 0);
  gpu_go(M * H / 256, 1, 1, 256, 1, 1);
  gpu_kern(down);
  gpu_buf(0, &tiles, 0), gpu_buf(1, &hh, 0), gpu_buf(2, &y, 0), gpu_bytes(3, &count, 4);
  gpu_go(1, K / 128, 1, 128, 1, 1);
  gpu_commit(), gpu_sync();
  int edge_down = 0;
  for (int r = 0; r < M; r++) {
    float expected = ref_from_bf16(ref_bf16(((float *)h.ptr)[r * H] * H));
    for (int d = 0; d < K; d++) {
      float actual = ((float *)y.ptr)[r * K + d];
      edge_down += !isfinite(actual) || memcmp(&expected, &actual, sizeof actual) != 0;
    }
  }
  printf("{\"mxfp4_activation_check\":{\"small_differences\":%d,\"ordinary_differences\":%d,"
         "\"down_differences\":%d,\"small_reference\":%.9g,\"small_prefill\":%.9g,"
         "\"gather_edge_differences\":%d,\"edge_down_differences\":%d}}\n",
         small, ordinary, down_errors, ((float *)reference.ptr)[0], ((float *)quantized.ptr)[0], gather_edge, edge_down);
  return small || ordinary || down_errors || gather_edge || edge_down;
}
