// Grouped experts must retain each product, BF16 boundary and ordered mixture.
#include "../../src/model/experts.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

enum { H = 2688, I = 1856, E = 3, N = 8 };
typedef struct { uint32_t row; float weight; } Entry;
static Linear matrices[E][3];
static Buf input, groups, entries, order, xg, xu, hidden, quantized, parts, output;
static Buf gathered, q, gate, up, middle, reference, expected, expected_middle, expected_parts, bf16;

static void fill_weight(Buf *buffer, const void *context) {
  unsigned char value = *(const unsigned char *)context;
  memset(buffer->ptr, value, buffer->size);
}
static int packed_weights(void) {
  static MemoryAccount account;
  memory_account_init(&account, "expert fixture");
  WeightSet *core = weights_new(&account, "backbone fixture");
  WeightCache cache = {0};
  WeightSet *expert = weights_cached(core, &cache, "expert fixture");
  Buf backbone = {0}, a = {0}, b = {0};
  unsigned char one = 0x35, two = 0xa9;
  weights_generated(core, &backbone, 256, fill_weight, &one, 1);
  weights_generated(expert, &a, 257, fill_weight, &one, 1);
  weights_generated(expert, &b, 700, fill_weight, &two, 1);
  if (!weights_load(core)) return 0;
  memory_pin(weights_slot(core), 1);
  for (int pass = 0; pass < 2; pass++) {
    if (!weights_load(expert)) return 0;
    int shared = a.obj == b.obj && b.off >= a.off + a.size;
    int values = a.size == 257 && b.size == 700;
    for (size_t i = 0; i < a.size; i++) values &= a.ptr[i] == one;
    for (size_t i = 0; i < b.size; i++) values &= b.ptr[i] == two;
    int charged = cache.slots == 1 && cache.bytes == gpu_allocation_size(512 + 700);
    printf("expert allocation pass %d: shared=%d, values=%d, single charge=%d\n", pass, shared, values, charged);
    if (!shared || !values || !charged) return 0;
    memory_reclaim(cache.bytes);
    if (cache.slots || cache.bytes || a.obj || b.obj) return 0;
  }
  memory_pin(weights_slot(core), 0);
  weights_free(expert);
  weights_free(core);
  return 1;
}

static void rounded(const Buf *b, int count) {
  gpu_kern(gpu_kernel("round_bf16"));
  gpu_buf(0, b, 0);
  gpu_bytes(1, &count, 4);
  gpu_go((count + 255) / 256, 1, 1, 256, 1, 1);
}
static void product(const Linear *m, const Buf *x, const Buf *y, int n) {
  struct { uint32_t count, bits; float scale; uint32_t pad; } p = {n * m->cols, 4, m->input_scale, 0};
  gpu_kern(gpu_kernel("quantize_input"));
  gpu_buf(0, x, 0); gpu_buf(1, &q, 0); gpu_bytes(2, &p, sizeof p);
  gpu_go((p.count + 255) / 256, 1, 1, 256, 1, 1);
  Linear scaled = *m;
  scaled.scale *= m->input_scale;
  linear_product(&scaled, &q, 0, y, 0, n, 0, &bf16, 0);
  rounded(y, n * m->rows);
}
static void baseline(int n, int relu) {
  memset(expected.ptr, 0, expected.size);
  gpu_begin();
  for (int e = 0; e < E; e++) {
    uint32_t dims[2] = {H, n};
    gpu_kern(gpu_kernel("gather_rows"));
    gpu_buf(0, &input, 0); gpu_buf(1, &entries, e * n * sizeof(Entry)); gpu_buf(2, &gathered, 0);
    gpu_bytes(3, dims, sizeof dims); gpu_go((H * n + 255) / 256, 1, 1, 256, 1, 1);
    uint32_t count = n * I;
    if (relu) {
      product(&matrices[e][1], &gathered, &middle, n);
      gpu_kern(gpu_kernel("nm_relu2"));
      gpu_buf(0, &middle, 0); gpu_bytes(1, &count, 4);
    } else {
      product(&matrices[e][0], &gathered, &gate, n);
      product(&matrices[e][1], &gathered, &up, n);
      gpu_kern(gpu_kernel("flash_swiglu"));
      gpu_buf(0, &gate, 0); gpu_buf(1, &up, 0); gpu_buf(2, &middle, 0); gpu_bytes(3, &count, 4);
    }
    gpu_go((count + 255) / 256, 1, 1, 256, 1, 1);
    product(&matrices[e][2], &middle, &reference, n);
    gpu_kern(gpu_kernel(relu ? "nm_scatter" : "flash_scatter"));
    gpu_buf(0, &reference, 0); gpu_buf(1, &entries, e * n * sizeof(Entry)); gpu_buf(2, &expected, 0);
    gpu_bytes(3, dims, sizeof dims); gpu_go((H * n + 255) / 256, 1, 1, 256, 1, 1);
    GpuCopy copies[] = {{&middle, &expected_middle, 0, (size_t)e * n * I * 4, (size_t)n * I * 4},
                        {&reference, &expected_parts, 0, (size_t)e * n * H * 4, (size_t)n * H * 4}};
    gpu_copies(copies, 2);
  }
  if (relu) rounded(&expected, n * H);
  gpu_commit(); gpu_sync();
}
static void grouped(int n, int relu) {
  uint32_t dims[2] = {H, I};
  gpu_begin();
  for (int e = 0; e < E; e++)
    for (int k = 0; k < 3; k++) { gpu_use(&matrices[e][k].w); gpu_use(&matrices[e][k].s); }
  gpu_kern(gpu_kernel(relu ? "expert_input_relu2" : "expert_input"));
  gpu_buf(0, &groups, 0); gpu_buf(1, &entries, 0); gpu_buf(2, &input, 0);
  gpu_buf(3, &xg, 0); gpu_buf(4, &xu, 0); gpu_bytes(5, dims, sizeof dims);
  gpu_go((H + 255) / 256, E, n, 256, 1, 1);
  gpu_kern(gpu_kernel(relu ? "expert_up_relu2" : "expert_up"));
  gpu_buf(0, &groups, 0); gpu_buf(1, &xg, 0); gpu_buf(2, &xu, 0); gpu_buf(3, &hidden, 0);
  gpu_bytes(4, dims, sizeof dims); gpu_go((I + 7) / 8, E, 1, 256, 1, 1);
  gpu_kern(gpu_kernel("expert_hidden"));
  gpu_buf(0, &groups, 0); gpu_buf(1, &hidden, 0); gpu_buf(2, &quantized, 0);
  gpu_bytes(3, dims, sizeof dims); gpu_go((I + 255) / 256, E, n, 256, 1, 1);
  gpu_kern(gpu_kernel("expert_down"));
  gpu_buf(0, &groups, 0); gpu_buf(1, &quantized, 0); gpu_buf(2, &parts, 0);
  gpu_bytes(3, dims, sizeof dims); gpu_go((H + 7) / 8, E, 1, 256, 1, 1);
  uint32_t sum[3] = {H, n, E};
  gpu_kern(gpu_kernel(relu ? "expert_sum_fp32" : "expert_sum"));
  gpu_buf(0, &parts, 0); gpu_buf(1, &entries, 0); gpu_buf(2, &order, 0); gpu_buf(3, &output, 0);
  gpu_bytes(4, sum, sizeof sum); gpu_go((H * n + 255) / 256, 1, 1, 256, 1, 1);
  gpu_commit(); gpu_sync();
}
int main(void) {
  gpu_start(); linear_init();
#define B(b, n) b = gpu_alloc((size_t)(n))
  B(input, N * H * 4); B(groups, E * sizeof(ExpertGroup)); B(entries, E * N * sizeof(Entry)); B(order, E * N * 4);
  B(xg, E * N * H * 4); B(xu, E * N * H * 4); B(hidden, E * N * I * 4); B(quantized, E * N * I * 4);
  B(parts, E * N * H * 4); B(output, N * H * 4); B(gathered, N * H * 4); B(q, N * H * 4);
  B(gate, N * I * 4); B(up, N * I * 4); B(middle, N * I * 4); B(reference, N * H * 4);
  B(expected_parts, E * N * H * 4); B(expected, N * H * 4); B(expected_middle, E * N * I * 4); B(bf16, 128 * H * 2);
#undef B
  for (int e = 0; e < E; e++) for (int k = 0; k < 3; k++) {
    Linear *m = &matrices[e][k];
    *m = (Linear){.rows = k == 2 ? H : I, .cols = k == 2 ? I : H, .kind = 'n',
                  .scale = .003f * (e + k + 1), .input_scale = .01337f * (1 + e + 2 * k)};
    m->w = gpu_alloc((size_t)m->rows * m->cols / 2); m->s = gpu_alloc((size_t)m->rows * m->cols / 16);
    for (size_t j = 0; j < m->w.size; j++) m->w.ptr[j] = (j * 71 + j / 93 + e * 11 + k * 7) & 255;
    for (size_t j = 0; j < m->s.size; j++) m->s.ptr[j] = 32 + (j * 13 + j / 17 + e + k) % 40;
  }
  for (int i = 0; i < N * H; i++) ((float *)input.ptr)[i] = sinf(i * .031f) * 3;
  for (int relu = 0; relu < 2; relu++) for (int n = 1; n <= N; n++) {
    ExpertGroup *g = (ExpertGroup *)groups.ptr;
    for (int e = 0; e < E; e++) {
      g[e] = (ExpertGroup){expert_matrix(&matrices[e][0]), expert_matrix(&matrices[e][1]),
                           expert_matrix(&matrices[e][2]), e * n, n};
      for (int r = 0; r < n; r++) {
        int row = (r + e) % n;
        ((Entry *)entries.ptr)[e * n + r] = (Entry){row, .25f + e * .125f};
        ((uint32_t *)order.ptr)[row * E + e] = e * n + r;
      }
    }
    baseline(n, relu); grouped(n, relu);
    int diff = 0, middle_diff = 0, parts_diff = 0;
    for (int i = 0; i < n * H; i++) diff += ((uint32_t *)output.ptr)[i] != ((uint32_t *)expected.ptr)[i];
    for (int i = 0; i < E * n * I; i++) middle_diff += ((uint32_t *)hidden.ptr)[i] != ((uint32_t *)expected_middle.ptr)[i];
    for (int i = 0; i < E * n * H; i++) parts_diff += ((uint32_t *)parts.ptr)[i] != ((uint32_t *)expected_parts.ptr)[i];
    printf("%s expert rows=%d: hidden %d, parts %d, ordered output %d differing values\n",
           relu ? "ReLU2" : "SwiGLU", n, middle_diff, parts_diff, diff);
    if (diff || middle_diff || parts_diff) return 1;
  }
  return packed_weights() ? 0 : 1;
}
