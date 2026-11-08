// Grouped experts must retain each product, BF16 boundary and ordered mixture.
#include "../../src/model/experts.h"
#include <math.h>
#include <fcntl.h>
#include <pthread.h>
#include <setjmp.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

enum { H = 2688, I = 1856, E = 3, N = 32 };
typedef struct { uint32_t row; float weight; } Entry;
static Linear matrices[E][3];
static Buf input, groups, entries, order, row_groups, tiles, xg, xu, hidden, quantized, parts, output;
static Buf gathered, q, gate, up, middle, reference, expected, expected_middle, expected_parts, bf16;
static jmp_buf read_failure;
static pthread_t engine_thread;

static void catch_read_failure(void) {
  if (!pthread_equal(engine_thread, pthread_self())) _Exit(3);
  longjmp(read_failure, 1);
}
static int failed_batch(const char *path) {
  int fd = open(path, O_RDONLY);
  struct stat stat;
  if (fd < 0 || fstat(fd, &stat)) return 0;
  static MemoryAccount account;
  memory_account_init(&account, "read-error fixture");
  WeightSet *core = weights_new(&account, "read-error backbone");
  static WeightCache cache;
  WeightSet *sets[] = {weights_cached(core, &cache, "readable expert"),
                       weights_cached(core, &cache, "truncated expert")};
  static Buf values[2];
  weights_tensor(sets[0], &values[0], (Tensor){.fd = fd, .off = 0, .size = 64}, 0);
  weights_tensor(sets[1], &values[1], (Tensor){.fd = fd, .off = stat.st_size + 1, .size = 64}, 0);
  uint64_t allocated = memory_usage().allocated;
  engine_thread = pthread_self();
  die_hook = catch_read_failure;
  int caught = 0;
  if (setjmp(read_failure) == 0) weights_load_many(sets, 2);
  else caught = 1;
  die_hook = NULL;
  int restored = !values[0].obj && !values[1].obj && !cache.slots && !cache.bytes &&
                  memory_usage().allocated == allocated;
  weights_free(sets[0]); weights_free(sets[1]); weights_free(core);
  close(fd);
  printf("parallel read failure: caller received error=%d, allocations restored=%d\n", caught != 0, restored);
  return caught && restored;
}

static void fill_weight(Buf *buffer, const void *context) {
  unsigned char value = *(const unsigned char *)context;
  memset(buffer->ptr, value, buffer->size);
}
static int held_selection(void) {
  static MemoryAccount account;
  memory_account_init(&account, "held-expert fixture");
  WeightSet *core = weights_new(&account, "held-expert backbone");
  WeightCache cache = {0};
  Expert fake[3] = {0};
  ExpertWork w = {0};
  experts_init(&w, 64, 64, 3, 2, EXPERT_SWIGLU, EXPERT_SUM_BF16);
  experts_workspace(&w, 1, 1);
  Buf x = gpu_alloc(64 * 4), a = gpu_alloc(64 * 4), b = gpu_alloc(64 * 4);
  Buf table = gpu_alloc(3 * sizeof(ExpertGroup)), selected = gpu_alloc(8), coefficients = gpu_alloc(8);
  Buf unique = gpu_alloc(4);
  for (int i = 0; i < 64; i++) ((float *)x.ptr)[i] = .015625f;
  ((uint32_t *)selected.ptr)[0] = 1; ((uint32_t *)selected.ptr)[1] = 0;
  ((float *)coefficients.ptr)[0] = .25f; ((float *)coefficients.ptr)[1] = .75f;
  for (int e = 0; e < 3; e++) {
    fake[e].weights = weights_cached(core, &cache, "held expert");
    Linear *m[] = {&fake[e].gate, &fake[e].up, &fake[e].down};
    for (int k = 0; k < 3; k++) {
      *m[k] = (Linear){.rows = 64, .cols = 64, .kind = 'n', .scale = .125f, .input_scale = 1};
      unsigned char value = 0x11 + e, scale = 0x38;
      weights_generated(fake[e].weights, &m[k]->w, 2048, fill_weight, &value, 1);
      weights_generated(fake[e].weights, &m[k]->s, 256, fill_weight, &scale, 1);
    }
    if (!weights_load(fake[e].weights)) return 0;
  }
  experts_table(&w, fake, &table, 0);
  if (!experts_acquire(&w, fake, (uint32_t *)selected.ptr, (float *)coefficients.ptr, 1, 0, &cache.stats))
    return 0;
  gpu_begin();
  experts_encode(&w, fake, &x); experts_sum(&w, &a);
  experts_resident(&w, &table, 0, &selected, &coefficients, &x, &unique, 0, 1, 0);
  experts_sum(&w, &b);
  gpu_commit();
  memory_reclaim(cache.bytes);
  int held = memory_ready(weights_slot(fake[0].weights)) && memory_ready(weights_slot(fake[1].weights)) &&
             !memory_ready(weights_slot(fake[2].weights));
  gpu_sync();
  int equal = !memcmp(a.ptr, b.ptr, a.size);
  ((uint32_t *)selected.ptr)[0] = 2;
  if (!experts_acquire(&w, fake, (uint32_t *)selected.ptr, (float *)coefficients.ptr, 1, 0, &cache.stats))
    return 0;
  memory_reclaim(cache.bytes);
  int replaced = memory_ready(weights_slot(fake[0].weights)) && !memory_ready(weights_slot(fake[1].weights)) &&
                 memory_ready(weights_slot(fake[2].weights));
  gpu_begin(); experts_encode(&w, fake, &x); experts_sum(&w, &a); gpu_commit(); gpu_sync();
  experts_release(&w);
  memory_reclaim(cache.bytes);
  int released = !cache.slots && !cache.bytes;
  for (int e = 0; e < 3; e++) weights_free(fake[e].weights);
  weights_free(core);
  Buf *buffers[] = {&w.plan, &w.entries, &w.order, &w.groups, &w.row_groups, &w.tiles, &w.input, &w.second,
                    &w.hidden_values, &w.quantized, &w.output, &x, &a, &b, &table, &selected, &coefficients, &unique};
  for (size_t i = 0; i < sizeof buffers / sizeof *buffers; i++) gpu_free(buffers[i]);
  printf("held selection: survives resident planning=%d, output equal=%d, next selection=%d, released=%d\n",
         held, equal, replaced, released);
  return held && equal && replaced && released;
}
static int packed_weights(void) {
  static MemoryAccount account;
  memory_account_init(&account, "expert fixture");
  WeightSet *core = weights_new(&account, "backbone fixture");
  WeightCache cache = {0};
  WeightSet *expert = weights_cached(core, &cache, "expert fixture");
  WeightPresence presence = {0};
  weights_track(expert, &presence);
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
    int charged = cache.slots == 1 && cache.bytes == gpu_allocation_size(512 + 700) &&
                   presence.count == 1 && presence.generation == (unsigned)(pass * 2 + 1);
    printf("expert allocation pass %d: shared=%d, values=%d, single charge=%d\n", pass, shared, values, charged);
    if (!shared || !values || !charged) return 0;
    memory_reclaim(cache.bytes);
    if (cache.slots || cache.bytes || a.obj || b.obj || presence.count ||
        presence.generation != (unsigned)(pass * 2 + 2)) return 0;
  }
  memory_pin(weights_slot(core), 0);
  weights_free(expert);
  weights_free(core);
  return 1;
}

static double prompt_product(int kernel, int tile, int rows, const Buf *x, const Buf *out, int repeats) {
  const Linear *m = &matrices[0][0];
  struct { uint32_t m, n, k, pad; float scale; uint32_t pad0, pad1, pad2; }
      p = {.m = rows, .n = I, .k = H, .scale = m->scale};
  double before = gpu_busy;
  gpu_begin();
  gpu_kern(kernel);
  gpu_buf(0, &m->w, 0); gpu_buf(1, &m->s, 0); gpu_buf(2, x, 0); gpu_buf(3, out, 0);
  gpu_bytes(4, &p, sizeof p);
  for (int i = 0; i < repeats; i++) gpu_go((I + 63) / 64, (rows + tile - 1) / tile, 1, 128, 1, 1);
  gpu_commit(); gpu_sync();
  return (gpu_busy - before) * 1e6 / repeats;
}
static int prompt_tiles(void) {
  Buf x = gpu_alloc(128 * H * 2), a = gpu_alloc(64 * I * 4), b = gpu_alloc(64 * I * 4);
  for (int i = 0; i < 128 * H; i++) {
    float value = sinf(i * .023f) * 2;
    uint32_t bits; memcpy(&bits, &value, 4);
    ((uint16_t *)x.ptr)[i] = (bits + 0x7fff + ((bits >> 16) & 1)) >> 16;
  }
  int old = gpu_kernel("gemm_nvfp4"), small = gpu_kernel("gemm16_nvfp4");
  int counts[] = {1, 2, 4, 8, 9, 16, 17, 32, 64};
  for (size_t n = 0; n < sizeof counts / sizeof *counts; n++) {
    int rows = counts[n];
    prompt_product(old, 128, rows, &x, &a, 1);
    prompt_product(small, 16, rows, &x, &b, 1);
    int different = 0;
    for (int i = 0; i < rows * I; i++) different += ((uint32_t *)a.ptr)[i] != ((uint32_t *)b.ptr)[i];
    double before = prompt_product(old, 128, rows, &x, &a, 16);
    double candidate = prompt_product(small, 16, rows, &x, &b, 16);
    double after = prompt_product(old, 128, rows, &x, &a, 16);
    printf("NVFP4 matrix rows=%d: %d differing values; %.2f / %.2f / %.2f us\n",
           rows, different, before, candidate, after);
    if (different) return 0;
  }
  gpu_free(&x); gpu_free(&a); gpu_free(&b);
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
  linear_product(&scaled, &q, 0, y, 0, n, n > 8, &bf16, 0);
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
  int jobs = 0;
  for (int e = 0; e < E; e++) for (int r = 0; r < n; r += 16)
    ((ExpertTile *)tiles.ptr)[jobs++] = (ExpertTile){e, r};
  gpu_begin();
  for (int e = 0; e < E; e++)
    for (int k = 0; k < 3; k++) { gpu_use(&matrices[e][k].w); gpu_use(&matrices[e][k].s); }
  gpu_kern(gpu_kernel(relu ? "expert_input_relu2" : "expert_input"));
  gpu_buf(0, &groups, 0); gpu_buf(1, &entries, 0); gpu_buf(2, &input, 0);
  gpu_buf(3, &xg, 0); gpu_buf(4, &xu, 0); gpu_bytes(5, dims, sizeof dims); gpu_buf(6, &row_groups, 0);
  gpu_go((H + 255) / 256, E * n, 1, 256, 1, 1);
  if (n <= 8) {
    gpu_kern(gpu_kernel(relu ? (n == 1 ? "expert_up_relu2_single" : "expert_up_relu2")
                             : (n == 1 ? "expert_up_single" : "expert_up")));
    gpu_buf(0, &groups, 0); gpu_buf(1, &xg, 0); gpu_buf(2, &xu, 0); gpu_buf(3, &hidden, 0);
    gpu_bytes(4, dims, sizeof dims); gpu_go((I + 7) / 8, E, 1, 256, 1, 1);
  } else {
    gpu_kern(gpu_kernel(relu ? "expert_up_relu2_matrix" : "expert_up_matrix"));
    gpu_buf(0, &groups, 0); gpu_buf(1, &xg, 0); gpu_buf(2, &xu, 0); gpu_buf(3, &hidden, 0);
    gpu_bytes(4, dims, sizeof dims); gpu_buf(5, &tiles, 0);
    gpu_go((I + 63) / 64, jobs, 1, 128, 1, 1);
  }
  gpu_kern(gpu_kernel("expert_hidden"));
  gpu_buf(0, &groups, 0); gpu_buf(1, &hidden, 0); gpu_buf(2, &quantized, 0);
  gpu_bytes(3, dims, sizeof dims); gpu_buf(4, &row_groups, 0);
  gpu_go((I + 255) / 256, E * n, 1, 256, 1, 1);
  if (n <= 8) {
    gpu_kern(gpu_kernel(n == 1 ? "expert_down_single" : "expert_down"));
    gpu_buf(0, &groups, 0); gpu_buf(1, &quantized, 0); gpu_buf(2, &parts, 0);
    gpu_bytes(3, dims, sizeof dims); gpu_go((H + 7) / 8, E, 1, 256, 1, 1);
  } else {
    gpu_kern(gpu_kernel("expert_down_matrix"));
    gpu_buf(0, &groups, 0); gpu_buf(1, &quantized, 0); gpu_buf(2, &quantized, 0); gpu_buf(3, &parts, 0);
    gpu_bytes(4, dims, sizeof dims); gpu_buf(5, &tiles, 0);
    gpu_go((H + 63) / 64, jobs, 1, 128, 1, 1);
  }
  uint32_t sum[3] = {H, n, E};
  gpu_kern(gpu_kernel(relu ? "expert_sum_fp32" : "expert_sum"));
  gpu_buf(0, &parts, 0); gpu_buf(1, &entries, 0); gpu_buf(2, &order, 0); gpu_buf(3, &output, 0);
  gpu_bytes(4, sum, sizeof sum); gpu_go((H * n + 255) / 256, 1, 1, 256, 1, 1);
  gpu_commit(); gpu_sync();
}

static int resident_groups(int n, int relu, int chosen) {
  static ExpertWork work[2][2];
  static Buf selected, coefficients, unique;
  ExpertWork *w = &work[relu][chosen == E];
  if (!w->hidden) experts_init(w, H, I, E, chosen, relu ? EXPERT_RELU2 : EXPERT_SWIGLU,
                              relu ? EXPERT_SUM_FP32 : EXPERT_SUM_BF16);
  experts_workspace(w, n, 1);
  if (!selected.obj) {
    selected = gpu_alloc(N * E * 4);
    coefficients = gpu_alloc(N * E * 4);
    unique = gpu_alloc(4);
  }
  int used[E] = {0}, expected_unique = 0;
  for (int r = 0; r < n; r++) for (int k = 0; k < chosen; k++) {
    int e = chosen == E ? (k + r * 2) % E : ((k ^ (r & 1)) ? (r % 7 ? 1 : 2) : 0);
    ((uint32_t *)selected.ptr)[r * chosen + k] = e;
    ((float *)coefficients.ptr)[r * chosen + k] = .25f + e * .125f;
    used[e] = 1;
  }
  for (int e = 0; e < E; e++) expected_unique += used[e];
  gpu_begin();
  experts_resident(w, &groups, 0, &selected, &coefficients, &input, &unique, 0, n, 1);
  experts_sum(w, &output);
  gpu_commit(); gpu_sync();
  int differences = 0;
  for (int i = 0; i < n * H; i++) differences += ((uint32_t *)output.ptr)[i] != ((uint32_t *)expected.ptr)[i];
  for (int i = 0; i < chosen * n; i++) {
    ExpertGroup *g = (ExpertGroup *)w->groups.ptr + ((uint32_t *)w->row_groups.ptr)[i];
    int e = 0;
    while (e < E && g->up.weight != matrices[e][1].w.gpu) e++;
    if (e == E) return 0;
    uint32_t row = ((ExpertRow *)w->entries.ptr)[i].row;
    int original = e * n + (row + n - e % n) % n;
    for (int k = 0; k < I; k++) differences += ((uint32_t *)w->hidden_values.ptr)[i * I + k] !=
                                               ((uint32_t *)expected_middle.ptr)[original * I + k];
    for (int k = 0; k < H; k++) differences += ((uint32_t *)w->output.ptr)[i * H + k] !=
                                               ((uint32_t *)expected_parts.ptr)[original * H + k];
  }
  printf("resident %s rows=%d chosen=%d: %d differing values, unique=%u\n", relu ? "ReLU2" : "SwiGLU",
         n, chosen, differences, *(uint32_t *)unique.ptr);
  return !differences && *(uint32_t *)unique.ptr == (unsigned)expected_unique;
}

int main(int argc, char **argv) {
  gpu_start(); linear_init();
#define B(b, n) b = gpu_alloc((size_t)(n))
  B(row_groups, E * N * 4); B(tiles, E * ((N + 15) / 16) * sizeof(ExpertTile)); B(input, N * H * 4); B(groups, E * sizeof(ExpertGroup)); B(entries, E * N * sizeof(Entry)); B(order, E * N * 4);
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
        ((uint32_t *)row_groups.ptr)[e * n + r] = e;
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
    if (diff || middle_diff || parts_diff || !resident_groups(n, relu, E)) return 1;
    if (n == 1 || n == 2 || n == 8 || n == 9 || n == 16 || n == 32) {
      for (int e = 0; e < E; e++) for (int r = 0; r < n; r++) {
        Entry *entry = (Entry *)entries.ptr + e * n + r;
        if (e && e != (entry->row % 7 ? 1 : 2)) entry->weight = 0;
      }
      baseline(n, relu);
      if (!resident_groups(n, relu, 2)) return 1;
    }
  }
  return packed_weights() && held_selection() && prompt_tiles() && failed_batch(argv[0]) ? 0 : 1;
}
