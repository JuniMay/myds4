// Inspect wide-verification dispatches on the CPU, without a Metal device.
#include "../../src/model/deepseek/model.h"
#include "myds4.h"
#include <string.h>

static int kernel, dispatches, kernels[12], row_counts[12];
static size_t input_offsets[12], output_offsets[12];
static void spy_kernel(int value) { kernel = value; }
static void spy_buffer(int index, const Buf *b, size_t offset) {
  if (index == 2) input_offsets[dispatches] = offset;
  if (index == 3) output_offsets[dispatches] = offset;
}
static void spy_bytes(int index, const void *bytes, size_t size) {
  if (index == 4 && kernel == 101) row_counts[dispatches] = ((const uint32_t *)bytes)[6];
}
static void spy_go(uint32_t x, uint32_t y, uint32_t z, uint32_t tx, uint32_t ty, uint32_t tz) {
  kernels[dispatches++] = kernel;
}
#define gpu_kern spy_kernel
#define gpu_buf spy_buffer
#define gpu_bytes spy_bytes
#define gpu_go spy_go
#include "../../src/model/deepseek/model.c"

static int checks, failures;
static void expect(const char *name, int ok) {
  checks++;
  if (!ok) failures++, fprintf(stderr, "FAIL: %s\n", name);
}

int main(void) {
  opt.lookup = 0;
  // the window and the rewind margin; the compressor two rows more at least
  expect("ordinary decoding keeps the window and the rewind margin", RING == WINDOW + REWIND && PENDING >= REWIND + 2);
  opt.draft = BLOCK;
  expect("DSpark's drafts fit within the rewind margin", RING == WINDOW + REWIND);
  opt.draft = 0, opt.lookup = 15;
  expect("wide verification retains overwritten local history", RING == 143 && PENDING >= 17);
  K_mvm_bf16 = 101;
  K_mm_bf16 = 102;
  Lin weights = {.n = EXPERTS, .k = DIM};
  Buf input = {0}, output = {0};
  Prod p = {&weights, &output, 128, EXPERTS};
  verifying = 1;
  products(1, &p, 16, &input, 64, DIM, 0);
  expect("sixteen rows use three bounded vector batches", dispatches == 3);
  expect("batch row counts are six, six and four", row_counts[0] == 6 && row_counts[1] == 6 && row_counts[2] == 4);
  expect("all segments use the vector kernel", kernels[0] == 101 && kernels[1] == 101 && kernels[2] == 101);
  expect("segments read their own input rows", input_offsets[0] == 64 && input_offsets[1] == 64 + 6u * DIM * 4 && input_offsets[2] == 64 + 12u * DIM * 4);
  expect("segments write their own output rows", output_offsets[0] == 128 && output_offsets[1] == 128 + 6u * EXPERTS * 4 && output_offsets[2] == 128 + 12u * EXPERTS * 4);
  dispatches = 0;
  verifying = 0;
  products(1, &p, 16, &input, 64, DIM, 0);
  expect("ordinary prefill retains the matrix path", dispatches == 1 && kernels[0] == 102);
  fprintf(stderr, "verification schedule: %d/%d CPU checks passed\n", checks - failures, checks);
  return failures != 0;
}
