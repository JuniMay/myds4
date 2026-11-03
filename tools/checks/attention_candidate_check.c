// Candidate masks must follow prompt row indices across decoder compaction.
// Observe dispatch offsets on the CPU without loading weights or using a GPU.
#include "myds4.h"
static int dispatch_kernel, observed;
static size_t candidate_offset;
static void spy_kernel(int k) { dispatch_kernel = k; }
static void spy_buffer(int index, const Buf *b, size_t offset) {
  if (dispatch_kernel == 101 && index == 3) candidate_offset = offset, observed++;
}
static void spy_bytes(int index, const void *data, size_t size) {}
static void spy_go(int x, int y, int z, int tx, int ty, int tz) {}
#define gpu_kern spy_kernel
#define gpu_buf spy_buffer
#define gpu_bytes spy_bytes
#define gpu_go spy_go
#include "../../src/engine/model.c"

int main(void) {
  opt.ctx = 32768, cand_words = 129, sc_rows = 256;
  L[24].ratio = L[24].index_source = 1, L[24].src = 20;
  K_index_topk = 101;
  int failed = 0;
  for (int row = 4; row <= MB + 4; row += MB) {
    observed = 0;
    attend_rows(24, 133, 16384 + row, 0, row);
    size_t expected = (size_t)row * cand_words * 4;
    int ok = observed == 1 && candidate_offset == expected;
    printf("candidate prompt row %d: offset=%zu expected=%zu %s\n", row, candidate_offset, expected, ok ? "PASS" : "FAIL");
    failed += !ok;
  }
  return failed != 0;
}
