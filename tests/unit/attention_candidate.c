// Candidate masks must follow prompt row indices across decoder compaction:
// both the scoring and the selection of a later indexer take a row's bits.
// Observe dispatch offsets on the CPU without loading weights or using a GPU.
#include "myds4.h"
enum { SCORE = 102, SELECT = 101 };  // stand-ins for the kernels' numbers
static int dispatch_kernel, observed, masked;
static size_t candidate_offset[2];  // as the scoring and the selection receive them
static void spy_kernel(int k) { dispatch_kernel = k; }
static void spy_buffer(int index, const Buf *b, size_t offset) {
  if (dispatch_kernel == SCORE && index == 6) candidate_offset[0] = offset, observed++;
  if (dispatch_kernel == SELECT && index == 3) candidate_offset[1] = offset, observed++;
}
static void spy_bytes(int index, const void *data, size_t size) {
  if (dispatch_kernel == SCORE && index == 5) masked = ((const uint32_t *)data)[3] != 0;  // Index.cand
}
static void spy_go(int x, int y, int z, int tx, int ty, int tz) {}
#define gpu_kern spy_kernel
#define gpu_buf spy_buffer
#define gpu_bytes spy_bytes
#define gpu_go spy_go
#include "../../src/model/deepseek/model.c"

int main(void) {
  opt.ctx = 32768, cand_words = 129, sc_rows = 256;
  L[24].ratio = L[24].index_source = 1, L[24].src = 20;
  K_index_score = SCORE, K_index_topk = SELECT;
  int failed = 0;
  for (int row = 4; row <= MB + 4; row += MB) {
    observed = masked = 0;
    attend_rows(24, 133, 16384 + row, 0, row);
    size_t expected = (size_t)row * cand_words * 4;
    int ok = observed == 2 && masked && candidate_offset[0] == expected && candidate_offset[1] == expected;
    printf("candidate prompt row %d: scoring offset=%zu selection offset=%zu expected=%zu %s\n", row,
           candidate_offset[0], candidate_offset[1], expected, ok ? "PASS" : "FAIL");
    failed += !ok;
  }
  return failed != 0;
}
