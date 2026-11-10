// The CPU cost of binding a DeepSeek pass's buffers into the model's globals
// (pipe_bind): what the pass scheduler pays between two passes' stages. Needs
// neither a model nor the GPU; the buffers bound are empty.
#include "../src/model/deepseek/model.h"
#include "../src/model/deepseek/model.c"

int main(void) {
  static PipeJob jobs[2];
  static State states[2];
  for (int l = 0; l < LAYERS; l++) L[l].kv_source = is_in(l, kv_sources, sizeof kv_sources / sizeof *kv_sources);
  jobs[0].state = &states[0], jobs[1].state = &states[1];
  Buf *live[PIPE_BUFFERS] = {PIPE_BUFFER_LIST};
  enum { TIMES = 10000000 };
  double t0 = now();
  for (int i = 0; i < TIMES; i++) {
    pipe_bind(&jobs[i & 1], live);
    __asm__ volatile("" : : : "memory");
  }
  double seconds = now() - t0;
  printf("{\"pipe_bind_ns\":%.1f,\"buffers\":%d,\"layers\":%d}\n", seconds / TIMES * 1e9, (int)PIPE_BUFFERS, (int)LAYERS);
  return 0;
}
