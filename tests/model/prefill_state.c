// Prefill must retain the window plus the server's allowed rewind margin.
#include "../../src/model/deepseek/model.h"
#include "../../src/model/deepseek/model.c"
static Tokenizer *tk;  // the checkpoint's tokenizer

int main(int argc, char **argv) {
  if (argc < 2 || argc > 3) die("usage: prefill-state-check CHECKPOINT [EXPERT_GGUF]");
  alarm(180);
  tk = tok_open(argv[1], deepseek_ops.media_token);
  int pattern[64], tokens[256];
  int pn = tok_encode(tk, " alpha beta gamma delta epsilon", pattern, 64);
  if (!pn) die("empty pattern");
  for (int i = 0; i < 256; i++) tokens[i] = pattern[i % pn];
  double model_memory_gib = 20;
  double model_reserve_gib = 4;
  Options o = {.ctx = 2048, .deepseek.experts = argc == 3 ? argv[2] : NULL, .dump_fd = -1, .routes_fd = -1, .quiet = 1};
  memory_configure(model_memory_gib, 8, model_reserve_gib);
  deepseek_load(argv[1], &o);
  deepseek_forward(tokens, 256, 0);
  const uint16_t *all = (const uint16_t *)WIN.ptr;
  const uint16_t *ring = (const uint16_t *)L[LAYERS - 1].ring.ptr;
  uint64_t bad_window = 0, bad_margin = 0;
  for (int p = 256 - RING; p < 256; p++)
    for (int d = 0; d < HEAD_DIM; d++) {
      int different = ring[(p % RING) * HEAD_DIM + d] != all[p * HEAD_DIM + d];
      if (p < 256 - WINDOW) bad_margin += different;
      else bad_window += different;
    }
  printf("{\"prefill_state_check\":{\"window_rows\":%d,\"rewind_rows\":%d,"
         "\"window_differences\":%llu,\"rewind_differences\":%llu}}\n",
         WINDOW, BLOCK, (unsigned long long)bad_window, (unsigned long long)bad_margin);
  return bad_window || bad_margin;
}
