// Prefill must retain the window plus the server's allowed rewind margin.
#include "../../src/engine/model.c"

int main(int argc, char **argv) {
  if (argc < 2 || argc > 3) die("usage: prefill-state-check CHECKPOINT [EXPERT_GGUF]");
  alarm(180);
  tok_load(argv[1]);
  int pattern[64], tokens[256];
  int pn = tok_encode(" alpha beta gamma delta epsilon", pattern, 64);
  if (!pn) die("empty pattern");
  for (int i = 0; i < 256; i++) tokens[i] = pattern[i % pn];
  Options o = {.ctx = 2048, .budget_gb = 20, .reserve_gb = 4,
               .experts = argc == 3 ? argv[2] : NULL, .dump_fd = -1, .routes_fd = -1, .quiet = 1};
  model_load(argv[1], &o);
  model_forward(tokens, 256, 0);
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
