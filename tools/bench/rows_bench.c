// Decoding B sequences together (model_forward_rows), for the decode bound:
// per step, the wall time, the GPU's busy and idle time, and per decoded row
// the experts read on demand, the expert bytes read and the slots mapped for
// the GPU again. Each sequence has its own prompt (an opening line, then a
// common text: a short request, or with --text the first --prompt-tokens
// tokens of FILE), prefilled alone; then the B sequences decode greedily in
// lockstep, WARM steps before the timed STEPS. One B per process, so that each
// starts from a cold cache.
//
// usage: rows-bench CHECKPOINT ROWS [--experts GGUF] [--cache GIB] [--steps N]
//                   [--text FILE --prompt-tokens N]
#include "../../src/engine/model.c"

int main(int argc, char **argv) {
  if (argc < 3) die("usage: rows-bench CHECKPOINT ROWS [--experts GGUF] [--cache GIB] [--steps N] [--text FILE --prompt-tokens N]");
  int rows = atoi(argv[2]), steps = 96, warm = 32, length = 0;
  const char *file = NULL;
  Options o = {.ctx = 4096, .budget_gb = 40, .reserve_gb = 10, .quiet = 1, .dump_fd = -1, .routes_fd = -1};
  for (int i = 3; i < argc; i++) {
    if (!strcmp(argv[i], "--experts") && i + 1 < argc) o.experts = argv[++i];
    else if (!strcmp(argv[i], "--cache") && i + 1 < argc) o.budget_gb = atof(argv[++i]);
    else if (!strcmp(argv[i], "--steps") && i + 1 < argc) steps = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--text") && i + 1 < argc) file = argv[++i];
    else if (!strcmp(argv[i], "--prompt-tokens") && i + 1 < argc) length = atoi(argv[++i]);
    else die("invalid rows-bench option %s", argv[i]);
  }
  if (rows < 1 || rows > FEW || steps < 1) die("rows-bench takes 1 to %d rows and at least one step", FEW);
  if (!file != !length || length < 0) die("--text and --prompt-tokens go together");
  o.ctx = length + 4096;
  tok_load(argv[1]);
  model_load(argv[1], &o);
  static const char *openings[] = {"A lighthouse keeper finds a bottle.", "A baker opens a shop at dawn.",
                                   "Two engineers debate a cache design.", "A sailor maps an unknown coast.",
                                   "A child learns to play the violin.", "A detective reads an old letter."};
  const char *body = " Write the next chapter of this story in vivid detail, with dialogue, and keep going.";
  int *common = xmalloc((size_t)(length + 256) * sizeof *common), ncommon = 0;
  if (file) {
    FILE *f = fopen(file, "rb");
    if (!f) die("cannot open %s", file);
    char *text = xmalloc(16u << 20);
    size_t got = fread(text, 1, (16u << 20) - 1, f);
    fclose(f);
    text[got] = 0;
    int *all = xmalloc((16u << 20) * sizeof *all);
    ncommon = tok_encode(text, all, 16 << 20);
    if (ncommon < length) die("%s has %d tokens, fewer than %d", file, ncommon, length);
    memcpy(common, all, (size_t)length * sizeof *common), ncommon = length;
    free(all), free(text);
  } else {
    ncommon = tok_encode(body, common, 256);
  }
  State *st[FEW];
  int tok[FEW], pos[FEW];
  int *prompt = xmalloc((size_t)(ncommon + 64) * sizeof *prompt);
  for (int r = 0; r < rows; r++) {
    int n = tok_encode(openings[r], prompt, 64);
    memcpy(prompt + n, common, (size_t)ncommon * sizeof *prompt), n += ncommon;
    st[r] = model_state_new();
    model_state_use(st[r]);
    tok[r] = sample_argmax(model_forward(prompt, n, 0), VOCAB);
    pos[r] = n;
  }
  CacheStats c0 = cache_stats;
  double t0 = now(), g0 = gpu_busy, i0 = gpu_gap;
  for (int s = 0; s < warm + steps; s++) {
    if (s == warm) c0 = cache_stats, t0 = now(), g0 = gpu_busy, i0 = gpu_gap, gpu_profile_reset();  // MYDS4_PROFILE=1 covers the timed passes
    const float *logits = model_forward_rows(st, tok, pos, rows);
    for (int r = 0; r < rows; r++) tok[r] = sample_argmax(logits + (size_t)r * VOCAB, VOCAB), pos[r]++;
  }
  double wall = now() - t0, n = (double)steps * rows;
  printf("{\"rows_bench\":{\"rows\":%d,\"context\":%d,\"steps\":%d,\"step_ms\":%.2f,\"tokens_per_s\":%.2f,\"gpu_busy_ms_per_step\":%.2f,"
         "\"gpu_idle_ms_per_step\":%.2f,\"demand_reads_per_row\":%.2f,\"prefetch_hits_per_row\":%.2f,\"read_mb_per_row\":%.1f,"
         "\"remaps_per_row\":%.2f,\"slots\":%d,\"mapped\":%d}}\n",
         rows, pos[0], steps, wall / steps * 1e3, n / wall, (gpu_busy - g0) / steps * 1e3, (gpu_gap - i0) / steps * 1e3,
         (cache_stats.misses - c0.misses) / n, (cache_stats.prefetch_hits - c0.prefetch_hits) / n,
         (cache_stats.bytes - c0.bytes) / n / 1e6, (cache_stats.remaps - c0.remaps) / n, cache_cap(), cache_mapped());
  return 0;
}
