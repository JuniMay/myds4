// Decoding B sequences together (model_forward_rows), for the decode bound:
// per step, the wall time, the GPU's busy and idle time, and per decoded row
// the experts read on demand, the expert bytes read and the slots mapped for
// the GPU again. Each sequence has its own prompt, prefilled alone: an
// opening line and a short request, or with --text sequence r's own part of
// FILE, its r-th run of --prompt-tokens tokens. Then the B sequences decode greedily in
// lockstep, --warm steps (default 32) before the timed --steps. One B per
// process, so that each starts from a cold cache. Greedy decoding can fall
// into a loop that repeats a few tokens, and a loop uses few experts and
// runs fast: distinct_8grams is the share of the first sequence's 8-token
// windows that had not occurred before in its generation (1: no repeats),
// and --show prints its text.
//
// usage: rows-bench CHECKPOINT ROWS [--experts GGUF] [--cache GIB] [--steps N]
//                   [--warm N] [--text FILE --prompt-tokens N] [--show]
#include "../../src/engine/model.c"

// The share of gen's 8-token windows that occurred nowhere before them.
static double distinct_8grams(const int *gen, int n) {
  enum { G = 8 };
  if (n < G) return 1;
  int windows = n - G + 1, fresh = 0;
  for (int i = 0; i < windows; i++) {
    int seen = 0;
    for (int j = 0; j < i && !seen; j++) seen = !memcmp(gen + i, gen + j, G * sizeof *gen);
    fresh += !seen;
  }
  return (double)fresh / windows;
}

int main(int argc, char **argv) {
  if (argc < 3) die("usage: rows-bench CHECKPOINT ROWS [--experts GGUF] [--cache GIB] [--steps N] [--warm N] [--text FILE --prompt-tokens N] [--show]");
  int rows = atoi(argv[2]), steps = 96, warm = 32, length = 0, show = 0;
  const char *file = NULL;
  Options o = {.ctx = 4096, .budget_gb = 40, .reserve_gb = 10, .quiet = 1, .dump_fd = -1, .routes_fd = -1};
  for (int i = 3; i < argc; i++) {
    if (!strcmp(argv[i], "--experts") && i + 1 < argc) o.experts = argv[++i];
    else if (!strcmp(argv[i], "--cache") && i + 1 < argc) o.budget_gb = atof(argv[++i]);
    else if (!strcmp(argv[i], "--steps") && i + 1 < argc) steps = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--warm") && i + 1 < argc) warm = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--text") && i + 1 < argc) file = argv[++i];
    else if (!strcmp(argv[i], "--prompt-tokens") && i + 1 < argc) length = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--show")) show = 1;
    else die("invalid rows-bench option %s", argv[i]);
  }
  if (rows < 1 || rows > BATCH_ROWS || steps < 1 || warm < 0)
    die("rows-bench takes 1 to %d rows, at least one step and no negative warm-up", BATCH_ROWS);
  if (!file != !length || length < 0) die("--text and --prompt-tokens go together");
  o.ctx = length + 4096;
  tok_load(argv[1]);
  model_load(argv[1], &o);
  static const char *openings[] = {"A lighthouse keeper finds a bottle.", "A baker opens a shop at dawn.",
                                   "Two engineers debate a cache design.", "A sailor maps an unknown coast.",
                                   "A child learns to play the violin.", "A detective reads an old letter."};
  const char *body = " Write the next chapter of this story in vivid detail, with dialogue, and keep going.";
  int *common = xmalloc(256 * sizeof *common), ncommon = 0, *all = NULL;
  if (file) {
    FILE *f = fopen(file, "rb");
    if (!f) die("cannot open %s", file);
    char *text = xmalloc(16u << 20);
    size_t got = fread(text, 1, (16u << 20) - 1, f);
    fclose(f);
    text[got] = 0;
    all = xmalloc((16u << 20) * sizeof *all);
    int count = tok_encode(text, all, 16 << 20);
    if (count < rows * length) die("%s has %d tokens, fewer than %d rows of %d", file, count, rows, length);
    free(text);
  } else {
    ncommon = tok_encode(body, common, 256);
  }
  State *st[FEW];
  int tok[FEW], pos[FEW];
  int *prompt = xmalloc((size_t)(length + ncommon + 64) * sizeof *prompt);
  for (int r = 0; r < rows; r++) {
    int n;
    if (all) {  // the r-th run of length tokens: each sequence its own part of the text
      memcpy(prompt, all + (size_t)r * length, (size_t)length * sizeof *prompt), n = length;
    } else {
      n = tok_encode(openings[r], prompt, 64);
      memcpy(prompt + n, common, (size_t)ncommon * sizeof *prompt), n += ncommon;
    }
    st[r] = model_state_new();
    model_state_use(st[r]);
    tok[r] = sample_argmax(model_forward(prompt, n, 0), VOCAB);
    pos[r] = n;
  }
  int *gen = xmalloc((size_t)(warm + steps + 1) * sizeof *gen), ng = 0;  // the first sequence's tokens
  gen[ng++] = tok[0];
  CacheStats c0 = cache_stats;
  double t0 = now(), g0 = gpu_busy, i0 = gpu_gap, ph0[7];
  for (int s = 0; s < warm + steps; s++) {
    if (s == warm) {  // MYDS4_PROFILE=1 covers the timed passes
      c0 = cache_stats, t0 = now(), g0 = gpu_busy, i0 = gpu_gap, gpu_profile_reset();
      memcpy(ph0, phase_s, sizeof ph0);
    }
    const float *logits = model_forward_rows(st, tok, pos, rows);
    for (int r = 0; r < rows; r++) tok[r] = sample_argmax(logits + (size_t)r * VOCAB, VOCAB), pos[r]++;
    gen[ng++] = tok[0];
  }
  double wall = now() - t0, n = (double)steps * rows;
  printf("{\"rows_bench\":{\"rows\":%d,\"context\":%d,\"steps\":%d,\"step_ms\":%.2f,\"tokens_per_s\":%.2f,\"gpu_busy_ms_per_step\":%.2f,"
         "\"gpu_idle_ms_per_step\":%.2f,\"demand_reads_per_row\":%.2f,\"prefetch_hits_per_row\":%.2f,\"read_mb_per_row\":%.1f,"
         "\"remaps_per_row\":%.2f,\"slots\":%d,\"mapped\":%d,\"distinct_8grams\":%.3f}}\n",
         rows, pos[0], steps, wall / steps * 1e3, n / wall, (gpu_busy - g0) / steps * 1e3, (gpu_gap - i0) / steps * 1e3,
         (cache_stats.misses - c0.misses) / n, (cache_stats.prefetch_hits - c0.prefetch_hits) / n,
         (cache_stats.bytes - c0.bytes) / n / 1e6, (cache_stats.remaps - c0.remaps) / n, cache_cap(), cache_mapped(),
         distinct_8grams(gen, ng));
  // the CPU's time per pass in decoding's phases (model.c's phase_s)
  static const char *phases[] = {"rows", "routing_waits", "expert_loads", "requests_and_prefetch", "encoding", "adapt", "drafting"};
  printf("{\"rows_bench_phases_ms_per_step\":{");
  for (int k = 0; k < 7; k++) printf("%s\"%s\":%.2f", k ? "," : "", phases[k], (phase_s[k] - ph0[k]) / steps * 1e3);
  printf("}}\n");
  if (show) {
    for (int i = 0; i < ng; i++) {
      int len;
      const char *piece = tok_piece(gen[i], &len);
      fwrite(piece, 1, (size_t)len, stderr);
    }
    fputc('\n', stderr);
  }
  return 0;
}
