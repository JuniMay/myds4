// Joint prefill and prefix restore against full-shape model execution.
// Keeps tensors in RAM and prints counts only; no prompt or tensor files.
#include "../src/model.c"

static size_t different(const void *a, const void *b, size_t n) {
  const unsigned char *x = a, *y = b;
  size_t bad = 0;
  for (size_t i = 0; i < n; i++) bad += x[i] != y[i];
  return bad;
}

int main(int argc, char **argv) {
  if (argc < 2) die("usage: prefill-prefix-check CHECKPOINT [--experts GGUF] [--long] [--no-draft]");
  Options o = {.ctx = 32768, .budget_gb = 20, .reserve_gb = 4, .draft = 2, .draft_cache = 32,
               .draft_trim = 1, .dump_fd = -1, .routes_fd = -1, .quiet = 1};
  int longer = 0;
  for (int i = 2; i < argc; i++) {
    if (!strcmp(argv[i], "--long")) longer = 1;
    else if (!strcmp(argv[i], "--no-draft")) o.draft = o.draft_cache = o.draft_trim = 0;
    else if (!strcmp(argv[i], "--experts") && i + 1 < argc) o.experts = argv[++i];
    else die("invalid check argument");
  }
  alarm(600);
  tok_load(argv[1]);
  int pattern[128], n = longer ? CHUNK + 137 : 640;
  int pn = tok_encode(" Inspect the scheduler and preserve cancellation. alpha beta gamma delta 12345.\n", pattern, 128);
  int *tokens = xmalloc((size_t)n * sizeof *tokens);
  for (int i = 0; i < n; i++) tokens[i] = pattern[i % pn];
  model_load(argv[1], &o);
  float *ref = xmalloc(VOCAB * 4);
  char *state = xmalloc(saved_bytes());
  size_t errors = 0;
  const int boundaries[] = {127, 128, 257, 512};
  {
    memcpy(ref, model_forward(tokens, n, 0), VOCAB * 4);
    model_state_save(0);
    memcpy(state, active->saved[0], saved_bytes());
    int cases = longer ? 1 : sizeof boundaries / sizeof *boundaries;
    for (int k = 0; k < cases; k++) {
      int prefix = longer ? CHUNK + 8 : boundaries[k];
      model_reset();
      const float *lg = model_forward_prefix(tokens, n, 0, prefix);
      size_t output = different(ref, lg, VOCAB * 4);
      model_state_save(0);
      size_t final = different(state, active->saved[0], saved_bytes());
      model_reset();
      model_forward(tokens, prefix, 0);
      model_state_save(0);
      size_t saved = different(active->saved[0], active->saved[1], saved_bytes());
      model_state_restore(1);
      model_seek(tokens, prefix);
      lg = model_forward(tokens + prefix, n - prefix, prefix);
      size_t resumed = different(ref, lg, VOCAB * 4);
      printf("prefix=%d draft=%d: logits=%zu final_state=%zu saved_prefix=%zu resumed_logits=%zu\n",
             prefix, o.draft, output, final, saved, resumed);
      fflush(stdout);
      errors += output + final + saved + resumed;
    }
  }
  printf("prefill-prefix-check: differing_bytes_and_failed_conditions=%zu\n", errors);
  free(ref), free(state), free(tokens);
  return errors != 0;
}
