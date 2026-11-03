// Joint prefill, prefix restore and CED against full-shape model execution.
// Keeps tensors in RAM and prints counts only; no prompt or tensor files.
#define MYDS4_CED_CHECK
#include "../src/model.c"

static void ced_check_routes(int layer, int row, int rows, const int *routes) {}

static size_t different(const void *a, const void *b, size_t n) {
  const unsigned char *x = a, *y = b;
  size_t bad = 0;
  for (size_t i = 0; i < n; i++) bad += x[i] != y[i];
  return bad;
}

static size_t encoder_bytes(void) {
  size_t n = 0;
  for (int l = 0; l <= 20; l++) n += L[l].ring.size + (L[l].kv_source ? L[l].cstate.size : 0);
  return n;
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
  char *prefix_state = xmalloc(saved_bytes());
  size_t errors = 0;
  const int boundaries[] = {127, 128, 257, 512};
  {
    memcpy(ref, model_forward(tokens, n, 0), VOCAB * 4);
    model_state_save(0);
    memcpy(state, active->saved[0], saved_bytes());
    int cases = longer ? 1 : sizeof boundaries / sizeof *boundaries;
    for (int k = 0; k < cases; k++) {
      int prefix = longer ? CHUNK + 8 : boundaries[k], encoder_only = -1;
      model_reset();
      const float *lg = model_forward_prefix(tokens, n, 0, prefix, &encoder_only);
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
      printf("prefix=%d draft=%d: logits=%zu final_state=%zu saved_prefix=%zu resumed_logits=%zu encoder_only=%d\n",
             prefix, o.draft, output, final, saved, resumed, encoder_only);
      fflush(stdout);
      errors += output + final + saved + resumed + (encoder_only != 0);
    }
  }
  for (int keep = 128; keep <= 512; keep *= 4) {
    int prefix = longer ? CHUNK + 1 : 257, encoder_only = -1;
    // Reference: retain every decoder row, masking only the same local keys.
    opt.ced = 0;
    ced_check_first = n - (keep + BLOCK), ced_check_compact = 0;
    model_reset();
    memcpy(ref, model_forward_prefix(tokens, n, 0, prefix, NULL), VOCAB * 4);
    memcpy(prefix_state, active->saved[1], saved_bytes());
    model_state_save(0);
    memcpy(state, active->saved[0], saved_bytes());
    ced_check_first = ced_check_compact = 0;
    opt.ced = keep;
    model_reset();
    const float *lg = model_forward_prefix(tokens, n, 0, prefix, &encoder_only);
    ModelForwardInfo info = model_forward_info();
    size_t output = different(ref, lg, VOCAB * 4);
    model_state_save(0);
    size_t final = different(state, active->saved[0], saved_bytes());
    // Compare identical matrix shapes. A one-token final chunk uses scalar
    // kernels and has separate rounding from a boundary inside a batch.
    size_t saved = different(prefix_state, active->saved[1], encoder_bytes());
    opt.ced = 0;
    if (!longer) {
      model_reset();
      model_forward(tokens, prefix, 0);
      model_state_save(0);
      saved += different(active->saved[0], active->saved[1], encoder_bytes());
    }
    printf("ced=%d n=%d: logits=%zu final_state=%zu saved_encoder=%zu decoder_tokens=%d chunks=%d encoder_only=%d\n",
           keep, n, output, final, saved, info.decoder_tokens, info.chunks, encoder_only);
    fflush(stdout);
    errors += output + final + saved + (encoder_only != 1) + (info.decoder_tokens != keep + BLOCK);
    if (n - prefix > keep + BLOCK) {
      opt.ced = keep;
      model_state_restore(1);
      model_seek(tokens, prefix);
      lg = model_forward(tokens + prefix, n - prefix, prefix);
      size_t resumed = different(ref, lg, VOCAB * 4);
      printf("ced=%d prefix=%d: resumed_logits=%zu\n", keep, prefix, resumed);
      fflush(stdout);
      errors += resumed;
    }
  }
  printf("prefill-prefix-check: differing_bytes_and_failed_conditions=%zu\n", errors);
  free(ref), free(state), free(prefix_state), free(tokens);
  return errors != 0;
}
