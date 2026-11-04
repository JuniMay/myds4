// Compare the first DSpark draft distribution with its target distribution.
// Reads a rendered text prompt on stdin. Reports probabilities, not throughput.
#include "../../src/engine/model.c"

enum { PROBE_TOKENS = 256, TEMPERATURES = 5 };
static const double draft_temperature[TEMPERATURES] = {.5, .75, 1, 1.5, 2};

int main(int argc, char **argv) {
  if (argc < 2 || argc > 4) die("usage: draft-distribution-probe CHECKPOINT [CACHE_GIB [EXPERT_GGUF]]");
  alarm(240);
  double cache = argc > 2 ? strtod(argv[2], NULL) : 56;
  if (!(cache >= 20 && cache <= 64)) die("probe cache must be 20 to 64 GiB");
  Options o = {.ctx = 32768, .budget_gb = cache, .reserve_gb = 4, .draft = 2,
               .draft_cache = 32, .draft_trim = 1, .expert_batch = 1,
               .experts = argc > 3 ? argv[3] : NULL, .dump_fd = -1, .routes_fd = -1, .quiet = 1};
  size_t limit = 512u << 10;
  char *text = xmalloc(limit + 1);
  size_t bytes = fread(text, 1, limit, stdin);
  if (!bytes || bytes == limit || ferror(stdin)) die("probe prompt is empty or too large");
  text[bytes] = 0;
  tok_load(argv[1]);
  int *tokens = xmalloc((size_t)o.ctx * sizeof *tokens);
  int n = tok_encode_chat(text, tokens, o.ctx);
  if (n < 1 || n + PROBE_TOKENS > o.ctx) die("probe prompt does not fit the context");
  free(text);
  model_load(argv[1], &o);
  Sampler sampler = {.temp = 1, .top_p = 1};
  sampler_seed(&sampler, 12345);
  int token = sample_token(model_forward(tokens, n, 0), VOCAB, &sampler), eos = tok_find("<｜end▁of▁sentence｜>");
  double *p = xmalloc(VOCAB * sizeof *p), *q = xmalloc(VOCAB * sizeof *q);
  double maximal[TEMPERATURES] = {0}, shared[TEMPERATURES] = {0}, greedy = 0;
  int passes = 0, first_accepted = 0, total_accepted = 0, gen = 0;
  while (gen < PROBE_TOKENS && token != eos && !shutdown_requested()) {
    int rows[3] = {token}, k = PROBE_TOKENS - gen - 1;
    if (k > 2) k = 2;
    if (!k) break;
    model_draft(token, n + gen, rows + 1, k);
    const float *target = model_verify(rows, k + 1, n + gen);
    const float *draft = (const float *)DLOG.ptr;
    int pb = sample_argmax(target, VOCAB), qb = sample_argmax(draft, VOCAB);
    double psum = 0;
    for (int i = 0; i < VOCAB; i++) psum += p[i] = exp((double)target[i] - target[pb]);
    for (int i = 0; i < VOCAB; i++) p[i] /= psum;
    greedy += p[qb];
    for (int t = 0; t < TEMPERATURES; t++) {
      double qsum = 0;
      for (int i = 0; i < VOCAB; i++) qsum += q[i] = exp(((double)draft[i] - draft[qb]) / draft_temperature[t]);
      double px = 0, qx = 0, a = 0, b = 0;
      for (int i = 0; i < VOCAB; i++) {
        double qv = q[i] / qsum;
        a += fmin(p[i], qv);
        b += fmax(0, fmin(px + p[i], qx + qv) - fmax(px, qx));
        px += p[i], qx += qv;
      }
      if (!isfinite(a) || !isfinite(b) || b > a + 1e-8 || a > 1 + 1e-8) die("invalid probability overlap");
      maximal[t] += a, shared[t] += b;
    }
    int accepted;
    token = sample_verify(target, VOCAB, rows + 1, k, eos, &sampler, &accepted);
    model_accept(rows, accepted + 1, n + gen);
    gen += accepted + 1, passes++, first_accepted += accepted > 0, total_accepted += accepted;
  }
  printf("{\"draft_distribution\":{\"prompt_tokens\":%d,\"generated_tokens\":%d,\"passes\":%d,"
         "\"observed_first_accepted\":%d,\"observed_total_accepted\":%d,\"greedy_first_probability\":%.9f,"
         "\"target_temperature\":1,\"top_p\":1,\"rows\":[", n, gen, passes, first_accepted, total_accepted,
         passes ? greedy / passes : 0);
  for (int t = 0; t < TEMPERATURES; t++)
    printf("%s{\"draft_temperature\":%.2f,\"maximal_first_acceptance\":%.9f,\"shared_uniform_first_acceptance\":%.9f}",
           t ? "," : "", draft_temperature[t], passes ? maximal[t] / passes : 0, passes ? shared[t] / passes : 0);
  printf("],\"scope\":\"first-draft distributions on the greedy-draft trajectory; not multi-token acceptance or throughput\"}}\n");
  free(tokens), free(p), free(q);
  return !passes;
}
