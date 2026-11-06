// CED numerical comparisons and prefill/DSpark timing checks.
// Including the model lets this diagnostic inspect caches without a public API.
#define MYDS4_CED_CHECK 1
#include "../../src/engine/model.c"

typedef struct { uint64_t count, different; double signal, error, max_abs; } Error;
static int prompt_n, observed[LAYERS][WINDOW][TOPK], reference_routes[LAYERS][WINDOW][TOPK];
static uint16_t reference_ring[LAYERS][(WINDOW + VERIFY_DRAFTS) * HEAD_DIM];
static uint16_t *reference_global[LAYERS][2];

static void benchmark(int *tokens, int generate, int keep) {
  ced_check_first = keep > 0 && prompt_n > keep ? prompt_n - keep : 0;
  CacheStats c0 = cache_stats;
  double g0 = gpu_busy, t0 = now();
  const float *lg = model_forward(tokens, prompt_n, 0);
  unsigned char prefill_digest[CC_SHA256_DIGEST_LENGTH];
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
  CC_SHA256(lg, VOCAB * 4, prefill_digest);
#pragma clang diagnostic pop
  double t1 = now(), g1 = gpu_busy;
  ModelForwardInfo plan = model_forward_info();
  CacheStats c1 = cache_stats;
  uint64_t rows0 = decode_rows, slots0 = cache_bytes(), minimum = slots0, maximum = slots0;
  Sampler sampler = {.temp = 1, .top_p = 1};
  sampler_seed(&sampler, 12345);
  int current = sample_token(lg, VOCAB, &sampler), eos = tok_find("<｜end▁of▁sentence｜>");
  int gen = 0, pos = prompt_n, proposals = 0, accepted = 0, passes = 0;
  while (gen < generate && current != eos && !shutdown_requested()) {
    int rows[BLOCK + 1] = {current}, k = opt.draft, a = 0, next;
    if (k > generate - gen - 1) k = generate - gen - 1;
    if (k > 0) {
      model_draft(current, pos, rows + 1, k);
      lg = model_verify(rows, k + 1, pos);
      next = sample_verify(lg, VOCAB, rows + 1, k, eos, &sampler, &a);
      model_accept(rows, a + 1, pos);
      passes++, proposals += k, accepted += a;
    } else {
      lg = model_forward(&current, 1, pos);
      next = sample_token(lg, VOCAB, &sampler);
    }
    for (int r = 0; r <= a; r++) tokens[pos + r] = rows[r];
    gen += a + 1, pos += a + 1, current = next;
    uint64_t bytes = cache_bytes();
    if (bytes < minimum) minimum = bytes;
    if (bytes > maximum) maximum = bytes;
  }
  double t2 = now(), g2 = gpu_busy;
  CacheStats c2 = cache_stats;
  unsigned char digest[CC_SHA256_DIGEST_LENGTH];
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
  CC_SHA256(tokens + prompt_n, gen * sizeof(int), digest);
#pragma clang diagnostic pop
  printf("{\"inference_bench\":{\"ced_keep\":%d,\"compact\":%s,\"draft_depth\":%d,\"prompt_tokens\":%d,"
         "\"generated\":%d,\"prefill_seconds\":%.9g,\"prefill_gpu_seconds\":%.9g,\"prefill_read_bytes\":%llu,"
         "\"prefill_read_wait_seconds\":%.9g,\"decode_seconds\":%.9g,\"decode_tokens_per_second\":%.9g,"
         "\"decode_gpu_seconds\":%.9g,\"decode_read_bytes\":%llu,\"decode_read_wait_seconds\":%.9g,"
         "\"computed_rows\":%llu,\"draft_passes\":%d,\"proposed_tokens\":%d,\"accepted_tokens\":%d,"
         "\"cache_bytes_start\":%llu,\"cache_bytes_min\":%llu,\"cache_bytes_max\":%llu,"
         "\"slots_released\":%llu,\"output_token_sha256\":\"",
         opt.ced ? opt.ced : keep, plan.ced_applied ? "true" : "false", opt.draft, prompt_n,
         gen, t1 - t0, g1 - g0, (unsigned long long)(c1.bytes - c0.bytes), c1.read_s - c0.read_s,
         t2 - t1, gen / fmax(t2 - t1, 1e-9), g2 - g1, (unsigned long long)(c2.bytes - c1.bytes),
         c2.read_s - c1.read_s, (unsigned long long)(decode_rows - rows0), passes, proposals, accepted,
         (unsigned long long)slots0, (unsigned long long)minimum, (unsigned long long)maximum,
         (unsigned long long)(c2.given_back - c1.given_back));
  for (size_t i = 0; i < sizeof digest; i++) printf("%02x", digest[i]);
  printf("\",\"configured_ced_depth\":%d,"
         "\"last_forward_chunks\":%d,\"last_forward_decoder_tokens\":%d,\"prefill_logits_sha256\":\"",
         opt.ced, plan.chunks, plan.decoder_tokens);
  for (size_t i = 0; i < sizeof prefill_digest; i++) printf("%02x", prefill_digest[i]);
  printf("\",\"proposal_policy\":\"%s\",\"decode_remaps\":%llu}}\n",
         opt.draft ? "dspark" : "none", (unsigned long long)(c2.remaps - c1.remaps));
  fflush(stdout);
  cache_metrics_report(stderr);
}

static void ced_check_routes(int layer, int row, int rows, const int *routes) {
  int first = prompt_n > WINDOW ? prompt_n - WINDOW : 0;
  for (int r = 0; r < rows; r++)
    if (row + r >= first && row + r < prompt_n)
      memcpy(observed[layer][row + r - first], routes + r * 16, TOPK * sizeof(int));
}

static float unpack_bf16(uint16_t x) {
  return (union { uint32_t u; float f; }){.u = (uint32_t)x << 16}.f;
}

static void add_error(Error *s, float a, float b, int different) {
  if (!isfinite(a) || !isfinite(b)) die("nonfinite numerical comparison");
  double d = (double)b - a;
  s->count++, s->different += different;
  s->signal += (double)a * a, s->error += d * d;
  if (fabs(d) > s->max_abs) s->max_abs = fabs(d);
}

static void print_error(const Error *s) {
  printf("{\"values\":%llu,\"different_values\":%llu,\"relative_rms\":%.9g,\"max_abs\":%.9g,\"snr_db\":",
         (unsigned long long)s->count, (unsigned long long)s->different,
         s->signal ? sqrt(s->error / s->signal) : 0, s->max_abs);
  if (s->error > 0 && s->signal > 0) printf("%.9g", 10 * log10(s->signal / s->error));
  else fputs("null", stdout); // Zero error has infinite SNR.
  putchar('}');
}

static size_t global_values(int l, int keys) {
  return (size_t)(prompt_n / L[l].ratio) * (keys ? IDX_DIM : HEAD_DIM);
}

static void snapshot(void) {
  gpu_sync();
  memcpy(reference_routes, observed, sizeof observed);
  for (int l = 0; l < LAYERS; l++) {
    memcpy(reference_ring[l], L[l].ring.ptr, L[l].ring.size);
    if (!L[l].kv_source) continue;
    for (int k = 0; k < 2; k++) {
      size_t bytes = global_values(l, k) * sizeof(uint16_t);
      reference_global[l][k] = xmalloc(bytes);
      memcpy(reference_global[l][k], (k ? L[l].keys : L[l].comp).ptr, bytes);
    }
  }
}

static uint64_t report_cache(int keep) {
  gpu_sync();
  uint64_t all_different = 0, global_different = 0, route_changed = 0, route_total = 0;
  Error encoder = {0}, decoder = {0};
  printf("{\"ced_cache\":{\"keep_tokens\":%d,\"first_decoder_layer\":%d,\"layers\":[", keep, ced_check_layer);
  for (int l = 0; l < LAYERS; l++) {
    Error local = {0};
    const uint16_t *actual = (const uint16_t *)L[l].ring.ptr;
    int first = prompt_n > WINDOW ? prompt_n - WINDOW : 0;
    for (int p = first; p < prompt_n; p++)
      for (int d = 0; d < HEAD_DIM; d++) {
        int i = (p % RING) * HEAD_DIM + d;
        uint16_t a = reference_ring[l][i], b = actual[i];
        add_error(&local, unpack_bf16(a), unpack_bf16(b), a != b);
        add_error(l < ced_check_layer ? &encoder : &decoder, unpack_bf16(a), unpack_bf16(b), a != b);
      }
    printf("%s{\"layer\":%d,\"local_kv\":", l ? "," : "", l);
    print_error(&local);
    putchar('}');
    all_different += local.different;
    if (L[l].kv_source)
      for (int k = 0; k < 2; k++) {
        const uint16_t *current = (const uint16_t *)(k ? L[l].keys : L[l].comp).ptr;
        for (size_t i = 0; i < global_values(l, k); i++) global_different += reference_global[l][k][i] != current[i];
      }
    if (l >= ced_check_layer)
      for (int r = 0; r < prompt_n - first; r++)
        for (int i = 0; i < TOPK; i++) {
          int found = 0;
          for (int j = 0; j < TOPK; j++) found |= reference_routes[l][r][i] == observed[l][r][j];
          route_changed += !found, route_total++;
        }
  }
  fputs("],\"encoder_local_kv\":", stdout), print_error(&encoder);
  fputs(",\"decoder_local_kv\":", stdout), print_error(&decoder);
  printf(",\"global_kv_and_index_different_values\":%llu,\"decoder_route_selections_changed\":%llu,"
         "\"decoder_route_selections\":%llu}}\n", (unsigned long long)global_different,
         (unsigned long long)route_changed, (unsigned long long)route_total);
  fflush(stdout);
  if (global_different || encoder.different) die("CED mask changed encoder or global cache");
  return all_different + route_changed;
}

static double logsum(const float *a) {
  double top = a[sample_argmax(a, VOCAB)], sum = 0;
  for (int i = 0; i < VOCAB; i++) sum += exp((double)a[i] - top);
  return top + log(sum);
}

static char *read_input(const char *path) {
  FILE *f = !strcmp(path, "-") ? stdin : fopen(path, "rb");
  if (!f) die("cannot open input");
  size_t limit = 2 * 1024 * 1024;
  char *text = xmalloc(limit + 1);
  size_t n = fread(text, 1, limit, f);
  if (ferror(f) || n == limit || memchr(text, 0, n)) die("input must be text smaller than 2 MiB");
  if (f != stdin) fclose(f);
  text[n] = 0;
  return text;
}

int main(int argc, char **argv) {
  if (argc < 3) die("usage: ced-check CHECKPOINT PROMPT_FILE|- [--chat] [--steps N] [--experts GGUF] "
                    "[--describe] [--compare-compact N|--bench N [--compact --keep N] [--dspark N]]");
  int steps = 16, mode = 0, describe_only = 0, bench_tokens = -1, keep_setting = 0;
  int compare_compact = 0;
  int runtime_ced = 0;
  int dspark_depth = 0;
  double cache_gib = 20, reserve_gib = 4;
  const char *experts = NULL;
  for (int i = 3; i < argc; i++) {
    if (!strcmp(argv[i], "--chat")) mode = 1;
    else if (!strcmp(argv[i], "--ced") && i + 1 < argc) {
      const char *v = argv[++i];
      char *end;
      long depth = !strcmp(v, "off") ? 0 : strtol(v, &end, 10);
      if (strcmp(v, "off") && (end == v || *end || depth < WINDOW || depth > 1 << 20))
        die("--ced takes off or 128 to 1048576 replay tokens");
      runtime_ced = depth;
    }
    else if (!strcmp(argv[i], "--describe")) describe_only = 1;
    else if (!strcmp(argv[i], "--compact")) ced_check_compact = 1;
    else if (!strcmp(argv[i], "--dspark") && i + 1 < argc) dspark_depth = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--cache") && i + 1 < argc) cache_gib = atof(argv[++i]);
    else if (!strcmp(argv[i], "--reserve") && i + 1 < argc) reserve_gib = atof(argv[++i]);
    else if (!strcmp(argv[i], "--bench") && i + 1 < argc) bench_tokens = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--keep") && i + 1 < argc) keep_setting = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--compare-compact") && i + 1 < argc) compare_compact = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--steps") && i + 1 < argc) steps = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--experts") && i + 1 < argc) experts = argv[++i];
    else die("unknown or incomplete option %s", argv[i]);
  }
  if (steps < 1 || steps > 64) die("steps must be 1 to 64");
  if (runtime_ced && (bench_tokens < 0 || ced_check_compact || keep_setting))
    die("--ced requires a benchmark without diagnostic mask settings");
  if (dspark_depth < 0 || dspark_depth > BLOCK || (dspark_depth && ced_check_compact))
    die("invalid or unsupported DSpark study combination");
  if (bench_tokens < -1 || bench_tokens > 4096 ||
      keep_setting < 0 || (keep_setting && keep_setting < WINDOW) || compare_compact < 0 ||
      (compare_compact && compare_compact < WINDOW)) die("invalid benchmark or replay bounds");
  if (!isfinite(cache_gib) || cache_gib < 16 || cache_gib > 80 || !isfinite(reserve_gib) ||
      reserve_gib < 2 || reserve_gib > 16) die("invalid cache or reserve budget");
  char *input = read_input(argv[2]), *text = input;
  if (mode == 0) {
    size_t size = strlen(input) + 512;
    text = xmalloc(size);
    snprintf(text, size, "<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: 75 (range 1-100, the higher "
             "the value, the more thorough the reasoning)\n\n<｜User｜>%s<｜Assistant｜><think>", input);
  }
  tok_load(argv[1]);
  int *tokens = xmalloc(32768 * sizeof(int));
  prompt_n = tok_encode_chat(text, tokens, 16384);
  if (prompt_n < 2 || prompt_n >= 16384 || prompt_n + steps >= 32768 || prompt_n + bench_tokens >= 32768)
    die("prompt is outside the diagnostic bounds");
  if (text != input) free(text);
  free(input);
  const char *scope = bench_tokens >= 0 ? "matched prefill and emitted-token throughput" :
                      compare_compact ? "compact decoder execution versus the same full-shape attention mask" :
                      "full-shape decoder attention mask; numerical study, not CED speed measurement";
  printf("{\"ced_setup\":{\"prompt_tokens\":%d,\"requested_steps\":%d,\"temperature\":1,\"top_p\":1,"
         "\"seed\":12345,\"cache_gib\":%.9g,\"reserve_gib\":%.9g,\"draft\":%d,\"substitution\":false,"
         "\"scope\":\"%s\"}}\n", prompt_n, steps, cache_gib, reserve_gib, dspark_depth, scope);
  fflush(stdout);
  if (describe_only) return 0;
  alarm(600);
  Options o = {.ctx = 32768, .budget_gb = cache_gib, .reserve_gb = reserve_gib, .experts = experts,
               .draft = dspark_depth, .draft_cache = dspark_depth ? 32 : 0, .draft_trim = dspark_depth != 0,
               .dump_fd = -1, .routes_fd = -1, .quiet = 1, .ced = runtime_ced};
  model_load(argv[1], &o);
  if (bench_tokens >= 0) {
    benchmark(tokens, bench_tokens, keep_setting);
    free(tokens);
    return 0;
  }
  if (compare_compact) {
    ced_check_first = prompt_n > compare_compact ? prompt_n - compare_compact : 0;
    ced_check_compact = 0;
  }
  float *reference_logits = xmalloc((size_t)steps * VOCAB * sizeof(float));
  int continuation[64], eos = tok_find("<｜end▁of▁sentence｜>");
  Sampler sampler = {.temp = 1, .top_p = 1}, draws[64];
  sampler_seed(&sampler, 12345);
  const float *lg = model_forward(tokens, prompt_n, 0);
  snapshot();
  for (int j = 0; j < steps; j++) {
    memcpy(reference_logits + (size_t)j * VOCAB, lg, VOCAB * sizeof(float));
    draws[j] = sampler;
    continuation[j] = sample_token(lg, VOCAB, &sampler);
    if (continuation[j] == eos) steps = j + 1;
    if (j + 1 < steps) lg = model_forward(continuation + j, 1, prompt_n + j);
  }
  printf("{\"ced_reference\":{\"teacher_forced_positions\":%d}}\n", steps), fflush(stdout);
  int keeps[] = {prompt_n, 128, 256, 512, 1024};
  size_t trials = sizeof keeps / sizeof *keeps;
  if (compare_compact) keeps[0] = compare_compact, trials = 1;
  for (size_t trial = 0; trial < trials; trial++) {
    int keep = keeps[trial];
    if (trial && keep >= prompt_n) continue;
    model_reset();
    if (compare_compact) ced_check_compact = 1;
    ced_check_first = prompt_n > keep ? prompt_n - keep : 0;
    memset(observed, 0, sizeof observed);
    lg = model_forward(tokens, prompt_n, 0);
    uint64_t cache_different = report_cache(keep);
    Error errors = {0};
    double kl_sum = 0, kl_max = 0, tv_sum = 0, tv_max = 0, nll_delta = 0;
    int argmax_equal = 0, sampled_equal = 0;
    for (int j = 0; j < steps; j++) {
      const float *ref = reference_logits + (size_t)j * VOCAB;
      double za = logsum(ref), zb = logsum(lg), kl = 0, tv = 0;
      for (int v = 0; v < VOCAB; v++) {
        add_error(&errors, ref[v], lg[v], memcmp(ref + v, lg + v, sizeof(float)) != 0);
        double la = ref[v] - za, lb = lg[v] - zb, p = exp(la), q = exp(lb);
        kl += p * (la - lb), tv += fabs(p - q) / 2;
      }
      kl_sum += kl, tv_sum += tv;
      if (kl > kl_max) kl_max = kl;
      if (tv > tv_max) tv_max = tv;
      int chosen = continuation[j];
      nll_delta += (ref[chosen] - za) - (lg[chosen] - zb);
      argmax_equal += sample_argmax(ref, VOCAB) == sample_argmax(lg, VOCAB);
      Sampler same_draw = draws[j];
      sampled_equal += sample_token(lg, VOCAB, &same_draw) == chosen;
      if (j + 1 < steps) lg = model_forward(continuation + j, 1, prompt_n + j);
    }
    printf("{\"ced_logits\":{\"keep_tokens\":%d,\"positions\":%d,\"argmax_equal\":%d,\"same_draw_sample_equal\":%d,"
           "\"mean_kl_nats\":%.9g,\"max_kl_nats\":%.9g,\"mean_total_variation\":%.9g,\"max_total_variation\":%.9g,"
           "\"mean_reference_token_nll_delta\":%.9g,\"logits\":", keep, steps, argmax_equal, sampled_equal,
           kl_sum / steps, kl_max, tv_sum / steps, tv_max, nll_delta / steps);
    print_error(&errors);
    fputs("}}\n", stdout), fflush(stdout);
    if (!trial && (cache_different || errors.different))
      die(compare_compact ? "compact replay did not match its full-shape mask" : "full-window repeat did not match the reference");
  }
  for (int l = 0; l < LAYERS; l++)
    for (int k = 0; k < 2; k++) free(reference_global[l][k]);
  free(reference_logits), free(tokens);
  return 0;
}
