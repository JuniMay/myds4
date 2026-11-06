// myds4 command line: generate text, or replay token ids for checks.
#include <math.h>
#include <errno.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "myds4.h"
#include "../experimental/context_lookup.h"
#ifdef MYDS4_PROTOCOL_DRAFT
#include "../experimental/protocol_draft.h"
#endif
#ifdef MYDS4_REJECTION_DRAFT
#include "../experimental/speculative_sample.h"
#endif
#ifdef MYDS4_STAGED_DRAFT
#include "../experimental/draft_queue.h"
static_assert(DRAFT_QUEUE_MAX == BLOCK, "draft queue matches DSpark's width");
static_assert(MYDS4_STAGED_DRAFT > 0 && MYDS4_STAGED_DRAFT <= BLOCK, "valid verification chunk");
#endif
#ifdef MYDS4_PREDICT_LAB
#include "../experimental/predictor_lab.h"
#endif

static void usage(void) {
  fputs("usage: myds4 -m MODEL [-p PROMPT | -f FILE | --tokens IDS] [options]\n"
        "  -n N          tokens to generate (default 128)\n"
        "  -t T, --temperature T  sampling temperature, 0 = greedy (default 0)\n"
        "  --top-p P     nucleus probability in [0, 1] (default 1, full distribution)\n"
        "  --seed N      nonnegative integer sampling seed\n"
        "  --raw         use the prompt as is, without the chat template\n"
        "  --rendered    use a server-rendered chat prompt; preserve literal image-marker text\n"
        "  --think N     thinking effort 1-100; 0 disables it (default off)\n"
        "  --image FILE  an image (any format macOS reads) before the prompt's text, up to 16; with --raw\n"
        "                each takes a <｜deepseek_image｜> of the prompt instead\n"
        "  --vision-fd FD  write the --image images' feature rows (BF16, 5120 per token) to FD and exit\n"
        "  --ctx N       context capacity in tokens, at most 1048576 (default 16384)\n"
        "  --prefill-prune  skip decoder batches outside required history (default off); CED and substitution must be off\n"
        "  --ced off|N   approximate prefill: replay N decoder tokens, 128-1048576 (default off);\n"
        "                keeps the verification rewind margin; decoding uses all layers\n"
        "  --cache GB    expert cache budget (default: from available memory)\n"
        "  --reserve GB  memory the expert cache leaves to other programs, 2-1024 (default 10)\n"
        "  --cache-half-life N  score half-life in forward passes, 1-1000000 (default 50)\n"
        "  --cache-read-cost   weight decode eviction priority by expert read bytes (default off)\n"
        "  --tokens IDS  comma-separated token ids, run without generation\n"
        "  --dump-fd FD  write every layer's streams and the logits to FD\n"
        "  --routes-fd FD  write every decoded token's selected experts, predictions and router logits to FD\n"
        "                (per layer, int32: 6 selected, the top 16 predicted for each of the next two layers,\n"
        "                the top 16 of this layer predicted from the residual before attention; 384 float32)\n"
        "  --tokenize    print the prompt's token ids and exit\n"
        "  --bench       time the kernels on the model's matrices\n"
        "  --check-prefill  compare the prompt's batched prefill with token-by-token decoding\n"
        "  --check-batch  compare a token decoded alone with the same token decoded beside another sequence\n"
        "                (and, with --draft, in a verify pass): the logits must be equal\n"
        "  --draft N     verify up to N (1-5) DSpark drafts per pass using the requested sampler\n"
        "  --lookup off|N  verify up to N (1-15) context-copy drafts; --draft supplies fallback (default off)\n"
        "  --expert-batch  share repeated expert weight loads across verification rows (exact)\n"
        "  --draft-cache N  with --draft: expert slots per stage, 15-128 (default: all 128 preloaded);\n"
        "                smaller caches load the native selections on demand, within --cache\n"
        "  --draft-trim  with --draft: compute only requested final-stage query/FFN rows; keep all five keys\n"
        "  --online-predictor  CPU prefetch correction learned online (default off)\n"
        "  --predictor-state FILE  enable online learning, load existing state and save on clean exit\n"
        "  --substitute D  approximate: when decoding, replace selected experts that are not in memory by the\n"
        "                resident experts with the highest selection keys at most D below theirs, the lowest-ranked\n"
        "                first, never the two with the highest scores\n"
        "  --gpu-route   decoding: the GPU looks up each layer's experts itself, so that a layer whose experts\n"
        "                are all in memory does not wait for the CPU (exact; --substitute applies on the GPU too)\n"
        "  --experts GGUF  read the routed experts from a GGUF file, or a set of shards given by one\n"
        "                (llama.cpp's layout, e.g. unsloth's UD-IQ2_XXS); approximate: smaller codes\n"
        "  --perplexity  decode the text token by token; print the perplexity of each next token\n"
        "  --prefill N   with --perplexity: prefill the text's first N tokens in one pass, then score the rest\n"
        "  --logits-fd FD  with --perplexity: write the logits (129280 float32) each token is scored on to FD\n",
        stderr);
  exit(2);
}

static int complete(void) {
  model_save_predictor();
  online_predictor_report(stderr);
#ifdef MYDS4_PREDICT_LAB
  predictor_lab_report(stderr);
#endif
  return 0;
}

static double sampling_option(const char *name, const char *value, double upper) {
  char *end;
  errno = 0;
  double n = strtod(value, &end);
  if (end == value || *end || errno || !isfinite(n) || n < 0 || n > upper)
    die("%s must be finite and in %s", name, isfinite(upper) ? "[0, 1]" : "[0, infinity)");
  return n;
}

int main(int argc, char **argv) {
  // Interactive work: the scheduler and the GPU's arbitration favor it over
  // background processes.
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
  const char *model = 0, *prompt = 0, *ids = 0;
  int ntokens = 128, raw = 0, think = 0, tokenize = 0, bench = 0, check = 0, check_batch = 0, perplexity = 0,
      prefill = 1, logits_fd = -1;
  double temp = 0, top_p = 1;
  uint64_t seed = 0;
  int seed_set = 0;
  Options o = {.ctx = 16384, .dump_fd = -1, .routes_fd = -1};
  const char *image_path[16];
  int nimg = 0, vision_fd = -1;
  for (int i = 1; i < argc; i++) {
    const char *a = argv[i], *v = i + 1 < argc ? argv[i + 1] : 0;
    if (!strcmp(a, "--raw")) {
      raw = 1;
      continue;
    }
    if (!strcmp(a, "--rendered")) {
      raw = 2;
      continue;
    }
    if (!strcmp(a, "--tokenize")) {
      tokenize = 1;
      if (!raw) raw = 1;
      continue;
    }
    if (!strcmp(a, "--bench")) {
      bench = 1;
      continue;
    }
    if (!strcmp(a, "--check-prefill")) {
      check = 1;
      continue;
    }
    if (!strcmp(a, "--check-batch")) {
      check_batch = 1;
      continue;
    }
    if (!strcmp(a, "--perplexity")) {
      perplexity = 1;
      if (!raw) raw = 1;
      continue;
    }
    if (!strcmp(a, "--gpu-route")) {
      o.gpu_route = 1;
      continue;
    }
    if (!strcmp(a, "--cache-read-cost")) {
      o.cache_read_cost = 1;
      continue;
    }
    if (!strcmp(a, "--draft-trim")) {
      o.draft_trim = 1;
      continue;
    }
    if (!strcmp(a, "--expert-batch")) {
      o.expert_batch = 1;
      continue;
    }
    if (!strcmp(a, "--prefill-prune")) {
      o.prefill_prune = 1;
      continue;
    }
    if (!strcmp(a, "--online-predictor")) {
      o.online_predictor = 64;
      continue;
    }
    if (!v) usage();
    i++;
    if (!strcmp(a, "-m")) model = v;
    else if (!strcmp(a, "-p")) prompt = v;
    else if (!strcmp(a, "-f")) prompt = read_file(v, 0);
    else if (!strcmp(a, "-n")) ntokens = atoi(v);
    else if (!strcmp(a, "-t") || !strcmp(a, "--temperature")) temp = sampling_option("temperature", v, INFINITY);
    else if (!strcmp(a, "--top-p")) top_p = sampling_option("top-p", v, 1);
    else if (!strcmp(a, "--seed")) {
      char *end;
      errno = 0;
      unsigned long long n = strtoull(v, &end, 10);
      if (*v < '0' || *v > '9' || *end || errno) die("--seed needs a nonnegative integer");
      seed = n;
      seed_set = 1;
    } else if (!strcmp(a, "--think")) {
      char *end;
      long n = strtol(v, &end, 10);
      if (end == v || *end || n < 0 || n > 100) die("--think takes 0 (off) or 1 to 100");
      think = (int)n;
    } else if (!strcmp(a, "--ced")) {
      char *end;
      errno = 0;
      long n = !strcmp(v, "off") ? 0 : strtol(v, &end, 10);
      if (strcmp(v, "off") && (end == v || *end || errno || n < WINDOW || n > 1 << 20))
        die("--ced takes off or 128 to 1048576 replay tokens");
      o.ced = (int)n;
    } else if (!strcmp(a, "--ctx")) o.ctx = atoi(v);
    else if (!strcmp(a, "--cache")) o.budget_gb = atof(v);
    else if (!strcmp(a, "--reserve")) {
      char *end;
      double gib = strtod(v, &end);
      if (end == v || *end || !(gib >= CACHE_RESERVE_MIN && gib <= CACHE_RESERVE_MAX))
        die("--reserve takes %d to %d GiB", CACHE_RESERVE_MIN, CACHE_RESERVE_MAX);
      o.reserve_gb = gib;
    }
    else if (!strcmp(a, "--cache-half-life")) {
      char *end;
      long half_life = strtol(v, &end, 10);
      if (end == v || *end || half_life < 1 || half_life > CACHE_HALF_LIFE_MAX)
        die("--cache-half-life takes 1 to %d forward passes", CACHE_HALF_LIFE_MAX);
      o.cache_half_life = (int)half_life;
    } else if (!strcmp(a, "--tokens")) ids = v;
    else if (!strcmp(a, "--dump-fd")) o.dump_fd = atoi(v);
    else if (!strcmp(a, "--routes-fd")) o.routes_fd = atoi(v);
    else if (!strcmp(a, "--draft")) o.draft = atoi(v);
    else if (!strcmp(a, "--lookup")) {
      char *end;
      errno = 0;
      long value = !strcmp(v, "off") ? 0 : strtol(v, &end, 10);
      if (strcmp(v, "off") && (end == v || *end || errno || value < 0 || value > VERIFY_DRAFTS))
        die("--lookup takes off or 1 to %d drafts", VERIFY_DRAFTS);
      o.lookup = (int)value;
    }
    else if (!strcmp(a, "--draft-cache")) {
      char *end;
      long slots = strtol(v, &end, 10);
      if (end == v || *end || slots < 0 || slots > DRAFT_EXPERTS || (slots && slots < DRAFT_CACHE_MIN))
        die("--draft-cache takes 15 to 128 slots per stage (0 keeps the default)");
      o.draft_cache = (int)slots;
    } else if (!strcmp(a, "--predictor-state")) {
      if (!*v) die("--predictor-state needs a nonempty path");
      o.predictor_state = v, o.online_predictor = 64;
    } else if (!strcmp(a, "--substitute")) o.substitute = atof(v);
    else if (!strcmp(a, "--experts")) o.experts = v;
    else if (!strcmp(a, "--prefill")) prefill = atoi(v);
    else if (!strcmp(a, "--logits-fd")) logits_fd = atoi(v);
    else if (!strcmp(a, "--image")) {
      if (nimg == 16) die("at most 16 images");
      image_path[nimg++] = v;
    } else if (!strcmp(a, "--vision-fd")) vision_fd = atoi(v);
    else usage();
  }
  if (vision_fd >= 0 && !prompt) prompt = "";
  if (!model || (!prompt && !ids && !bench)) usage();
  if (nimg && (ids || bench || check || check_batch || perplexity || o.dump_fd >= 0))
    die("--image works with a prompt's generation and with --vision-fd");
  if (o.ctx < 1 || o.ctx > 1 << 20) die("--ctx takes 1 to %d tokens (the model's longest context)", 1 << 20);
  if (o.draft < 0 || o.draft > BLOCK) die("--draft takes 1 to %d drafts", BLOCK);
  if (o.draft_cache && !o.draft) die("--draft-cache needs --draft");
  if (o.draft_trim && !o.draft) die("--draft-trim needs --draft");
  if (o.ced && (o.dump_fd >= 0 || check)) die("--dump-fd and --check-prefill require --ced off; they inspect every prompt row");
  if (o.predictor_state) shutdown_watch();
  if (bench) {
    o.ctx = 2048;
    model_load(model, &o);
    if (shutdown_requested()) return complete();
    model_bench();
    return complete();
  }
  tok_load(model);
  int *tokens = xmalloc((size_t)o.ctx * sizeof *tokens), n = 0;
  if (ids) {
    for (const char *s = ids; *s && n < o.ctx; s += strspn(s, ",")) tokens[n++] = (int)strtol(s, (char **)&s, 10);
  } else {
    size_t len = strlen(prompt) + 512 + 32 * (size_t)nimg;
    char *text = xmalloc(len), user[16 * 32 + 1] = "";
    for (int i = 0; i < nimg && !raw; i++) strcat(user, IMAGE_MARK "\n\n"); // each image's place, then the text
    if (raw) snprintf(text, len, "%s", prompt);
    else if (think)
      snprintf(text, len,
               "<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: %d (range 1-100, the higher "
               "the value, the more thorough the reasoning)\n\n<｜User｜>%s%s<｜Assistant｜><think>",
               think, user, prompt);
    else snprintf(text, len, "<｜begin▁of▁sentence｜><｜User｜>%s%s<｜Assistant｜></think>", user, prompt);
    n = raw == 1 ? tok_encode(text, tokens, o.ctx)
            : tok_encode_chat(text, tokens, o.ctx); // --raw: the placeholders place images
  }
  Image ims[16] = {0};
  if (nimg) { // each placeholder becomes its image's span
    int total = n;
    for (int i = 0; i < nimg; i++) {
      size_t len;
      const char *err;
      char *d = read_file(image_path[i], &len);
      if (!image_open(&ims[i], d, len, &err)) die("%s: %s", image_path[i], err);
      free(d);
      total += image_span(&ims[i]) - 1;
    }
    int *p = xmalloc((size_t)(total + 1) * sizeof *p), m = image_expand(tokens, n, ims, nimg, p, total);
    if (m < 0) die("the prompt holds another number of <｜deepseek_image｜> than --image images");
    if (m > o.ctx) die("the prompt has %d tokens with its images; --ctx holds %d", m, o.ctx);
    free(tokens), tokens = p, n = m;
  }
  if (tokenize) {
    for (int i = 0; i < n; i++) printf("%d%c", tokens[i], i + 1 < n ? ',' : '\n');
    return 0;
  }
  model_load(model, &o);
  if (shutdown_requested()) return complete();
  for (int i = 0; i < nimg; i++) {
    if (shutdown_requested()) return complete();
    double t = now();
    model_image(&ims[i]);
    fprintf(stderr, "image %s: %d x %d patches, %d tokens, encoded in %.2f s\n", image_path[i], ims[i].vh, ims[i].vw,
            image_span(&ims[i]), now() - t);
    if (vision_fd >= 0 && write(vision_fd, ims[i].rows, (size_t)ims[i].th * ims[i].tw * DIM * 2) < 0)
      die("vision: write failed");
  }
  if (vision_fd >= 0) return complete();
  model_images(ims, nimg);
  if (check) { // every position's logits: the batched prefill against token-by-token decoding
    extern float *all_logits;
    all_logits = xmalloc((size_t)n * VOCAB * 4);
    model_forward(tokens, n, 0);
    float *a = all_logits;
    all_logits = 0;
    if (shutdown_requested()) return complete();
    model_reset();
    if (n >= 4) { // two prompt batches, the first of odd length, against one
      int n1 = (n / 2) | 1, diff = 0;
      model_forward(tokens, n1, 0);
      const float *c = model_forward(tokens + n1, n - n1, n1), *an = a + (size_t)(n - 1) * VOCAB;
      for (int k = 0; k < VOCAB; k++) diff += c[k] != an[k];
      printf("two batches (%d + %d) against one: %d of %d final logits differ\n", n1, n - n1, diff, VOCAB);
      model_reset();
    }
    if (shutdown_requested()) return complete();
    int agree[2] = {0}, count[2] = {0};
    double snr[2] = {0};
    for (int i = 0; i < n && !shutdown_requested(); i++) {
      const float *b = model_forward(tokens + i, 1, i), *ai = a + (size_t)i * VOCAB;
      double err = 0, ref = 0;
      for (int k = 0; k < VOCAB; k++) err += (ai[k] - b[k]) * (double)(ai[k] - b[k]), ref += (double)b[k] * b[k];
      int part = i >= WINDOW;
      count[part]++, agree[part] += sample_argmax(ai, VOCAB) == sample_argmax(b, VOCAB),
          snr[part] += 10 * log10(ref / err);
    }
    for (int part = 0; part < 2; part++)
      if (count[part])
        printf("positions %s 128: argmax equal at %d of %d, mean logit SNR %.1f dB\n", part ? ">=" : "<", agree[part],
               count[part], snr[part] / count[part]);
    return complete();
  }
  if (check_batch) { // the next token of the prompt (a) and of the prompt less 3 tokens (b), alone and together
    if (n < 5) die("--check-batch needs a prompt of at least 5 tokens");
    State *st[2] = {model_state_new(), model_state_new()};
    int len[2] = {n, n - 3}, next[2];
    float *alone = xmalloc((size_t)2 * VOCAB * 4);
    for (int s = 0; s < 2; s++) {
      if (shutdown_requested()) return complete();
      model_state_use(st[s]);
      next[s] = sample_argmax(model_forward(tokens, len[s], 0), VOCAB);
      model_state_save();
      memcpy(alone + (size_t)s * VOCAB, model_forward(&next[s], 1, len[s]), VOCAB * 4);
      model_state_restore();
    }
    if (shutdown_requested()) return complete();
    const float *both = model_forward_rows(st, next, len, 2);
    int diff[2] = {0};
    for (int s = 0; s < 2; s++)
      for (int k = 0; k < VOCAB; k++) diff[s] += both[(size_t)s * VOCAB + k] != alone[(size_t)s * VOCAB + k];
    printf("two sequences decoded together against alone: %d and %d of %d logits differ\n", diff[0], diff[1], VOCAB);
    if (o.draft) { // a verify pass's first row is the token alone
      if (shutdown_requested()) return complete();
      model_state_use(st[0]);
      model_state_restore();
      int rows[BLOCK + 1] = {next[0]}, d = 0;
      model_draft(next[0], n, rows + 1, o.draft);
      const float *v = model_verify(rows, o.draft + 1, n);
      for (int k = 0; k < VOCAB; k++) d += v[k] != alone[k];
      printf("a verify pass of %d drafts against the token alone: %d of %d first-row logits differ\n", o.draft, d,
             VOCAB);
    }
    return complete();
  }
  if (ids) { // replay: one token at a time, every position's logits dumped
    for (int i = 0; i < n && !shutdown_requested(); i++) model_forward(tokens + i, 1, i);
    cache_metrics_report(stderr);
    return complete();
  }
  if (perplexity) { // tokens prefill.. scored, each on the logits after the token before it, decoded one at a time
    if (prefill < 1 || prefill >= n) die("--prefill takes 1 to %d tokens", n - 1);
    const float *lg = prefill > 1 ? model_forward(tokens, prefill, 0) : 0; // the logits after token prefill - 1
    double nll = 0, t0 = now();
    int top = 0, steps = 0, scored = 0;
    CacheStats c0 = cache_stats;
    for (int i = prefill; i < n && !shutdown_requested(); i++, lg = 0) {
      if (!lg) lg = model_forward(tokens + i - 1, 1, i - 1), steps++;
      if (logits_fd >= 0 && write(logits_fd, lg, VOCAB * 4) < 0) die("logits: write failed");
      double mx = lg[0], sum = 0;
      for (int k = 1; k < VOCAB; k++) mx = lg[k] > mx ? lg[k] : mx;
      for (int k = 0; k < VOCAB; k++) sum += exp(lg[k] - mx);
      nll += mx + log(sum) - lg[tokens[i]];
      top += sample_argmax(lg, VOCAB) == tokens[i];
      scored++;
    }
    if (scored)
      printf("perplexity %.4f over %d tokens; the next token is the most likely at %d of them\n", exp(nll / scored),
             scored, top);
    else if (shutdown_requested()) fprintf(stderr, "perplexity interrupted before scoring\n");
    fprintf(stderr, "%.2f tokens/s; %.1f experts substituted per token\n", steps / (now() - t0),
            (cache_stats.substituted - c0.substituted) / (double)(steps ? steps : 1));
    cache_metrics_report(stderr);
    return complete();
  }
  int eos = tok_find("<｜end▁of▁sentence｜>");
  Sampler sampler = {.temp = temp, .top_p = top_p};
  if (seed_set) sampler_seed(&sampler, seed);
  else arc4random_buf(sampler.seed, sizeof sampler.seed);
  CacheStats c0 = cache_stats;
  uint64_t swap0 = mem_swapouts(), comp0 = mem_compressions();
  extern double phase_s[7];
  double g0 = gpu_busy, t0 = now(), p1[7];
  const float *logits = model_forward(tokens, n, 0);
  uint64_t prefill_workspace = model_workspace_bytes();
  ModelForwardInfo prune_info = model_forward_info();
  fprintf(stderr, "{\"prefill_prune\":{\"applied\":%s,\"tokens\":%d,\"decoder_tokens\":%d,\"decoder_row_layers\":%d,\"chunks\":%d,\"prefix_tokens\":%d}}\n",
          prune_info.prune_applied ? "true" : "false", prune_info.tokens, prune_info.decoder_tokens,
          prune_info.decoder_row_layers, prune_info.chunks, prune_info.prefix_tokens);
  double t1 = now();
  CacheStats c1 = cache_stats;
  uint64_t rows_before_decode = decode_rows;
#ifdef MYDS4_IO_DIAGNOSTICS
  cache_io_diagnostic_decode();
#endif
  const char *prof = getenv("MYDS4_PROFILE");
  if (!prof || strcmp(prof, "prefill")) gpu_profile_reset(); // a profile covers decoding, or the prefill
  memcpy(p1, phase_s, sizeof p1);
  double g1 = gpu_busy, gap1 = gpu_gap;
  long gaps1 = gpu_gaps;
  uint64_t fix1 = spec_fixups;
  int pos = n, gen = 0, passes = 0, accepted = 0, t = sample_token(logits, VOCAB, &sampler), len;
  ContextLookup *lookup = NULL;
#ifdef MYDS4_REJECTION_DRAFT
  double *draft_probability = o.draft ? xmalloc((size_t)o.draft * VOCAB * sizeof(double)) : NULL;
  int rejection_passes = 0, accepted_prefix[BLOCK + 1] = {0};
#endif
#ifdef MYDS4_PROTOCOL_DRAFT
  ProtocolDraft protocol = {0};
  if (o.lookup) protocol_draft_init(&protocol);
#else
  lookup = o.lookup ? context_lookup_new(o.ctx) : NULL;
  if (o.lookup && (!lookup || !context_lookup_append(lookup, tokens, n))) die("cannot initialize context lookup");
#endif
  int lookup_attempts = 0, lookup_passes = 0, lookup_offered = 0, lookup_kept = 0;
#ifdef MYDS4_STAGED_DRAFT
  DraftQueue pending_draft = {0};
  int draft_generations = 0, draft_reuses = 0;
#endif
  uint64_t cache_start = cache_bytes(), cache_min = cache_start, cache_max = cache_start;
  while (gen < ntokens && pos < o.ctx && t != eos && !shutdown_requested()) {
    // A verification pass runs the current token and its proposed successors.
    // Target samples confirm a prefix and supply the next token, consuming
    // exactly the draws used by ordinary sampling.
    int rows[VERIFY_DRAFTS + 1] = {t}, k = 0, a = 0, next;
    int room = ntokens - gen - 1;
    if (room > o.ctx - pos - 1) room = o.ctx - pos - 1;
    ContextDraft proposal = {0};
#ifdef MYDS4_PROTOCOL_DRAFT
    if (o.lookup) {
      protocol_draft_push(&protocol, &t, 1);
      proposal = protocol_draft_propose(&protocol, o.lookup < room ? o.lookup : room);
      lookup_attempts++;
    }
#endif
    if (lookup) {
      if (!context_lookup_append(lookup, &t, 1)) die("context lookup exceeds the configured context");
#ifdef MYDS4_CONSENSUS_LOOKUP
      proposal = context_lookup_propose_consensus(lookup, o.lookup < room ? o.lookup : room);
#else
      proposal = context_lookup_propose(lookup, o.lookup < room ? o.lookup : room);
#endif
      lookup_attempts++;
    }
    if (proposal.count) {
#ifdef MYDS4_STAGED_DRAFT
      pending_draft.count = 0;
#endif
      k = proposal.count;
      memcpy(rows + 1, proposal.tokens, (size_t)k * sizeof *rows);
      lookup_passes++;
      lookup_offered += k;
    } else {
#ifdef MYDS4_STAGED_DRAFT
      if (room && o.draft && !pending_draft.count) {
        int guesses[BLOCK], generated = o.draft < room ? o.draft : room;
#ifdef MYDS4_COUPLED_DRAFT
        model_draft_coupled(t, pos, guesses, generated, &sampler);
#else
        model_draft(t, pos, guesses, generated);
#endif
        if (!draft_queue_set(&pending_draft, guesses, generated)) die("invalid staged draft block");
        draft_generations++;
      } else if (room && pending_draft.count) draft_reuses++;
      k = draft_queue_take(&pending_draft, rows + 1, room < MYDS4_STAGED_DRAFT ? room : MYDS4_STAGED_DRAFT);
#else
      k = o.draft < room ? o.draft : room;
      if (k) {
#ifdef MYDS4_REJECTION_DRAFT
        model_draft_sampled(t, pos, rows + 1, k, &sampler, draft_probability);
#elif defined(MYDS4_COUPLED_DRAFT)
        model_draft_coupled(t, pos, rows + 1, k, &sampler);
#else
        model_draft(t, pos, rows + 1, k);
#endif
      }
#endif
    }
    if (k > 0) {
#ifdef MYDS4_CAUSAL_EARLY_STOP
      logits = model_verify_early(rows, k + 1, pos, eos, &sampler, &next, &a);
#else
      logits = model_verify(rows, k + 1, pos);
#ifdef MYDS4_REJECTION_DRAFT
      if (!proposal.count) {
        next = speculative_verify(logits, VOCAB, rows + 1, k, draft_probability, eos, &sampler, &a);
        rejection_passes++, accepted_prefix[a]++;
      } else
#endif
      next = sample_verify(logits, VOCAB, rows + 1, k, eos, &sampler, &a);
#endif
      model_accept(rows, a + 1, pos);
      logits += (size_t)a * VOCAB;
      passes++, accepted += a;
    } else {
      logits = model_forward(&t, 1, pos);
      next = sample_token(logits, VOCAB, &sampler);
    }
#ifdef MYDS4_STAGED_DRAFT
    if (k && !proposal.count && !draft_queue_accept(&pending_draft, k, a, next))
      die("invalid staged draft acceptance");
    if (next == eos) pending_draft.count = 0;
#endif
    if (lookup && a && !context_lookup_append(lookup, rows + 1, a)) die("context lookup exceeds the configured context");
#ifdef MYDS4_PROTOCOL_DRAFT
    if (o.lookup && a) protocol_draft_push(&protocol, rows + 1, a);
#endif
    if (proposal.count) {
      if (lookup) context_lookup_feedback(lookup, k, a);
#ifdef MYDS4_PROTOCOL_DRAFT
      protocol_draft_feedback(&protocol, proposal.source, a);
#endif
      lookup_kept += a;
    }
    uint64_t current_cache = cache_bytes();
    if (current_cache < cache_min) cache_min = current_cache;
    if (current_cache > cache_max) cache_max = current_cache;
    int i = 0;
    for (; i <= a && rows[i] != eos; i++, gen++) {
      const char *s = tok_piece(rows[i], &len);
      fwrite(s, 1, len, stdout);
    }
    fflush(stdout);
    if (i <= a) break; // a kept draft ends the text
    pos += a + 1;
    t = next;
  }
  context_lookup_free(lookup);
  if (!gen && shutdown_requested()) {
    fprintf(stderr, "generation interrupted before decoding\n");
    return complete();
  }
  double t2 = now();
  int verify_limit = o.draft;
#ifdef MYDS4_STAGED_DRAFT
  if (verify_limit > MYDS4_STAGED_DRAFT) verify_limit = MYDS4_STAGED_DRAFT;
#endif
  if (verify_limit < o.lookup) verify_limit = o.lookup;
  if (passes)
    fprintf(stderr, "speculative: %d passes, %.2f of up to %d drafts kept per pass\n", passes,
            (double)accepted / passes, verify_limit);
  if (o.lookup)
#ifdef MYDS4_PROTOCOL_DRAFT
    fprintf(stderr, "{\"protocol_draft\":{\"attempts\":%d,\"passes\":%d,\"offered\":%d,\"accepted\":%d}}\n",
#else
    fprintf(stderr, "{\"context_lookup\":{\"attempts\":%d,\"passes\":%d,\"offered\":%d,\"accepted\":%d}}\n",
#endif
            lookup_attempts, lookup_passes, lookup_offered, lookup_kept);
#ifdef MYDS4_STAGED_DRAFT
  fprintf(stderr, "{\"staged_draft\":{\"generated_drafts\":%d,\"chunk_drafts\":%d,\"generations\":%d,\"reused_passes\":%d}}\n",
          o.draft, MYDS4_STAGED_DRAFT, draft_generations, draft_reuses);
#endif
  CacheStats c2 = cache_stats;
#ifdef MYDS4_REJECTION_DRAFT
  fprintf(stderr, "{\"rejection_sampling\":{\"enabled\":true,\"drafts\":%d,\"same_seed_output\":false,\"passes\":%d,\"accepted_prefix_counts\":[",
          o.draft, rejection_passes);
  for (int r = 0; r <= BLOCK; r++) fprintf(stderr, "%s%d", r ? "," : "", accepted_prefix[r]);
  fprintf(stderr, "]}}\n");
  free(draft_probability);
#endif
  uint64_t computed_rows = decode_rows - rows_before_decode;
#ifdef MYDS4_CAUSAL_EARLY_STOP
  EarlyVerifyStats early = model_early_stats();
  uint64_t scheduled_rows = computed_rows;
  uint64_t selected = ((scheduled_rows - early.attempted_rows) * LAYERS + early.requested_row_layers) * TOPK;
  computed_rows -= early.cancelled_rows;
  fprintf(stderr, "{\"decode_experts\":{\"computed_rows\":%llu,\"scheduled_rows\":%llu,\"selected\":%llu,\"unique_layer_requests\":%llu}}\n",
          (unsigned long long)computed_rows, (unsigned long long)scheduled_rows, (unsigned long long)selected,
          (unsigned long long)(c2.row_unique - c1.row_unique));
#else
  fprintf(stderr, "{\"decode_experts\":{\"computed_rows\":%llu,\"selected\":%llu,\"unique_layer_requests\":%llu}}\n",
          (unsigned long long)computed_rows, (unsigned long long)(computed_rows * LAYERS * TOPK),
          (unsigned long long)(c2.row_unique - c1.row_unique));
#endif
  fprintf(stderr, "{\"decode_cache\":{\"bytes_at_start\":%llu,\"min_bytes\":%llu,\"max_bytes\":%llu,\"slots_released\":%llu}}\n",
          (unsigned long long)cache_start, (unsigned long long)cache_min, (unsigned long long)cache_max,
          (unsigned long long)(c2.given_back - c1.given_back));
  double g2 = gpu_busy;
  fprintf(stderr, "{\"workspace\":{\"resizable_after_prefill_bytes\":%llu,\"resizable_after_decode_bytes\":%llu}}\n",
          (unsigned long long)prefill_workspace, (unsigned long long)model_workspace_bytes());
  fputc('\n', stdout);
  // hits: requested experts that needed no read on demand (resident, or prefetched)
  double hit1 = 1 - (double)(c1.misses - c0.misses) /
                        fmax(1, c1.hits - c0.hits + c1.misses - c0.misses + c1.prefetch_hits - c0.prefetch_hits);
  double hit2 = 1 - (double)(c2.misses - c1.misses) /
                        fmax(1, c2.hits - c1.hits + c2.misses - c1.misses + c2.prefetch_hits - c1.prefetch_hits);
  fprintf(
      stderr,
      "prefill: %d tokens in %.2f s, %.2f tok/s, expert hits %.1f%%, %.2f GB read in %.2f s, GPU %.2f s\n"
      "decode:  %d tokens in %.2f s, %.2f tok/s, expert hits %.1f%%, %.2f GB read in %.2f s, GPU %.2f s\n"
      "cache:   %d slots, %.1f GiB, %llu evictions; available memory %.1f GiB, "
      "footprint %.1f GiB, GPU resident memory (all processes) %.1f GiB; %llu pages swapped out and %llu compressed "
      "by the system meanwhile, this process's memory compressed at most %.0f MiB\n",
      n, t1 - t0, n / (t1 - t0), 100 * hit1, (c1.bytes - c0.bytes) / 1e9, c1.read_s - c0.read_s, g1 - g0, gen, t2 - t1,
      gen / (t2 - t1), 100 * hit2, (c2.bytes - c1.bytes) / 1e9, c2.read_s - c1.read_s, g2 - g1, cache_slots(),
      cache_bytes() / (1024.0 * 1024 * 1024), (unsigned long long)c2.evictions,
      mem_available() / (1024.0 * 1024 * 1024), mem_footprint() / (1024.0 * 1024 * 1024),
      mem_gpu() / (1024.0 * 1024 * 1024), (unsigned long long)(mem_swapouts() - swap0),
      (unsigned long long)(mem_compressions() - comp0), mem_compressed(1) / (1024.0 * 1024));
  fprintf(stderr,
          "decode reads per token: %.1f on demand, %.1f ahead (%.1f used, %.1f of them late), %.1f cancelled; "
          "%.1f experts substituted, %.1f mapped for the GPU again\n",
          (c2.misses - c1.misses) / (double)gen, (c2.prefetches - c1.prefetches) / (double)gen,
          (c2.prefetch_hits - c1.prefetch_hits) / (double)gen, (c2.late - c1.late) / (double)gen,
          (c2.cancelled - c1.cancelled) / (double)gen, (c2.substituted - c1.substituted) / (double)gen,
          (c2.remaps - c1.remaps) / (double)gen);
  fprintf(stderr,
          "decode per token (ms): rows %.1f, waits %.1f, loads %.1f, prefetch %.1f, encode %.1f, adapt %.1f, "
          "drafting %.1f; "
          "GPU idle between command buffers %.1f (%.1f gaps over 100 us); %.1f layers' experts recomputed\n",
          (phase_s[0] - p1[0]) / gen * 1e3, (phase_s[1] - p1[1]) / gen * 1e3, (phase_s[2] - p1[2]) / gen * 1e3,
          (phase_s[3] - p1[3]) / gen * 1e3, (phase_s[4] - p1[4]) / gen * 1e3, (phase_s[5] - p1[5]) / gen * 1e3,
          (phase_s[6] - p1[6]) / gen * 1e3, (gpu_gap - gap1) / gen * 1e3, (double)(gpu_gaps - gaps1) / gen,
          (double)(spec_fixups - fix1) / gen);
  cache_metrics_report(stderr);
  return complete();
}
