// Compare DSpark weight sharing with its scalar expert path on checkpoint
// weights. Check all requested logits, final streams and proposal tokens.
#include "../../src/engine/model.c"
#include <unistd.h>

static size_t differences(const void *a, const void *b, size_t n) {
  const unsigned char *x = a, *y = b;
  size_t bad = 0;
  for (size_t i = 0; i < n; i++) bad += x[i] != y[i];
  return bad;
}

static double middle(double *values, int count) {
  for (int i = 1; i < count; i++)
    for (int j = i; j > 0 && values[j] < values[j - 1]; j--) {
      double t = values[j]; values[j] = values[j - 1], values[j - 1] = t;
    }
  return values[count / 2];
}

// Finish reads already issued during prefill before measuring resident
// drafts. This neither requests new experts nor changes their use scores.
static void finish_reads(void) {
  for (int l = 0; l < CACHE_LAYERS; l++) {
    int ids[EXPERTS], n = 0, limit = l < LAYERS ? EXPERTS : DRAFT_EXPERTS;
    for (int e = 0; e < limit; e++) if (cache_slot(l, e)) ids[n++] = e;
    if (n) cache_wait(l, ids, n);
  }
}

static int timing(int token) {
  const int repeats = 8, samples = 7;
  int failed = 0, drafts[BLOCK];
  opt.draft_trim = 1;
  for (int n = 2; n <= BLOCK; n += 3) {
    double wall[2][7], gpu[2][7];
    uint64_t bytes[2] = {0};
    for (int v = 0; v < 2; v++) opt.expert_batch = v, model_draft(token, 257, drafts, n);
    finish_reads();
    for (int sample = 0; sample < samples; sample++)
      for (int turn = 0; turn < 2; turn++) {
        int v = (sample + turn) % 2;
        opt.expert_batch = v;
        uint64_t before = cache_stats.bytes;
        double t = now(), g = gpu_busy;
        for (int i = 0; i < repeats; i++) model_draft(token, 257, drafts, n);
        wall[v][sample] = (now() - t) * 1000 / repeats;
        gpu[v][sample] = (gpu_busy - g) * 1000 / repeats;
        bytes[v] += cache_stats.bytes - before;
      }
    printf("{\"draft_expert_timing\":{\"depth\":%d,\"trim\":true,\"samples\":%d,\"repeats\":%d,"
           "\"control_wall_ms\":%.6f,\"candidate_wall_ms\":%.6f,\"control_gpu_ms\":%.6f,\"candidate_gpu_ms\":%.6f,"
           "\"control_expert_read_bytes\":%llu,\"candidate_expert_read_bytes\":%llu}}\n",
           n, samples, repeats, middle(wall[0], samples), middle(wall[1], samples),
           middle(gpu[0], samples), middle(gpu[1], samples),
           (unsigned long long)bytes[0], (unsigned long long)bytes[1]);
    fflush(stdout);
    failed |= bytes[0] || bytes[1];
  }
  return failed;
}

int main(int argc, char **argv) {
  if (argc < 2 || argc > 4) die("usage: draft-expert-batch-check CHECKPOINT [EXPERT_GGUF] [--bench]");
  alarm(240);
  Options o = {.ctx = 2048, .budget_gb = 20, .reserve_gb = 4, .draft = 5, .draft_cache = 32,
               .expert_batch = 1, .dump_fd = -1, .routes_fd = -1, .quiet = 1};
  int bench = 0;
  for (int i = 2; i < argc; i++) {
    if (!strcmp(argv[i], "--bench")) bench = 1;
    else if (!o.experts) o.experts = argv[i];
    else die("invalid check argument");
  }
  tok_load(argv[1]);
  int seed[128], tokens[257];
  int count = tok_encode(" Review the queue and preserve cancellation order while sharing expert weights.\n", seed, 128);
  if (!count) die("empty fixture");
  for (int i = 0; i < 257; i++) tokens[i] = seed[i % count];
  model_load(argv[1], &o);
  const float *lg = model_forward(tokens, 257, 0);
  int token = sample_argmax(lg, VOCAB);
  model_state_save();
  float *expected_logits = xmalloc((size_t)BLOCK * VOCAB * 4);
  uint16_t *expected_streams = xmalloc((size_t)BLOCK * HC * DIM * STREAM_BYTES);
  int failed = 0, cases = 0;
  const int depths[] = {1, 2, 5};
  for (int trim = 0; trim < 2; trim++)
    for (unsigned at = 0; at < sizeof depths / sizeof *depths; at++) {
      int n = depths[at], expected[BLOCK], got[BLOCK];
      opt.draft_trim = trim;
      model_state_restore();
      opt.expert_batch = 0;
      model_draft(token, 257, expected, n);
      memcpy(expected_logits, DLOG.ptr, (size_t)n * VOCAB * 4);
      memcpy(expected_streams, DHS.ptr, (size_t)BLOCK * HC * DIM * STREAM_BYTES);
      model_state_restore();
      opt.expert_batch = 1;
      uint64_t calls = draft_batch_calls, pairs = draft_batch_pairs, groups = draft_batch_groups;
      model_draft(token, 257, got, n);
      size_t logits_bad = differences(expected_logits, DLOG.ptr, (size_t)n * VOCAB * 4);
      size_t streams_bad = differences(expected_streams, DHS.ptr, (size_t)BLOCK * HC * DIM * STREAM_BYTES);
      size_t tokens_bad = differences(expected, got, (size_t)n * sizeof(int));
      int nonfinite = 0;
      for (size_t i = 0; i < (size_t)n * VOCAB; i++) nonfinite += !isfinite(((float *)DLOG.ptr)[i]);
      printf("{\"draft_expert_check\":{\"depth\":%d,\"trim\":%d,\"logit_bytes_different\":%zu,"
             "\"stream_bytes_different\":%zu,\"token_bytes_different\":%zu,\"nonfinite\":%d,"
             "\"grouped_stages\":%llu,\"row_expert_pairs\":%llu,\"weight_groups\":%llu}}\n",
             n, trim, logits_bad, streams_bad, tokens_bad, nonfinite,
             (unsigned long long)(draft_batch_calls - calls), (unsigned long long)(draft_batch_pairs - pairs),
             (unsigned long long)(draft_batch_groups - groups));
      fflush(stdout);
      failed |= logits_bad || streams_bad || tokens_bad || nonfinite || draft_batch_calls == calls;
      cases++;
    }
  free(expected_logits), free(expected_streams);
  fprintf(stderr, "DSpark expert sharing: %d cases, %s\n", cases, failed ? "FAIL" : "PASS");
  if (bench && !failed) failed |= timing(token);
  return failed;
}
