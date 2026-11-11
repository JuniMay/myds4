// Passes of a model interleaved at every yield must each compute what they
// compute alone, byte for byte: every logit and, with drafts, every proposed
// token. Two prompts are prefilled and decoded greedily alone, then together,
// each on its own stack with its own buffers and signals. For DeepSeek, whose
// passes share some work buffers, those buffers
// are filled with garbage at every switch: nothing a pass needs may be left in
// them across a wait.
//
// usage: passes CHECKPOINT --text FILE [--experts GGUF] [--memory GIB]
//               [--tokens N] [--decode N] [--draft N] [--draft-model DIR]
//               [--no-garbage] [--verbose]
#include "../../src/model/deepseek/model.h"
#include <math.h>

#include "../../src/model/deepseek/model.c"

enum { MAX_DRAFTS = 15 };

static int use_garbage = 1, drafts_wanted, vocab;
static uint64_t garbage_filled;
static const ModelOps *ops;

typedef struct {
  const int *tokens;
  int n, decode;
  ModelState *state;
  uint64_t digest, proposals;
  int resumes, last, accepted;
} Job;

static uint64_t fnv(uint64_t h, const void *data, size_t n) {
  const unsigned char *p = data;
  for (size_t i = 0; i < n; i++) h = (h ^ p[i]) * 1099511628211ull;
  return h;
}

static int best(const float *logits) {
  int b = 0;
  for (int i = 1; i < vocab; i++)
    if (logits[i] > logits[b]) b = i;
  return b;
}

static void job_body(void *argument) {
  Job *j = argument;
  ops->pass_begin();
  ModelState *state = j->state;
  const float *logits = ops->prefill(state, j->tokens, j->n, 0, 0, -1, 0, 0);
  if (!logits) die("the prefill did not fit in memory");
  j->digest = fnv(14695981039346656037ull, logits, (size_t)vocab * 4);
  int pos = j->n, token = best(logits);
  Sampler greedy = {.temp = 0, .top_p = 1};
  for (int done = 0; done < j->decode;) {
    if (drafts_wanted) {
      int rows[MAX_DRAFTS + 1] = {token}, kept;
      ops->draft(state, token, pos, rows + 1, drafts_wanted, &greedy);
      j->proposals = fnv(j->proposals, rows + 1, (size_t)drafts_wanted * sizeof *rows);
      logits = ops->verify(state, rows, drafts_wanted + 1, pos);
      int next = sample_verify_probabilities(logits, vocab, rows + 1, drafts_wanted, NULL, ops->stop, &greedy, &kept);
      ops->accept(state, rows, kept + 1, pos);
      j->digest = fnv(j->digest, logits, (size_t)(kept + 1) * vocab * 4);
      j->accepted += kept;
      done += kept + 1, pos += kept + 1;
      logits += (size_t)kept * vocab;
      token = next;
    } else {
      logits = ops->rows(&state, &token, &pos, 1);
      if (!logits) die("a decoded token did not fit in memory");
      j->digest = fnv(j->digest, logits, (size_t)vocab * 4);
      done++, pos++;
      token = best(logits);
    }
  }
  j->last = token;
  ops->pass_end();
}

// Between two passes' turns, once queued work has run, the buffers that
// DeepSeek's passes share hold nothing that a pass needs.
static void garbage(void) {
  gpu_sync();
  for (size_t i = 0; i < sizeof work_buffers / sizeof *work_buffers; i++) {
    Buf *b = work_buffers[i];
    if (is_pass_work(b) || !b->obj) continue;
    size_t n = b->size < (1u << 20) ? b->size : (1u << 20);
    memset(b->ptr, 0xAB, n);
    garbage_filled += n;
  }
}

// Runs the jobs' passes, resuming each as soon as it can continue; alone
// (count 1) each waits as a single pass does.
static void run(Job *jobs, int count) {
  Pass *passes[4];
  int ended[4] = {0}, alive = count;
  for (int i = 0; i < count; i++) {
    jobs[i].state = ops->state_new();
    jobs[i].resumes = 0, jobs[i].accepted = 0, jobs[i].proposals = 0;
    passes[i] = ops->pass_new(job_body, &jobs[i]);
  }
  if (count == 1) {
    while (!pass_resume(passes[0])) jobs[0].resumes++, pass_block(passes[0]);
    jobs[0].resumes++;
    ended[0] = 1, alive = 0;
  }
  while (alive) {
    int ran = 0;
    for (int i = 0; i < count; i++) {
      if (ended[i] || !pass_ready(passes[i])) continue;
      if (use_garbage && ops == &deepseek_ops) garbage();
      jobs[i].resumes++;
      ended[i] = pass_resume(passes[i]);
      alive -= ended[i];
      ran = 1;
    }
    if (!ran) usleep(10);
  }
  for (int i = 0; i < count; i++) pass_free(passes[i]), ops->state_free(jobs[i].state);
}

int main(int argc, char **argv) {
  if (argc < 4)
    die("usage: passes CHECKPOINT --text FILE [--experts GGUF] [--memory GIB] [--tokens N] [--decode N] [--draft N] [--draft-model DIR] [--no-garbage]");
  double memory_gib = 40;
  Options o = {.ctx = 32768, .dump_fd = -1, .routes_fd = -1, .quiet = 1};
  const char *text = 0;
  int tokens_each = 600, decode = 16;
  for (int i = 2; i < argc; i++) {
    if (!strcmp(argv[i], "--text") && i + 1 < argc) text = argv[++i];
    else if (!strcmp(argv[i], "--experts") && i + 1 < argc) o.deepseek.experts = argv[++i];
    else if (!strcmp(argv[i], "--draft-model") && i + 1 < argc) o.nemotron.draft_model = argv[++i];
    else if (!strcmp(argv[i], "--memory") && i + 1 < argc) memory_gib = atof(argv[++i]);
    else if (!strcmp(argv[i], "--tokens") && i + 1 < argc) tokens_each = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--decode") && i + 1 < argc) decode = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--draft") && i + 1 < argc) o.draft = drafts_wanted = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--no-garbage")) use_garbage = 0;
    else if (!strcmp(argv[i], "--verbose")) o.quiet = 0;
    else die("invalid argument: %s", argv[i]);
  }
  if (!text) die("--text FILE is required");
  if (drafts_wanted > MAX_DRAFTS) die("--draft takes at most %d", MAX_DRAFTS);
  ops = model_find(argv[1]);
  Tokenizer *tk = tok_open(argv[1], ops->media_token);
  size_t size;
  char *t = read_file(text, &size);
  int *tokens = xmalloc((size + 2) * sizeof *tokens);
  int count = tok_encode(tk, t, tokens, (int)size + 2);
  free(t);
  if (count < 2 * tokens_each) die("the text has %d tokens; %d are needed", count, 2 * tokens_each);
  memory_configure(memory_gib, 4);
  if (!model_open(argv[1], &o)) die("the model did not load");
  vocab = ops->vocab();

  Job alone[2], together[2];
  for (int i = 0; i < 2; i++) {
    alone[i] = (Job){.tokens = tokens + (size_t)i * tokens_each, .n = tokens_each - i * tokens_each / 3, .decode = decode};
    together[i] = alone[i];
  }
  // An idle run first: the expert cache and the GPU's clock start warm for
  // every timed run below.
  run(&alone[0], 1);
  double t0 = now(), busy0 = gpu_busy;
  run(&alone[0], 1);
  double a = now() - t0, a_busy = gpu_busy - busy0;
  t0 = now(), busy0 = gpu_busy;
  run(&alone[1], 1);
  double b = now() - t0, b_busy = gpu_busy - busy0;
  t0 = now(), busy0 = gpu_busy;
  ModelCounters before, after;
  ops->counters(&before);
  run(together, 2);
  ops->counters(&after);
  double both = now() - t0, both_busy = gpu_busy - busy0;
  printf("alone: %.2f s and %.2f s (GPU busy %.2f and %.2f s), one after the other %.2f s; interleaved %.2f s (GPU busy %.2f s)\n", a, b, a_busy, b_busy,
         a + b, both, both_busy);
  printf("while interleaved: %llu layer passes through tables, %llu experts evicted\n", (unsigned long long)(after.table_layers - before.table_layers),
         (unsigned long long)(after.cache.evictions - before.cache.evictions));
  int failures = 0;
  for (int i = 0; i < 2; i++) {
    printf("pass %d: %d prompt tokens, %d decoded", i, alone[i].n, decode);
    if (drafts_wanted)
      printf(" (%d drafts accepted alone, %d together; proposals %016llx and %016llx)", alone[i].accepted, together[i].accepted,
             (unsigned long long)alone[i].proposals, (unsigned long long)together[i].proposals);
    printf("; alone %016llx in %d resumes, together %016llx in %d resumes; last token %d\n", (unsigned long long)alone[i].digest,
           alone[i].resumes, (unsigned long long)together[i].digest, together[i].resumes, alone[i].last);
    if (alone[i].digest != together[i].digest || alone[i].last != together[i].last || alone[i].proposals != together[i].proposals) {
      printf("FAIL: pass %d's logits differ when interleaved\n", i);
      failures++;
    }
  }
  if (use_garbage && ops == &deepseek_ops) printf("garbage written between passes: %.1f GiB\n", (double)garbage_filled / (1u << 30));
  printf("%s\n", failures ? "interleaved passes: FAILED" : "interleaved passes: every logit equal to the pass alone");
  return failures != 0;
}
