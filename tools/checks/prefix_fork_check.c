// The fork test's engine part (docs/server.md, the KV prefix cache). A
// conversation is forked at many positions, by each way the cache resumes
// there: an exact replay at the fork, and the latest snapshot at or before it.
// - Where the history before the fork came from prompts, both must give what
//   a cold prefill of the fork's prompt gives: the same bytes in the logits,
//   in every state tensor and in the next 16 greedy tokens.
// - Where it holds decoded rows, a cold prefill differs from them by design:
//   decoding and prefilling round differently. A snapshot resume must give
//   the same bytes as the conversation itself, computed again as it was (its
//   replies decoded, its prompts prefilled) up to the snapshot and then
//   prefilled. A replay recomputes the rows near the fork with the prefill
//   kernels, so it matches neither; it must be about as close to the cold
//   prefill as the conversation continued at the fork is: for each kind of
//   state tensor and the logits, an SNR at most MARGIN_DB dB lower (see near()).
// - Each fork captures the rolling rows at its branch point, as the server
//   does: a later fork there, resumed from that snapshot over the first
//   fork's rows, must give the first fork's bytes.
// With --ced N, forks are approximate by design (docs/usage.md): the replay
// is bounded to the last ring's positions. It must keep the source's
// compressed rows and index keys that positions before the fork completed,
// byte for byte, and give finite logits; its distance from the cold prefill
// (also compacted) is printed, as is the snapshot resume's.
// --measure prints how far decoding a reply and prefilling the same tokens
// differ. The check reads the state's buffers directly and writes no files.
#include <math.h>

#include "../../src/engine/model.c"

// How much further from the cold prefill a replay over decoded rows may be
// than the conversation continued at the fork, in dB per kind of tensor; and
// the SNR taken as "equal" when the continued conversation equals the cold
// prefill in a kind. Rounding differences in this model stay below 30 dB
// (--measure); wrong rows give about 0 dB.
enum { MARGIN_DB = 6, EQUAL_DB = 40 };

typedef struct {
  double sig, err;
} Snr;

static void add(Snr *s, double ref, double x) {
  s->sig += ref * ref, s->err += (ref - x) * (ref - x);
}

static double db(Snr s) {
  return s.err == 0 ? INFINITY : 10 * log10(s.sig / s.err);
}

static float bf16(const uint8_t *p) {
  uint32_t u = (uint32_t)(p[0] | p[1] << 8) << 16;
  float f;
  memcpy(&f, &u, 4);
  return f;
}

static Snr snr_bf16(const uint8_t *ref, const uint8_t *x, size_t bytes) {
  Snr s = {0};
  for (size_t i = 0; i + 1 < bytes; i += 2) add(&s, bf16(ref + i), bf16(x + i));
  return s;
}

static Snr snr_f32(const float *ref, const float *x, size_t n) {
  Snr s = {0};
  for (size_t i = 0; i < n; i++) add(&s, ref[i], x[i]);
  return s;
}

static size_t differ(const void *a, const void *b, size_t n) {
  const uint8_t *x = a, *y = b;
  size_t bad = 0;
  for (size_t i = 0; i < n; i++) bad += x[i] != y[i];
  return bad;
}

static uint8_t *host_copy(const void *p, size_t n) {
  uint8_t *q = xmalloc(n ? n : 1);
  memcpy(q, p, n);
  return q;
}

static int finite(const float *x, size_t n) {
  for (size_t i = 0; i < n; i++)
    if (!isfinite(x[i])) return 0;
  return 1;
}

// A host copy of the state in use's tensors and its next token's logits.
typedef struct {
  uint8_t *ring[LAYERS], *comp[LAYERS], *keys[LAYERS], *cstate[LAYERS], *dring[DRAFTS];
  int32_t history[3];
  float *logits;
} Tensors;

static void take(Tensors *c, const float *lg) {
  gpu_sync();
  for (int l = 0; l < LAYERS; l++) {
    c->ring[l] = host_copy(L[l].ring.ptr, L[l].ring.size);
    if (L[l].kv_source)
      c->comp[l] = host_copy(L[l].comp.ptr, L[l].comp.size), c->keys[l] = host_copy(L[l].keys.ptr, L[l].keys.size),
      c->cstate[l] = host_copy(L[l].cstate.ptr, L[l].cstate.size);
  }
  for (int d = 0; opt.draft && d < DRAFTS; d++) c->dring[d] = host_copy(D[d].ring.ptr, D[d].ring.size);
  memcpy(c->history, history, sizeof history);
  c->logits = (float *)host_copy(lg, VOCAB * 4);
}

static void drop(Tensors *c) {
  for (int l = 0; l < LAYERS; l++) free(c->ring[l]), free(c->comp[l]), free(c->keys[l]), free(c->cstate[l]);
  for (int d = 0; d < DRAFTS; d++) free(c->dring[d]);
  free(c->logits);
  *c = (Tensors){0};
}

// The bytes in which two states at position n differ: the compressed rows and
// index keys that n positions completed, the rings, compressor rows, draft
// rings, the history and the logits.
static size_t differ_state(const Tensors *a, const Tensors *b, int n) {
  size_t bad = differ(a->logits, b->logits, VOCAB * 4) + differ(a->history, b->history, sizeof a->history);
  for (int l = 0; l < LAYERS; l++) {
    bad += differ(a->ring[l], b->ring[l], L[l].ring.size);
    if (!L[l].kv_source) continue;
    size_t rows = (size_t)(n / L[l].ratio);
    bad += differ(a->comp[l], b->comp[l], rows * HEAD_DIM * 2) + differ(a->keys[l], b->keys[l], rows * IDX_DIM * 2);
    bad += differ(a->cstate[l], b->cstate[l], L[l].cstate.size);
  }
  for (int d = 0; opt.draft && d < DRAFTS; d++) bad += differ(a->dring[d], b->dring[d], D[d].ring.size);
  return bad;
}

// Where two states at position n differ, layer by layer: the ring's
// positions, the compressed rows' and index keys' first and last differing
// positions, the compressor rows, and the logits.
static void explain(const Tensors *a, const Tensors *b, int n) {
  printf("    logits: %zu bytes differ, SNR %.1f dB\n", differ(a->logits, b->logits, VOCAB * 4),
         db(snr_f32(a->logits, b->logits, VOCAB)));
  for (int l = 0; l < LAYERS; l++) {
    int count = 0, lo = -1, hi = -1;
    for (int q = n > RING ? n - RING : 0; q < n; q++) {
      size_t off = (size_t)(q % RING) * HEAD_DIM * 2;
      if (memcmp(a->ring[l] + off, b->ring[l] + off, HEAD_DIM * 2)) count++, hi = q, lo = lo < 0 ? q : lo;
    }
    char line[256];
    int w = snprintf(line, sizeof line, "    layer %d: ring %d positions (%d..%d), SNR %.1f dB", l, count, lo, hi,
                     db(snr_bf16(a->ring[l], b->ring[l], L[l].ring.size)));
    int bad = count > 0;
    if (L[l].kv_source) {
      int rows = n / L[l].ratio, kinds[2] = {HEAD_DIM, IDX_DIM};
      const uint8_t *x[2] = {a->comp[l], a->keys[l]}, *y[2] = {b->comp[l], b->keys[l]};
      for (int k = 0; k < 2; k++) {
        int first = -1, last = -1;
        for (int r = 0; r < rows; r++)
          if (memcmp(x[k] + (size_t)r * kinds[k] * 2, y[k] + (size_t)r * kinds[k] * 2, (size_t)kinds[k] * 2))
            last = r, first = first < 0 ? r : first;
        if (first >= 0) bad = 1;
        w += snprintf(line + w, sizeof line - w, "; %s rows %d..%d", k ? "keys" : "compressed", first * L[l].ratio,
                      last * L[l].ratio);
      }
      size_t c = differ(a->cstate[l], b->cstate[l], L[l].cstate.size);
      if (c) bad = 1;
      snprintf(line + w, sizeof line - w, "; compressor %zu bytes", c);
    }
    if (bad) printf("%s\n", line);
  }
}

// The lowest SNR of each kind of tensor between a and b: the compressed rows
// and index keys that positions [from, to) completed, the ring, compressor
// and draft buffers whole, and the logits.
typedef struct {
  double logits, comp, keys, ring, cstate, dring;
  int history_equal, argmax_equal;
} Diff;

static Diff compare(const Tensors *a, const Tensors *b, int from, int to) {
  Diff d = {.logits = db(snr_f32(a->logits, b->logits, VOCAB)), .comp = INFINITY, .keys = INFINITY,
            .ring = INFINITY, .cstate = INFINITY, .dring = INFINITY};
  for (int l = 0; l < LAYERS; l++) {
    d.ring = fmin(d.ring, db(snr_bf16(a->ring[l], b->ring[l], L[l].ring.size)));
    if (!L[l].kv_source) continue;
    size_t first = (size_t)(from / L[l].ratio), last = (size_t)(to / L[l].ratio);
    d.comp = fmin(d.comp, db(snr_bf16(a->comp[l] + first * HEAD_DIM * 2, b->comp[l] + first * HEAD_DIM * 2, (last - first) * HEAD_DIM * 2)));
    d.keys = fmin(d.keys, db(snr_bf16(a->keys[l] + first * IDX_DIM * 2, b->keys[l] + first * IDX_DIM * 2, (last - first) * IDX_DIM * 2)));
    d.cstate = fmin(d.cstate, db(snr_f32((const float *)a->cstate[l], (const float *)b->cstate[l], L[l].cstate.size / 4)));
  }
  for (int i = 0; opt.draft && i < DRAFTS; i++) d.dring = fmin(d.dring, db(snr_bf16(a->dring[i], b->dring[i], D[i].ring.size)));
  d.history_equal = !memcmp(a->history, b->history, sizeof a->history);
  d.argmax_equal = sample_argmax(a->logits, VOCAB) == sample_argmax(b->logits, VOCAB);
  return d;
}

// Whether x (a replay against the cold prefill) is about as close as ref (the
// conversation continued at the fork against the cold prefill), kind by kind.
static int near(Diff x, Diff ref) {
  double e[] = {x.logits, x.comp, x.keys, x.ring, x.cstate, opt.draft ? x.dring : INFINITY};
  double r[] = {ref.logits, ref.comp, ref.keys, ref.ring, ref.cstate, opt.draft ? ref.dring : INFINITY};
  for (int i = 0; i < 6; i++)
    if (e[i] < fmin(r[i], EQUAL_DB) - MARGIN_DB) return 0;
  return x.history_equal;
}

static void print_diff(const char *what, Diff d) {
  printf("%s: SNR dB logits %.1f  compressed %.1f  index keys %.1f  rings %.1f  compressor %.1f  draft rings %.1f;"
         "  history %s  next token %s\n",
         what, d.logits, d.comp, d.keys, d.ring, d.cstate, d.dring, d.history_equal ? "equal" : "DIFFERENT",
         d.argmax_equal ? "equal" : "different");
  fflush(stdout);
}

static int *read_text(const char *path, int rendered, int *count) {
  FILE *f = fopen(path, "rb");
  if (!f) die("cannot read %s", path);
  fseek(f, 0, SEEK_END);
  long size = ftell(f);
  fseek(f, 0, SEEK_SET);
  char *t = xmalloc((size_t)size + 1);
  if (fread(t, 1, (size_t)size, f) != (size_t)size) die("cannot read %s", path);
  t[size] = 0;
  fclose(f);
  int *p = xmalloc(((size_t)size + 2) * sizeof *p);
  *count = rendered ? tok_encode_chat(t, p, (int)size + 2) : tok_encode(t, p, (int)size + 2);
  free(t);
  return p;
}

// ---- --measure: decoding a reply against prefilling it ----

static Diff decoded_against_prefilled(const int *text, int n, int m) {
  int *tokens = xmalloc((size_t)(n + m) * sizeof *tokens);
  memcpy(tokens, text, (size_t)n * sizeof *tokens);
  State *a = model_state_new(), *b = model_state_new();
  model_state_use(a);
  model_reset();
  const float *lg = model_forward(tokens, n, 0);
  for (int i = 0; i < m; i++) {
    tokens[n + i] = sample_argmax(lg, VOCAB);
    lg = model_forward(tokens + n + i, 1, n + i);
  }
  Tensors decoded = {0}, prefilled = {0};
  take(&decoded, lg);
  model_state_use(b);
  model_reset();
  lg = model_forward(tokens, n + m, 0);
  take(&prefilled, lg);
  model_state_use(a);
  model_state_free(b);
  free(tokens);
  Diff d = compare(&prefilled, &decoded, n, n + m);
  drop(&decoded), drop(&prefilled);
  return d;
}

static int measure(const int *tokens, int count, int rendered) {
  const int m = 256, n = rendered ? count : 2000;
  Diff worst = {INFINITY, INFINITY, INFINITY, INFINITY, INFINITY, INFINITY, 1, 1};
  for (int k = 0; k < (rendered ? 1 : 2); k++) {
    int start = rendered ? 0 : k * (count / 2 > n ? count / 2 : 0);
    if (start + n > count) die("the text has %d tokens; it needs %d", count, start + n);
    Diff d = decoded_against_prefilled(tokens + start, n, m);
    char what[64];
    snprintf(what, sizeof what, "prompt at token %d, %d decoded", start, m);
    print_diff(what, d);
    worst.logits = fmin(worst.logits, d.logits), worst.comp = fmin(worst.comp, d.comp);
    worst.keys = fmin(worst.keys, d.keys), worst.ring = fmin(worst.ring, d.ring);
    worst.cstate = fmin(worst.cstate, d.cstate), worst.dring = fmin(worst.dring, d.dring);
    worst.history_equal &= d.history_equal, worst.argmax_equal &= d.argmax_equal;
  }
  print_diff("lowest", worst);
  return 0;
}

// ---- --fork: the conversation, its forks and their checks ----

enum { NEXT = 16, REPLY1 = 256, REPLY2 = 128, MAXSAVED = 16 };

static int *conv, nconv, prompt1, reply1, prompt2, reply2;  // the parts' ends
static int prompt_only;
static State *source;
static uint8_t *source_comp[LAYERS], *source_keys[LAYERS];  // its compressed rows and index keys, for --ced
static struct {
  int pos;
  Snapshot *snap;
} saved[MAXSAVED];
static int nsaved;

// A fork's state after its prompt and the 16 greedy tokens after it.
typedef struct {
  Tensors t;
  int next[NEXT];
} Outcome;

static int failures;

static void fail(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
static void fail(const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  printf("  FAIL: ");
  vprintf(fmt, ap);
  printf("\n");
  va_end(ap);
  fflush(stdout);
  failures++;
}

static int same_next(const Outcome *a, const Outcome *b) {
  return !memcmp(a->next, b->next, sizeof a->next);
}

static int same_count(const Outcome *a, const Outcome *b) {
  int i = 0;
  while (i < NEXT && a->next[i] == b->next[i]) i++;
  return i;
}

// The bytes of the compressed rows and index keys that positions before p
// completed in which t differs from the source's.
static size_t differ_reused(const Tensors *t, int p) {
  size_t bad = 0;
  for (int l = 0; l < LAYERS; l++) {
    if (!L[l].kv_source) continue;
    size_t rows = (size_t)(p / L[l].ratio);
    bad += differ(t->comp[l], source_comp[l], rows * HEAD_DIM * 2) + differ(t->keys[l], source_keys[l], rows * IDX_DIM * 2);
  }
  return bad;
}

static Outcome finish(const float *lg, int n) {
  Outcome o = {0};
  take(&o.t, lg);
  for (int i = 0; i < NEXT; i++) {
    int t = o.next[i] = sample_argmax(lg, VOCAB);
    lg = model_forward(&t, 1, n + i);
  }
  return o;
}

static State *fresh(void) {
  State *s = model_state_new();
  model_state_use(s);
  model_reset();
  model_state_use(source);
  return s;
}

static void done_with(State *s) {
  model_state_use(source);
  model_state_free(s);
}

// The fork's prompt with no cache: a cold prefill from position 0.
static Outcome cold(const int *f, int n) {
  State *c = fresh();
  model_state_use(c);
  Outcome o = finish(model_prefill(f, n, 0, 0, -1, 0), n);
  done_with(c);
  return o;
}

// A later fork at the branch point p, resumed from the snapshot that the
// fork in state s captured there, over s's rows: it must give that fork's
// bytes. The server keeps no snapshot of the encoder layers only (--ced).
static void check_branch(State *s, Snapshot *branch, const int *f, int n, int p, const Outcome *first, const char *what) {
  if (!branch) return;
  if (model_snapshot_extent(branch) != LAYERS - 1) {
    printf("  %s: the branch snapshot at %d holds the encoder layers only\n", what, p);
    return;
  }
  State *q = fresh();
  model_state_copy(q, s, p);
  model_state_use(q);
  model_snapshot_restore(q, branch);
  Outcome o = finish(model_prefill(f, n, p, 0, -1, 0), n);
  done_with(q);
  size_t bad = differ_state(&o.t, &first->t, n);
  printf("  %s: its branch snapshot at %d resumed against the %s itself: %zu bytes differ, next tokens %s\n", what, p,
         what, bad, same_next(&o, first) ? "equal" : "DIFFERENT");
  if (bad) explain(&first->t, &o.t, n);
  if (bad || !same_next(&o, first)) fail("a resume from the %s's branch snapshot at %d differs from the %s", what, p, what);
  drop(&o.t);
}

// The cache's replay at p (exact, or bounded with --ced): the source's global
// rows before p, then a prefill that rebuilds the rolling rows at p and
// captures them there, as the server does (its branch-point snapshot).
static Outcome replay(const int *f, int n, int p) {
  State *r = fresh();
  model_state_copy(r, source, p);
  model_state_use(r);
  Snapshot *branch = 0;
  Outcome o = finish(model_prefill(f, n, p, 1, p > 0 ? p : -1, &branch), n);
  check_branch(r, branch, f, n, p, &o, "replay");
  model_snapshot_free(branch);
  done_with(r);
  return o;
}

// The cache's resume from the source's snapshot at b <= p, which captures the
// rolling rows at p when b < p (the server's branch point after a snapshot).
static Outcome resume(const int *f, int n, int k, int p) {
  int b = saved[k].pos;
  State *q = fresh();
  model_state_copy(q, source, b);
  model_state_use(q);
  model_snapshot_restore(q, saved[k].snap);
  Snapshot *branch = 0;
  Outcome o = finish(model_prefill(f, n, b, 0, b < p ? p : -1, b < p ? &branch : 0), n);
  check_branch(q, branch, f, n, p, &o, "snapshot resume");
  model_snapshot_free(branch);
  done_with(q);
  return o;
}

// The conversation itself continued at b (b > prompt1): its history from the
// first prompt's end computed again as build() computed it (the first reply
// decoded, the tool result prefilled, the second reply decoded) up to b, then
// the fork's tokens from b prefilled in that same state.
static Outcome continued(const int *f, int n, int b) {
  int k = 0;
  while (saved[k].pos != prompt1) k++;
  State *c = fresh();
  model_state_copy(c, source, prompt1);
  model_state_use(c);
  model_snapshot_restore(c, saved[k].snap);
  model_seek(conv, prompt1);
  for (int i = prompt1; i < b && i < reply1; i++) model_forward(conv + i, 1, i);
  if (b > reply1) model_prefill(conv, b < prompt2 ? b : prompt2, reply1, 0, -1, 0);
  for (int i = prompt2; i < b; i++) model_forward(conv + i, 1, i);
  Outcome o = finish(model_prefill(f, n, b, 0, -1, 0), n);
  done_with(c);
  return o;
}

static void keep_source_rows(void) {
  gpu_sync();
  for (int l = 0; l < LAYERS; l++)
    if (L[l].kv_source)
      source_comp[l] = host_copy(L[l].comp.ptr, L[l].comp.size), source_keys[l] = host_copy(L[l].keys.ptr, L[l].keys.size);
}

static void snapshot_source(int pos, int with_logits) {
  if (nsaved == MAXSAVED) die("too many snapshots");
  saved[nsaved].pos = pos, saved[nsaved++].snap = model_snapshot(source, pos, with_logits);
}

// The conversation: the prompt, a decoded reply, a tool-like result, a second
// decoded reply. Snapshots at both prompts' ends (with logits), the replies'
// ends, and some positions inside the replies.
static void build(const int *prompt, int n) {
  static const char *result =
      "\n\nTool result:\n$ make -s server-check\ntest result: ok. 23 passed; 0 failed\n$ git status --short\n"
      " M src/engine/model.c\n M src/engine/myds4.h\n?? tools/checks/prefix_fork_check.c\n"
      "The build is clean. Summarize what changed and what is left to check, in two sentences.\n";
  int extra[1024], ne = tok_encode(result, extra, 1024);
  conv = xmalloc((size_t)(n + REPLY1 + ne + REPLY2) * sizeof *conv);
  memcpy(conv, prompt, (size_t)n * sizeof *conv);
  source = model_state_new();
  model_state_use(source);
  model_reset();
  const float *lg = model_prefill(conv, n, 0, 0, -1, 0);
  prompt1 = n;
  snapshot_source(prompt1, 1);
  if (prompt_only) {  // --prompt-only: the prompt alone, for quick checks of prompt rows
    reply1 = prompt2 = reply2 = nconv = prompt1;
    keep_source_rows();
    printf("conversation: prompt 0..%d only\n", prompt1);
    return;
  }
  const int inside1[] = {prompt1 + 1, prompt1 + REPLY1 / 2}, inside2[] = {1, REPLY2 / 2};
  for (int i = 0; i < REPLY1; i++) {
    int pos = prompt1 + i;
    conv[pos] = sample_argmax(lg, VOCAB);
    lg = model_forward(conv + pos, 1, pos);
    for (int k = 0; k < 2; k++)
      if (pos + 1 == inside1[k]) snapshot_source(pos + 1, 0);
  }
  reply1 = prompt1 + REPLY1;
  snapshot_source(reply1, 0);
  memcpy(conv + reply1, extra, (size_t)ne * sizeof *conv);
  prompt2 = reply1 + ne;
  lg = model_prefill(conv, prompt2, reply1, 0, -1, 0);
  snapshot_source(prompt2, 1);
  for (int i = 0; i < REPLY2; i++) {
    int pos = prompt2 + i;
    conv[pos] = sample_argmax(lg, VOCAB);
    lg = model_forward(conv + pos, 1, pos);
    for (int k = 0; k < 2; k++)
      if (pos + 1 == prompt2 + inside2[k]) snapshot_source(pos + 1, 0);
  }
  reply2 = prompt2 + REPLY2;
  snapshot_source(reply2, 0);
  nconv = reply2;
  keep_source_rows();
  printf("conversation: prompt 0..%d, reply %d..%d, result %d..%d, reply %d..%d; %d snapshots\n", prompt1, prompt1,
         reply1, reply1, prompt2, prompt2, reply2, nsaved);
  fflush(stdout);
}

// --ced: the bounded replay at p and the resume from the latest snapshot at
// or before it, against the cold prefill (compacted too).
static void check_fork_ced(const int *f, int n, int p) {
  Outcome c = cold(f, n), r = replay(f, n, p);
  size_t kept = differ_reused(&r.t, p);
  print_diff("  bounded replay against cold", compare(&c.t, &r.t, 0, n));
  printf("  bounded replay: %zu bytes of the reused rows differ from the source's; %d of the next %d tokens equal "
         "to cold's\n", kept, same_count(&r, &c), NEXT);
  if (kept) fail("the bounded replay at %d changed the compressed rows or index keys before it", p);
  if (!finite(r.t.logits, VOCAB)) fail("the bounded replay at %d gives logits that are not finite", p);
  int k = -1;
  for (int i = 0; i < nsaved; i++)
    if (saved[i].pos <= p && (k < 0 || saved[i].pos > saved[k].pos)) k = i;
  if (k >= 0 && saved[k].pos < n) {
    Outcome q = resume(f, n, k, p);
    char what[64];
    snprintf(what, sizeof what, "  snapshot at %d against cold", saved[k].pos);
    print_diff(what, compare(&c.t, &q.t, 0, n));
    printf("  snapshot at %d: %d of the next %d tokens equal to cold's\n", saved[k].pos, same_count(&q, &c), NEXT);
    if (!finite(q.t.logits, VOCAB)) fail("the snapshot resume at %d gives logits that are not finite", p);
    drop(&q.t);
  }
  drop(&c.t), drop(&r.t);
}

static void check_fork(int p, const int *suffix, int ns) {
  int n = p + ns, *f = xmalloc((size_t)n * sizeof *f);
  memcpy(f, conv, (size_t)p * sizeof *f);
  memcpy(f + p, suffix, (size_t)ns * sizeof *f);
  int decoded = p > prompt1;  // the history before p holds decoded rows
  printf("fork at %d (%s):\n", p, decoded ? "after decoded rows" : "prompt rows only");
  if (opt.ced) {
    check_fork_ced(f, n, p);
    free(f);
    return;
  }
  Outcome c = cold(f, n), r = replay(f, n, p), o = {0};  // o: with decoded rows, the conversation continued at p
  if (!decoded) {
    size_t bad = differ_state(&r.t, &c.t, n);
    printf("  replay against cold: %zu bytes differ, next %d tokens %s\n", bad, NEXT, same_next(&r, &c) ? "equal" : "DIFFERENT");
    if (bad) explain(&c.t, &r.t, n);
    if (bad || !same_next(&r, &c)) fail("replay at %d differs from the cold prefill", p);
  } else {
    o = continued(f, n, p);
    Diff dc = compare(&c.t, &o.t, prompt1, n), dr = compare(&c.t, &r.t, prompt1, n);
    print_diff("  continued against cold", dc);
    print_diff("  replay against cold", dr);
    if (!near(dr, dc)) fail("replay at %d is further from the cold prefill than the conversation continued there", p);
  }
  // the snapshot method: the latest snapshot at or before p
  int k = -1;
  for (int i = 0; i < nsaved; i++)
    if (saved[i].pos <= p && (k < 0 || saved[i].pos > saved[k].pos)) k = i;
  if (k >= 0 && saved[k].pos < n) {
    Outcome q = resume(f, n, k, p);
    if (!decoded) {
      size_t bad = differ_state(&q.t, &c.t, n);
      printf("  snapshot at %d against cold: %zu bytes differ, next tokens %s\n", saved[k].pos, bad,
             same_next(&q, &c) ? "equal" : "DIFFERENT");
      if (bad) explain(&c.t, &q.t, n);
      if (bad || !same_next(&q, &c)) fail("snapshot resume at %d differs from the cold prefill", p);
    } else {
      Diff d = compare(&c.t, &q.t, prompt1, n);
      char what[64];
      snprintf(what, sizeof what, "  snapshot at %d against cold", saved[k].pos);
      print_diff(what, d);
      Outcome b = saved[k].pos == p ? o : continued(f, n, saved[k].pos);
      size_t bad = differ_state(&q.t, &b.t, n);
      printf("  snapshot at %d against the conversation continued there: %zu bytes differ, next tokens %s\n",
             saved[k].pos, bad, same_next(&q, &b) ? "equal" : "DIFFERENT");
      if (bad) explain(&b.t, &q.t, n);
      if (bad || !same_next(&q, &b)) fail("snapshot resume at %d differs from the conversation continued there", p);
      if (saved[k].pos != p) drop(&b.t);
    }
    drop(&q.t);
  }
  if (decoded) drop(&o.t);
  drop(&c.t), drop(&r.t);
  free(f);
}

static int only[64], nonly;  // --at P: these fork positions only
static int suffix_len;       // --suffix N: the first N tokens of the suffix only

// The forks' divergent suffix: a fixed question, or its first --suffix tokens.
static int fork_suffix(int *suffix, int max) {
  int ns = tok_encode(" Ignore the rest and answer in five words: what is this about?", suffix, max);
  return suffix_len > 0 && suffix_len < ns ? suffix_len : ns;
}

static int fork_test(const int *prompt, int n, int randoms) {
  build(prompt, n);
  int suffix[64], ns = fork_suffix(suffix, 64);
  if (nonly) {
    for (int i = 0; i < nonly; i++)
      if (only[i] > 0 && only[i] <= nconv) check_fork(only[i], suffix, ns);
    printf("%d forks, %d failures\n", nonly, failures);
    return failures != 0;
  }
  int classes[] = {1, 2, 100, MB - 1, MB, MB + 1, 2 * MB - 1, 2 * MB, 2 * MB + 1, prompt1 / 2, prompt1 - 1, prompt1,
                   prompt1 + 1, prompt1 + REPLY1 / 2, reply1 - 1, reply1, reply1 + 1, (reply1 + prompt2) / 2, prompt2 - 1,
                   prompt2, prompt2 + 1, prompt2 + REPLY2 / 2, reply2 - 1, reply2};
  int positions[64], np = 0;
  for (size_t i = 0; i < sizeof classes / sizeof *classes; i++)
    if (classes[i] > 0 && classes[i] <= nconv) positions[np++] = classes[i];
  uint64_t x = 0x9E3779B97F4A7C15ull;
  for (int i = 0; i < randoms && np < 64; i++) {
    x ^= x << 13, x ^= x >> 7, x ^= x << 17;
    positions[np++] = 1 + (int)(x % (uint64_t)nconv);
  }
  for (int i = 0; i < np; i++) check_fork(positions[i], suffix, ns);
  printf("%d forks, %d failures\n", np, failures);
  return failures != 0;
}

int main(int argc, char **argv) {
  if (argc < 4) die("usage: prefix-fork-check CHECKPOINT --text FILE [--rendered] [--experts GGUF] [--draft N] [--cache GB] [--ced N] [--limit N] (--measure | --fork [--random N] [--at P]... [--prompt-only] [--suffix N])");
  Options o = {.ctx = 32768, .budget_gb = 40, .reserve_gb = 4, .dump_fd = -1, .routes_fd = -1, .quiet = 1};
  const char *text = 0;
  int mode = 0, rendered = 0, randoms = 16, limit = 0;
  for (int i = 2; i < argc; i++) {
    if (!strcmp(argv[i], "--text") && i + 1 < argc) text = argv[++i];
    else if (!strcmp(argv[i], "--limit") && i + 1 < argc) limit = atoi(argv[++i]);  // the text's first N tokens only
    else if (!strcmp(argv[i], "--rendered")) rendered = 1;
    else if (!strcmp(argv[i], "--experts") && i + 1 < argc) o.experts = argv[++i];
    else if (!strcmp(argv[i], "--draft") && i + 1 < argc) o.draft = atoi(argv[++i]), o.draft_cache = o.draft ? 32 : 0;
    else if (!strcmp(argv[i], "--cache") && i + 1 < argc) o.budget_gb = atof(argv[++i]);
    else if (!strcmp(argv[i], "--ced") && i + 1 < argc) o.ced = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--random") && i + 1 < argc) randoms = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--at") && i + 1 < argc && nonly < 64) only[nonly++] = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--prompt-only")) prompt_only = 1;
    else if (!strcmp(argv[i], "--suffix") && i + 1 < argc) suffix_len = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--measure")) mode = 1;
    else if (!strcmp(argv[i], "--fork")) mode = 2;
    else die("invalid check argument: %s", argv[i]);
  }
  if (!text || !mode) die("--text FILE and --measure or --fork is required");
  tok_load(argv[1]);
  int count, *tokens = read_text(text, rendered, &count);
  if (limit > 0 && count > limit) count = limit;
  model_load(argv[1], &o);
  return mode == 1 ? measure(tokens, count, rendered) : fork_test(tokens, count, randoms);
}
