// The engine behind the C ABI of engine.h: the process's log and stop, and
// for each model (src/model/model.h) its loading, its work on states and
// snapshots, sampling, context-copy drafts, images, statistics and its
// tokenizer. Requests, their scheduling and the KV prefix cache are the
// server's (server/src/worker.rs, server/src/kv.rs).
#include "../model/deepseek/model.h"
#include <math.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdlib.h>
#include <stdatomic.h>
#include <string.h>
#include <sys/qos.h>
#include <sys/time.h>
#include <time.h>

#include "engine.h"
#include "myds4.h"
#include "pass.h"
#include "state_file.h"
#include "../model/model.h"
#include "../model/draft.h"

struct myds4_sampler {
  Sampler sp;
};

struct myds4_model {
  Model *m;
  const ModelOps *ops;
  Tokenizer *tk;
  Options opt;
  char *dir;
  atomic_int loaded;
  uint8_t identity[32];  // myds4_state_identity's, once known
  int identified;
};

struct myds4_media {
  myds4_model *m;
  Media *items;
  int n;
};

static myds4_process proc;
static int initialized;
static const ModelOps *families[8];  // the families loaded: one model each
static int nfamilies;

// The log is say() in util.c: one line per event, to the caller's on_log
// (log_hook) or stderr.
void myds4_log(const char *line) { say("%s", line); }

// ---- the process ----

int myds4_init(const myds4_process *p) {
  if (!p || p->abi != MYDS4_ABI || initialized) return -1;
  if (p->reserve_gb && !(p->reserve_gb >= CACHE_RESERVE_MIN && p->reserve_gb <= CACHE_RESERVE_MAX)) return -1;
  memory_configure(p->memory_gb, p->reserve_gb);
  proc = *p;
  die_hook = p->on_die;
  log_hook = p->on_log;
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);  // the models' GPU work goes before background work
  shutdown_watch();  // an interrupt while loading still saves a predictor's state
  initialized = 1;
  return 0;
}

int myds4_stopping(void) { return shutdown_requested(); }

int myds4_stop_requests(void) { return shutdown_requests(); }

void myds4_request_stop(void) { shutdown_request(); }

// ---- models ----

static char *copy(const char *s) { return s ? strdup(s) : 0; }

myds4_model *myds4_open(const myds4_config *c) {
  if (!initialized || !c || c->abi != MYDS4_ABI || !c->model) return NULL;
  const ModelOps *ops = model_find(c->model);
  myds4_architecture architecture = ops == &deepseek_ops ? MYDS4_DEEPSEEK : ops == &nemotron_ops ? MYDS4_NEMOTRON : MYDS4_QWEN;
  if (c->architecture != architecture) {
    say("%s: option group does not match checkpoint architecture", c->model);
    return NULL;
  }
  const myds4_deepseek_config empty = {0};
  const myds4_deepseek_config *d = architecture == MYDS4_DEEPSEEK ? &c->options.deepseek : &empty;
  Options options = {.family = c->architecture, .ctx = c->ctx, .lookup = c->lookup, .dump_fd = -1, .routes_fd = -1, .quiet = 1,
    .deepseek.cache_half_life = d->cache_half_life, .deepseek.cache_read_cost = d->cache_read_cost,
    .draft = architecture == MYDS4_DEEPSEEK ? d->draft : architecture == MYDS4_NEMOTRON ? c->options.nemotron.draft : c->options.qwen.draft,
    .deepseek.draft_cache = d->draft_cache, .deepseek.predictor_state = d->predictor_state, .deepseek.substitute = d->substitute,
    .deepseek.experts = d->experts, .deepseek.online_predictor = d->online_predictor || d->predictor_state ? 64 : 0,
    .deepseek.ced = d->ced, .deepseek.prefill_prune = d->prefill_prune, .deepseek.runahead = d->runahead, .deepseek.pipeline = d->pipeline};
  if (architecture == MYDS4_NEMOTRON) options.nemotron.draft_model = c->options.nemotron.draft_model;
  const char *bad = options.ctx < 1 || options.ctx > 1 << 20 ? "ctx takes 1 to 1048576 tokens"
    : options.draft < 0 || options.draft > (architecture == MYDS4_NEMOTRON ? 15 : BLOCK) ? "draft exceeds this model's proposal limit"
    : options.lookup < 0 || options.lookup > VERIFY_DRAFTS ? "lookup takes off or 1 to 15 proposals"
    : d->draft_cache && !d->draft ? "draft-cache needs draft"
    : d->draft_cache && (d->draft_cache < DRAFT_CACHE_MIN || d->draft_cache > DRAFT_EXPERTS) ? "draft-cache takes 15 to 128 slots"
    : d->ced && (d->ced < WINDOW || d->ced > 1 << 20) ? "ced takes off or 128 to 1048576 replay tokens"
    : d->cache_half_life < 0 || d->cache_half_life > CACHE_HALF_LIFE_MAX ? "cache-half-life takes 1 to 1000000 passes"
    : d->runahead && (d->draft || d->online_predictor || d->substitute > 0) ? "runahead requires draft off, no online predictor and no substitution"
    : d->pipeline && (d->experts || d->runahead || d->draft || d->online_predictor || d->predictor_state || d->substitute > 0)
      ? "pipeline requires native MXFP4, no runahead, draft, online predictor or substitution" : ops->refuse(&options);
  if (bad) { say("%s: %s", c->model, bad); return NULL; }
  for (int i = 0; i < nfamilies; i++) if (families[i] == ops) {
    say("%s: a %s model is registered already", c->model, ops->family);
    return NULL;
  }
  myds4_model *x = xmalloc(sizeof *x);
  atomic_init(&x->loaded, 0);
  x->ops = ops;
  x->opt = options;
  if (architecture == MYDS4_DEEPSEEK) {
    x->opt.deepseek.predictor_state = copy(options.deepseek.predictor_state);
    x->opt.deepseek.experts = copy(options.deepseek.experts);
  } else if (architecture == MYDS4_NEMOTRON) x->opt.nemotron.draft_model = copy(options.nemotron.draft_model);
  families[nfamilies++] = ops;
  x->tk = tok_open(c->model, ops->media_token);
  x->dir = copy(c->model);
  x->m = model_create(x->dir);
  say("%s: registered %s; weights load on first inference request", c->model, ops->family);
  return x;
}

int myds4_loaded(const myds4_model *m) {
  if (!atomic_load_explicit(&m->loaded, memory_order_acquire)) return 0;
  MemorySlot *slot = m->ops->weight_slot ? m->ops->weight_slot() : NULL;
  return !slot || memory_ready(slot);
}
void myds4_active(myds4_model *m, int active) { model_active(m->m, active); }


int myds4_load(myds4_model *m) {
  int was_loaded = myds4_loaded(m);
  if (!model_load(m->m, &m->opt)) {
    model_active(m->m, 0);
    MemoryUsage u = memory_usage();
    uint64_t limit = memory_current_limit();
    if (shutdown_requested()) say("%s: loading stopped", m->dir);
    else say("%s: not loaded: it does not fit; %.1f GiB more model memory fits now (%.1f GiB held, %.1f GiB of KV)",
             m->dir, (limit > u.committed ? limit - u.committed : 0) / 1073741824.0, u.committed / 1073741824.0,
             u.kv / 1073741824.0);
    return 0;
  }
  model_active(m->m, 1);
  atomic_store_explicit(&m->loaded, 1, memory_order_release);
  if (!was_loaded && m->ops == &deepseek_ops) say("%s: %s, context %d, drafts %d, rewind margin %d, context lookup %d; prefill: ced_depth=%d (%s), dependency "
      "pruning=%s",
      m->dir, m->ops->family, m->opt.ctx, m->ops->draft_block(), m->ops->rewind(), m->opt.lookup, m->opt.deepseek.ced,
      m->opt.deepseek.ced ? "approximate" : "off", m->opt.deepseek.prefill_prune ? "on" : "off");
  else if (!was_loaded) say("%s: %s, context %d, drafts %d", m->dir, m->ops->family, m->opt.ctx, m->ops->draft_block());
  return 1;
}

void myds4_finish(myds4_model *m) {
  if (atomic_load_explicit(&m->loaded, memory_order_acquire) && m->ops->finish) m->ops->finish();
}

int myds4_idle(myds4_model *m) {
  if (!atomic_load_explicit(&m->loaded, memory_order_acquire)) return 0;
  memory_scope(&m->m->work);
  ModelCounters c;
  m->ops->counters(&c);
  int before = c.cache_slots, after = m->ops->idle();
  if (after < before) say("idle: memory short, expert cache %d -> %d slots", before, after);
  return after;
}

int myds4_max_rows(const myds4_model *m) { return m->ops->max_rows(); }
int myds4_draft_block(const myds4_model *m) { return myds4_loaded(m) ? m->ops->draft_block() : m->opt.draft; }
int myds4_max_verify(const myds4_model *m) { return m->ops->max_verify(); }
int myds4_rewind(const myds4_model *m) { return m->ops->rewind(); }
int myds4_vocab(const myds4_model *m) { return m->ops->vocab(); }
int myds4_stop(const myds4_model *m, int32_t token) { return m->ops->stop(token); }
int myds4_prefill_rows(const myds4_model *m) { return m->ops->prefill_rows(); }

// ---- attention states and snapshots ----

myds4_state *myds4_state_new(myds4_model *m) { memory_scope(&m->m->work); return m->ops->state_new(); }
void myds4_state_free(myds4_model *m, myds4_state *s) { m->ops->state_free(s); }
size_t myds4_state_bytes(const myds4_model *m, const myds4_state *s, int positions) {
  return m->ops->state_bytes(s, positions);
}
void myds4_state_copy(myds4_model *m, myds4_state *dst, const myds4_state *src, int positions) {
  memory_scope(&m->m->work);
  m->ops->state_copy(dst, src, positions);
}

myds4_snapshot *myds4_snapshot_take(myds4_model *m, myds4_state *s, int pos, const float *logits) {
  memory_scope(&m->m->work);
  return m->ops->snapshot_take(s, pos, logits);
}
void myds4_snapshot_restore(myds4_model *m, myds4_state *s, const myds4_snapshot *x) {
  memory_scope(&m->m->work);
  m->ops->snapshot_restore(s, x);
}
void myds4_snapshot_free(myds4_model *m, myds4_snapshot *x) { m->ops->snapshot_free(x); }
size_t myds4_snapshot_bytes(const myds4_model *m, int with_logits) { return m->ops->snapshot_bytes(with_logits); }
const float *myds4_snapshot_logits(const myds4_model *m, const myds4_snapshot *x) { return m->ops->snapshot_logits(x); }

// ---- saved states ----

int64_t myds4_state_write(myds4_model *m, myds4_state *s, int from, int to, int fd) {
  memory_scope(&m->m->work);
  return state_file_write(m->ops, s, from, to, fd);
}
int myds4_state_read(myds4_model *m, myds4_state *s, int positions, const int *fds, int count) {
  memory_scope(&m->m->work);
  return state_file_read(m->ops, s, positions, fds, count);
}
int64_t myds4_snapshot_write(myds4_model *m, const myds4_snapshot *x, myds4_state *s, int pos, int fd) {
  memory_scope(&m->m->work);
  return snapshot_file_write(m->ops, (myds4_snapshot *)x, s, pos, fd);
}
myds4_snapshot *myds4_snapshot_read(myds4_model *m, int fd) {
  memory_scope(&m->m->work);
  return snapshot_file_read(m->ops, fd);
}

static void identity_number(CC_SHA256_CTX *h, int64_t v) { CC_SHA256_Update(h, &v, sizeof v); }

void myds4_state_identity(myds4_model *m, uint8_t out[32]) {
  if (!m->identified) {
    CC_SHA256_CTX h;
    CC_SHA256_Init(&h);
    static const char tag[] = "myds4 saved state";
    CC_SHA256_Update(&h, tag, sizeof tag);
    CC_SHA256_Update(&h, m->ops->family, (CC_LONG)strlen(m->ops->family) + 1);
    identity_number(&h, m->ops->state_version);
    identity_folder(&h, m->dir);
    identity_number(&h, m->opt.draft);
    identity_number(&h, m->opt.lookup);
    if (m->ops == &deepseek_ops) {
      const DeepSeekOptions *d = &m->opt.deepseek;
      uint32_t substitute;
      memcpy(&substitute, &d->substitute, sizeof substitute);
      identity_number(&h, d->ced);
      identity_number(&h, d->prefill_prune);
      identity_number(&h, substitute);
      identity_number(&h, d->experts != NULL);
      if (d->experts) identity_gguf_set(&h, d->experts);
    }
    if (m->ops == &nemotron_ops) {
      identity_number(&h, m->opt.nemotron.draft_model != NULL);
      if (m->opt.nemotron.draft_model) identity_folder(&h, m->opt.nemotron.draft_model);
    }
    CC_SHA256_Final(m->identity, &h);
    m->identified = 1;
  }
  memcpy(out, m->identity, 32);
}

// ---- running a model ----

const float *myds4_prefill(myds4_model *m, myds4_state *s, const int32_t *tokens, int n, int from, int replay,
                           int capture, myds4_snapshot **captured, myds4_prefill_info *info, const myds4_media *images) {
  memory_scope(&m->m->work);
  if (captured) *captured = 0;
  int start = replay && from > 0 ? m->ops->replay_start(from) : from;
  if (m->ops->media) m->ops->media(images ? images->items : NULL, images ? images->n : 0);
  const float *lg = m->ops->prefill(s, tokens, n, from, replay, capture, captured, 0);
  if (info) {  // the family's plan, or every row through every layer
    ModelPlan p = {.chunks = 1, .tokens = n - start, .decoder_tokens = n - start};
    if (m->ops->plan) m->ops->plan(&p);
    *info = (myds4_prefill_info){.chunks = p.chunks, .tokens = p.tokens, .decoder_tokens = p.decoder_tokens,
                                 .ced_applied = p.ced_applied, .prune_applied = p.prune_applied,
                                 .prefix_tokens = p.prefix_tokens};
  }
  return lg;
}

// ---- streams ----

enum { STREAM_ROWS = 16 };
typedef enum { STREAM_NONE, STREAM_PREFILL, STREAM_DECODE, STREAM_VERIFY, STREAM_QUIT } StreamCall;

struct myds4_stream {
  myds4_model *m;
  Pass *pass;
  StreamCall call;  // the call posted and not finished
  int finished;     // it finished, and myds4_run has not reported that yet
  // the call's arguments
  myds4_state *states[STREAM_ROWS];
  int32_t row_tokens[STREAM_ROWS], row_pos[STREAM_ROWS];
  const int32_t *tokens;  // a prefill's prompt; a verify's rows (the token, then the drafts)
  int n, from, replay, capture, count, drafts, proposed, pos;
  const myds4_media *images;
  myds4_sampler *sampler;
  // its results
  const float *logits;
  myds4_snapshot *captured;
  myds4_prefill_info info;
  int32_t next;
  int accepted;
};

static int stream_has_call(void *context) { return ((myds4_stream *)context)->call != STREAM_NONE; }

static void stream_run(myds4_stream *x) {
  myds4_model *m = x->m;
  x->logits = NULL;
  switch (x->call) {
    case STREAM_PREFILL:
      x->logits = myds4_prefill(m, x->states[0], x->tokens, x->n, x->from, x->replay, x->capture, &x->captured, &x->info, x->images);
      break;
    case STREAM_DECODE:
      x->logits = myds4_rows(m, x->states, x->row_tokens, x->row_pos, x->count);
      break;
    case STREAM_VERIFY: {
      int32_t *rows = (int32_t *)x->tokens;
      const double *q = x->proposed ? NULL : myds4_draft(m, x->states[0], rows[0], x->pos, rows + 1, x->drafts, x->sampler);
      x->logits = myds4_verify(m, x->states[0], rows, x->drafts + 1, x->pos);
      if (x->logits) {
        x->next = myds4_sample_verify(m, x->logits, rows + 1, x->drafts, q, x->sampler, &x->accepted);
        myds4_accept(m, x->states[0], rows, x->accepted + 1, x->pos);
      }
      break;
    }
    default:
      break;
  }
}

// The stream's pass: a call after another, with its own buffers; between
// calls no state stays bound to it.
static void stream_body(void *argument) {
  myds4_stream *x = argument;
  x->m->ops->pass_begin();
  for (;;) {
    pass_wait(&(PassWait){stream_has_call, NULL, x});
    if (x->call == STREAM_QUIT) break;
    stream_run(x);
    if (x->m->ops->pass_release) x->m->ops->pass_release();
    x->call = STREAM_NONE;
    x->finished = 1;
  }
  x->m->ops->pass_end();
}

myds4_stream *myds4_stream_new(myds4_model *m) {
  myds4_stream *x = xmalloc(sizeof *x);
  x->m = m;
  memory_scope(&m->m->work);  // the scope the pass keeps as its own
  x->pass = m->ops->pass_new(stream_body, x);
  return x;
}

void myds4_stream_free(myds4_stream *x) {
  if (!x) return;
  x->call = STREAM_QUIT;
  pass_run(x->pass);
  pass_free(x->pass);
  free(x);
}

void myds4_stream_prefill(myds4_stream *x, myds4_state *s, const int32_t *tokens, int n, int from, int replay, int capture,
                          const myds4_media *images) {
  x->states[0] = s, x->tokens = tokens, x->n = n, x->from = from, x->replay = replay, x->capture = capture, x->images = images;
  x->captured = NULL;
  x->call = STREAM_PREFILL;
}

void myds4_stream_rows(myds4_stream *x, myds4_state **states, const int32_t *tokens, const int32_t *pos, int count) {
  if (count < 1 || count > STREAM_ROWS) die("a stream's decoding call takes 1 to %d rows", STREAM_ROWS);
  memcpy(x->states, states, (size_t)count * sizeof *states);
  memcpy(x->row_tokens, tokens, (size_t)count * sizeof *tokens);
  memcpy(x->row_pos, pos, (size_t)count * sizeof *pos);
  x->count = count;
  x->call = STREAM_DECODE;
}

void myds4_stream_verify(myds4_stream *x, myds4_state *s, int32_t *rows, int drafts, int proposed, int pos, myds4_sampler *sampler) {
  x->states[0] = s, x->tokens = rows, x->drafts = drafts, x->proposed = proposed, x->pos = pos, x->sampler = sampler;
  x->accepted = 0;
  x->call = STREAM_VERIFY;
}

// The streams' calls advance on this thread: each stream that can go on runs
// until it would wait, and a thread that finds none ready looks again at once
// (a GPU wait is mostly shorter than the time a blocking wait takes to wake up)
// and after a moment sleeps briefly between its looks.
enum { RUN_SLEEP_US = 50 };
#define RUN_SPIN_SECONDS 1e-4
int myds4_run(myds4_stream *const *streams, int n, double seconds) {
  static unsigned first;
  double started = now(), deadline = started + seconds, idle_since = started;
  int posted = 0;
  for (int k = 0; k < n; k++) posted += streams[k]->call != STREAM_NONE;
  __atomic_store_n(&gpu_other_passes, posted - 1, __ATOMIC_RELAXED);
  for (;;) {
    int ran = 0;
    for (int k = 0; k < n; k++) {
      int i = (int)(((unsigned)first + (unsigned)k) % (unsigned)n);
      myds4_stream *x = streams[i];
      if (pass_ready(x->pass)) pass_resume(x->pass), ran = 1;
      if (x->finished) {
        x->finished = 0;
        first = (unsigned)i + 1;
        return i;
      }
    }
    double t = now();
    if (t >= deadline) return -1;
    if (ran) idle_since = t;
    // A short wait is polled, a longer one sleeps between looks: a spinning
    // core takes power and memory bandwidth that the GPU would otherwise use.
    else if (t - idle_since < RUN_SPIN_SECONDS) __asm__ __volatile__("yield");
    else usleep(RUN_SLEEP_US);
  }
}

const float *myds4_stream_logits(const myds4_stream *x) { return x->logits; }
myds4_snapshot *myds4_stream_captured(const myds4_stream *x) { return x->captured; }
void myds4_stream_info(const myds4_stream *x, myds4_prefill_info *info) { *info = x->info; }
int32_t myds4_stream_next(const myds4_stream *x, int *accepted) {
  if (accepted) *accepted = x->accepted;
  return x->next;
}

int myds4_replay_rows(const myds4_model *m, int k) { return m->ops->replay_rows(k); }
int myds4_replay_start(const myds4_model *m, int k) { return m->ops->replay_start(k); }

const float *myds4_rows(myds4_model *m, myds4_state **states, const int32_t *tokens, const int32_t *pos, int count) {
  memory_scope(&m->m->work);
  return m->ops->rows(states, tokens, pos, count);
}

const double *myds4_draft(myds4_model *m, myds4_state *s, int32_t token, int pos, int32_t *drafts, int n, myds4_sampler *sampler) {
  memory_scope(&m->m->work);
  return m->ops->draft(s, token, pos, drafts, n, &sampler->sp);
}

const float *myds4_verify(myds4_model *m, myds4_state *s, const int32_t *rows, int count, int pos) {
  memory_scope(&m->m->work);
  return m->ops->verify(s, rows, count, pos);
}

void myds4_accept(myds4_model *m, myds4_state *s, const int32_t *rows, int n, int pos) {
  memory_scope(&m->m->work);
  m->ops->accept(s, rows, n, pos);
}

void myds4_seek(myds4_model *m, myds4_state *s, const int32_t *tokens, int pos) { memory_scope(&m->m->work); m->ops->seek(s, tokens, pos); }

// ---- sampling ----


myds4_sampler *myds4_sampler_new(double temperature, double top_p, int top_k, uint64_t seed, int seed_set) {
  Sampler sp = {.temp = temperature, .top_p = top_p, .top_k = top_k};
  if (!sampler_valid(&sp)) return 0;
  myds4_sampler *s = xmalloc(sizeof *s);
  s->sp = sp;
  if (seed_set) sampler_seed(&s->sp, seed);
  else arc4random_buf(s->sp.seed, sizeof s->sp.seed);
  return s;
}

void myds4_sampler_free(myds4_sampler *s) { free(s); }

int32_t myds4_sample(const myds4_model *m, const float *logits, myds4_sampler *s) {
  return sample_token(logits, m->ops->vocab(), &s->sp);
}

int32_t myds4_sample_verify(const myds4_model *m, const float *logits, const int32_t *drafts, int n, const double *q, myds4_sampler *s,
                            int *accepted) {
  return sample_verify_probabilities(logits, m->ops->vocab(), drafts, n, q, m->ops->stop, &s->sp, accepted);
}

// ---- context-copy drafts ----

myds4_lookup *myds4_lookup_new(int capacity) { return context_lookup_new(capacity); }
void myds4_lookup_free(myds4_lookup *x) { context_lookup_free(x); }
int myds4_lookup_append(myds4_lookup *x, const int32_t *tokens, int n) { return context_lookup_append(x, tokens, n); }

int myds4_lookup_propose(const myds4_lookup *x, int max, int32_t *drafts) {
  ContextDraft d = context_lookup_propose(x, max);
  memcpy(drafts, d.tokens, (size_t)d.count * sizeof *drafts);
  return d.count;
}

void myds4_lookup_feedback(myds4_lookup *x, int offered, int accepted) { context_lookup_feedback(x, offered, accepted); }

myds4_draft_choice *myds4_draft_choice_new(void) {
  myds4_draft_choice *x = xmalloc(sizeof *x);
  draft_choice_init(x);
  return x;
}
void myds4_draft_choice_free(myds4_draft_choice *x) { free(x); }
int myds4_draft_choice_count(myds4_draft_choice *x, int most) { return draft_choice_count(x, most); }
void myds4_draft_choice_feedback(myds4_draft_choice *x, int count, int kept, double seconds) {
  draft_choice_feedback(x, count, kept, seconds);
}

// ---- images ----

myds4_media *myds4_media_open(myds4_model *m, const uint8_t *const *data, const size_t *len, int n, char *message,
                                size_t cap) {
  memory_scope(&m->m->work);
  if (!m->ops->encode_media || !m->ops->prepare_media) {
    snprintf(message, cap, "this model has no media encoder");
    return 0;
  }
  myds4_media *x = xmalloc(sizeof *x);
  x->m = m;
  x->items = xmalloc((size_t)n * sizeof *x->items), x->n = 0;
  for (int i = 0; i < n; i++) {
    const char *why;
    if (!m->ops->prepare_media(m->dir, &x->items[i], data[i], len[i], &why)) {
      snprintf(message, cap, "media %d: %s", i + 1, why);
      myds4_media_close(x);
      return 0;
    }
    x->n = i + 1;
  }
  return x;
}

int myds4_media_expand(myds4_media *x, const int32_t *in, int n, int32_t *out, int max) {
  int token = tok_find(x->m->tk, x->m->ops->media_token);
  return media_expand(in, n, x->items, x->n, out, max, token);
}

int myds4_media_encode(myds4_media *x, int from) {
  memory_scope(&x->m->m->work);
  int encoded = 0;  // the encoder runs for the images the prefill reaches; the others are in the state's rows
  for (int i = 0; i < x->n; i++)
    if (x->items[i].start + media_span(&x->items[i]) > from) {
      if (!x->m->ops->encode_media(&x->items[i])) return -1;
      encoded++;
    }
  return encoded;
}

void myds4_media_close(myds4_media *x) {
  if (!x) return;
  x->m->ops->media(0, 0);
  for (int i = 0; i < x->n; i++) media_free(&x->items[i]);
  free(x->items), free(x);
}

// ---- statistics ----

static myds4_cache_counts counts(const CacheStats *c) {
  return (myds4_cache_counts){.hits = c->hits, .misses = c->misses, .prefetch_hits = c->prefetch_hits,
                              .prefetches = c->prefetches, .bytes = c->bytes, .evictions = c->evictions,
                              .late = c->late, .cancelled = c->cancelled, .substituted = c->substituted,
                              .remaps = c->remaps, .given_back = c->given_back, .row_own = c->row_own,
                              .row_ahead = c->row_ahead, .row_late = c->row_late, .row_sub = c->row_sub,
                              .row_ssd = c->row_ssd, .read_s = c->read_s};
}

int myds4_routes(const myds4_model *m, int row, uint16_t *experts, uint8_t *sources) {
  return m->ops->routes ? m->ops->routes(row, experts, sources) : 0;
}

static myds4_stats process_stats;
void myds4_refresh_stats(void) {
  MemInfo mi;
  mem_info(&mi);
  process_stats = (myds4_stats){.footprint = mem_footprint(), .gpu_in_use = mem_gpu(),
    .available = mem_available(), .gpu_limit = mem_gpu_limit(), .total = mi.total,
    .free = mi.free, .file = mi.file, .file_kept = mi.file_kept, .wired = mi.wired,
    .compressor = mi.compressor, .compressed = mi.compressed, .swap_used = mi.swap_used,
    .swapouts = mi.swapouts, .pressure = mi.pressure};
}

uint64_t myds4_reclaim(uint64_t bytes) { return memory_reclaim(bytes); }
uint64_t myds4_unload(myds4_model *m) {
  memory_scope(&m->m->work);
  return memory_unload(&m->m->memory);
}
myds4_memory_info myds4_memory(void) {
  MemoryUsage usage = memory_usage();
  return (myds4_memory_info){usage.limit, usage.allocated, usage.kv};
}
void myds4_memory_limit(uint64_t bytes) { memory_set_limit(bytes); }
void myds4_drain(void) { gpu_drain(); cache_drain(); }

void myds4_stats_get(const myds4_model *m, myds4_stats *s) {
  ModelCounters c = {0};
  if (atomic_load_explicit(&m->loaded, memory_order_acquire)) m->ops->counters(&c);
  *s = process_stats;
  myds4_stats model = {.decode = counts(&c.decode), .prefill = counts(&c.prefill), .decode_rows = c.decode_rows,
                     .prefill_rows = c.prefill_rows, .decode_seconds = c.decode_seconds,
                     .decode_gpu_seconds = c.decode_gpu_seconds, .prefill_layer = c.prefill_layer,
                     .prefill_rows_now = c.prefill_rows_now, .dense_bytes = c.dense_bytes,
                     .layers = c.layers, .experts_per_row = c.experts_per_row,
                     .routed_layers = c.routed_layers, .experts_per_layer = c.experts_per_layer};
  s->decode = model.decode; s->prefill = model.prefill;
  s->decode_rows = model.decode_rows; s->prefill_rows = model.prefill_rows;
  s->decode_seconds = model.decode_seconds; s->decode_gpu_seconds = model.decode_gpu_seconds;
  s->prefill_layer = model.prefill_layer; s->prefill_rows_now = model.prefill_rows_now;
  s->dense_bytes = model.dense_bytes; s->layers = model.layers; s->experts_per_row = model.experts_per_row;
  s->routed_layers = model.routed_layers; s->experts_per_layer = model.experts_per_layer;
  MemoryUsage memory = memory_usage();
  s->model_memory_limit = memory_current_limit(); s->model_memory_used = memory.allocated;
  s->model_memory_committed = memory.committed; s->model_memory_peak = memory.peak;
  s->kv_allocated = memory.kv;
  s->model_workspace = m->m->memory.bytes - c.dense_bytes - c.cache_bytes;
  s->cache = counts(&c.cache);
  s->slots = c.cache_slots; s->mapped = c.mapped; s->cap = c.cap; s->max_slots = c.max_slots;
  s->experts = c.experts; s->slot_bytes = c.slot_bytes; s->cache_bytes = c.cache_bytes;
  s->uncached_bytes = c.uncached_bytes; s->uncached = c.uncached;
}

// ---- tokenizers, from any thread ----

myds4_tokenizer *myds4_model_tokenizer(const myds4_model *m) { return m->tk; }
myds4_tokenizer *myds4_tokenizer_open(const char *dir, const char *image) { return tok_open(dir, image); }

int myds4_tokenize(myds4_tokenizer *t, const char *text, size_t len, int mode, int32_t *out, int max) {
  if (mode < TOK_TEXT || mode > TOK_PROMPT) return -1;
  int *p = xmalloc((len + 2) * sizeof *p);  // a token takes at least one byte
  int n = tok_encode_bytes(t, text, len, mode, p, (int)len + 2);
  if (n <= max) memcpy(out, p, (size_t)n * sizeof *p);
  free(p);
  return n <= max ? n : -1;
}

const char *myds4_piece(const myds4_tokenizer *t, int32_t id, size_t *len) {
  int n;
  const char *s = tok_piece(t, id, &n);
  *len = (size_t)n;
  return s;
}

int myds4_added(const myds4_tokenizer *t, int32_t id) { return tok_added(t, id); }

int32_t myds4_special(const myds4_tokenizer *t, const char *text, size_t len) { return tok_special(t, text, len); }
