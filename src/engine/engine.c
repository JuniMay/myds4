// The engine behind the C ABI of engine.h: the process's log and stop, and
// for each model (src/model/model.h) its loading, its work on states and
// snapshots, sampling, context-copy drafts, images, statistics and its
// tokenizer. Requests, their scheduling and the KV prefix cache are the
// server's (server/src/worker.rs, server/src/kv.rs).
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
#include "../model/model.h"
#include "../experimental/context_lookup.h"

struct myds4_model {
  Model *m;
  const ModelOps *ops;
  Tokenizer *tk;
  Options opt;
  char *dir;
  atomic_int loaded;
};

static myds4_process proc;
static int initialized;
static const ModelOps *families[8];  // the families loaded: one model each
static int nfamilies;

// ---- the log: one line per event, on stderr (or to the caller's on_log) ----

static pthread_mutex_t log_lock = PTHREAD_MUTEX_INITIALIZER;

static void say(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
static void say(const char *fmt, ...) {
  struct timeval tv;
  struct tm tm;
  char t[16];
  gettimeofday(&tv, 0);
  localtime_r(&tv.tv_sec, &tm);
  strftime(t, sizeof t, "%H:%M:%S", &tm);
  va_list ap;
  va_start(ap, fmt);
  pthread_mutex_lock(&log_lock);
  if (proc.on_log) {
    char line[1024];
    int k = snprintf(line, sizeof line, "%s.%03d ", t, (int)(tv.tv_usec / 1000));
    vsnprintf(line + k, sizeof line - k, fmt, ap);
    proc.on_log(line);
  } else {
    fprintf(stderr, "%s.%03d ", t, (int)(tv.tv_usec / 1000));
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
  }
  pthread_mutex_unlock(&log_lock);
  va_end(ap);
}

void myds4_log(const char *line) { say("%s", line); }

// ---- the process ----

int myds4_init(const myds4_process *p) {
  if (!p || p->abi != MYDS4_ABI || initialized) return -1;
  if (p->reserve_gb && !(p->reserve_gb >= CACHE_RESERVE_MIN && p->reserve_gb <= CACHE_RESERVE_MAX)) return -1;
  cache_reserve(p->reserve_gb);
  proc = *p;
  die_hook = p->on_die;
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);  // the models' GPU work goes before background work
  shutdown_watch();  // an interrupt while loading still saves a predictor's state
  initialized = 1;
  return 0;
}

int myds4_stopping(void) { return shutdown_requested(); }

void myds4_request_stop(void) { shutdown_request(); }

// ---- models ----

static char *copy(const char *s) { return s ? strdup(s) : 0; }

myds4_model *myds4_open(const myds4_config *c) {
  const char *bad = !initialized                    ? "the engine is not initialized (myds4_init)"
                  : !c || c->abi != MYDS4_ABI       ? "the engine's interface version differs from the caller's"
                  : !c->model                       ? "no checkpoint directory"
                  : c->ctx < 1 || c->ctx > 1 << 20  ? "ctx takes 1 to 1048576 tokens (the longest context)"
                  : c->draft < 0 || c->draft > BLOCK ? "draft takes 0 to 5 drafts"
                  : c->lookup < 0 || c->lookup > VERIFY_DRAFTS ? "lookup takes off or 1 to 15 drafts"
                  : c->draft_cache && !c->draft     ? "draft-cache needs draft"
                  : c->draft_cache && (c->draft_cache < DRAFT_CACHE_MIN || c->draft_cache > DRAFT_EXPERTS)
                      ? "draft-cache takes 15 to 128 slots per stage (0 keeps the default)"
                  : c->ced && (c->ced < WINDOW || c->ced > 1 << 20) ? "ced takes off or 128 to 1048576 replay tokens"
                  : c->cache_half_life < 0 || c->cache_half_life > CACHE_HALF_LIFE_MAX
                      ? "cache-half-life takes 1 to 1000000 forward passes"
                  : c->runahead && (c->draft || c->online_predictor || c->substitute > 0)
                      ? "runahead requires draft off, no online predictor and no substitution"
                  : c->pipeline && (c->experts || c->runahead || c->draft || c->online_predictor || c->predictor_state || c->substitute > 0)
                      ? "pipeline requires native MXFP4, no runahead, draft, online predictor or substitution"
                                                    : 0;
  if (bad) {
    fprintf(stderr, "myds4: %s\n", bad);
    return 0;
  }
  const ModelOps *ops = model_find(c->model);
  for (int i = 0; i < nfamilies; i++)
    if (families[i] == ops) {
      fprintf(stderr, "myds4: %s: a %s model is loaded already (one model of a family per process)\n", c->model,
              ops->family);
      return 0;
    }
  myds4_model *x = xmalloc(sizeof *x);
  atomic_init(&x->loaded, 0);
  x->ops = ops;
  x->opt = (Options){.ctx = c->ctx, .budget_gb = c->cache_gb,
                     .cache_half_life = c->cache_half_life, .cache_read_cost = c->cache_read_cost, .dump_fd = -1,
                     .routes_fd = -1, .draft = c->draft, .draft_cache = c->draft_cache,
                     .predictor_state = copy(c->predictor_state), .substitute = c->substitute,
                     .experts = copy(c->experts), .quiet = 1,
                     .online_predictor = c->online_predictor || c->predictor_state ? 64 : 0, .ced = c->ced,
                     .lookup = c->lookup, .prefill_prune = c->prefill_prune, .runahead = c->runahead, .pipeline = c->pipeline};
  const char *why = ops->refuse(&x->opt);
  if (why) {
    fprintf(stderr, "myds4: %s: %s\n", c->model, why);
    free(x);
    return 0;
  }
  families[nfamilies++] = ops;
  x->tk = tok_open(c->model, ops->image_token);
  x->dir = copy(c->model);
  say("%s: registered %s; weights load on first inference request", c->model, ops->family);
  return x;
}

int myds4_loaded(const myds4_model *m) { return atomic_load_explicit(&m->loaded, memory_order_acquire); }

int myds4_load(myds4_model *m) {
  if (myds4_loaded(m)) return 1;
  m->m = model_open(m->dir, &m->opt);
  atomic_store_explicit(&m->loaded, 1, memory_order_release);
  say("%s: %s, context %d, drafts %d, rewind margin %d, context lookup %d; prefill: ced_depth=%d (%s), dependency "
      "pruning=%s",
      m->dir, m->ops->family, m->opt.ctx, m->ops->draft_block(), m->ops->rewind(), m->opt.lookup, m->opt.ced,
      m->opt.ced ? "approximate" : "off", m->opt.prefill_prune ? "on" : "off");
  return 1;
}

void myds4_finish(myds4_model *m) {
  if (myds4_loaded(m) && m->ops->finish) m->ops->finish();
}

int myds4_idle(myds4_model *m) {
  if (!myds4_loaded(m)) return 0;
  // the expert cache (io.c) is the slots of the model with routed experts;
  // another model's idle returns 0 and leaves the cache as it is
  ModelCounters c;
  m->ops->counters(&c);
  int before = c.experts_per_row ? cache_slots() : 0, after = m->ops->idle();
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

myds4_state *myds4_state_new(myds4_model *m) { return m->ops->state_new(); }
void myds4_state_free(myds4_model *m, myds4_state *s) { m->ops->state_free(s); }
size_t myds4_state_bytes(const myds4_model *m, const myds4_state *s, int positions) {
  return m->ops->state_bytes(s, positions);
}
void myds4_state_copy(myds4_model *m, myds4_state *dst, const myds4_state *src, int positions) {
  m->ops->state_copy(dst, src, positions);
}

myds4_snapshot *myds4_snapshot_take(myds4_model *m, myds4_state *s, int pos, int with_logits) {
  return m->ops->snapshot_take(s, pos, with_logits);
}
void myds4_snapshot_restore(myds4_model *m, myds4_state *s, const myds4_snapshot *x) {
  m->ops->snapshot_restore(s, x);
}
void myds4_snapshot_free(myds4_model *m, myds4_snapshot *x) { m->ops->snapshot_free(x); }
size_t myds4_snapshot_bytes(const myds4_model *m, int with_logits) { return m->ops->snapshot_bytes(with_logits); }
const float *myds4_snapshot_logits(const myds4_model *m, const myds4_snapshot *x) { return m->ops->snapshot_logits(x); }

// ---- running a model ----

// Tokens of the vocabulary; with images, also the negative keys that
// myds4_images_expand writes at an image's positions.
static int in_vocabulary(const myds4_model *m, const int32_t *tokens, int n, int images) {
  int vocab = m->ops->vocab();
  for (int i = 0; i < n; i++)
    if (tokens[i] >= vocab || (tokens[i] < 0 && !images)) return 0;
  return 1;
}

const float *myds4_prefill(myds4_model *m, myds4_state *s, const int32_t *tokens, int n, int from, int replay,
                           int capture, myds4_snapshot **captured, myds4_prefill_info *info) {
  if (captured) *captured = 0;
  if (n < 1 || from < 0 || from >= n || n > m->opt.ctx || !in_vocabulary(m, tokens, n, m->ops->image != 0)) return 0;
  if (replay && from > 0 && m->ops->replay_rows(from) < 0) return 0;
  int start = replay && from > 0 ? m->ops->replay_start(from) : from;
  if (capture >= 0 && (!captured || capture <= start || capture >= n)) return 0;
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

int myds4_replay_rows(const myds4_model *m, int k) { return m->ops->replay_rows(k); }
int myds4_replay_start(const myds4_model *m, int k) { return m->ops->replay_start(k); }

const float *myds4_rows(myds4_model *m, myds4_state **states, const int32_t *tokens, const int32_t *pos, int count) {
  if (count < 1 || count > m->ops->max_rows() || !in_vocabulary(m, tokens, count, 0)) return 0;
  for (int r = 0; r < count; r++)
    if (pos[r] < 0 || pos[r] >= m->opt.ctx) return 0;
  return m->ops->rows(states, tokens, pos, count);
}

void myds4_draft(myds4_model *m, myds4_state *s, int32_t token, int pos, int32_t *drafts, int n) {
  m->ops->draft(s, token, pos, drafts, n);
}

const float *myds4_verify(myds4_model *m, myds4_state *s, const int32_t *rows, int count, int pos) {
  if (count < 1 || count > m->ops->max_verify() || pos < 0 || pos + count > m->opt.ctx ||
      !in_vocabulary(m, rows, count, 0))
    return 0;
  return m->ops->verify(s, rows, count, pos);
}

void myds4_accept(myds4_model *m, myds4_state *s, const int32_t *rows, int n, int pos) {
  m->ops->accept(s, rows, n, pos);
}

void myds4_seek(myds4_model *m, myds4_state *s, const int32_t *tokens, int pos) { m->ops->seek(s, tokens, pos); }

// ---- sampling ----

struct myds4_sampler {
  Sampler sp;
};

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

int32_t myds4_sample_verify(const myds4_model *m, const float *logits, const int32_t *drafts, int n, myds4_sampler *s,
                            int *accepted) {
  return sample_verify(logits, m->ops->vocab(), drafts, n, m->ops->stop, &s->sp, accepted);
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

// ---- images ----

struct myds4_images {
  myds4_model *m;
  Image *ims;
  int n;
};

myds4_images *myds4_images_open(myds4_model *m, const uint8_t *const *data, const size_t *len, int n, char *message,
                                size_t cap) {
  if (!m->ops->image) {
    snprintf(message, cap, "this model has no vision encoder");
    return 0;
  }
  myds4_images *x = xmalloc(sizeof *x);
  x->m = m;
  x->ims = calloc((size_t)n + 1, sizeof *x->ims), x->n = 0;
  if (!x->ims) die("out of memory");
  for (int i = 0; i < n; i++) {
    const char *why;
    if (!image_open(&x->ims[i], data[i], len[i], &why)) {
      snprintf(message, cap, "image %d: %s", i + 1, why);
      myds4_images_close(x);
      return 0;
    }
    x->n = i + 1;
  }
  return x;
}

int myds4_images_expand(myds4_images *x, const int32_t *in, int n, int32_t *out, int max) {
  return image_expand(in, n, x->ims, x->n, out, max);
}

int myds4_images_encode(myds4_images *x, int from) {
  int encoded = 0;  // the encoder runs for the images the prefill reaches; the others are in the state's rows
  for (int i = 0; i < x->n; i++)
    if (x->ims[i].start + image_span(&x->ims[i]) > from) x->m->ops->image(&x->ims[i]), encoded++;
  x->m->ops->images(x->ims, x->n);
  return encoded;
}

void myds4_images_close(myds4_images *x) {
  if (!x) return;
  x->m->ops->images(0, 0);
  for (int i = 0; i < x->n; i++) image_free(&x->ims[i]);
  free(x->ims), free(x);
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

void myds4_stats_get(const myds4_model *m, myds4_stats *s) {
  MemInfo mi;
  mem_info(&mi);
  ModelCounters c = {0};
  if (myds4_loaded(m)) m->ops->counters(&c);
  *s = (myds4_stats){.decode = counts(&c.decode), .prefill = counts(&c.prefill), .decode_rows = c.decode_rows,
                     .prefill_rows = c.prefill_rows, .decode_seconds = c.decode_seconds,
                     .decode_gpu_seconds = c.decode_gpu_seconds, .prefill_layer = c.prefill_layer,
                     .prefill_rows_now = c.prefill_rows_now, .dense_bytes = c.dense_bytes,
                     .footprint = mem_footprint(), .gpu_in_use = mem_gpu(), .available = mem_available(),
                     .gpu_limit = mem_gpu_limit(), .total = mi.total, .free = mi.free, .file = mi.file,
                     .file_kept = mi.file_kept, .wired = mi.wired, .compressor = mi.compressor,
                     .compressed = mi.compressed, .swap_used = mi.swap_used, .swapouts = mi.swapouts,
                     .pressure = mi.pressure, .layers = c.layers, .experts_per_row = c.experts_per_row};
  if (c.experts_per_row) {  // the expert cache serves this model
    s->cache = counts(&cache_stats);
    s->slots = cache_slots(), s->mapped = cache_mapped(), s->cap = cache_cap(), s->max_slots = cache_max();
    s->experts = c.layers * EXPERTS, s->slot_bytes = cache_slot_bytes(), s->cache_bytes = cache_bytes();
    s->uncached_bytes = cache_uncached_bytes(&s->uncached);
  }
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
