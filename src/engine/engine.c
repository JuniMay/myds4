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
  if (!memory_configure(p->memory_gb, p->kv_gb, p->reserve_gb)) return -1;
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
    say("%s: the global model-memory budget cannot fit this request", m->dir);
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

myds4_snapshot *myds4_snapshot_take(myds4_model *m, myds4_state *s, int pos, int with_logits) {
  memory_scope(&m->m->work);
  return m->ops->snapshot_take(s, pos, with_logits);
}
void myds4_snapshot_restore(myds4_model *m, myds4_state *s, const myds4_snapshot *x) {
  memory_scope(&m->m->work);
  m->ops->snapshot_restore(s, x);
}
void myds4_snapshot_free(myds4_model *m, myds4_snapshot *x) { m->ops->snapshot_free(x); }
size_t myds4_snapshot_bytes(const myds4_model *m, int with_logits) { return m->ops->snapshot_bytes(with_logits); }
const float *myds4_snapshot_logits(const myds4_model *m, const myds4_snapshot *x) { return m->ops->snapshot_logits(x); }

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
    .available = mi.free + mi.file, .gpu_limit = mem_gpu_limit(), .total = mi.total,
    .free = mi.free, .file = mi.file, .file_kept = mi.file_kept, .wired = mi.wired,
    .compressor = mi.compressor, .compressed = mi.compressed, .swap_used = mi.swap_used,
    .swapouts = mi.swapouts, .pressure = mi.pressure};
}

uint64_t myds4_reclaim(uint64_t bytes) { return memory_reclaim(bytes); }

void myds4_stats_get(const myds4_model *m, myds4_stats *s) {
  ModelCounters c = {0};
  if (atomic_load_explicit(&m->loaded, memory_order_acquire)) m->ops->counters(&c);
  *s = process_stats;
  myds4_stats model = {.decode = counts(&c.decode), .prefill = counts(&c.prefill), .decode_rows = c.decode_rows,
                     .prefill_rows = c.prefill_rows, .decode_seconds = c.decode_seconds,
                     .decode_gpu_seconds = c.decode_gpu_seconds, .prefill_layer = c.prefill_layer,
                     .prefill_rows_now = c.prefill_rows_now, .dense_bytes = c.dense_bytes,
                     .layers = c.layers, .experts_per_row = c.experts_per_row};
  s->decode = model.decode; s->prefill = model.prefill;
  s->decode_rows = model.decode_rows; s->prefill_rows = model.prefill_rows;
  s->decode_seconds = model.decode_seconds; s->decode_gpu_seconds = model.decode_gpu_seconds;
  s->prefill_layer = model.prefill_layer; s->prefill_rows_now = model.prefill_rows_now;
  s->dense_bytes = model.dense_bytes; s->layers = model.layers; s->experts_per_row = model.experts_per_row;
  MemoryUsage memory = memory_usage();
  s->model_memory_limit = memory.limit; s->model_memory_used = memory.allocated;
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
