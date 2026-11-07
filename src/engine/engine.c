// The engine behind the C ABI of engine.h: loading, the model's work on
// attention states and snapshots, sampling, context-copy drafts, images,
// statistics and the tokenizer. Requests, their scheduling and the KV prefix
// cache are the server's (server/src/worker.rs, server/src/kv.rs).
#include <math.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <sys/qos.h>
#include <sys/time.h>
#include <time.h>

#include "engine.h"
#include "myds4.h"
#include "../experimental/context_lookup.h"

static Options opt;
static myds4_config cfg;
static int loaded;
static int requested_runahead;

int myds4_set_runahead(int enabled) {
  if (loaded || (enabled != 0 && enabled != 1)) return -1;
  requested_runahead = enabled;
  return 0;
}

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
  if (cfg.on_log) {
    char line[1024];
    int k = snprintf(line, sizeof line, "%s.%03d ", t, (int)(tv.tv_usec / 1000));
    vsnprintf(line + k, sizeof line - k, fmt, ap);
    cfg.on_log(line);
  } else {
    fprintf(stderr, "%s.%03d ", t, (int)(tv.tv_usec / 1000));
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
  }
  pthread_mutex_unlock(&log_lock);
  va_end(ap);
}

void myds4_log(const char *line) { say("%s", line); }

// ---- the tokenizer, from any thread ----

static pthread_mutex_t tok_lock = PTHREAD_MUTEX_INITIALIZER;
static int tok_loaded;

int myds4_tokenizer(const char *dir) {
  pthread_mutex_lock(&tok_lock);
  if (!tok_loaded) tok_load(dir), tok_loaded = 1;
  pthread_mutex_unlock(&tok_lock);
  return 0;
}

int myds4_tokenize(const char *text, size_t len, int chat, int32_t *out, int max) {
  char *t = xmalloc(len + 1);
  memcpy(t, text, len);
  int *p = xmalloc((len + 2) * sizeof *p);  // a token takes at least one byte
  pthread_mutex_lock(&tok_lock);
  int n = chat ? tok_encode_chat(t, p, (int)len + 2) : tok_encode(t, p, (int)len + 2);
  pthread_mutex_unlock(&tok_lock);
  if (n <= max) memcpy(out, p, (size_t)n * sizeof *p);
  free(t), free(p);
  return n <= max ? n : -1;
}

const char *myds4_piece(int32_t id, size_t *len) {
  int n;
  const char *s = tok_piece(id, &n);
  *len = (size_t)n;
  return s;
}

int myds4_added(int32_t id) { return tok_added(id); }

int32_t myds4_special(const char *text, size_t len) { return tok_special(text, len); }

// ---- loading and the stop ----

static char *copy(const char *s) { return s ? strdup(s) : 0; }

int myds4_load(const myds4_config *c) {
  const char *bad = !c || c->abi != MYDS4_ABI       ? "the engine's interface version differs from the caller's"
                  : !c->model                       ? "-m: no checkpoint directory"
                  : c->ctx < 1 || c->ctx > 1 << 20  ? "--ctx takes 1 to 1048576 tokens (the model's longest context)"
                  : c->draft < 0 || c->draft > BLOCK ? "--draft takes 1 to 5 drafts"
                  : c->lookup < 0 || c->lookup > VERIFY_DRAFTS ? "--lookup takes off or 1 to 15 drafts"
                  : c->draft_cache && !c->draft     ? "--draft-cache needs --draft"
                  : c->draft_cache && (c->draft_cache < DRAFT_CACHE_MIN || c->draft_cache > DRAFT_EXPERTS)
                      ? "--draft-cache takes 15 to 128 slots per stage (0 keeps the default)"
                  : c->ced && (c->ced < WINDOW || c->ced > 1 << 20)
                      ? "--ced takes off or 128 to 1048576 replay tokens"
                  : c->cache_half_life < 0 || c->cache_half_life > CACHE_HALF_LIFE_MAX
                      ? "--cache-half-life takes 1 to 1000000 forward passes"
                  : requested_runahead && (c->draft || c->online_predictor || c->substitute > 0)
                      ? "--runahead requires --draft off, no online predictor and no substitution"
                  : c->reserve_gb && !(c->reserve_gb >= CACHE_RESERVE_MIN && c->reserve_gb <= CACHE_RESERVE_MAX)
                      ? "--reserve takes 2 to 1024 GiB"
                  : loaded                          ? "the model is loaded already"
                                                    : 0;
  if (bad) {
    fprintf(stderr, "myds4: %s\n", bad);
    return -1;
  }
  cfg = *c;
  cfg.model = copy(c->model), cfg.experts = copy(c->experts), cfg.predictor_state = copy(c->predictor_state);
  die_hook = c->on_die;
  opt = (Options){.ctx = c->ctx, .budget_gb = c->cache_gb, .reserve_gb = c->reserve_gb, .cache_half_life = c->cache_half_life,
                  .cache_read_cost = c->cache_read_cost, .dump_fd = -1, .routes_fd = -1, .draft = c->draft,
                  .draft_cache = c->draft_cache, .predictor_state = cfg.predictor_state,
                  .substitute = c->substitute, .experts = cfg.experts, .quiet = 1,
                  .online_predictor = c->online_predictor || c->predictor_state ? 64 : 0, .ced = c->ced,
                  .lookup = c->lookup, .prefill_prune = c->prefill_prune, .runahead = requested_runahead};
  if (opt.predictor_state) shutdown_watch();  // an interrupt while loading still saves the state
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);  // the model's GPU work goes before background work
  myds4_tokenizer(cfg.model);
  model_load(cfg.model, &opt);
  shutdown_watch();
  loaded = 1;
  say("prefill: ced_depth=%d (%s); dependency pruning=%s; rewind margin=%d; context lookup=%d", opt.ced,
      opt.ced ? "approximate" : "off", opt.prefill_prune ? "on" : "off", myds4_rewind(), opt.lookup);
  return 0;
}

void myds4_finish(void) {
  model_save_predictor();
  online_predictor_report(stderr);
}

int myds4_stopping(void) { return shutdown_requested(); }

void myds4_request_stop(void) { shutdown_request(); }

int myds4_idle(void) {
  int n = cache_slots();
  cache_idle();
  if (cache_slots() < n) say("idle: memory short, expert cache %d -> %d slots", n, cache_slots());
  return cache_slots();
}

int myds4_max_rows(void) { return BLOCK + 1; }
int myds4_draft_block(void) { return opt.draft; }
int myds4_max_verify(void) { return VERIFY_DRAFTS + 1; }
int myds4_rewind(void) { return opt.lookup > BLOCK ? opt.lookup : BLOCK; }
int myds4_vocab(void) { return VOCAB; }

// ---- attention states and snapshots ----

myds4_state *myds4_state_new(void) { return model_state_new(); }
void myds4_state_free(myds4_state *s) { model_state_free(s); }
size_t myds4_state_bytes(const myds4_state *s, int positions) { return model_state_bytes(s, positions); }
void myds4_state_copy(myds4_state *dst, const myds4_state *src, int positions) { model_state_copy(dst, src, positions); }
void myds4_state_use(myds4_state *s) { model_state_use(s); }
void myds4_reset(void) { model_reset(); }

myds4_snapshot *myds4_snapshot_take(myds4_state *s, int pos, int with_logits) {
  return model_snapshot(s, pos, with_logits);
}
void myds4_snapshot_restore(myds4_state *s, const myds4_snapshot *x) { model_snapshot_restore(s, x); }
void myds4_snapshot_free(myds4_snapshot *x) { model_snapshot_free(x); }
size_t myds4_snapshot_bytes(int with_logits) { return model_snapshot_bytes(with_logits); }
const float *myds4_snapshot_logits(const myds4_snapshot *x) { return model_snapshot_logits(x); }

// ---- running the model, in the state in use ----

// Tokens of the vocabulary; with images, also the negative keys that
// myds4_images_expand writes at an image's positions.
static int in_vocabulary(const int32_t *tokens, int n, int images) {
  for (int i = 0; i < n; i++)
    if (tokens[i] >= VOCAB || (tokens[i] < 0 && !images)) return 0;
  return 1;
}

const float *myds4_prefill(const int32_t *tokens, int n, int from, int replay, int capture,
                           myds4_snapshot **captured, myds4_prefill_info *info) {
  if (captured) *captured = 0;
  if (n < 1 || from < 0 || from >= n || n > opt.ctx || !in_vocabulary(tokens, n, 1)) return 0;
  int start = replay && from > 0 ? model_replay_start(from) : from;
  if (capture >= 0 && (!captured || capture <= start || capture >= n)) return 0;
  Snapshot *x = 0;
  const float *lg = model_prefill(tokens, n, from, replay, capture, &x);
  if (x && model_snapshot_extent(x) != LAYERS - 1) model_snapshot_free(x), x = 0;  // no whole state to restore
  if (captured) *captured = x;
  if (info) {
    ModelForwardInfo f = model_forward_info();
    *info = (myds4_prefill_info){.chunks = f.chunks, .tokens = f.tokens, .decoder_tokens = f.decoder_tokens,
                                 .ced_applied = f.ced_applied, .prune_applied = f.prune_applied,
                                 .prefix_tokens = f.prefix_tokens};
  }
  return lg;
}

int myds4_replay_rows(int k) { return model_replay_rows(k); }
int myds4_replay_start(int k) { return model_replay_start(k); }

const float *myds4_step(int32_t token, int pos) {
  if (!in_vocabulary(&token, 1, 0) || pos < 0 || pos >= opt.ctx) return 0;
  return model_forward(&token, 1, pos);
}

const float *myds4_rows(myds4_state **states, const int32_t *tokens, const int32_t *pos, int m) {
  if (m < 1 || m > BLOCK + 1 || !in_vocabulary(tokens, m, 0)) return 0;
  for (int r = 0; r < m; r++)
    if (pos[r] < 0 || pos[r] >= opt.ctx) return 0;
  return model_forward_rows(states, tokens, pos, m);
}

void myds4_draft(int32_t token, int pos, int32_t *drafts, int n) { model_draft(token, pos, drafts, n); }

const float *myds4_verify(const int32_t *rows, int m, int pos) {
  if (m < 1 || m > VERIFY_DRAFTS + 1 || pos < 0 || pos + m > opt.ctx || !in_vocabulary(rows, m, 0)) return 0;
  return model_verify(rows, m, pos);
}

void myds4_accept(const int32_t *rows, int n, int pos) { model_accept(rows, n, pos); }

void myds4_seek(const int32_t *tokens, int pos) { model_seek(tokens, pos); }

// ---- sampling ----

struct myds4_sampler {
  Sampler sp;
};

myds4_sampler *myds4_sampler_new(double temperature, double top_p, uint64_t seed, int seed_set) {
  Sampler sp = {.temp = temperature, .top_p = top_p};
  if (!sampler_valid(&sp)) return 0;
  myds4_sampler *s = xmalloc(sizeof *s);
  s->sp = sp;
  if (seed_set) sampler_seed(&s->sp, seed);
  else arc4random_buf(s->sp.seed, sizeof s->sp.seed);
  return s;
}

void myds4_sampler_free(myds4_sampler *s) { free(s); }

int32_t myds4_sample(const float *logits, myds4_sampler *s) { return sample_token(logits, VOCAB, &s->sp); }

int32_t myds4_sample_verify(const float *logits, const int32_t *drafts, int n, int32_t eos, myds4_sampler *s,
                            int *accepted) {
  return sample_verify(logits, VOCAB, drafts, n, eos, &s->sp, accepted);
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
  Image *ims;
  int n;
};

myds4_images *myds4_images_open(const uint8_t *const *data, const size_t *len, int n, char *message, size_t cap) {
  myds4_images *x = xmalloc(sizeof *x);
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
    if (x->ims[i].start + image_span(&x->ims[i]) > from) model_image(&x->ims[i]), encoded++;
  model_images(x->ims, x->n);
  return encoded;
}

void myds4_images_close(myds4_images *x) {
  if (!x) return;
  model_images(0, 0);
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

int myds4_routes(int row, uint16_t *experts, uint8_t *sources) { return model_routes(row, experts, sources); }

void myds4_stats_get(myds4_stats *s) {
  MemInfo m;
  mem_info(&m);
  *s = (myds4_stats){.cache = counts(&cache_stats), .decode = counts(&decode_stats), .prefill = counts(&prefill_stats),
                     .decode_rows = decode_rows, .prefill_rows = prefill_rows, .decode_seconds = decode_s,
                     .decode_gpu_seconds = decode_gpu_s, .prefill_layer = prefill_layer, .prefill_rows_now = prefill_n,
                     .slots = cache_slots(), .mapped = cache_mapped(), .cap = cache_cap(), .max_slots = cache_max(),
                     .experts = LAYERS * EXPERTS, .slot_bytes = cache_slot_bytes(), .cache_bytes = cache_bytes(),
                     .dense_bytes = model_dense_bytes(), .footprint = mem_footprint(), .gpu_in_use = mem_gpu(),
                     .available = mem_available(), .gpu_limit = mem_gpu_limit(), .total = m.total, .free = m.free,
                     .file = m.file, .file_kept = m.file_kept, .wired = m.wired, .compressor = m.compressor,
                     .compressed = m.compressed, .swap_used = m.swap_used, .swapouts = m.swapouts,
                     .pressure = m.pressure, .layers = LAYERS, .experts_per_row = LAYERS * TOPK};
  s->uncached_bytes = cache_uncached_bytes(&s->uncached);
}
