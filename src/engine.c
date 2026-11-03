// The engine behind the C ABI of engine.h: the C server's scheduler
// (serve.c) without HTTP or JSON. Requests arrive whole, as token ids or as
// rendered text, and wait in a queue (at most 16). The engine's thread loads
// the model and serves the queue in order: it admits waiting requests while
// fewer than --batch decode, prefilling each while the others wait, then
// advances every decoding request by a token, alone with its drafts
// (--draft), or together with the others in one step of model_forward_rows.
// Each conversation keeps its attention state in a session, so a request
// that extends an earlier one runs only its new tokens.
#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>

#include "engine.h"
#include "myds4.h"

#define WAITING 16           // requests waiting for the model
#define SESSIONS 256         // at most, whatever --kv-cache allows
#define FEW (BLOCK + 1)      // requests decoding together at most (model_forward_rows)
#define MODEL_ID "deepseek-v4.1-flash"
#define GiB (1024.0 * 1024 * 1024)

static Options opt;
static myds4_config cfg;
static int eos, user, think_start, think_end;
static long started;

// ---- the log: one line per event, on stderr (or in the live view's pane) ----

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
  if (cfg.tui) {
    char line[1024];
    int k = snprintf(line, sizeof line, "%s.%03d ", t, (int)(tv.tv_usec / 1000));
    vsnprintf(line + k, sizeof line - k, fmt, ap);
    tui_log(line);
  } else {
    fprintf(stderr, "%s.%03d ", t, (int)(tv.tv_usec / 1000));
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
  }
  pthread_mutex_unlock(&log_lock);
  va_end(ap);
}

void myds4_log(const char *line) { say("%s", line); }

// ---- sessions: each conversation's attention state ----

typedef struct {
  State *st;      // null: a free slot
  int *ids, n;    // the tokens whose state st holds
  int saved;      // the position of the rows saved in slot 0 (its last prompt's end), -1 if none
  float *logits;  // the logits of that prompt's last token
  int base;       // the position of the rows saved in slot 1 (its first user turn), -1 if none
  int busy;       // a request decodes in it
  int number;
  double used;
  int hits;          // requests that reused its rows (continued it, or forked from it)
  int hit_at[32];    // the positions reused, of the last 32 of them
} Session;

static Session sessions[SESSIONS];
static Session *current;  // the session whose state the layers use
static int live, session_numbers;
static double kv_limit;  // bytes the sessions' states may take (--kv-cache)

// What the worker is doing, for myds4_status and the live view.
static pthread_mutex_t st_lock = PTHREAD_MUTEX_INITIALIZER;
static struct {
  const char *phase;  // idle, prefill, decode
  int active;         // requests decoding
  int request, prompt, reused, generated;  // the first of them, or the one prefilling
  double since;
  uint64_t requests, prompt_tokens, generated_tokens, reused_tokens;
  int kinds[4];          // requests by session: new, kept, saved, forked
  int nlive;             // the requests decoding (for the live view)
  ViewReq live[FEW];
} status = {.phase = "idle"};

static void hit(Session *s, int at) {  // a request reused s's rows up to position at
  s->hit_at[s->hits++ % 32] = at;
}

static int common(const int *a, int n, const int *b, int m) {
  int k = 0;
  while (k < n && k < m && a[k] == b[k]) k++;
  return k;
}

static Session *least_used(const Session *except) {  // of the sessions no request decodes in
  Session *v = 0;
  for (int i = 0; i < SESSIONS; i++)
    if (sessions[i].st && !sessions[i].busy && &sessions[i] != except && (!v || sessions[i].used < v->used))
      v = &sessions[i];
  return v;
}

// The prompt tokens a session can skip: up to a position its rows cover (its
// last BLOCK positions, or the BLOCK before its saved one), or its saved
// position. The last prompt token always runs, for its logits.
static int reusable(const Session *s, const int *p, int n, int *saved) {
  int k = common(p, n, s->ids, s->n), r = k < n - 1 ? k : n - 1;
  *saved = 0;
  if (k <= 1) return 0;  // the begin token alone is no prefix
  if (s->n - r <= BLOCK) return r;
  *saved = s->saved >= 0;
  if (s->saved >= r && s->saved - r <= BLOCK) return r;
  if (s->saved >= 0 && s->saved <= r) return s->saved;
  return *saved = 0;
}

static double kv_bytes(void) {
  double n = 0;
  for (int i = 0; i < SESSIONS; i++) n += sessions[i].st ? model_state_bytes(sessions[i].st, 0) : 0;
  return n;
}

static void release(Session *v, const char *why) {
  pthread_mutex_lock(&st_lock);
  model_state_free(v->st), free(v->ids), free(v->logits);
  *v = (Session){0};
  live--;
  pthread_mutex_unlock(&st_lock);
  say("kv: %s: released a session (%d left, %.2f GiB)", why, live, kv_bytes() / GiB);
}

// Releases the least recently used sessions but keep until `need` more
// bytes fit in --kv-cache; 0 if they cannot.
static int fits(const Session *keep, double need) {
  for (;;) {
    if (kv_bytes() - (keep ? model_state_bytes(keep->st, 0) : 0) + need <= kv_limit) return 1;
    Session *v = least_used(keep);
    if (!v) return 0;
    release(v, "over --kv-cache");
  }
}

// A session to fill: a new one if --kv-cache has room, else the least
// recently used session other than keep.
static Session *empty(const Session *keep) {
  Session *s = live == SESSIONS || kv_bytes() + model_state_bytes(0, 0) > kv_limit ? least_used(keep) : 0;
  pthread_mutex_lock(&st_lock);
  if (!s) {
    for (s = sessions; s->st; s++) {}
    s->st = model_state_new(), s->ids = xmalloc((size_t)opt.ctx * sizeof *s->ids);
    live++;
  }
  s->n = 0, s->saved = s->base = -1, s->number = ++session_numbers, s->hits = 0;
  pthread_mutex_unlock(&st_lock);
  return s;
}

// The session of a prompt of n tokens. A session that holds the prompt's
// start continues (or rewinds to its saved rows) when that skips the most;
// a prompt that shares only a session's first system part (another
// conversation with the same instructions and tools) gets a copy of that
// session from its first user turn, which leaves the session as it was.
// Else a new session, or the least recently used one emptied. *from: the
// first token to run.
static Session *pick(const int *p, int n, int *from, const char **how) {
  Session *v = live > 1 && mem_pressure() >= 2 ? least_used(current) : 0;
  if (v) release(v, "memory pressure");
  Session *s = 0, *src = 0;
  int saved = 0;
  *from = 0;
  for (int i = 0; i < SESSIONS; i++) {
    Session *c = &sessions[i];
    if (!c->st) continue;
    int w, r = c->busy ? 0 : reusable(c, p, n, &w), k = common(p, n, c->ids, c->n);  // a busy one only forks
    int f = c->base > 1 && c->base < n && k >= c->base ? c->base : 0;
    if (r > *from) *from = r, saved = w, s = c, src = 0;
    if (f > *from) *from = f, src = c, s = 0;
  }
  if (s) {
    model_state_use(s->st);
    if (saved) model_state_restore(0);
    *how = saved ? "saved" : "kept";
    pthread_mutex_lock(&st_lock);
    hit(s, *from), status.kinds[saved ? 2 : 1]++;
    pthread_mutex_unlock(&st_lock);
  } else if (src) {
    model_state_use(src->st);
    s = empty(src);
    model_state_copy(s->st, src->st);
    model_state_use(s->st);
    model_state_restore(1);
    pthread_mutex_lock(&st_lock);
    s->base = src->base;
    memcpy(s->ids, p, (size_t)*from * sizeof *p), s->n = *from;
    hit(src, *from), status.kinds[3]++;
    pthread_mutex_unlock(&st_lock);
    *how = "forked";
  } else {
    s = empty(0);
    model_state_use(s->st);
    model_reset();
    *how = "new";
    pthread_mutex_lock(&st_lock);
    status.kinds[0]++;
    pthread_mutex_unlock(&st_lock);
  }
  model_seek(p, *from);
  current = s;
  return s;
}

// ---- the tokenizer: its work buffers are static, so one call at a time ----

static pthread_mutex_t tok_lock = PTHREAD_MUTEX_INITIALIZER;
static int tok_loaded;

int myds4_tokenizer(const char *dir) {
  pthread_mutex_lock(&tok_lock);
  if (!tok_loaded) tok_load(dir), tok_loaded = 1;
  pthread_mutex_unlock(&tok_lock);
  return 0;
}

// The tokens of n bytes of text (chat: a rendered prompt), allocated; *count.
static int *tokenize(const char *text, size_t n, int chat, int *count) {
  char *t = xmalloc(n + 1);
  memcpy(t, text, n);
  int *p = xmalloc((n + 2) * sizeof *p);  // a token takes at least one byte
  pthread_mutex_lock(&tok_lock);
  *count = chat ? tok_encode_chat(t, p, (int)n + 2) : tok_encode(t, p, (int)n + 2);
  pthread_mutex_unlock(&tok_lock);
  free(t);
  return p;
}

int myds4_tokenize(const char *text, size_t len, int chat, int32_t *out, int max) {
  int n, *p = tokenize(text, len, chat, &n);
  if (n <= max) memcpy(out, p, (size_t)n * sizeof *p);
  free(p);
  return n <= max ? n : -1;
}

const char *myds4_piece(int32_t id, size_t *len) {
  int n;
  const char *s = tok_piece(id, &n);
  *len = (size_t)n;
  return s;
}

int myds4_added(int32_t id) { return tok_added(id); }

// ---- requests: waiting, then decoding ----

typedef struct Req {
  struct Req *next;       // in the queue
  struct Req *next_live;  // among the requests not yet ended
  uint64_t id;
  int *p, n;              // the prompt's tokens
  myds4_image *img;       // copies of the images' bytes
  int nimg;
  Sampler sp;
  int seed_set;
  uint64_t seed;
  long max_new;
  void (*on_event)(void *, const myds4_event *);
  void *user;
  _Atomic int cancelled;
  double queued;
} Req;

static Req *first, *last, *alive;  // the queue; every request not yet ended
static int waiting, closed;        // closed: the engine stopped and takes no requests
static uint64_t numbers;
static pthread_mutex_t q_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t q_job = PTHREAD_COND_INITIALIZER;

static void event(Req *r, myds4_event e) {
  e.id = r->id;
  r->on_event(r->user, &e);
}

static void drop_req(Req *r) {  // after its last event
  pthread_mutex_lock(&q_lock);
  Req **at = &alive;
  while (*at && *at != r) at = &(*at)->next_live;
  if (*at) *at = r->next_live;
  pthread_mutex_unlock(&q_lock);
  for (int i = 0; i < r->nimg; i++) free((void *)r->img[i].data);
  free(r->img), free(r->p), free(r);
}

static void end_waiting(Req *r, int finish) {  // a request that never started
  event(r, (myds4_event){.kind = MYDS4_DONE, .prompt = r->n, .finish = finish});
  drop_req(r);
}

int myds4_submit(const myds4_request *q, uint64_t *id, char *message, size_t cap) {
  Sampler sp = {.temp = q->temperature, .top_p = q->top_p};
  const char *bad = !q->on_event                                    ? "the request has no event callback"
                  : !(q->tokens ? q->ntokens > 0 : q->text && q->len) ? "the prompt is empty"
                  : q->nimages < 0 || (q->nimages && !q->images)    ? "the images are missing"
                  : !sampler_valid(&sp) ? "temperature must be a finite nonnegative number and top_p must be in [0, 1]"
                  : q->max_tokens < 1   ? "max_tokens must be at least 1"
                                        : 0;
  for (int i = 0; !bad && q->tokens && i < q->ntokens; i++)
    if (q->tokens[i] < 0 || q->tokens[i] >= VOCAB) bad = "a prompt token is not in the vocabulary";
  if (bad) {
    snprintf(message, cap, "%s", bad);
    return MYDS4_INVALID;
  }
  Req *r = xmalloc(sizeof *r);
  if (q->tokens) {
    r->p = xmalloc((size_t)q->ntokens * sizeof *r->p), r->n = q->ntokens;
    memcpy(r->p, q->tokens, (size_t)q->ntokens * sizeof *r->p);
  } else {
    r->p = tokenize(q->text, q->len, 1, &r->n);
  }
  int places = 0;
  for (int i = 0; i < r->n; i++) places += r->p[i] == IMAGE_TOKEN;
  if (!r->n || places != q->nimages) {
    if (!r->n) snprintf(message, cap, "the prompt is empty");
    else snprintf(message, cap, "the prompt has %d image places and %d images", places, q->nimages);
    free(r->p), free(r);
    return MYDS4_INVALID;
  }
  r->img = xmalloc((size_t)(q->nimages + 1) * sizeof *r->img), r->nimg = q->nimages;
  for (int i = 0; i < q->nimages; i++) {
    uint8_t *d = xmalloc(q->images[i].len + 1);
    memcpy(d, q->images[i].data, q->images[i].len);
    r->img[i] = (myds4_image){d, q->images[i].len};
  }
  r->sp = sp, r->seed_set = q->seed_set, r->seed = q->seed, r->max_new = q->max_tokens;
  r->on_event = q->on_event, r->user = q->user, r->queued = now();
  pthread_mutex_lock(&q_lock);
  int refused = closed || waiting >= WAITING;
  if (!refused) {
    r->id = *id = ++numbers;
    if (q->note) say("req %llu %s", (unsigned long long)r->id, q->note);  // before the worker can take it
    if (last) last->next = r;
    else first = r;
    last = r;
    r->next_live = alive, alive = r;
    waiting++;
    pthread_cond_signal(&q_job);
  }
  pthread_mutex_unlock(&q_lock);
  if (refused) {
    snprintf(message, cap, "%s", closed ? "the server is stopping" : "the server is busy: too many requests are waiting");
    r->next_live = 0, drop_req(r);
    return MYDS4_BUSY;
  }
  return MYDS4_QUEUED;
}

void myds4_cancel(uint64_t id) {
  pthread_mutex_lock(&q_lock);
  for (Req *r = alive; r; r = r->next_live)
    if (r->id == id) atomic_store(&r->cancelled, 1);
  pthread_mutex_unlock(&q_lock);
}

// ---- the requests being served ----

// A request from its prefill to its end: its sampler, its session, and t,
// the token at pos it runs next.
typedef struct {
  Req *r;
  Sampler sp;
  Session *s;
  int *p, n, from, pos, gen, reasoning, thinking, ended, cancelled, t;
  int32_t out[BLOCK + 1];  // the tokens kept in this step, for MYDS4_TOKENS
  int nout;
  int draft_passes, draft_accepted;
  long max_new;
  double t1, tick;
  CacheStats c1;
  Image *ims;  // the prompt's images, until its prefill
  int nims;
} Gen;

static Gen gens[FEW];  // decoding together, at most cfg.batch
static int ngens;

static void set_status(const char *phase, const Gen *g) {
  pthread_mutex_lock(&st_lock);
  if (strcmp(status.phase, phase)) status.since = now();
  status.phase = phase, status.active = ngens;
  status.request = g ? (int)g->r->id : 0, status.prompt = g ? g->n : 0;
  status.reused = g ? g->from : 0, status.generated = g ? g->gen : 0;
  status.nlive = !strcmp(phase, "decode") ? ngens : 0;
  for (int r = 0; r < status.nlive; r++)
    status.live[r] = (ViewReq){(int)gens[r].r->id, gens[r].s->number, gens[r].n, gens[r].from, gens[r].gen, gens[r].t1};
  if (!strcmp(phase, "prefill") && g)
    status.nlive = 1, status.live[0] = (ViewReq){(int)g->r->id, g->s->number, g->n, g->from, 0, 0};
  pthread_mutex_unlock(&st_lock);
}

// Decodes the request's images and replaces each image place of g->p by its
// image's span: the prompt's new length, or -1 with a message.
static int open_images(Gen *g, int n, char *msg, size_t cap) {
  Req *r = g->r;
  g->ims = calloc((size_t)r->nimg, sizeof *g->ims), g->nims = r->nimg;
  int total = n;
  for (int i = 0; i < r->nimg; i++) {
    const char *why;
    if (!image_open(&g->ims[i], r->img[i].data, r->img[i].len, &why)) {
      snprintf(msg, cap, "image %d: %s", i + 1, why);
      return -1;
    }
    total += image_span(&g->ims[i]) - 1;
  }
  int *p = xmalloc((size_t)(total + 2) * sizeof *p), m = image_expand(g->p, n, g->ims, r->nimg, p, total);
  if (m < 0) {
    snprintf(msg, cap, "the prompt's image places and its %d images differ", r->nimg);
    free(p);
    return -1;
  }
  free(g->p), g->p = p;
  return m;
}

static void close_images(Gen *g) {
  model_images(0, 0);
  for (int i = 0; i < g->nims; i++) image_free(&g->ims[i]);
  free(g->ims), g->ims = 0, g->nims = 0;
}

// A request refused before its prefill.
static void refuse(Gen *g, const char *message) {
  say("req %llu: refused: %s", (unsigned long long)g->r->id, message);
  event(g->r, (myds4_event){.kind = MYDS4_REFUSED, .message = message});
  close_images(g);
  free(g->p), g->p = 0;
}

// Prefills a request: 1 if it decodes from now on, 0 if it was refused.
static int begin(Req *r, Gen *g) {
  *g = (Gen){.r = r, .sp = r->sp, .p = r->p, .n = r->n, .max_new = r->max_new};
  r->p = 0;  // the generation's from now on
  if (r->seed_set) sampler_seed(&g->sp, r->seed);
  else arc4random_buf(g->sp.seed, sizeof g->sp.seed);
  char message[200] = "";
  int n = g->n;
  if (r->nimg) n = g->n = open_images(g, n, message, sizeof message);  // -1 with its message
  if (!*message && n >= opt.ctx) snprintf(message, sizeof message, "the prompt has %d tokens; the context holds %d", n, opt.ctx);
  if (*message) {
    refuse(g, message);
    return 0;
  }
  if (g->max_new > opt.ctx - n) g->max_new = opt.ctx - n;

  const char *how;
  double t0 = now(), wait = t0 - r->queued;
  Session *s = g->s = pick(g->p, n, &g->from, &how);
  if (!fits(s, model_state_bytes(s->st, n + 1))) {
    snprintf(message, sizeof message, "the conversation's attention state needs %.1f GiB; --kv-cache allows %.1f GiB",
             model_state_bytes(s->st, n + 1) / GiB, cfg.kv_gb);
    refuse(g, message);
    return 0;
  }
  s->busy = 1;
  int same_prompt = g->from == n - 1 && s->saved == n && s->n >= n && g->p[n - 1] == s->ids[n - 1];
  if (same_prompt) g->from = n;
  int thinking = g->p[n - 1] == think_start;
  say("req %llu: %d prompt tokens, session %d (%s), %d reused; thinking %s; at most %ld new tokens, %s; waited %.1f s",
      (unsigned long long)r->id, n, s->number, how, g->from, thinking ? "on" : "off", g->max_new,
      g->sp.temp > 0 ? "sampled" : "greedy", wait);
  set_status("prefill", g);
  int encoded = 0;  // the ViT runs for the images the prefill reaches; the others are in the session's rows
  for (int i = 0; i < g->nims; i++)
    if (g->ims[i].start + image_span(&g->ims[i]) > g->from) model_image(&g->ims[i]), encoded++;
  model_images(g->ims, g->nims);
  if (encoded) say("req %llu: %d of %d images encoded in %.2f s", (unsigned long long)r->id, encoded, g->nims, now() - t0);
  CacheStats c0 = cache_stats;
  const float *lg;
  if (same_prompt) {
    model_state_restore(0);  // a short reply may have selected the current rewindable state
    lg = s->logits;
    model_seek(g->p, n);
  } else {
    int z = 0, start = g->from;
    while (!g->from && z < n && g->p[z] != user) z++;
    if (z > 1 && z < n - 1)  // a new conversation: its rows at its first user turn are saved too
      model_forward(g->p, z, 0), model_state_save(1), s->base = start = z;
    lg = model_forward(g->p + start, n - start, start);
    model_state_save(0);
    if (!s->logits) s->logits = xmalloc(VOCAB * 4);
    memcpy(s->logits, lg, VOCAB * 4);
  }
  close_images(g);
  pthread_mutex_lock(&st_lock);
  memcpy(s->ids, g->p, (size_t)n * sizeof *g->p);
  s->n = n, s->saved = n, s->used = now();
  pthread_mutex_unlock(&st_lock);
  g->t1 = g->tick = now();
  say("req %llu: prefill of %d tokens in %.2f s (%.0f tokens/s), %.2f GB of experts read", (unsigned long long)r->id,
      n - g->from, g->t1 - t0, (n - g->from) / fmax(g->t1 - t0, 1e-9), (cache_stats.bytes - c0.bytes) / 1e9);
  event(r, (myds4_event){.kind = MYDS4_STARTED, .prompt = n, .reused = g->from});
  g->thinking = thinking;
  g->pos = n, g->t = sample_token(lg, VOCAB, &g->sp), g->c1 = cache_stats;
  return 1;
}

// The token t ran at pos: into the session's tokens and the step's output.
static void emit(Gen *g, int t) {
  if (g->thinking) g->reasoning++, g->thinking = t != think_end;
  g->s->ids[g->pos++] = t;
  g->s->n = g->pos;
  g->gen++;
  g->ended = t == eos;  // a kept draft ends the text
  if (!g->ended) g->out[g->nout++] = t;
}

// Whether the generation ends before its next token: at the end token, its
// budget, the context's end, or when it was cancelled.
static int stop(Gen *g) {
  if (atomic_load(&g->r->cancelled)) g->cancelled = 1;
  if (g->ended || g->cancelled) return 1;
  if (g->t == eos) g->gen++, g->ended = 1;
  return g->ended || g->gen >= g->max_new || g->pos >= opt.ctx;
}

static void finish(Gen *g, int shutdown) {
  Req *r = g->r;
  int why = g->cancelled ? MYDS4_CANCELLED : shutdown ? MYDS4_SHUTDOWN : g->ended ? MYDS4_STOP : MYDS4_LENGTH;
  static const char *const name[] = {"stop", "length", "client gone", "stopped by the server's stop"};
  double t2 = now();
  CacheStats c2 = cache_stats;
  event(r, (myds4_event){.kind = MYDS4_DONE, .prompt = g->n, .reused = g->from, .generated = g->gen,
                         .reasoning = g->reasoning, .finish = why});
  g->s->used = now(), g->s->busy = 0;
  uint64_t hits = c2.hits - g->c1.hits + c2.prefetch_hits - g->c1.prefetch_hits, misses = c2.misses - g->c1.misses;
  say("req %llu: %s: %d tokens (%d reasoning) in %.1f s, %.1f tokens/s; experts %.1f%% hit, %.2f GB read; "
      "kv %d sessions, %.2f GiB; available %.1f GiB, footprint %.1f GiB",
      (unsigned long long)r->id, name[why], g->gen, g->reasoning, t2 - g->t1, g->gen / fmax(t2 - g->t1, 1e-9),
      100.0 * hits / fmax(hits + misses, 1), (c2.bytes - g->c1.bytes) / 1e9, live, kv_bytes() / GiB,
      mem_available() / GiB, mem_footprint() / GiB);
  say("req %llu: sampling temperature %.6g, top_p %.6g; %d draft passes, %d drafts accepted", (unsigned long long)r->id,
      g->sp.temp, g->sp.top_p, g->draft_passes, g->draft_accepted);
  fits(g->s, 0);  // a state that grew while decoding may push others out
  pthread_mutex_lock(&st_lock);
  status.requests++, status.prompt_tokens += g->n, status.generated_tokens += g->gen, status.reused_tokens += g->from;
  pthread_mutex_unlock(&st_lock);
  free(g->p), g->p = 0;
}

// One token for each generation: alone with its drafts (--draft),
// or together, each row as it would run alone.
static void advance(void) {
  const float *lg;
  if (ngens == 1) {
    Gen *g = gens;
    int rows[BLOCK + 1] = {g->t}, k = opt.draft, a = 0, next;
    if (k > g->max_new - g->gen - 1) k = (int)g->max_new - g->gen - 1;
    if (k > opt.ctx - g->pos - 1) k = opt.ctx - g->pos - 1;
    model_state_use(g->s->st);
    if (k > 0) { // Sample the target on a matching draft prefix; stop drawing at the first mismatch.
      model_draft(g->t, g->pos, rows + 1, k);
      lg = model_verify(rows, k + 1, g->pos);
      next = sample_verify(lg, VOCAB, rows + 1, k, eos, &g->sp, &a);
      g->draft_passes++;
      g->draft_accepted += a;
      model_accept(rows, a + 1, g->pos);
      lg += (size_t)a * VOCAB;
    } else {
      lg = model_forward(&g->t, 1, g->pos);
      next = sample_token(lg, VOCAB, &g->sp);
    }
    for (int i = 0; i <= a && !g->ended; i++) emit(g, rows[i]);
    g->t = next;
    return;
  }
  State *st[FEW];
  int tokens[FEW], pos[FEW];
  for (int r = 0; r < ngens; r++) st[r] = gens[r].s->st, tokens[r] = gens[r].t, pos[r] = gens[r].pos;
  lg = model_forward_rows(st, tokens, pos, ngens);
  for (int r = 0; r < ngens; r++)
    emit(&gens[r], tokens[r]), gens[r].t = sample_token(lg + (size_t)r * VOCAB, VOCAB, &gens[r].sp);
}

// Admits waiting requests while the batch has room (prefilling each while
// the others wait), ends the generations that are done, and advances the
// others by a token. While no request runs, the experts stay in memory, and
// every second the cache gives slots back if other programs need the memory
// (cache_idle). On the stop, every request ends with MYDS4_SHUTDOWN.
static void worker(void) {
  int shown = 0;
  while (!shutdown_requested()) {
    pthread_mutex_lock(&q_lock);
    while (!first && !ngens && !shutdown_requested()) {
      struct timespec ts;
      clock_gettime(CLOCK_REALTIME, &ts);
      ts.tv_sec += 1;
      if (pthread_cond_timedwait(&q_job, &q_lock, &ts) != ETIMEDOUT) continue;
      pthread_mutex_unlock(&q_lock);
      int n = cache_slots();
      cache_idle();
      if (cache_slots() < n) say("idle: memory short, expert cache %d -> %d slots", n, cache_slots());
      pthread_mutex_lock(&q_lock);
    }
    Req *admit[FEW];
    int na = 0;
    while (first && ngens + na < cfg.batch && !shutdown_requested()) {
      Req *r = admit[na++] = first;
      if (!(first = r->next)) last = 0;
      waiting--;
    }
    pthread_mutex_unlock(&q_lock);
    for (int i = 0; i < na; i++) {
      Req *r = admit[i];
      if (shutdown_requested()) {
        end_waiting(r, MYDS4_SHUTDOWN);
      } else if (atomic_load(&r->cancelled)) {
        say("req %llu: client gone while waiting", (unsigned long long)r->id);
        end_waiting(r, MYDS4_CANCELLED);
      } else if (begin(r, &gens[ngens])) {
        ngens++;
      } else {
        drop_req(r);
      }
    }
    if (shutdown_requested()) break;
    for (int i = 0; i < ngens;)
      if (stop(&gens[i])) {
        Req *r = gens[i].r;
        finish(&gens[i], 0);
        drop_req(r);
        gens[i] = gens[--ngens];
      } else {
        i++;
      }
    if (ngens > 1 && ngens != shown) say("batch: %d requests decoding together", ngens);
    shown = ngens;
    if (!ngens) {
      set_status("idle", 0);
      continue;
    }
    if (shutdown_requested()) break;
    advance();
    for (int r = 0; r < ngens; r++) {
      Gen *g = &gens[r];
      if (g->nout) event(g->r, (myds4_event){.kind = MYDS4_TOKENS, .tokens = g->out, .ntokens = g->nout}), g->nout = 0;
      if (now() - g->tick >= 10)
        g->tick = now(), say("req %llu: %d tokens, %.1f tokens/s", (unsigned long long)g->r->id, g->gen,
                             g->gen / (g->tick - g->t1));
    }
    set_status("decode", gens);
  }
  for (int i = 0; i < ngens; i++) {
    Req *r = gens[i].r;
    if (gens[i].nout) event(r, (myds4_event){.kind = MYDS4_TOKENS, .tokens = gens[i].out, .ntokens = gens[i].nout});
    finish(&gens[i], 1);
    drop_req(r);
  }
  ngens = 0;
  pthread_mutex_lock(&q_lock);
  closed = 1;
  Req *q = first;
  first = last = 0, waiting = 0;
  pthread_mutex_unlock(&q_lock);
  while (q) {
    Req *next = q->next;
    end_waiting(q, MYDS4_SHUTDOWN);
    q = next;
  }
  set_status("idle", 0);
}

// ---- the live view (--tui) ----

// The first position at or after `from` where the sessions' tokens differ,
// or the shorter one's end.
static int differ(const Session *a, const Session *b, int from) {
  int n = a->n < b->n ? a->n : b->n, k = from;
  while (k + 1024 <= n && !memcmp(a->ids + k, b->ids + k, 1024 * sizeof *a->ids)) k += 1024;
  while (k < n && a->ids[k] == b->ids[k]) k++;
  return k;
}

// The sessions' prefix tree from position `from`, depth first: a node is a
// run of positions that all of set share; the sessions part where their
// tokens differ. indent: the tree's lines above this level.
static void tree(View *v, Session **set, int n, int from, const char *indent, int last, int root) {
  int cap = (int)(sizeof v->node / sizeof *v->node), end = set[0]->n;
  if (v->nnodes >= cap || strlen(indent) > sizeof v->node[0].tree - 12) return;
  for (int i = 1; i < n; i++) {
    int d = differ(set[0], set[i], from);
    end = d < end ? d : end;
  }
  char next[64];
  snprintf(next, sizeof next, "%s", indent);
  if (end > from || n == 1) {
    ViewNode *x = &v->node[v->nnodes++];
    *x = (ViewNode){.start = from, .end = end, .sessions = n, .saved = -1};
    snprintf(x->tree, sizeof x->tree, "%s%s", indent, root ? "" : last ? "└─ " : "├─ ");
    for (int i = 0; i < n; i++)
      for (int k = 0; k < set[i]->hits && k < 32; k++) x->hits += set[i]->hit_at[k] > from;
    for (int i = 0; i < n; i++)
      if (set[i]->n == end && !x->session) {
        const Session *s = set[i];
        x->session = s->number, x->busy = s->busy, x->saved = s->saved, x->idle = v->now - s->used;
        x->gib = model_state_bytes(s->st, 0) / GiB;
      }
    snprintf(next, sizeof next, "%s%s", indent, root ? "" : last ? "   " : "│  ");
  }
  Session *rest[24];  // the sessions that go on, in groups by their token at end
  char taken[24] = {0};
  int m = 0;
  for (int i = 0; i < n; i++)
    if (set[i]->n > end) rest[m++] = set[i];
  for (int i = 0; i < m; i++) {
    if (taken[i]) continue;
    Session *group[24];
    int g = 0, left = 0;
    for (int k = i; k < m; k++)
      if (!taken[k] && rest[k]->ids[end] == rest[i]->ids[end]) group[g++] = rest[k], taken[k] = 1;
    for (int k = i + 1; k < m; k++) left += !taken[k];
    tree(v, group, g, end, next, !left, 0);
  }
}

void serve_view(View *v) {
  v->now = now(), v->uptime = (double)(time(0) - started);
  char sub[32] = "", draft[16] = "";
  if (opt.substitute > 0) snprintf(sub, sizeof sub, " · --substitute %.2g", opt.substitute);
  if (opt.draft) snprintf(draft, sizeof draft, " · --draft %d", opt.draft);
  snprintf(v->title, sizeof v->title, "myds4 · %s · %s · %s experts%s%s", cfg.address ? cfg.address : "", MODEL_ID,
           opt.experts ? "GGUF" : "MXFP4", sub, draft);
  pthread_mutex_lock(&q_lock);
  v->queue = waiting;
  pthread_mutex_unlock(&q_lock);
  pthread_mutex_lock(&st_lock);
  v->phase = status.phase, v->phase_s = v->now - status.since, v->nreq = status.nlive;
  memcpy(v->req, status.live, (size_t)status.nlive * sizeof *status.live);
  v->requests = status.requests, v->prompt_tokens = status.prompt_tokens, v->reused_tokens = status.reused_tokens;
  v->generated_tokens = status.generated_tokens;
  memcpy(v->kinds, status.kinds, sizeof v->kinds);
  Session *set[24];  // the most recently used sessions, for the tree
  int n = 0;
  v->sessions = 0;
  for (int i = 0; i < SESSIONS; i++) {
    Session *s = &sessions[i];
    if (!s->st) continue;
    v->sessions++;
    if (n < 24) set[n++] = s;
    else {
      int old = 0;
      for (int k = 1; k < n; k++) old = set[k]->used < set[old]->used ? k : old;
      if (s->used > set[old]->used) set[old] = s;
    }
  }
  v->kv_gib = kv_bytes() / GiB, v->kv_limit_gib = cfg.kv_gb;
  v->nnodes = 0;
  if (n) tree(v, set, n, 0, "", 1, 1);
  pthread_mutex_unlock(&st_lock);
  v->prefill_layer = prefill_layer, v->prefill_n = prefill_n;
  v->decode = decode_stats, v->prefill = prefill_stats, v->cache = cache_stats;
  v->decode_rows = decode_rows, v->prefill_rows = prefill_rows;
  v->decode_gpu_s = decode_gpu_s, v->decode_s = decode_s;
  v->slots = cache_slots(), v->mapped = cache_mapped(), v->cap = cache_cap(), v->max_slots = cache_max();
  v->slot_bytes = cache_slot_bytes(), v->experts = LAYERS * EXPERTS;
  v->uncached_bytes = cache_uncached_bytes(&v->uncached);
  v->dense_bytes = model_dense_bytes(), v->pinned_bytes = cache_bytes() - (uint64_t)v->slots * v->slot_bytes;
  v->footprint = mem_footprint(), v->gpu_inuse = mem_gpu(), v->gpu_limit = mem_gpu_limit();
  mem_info(&v->mem);
}

// ---- GET /v1/status ----

char *myds4_status(void) {
  Str b = {0};
  double t = now();
  pthread_mutex_lock(&q_lock);
  int queue = waiting;
  pthread_mutex_unlock(&q_lock);
  pthread_mutex_lock(&st_lock);
  str_printf(&b, "{\"phase\": \"%s\", \"phase_seconds\": %.1f, \"decoding\": %d, \"request\": %d, "
             "\"prompt_tokens\": %d, \"reused_tokens\": %d, \"generated_tokens\": %d, \"waiting\": %d, \"sessions\": [",
             status.phase, status.since ? t - status.since : 0, status.active, status.request, status.prompt,
             status.reused, status.generated, queue);
  int k = 0;
  for (int i = 0; i < SESSIONS; i++)
    if (sessions[i].st)
      str_printf(&b, "%s{\"session\": %d, \"tokens\": %d, \"idle_seconds\": %.0f}", k++ ? ", " : "",
                 sessions[i].number, sessions[i].n, sessions[i].used ? t - sessions[i].used : 0);
  str_printf(&b, "], \"kv_cache_gib\": %.3f, \"kv_cache_limit_gib\": %.3f, "
             "\"totals\": {\"requests\": %llu, \"prompt_tokens\": %llu, \"generated_tokens\": %llu}, ",
             kv_bytes() / GiB, cfg.kv_gb, (unsigned long long)status.requests,
             (unsigned long long)status.prompt_tokens, (unsigned long long)status.generated_tokens);
  pthread_mutex_unlock(&st_lock);
  CacheStats c = cache_stats;
  str_printf(&b, "\"expert_cache\": {\"slots\": %d, \"gib\": %.1f, \"hits\": %llu, \"misses\": %llu, \"read_gb\": %.1f}, "
             "\"memory\": {\"available_gib\": %.1f, \"footprint_gib\": %.1f}}",
             cache_slots(), cache_bytes() / GiB, (unsigned long long)(c.hits + c.prefetch_hits),
             (unsigned long long)c.misses, c.bytes / 1e9, mem_available() / GiB, mem_footprint() / GiB);
  return b.p;
}

void myds4_free(void *p) { free(p); }

// ---- start and stop ----

static pthread_t thread;
static int running;
static pthread_mutex_t ready_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t ready_cond = PTHREAD_COND_INITIALIZER;
static int ready;

static void *engine(void *arg) {
  myds4_tokenizer(cfg.model);
  eos = tok_find("<｜end▁of▁sentence｜>"), user = tok_find("<｜User｜>");
  think_start = tok_find("<think>"), think_end = tok_find("</think>");
  model_load(cfg.model, &opt);
  kv_limit = cfg.kv_gb * GiB;
  started = time(0), status.since = now();
  if (cfg.tui && !tui_start()) cfg.tui = 0, say("--tui: stderr is not a terminal; the log stays on stderr");
  shutdown_watch();
  say("kv cache: up to %.1f GiB (--kv-cache); a session takes %.0f MiB at first, %.2f GiB at %d tokens", cfg.kv_gb,
      model_state_bytes(0, 0) / 1048576.0, model_state_bytes(0, opt.ctx) / GiB, opt.ctx);
  pthread_mutex_lock(&ready_lock);
  ready = 1;
  pthread_cond_broadcast(&ready_cond);
  pthread_mutex_unlock(&ready_lock);
  worker();
  model_save_predictor();
  online_predictor_report(stderr);
  return arg;
}

static char *copy(const char *s) { return s ? strdup(s) : 0; }

int myds4_start(const myds4_config *c) {
  const char *bad = !c || c->abi != MYDS4_ABI       ? "the engine's interface version differs from the caller's"
                  : !c->model                       ? "-m: no checkpoint directory"
                  : c->ctx < 1 || c->ctx > 1 << 20  ? "--ctx takes 1 to 1048576 tokens (the model's longest context)"
                  : c->batch < 1 || c->batch > FEW  ? "--batch takes 1 to 6 requests"
                  : c->draft < 0 || c->draft > BLOCK ? "--draft takes 1 to 5 drafts"
                  : c->draft_cache && !c->draft     ? "--draft-cache needs --draft"
                  : c->draft_cache && (c->draft_cache < DRAFT_CACHE_MIN || c->draft_cache > DRAFT_EXPERTS)
                      ? "--draft-cache takes 15 to 128 slots per stage (0 keeps the default)"
                  : c->draft_trim && !c->draft      ? "--draft-trim needs --draft"
                  : c->cache_half_life < 0 || c->cache_half_life > CACHE_HALF_LIFE_MAX
                      ? "--cache-half-life takes 1 to 1000000 forward passes"
                  : c->reserve_gb && !(c->reserve_gb >= CACHE_RESERVE_MIN && c->reserve_gb <= CACHE_RESERVE_MAX)
                      ? "--reserve takes 2 to 1024 GiB"
                  : running                         ? "the engine runs already"
                                                    : 0;
  if (bad) {
    fprintf(stderr, "myds4: %s\n", bad);
    return -1;
  }
  cfg = *c;
  cfg.model = copy(c->model), cfg.experts = copy(c->experts), cfg.predictor_state = copy(c->predictor_state);
  cfg.address = copy(c->address);
  opt = (Options){.ctx = c->ctx, .budget_gb = c->cache_gb, .reserve_gb = c->reserve_gb, .cache_half_life = c->cache_half_life,
                  .cache_read_cost = c->cache_read_cost, .dump_fd = -1, .routes_fd = -1, .draft = c->draft,
                  .draft_cache = c->draft_cache, .draft_trim = c->draft_trim, .predictor_state = cfg.predictor_state,
                  .substitute = c->substitute, .gpu_route = c->gpu_route, .experts = cfg.experts, .quiet = 1,
                  .online_predictor = c->online_predictor || c->predictor_state ? 64 : 0};
  if (opt.predictor_state) shutdown_watch();  // an interrupt while loading still saves the state
  pthread_attr_t at;
  pthread_attr_init(&at);
  pthread_attr_setstacksize(&at, 16 << 20);
  pthread_attr_set_qos_class_np(&at, QOS_CLASS_USER_INTERACTIVE, 0);  // its GPU work goes before background work
  if (pthread_create(&thread, &at, engine, 0)) die("cannot start the engine's thread");
  pthread_attr_destroy(&at);
  running = 1;
  pthread_mutex_lock(&ready_lock);
  while (!ready) pthread_cond_wait(&ready_cond, &ready_lock);
  pthread_mutex_unlock(&ready_lock);
  return 0;
}

int myds4_stopping(void) { return shutdown_requested(); }

void myds4_stop(void) {
  shutdown_request();
  pthread_mutex_lock(&q_lock);
  pthread_cond_broadcast(&q_job);
  pthread_mutex_unlock(&q_lock);
  if (running) pthread_join(thread, 0), running = 0;
}
