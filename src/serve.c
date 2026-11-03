// An OpenAI Chat Completions server: POST /v1/chat/completions (streamed as
// server-sent events or as one JSON response), GET /v1/models, GET
// /v1/status and GET /health. HTTP/1.1 with one request per connection, an
// API key on every request but /health, and one model worker (the main
// thread) that serves the queued requests in order. Each conversation keeps
// its attention state in a session, so a request that extends an earlier
// one runs only its new tokens.
#include <CommonCrypto/CommonDigest.h>
#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include <yyjson.h>

#include "myds4.h"

#define HEAD_MAX (64 << 10)  // a request's line and headers
#define BODY_MAX (32 << 20)  // a request's body
#define CONNS 64             // connections at once
#define WAITING 16           // requests waiting for the model
#define SESSIONS 256         // at most, whatever --kv-cache allows
#define FEW (BLOCK + 1)      // requests decoding together at most (model_forward_rows)
#define MODEL_ID "deepseek-v4.1-flash"
#define GiB (1024.0 * 1024 * 1024)

static Options opt;
static Serve cfg;
static int eos, user;
static unsigned char key_hash[CC_SHA256_DIGEST_LENGTH];
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

// What the worker is doing, for GET /v1/status.
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

// ---- HTTP ----

typedef struct {
  char method[8], path[256], key[512];
  int has_key;
  long length;
  char *body;
} Http;

static int send_all(int fd, const char *s, size_t n) {
  while (n) {
    ssize_t k = send(fd, s, n, 0);
    if (k < 0 && errno == EINTR) continue;
    if (k <= 0) return -1;
    s += k, n -= k;
  }
  return 0;
}

static const char *reason(int status) {
  switch (status) {
  case 200: return "OK";
  case 400: return "Bad Request";
  case 401: return "Unauthorized";
  case 404: return "Not Found";
  case 405: return "Method Not Allowed";
  case 408: return "Request Timeout";
  case 411: return "Length Required";
  case 413: return "Content Too Large";
  case 429: return "Too Many Requests";
  case 431: return "Request Header Fields Too Large";
  default: return "Service Unavailable";
  }
}

static void respond(int fd, int status, const char *body, size_t n) {
  Str h = {0};
  str_printf(&h, "HTTP/1.1 %d %s\r\nContent-Type: application/json\r\nContent-Length: %zu\r\nConnection: close\r\n\r\n",
             status, reason(status), n);
  if (!send_all(fd, h.p, h.n)) send_all(fd, body, n);
  free(h.p);
}

static void error_json(Str *b, const char *type, const char *message) {
  str_cat(b, "{\"error\": {\"message\": ");
  json_str(b, message, strlen(message));
  str_printf(b, ", \"type\": \"%s\", \"code\": null}}", type);
}

static void respond_error(int fd, int status, const char *type, const char *message) {
  Str b = {0};
  error_json(&b, type, message);
  respond(fd, status, b.p, b.n);
  free(b.p);
}

static const char SSE[] = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n"
                          "Connection: close\r\n\r\n";

// The request line, the headers this server reads and the body. 0, or the
// status of a malformed request, or -1 when the client left.
static int read_http(int fd, Http *h) {
  char *buf = xmalloc(HEAD_MAX + 1), *end;
  size_t n = 0;
  int chunked = 0, expect = 0, status = 0;
  while (!(end = memmem(buf, n, "\r\n\r\n", 4))) {
    ssize_t k = n == HEAD_MAX ? 0 : recv(fd, buf + n, HEAD_MAX - n, 0);
    if (k <= 0) {
      free(buf);
      return n == HEAD_MAX ? 431 : k < 0 && errno == EAGAIN ? 408 : -1;
    }
    n += k;
  }
  *end = 0;
  char *line = buf, *next = strstr(line, "\r\n");
  if (next) *next = 0, next += 2;
  if (sscanf(line, "%7s %255s HTTP/1.", h->method, h->path) != 2) status = 400;
  char *q = strchr(h->path, '?');
  if (q) *q = 0;
  h->length = -1;
  for (char *l = next; l && l < end && !status; l = next) {
    next = strstr(l, "\r\n");
    if (next) *next = 0, next += 2;
    char *v = strchr(l, ':');
    if (!v) continue;
    *v++ = 0;
    while (*v == ' ' || *v == '\t') v++;
    for (char *e = v + strlen(v); e > v && (e[-1] == ' ' || e[-1] == '\t'); e--) e[-1] = 0;
    if (!strcasecmp(l, "content-length")) {
      char *e;
      h->length = strtol(v, &e, 10);
      if (*e || e == v || h->length < 0) status = 400;
    } else if ((!strcasecmp(l, "authorization") && !strncasecmp(v, "bearer ", 7)) || !strcasecmp(l, "x-api-key")) {
      const char *key = *l == 'a' || *l == 'A' ? v + 7 : v;
      while (*key == ' ') key++;
      h->has_key = strlen(key) < sizeof h->key;
      if (h->has_key) strcpy(h->key, key);
    } else if (!strcasecmp(l, "transfer-encoding")) {
      chunked = 1;
    } else if (!strcasecmp(l, "expect") && !strcasecmp(v, "100-continue")) {
      expect = 1;
    }
  }
  if (!status && chunked) status = 411;
  if (!status && h->length > BODY_MAX) status = 413;
  if (!status && h->length > 0) {
    size_t have = n - (end + 4 - buf);
    if (have > (size_t)h->length) have = h->length;
    h->body = xmalloc(h->length + 1);
    memcpy(h->body, end + 4, have);
    if (expect && have < (size_t)h->length) send_all(fd, "HTTP/1.1 100 Continue\r\n\r\n", 25);
    while (have < (size_t)h->length) {
      ssize_t k = recv(fd, h->body + have, h->length - have, 0);
      if (k <= 0) {
        status = k < 0 && errno == EAGAIN ? 408 : -1;
        break;
      }
      have += k;
    }
  }
  free(buf);
  return status;
}

static int authorized(const Http *h) {
  unsigned char d[CC_SHA256_DIGEST_LENGTH];
  CC_SHA256(h->key, (CC_LONG)strlen(h->key), d);
  return h->has_key && !timingsafe_bcmp(d, key_hash, sizeof d);
}

static void random_id(char *out, const char *prefix) {
  static const char a[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
  size_t p = strlen(prefix);
  memcpy(out, prefix, p);
  for (int i = 0; i < 24; i++) out[p + i] = a[arc4random_uniform(sizeof a - 1)];
  out[p + 24] = 0;
}

static double number(yyjson_val *v, double dflt) {  // numbers are read as raw text
  return yyjson_is_raw(v) ? strtod(yyjson_get_raw(v), 0) : yyjson_is_num(v) ? yyjson_get_num(v) : dflt;
}

// ---- requests waiting for the model ----

enum { QUEUED, RUNNING, DONE };

typedef struct Job {
  struct Job *next;
  int fd, stream, usage, number;
  _Atomic int state, gone;
  yyjson_doc *doc;
  char peer[48], id[40];
  double queued;
  pthread_mutex_t write;  // held by whoever writes to the client
} Job;

static Job *first, *last;
static int waiting;
static pthread_mutex_t q_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t q_job = PTHREAD_COND_INITIALIZER, q_done = PTHREAD_COND_INITIALIZER;
static atomic_int conns, numbers;

static int gone(int fd) {  // the client closed its end
  char c;
  ssize_t k = recv(fd, &c, 1, MSG_PEEK | MSG_DONTWAIT);
  return k == 0 || (k < 0 && errno != EAGAIN && errno != EINTR);
}

// A request's output, streamed as chunks or kept for one response.
typedef struct {
  Job *j;
  const char *model;
  long created;
  Str reasoning, content, calls;
  int ncalls, failed;
} Out;

static void chunk(Out *o, const char *delta, const char *finish) {
  Str s = {0};
  str_printf(&s, "data: {\"id\": \"%s\", \"object\": \"chat.completion.chunk\", \"created\": %ld, \"model\": ",
             o->j->id, o->created);
  json_str(&s, o->model, strlen(o->model));
  str_printf(&s, ", \"choices\": [{\"index\": 0, \"delta\": %s, \"finish_reason\": ", delta);
  if (finish) json_str(&s, finish, strlen(finish));
  else str_cat(&s, "null");
  str_cat(&s, "}]}\n\n");
  if (!o->failed && send_all(o->j->fd, s.p, s.n)) o->failed = 1;
  free(s.p);
}

static void on_text(void *ctx, int reasoning, const char *s, size_t n) {
  Out *o = ctx;
  if (!o->j->stream) {
    str_put(reasoning ? &o->reasoning : &o->content, s, n);
    return;
  }
  Str d = {0};
  str_cat(&d, reasoning ? "{\"reasoning_content\": " : "{\"content\": ");
  json_str(&d, s, n);
  str_cat(&d, "}");
  chunk(o, d.p, 0);
  free(d.p);
}

static void on_call(void *ctx, const char *name, const char *args) {
  Out *o = ctx;
  char id[40];
  random_id(id, "call_");
  Str c = {0};
  if (o->j->stream) str_printf(&c, "{\"tool_calls\": [{\"index\": %d, ", o->ncalls);
  else str_cat(&c, o->ncalls ? ", {" : "{");
  str_printf(&c, "\"id\": \"%s\", \"type\": \"function\", \"function\": {\"name\": ", id);
  json_str(&c, name, strlen(name));
  str_cat(&c, ", \"arguments\": ");
  json_str(&c, args, strlen(args));
  str_cat(&c, o->j->stream ? "}}]}" : "}}");
  if (o->j->stream) chunk(o, c.p, 0);
  else str_put(&o->calls, c.p, c.n);
  o->ncalls++;
  free(c.p);
}

static void usage_json(Str *u, int prompt, int reused, int generated, int reasoning) {
  str_printf(u, "{\"prompt_tokens\": %d, \"completion_tokens\": %d, \"total_tokens\": %d, "
             "\"prompt_tokens_details\": {\"cached_tokens\": %d}, "
             "\"completion_tokens_details\": {\"reasoning_tokens\": %d}, "
             "\"prompt_cache_hit_tokens\": %d, \"prompt_cache_miss_tokens\": %d}",
             prompt, generated, prompt + generated, reused, reasoning, reused, prompt - reused);
}

// A request refused before its output began.
static void refuse(Job *j, const char *message) {
  say("req %d: refused: %s", j->number, message);
  if (!j->stream) {
    respond_error(j->fd, 400, "invalid_request_error", message);
    return;
  }
  Str b = {0};
  str_cat(&b, "data: ");
  error_json(&b, "invalid_request_error", message);
  str_cat(&b, "\n\ndata: [DONE]\n\n");
  send_all(j->fd, b.p, b.n);
  free(b.p);
}

// ---- the requests being served ----

// A request from its prefill to its end: its output, parser and sampler, its
// session, and t, the token at pos it runs next.
typedef struct {
  Job *j;
  Out o;
  Parser ps;
  Sampler sp;
  Session *s;
  char *text;
  int *p, n, from, pos, gen, reasoning, ended, t;
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
  status.request = g ? g->j->number : 0, status.prompt = g ? g->n : 0;
  status.reused = g ? g->from : 0, status.generated = g ? g->gen : 0;
  status.nlive = !strcmp(phase, "decode") ? ngens : 0;
  for (int r = 0; r < status.nlive; r++)
    status.live[r] = (ViewReq){gens[r].j->number, gens[r].s->number, gens[r].n, gens[r].from, gens[r].gen, gens[r].t1};
  if (!strcmp(phase, "prefill") && g)
    status.nlive = 1, status.live[0] = (ViewReq){g->j->number, g->s->number, g->n, g->from, 0, 0};
  pthread_mutex_unlock(&st_lock);
}

// An image part's bytes: an image_url (a string, or an object's url) or a
// source or part holding base64 data. Only data: URLs are read: the server
// fetches no URLs and reads no files for a client.
static uint8_t *image_bytes(yyjson_val *part, size_t *len, const char **why) {
  yyjson_val *iu = yyjson_obj_get(part, "image_url"), *src = yyjson_obj_get(part, "source"), *o = src ? src : part;
  const char *url = iu ? (yyjson_is_str(iu) ? yyjson_get_str(iu) : yyjson_get_str(yyjson_obj_get(iu, "url")))
                       : yyjson_get_str(yyjson_obj_get(o, "url"));
  const char *data = iu ? 0 : yyjson_get_str(yyjson_obj_get(o, "data"));
  if (url && !data) {
    const char *comma = strchr(url, ',');
    if (strncmp(url, "data:", 5) || !comma || !memmem(url, (size_t)(comma - url), ";base64", 7)) {
      *why = "only data: URLs with base64 data are read (the server fetches no URLs and reads no files)";
      return 0;
    }
    data = comma + 1;
  }
  if (!data) {
    *why = "the part holds no image data";
    return 0;
  }
  size_t n = strlen(data);
  uint8_t *out = xmalloc(n / 4 * 3 + 3);
  long k = b64_decode(data, n, out);
  if (k <= 0) {
    free(out);
    *why = "the image data is not base64";
    return 0;
  }
  *len = (size_t)k;
  return out;
}

// Decodes the request's images and replaces each placeholder of g->p by its
// image's span: the prompt's new length, or -1 with a message.
static int open_images(Gen *g, yyjson_val **parts, int np, int n, char *msg, size_t cap) {
  g->ims = calloc((size_t)np, sizeof *g->ims), g->nims = np;
  int total = n;
  for (int i = 0; i < np; i++) {
    size_t len;
    const char *why;
    uint8_t *bytes = image_bytes(parts[i], &len, &why);
    int ok = bytes && image_open(&g->ims[i], bytes, len, &why);
    free(bytes);
    if (!ok) {
      snprintf(msg, cap, "image %d: %s", i + 1, why);
      return -1;
    }
    total += image_span(&g->ims[i]) - 1;
  }
  int *p = xmalloc((size_t)(total + 2) * sizeof *p), m = image_expand(g->p, n, g->ims, np, p, total);
  if (m < 0) {  // each image part has one IMAGE_MARK: not expected
    snprintf(msg, cap, "the prompt's image places and its %d images differ", np);
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

// Renders, tokenizes and prefills a request: 1 if it decodes from now on,
// 0 if it was refused.
static int begin(Job *j, Gen *g) {
  yyjson_val *root = yyjson_doc_get_root(j->doc), *th = yyjson_obj_get(root, "thinking");
  *g = (Gen){.j = j, .o = {.j = j, .created = time(0)}};
  yyjson_val *temperature = yyjson_obj_get(root, "temperature"), *top_p = yyjson_obj_get(root, "top_p");
  yyjson_val *seed = yyjson_obj_get(root, "seed");
  if (yyjson_is_null(temperature)) temperature = 0;
  if (yyjson_is_null(top_p)) top_p = 0;
  if (yyjson_is_null(seed)) seed = 0;
  if (yyjson_is_null(th)) th = 0;
  g->sp.temp = number(temperature, cfg.temp);
  g->sp.top_p = number(top_p, cfg.top_p);
  if ((temperature && !yyjson_is_num(temperature) && !yyjson_is_raw(temperature)) ||
      (top_p && !yyjson_is_num(top_p) && !yyjson_is_raw(top_p)) || !sampler_valid(&g->sp)) {
    refuse(j, "temperature must be a finite nonnegative number and top_p must be in [0, 1]");
    return 0;
  }
  uint64_t seed_value = cfg.seed;
  if (seed) {
    int valid = yyjson_is_uint(seed);
    if (valid) seed_value = yyjson_get_uint(seed);
    else if (yyjson_is_raw(seed)) {
      const char *text = yyjson_get_raw(seed);
      char *end;
      errno = 0;
      seed_value = strtoull(text, &end, 10);
      valid = *text >= '0' && *text <= '9' && (size_t)(end - text) == yyjson_get_len(seed) && !errno;
    }
    if (!valid) {
      refuse(j, "seed must be a nonnegative integer");
      return 0;
    }
  }
  if (seed || cfg.seed_set) sampler_seed(&g->sp, seed_value);
  else arc4random_buf(g->sp.seed, sizeof g->sp.seed);
  g->o.model = yyjson_get_str(yyjson_obj_get(root, "model")) ? yyjson_get_str(yyjson_obj_get(root, "model")) : MODEL_ID;
  const char *type = yyjson_get_str(yyjson_obj_get(th, "type")), *err;
  yyjson_val *re = yyjson_obj_get(root, "reasoning_effort");
  if (yyjson_is_null(re)) re = 0;
  // as in deepseek-recipe: a thinking setting decides; without one, effort "none" turns thinking off
  int thinking = yyjson_is_bool(th) ? yyjson_get_bool(th)
                 : th               ? !type || strcmp(type, "disabled")
                 : re               ? !(yyjson_get_str(re) && !strcmp(yyjson_get_str(re), "none"))
                                    : cfg.thinking;
  int effort = chat_effort(re, cfg.effort);
  yyjson_val **parts = 0;
  int nparts = 0;
  g->text = chat_prompt(yyjson_obj_get(root, "messages"), yyjson_obj_get(root, "tools"), thinking, effort, IMAGE_MARK,
                        &err, &parts, &nparts);
  size_t len = g->text ? strlen(g->text) : 0;
  g->p = xmalloc((len + 2) * sizeof *g->p);
  int n = g->n = g->text ? tok_encode_chat(g->text, g->p, (int)len + 2) : 0;
  g->max_new =
      (long)number(yyjson_obj_get(root, "max_completion_tokens"), number(yyjson_obj_get(root, "max_tokens"), opt.ctx));
  char message[200] = "";
  if (!g->text) snprintf(message, sizeof message, "%s", err);
  else n = g->n = open_images(g, parts, nparts, n, message, sizeof message); // -1 with its message
  free(parts);
  if (*message) {
    refuse(j, message);
    close_images(g);
    free(g->text), free(g->p);
    return 0;
  }
  if (n >= opt.ctx) snprintf(message, sizeof message, "the prompt has %d tokens; the context holds %d", n, opt.ctx);
  else if (g->max_new < 1) snprintf(message, sizeof message, "max_tokens must be at least 1");
  if (*message) {
    refuse(j, message);
    close_images(g);
    free(g->text), free(g->p);
    return 0;
  }
  if (g->max_new > opt.ctx - n) g->max_new = opt.ctx - n;

  const char *how;
  double t0 = now(), wait = t0 - j->queued;
  Session *s = g->s = pick(g->p, n, &g->from, &how);
  if (!fits(s, model_state_bytes(s->st, n + 1))) {
    snprintf(message, sizeof message, "the conversation's attention state needs %.1f GiB; --kv-cache allows %.1f GiB",
             model_state_bytes(s->st, n + 1) / GiB, cfg.kv_gb);
    refuse(j, message);
    close_images(g);
    free(g->text), free(g->p);
    return 0;
  }
  s->busy = 1;
  int same_prompt = g->from == n - 1 && s->saved == n && s->n >= n && g->p[n - 1] == s->ids[n - 1];
  if (same_prompt) g->from = n;
  char mode[24] = "off";
  if (thinking) snprintf(mode, sizeof mode, "effort %d", effort);
  say("req %d: %d prompt tokens, session %d (%s), %d reused; thinking %s; at most %ld new tokens, %s; waited %.1f s",
      j->number, n, s->number, how, g->from, mode, g->max_new, g->sp.temp > 0 ? "sampled" : "greedy", wait);
  set_status("prefill", g);
  int encoded = 0;  // the ViT runs for the images the prefill reaches; the others are in the session's rows
  for (int i = 0; i < g->nims; i++)
    if (g->ims[i].start + image_span(&g->ims[i]) > g->from) model_image(&g->ims[i]), encoded++;
  model_images(g->ims, g->nims);
  if (encoded) say("req %d: %d of %d images encoded in %.2f s", j->number, encoded, g->nims, now() - t0);
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
  say("req %d: prefill of %d tokens in %.2f s (%.0f tokens/s), %.2f GB of experts read", j->number, n - g->from,
      g->t1 - t0, (n - g->from) / fmax(g->t1 - t0, 1e-9), (cache_stats.bytes - c0.bytes) / 1e9);
  if (j->stream) chunk(&g->o, "{\"role\": \"assistant\", \"content\": \"\"}", 0);
  parser_init(&g->ps, len >= 7 && !strcmp(g->text + len - 7, "<think>"));
  g->ps.text = on_text, g->ps.call = on_call, g->ps.ctx = &g->o;
  g->pos = n, g->t = sample_token(lg, VOCAB, &g->sp), g->c1 = cache_stats;
  return 1;
}

// The token t ran at pos: into the session's tokens and the output.
static void emit(Gen *g, int t) {
  int len;
  g->reasoning += g->ps.state == P_REASON;
  g->s->ids[g->pos++] = t;
  g->s->n = g->pos;
  parser_feed(&g->ps, tok_piece(t, &len), len);
  g->gen++;
  g->ended = t == eos;  // a kept draft ends the text
}

// Whether the generation ends before its next token: at the end token, its
// budget, the context's end, or when its client left.
static int stop(Gen *g) {
  int len;
  if (!g->o.failed && g->gen % 16 == 0 && gone(g->j->fd)) g->o.failed = 1;
  if (g->ended || g->o.failed) return 1;
  if (g->t == eos) {
    parser_feed(&g->ps, tok_piece(eos, &len), len);
    g->gen++, g->ended = 1;
  }
  return g->ended || g->gen >= g->max_new || g->pos >= opt.ctx;
}

static void finish(Gen *g) {
  Job *j = g->j;
  Out *o = &g->o;
  int whole = parser_end(&g->ps);
  const char *finish = o->failed ? "cancelled" : !g->ended ? "length" : o->ncalls ? "tool_calls" : "stop";
  double t2 = now();
  CacheStats c2 = cache_stats;
  if (!o->failed) {
    Str u = {0}, b = {0};
    usage_json(&u, g->n, g->from, g->gen, g->reasoning);
    if (j->stream) {
      chunk(o, "{}", finish);
      if (j->usage) {
        str_printf(&b, "data: {\"id\": \"%s\", \"object\": \"chat.completion.chunk\", \"created\": %ld, \"model\": ",
                   j->id, o->created);
        json_str(&b, o->model, strlen(o->model));
        str_printf(&b, ", \"choices\": [], \"usage\": %s}\n\n", u.p);
      }
      str_cat(&b, "data: [DONE]\n\n");
      if (send_all(j->fd, b.p, b.n)) o->failed = 1;
    } else {
      str_printf(&b, "{\"id\": \"%s\", \"object\": \"chat.completion\", \"created\": %ld, \"model\": ", j->id, o->created);
      json_str(&b, o->model, strlen(o->model));
      str_cat(&b, ", \"choices\": [{\"index\": 0, \"message\": {\"role\": \"assistant\", \"content\": ");
      json_str(&b, o->content.p ? o->content.p : "", o->content.n);
      if (o->reasoning.n) str_cat(&b, ", \"reasoning_content\": "), json_str(&b, o->reasoning.p, o->reasoning.n);
      if (o->ncalls) str_printf(&b, ", \"tool_calls\": [%s]", o->calls.p);
      str_printf(&b, "}, \"finish_reason\": \"%s\"}], \"usage\": %s}", finish, u.p);
      respond(j->fd, 200, b.p, b.n);
    }
    free(u.p), free(b.p);
  }
  g->s->used = now(), g->s->busy = 0;
  uint64_t hits = c2.hits - g->c1.hits + c2.prefetch_hits - g->c1.prefetch_hits, misses = c2.misses - g->c1.misses;
  say("req %d: %s%s: %d tokens (%d reasoning, %d tool calls) in %.1f s, %.1f tokens/s; experts %.1f%% hit, "
      "%.2f GB read; kv %d sessions, %.2f GiB; available %.1f GiB, footprint %.1f GiB",
      j->number, o->failed ? "client gone" : finish, whole ? "" : " (tool block unfinished)", g->gen, g->reasoning,
      o->ncalls, t2 - g->t1, g->gen / fmax(t2 - g->t1, 1e-9), 100.0 * hits / fmax(hits + misses, 1),
      (c2.bytes - g->c1.bytes) / 1e9, live, kv_bytes() / GiB, mem_available() / GiB, mem_footprint() / GiB);
  say("req %d: sampling temperature %.6g, top_p %.6g; %d draft passes, %d drafts accepted",
      j->number, g->sp.temp, g->sp.top_p, g->draft_passes, g->draft_accepted);
  fits(g->s, 0);  // a state that grew while decoding may push others out
  pthread_mutex_lock(&st_lock);
  status.requests++, status.prompt_tokens += g->n, status.generated_tokens += g->gen, status.reused_tokens += g->from;
  pthread_mutex_unlock(&st_lock);
  free(o->reasoning.p), free(o->content.p), free(o->calls.p), free(g->text), free(g->p);
}

static void done(Job *j) {  // its client's thread may answer and close
  pthread_mutex_unlock(&j->write);
  pthread_mutex_lock(&q_lock);
  j->state = DONE;
  pthread_cond_broadcast(&q_done);
  pthread_mutex_unlock(&q_lock);
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
// (cache_idle).
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
    if (shutdown_requested()) {
      pthread_mutex_unlock(&q_lock);
      return;
    }
    Job *admit[FEW];
    int na = 0;
    while (first && ngens + na < cfg.batch && !shutdown_requested()) {
      Job *j = admit[na++] = first;
      if (!(first = j->next)) last = 0;
      waiting--;
      j->state = RUNNING;
    }
    pthread_mutex_unlock(&q_lock);
    for (int i = 0; i < na; i++) {
      if (shutdown_requested()) return;
      pthread_mutex_lock(&admit[i]->write);
      if (admit[i]->gone) say("req %d: client gone while waiting", admit[i]->number), done(admit[i]);
      else if (begin(admit[i], &gens[ngens])) ngens++;
      else done(admit[i]);
    }
    if (shutdown_requested()) return;
    for (int i = 0; i < ngens;)
      if (stop(&gens[i])) {
        finish(&gens[i]), done(gens[i].j);
        gens[i] = gens[--ngens];
        gens[i].ps.ctx = &gens[i].o;  // the parser writes to its own generation's output
      } else {
        i++;
      }
    if (ngens > 1 && ngens != shown) say("batch: %d requests decoding together", ngens);
    shown = ngens;
    if (!ngens) {
      set_status("idle", 0);
      continue;
    }
    if (shutdown_requested()) return;
    advance();
    for (int r = 0; r < ngens; r++) {
      Gen *g = &gens[r];
      if (now() - g->tick >= 10)
        g->tick = now(), say("req %d: %d tokens, %.1f tokens/s", g->j->number, g->gen, g->gen / (g->tick - g->t1));
    }
    set_status("decode", gens);
  }
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
  snprintf(v->title, sizeof v->title, "myds4 · %s:%d · %s · %s experts%s%s", cfg.host, cfg.port, MODEL_ID,
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

// ---- the endpoints ----

static void chat(int fd, const Http *h, const char *peer) {
  yyjson_read_err e;
  yyjson_doc *doc = h->length > 0 ? yyjson_read_opts(h->body, h->length, YYJSON_READ_NUMBER_AS_RAW, 0, &e) : 0;
  yyjson_val *root = yyjson_doc_get_root(doc), *tools = yyjson_obj_get(root, "tools"),
             *msgs = yyjson_obj_get(root, "messages"), *role = yyjson_obj_get(yyjson_arr_get_last(msgs), "role");
  const char *bad = !yyjson_is_obj(root) ? "the body is not a JSON object"
                  : !yyjson_arr_size(msgs) ? "messages must be a non-empty array"
                  : yyjson_equals_str(role, "assistant") ? "the last message must not be the assistant's"
                  : number(yyjson_obj_get(root, "n"), 1) != 1 ? "only n = 1 is supported" : 0;
  if (bad) {
    say("%s: 400: %s", peer, bad);
    respond_error(fd, 400, "invalid_request_error", bad);
    yyjson_doc_free(doc);
    return;
  }
  Job j = {.fd = fd, .doc = doc, .queued = now(), .number = ++numbers};
  j.stream = yyjson_get_bool(yyjson_obj_get(root, "stream"));
  j.usage = yyjson_get_bool(yyjson_obj_get(yyjson_obj_get(root, "stream_options"), "include_usage"));
  random_id(j.id, "chatcmpl-");
  snprintf(j.peer, sizeof j.peer, "%s", peer);
  pthread_mutex_init(&j.write, 0);
  pthread_mutex_lock(&q_lock);
  int full = waiting >= WAITING;
  waiting += !full;
  pthread_mutex_unlock(&q_lock);
  say("req %d from %s: %zu messages, %zu tools, %s%s", j.number, peer, yyjson_arr_size(yyjson_obj_get(root, "messages")),
      yyjson_arr_size(tools), j.stream ? "streamed" : "one response", full ? "; busy, refused" : "");
  if (full) {
    respond_error(fd, 429, "rate_limit_error", "the server is busy: too many requests are waiting");
  } else {
    if (j.stream) send_all(fd, SSE, sizeof SSE - 1);
    pthread_mutex_lock(&q_lock);
    if (last) last->next = &j;
    else first = &j;
    last = &j;
    pthread_cond_signal(&q_job);
    while (j.state != DONE) {
      if (!j.stream || j.state != QUEUED) {
        pthread_cond_wait(&q_done, &q_lock);
        continue;
      }
      struct timespec ts;  // a comment every 15 s keeps a waiting stream open
      clock_gettime(CLOCK_REALTIME, &ts);
      ts.tv_sec += 15;
      if (pthread_cond_timedwait(&q_done, &q_lock, &ts) != ETIMEDOUT) continue;
      pthread_mutex_unlock(&q_lock);
      pthread_mutex_lock(&j.write);
      if (j.state == QUEUED && send_all(fd, ": waiting\n\n", 11)) j.gone = 1;
      pthread_mutex_unlock(&j.write);
      pthread_mutex_lock(&q_lock);
    }
    pthread_mutex_unlock(&q_lock);
  }
  pthread_mutex_destroy(&j.write);
  yyjson_doc_free(doc);
}

static void models(int fd) {
  Str b = {0};
  str_printf(&b, "{\"object\": \"list\", \"data\": [{\"id\": \"%s\", \"object\": \"model\", \"created\": %ld, "
             "\"owned_by\": \"myds4\", \"name\": \"DeepSeek V4.1 Flash (myds4)\", \"context_length\": %d, "
             "\"max_model_len\": %d, \"max_output_tokens\": %d}]}",
             MODEL_ID, started, opt.ctx, opt.ctx, opt.ctx < 65536 ? opt.ctx : 65536);
  respond(fd, 200, b.p, b.n);
  free(b.p);
}

static void report(int fd) {
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
  respond(fd, 200, b.p, b.n);
  free(b.p);
}

typedef struct {
  int fd;
  char peer[48];
} Conn;

static void *connection(void *arg) {
  Conn *c = arg;
  struct timeval tv = {60, 0};
  setsockopt(c->fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
  setsockopt(c->fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof tv);
  setsockopt(c->fd, IPPROTO_TCP, TCP_NODELAY, &(int){1}, sizeof(int));
  Http h = {0};
  int status = read_http(c->fd, &h), get = !strcmp(h.method, "GET"), post = !strcmp(h.method, "POST");
  const char *path = !strncmp(h.path, "/v1/", 4) ? h.path + 3 : h.path;
  if (status > 0) {
    say("%s: %d %s", c->peer, status, reason(status));
    respond_error(c->fd, status, "invalid_request_error", reason(status));
  } else if (status < 0) {
  } else if (!strcmp(path, "/health")) {
    respond(c->fd, 200, "{\"status\": \"ok\"}", 16);
  } else if (!authorized(&h)) {
    say("%s: 401: %s %s with %s", c->peer, h.method, h.path, h.has_key ? "a wrong API key" : "no API key");
    respond_error(c->fd, 401, "authentication_error", "a valid API key is required (Authorization: Bearer KEY)");
  } else if (!strcmp(path, "/chat/completions")) {
    if (post) chat(c->fd, &h, c->peer);
    else respond_error(c->fd, 405, "invalid_request_error", "use POST");
  } else if (!strcmp(path, "/models") || !strcmp(path, "/status")) {
    if (!get) respond_error(c->fd, 405, "invalid_request_error", "use GET");
    else if (path[1] == 'm') models(c->fd);
    else report(c->fd);
  } else {
    respond_error(c->fd, 404, "invalid_request_error", "no such endpoint");
  }
  close(c->fd);
  free(h.body), free(c);
  atomic_fetch_sub(&conns, 1);
  return 0;
}

static void *acceptor(void *arg) {
  int ls = *(int *)arg;
  pthread_attr_t at;
  pthread_attr_init(&at);
  pthread_attr_setdetachstate(&at, PTHREAD_CREATE_DETACHED);
  while (!shutdown_requested()) {
    struct pollfd ready = {.fd = ls, .events = POLLIN};
    int n = poll(&ready, 1, 1000);
    if (shutdown_requested()) break;
    if (n < 0) {
      if (errno != EINTR) say("listen: %s", strerror(errno));
      continue;
    }
    if (!n) continue;
    if (ready.revents & (POLLERR | POLLHUP | POLLNVAL)) break;
    struct sockaddr_in a;
    socklen_t al = sizeof a;
    int fd = accept(ls, (struct sockaddr *)&a, &al);
    if (fd < 0) {
      if (errno != EINTR && errno != ECONNABORTED && errno != EAGAIN && errno != EWOULDBLOCK)
        say("accept: %s", strerror(errno)), usleep(100000);
      continue;
    }
    int flags = fcntl(fd, F_GETFL);
    if (shutdown_requested() || flags < 0 || fcntl(fd, F_SETFL, flags & ~O_NONBLOCK)) {
      close(fd);
      continue;
    }
    Conn *c = xmalloc(sizeof *c);
    c->fd = fd;
    inet_ntop(AF_INET, &a.sin_addr, c->peer, sizeof c->peer);
    pthread_t t;
    if (atomic_fetch_add(&conns, 1) >= CONNS || pthread_create(&t, &at, connection, c)) {
      respond_error(fd, 503, "server_error", "too many connections");
      close(fd), free(c);
      atomic_fetch_sub(&conns, 1);
    }
  }
  pthread_attr_destroy(&at);
  return 0;
}

void serve(const char *model, const Options *o, const Serve *s) {
  opt = *o, cfg = *s;
  opt.quiet = 1;
  signal(SIGPIPE, SIG_IGN);
  char *key = cfg.key_file ? read_file(cfg.key_file, 0) : getenv("MYDS4_API_KEY");
  size_t kn = key ? strlen(key) : 0;
  while (kn && (key[kn - 1] == '\n' || key[kn - 1] == '\r' || key[kn - 1] == ' ')) kn--;
  if (kn < 16 || kn > 256) die("the API key (MYDS4_API_KEY or --api-key-file) must have 16 to 256 characters");
  CC_SHA256(key, (CC_LONG)kn, key_hash);
  struct sockaddr_in addr = {.sin_family = AF_INET, .sin_port = htons(cfg.port)};
  const char *host = !strcmp(cfg.host, "localhost") ? "127.0.0.1" : cfg.host;
  if (inet_pton(AF_INET, host, &addr.sin_addr) != 1) die("--host takes an IPv4 address: %s", cfg.host);
  int ls = socket(AF_INET, SOCK_STREAM, 0);
  setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &(int){1}, sizeof(int));
  if (ls < 0 || bind(ls, (struct sockaddr *)&addr, sizeof addr) || listen(ls, 64))
    die("cannot listen on %s:%d: %s", host, cfg.port, strerror(errno));
  int flags = fcntl(ls, F_GETFL);
  if (flags < 0 || fcntl(ls, F_SETFL, flags | O_NONBLOCK)) die("cannot configure listening socket");
  tok_load(model);
  eos = tok_find("<｜end▁of▁sentence｜>"), user = tok_find("<｜User｜>");
  model_load(model, &opt);
  kv_limit = cfg.kv_gb * GiB;
  started = time(0), status.since = now();
  if (cfg.tui && !tui_start()) cfg.tui = 0, say("--tui: stderr is not a terminal; the log stays on stderr");
  shutdown_watch();
  say("kv cache: up to %.1f GiB (--kv-cache); a session takes %.0f MiB at first, %.2f GiB at %d tokens", cfg.kv_gb,
      model_state_bytes(0, 0) / 1048576.0, model_state_bytes(0, opt.ctx) / GiB, opt.ctx);
  if (ntohl(addr.sin_addr.s_addr) >> 24 != 127)
    say("warning: plain HTTP on %s: keys and conversations cross the network unencrypted", host);
  say("serving %s at http://%s:%d/v1 (context %d tokens%s%s)", MODEL_ID, host, cfg.port, opt.ctx,
      opt.experts ? ", GGUF experts" : "", opt.draft ? ", drafts" : "");
  say("request defaults: temperature %.6g, top_p %.6g, thinking %s, effort %d", cfg.temp, cfg.top_p,
      cfg.thinking ? "on" : "off", cfg.effort);
  pthread_t t;
  if (pthread_create(&t, 0, acceptor, &ls)) die("cannot start the server's thread");
  worker();
  shutdown(ls, SHUT_RDWR);
  close(ls);
  pthread_join(t, 0);
  model_save_predictor();
  online_predictor_report(stderr);
}
