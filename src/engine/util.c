// Errors, time, a small JSON parser, the safetensors index and GGUF headers.
#include <fcntl.h>
#include <libgen.h>
#include <limits.h>
#include <mach-o/dyld.h>
#include <signal.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "myds4.h"

void (*die_hook)(void);

static volatile sig_atomic_t stop_requested;

// Signal handlers only request a stop. The model owner saves state after
// its current forward pass, outside the signal handler.
static void request_shutdown(int signal_number) { stop_requested = 1; }
void shutdown_request(void) { stop_requested = 1; }  // myds4_request_stop: the live view's q

void shutdown_watch(void) {
  struct sigaction action = {.sa_handler = request_shutdown, .sa_flags = SA_RESTART};
  sigemptyset(&action.sa_mask);
  // The engine installs these again once the model is loaded; an interrupt
  // received while it was loading stays requested.
  if (sigaction(SIGINT, &action, 0) || sigaction(SIGTERM, &action, 0))
    die("cannot install clean shutdown handlers");
}

int shutdown_requested(void) { return stop_requested != 0; }

_Noreturn void die(const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  if (die_hook) die_hook();
  fputs("myds4: ", stderr);
  vfprintf(stderr, fmt, ap);
  fputc('\n', stderr);
  exit(1);
}

double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + t.tv_nsec * 1e-9;
}

void *xmalloc(size_t n) {
  void *p = calloc(1, n ? n : 1);
  if (!p) die("out of memory (%zu bytes)", n);
  return p;
}

void str_put(Str *s, const char *t, size_t n) {
  if (s->n + n + 1 > s->cap) {
    s->cap = (s->n + n + 1) * 2;
    if (!(s->p = realloc(s->p, s->cap))) die("out of memory (%zu bytes)", s->cap);
  }
  memcpy(s->p + s->n, t, n);
  s->n += n;
  s->p[s->n] = 0;
}

void str_cat(Str *s, const char *t) { str_put(s, t, strlen(t)); }

void str_printf(Str *s, const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  int n = vsnprintf(0, 0, fmt, ap);
  va_end(ap);
  str_put(s, "", 0);
  if (s->n + n + 1 > s->cap && !(s->p = realloc(s->p, s->cap = (s->n + n + 1) * 2))) die("out of memory");
  va_start(ap, fmt);
  vsnprintf(s->p + s->n, n + 1, fmt, ap);
  va_end(ap);
  s->n += n;
}

// One process that loads the model at a time, since each takes most of the
// memory: an exclusive lock (flock, as lockf(1) takes it) on myds4.lock
// beside the executable, or on $MYDS4_LOCK, held until the process ends.
void lock_model(void) {
  char exe[PATH_MAX], real[PATH_MAX], path[PATH_MAX + 16];
  uint32_t n = sizeof exe;
  if (getenv("MYDS4_LOCK")) snprintf(path, sizeof path, "%s", getenv("MYDS4_LOCK"));
  else if (_NSGetExecutablePath(exe, &n) || !realpath(exe, real)) die("cannot find the executable's directory");
  else snprintf(path, sizeof path, "%s/myds4.lock", dirname(real));
  int fd = open(path, O_RDWR | O_CREAT, 0644);
  if (fd < 0) die("cannot open the lock %s", path);
  if (flock(fd, LOCK_EX | LOCK_NB)) {
    fprintf(stderr, "myds4: another process holds %s; waiting for it to end\n", path);
    if (flock(fd, LOCK_EX)) die("cannot lock %s", path);
  }
}  // the descriptor stays open: the lock lasts as long as the process

// Base64 (RFC 4648, also its URL alphabet), up to padding, skipping
// whitespace: the bytes written to out, or -1 for another character.
long b64_decode(const char *s, size_t n, uint8_t *out) {
  uint32_t acc = 0;
  int bits = 0;
  long k = 0;
  for (size_t i = 0; i < n && s[i] != '='; i++) {
    unsigned char c = (unsigned char)s[i];
    int v = c >= 'A' && c <= 'Z' ? c - 'A' : c >= 'a' && c <= 'z' ? c - 'a' + 26 : c >= '0' && c <= '9' ? c - '0' + 52
          : c == '+' || c == '-' ? 62 : c == '/' || c == '_' ? 63 : -1;
    if (v < 0) {
      if (c == ' ' || c == '\n' || c == '\r' || c == '\t') continue;
      return -1;
    }
    acc = acc << 6 | (uint32_t)v, bits += 6;
    if (bits >= 8) bits -= 8, out[k++] = (uint8_t)(acc >> bits);
  }
  return k;
}

char *read_file(const char *path, size_t *len) {
  int fd = open(path, O_RDONLY);
  struct stat st;
  if (fd < 0 || fstat(fd, &st)) die("cannot open %s", path);
  char *p = xmalloc(st.st_size + 1);
  for (off_t got = 0; got < st.st_size;) {
    ssize_t r = pread(fd, p + got, st.st_size - got, got);
    if (r <= 0) die("cannot read %s", path);
    got += r;
  }
  close(fd);
  if (len) *len = st.st_size;
  return p;
}

// ---- JSON: parsed in place; strings are unescaped into the source text ----

static char *jp;

static void skip(void) {
  while (*jp == ' ' || *jp == '\n' || *jp == '\r' || *jp == '\t') jp++;
}

static int utf8(char *o, unsigned c) {
  if (c < 0x80) return o[0] = c, 1;
  if (c < 0x800) return o[0] = 0xC0 | c >> 6, o[1] = 0x80 | (c & 63), 2;
  if (c < 0x10000)
    return o[0] = 0xE0 | c >> 12, o[1] = 0x80 | (c >> 6 & 63), o[2] = 0x80 | (c & 63), 3;
  o[0] = 0xF0 | c >> 18, o[1] = 0x80 | (c >> 12 & 63), o[2] = 0x80 | (c >> 6 & 63);
  return o[3] = 0x80 | (c & 63), 4;
}

static const char *jstring(int *n) {
  char *o = ++jp, *start = o;
  while (*jp != '"') {
    if (*jp != '\\') { *o++ = *jp++; continue; }
    char e = *++jp;
    jp++;
    if (e == 'u') {
      unsigned c = strtoul((char[5]){jp[0], jp[1], jp[2], jp[3], 0}, 0, 16);
      jp += 4;
      if (c >= 0xD800 && c < 0xDC00 && jp[0] == '\\' && jp[1] == 'u') {
        unsigned lo = strtoul((char[5]){jp[2], jp[3], jp[4], jp[5], 0}, 0, 16);
        c = 0x10000 + ((c - 0xD800) << 10) + (lo - 0xDC00);
        jp += 6;
      }
      o += utf8(o, c);
    } else {
      *o++ = e == 'n' ? '\n' : e == 't' ? '\t' : e == 'r' ? '\r' : e == 'b' ? '\b'
           : e == 'f' ? '\f' : e;
    }
  }
  jp++;
  *n = o - start;
  *o = 0;
  return start;
}

static void jvalue(Json *v) {
  skip();
  memset(v, 0, sizeof *v);
  if (*jp == '{' || *jp == '[') {
    int obj = *jp++ == '{', cap = 8;
    v->type = obj ? J_OBJ : J_ARR;
    v->kid = xmalloc(cap * sizeof(Json));
    for (skip(); *jp != '}' && *jp != ']'; skip()) {
      if (*jp == ',') { jp++; continue; }
      if (v->n == cap) v->kid = realloc(v->kid, (cap *= 2) * sizeof(Json));
      const char *key = 0;
      int kn;
      if (obj) {
        skip();
        key = jstring(&kn);
        skip();
        jp++;  // ':'
      }
      jvalue(&v->kid[v->n]);
      v->kid[v->n++].key = key;
    }
    jp++;
  } else if (*jp == '"') {
    v->type = J_STR;
    v->s = jstring(&v->n);
  } else if (*jp == 't' || *jp == 'f') {
    v->type = J_BOOL;
    v->num = *jp == 't';
    jp += *jp == 't' ? 4 : 5;
  } else if (*jp == 'n') {
    jp += 4;
  } else {
    v->type = J_NUM;
    v->num = strtod(jp, &jp);
  }
}

Json *json_parse(char *text) {
  Json *v = xmalloc(sizeof *v);
  jp = text;
  jvalue(v);
  return v;
}

Json *json_get(const Json *o, const char *key) {
  for (int i = 0; o && i < o->n; i++)
    if (o->kid[i].key && !strcmp(o->kid[i].key, key)) return &o->kid[i];
  return 0;
}

// ---- safetensors: every tensor of every shard, by name ----

typedef struct { const char *name; Tensor t; } Entry;
static Entry *table;
static size_t nentries, tcap;

static uint64_t hash(const char *s) {
  uint64_t h = 1469598103934665603ull;
  while (*s) h = (h ^ (uint8_t)*s++) * 1099511628211ull;
  return h;
}

static Entry *slot(const char *name) {
  size_t i = hash(name) & (tcap - 1);
  while (table[i].name && strcmp(table[i].name, name)) i = (i + 1) & (tcap - 1);
  return &table[i];
}

void st_open(const char *dir) {
  char path[4096];
  snprintf(path, sizeof path, "%s/model.safetensors.index.json", dir);
  Json *idx = json_get(json_parse(read_file(path, 0)), "weight_map");
  if (!idx) die("%s has no weight_map", path);
  tcap = 1;
  while (tcap < 2 * (size_t)idx->n) tcap *= 2;
  table = xmalloc(tcap * sizeof *table);
  const char *done[256];
  int ndone = 0;
  for (int i = 0; i < idx->n; i++) {
    const char *file = idx->kid[i].s;
    int seen = 0;
    for (int j = 0; j < ndone; j++) seen |= !strcmp(done[j], file);
    if (seen) continue;
    done[ndone++] = file;
    snprintf(path, sizeof path, "%s/%s", dir, file);
    int fd = open(path, O_RDONLY);
    uint64_t n;
    if (fd < 0 || pread(fd, &n, 8, 0) != 8) die("cannot read %s", path);
    char *h = xmalloc(n + 1);
    if (pread(fd, h, n, 8) != (ssize_t)n) die("cannot read %s", path);
    fcntl(fd, F_NOCACHE, 1);
    Json *hdr = json_parse(h);
    for (int k = 0; k < hdr->n; k++) {
      Json *e = &hdr->kid[k];
      if (!strcmp(e->key, "__metadata__")) continue;
      Entry *en = slot(e->key);
      if (en->name) continue;
      en->name = e->key;
      nentries++;
      const char *dt = json_get(e, "dtype")->s;
      Json *sh = json_get(e, "shape"), *of = json_get(e, "data_offsets");
      Tensor *t = &en->t;
      t->fd = fd;
      t->off = 8 + n + (uint64_t)of->kid[0].num;
      t->size = (uint64_t)of->kid[1].num - (uint64_t)of->kid[0].num;
      t->dtype = !strcmp(dt, "BF16") ? 'b' : !strcmp(dt, "F32") ? 'f' : !strcmp(dt, "F8_E4M3") ? '8'
               : !strcmp(dt, "F8_E8M0") ? 'e' : !strcmp(dt, "I8") ? 'i' : '?';
      t->ndim = sh->n;
      for (int d = 0; d < sh->n && d < 4; d++) t->shape[d] = (int64_t)sh->kid[d].num;
    }
  }
}

static const char *format(const char *fmt, va_list ap) {
  static char name[256];
  vsnprintf(name, sizeof name, fmt, ap);
  return name;
}

Tensor st_get(const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  const char *name = format(fmt, ap);
  va_end(ap);
  Entry *e = slot(name);
  if (!e->name) die("checkpoint has no tensor %s", name);
  return e->t;
}

int st_has(const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  const char *name = format(fmt, ap);
  va_end(ap);
  return slot(name)->name != 0;
}

// ---- GGUF: the header's key-value pairs (skipped, but for the data
// alignment) and tensor infos of every shard ----

typedef struct { const char *name; Gguf t; } GgEntry;
static GgEntry *gg;
static int ngg;
typedef struct { int fd; uint8_t *b; uint64_t n, at; } Rd;

static const uint8_t *take_bytes(Rd *r, uint64_t n) {
  while (r->at + n > r->n) {  // the header is read in growing chunks
    uint64_t want = r->n ? 2 * r->n : 1 << 20;
    r->b = realloc(r->b, want);
    ssize_t got = pread(r->fd, r->b + r->n, want - r->n, r->n);
    if (got <= 0) die("GGUF header truncated");
    r->n += got;
  }
  r->at += n;
  return r->b + r->at - n;
}
static uint64_t rd_u(Rd *r, int n) {
  uint64_t v = 0;
  memcpy(&v, take_bytes(r, n), n);
  return v;
}
static char *rd_str(Rd *r) {
  uint64_t n = rd_u(r, 8);
  char *s = xmalloc(n + 1);
  memcpy(s, take_bytes(r, n), n);
  s[n] = 0;
  return s;
}
static void rd_skip(Rd *r, uint32_t type) {
  static const uint8_t size[13] = {1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8};
  if (type == 8) take_bytes(r, rd_u(r, 8));
  else if (type == 9) {
    uint32_t t = rd_u(r, 4);
    uint64_t n = rd_u(r, 8);
    for (uint64_t i = 0; i < n; i++)
      if (t == 8 || t == 9) rd_skip(r, t);
      else take_bytes(r, size[t]);
  } else if (type < 13) take_bytes(r, size[type]);
  else die("GGUF: unknown value type %u", type);
}

static void gguf_file(const char *path) {
  Rd r = {open(path, O_RDONLY), 0, 0, 0};
  if (r.fd < 0) die("cannot open %s", path);
  if (memcmp(take_bytes(&r, 4), "GGUF", 4) || rd_u(&r, 4) != 3) die("%s: not a GGUF file of version 3", path);
  uint64_t nt = rd_u(&r, 8), nkv = rd_u(&r, 8), align = 32;
  for (uint64_t i = 0; i < nkv; i++) {
    char *key = rd_str(&r);
    uint32_t type = rd_u(&r, 4);
    if (!strcmp(key, "general.alignment") && type == 4) align = rd_u(&r, 4);
    else rd_skip(&r, type);
    free(key);
  }
  gg = realloc(gg, (ngg + nt) * sizeof *gg);
  GgEntry *first = gg + ngg;
  for (uint64_t i = 0; i < nt; i++) {
    GgEntry *e = &gg[ngg++];
    e->name = rd_str(&r);
    uint32_t nd = rd_u(&r, 4);
    memset(e->t.ne, 0, sizeof e->t.ne);
    for (uint32_t d = 0; d < nd; d++) {
      uint64_t v = rd_u(&r, 8);
      if (d < 4) e->t.ne[d] = v;
    }
    e->t.type = rd_u(&r, 4);
    e->t.off = rd_u(&r, 8);
    e->t.fd = r.fd;
  }
  uint64_t data = (r.at + align - 1) / align * align;
  for (GgEntry *e = first; e < gg + ngg; e++) e->t.off += data;
  free(r.b);
  fcntl(r.fd, F_NOCACHE, 1);
}

void gguf_open(const char *path) {
  const char *of = strstr(path, "-of-");
  int n = of ? atoi(of + 4) : 0;
  if (!of || of - path < 5 || n <= 0) {  // a single file
    gguf_file(path);
    return;
  }
  char p[4096];
  for (int i = 1; i <= n; i++) {  // a shard not downloaded is left out (the Engram tables' one)
    snprintf(p, sizeof p, "%.*s%05d%s", (int)(of - path - 5), path, i, of);
    if (!access(p, R_OK)) gguf_file(p);
  }
}

int gguf_get(const char *name, Gguf *t) {
  for (int i = 0; i < ngg; i++)
    if (!strcmp(gg[i].name, name)) return *t = gg[i].t, 1;
  return 0;
}
