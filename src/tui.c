// The server's live view (--tui): the terminal's alternate screen, redrawn
// twice a second from serve_view. It shows the requests and their decoding
// speed, the experts per decoded row (with what substitution adds), memory on
// the GPU, in RAM and on the SSD, the sessions' prefix tree and the latest log
// lines. Nothing is written to files: on exit the normal screen comes back
// with the last log lines. The q key stops the server as Ctrl-C does.
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <termios.h>
#include <unistd.h>

#include "myds4.h"

#define GiB (1024.0 * 1024 * 1024)
#define RING 256   // log lines kept
#define FRAMES 21  // statistics kept: 10 s at two frames a second
#define BOLD "\x1b[1m"
#define DIM "\x1b[2m"
#define OFF "\x1b[0m"
#define RED "\x1b[31m"
#define GREEN "\x1b[32m"
#define YELLOW "\x1b[33m"
#define BLUE "\x1b[34m"
#define MAGENTA "\x1b[35m"
#define CYAN "\x1b[36m"

static char ring[RING][240];
static int logged;
static pthread_mutex_t ring_lock = PTHREAD_MUTEX_INITIALIZER;
static volatile sig_atomic_t on;
static struct termios tty;
static int tty_kept, keyboard;  // the terminal's settings to restore; stdin still delivers keys

void tui_log(const char *line) {
  pthread_mutex_lock(&ring_lock);
  snprintf(ring[logged % RING], sizeof ring[0], "%s", line);
  logged++;
  pthread_mutex_unlock(&ring_lock);
}

// Back to the normal screen, then the last log lines: only calls that a
// signal handler may make.
static void leave(void) {
  if (!on) return;
  on = 0;
  static const char back[] = OFF "\x1b[?25h\x1b[?1049l";
  if (tty_kept) tcsetattr(0, TCSANOW, &tty);
  if (write(2, back, sizeof back - 1) < 0) return;
  for (int i = logged > 30 ? logged - 30 : 0; i < logged; i++)
    if (write(2, ring[i % RING], strlen(ring[i % RING])) < 0 || write(2, "\n", 1) < 0) return;
}

static void on_signal(int sig) {
  leave();
  signal(sig, SIG_DFL);
  raise(sig);
}

// ---- formatting ----

static char *fmt(const char *f, ...) __attribute__((format(printf, 1, 2)));
static char *fmt(const char *f, ...) {  // into one of 64 rotating buffers
  static char buf[64][192];
  static int k;
  char *b = buf[k++ % 64];
  va_list ap;
  va_start(ap, f);
  vsnprintf(b, sizeof buf[0], f, ap);
  va_end(ap);
  return b;
}

static char *count(double v) {  // an integer with thousands separators
  char d[32], *o = fmt("%s", "");
  int n = snprintf(d, sizeof d, "%.0f", v < 0 ? 0 : v), k = 0;
  for (int i = 0; i < n; i++) {
    o[k++] = d[i];
    if ((n - i - 1) % 3 == 0 && i < n - 1) o[k++] = ',';
  }
  o[k] = 0;
  return o;
}

static char *span(double s) {
  int t = (int)s;
  return t >= 3600 ? fmt("%dh%02dm", t / 3600, t / 60 % 60) : t >= 60 ? fmt("%dm%02ds", t / 60, t % 60) : fmt("%ds", t);
}

static char *pct(double part, double whole) { return fmt("%.1f%%", whole > 0 ? 100 * part / whole : 0); }

#define L(s) DIM s OFF  // a figure's label

// The columns s takes: escape sequences none, a UTF-8 character one.
static int width(const char *s) {
  int n = 0;
  for (const char *p = s; *p; p++) {
    if (p[0] == '\x1b' && p[1] == '[') {
      for (p += 2; *p && !(*p >= '@' && *p <= '~'); p++) {}
      if (!*p) break;
    } else {
      n += (*p & 0xC0) != 0x80;
    }
  }
  return n;
}

static char *pad(const char *s, int n) {  // s, then spaces to n columns
  int k = n - width(s);
  return fmt("%s%*s", s, k > 1 ? k : 1, "");
}

// A bar of w cells: each part's share in its color, the rest dim.
static char *bar(int w, int n, const double *share, const char *const *color) {
  static char b[8][2048];
  static int k;
  char *o = b[k++ % 8];
  size_t at = 0;
  double sum = 0;
  int done = 0;
  for (int i = 0; i < n; i++) {
    sum += share[i] > 0 ? share[i] : 0;
    int to = (int)(sum * w + 0.5);
    to = to > w ? w : to;
    at += snprintf(o + at, sizeof b[0] - at, "%s", color[i]);
    for (; done < to && at + 4 < sizeof b[0]; done++) at += snprintf(o + at, sizeof b[0] - at, "█");
  }
  at += snprintf(o + at, sizeof b[0] - at, OFF DIM);
  for (; done < w && at + 4 < sizeof b[0]; done++) at += snprintf(o + at, sizeof b[0] - at, "░");
  snprintf(o + at, sizeof b[0] - at, OFF);
  return o;
}

static char *spark(const double *v, int n) {  // the values' heights in eighths
  static const char *h[] = {"▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"};
  char *o = fmt("%s", "");
  double top = 0;
  for (int i = 0; i < n; i++) top = v[i] > top ? v[i] : top;
  size_t at = 0;
  for (int i = 0; i < n && at + 4 < 128; i++) at += snprintf(o + at, 128 - at, "%s", h[top > 0 ? (int)(7.99 * v[i] / top) : 0]);
  return o;
}

typedef struct {
  char *s[512];
  int n;
} Lines;

static void add(Lines *l, const char *f, ...) __attribute__((format(printf, 2, 3)));
static void add(Lines *l, const char *f, ...) {
  if (l->n == 512) return;
  va_list ap;
  va_start(ap, f);
  if (vasprintf(&l->s[l->n], f, ap) >= 0) l->n++;
  va_end(ap);
}

static int utf8_len(const char *p) {  // the bytes of the character at p
  int len = (*p & 0x80) == 0 ? 1 : (*p & 0xE0) == 0xC0 ? 2 : (*p & 0xF0) == 0xE0 ? 3 : 4;
  for (int k = 1; k < len; k++)
    if (!p[k]) return k;
  return len;
}

static char *clip(const char *s, int n) {  // s (no escape sequences) in at most n columns, "…" at the cut
  static char b[256];
  if (width(s) <= n) return fmt("%s", s);
  size_t at = 0;
  for (int col = 0; *s && col < n - 1 && at + utf8_len(s) + 4 < sizeof b; col++) {
    int len = utf8_len(s);
    memcpy(b + at, s, (size_t)len), at += (size_t)len, s += len;
  }
  snprintf(b + at, sizeof b - at, "…");
  return b;
}

// A line of at most w columns: escape sequences take none, a UTF-8
// character one. A longer line ends after its last whole figure (before
// two spaces or " · "), else its last whole word, with "…".
static void put_line(Str *o, int w, const char *s) {
  int col = 0, fits = width(s) <= w, word_col = 0, figure_col = 0;
  size_t start = o->n, word = start, figure = start;  // where the last word, the last figure before the cut ends
  for (const char *p = s; *p;) {
    if (*p == '\x1b') {
      const char *q = p + 1;
      if (*q == '[') {
        q++;
        while (*q && !(*q >= '@' && *q <= '~')) q++;
        if (*q) q++;
      }
      str_put(o, p, (size_t)(q - p));
      p = q;
      continue;
    }
    int len = utf8_len(p);
    if (!fits && col == w - 2) break;
    if (!fits && *p == ' ' && col > 0 && p[-1] != ' ') {
      word = o->n, word_col = col;
      if (p[1] == ' ' || !strncmp(p + 1, "·", strlen("·"))) figure = o->n, figure_col = col;
    }
    str_put(o, p, (size_t)len), col++;
    p += len;
  }
  if (!fits) {
    if (figure > start) o->n = figure, col = figure_col;
    else if (word > start) o->n = word, col = word_col;
    str_cat(o, OFF DIM " …");
    col += 2;
  }
  // The rest of the row is erased; a full row leaves the cursor on its last
  // column, which the erase would clear.
  str_cat(o, col < w ? OFF "\x1b[K" : OFF);
}

static void head(Lines *l, int w, const char *title) {  // a section's rule
  char o[1024];
  size_t at = (size_t)snprintf(o, sizeof o, BOLD CYAN "─ %s " OFF CYAN, title);
  for (int c = width(title) + 3; c < w && at + 4 < sizeof o; c++) at += (size_t)snprintf(o + at, sizeof o - at, "─");
  add(l, "%s", o);
}

// ---- the frames ----

typedef struct { double t; CacheStats c; uint64_t rows; double gpu, wall; } Snap;
static Snap snaps[FRAMES];
static int nsnap;
typedef struct { int number, k; double t[10]; int gen[10]; } Track;  // a request's (time, tokens) samples
static Track track[8];
static double speeds[60];  // the requests' tokens/s, the last 30 s
static uint64_t swap0;  // the Mac's swap-outs when the view started
static int swap_set;

static double speed(const ViewReq *r, double now) {
  Track *t = 0;
  for (int i = 0; i < 8 && !t; i++)
    if (track[i].number == r->number) t = &track[i];
  for (int i = 0; i < 8 && !t; i++)
    if (!track[i].number) t = &track[i];
  if (!t) t = &track[r->number % 8];
  if (t->number != r->number) *t = (Track){.number = r->number};
  t->t[t->k % 10] = now, t->gen[t->k % 10] = r->generated, t->k++;
  int old = t->k > 10 ? t->k % 10 : 0, last = (t->k - 1) % 10;
  return t->t[last] > t->t[old] ? (t->gen[last] - t->gen[old]) / (t->t[last] - t->t[old]) : 0;
}

static void draw(const View *v) {
  struct winsize ws = {0};
  int ok = !ioctl(2, TIOCGWINSZ, &ws);
  int w = ok && ws.ws_col >= 40 ? ws.ws_col : 100, h = ok && ws.ws_row >= 12 ? ws.ws_row : 40;
  int gw = w >= 110 ? 24 : 10;  // a gauge's cells
  Lines l = {0};
  char *up = fmt("q quits · up %s ", span(v->uptime)), *title = fmt(" %s", clip(v->title, w - width(up) - 2));
  int gap = w - width(title) - width(up);
  add(&l, "\x1b[7m%s%*s%s" OFF, title, gap > 1 ? gap : 1, "", up);

  // now: the phase, the requests and their speed
  double sum = 0, sp[8];
  for (int i = 0; i < v->nreq; i++) sum += sp[i] = v->req[i].generated ? speed(&v->req[i], v->now) : 0;
  memmove(speeds, speeds + 1, sizeof speeds - sizeof *speeds);
  speeds[59] = !strcmp(v->phase, "decode") ? sum : 0;
  head(&l, w, "Now");
  char *queue = v->queue ? fmt("   " YELLOW "%d waiting" OFF, v->queue) : "";
  int ns = w > 100 ? 40 : 20;
  if (!strcmp(v->phase, "decode")) {
    add(&l, " %s%d request%s   " BOLD "%.1f" OFF " tok/s   " GREEN "%s" OFF "%s", pad(BOLD GREEN "decoding" OFF, 12), v->nreq,
        v->nreq == 1 ? "" : "s", sum, spark(speeds + 60 - ns, ns), queue);
  } else if (!strcmp(v->phase, "prefill") && v->prefill_n) {
    double share[1] = {(double)v->prefill_layer / LAYERS};
    const char *const color[1] = {CYAN};
    add(&l, " %s%s  " L("layer") " " BOLD "%d" OFF "/%d   " L("tokens") " %s%s", pad(BOLD CYAN "prefilling" OFF, 12),
        bar(gw, 1, share, color), v->prefill_layer, LAYERS, count(v->prefill_n), queue);
  } else if (!strcmp(v->phase, "prefill")) {
    add(&l, " %s" L("starting; images are encoded first") "%s", pad(BOLD CYAN "prefilling" OFF, 12), queue);
  } else {
    add(&l, " %s%s%s", pad(BOLD "idle" OFF, 12), span(v->phase_s), queue);
  }
  for (int i = 0; i < v->nreq && i < 8; i++) {
    const ViewReq *r = &v->req[i];
    add(&l, "   %s" L("session") " %-5d" L("prompt") " %-8s" L("reused") " %-8s" L("generated") " %-8s%s",
        pad(fmt(BOLD "#%d" OFF, r->number), 7), r->session, count(r->prompt), count(r->reused), count(r->generated),
        r->generated ? fmt(GREEN "%.1f tok/s" OFF, sp[i]) : "");
  }

  // experts per decoded token, the last 10 s (or since start)
  Snap now = {v->now, v->decode, v->decode_rows, v->decode_gpu_s, v->decode_s}, *old = 0;
  memmove(snaps, snaps + 1, (FRAMES - 1) * sizeof *snaps);
  snaps[FRAMES - 1] = now;
  nsnap += nsnap < FRAMES;
  for (int i = FRAMES - nsnap; i < FRAMES && !old; i++)
    if (snaps[i].rows < now.rows && now.t - snaps[i].t <= 11) old = &snaps[i];
  Snap zero = {0}, *base = old ? old : &zero;
  double rows = (double)(now.rows - base->rows);
  const CacheStats *c = &now.c, *b = &base->c;
  double part[5] = {(double)(c->row_own - b->row_own), (double)(c->row_ahead - b->row_ahead), (double)(c->row_late - b->row_late),
                    (double)(c->row_sub - b->row_sub), (double)(c->row_ssd - b->row_ssd)};
  double counted = part[0] + part[1] + part[2] + part[3] + part[4], bytes = (double)(c->bytes - b->bytes);
  head(&l, w, rows > 0 ? fmt("Experts per token · %s · %s tokens", old ? "last 10 s" : "since start", count(rows))
                       : "Experts per token");
  if (rows > 0 && counted > 0) {
    double share[5], each = counted / rows;
    for (int i = 0; i < 5; i++) share[i] = part[i] / counted;
    const char *const color[5] = {GREEN, CYAN, YELLOW, MAGENTA, RED};
    static const char *const name[5] = {"in memory", "read ahead", "late", "substituted", "from SSD"};
    add(&l, " %s   " L("hit rate") " " BOLD "%s" OFF "%s%s", bar(w >= 110 ? 40 : 20, 5, share, color),
        pct(part[0] + part[1] + part[2], counted),
        part[3] > 0 ? fmt("   " L("with substitutes") " " BOLD "%s" OFF, pct(counted - part[4], counted)) : "",
        each < LAYERS * TOPK - 0.05 ? fmt("   " L("counted") " %.1f/%d", each, LAYERS * TOPK) : "");
    Str t[3] = {0};  // a table: the sources' names, experts per token, shares
    str_printf(&t[0], " %10s", "");
    str_printf(&t[1], " " L("%-10s"), "experts");
    str_printf(&t[2], " " L("%-10s"), "share");
    for (int i = 0; i < 5; i++) {
      int cw = (int)strlen(name[i]) + 4;
      str_printf(&t[0], "  %s■" OFF " " L("%s"), color[i], name[i]);
      str_printf(&t[1], "%*.1f", cw, part[i] / rows);
      str_printf(&t[2], "%*s", cw, pct(part[i], counted));
    }
    for (int i = 0; i < 3; i++) add(&l, "%s", t[i].p), free(t[i].p);
    double wall = now.wall - base->wall, gpu = now.gpu - base->gpu;
    double remaps = (c->remaps - b->remaps) / rows;  // slots mapped for the GPU again: its memory was short
    add(&l, " " L("%-10s") " " L("SSD") " %.2f GB · %.1f GB/s   " L("wait") " %.1f ms   " L("GPU") " %.1f ms · %s busy   "
        L("read ahead") " %.1f · %.1f cancelled%s", "per token", bytes / rows / 1e9, wall > 0 ? bytes / wall / 1e9 : 0,
        (c->read_s - b->read_s) / rows * 1e3, gpu / rows * 1e3, pct(gpu, wall), (c->prefetches - b->prefetches) / rows,
        (c->cancelled - b->cancelled) / rows, remaps >= 0.05 ? fmt("   " YELLOW "remapped %.1f" OFF, remaps) : "");
  } else if (rows > 0) {  // only --gpu-route layers, which the GPU looks up without counting
    add(&l, " " L("%s tokens; the GPU looked up all their experts (--gpu-route)"), count(rows));
  } else {
    add(&l, " " L("no decoded tokens yet"));
  }
  if (v->prefill_rows) {  // a prompt batch reads each expert once for all its rows
    const CacheStats *p = &v->prefill;
    double used = (double)(p->hits + p->prefetch_hits + p->misses), sub = (double)p->substituted / v->prefill_rows;
    add(&l, " " L("%-10s") " %s tokens%s   " L("in memory") " %s   " L("read") " %.1f GB", "prompts", count((double)v->prefill_rows),
        p->substituted ? fmt("   " L("substituted") " %.1f/%d · %s", sub, LAYERS * TOPK, pct(sub, LAYERS * TOPK)) : "",
        pct((double)(p->hits + p->prefetch_hits), used), p->bytes / 1e9);
  }

  // memory: a row each for the GPU, RAM, the SSD and the Mac: a gauge, the main figure, the parts
  head(&l, w, "Memory");
  const MemInfo *m = &v->mem;
  double gb = v->slot_bytes / GiB, mapped = v->mapped * gb, weights = v->dense_bytes / GiB, inuse = v->gpu_inuse / GiB;
  double mine = mapped + weights, lim = v->gpu_limit < INT64_MAX ? v->gpu_limit / GiB : 0, ram = m->total / GiB;
  int fw = 18;  // the main figure's columns
  {  // the driver's resident memory (all apps): myds4's mapped experts and weights while the GPU works; it releases them while it idles
    int all = inuse >= mine;
    const char *const color[3] = {BLUE, CYAN, DIM}, *const plain[1] = {""};
    double share[3] = {all ? mapped : inuse, all ? weights : 0, all ? inuse - mine : 0};
    for (int i = 0; i < 3; i++) share[i] = lim > 0 ? share[i] / lim : 0;
    char *fig = pad(lim > 0 ? fmt(BOLD "%.1f" OFF " / %.1f GiB", inuse, lim) : fmt(BOLD "%.1f" OFF " GiB", inuse), fw);
    if (all)
      add(&l, " " BOLD "GPU" OFF "  %s  %s" BLUE "■" OFF " " L("experts") " %.1f " L("(%s slots)") "   " CYAN "■" OFF " " L("weights")
          " %.1f   " L("other") " %.1f", bar(gw, 3, share, color), fig, mapped, count(v->mapped), weights, inuse - mine);
    else
      add(&l, " " BOLD "GPU" OFF "  %s  %s" L("idle; resident when busy:") " " L("experts") " %.1f " L("(%s slots)") "   " L("weights")
          " %.1f", bar(gw, 1, share, plain), fig, mapped, count(v->mapped), weights);
  }
  {  // myds4's footprint against the RAM: its experts and the rest
    double foot = v->footprint / GiB, ex = v->slots * gb, rest = foot > ex ? foot - ex : 0;
    const char *const color[2] = {BLUE, CYAN};
    double share[2] = {ram > 0 ? ex / ram : 0, ram > 0 ? rest / ram : 0};
    add(&l, " " BOLD "RAM" OFF "  %s  %s" BLUE "■" OFF " " L("experts") " %.1f   " CYAN "■" OFF " " L("rest") " %.1f   " L("slots")
        " %s / %s " L("(max %s)"), bar(gw, 2, share, color), pad(fmt(L("myds4") " " BOLD "%.1f" OFF " GiB", foot), fw), ex, rest,
        count(v->slots), count(v->cap), count(v->max_slots));
  }
  {  // the experts that are only on the SSD
    const char *const plain[1] = {""};
    double share[1] = {v->experts ? (double)v->uncached / v->experts : 0};
    add(&l, " " BOLD "SSD" OFF "  %s  %s" L("experts") " %s / %s   " L("evicted") " %s   " L("given back") " %s",
        bar(gw, 1, share, plain), pad(fmt(BOLD "%.1f" OFF " GB", v->uncached_bytes / 1e9), fw), count(v->uncached),
        count(v->experts), count((double)v->cache.evictions), count((double)v->cache.given_back));
  }
  {  // the Mac's memory in use: without the file cache, then the file cache
    if (!swap_set) swap0 = m->swapouts, swap_set = 1;
    double spare = m->free / GiB, file = m->file / GiB, used = ram - spare, held = used - file > 0 ? used - file : 0;
    const char *const color[2] = {"", DIM};
    double share[2] = {ram > 0 ? held / ram : 0, ram > 0 ? file / ram : 0};
    const char *level = m->pressure >= 4 ? RED "critical" OFF : m->pressure >= 2 ? YELLOW "warning" OFF : GREEN "normal" OFF;
    add(&l, " " BOLD "Mac" OFF "  %s  %s" L("free") " " BOLD "%.1f" OFF "   " L("pressure") " %s   " L("file cache") " %.1f   "
        L("wired") " %.1f   " L("swap") " %.1f", bar(gw, 2, share, color), pad(fmt(BOLD "%.1f" OFF " / %.0f GiB", used, ram), fw),
        spare, level, file, m->wired / GiB, m->swap_used / GiB);
    add(&l, "%*s" L("compressor %.1f GiB holding %.1f · file cache minimum %.1f · %s swap-outs since start"), 6, "",
        m->compressor / GiB, m->compressed / GiB, m->file_kept / GiB, count((double)(m->swapouts - swap0)));
  }

  // the sessions' prefix tree
  head(&l, w, "KV cache");
  add(&l, " " L("sessions") " %d   %.2f / %.1f GiB   " L("reused") " %s   " L("new") " %d   " L("continued")
      " %d   " L("saved") " %d   " L("forked") " %d", v->sessions, v->kv_gib, v->kv_limit_gib,
      pct((double)v->reused_tokens, (double)v->prompt_tokens), v->kinds[0], v->kinds[1], v->kinds[2], v->kinds[3]);
  int fixed = l.n, room = h - fixed - 3, nodes = v->nnodes < room - room / 3 ? v->nnodes : room - room / 3;
  for (int i = 0; i < nodes; i++) {
    const ViewNode *x = &v->node[i];
    char *who = x->session ? fmt("   " L("session") " %d   %s%s   %.2f GiB", x->session,
                                 x->busy ? GREEN "decoding" OFF : fmt(L("idle") " %s", span(x->idle)),
                                 x->saved >= 0 ? fmt("   " L("saved at") " %s", count(x->saved)) : "", x->gib)
                           : "";
    add(&l, " " DIM "%s" OFF "%s %s–%s " L("(%s tok)") "%s%s%s", x->tree, x->session ? "●" : "○", count(x->start),
        count(x->end), count(x->end - x->start), x->sessions > 1 ? fmt("   %d " L("sessions"), x->sessions) : "",
        x->hits ? fmt("   " BOLD "%d" OFF " " L("%s"), x->hits, x->hits == 1 ? "hit" : "hits") : "", who);
  }
  if (nodes < v->nnodes) add(&l, DIM "   … %d more" OFF, v->nnodes - nodes);

  // the log's last lines
  head(&l, w, "Log");
  pthread_mutex_lock(&ring_lock);
  int show = h - l.n > 0 ? h - l.n : 0, first = logged > show ? logged - show : 0;
  for (int i = first; i < logged; i++) add(&l, DIM " %s" OFF, ring[i % RING]);
  pthread_mutex_unlock(&ring_lock);

  Str o = {0};
  str_cat(&o, "\x1b[H");
  for (int i = 0; i < l.n && i < h; i++) {
    if (i) str_cat(&o, "\r\n");  // between rows only: one after the last row scrolls the terminal
    put_line(&o, w, l.s[i]);
  }
  str_cat(&o, "\x1b[J");
  if (on && write(2, o.p, o.n) < 0) {}
  for (int i = 0; i < l.n; i++) free(l.s[i]);
  free(o.p);
}

// The wait for the next frame, reading the keys typed meanwhile: q, or a
// Ctrl-C that reaches stdin as a character, requests the stop that Ctrl-C's
// signal requests, for terminals that keep Ctrl-C for themselves.
static void keys(int ms) {
  struct pollfd in = {.fd = 0, .events = POLLIN};
  if (!keyboard) {
    usleep(ms * 1000);
    return;
  }
  if (poll(&in, 1, ms) <= 0) return;
  char c[16];
  ssize_t n = read(0, c, sizeof c);
  if (n <= 0) keyboard = 0;  // stdin closed: wait by sleeping from now on
  for (ssize_t i = 0; i < n; i++)
    if (c[i] == 'q' || c[i] == 'Q' || c[i] == 3) {
      if (!shutdown_requested()) tui_log("q: stopping after the current step");
      shutdown_request();
    }
}

static void *loop(void *arg) {
  View *v = calloc(1, sizeof *v);
  if (!v) return 0;
  while (on) {
    serve_view(v);
    draw(v);
    keys(500);
  }
  free(v);
  return arg;
}

int tui_start(void) {
  if (!isatty(2)) return 0;
  if (isatty(0) && !tcgetattr(0, &tty)) {  // no echo of typed keys over the view
    struct termios quiet = tty;
    quiet.c_lflag &= ~(tcflag_t)(ECHO | ICANON);
    tty_kept = keyboard = !tcsetattr(0, TCSANOW, &quiet);
  }
  static const char enter[] = "\x1b[?1049h\x1b[?25l\x1b[2J";
  if (write(2, enter, sizeof enter - 1) < 0) return 0;
  on = 1;
  atexit(leave);
  die_hook = leave;
  int sig[] = {SIGINT, SIGTERM, SIGHUP, SIGQUIT};
  for (int i = 0; i < 4; i++) signal(sig[i], on_signal);
  pthread_t t;
  if (pthread_create(&t, 0, loop, 0)) {
    leave();
    return 0;
  }
  pthread_detach(t);
  return 1;
}
