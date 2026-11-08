// A checkpoint's tokenizer (tokenizer.json), one per model: byte-level BPE
// after the file's normalizer (none or NFC) and its sequence of Split
// regexes, each isolating its matches within the pieces of the one before;
// in a rendered prompt the added tokens' spellings are those tokens. The
// regexes and NFC run in the system's ICU library (libicucore), as the
// reference tokenizer's do in their own engines; tok_open refuses a pipeline
// with other steps.
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

#include "myds4.h"

// ---- ICU (libicucore), declared as the library exports it ----

typedef uint16_t UChar;
typedef struct URegularExpression URegularExpression;
typedef struct UNormalizer2 UNormalizer2;
extern URegularExpression *uregex_open(const UChar *pattern, int32_t length, uint32_t flags, void *parse_error,
                                       int32_t *status);
extern void uregex_close(URegularExpression *regex);
extern void uregex_setText(URegularExpression *regex, const UChar *text, int32_t length, int32_t *status);
extern int8_t uregex_findNext(URegularExpression *regex, int32_t *status);
extern int32_t uregex_start(URegularExpression *regex, int32_t group, int32_t *status);
extern int32_t uregex_end(URegularExpression *regex, int32_t group, int32_t *status);
extern const UNormalizer2 *unorm2_getNFCInstance(int32_t *status);
extern int32_t unorm2_normalize(const UNormalizer2 *n, const UChar *src, int32_t length, UChar *dest, int32_t capacity,
                                int32_t *status);

static void icu_check(int32_t status, const char *what) {
  if (status > 0) die("tokenizer: ICU %s failed (error %d)", what, status);  // negative values are warnings
}

// ---- the tokenizer ----

typedef struct { const char *s; int n, id; } Key;
typedef struct { uint64_t pair; int rank, id; } Merge;
typedef struct { int rank, pos; } Pair;

enum { SPLITS = 8 };

struct Tokenizer {
  Json *document;
  Key *vocab;
  uint64_t vcap;
  Merge *merges;
  uint64_t mcap;
  char **pieces;  // each id's raw bytes
  int *piece_len, nvocab;
  Key *added;     // the added tokens' spellings
  unsigned char *special;  // each added token's tokenizer.json "special"
  int nadded, image;  // image: the image token's id (IMAGE_MARK's), or -1
  URegularExpression *split[SPLITS];
  int nsplit, nfc;
  pthread_mutex_t lock;  // encoding's scratch below
  int *sym, *prv, *nxt, cap_bpe, nheap;
  Pair *heap;
};

static uint64_t fnv(const char *s, int n) {
  uint64_t h = 1469598103934665603ull;
  for (int i = 0; i < n; i++) h = (h ^ (uint8_t)s[i]) * 1099511628211ull;
  return h;
}

static int lookup(const Tokenizer *t, const char *s, int n) {
  for (uint64_t i = fnv(s, n) & (t->vcap - 1);; i = (i + 1) & (t->vcap - 1)) {
    if (!t->vocab[i].s) return -1;
    if (t->vocab[i].n == n && !memcmp(t->vocab[i].s, s, n)) return t->vocab[i].id;
  }
}

static void insert(Tokenizer *t, const char *s, int n, int id) {
  uint64_t i = fnv(s, n) & (t->vcap - 1);
  while (t->vocab[i].s) i = (i + 1) & (t->vcap - 1);
  t->vocab[i] = (Key){s, n, id};
}

static uint64_t mix(uint64_t x) {
  x ^= x >> 33;
  x *= 0xff51afd7ed558ccdull;
  return x ^ x >> 33;
}

static int merge_of(const Tokenizer *t, int a, int b, int *id) {
  uint64_t key = (uint64_t)a << 32 | (uint32_t)b;
  for (uint64_t i = mix(key) & (t->mcap - 1);; i = (i + 1) & (t->mcap - 1)) {
    if (!t->merges[i].pair) return -1;
    if (t->merges[i].pair == key + 1) return *id = t->merges[i].id, t->merges[i].rank;
  }
}

static int utf8_next(const unsigned char *s, int *len) {
  if (s[0] < 0x80) return *len = 1, s[0];
  if (s[0] < 0xE0) return *len = 2, (s[0] & 31) << 6 | (s[1] & 63);
  if (s[0] < 0xF0) return *len = 3, (s[0] & 15) << 12 | (s[1] & 63) << 6 | (s[2] & 63);
  return *len = 4, (s[0] & 7) << 18 | (s[1] & 63) << 12 | (s[2] & 63) << 6 | (s[3] & 63);
}

// GPT-2's byte-level alphabet, read backwards: code point to byte.
static int unbyte[324];

// A byte-level token string to its raw bytes; returns the length.
static int raw(const char *s, int n, char *out) {
  int m = 0;
  for (int i = 0, k; i < n; i += k) {
    int cp = utf8_next((const unsigned char *)s + i, &k);
    if ((unsigned)cp >= sizeof unbyte / sizeof *unbyte) die("tokenizer: invalid byte-alphabet token");
    out[m++] = unbyte[cp];
  }
  return m;
}

static UChar *utf16(const char *s, int n, int *len);

static int is_str(const Json *j, const char *s) { return j && j->type == J_STR && !strcmp(j->s, s); }
static int is_false(const Json *j) { return !j || j->type == J_NULL || (j->type == J_BOOL && !j->num); }
static int is_empty(const Json *j) { return !j || j->type == J_NULL || (j->type == J_STR && !j->n); }

// The pipeline's steps this tokenizer runs: the normalizer, the Split
// regexes and ByteLevel, and plain BPE settings.
static void pipeline(Tokenizer *t, const Json *root, const char *path) {
  const Json *norm = json_get(root, "normalizer"), *list;
  if (norm && norm->type != J_NULL) {
    if (is_str(json_get(norm, "type"), "NFC")) t->nfc = 1;
    else if (!is_str(json_get(norm, "type"), "Sequence") || !(list = json_get(norm, "normalizers")) ||
             list->type != J_ARR || list->n)
      die("%s: a normalizer other than NFC", path);
  }
  const Json *pre = json_get(root, "pre_tokenizer"), *steps = json_get(pre, "pretokenizers");
  if (!is_str(json_get(pre, "type"), "Sequence") || !steps || steps->type != J_ARR || steps->n < 1)
    die("%s: a pre-tokenizer other than Split regexes and ByteLevel", path);
  for (int i = 0; i < steps->n; i++) {
    const Json *s = &steps->kid[i];
    if (i == steps->n - 1) {
      if (!is_str(json_get(s, "type"), "ByteLevel") || !is_false(json_get(s, "add_prefix_space")) ||
          !is_false(json_get(s, "use_regex")))
        die("%s: the pre-tokenizer does not end with ByteLevel (no prefix space, no regex)", path);
      continue;
    }
    const Json *re = json_get(json_get(s, "pattern"), "Regex");
    if (!is_str(json_get(s, "type"), "Split") || !is_str(json_get(s, "behavior"), "Isolated") ||
        !is_false(json_get(s, "invert")) || !re || re->type != J_STR || t->nsplit == SPLITS)
      die("%s: pre-tokenizer step %d is not an isolating Split regex", path, i + 1);
    int len, status = 0;
    UChar *u = utf16(re->s, re->n, &len);
    t->split[t->nsplit++] = uregex_open(u, len, 0, 0, &status);
    icu_check(status, "regex compilation");
    free(u);
  }
  const Json *model = json_get(root, "model");
  if (!is_str(json_get(model, "type"), "BPE") || !is_false(json_get(model, "byte_fallback")) ||
      !is_empty(json_get(model, "continuing_subword_prefix")) || !is_empty(json_get(model, "end_of_word_suffix")) ||
      !is_false(json_get(model, "fuse_unk")) || !is_false(json_get(model, "ignore_merges")) ||
      !is_empty(json_get(model, "unk_token")))
    die("%s: BPE settings this tokenizer does not run", path);
}

Tokenizer *tok_open(const char *dir, const char *image) {
  static pthread_once_t once = PTHREAD_ONCE_INIT;
  void alphabet(void);
  pthread_once(&once, alphabet);
  char path[4096];
  snprintf(path, sizeof path, "%s/tokenizer.json", dir);
  char *text = read_file(path, 0);
  Json *root = json_parse(text), *model = json_get(root, "model");
  free(text);
  Json *v = json_get(model, "vocab"), *m = json_get(model, "merges"), *add = json_get(root, "added_tokens");
  if (!v || v->type != J_OBJ || !m || m->type != J_ARR || !add || add->type != J_ARR)
    die("%s: no vocabulary, merges or added tokens", path);
  Tokenizer *t = xmalloc(sizeof *t);
  t->document = root;
  pthread_mutex_init(&t->lock, 0);
  pipeline(t, root, path);
  for (int i = 0; i < v->n; i++) t->nvocab = (int)v->kid[i].num >= t->nvocab ? (int)v->kid[i].num + 1 : t->nvocab;
  t->added = xmalloc((size_t)add->n * sizeof *t->added);
  t->special = xmalloc((size_t)add->n);
  t->image = -1;
  for (int i = 0; i < add->n; i++) {
    Json *a = &add->kid[i], *c = json_get(a, "content");
    int id = (int)json_get(a, "id")->num;
    if (!is_false(json_get(a, "lstrip")) || !is_false(json_get(a, "rstrip")) ||
        !is_false(json_get(a, "single_word")) || (t->nfc && !is_false(json_get(a, "normalized"))))
      die("%s: added token %s strips, matches words or is normalized", path, c->s);
    const Json *sp = json_get(a, "special");
    t->special[t->nadded] = sp && sp->type == J_BOOL && sp->num;
    t->added[t->nadded++] = (Key){c->s, c->n, id};
    t->nvocab = id >= t->nvocab ? id + 1 : t->nvocab;
    if (image && !strcmp(c->s, image)) t->image = id;
  }
  if (image && t->image < 0) die("%s: no image token %s", path, image);
  t->pieces = xmalloc((size_t)t->nvocab * sizeof *t->pieces);
  t->piece_len = xmalloc((size_t)t->nvocab * sizeof *t->piece_len);
  // Added tokens are literal UTF-8. Some also occur in model.vocab, where
  // converting them through the byte-level alphabet would read out of bounds.
  for (int i = 0; i < t->nadded; i++) {
    int id = t->added[i].id;
    if (id >= 0) t->pieces[id] = (char *)t->added[i].s, t->piece_len[id] = t->added[i].n;
  }
  for (t->vcap = 1; t->vcap < 2ull * v->n; t->vcap *= 2) {}
  t->vocab = xmalloc(t->vcap * sizeof *t->vocab);
  for (int i = 0; i < v->n; i++) {
    int id = (int)v->kid[i].num, len = strlen(v->kid[i].key);
    if (id < 0) die("tokenizer: vocabulary id out of range");
    if (t->pieces[id]) continue;
    char *r = xmalloc(len + 1);
    int n = raw(v->kid[i].key, len, r);
    insert(t, r, n, id);
    t->pieces[id] = r, t->piece_len[id] = n;
  }
  for (t->mcap = 1; t->mcap < 2ull * m->n; t->mcap *= 2) {}
  t->merges = xmalloc(t->mcap * sizeof *t->merges);
  for (int i = 0; i < m->n; i++) {
    const Json *e = &m->kid[i];
    const char *l, *r;
    int nl, nr;
    if (e->type == J_STR) {
      const char *sp = strchr(e->s, ' ');
      if (!sp) die("tokenizer: a merge without a space");
      l = e->s, nl = (int)(sp - e->s), r = sp + 1, nr = (int)strlen(sp + 1);
    } else if (e->type == J_ARR && e->n == 2 && e->kid[0].type == J_STR && e->kid[1].type == J_STR) {
      l = e->kid[0].s, nl = e->kid[0].n, r = e->kid[1].s, nr = e->kid[1].n;
    } else die("tokenizer: a merge of an unknown form");
    char a[256], b[256], ab[512];
    if (nl > 255 || nr > 255) die("tokenizer: a merge part longer than 255 bytes");
    int na = raw(l, nl, a), nb = raw(r, nr, b);
    memcpy(ab, a, na);
    memcpy(ab + na, b, nb);
    int ia = lookup(t, a, na), ib = lookup(t, b, nb), id = lookup(t, ab, na + nb);
    if (ia < 0 || ib < 0 || id < 0) die("tokenizer: bad merge %.*s %.*s", nl, l, nr, r);
    uint64_t key = (uint64_t)ia << 32 | (uint32_t)ib, j = mix(key) & (t->mcap - 1);
    while (t->merges[j].pair) j = (j + 1) & (t->mcap - 1);
    t->merges[j] = (Merge){key + 1, i, id};
  }
  for (int b = 0; b < 256; b++)
    if (lookup(t, (char[]){(char)b}, 1) < 0) die("%s: no token for byte %d", path, b);
  return t;
}

void alphabet(void) {
  for (int b = 0, n = 0; b < 256; b++) {
    int keep = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || b >= 174;
    unbyte[keep ? b : 256 + n++] = b;
  }
}

int tok_vocab(const Tokenizer *t) { return t->nvocab; }

void tok_close(Tokenizer *t) {
  if (!t) return;
  for (int i = 0; i < t->nsplit; i++) uregex_close(t->split[i]);
  for (uint64_t i = 0; i < t->vcap; i++) free((char *)t->vocab[i].s);
  free(t->vocab); free(t->merges); free(t->pieces); free(t->piece_len);
  free(t->added); free(t->special); free(t->sym); free(t->prv); free(t->nxt); free(t->heap);
  json_free(t->document);
  pthread_mutex_destroy(&t->lock);
  free(t);
}

// ---- encoding ----

// UTF-8 to UTF-16 (len units); off, when not null, gets each unit's byte offset.
static UChar *utf16(const char *s, int n, int *len) {
  UChar *u = xmalloc(((size_t)n + 1) * sizeof *u);
  int m = 0;
  for (int i = 0, k; i < n; i += k) {
    int c = utf8_next((const unsigned char *)s + i, &k);
    if (c >= 0x10000) c -= 0x10000, u[m++] = (UChar)(0xD800 + (c >> 10)), u[m++] = (UChar)(0xDC00 + (c & 0x3FF));
    else u[m++] = (UChar)c;
  }
  *len = m;
  return u;
}

// UTF-16 to UTF-8; returns the bytes written into out (4 per unit at most).
static int utf8(const UChar *u, int n, char *out) {
  int m = 0;
  for (int i = 0; i < n; i++) {
    uint32_t c = u[i];
    if (c >= 0xD800 && c < 0xDC00 && i + 1 < n && u[i + 1] >= 0xDC00 && u[i + 1] < 0xE000)
      c = 0x10000 + ((c - 0xD800) << 10) + (u[++i] - 0xDC00);
    if (c < 0x80) out[m++] = (char)c;
    else if (c < 0x800) out[m++] = (char)(0xC0 | c >> 6), out[m++] = (char)(0x80 | (c & 63));
    else if (c < 0x10000)
      out[m++] = (char)(0xE0 | c >> 12), out[m++] = (char)(0x80 | (c >> 6 & 63)), out[m++] = (char)(0x80 | (c & 63));
    else
      out[m++] = (char)(0xF0 | c >> 18), out[m++] = (char)(0x80 | (c >> 12 & 63)),
      out[m++] = (char)(0x80 | (c >> 6 & 63)), out[m++] = (char)(0x80 | (c & 63));
  }
  return m;
}

static int before(Pair a, Pair b) { return a.rank < b.rank || (a.rank == b.rank && a.pos < b.pos); }

static void push(Tokenizer *t, int rank, int pos) {
  Pair p = {rank, pos};
  int i = t->nheap++;
  for (; i > 0 && before(p, t->heap[(i - 1) / 2]); i = (i - 1) / 2) t->heap[i] = t->heap[(i - 1) / 2];
  t->heap[i] = p;
}

static Pair pop(Tokenizer *t) {
  Pair top = t->heap[0], last = t->heap[--t->nheap];
  int i = 0;
  for (int c; (c = 2 * i + 1) < t->nheap; i = c) {
    if (c + 1 < t->nheap && before(t->heap[c + 1], t->heap[c])) c++;
    if (!before(t->heap[c], last)) break;
    t->heap[i] = t->heap[c];
  }
  t->heap[i] = last;
  return top;
}

// The pair the symbol at i starts, onto the heap if a merge joins it.
static void offer(Tokenizer *t, int i) {
  int id, j = t->nxt[i], rank = j < 0 ? -1 : merge_of(t, t->sym[i], t->sym[j], &id);
  if (rank >= 0) push(t, rank, i);
}

// Byte-level BPE over a whole piece of n bytes, as the reference merges: the
// lowest-ranked adjacent pair first, the leftmost among equals, until no
// merge applies. A heap keeps a long piece at O(n log n).
static int bpe(Tokenizer *t, const char *s, int n, int *out, int max) {
  if (n <= 0) return 0;
  if (n > t->cap_bpe) {
    t->cap_bpe = n;
    t->sym = realloc(t->sym, n * sizeof *t->sym), t->prv = realloc(t->prv, n * sizeof *t->prv);
    t->nxt = realloc(t->nxt, n * sizeof *t->nxt);
    t->heap = realloc(t->heap, 3ull * n * sizeof *t->heap);  // n - 1 pairs, then at most 2 per merge
    if (!t->sym || !t->prv || !t->nxt || !t->heap) die("out of memory");
  }
  t->nheap = 0;
  for (int i = 0; i < n; i++) t->sym[i] = lookup(t, s + i, 1), t->prv[i] = i - 1, t->nxt[i] = i + 1 < n ? i + 1 : -1;
  for (int i = 0; i + 1 < n; i++) offer(t, i);
  while (t->nheap) {
    Pair p = pop(t);
    int id, i = p.pos, j = t->nxt[i];
    if (t->sym[i] < 0 || j < 0 || merge_of(t, t->sym[i], t->sym[j], &id) != p.rank) continue;  // merged since
    t->sym[i] = id, t->sym[j] = -1, t->nxt[i] = t->nxt[j];
    if (t->nxt[i] >= 0) t->prv[t->nxt[i]] = i;
    if (t->prv[i] >= 0) offer(t, t->prv[i]);
    offer(t, i);
  }
  int k = 0;
  for (int i = 0; i >= 0; i = t->nxt[i]) {
    if (k >= max) die("prompt too long");
    out[k++] = t->sym[i];
  }
  return k;
}

typedef struct { int a, b; } Span;  // UTF-16 unit indices [a, b)

// Text without added tokens: normalized, split by each regex within the
// pieces of the one before, each piece through BPE.
static int encode_text(Tokenizer *t, const char *text, int len, int *out, int max) {
  if (!len) return 0;
  int n, status = 0;
  UChar *u = utf16(text, len, &n);
  if (t->nfc) {
    const UNormalizer2 *nfc = unorm2_getNFCInstance(&status);
    icu_check(status, "NFC");
    int need = unorm2_normalize(nfc, u, n, 0, 0, &status);
    if (status != 15) icu_check(status, "NFC");  // 15: U_BUFFER_OVERFLOW_ERROR, the expected preflight
    UChar *v = xmalloc(((size_t)need + 1) * sizeof *v);
    status = 0;
    n = unorm2_normalize(nfc, u, n, v, need, &status);
    icu_check(status, "NFC");
    free(u), u = v;
  }
  Span *a = xmalloc(((size_t)2 * n + 2) * sizeof *a), *b = xmalloc(((size_t)2 * n + 2) * sizeof *b);
  int na = 1;
  a[0] = (Span){0, n};
  for (int r = 0; r < t->nsplit; r++) {
    int nb = 0;
    for (int i = 0; i < na; i++) {
      int base = a[i].a, consumed = 0, size = a[i].b - a[i].a;
      if (!size) continue;
      status = 0;
      uregex_setText(t->split[r], u + base, size, &status);
      icu_check(status, "regex input");
      while (uregex_findNext(t->split[r], &status)) {
        int s = uregex_start(t->split[r], 0, &status), e = uregex_end(t->split[r], 0, &status);
        icu_check(status, "regex matching");
        if (s > consumed) b[nb++] = (Span){base + consumed, base + s};
        if (e > s) b[nb++] = (Span){base + s, base + e};
        consumed = e;
      }
      icu_check(status, "regex matching");
      if (consumed < size) b[nb++] = (Span){base + consumed, base + size};
    }
    Span *x = a;
    a = b, b = x, na = nb;
  }
  char *piece = xmalloc((size_t)4 * n + 4);
  int m = 0;
  for (int i = 0; i < na; i++) m += bpe(t, piece, utf8(u + a[i].a, a[i].b - a[i].a, piece), out + m, max - m);
  free(u), free(a), free(b), free(piece);
  return m;
}

// TOK_TEXT: text alone; TOK_ADDED: the added tokens' spellings are those
// tokens; TOK_PROMPT: also IMAGE_MARK is the image token, and the image
// token's own spelling is text (a rendered prompt).
static int encode(Tokenizer *t, const char *text, int len, int *out, int max, int mode) {
  int n = 0, start = 0;
  pthread_mutex_lock(&t->lock);
  for (int i = 0; mode && i < len;) {
    int best = -1, id, skip;
    if (mode == TOK_PROMPT && text[i] == IMAGE_MARK[0]) {
      if (t->image < 0) die("an image's place in a prompt for a model without images");
      id = t->image, skip = 1;
    } else {
      for (int k = 0; k < t->nadded; k++)
        if (t->added[k].n <= len - i && !memcmp(text + i, t->added[k].s, t->added[k].n) &&
            !(mode == TOK_PROMPT && t->added[k].id == t->image) &&
            (best < 0 || t->added[k].n > t->added[best].n))
          best = k;
      if (best < 0) {
        i++;
        continue;
      }
      id = t->added[best].id, skip = t->added[best].n;
    }
    n += encode_text(t, text + start, i - start, out + n, max - n);
    if (n >= max) die("prompt too long");
    out[n++] = id;
    i += skip;
    start = i;
  }
  n += encode_text(t, text + start, len - start, out + n, max - n);
  pthread_mutex_unlock(&t->lock);
  return n;
}

int tok_encode_bytes(Tokenizer *t, const char *text, size_t len, int mode, int *out, int max) {
  if (mode < TOK_TEXT || mode > TOK_PROMPT) die("tokenizer: encode mode %d", mode);
  return encode(t, text, (int)len, out, max, mode);
}
int tok_encode_text(Tokenizer *t, const char *text, int *out, int max) {
  return encode(t, text, (int)strlen(text), out, max, TOK_TEXT);
}
int tok_encode(Tokenizer *t, const char *text, int *out, int max) {
  return encode(t, text, (int)strlen(text), out, max, TOK_ADDED);
}
int tok_encode_chat(Tokenizer *t, const char *text, int *out, int max) {
  return encode(t, text, (int)strlen(text), out, max, TOK_PROMPT);
}

const char *tok_piece(const Tokenizer *t, int id, int *len) {
  *len = id >= 0 && id < t->nvocab && t->pieces[id] ? t->piece_len[id] : 0;
  return *len ? t->pieces[id] : "";
}

int tok_added(const Tokenizer *t, int id) {
  for (int k = 0; k < t->nadded; k++)
    if (t->added[k].id == id) return t->special[k] ? 2 : 1;
  return 0;
}

int tok_special(const Tokenizer *t, const char *text, size_t len) {
  for (int k = 0; k < t->nadded; k++)
    if ((size_t)t->added[k].n == len && !memcmp(t->added[k].s, text, len)) return t->added[k].id;
  return -1;
}

int tok_find(const Tokenizer *t, const char *special) {
  int id = tok_special(t, special, strlen(special));
  if (id < 0) die("no special token %s", special);
  return id;
}
