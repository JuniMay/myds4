// The checkpoint's tokenizer (tokenizer.json): added tokens matched first,
// then the pre-tokenizer's three splits, then byte-level BPE.
//
//   split 1  \p{N}{1,3}
//   split 2  [一-龥぀-ゟ゠-ヿ]+
//   split 3  [!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+
//            |[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+[\r\n]*
//            |\s*[\r\n]+|\s+(?!\S)|\s+
//
// Each split isolates its matches within the pieces of the previous one.
#include <stdlib.h>
#include <string.h>

#include "myds4.h"

_Alignas(8) static const unsigned char engram_data[] = {
#embed "../../data/engram.bin"
};
_Alignas(4) static const unsigned char unicode_data[] = {
#embed "../../data/unicode.bin"
};
#define unicode_runs ((const uint32_t *)unicode_data)
const int32_t *engram_map;
const int64_t *engram_mult = (const int64_t *)engram_data;
const int64_t *engram_primes = (const int64_t *)engram_data + 8;
const int64_t *engram_offsets = (const int64_t *)engram_data + 56;
int engram_pad(void) { return *(const int32_t *)(engram_data + 104 * 8); }

enum { OTHER, LETTER, MARK, NUMBER, PUNCT, SYMBOL };

static int uclass(uint32_t c) {
  int lo = 0, hi = sizeof unicode_data / 4 - 1;
  while (lo < hi) {  // last run starting at or before c
    int mid = (lo + hi + 1) / 2;
    if ((unicode_runs[mid] >> 3) <= c) lo = mid; else hi = mid - 1;
  }
  return unicode_runs[lo] & 7;
}

static int space(uint32_t c) {  // White_Space
  return (c >= 9 && c <= 13) || c == 32 || c == 0x85 || c == 0xA0 || c == 0x1680 ||
         (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 || c == 0x202F ||
         c == 0x205F || c == 0x3000;
}

// ---- strings to ids ----

typedef struct { const char *s; int n, id; } Key;
static Key *vocab;
static uint64_t vcap;
static struct { uint64_t pair; int rank, id; } *merges;
static uint64_t mcap;
static char **pieces;  // raw bytes of each id
static int *piece_len, nvocab;
static struct { const char *s; int n, id; } added[2048];
static int nadded;

static uint64_t fnv(const char *s, int n) {
  uint64_t h = 1469598103934665603ull;
  for (int i = 0; i < n; i++) h = (h ^ (uint8_t)s[i]) * 1099511628211ull;
  return h;
}

static int lookup(const char *s, int n) {
  for (uint64_t i = fnv(s, n) & (vcap - 1);; i = (i + 1) & (vcap - 1)) {
    if (!vocab[i].s) return -1;
    if (vocab[i].n == n && !memcmp(vocab[i].s, s, n)) return vocab[i].id;
  }
}

static void insert(const char *s, int n, int id) {
  uint64_t i = fnv(s, n) & (vcap - 1);
  while (vocab[i].s) i = (i + 1) & (vcap - 1);
  vocab[i] = (Key){s, n, id};
}

static uint64_t mix(uint64_t x) {
  x ^= x >> 33;
  x *= 0xff51afd7ed558ccdull;
  return x ^ x >> 33;
}

static int merge_of(int a, int b, int *id) {
  uint64_t key = (uint64_t)a << 32 | (uint32_t)b;
  for (uint64_t i = mix(key) & (mcap - 1);; i = (i + 1) & (mcap - 1)) {
    if (!merges[i].pair) return -1;
    if (merges[i].pair == key + 1) return *id = merges[i].id, merges[i].rank;
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

void tok_load(const char *dir) {
  // the Engram table's compressed token map follows its hashing constants
  engram_map = (const int32_t *)(engram_data + (8 + 48 + 48) * 8 + 8);
  for (int b = 0, n = 0; b < 256; b++) {
    int keep = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || b >= 174;
    unbyte[keep ? b : 256 + n++] = b;
  }
  char path[4096];
  snprintf(path, sizeof path, "%s/tokenizer.json", dir);
  Json *root = json_parse(read_file(path, 0)), *model = json_get(root, "model");
  Json *v = json_get(model, "vocab"), *m = json_get(model, "merges"), *add = json_get(root, "added_tokens");
  nvocab = VOCAB;
  pieces = xmalloc(nvocab * sizeof *pieces);
  piece_len = xmalloc(nvocab * sizeof *piece_len);
  // Added tokens use literal UTF-8. Some also occur in model.vocab, where
  // converting them through the byte-level alphabet would read out of bounds.
  for (int i = 0; i < add->n && nadded < 2048; i++) {
    Json *t = &add->kid[i];
    Json *c = json_get(t, "content");
    int id = (int)json_get(t, "id")->num;
    added[nadded].s = c->s;
    added[nadded].n = c->n;
    added[nadded++].id = id;
    if (id >= 0 && id < nvocab) pieces[id] = (char *)c->s, piece_len[id] = c->n;
  }
  for (vcap = 1; vcap < 2ull * v->n; vcap *= 2) {}
  vocab = xmalloc(vcap * sizeof *vocab);
  for (int i = 0; i < v->n; i++) {
    int id = (int)v->kid[i].num, len = strlen(v->kid[i].key);
    if (id < 0 || id >= nvocab) die("tokenizer: vocabulary id out of range");
    if (pieces[id]) continue;
    char *r = xmalloc(len + 1);
    int n = raw(v->kid[i].key, len, r);
    insert(r, n, id);
    pieces[id] = r;
    piece_len[id] = n;
  }
  for (mcap = 1; mcap < 2ull * m->n; mcap *= 2) {}
  merges = xmalloc(mcap * sizeof *merges);
  for (int i = 0; i < m->n; i++) {
    const char *s = m->kid[i].s, *sp = strchr(s, ' ');
    char a[256], b[256], ab[512];
    int na = raw(s, sp - s, a), nb = raw(sp + 1, strlen(sp + 1), b);
    memcpy(ab, a, na);
    memcpy(ab + na, b, nb);
    int ia = lookup(a, na), ib = lookup(b, nb), id = lookup(ab, na + nb);
    if (ia < 0 || ib < 0 || id < 0) die("tokenizer: bad merge %s", s);
    uint64_t key = (uint64_t)ia << 32 | (uint32_t)ib, j = mix(key) & (mcap - 1);
    while (merges[j].pair) j = (j + 1) & (mcap - 1);
    merges[j].pair = key + 1;
    merges[j].rank = i;
    merges[j].id = id;
  }
}

// ---- encoding ----

typedef struct { int a, b; } Span;  // code point indices [a, b)

static uint32_t *cp;   // the segment's code points
static int *cpoff;     // and their byte offsets

// The piece's symbols as BPE merges them: each symbol's id (-1 once merged
// into the one before it) and its neighbors; and a heap of the adjacent pairs
// a merge joins, the lowest rank first, the leftmost among equals. Sized for
// the longest piece so far (tok_encode runs under the tokenizer's lock).
typedef struct { int rank, pos; } Pair;
static int *sym, *prv, *nxt, cap_bpe, nheap;
static Pair *heap;

static int before(Pair a, Pair b) { return a.rank < b.rank || (a.rank == b.rank && a.pos < b.pos); }

static void push(int rank, int pos) {
  Pair p = {rank, pos};
  int i = nheap++;
  for (; i > 0 && before(p, heap[(i - 1) / 2]); i = (i - 1) / 2) heap[i] = heap[(i - 1) / 2];
  heap[i] = p;
}

static Pair pop(void) {
  Pair top = heap[0], last = heap[--nheap];
  int i = 0;
  for (int c; (c = 2 * i + 1) < nheap; i = c) {
    if (c + 1 < nheap && before(heap[c + 1], heap[c])) c++;
    if (!before(heap[c], last)) break;
    heap[i] = heap[c];
  }
  heap[i] = last;
  return top;
}

// The pair the symbol at i starts, onto the heap if a merge joins it.
static void offer(int i) {
  int id, j = nxt[i], rank = j < 0 ? -1 : merge_of(sym[i], sym[j], &id);
  if (rank >= 0) push(rank, i);
}

// Byte-level BPE over the whole piece of n bytes, as the reference merges:
// the lowest-ranked adjacent pair first, the leftmost among equals, until no
// merge applies. A heap keeps a long piece at O(n log n).
static int bpe(const char *s, int n, int *out, int max) {
  if (n <= 0) return 0;
  if (n > cap_bpe) {
    cap_bpe = n;
    sym = realloc(sym, n * sizeof *sym), prv = realloc(prv, n * sizeof *prv), nxt = realloc(nxt, n * sizeof *nxt);
    heap = realloc(heap, 3ull * n * sizeof *heap);  // n - 1 pairs, then at most 2 per merge
    if (!sym || !prv || !nxt || !heap) die("out of memory");
  }
  nheap = 0;
  for (int i = 0; i < n; i++) sym[i] = lookup(s + i, 1), prv[i] = i - 1, nxt[i] = i + 1 < n ? i + 1 : -1;
  for (int i = 0; i + 1 < n; i++) offer(i);
  while (nheap) {
    Pair p = pop();
    int id, i = p.pos, j = nxt[i];
    if (sym[i] < 0 || j < 0 || merge_of(sym[i], sym[j], &id) != p.rank) continue;  // merged since
    sym[i] = id, sym[j] = -1, nxt[i] = nxt[j];
    if (nxt[i] >= 0) prv[nxt[i]] = i;
    if (prv[i] >= 0) offer(prv[i]);
    offer(i);
  }
  int k = 0;
  for (int i = 0; i >= 0; i = nxt[i]) {
    if (k >= max) die("prompt too long");
    out[k++] = sym[i];
  }
  return k;
}

static int cls(int i) { return uclass(cp[i]); }
static int lm(int i) { int c = cls(i); return c == LETTER || c == MARK; }
static int ps(int i) { int c = cls(i); return c == PUNCT || c == SYMBOL; }
static int nl(int i) { return cp[i] == '\r' || cp[i] == '\n'; }
static int ascii_letter(uint32_t c) { return (c | 32) >= 'a' && (c | 32) <= 'z'; }

// Length of split 3's match at i within [i, b), or 0.
static int match3(int i, int b) {
  int j;
  if (i + 1 < b && cp[i] < 128 && strchr("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~", cp[i]) && cp[i] &&
      ascii_letter(cp[i + 1])) {
    for (j = i + 1; j < b && ascii_letter(cp[j]); j++) {}
    return j - i;
  }
  int c = cls(i);
  j = i + (!nl(i) && c != LETTER && c != PUNCT && c != SYMBOL && i + 1 < b && lm(i + 1));
  if (lm(j)) {
    while (j < b && lm(j)) j++;
    return j - i;
  }
  j = i + (cp[i] == ' ' && i + 1 < b && ps(i + 1));
  if (ps(j)) {
    while (j < b && ps(j)) j++;
    while (j < b && nl(j)) j++;
    return j - i;
  }
  if (!space(cp[i])) return 0;
  int e = i;
  while (e < b && space(cp[e])) e++;
  for (j = e - 1; j >= i; j--)
    if (nl(j)) return j + 1 - i;
  if (e == b) return e - i;
  return e - i > 1 ? e - 1 - i : e - i;
}

static int cjk(uint32_t c) {
  return (c >= 0x4E00 && c <= 0x9FA5) || (c >= 0x3040 && c <= 0x309F) || (c >= 0x30A0 && c <= 0x30FF);
}

// Isolates the matches of split k within s, appending pieces to out.
static int split(int k, Span s, Span *out) {
  int n = 0, gap = s.a;
  for (int i = s.a; i < s.b;) {
    int len = 0;
    if (k == 1) while (len < 3 && i + len < s.b && cls(i + len) == NUMBER) len++;
    else if (k == 2) while (i + len < s.b && cjk(cp[i + len])) len++;
    else len = match3(i, s.b);
    if (!len) { i++; continue; }
    if (gap < i) out[n++] = (Span){gap, i};
    out[n++] = (Span){i, i + len};
    i += len;
    gap = i;
  }
  if (gap < s.b) out[n++] = (Span){gap, s.b};
  return n;
}

static int encode_text(const char *text, int len, int *out, int max) {
  if (!len) return 0;
  cp = xmalloc((len + 1) * sizeof *cp);
  cpoff = xmalloc((len + 1) * sizeof *cpoff);
  int n = 0;
  for (int i = 0, k; i < len; i += k) cpoff[n] = i, cp[n++] = utf8_next((const unsigned char *)text + i, &k);
  cpoff[n] = len;
  Span *a = xmalloc(4 * (n + 1) * sizeof *a), *b = xmalloc(4 * (n + 1) * sizeof *b);
  int na = split(1, (Span){0, n}, a), nb = 0;
  for (int i = 0; i < na; i++) nb += split(2, a[i], b + nb);
  na = 0;
  for (int i = 0; i < nb; i++) na += split(3, b[i], a + na);
  int m = 0;
  for (int i = 0; i < na; i++)
    m += bpe(text + cpoff[a[i].a], cpoff[a[i].b] - cpoff[a[i].a], out + m, max - m);
  free(cp), free(cpoff), free(a), free(b);
  return m;
}

// chat: IMAGE_MARK is the image token, and the image token's text is text.
static int encode(const char *text, int *out, int max, int chat) {
  int n = 0, len = strlen(text), start = 0;
  for (int i = 0; i < len;) {
    int best = -1, id, skip;
    if (chat && text[i] == IMAGE_MARK[0]) {
      id = IMAGE_TOKEN, skip = 1;
    } else {
      for (int k = 0; k < nadded; k++)
        if (added[k].n <= len - i && !memcmp(text + i, added[k].s, added[k].n) && !(chat && added[k].id == IMAGE_TOKEN) &&
            (best < 0 || added[k].n > added[best].n))
          best = k;
      if (best < 0) { i++; continue; }
      id = added[best].id, skip = added[best].n;
    }
    n += encode_text(text + start, i - start, out + n, max - n);
    if (n >= max) die("prompt too long");
    out[n++] = id;
    i += skip;
    start = i;
  }
  return n + encode_text(text + start, len - start, out + n, max - n);
}

int tok_encode(const char *text, int *out, int max) { return encode(text, out, max, 0); }
int tok_encode_chat(const char *text, int *out, int max) { return encode(text, out, max, 1); }

const char *tok_piece(int id, int *len) {
  *len = id >= 0 && id < nvocab && pieces[id] ? piece_len[id] : 0;
  return *len ? pieces[id] : "";
}

int tok_added(int id) {
  for (int k = 0; k < nadded; k++)
    if (added[k].id == id) return 1;
  return 0;
}

int tok_special(const char *text, size_t len) {
  for (int k = 0; k < nadded; k++)
    if (strlen(added[k].s) == len && !memcmp(added[k].s, text, len)) return added[k].id;
  return -1;
}

int tok_find(const char *special) {
  int id = tok_special(special, strlen(special));
  if (id < 0) die("no special token %s", special);
  return id;
}
