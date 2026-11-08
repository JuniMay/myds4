// Engram's token normalization is computed from the configured tokenizer.
// The supported Flash layout uses two layers, n-grams 2..4 and eight heads.
#include "engram.h"
#include "../../engine/myds4.h"
#include <stdlib.h>
#include <string.h>

typedef uint16_t UChar;
typedef struct UNormalizer2 UNormalizer2;
extern const UNormalizer2 *unorm2_getNFKCInstance(int32_t *status);
extern const UNormalizer2 *unorm2_getNFDInstance(int32_t *status);
extern int32_t unorm2_normalize(const UNormalizer2 *, const UChar *, int32_t, UChar *, int32_t, int32_t *);
extern UChar *u_strFromUTF8(UChar *, int32_t, int32_t *, const char *, int32_t, int32_t *);
extern int8_t u_charType(int32_t codepoint);
extern int32_t u_tolower(int32_t codepoint);
extern int8_t u_isUWhiteSpace(int32_t codepoint);

typedef struct { char *text; int length, id; } Key;

static uint64_t hash(const char *text, int length) {
  uint64_t h = 1469598103934665603ull;
  for (int i = 0; i < length; i++) h = (h ^ (uint8_t)text[i]) * 1099511628211ull;
  return h;
}
static void append(Str *s, uint32_t c) {
  char text[4]; int n;
  if (c < 128) text[0] = (char)c, n = 1;
  else if (c < 2048) text[0] = (char)(0xc0 | c >> 6), text[1] = (char)(0x80 | (c & 63)), n = 2;
  else if (c < 65536) text[0] = (char)(0xe0 | c >> 12), text[1] = (char)(0x80 | (c >> 6 & 63)), text[2] = (char)(0x80 | (c & 63)), n = 3;
  else text[0] = (char)(0xf0 | c >> 18), text[1] = (char)(0x80 | (c >> 12 & 63)), text[2] = (char)(0x80 | (c >> 6 & 63)), text[3] = (char)(0x80 | (c & 63)), n = 4;
  str_put(s, text, n);
}
static UChar *normalize(const UNormalizer2 *normalizer, UChar *text, int32_t *length) {
  int32_t error = 0, size = unorm2_normalize(normalizer, text, *length, NULL, 0, &error);
  if (error > 0 && error != 15) die("Engram normalization failed");
  UChar *out = xmalloc((size_t)(size + 1) * 2);
  error = 0;
  *length = unorm2_normalize(normalizer, text, *length, out, size + 1, &error);
  free(text);
  if (error > 0) die("Engram normalization failed");
  return out;
}
static Str original(const Tokenizer *tokenizer, int id, const char *text, int length) {
  Str out = {0};
  if (tok_added(tokenizer, id)) { str_put(&out, text, length); return out; }
  int map[256], next = 256;
  for (int b = 0; b < 256; b++) map[b] = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || b >= 174 ? b : next++;
  for (int i = 0; i < length; i++) append(&out, (uint32_t)map[(uint8_t)text[i]]);
  return out;
}
static Str key(const Tokenizer *tokenizer, int id, const UNormalizer2 *nfkc, const UNormalizer2 *nfd) {
  int length;
  const char *text = tok_piece(tokenizer, id, &length);
  int32_t error = 0, size = 0;
  u_strFromUTF8(NULL, 0, &size, text, length, &error);
  int replacement = 0;
  for (int i = 0; i + 2 < length; i++)
    if ((uint8_t)text[i] == 0xef && (uint8_t)text[i+1] == 0xbf && (uint8_t)text[i+2] == 0xbd) replacement = 1;
  if ((error > 0 && error != 15) || replacement) return original(tokenizer, id, text, length);
  UChar *u = xmalloc((size_t)(size + 1) * 2);
  error = 0;
  u_strFromUTF8(u, size + 1, &size, text, length, &error);
  if (error > 0) die("Engram token decoding failed");
  u = normalize(nfkc, u, &size); u = normalize(nfd, u, &size);
  int32_t *points = xmalloc((size_t)(size + 1) * 4); int count = 0;
  for (int i = 0; i < size; i++) {
    int32_t c = u[i];
    if (c >= 0xd800 && c <= 0xdbff && i + 1 < size) c = 0x10000 + ((c - 0xd800) << 10) + (u[++i] - 0xdc00);
    int category = u_charType(c);
    if (category >= 6 && category <= 8) continue;
    c = u_tolower(c);
    if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
      if (count && points[count - 1] == ' ') continue;
      c = ' ';
    }
    points[count++] = c;
  }
  free(u);
  int first = 0, last = count;
  if (!(count == 1 && points[0] == ' ')) {
    while (first < last && u_isUWhiteSpace(points[first])) first++;
    while (last > first && u_isUWhiteSpace(points[last - 1])) last--;
  }
  Str out = {0};
  for (int i = first; i < last; i++) append(&out, points[i] == 0xe000 ? ' ' : (uint32_t)points[i]);
  free(points);
  if (!out.n) str_put(&out, text, length);
  return out;
}

static int prime(int64_t n) {
  if (n < 2 || !(n & 1)) return n == 2;
  for (int64_t d = 3; d * d <= n; d += 2) if (n % d == 0) return 0;
  return 1;
}

void engram_init(Engram *layout, const char *checkpoint) {
  static const int64_t multipliers[8] = {
    76632096046245, 4839876093313, 35959672319349, 73987337458391,
    67716810739261, 51510806800915, 30921347202721, 82619226485591
  }; // Reference seeds 10007 * layer_id, compressed vocabulary 99092.
  memset(layout, 0, sizeof *layout);
  Tokenizer *tokenizer = tok_open(checkpoint, NULL);
  layout->count = tok_vocab(tokenizer);
  if (layout->count != 129280) die("unsupported Engram vocabulary size");
  layout->map = xmalloc((size_t)layout->count * 4);
  int capacity = 1;
  while (capacity < 2 * layout->count) capacity *= 2;
  Key *table = xmalloc((size_t)capacity * sizeof *table);
  int32_t error = 0;
  const UNormalizer2 *nfkc = unorm2_getNFKCInstance(&error), *nfd = unorm2_getNFDInstance(&error);
  if (error > 0) die("Engram normalizers are unavailable");
  int unique = 0;
  for (int id = 0; id < layout->count; id++) {
    Str s = key(tokenizer, id, nfkc, nfd);
    size_t at = hash(s.p, (int)s.n) & (capacity - 1);
    while (table[at].text && (table[at].length != (int)s.n || memcmp(table[at].text, s.p, s.n))) at = (at + 1) & (capacity - 1);
    if (!table[at].text) table[at] = (Key){s.p, (int)s.n, unique++}; else free(s.p);
    layout->map[id] = table[at].id;
  }
  for (int i = 0; i < capacity; i++) free(table[i].text);
  free(table); tok_close(tokenizer);
  if (unique != 99092) die("Engram normalized vocabulary differs from the supported checkpoint: %d", unique);
  memcpy(layout->multipliers, multipliers, sizeof multipliers);
  int64_t current = 15999999;
  for (int i = 0; i < 48; i++) {
    do current++; while (!prime(current));
    layout->primes[i] = current;
    layout->offsets[i] = i % 24 ? layout->offsets[i-1] + layout->primes[i-1] : 0;
  }
  layout->pad = layout->map[2];
}

void engram_free(Engram *layout) {
  free(layout->map);
  memset(layout, 0, sizeof *layout);
}
