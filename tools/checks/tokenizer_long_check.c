#include "myds4.h"
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

// A pre-token longer than 1,024 bytes must tokenize as the checkpoint's
// tokenizer.json does in the reference implementation (Hugging Face tokenizers
// 0.23.3, encode without special tokens): byte-level BPE over the whole
// piece. The expected ids below are that implementation's. Run with address
// and undefined-behavior sanitizers.

static int failures;

static void check(const char *name, const char *text, const int *want, int nwant) {
  int max = 8192, *ids = xmalloc(max * sizeof *ids), n = tok_encode(text, ids, max), first = -1;
  for (int i = 0; i < n && i < nwant && first < 0; i++)
    if (ids[i] != want[i]) first = i;
  if (first < 0 && n != nwant) first = n < nwant ? n : nwant;
  if (first >= 0) {
    fprintf(stderr, "FAIL: %s: %d ids, the reference %d; the first difference at id %d", name, n, nwant, first);
    if (first < n && first < nwant) fprintf(stderr, " (%d, the reference %d)", ids[first], want[first]);
    fprintf(stderr, "\n");
    failures++;
  }
  free(ids);
}

// n copies of s
static char *repeat(const char *s, int n) {
  size_t k = strlen(s);
  char *r = xmalloc(k * n + 1);
  for (int i = 0; i < n; i++) memcpy(r + k * i, s, k);
  r[k * n] = 0;
  return r;
}

int main(int argc, char **argv) {
  if (argc != 2) die("usage: tokenizer-long-check CHECKPOINT");
  alarm(20);
  tok_load(argv[1]);
  int *want = xmalloc(1024 * sizeof *want);

  // 400 CJK characters, one 1,200-byte piece: byte 1,024 falls inside the
  // 342nd character
  char *cjk = repeat("中", 400);
  for (int i = 0; i < 400; i++) want[i] = 525;
  check("400 x 中", cjk, want, 400);

  // a letters-only run of 1,400 bytes: a token crosses byte 1,024
  static const int fox[13] = {1805, 69582, 68, 4775, 47971, 76, 3798, 821, 432, 72016, 1521, 2628, 520};
  char *letters = repeat("thequickbrownfoxjumpsoverthelazydog", 40);
  for (int i = 0; i < 520; i++) want[i] = fox[i % 13];
  check("40 x thequickbrownfoxjumpsoverthelazydog", letters, want, 520);

  // 1,500 letters a, and 1,099 spaces between two letters: runs whose merges
  // happen to align with 1,024 bytes
  char *a = repeat("a", 1500);
  for (int i = 0; i < 187; i++) want[i] = 89086;
  want[187] = 39535;
  check("1500 x a", a, want, 188);
  char *spaces = repeat(" ", 1100), *x = xmalloc(1103);
  snprintf(x, 1103, "x%sy", spaces);
  int sp[11] = {90, 69933, 69933, 69933, 69933, 69933, 69933, 69933, 69933, 124189, 383};
  check("x, 1100 spaces, y", x, sp, 11);

  printf("tokenizer pieces over 1,024 bytes: %s\n", failures ? "FAIL" : "PASS");
  free(cjk), free(letters), free(a), free(spaces), free(x), free(want);
  return failures != 0;
}
