// A checkpoint's tokenizer against reference ids: a JSON fixture of
// {"text", "ids"} cases (the reference tokenizer's ids, its added tokens'
// spellings taken as those tokens). Texts may hold any character, NUL
// included. No model or GPU.
//
// usage: tokenizer-reference-check CHECKPOINT FIXTURE
#include <stdlib.h>
#include <string.h>

#include "myds4.h"

int main(int argc, char **argv) {
  if (argc != 3) die("usage: tokenizer-reference-check CHECKPOINT FIXTURE");
  Tokenizer *tk = tok_open(argv[1], 0);
  char *source = read_file(argv[2], 0);
  Json *cases = json_parse(source);
  free(source);
  if (!cases || cases->type != J_ARR) die("%s: not a list of cases", argv[2]);
  int failed = 0, checked = 0;
  for (int c = 0; c < cases->n; c++) {
    const Json *text = json_get(&cases->kid[c], "text"), *ids = json_get(&cases->kid[c], "ids");
    if (!text || text->type != J_STR || !ids || ids->type != J_ARR) die("%s: case %d without text or ids", argv[2], c);
    int max = 4 * text->n + 4, *got = xmalloc((size_t)max * sizeof *got);
    int n = tok_encode_bytes(tk, text->s, (size_t)text->n, TOK_ADDED, got, max), same = n == ids->n;
    for (int i = 0; same && i < n; i++) same = got[i] == (int)ids->kid[i].num;
    const Json *roundtrip = json_get(&cases->kid[c], "roundtrip"), *added = json_get(&cases->kid[c], "added");
    if (roundtrip && roundtrip->num) {
      int offset = 0;
      for (int i = 0; same && i < n; i++) {
        int length;
        const char *piece = tok_piece(tk, got[i], &length);
        same = offset + length <= text->n && !memcmp(text->s + offset, piece, length);
        offset += length;
      }
      same &= offset == text->n;
    }
    if (added && added->num) same &= n == 1 && tok_added(tk, got[0]);
    if (!same) {
      failed++;
      printf("case %d (%d bytes): %d ids, the reference's %d\n", c, text->n, n, ids->n);
    }
    checked++;
    free(got);
  }
  printf("tokenizer against reference ids: %d of %d cases match\n", checked - failed, checked);
  json_free(cases);
  tok_close(tk);
  return failed != 0;
}
