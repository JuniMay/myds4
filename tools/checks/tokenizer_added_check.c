#include "myds4.h"
#include <string.h>
#include <unistd.h>

// Run with address and undefined-behavior sanitizers. The checkpoint lists
// three Unicode added tokens in the BPE vocabulary as well as added_tokens.
int main(int argc, char **argv) {
  if (argc != 2) die("usage: tokenizer-added-check CHECKPOINT");
  alarm(20);
  tok_load(argv[1]);
  const char *text[] = {"<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>", "<｜▁pad▁｜>", "｜DSML｜"};
  int expected[] = {0, 1, 2, 128825}, failures = 0;
  for (int i = 0; i < 4; i++) {
    int ids[32], length = 0, n = tok_encode(text[i], ids, 32);
    const char *piece = tok_piece(expected[i], &length);
    if (n != 1 || ids[0] != expected[i] || length != (int)strlen(text[i]) ||
        memcmp(piece, text[i], length) || !tok_added(expected[i])) {
      fprintf(stderr, "FAIL: added token %d\n", expected[i]);
      failures++;
    }
  }
  const char *ordinary = " Review the queue. 文件 12345.\n";
  int ids[64], n = tok_encode(ordinary, ids, 64), offset = 0;
  for (int i = 0; i < n; i++) {
    int length;
    const char *piece = tok_piece(ids[i], &length);
    if (offset + length > (int)strlen(ordinary) || memcmp(ordinary + offset, piece, length)) failures++;
    offset += length;
  }
  if (offset != (int)strlen(ordinary)) failures++;
  printf("tokenizer added-token load and round trip: %s\n", failures ? "FAIL" : "PASS");
  return failures != 0;
}
