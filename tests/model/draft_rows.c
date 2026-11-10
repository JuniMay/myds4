// Decoding several sequences together (deepseek_forward_rows) must write each
// one's draft window keys as decoding it alone does (deepseek_forward with
// --draft); otherwise a sequence that drafts again after decoding in a batch
// attends over a window that lacks those positions. Two states decode their
// next token alone, their draft rings kept, then together: each state's rings
// must be the same, byte for byte.
//
// usage: draft-rows-check CHECKPOINT GGUF (one model process, about a minute)
#include "../../src/model/deepseek/model.h"
#include "../../src/model/deepseek/model.c"
static Tokenizer *tk;  // the checkpoint's tokenizer

int main(int argc, char **argv) {
  if (argc != 3) die("usage: draft-rows-check CHECKPOINT GGUF");
  double model_memory_gib = 40;
  double model_reserve_gib = 4;
  Options o = {.ctx = 4096, .dump_fd = -1, .routes_fd = -1, .quiet = 1,
               .deepseek.experts = argv[2], .draft = 2, .deepseek.draft_cache = 32};
  tk = tok_open(argv[1], deepseek_ops.media_token);
  const char *prompt = "Write a short story about a lighthouse keeper who discovers a message in a bottle.";
  int tokens[256], n = tok_encode(tk, prompt, tokens, 256);
  memory_configure(model_memory_gib, model_reserve_gib);
  deepseek_load(argv[1], &o);
  State *st[2] = {deepseek_state_new(), deepseek_state_new()};
  int len[2] = {n, n - 3}, next[2];
  uint8_t *alone[2][DRAFTS];
  for (int s = 0; s < 2; s++) {
    deepseek_state_use(st[s]);
    next[s] = sample_argmax(deepseek_forward(tokens, len[s], 0), VOCAB);
    deepseek_state_save();
    deepseek_forward(&next[s], 1, len[s]);  // alone: its draft keys at len[s]
    for (int d = 0; d < DRAFTS; d++) alone[s][d] = xmalloc(D[d].ring.size), memcpy(alone[s][d], D[d].ring.ptr, D[d].ring.size);
    deepseek_state_restore();
  }
  deepseek_forward_rows(st, next, len, 2);
  int bad = 0;
  for (int s = 0; s < 2; s++)
    for (int d = 0; d < DRAFTS; d++) {
      size_t differ = 0;
      const uint8_t *got = st[s]->dring[d].ptr;
      for (size_t i = 0; i < D[d].ring.size; i++) differ += got[i] != alone[s][d][i];
      if (differ) fprintf(stderr, "FAIL: sequence %d, stage %d: %zu of %zu draft ring bytes differ\n", s, d, differ, D[d].ring.size);
      bad += differ != 0;
    }
  printf("draft rings after decoding together against alone: %s\n", bad ? "FAIL" : "PASS");
  return bad != 0;
}
