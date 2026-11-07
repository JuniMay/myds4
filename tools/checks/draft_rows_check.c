// Decoding several sequences together (model_forward_rows) must write each
// one's draft window keys as decoding it alone does (model_forward with
// --draft); otherwise a sequence that drafts again after decoding in a batch
// attends over a window that lacks those positions. Two states decode their
// next token alone, their draft rings kept, then together: each state's rings
// must be the same, byte for byte.
//
// usage: draft-rows-check CHECKPOINT GGUF (one model process, about a minute)
#include "../../src/engine/model.c"

int main(int argc, char **argv) {
  if (argc != 3) die("usage: draft-rows-check CHECKPOINT GGUF");
  Options o = {.ctx = 4096, .budget_gb = 40, .reserve_gb = 4, .dump_fd = -1, .routes_fd = -1, .quiet = 1,
               .experts = argv[2], .draft = 2, .draft_cache = 32};
  tok_load(argv[1]);
  const char *prompt = "Write a short story about a lighthouse keeper who discovers a message in a bottle.";
  int tokens[256], n = tok_encode(prompt, tokens, 256);
  model_load(argv[1], &o);
  State *st[2] = {model_state_new(), model_state_new()};
  int len[2] = {n, n - 3}, next[2];
  uint8_t *alone[2][DRAFTS];
  for (int s = 0; s < 2; s++) {
    model_state_use(st[s]);
    next[s] = sample_argmax(model_forward(tokens, len[s], 0), VOCAB);
    model_state_save();
    model_forward(&next[s], 1, len[s]);  // alone: its draft keys at len[s]
    for (int d = 0; d < DRAFTS; d++) alone[s][d] = xmalloc(D[d].ring.size), memcpy(alone[s][d], D[d].ring.ptr, D[d].ring.size);
    model_state_restore();
  }
  model_forward_rows(st, next, len, 2);
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
