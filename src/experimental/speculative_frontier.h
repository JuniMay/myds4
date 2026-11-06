#ifndef MYDS4_SPECULATIVE_FRONTIER_H
#define MYDS4_SPECULATIVE_FRONTIER_H

#include "myds4.h"

// Sample completed verification rows in token order, even when GPU work
// finishes out of order. The caller owns immutable drafts and logits. The
// copied sampler is committed only after done becomes true.
typedef struct {
  Sampler sampler;
  const int *drafts;
  int count, eos, sampled, accepted, next, done;
} SpeculativeFrontier;

static inline int speculative_frontier_start(SpeculativeFrontier *f, const int *drafts,
                                             int count, int eos, const Sampler *sampler) {
  if (!f || !sampler || count < 0 || count > VERIFY_DRAFTS || (count && !drafts)) return 0;
  *f = (SpeculativeFrontier){.sampler = *sampler, .drafts = drafts, .count = count, .eos = eos};
  return 1;
}

// Bit r means that logits row r has completed. There are count + 1 rows:
// the last provides the next token when every draft is accepted.
static inline int speculative_frontier_poll(SpeculativeFrontier *f, const float *logits,
                                            int vocab, uint32_t ready) {
  if (f->done) return 1;
  while (ready & (UINT32_C(1) << f->sampled)) {
    int row = f->sampled++;
    f->next = sample_token(logits + (size_t)row * vocab, vocab, &f->sampler);
    if (row == f->count || f->next == f->eos || f->next != f->drafts[row]) {
      f->accepted = row;
      f->done = 1;
      return 1;
    }
    f->accepted = row + 1;
  }
  return 0;
}

#endif
