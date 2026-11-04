#ifndef MYDS4_DRAFT_QUEUE_H
#define MYDS4_DRAFT_QUEUE_H

#include <string.h>

enum { DRAFT_QUEUE_MAX = 5 };
typedef struct { int count, tokens[DRAFT_QUEUE_MAX]; } DraftQueue;

static inline int draft_queue_set(DraftQueue *q, const int *tokens, int count) {
  q->count = 0;
  if (count < 0 || count > DRAFT_QUEUE_MAX) return 0;
  if (count) memcpy(q->tokens, tokens, (size_t)count * sizeof *tokens);
  q->count = count;
  return 1;
}

static inline int draft_queue_take(const DraftQueue *q, int *tokens, int limit) {
  if (limit <= 0) return 0;
  int count = q->count < limit ? q->count : limit;
  if (count) memcpy(tokens, q->tokens, (size_t)count * sizeof *tokens);
  return count;
}

// The target sampled the offered prefix and a next token. The next token
// becomes the following pass's current token, so a matching queued token
// is consumed too. Any mismatch invalidates all remaining guesses.
static inline int draft_queue_accept(DraftQueue *q, int offered, int accepted, int next) {
  if (offered < 1 || offered > q->count || accepted < 0 || accepted > offered) {
    q->count = 0;
    return 0;
  }
  if (accepted != offered || (q->count > offered && q->tokens[offered] != next)) {
    q->count = 0;
    return 1;
  }
  int used = offered + (q->count > offered);
  q->count -= used;
  if (q->count) memmove(q->tokens, q->tokens + used, (size_t)q->count * sizeof *q->tokens);
  return 1;
}

#endif
