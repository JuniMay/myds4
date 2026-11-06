#ifndef MYDS4_PROTOCOL_DRAFT_H
#define MYDS4_PROTOCOL_DRAFT_H

#include "context_lookup.h"

enum { PROTOCOL_RULES = 3, PROTOCOL_TAIL = 32 };
typedef struct {
  struct { int prefix[PROTOCOL_TAIL], suffix[CONTEXT_LOOKUP_DRAFTS], nprefix, nsuffix, misses; } rule[PROTOCOL_RULES];
  int tail[PROTOCOL_TAIL], count;
} ProtocolDraft;

// The tokenizer must be loaded. Only fixed DSML header fragments are proposed;
// every proposal still requires target-model verification.
void protocol_draft_init(ProtocolDraft *s);
void protocol_draft_push(ProtocolDraft *s, const int *tokens, int count);
ContextDraft protocol_draft_propose(const ProtocolDraft *s, int limit);
void protocol_draft_feedback(ProtocolDraft *s, int rule, int accepted);

#endif
