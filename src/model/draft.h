#ifndef MYDS4_CONTEXT_LOOKUP_H
#define MYDS4_CONTEXT_LOOKUP_H

#include <stdint.h>

enum { CONTEXT_LOOKUP_DRAFTS = 15, CONTEXT_LOOKUP_MATCH = 64, CONTEXT_LOOKUP_SEARCH = 64 }; // search limit per index
typedef struct ContextLookup ContextLookup;
typedef struct {
  int count, matched, source, examined;
  int32_t tokens[CONTEXT_LOOKUP_DRAFTS];
} ContextDraft;

// CPU candidate generator. Callers append only committed history and
// verify every proposal with the target model.
ContextLookup *context_lookup_new(int capacity); // 1..1048576 tokens, NULL on failure
void context_lookup_free(ContextLookup *lookup);
void context_lookup_reset(ContextLookup *lookup);
int context_lookup_append(ContextLookup *lookup, const int32_t *tokens, int count);
ContextDraft context_lookup_propose(const ContextLookup *lookup, int max_drafts);
// Record the actual offered count and the prefix accepted by the target.
// An empty proposal does not count as failed verification.
int context_lookup_feedback(ContextLookup *lookup, int offered, int accepted);

#endif
