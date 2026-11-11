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

// How many of a draft model's drafts a verification pass runs. The outputs
// do not depend on it, since a pass keeps only the drafts that the target's
// own samples confirm; the speed does, because every verified row costs GPU
// work and expert reads, kept or not. The choice is the count k with the
// most expected tokens per second: 1 plus, for each position j up to k, the
// rate at which drafts 1 to j were all kept, over the measured time of a pass
// with k drafts. Before choosing it measures every count DRAFT_CHOICE_WARM
// times, in rounds from the most down, so that no count's figure rests on one
// pass or on the cold cache after a prompt alone; afterwards every
// DRAFT_CHOICE_EXPLORE-th pass tries a count next to the best, so that their
// figures follow the text.
enum { DRAFT_CHOICE_MAX = 15, DRAFT_CHOICE_WARM = 3, DRAFT_CHOICE_EXPLORE = 8, DRAFT_CHOICE_MEMORY = 8 };
typedef struct DraftChoice {
  double kept[DRAFT_CHOICE_MAX + 1];     // [j], j >= 1: the rate at which drafts 1 to j were all kept
  double seconds[DRAFT_CHOICE_MAX + 1];  // [k]: the time of a pass with k drafts
  int kept_n[DRAFT_CHOICE_MAX + 1], seconds_n[DRAFT_CHOICE_MAX + 1];  // observations, up to DRAFT_CHOICE_MEMORY
  int passes;
} DraftChoice;
void draft_choice_init(DraftChoice *choice);
int draft_choice_count(DraftChoice *choice, int most);  // 0 to most (at most DRAFT_CHOICE_MAX)
// A pass with `count` drafts kept the first `kept` of them and took `seconds`.
void draft_choice_feedback(DraftChoice *choice, int count, int kept, double seconds);

#endif
