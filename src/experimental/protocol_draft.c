#include "myds4.h"
#include "protocol_draft.h"
#include <string.h>

void protocol_draft_init(ProtocolDraft *s) {
  memset(s, 0, sizeof *s);
  const char *prefix[] = {"<｜DSML｜ calls", "<｜DSML｜ invoke", "<｜DSML｜ parameter"};
  const char *suffix[] = {">\n<｜DSML｜ invoke name=\"", " name=\"", " name=\""};
  // Keep only tokens whose encoding is independent of the following name.
  // A quote or punctuation token can merge with the first value character.
  const char *value[] = {"apple", "zebra", "functions.exec", "_name", "0", "工具"};
  for (int r = 0; r < PROTOCOL_RULES; r++) {
    s->rule[r].nprefix = tok_encode(prefix[r], s->rule[r].prefix, PROTOCOL_TAIL);
    int reference[64], common = 0;
    for (unsigned v = 0; v < sizeof value / sizeof *value; v++) {
      char text[256];
      int ids[64];
      snprintf(text, sizeof text, "%s%s%s", prefix[r], suffix[r], value[v]);
      int count = tok_encode(text, ids, 64);
      if (!v) memcpy(reference, ids, count * sizeof(int)), common = count;
      else {
        if (count < common) common = count;
        int i = 0;
        while (i < common && ids[i] == reference[i]) i++;
        common = i;
      }
    }
    int n = s->rule[r].nprefix;
    if (common <= n || memcmp(reference, s->rule[r].prefix, n * sizeof(int))) continue;
    int keep = common - n;
    if (keep > CONTEXT_LOOKUP_DRAFTS) keep = CONTEXT_LOOKUP_DRAFTS;
    s->rule[r].nsuffix = keep;
    memcpy(s->rule[r].suffix, reference + n, keep * sizeof(int));
  }
}

void protocol_draft_push(ProtocolDraft *s, const int *tokens, int count) {
  if (count < 0) die("invalid protocol history append");
  if (count >= PROTOCOL_TAIL) {
    memcpy(s->tail, tokens + count - PROTOCOL_TAIL, sizeof s->tail);
    s->count = PROTOCOL_TAIL;
    return;
  }
  int keep = s->count;
  if (keep > PROTOCOL_TAIL - count) keep = PROTOCOL_TAIL - count;
  memmove(s->tail, s->tail + s->count - keep, keep * sizeof(int));
  memcpy(s->tail + keep, tokens, count * sizeof(int));
  s->count = keep + count;
}

ContextDraft protocol_draft_propose(const ProtocolDraft *s, int limit) {
  ContextDraft out = {.source = -1};
  if (limit <= 0) return out;
  for (int r = 0; r < PROTOCOL_RULES; r++) {
    int n = s->rule[r].nprefix;
    if (!n || !s->rule[r].nsuffix || s->rule[r].misses >= 2 || n > s->count ||
        memcmp(s->tail + s->count - n, s->rule[r].prefix, n * sizeof(int))) continue;
    out.count = s->rule[r].nsuffix < limit ? s->rule[r].nsuffix : limit;
    out.matched = n, out.source = r, out.examined = r + 1;
    memcpy(out.tokens, s->rule[r].suffix, out.count * sizeof(int));
    return out;
  }
  return out;
}

void protocol_draft_feedback(ProtocolDraft *s, int rule, int accepted) {
  if (rule < 0 || rule >= PROTOCOL_RULES || accepted < 0) die("invalid protocol draft feedback");
  s->rule[rule].misses = accepted ? 0 : s->rule[rule].misses + 1;
}
