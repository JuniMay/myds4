#include "myds4.h"
#include "../../src/experimental/protocol_draft.h"
#include <string.h>
#include <unistd.h>

static int checks, failures;
static void expect(const char *name, int ok) {
  checks++;
  if (!ok) failures++, fprintf(stderr, "FAIL: %s\n", name);
}

int main(int argc, char **argv) {
  if (argc != 2) die("usage: protocol-draft-check CHECKPOINT");
  alarm(20);
  tok_load(argv[1]);
  ProtocolDraft s;
  protocol_draft_init(&s);
  int active = 0;
  for (int r = 0; r < PROTOCOL_RULES; r++) {
    active += s.rule[r].nprefix > 0 && s.rule[r].nsuffix > 0;
    printf("protocol rule %d: prefix=%d suffix=%d tokens\n", r, s.rule[r].nprefix, s.rule[r].nsuffix);
  }
  expect("all canonical header rules have stable token boundaries", active == PROTOCOL_RULES);
  expect("empty output has no proposal", protocol_draft_propose(&s, 15).count == 0);
  const char *examples[] = {
    "<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"functions.exec\">\n<｜DSML｜ parameter name=\"cmd\" string=\"true\">pwd</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>",
    "<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"read_file\">\n<｜DSML｜ parameter name=\"path\" string=\"true\">src/main.c</｜DSML｜ parameter>\n<｜DSML｜ parameter name=\"limit\" string=\"false\">42</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>",
    "<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"工具\">\n<｜DSML｜ parameter name=\"文件\" string=\"true\">文档</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>"
  };
  for (unsigned example = 0; example < sizeof examples / sizeof *examples; example++) {
    int ids[256], n = tok_encode(examples[example], ids, 256), proposed = 0, correct = 1, immutable = 1;
    protocol_draft_init(&s);
    for (int i = 0; i < n; i++) {
      protocol_draft_push(&s, ids + i, 1);
      ProtocolDraft saved = s;
      ContextDraft d = protocol_draft_propose(&s, 15);
      immutable &= !memcmp(&saved, &s, sizeof s);
      if (d.count) {
        proposed++;
        correct &= i + d.count < n && !memcmp(d.tokens, ids + i + 1, d.count * sizeof(int));
        protocol_draft_feedback(&s, d.source, d.count);
      }
    }
    expect("canonical DSML headers produce matching proposals", proposed >= 3 && correct);
    expect("proposing never appends unverified tokens", immutable);
  }
  protocol_draft_init(&s);
  int unrelated[64];
  for (int i = 0; i < 64; i++) unrelated[i] = i;
  protocol_draft_push(&s, unrelated, 64);
  expect("history stays bounded", s.count == PROTOCOL_TAIL && s.tail[0] == 32);
  protocol_draft_push(&s, s.rule[0].prefix, s.rule[0].nprefix);
  ContextDraft d = protocol_draft_propose(&s, 1);
  expect("output/context budget truncates the proposal", d.count == 1 && d.tokens[0] == s.rule[0].suffix[0]);
  expect("zero remaining output budget proposes nothing", protocol_draft_propose(&s, 0).count == 0);
  protocol_draft_feedback(&s, 0, 0), protocol_draft_feedback(&s, 0, 0);
  expect("two empty acceptances disable the rule for this request", protocol_draft_propose(&s, 15).count == 0);
  ProtocolDraft other;
  protocol_draft_init(&other);
  expect("another request starts without history or failures", !other.count && !other.rule[0].misses && protocol_draft_propose(&other, 15).count == 0);
  fprintf(stderr, "protocol drafts: %d/%d CPU checks passed\n", checks - failures, checks);
  return failures != 0;
}
