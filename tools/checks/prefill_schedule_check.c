// One submitted prompt should use one forward call while retaining its prefix.
// The engine runs against CPU-only model spies; no weights or GPU are used.
#include "myds4.h"
#include <stdlib.h>
#include <string.h>

static int forwards, saved_mask, started_events;
static struct { int n, pos; } forward_calls[8];
static float fixture_logits[VOCAB];
static ModelForwardInfo spy_info;
static const float *spy_forward(const int *tokens, int n, int pos) {
  if (forwards < 8) forward_calls[forwards] = (typeof(*forward_calls)){n, pos};
  forwards++;
  spy_info = (ModelForwardInfo){.tokens = n, .chunks = 1, .decoder_tokens = n};
  return fixture_logits;
}
static State *spy_new(void) { return (State *)(uintptr_t)1; }
static void spy_use(State *s) {}
static void spy_free(State *s) {}
static void spy_copy(State *to, const State *from) {}
static size_t spy_bytes(const State *s, int positions) { return 8192; }
static void spy_reset(void) {}
static void spy_save(int slot) { saved_mask |= 1 << slot; }
static void spy_restore(int slot) {}
static void spy_seek(const int *tokens, int pos) {}
static void spy_images(const Image *images, int count) {}
static const float *spy_prefix(const int *tokens, int n, int pos, int prefix, int *encoder_only) {
  spy_save(1);
  if (encoder_only) *encoder_only = 0;
  const float *lg = spy_forward(tokens, n, pos);
  spy_info.prefix_tokens = prefix;
  return lg;
}
static ModelForwardInfo spy_forward_info(void) { return spy_info; }

#define model_forward spy_forward
#define model_forward_prefix spy_prefix
#define model_forward_info spy_forward_info
#define model_state_new spy_new
#define model_state_use spy_use
#define model_state_free spy_free
#define model_state_copy spy_copy
#define model_state_bytes spy_bytes
#define model_reset spy_reset
#define model_state_save spy_save
#define model_state_restore spy_restore
#define model_seek spy_seek
#define model_images spy_images
#include "../../src/engine/engine.c"

static void fixture_event(void *context, const myds4_event *e) { started_events += e->kind == MYDS4_STARTED; }

int main(void) {
  const int prompt[] = {1, 2, 3, 4, 42, 6, 42, 8, 42, 10, 11};
  opt.ctx = 512, cfg.kv_gb = 1, kv_limit = GiB, user = 42, think_start = 99;
  fixture_logits[0] = 1;
  Req r = {.id = 1, .n = sizeof prompt / sizeof *prompt, .sp = {.temp = 0, .top_p = 1},
           .seed_set = 1, .max_new = 1, .on_event = fixture_event, .queued = now()};
  r.p = xmalloc(sizeof prompt);
  memcpy(r.p, prompt, sizeof prompt);
  Gen g;
  int accepted = begin(&r, &g);
  int one_prompt = forwards == 1 && forward_calls[0].n == r.n && forward_calls[0].pos == 0;
  int retained = g.s->base == 4 && (saved_mask & 2) && (saved_mask & 1);
  printf("prefill scheduling: accepted=%d, forward_calls=%d (expected 1), prefix_retained=%d, started=%d\n",
         accepted, forwards, retained, started_events);
  for (int i = 0; i < forwards && i < 8; i++)
    printf("  forward %d: position=%d tokens=%d\n", i + 1, forward_calls[i].pos, forward_calls[i].n);
  // A short reply can still fit the rewind margin. A retry must restore
  // saved logits, while a changed last prompt token must run again.
  g.s->busy = 0;
  for (int i = 0; i < 4; i++) g.s->ids[g.s->n++] = 7;
  Req retry = {.id = 2, .n = r.n, .sp = r.sp, .seed_set = 1, .max_new = 1,
               .on_event = fixture_event, .queued = now()};
  retry.p = xmalloc(sizeof prompt), memcpy(retry.p, prompt, sizeof prompt);
  Gen retried;
  int before = forwards;
  begin(&retry, &retried);
  int retry_ok = forwards == before && retried.from == r.n;
  printf("short-reply retry: forward_calls=%d expected=0\n", forwards - before);
  retried.s->busy = 0;
  Req changed = {.id = 3, .n = r.n, .sp = r.sp, .seed_set = 1, .max_new = 1,
                 .on_event = fixture_event, .queued = now()};
  changed.p = xmalloc(sizeof prompt), memcpy(changed.p, prompt, sizeof prompt);
  changed.p[changed.n - 1]++;
  Gen rerun;
  before = forwards;
  begin(&changed, &rerun);
  int changed_ok = forwards == before + 1 && rerun.from == r.n - 1;
  printf("changed last token: forward_calls=%d expected=1\n", forwards - before);
  free(retried.p), free(rerun.p);
  int from;
  const char *how;
  pick(prompt, 4, &from, &how);
  int short_prompt = from < 4;
  printf("prefix-only request: reused=%d (must leave a token for logits)\n", from);
  int fork_tokens[256] = {1, 2, 3, 4, 77};
  opt.ced = 128;
  g.s->base_encoder_only = 1;
  pick(fork_tokens, 4 + 128 + BLOCK, &from, &how);
  int guarded = from == 0;
  pick(fork_tokens, 4 + 128 + BLOCK + 1, &from, &how);
  guarded &= from == 4 && !strcmp(how, "forked");
  printf("encoder-only prefix: short suffix skipped, full replay forked=%d\n", guarded);
  free(g.p);
  for (int i = 0; i < SESSIONS; i++) free(sessions[i].ids), free(sessions[i].logits);
  return !(accepted && one_prompt && retained && short_prompt && guarded && retry_ok && changed_ok && started_events == 3);
}
