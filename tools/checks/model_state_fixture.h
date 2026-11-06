// Numerical fixtures copy model buffers directly, without the server's
// snapshot API. Include after model.c so both older and newer APIs work.
typedef struct { size_t size; void *data; } FixtureBuffer;
typedef struct {
  void *local;
  FixtureBuffer global[LAYERS][2];
} FixtureState;

static void fixture_rolling(void *data, int restore) {
  char *p = data;
#define FIXTURE_ROWS(b) do { \
    if (restore) memcpy((b).ptr, p, (b).size); else memcpy(p, (b).ptr, (b).size); \
    p += (b).size; \
  } while (0)
  for (int l = 0; l < LAYERS; l++) {
    FIXTURE_ROWS(L[l].ring);
    if (L[l].kv_source) FIXTURE_ROWS(L[l].cstate);
  }
  for (int d = 0; opt.draft && d < DRAFTS; d++) FIXTURE_ROWS(D[d].ring);
#undef FIXTURE_ROWS
  if (restore) memcpy(history, p, sizeof history);
  else memcpy(p, history, sizeof history);
}

static FixtureState *fixture_capture(void) {
  gpu_sync();
  FixtureState *s = calloc(1, sizeof *s);
  if (!s) die("cannot allocate fixture state");
  s->local = xmalloc(saved_bytes());
  fixture_rolling(s->local, 0);
  for (int l = 0; l < LAYERS; l++) if (L[l].kv_source)
    for (int k = 0; k < 2; k++) {
      Buf *b = k ? &L[l].keys : &L[l].comp;
      FixtureBuffer *copy = &s->global[l][k];
      copy->size = b->size, copy->data = xmalloc(b->size);
      memcpy(copy->data, b->ptr, b->size);
    }
  return s;
}

static void fixture_restore(const FixtureState *s) {
  gpu_sync();
  fixture_rolling(s->local, 1);
  for (int l = 0; l < LAYERS; l++) if (L[l].kv_source)
    for (int k = 0; k < 2; k++) {
      Buf *b = k ? &L[l].keys : &L[l].comp;
      const FixtureBuffer *copy = &s->global[l][k];
      if (copy->size != b->size) die("fixture state capacity changed");
      memcpy(b->ptr, copy->data, b->size);
    }
}

static void fixture_free(FixtureState *s) {
  for (int l = 0; l < LAYERS; l++) for (int k = 0; k < 2; k++) free(s->global[l][k].data);
  free(s->local), free(s);
}
