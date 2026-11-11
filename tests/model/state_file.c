// Saved states (src/engine/state_file.c) with a real model of any family. A
// state's parts, read back for a prefix, must hold the rows a state_copy of
// the live state held when they were written, also in a state of another
// capacity, across an append and across a part from an earlier position (a
// rewind). A state read back with a saved snapshot, or with its end written
// straight from the live state, must decode the same logits, byte for byte,
// as the state itself did. A damaged block must be refused. With --unload,
// a state written while the model's weights are unloaded must continue as
// the state does once they are back. The files go to a temporary folder,
// removed before exit; the check prints hashes of the logits it compares.
#include "../../src/engine/state_file.h"

#include <dirent.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define CHECK(x)                                                                                                       \
  do {                                                                                                                 \
    if (!(x))                                                                                                          \
      die("state file check: %s", #x);                                                                                 \
  } while (0)

enum { ROWS = 24, MAX_PROMPT = 4096 };

static char directory[4096];
static const ModelOps *ops;
static int vocab;

static void cleanup(void) {
  if (!*directory) return;
  DIR *d = opendir(directory);
  if (d) {
    for (struct dirent *e; (e = readdir(d));) {
      if (e->d_name[0] == '.') continue;
      char path[8192];
      snprintf(path, sizeof path, "%s/%s", directory, e->d_name);
      unlink(path);
    }
    closedir(d);
  }
  rmdir(directory);
}

static int open_new(const char *name) {
  char path[8192];
  snprintf(path, sizeof path, "%s/%s", directory, name);
  int fd = open(path, O_RDWR | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
  CHECK(fd >= 0);
  return fd;
}

static uint64_t logits_hash(const void *data, size_t bytes) {
  const uint8_t *p = data;
  uint64_t h = 0xcbf29ce484222325ull;
  for (size_t i = 0; i < bytes; i++) h = (h ^ p[i]) * 0x100000001b3ull;
  return h;
}

static int argmax(const float *logits) {
  int best = 0;
  for (int i = 1; i < vocab; i++)
    if (logits[i] > logits[best]) best = i;
  return best;
}

// A state's rows before `positions`, region after region, whatever its
// capacity.
static uint8_t *rows_of(ModelState *s, int positions, size_t *bytes) {
  StateRegion r[STATE_REGIONS_MAX];
  int n = ops->state_regions(s, positions, 0, r);
  *bytes = 0;
  for (int i = 0; i < n; i++) *bytes += (size_t)region_rows(&r[i], positions) * r[i].unit;
  uint8_t *out = xmalloc(*bytes ? *bytes : 1), *p = out;
  for (int i = 0; i < n; i++) {
    size_t len = (size_t)region_rows(&r[i], positions) * r[i].unit;
    memcpy(p, r[i].data, len), p += len;
  }
  return out;
}

static int same_rows(ModelState *a, ModelState *b, int positions) {
  size_t na, nb;
  uint8_t *x = rows_of(a, positions, &na), *y = rows_of(b, positions, &nb);
  int same = na == nb && !memcmp(x, y, na);
  free(x), free(y);
  return same;
}

// A new state of another capacity than a fresh one's, that reads `count`
// parts' rows before `positions`; null when the read fails.
static ModelState *read_rows(const int *fds, int count, int positions) {
  ModelState *s = ops->state_new();
  StateRegion r[STATE_REGIONS_MAX];
  ops->state_regions(s, 3 * MAX_PROMPT / 2, 1, r);  // a larger capacity first
  if (state_file_read(ops, s, positions, fds, count)) {
    ops->state_free(s);
    return NULL;
  }
  return s;
}

// The snapshot in fd, restored into s at pos.
static void restore(ModelState *s, int fd, const int *tokens, int pos, ModelSnapshot **kept) {
  ModelSnapshot *x = snapshot_file_read(ops, fd);
  CHECK(x);
  ops->snapshot_restore(s, x);
  if (ops->seek) ops->seek(s, tokens, pos);
  if (kept) *kept = x;
  else ops->snapshot_free(x);
}

// Decodes tokens[from .. to - 1] at their positions (start + i) and requires
// each row's logits to be expected's, byte for byte; their count.
static int decodes_as(ModelState *s, const int *tokens, int start, int from, int to, const float *expected) {
  int same = 0;
  for (int i = from; i < to; i++) {
    int pos = start + i;
    const float *logits = ops->rows(&s, &tokens[i], &pos, 1);
    CHECK(logits);
    same += !memcmp(logits, expected + (size_t)i * vocab, (size_t)vocab * 4);
  }
  return same;
}

// A 96 x 96 BMP of colored stripes, in memory: an image for the prompt.
static uint8_t *test_image(size_t *len) {
  enum { W = 96, H = 96, ROW = W * 3 };
  *len = 54 + (size_t)ROW * H;
  uint8_t *b = xmalloc(*len);
  uint32_t v[] = {(uint32_t)*len, 0, 54, 40, W, H};
  memcpy(b, "BM", 2), memcpy(b + 2, v, 12), memcpy(b + 14, v + 3, 12);
  uint16_t planes[2] = {1, 24};
  memcpy(b + 26, planes, 4);
  uint32_t rest[6] = {0, (uint32_t)ROW * H, 2835, 2835, 0, 0};
  memcpy(b + 30, rest, 24);
  for (int y = 0; y < H; y++)
    for (int x = 0; x < W; x++) {
      uint8_t *p = b + 54 + (size_t)y * ROW + x * 3;
      p[0] = (uint8_t)(x * 255 / W), p[1] = (uint8_t)(y * 255 / H), p[2] = (uint8_t)((x / 16 + y / 16) % 2 * 200);
    }
  return b;
}

static void flip(int fd, off_t at) {
  uint8_t b;
  CHECK(pread(fd, &b, 1, at) == 1);
  b ^= 0x40;
  CHECK(pwrite(fd, &b, 1, at) == 1);
}

int main(int argc, char **argv) {
  double memory = 32;
  int draft = 0, lookup = 0, unload = 0, image = 0;
  const char *experts = NULL, *draft_model = NULL;
  for (int i = 2; i < argc; i++) {
    if (!strcmp(argv[i], "--memory") && i + 1 < argc) memory = atof(argv[++i]);
    else if (!strcmp(argv[i], "--draft") && i + 1 < argc) draft = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--lookup") && i + 1 < argc) lookup = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--experts") && i + 1 < argc) experts = argv[++i];
    else if (!strcmp(argv[i], "--draft-model") && i + 1 < argc) draft_model = argv[++i];
    else if (!strcmp(argv[i], "--image")) image = 1;
    else if (!strcmp(argv[i], "--unload")) unload = 1;
    else argc = 0;
  }
  if (argc < 2)
    die("usage: state-file-model CHECKPOINT [--memory GIB] [--draft N] [--lookup N] [--experts GGUF] "
        "[--draft-model DIR] [--image] [--unload]  (--image: a test image in the prompt)");
  const char *base = getenv("TMPDIR");
  if (!base || !*base) base = "/tmp";
  snprintf(directory, sizeof directory, "%s/myds4-state-file-model.XXXXXX", base);
  CHECK(mkdtemp(directory));
  atexit(cleanup);

  memory_configure(memory, 10);
  Options o = {.ctx = 8192, .dump_fd = -1, .routes_fd = -1, .quiet = 1, .draft = draft, .lookup = lookup};
  if (experts) o.family = MODEL_DEEPSEEK, o.deepseek.experts = experts;
  if (draft_model) o.family = MODEL_NEMOTRON, o.nemotron.draft_model = draft_model;
  Model *model = model_open(argv[1], &o);
  CHECK(model);
  ops = model->ops;
  vocab = ops->vocab();
  CHECK(ops->state_regions && ops->snapshot_parts && ops->snapshot_alloc);

  // the prompt, with an image when one is given
  Tokenizer *tk = tok_open(argv[1], ops->media_token);
  char text[4096];
  ops->chat(text, sizeof text, "Tell a short story about a lighthouse keeper who finds a letter in a bottle.",
            image ? 1 : 0, 0);
  int *tokens = xmalloc((MAX_PROMPT + ROWS + 1) * sizeof *tokens), n = tok_encode_chat(tk, text, tokens, MAX_PROMPT);
  Media im = {0};
  if (image) {
    size_t len;
    const char *error = "";
    uint8_t *data = test_image(&len);
    if (!ops->prepare_media || !ops->prepare_media(argv[1], &im, data, len, &error))
      die("state file check: the test image: %s", error);
    free(data);
    int *expanded = xmalloc((MAX_PROMPT + ROWS + 1) * sizeof *expanded);
    n = media_expand(tokens, n, &im, 1, expanded, MAX_PROMPT, tok_find(tk, ops->media_token));
    CHECK(n > 0);
    free(tokens), tokens = expanded;
    CHECK(ops->encode_media(&im));
    ops->media(&im, 1);
  }
  CHECK(n > 8 && n + ROWS <= MAX_PROMPT);
  printf("%s: a prompt of %d tokens%s\n", ops->family, n, image ? " with an image" : "");

  // the state, its prompt end saved, and decoding: the logits to match
  ModelState *live = ops->state_new();
  const float *logits = ops->prefill(live, tokens, n, 0, 0, -1, NULL, NULL);
  CHECK(logits);
  ModelSnapshot *prompt_end = ops->snapshot_take(live, n, logits);
  int rows0 = open_new("rows-0"), snap0 = open_new("snap-0"), end0 = open_new("end-0");
  CHECK(state_file_write(ops, live, 0, n, rows0) > 0);
  CHECK(snapshot_file_write(ops, prompt_end, NULL, 0, snap0) > 0);
  CHECK(snapshot_file_write(ops, NULL, live, n, end0) > 0);
  int prefix[4] = {1, n / 2, n - 1, n};
  ModelState *copy[4], *later[4];
  for (int k = 0; k < 4; k++) copy[k] = ops->state_new(), ops->state_copy(copy[k], live, prefix[k]);
  float *expected = xmalloc((size_t)ROWS * vocab * 4);
  int *t = tokens + n;  // the decoded tokens follow the prompt's
  t[0] = argmax(logits);
  ModelSnapshot *at6 = NULL;
  int rows1 = open_new("rows-1"), end1 = open_new("end-1");
  int prefix1[4] = {n, n + 1, n + 6, n + 12};
  for (int i = 0; i < ROWS; i++) {
    if (i == 6) at6 = ops->snapshot_take(live, n + 6, 0);
    if (i == 12) {
      CHECK(state_file_write(ops, live, n, n + 12, rows1) > 0);
      CHECK(snapshot_file_write(ops, NULL, live, n + 12, end1) > 0);
      for (int k = 0; k < 4; k++) later[k] = ops->state_new(), ops->state_copy(later[k], live, prefix1[k]);
    }
    int pos = n + i;
    logits = ops->rows(&live, &t[i], &pos, 1);
    CHECK(logits);
    memcpy(expected + (size_t)i * vocab, logits, (size_t)vocab * 4);
    if (i + 1 <= ROWS) t[i + 1] = argmax(logits);
  }
  printf("decoded %d rows: logits %016llx\n", ROWS,
         (unsigned long long)logits_hash(expected, (size_t)ROWS * vocab * 4));

  // prefixes: the rows a state_copy held when the parts were written
  int same = 0;
  for (int k = 0; k < 4; k++) {
    ModelState *s = read_rows(&rows0, 1, prefix[k]);
    CHECK(s);
    same += same_rows(s, copy[k], prefix[k]);
    ops->state_free(s), ops->state_free(copy[k]);
  }
  printf("one part: %d of 4 prefixes (1, %d, %d, %d) hold state_copy's rows\n", same, n / 2, n - 1, n);
  CHECK(same == 4);
  int chain[3] = {rows0, rows1, -1};
  same = 0;
  for (int k = 0; k < 4; k++) {
    ModelState *s = read_rows(chain, 2, prefix1[k]);
    CHECK(s);
    same += same_rows(s, later[k], prefix1[k]);
    ops->state_free(s), ops->state_free(later[k]);
  }
  printf("an append: %d of 4 prefixes (%d, %d, %d, %d) hold state_copy's rows\n", same, n, n + 1, n + 6, n + 12);
  CHECK(same == 4);

  // continuing from the files: the prompt end's snapshot, the live end, the append
  ModelState *s = read_rows(&rows0, 1, n);
  CHECK(s);
  ModelSnapshot *read_back;
  restore(s, snap0, tokens, n, &read_back);
  const float *saved_logits = ops->snapshot_logits(read_back);
  CHECK(saved_logits && !memcmp(saved_logits, ops->snapshot_logits(prompt_end), (size_t)vocab * 4));
  same = decodes_as(s, t, n, 0, ROWS, expected);
  printf("from the saved prompt end: its logits as taken; %d of %d decoded rows' logits identical\n", same, ROWS);
  CHECK(same == ROWS);
  ops->snapshot_free(read_back), ops->state_free(s);
  s = read_rows(&rows0, 1, n);
  CHECK(s);
  restore(s, end0, tokens, n, NULL);
  same = decodes_as(s, t, n, 0, ROWS, expected);
  printf("from the end written from the live state: %d of %d rows identical\n", same, ROWS);
  CHECK(same == ROWS);
  ops->state_free(s);
  s = read_rows(chain, 2, n + 12);
  CHECK(s);
  restore(s, end1, tokens, n + 12, NULL);
  same = decodes_as(s, t, n, 12, ROWS, expected);
  printf("from an append's end: %d of %d rows identical\n", same, ROWS - 12);
  CHECK(same == ROWS - 12);
  ops->state_free(s);

  // a rewind: back at n + 6 the state takes another token and six more rows,
  // and a part from n + 6 holds them
  ModelState *other = read_rows(chain, 2, n + 6);
  CHECK(other);
  ops->snapshot_restore(other, at6);
  int u[ROWS + 1];
  memcpy(u, t, sizeof u);
  for (int i = 0; i < 6 && u[6] == t[6]; i++) u[6] = t[i];  // another decoded token than the one there
  if (u[6] == t[6]) u[6] = t[6] + 1 < vocab ? t[6] + 1 : t[6] - 1;
  int rows2 = open_new("rows-2"), end2 = open_new("end-2");
  ModelState *other_copy = ops->state_new(), *other_early = ops->state_new();
  float *other_expected = xmalloc((size_t)ROWS * vocab * 4);
  for (int i = 6; i < 16; i++) {
    if (i == 12) {
      CHECK(state_file_write(ops, other, n + 6, n + 12, rows2) > 0);
      CHECK(snapshot_file_write(ops, NULL, other, n + 12, end2) > 0);
      ops->state_copy(other_copy, other, n + 12), ops->state_copy(other_early, other, n + 9);
    }
    int pos = n + i;
    logits = ops->rows(&other, &u[i], &pos, 1);
    CHECK(logits);
    memcpy(other_expected + (size_t)i * vocab, logits, (size_t)vocab * 4);
    u[i + 1] = argmax(logits);
  }
  chain[2] = rows2;
  s = read_rows(chain, 3, n + 12);
  CHECK(s && same_rows(s, other_copy, n + 12));
  ModelState *early = read_rows(chain, 3, n + 9);
  CHECK(early && same_rows(early, other_early, n + 9));
  int *rewound = xmalloc((size_t)(n + ROWS + 1) * sizeof *rewound);  // the prompt, then the other tokens
  memcpy(rewound, tokens, (size_t)n * sizeof *rewound), memcpy(rewound + n, u, sizeof u);
  restore(s, end2, rewound, n + 12, NULL);
  free(rewound);
  same = decodes_as(s, u, n, 12, 16, other_expected);
  printf("a part from an earlier position: rows as state_copy's at %d and %d; %d of 4 rows identical\n", n + 12,
         n + 9, same);
  CHECK(same == 4);
  ops->state_free(s), ops->state_free(early), ops->state_free(other_copy), ops->state_free(other_early);

  // a damaged block is refused
  flip(rows0, 16384 + 8);
  CHECK(!read_rows(&rows0, 1, n));
  flip(rows0, 16384 + 8);
  s = read_rows(&rows0, 1, n);
  CHECK(s);
  ops->state_free(s);
  puts("a damaged block is refused");

  if (unload) {
    int length = n + ROWS, rows9 = open_new("rows-9"), end9 = open_new("end-9");
    uint64_t freed = memory_unload(&model->memory);
    printf("unloaded %.2f GiB\n", freed / 1073741824.0);
    CHECK(state_file_write(ops, live, 0, length, rows9) > 0);
    CHECK(snapshot_file_write(ops, NULL, live, length, end9) > 0);
    CHECK(model_load(model, &o));
    s = read_rows(&rows9, 1, length);
    CHECK(s);
    restore(s, end9, tokens, length, NULL);
    same = 0;
    int w = t[ROWS];  // the next token, after the decoded rows
    for (int i = 0; i < 4; i++) {
      int pos = length + i;
      logits = ops->rows(&live, &w, &pos, 1);
      CHECK(logits);
      float *want = xmalloc((size_t)vocab * 4);
      memcpy(want, logits, (size_t)vocab * 4);
      logits = ops->rows(&s, &w, &pos, 1);
      same += !memcmp(logits, want, (size_t)vocab * 4);
      w = argmax(want);
      free(want);
    }
    printf("written while unloaded: %d of 4 rows identical after the reload\n", same);
    CHECK(same == 4);
    ops->state_free(s);
    close(rows9), close(end9);
  }
  close(rows0), close(snap0), close(end0), close(rows1), close(end1), close(rows2), close(end2);
  ops->snapshot_free(prompt_end), ops->snapshot_free(at6);
  ops->state_free(other), ops->state_free(live);
  free(expected), free(other_expected), free(tokens);
  puts("state file checks passed");
  return 0;
}
