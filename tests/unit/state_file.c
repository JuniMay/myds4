// Saved states' files (src/engine/state_file.c) without a model: a fake
// family keeps its rows and snapshot parts in plain memory. A saved state's
// parts must give back, for every prefix, the rows a state_copy of the live
// state gives, also across appends and a part that starts earlier (a
// rewind); damaged, short or differently shaped files must be refused; a
// snapshot, and a live state's end, must come back byte for byte; a second
// stop request must end a write; and a checkpoint's identity must follow its
// weights and nothing else. Its temporary folder is removed before exit.
#include "../../src/engine/state_file.h"

#include <dirent.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>

static int checks, failures;
static char directory[4096];

static void expect(const char *name, int ok) {
  checks++;
  if (!ok) failures++, fprintf(stderr, "FAIL: %s\n", name);
}

static void remove_tree(const char *path) {
  DIR *d = opendir(path);
  if (d) {
    for (struct dirent *e; (e = readdir(d));) {
      if (!strcmp(e->d_name, ".") || !strcmp(e->d_name, "..")) continue;
      char sub[8192];
      snprintf(sub, sizeof sub, "%s/%s", path, e->d_name);
      struct stat st;
      if (!lstat(sub, &st) && S_ISDIR(st.st_mode)) remove_tree(sub);
      else unlink(sub);
    }
    closedir(d);
  }
  rmdir(path);
}

static void cleanup(void) {
  if (*directory) remove_tree(directory);
}

// ---- a fake family ----

// Four regions like the real ones: plain rows (24 bytes), compressed pairs
// with a guard row (1 KiB, more than one 4 MiB block at these lengths), one
// index row per four positions (512 bytes), and rows a position late (64
// bytes; Qwen's MTP keys).
enum { REGIONS = 4, ROLL = 100000, VOCAB = 250000 };
static const uint32_t UNIT[REGIONS] = {24, 1024, 512, 64}, RATIO[REGIONS] = {1, 2, 4, 1}, GUARD[REGIONS] = {0, 1, 0, 1},
                      LAG[REGIONS] = {0, 0, 0, 1};
static uint32_t unit_of[REGIONS] = {24, 1024, 512, 64};  // the shape a "build" lists; one test changes it

// Region i's shape, its rows nowhere.
static StateRegion shape(int i) { return (StateRegion){NULL, UNIT[i], RATIO[i], GUARD[i], LAG[i]}; }

typedef struct {
  uint8_t *rows[REGIONS];
  int cap, pos, generation;
  uint8_t roll[ROLL];
} Fake;

typedef struct {
  int pos;
  uint8_t roll[ROLL];
  float *logits;
} FakeSnap;

static size_t region_bytes(int i, int cap) { return (size_t)(cap / RATIO[i] + GUARD[i] + 1) * UNIT[i]; }

static void reserve(Fake *s, int positions) {
  if (positions < 1) positions = 1;  // a new state has room, as the real ones do
  if (positions <= s->cap) return;
  for (int i = 0; i < REGIONS; i++) {
    size_t old = s->cap ? region_bytes(i, s->cap) : 0, now = region_bytes(i, positions);
    s->rows[i] = realloc(s->rows[i], now);
    memset(s->rows[i] + old, 0xEE, now - old);
  }
  s->cap = positions;
}

static int fake_regions(ModelState *state, int positions, int fill, StateRegion *out) {
  Fake *s = (Fake *)state;
  if (fill) reserve(s, positions), s->pos = positions;
  for (int i = 0; i < REGIONS; i++) out[i] = (StateRegion){s->rows[i], unit_of[i], RATIO[i], GUARD[i], LAG[i]};
  return REGIONS;
}

static int fake_parts(ModelSnapshot *snap, ModelState *state, int pos, SnapshotPart *out, int32_t *meta, int *nmeta) {
  FakeSnap *x = (FakeSnap *)snap;
  Fake *s = (Fake *)state;
  meta[0] = x ? x->pos : pos, *nmeta = 1;
  int n = 0;
  out[n++] = (SnapshotPart){x ? x->roll : s->roll, ROLL};
  if (x && x->logits) out[n++] = (SnapshotPart){x->logits, VOCAB * 4};
  return n;
}

static ModelSnapshot *fake_alloc(const int32_t *meta, int nmeta, int with_logits) {
  if (nmeta != 1) return NULL;
  FakeSnap *x = calloc(1, sizeof *x);
  x->pos = meta[0];
  if (with_logits) x->logits = calloc(VOCAB, 4);
  return (ModelSnapshot *)x;
}

static void fake_free(ModelSnapshot *snap) {
  FakeSnap *x = (FakeSnap *)snap;
  if (x) free(x->logits), free(x);
}

static const float *fake_logits(const ModelSnapshot *snap) { return ((const FakeSnap *)snap)->logits; }

static const ModelOps fake = {.family = "fake",
                              .snapshot_free = fake_free,
                              .snapshot_logits = fake_logits,
                              .state_version = 1,
                              .state_regions = fake_regions,
                              .snapshot_parts = fake_parts,
                              .snapshot_alloc = fake_alloc};

static uint8_t byte_of(int region, int64_t row, int generation, size_t k) {
  uint64_t x = (uint64_t)region * 0x9e3779b97f4a7c15ull ^ (uint64_t)row * 0xbf58476d1ce4e5b9ull ^
               (uint64_t)generation * 0x94d049bb133111ebull ^ k * 0x2545f4914f6cdd1dull;
  x ^= x >> 31, x *= 0xd6e8feb86659fd93ull, x ^= x >> 32;
  return (uint8_t)x;
}

// The live state runs positions from .. to - 1: every row from those that
// `from` positions leave unchanged is written again, as a prefill that
// continues there writes it (a guard row, and rows after a rewind).
static void run(Fake *s, int from, int to) {
  reserve(s, to);
  s->generation++;
  for (int i = 0; i < REGIONS; i++) {
    StateRegion r0 = shape(i);
    for (int64_t r = region_stable(&r0, from); r < region_rows(&r0, to); r++)
      for (size_t k = 0; k < UNIT[i]; k++) s->rows[i][r * UNIT[i] + k] = byte_of(i, r, s->generation, k);
  }
  for (int k = 0; k < ROLL; k++) s->roll[k] = byte_of(9, to, s->generation, (size_t)k);
  s->pos = to;
}

// Whether d holds s's rows before `positions`, as a state_copy gives them.
static int same_rows(const Fake *d, const Fake *s, int positions) {
  for (int i = 0; i < REGIONS; i++) {
    StateRegion r0 = shape(i);
    if (memcmp(d->rows[i], s->rows[i], (size_t)region_rows(&r0, positions) * UNIT[i])) return 0;
  }
  return 1;
}

static void fake_release(Fake *s) {
  for (int i = 0; i < REGIONS; i++) free(s->rows[i]);
  free(s);
}

static int open_new(const char *name) {
  char path[8192];
  snprintf(path, sizeof path, "%s/%s", directory, name);
  return open(path, O_RDWR | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
}

// Reads the first `count` of the files' rows prefix into a new state and
// compares it with s's.
static int prefix_ok(const int *fds, int count, int positions, const Fake *s) {
  Fake *d = calloc(1, sizeof *d);
  int ok = !state_file_read(&fake, (ModelState *)d, positions, fds, count) && d->pos == positions &&
           same_rows(d, s, positions);
  fake_release(d);
  return ok;
}

// Whether a prefix reads at all (its rows not compared).
static int reads(const int *fds, int count, int positions) {
  Fake *d = calloc(1, sizeof *d);
  int ok = !state_file_read(&fake, (ModelState *)d, positions, fds, count);
  fake_release(d);
  return ok;
}

static void flip(int fd, off_t at) {
  uint8_t b;
  if (pread(fd, &b, 1, at) == 1) {
    b ^= 0x40;
    if (pwrite(fd, &b, 1, at) != 1) perror("pwrite");
  }
}

// ---- identity ----

static void write_file(const char *name, const char *text, size_t size) {
  char path[8192];
  snprintf(path, sizeof path, "%s/%s", directory, name);
  FILE *f = fopen(path, "wb");
  if (!f) {
    perror(path);
    exit(1);
  }
  for (size_t i = 0; i < size; i++) fputc(text[i % strlen(text)], f);
  fclose(f);
}

static void folder_identity(const char *folder, uint8_t out[32]) {
  char path[8192];
  snprintf(path, sizeof path, "%s/%s", directory, folder);
  CC_SHA256_CTX h;
  CC_SHA256_Init(&h);
  identity_folder(&h, path);
  CC_SHA256_Final(out, &h);
}

static void set_identity(const char *first, uint8_t out[32]) {
  char path[8192];
  snprintf(path, sizeof path, "%s/%s", directory, first);
  CC_SHA256_CTX h;
  CC_SHA256_Init(&h);
  identity_gguf_set(&h, path);
  CC_SHA256_Final(out, &h);
}

static void identity_checks(void) {
  char path[8192];
  snprintf(path, sizeof path, "%s/model", directory), mkdir(path, 0700);
  snprintf(path, sizeof path, "%s/model/sub", directory), mkdir(path, 0700);
  snprintf(path, sizeof path, "%s/model/.cache", directory), mkdir(path, 0700);
  write_file("model/config.json", "{\"a\": 1}", 8);
  write_file("model/a.safetensors", "weights-a", 300000);
  write_file("model/sub/b.safetensors", "weights-b", 1000);
  write_file("model/README.md", "read me", 7);
  write_file("model/.cache/c.safetensors", "hidden", 6);
  uint8_t base[32], again[32];
  folder_identity("model", base);
  write_file("model/README.md", "read me again", 13);
  write_file("model/.cache/c.safetensors", "hidden, changed", 15);
  write_file("model/.DS_Store", "finder", 6);
  folder_identity("model", again);
  expect("documents and dot files leave the identity as it is", !memcmp(base, again, 32));
  write_file("model/a.safetensors", "weights-A", 300000);  // the first 64 KiB differ
  folder_identity("model", again);
  expect("a weights file's first bytes change the identity", memcmp(base, again, 32));
  write_file("model/a.safetensors", "weights-a", 300000);
  folder_identity("model", again);
  expect("the same weights give the same identity again", !memcmp(base, again, 32));
  write_file("model/sub/b.safetensors", "weights-b", 1001);  // its size
  folder_identity("model", again);
  expect("a weights file's size changes the identity", memcmp(base, again, 32));
  write_file("model/sub/b.safetensors", "weights-b", 1000);
  write_file("model/config.json", "{\"a\": 2}", 8);
  folder_identity("model", again);
  expect("config.json changes the identity", memcmp(base, again, 32));
  write_file("x-00001-of-00002.gguf", "first", 5000);
  write_file("x-00002-of-00002.gguf", "second", 5000);
  set_identity("x-00001-of-00002.gguf", base);
  write_file("x-00002-of-00002.gguf", "SECOND", 5000);
  set_identity("x-00001-of-00002.gguf", again);
  expect("every file of a GGUF set counts", memcmp(base, again, 32));
}

int main(void) {
  alarm(120);
  const char *base = getenv("TMPDIR");
  if (!base || !*base) base = "/tmp";
  if (snprintf(directory, sizeof directory, "%s/myds4-state-file-check.XXXXXX", base) >= (int)sizeof directory ||
      !mkdtemp(directory)) {
    perror("mkdtemp");
    return 1;
  }
  atexit(cleanup);

  // one part, every prefix
  Fake *s = calloc(1, sizeof *s);
  run(s, 0, 9000);
  int fds[3] = {open_new("rows-0"), open_new("rows-1"), open_new("rows-2")};
  int64_t written = state_file_write(&fake, (ModelState *)s, 0, 9000, fds[0]);
  expect("a part is written in whole 16 KiB spans", written > 0 && written % 16384 == 0 && lseek(fds[0], 0, SEEK_END) == written);
  int prefixes[] = {0, 1, 2, 3, 4, 5, 4095, 4096, 8191, 8999, 9000};
  int all = 1;
  for (size_t k = 0; k < sizeof prefixes / sizeof *prefixes; k++) all &= prefix_ok(fds, 1, prefixes[k], s);
  expect("every prefix of one part reads back as state_copy gives it", all);
  expect("a prefix past the part is refused", !prefix_ok(fds, 1, 9001, s));

  // an append from where the part ended: its guard rows come from the append
  run(s, 9000, 12001);
  state_file_write(&fake, (ModelState *)s, 9000, 12001, fds[1]);
  int appended[] = {0, 5000, 8999, 9000, 9001, 9002, 11999, 12000, 12001};
  all = 1;
  for (size_t k = 0; k < sizeof appended / sizeof *appended; k++) all &= prefix_ok(fds, 2, appended[k], s);
  expect("an append gives every prefix of the state that continued", all);

  // a rewind: the state continues from 10,000 again, and a part from there
  run(s, 10000, 11003);
  state_file_write(&fake, (ModelState *)s, 10000, 11003, fds[2]);
  int rewound[] = {0, 8999, 9500, 9999, 10000, 10001, 11002, 11003};
  all = 1;
  for (size_t k = 0; k < sizeof rewound / sizeof *rewound; k++) all &= prefix_ok(fds, 3, rewound[k], s);
  expect("a part from an earlier position replaces the rows after it", all);
  expect("a prefix past the latest part is refused", !prefix_ok(fds, 3, 11004, s));
  int reordered[3] = {fds[1], fds[0], fds[2]};
  expect("parts out of order are refused", !prefix_ok(reordered, 3, 100, s));

  // damage: a block the prefix needs, and one it does not. The first part's
  // compressed pairs (4,501 rows of 1 KiB) fill two blocks; the second holds
  // the rows from 4,096 on, those of positions 8,192 and later.
  off_t region1 = 0;
  {
    uint8_t h[16384];
    if (pread(fds[0], h, sizeof h, 0) == (ssize_t)sizeof h) {
      uint64_t at;
      memcpy(&at, h + 40 + 48 * 1 + 40, 8);  // region 1's offset
      region1 = (off_t)at;
    }
  }
  expect("the first part reads before any damage", reads(fds, 1, 9000));
  flip(fds[0], region1 + (4 << 20) + 1000);
  expect("a prefix that needs a damaged block is refused", !reads(fds, 1, 9000));
  expect("a prefix before the damaged block still reads", prefix_ok(fds, 1, 4000, s));
  flip(fds[0], region1 + (4 << 20) + 1000);
  flip(fds[0], 100);  // the header
  expect("a damaged header is refused", !reads(fds, 1, 10));
  flip(fds[0], 100);
  expect("the part reads again once undamaged", prefix_ok(fds, 1, 8999, s));
  if (ftruncate(fds[1], lseek(fds[1], 0, SEEK_END) - 16384)) perror("ftruncate");
  expect("a short part is refused", !reads(fds, 2, 12001));
  unit_of[2] = 256;  // a build that lists another shape
  expect("a part of another shape is refused", !reads(fds, 1, 100));
  unit_of[2] = UNIT[2];

  // snapshots: one with logits, and a live state's end
  FakeSnap *x = (FakeSnap *)fake_alloc((int32_t[]){777}, 1, 1);
  for (int k = 0; k < ROLL; k++) x->roll[k] = byte_of(7, k, 1, 0);
  for (int k = 0; k < VOCAB; k++) x->logits[k] = (float)k * 0.5f;
  int snap = open_new("snap"), end = open_new("end");
  expect("a snapshot is written", snapshot_file_write(&fake, (ModelSnapshot *)x, NULL, 0, snap) > 0);
  FakeSnap *y = (FakeSnap *)snapshot_file_read(&fake, snap);
  expect("a snapshot reads back with its position, rows and logits",
         y && y->pos == 777 && !memcmp(y->roll, x->roll, ROLL) && y->logits && !memcmp(y->logits, x->logits, VOCAB * 4));
  fake_free((ModelSnapshot *)y);
  expect("a live state's end is written", snapshot_file_write(&fake, NULL, (ModelState *)s, s->pos, end) > 0);
  y = (FakeSnap *)snapshot_file_read(&fake, end);
  expect("a live state's end reads back as a snapshot without logits",
         y && y->pos == s->pos && !memcmp(y->roll, s->roll, ROLL) && !y->logits);
  fake_free((ModelSnapshot *)y);
  flip(snap, 16384 + 4096);
  expect("a damaged snapshot is refused", !snapshot_file_read(&fake, snap));

  identity_checks();

  // a second stop request ends a write
  shutdown_request(), shutdown_request();
  int last = open_new("rows-3");
  expect("a second stop request ends a write", state_file_write(&fake, (ModelState *)s, 0, 9000, last) < 0);
  close(last), close(snap), close(end);
  for (int i = 0; i < 3; i++) close(fds[i]);
  fake_free((ModelSnapshot *)x);
  fake_release(s);
  printf("state file checks: %d, failures: %d\n", checks, failures);
  return failures != 0;
}
