// Saved states (state_file.h): the layout of a state's parts and of its
// snapshots in files, and the checks of what is read back.
//
// A part file is a header, then each region's rows from a 16 KiB-aligned
// offset. The header holds the part's positions (from, to), each region's
// shape (row bytes, positions per row, guard row), first row, row count and
// offset, and a CRC32C for each 4 MiB block of each region's rows. A
// snapshot file is a header (its integers, whether it holds logits, a
// CRC32C for each 4 MiB block), then its parts' bytes as one stream. Every
// header ends in its own CRC32C. Numbers are little-endian.
//
// Uncached reads and writes go past the page cache only in whole 16 KiB-
// aligned spans to and from page-aligned memory (weights.c). So each block
// moves through an aligned buffer of the worker that writes or reads it, and
// files are padded to whole spans.
#include "state_file.h"

#include <arm_acle.h>
#include <dirent.h>
#include <dispatch/dispatch.h>
#include <errno.h>
#include <fcntl.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include "myds4.h"

_Static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__, "saved states are little-endian");

enum {
  ALIGN = 16384,        // file offsets and spans
  BLOCK = 4 << 20,      // a checksummed block, also the unit a worker moves
  WORKERS = 4,          // blocks written or read at once
  VERSION = 1,          // of both file layouts
  PARTS_MAX = 4096,     // parts a saved state may have
  HEADER_MAX = 64 << 20 // the largest header read
};
static const char ROWS_MAGIC[8] = {'M', 'Y', 'D', 'S', '4', 'K', 'V', 'R'};
static const char SNAP_MAGIC[8] = {'M', 'Y', 'D', 'S', '4', 'K', 'V', 'S'};

uint32_t crc32c(uint32_t crc, const void *data, size_t n) {
  const uint8_t *p = data;
  crc = ~crc;
  for (; n >= 8; p += 8, n -= 8) {
    uint64_t v;
    memcpy(&v, p, 8);
    crc = __crc32cd(crc, v);
  }
  while (n--) crc = __crc32cb(crc, *p++);
  return ~crc;
}

static size_t aligned(uint64_t n) { return (size_t)((n + ALIGN - 1) / ALIGN * ALIGN); }

static int write_all(int fd, const void *p, size_t n, off_t at) {
  for (size_t done = 0; done < n;) {
    ssize_t w = pwrite(fd, (const uint8_t *)p + done, n - done, at + (off_t)done);
    if (w < 0 && errno == EINTR) continue;
    if (w <= 0) return -1;
    done += (size_t)w;
  }
  return 0;
}

static int read_all(int fd, void *p, size_t n, off_t at) {
  for (size_t done = 0; done < n;) {
    ssize_t r = pread(fd, (uint8_t *)p + done, n - done, at + (off_t)done);
    if (r < 0 && errno == EINTR) continue;
    if (r <= 0) return -1;
    done += (size_t)r;
  }
  return 0;
}

static void put32(uint8_t *p, uint32_t v) { memcpy(p, &v, 4); }
static void put64(uint8_t *p, uint64_t v) { memcpy(p, &v, 8); }
static uint32_t get32(const uint8_t *p) {
  uint32_t v;
  memcpy(&v, p, 4);
  return v;
}
static uint64_t get64(const uint8_t *p) {
  uint64_t v;
  memcpy(&v, p, 8);
  return v;
}

// ---- blocks, written and read by workers ----

// One block: its file offset and bytes (before padding). Rows move between
// the file and memory at mem; a snapshot's stream (mem null) between the file
// and the job's parts, from stream on. A read copies only the block's bytes
// lo .. hi into memory, after checking all of them.
typedef struct {
  int fd;
  off_t at;
  uint32_t len, crc, lo, hi;
  uint8_t *mem;
  uint64_t stream;
} Block;

typedef struct {
  Block *blocks;
  int n, writing;
  const SnapshotPart *parts;
  int nparts;
  atomic_int failed;
} Job;

// The stream's bytes [at, at + len) from the parts, or (put) into them.
static void stream_copy(const SnapshotPart *parts, int n, uint64_t at, uint8_t *buf, size_t len, int put) {
  uint64_t base = 0;
  for (int i = 0; i < n && len; base += parts[i].bytes, i++) {
    if (at >= base + parts[i].bytes) continue;
    size_t off = (size_t)(at - base), take = parts[i].bytes - off < len ? parts[i].bytes - off : len;
    if (put) memcpy((uint8_t *)parts[i].data + off, buf, take);
    else memcpy(buf, (const uint8_t *)parts[i].data + off, take);
    buf += take, at += take, len -= take;
  }
}

static void worker(void *context, size_t w) {
  Job *j = context;
  uint8_t *buf = NULL;
  if (posix_memalign((void **)&buf, ALIGN, BLOCK)) {
    atomic_store(&j->failed, 1);
    return;
  }
  for (int i = (int)w; i < j->n && !atomic_load(&j->failed); i += WORKERS) {
    Block *b = &j->blocks[i];
    size_t span = aligned(b->len);
    if (j->writing) {
      if (shutdown_requests() >= 2) {  // a second stop request ends saving at quit
        atomic_store(&j->failed, 1);
        break;
      }
      if (b->mem) memcpy(buf, b->mem, b->len);
      else stream_copy(j->parts, j->nparts, b->stream, buf, b->len, 0);
      memset(buf + b->len, 0, span - b->len);
      b->crc = crc32c(0, buf, b->len);
      if (write_all(b->fd, buf, span, b->at)) atomic_store(&j->failed, 1);
    } else {
      if (read_all(b->fd, buf, span, b->at) || crc32c(0, buf, b->len) != b->crc) {
        atomic_store(&j->failed, 1);
        break;
      }
      if (b->mem) memcpy(b->mem + b->lo, buf + b->lo, b->hi - b->lo);
      else stream_copy(j->parts, j->nparts, b->stream, buf, b->len, 1);
    }
  }
  free(buf);
}

static int run(Job *j) {
  if (j->n) dispatch_apply_f(WORKERS, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), j, worker);
  return atomic_load(&j->failed) ? -1 : 0;
}

// A header of `bytes` (aligned) with its CRC32C at crc_at, written at 0.
static int write_header(int fd, uint8_t *h, size_t bytes, size_t crc_at) {
  put32(h + crc_at, 0);
  put32(h + crc_at, crc32c(0, h, bytes));
  return write_all(fd, h, bytes, 0);
}

// A file's header: its first span, then all of it (its size at size_at);
// null when it is short, too large, of another kind or damaged.
static uint8_t *read_header(int fd, const char magic[8], size_t size_at, size_t crc_at, size_t *bytes) {
  uint8_t *h = NULL;
  if (posix_memalign((void **)&h, ALIGN, ALIGN)) return NULL;
  if (read_all(fd, h, ALIGN, 0) || memcmp(h, magic, 8) || get32(h + 8) != VERSION) goto bad;
  *bytes = get32(h + size_at);
  if (*bytes < ALIGN || *bytes > HEADER_MAX || *bytes % ALIGN) goto bad;
  if (*bytes > ALIGN) {
    uint8_t *all = NULL;
    if (posix_memalign((void **)&all, ALIGN, *bytes)) goto bad;
    free(h), h = all;
    if (read_all(fd, h, *bytes, 0)) goto bad;
  }
  uint32_t crc = get32(h + crc_at);
  put32(h + crc_at, 0);
  if (crc32c(0, h, *bytes) != crc) goto bad;
  return h;
bad:
  free(h);
  return NULL;
}

// ---- a state's parts ----

// The part header: magic, version, header bytes, from, to, regions, blocks,
// its CRC32C, then for each region its row bytes, positions per row, guard
// rows, lag, first block, a reserved word, first row, row count and offset
// (48 bytes), then each block's CRC32C.
enum { ROWS_FIXED = 40, ROWS_REGION = 48, ROWS_CRC_AT = 32 };

int64_t state_file_write(const ModelOps *ops, ModelState *s, int from, int to, int fd) {
  if (!ops->state_regions || from < 0 || to < from) return -1;
  StateRegion r[STATE_REGIONS_MAX];
  int n = ops->state_regions(s, to, 0, r);
  int64_t first[STATE_REGIONS_MAX], rows[STATE_REGIONS_MAX];
  uint32_t first_block[STATE_REGIONS_MAX];
  uint64_t blocks = 0;
  for (int i = 0; i < n; i++) {
    first[i] = region_stable(&r[i], from), rows[i] = region_rows(&r[i], to) - first[i];
    first_block[i] = (uint32_t)blocks;
    blocks += ((uint64_t)rows[i] * r[i].unit + BLOCK - 1) / BLOCK;
  }
  size_t header = aligned(ROWS_FIXED + (uint64_t)ROWS_REGION * n + 4 * blocks);
  if (header > HEADER_MAX) return -1;
  if (fd < 0) {  // the size alone
    uint64_t size = header;
    for (int i = 0; i < n; i++) size += aligned((uint64_t)rows[i] * r[i].unit);
    return (int64_t)size;
  }
  uint8_t *h = NULL;
  Block *b = xmalloc((blocks ? blocks : 1) * sizeof *b);
  if (posix_memalign((void **)&h, ALIGN, header)) die("out of memory for a saved state's header");
  memset(h, 0, header);
  uint64_t at = header;
  int k = 0;
  for (int i = 0; i < n; i++) {
    uint64_t bytes = (uint64_t)rows[i] * r[i].unit;
    uint8_t *row = (uint8_t *)r[i].data + (size_t)first[i] * r[i].unit;
    uint8_t *e = h + ROWS_FIXED + (size_t)ROWS_REGION * i;
    put32(e, r[i].unit), put32(e + 4, r[i].ratio), put32(e + 8, r[i].guard), put32(e + 12, r[i].lag);
    put32(e + 16, first_block[i]);
    put64(e + 24, (uint64_t)first[i]), put64(e + 32, (uint64_t)rows[i]), put64(e + 40, at);
    for (uint64_t o = 0; o < bytes; o += BLOCK)
      b[k++] = (Block){.fd = fd, .at = (off_t)(at + o), .len = (uint32_t)(bytes - o < BLOCK ? bytes - o : BLOCK),
                       .mem = row + o};
    at += aligned(bytes);
  }
  fcntl(fd, F_NOCACHE, 1);
  Job job = {.blocks = b, .n = k, .writing = 1};
  int64_t written = -1;
  if (!run(&job)) {
    memcpy(h, ROWS_MAGIC, 8);
    put32(h + 8, VERSION), put32(h + 12, (uint32_t)header), put32(h + 16, (uint32_t)from), put32(h + 20, (uint32_t)to);
    put32(h + 24, (uint32_t)n), put32(h + 28, (uint32_t)blocks);
    for (int i = 0; i < k; i++) put32(h + ROWS_FIXED + (size_t)ROWS_REGION * n + 4 * (size_t)i, b[i].crc);
    if (!write_header(fd, h, header, ROWS_CRC_AT)) written = (int64_t)at;
  }
  free(h);
  free(b);
  return written;
}

typedef struct {
  int fd, from, to;
  uint8_t *h;  // its header
  int regions;
  uint32_t blocks;
} Part;

// Region i of a part: its fields in the header.
static const uint8_t *part_region(const Part *p, int i) { return p->h + ROWS_FIXED + (size_t)ROWS_REGION * i; }

int state_file_read(const ModelOps *ops, ModelState *s, int positions, const int *fds, int count) {
  if (!ops->state_regions || count < 1 || count > PARTS_MAX || positions < 0) return -1;
  Part *p = xmalloc((size_t)count * sizeof *p);
  Block *b = NULL;
  int ok = 0, nb = 0, cap = 0;
  for (int j = 0; j < count; j++) {
    size_t header;
    p[j].fd = fds[j];
    if (!(p[j].h = read_header(fds[j], ROWS_MAGIC, 12, ROWS_CRC_AT, &header))) goto done;
    p[j].from = (int)get32(p[j].h + 16), p[j].to = (int)get32(p[j].h + 20);
    p[j].regions = (int)get32(p[j].h + 24), p[j].blocks = get32(p[j].h + 28);
    if (p[j].regions < 0 || p[j].regions > STATE_REGIONS_MAX || p[j].from < 0 || p[j].to < p[j].from ||
        ROWS_FIXED + (uint64_t)ROWS_REGION * p[j].regions + 4ull * p[j].blocks > header)
      goto done;
    // each part continues the rows before it
    if (j ? p[j].from > p[j - 1].to : p[j].from != 0) goto done;
    fcntl(fds[j], F_NOCACHE, 1);
  }
  if (positions > p[count - 1].to) goto done;
  StateRegion r[STATE_REGIONS_MAX];
  int n = ops->state_regions(s, positions, 1, r);
  int64_t covered[STATE_REGIONS_MAX];
  for (int i = 0; i < n; i++) covered[i] = region_rows(&r[i], positions);
  // the parts' shapes and their rows' places
  for (int j = 0; j < count; j++) {
    if (p[j].regions != n) goto done;
    for (int i = 0; i < n; i++) {
      const uint8_t *e = part_region(&p[j], i);
      if (get32(e) != r[i].unit || get32(e + 4) != r[i].ratio || get32(e + 8) != r[i].guard ||
          get32(e + 12) != r[i].lag || (int64_t)get64(e + 24) != region_stable(&r[i], p[j].from) ||
          (int64_t)get64(e + 32) != region_rows(&r[i], p[j].to) - region_stable(&r[i], p[j].from))
        goto done;
    }
  }
  // the latest part holding a row gives it: from the last part back, each
  // gives the rows from its first up to those later parts gave
  for (int j = count - 1; j >= 0; j--)
    for (int i = 0; i < n; i++) {
      const uint8_t *e = part_region(&p[j], i);
      int64_t first = (int64_t)get64(e + 24), rows = (int64_t)get64(e + 32);
      if (covered[i] <= first) continue;          // later parts gave every row from here
      if (first + rows < covered[i]) goto done;   // rows between this part's and the later ones'
      uint64_t hi_byte = (uint64_t)(covered[i] - first) * r[i].unit, bytes = (uint64_t)rows * r[i].unit;
      uint64_t at = get64(e + 40);
      uint32_t block = get32(e + 16);
      if (block + (bytes + BLOCK - 1) / BLOCK > p[j].blocks) goto done;
      for (uint64_t o = 0; o < hi_byte; o += BLOCK) {
        if (nb == cap) b = realloc(b, (size_t)(cap = cap ? 2 * cap : 256) * sizeof *b);
        if (!b) die("out of memory for a saved state's blocks");
        uint32_t len = (uint32_t)(bytes - o < BLOCK ? bytes - o : BLOCK);
        b[nb++] = (Block){.fd = p[j].fd,
                          .at = (off_t)(at + o),
                          .len = len,
                          .crc = get32(p[j].h + ROWS_FIXED + (size_t)ROWS_REGION * n + 4 * (size_t)(block + o / BLOCK)),
                          .lo = 0,
                          .hi = (uint32_t)(hi_byte - o < len ? hi_byte - o : len),
                          .mem = (uint8_t *)r[i].data + (size_t)first * r[i].unit + o};
      }
      covered[i] = first;
    }
  for (int i = 0; i < n; i++)
    if (covered[i] > 0) goto done;  // rows no part holds
  Job job = {.blocks = b, .n = nb};
  ok = !run(&job);
done:
  for (int j = 0; j < count; j++) free(p[j].h);
  free(p);
  free(b);
  return ok ? 0 : -1;
}

// ---- snapshots ----

// The snapshot header: magic, version, header bytes, stream bytes, integers
// and their count, whether it holds logits, blocks, its CRC32C, then each
// block's CRC32C.
enum { SNAP_FIXED = 80, SNAP_CRC_AT = 72 };

static uint64_t parts_bytes(const SnapshotPart *p, int n) {
  uint64_t total = 0;
  for (int i = 0; i < n; i++) total += p[i].bytes;
  return total;
}

int64_t snapshot_file_write(const ModelOps *ops, ModelSnapshot *x, ModelState *s, int pos, int fd) {
  if (!ops->snapshot_parts) return -1;
  SnapshotPart parts[SNAPSHOT_PARTS_MAX];
  int32_t meta[SNAPSHOT_META_MAX] = {0};
  int nmeta = 0, n = ops->snapshot_parts(x, s, pos, parts, meta, &nmeta);
  if (n < 0) return -1;
  uint64_t total = parts_bytes(parts, n), blocks = (total + BLOCK - 1) / BLOCK;
  size_t header = aligned(SNAP_FIXED + 4 * blocks);
  if (header > HEADER_MAX) return -1;
  if (fd < 0) return (int64_t)(header + aligned(total));  // the size alone
  uint8_t *h = NULL;
  if (posix_memalign((void **)&h, ALIGN, header)) die("out of memory for a snapshot's header");
  memset(h, 0, header);
  Block *b = xmalloc((blocks ? blocks : 1) * sizeof *b);
  for (uint64_t i = 0; i < blocks; i++)
    b[i] = (Block){.fd = fd,
                   .at = (off_t)(header + i * BLOCK),
                   .len = (uint32_t)(total - i * BLOCK < BLOCK ? total - i * BLOCK : BLOCK),
                   .stream = i * BLOCK};
  fcntl(fd, F_NOCACHE, 1);
  Job job = {.blocks = b, .n = (int)blocks, .writing = 1, .parts = parts, .nparts = n};
  int64_t written = -1;
  if (!run(&job)) {
    memcpy(h, SNAP_MAGIC, 8);
    put32(h + 8, VERSION), put32(h + 12, (uint32_t)header), put64(h + 16, total), put32(h + 24, (uint32_t)nmeta);
    put32(h + 28, x && ops->snapshot_logits(x) ? 1 : 0);
    for (int i = 0; i < nmeta; i++) put32(h + 32 + 4 * (size_t)i, (uint32_t)meta[i]);
    put32(h + 64, (uint32_t)blocks);
    for (uint64_t i = 0; i < blocks; i++) put32(h + SNAP_FIXED + 4 * i, b[i].crc);
    if (!write_header(fd, h, header, SNAP_CRC_AT)) written = (int64_t)(header + aligned(total));
  }
  free(h);
  free(b);
  return written;
}

ModelSnapshot *snapshot_file_read(const ModelOps *ops, int fd) {
  if (!ops->snapshot_alloc) return NULL;
  size_t header;
  uint8_t *h = read_header(fd, SNAP_MAGIC, 12, SNAP_CRC_AT, &header);
  if (!h) return NULL;
  ModelSnapshot *x = NULL;
  Block *b = NULL;
  uint64_t total = get64(h + 16), blocks = get32(h + 64);
  int nmeta = (int)get32(h + 24), logits = (int)get32(h + 28);
  int32_t meta[SNAPSHOT_META_MAX], again[SNAPSHOT_META_MAX];
  if (nmeta < 0 || nmeta > SNAPSHOT_META_MAX || logits > 1 || blocks != (total + BLOCK - 1) / BLOCK ||
      SNAP_FIXED + 4 * blocks > header)
    goto bad;
  for (int i = 0; i < nmeta; i++) meta[i] = (int32_t)get32(h + 32 + 4 * (size_t)i);
  if (!(x = ops->snapshot_alloc(meta, nmeta, logits))) goto bad;
  SnapshotPart parts[SNAPSHOT_PARTS_MAX];
  int nagain = 0, n = ops->snapshot_parts(x, NULL, 0, parts, again, &nagain);
  if (parts_bytes(parts, n) != total || nagain != nmeta || memcmp(meta, again, (size_t)nmeta * 4)) goto bad;
  b = xmalloc((blocks ? blocks : 1) * sizeof *b);
  for (uint64_t i = 0; i < blocks; i++) {
    uint32_t len = (uint32_t)(total - i * BLOCK < BLOCK ? total - i * BLOCK : BLOCK);
    b[i] = (Block){.fd = fd, .at = (off_t)(header + i * BLOCK), .len = len, .crc = get32(h + SNAP_FIXED + 4 * i),
                   .stream = i * BLOCK};
  }
  fcntl(fd, F_NOCACHE, 1);
  Job job = {.blocks = b, .n = (int)blocks, .parts = parts, .nparts = n};
  if (run(&job)) goto bad;
  free(b);
  free(h);
  return x;
bad:
  if (x) ops->snapshot_free(x);
  free(b);
  free(h);
  return NULL;
}

// ---- identity ----

static int weights_file(const char *name) {
  size_t n = strlen(name);
  return !strcmp(name, "config.json") || (n > 12 && !strcmp(name + n - 12, ".safetensors")) ||
         (n > 5 && !strcmp(name + n - 5, ".gguf"));
}

// The file at path, as `name`: its name, size, and first and last 64 KiB.
static void identity_file(CC_SHA256_CTX *h, const char *path, const char *name) {
  enum { EDGE = 65536 };
  CC_SHA256_Update(h, name, (CC_LONG)strlen(name) + 1);
  struct stat st;
  int fd = open(path, O_RDONLY | O_CLOEXEC);
  uint64_t size = fd >= 0 && !fstat(fd, &st) ? (uint64_t)st.st_size : UINT64_MAX;
  CC_SHA256_Update(h, &size, sizeof size);
  if (fd < 0) return;
  fcntl(fd, F_NOCACHE, 1);
  uint8_t *buf = xmalloc(EDGE);
  size_t head = size < EDGE ? (size_t)size : EDGE;
  if (!read_all(fd, buf, head, 0)) CC_SHA256_Update(h, buf, (CC_LONG)head);
  if (size > EDGE) {
    size_t tail = size - EDGE < EDGE ? (size_t)(size - EDGE) : EDGE;
    if (!read_all(fd, buf, tail, (off_t)(size - tail))) CC_SHA256_Update(h, buf, (CC_LONG)tail);
  }
  free(buf);
  close(fd);
}

typedef struct {
  char **names;
  int n, cap;
} Names;

static void collect(const char *root, const char *rel, Names *out, int depth) {
  char path[4096];
  snprintf(path, sizeof path, "%s%s%s", root, *rel ? "/" : "", rel);
  DIR *d = opendir(path);
  if (!d) return;
  for (struct dirent *e; (e = readdir(d));) {
    if (e->d_name[0] == '.') continue;
    char sub[4096], full[4096];
    snprintf(sub, sizeof sub, "%s%s%s", rel, *rel ? "/" : "", e->d_name);
    snprintf(full, sizeof full, "%s/%s", root, sub);
    struct stat st;
    if (lstat(full, &st)) continue;
    if (S_ISDIR(st.st_mode)) {
      if (depth < 8) collect(root, sub, out, depth + 1);
      continue;
    }
    if (!weights_file(e->d_name)) continue;
    if (out->n == out->cap && !(out->names = realloc(out->names, (size_t)(out->cap = out->cap ? 2 * out->cap : 64) * sizeof *out->names)))
      die("out of memory listing %s", root);
    out->names[out->n++] = strdup(sub);
  }
  closedir(d);
}

static int by_name(const void *a, const void *b) { return strcmp(*(char *const *)a, *(char *const *)b); }

void identity_folder(CC_SHA256_CTX *h, const char *dir) {
  Names names = {0};
  collect(dir, "", &names, 0);
  qsort(names.names, (size_t)names.n, sizeof *names.names, by_name);
  CC_SHA256_Update(h, &names.n, sizeof names.n);
  for (int i = 0; i < names.n; i++) {
    char full[4096];
    snprintf(full, sizeof full, "%s/%s", dir, names.names[i]);
    identity_file(h, full, names.names[i]);
    free(names.names[i]);
  }
  free(names.names);
}

void identity_gguf_set(CC_SHA256_CTX *h, const char *path) {
  const char *slash = strrchr(path, '/'), *base = slash ? slash + 1 : path;
  size_t n = strlen(base), dir = (size_t)(base - path);
  int k, total;
  char tail[32];
  // NAME-00001-of-00005.gguf: every file of the set
  if (n > 20 && sscanf(base + n - 20, "-%5d-of-%5d%31s", &k, &total, tail) == 3 && !strcmp(tail, ".gguf") && total > 0 &&
      total < 100000) {
    for (int i = 1; i <= total; i++) {
      char sibling[4096];
      snprintf(sibling, sizeof sibling, "%.*s%.*s-%05d-of-%05d.gguf", (int)dir, path, (int)(n - 20), base, i, total);
      identity_file(h, sibling, sibling + dir);
    }
  } else identity_file(h, path, base);
}
