// Request workspace must release prompt-sized buffers before decoding.
// Host allocations stand in for Metal buffers; no device or model is opened.
#include "myds4.h"
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static size_t live_bytes, peak_bytes;
static int allocations, pending, unsafe_frees, damaged;
static int cache_clock_advances, cache_maintenance;

static Buf fixture_alloc(size_t n) {
  if (n > (32u << 20) || live_bytes + n > (128u << 20)) die("workspace fixture exceeded its memory limit");
  unsigned char *p = xmalloc(n + 32);
  memset(p, 0xA5, n + 32);
  live_bytes += n;
  if (live_bytes > peak_bytes) peak_bytes = live_bytes;
  allocations++;
  return (Buf){.obj = p, .ptr = p + 16, .size = n};
}

static void fixture_free(Buf *b) {
  if (!b->obj) return;
  unsafe_frees += pending;
  unsigned char *p = b->obj;
  for (int i = 0; i < 16; i++) damaged |= p[i] != 0xA5 || p[16 + b->size + i] != 0xA5;
  live_bytes -= b->size;
  free(p);
  *b = (Buf){0};
}

static void fixture_sync(void) { pending = 0; }
static void fixture_adapt(void) { cache_clock_advances++, cache_maintenance++; }
static void fixture_idle(void) { cache_maintenance++; }

#define gpu_alloc fixture_alloc
#define gpu_free fixture_free
#define gpu_sync fixture_sync
#define cache_adapt fixture_adapt
#define cache_idle fixture_idle
#include "../../src/model/deepseek.c"

static int checks, failures;
static void expect(const char *name, int ok) {
  checks++;
  if (!ok) failures++, fprintf(stderr, "FAIL: %s\n", name);
}

int main(void) {
  alarm(10);
  opt.ctx = 4096;
  opt.draft = 1;
  cand_words = 17;
  Q = fixture_alloc(128u * HEADS * HEAD_DIM * sizeof(float));
  O = fixture_alloc(Q.size);
  request_capacity(64);
  size_t prompt_bytes = live_bytes;
  expect("prompt rows fit", request_rows >= 64 && HS.size >= 64u * HC * DIM * STREAM_BYTES);
  memset(HS.ptr, 0x19, HS.size);
  pending = 1;
  request_capacity(1);
  size_t decode_bytes = live_bytes;
  // a verify pass's rows (the token and the rewind margin's), or batched rows
  size_t small = replay_margin() + 1 > BATCH_ROWS ? replay_margin() + 1 : BATCH_ROWS;
  expect("decoding retains only the small-batch row capacity", request_rows == (int)small);
  expect("prompt allocation is released", decode_bytes < prompt_bytes);
  expect("submitted work completes before prompt buffers are freed", !pending && !unsafe_frees);
  expect("residual storage fits every verification row", HS.size == small * HC * DIM * STREAM_BYTES);
  expect("candidate masks retain only the rows being served", CAND.size == small * cand_words * sizeof(uint32_t));
  expect("query workspace releases unused prompt rows", Q.size == small * HEADS * HEAD_DIM * sizeof(float));
  expect("attention output workspace releases unused prompt rows", O.size == Q.size);
  expect("matrix input staging also fits the draft history projection", XB.size >= 128u * 3 * DIM * 2);
  expect("new residual storage is cleared", HS.ptr[0] == 0 && HS.ptr[HS.size - 1] == 0);
  int before = allocations;
  for (int m = 1; m <= (int)small; m++) request_capacity(m);
  expect("switching among decode batch sizes does not reallocate", allocations == before);
  pending = 1;
  request_capacity(32);
  expect("a later prompt grows the workspace", request_rows >= 32 && HS.size >= 32u * HC * DIM * STREAM_BYTES);
  expect("growth also waits before freeing buffers", !pending && !unsafe_frees);
  request_capacity(1);
  expect("later decoding releases prompt storage again", live_bytes == decode_bytes);
  opt.lookup = VERIFY_DRAFTS;
  request_capacity(1);
  expect("context lookup reserves its wider verification capacity", request_rows == VERIFY_DRAFTS + 1);
  before = allocations;
  for (int m = 1; m <= VERIFY_DRAFTS + 1; m++) request_capacity(m);
  expect("wide verification changes do not reallocate", allocations == before);
  expect("workspace maintenance re-evaluates cache capacity", cache_maintenance > 0);
  expect("workspace maintenance does not advance the cache-use clock", cache_clock_advances == 0);
  fixture_sync();
  Buf *all[] = {&HS, &MF, &SEL, &CAND, &IMG, &EC[0], &ES[0], &EC[1], &ES[1]};
  for (size_t i = 0; i < sizeof all / sizeof *all; i++) fixture_free(all[i]);
  Buf *work[] = {&H1, &MA, &PART, &X, &XQ, &QA, &QR, &QRQ, &Q, &O, &OA, &AO, &KV, &CK, &CS, &LAT,
                &IK, &QI, &IW, &SC, &GL, &GP, &RT, &PR, &SHG, &SHU, &SHH, &SHO, &EROW, &EX, &EKV,
                &MH, &YK, &TILES, &XH, &MHH, &XB, &PTOK, &PW, &TP, &WIN};
  for (size_t i = 0; i < sizeof work / sizeof *work; i++) fixture_free(work[i]);
  expect("all host allocations are released", !live_bytes);
  expect("buffer guards are intact", !damaged);
  fprintf(stderr, "workspace: %d/%d CPU checks passed; prompt=%zu, decode=%zu, peak=%zu bytes; cache clock advances=%d (expected 0)\n",
          checks - failures, checks, prompt_bytes, decode_bytes, peak_bytes, cache_clock_advances);
  return failures != 0;
}
