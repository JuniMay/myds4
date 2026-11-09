// Quitting (SIGINT, SIGTERM, the live view's q) ends a model's load between
// weights: the server waits for its engine thread's current call before it
// exits, so a load that runs to its end keeps a quitting server alive for
// the whole load. The stopped load returns its memory and leaves the weights
// unloaded. Weights that a pass loads complete with the pass. Each case runs
// in its own child process, since the quit request stays set.
//
// weights.c runs with fixture buffers: no Metal device, no files.
#include "myds4.h"
#include "../../src/engine/memory.h"
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>

static Buf fixture_new(size_t size) {
  Buf b = {.allocated = size, .size = size};
  b.memory = memory_current() ? memory_current() : memory_shared();
  if (!memory_claim(b.memory, b.allocated, MEMORY_REQUIRED)) die("the fixture budget refused %zu bytes", size);
  b.ptr = calloc(1, size ? size : 1);
  b.obj = b.ptr;
  return b;
}
static void fixture_free(Buf *b) {
  if (b->obj) {
    memory_release(b->memory, b->allocated);
    free(b->ptr);
  }
  *b = (Buf){0};
}
static size_t fixture_allocation_size(size_t size) { return size; }
static void fixture_resident(const Buf *b, int on) {}
static void fixture_commit(void) {}
static void fixture_use(const Buf *b) {}
static void fixture_read(int fd, uint64_t off, uint64_t len, Buf *dst) {}
#define gpu_new fixture_new
#define gpu_free fixture_free
#define gpu_allocation_size fixture_allocation_size
#define gpu_resident fixture_resident
#define gpu_residency_commit fixture_commit
#define gpu_use fixture_use
#define io_read fixture_read
#include "../../src/model/weights.c"
#undef gpu_new
#undef gpu_free
#undef gpu_allocation_size
#undef gpu_resident
#undef gpu_residency_commit
#undef gpu_use
#undef io_read

enum { WEIGHTS = 8, BYTES = 4096 };
static int fills;
static void fill(Buf *buffer, const void *parameters) {
  if (++fills == 1) shutdown_request();  // quitting starts while the first weight is read
}

static int failures;
static void expect(const char *name, int ok) {
  if (!ok) failures++, fprintf(stderr, "FAIL: %s\n", name);
}

// Eight weights load; quitting starts while the first one is read.
static int load(int model_load) {
  memory_init(1ull << 30);
  MemoryAccount account;
  memory_account_init(&account, "fixture model");
  WeightSet *set = weights_new(&account, "fixture weights");
  Buf targets[WEIGHTS] = {0};
  for (int i = 0; i < WEIGHTS; i++) weights_generated(set, &targets[i], BYTES, fill, NULL, 0);
  weights_stop_on_quit(model_load);
  int loaded = weights_load(set);
  if (model_load) {
    expect("no weight of a model's load is read after quitting starts", fills == 1);
    expect("the stopped load reports that the weights did not load", !loaded);
    expect("the stopped load leaves the weights unloaded", !memory_ready(weights_slot(set)));
    expect("the stopped load returns its memory", memory_usage().allocated == 0 && memory_usage().committed == 0);
  } else {
    expect("a pass's load completes", fills == WEIGHTS && loaded && memory_ready(weights_slot(set)));
  }
  fprintf(stderr, "  %s: %d of %d weights read, %s\n", model_load ? "a model's load" : "a pass's load", fills, WEIGHTS,
          loaded ? "loaded" : "not loaded");
  return failures == 0;
}

int main(void) {
  int passed = 0;
  for (int model_load = 1; model_load >= 0; model_load--) {
    fflush(stderr);
    pid_t child = fork();
    if (child == 0) _exit(load(model_load) ? 0 : 1);
    int status = 0;
    waitpid(child, &status, 0);
    passed += WIFEXITED(status) && WEXITSTATUS(status) == 0;
  }
  fprintf(stderr, "weight load stop: %d/2 CPU checks passed\n", passed);
  return passed != 2;
}
