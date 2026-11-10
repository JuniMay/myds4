// This program's memory is the installed memory less the reserve less what
// other programs hold at that moment: it follows them and is never a fixed
// number. KV comes first: the experts may use the space that KV does not
// occupy and give it back when KV grows, and never take the space KV
// occupies. Model memory is GPU-resident, so the kernel keeps it wired;
// anything added past the available memory is met by swapping out other
// programs.
//
// The engine's memory code runs on a simulated Mac: 128 GiB installed, other
// programs holding 28 GiB, so 100 GiB is available while this program holds
// nothing. The simulated kernel counts this program's model and KV memory as
// used once it is written, as the real one does. Each case runs in its own
// child process.
#include <mach/mach.h>
#include <sys/sysctl.h>
#include <sys/wait.h>
#include <string.h>
#include <unistd.h>
#include "myds4.h"
#include "../../src/engine/memory.h"

#define GB (1ull << 30)
static uint64_t others_growth;  // what other programs took after the start
static uint64_t base_free = 100 * GB, file_cache, kept_share;  // kept_share 0: unreadable
static uint64_t unwritten;  // charged model memory whose pages are not written yet

static kern_return_t fixture_host(host_t host, host_flavor_t flavor, host_info64_t out, mach_msg_type_number_t *n) {
  MemoryUsage u = memory_usage();
  uint64_t used = u.allocated - unwritten + u.kv + others_growth, available = used < base_free ? base_free - used : 0;
  vm_statistics64_data_t s = {0};
  s.free_count = (natural_t)(available / vm_page_size);
  s.external_page_count = (natural_t)(file_cache / vm_page_size);
  memcpy(out, &s, sizeof s);
  return KERN_SUCCESS;
}
static int fixture_sysctl(const char *name, void *out, size_t *n, void *in, size_t len) {
  uint64_t value = !strcmp(name, "hw.memsize") ? 128 * GB :
                   !strcmp(name, "kern.memorystatus_vm_pressure_level") ? 1 :
                   !strcmp(name, "vm.vm_page_filecache_min") ? kept_share / vm_page_size : 0;
  if (!out || !n) return -1;
  size_t size = *n < sizeof value ? *n : sizeof value;
  memcpy(out, &value, size), *n = size;
  return 0;
}
#define host_statistics64 fixture_host
#define sysctlbyname fixture_sysctl
#include "../../src/engine/io.c"
#include "cache_layout.h"
#undef host_statistics64
#undef sysctlbyname

typedef struct { MemorySlot slot; int evicted; } Entry;
static void evict_entry(void *context) {
  Entry *e = context;
  e->evicted++;
  memory_release(&e->slot, e->slot.payload);
}
static Entry *entry(MemoryAccount *account, Entry *parent, uint64_t bytes) {
  Entry *e = calloc(1, sizeof *e);
  memory_slot_init(&e->slot, account, parent ? &parent->slot : NULL, "weight group", evict_entry, e);
  if (memory_claim(&e->slot, bytes, MEMORY_REQUIRED)) memory_publish(&e->slot, 1);
  return e;
}

// Loads thirty 4 GiB weight groups one after another and returns the most
// model memory this program held at any point.
static uint64_t fill(void) {
  MemoryAccount *model = calloc(1, sizeof *model);
  memory_account_init(model, "model");
  uint64_t most = 0;
  for (int i = 0; i < 30; i++) {
    entry(model, NULL, 4 * GB);
    if (memory_usage().allocated > most) most = memory_usage().allocated;
  }
  return most;
}
static double gib(uint64_t bytes) { return (double)bytes / GB; }

static int your_budgets_with_kv_empty(void) {
  memory_configure(110, 4);
  uint64_t most = fill();
  fprintf(stderr, "  memory 110, reserve 4, no KV: model memory reached %.0f GiB (at most 96)\n", gib(most));
  return most <= 96 * GB;
}

static int your_budgets_with_kv_in_use(void) {
  memory_configure(110, 4);
  memory_kv_change(20 * GB, 1);
  uint64_t most = fill();
  fprintf(stderr, "  memory 110, reserve 4, 20 GiB of KV: model memory reached %.0f GiB (at most 76)\n",
          gib(most));
  return most <= 76 * GB;
}

static int the_defaults(void) {
  memory_configure(0, 0);
  uint64_t most = fill();
  fprintf(stderr, "  defaults (reserve %d): model memory reached %.0f GiB (at most 90)\n", CACHE_RESERVE_DEFAULT, gib(most));
  return most <= 90 * GB;
}

static int a_lower_memory_setting(void) {
  memory_configure(64, 4);
  uint64_t most = fill();
  fprintf(stderr, "  memory 64: model memory reached %.0f GiB (exactly 64)\n", gib(most));
  return most == 64 * GB;
}

// DeepSeek holds an 8 GiB backbone and 44 GiB of idle experts. Other
// programs then take 40 GiB, which leaves 8 GiB available, and Qwen's 20 GiB
// load arrives: it first takes back at least 20 - (8 - 4) = 16 GiB of idle
// experts, or it is refused.
static int a_load_after_other_programs_grew(void) {
  memory_configure(110, 4);
  MemoryAccount deepseek, qwen;
  memory_account_init(&deepseek, "DeepSeek"), memory_account_init(&qwen, "Qwen");
  Entry *backbone = entry(&deepseek, NULL, 8 * GB), *experts[11];
  memory_pin(&backbone->slot, 1);
  for (int i = 0; i < 11; i++) experts[i] = entry(&deepseek, backbone, 4 * GB);
  others_growth = 40 * GB;
  Entry weights = {0};
  memory_slot_init(&weights.slot, &qwen, NULL, "Qwen weights", evict_entry, &weights);
  MemorySlot *wanted[] = {&weights.slot};
  uint64_t bytes = 20 * GB, back = 0;
  int admitted = memory_admit(wanted, &bytes, 1, MEMORY_REQUIRED);
  for (int i = 0; i < 11; i++) back += experts[i]->evicted ? 4 * GB : 0;
  fprintf(stderr, "  a 20 GiB load with 8 GiB available: %s, %.0f GiB of experts taken back (at least 16, or refused)\n",
          admitted ? "admitted" : "refused", gib(back));
  return !admitted || back >= 16 * GB;
}

// DeepSeek holds an 8 GiB backbone and 88 GiB of idle experts: 96 GiB, so
// 4 GiB is available, the reserve. KV then grows by 1 GiB within its budget:
// the experts give that space back.
static int kv_growth_takes_its_space_from_the_experts(void) {
  memory_configure(110, 4);
  MemoryAccount deepseek;
  memory_account_init(&deepseek, "DeepSeek");
  Entry *backbone = entry(&deepseek, NULL, 8 * GB), *experts[22];
  memory_pin(&backbone->slot, 1);
  for (int i = 0; i < 22; i++) experts[i] = entry(&deepseek, backbone, 4 * GB);
  uint64_t held = memory_usage().allocated, back = 0;
  memory_kv_change(1 * GB, 1);
  for (int i = 0; i < 22; i++) back += experts[i]->evicted ? 4 * GB : 0;
  fprintf(stderr, "  1 GiB of KV growth with %.0f GiB of model memory held: %.0f GiB of experts taken back (at least 1)\n",
          gib(held), gib(back));
  return back >= 1 * GB;
}

// 10 GiB free and 30 GiB of file cache, of which the kernel keeps 24 GiB:
// 16 GiB is available, so model memory stops at 12 GiB above the 4 GiB
// reserve. The file cache within the kept share is not room.
static int the_kept_file_cache_is_not_room(void) {
  base_free = 10 * GB, file_cache = 30 * GB, kept_share = 24 * GB;
  memory_configure(110, 4);
  uint64_t most = fill();
  fprintf(stderr, "  10 GiB free, 30 GiB of file cache, 24 GiB kept: model memory reached %.0f GiB (at most 12)\n",
          gib(most));
  return most <= 12 * GB;
}

// The expert cache charges a slot when it allocates it, but macOS counts the
// slot's pages as in use only once its read writes them. DeepSeek holds
// 80 GiB of written experts, so 20 GiB is available. The cache then adds
// 1 GiB slots whose reads are all still under way: it stops 16 GiB later, at
// the 4 GiB reserve, not at the 110 GiB setting.
static uint64_t fixture_unwritten(void) { return unwritten; }
static int slots_not_yet_written_are_not_room(void) {
  memory_configure(110, 4);
  memory_unwritten(fixture_unwritten);
  MemoryAccount *deepseek = calloc(1, sizeof *deepseek);
  memory_account_init(deepseek, "DeepSeek");
  for (int i = 0; i < 20; i++) entry(deepseek, NULL, 4 * GB);
  for (int i = 0; i < 40; i++) {
    MemorySlot *slot = calloc(1, sizeof *slot);
    memory_slot_init(slot, deepseek, NULL, "expert slot", NULL, NULL);
    if (!memory_claim(slot, GB, MEMORY_SPECULATIVE)) break;
    unwritten += GB;
  }
  uint64_t held = memory_usage().allocated;
  unwritten = 0;  // the reads finish
  fprintf(stderr, "  1 GiB slots added while their reads are under way: model memory reached %.0f GiB "
                  "once written (at most 96)\n", gib(held));
  return held <= 96 * GB;
}

// The expert cache grows into the room above the reserve, slot by slot, as
// far as the memory allows (a speculative claim).
static Entry *grown_slot(MemoryAccount *account, uint64_t bytes) {
  Entry *e = calloc(1, sizeof *e);
  memory_slot_init(&e->slot, account, NULL, "routed expert", evict_entry, e);
  if (!memory_claim(&e->slot, bytes, MEMORY_SPECULATIVE)) {
    memory_slot_remove(&e->slot);
    free(e);
    return NULL;
  }
  memory_publish(&e->slot, 1);
  return e;
}

// Before a pass, a model reserves its expert store's fewest slots
// (memory_minimum with cache_required's bytes), and the store's slot limit
// rises to them. The slots within that reservation are then admitted even
// with the available memory at the reserve, so the store can fill them;
// slots beyond it are refused there. A model with an 8 GiB backbone, with
// 6 GiB available (the 4 GiB reserve and 2 GiB), reserves 2 GiB of 64 MiB
// slots.
static int reserved_slots_are_admitted_at_the_reserve(void) {
  memory_configure(110, 4);
  MemoryAccount *flash = calloc(1, sizeof *flash);
  memory_account_init(flash, "Flash-Next");
  entry(flash, NULL, 8 * GB);
  others_growth = 100 * GB - 8 * GB - 6 * GB;
  if (!memory_minimum(flash, 10 * GB)) return 0;
  int within = 0, beyond = 0;
  for (int i = 0; i < 32; i++) within += grown_slot(flash, 64ull << 20) != NULL;
  for (int i = 0; i < 4; i++) beyond += grown_slot(flash, 64ull << 20) != NULL;
  fprintf(stderr, "  at the reserve: %d of 32 slots within a reserved 2 GiB admitted (all), %d of 4 beyond it (none)\n",
          within, beyond);
  return within == 32 && beyond == 0;
}

// A pass takes small host buffers in the model's workspace and frees them
// (a token's embedding and Engram rows), and the cache grows by a slot when
// an expert misses. Once the cache has filled the room above the reserve,
// the small buffers must not take an expert back on every pass: of 20
// passes, at most one may.
static int small_buffers_do_not_take_experts_back_each_pass(void) {
  memory_configure(110, 4);
  MemoryAccount *deepseek = calloc(1, sizeof *deepseek);
  memory_account_init(deepseek, "DeepSeek");
  MemorySlot *work = calloc(1, sizeof *work);
  memory_slot_init(work, deepseek, NULL, "model workspace", NULL, NULL);
  work->automatic = 1;
  enum { MOST = 2048 };
  Entry **grown = calloc(MOST, sizeof *grown);
  int n = 0, taken = 0;
  while (n < MOST && (grown[n] = grown_slot(deepseek, 64ull << 20))) n++;
  for (int pass = 0; pass < 20; pass++) {
    MemorySlot *before = memory_scope(work);
    memory_host_free(memory_host_alloc(64, 0));
    memory_scope(before);
    if (n < MOST && (grown[n] = grown_slot(deepseek, 64ull << 20))) n++;
  }
  for (int i = 0; i < n; i++) taken += grown[i]->evicted;
  fprintf(stderr, "  20 passes with small buffers at the reserve: %d expert slots taken back (at most 1)\n", taken);
  return taken <= 1;
}

// Memory messages go to the engine's log (the server's live view shows it),
// and only take-backs of 1 GiB or more: a small buffer that takes back one
// 64 MiB slot logs nothing, and a 4 GiB load that takes back 4 GiB logs one
// line.
static int lines_logged;
static char last_line[512];
static void capture(const char *line) {
  lines_logged++;
  snprintf(last_line, sizeof last_line, "%s", line);
}
static int memory_messages_go_to_the_log(void) {
  memory_configure(110, 4);
  log_hook = capture;
  MemoryAccount *deepseek = calloc(1, sizeof *deepseek), *qwen = calloc(1, sizeof *qwen);
  memory_account_init(deepseek, "DeepSeek"), memory_account_init(qwen, "Qwen");
  MemorySlot *work = calloc(1, sizeof *work);
  memory_slot_init(work, deepseek, NULL, "model workspace", NULL, NULL);
  work->automatic = 1;
  for (int i = 0; i < 1536; i++) entry(deepseek, NULL, 64ull << 20);  // 96 GiB: the room above the reserve
  MemorySlot *before = memory_scope(work);
  memory_host_free(memory_host_alloc(64, 0));
  memory_scope(before);
  int small = lines_logged;
  entry(qwen, NULL, 4 * GB);
  fprintf(stderr, "  memory messages in the log: %d for a 64 MiB take-back (none), %d for a 4 GiB one (one): %s\n",
          small, lines_logged - small, lines_logged ? last_line : "-");
  return small == 0 && lines_logged == 1 && strstr(last_line, "took back");
}

static int checks, failures;
static void run_case(const char *name, int (*check)(void)) {
  fflush(stderr);
  pid_t child = fork();
  if (child == 0) _exit(check() ? 0 : 1);
  int status = 0;
  waitpid(child, &status, 0);
  checks++;
  if (!WIFEXITED(status) || WEXITSTATUS(status)) failures++, fprintf(stderr, "FAIL: %s\n", name);
}

int main(void) {
  st = deepseek_cache_create();
  run_case("with your budgets and no KV, model memory stays within 128 - 4 - 28 GiB", your_budgets_with_kv_empty);
  run_case("model memory never takes the space that KV occupies", your_budgets_with_kv_in_use);
  run_case("with the defaults, model memory stays within 128 - 10 - 28 GiB", the_defaults);
  run_case("a memory setting below the free memory still caps model memory", a_lower_memory_setting);
  run_case("a load after other programs grew first takes back idle experts, or is refused", a_load_after_other_programs_grew);
  run_case("KV growth takes its space back from idle experts", kv_growth_takes_its_space_from_the_experts);
  run_case("the file cache the kernel keeps is not room for model memory", the_kept_file_cache_is_not_room);
  run_case("charged slots that their reads have not written yet are not room", slots_not_yet_written_are_not_room);
  run_case("a pass's small buffers do not take an expert back on every pass", small_buffers_do_not_take_experts_back_each_pass);
  run_case("slots within a pass's reservation are admitted at the reserve", reserved_slots_are_admitted_at_the_reserve);
  run_case("memory messages go to the engine's log, and only large take-backs", memory_messages_go_to_the_log);
  fprintf(stderr, "memory budget: %d/%d CPU checks passed\n", checks - failures, checks);
  return failures != 0;
}
