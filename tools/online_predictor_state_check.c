// CPU-only persistence checks. Owned temporary files are removed before exit.
#include "../src/online_predictor.c"
#include <dirent.h>
#include <signal.h>
#include <sys/wait.h>

static int checks, failures;
static char directory[4096];

static void expect(const char *name, int ok) {
  checks++;
  if (!ok) failures++, fprintf(stderr, "FAIL: %s\n", name);
}

static void cleanup(void) {
  if (!*directory) return;
  DIR *d = opendir(directory);
  if (d) {
    struct dirent *entry;
    while ((entry = readdir(d))) {
      if (!strcmp(entry->d_name, ".") || !strcmp(entry->d_name, "..")) continue;
      char path[8192];
      snprintf(path, sizeof path, "%s/%s", directory, entry->d_name);
      unlink(path);
    }
    closedir(d);
  }
  rmdir(directory);
}

static int temporary_count(void) {
  DIR *d = opendir(directory);
  if (!d) return -1;
  int count = 0;
  struct dirent *entry;
  while ((entry = readdir(d))) count += strstr(entry->d_name, ".tmp.") != NULL;
  closedir(d);
  return count;
}

static int patch_file(const char *path, size_t offset, const void *value, size_t bytes) {
  int fd = open(path, O_WRONLY);
  if (fd < 0) return 0;
  int ok = pwrite(fd, value, bytes, (off_t)offset) == (ssize_t)bytes;
  close(fd);
  return ok;
}

static int fifo_rejected_without_waiting(const char *path) {
  pid_t child = fork();
  if (child < 0) return 0;
  if (!child) {
    char error[256];
    _exit(online_predictor_load(path, error, sizeof error) == -1 ? 0 : 1);
  }
  int status;
  for (int attempt = 0; attempt < 30; attempt++) {
    if (waitpid(child, &status, WNOHANG) == child) return WIFEXITED(status) && WEXITSTATUS(status) == 0;
    usleep(10000);
  }
  kill(child, SIGKILL);
  waitpid(child, &status, 0);
  return 0;
}

static void covariance_runtime_checks(void) {
  OpPair sample = op_create_pair(EXPERTS, ONLINE_RANK, 123);
  float phi[ONLINE_RANK], target[EXPERTS];
  for (int k = 0; k < ONLINE_RANK; k++) phi[k] = .01f * (k % 7);
  for (int k = 0; k < EXPERTS; k++) target[k] = .001f * (k % 23 - 11);
  int invalid = 0, inspected = 0;
  for (int row = 0; row < 10000; row++) {
    op_update(&sample, &online.work, phi, target);
    if (row % 128 == 0) inspected++, invalid += !state_covariance(sample.p);
  }
  expect("live-produced covariance remains valid for persistence", invalid == 0);
  printf("covariance diagnostic: %d/%d sampled states rejected, %llu guarded resets\n", invalid, inspected,
         (unsigned long long)sample.covariance_resets);
  op_free_pair(&sample);
}

static void reject_mutation(const char *name, const char *path, unsigned char *golden, size_t offset,
                            const void *replacement, size_t bytes, int resign) {
  unsigned char saved[32], digest[32];
  if (bytes > sizeof saved) op_die("test mutation too large");
  memcpy(saved, golden + offset, bytes);
  memcpy(digest, golden + STATE_DIGEST, sizeof digest);
  memcpy(golden + offset, replacement, bytes);
  if (resign) state_digest(golden, golden + STATE_DIGEST);
  int patched =
      patch_file(path, offset, replacement, bytes) && patch_file(path, STATE_DIGEST, golden + STATE_DIGEST, 32);
  char error[256];
  unsigned char *before = state_encode();
  int dirty_before = online.dirty;
  int result = patched ? online_predictor_load(path, error, sizeof error) : 1;
  unsigned char *after = state_encode();
  expect(name,
         patched && result == -1 && *error && !memcmp(before, after, STATE_BYTES) && online.dirty == dirty_before);
  free(before);
  free(after);
  memcpy(golden + offset, saved, bytes);
  memcpy(golden + STATE_DIGEST, digest, sizeof digest);
  if (!patch_file(path, offset, saved, bytes) || !patch_file(path, STATE_DIGEST, digest, 32))
    op_die("cannot restore state test fixture");
}

int main(int argc, char **argv) {
  alarm(90);
  const char *base = getenv("TMPDIR");
  if (!base || !*base) base = "/tmp";
  if (snprintf(directory, sizeof directory, "%s/myds4-state-check.XXXXXX", base) >= (int)sizeof directory ||
      !mkdtemp(directory))
    op_die("cannot create state-check directory");
  atexit(cleanup);
  char path[8192], unrelated[8192], link[8192], absent_parent[8192], fifo[8192], error[256];
  snprintf(path, sizeof path, "%s/learner.bin", directory);
  snprintf(unrelated, sizeof unrelated, "%s/notes.txt", directory);
  snprintf(link, sizeof link, "%s/linked.bin", directory);
  snprintf(absent_parent, sizeof absent_parent, "%s/absent/learner.bin", directory);
  snprintf(fifo, sizeof fifo, "%s/input.pipe", directory);
  unsetenv("MYDS4_ONLINE_FREEZE_ROWS");
  unsetenv("MYDS4_ONLINE_WINDOW_PASSES");
  static float biases[LAYERS][EXPERTS];
  const float *bias[LAYERS];
  for (int l = 0; l < LAYERS; l++) {
    bias[l] = biases[l];
    for (int e = 0; e < EXPERTS; e++) biases[l][e] = .0001f * (l + e % 17);
  }
  online_predictor_init(64, bias);
  expect("persistence requires a model binding", online_predictor_save(path, error, sizeof error) == -1);
  unsigned char identity[32];
  for (int k = 0; k < 32; k++) identity[k] = (unsigned char)(k * 7 + 1);
  online_predictor_bind(identity);
  expect("FIFO fixture created", !mkfifo(fifo, 0600));
  expect("FIFO without a writer is rejected promptly", fifo_rejected_without_waiting(fifo));
  if (argc == 2 && !strcmp(argv[1], "--fifo-only")) {
    printf("online predictor FIFO: %d/%d checks passed\n", checks - failures, checks);
    return failures != 0;
  }
  if (argc == 2 && !strcmp(argv[1], "--covariance-only")) {
    covariance_runtime_checks();
    return failures != 0;
  }
  expect("new initialized learner is dirty", online_predictor_dirty());
  expect("missing state is distinct from invalid state", online_predictor_load(path, error, sizeof error) == 0);
  expect("missing parent is rejected before a fresh run",
         online_predictor_load(absent_parent, error, sizeof error) == -1);

  float phi[ONLINE_RANK], target[EXPERTS];
  for (int k = 0; k < ONLINE_RANK; k++) phi[k] = .01f * (k % 7);
  for (int k = 0; k < EXPERTS; k++) target[k] = .001f * (k % 23 - 11);
  for (int p = 0; p < ONLINE_PAIRS; p++)
    for (int row = 0; row < 3; row++)
      if (!op_update(&online.models[p], &online.work, phi, target)) op_die("fixture update failed");
  online.epoch_rows = 27;
  expect("trained state saves successfully", online_predictor_save(path, error, sizeof error) == 1);
  struct stat st;
  expect("state has a fixed bounded size and private permissions",
         !stat(path, &st) && st.st_size == STATE_BYTES && (st.st_mode & 0777) == 0600);
  expect("successful save clears dirty state", !online_predictor_dirty());
  unsigned char *golden = state_encode();
  float expected_b[STATE_B], expected_p[STATE_P];
  expect("reference continuation accepts next observation", op_update(&online.models[0], &online.work, phi, target));
  memcpy(expected_b, online.models[0].b, sizeof expected_b);
  memcpy(expected_p, online.models[0].p, sizeof expected_p);
  online_predictor_reset();
  online.models[0].a[0] += 1;
  online.records[0].observed = 1;
  expect("trained state loads successfully", online_predictor_load(path, error, sizeof error) == 1);
  unsigned char *restored = state_encode();
  expect("all A B P biases active flags and progress round-trip exactly", !memcmp(golden, restored, STATE_BYTES));
  free(restored);
  expect("load clears pending observations and dirty state", !online.records[0].observed && !online_predictor_dirty());
  expect("restored covariance produces identical next update",
         op_update(&online.models[0], &online.work, phi, target) &&
             !memcmp(expected_b, online.models[0].b, sizeof expected_b) &&
             !memcmp(expected_p, online.models[0].p, sizeof expected_p));
  expect("reload restores the original training prefix", online_predictor_load(path, error, sizeof error) == 1);

  unsigned char wrong = 'X';
  reject_mutation("wrong magic rejected without changing live state", path, golden, 0, &wrong, 1, 1);
  uint32_t version = 2;
  reject_mutation("unknown version rejected", path, golden, 8, &version, 4, 1);
  uint32_t rank = 128;
  reject_mutation("wrong dimensions rejected", path, golden, 36, &rank, 4, 1);
  unsigned char wrong_identity = identity[0] ^ 1;
  reject_mutation("different model or quantization binding rejected", path, golden, 56, &wrong_identity, 1, 1);
  float nan = NAN, changed = .123f;
  size_t a_offset = STATE_HEADER + STATE_META;
  size_t b_offset = a_offset + STATE_A * 4;
  size_t p_offset = b_offset + STATE_B * 4;
  size_t bias_offset = p_offset + STATE_P * 4;
  reject_mutation("corrupt payload checksum rejected", path, golden, b_offset, &changed, 4, 0);
  reject_mutation("nonfinite parameter rejected with a valid checksum", path, golden, a_offset, &nan, 4, 1);
  reject_mutation("native router bias mismatch rejected", path, golden, bias_offset, &changed, 4, 1);
  uint32_t inactive = 0;
  reject_mutation("active flag inconsistent with learned weights rejected", path, golden, STATE_HEADER + 8, &inactive,
                  4, 1);
  float zero = 0, huge = OP_COV_LIMIT * 2;
  reject_mutation("nonpositive covariance diagonal rejected", path, golden, p_offset, &zero, 4, 1);
  reject_mutation("out-of-range covariance rejected", path, golden, p_offset, &huge, 4, 1);
  reject_mutation("asymmetric covariance rejected", path, golden, p_offset + 4, &changed, 4, 1);
  float negative = -1;
  reject_mutation("negative covariance diagonal rejected", path, golden, p_offset, &negative, 4, 1);

  int fd = open(path, O_WRONLY);
  int shortened = fd >= 0 && !ftruncate(fd, STATE_BYTES - 1);
  if (fd >= 0) close(fd);
  expect("truncated file rejected", shortened && online_predictor_load(path, error, sizeof error) == -1);
  expect("truncate fixture restored", patch_file(path, STATE_BYTES - 1, golden + STATE_BYTES - 1, 1));
  expect("symlink fixture created", !symlink(path, link));
  expect("symlink load rejected", online_predictor_load(link, error, sizeof error) == -1);
  expect("symlink save rejected", online_predictor_save(link, error, sizeof error) == -1);
  fd = open(unrelated, O_WRONLY | O_CREAT | O_EXCL, 0600);
  const char note[] = "keep this file";
  int noted = fd >= 0 && state_write(fd, note, sizeof note);
  if (fd >= 0) close(fd);
  expect("unrelated destination cannot be overwritten",
         noted && online_predictor_save(unrelated, error, sizeof error) == -1);
  char readback[sizeof note];
  fd = open(unrelated, O_RDONLY);
  int unchanged = fd >= 0 && state_read(fd, readback, sizeof readback) && !memcmp(readback, note, sizeof note);
  if (fd >= 0) close(fd);
  expect("unrelated destination retained exactly", unchanged);
  expect("missing destination directory fails without temp residue",
         online_predictor_save(absent_parent, error, sizeof error) == -1 && temporary_count() == 0);
  pid_t writer = fork();
  if (!writer) {
    alarm(5);
    signal(SIGXFSZ, SIG_IGN);
    struct rlimit limit;
    if (getrlimit(RLIMIT_FSIZE, &limit)) _exit(1);
    limit.rlim_cur = 64;
    if (setrlimit(RLIMIT_FSIZE, &limit)) _exit(1);
    _exit(online_predictor_save(path, error, sizeof error) == -1 && temporary_count() == 0 ? 0 : 1);
  }
  int writer_status = 0;
  expect("partial write failure removes the sibling temporary file",
         writer > 0 && waitpid(writer, &writer_status, 0) == writer && WIFEXITED(writer_status) &&
             WEXITSTATUS(writer_status) == 0 && temporary_count() == 0);
  expect("partial write failure leaves the prior checkpoint loadable",
         online_predictor_load(path, error, sizeof error) == 1);
  online_predictor_begin(1, 1);
  expect("saving an active pass is refused", online_predictor_save(path, error, sizeof error) == -1);
  expect("loading an active pass is refused", online_predictor_load(path, error, sizeof error) == -1);
  online_predictor_end();
  online.dirty = 1;
  expect("atomic replacement succeeds without residue",
         online_predictor_save(path, error, sizeof error) == 1 && temporary_count() == 0);
  expect("atomically replaced state reloads", online_predictor_load(path, error, sizeof error) == 1);
  covariance_runtime_checks();
  printf("online predictor persistence: %d/%d checks passed; state file %d bytes\n", checks - failures, checks,
         STATE_BYTES);
  free(golden);
  for (int p = 0; p < online.count; p++) op_free_pair(&online.models[p]);
  op_free_work(&online.work);
  free(online.models);
  free(online.records);
  return failures != 0;
}
