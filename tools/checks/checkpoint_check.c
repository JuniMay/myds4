// Reading a second checkpoint must leave the first model's tensors usable.
// This reads safetensors headers only; no weights or GPU work are needed.
#include "myds4.h"
#include "../../src/model/model.h"
#include <fcntl.h>
#include <sys/file.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

static int loads;
static void load_stub(const char *directory, const Options *options) { loads++; }
static const char *const qwen_types[] = {"qwen3_5", 0};
static const char *const deepseek_types[] = {"deepseek_v41", 0};
const ModelOps qwen_ops = {.family = "Qwen", .model_types = qwen_types, .load = load_stub};
const ModelOps deepseek_ops = {.family = "DeepSeek", .model_types = deepseek_types, .load = load_stub};

static int model_lock(const char *first, const char *second) {
  const char *path = getenv("MYDS4_LOCK");
  if (!path) return 2;
  Options options = {0};
  Model *a = model_open(first, &options), *b = model_open(second, &options);
  int fd = open(path, O_RDWR | O_CREAT, 0600);
  int locked = fd >= 0 && flock(fd, LOCK_EX | LOCK_NB) != 0;
  if (fd >= 0) close(fd);
  free(a), free(b);
  if (!locked || loads != 2) {
    fputs("FAIL: every model family must share the process inference lock\n", stderr);
    return 1;
  }
  puts("model ownership: both families share one process lock");
  return 0;
}

static int malformed_json(void) {
  pid_t child = fork();
  if (!child) {
    char text[] = "{\"value\":1,}";
    json_free(json_parse(text));
    _exit(0);
  }
  int status;
  if (child < 0 || waitpid(child, &status, 0) != child) return 1;
  if (!WIFEXITED(status) || WEXITSTATUS(status) != 1) {
    fputs("FAIL: malformed JSON was accepted\n", stderr);
    return 1;
  }
  puts("metadata parsing: malformed JSON rejected");
  return 0;
}

int main(int argc, char **argv) {
  if (argc == 1) return malformed_json();
  if (argc != 3) return 2;
  TensorIndex *first = st_open(argv[1]);
  Tensor before = st_get(first, "vision.patch_embed.proj.weight");
  TensorIndex *second = st_open(argv[2]);
  Tensor after = st_get(first, "vision.patch_embed.proj.weight");
  if (before.fd != after.fd || before.off != after.off || before.size != after.size) return 1;
  if (st_has(second, "vision.patch_embed.proj.weight")) return 1;
  st_close(second);
  after = st_get(first, "vision.patch_embed.proj.weight");
  if (before.fd != after.fd || before.off != after.off || before.size != after.size) return 1;
  st_close(first);
  puts("checkpoint ownership: first model's tensor remains available");
  return model_lock(argv[2], argv[1]);
}
