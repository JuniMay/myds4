#include "myds4.h"
#include <signal.h>
#include <spawn.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

// A wait for a value the GPU never signals must end the program with the
// reason after a minute: not return as if the GPU had written the results it
// was waiting for (gpu_wait_signal), and not spin forever (gpu_spin_signal).
// Two spawned copies of this program (Metal does not survive fork) wait at
// once, one each way, and print "returned" if their wait returns. Takes a
// minute. Uses the GPU for nothing but the waits: no model, no kernels run.

extern char **environ;

static pid_t start(const char *self, const char *how, int *fd) {
  int out[2];
  if (pipe(out)) die("pipe");
  posix_spawn_file_actions_t fa;
  posix_spawn_file_actions_init(&fa);
  posix_spawn_file_actions_adddup2(&fa, out[1], 1);
  posix_spawn_file_actions_addclose(&fa, out[0]);
  char *args[] = {(char *)self, (char *)how, 0};
  pid_t pid;
  if (posix_spawn(&pid, self, &fa, 0, args, environ)) die("spawn %s", self);
  close(out[1]);
  *fd = out[0];
  return pid;
}

// Whether the copy ended with an error, and did not say its wait returned;
// a copy still waiting after `seconds` is stopped and fails.
static int ended(const char *name, pid_t pid, int fd, int seconds) {
  int status = 0, done = 0;
  for (int i = 0; i < seconds * 10 && !(done = waitpid(pid, &status, WNOHANG) == pid); i++) usleep(100000);
  if (!done) kill(pid, SIGKILL), waitpid(pid, &status, 0);
  char said[64] = {0};
  for (ssize_t n = 0, k; n + 1 < (ssize_t)sizeof said && (k = read(fd, said + n, sizeof said - 1 - n)) > 0;) n += k;
  close(fd);
  int returned = strstr(said, "returned") != 0, error = done && WIFEXITED(status) && WEXITSTATUS(status) != 0;
  if (!done) fprintf(stderr, "FAIL: %s: still waiting after %d s\n", name, seconds);
  if (returned) fprintf(stderr, "FAIL: %s: returned without the GPU's signal\n", name);
  return !returned && error;
}

int main(int argc, char **argv) {
  if (argc == 2 && (!strcmp(argv[1], "--wait") || !strcmp(argv[1], "--spin"))) {
    static const char source[] = "kernel void nothing() {}\n";
    gpu_init(source, sizeof source - 1);
    if (!strcmp(argv[1], "--wait")) gpu_wait_signal(1ull << 40);  // nothing signals it
    else gpu_spin_signal(1ull << 40);
    printf("returned\n");
    fflush(stdout);
    return 0;
  }
  alarm(200);
  int wfd, sfd;
  pid_t w = start(argv[0], "--wait", &wfd), s = start(argv[0], "--spin", &sfd);
  int ok = ended("gpu_wait_signal", w, wfd, 90);
  ok &= ended("gpu_spin_signal", s, sfd, 20);  // started with the other: 110 s in all
  printf("GPU waits that are never signaled end the program: %s\n", ok ? "PASS" : "FAIL");
  return !ok;
}
