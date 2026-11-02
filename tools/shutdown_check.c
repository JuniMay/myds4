// CPU-only checks of the stop flags used by CLI and server shutdown.
#include "../src/myds4.h"
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

static int checks, failures;

static void expect(const char *name, int ok) {
  checks++;
  if (!ok) failures++, fprintf(stderr, "FAIL: %s\n", name);
}

int main(void) {
  alarm(10);
  const int signals[] = {SIGINT, SIGTERM};
  for (int i = 0; i < 2; i++) {
    pid_t child = fork();
    if (child < 0) {
      perror("fork");
      return 1;
    }
    if (!child) {
      alarm(5);
      checks = failures = 0;
      shutdown_watch();
      expect("watch starts without a stop request", !shutdown_requested());
      expect("signal handler returns to the owner", raise(signals[i]) == 0);
      expect("signal requests a stop", shutdown_requested());
      shutdown_watch();
      expect("reinstalling handlers preserves a pending stop", shutdown_requested());
      _exit(failures);
    }
    int status;
    if (waitpid(child, &status, 0) != child) {
      perror("waitpid");
      return 1;
    }
    checks += 4;
    failures += WIFEXITED(status) ? WEXITSTATUS(status) : 4;
  }
  fprintf(stderr, "shutdown: %d/%d CPU checks passed\n", checks - failures, checks);
  return failures != 0;
}
