// A stand-in for the server's state (serve_view) for `tools/check.py view`:
// the live view of src/tui.c draws it in a pseudo-terminal, with more log
// lines than the screen holds, until q requests the stop (at most 5 s).
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "myds4.h"

void serve_view(View *v) {
  memset(v, 0, sizeof *v);
  snprintf(v->title, sizeof v->title, "myds4 · view check");
  v->phase = "idle";
  v->gpu_limit = (int64_t)52 << 30;
  v->kv_limit_gib = 8;
  v->mem.total = (uint64_t)128 << 30, v->mem.free = (uint64_t)40 << 30;
}

int main(void) {
  char line[64];
  for (int i = 0; i < 100; i++) snprintf(line, sizeof line, "log line %d", i), tui_log(line);
  if (!tui_start()) return 1;
  shutdown_watch();  // as the server does once the view runs
  for (int i = 0; i < 500 && !shutdown_requested(); i++) usleep(10000);
  return shutdown_requested() ? 0 : 2;
}
