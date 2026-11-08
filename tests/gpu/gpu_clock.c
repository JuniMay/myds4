// Tiny GPU check of clock-keeper calibration; no model or expert files.
#include "../../src/engine/myds4.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int host_clock_calls;
static double delayed_host_clock(void) {
  double t = now();
  // Simulate 10 ms of host scheduling delay around the first calibration.
  // A GPU execution rate must not include that delay.
  return ++host_clock_calls == 2 ? t + .010 : t;
}

#define now delayed_host_clock
#include "../../src/engine/gpu.c"
#undef now

static int ascending(const void *a, const void *b) {
  double x = *(const double *)a, y = *(const double *)b;
  return (x > y) - (x < y);
}

int main(void) {
  (void)delayed_host_clock;
  alarm(30);
  static const char kernels[] = {
#embed "../../src/kernels/kernels.metal"
      , 0};
  const char *first = strstr(kernels, "kernel void spin("), *last = first ? strstr(first, "\n}") : NULL;
  if (!first || !last) return 1;
  Str source = {0};
  str_cat(&source, "#include <metal_stdlib>\nusing namespace metal;\n");
  str_put(&source, first, (size_t)(last + 2 - first));
  gpu_init(source.p, source.n);
  free(source.p);
  gpu_keep_clock(.002);
  double samples[5];
  for (int i = 0; i < 5; i++) {
    id cb = spin((uint32_t)(.002 / spin_s));
    MSG(void (*)(id, SEL), cb, S("waitUntilCompleted"));
    samples[i] = MSG(double (*)(id, SEL), cb, S("GPUEndTime")) - MSG(double (*)(id, SEL), cb, S("GPUStartTime"));
  }
  qsort(samples, 5, sizeof samples[0], ascending);
  int accurate = samples[2] >= .002 * .75 && samples[2] <= .002 * 1.5;
  fprintf(stderr, "GPU clock: requested 2.000 ms, measured median %.3f ms; %s\n",
          samples[2] * 1000, accurate ? "passed" : "FAILED (host delay contaminated calibration)");
  return !accurate;
}
