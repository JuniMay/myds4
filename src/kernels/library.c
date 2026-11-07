// The GPU kernels of every model, compiled into one Metal library per
// process: the shared number formats first, then each source. A model's
// load calls gpu_start, which compiles the library the first time.
#include "myds4.h"

static const char source[] = {
#embed "formats.metal"
    , '\n',
#embed "kernels.metal"
    , '\n',
#embed "../experimental/moe_tasks.metal"
    , '\n',
#embed "gguf_grids.metal"
    , '\n',
#embed "gguf.metal"
    , '\n',
#embed "vision.metal"
    , '\n',
#embed "rows.metal"
    , '\n',
#embed "products.metal"
    , '\n',
#embed "deltanet.metal"
    , '\n',
#embed "gqa.metal"
};

void gpu_start(void) {
  static int started;
  if (!started) gpu_init(source, sizeof source), started = 1;
}
