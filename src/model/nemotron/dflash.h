#pragma once
#include "../linear.h"
#include "../../engine/pass.h"

typedef struct DFlash DFlash;
typedef struct {
  Buf *kv;
  int capacity, position;
} DFlashRow;
DFlash *dflash_open(const char *directory, WeightSet *backbone, const Buf *embedding, const Linear *head);
// A pass's own buffers (src/engine/pass.h): dflash_pass_variable is what the
// pass swaps; begin and end run inside the pass, after its GPU work for end.
PassVariable dflash_pass_variable(DFlash *draft);
void dflash_pass_begin(DFlash *draft);  // draft may be null
void dflash_pass_end(DFlash *draft);    // draft may be null
int dflash_load(DFlash *draft);
void dflash_active(DFlash *draft, int active);
uint64_t dflash_weight_bytes(const DFlash *draft);
uint64_t dflash_required(const DFlash *draft);
uint64_t dflash_kv_bytes(int capacity);
int dflash_prepare(DFlash *draft, int rows, int positions);
void dflash_capture(DFlash *draft, int layer, const Buf *hidden, const Buf *captured, int rows);
void dflash_context(DFlash *draft, const Buf *captured, const DFlashRow *rows, int count);
void dflash_predict(DFlash *draft, const DFlashRow *state, int anchor, int *tokens, int count);
