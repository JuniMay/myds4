// Checkpoint matrices and their stored quantization formats.
#pragma once
#include "weights.h"

typedef struct {
  Buf w, s;
  float scale, input_scale;
  int rows, cols, kind; // 'n' NVFP4, '8' scalar FP8, 'm' MXFP8, 'b' BF16
} Linear;

void linear_init(void);
void linear_load(WeightSet *weights, Linear *matrix, TensorIndex *index, const char *stem);
// A matrix that the expert store reads (NVFP4): its shape, format and scalar
// scales, and the tensors of its weights and scales. Compressed scales (CSF)
// give their fixed rows as `scales` and their exceptions; stored scales give
// an empty `exceptions`. Its buffers are views of an expert's slot, set while
// the expert is held.
void linear_describe(Linear *matrix, TensorIndex *index, const char *stem, Tensor *weight, Tensor *scales,
                     Tensor *exceptions);
// Float activation rows; prompt inputs stage through the supplied BF16 buffer.
// Rounded outputs retain FP32 storage with BF16 values.
void linear_product(const Linear *matrix, const Buf *x, size_t x_offset, const Buf *y, size_t y_offset, int rows,
                    int prompt, const Buf *bf16, int staged, int rounded);
