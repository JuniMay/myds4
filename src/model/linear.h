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
// Float activation rows; prompt inputs stage through the supplied BF16 buffer.
// Rounded outputs retain FP32 storage with BF16 values.
void linear_product(const Linear *matrix, const Buf *x, size_t x_offset, const Buf *y, size_t y_offset, int rows,
                    int prompt, const Buf *bf16, int staged, int rounded);
