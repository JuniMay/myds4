#pragma once
#include "../../engine/myds4.h"

typedef struct QwenVision QwenVision;
QwenVision *qwen_vision_open(TensorIndex *index, const Json *config);
int qwen_prepare_image(const char *dir, Image *image, const void *data, size_t size, const char **error);
int qwen_vision_encode(QwenVision *vision, Image *image);
int qwen_vision_trace(QwenVision *vision, Image *image, int fd); // per-block FP32 values for reference checks
void qwen_vision_reside(QwenVision *vision, int resident);
uint64_t qwen_vision_bytes(const QwenVision *vision);
void qwen_vision_close(QwenVision *vision);
uint64_t qwen_image_embedding(const Image *images, int count, int position);
void qwen_image_positions(const Image *images, int count, int position, int delta, int prompt, int32_t out[3]);
