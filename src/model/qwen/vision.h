#pragma once
#include "../../engine/myds4.h"

typedef struct QwenVision QwenVision;
QwenVision *qwen_vision_open(TensorIndex *index, const Json *config);
int qwen_prepare_image(const char *dir, Media *image, const void *data, size_t size, const char **error);
int qwen_vision_encode(QwenVision *vision, Media *image);
int qwen_vision_trace(QwenVision *vision, Media *image, int fd); // per-block FP32 values for reference checks
void qwen_vision_reside(QwenVision *vision, int resident);
uint64_t qwen_vision_bytes(const QwenVision *vision);
void qwen_vision_close(QwenVision *vision);
void qwen_image_positions(const Media *images, int count, int position, int delta, int prompt, int32_t out[3]);
