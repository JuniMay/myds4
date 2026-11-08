#pragma once
#include "../../engine/myds4.h"

typedef struct QwenVision QwenVision;
QwenVision *qwen_vision_open(TensorIndex *index, const Json *config);
int qwen_prepare_image(const char *dir, Image *image, const void *data, size_t size, const char **error);
int qwen_vision_encode(QwenVision *vision, Image *image);
void qwen_vision_reside(QwenVision *vision, int resident);
uint64_t qwen_vision_bytes(const QwenVision *vision);
void qwen_vision_close(QwenVision *vision);
