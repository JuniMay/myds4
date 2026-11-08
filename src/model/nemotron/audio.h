#pragma once
#include "../../engine/myds4.h"

typedef struct SpeechEncoder SpeechEncoder;
SpeechEncoder *speech_open(TensorIndex *checkpoint, const Json *configuration);
int speech_prepare(const char *directory, Media *media, const void *data, size_t length, const char **error);
int speech_encode(SpeechEncoder *encoder, Media *media);
uint64_t speech_weight_bytes(const SpeechEncoder *encoder);
void speech_reside(SpeechEncoder *encoder, int active);
