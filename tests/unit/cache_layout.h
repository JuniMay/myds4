// The expert-cache tests include io.c without DeepSeek's model code. They
// keep the span table that DeepSeek's model code keeps, and create a store
// with DeepSeek's layout (st = deepseek_cache_create()) before they use it.
#pragma once
#include "../../src/model/deepseek/model.h"

FileSpan expert_span[CACHE_LAYERS][EXPERTS][3];
int expert_spans[CACHE_LAYERS];
static ExpertStore *st;
