// The expert-cache tests include io.c without DeepSeek's model code. They
// keep the span table that DeepSeek's model code keeps, and give the store
// DeepSeek's layout (deepseek_cache_layout) before they use it.
#pragma once
#include "../../src/model/deepseek/model.h"

FileSpan expert_span[CACHE_LAYERS][EXPERTS][3];
int expert_spans[CACHE_LAYERS];
