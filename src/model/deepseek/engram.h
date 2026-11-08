#pragma once
#include <stdint.h>

typedef struct {
  int64_t multipliers[8], primes[48], offsets[48];
  int32_t pad, count, *map;
} Engram;

void engram_init(Engram *layout, const char *checkpoint);
void engram_free(Engram *layout);
