// Shared by the model checks of families with routed experts: hashes of
// logits, so that two builds can be compared, and which routed experts a
// pass marked used.
#pragma once
#include "../../src/model/model.h"
#include <string.h>

static uint64_t logits_hash(const void *data, size_t bytes) {
  const uint8_t *p = data;
  uint64_t h = 0xcbf29ce484222325ull;
  for (size_t i = 0; i < bytes; i++)
    h = (h ^ p[i]) * 0x100000001b3ull;
  return h;
}

// Eviction takes the experts used longest ago first, so a pass must mark
// every expert it uses, and only those: a decoded row marks the experts each
// layer chose for it. The memory pool lists its entries newest first, so an
// entry made after the model's leads to all of them. Counts the routed
// experts in memory (entries holding memory: an expert's own weight group,
// or a slot of the expert store), and those touched (marked used) at or
// after `since`.
static int touched_experts(double since, int *ready) {
  static const char *const names[] = {"Flash-Next expert", "routed expert"};
  static MemoryAccount account;
  if (!account.name)
    memory_account_init(&account, "recency probe");
  MemorySlot probe;
  memory_slot_init(&probe, &account, NULL, "recency probe", NULL, NULL);
  int touched = 0;
  *ready = 0;
  for (const MemorySlot *s = probe.next; s; s = s->next)
    for (size_t i = 0; i < sizeof names / sizeof *names; i++)
      if (!strcmp(s->name, names[i]) && s->bytes) {
        ++*ready;
        touched += s->touched >= since;
      }
  memory_slot_remove(&probe);
  return touched;
}
