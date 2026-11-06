// The six original expert indices fit after their records in a 256-byte row.
// Kernels retain those indices for hidden values and the final ordered sum.
static int expert_active_slots(ExpertRef *refs) {
  static_assert(6 * sizeof(ExpertRef) + 6 * sizeof(uint32_t) <= 256);
  uint32_t *active = (uint32_t *)(refs + 6);
  int n = 0;
  for (int e = 0; e < 6; e++) if (refs[e].p[0]) active[n++] = e;
  return n;
}
