// Included by model.c. Execute completed whole experts while other selected
// experts are still loading. Each expert writes its original PARTS index;
// one final reduction preserves the selected experts' summation order.
static int experts_as_ready(const Batch *b, int layer, const int *need, int count, void *rest, double *waited) {
  char done[EXPERTS] = {0};
  int finished = 0, wave = 0;
  const int *routes = (const int *)RT.ptr;
  const float *weights = (const float *)RT.ptr + 8;
  expert_wave_layers++;
  while (finished < count) {
    int available[TOPK], pending[TOPK], nr = 0, np = 0;
    char chosen[EXPERTS] = {0};
    int final_group = expert_wave_cap && wave + 1 >= expert_wave_cap;
    if (final_group) {
      for (int k = 0; k < count; k++) if (!done[need[k]]) pending[np++] = need[k];
      double began = now();
      cache_wait(layer, pending, np);
      *waited += now() - began;
      np = 0;
    }
    for (int k = 0; k < count; k++) if (!done[need[k]]) {
      int e = need[k];
      if (cache_ready(layer, e) && (final_group || !expert_waves_limit || nr < expert_waves_limit))
        available[nr++] = e, chosen[e] = 1;
      else pending[np++] = e;
    }
    if (!nr) {
      double began = now();
      cache_wait_any(layer, pending, np);
      *waited += now() - began;
      expert_wave_waits++;
      continue;
    }
    double began = now();
    cache_wait(layer, available, nr);
    *waited += now() - began;
    if (!wave && nr == count) {
      encode_experts(layer, 0, routes, weights, mf_off(b), hs_off(b));
      expert_wave_batches++;
      return 0;
    }
    int last = finished + nr == count;
    if (wave) {
      if (last) gpu_encode(rest);
      else gpu_begin();
    }
    if (wave >= TOPK) die("too many expert completion waves");
    size_t eo = (size_t)(3 * LAYERS * FEW + DRAFTS * BLOCK + layer * TOPK + wave) * 256;
    fill_entries(layer, eo, routes, weights, chosen, 1);
    experts_go(layer, 0, eo, mf_off(b), hs_off(b), 1);
    for (int k = 0; k < nr; k++) done[available[k]] = 1;
    finished += nr, wave++, expert_wave_batches++;
    if (last) {
      experts_sum(0, mf_off(b), hs_off(b));
      return 1; // the caller appends the head and commits the final buffer
    }
    gpu_commit();
  }
  die("empty expert completion schedule");
}
