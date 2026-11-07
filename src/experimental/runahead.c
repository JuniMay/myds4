// Included by model.c after its state helpers. Each request retains the
// buffers that survive a layer front, including CPU-written descriptors.
// Shared GPU scratch follows command-queue order. The event-based variant
// reads routing after its signal and retires command buffers once per step.
// Requests can advance at different layers while SSD reads run in parallel.
static void run_wait_signal(uint64_t value) {
#ifdef MYDS4_RUNAHEAD_SPIN
  gpu_spin_signal(value);
#else
  gpu_wait_signal(value);
#endif
}

enum { RUN_BUFFERS = 14 };
#define RUN_BUFFER_LIST &HS, &MF, &SEL, &CAND, &XQ, &H1, &SHO, &RT, &EC[0], &EC[1], &ES[0], &ES[1], &ETAB, &EROW
typedef struct {
  Buf saved[RUN_BUFFERS];
  State *state;
  int layer, pos, experts[TOPK];
#ifdef MYDS4_CAUSAL_PRIORITY_READS
  int speculative;
#endif
} RunJob;
static RunJob run_jobs[FEW];
static float *run_logits;
static uint64_t runahead_after_wait_groups;
static uint64_t runahead_head_batches, runahead_head_rows;

static void run_bind(RunJob *job, Buf **live) {
  if (runahead_enabled >= 2) bind(job->state);
  else model_state_use(job->state);
  for (int k = 0; k < RUN_BUFFERS; k++) *live[k] = job->saved[k];
}

static void run_issue(RunJob *job) {
#ifdef MYDS4_CAUSAL_EARLY_STOP
  if (causal_frontier) early_stats.requested_row_layers++;
#endif
  const int *rt = (const int *)RT.ptr;
  CacheObservation observation;
  cache_observe_snapshot(job->layer, &observation);
  cache_observe_routes(&observation, rt, 1, 16, 0);
  cache_observe_routes(&observation, rt, 1, 16, 1);
  uint64_t *tally[] = {&cache_stats.row_ssd, &cache_stats.row_own, &cache_stats.row_ahead, &cache_stats.row_late};
  for (int k = 0; k < TOPK; k++) {
    job->experts[k] = rt[k];
    (*tally[cache_where(job->layer, rt[k])])++;
  }
  for (int i = 1; i < TOPK; i++)
    for (int k = i; k > 0 && job->experts[k] < job->experts[k - 1]; k--) {
      int e = job->experts[k]; job->experts[k] = job->experts[k - 1], job->experts[k - 1] = e;
    }
  cache_stats.row_unique += TOPK;
#ifdef MYDS4_CAUSAL_PRIORITY_READS
  job->speculative = causal_frontier && job->pos > causal_base_pos + causal_frontier->sampled;
  if (job->speculative) {
    early_stats.background_row_layers++;
    cache_hold_speculative(job->layer, job->experts, TOPK);
  } else
#endif
  cache_hold(job->layer, job->experts, TOPK);
  cache_demote(job->layer);
}

// Gather ready jobs at one layer into the existing small-batch kernels.
// Blit copies carry tracked source/destination dependencies across buffers.
static int run_group(RunJob **jobs, const int *ids, int count, Buf **live, const Buf *original,
                     float *outputs, Buf **captures, const Buf *original_capture) {
  const size_t bytes[RUN_BUFFERS] = {HC * DIM * STREAM_BYTES, MIX * 4, (IDX_TOPK + 1) * 4,
    (size_t)cand_words * 4, DIM * 4, HC * DIM * STREAM_BYTES, DIM * 4, 0,
    ENGRAM_ROWS * ENGRAM_DIM, ENGRAM_ROWS * ENGRAM_DIM, ENGRAM_ROWS * 8, ENGRAM_ROWS * 8, 0, 0};
  GpuCopy copies[FEW * (RUN_BUFFERS + 1)];
  int ncopy = 0, layer = jobs[0]->layer;
  runahead_group_count++, runahead_group_rows += count;
  for (int k = 0; k < RUN_BUFFERS; k++) *live[k] = original[k];
  if (captures) MAINH = *original_capture;
  for (int r = 0; r < count; r++) {
    RunJob *job = jobs[r];
    cache_wait(layer, job->experts, TOPK);
    for (int k = 0; k < RUN_BUFFERS; k++) if (bytes[k])
      copies[ncopy++] = (GpuCopy){&job->saved[k], live[k], 0, r * bytes[k], bytes[k]};
    if (captures)
      copies[ncopy++] = (GpuCopy){captures[r], &MAINH, 0, (size_t)r * 3 * DIM * 4, 3 * DIM * 4};
  }
  gpu_begin();
  gpu_copies(copies, ncopy);
  size_t entries = (size_t)layer * FEW * 256;
  for (int r = 0; r < count; r++) {
    const int *rt = (const int *)jobs[r]->saved[7].ptr;
    fill_entries(layer, entries + r * 256, rt, (const float *)rt + 8, NULL, 0);
  }
  for (int r = 0; r < count; r++)
    experts_go(layer, r, entries + r * 256, (size_t)r * MIX * 4, (size_t)r * HC * DIM * STREAM_BYTES, 0);
  uint64_t signal;
  int deferred_head = runahead_enabled == 5 && !captures;
  if (layer == LAYERS - 1) {
    if (deferred_head) {
      ncopy = 0;
      for (int r = 0; r < count; r++) {
        copies[ncopy++] = (GpuCopy){&HS, &jobs[r]->saved[0], r * bytes[0], 0, bytes[0]};
        copies[ncopy++] = (GpuCopy){&MF, &jobs[r]->saved[1], r * bytes[1], 0, bytes[1]};
      }
      gpu_copies(copies, ncopy);
    } else {
      hc_pre_norm(count, &HS, 0, &MF, 0, &final_norm);
      head_out(count, &logits);
    }
    signal = gpu_signal();
    gpu_commit();
  } else {
    gpu_commit();
    for (int r = 0; r < count; r++) row_state[r] = jobs[r]->state, row_pos[r] = jobs[r]->pos;
    nrows = count;
    signal = layer_front(&(Batch){count, jobs[0]->pos, 0}, layer + 1);
    nrows = 0;
  }
  run_wait_signal(signal);
  if (layer != LAYERS - 1) {
    ncopy = 0;
    for (int r = 0; r < count; r++) {
      RunJob *job = jobs[r];
      memcpy(job->saved[7].ptr, original[7].ptr + (size_t)r * 16 * 4, 16 * 4);
      if (captures) memcpy(captures[r]->ptr, original_capture->ptr + (size_t)r * 3 * DIM * 4, 3 * DIM * 4);
      for (int k = 0; k < 7; k++)
        copies[ncopy++] = (GpuCopy){live[k], &job->saved[k], r * bytes[k], 0, bytes[k]};
    }
    gpu_begin();
    gpu_copies(copies, ncopy);
    gpu_commit();
  }
  for (int r = 0; r < count; r++) {
    RunJob *job = jobs[r];
    cache_unhold(layer, job->experts, TOPK);
    job->layer++;
    if (layer == LAYERS - 1) {
      if (!deferred_head) memcpy(outputs + (size_t)ids[r] * VOCAB, logits.ptr + (size_t)r * VOCAB * 4, VOCAB * 4);
    } else RT = job->saved[7], run_issue(job);
  }
  RT = original[7];
  return layer == LAYERS - 1 ? count : 0;
}

static const float *runahead_rows(State **states, const int *tokens, const int *positions, int count) {
  if (nrows || verifying || count < 2 || count > FEW) die("invalid runahead batch");
  for (int r = 0; r < count; r++) if (tokens[r] < 0) die("runahead needs text decode tokens");
#ifdef MYDS4_RUNAHEAD_NO_FORECASTS
  runahead_route_only = 1;
#endif
  request_capacity(count);
  State *original_state = active;
  int signals = runahead_enabled >= 2;
  Buf *live[RUN_BUFFERS] = {RUN_BUFFER_LIST};
  Buf original[RUN_BUFFERS];
  for (int k = 0; k < RUN_BUFFERS; k++) original[k] = *live[k];
  if (!run_logits) {
    size_t bytes = (size_t)FEW * VOCAB * sizeof(float);
    run_logits = xmalloc(bytes), experimental_workspace_bytes += bytes;
  }
  for (int r = 0; r < count; r++) {
    RunJob *job = &run_jobs[r];
    job->state = states[r], job->pos = positions[r], job->layer = 0;
    for (int k = 0; k < RUN_BUFFERS; k++) if (job->saved[k].size != original[k].size) {
      experimental_workspace_bytes -= job->saved[k].size;
      gpu_free(&job->saved[k]);
      job->saved[k] = zeros(original[k].size);
      experimental_workspace_bytes += job->saved[k].size;
    }
  }
  CacheStats before = cache_stats;
  double began = now(), gpu_before = gpu_busy;
  cache_metrics_begin(PF_DECODE, count);
  cache_adapt(); // no job holds a slot or a GPU command at a step boundary
  for (int r = 0; r < count; r++) {
    RunJob *job = &run_jobs[r];
    run_bind(job, live);
    int64_t rows[1][2][ENGRAM_ROWS];
    memcpy(history, job->state->history, sizeof history);
    engram_hash(tokens[r], job->pos, rows[0]);
    memcpy(job->state->history, history, sizeof history);
    // These small reads finish before switching EC/ES to another job.
    // Expert reads from preceding jobs remain in flight meanwhile.
    engram_read(0, rows, 1), engram_read(1, rows, 1);
    embed_read(tokens + r, 1, 0, job->pos);
    uint64_t signal = layer_front(&(Batch){1, job->pos, 0}, 0);
    if (signals) run_wait_signal(signal);
    else gpu_sync();
    run_issue(job);
  }
  int remaining = count;
  while (remaining) {
    int ready = -1, wait = -1, wait_misses = TOPK + 1;
    for (int r = 0; r < count; r++) {
      RunJob *job = &run_jobs[r];
      if (job->layer == LAYERS) continue;
      int missing = 0;
      for (int k = 0; k < TOPK; k++) missing += !cache_ready(job->layer, job->experts[k]);
      if (!missing && (ready < 0 || job->layer < run_jobs[ready].layer)) ready = r;
      if (wait < 0 || job->layer < run_jobs[wait].layer ||
          (job->layer == run_jobs[wait].layer && missing < wait_misses)) wait = r, wait_misses = missing;
    }
    int r = ready >= 0 ? ready : wait;
    if (r < 0) die("runahead has no unfinished job");
    int after_wait = 0;
    if (runahead_enabled == 4 && ready < 0) {
      gpu_keep_clock(.001);
      cache_wait(run_jobs[r].layer, run_jobs[r].experts, TOPK);
      ready = r, after_wait = 1;
    }
    if (runahead_enabled >= 3 && ready >= 0) {
      int ids[FEW], group = 0;
      RunJob *jobs[FEW];
      for (int j = 0; j < count; j++) if (run_jobs[j].layer == run_jobs[ready].layer) {
        int available = 1;
        for (int k = 0; k < TOPK; k++) available &= cache_ready(run_jobs[j].layer, run_jobs[j].experts[k]);
        if (available) ids[group] = j, jobs[group++] = &run_jobs[j];
      }
      if (group > 1) {
        runahead_after_wait_groups += after_wait;
        remaining -= run_group(jobs, ids, group, live, original, run_logits, NULL, NULL);
        continue;
      }
    }
    RunJob *job = &run_jobs[r];
    if (ready < 0) gpu_keep_clock(.001);
    cache_wait(job->layer, job->experts, TOPK);
    run_bind(job, live);
    const int *rt = (const int *)RT.ptr;
    uint64_t signal = 0;
    gpu_begin();
    encode_experts(job->layer, 0, rt, (const float *)RT.ptr + 8, 0, 0);
    if (job->layer == LAYERS - 1) {
      if (runahead_enabled != 5) {
        hc_pre_norm(1, &HS, 0, &MF, 0, &final_norm);
        head_out(1, &logits);
      }
      if (signals) signal = gpu_signal();
    }
    gpu_commit();
    if (job->layer + 1 < LAYERS) signal = layer_front(&(Batch){1, job->pos, 0}, job->layer + 1);
    if (signals) run_wait_signal(signal);
    else gpu_sync();
    cache_unhold(job->layer, job->experts, TOPK);
    if (++job->layer == LAYERS) {
      if (runahead_enabled != 5) memcpy(run_logits + (size_t)r * VOCAB, logits.ptr, VOCAB * sizeof(float));
      remaining--;
    } else run_issue(job);
  }
  if (runahead_enabled == 5) {
    // Sampling begins only after all rows finish. Preserve each row's final
    // streams and mixing weights, then read the output head once per batch.
    for (int k = 0; k < RUN_BUFFERS; k++) *live[k] = original[k];
    GpuCopy copies[2 * FEW];
    size_t hs = HC * DIM * STREAM_BYTES, mf = MIX * sizeof(float);
    for (int r = 0; r < count; r++) {
      copies[2 * r] = (GpuCopy){&run_jobs[r].saved[0], &HS, 0, r * hs, hs};
      copies[2 * r + 1] = (GpuCopy){&run_jobs[r].saved[1], &MF, 0, r * mf, mf};
    }
    gpu_begin();
    gpu_copies(copies, 2 * count);
    hc_pre_norm(count, &HS, 0, &MF, 0, &final_norm);
    head_out(count, &logits);
    gpu_commit();
    runahead_head_batches++, runahead_head_rows += count;
  }
  gpu_sync();
  if (runahead_enabled == 5) memcpy(run_logits, logits.ptr, (size_t)count * VOCAB * sizeof(float));
  if (!signals) memcpy(active->history, history, sizeof history);
  for (int k = 0; k < RUN_BUFFERS; k++) *live[k] = original[k];
  if (signals) bind(original_state), memcpy(history, original_state->history, sizeof history);
  else model_state_use(original_state);
  cache_metrics_end();
  stats_add(&decode_stats, &before);
  decode_rows += count, decode_gpu_s += gpu_busy - gpu_before, decode_s += now() - began;
#ifdef MYDS4_RUNAHEAD_NO_FORECASTS
  runahead_route_only = 0;
#endif
  return run_logits;
}
