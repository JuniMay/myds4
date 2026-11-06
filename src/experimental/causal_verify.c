// Included after runahead.c. Verification rows share one causal KV state.
// Each front waits for the preceding row's front at that layer. Routed
// products can finish earlier and wait with their output in private HS.
// Without a sampling frontier, every row is completed. With one, rows after
// the first rejection or EOS may be cancelled before their remaining layers.
typedef struct { RunJob job; Buf mainh; int products_done; } CausalJob;
static CausalJob causal_jobs[FEW];
static float *causal_logits;

static void causal_bind(CausalJob *job, Buf **live) {
  for (int k = 0; k < RUN_BUFFERS; k++) *live[k] = job->job.saved[k];
  if (opt.draft) MAINH = job->mainh;
}

static int causal_front_ready(int row) {
  return !row || causal_jobs[row - 1].job.layer > causal_jobs[row].job.layer;
}

static int causal_prefer(int row, int previous) {
  if (previous < 0) return 1;
#ifdef MYDS4_CAUSAL_LEAD_FIRST
  return row < previous;
#else
  return causal_jobs[row].job.layer < causal_jobs[previous].job.layer;
#endif
}

#ifdef MYDS4_CAUSAL_EARLY_STOP
static int causal_decision(int count) {
  if (!causal_frontier) return 0;
  uint32_t ready = 0;
  for (int r = 0; r < count; r++)
    if (causal_jobs[r].job.layer == LAYERS) ready |= UINT32_C(1) << r;
  return speculative_frontier_poll(causal_frontier, causal_logits, VOCAB, ready);
}
#endif

static const float *causal_verify(const int *tokens, int count, int pos) {
#ifdef MYDS4_CAUSAL_SIGNALS
  const int signals = 1;
#else
  const int signals = 0;
#endif
  if (!verifying || nrows || count < 2 || count > FEW || pos + count > opt.ctx)
    die("invalid causal verification batch");
#ifdef MYDS4_RUNAHEAD_NO_FORECASTS
  // This scheduler requests only selected experts; it never reads PC/PR.
  // Keep the actual router while omitting those unused projections.
  runahead_route_only = 1;
#endif
  reserve(pos + count, pos);
  request_capacity(count);
  Buf *live[RUN_BUFFERS] = {RUN_BUFFER_LIST};
  Buf original[RUN_BUFFERS], original_mainh = MAINH;
  for (int k = 0; k < RUN_BUFFERS; k++) original[k] = *live[k];
  if (!causal_logits) {
    size_t bytes = (size_t)FEW * VOCAB * sizeof(float);
    causal_logits = xmalloc(bytes), experimental_workspace_bytes += bytes;
  }
  for (int r = 0; r < count; r++) {
    if (tokens[r] < 0) die("causal verification needs text tokens");
    CausalJob *job = &causal_jobs[r];
    job->job.state = active, job->job.pos = pos + r, job->job.layer = job->products_done = 0;
    for (int k = 0; k < RUN_BUFFERS; k++) if (job->job.saved[k].size != original[k].size) {
      experimental_workspace_bytes -= job->job.saved[k].size;
      gpu_free(&job->job.saved[k]);
      job->job.saved[k] = zeros(original[k].size);
      experimental_workspace_bytes += job->job.saved[k].size;
    }
    if (opt.draft && job->mainh.size != original_mainh.size) {
      experimental_workspace_bytes -= job->mainh.size;
      gpu_free(&job->mainh);
      job->mainh = zeros(original_mainh.size);
      experimental_workspace_bytes += job->mainh.size;
    }
  }
  CacheStats before = cache_stats;
  double began = now(), gpu_before = gpu_busy;
  cache_metrics_begin(PF_DECODE, count);
  cache_adapt();
  for (int r = 0; r < count; r++) {
    CausalJob *job = &causal_jobs[r];
    causal_bind(job, live);
    int64_t rows[1][2][ENGRAM_ROWS];
    engram_hash(tokens[r], pos + r, rows[0]);
    engram_read(0, rows, 1), engram_read(1, rows, 1);
    embed_read(tokens + r, 1, 0, pos + r);
    uint64_t signal = layer_front(&(Batch){1, pos + r, 0}, 0);
    if (signals) run_wait_signal(signal);
    else gpu_sync();
    run_issue(&job->job);
  }
  int remaining = count;
  while (remaining) {
#ifdef MYDS4_CAUSAL_EARLY_STOP
    if (causal_decision(count)) break;
#endif
    int ready = -1, wait = -1;
#ifndef MYDS4_CAUSAL_LEAD_FIRST
    int wait_misses = TOPK + 1;
#endif
    for (int r = 0; r < count; r++) {
      CausalJob *job = &causal_jobs[r];
      if (job->job.layer == LAYERS) continue;
#ifdef MYDS4_CAUSAL_PRIORITY_READS
      if (causal_frontier && r == causal_frontier->sampled && !job->products_done && job->job.speculative) {
        cache_promote_selected(job->job.layer, job->job.experts, TOPK);
        job->job.speculative = 0;
        early_stats.promoted_row_layers++;
      }
#endif
      if (job->products_done) {
        if (causal_front_ready(r) && causal_prefer(r, ready)) ready = r;
        continue;
      }
      int missing = 0;
      for (int k = 0; k < TOPK; k++) missing += !cache_ready(job->job.layer, job->job.experts[k]);
      if (!missing && causal_prefer(r, ready)) ready = r;
#ifdef MYDS4_CAUSAL_LEAD_FIRST
      if (wait < 0) wait = r;
#else
      if (wait < 0 || job->job.layer < causal_jobs[wait].job.layer ||
          (job->job.layer == causal_jobs[wait].job.layer && missing < wait_misses)) wait = r, wait_misses = missing;
#endif
    }
    int r = ready >= 0 ? ready : wait;
    if (r < 0) die("causal verification cannot make progress");
#ifdef MYDS4_CAUSAL_GROUP
    if (ready >= 0 && !causal_jobs[r].products_done &&
        (causal_jobs[r].job.layer == LAYERS - 1 || causal_front_ready(r))) {
      int ids[FEW], group = 0;
      RunJob *jobs[FEW];
      Buf *captures[FEW];
      // Consecutive rows can form one front: attention encodes them in
      // token order, so each row sees its predecessor's completed keys.
      for (int j = r; j < count && causal_jobs[j].job.layer == causal_jobs[r].job.layer &&
                         !causal_jobs[j].products_done; j++) {
        int available = 1;
        for (int k = 0; k < TOPK; k++) available &= cache_ready(causal_jobs[j].job.layer, causal_jobs[j].job.experts[k]);
        if (!available) break;
        ids[group] = j, jobs[group] = &causal_jobs[j].job, captures[group++] = &causal_jobs[j].mainh;
      }
      if (group > 1) {
        remaining -= run_group(jobs, ids, group, live, original, causal_logits, opt.draft ? captures : NULL, &original_mainh);
        continue;
      }
    }
#endif
    CausalJob *job = &causal_jobs[r];
    RunJob *base = &job->job;
    int products = !job->products_done, last = base->layer == LAYERS - 1;
    int front = !last && causal_front_ready(r);
    uint64_t signal = 0;
    if (products) {
      if (ready < 0) gpu_keep_clock(.001);
      cache_wait(base->layer, base->experts, TOPK);
    }
    causal_bind(job, live);
    if (products) {
      const int *rt = (const int *)RT.ptr;
      gpu_begin();
      encode_experts(base->layer, 0, rt, (const float *)RT.ptr + 8, 0, 0);
      if (last) hc_pre_norm(1, &HS, 0, &MF, 0, &final_norm), head_out(1, &logits);
      if (signals && !front) signal = gpu_signal();
      gpu_commit();
    }
    if (front) signal = layer_front(&(Batch){1, base->pos, 0}, base->layer + 1);
    if (signals) run_wait_signal(signal);
    else gpu_sync();
    if (products) cache_unhold(base->layer, base->experts, TOPK);
    if (last) {
      memcpy(causal_logits + (size_t)r * VOCAB, logits.ptr, VOCAB * sizeof(float));
      base->layer = LAYERS;
      remaining--;
    } else if (front) {
      base->layer++, job->products_done = 0;
      run_issue(base);
    } else job->products_done = 1;
  }
  gpu_sync();
#ifdef MYDS4_CAUSAL_EARLY_STOP
  if (causal_frontier) {
    if (!causal_decision(count)) die("completed verification has no sampling decision");
    early_stats.sampled_rows += causal_frontier->sampled;
    for (int r = 0; r < count; r++) {
      CausalJob *job = &causal_jobs[r];
      if (job->job.layer == LAYERS) {
        early_stats.finished_rows++;
      } else {
        // Products that completed already released their reservation. A
        // pending read keeps ownership of its buffer after the job lets go.
        if (!job->products_done) cache_unhold(job->job.layer, job->job.experts, TOPK);
        early_stats.cancelled_rows++;
        early_stats.skipped_expert_row_layers += LAYERS - job->job.layer - job->products_done;
      }
    }
  }
#endif
  for (int k = 0; k < RUN_BUFFERS; k++) *live[k] = original[k];
  MAINH = original_mainh;
  if (opt.draft)
    for (int r = 0; r < count; r++)
      memcpy(MAINH.ptr + (size_t)r * 3 * DIM * 4, causal_jobs[r].mainh.ptr, 3 * DIM * 4);
  cache_metrics_end();
  stats_add(&decode_stats, &before);
  decode_rows += count, decode_gpu_s += gpu_busy - gpu_before, decode_s += now() - began;
#ifdef MYDS4_RUNAHEAD_NO_FORECASTS
  runahead_route_only = 0;
#endif
  return causal_logits;
}
