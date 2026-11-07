// Included by deepseek.c. Requests retain their layer-front outputs and
// expert-stage buffers. GPU completions advance stages without a per-job
// blocking wait. Ready expert stages can share one dispatch across layers.
#include "moe_tasks.h"
enum { PIPE_BUFFERS = 17, PIPE_BATCHES = 2 * LAYERS * BATCH_ROWS };
#define PIPE_BUFFER_LIST &HS, &MF, &SEL, &CAND, &XQ, &H1, &SHO, &RT, &EC[0], &EC[1], &ES[0], &ES[1], &ETAB, &EROW, &MH, &logits, &IMG
enum { PIPE_FRONT, PIPE_READ, PIPE_DOWN, PIPE_FINISH, PIPE_DONE };
typedef struct {
  Buf saved[PIPE_BUFFERS];
  State *state;
  int row, layer, pos, phase, held_layer, experts[TOPK];
  uint64_t signal, release_at;
} PipeJob;
static PipeJob pipe_jobs[BATCH_ROWS];
static Buf pipe_tasks;
static float *pipe_logits;
static int K_pipe = -1;
static struct {
  uint64_t steps, batches, tasks, mixed_stages, mixed_layers, fronts, max_inflight;
  double read_wait, gpu_wait;
} pipe_stats;

static void pipeline_report(FILE *out) {
  if (!pipe_stats.steps) return;
  fprintf(out, "{\"pipeline\":{\"steps\":%llu,\"stage_batches\":%llu,\"tasks\":%llu,\"mixed_stages\":%llu,"
               "\"mixed_layers\":%llu,\"fronts\":%llu,\"max_inflight\":%llu,\"read_wait_seconds\":%.6f,\"gpu_wait_seconds\":%.6f}}\n",
          (unsigned long long)pipe_stats.steps, (unsigned long long)pipe_stats.batches, (unsigned long long)pipe_stats.tasks,
          (unsigned long long)pipe_stats.mixed_stages, (unsigned long long)pipe_stats.mixed_layers,
          (unsigned long long)pipe_stats.fronts, (unsigned long long)pipe_stats.max_inflight,
          pipe_stats.read_wait, pipe_stats.gpu_wait);
}

static void pipe_bind(PipeJob *job, Buf **live) {
  bind(job->state);
  for (int k = 0; k < PIPE_BUFFERS; k++) *live[k] = job->saved[k];
}

static void pipe_front(PipeJob *job, Buf **live) {
  pipe_bind(job, live);
  job->phase = PIPE_FRONT;
  job->signal = layer_front(&(Batch){1, job->pos, 0}, job->layer);
  pipe_stats.fronts++;
}

static void pipe_issue(PipeJob *job) {
  const int *rt = (const int *)job->saved[7].ptr;
  CacheObservation observed;
  cache_observe_snapshot(job->layer, &observed);
  cache_observe_routes(&observed, rt, 1, 16, 0);
  cache_observe_routes(&observed, rt, 1, 16, 1);
  uint64_t *tally[] = {&cache_stats.row_ssd, &cache_stats.row_own, &cache_stats.row_ahead, &cache_stats.row_late};
  for (int k = 0; k < TOPK; k++) {
    int e = rt[k], where = cache_where(job->layer, e);
    job->experts[k] = e;
    (*tally[where])++;
    route_id[job->row][job->layer][k] = e;
    route_from[job->row][job->layer][k] = (where + 4) % 5;
    route_step[job->row][job->layer] = route_steps;
  }
  for (int i = 1; i < TOPK; i++)
    for (int k = i; k && job->experts[k] < job->experts[k - 1]; k--) {
      int e = job->experts[k]; job->experts[k] = job->experts[k - 1], job->experts[k - 1] = e;
    }
  cache_stats.row_unique += TOPK;
  cache_hold(job->layer, job->experts, TOPK);
  cache_demote(job->layer);
  job->held_layer = job->layer;
  job->phase = PIPE_READ;
}

static void pipe_single(int kind, int layer, size_t entries) {
  gpu_kern(kind ? K_moe_down : K_moe_up);
  gpu_buf(0, &ETAB, entries);
  if (!kind) {
    gpu_buf(1, &XQ, 0), gpu_buf(2, &MH, 0);
    gpu_go(INTER / 32, TOPK, 1, 256, 1, 1);
  } else {
    uint32_t n = TOPK, pm = 0;
    gpu_buf(1, &MH, 0), gpu_buf(2, &SHO, 0), gpu_buf(3, &MF, 0);
    gpu_buf(4, &H1, 0), gpu_buf(5, &HS, 0), gpu_bytes(6, &n, 4), gpu_bytes(7, &pm, 4), gpu_buf(8, &PARTS, 0);
    gpu_go(DIM / 4, 1, 1, 32 * TOPK, 1, 1);
  }
}

static const float *pipeline_rows(State **states, const int *tokens, const int *positions, int count) {
  if (nrows || verifying || count < 2 || count > BATCH_ROWS) die("invalid pipeline batch");
  if (opt.experts || opt.draft || opt.substitute > 0 || opt.online_predictor) die("pipeline needs native MXFP4 exact routing");
  request_capacity(1);
  State *original_state = active;
  Buf *live[PIPE_BUFFERS] = {PIPE_BUFFER_LIST}, original[PIPE_BUFFERS];
  for (int k = 0; k < PIPE_BUFFERS; k++) original[k] = *live[k];
  if (K_pipe < 0) K_pipe = gpu_kernel("moe_tasks");
  if (!pipe_tasks.obj) {
    pipe_tasks = zeros((size_t)PIPE_BATCHES * BATCH_ROWS * sizeof(MoeTask));
    pipe_logits = xmalloc((size_t)BATCH_ROWS * VOCAB * 4);
    experimental_workspace_bytes += pipe_tasks.size + (size_t)BATCH_ROWS * VOCAB * 4;
  }
  for (int r = 0; r < count; r++) {
    PipeJob *job = pipe_jobs + r;
    job->state = states[r], job->row = r, job->pos = positions[r], job->layer = 0;
    job->held_layer = -1, job->release_at = 0;
    for (int k = 0; k < PIPE_BUFFERS; k++) {
      size_t bytes = k == 14 ? TOPK * INTER * 4 : k == 15 ? VOCAB * 4 : original[k].size;
      if (job->saved[k].size == bytes) continue;
      experimental_workspace_bytes -= job->saved[k].size;
      gpu_free(job->saved + k);
      job->saved[k] = zeros(bytes);
      experimental_workspace_bytes += bytes;
    }
    memset(job->saved[16].ptr, 0, job->saved[16].size);
  }
  CacheStats before = cache_stats;
  double began = now(), gpu_before = gpu_busy;
  cache_metrics_begin(PF_DECODE, count);
  cache_adapt();
  runahead_route_only = 1;
  route_steps++;
  pipe_stats.steps++;
  for (int r = 0; r < count; r++) {
    PipeJob *job = pipe_jobs + r;
    if (tokens[r] < 0) die("pipeline needs decoded text tokens");
    pipe_bind(job, live);
    int64_t rows[1][2][ENGRAM_ROWS];
    memcpy(history, job->state->history, sizeof history);
    engram_hash(tokens[r], job->pos, rows[0]);
    memcpy(job->state->history, history, sizeof history);
    engram_read(0, rows, 1), engram_read(1, rows, 1);
    embed_read(tokens + r, 1, 0, job->pos);
    pipe_front(job, live);
  }
  int remaining = count, batch = 0;
  while (remaining) {
    uint64_t epoch = cache_read_epoch(), completed = gpu_signal_value(), wait = UINT64_MAX;
    PipeJob *ready[BATCH_ROWS];
    int kind[BATCH_ROWS], nr = 0, pending = 0;
    for (int r = 0; r < count; r++) {
      PipeJob *job = pipe_jobs + r;
      if (job->release_at && job->release_at <= completed) {
        cache_unhold(job->held_layer, job->experts, TOPK);
        job->held_layer = -1, job->release_at = 0;
      }
      if (job->phase == PIPE_DONE) continue;
      if ((job->phase == PIPE_FRONT || job->phase == PIPE_FINISH) && job->signal <= completed) {
        if (job->phase == PIPE_FRONT) pipe_issue(job);
        else {
          memcpy(pipe_logits + (size_t)r * VOCAB, job->saved[15].ptr, VOCAB * 4);
          job->phase = PIPE_DONE, remaining--;
          continue;
        }
      }
      if (job->phase == PIPE_READ) {
        int all = 1;
        for (int k = 0; k < TOPK; k++) all &= cache_ready(job->layer, job->experts[k]);
        if (all) ready[nr] = job, kind[nr++] = 0;
      } else if (job->phase == PIPE_DOWN) {
        ready[nr] = job, kind[nr++] = 1;
        pending += job->signal > completed;
      } else {
        pending++;
        if (job->signal < wait) wait = job->signal;
      }
    }
    if ((uint64_t)pending > pipe_stats.max_inflight) pipe_stats.max_inflight = pending;
    if (!remaining) break;
    if (!nr) {
      double start = now();
      if (wait != UINT64_MAX) gpu_wait_signal(wait), pipe_stats.gpu_wait += now() - start;
      else {
        gpu_keep_clock(.001);
        cache_wait_progress(epoch);
        double elapsed = now() - start;
        pipe_stats.read_wait += elapsed, cache_stats.read_s += elapsed;
      }
      continue;
    }
    if (batch == PIPE_BATCHES) die("pipeline stage count exceeded its bound");
    size_t offset = (size_t)batch++ * BATCH_ROWS * sizeof(MoeTask);
    MoeTask *tasks = (MoeTask *)(pipe_tasks.ptr + offset);
    uint32_t groups = 0;
    int mixed_kind = 0, mixed_layer = 0;
    gpu_begin();
    for (int i = 0; i < nr; i++) {
      PipeJob *job = ready[i];
      pipe_bind(job, live);
      size_t entries = (size_t)job->layer * FEW * 256;
      if (!kind[i]) {
        cache_wait(job->layer, job->experts, TOPK);
        const int *rt = (const int *)RT.ptr;
        fill_entries(job->layer, entries, rt, (const float *)rt + 8, NULL, 0);
      } else {
        // A first-use mapping might not be committed to the residency set
        // yet. This new encoder must declare the down stage's resources too.
        for (int k = 0; k < TOPK; k++) {
          uint32_t at[3]; int fresh;
          const Buf *b = cache_get(job->layer, job->experts[k], at, &fresh);
          if (fresh) gpu_use(b);
        }
      }
      tasks[i] = (MoeTask){ETAB.gpu + entries, XQ.gpu, MH.gpu, SHO.gpu, MF.gpu, H1.gpu, HS.gpu,
                            groups, kind[i] ? DIM / 4 : TOPK * (INTER / 32), kind[i], TOPK};
      if (nr > 1) {
        gpu_use(&ETAB);
        if (!kind[i]) gpu_use(&XQ), gpu_use_rw(&MH);
        else {
          gpu_use(&MH), gpu_use(&SHO), gpu_use(&MF), gpu_use(&H1), gpu_use_rw(&HS);
        }
      }
      groups += tasks[i].groups;
      mixed_kind |= kind[i] != kind[0];
      mixed_layer |= job->layer != ready[0]->layer;
    }
    if (nr == 1) {
      pipe_bind(ready[0], live);
      pipe_single(kind[0], ready[0]->layer, (size_t)ready[0]->layer * FEW * 256);
    } else {
      uint32_t n = nr;
      gpu_kern(K_pipe), gpu_buf(0, &pipe_tasks, offset), gpu_bytes(1, &n, 4);
      gpu_go(groups, 1, 1, 256, 1, 1);
    }
    for (int i = 0; i < nr; i++) if (kind[i] && ready[i]->layer == LAYERS - 1) {
      pipe_bind(ready[i], live);
      hc_pre_norm(1, &HS, 0, &MF, 0, &final_norm);
      head_out(1, &logits);
    }
    uint64_t finished = gpu_signal();
    gpu_commit();
    pipe_stats.batches++, pipe_stats.tasks += nr;
    pipe_stats.mixed_stages += mixed_kind, pipe_stats.mixed_layers += mixed_layer;
    for (int i = 0; i < nr; i++) {
      PipeJob *job = ready[i];
      job->signal = finished;
      // The down stage only consumes GPU data. Its up stage is already
      // earlier on this queue, and declared resource dependencies order
      // their accesses. Do not wait for a CPU-visible completion here.
      if (!kind[i]) job->phase = PIPE_DOWN;
      else {
        job->release_at = finished;
        if (++job->layer < LAYERS) pipe_front(job, live);
        else job->phase = PIPE_FINISH;
      }
    }
  }
  gpu_sync();
  for (int k = 0; k < PIPE_BUFFERS; k++) *live[k] = original[k];
  bind(original_state), memcpy(history, original_state->history, sizeof history);
  runahead_route_only = 0;
  cache_metrics_end();
  stats_add(&decode_stats, &before);
  decode_rows += count, decode_gpu_s += gpu_busy - gpu_before, decode_s += now() - began;
  return pipe_logits;
}
