// Decode independent prepared states through the model API. Prompts arrive
// over stdin; only timings, counters and token digests leave the process.
#include <pthread.h>
#include <sys/qos.h>
#include "../../src/engine/model.c"

typedef struct {
  State *state;
  int *prompt, *output, length, pos, token, generated;
  double prefill_seconds, finished_seconds;
  Sampler sampler;
} SessionJob;

static uint32_t input_number(void) {
  unsigned char bytes[4];
  if (fread(bytes, 1, sizeof bytes, stdin) != sizeof bytes) die("short session input");
  return (uint32_t)bytes[0] | (uint32_t)bytes[1] << 8 | (uint32_t)bytes[2] << 16 | (uint32_t)bytes[3] << 24;
}

static double argument(const char *text, double low, double high, int integer) {
  char *end;
  double value = strtod(text, &end);
  if (end == text || *end || !isfinite(value) || value < low || value > high || (integer && value != floor(value)))
    die("invalid session-probe argument");
  return value;
}

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
static void token_digest(const int *tokens, int count, char hex[65]) {
  CC_SHA256_CTX context;
  unsigned char digest[CC_SHA256_DIGEST_LENGTH];
  CC_SHA256_Init(&context);
  for (int i = 0; i < count; i++) {
    uint32_t token = (uint32_t)tokens[i];
    unsigned char bytes[4] = {token, token >> 8, token >> 16, token >> 24};
    CC_SHA256_Update(&context, bytes, sizeof bytes);
  }
  CC_SHA256_Final(digest, &context);
  for (int i = 0; i < CC_SHA256_DIGEST_LENGTH; i++) sprintf(hex + 2 * i, "%02x", digest[i]);
}
#pragma clang diagnostic pop

int main(int argc, char **argv) {
  if (argc < 9 || argc > 10)
    die("usage: session-probe CHECKPOINT CONTEXT TOKENS CACHE RESERVE BATCH SHARE PRUNE [EXPERT_GGUF]");
  alarm(1800);
  shutdown_watch();
  if (pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0)) die("cannot set model thread priority");
  qos_class_t qos;
  int relative_priority;
  if (pthread_get_qos_class_np(pthread_self(), &qos, &relative_priority) || qos != QOS_CLASS_USER_INTERACTIVE)
    die("model thread priority does not match the server");
  int context = (int)argument(argv[2], 16384, 131072, 1);
  int limit = (int)argument(argv[3], 128, 8192, 1);
  double cache = argument(argv[4], 20, 80, 0), reserve_gib = argument(argv[5], 4, 16, 0);
  int batch = (int)argument(argv[6], 1, BLOCK + 1, 1);
  int share = (int)argument(argv[7], 0, 1, 1), prune = (int)argument(argv[8], 0, 1, 1);
  uint32_t count = input_number();
  if (!count || count > BLOCK + 1) die("session count outside the supported range");
  SessionJob jobs[BLOCK + 1] = {0};
  tok_load(argv[1]);
  for (uint32_t i = 0; i < count; i++) {
    uint32_t bytes = input_number();
    if (!bytes || bytes > (2u << 20)) die("session prompt exceeds 2 MiB");
    char *text = xmalloc((size_t)bytes + 1);
    if (fread(text, 1, bytes, stdin) != bytes || memchr(text, 0, bytes)) die("invalid session prompt bytes");
    if (strstr(text, "<｜deepseek_image｜>") || memchr(text, '\x1d', bytes)) die("session probe requires text prompts");
    jobs[i].prompt = xmalloc((size_t)context * sizeof(int));
    jobs[i].length = tok_encode_chat(text, jobs[i].prompt, context);
    if (jobs[i].length < 1 || jobs[i].length + limit > context) die("session prompt exceeds the context budget");
    jobs[i].output = xmalloc((size_t)limit * sizeof(int));
    jobs[i].sampler = (Sampler){.temp = 1, .top_p = 1};
    sampler_seed(&jobs[i].sampler, 12345);
    free(text);
  }
  if (fgetc(stdin) != EOF) die("trailing session input");
  Options options = {.ctx = context, .budget_gb = cache, .reserve_gb = reserve_gib,
                     .quiet = 1, .dump_fd = -1, .routes_fd = -1, .expert_batch = share,
                     .prefill_prune = prune, .experts = argc == 10 ? argv[9] : NULL};
  model_load(argv[1], &options);
  int eos = tok_find("<｜end▁of▁sentence｜>");
  double prefilling = now();
  for (uint32_t i = 0; i < count; i++) {
    if (shutdown_requested()) return 130;
    SessionJob *job = &jobs[i];
    job->state = model_state_new();
    model_state_use(job->state);
    double began = now();
    job->token = sample_token(model_forward(job->prompt, job->length, 0), VOCAB, &job->sampler);
    job->prefill_seconds = now() - began;
    job->pos = job->length;
    free(job->prompt), job->prompt = NULL;
    fprintf(stderr, "session probe: prompt %u/%u, %d tokens, %.2f seconds\n", i + 1, count, job->length, job->prefill_seconds);
  }
  double prefill_seconds = now() - prefilling;
  uint64_t min_cache = cache_bytes(), max_cache = min_cache, start_cache = min_cache;
  uint64_t first_rows = decode_rows, first_pairs = expert_batch_pairs, first_groups = expert_batch_groups;
  uint64_t first_calls = expert_batch_calls, batches[BLOCK + 2] = {0};
  int schedule = 0;
  uint64_t ready_groups = 0, ready_rows = 0, after_wait_groups = 0, head_batches = 0, head_rows = 0;
#ifdef MYDS4_RUNAHEAD
  schedule = runahead_enabled;
  ready_groups = runahead_group_count, ready_rows = runahead_group_rows, after_wait_groups = runahead_after_wait_groups;
  head_batches = runahead_head_batches, head_rows = runahead_head_rows;
#endif
  CacheStats first = decode_stats;
  double first_decode = decode_s, first_gpu = decode_gpu_s, began = now();
  int total = 0;
  while (!shutdown_requested()) {
    State *states[BLOCK + 1];
    int ids[BLOCK + 1], tokens[BLOCK + 1], positions[BLOCK + 1], rows = 0;
    for (uint32_t i = 0; i < count && rows < batch; i++) {
      SessionJob *job = &jobs[i];
      if (job->token == eos || job->generated == limit) continue;
      states[rows] = job->state, tokens[rows] = job->token, positions[rows] = job->pos, ids[rows++] = i;
    }
    if (!rows) break;
    const float *lg;
    if (rows == 1) model_state_use(states[0]), lg = model_forward(tokens, 1, positions[0]);
    else lg = model_forward_rows(states, tokens, positions, rows);
    batches[rows]++;
    for (int r = 0; r < rows; r++) {
      SessionJob *job = &jobs[ids[r]];
      job->output[job->generated++] = job->token;
      job->token = sample_token(lg + (size_t)r * VOCAB, VOCAB, &job->sampler);
      job->pos++, total++;
      if (job->generated == limit || job->token == eos) job->finished_seconds = now() - began;
    }
    uint64_t bytes = cache_bytes();
    if (bytes < min_cache) min_cache = bytes;
    if (bytes > max_cache) max_cache = bytes;
  }
  if (shutdown_requested()) return 130;
  double seconds = now() - began;
  uint64_t selected = (decode_stats.row_own - first.row_own) + (decode_stats.row_ahead - first.row_ahead) +
                      (decode_stats.row_late - first.row_late) + (decode_stats.row_ssd - first.row_ssd);
  uint64_t pairs = expert_batch_pairs - first_pairs, groups = expert_batch_groups - first_groups;
#ifdef MYDS4_RUNAHEAD
  ready_groups = runahead_group_count - ready_groups, ready_rows = runahead_group_rows - ready_rows;
  after_wait_groups = runahead_after_wait_groups - after_wait_groups;
  head_batches = runahead_head_batches - head_batches, head_rows = runahead_head_rows - head_rows;
#endif
  printf("{\"session_model\":{\"thread_qos\":\"user_interactive\",\"sessions\":%u,\"batch_limit\":%d,\"prefetch_budget_units\":%d,\"usage_credit\":%d,"
         "\"generated_tokens\":%d,\"decode_wall_seconds\":%.6f,\"aggregate_tokens_per_second\":%.6f,"
         "\"prefill_seconds\":%.6f,\"model_decode_seconds\":%.6f,\"main_queue_gpu_seconds\":%.6f,"
         "\"requested_expert_gb\":%.6f,\"read_wait_seconds\":%.6f,\"computed_rows\":%llu,\"selected_experts\":%llu,"
         "\"unique_expert_requests\":%llu,\"grouped_calls\":%llu,\"grouped_pairs\":%llu,\"weight_groups\":%llu,"
         "\"weight_load_groups\":%llu,\"cache_start_bytes\":%llu,\"cache_min_bytes\":%llu,\"cache_max_bytes\":%llu,"
         "\"slots_released\":%llu,\"substituted\":%llu,\"runahead\":%d,\"ready_groups\":%llu,\"ready_rows\":%llu,"
         "\"groups_after_wait\":%llu,\"head_batches\":%llu,\"head_rows\":%llu,"
         "\"nonforecast_slot_selections\":%llu,\"on_time_prefetch_selections\":%llu,"
         "\"late_prefetch_selections\":%llu,\"demand_selections\":%llu,\"prefetch_reads\":%llu,"
         "\"prefetch_first_uses\":%llu,\"prefetch_late_first_uses\":%llu,\"cancelled_forecasts\":%llu,\"batches\":[",
         count, batch, batch_prefetch_limit, batch_usage_credit, total, seconds, total / seconds,
         prefill_seconds, decode_s - first_decode, decode_gpu_s - first_gpu,
         (decode_stats.bytes - first.bytes) / 1e9, decode_stats.read_s - first.read_s,
         (unsigned long long)(decode_rows - first_rows), (unsigned long long)selected,
         (unsigned long long)(decode_stats.row_unique - first.row_unique),
         (unsigned long long)(expert_batch_calls - first_calls), (unsigned long long)pairs, (unsigned long long)groups,
         (unsigned long long)(selected - pairs + groups), (unsigned long long)start_cache,
         (unsigned long long)min_cache, (unsigned long long)max_cache, (unsigned long long)cache_stats.given_back,
         (unsigned long long)(decode_stats.row_sub - first.row_sub), schedule,
         (unsigned long long)ready_groups, (unsigned long long)ready_rows, (unsigned long long)after_wait_groups,
         (unsigned long long)head_batches, (unsigned long long)head_rows,
         (unsigned long long)(decode_stats.row_own - first.row_own),
         (unsigned long long)(decode_stats.row_ahead - first.row_ahead),
         (unsigned long long)(decode_stats.row_late - first.row_late),
         (unsigned long long)(decode_stats.row_ssd - first.row_ssd),
         (unsigned long long)(decode_stats.prefetches - first.prefetches),
         (unsigned long long)(decode_stats.prefetch_hits - first.prefetch_hits),
         (unsigned long long)(decode_stats.late - first.late), (unsigned long long)(decode_stats.cancelled - first.cancelled));
  for (int n = 1; n <= BLOCK + 1; n++) printf("%s%llu", n > 1 ? "," : "", (unsigned long long)batches[n]);
  printf("],\"outputs\":[");
  for (uint32_t i = 0; i < count; i++) {
    char digest[65];
    SessionJob *job = &jobs[i];
    token_digest(job->output, job->generated, digest);
    printf("%s{\"prompt_tokens\":%d,\"generated_tokens\":%d,\"prefill_seconds\":%.6f,\"finished_seconds\":%.6f,"
           "\"token_sha256\":\"%s\"}", i ? "," : "", job->length, job->generated, job->prefill_seconds,
           job->finished_seconds, digest);
    free(job->output);
  }
  printf("]}}\n");
  return total < 1;
}
