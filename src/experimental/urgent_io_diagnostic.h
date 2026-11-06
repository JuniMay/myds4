// Included by io.c. These counters instrument urgent read callbacks; they
// do not measure physical SSD queue depth or uninstrumented throughput.
enum { DIAGNOSTIC_BINS = 512, DIAGNOSTIC_CONCURRENCY = 129 };
#define DIAGNOSTIC_BIN_NS 50000ull
typedef struct {
  _Atomic uint64_t count, total, maximum, bins[DIAGNOSTIC_BINS];
} DiagnosticLatency;
typedef struct {
  DiagnosticLatency queue, service;
  _Atomic uint64_t bytes, active_peak, pending_peak, active_starts[DIAGNOSTIC_CONCURRENCY];
} DiagnosticReads;
static DiagnosticReads diagnostic_reads[2];
static _Atomic unsigned diagnostic_active;
static int diagnostic_phase;
static int diagnostic_request_active;
static uint64_t diagnostic_reclaim_ns, diagnostic_reclaim_requests, diagnostic_reclaim_total_ns;
static uint64_t diagnostic_delayed_promotions, diagnostic_delayed_pieces;
static struct {
  int active, layer, count, ids[TOPK];
  uint64_t began, requests, completed, unmatched, pending[TOPK + 1], last_worker[3];
  DiagnosticLatency arrival, host_wait;
} diagnostic_bundle;

static void diagnostic_max(_Atomic uint64_t *value, uint64_t n) {
  uint64_t old = atomic_load_explicit(value, memory_order_relaxed);
  while (old < n && !atomic_compare_exchange_weak_explicit(value, &old, n,
          memory_order_relaxed, memory_order_relaxed)) {}
}

static void diagnostic_add(DiagnosticLatency *s, uint64_t ns) {
  uint64_t bin = ns / DIAGNOSTIC_BIN_NS;
  if (bin >= DIAGNOSTIC_BINS) bin = DIAGNOSTIC_BINS - 1;
  atomic_fetch_add_explicit(&s->count, 1, memory_order_relaxed);
  atomic_fetch_add_explicit(&s->total, ns, memory_order_relaxed);
  atomic_fetch_add_explicit(&s->bins[bin], 1, memory_order_relaxed);
  diagnostic_max(&s->maximum, ns);
}

static uint64_t diagnostic_read_begin(Slot *s, int piece, unsigned pending) {
  uint64_t queued = s->diagnostic_queued[piece];
  if (!queued) return 0; // permits fixtures with manually completed groups
  DiagnosticReads *d = &diagnostic_reads[s->diagnostic_phase[piece]];
  uint64_t began = metrics_ns();
  diagnostic_add(&d->queue, began - queued);
  unsigned active = atomic_fetch_add_explicit(&diagnostic_active, 1, memory_order_relaxed) + 1;
  diagnostic_max(&d->active_peak, active);
  diagnostic_max(&d->pending_peak, pending);
  unsigned bin = active < DIAGNOSTIC_CONCURRENCY ? active : DIAGNOSTIC_CONCURRENCY - 1;
  atomic_fetch_add_explicit(&d->active_starts[bin], 1, memory_order_relaxed);
  return metrics_ns();
}

static void diagnostic_read_end(Slot *s, int piece, uint64_t began) {
  if (!began) return;
  uint64_t ended = metrics_ns();
  atomic_fetch_sub_explicit(&diagnostic_active, 1, memory_order_relaxed);
  DiagnosticReads *d = &diagnostic_reads[s->diagnostic_phase[piece]];
  diagnostic_add(&d->service, ended - began);
  atomic_fetch_add_explicit(&d->bytes, s->piece[piece].len, memory_order_relaxed);
}

void cache_io_diagnostic_decode(void) { diagnostic_phase = 1; }

static void diagnostic_request_begin(int layer, const int *experts, int n, int urgent) {
  diagnostic_request_active = 0, diagnostic_reclaim_ns = 0;
  if (!diagnostic_phase || !urgent || layer >= LAYERS || n < 1 || n > TOPK) return;
  diagnostic_request_active = 1;
  if (diagnostic_bundle.active) diagnostic_bundle.unmatched++;
  diagnostic_bundle.active = 1, diagnostic_bundle.layer = layer, diagnostic_bundle.count = n;
  diagnostic_bundle.began = metrics_ns();
  memcpy(diagnostic_bundle.ids, experts, (size_t)n * sizeof *experts);
  int pending = 0;
  for (int k = 0; k < n; k++) {
    int i = where[layer][experts[k]];
    pending += i < 0 || slot_reading(i);
  }
  diagnostic_bundle.requests++, diagnostic_bundle.pending[pending]++;
}

static void diagnostic_promote(Slot *s) {
  if (!diagnostic_request_active || !diagnostic_reclaim_ns) return;
  int unclaimed = 0;
  for (int k = 0; k < s->npiece; k++) unclaimed += !atomic_load(&s->taken[k]);
  if (unclaimed) diagnostic_delayed_promotions++, diagnostic_delayed_pieces += unclaimed;
}

static void diagnostic_request_return(void) {
  if (diagnostic_request_active && diagnostic_reclaim_ns)
    diagnostic_reclaim_requests++, diagnostic_reclaim_total_ns += diagnostic_reclaim_ns;
  diagnostic_request_active = 0;
}

static void diagnostic_request_end(int layer, const int *experts, int n, uint64_t wait_started) {
  if (!diagnostic_bundle.active || layer != diagnostic_bundle.layer || n != diagnostic_bundle.count ||
      memcmp(experts, diagnostic_bundle.ids, (size_t)n * sizeof *experts)) return;
  uint64_t ready = diagnostic_bundle.began;
  int worker = 0;
  for (int k = 0; k < n; k++) {
    uint64_t t = atomic_load_explicit(&slots[where[layer][experts[k]]].diagnostic_ready, memory_order_acquire);
    if (t > ready) {
      ready = t;
      worker = 1 + atomic_load_explicit(&slots[where[layer][experts[k]]].diagnostic_ready_urgent, memory_order_relaxed);
    }
  }
  diagnostic_add(&diagnostic_bundle.arrival, ready - diagnostic_bundle.began);
  diagnostic_add(&diagnostic_bundle.host_wait, metrics_ns() - wait_started);
  diagnostic_bundle.active = 0, diagnostic_bundle.completed++;
  diagnostic_bundle.last_worker[worker]++;
}

static double diagnostic_percentile(DiagnosticLatency *s, unsigned percent) {
  uint64_t count = atomic_load(&s->count), cumulative = 0;
  if (!count) return 0;
  uint64_t target = (count * percent + 99) / 100;
  for (int k = 0; k < DIAGNOSTIC_BINS - 1; k++) {
    cumulative += atomic_load(&s->bins[k]);
    if (cumulative >= target) return (k + 1) * DIAGNOSTIC_BIN_NS / 1e3;
  }
  return atomic_load(&s->maximum) / 1e3; // conservative upper bound for the overflow bin
}

static void diagnostic_latency_report(FILE *out, DiagnosticLatency *s) {
  uint64_t n = atomic_load(&s->count);
  fprintf(out, "{\"count\":%llu,\"mean_us\":%.3f,\"p50_upper_us\":%.3f,\"p95_upper_us\":%.3f,"
          "\"p99_upper_us\":%.3f,\"max_us\":%.3f,\"overflow_count\":%llu}",
          (unsigned long long)n, n ? atomic_load(&s->total) / (1e3 * n) : 0,
          diagnostic_percentile(s, 50), diagnostic_percentile(s, 95), diagnostic_percentile(s, 99),
          atomic_load(&s->maximum) / 1e3, (unsigned long long)atomic_load(&s->bins[DIAGNOSTIC_BINS - 1]));
}

static void diagnostic_report(FILE *out) {
  fprintf(out, "{\"demand_promotion_diagnostic\":{\"requests_waiting_for_reclamation\":%llu,"
          "\"reclamation_wait_us\":%.3f,\"promotions_delayed_by_reclamation\":%llu,\"pieces_delayed\":%llu}}\n",
          (unsigned long long)diagnostic_reclaim_requests, diagnostic_reclaim_total_ns / 1e3,
          (unsigned long long)diagnostic_delayed_promotions, (unsigned long long)diagnostic_delayed_pieces);
  fprintf(out, "{\"urgent_io_diagnostic\":{\"instrumented\":true,\"histogram_bin_us\":50,"
          "\"overflow_from_us\":25550,\"active_at_report\":%u,\"phases\":[", atomic_load(&diagnostic_active));
  for (int phase = 0; phase < 2; phase++) {
    DiagnosticReads *d = &diagnostic_reads[phase];
    fprintf(out, "%s{\"phase\":\"%s\",\"bytes\":%llu,\"active_peak\":%llu,\"pending_peak\":%llu,\"queue\":",
            phase ? "," : "", phase ? "decode" : "startup_prefill", (unsigned long long)atomic_load(&d->bytes),
            (unsigned long long)atomic_load(&d->active_peak), (unsigned long long)atomic_load(&d->pending_peak));
    diagnostic_latency_report(out, &d->queue);
    fprintf(out, ",\"service\":");
    diagnostic_latency_report(out, &d->service);
    fprintf(out, ",\"active_at_start_histogram\":[");
    for (int k = 0; k < DIAGNOSTIC_CONCURRENCY; k++)
      fprintf(out, "%s%llu", k ? "," : "", (unsigned long long)atomic_load(&d->active_starts[k]));
    fprintf(out, "]}");
  }
  fprintf(out, "],\"selected_bundles\":{\"requests\":%llu,\"completed\":%llu,\"unmatched\":%llu,"
          "\"pending_at_request_histogram\":[", (unsigned long long)diagnostic_bundle.requests,
          (unsigned long long)diagnostic_bundle.completed, (unsigned long long)diagnostic_bundle.unmatched);
  for (int k = 0; k <= TOPK; k++) fprintf(out, "%s%llu", k ? "," : "", (unsigned long long)diagnostic_bundle.pending[k]);
  fprintf(out, "],\"last_ready_worker\":{\"already_complete\":%llu,\"background\":%llu,\"urgent\":%llu},\"last_selected_io_ready\":",
          (unsigned long long)diagnostic_bundle.last_worker[0], (unsigned long long)diagnostic_bundle.last_worker[1],
          (unsigned long long)diagnostic_bundle.last_worker[2]);
  diagnostic_latency_report(out, &diagnostic_bundle.arrival);
  fprintf(out, ",\"host_wait_after_setup\":");
  diagnostic_latency_report(out, &diagnostic_bundle.host_wait);
  fprintf(out, "},\"scope\":\"Urgent callback read intervals, not physical SSD queue depth. Phase follows urgent submission. "
          "Readiness is timestamped before group notification; host wait can overlap GPU work. "
          "Histogram values are upper bounds. Instrumented timing is not a throughput result.\"}}\n");
}
