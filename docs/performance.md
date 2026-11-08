# Performance measurement

Keep measurements separate from runtime source and evaluation inputs.
Drivers print results by default; save an aggregate only when an output
path is explicitly supplied. Use an ignored directory such as
`build/results/`. Do not commit raw logs, model replies or a continuing
archive of experimental runs.

## Comparable runs

Every result must identify the model, weight format, global model-memory and KV budgets, actual
expert-cache size, peak process footprint, context, prompt, generated-token count, request count,
sampling settings, drafts, approximations, binary and measurement scope.
Use a real workload appropriate to the proposed change. A small kernel
check establishes only that kernel's behavior.

Run one model process at a time on AC power. Record competing GPU/CPU work,
physical memory availability and cache adaptation. Count reclaimable file
pages as available. For `vm_stat` text, add "Pages free" and "File-backed
pages": the printed free count already excludes speculative pages.
In-memory compression is allowed; new swap-outs
invalidate a run. A warm file cache changes the meaning of an I/O result.

Alternate control and candidate runs. A short run can reject a large
regression; a small apparent gain needs enough repetitions and workload
length to distinguish it from drift. Do not time repeated identical greedy
requests as representative performance: they can replay the same expert
working set and produce misleading cache hits.

## Metrics

- End-to-end latency includes loading when the model is cold, prefill and
  generation. Report those phases separately as well.
- Decode throughput counts emitted tokens over the chosen decode interval.
- Multi-request throughput counts all emitted tokens over the same wall
  interval; also report each request's latency and throughput.
- Computed rows include rejected speculative work. They are not emitted
  tokens and should not be used interchangeably with them.
- GPU time and waiting for reads can overlap. Their sum is not a wall-time
  breakdown.
- Logical weight bytes divided by time do not prove physical DRAM or SSD
  saturation. Use hardware counters for that claim.

For prefetching, distinguish already-resident hits, newly useful reads
completed before demand, and pending reads that were late. Count wasted
reads and eviction damage. Aggregate cache hit rate alone does not measure
predictor benefit or timeliness.

## Drivers

`MYDS4_PROFILE=stages` records GPU timestamps at compute-encoder boundaries.
It keeps the command-buffer submission points and resolves samples at GPU
synchronization points. Matrix-vector results are grouped by kernel and
matrix dimensions. The sample buffer holds 2,048 encoders between collections.
This mode adds encoder and counter overhead; use it to locate expensive work,
and measure throughput with the variable unset. It has been checked against
the expert fixtures and Flash-Next output. Command-buffer schedules that keep
future work uncommitted need separate validation before their stage totals
can be used.

The existing `decode` and `prefill` profiling modes wait after each dispatch.
Their throughput does not represent ordinary inference.

| Driver | Scope |
| --- | --- |
| `benchmarks/compare.py` | Matched CLI control/candidate generations and output checks. |
| `benchmarks/concurrency.py` | HTTP throughput and latency on Harness requests or local audio fixtures. |
| `benchmarks/turns.py` | A conversation followed by a longer tool-like input. |
| `benchmarks/prefill.py` | CED prefill and continuation comparisons. |
| `benchmarks/decode.c` | Batched model-row timing. |
| `benchmarks/products.c` | Common matrix-product correctness and warmed vector-kernel timing; these are not model throughput or physical memory-bandwidth measurements. |
| `benchmarks/expert_reads.py` | Expert-sized reads from checkpoint storage, with file-cache caveats. |
| `benchmarks/evaluate.py` | Bounded public-dataset and conversation diagnostics. |

Use each driver's `--help` for required paths and supported modes. Supply
checkpoint and workload paths explicitly. Private Harness conversations
and generated text stay in memory; reports should contain measurements
and digests rather than the conversation itself.

For Transcribe, use `--audio FILE` in place of `--harness-session`, and
`--draft-model DIR` with `--draft N` to measure DFlash2. Set
`--min-output-tokens` to suit transcripts that end before 128 tokens.
`--kv` sets the shared KV budget. Audio comparisons use sampled requests
at temperature and top-p 1; the production transcription default remains
greedy. Count the entire generation interval, including the drafter;
the target's `model_decode.seconds` excludes separate proposal work.

GSM8K and HumanEval inputs live under `benchmarks/datasets/`, with their
licenses and source metadata. HumanEval syntax or entry-point checks are
not functional pass@1. A diagnostic subset is not a full benchmark score,
and numerical similarity is not proof of reasoning or coding quality.
