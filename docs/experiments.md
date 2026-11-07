# Removed experiments

This document records the experiments whose code was removed from myds4's
main branch on 2026-11-06. Each experiment was a study: a compile-time
switch, a research build, a runtime option, a probe or a check that tested
one idea for making DeepSeek-V4.1-Flash faster on this machine. The
experiments here were removed because their measurements showed no useful
gain, or because the user decided that they did not help (`--draft-trim`,
`--expert-batch`, `--gpu-route` and the priority-read studies). One section
records two policies that ran only in study builds that were never merged
into the main branch; their measurement records remain, and that section
says so.

Each section explains the problem the experiment addressed, how it worked,
how it was measured, what the measurements showed, why it did not help and
what was removed. The aim is that a later reader can understand, and if
needed rebuild, each experiment without the old code.

The measurement records remain in `data/benchmarks/`. Each section names
the records it draws on; they add detail, such as every run's settings and
counters. The README and the design, usage, server and performance
documents describe only the code that remains, and point here for these
experiments. Two studies remain in the code: causal verification, which is
under evaluation, and the prediction probe, a diagnostic capture tool. The
design and performance documents describe them.

## Background

### The machine and the model

The measurements ran on an Apple M5 Max with 18 CPU cores, 128 GiB of
unified memory (memory shared by the CPU and the GPU) and a 2 TB internal
SSD, under macOS 27.0.1.

DeepSeek-V4.1-Flash-1017 (`deepseek-ai/DeepSeek-V4.1-Flash-1017`) has 40
layers. Each layer has 384 routed experts and a router that selects 6 of
them for each token. An expert is a small feed-forward network (FFN) of
three matrices. In the checkpoint's own format one expert takes 18.8 MB,
and all experts take 269 GB, so they do not fit in memory. myds4 reads
experts from the SSD into a cache in RAM. The SSD reads about 14.1 GB/s,
so one expert takes about 1.34 ms to read. With every expert in memory, a
token would need about 25 ms of GPU time.

### Terms

| Term | Meaning |
| --- | --- |
| MXFP4 | The checkpoint's own 4-bit expert format. "MXFP4" in a result means the checkpoint's experts. |
| IQ2 | The routed experts of unsloth's `UD-IQ2_XXS` GGUF build (`unsloth/DeepSeek-V4.1-Flash-1017-GGUF`). Its experts take 7.6 to 12.1 MB. Its codes change the model's output, so IQ2 results are compared only with IQ2 results. |
| Expert cache, slot | The RAM that holds experts. A slot holds one expert. A fixed cache size (for example "56 GiB") is the budget of all slots. |
| Demand read | A read of a selected expert that is not in memory. The layer waits for it. |
| Forecast, prefetch, read-ahead | A read started from a prediction, before the layer's router has selected its experts. Three predictions exist: from the residual part of the layer's own input, before its attention ("pc"); from the next layer's router applied one layer earlier ("h1"); and two layers earlier ("h2"). Each reads its top 3 uncached experts while fewer than 6 forecast reads per row are in flight. |
| Refinement | A later forecast for a layer cancels that layer's forecast reads that no reader has started and that are not in its own top 6. |
| Retention score | Each slot's eviction score. Each request of the slot's expert adds a weight. The weight grows every forward pass so that older credit halves after 50 passes. The decode victim is the eligible slot with the lowest score. |
| Piece, reader | An expert is read in pieces of 2 MiB. Four background reader threads read forecast pieces. Demand pieces go to Grand Central Dispatch (GCD, the system's thread pool) at the user-interactive priority. Background readers start a piece only while no demand piece is in flight. |
| Decode, prefill | Decoding produces one token per forward pass. Prefill runs a prompt's tokens layer by layer in batches of up to 2,048 rows. |
| Command buffer | A unit of GPU work that the CPU encodes and then commits (submits). |
| Front | Everything of a decoded layer before its routed experts: the attention, the FFN's mixing step, the router, then the shared expert. The router signals the CPU through a shared event. |
| Split path | When some selected experts are still being read, the experts in memory run first. Each writes its part of each row's sum to a buffer (`PARTS`). A second command buffer later runs the late experts and adds all parts in expert order (`moe_parts`), so the result is bit for bit that of one fused pass. |
| DSpark, draft | The checkpoint's draft model: three small stages that propose up to 5 next tokens. |
| Verification pass | One forward pass of the main model over the current token and its proposed tokens. A proposal is kept while it equals the token the main model samples. |
| Context copy (`--lookup`) | Proposals copied from earlier text in the context, verified the same way. |
| DSH request | A request reconstructed from a recorded DeepSeek Harness session, the agent client used with the server. Generated tool calls were never executed. |
| Control, candidate | A comparison usually ran a control (the ordinary build), the candidate, then a closing control. |
| Logical read GB | The engine's count of requested expert-read bytes. It is not a measurement of physical SSD traffic. |
| Read wait | Time the CPU spent waiting for expert reads. GPU time and read wait overlap; they must not be added. |

### Workloads

| Name used here | Content |
| --- | --- |
| 13,772-token DSH request | Seven messages, a tool schema and a real tool-result round. Its reply ends after 308 tokens with MXFP4 and 295 tokens with IQ2 (temperature 1, top-p 1, seed 12345). |
| 51,895-token and 107,934-token DSH requests | Later requests of the same recorded session. With an 8,192-token limit, the 107,934-token generation stays inside the model's thinking section. |
| Four-state workload | Four related continuations of one DSH task with 13,014, 13,772, 19,129 and 22,481 input tokens. |
| Repository-review prompt | An 8,171- or 8,187-token request built from source excerpts. |
| Scheduler fixture | `data/benchmarks/long_scheduler.txt`: 1,229 prompt tokens that ask for a Rust task scheduler. |

### How the comparisons were made

Unless a section says otherwise, the runs used temperature 1, top-p 1 and
seed 12345, a fixed expert-cache budget and usually a 4 GiB reserve (the
memory the cache leaves to other programs). Each model run was a fresh
process, and runs ran one at a time. A guard required a quiet GPU before
each run and rejected runs whose cache shrank or whose memory was swapped
out. Desktop graphics activity was recorded; the GPU was never reserved
for the runs. Every listed comparison produced the same output bytes in
candidate and controls, unless a section says otherwise.

Paired controls usually differed by up to about 1%. Some pairs differed
by 2 to 3%, and a few by more. This document therefore treats a
difference within the controls' spread as no measured gain.

## Runtime options removed from the CLI and server

### `--draft-trim`: trimming the final DSpark stage

**The option as users saw it.** `--draft-trim` was a flag of `build/myds4`
and `build/myds4-server`, off by default. It required `--draft`
(`--draft-trim needs --draft`). The help text said: "with --draft: compute
only requested final-stage query/FFN rows; keep all five keys". The engine's
C interface carried it as the `draft_trim` field of `myds4_config`.

**Problem.** DSpark always computes five rows in each of its three stages:
the current token's row and four rows of a fixed noise token. A pass that
wants fewer than five drafts, for example `--draft 2`, does not use the
final stage's extra rows. Their queries, attention outputs and FFNs are
computed and then ignored.

**How it worked.** With the flag, the final stage computed its query
projections, attention, output projection and FFN for only the requested
rows. Its key/value projection still covered all five rows, so attention
saw the same keys. The first two stages still computed all five rows. The
head already used only the requested rows. Trimming split one paired
product (the query projection `wq_a` with the key/value projection `wkv`
in one dispatch) into two separate products of different row counts.

**How it was measured.**

- Exactness: direct captures with one, two, three and five drafts, in
  MXFP4 and IQ2, with draft caches of 128 and of 32 slots per stage. Every
  retained stage and head value, proposal ID and accepted prefix was
  compared bit for bit.
- Speed: a 16-run screen with `tools/bench/benchmark_suite.py` (GSM8K and
  HumanEval prompts and two context diagnostics), trimming on and off, two
  drafts, 32 draft slots per stage, 2026-11-01.

**Results.** All captures were bit-identical. All eight output comparisons
of the speed screen matched. The performance record gives aggregate decode
throughput for trimming on and off as 12.334 versus 12.340 tokens/s for
IQ2, and 5.768 versus 5.533 tokens/s for MXFP4. One MXFP4 context run had a
large additional read wait, with slightly fewer logical read bytes and
unchanged GPU time. No JSON record exists for this screen.

**Why it did not help.** The trimmed work is at most four rows of one
stage's per-row work. Drafting as a whole took 2.2 to 3.9 ms per token in
the 2026-10-31 IQ2 measurements. IQ2 showed no change. The MXFP4
difference came with an unexplained read wait in one run and was not
repeatable evidence. The user removed the option as exact but without a
measurable gain.

**What was removed.** The option in `src/cli/main.c` (parsing, help text,
the `--draft` requirement), `server/src/main.rs`, the `draft_trim` fields
of `Options` (`src/engine/myds4.h`), `myds4_config` (`src/engine/engine.h`)
and `server/src/engine.rs`, and its validation in `src/engine/engine.c`.
In `src/engine/model.c`: the trimmed final-stage call in `draft_generate`,
the separate-products branch of `draft_stage` for fewer than five rows, and
the trimmed stream sizes of the prediction probe. Checks and scripts that
set the flag were changed to omit it.

### `--expert-batch`: sharing expert weight loads across rows

**The option as users saw it.** `--expert-batch` was a flag of
`build/myds4` and `build/myds4-server`, off by default: "share repeated
expert weight loads across verification rows (exact)". The engine's C
interface carried it as `myds4_config.expert_batch`, and the server's
`GET /v1/status` reported it as `expert_batch`. The diagnostic build
`build/myds4-expert-batch` (switch `MYDS4_EXPERT_BATCH`) turned it on
without the flag.

**Problem.** A verification pass runs up to sixteen rows, and concurrent
requests run one row per request. The rows often select the same expert.
The ordinary kernels run each row's experts separately, so a shared
expert's weights are loaded from memory and decoded once per row.

**How it worked.** After routing (and substitution, if enabled), the CPU
grouped each layer's (row, expert) pairs by expert, with up to three rows
per group (`expert_batch_plan` in `src/engine/expert_batch.h`). Grouped
kernels (`moe_up_rows`, `moe_down_rows`, and `gg_up_rows`/`gg_down_rows`
for the GGUF codes Q2_K, IQ1_S, IQ2_XXS, IQ2_XS, IQ2_S and IQ3_XXS) loaded
and decoded each group's weights once. Each row kept its own accumulator,
routing weight, lane assignment, order of additions, BF16 rounding and FP8
activation quantization. Down products went to `PARTS`, and `moe_parts`
summed each row's experts in expert order. A layer without a repeated
expert used the ordinary kernels. Ready experts ran before the SSD wait and
late experts after it, each phase with its own descriptors. The
descriptors took 480 KiB.

**How it was measured.**

- Kernels, synthetic resident weights: one to six rows, three sharing
  patterns, whole, split and empty-first read phases.
- Logits: all six verification rows against sequential decoding.
- Model runs: the repository-review prompt with two DSpark drafts, and
  concurrent DSH requests.

**Results.**

| Measurement | Ordinary | Shared loads | Record |
| --- | ---: | ---: | --- |
| Three rows with the same six experts, synthetic, ms | 0.709 | 0.380 | `expert_batch_results.json` |
| 8,171-token review, 1,024 tokens, two drafts, 20 GiB: decode seconds | 187.36, 188.68 | 186.25 | `expert_batch_results.json` |
| The same: main-queue GPU seconds | 35.98 | 31.37 | `expert_batch_results.json` |
| The same: read-wait seconds | 154.17, 154.99 | 153.99 | `expert_batch_results.json` |
| 8,187-token review, 256 tokens, two drafts, 56 GiB: tokens/s | 10.19, 10.00 | 10.25 | `shared_expert_runtime_results.json` |
| The same: GPU seconds | 9.16 to 9.72 | 8.12 | `shared_expert_runtime_results.json` |

All outputs matched, and the verification rows' logits had no differing
values. In the 56 GiB comparison, which also used 32 draft slots per stage
and final-stage trimming, read waiting stayed at 16.28 to 16.63 s. An earlier kernel design, whose loops depended on the runtime row
count, passed all 180 checks but was slower than the per-row kernels in
all 18 timed cases; the fixed-row-count kernels replaced it.

For concurrent requests, two DSH requests (13,772 and 51,895 tokens) at a
56 GiB cache decoded at about 10.51 aggregate tokens/s when served one
after the other. Decoded together, they reached 9.54 tokens/s without
shared loads and 9.53 with them (`session_throughput_results.json`;
rates exclude the logged prefill time). In the four-state workload,
215,520 row/expert selections became 187,155 distinct requests within
layer batches, and the GPU ran 187,716 weight-load groups: the three-row
group limit added only 561 groups, about 0.3%
(`session_four_model_results.json`).

**Why it did not help.** The model runs were bound by expert reads, not
by GPU work. In the 20 GiB run the CPU waited about 154 s for reads out of
187 s, and the 12.8% GPU saving changed the total time by about 0.9%. With
concurrent requests, little repeated weight traffic was left to remove.
The user removed the option as no help.

**What was removed.** The option in the CLI, the server, `myds4_config`,
`server/src/engine.rs` and `server/src/worker.rs` (the `expert_batch` field
of `GET /v1/status`); the switch `MYDS4_EXPERT_BATCH` and the targets
`build/myds4-expert-batch`, `build/expert-batch-check`
(`tools/checks/expert_batch_check.c`), `build/expert-batch-model-check` and
`build/expert-batch-model-control` (`tools/checks/expert_batch_model_check.c`);
the grouped kernels `moe_up_rows` and `moe_down_rows` in
`src/kernels/kernels.metal` and `gg_up_rows` and `gg_down_rows` in
`src/kernels/gguf.metal`; `src/engine/expert_batch.h` with its group
descriptors; and, in `src/engine/model.c`, `experts_rows`, the `EROWS`
descriptor buffer, the `expert_batch` exit record and the grouped branches
of `routed_rows` that served verification passes and several sequences
decoded together.

**How runahead changed.** Until this removal, the server's `--runahead`
turned `--expert-batch` on, and a group of ready requests at one layer ran
its experts with the grouped kernels. Now each request of such a group runs
its own experts' products, with its own weight loads; the group still
shares one command buffer and the next layer's front. Runahead's measured
gain (6.3% more decode throughput in the HTTP comparison) was taken with
shared loads on in both arms, so it does not measure the current form.

### `--gpu-route`: experts looked up by the GPU

**The option as users saw it.** `--gpu-route` was a flag of `build/myds4`
and `build/myds4-server`, off by default: "decoding: the GPU looks up each
layer's experts itself". It applied to a token decoded alone, not to
verification passes or concurrent requests. It could not be combined with
`--runahead`. The live view noted tokens whose experts the GPU had looked
up, because their cache counters were not collected.

**Problem.** A decoded layer normally waits for the CPU between the
router and the routed experts: the router signals the CPU, the CPU reads
the selection, requests the experts and encodes their products. When the
experts are already in memory, that round trip leaves the GPU idle.

**How it worked.** Each layer became one command buffer: the front, a
`resolve` kernel, the routing signal, the shared expert, then the routed
experts as `resolve` named them. Before committing a layer's buffer, the
CPU published a table with, for every expert of the layer, the GPU
addresses of its slot if it was read in full, 1 if it was still being
read, else 0. `resolve` computed the router's selection on the GPU, looked
the experts up in that table, and wrote their descriptors. If any selected
expert was not ready, all descriptors were null, the expert kernels did
nothing, and the CPU computed that layer's experts again in a command
buffer of its own (`fix_layer`). The CPU encoded each layer's buffer two
layers ahead. Until the next layer's table was published, no slot of the
published layer could be evicted. With `--substitute`, `resolve` applied
the substitution rule to the published table.

**How it was measured.** In two series, each against the ordinary
two-buffer path:

- 2026-10-31: IQ2 experts, a 40 GiB cache, 150 tokens of a story prompt and
  a Python prompt, runs alternated, other applications using 18 to 28% of
  the GPU. One pair of story runs that other GPU work slowed (GPU time 4.8
  to 5.4 s) is left out.
- 2026-11-04: MXFP4, the 51,895-token DSH request, 512 tokens, a 56 GiB
  cache.

**Results.**

| Measurement | Two-buffer path | `--gpu-route` |
| --- | ---: | ---: |
| Story, tokens/s (2026-10-31) | 20.86 | 20.97 |
| Python, tokens/s (2026-10-31) | 18.64 to 18.65 | 18.81 to 18.88 |
| GPU time, story; Python | 4.33; 4.32 to 4.33 s | 4.37; 4.37 to 4.39 s |
| GPU idle between command buffers per token, story; Python | 18.4; 24.1 ms | 17.9; 23.2 to 23.3 ms |
| Gaps over 100 µs between command buffers per token, story; Python | 16.4; 20.2 to 20.3 | 14.5; 18.3 to 18.4 |
| Layers computed again by the CPU per token, story; Python | | 16.5; 20.1 |
| 51,895-token DSH, tokens/s (2026-11-04) | 10.45, 10.43 | 10.26 |
| The same: logical read GB | 523.52, 524.04 | 529.04 |

In the two-buffer path, the gaps over 100 µs were about as many as the
layers that `--gpu-route` computed again because its table did not mark all
their experts ready. The texts and the logits of a 649-position perplexity
run were the same bit for bit in both paths, also with a 12 GiB cache. The
2026-10-31 runs predate a correction of the cache's readiness check; at
that time a completed read could still be counted as pending. The
2026-11-04 run had no competing compute job and a fixed cache; its record
is `gpu_route_night_results.json`.

**Why it did not help.** In the two-buffer path, a layer whose experts
are in memory already starts them within 100 µs of its front's end. The
measured reduction of GPU idle time was about 0.9 ms per token. In the
2026-10-31 runs, 16.5 to 20.1 of a token's 40 layers ran their experts
again on the CPU path; because those runs predate the readiness
correction, these counts do not show how many reads were really
unfinished. The later MXFP4 run was about 1.7% slower than the mean of
its controls. The user removed the option as no help.

**What was removed.** The option in the CLI, the server and
`myds4_config`. In `src/engine/model.c`: `step_spec`, `layer_spec`,
`encode_spec`, `fix_layer`, `resolve`, `spec_entries`, the per-layer rows
of routing and predictions (`gpu_step`, `slice`), the `RU` buffer,
`ready_tab` and the `spec_fixups` counter. The `resolve` kernel in
`src/kernels/kernels.metal`. In `src/engine/io.c`: `cache_table`,
`cache_publish`, `cache_unpublish`, the published-layer protection, and the
`unsupported_gpu_route_passes` and `unsupported_gpu_route_rows` counters of
the prefetch metrics. The live view's note about GPU-routed tokens. The
checks' GPU-routing cases: two of the prefetch-metrics check (now 54
checks), one each of the cache-policy and workspace checks (the workspace
check now has 20) and two of the demotion check.

## DSpark and verification studies

### Shared-uniform DSpark drafts

**Problem.** DSpark proposes each draft as its highest-probability token.
When the main model samples at temperature 1, its token often differs from
that choice, so drafts are rejected.

**How it worked.** The build `build/myds4-coupled-draft` (switch
`MYDS4_COUPLED_DRAFT`) sampled each draft on the CPU from DSpark's
probabilities, using a copy of the main sampler's random-generator state,
so that a draft used the same uniform random number the main model would
use at that position. Verification still drew each target token from the
original random stream and stopped at the first mismatch, so the output
was the same as ordinary decoding. Each draft position needed a GPU
synchronization and CPU sampling.

**How it was measured.** The 8,187-token review prompt, 256 tokens, two
drafts, 32 draft slots per stage, a 56 GiB cache, shared expert loads in
both arms. A separate probe (`build/draft-distribution-probe`) measured
how often the first draft could agree with the target's sample.

**Results.** Accepted drafts per pass rose from 0.99 to 1.07. Throughput
was 10.28 tokens/s against controls of 10.26 and 9.98. All 256 output
tokens matched (`coupled_draft_results.json`). The probe found a mean
target probability of 0.625 for the greedy first draft; a shared-uniform
sampler would agree with mean probability 0.687 at draft temperature 1,
and the best possible coupling 0.766. Draft temperature 0.75 gave 0.691
(`draft_distribution_results.json`). These are first-token statistics, not
estimates of multi-token acceptance.

**Why it did not help.** The 0.08 more accepted drafts per pass did not
outweigh the added synchronization and CPU sampling.

**What was removed.** The switch, `model_draft_coupled` and the sampled
branch of `draft_generate` in `src/engine/model.c`, the targets
`build/myds4-coupled-draft` and `build/draft-distribution-probe`, and
`tools/experimental/draft_distribution_probe.c`.

### Staged DSpark verification

**Problem.** Verifying five drafts in one pass computes five extra rows,
and each row reads its own experts. When early drafts are rejected, the
later rows were computed for nothing.

**How it worked.** The builds `build/myds4-staged-draft` (switch
`MYDS4_STAGED_DRAFT=2`) and `build/myds4-staged-coupled-draft` (also
`MYDS4_COUPLED_DRAFT`) generated five drafts but verified only two per
pass. If both matched and the target's next sampled token also matched
the next held draft, a later pass verified the remaining held drafts
without generating new ones. A mismatch discarded the held drafts. A small
queue (`src/experimental/draft_queue.h`) held them.

**How it was measured.** The 8,187-token review prompt, 256 tokens, a
56 GiB total cache, 32 draft slots per stage, shared expert loads.

**Results.**

| Configuration | Decode tokens/s | Computed target rows | Logical read GB |
| --- | ---: | ---: | ---: |
| Plain decoding, controls | 11.73 to 11.81 | 256 | 220.30 to 220.79 |
| Five greedy drafts, one pass | 7.84 | 646 | 392.12 |
| Five shared-uniform drafts, one pass | 7.46 | 596 | 391.95 |
| Five greedy drafts, verified two at a time | 10.17 | 390 | 287.56 |
| Five shared-uniform drafts, verified two at a time | 10.15 | 373 | 285.71 |

The staged comparison's own plain controls reached 11.60 and 11.81
tokens/s. Outputs matched (`staged_draft_results.json`,
`draft_depth_current_results.json`). The queue passed 11 CPU checks.

**Why it did not help.** Staging reduced wasted rows and reads, but every
draft configuration still computed more rows and read more experts than
plain decoding on this request.

**What was removed.** The switch, `draft_queue.h`, its check
`tools/checks/draft_queue_check.c` (`build/draft-queue-check`), the two
builds and the staged branches of the CLI's decode loop.

### Modified rejection sampling for DSpark drafts

**Problem.** Drafts that must equal the target's sample are often
rejected at temperature 1.

**How it worked.** The build `build/myds4-rejection-draft` (switch
`MYDS4_REJECTION_DRAFT`, code in `src/experimental/speculative_sample.c`)
sampled each DSpark proposal from its own conditional distribution,
including the bias that DSpark's Markov head adds for the preceding
proposal. Verification accepted a proposal with probability min(1, p/q),
where p is the target's and q the draft's probability of that token.
After a rejection it drew the next
token from the normalized positive part of p − q. This method, modified
rejection sampling ([arXiv 2302.01318](https://arxiv.org/abs/2302.01318)),
keeps the target's distribution exactly, but a given seed no longer gives
the same text as ordinary decoding. Variants combined it with the causal
verification scheduler, a study that remains (see
[Early rejection, priority reads and spinning](#early-rejection-priority-reads-and-spinning-in-causal-verification))
(`build/myds4-causal-rejection-draft`,
`build/myds4-causal-signal-rejection-draft`,
`build/myds4-causal-group-rejection-draft`) and with a two-row forecast
limit (`build/myds4-rejection-prefetch2`, see
[Verification forecast admission by row](#verification-forecast-admission-by-row)).

**How it was measured.** A CPU fixture checked the sampler (22 checks:
probability conservation, one- and two-token frequencies, nucleus
filtering, greedy sampling, infinities, end of sequence and bonus tokens).
A model fixture compared all 16 verification rows and the state after
keeping 1, 2, 6, 15 or 16 rows. Live runs used the 51,895-token DSH
request, up to 1,024 tokens, two seeds and a fixed 56 GiB cache, of which
1.7 GiB held 32 experts per draft stage.

**Results.**

| Seed | Ordinary decoding, tokens/s | Five greedy drafts | Five rejection-sampled drafts | Accepted per pass, greedy; rejection |
| --- | ---: | ---: | ---: | ---: |
| 12345 | 10.91 | 6.88 | 6.96 | 1.80; 2.17 |
| 23456 | 10.01 | 6.32 | 6.96 | 1.85; 2.19 |

Ordinary and greedy-draft outputs matched within each seed; the second
seed ended after 990 tokens there, while rejection sampling reached the
1,024-token limit with different text, as expected. Rejection sampling
requested 1,818 to 1,851 GB of decode reads, against 994 to 1,024 GB for
ordinary decoding (`rejection_draft_speed_results.json`).
With the causal scheduler (`build/myds4-causal-signal-rejection-draft`)
and two drafts, it reached 8.69 and 8.86 tokens/s for the two seeds,
against plain decoding at 10.39 and 10.08
(`causal_rejection_speed_results.json`). The fixtures passed
(`rejection_verification_results.json`).

**Why it did not help.** Acceptance rose, but each pass still read about
twice the experts of ordinary decoding, and the SSD bound the speed.

**What was removed.** The switch, `src/experimental/speculative_sample.c`
and `.h`, `model_draft_sampled`, the CLI's rejection branches and its
`rejection_sampling` JSON line, the five builds named above,
`build/speculative-sample-check` (`tools/checks/speculative_sample_check.c`)
and `tools/bench/speculative_bench.py`, the runner of these comparisons.

### Fixed tool-header proposals

**Problem.** The model writes the same tokens at the start of every tool
call. They could be proposed without a draft model.

**How it worked.** The build `build/myds4-protocol-draft` (switch
`MYDS4_PROTOCOL_DRAFT`, code in `src/experimental/protocol_draft.c`)
replaced the `--lookup` proposer with three rules for the markup the model
writes around tool calls. After `<｜DSML｜ calls`, `<｜DSML｜ invoke` or
`<｜DSML｜ parameter`, it proposed the fixed tokens that follow, up to the
first token that depends on the tool or parameter name. A rule turned off
after two passes that accepted nothing. The target model verified every
proposal.

**How it was measured.** The 13,772-token DSH request, MXFP4, a 56 GiB
cache, `--lookup 5`, shared expert loads.

**Results.** All seven proposals across three headers were accepted, and
the 308-token reply matched ordinary decoding. Throughput was 10.83 tokens/s
against controls of 10.78 and 10.74 (`protocol_draft_speed_results.json`).
A CPU fixture checked the rules (`build/protocol-draft-check`).

**Why it did not help.** Headers are a few tokens per tool call. Seven
accepted tokens in a 308-token reply cannot change throughput measurably.

**What was removed.** The switch, `protocol_draft.c` and `.h`, the CLI's
protocol branches, the two targets and `tools/checks/protocol_draft_check.c`.

### Consensus fallback for context-copy drafts

**Problem.** `--lookup` copies proposals only from long matches: at least
12 tokens at first, 16 after two verifications that accept nothing, and 4
only after a fully accepted proposal. Short repeated phrases are therefore
rarely proposed.

**How it worked.** The build `build/myds4-consensus-lookup` (switch
`MYDS4_CONSENSUS_LOOKUP`, function `context_lookup_propose_consensus` in
`src/experimental/context_lookup.c`) kept the long-match path and added a
fallback of up to four tokens. The fallback required at least two earlier
occurrences of the four-token context and 80% agreement among them on the
whole proposed continuation. Two failed verifications disabled it until a
longer match succeeded. `build/myds4-prefix-prune-consensus` combined it
with dependency-pruned prefill and served as the control of two follow-up
studies. [SuffixDecoding](https://arxiv.org/html/2411.04975v3) also studies
frequency-based selection and adaptive proposal lengths; this fallback
verified one linear chain with the engine's ordinary causal attention.

**How it was measured.** Offline, recorded DSH token streams were replayed
through the matcher (`build/context-lookup-replay --consensus`). Live runs
used the 13,772-token request (256 tokens, 56 GiB, `--lookup 15`) and the
107,934-token request (8,192 tokens, 56 GiB).

**Results.** On an 8,190-token recorded continuation, the fallback
accepted 1,191 of 2,087 proposals, against 324 of 1,185 without it. Its
computed-row proxy was 9,086 against 9,051 (`context_lookup_replay_results.json`).
The live 13,772-token run decoded at 10.28 tokens/s; that record has no
control (`consensus_lookup_model_results.json`). The same build reached
10.34 to 10.41 tokens/s as the control of the two follow-up studies below.
On the 107,934-token request, with pruned prefill in every arm:

| Configuration | Decode seconds | Decode tokens/s |
| --- | ---: | ---: |
| No lookup | 705.91 | 11.60 |
| Consensus lookup | 731.87 | 11.19 |
| Consensus lookup with repeated-row cache credit | 719.02 | 11.39 |

All output bytes matched (`long_optimization_results.json`).

**Why it did not help.** More accepted tokens offline did not save time
live: on the long request, decoding with the fallback was 3.5% slower than
decoding without lookup.

**What was removed.** The switch, `context_lookup_propose_consensus` and
its declaration, the six consensus cases of
`tools/checks/context_lookup_check.c` (the check now runs 27), the
`--consensus` mode of `tools/experimental/context_lookup_replay.c`, and the
targets `build/myds4-consensus-lookup`, `build/myds4-prefix-prune-consensus`,
`build/myds4-consensus-prefetch1` and `build/myds4-consensus-row-credit`.
The ordinary `--lookup` remains.

### Repeated-row cache credit in verification

**Problem.** In a verification pass, one request credits an expert once,
even when several rows use it. An expert used by many rows may deserve a
higher retention score.

**How it worked.** The switch `MYDS4_VERIFY_ROW_CREDIT`
(`build/myds4-consensus-row-credit`, on top of the consensus build) added
one credit to the retention score for each additional row that used an
expert in an exact verification pass (`cache_repeat_credit`). Reads stayed
deduplicated, and rejected rows counted too.

**How it was measured.** The 13,772-token request, 256 tokens, 56 GiB,
`--lookup 15`, against the consensus build; and the long request above.

**Results.** 10.45 tokens/s against controls of 10.41 and 10.35
(`verification_row_credit_results.json`). On the 107,934-token request,
11.39 tokens/s against 11.19 for consensus lookup alone and 11.60 for no
lookup (`long_optimization_results.json`).

**Why it did not help.** The short gain was below 1%, within the
controls' spread, and the long run stayed slower than decoding without
lookup.

**What was removed.** The switch, its code in `routed_rows`,
`cache_repeat_credit` in `src/engine/io.c`, which the concurrent usage
credit below also used, and the four cases of
`tools/checks/cache_policy_check.c` that checked it.

### Verification forecast admission by row

**Problem.** Each row of an exact verification pass issues forecast reads
like a decoded token, so a pass admits forecasts for every row. The
forecasts of rows that are later rejected may be wasted.

**How it worked.** The switch `MYDS4_VERIFY_PREFETCH_ROWS=N` admitted
new forecast reads only from the first N rows of a verification pass.
Refinement still used every row's forecasts, and every selected expert was
still read. Builds: `build/myds4-verify-no-prefetch` (N = 0),
`build/myds4-verify-prefetch1`, `build/myds4-verify-prefetch2`,
`build/myds4-consensus-prefetch1` and `build/myds4-rejection-prefetch2`.

**How it was measured.** The 13,772-token request, MXFP4, 308 tokens, two
drafts, a fixed 20 GiB cache, shared expert loads and pruned prefill.

**Results.**

| Rows supplying forecast reads | Tokens/s | Admitted reads per token | Logical decode read GB |
| --- | ---: | ---: | ---: |
| All three (controls) | 6.13 to 6.14 | 75.7 to 75.8 | 642.81 to 644.10 |
| First row | 6.11 | 28.3 | 628.00 |
| First two rows | 6.17 | 54.1 | 637.01 |
| None | 5.71 | | 596.51 |

Records: `verification_prefetch_budget_results.json` and
`verify_no_prefetch_results.json`. With consensus lookup and one row,
10.35 tokens/s against 10.39 and 10.34 (`verification_prefetch_results.json`).

**Why it did not help.** Fewer forecasts meant fewer useful reads in time:
with none, read waiting rose from about 42 s to 45.46 s. The two-row limit
gained about 0.5%, within the controls' spread.

**What was removed.** The switch, its limits in `forecast_rows` and
`read_residual_ahead`, the five builds, and
`tools/checks/verification_prefetch_check.c` (`build/verification-prefetch-check`).

### Shared kernels for DSpark's experts

**Problem.** DSpark's three stages each select 3 of 128 experts for each
of five rows, and rows often share experts. Like the backbone's rows, they
loaded shared weights once per row.

**How it worked.** The build `build/myds4-draft-expert-batch` (switch
`MYDS4_DRAFT_EXPERT_BATCH`) applied the grouped MXFP4 kernels of
`--expert-batch` to DSpark's routed experts when that option was on. Each
row used three of the kernels' six expert slots. Each stage kept its own
descriptors until its GPU work finished. The draft's experts stay MXFP4
even with a GGUF backbone.

**How it was measured.** `build/draft-expert-batch-check` compared the
path with the ordinary draft experts on checkpoint weights. Its `--bench`
mode timed repeated drafts from one state with no expert reads (seven
alternating samples of eight generations). A live screen used the
13,772-token request, up to 512 tokens, two drafts and a fixed 20 GiB
total cache.

**Results.** All six cases per format (one, two or five drafts, each with
final-stage trimming on and off) were bit-identical in their logits,
residual streams and draft token IDs; with two trimmed drafts the fixture
shared 36 row/expert pairs in 24 groups
(`draft_expert_batch_validation.json`). Drafting alone took 5.6 to 7.9%
less time:

| Backbone format | Drafts | Ordinary, ms | Shared, ms |
| --- | ---: | ---: | ---: |
| MXFP4 | 2 | 7.282 | 6.878 |
| MXFP4 | 5 | 9.635 | 9.032 |
| IQ2 | 2 | 6.123 | 5.777 |
| IQ2 | 5 | 8.089 | 7.450 |

Record: `draft_expert_batch_kernel_results.json`. Live decoding reached
6.06 tokens/s for MXFP4 against controls of 6.14 and 6.13, and 12.09 for
IQ2 against two controls of 12.10 (`draft_expert_batch_speed_results.json`).

**Why it did not help.** The saving was 0.3 to 0.6 ms per drafting step.
In the same MXFP4 runs, the CPU waited about 42 s for expert reads over
308 emitted tokens, about 137 ms per token.

**What was removed.** With `--expert-batch`, whose kernels it used: the
switch, `draft_experts_grouped` in `src/engine/model.c`, the target
`build/myds4-draft-expert-batch`, and `tools/checks/draft_expert_batch_check.c`
(`build/draft-expert-batch-check`).

### One draft-slot pool for the three DSpark stages

**Problem.** With `--draft-cache 32`, each DSpark stage has its own pool of
32 slots. The stages use different numbers of distinct experts: earlier
MXFP4 captures of a reasoning prompt used 57, 28 and 20. Stage 0 therefore
missed more often while the other pools had spare slots.

**How it worked.** The build `build/myds4-draft-shared-pool` (switch
`MYDS4_DRAFT_SHARED_POOL`) let the three stages borrow from one pool of
the same total size (96 slots for 32 per stage). Replacement chose the
least recently used slot across the pool, protected every expert of the
current request, and skipped slots with pending reads. Backbone slots and
the total cache budget were unchanged. The build
`build/myds4-draft-cache-control` (switch `MYDS4_DRAFT_CACHE_METRICS`)
added only per-stage hit, miss and residency counters, as the control.

**How it was measured.** 69 host-side checks of borrowing, pending-read
ownership, current-request protection and constant memory
(`build/draft-shared-pool-check`). Live: the 13,772-token request, a fixed
20 GiB total cache.

**Results.** With MXFP4, draft-expert misses fell from 391 to 363. Stage
0's misses fell from 231 to 171, and the other stages' rose from 160 to
192. The pool ended with 39, 28 and 29 experts across the stages.
Throughput was 6.14 tokens/s for the candidate and both controls. With
IQ2, misses fell from 361 to 337, at 12.08 tokens/s against two controls
of 12.07 (`draft_shared_pool_speed_results.json`).

**Why it did not help.** Draft misses were about 1.3 per emitted token.
The same runs made about 55 demand reads per emitted token with MXFP4 and
46 with IQ2, so removing 28 draft misses in a whole run changed nothing
measurable.

**What was removed.** Both switches, the shared-pool branch of
`draft_victim` and the per-stage counters (the `draft_cache_stages` JSON
line) in `src/engine/io.c`, the two builds, `build/draft-shared-pool-check`
and `tools/checks/draft_shared_pool_check.c`. The ordinary per-stage pools
of `--draft-cache` remain.

### Early rejection, priority reads and spinning in causal verification

These experiments extended the causal verification study
(`MYDS4_CAUSAL_VERIFY`, with `MYDS4_CAUSAL_SIGNALS` and
`MYDS4_CAUSAL_GROUP`). That study remains in the code as a study under
evaluation: its performance with large caches is still to be measured
before it is kept or removed. The design and performance documents
describe it. This section covers only the variants built on it that were
removed.

The study verifies a pass's rows as separate jobs on one attention state.
A row runs a layer's front only after the row before it has run that
front, so each row sees its predecessor's keys. A row's routed experts may
finish earlier and wait in a private buffer, so a row whose experts are
ready can advance while another row waits for reads. The grouped variant
runs ready rows at one layer together. Rows are still all computed before
sampling decides which prefix to keep.

**Problem.** When the target rejects an early draft, the later rows'
remaining layers are wasted. Their expert reads also compete with the
reads of rows that will be kept.

**How it worked.**

- Early stop (switches `MYDS4_CAUSAL_EARLY_STOP` and
  `MYDS4_CAUSAL_LEAD_FIRST`, build `build/myds4-causal-early`): rows were
  sampled in token order as each finished
  (`src/experimental/speculative_frontier.h`). The earliest ready row ran
  first, and later rows ran while it waited for reads. A rejected draft or
  the end of sequence stopped the remaining rows. Unfinished rows held NaN
  logits that were never sampled. This is a single-GPU form of the early
  cancellation that [PipeInfer](https://arxiv.org/html/2407.11798v1)
  studies in a distributed pipeline; its published speedups are not
  predictions for this engine.
- Priority reads (switch `MYDS4_CAUSAL_PRIORITY_READS`, build
  `build/myds4-causal-priority`): rows of the already-required prefix used
  demand reads. The selected experts of unconfirmed rows used the
  background queue (`cache_hold_speculative`). When such a row became
  necessary, its queued pieces were promoted to demand priority without a
  second request (`cache_promote_selected`).
- Spinning (switch `MYDS4_RUNAHEAD_SPIN`, build
  `build/myds4-causal-priority-spin`): the scheduler spun on the GPU's
  event instead of sleeping, to resume without a wake-up delay.

**How it was measured.** A CPU fixture checked the token-order sampler on
2,716 cases. A model fixture checked accepted prefixes of 0, 1, 2, 5, 14
and 15 drafts in MXFP4 and IQ2 at 257 and 16,521 prompt tokens: 24 cases
matched the target's samples, random state, logits and next drafts
(`early_verify_validation.json`, `early_priority_validation.json`). A host
check of the read queue passed 23 checks (`build/speculative-read-check`).
The spinning variant's six short-context sampler and rollback cases passed
for each expert format (`early_spin_validation.json`). Live runs used the
13,772-token request with 308 tokens.

**Results.**

| Build, drafts, cache | Tokens/s | Ordinary-decoding controls | Logical read GB | Record |
| --- | ---: | ---: | ---: | --- |
| Early stop, 5, 20 GiB | 5.53 | 6.38, 6.38 | 683.17 (ordinary 599.43) | `early_verify_speed_results.json` |
| Priority reads, 5, 20 GiB | 6.02 | 6.40, 6.38 | 639.32 | `early_priority_speed_results.json` |
| Priority reads, 2, 20 GiB | 5.90 | 6.31, 6.38 | | `early_priority_speed_results.json` |
| Priority reads, 2, 32 GiB | 7.58 | 8.01, 7.96 | about 458 in both | `early_priority_speed_results.json` |
| Priority reads with spinning, 2, 32 GiB | 7.63 | 8.06, 8.07 | | `early_spin_speed_results.json` |

Early stop scheduled 414 target rows, cancelled 106 and skipped 1,423
expert row-layer computations. Priority reads skipped 3,142. The spinning
run used 14.02 s of user CPU time and 37.98 s of system time, against
about 2.34 and 39.19 s for the controls, over the whole process.

**Why it did not help.** Cancelled rows had already requested their reads,
so stopping before the head did not recover the read work. Every variant
also stayed slower than ordinary decoding without drafts. At 32 GiB both
paths read about 458 GB, yet the speculative path took longer. The likely
cause is the scheduler's coordination work between GPU jobs; this is
inferred from equal reads at lower speed and was not measured directly.

**What was removed.** The switches `MYDS4_CAUSAL_EARLY_STOP`,
`MYDS4_CAUSAL_LEAD_FIRST`, `MYDS4_CAUSAL_PRIORITY_READS` and
`MYDS4_RUNAHEAD_SPIN`; `speculative_frontier.h`, `model_verify_early`, the
early statistics and the CLI's early-stop branch; in `src/engine/io.c`,
`cache_hold_speculative`, `cache_promote_selected`, `cache_request_priority`
and the third read mode of `start()`, which put a router-selected
speculative row's reads on the background queue without counting them as
prefetches (`start()` now takes 0 for a forecast and 1 for a demand);
`gpu_spin_signal` in `src/engine/gpu.c`, which this spin and the spin
router below used; the targets `build/myds4-causal-early`,
`build/early-verify-check`, `build/speculative-frontier-check`,
`build/myds4-causal-priority`, `build/myds4-causal-priority-spin`,
`build/early-priority-check`, `build/early-priority-spin-check` and
`build/speculative-read-check`; the checks
`tools/checks/early_verify_check.c`, `tools/checks/speculative_frontier_check.c`
and `tools/checks/speculative_read_check.c`. `build/gpu-wait-check` now
checks only the blocking wait, `gpu_wait_signal`.

**The separate route-only build.** `build/myds4-causal-route-only` and its
check `build/causal-route-only-check` built the grouped study with
`MYDS4_RUNAHEAD_NO_FORECASTS`, which omits the residual and next-layer
router projections that the scheduler never reads. By the time of this
cleanup the default compiler flags defined that switch for every build, so
the other causal builds behaved the same. The code now omits those
projections in every causal build without a switch, and the two separate
targets are gone. The performance document keeps the measurement of that
change.

## Expert-cache and read-ahead studies

### Read-ahead width and order

**Problem.** Each forecast reads its top 3 uncached experts. Fewer reads
leave the SSD free for demand reads. More reads catch more misses. The
order of the one- and two-layer forecasts decides which reads start first.

**How it worked.** The switch `MYDS4_PREFETCH_AHEAD=N` changed the number
of forecast candidates read from 3 to N (builds `build/myds4-ahead1`,
`build/myds4-ahead2`, `build/myds4-ahead4`). The switch `MYDS4_H2_FIRST`
(`build/myds4-h2-first`) issued the two-layer forecast's reads before the
one-layer forecast's.

**How it was measured.** The 13,772-token request, MXFP4, 256 tokens, a
56 GiB cache.

**Results.**

| Build | Tokens/s | Controls | Logical decode read GB | Record |
| --- | ---: | ---: | ---: | --- |
| 1 candidate | 10.34 | 10.66, 10.66 | 214.56 (controls about 253) | `native_ahead1_results.json` |
| 2 candidates | 10.58 | 10.66, 10.65 | 231.44 | `native_ahead2_results.json` |
| 4 candidates | 10.55 | 10.70, 10.61 | 273.70 | `single_decode_screen_results.json` |
| Two-layer forecast first | 10.63 | 10.70, 10.61 | 253.34 | `single_decode_screen_results.json` |

**Why it did not help.** Fewer candidates cut reads but raised read
waiting (15.19 s against about 14.2 s with one candidate). Four candidates
read 8% more without a gain. The order made no measurable difference.

**What was removed.** The override of `MYDS4_PREFETCH_AHEAD` (the engine
keeps 3 candidates), the switch `MYDS4_H2_FIRST` and the four builds.

### Eviction by stale layer

**Problem.** The retention score ignores where a slot's layer lies
relative to the layer being served. An expert of a layer already passed
in this token cannot be needed again until the next token.

**How it worked.** The switch `MYDS4_STALE_PRIORITY` (`build/myds4-stale`)
chose decode victims in classes: empty slots first, then experts of
earlier layers that the current pass had not selected, then such experts
of later layers, then experts already used in this pass. The retention
score ordered each class. Protected forecasts and experts in use stayed
ineligible, as before.

**How it was measured.** `tools/checks/cache_stale_check.c`
(`build/cache-stale-check`) checked the classes. A live screen used the
51,895-token request, MXFP4, 512 tokens, a 56 GiB cache.

**Results.** 9.08 tokens/s against controls of 10.43 and 10.44. Logical
decode reads rose from about 524 GB to 624.15 GB, and read waiting from
29.24 s to 36.82 s (`decode_long_screen_results.json`).

**Why it did not help.** Reads rose by 19% and speed fell by 13%. The
layer classes overrode the retention score; the extra reads indicate that
they evicted experts that later tokens needed again (an inference from
the read counts).

**What was removed.** The switch, its victim classes and `demanded` field
in `src/engine/io.c`, `build/myds4-stale`, and `build/cache-stale-check`
(`tools/checks/cache_stale_check.c`).

### A lower GPU residency limit

**Problem.** The GPU stalls once the resident memory of all processes
passes about 55 GiB on this machine. The engine keeps mapped slots below
41% of physical memory (52.5 GiB), about 2 GiB below that point.

**How it worked.** The switch `MYDS4_GPU_LIMIT_GIB=48` lowered the limit
to 48 GiB, so fewer slots stayed mapped for the GPU; the others were
mapped again when needed (builds `build/myds4-gpu48` and
`build/myds4-prune-gpu48`).

**How it was measured.** The 51,895-token request, MXFP4, 512 tokens, a
56 GiB cache.

**Results.** 10.40 tokens/s against controls of 10.43 and 10.44, with the
same logical reads (524.44 GB) (`decode_long_screen_results.json`).

**Why it did not help.** Speed and reads did not change. This suggests
that the default limit already kept the GPU below its stall point in this
configuration; the run did not measure the GPU's resident memory against
that point.

**What was removed.** The switch in `mem_gpu_limit` and the two builds.

### Prefill credit and decode-only scores

**Problem.** A long prompt selects nearly every expert, so prefill gives
almost every requested expert the same credit. This credit can keep
experts that decoding will not use.

**How it worked.** The switch `MYDS4_PREFILL_CREDIT_ROWS` replaced the one
credit that a large prompt batch gives each requested expert with the
share of the batch's rows that selected it (`cache_prefill_credit`). A
batch's credits then total 6, as one decoded row's do. Value 0 counted
the whole batch (`build/myds4-prefill-frequency`); 256 counted only the
last 256 rows (`build/myds4-prefill-tail-frequency`). The switch
`MYDS4_DECODE_ONLY_SCORE` gave prefill requests no credit at all (builds
`build/myds4-decode-score`, `build/myds4-prefix-decode-score` and
`build/myds4-prune-decode-score`).

**How it was measured.** A CPU fixture passed 57 checks of frequency
totals, retained history, window boundaries, active-expert protection and
unchanged read accounting (`build/prefill-credit-check`); the retention
fixture of that time passed its 48; `build/prefill-priority-check` checked
the decode-only rule. Live runs used the 13,772-token request (512 tokens
requested, a fixed 32 GiB cache, pruned prefill) and the 107,934-token
request.

**Results.**

| Build | Tokens/s | Controls | Logical decode read GB | Record |
| --- | ---: | ---: | ---: | --- |
| Whole-batch frequency, MXFP4 | 8.27 | 8.08, 8.05 | 445.75 (controls about 458) | `prefill_frequency_speed_results.json` |
| Last-256-row frequency, MXFP4 | 8.26 | 8.08, 8.08 | 445.93 | `prefill_frequency_speed_results.json` |
| Decode-only score, MXFP4 | 8.14 | 8.07, 8.05 | 447.41 | `prefill_frequency_speed_results.json` |
| Whole-batch frequency, IQ2 | 16.39 | 15.98, 15.98 | 186.62 (controls 194.3) | `prefill_frequency_speed_results.json` |

All outputs matched their controls. One closing control's prefill took
29.57 s against about 22 s in the other arms, so these runs support no
prefill claim. On the 107,934-token request with 8,192 tokens and a
32 GiB cache, the control decoded in 948.28 s (8.64 tokens/s) and
whole-batch frequency in 985.53 s (8.31). Logical reads fell from 10,821.7
to 10,697.2 GB, but read waiting rose from 664.47 to 691.31 s, and GPU time
from 290.67 to 293.85 s (`prefill_frequency_long_results.json`).
A decode-only run on that request (512 tokens, 56 GiB) completed only its
control: 138.82 s of prefill and 10.88 tokens/s; background CPU activity
stopped the candidate (`prefill_priority_results.json`).

**Why it did not help.** The decode-only build, which only removes
prefill credit, cut reads nearly as much (447.41 GB). This suggests that
less prefill credit at the start of decoding explains most of the short
gains. Over a long generation the frequency policy was 3.8% slower. One
long pair cannot separate drift during the run from the policy's effect.

**What was removed.** Both switches, `cache_prefill_credit`, the five
builds named above, and the checks `build/prefill-credit-check`
(`tools/checks/prefill_credit_check.c`) and `build/prefill-priority-check`
(`tools/checks/prefill_priority_check.c`).

### Cancelling unused forecasts after routing

**Problem.** When a layer's router selects its experts, forecast reads for
that layer that it did not select are useless. Unstarted ones still
occupy the queue and the SSD.

**How it worked.** The switch `MYDS4_PREFETCH_DEMOTE_CANCEL`
(`build/myds4-prefetch-demote`, and the server build
`build/myds4-server-prefetch-demote`) cancelled, after routing, every
forecast read of that layer that routing had not selected and that no
reader had started. It used the refinement's cancellation with an empty
keep list. Reader claims and the decision shared the queue's lock. Prefill
kept its policy, because later prompt batches can still need the experts.

**How it was measured.** A CPU fixture failed 7 of 26 checks before the
implementation; the final fixture passed 40 (`build/prefetch-demote-check`).
Live runs used the 13,772-token request at a fixed 24 GiB cache, the
107,934-token request with 8,192 tokens, and the server with one request.

**Results.**

| Run | Candidate tokens/s | Controls | Logical decode read GB |
| --- | ---: | ---: | ---: |
| 13,772 tokens, MXFP4 | 7.04 | 6.91, 6.92 | 533.34 (controls about 546) |
| 13,772 tokens, IQ2 | 13.89 | 13.71, 13.47 | 237.17 (controls 241.71 to 243.06) |
| 107,934 tokens, MXFP4, 8,192 tokens | 7.24 | 7.17, 7.40 | 12,815.07 (controls 13,172.27, 13,141.92) |
| Server, MXFP4, with the refinement fix | 7.01 | 6.85, 6.94 | |
| Server, IQ2, with the refinement fix | 13.63 | 13.81, 13.55 | |

Records: `prefetch_demote_speed_results.json`,
`prefetch_demote_long_results.json`,
`prefetch_refine_narrow_server_mxfp4_results.json` and
`prefetch_refine_narrow_server_iq2_results.json`. The long run removed
42,872 unstarted forecasts, and 95.6% of its useful forecast reads still
arrived late.

**Why it did not help.** Reads fell by 2.5 to 2.7%, but the reads that
decide the speed, those of experts that are used, still arrived late. The
long run lay between its controls, 0.9% faster than the first and 2.2%
slower than the closing one.

**What was removed.** The switch and its `prefetch_demote` counter, the
build, and the server build with its library
(`build/libmyds4-prefetch-demote.a`, `build/experimental/io-prefetch-demote.o`).
The refinement's cancellation of unstarted reads remains.
`tools/checks/prefetch_demote_check.c` (`make prefetch-demote-check`)
remains with 37 checks: demotion after routing cancels nothing, in decoding
or in prefill, and a newer prediction (`cache_refine`) cancels only wholly
unstarted forecasts that it excludes, keeping the queue's order, the
accounting and the reuse of the emptied slots.

### A bounded segment for forecast reads

**Problem.** A forecast read takes a slot from the main cache, so a wrong
forecast evicts an expert that might be used again.

**How it worked.** The builds `build/myds4-prefetch-pool4` and
`build/myds4-prefetch-pool8` (switch `MYDS4_PREFETCH_POOL`) limited
decode-mode forecast victims to a probationary segment of up to 4 or 8
slots inside the cache. Establishing the segment cost that many main
slots. When routing selected a forecast expert, the slot joined the main
cache and an eligible main slot joined the segment; no weights moved.
Required reads could still reclaim segment slots.

**How it was measured.** 42 CPU checks of both limits, held and published
entries, pending reads, buffer identity, refinement, prompt-pass
reclamation and pressure release (`build/prefetch-pool-check`). Live runs
used the 13,772-token request (308 tokens with MXFP4, 295 with IQ2). A
`prefetch_pool` JSON record reported the limit, membership, admissions,
declined forecasts, promotions and the main entries displaced while the
segment was set up.

**Results.**

| Format, cache | Segment slots | First control | Candidate | Closing control |
| --- | ---: | ---: | ---: | ---: |
| MXFP4, 20 GiB | 4 | 6.33 | 6.31 | 6.34 |
| MXFP4, 20 GiB | 8 | 6.35 | 6.23 | not run |
| IQ2, 16 GiB | 4 | 11.29 | 11.35 | 11.29 |
| IQ2, 16 GiB | 8 | 11.26 | 11.33 | 11.34 |

The four-slot MXFP4 run read 605.76 GB against 599.91 to 600.56 GB, and
the eight-slot one 612.35 GB against its first control's 599.49 GB. The
eight-slot MXFP4 comparison is incomplete: the memory guard blocked its
closing control. With IQ2 the four-slot run read 308.41 GB against 309.92
to 310.03 GB, and the eight-slot one 311.18 GB against 309.75 to 310.16 GB
(`prefetch_pool_speed_results.json`).

**Why it did not help.** Protecting the main cache from forecasts did not
reduce reads in a useful way. With MXFP4 it raised them by about 1 to 2%.
With IQ2 the four-slot segment read about 0.5% less and was about 0.5%
faster in one request, and the eight-slot segment did not consistently
improve throughput.

**What was removed.** The switch, its segment code and `prefetch_pool`
record in `src/engine/io.c`, the two builds, `build/prefetch-pool-check`
and `tools/checks/prefetch_pool_check.c`.

### A bounded urgent-read window, and read diagnostics

**Problem.** Every demand piece goes to GCD at once; up to 60 read
callbacks were active together. Many parallel reads might slow each
read, so the last expert a layer needs could arrive later.

**How it worked.** The builds `build/myds4-io-window4`, `-window8` and
`-window16` (switch `MYDS4_URGENT_WINDOW=N`) let at most N workers read
demand pieces, also during prefill. Queued and active demand pieces both
kept background readers paused; pieces kept their size, destination and
single owner. The diagnostic builds `build/myds4-io-diagnostic` and
`build/myds4-io-windowN-diagnostic` (switch `MYDS4_IO_DIAGNOSTICS`, code
in `src/experimental/urgent_io_diagnostic.h`) recorded software timestamps
of queue delay, read service time, active callbacks, and the interval
from a layer's demand to the arrival of its last selected expert, in
50 µs bins. These timestamps describe software callbacks, not the SSD's
hardware queue.

**How it was measured.** A CPU fixture passed 144 checks of repeated
promotion, queue-counter wrap, completion accounting and exact destination
bytes (`build/urgent-window-check`). The diagnostic screen used the
13,772-token request, MXFP4, 128 tokens and a 16 GiB cache; all five
outputs matched and every run kept 17,175,674,880 cache bytes. A second
screen used the uninstrumented builds at 56 GiB with the 308-token reply.

**Results.**

| Urgent window | Peak active callbacks | Mean last-expert arrival | Mean queue delay | Mean read service |
| --- | ---: | ---: | ---: | ---: |
| Unrestricted, first control | 60 | 3.561 ms | 0.123 ms | 2.816 ms |
| 8 | 8 | 3.548 ms | 1.739 ms | 1.119 ms |
| 4 | 4 | 3.551 ms | 2.164 ms | 0.591 ms |
| 16 | 16 | 3.553 ms | 0.998 ms | 1.891 ms |
| Unrestricted, closing control | 60 | 3.585 ms | 0.126 ms | 2.834 ms |

Records: `urgent_window_diagnostic_results.json` and
`urgent_io_diagnostic_results.json`. At 56 GiB the windows changed the
mean control duration by +0.09% (8), −0.94% (4) and +0.26% (16)
(`five_proposals_56g_results.json`).

**Why it did not help.** Shorter service times were offset by longer
queue waits, and the last expert's arrival changed by less than 1%. This
suggests that the SSD's total throughput, not the number of parallel
requests, set the arrival time.

**What was removed.** Both switches, `urgent_io_diagnostic.h`, the window
code and the diagnostic hooks in `src/engine/io.c` and `src/cli/main.c`,
the pattern targets `build/myds4-io-window%` and
`build/myds4-io-window%-diagnostic`, `build/myds4-io-diagnostic`,
`build/prefetch-queue-diag-check` (the queue check built with diagnostics)
and `tools/checks/urgent_window_check.c`.

### Promoting selected pending reads earlier

**Problem.** When a layer selects an expert whose forecast read is still
queued, the request promotes its queued pieces to demand priority. In the
ordinary order, a missing expert's slot is allocated first, and that
allocation can wait for its victim's started read; the selected forecast
stays at background priority during that wait.

**How it worked.** The switch `MYDS4_EARLY_DEMAND_PROMOTION`
(`build/myds4-early-promotion`, `build/myds4-early-promotion-diagnostic`)
promoted every selected pending read before allocating slots for missing
experts. Hit and late attribution stayed with the original read.

**How it was measured.** A controlled CPU fixture: the ordinary path
passed 11 of 12 checks and the new order all 12
(`build/prefetch-promote-check`). A 128-token diagnostic used the
13,772-token request at 16 GiB; a timing screen used 56 GiB.

**Results.** The diagnostic found 245 promotions, covering 2,078 pieces,
delayed behind reclamation in the ordinary path, and none with the
switch. Reclamation still waited 167.949 ms in total in the control and
167.127 ms in the candidate. The mean arrival of a layer's last required
expert was 3.561 and 3.558 ms (`demand_promotion_diagnostic_results.json`).
At 56 GiB the candidate reached 10.67 tokens/s against controls of 10.57
and 10.59, +0.81% (`five_proposals_56g_results.json`).

**Why it did not help.** The reordering worked, but the arrival of a
layer's last required expert did not change (3.561 against 3.558 ms), and
reclamation still waited as long as before.

**What was removed.** The switch, its code in `cache_request_priority`,
both builds, `build/prefetch-promote-check` and
`tools/checks/prefetch_promote_check.c`.

### Metal's file-loading API

**Problem.** macOS offers a Metal API that loads file data directly into
GPU buffers. It might read experts faster than `pread` with `F_NOCACHE`
(reads that bypass the file cache) into shared buffers.

**How it worked.** `build/metal-io-probe CHECKPOINT`
(`tools/experimental/metal_io_probe.c`) read 18 selected expert spans
three ways: parallel 2 MiB `pread` pieces, Metal I/O into shared buffers,
and Metal I/O into private (GPU-only) buffers, and compared the bytes.

**Results.** All paths matched byte for byte. First reads of new regions,
median of two samples:

| Experts read together | Parallel pread | Metal I/O, shared | Metal I/O, private |
| --- | ---: | ---: | ---: |
| 1 | 1.673 ms | 1.730 ms | 2.645 ms |
| 3 | 4.495 ms | 4.651 ms | 5.362 ms |
| 6 | 8.777 ms | 8.859 ms | 9.885 ms |

Record: `metal_io_probe_results.json`. Reads of regions that another path
had just read were much faster (about 1.16 to 1.28 ms for six experts
through `pread`). These timings are not evidence of SSD throughput;
caching may explain them.

**Why it did not help.** Metal I/O was slower in every case.

**What was removed.** The probe and `build/metal-io-probe`.

### Offering forecasts again, and releasing admissions early (never merged)

These two policies ran only in study builds that were never merged into
the main branch; only their records remain.

**Problem.** A forecast that finds the in-flight limit full is skipped. And
a forecast that its layer then selects keeps its admission, counting
against the limit, until its read completes.

**How it worked.** One build offered the same one- and two-layer forecasts
again after the required reads completed. The other released a forecast's
admission as soon as the forecast became a required read. Both kept the
selected experts.

**How it was measured.** The 13,772-token DSH request, a fixed 24 GiB
cache, temperature 1 and top-p 1; MXFP4 generated 308 tokens and IQ2 295,
with the same output as the controls.

**Results.**

| Experiment | MXFP4 tokens/s | MXFP4 controls | IQ2 tokens/s | IQ2 controls |
| --- | ---: | ---: | ---: | ---: |
| Offer forecasts after the wait | 6.91 | 6.83 / 6.92 | 13.68 | 13.70 / 13.48 |
| Release admission on promotion | 6.87 | 6.89 / 6.93 | 13.68 | 13.75 / 13.65 |

The second offer admitted 1,825 MXFP4 forecasts after 10,352 waits, using
0.014 s of host scheduling time, and raised logical reads from about 545.8
to 546.7 GB. Of 36.1 first uses of read-ahead experts per token, 35.2 were
still late. With IQ2 it admitted 920 more forecasts. Records:
`prefetch_retry_speed_results.json` and
`prefetch_promotion_speed_results.json`.

**Why it did not help.** Neither changed throughput beyond the controls'
spread; the reads that set the speed still arrived late.

## Expert execution studies

The ordinary single-row decode already runs the experts in memory while
other selected experts are read (the split path). These studies tried to
overlap more of the remaining wait.

### Phased expert reads

**Problem.** A layer's down projection cannot start before its gate/up
projections finish. The gate/up weights could be used before the down
weights arrive.

**How it worked.** The build `build/myds4-expert-phase` (switch
`MYDS4_EXPERT_PHASED`) ordered each native MXFP4 expert's read pieces so
that the gate/up weights and all scales came first and the pieces wholly
inside the down matrix came last, without changing any piece's file range
or destination. A layer's gate/up kernels ran once their pieces were in;
its down kernels waited for the full read. The variant
`build/myds4-expert-phase-preencode` (`MYDS4_EXPERT_PHASED_PREENCODE`)
also encoded both commands before the waits and submitted each when its
inputs were ready. Prompt batches, GGUF experts and DSpark kept their
ordinary paths. A separate probe (`build/expert-stream-probe`) measured
the best case: six experts missing and shared by every row, with synthetic
activations. Its alignment pages were split so that a late CPU read never
wrote bytes that the GPU's gate/up kernels read.

**How it was measured.** A CPU fixture passed 803 checks of read ranges,
completion order, cancellation and group lifetimes
(`build/expert-phase-check`). Model fixtures compared 16 forced
continuation positions after 257- and 16,521-token prompts in both
formats. A live screen used the 13,772-token request with 308 tokens and a
fixed 32 GiB cache.

**Results.** All model cases had zero differing logit and state bytes,
and the next DSpark drafts matched (`expert_phase_validation.json`,
`expert_phase_preencode_validation.json`). The MXFP4 cases ran 447 and 436
phased layers (449 and 448 pre-encoded); IQ2 kept its ordinary read path.
The probe's medians, all bytes first against gate/up first, were
9.158/8.986 ms for one row, 9.324/9.001 ms for three and 9.632/9.165 ms
for six, without the first sample of each size; all outputs matched
(`expert_stream_results.json`).

| Run | Tokens/s | Controls | Main-queue GPU seconds | Read-wait seconds |
| --- | ---: | ---: | ---: | ---: |
| Phased reads | 8.04 | 8.07, 8.08 | 9.78 (controls 10.41 to 10.42) | about 28.07 |
| Phased reads, pre-encoded | 8.04 | 7.98, 8.08 | 9.78 | 28.06 |

Record: `expert_phase_speed_results.json`. The candidate reported 9,001
phased layers and 858 layers whose reads were complete after the first
wait.

**Why it did not help.** GPU time fell by about 6%, but read waiting
stayed at about 28 s, and throughput did not change.

**What was removed.** Both switches, the phased piece ordering and
`cache_wait_up` in `src/engine/io.c`, the phased branches of `routed_rows`,
the builds `build/myds4-expert-phase`, `build/myds4-expert-phase-preencode`,
`build/expert-phase-check`, `build/expert-phase-model-check`,
`build/expert-phase-preencode-model-check` and `build/expert-stream-probe`,
and `tools/checks/expert_phase_check.c`,
`tools/checks/expert_phase_model_check.c` and
`tools/experimental/expert_stream_probe.c`.

### Running experts in completion waves

**Problem.** The split path runs the experts in memory once, then waits
for all the others. Experts that arrive early during that wait stay idle
until the last one arrives.

**How it worked.** The build `build/myds4-expert-waves` (switch
`MYDS4_EXPERT_WAVES`, code in `src/experimental/expert_waves.c`) waited
for any selected expert to finish loading, then submitted every expert
that was ready, as a wave, while the other reads continued. Each wave
wrote the experts' original positions in `PARTS`, and one final reduction
summed them in expert order. Each wave had its own descriptors, 60 KiB in
all. The next layer's front was encoded early but submitted after the last
wave. The variant `build/myds4-expert-waves3` (`MYDS4_EXPERT_WAVE_CAP=3`)
allowed at most three waves per layer. The server build
`build/myds4-server-expert-waves` tested it through HTTP. It applied to
single-row decoding without substitution.

**How it was measured.** A CPU fixture passed 10 checks
(`build/expert-ready-check`). Model fixtures forced one expert per wave
after 257- and 16,521-token prompts in both formats: 1,920 waves over 320
layers per case, with all logit and state bytes matching
(`expert_waves_validation.json`); the three-wave cap passed the same cases
with 960 waves over 320 layers (`expert_waves3_validation.json`). Live
screens used the 13,772-token request (20 GiB and 16 GiB caches), a
22,481-token request, the server (one request at a time; its counters had
to confirm that the path ran), and the 107,934-token request with 8,192
tokens. The CLI screens used pruned prefill and shared expert loads.

**Results.**

| Run | First control | Waves | Closing control | Record |
| --- | ---: | ---: | ---: | --- |
| CLI, 13,772 tokens, 20 GiB, MXFP4 | 6.37 | 6.44 | 6.36 | `expert_waves_speed_results.json` |
| CLI, 22,481 tokens, 512 out, MXFP4 | 5.90 | 5.95 | 5.89 | `expert_waves_speed_results.json` |
| Server, 13,772 tokens, 16 GiB, MXFP4 | 5.792 | 5.850 | 5.825 | `expert_waves_server_mxfp4_results.json` |
| Server, 13,772 tokens, 16 GiB, IQ2 | 11.342 | 11.400 | 11.328 | `expert_waves_server_iq2_results.json` |
| CLI, 107,934 tokens, 8,192 out, 20 GiB, MXFP4 | 6.52 | 6.60 | 6.65 | `expert_waves_long_results.json` |
| Capped at three waves, 16 GiB, MXFP4 | 5.87 (uncapped) | 5.82 | 5.88 (uncapped) | `expert_waves3_speed_results.json` |
| Capped at three waves, 16 GiB, IQ2 | 11.34 (uncapped) | 11.35 | 11.34 (uncapped) | `expert_waves3_speed_results.json` |

The short runs were 0.4 to 1.2% faster. The 13,772-token CLI run executed
38,991 waves over 11,058 layers, and the 22,481-token run 70,589 waves over
18,885 layers; their GPU time rose from 10.77 to 10.78 s to 11.41 s and
from 18.05 to 18.08 s to 19.22 s, and their reads fell by less than 1%.
The long run executed 966,787 waves over 286,484 layers; it was 1.25%
faster than its first control and 0.76% slower than its closing control.
Main-queue GPU time rose from 298.04 and 299.04 s to 312.60 s, and logical
reads fell by about 0.6%. The cap reduced GPU work without a further gain.

**Why it did not help.** Waves added GPU work and command buffers, and
the gain in short runs was not confirmed by the long run. The time of a
layer still ended with its last expert's read.

**What was removed.** Both switches, `expert_waves.c`, `cache_wait_any`
and its wake-up signal in `src/engine/io.c`, `gpu_open_late` in
`src/engine/gpu.c` (a command buffer that joins the queue only when it is
committed, which the early-encoded next front used), the wave branches of
`step_rows` and `layer_front`, the builds `build/myds4-expert-waves`,
`build/myds4-expert-waves3`, `build/expert-waves-model-check`,
`build/expert-waves3-model-check`, `build/expert-ready-check`, the server
build with its library (`build/libmyds4-expert-waves.a`,
`build/experimental/model-expert-waves.o`,
`build/experimental/io-expert-waves.o`), and the checks
`tools/checks/expert_waves_model_check.c` and
`tools/checks/expert_ready_check.c`.

### Compact partial batches

**Problem.** When only some of a row's six experts are ready, the first
expert batch still launches GPU work for all six, and the inactive ones
return at once. The empty work still costs dispatch time.

**How it worked.** The build `build/myds4-expert-compact` (switch
`MYDS4_EXPERT_COMPACT`, code in `src/experimental/expert_compact.h` and
`.metal`) wrote the active experts' original indices into the unused end
of each 256-byte descriptor. The up kernel then dispatched only the active
experts, and the down kernel one SIMD group (a group of 32 GPU threads that
run together) per active expert. Results kept their original indices, so
the final ordered sum was unchanged. The all-ready path and GGUF experts
kept their kernels.

**How it was measured.** A resident fixture passed 768 checks across all
64 masks (`build/expert-compact-check`); model fixtures compared eight
forced steps after 257- and 16,521-token prefixes with zero differing
bytes (`expert_compact_validation.json`). Live screens used the
13,772-token request at 16 GiB and at 56 GiB.

**Results.** Resident kernel times for both partial batches and the final
sum (medians of seven alternated samples of 64 iterations, over a
902,430,720-byte weight set): 1/5 active experts 0.2840 against 0.2536 ms;
2/4 0.2877 against 0.2683 ms; 3/3 0.3052 against 0.2780 ms. At 16 GiB:
5.83 tokens/s against controls of 5.76 and 5.83, with GPU time 10.46 s
against 10.68 and 10.65 s, over 22,090 partial batches
(`expert_compact_speed_results.json`). At 56 GiB: 10.69 against 10.62 and
10.66, +0.45% (`expert_compact_56g_speed_results.json`; that record also
holds one earlier run whose cache stayed below 56 GiB, which is excluded).

**Why it did not help.** It saved 0.02 to 0.03 ms per partial layer in
the fixture, and GPU time fell about 1.9% in the 16 GiB model run. Read
waiting, about 42 s of the 53 s decode, did not change.

**What was removed.** The switch, `expert_compact.h`, `expert_compact.metal`,
the compact branch of `experts_go` and its `expert_compact` JSON line, the
builds `build/myds4-expert-compact`,
`build/expert-compact-check` and `build/expert-compact-model-check`, and
`tools/checks/expert_compact_check.c` and
`tools/checks/expert_compact_model_check.c`.

### Encoding late expert work while reads finish

**Problem.** After the late experts' reads finish, the CPU still encodes
their command buffer. That encoding delays the GPU.

**How it worked.** The build `build/myds4-late-preencode` (switch
`MYDS4_LATE_PREENCODE`) encoded the late experts' work, or the whole fused
batch when all experts were missing, before waiting for their reads. It
submitted the buffer only after the reads finished, so the GPU never read
unfinished data. The check build defined `MYDS4_LATE_PREENCODE_CHECK` to
expose the command buffer's status (`gpu_command_status`).

**How it was measured.** Model fixtures compared eight forced steps in
three cases with zero differing bytes, and 960 layer checks confirmed that
the command stayed unsubmitted until the final wait
(`late_preencode_validation.json`). Live screens used the 13,772-token
request at 16 GiB and 56 GiB.

**Results.** At 16 GiB: 5.84 tokens/s against controls of 5.83 and 5.84;
GPU time 10.65 s in all arms; read wait 42.10 s against 42.07 and 42.05 s;
11,054 mixed and 346 all-missing layers pre-encoded
(`late_preencode_speed_results.json`). At 56 GiB: −0.43%
(`five_proposals_56g_results.json`).

**Why it did not help.** In an earlier MXFP4 measurement of the ordinary
engine, encoding GPU work took about 0.4 ms per emitted token, against
about 71 ms of waiting for expert reads. Moving the encoding earlier had
little to gain.

**What was removed.** Both switches, the pre-encoding branch of
`routed_rows`, `gpu_command_status`, `build/myds4-late-preencode`,
`build/late-preencode-model-check` and
`tools/checks/late_preencode_model_check.c`.

### Compiler thread limits for the expert kernels

**Problem.** The Metal compiler accepts the largest threadgroup size (a
threadgroup is a set of GPU threads that share fast memory) that a kernel
will be dispatched with, and may compile the kernel differently for it,
for example in its register use.

**How it worked.** The switches `MYDS4_PIPELINE_LIMIT` and
`MYDS4_PIPELINE_DOWN_LIMIT` built the native `moe_up` and `moe_down`
kernels with explicit limits of 256/192 or 512/256 threads
(`build/myds4-limit256`, `build/myds4-limit512` and the checks
`build/expert-limit256-check`, `build/expert-limit512-check`). The kernels'
dispatch sizes stayed 256 and 192.

**Results.** Both variants passed 769 checks. Metal reported the requested
limits and 128 bytes of static threadgroup memory for each kernel;
register use and spills were not captured. With 256/192, the partial
sequences 1/5, 2/4 and 3/3 took 0.285, 0.290 and 0.305 ms against 0.283,
0.289 and 0.303 ms. With 512/256, two cases improved by less than 1% and
two regressed (`expert_pipeline_limit_results.json`). At 56 GiB the model
changed by −0.10% and −0.17% (`five_proposals_56g_results.json`).

**Why it did not help.** The limits did not change the kernels' speed.

**What was removed.** Both switches, `gpu_kernel_limited` and its
`pipeline_limit` JSON line in `src/engine/gpu.c`, and the four targets.

### Combined scheduling variants

**Problem.** The single changes above each gained less than 1%. Together
they might add up.

**How it worked.** `build/myds4-combined` joined compact partial batches,
earlier promotion and late pre-encoding. `build/myds4-combined-all` added
a 16-reader urgent window and the 256/192 thread limits.
`build/myds4-combined-waves` joined completion waves, compact batches and
earlier promotion (waves bypass late pre-encoding).
`build/combined-all-model-check` and `build/combined-waves-model-check`
checked them.

**How it was measured.** The 13,772-token request with its 308-token
reply, MXFP4, a 56 GiB cache, a 32,768-token context, pruned prefill and
shared expert loads. Each candidate had controls before and after. The
single changes' 56 GiB screens (`five_proposals_56g_results.json`, cited in
their sections) used the same conditions: every valid run kept 3,185 slots
(60,114,862,080 bytes), matched the output and recorded no new compression
or swap-outs. A standalone validation of the five single changes passed
11/11 memory-accounting, 8/8 allocation, 144/144 urgent-read and 12/12
promotion checks, the 768 kernel checks, five model cases at 257- and
16,521-token prefixes, and 960 pre-encoding submission checks
(`expert_commit_validation.json`).

**Results.** The numerical checks of the combinations matched in every
byte: mixed and all-missing cases for the all-five build and forced
single-expert waves for the wave combination, eight steps at a 257-token
prefix (`combined_validation.json`).

| Combination | Controls, tokens/s | Candidate | Change |
| --- | ---: | ---: | ---: |
| All five | 10.64, 10.54 | 10.56 | −0.26% |
| Compaction, promotion, pre-encoding | 10.64, 10.54 | 10.53 | −0.63% |
| Waves, compaction, promotion | 10.54, 10.60 | 10.79 | +2.05% |

Record: `combined_56g_results.json`. The wave combination ran 19,769
partial groups over 7,369 layers.

**Why it did not help.** The changes did not add up. The 2.05% of the wave
combination is one short comparison, and the waves' own long run showed
no consistent gain.

**What was removed.** The five targets above and the Makefile's
`COMBINED_*` variables.

### MXFP4 expert kernel load schedules

**Problem.** A GPU load that is used as soon as it is issued waits a whole
memory latency. Issuing an expert's weight and scale loads earlier,
relative to their use, might hide that latency in the decode expert
kernels.

**How it worked.** The switch `MYDS4_FP4_LOADS` replaced the native
`moe_up`/`moe_down` kernels with variants from
`src/experimental/mxfp4_loads.metal`: 1, early loads of a whole block
(`build/myds4-fp4-early`); 2, a next-block pipeline (`build/myds4-fp4-pipeline`);
3, row-wise lookahead (`build/myds4-fp4-row`); 4, block-scale factoring
(`build/myds4-fp4-factor`). `build/expert-batch-check --loads` also tested
matrix-unit tiles of one pair (`matrix`) and of eight rows (`matrix8`).
`--specialize` tested separate grouped pipelines for one, two and three
rows (`src/experimental/expert_rows_specialized.metal`).

**Results.**

- Early loads and the pipeline passed 324 checks each, but one row rose
  from about 0.233 ms to 0.357 to 0.386 ms.
- Row-wise lookahead and scale factoring passed and took about 0.230 ms.
  Their model check on a 13,772-token DSH prefix found zero differing bytes
  in the logits of 16 forced continuation positions, the local state and
  the global KV. On the 13,772-token request (256 tokens, 56 GiB) they
  decoded at 10.72 and 10.63 tokens/s against controls of 10.70 and 10.61,
  with identical output (`mxfp4_load_kernel_results.json`,
  `mxfp4_load_model_results.json`, `single_decode_screen_results.json`).
- The one-pair matrix tile took about 0.494 ms for one row, and the
  eight-row tile 0.358 ms against 0.231 ms for the ordinary kernel. Both
  produced values that differed from the scalar kernels, so they were not
  valid (`mxfp4_matrix8_results.json`).
- Separate row-count pipelines passed 180 checks, including early and late
  read phases, and took 0.397 ms against 0.381 ms with three rows sharing
  every expert, and 0.577 against 0.507 ms with half shared; across the 18
  tested shapes they gave no consistent gain (`expert_specialize_results.json`).

**Why it did not help.** The valid variants were as fast as the ordinary
kernels, and the faster-looking ones changed results or were slower.

**What was removed.** The switch, `mxfp4_loads.metal`,
`expert_rows_specialized.metal`, the four builds and
`build/mxfp4-load-model-check` (`tools/checks/mxfp4_load_model_check.c`).
The `--loads` and `--specialize` modes went with
`tools/checks/expert_batch_check.c`, removed with `--expert-batch`.

### Spinning on the routing signal

**Problem.** The CPU sleeps until the router's event, and waking takes
time.

**How it worked.** The switch `MYDS4_SPIN_ROUTER` (`build/myds4-spin-router`)
made the decode loop spin on the event's value instead of sleeping.

**Results.** On the 51,895-token request (MXFP4, 512 tokens, 56 GiB),
10.32 tokens/s against controls of 10.29 and 10.25
(`spin_router_results.json`).

**Why it did not help.** The difference was within the controls' spread.
This suggests that the wake-up delay was not on the path that set the
decode time.

**What was removed.** The switch, `wait_front_signal`'s spinning branch,
the build, and `gpu_spin_signal` in `src/engine/gpu.c`, which this study
and the causal spinning variant used.

## Dense kernel and activation probes

These probes used synthetic weights held in memory and were never part of
inference.

### Shared-expert gate fusion

`build/shared-gate-check` (`src/experimental/shared_gate.metal`) compared
the shared expert's gate/up projections and activation, now two
dispatches, with three one-dispatch schedules. All 36,864 output values per
variant matched across 16 input cases. Median GPU times per row were
0.028475 ms for the existing two dispatches, 0.036699 ms with register
partial sums, 0.038620 ms with 1,024 threads and 0.031918 ms with 128
threads and separate FP8 rounding (`shared_gate_results.json`). Every
fused schedule was slower. Removed: the kernel file, the check and its
target.

### Dense row padding

`build/dense-padding-check` (`src/experimental/dense_padding.metal`)
added 0, 16, 32 or 64 unused elements after each weight row of the FP8 and
BF16 decode kernels, to test whether another row alignment reads memory
faster. All 456 comparisons were bit-identical. Padding did not speed up
the FP8 output
projection, query projection or shared-expert gate. The BF16 output head
with 64 elements took 2.296 ms against 2.354 ms, about 0.058 ms of a token
that takes tens of milliseconds (`dense_padding_results.json`). Removed:
the kernel file, the check and its target.

### Direct FP8 tensor operands

`build/fp8-tensor-check` (`src/experimental/fp8_tensor.metal`) fed FP8
weights directly to the matrix units instead of staging them as BF16,
with the checkpoint's 32-by-32 block scales expanded to the API's layout.
Some BF16 outputs differed: unrounded sums had about 122 to 130 dB SNR
(signal-to-noise ratio), and rounding lowered it to 84 to 88 dB. The first
64-row tile was faster for one 128-row shape and slower for the others; a
128-row tile with weight reuse and a scaled-half input did not give a
general gain (`fp8_tensor_results.json`, `fp8_tensor_reuse_results.json`).
None gave a general speed gain, and all changed some output values, so
none replaced the dense path. Removed: the kernel file, the check and its
target.

### Packed FP8 activations

`build/packed-activation-check` (`tools/checks/packed_activation.metal`)
read activations as E4M3 codes (an 8-bit float format) with E8M0 scales
(8-bit powers of two) directly in the MXFP4 expansion kernel. This cut
input storage from 20,480 to 5,280 bytes per row and preserved all tested
values. With packing costs excluded, expanded against packed inputs took
0.146656/0.148849 ms for one row, 0.440227/0.446740 ms for three and
0.879669/0.891102 ms for six (`activation_storage_results.json`). The
packed reader was 1.3 to 1.5% slower. Removed: the kernel, the check and
its target. The activation-format reference
(`tools/reference/activation_format.h`) and its checks remain, because
they check the engine's own MXFP4 activation staging.

## Prefill studies

### Candidate-masked indexer

**Problem.** The indexer chooses which compressed positions a row's
sparse attention reads. For long contexts, layer 20 selects up to 2,048
candidate blocks of 8 positions per row, recorded in a bitmap, and the
later indexer layers select only among them. Those layers still scored
every position.

**How it worked.** The switch `MYDS4_INDEX_PRUNE` (code in
`src/experimental/index_prune.metal`) used `index_score_pruned`, which
skipped score products for positions outside the bitmap. With value 2 it
also used `index_topk_masked`, whose top-k histogram counted the excluded
positions (scored as negative infinity) in bulk, keeping ties and
nonfinite values in order. Builds: `build/myds4-index-prune` (1),
`build/myds4-index-both` (2), and with pruned prefill
`build/myds4-prune-index` and `build/myds4-prune-index-both`.

**How it was measured.** A GPU fixture passed 280 cases
(`build/index-mask-check`). Model checks in MXFP4 and IQ2 found zero
differing bytes in logits, attention state, saved prefixes and resumed
state (`build/index-both-model-check`, `build/index-prune-model-check`;
`index_both_validation_results.json`, `index_prune_validation_results.json`).
In the real 51,895-token MXFP4 case the bitmap kept 16,383 positions for
the final query. Live runs used the 51,895-token request with 512 tokens
and 56 GiB.

**Results.** The masked histogram kernel fell from 0.190 to 0.142 ms at
107,934 positions (`index_mask_kernel_results.json`). With both kernels,
decoding reached 10.33 tokens/s against 10.35 and 10.28, and prefill took
76.38 s against 74.43 and 78.92 s (`index_both_speed_results.json`). With
the score kernel alone, decoding reached 10.50 against 10.43 and 10.44,
and prefill took 113.69 s against 106.84 and 114.46 s
(`decode_long_screen_results.json`).

**Why it did not help.** The masked histogram kernel saved about 0.05 ms
per call at 107,934 positions, and neither decode nor prefill time moved
beyond the controls' spread. This suggests that indexer scoring is a
small part of a pass at these lengths.

**What was removed.** The switch, `index_prune.metal`, its kernel
selection in `src/engine/model.c` and in
`tools/checks/prefill_prune_model_check.c`, the seven targets named above
and `tools/checks/index_mask_check.c`.

### Layer-dependent prune interval and 32K outer chunks

**Problem.** `--prefill-prune` keeps a fixed dependency interval of
`2668 + max(5, lookup depth)` positions before the final state in layers
21 to 39. Later layers need less history. Separately, prefill runs in
outer chunks of 16,384 tokens, and every chunk reads the experts again.

**How it worked.** The switch `MYDS4_LAYER_PRUNE` (`build/myds4-layer-prune`)
shrank the retained interval with decoder depth. `build/myds4-chunk32`
set `MYDS4_PREFILL_CHUNK=32768`. Both kept the 2,048-row matrix batches.
`build/myds4-prefill-prune` (switch `MYDS4_PREFILL_PRUNE`) only turned
`--prefill-prune` on and served as these builds' base.

**How it was measured.** The plans passed 218,880 small causal graphs.
Model checks compared the layer-dependent plan on 16,521 tokens and 32K
chunks on 32,905 tokens with full 16K-chunk prefill, with zero differing
bytes (`prefill_night_validation_results.json`). Timing used the
107,934-token request with a saved prefix at 12,453 tokens, 256 tokens
and 56 GiB.

**Results.**

| Configuration | Prefill seconds | Logical prefill read GB | Decode tokens/s |
| --- | ---: | ---: | ---: |
| Fixed interval, 16K chunks, first control | 141.19 | 1131.38 | 9.49 |
| Layer-dependent interval | 151.89 | 1115.74 | 9.51 |
| Fixed interval, 32K chunks | 149.31 | 792.43 | 9.45 |
| Fixed interval, 16K chunks, closing control | 153.62 | 1131.38 | 9.47 |

Record: `prefill_night_speed_results.json`. 32K chunks cut reads by about
30% and raised the prefill workspace from 3.09 to 4.03 GB.

**Why it did not help.** The controls differed by 12.43 s, and neither
candidate was faster than both controls. Fewer reads did not give a
reliable prefill speedup in this screen.

**What was removed.** The switches `MYDS4_LAYER_PRUNE`,
`MYDS4_PREFILL_CHUNK` and `MYDS4_PREFILL_PRUNE`, and the targets
`build/myds4-layer-prune`, `build/myds4-chunk32`, `build/myds4-prefill-prune`
and `build/prefill-layer-prune-model-check`. `--prefill-prune` and the
16,384-token chunk remain. The CPU planner check
(`build/prefill-prune-check`) still covers the layer-dependent interval,
and `build/prefill-prune-model-check --chunk32` still sets 32K chunks
inside that check.

## Concurrent request studies

### The session probe

**What it was.** `build/session-probe` (`tools/experimental/session_probe.c`)
decoded several sessions in one model process, without the server, to
measure model work directly. It read independently rendered text prompts
through a bounded input pipe and prefilled each state from an empty cache,
in the same order, before it started the decode timer. Then it completed
the states one after the other or advanced up to four rows together. It
set and checked the user-interactive thread priority that the CLI and the
server use; earlier records did not record their inherited priority. Each
state had its own sampler (temperature 1, top-p 1, seed 12345) and stopped
at the end of sequence or the output limit. The probe was built with
`MYDS4_BATCH_PREFETCH_STUDY` (below); `build/session-probe-runahead` added
the runahead schedules. `tools/bench/session_bench.py --direct-binary`
drove it, with `--prefetch-budgets` and `--usage-credits`.

**What it measured.** Aggregate model decode time and GPU time, expert-read
bytes, read waiting, the cache's smallest and largest allocation, slot
releases, distinct expert requests and GPU weight-load groups; the last two
separate the cache's deduplication from repeated weight loads. It saved
only these aggregates and hashes of the generated token IDs; generated
tool calls were never executed. The driver required equal token hashes for
every state, a fixed cache, no substitution and the resource guards. HTTP
work, request admission, prefix reuse and client latency were outside the
probe. The runahead variant reported the requests that found a slot no
forecast had filled separately from the first uses of forecasts; such a
slot could still hold another request's pending demand read, so that count
is not a hit rate.

**Why it was removed.** It served only the studies below. The server's
scheduling is compared through HTTP with `tools/bench/session_bench.py`.

**What was removed.** `tools/experimental/session_probe.c`,
`build/session-probe`, `build/session-probe-runahead`, and the direct mode
of `tools/bench/session_bench.py` (`--direct-binary`, `--prefetch-budgets`,
`--usage-credits`).

### A shared forecast budget and usage credit across requests

**Problem.** When several requests decode together, each row issues its
own forecasts, so concurrent decoding read 12% more than serving the
requests one after the other in the four-state workload. Separately, a
shared expert receives one retention credit, however many requests use it.

**How it worked.** The switch `MYDS4_BATCH_PREFETCH_STUDY` enabled three
environment variables, read at model load:

- `MYDS4_BATCH_PREFETCH_LIMIT=0..4` capped new forecast reads across all
  concurrent rows; one unit allowed three new predictions with six reads
  in flight. Refinement still used every row.
- `MYDS4_BATCH_USAGE_CREDIT=1` added one retention credit for each
  additional request that used an expert.
- `MYDS4_RUNAHEAD=0..5` selected a runahead schedule (see the next
  section).

They were used through the session probe and through the server build
`build/myds4-server-prefetch-study`, whose engine library `server/build.rs`
selected with the environment variable `MYDS4_ENGINE_VARIANT`. The cap
passed a count of reads to admit to `cache_prefetch`, which no other caller
limited.

**Results.** In the four-state workload (the session probe, 32 GiB, up to
256 tokens per state, 898 tokens in all, identical tokens in every run, a
fixed cache and no slot releases):

| Execution | Aggregate tokens/s | Logical read GB | GPU seconds | Read-wait seconds |
| --- | ---: | ---: | ---: | ---: |
| Serial controls | 7.34 to 7.43 | 1475.14 to 1475.50 | 30.59 to 30.67 | 91.12 to 92.20 |
| Concurrent controls | 6.95 to 7.16 | 1648.20 to 1650.84 | 18.35 to 18.43 | 107.16 to 110.68 |
| Concurrent, two-row budget | 7.20 | 1636.77 | 18.35 | 106.28 |
| Concurrent, one-row budget | 7.25 | 1614.20 | 18.32 | 105.53 |

Record: `session_four_model_results.json`. The concurrent controls varied
by about 3%, and the capped runs only narrowly exceeded the faster one. The
one-row run saw background CPU activity of up to one core, within the
guard's threshold. With two concurrent DSH requests, the one-row cap took
85.17 s of model time against 86.54 and 86.73 s; serial controls took
77.54 and 78.81 s, and no concurrent forecasts at all took 91.81 s
(`session_budget_results.json`). Usage credit took 86.20 s against
controls of 85.86 and 93.79 s (`session_scheduling_results.json`, in the
table of the next section).

**Why it did not help.** The cap reduced reads a little but stayed slower
than serial decoding. On the same four-state workload, the public
`--runahead` schedule reached 7.70 tokens/s
(`session_four_runahead_results.json`), and it remains.

**What was removed.** The switch, the three environment variables,
`credit_batch_uses` and the budget wrappers (`batch_budget_active`,
`forecast_admission`, `forecast_inflight`) in `src/engine/model.c`,
`cache_repeat_credit` in `src/engine/io.c`, the `admit` argument of
`cache_prefetch` and `cache_prefetch_from` (every remaining caller admits
the whole list), the targets `build/libmyds4-prefetch-study.a`,
`build/experimental/model-prefetch.o`, `build/experimental/io-runahead.o`,
`build/myds4-server-prefetch-study` and `build/batch-prefetch-check`
(`tools/checks/batch_prefetch_check.c`), and `MYDS4_ENGINE_VARIANT` in
`server/build.rs`, which now always links `build/libmyds4.a`. The other
study servers went the same way (see the sections on cancelling unused
forecasts and on completion waves).

### Runahead schedules other than the public one

**Problem.** With several requests, a request whose experts are in memory
waits for others' reads in an ordinary batched step. Runahead lets each
request advance through the layers on its own while reads continue.

**How it worked.** Each request kept the buffers that survive a pause
between layers. The schedules were:

| Mode | Behavior |
| --- | --- |
| 1 | Wait for each GPU job to finish (the first prototype). |
| 2 | Wait for routing events and retire command buffers once per step. |
| 3 | Mode 2, and run ready requests at the same layer together. This is the public `--runahead`; until `--expert-batch` was removed, it ran their experts with the grouped kernels. |
| 4 | Mode 3, and look for such a group again after a blocking read finishes. |
| 5 | Mode 3, and run the output head once for all requests after they finish their last layer, keeping each row's final streams and mixing weights until then. |

The modes were compiled in with `MYDS4_RUNAHEAD` and chosen with the
environment variable of the same name in the study builds (the session
probe and the study server). The switch `MYDS4_RUNAHEAD_NO_FORECASTS` left
out the forecast projections during the concurrent steps; the default
compiler flags defined both.

**How it was measured.** The first screen served two DSH requests (13,772
and 51,895 tokens) through HTTP at a 56 GiB cache with pruned prefill and
shared expert loads; all replies matched, and these timings precede a fix
that kept the caller's active session unchanged:

| Schedule | Model execution seconds | GPU seconds | Logical decode read GB | Read-wait seconds |
| --- | ---: | ---: | ---: | ---: |
| Serial requests | 77.79 | 27.48 | 831.31 | 47.10 |
| Concurrent rows, first control | 85.86 | 22.99 | 1043.53 | 63.00 |
| Concurrent rows with per-request usage credit | 86.20 | 22.99 | 1042.80 | 63.29 |
| 32-token scheduling quanta (an earlier scheduler study) | 88.39 | 27.77 | 998.89 | 58.07 |
| Mode 1 | 86.71 | 27.23 | 942.65 | 49.00 |
| Concurrent rows, closing control | 93.79 | 23.33 | 1039.32 | 69.45 |

The closing control slowed substantially, so this screen established no
gain for any candidate (`session_scheduling_results.json`). A model
fixture for mode 5 advanced four states, the shortest with 16,521 prompt
tokens, four times each in MXFP4 and IQ2: every logit and state byte
matched, the caller's active state stayed unchanged, and each check ran
four head batches of 16 rows (`session_head_current_validation.json`).

**Results.**

- Mode 1: see the table.
- Mode 2, the same two requests through HTTP: 9.37 tokens/s, excluding
  prefill, against 9.23 and 8.84 (`session_signal_results.json`).
- Mode 4, the four-state workload at 32 GiB: 7.65 tokens/s against 7.70
  for mode 3 (`session_four_runahead_results.json`).
- Mode 3 through HTTP: 6.508 tokens/s against 6.010 and 6.124, 6.3% above
  the closing control (`server_runahead_repeat_results.json`), with shared
  expert loads in both arms.

Mode 5, four DSH requests at 24 GiB and 128 tokens each, in the session
probe, against mode 3 before and after (`session_head_qos_speed_results.json`):

| Measurement | Mode 3, first | Mode 5 | Mode 3, closing |
| --- | ---: | ---: | ---: |
| Aggregate decode tokens/s | 6.450 | 6.351 | 6.460 |
| Decode wall seconds | 79.374 | 80.618 | 79.262 |
| Main-queue GPU seconds | 15.739 | 15.522 | 16.010 |
| Logical read GB | 1036.831 | 1036.907 | 1036.907 |
| Read-wait seconds | 56.609 | 58.167 | 56.338 |

**Why the others did not help.** Mode 1 reduced reads and read waiting
but added waits between GPU jobs. Mode 4 formed about twice as many
groups as mode 3 (3,365 grouped calls against 1,617) and used less GPU
time (25.59 s against 27.40 s), but read waiting rose (81.49 s against
78.85 s) and throughput did not improve. Mode 5 removed a little GPU
work, but read waiting rose and throughput fell 1.5 to 1.7%; the records
do not establish why. Mode 2's event waits remain as part of mode 3.

**What was removed.** The selection of modes 1, 2, 4 and 5 and their code
in `src/experimental/runahead.c` (with their `groups_after_wait` and
head-batch counters); both switches, `MYDS4_RUNAHEAD` and
`MYDS4_RUNAHEAD_NO_FORECASTS`, since runahead is now always compiled in and
always leaves out the forecast projections in its concurrent steps; the
target `build/runahead-route-model-check`; and the options `--signal`,
`--grouped`, `--grouped-after-wait` and `--batch-head` of
`tools/checks/runahead_model_check.c`, which now always checks runahead and
requires ready groups. Mode 3 remains as `--runahead`, now with each
request's own expert loads (see `--expert-batch` above).

## Learned prediction and cache actions: the predictor lab

The online CPU predictor (`--online-predictor`) is a separate, smaller
learner and remains in the engine. The predictor lab tested two larger
ideas in a build of its own.

**Problem.** The router forecasts name a layer's experts at most two
layers ahead, and most of their reads arrive late. A learned predictor fed
by DSpark's internal signals might name every layer's experts when a
verification pass starts. Separately, a learned controller might decide
better which forecast reads to admit and which slots to evict.

**How it worked.** `build/myds4-predict-lab` (switch `MYDS4_PREDICT_LAB`,
code in `src/experimental/predictor_lab.c` and `.h`) linked the experiment
into the CLI. Environment variables set it up (`MYDS4_LAB_MODE`, `_VARIANT`,
`_BASE`, `_RANK`, `_GAIN`, `_LEARN`, `_WARMUP`, `_POLICY`, `_VICTIMS`,
`_WASTE`, `_WIDTH`, `_DISTANCE`, `_HORIZON`, `_LONG`, `_LONG_MISSES`), and
the runner `tools/experimental/predictor_lab.py` set them from its options.

- The runner ran controls and experiments one at a time, kept all output
  in memory, and saved no traces, generated text or learned weights. Before
  each run it required three quiet samples: GPU activity at most 20%, the
  GPU's resident memory at most 8 GiB, available memory of at least the
  cache plus 14 GiB, no other model or large Python job, and AC power for
  large runs. It waited up to 900 s for them (`--gpu-wait`). It stopped on
  changed output, a failed process, a shrinking cache or swapping, and on
  interference it stopped only its own child. Its trials used two DSpark
  drafts, 32 slots per draft stage, temperature 1, top-p 1, seed 12,345,
  thinking effort 75, a context of 8,192 and 512 generated tokens of the
  scheduler fixture (or of a scheduling-reasoning prompt).
- The joint predictor projected 584 input features to 127 ReLU features
  and a constant, with a fixed projection, and recursive least squares
  learned one 128-by-15,360 output matrix for all 40 backbone routers (rank
  16 to 256, default 128; `--feature-gain` scaled the latent features). The
  inputs were normalized and compressed into signed 64-value sketches:
  token embeddings, and one or all of eight groups of DSpark signals (its
  FFN inputs, four residual streams, mixing coefficients with stream norms
  and correlations, router logits, attention queries and outputs, summaries
  of its window keys, the accepted rows' routing history, and the final
  head input with its streams). Each pass predicted all its rows before it
  learned from their labels; the first 16 passes only trained. With
  `--base draft` or `--base embedding`, the GPU also projected the draft
  head state or the embeddings through each layer's router, and the
  learner fitted the residual of those keys.
- `--phase shadow` fitted ten such predictors on the same real signals and
  issued no reads. `--phase live` let one predictor's reads into the cache,
  under a policy: `rank` (each row's leading predictions), `miss` (only
  uncached candidates) or `bandit` (learned). `--long-range` added 68 heads
  that forecast four and eight layers ahead from the actual FFN input; the
  eight-layer forecast could start a read and the four-layer one cancelled
  unstarted reads outside its top six.
- The cache-action controller (`--policy native --native-router-only`)
  used the engine's own residual and one- and two-layer forecasts as
  candidates, with the online corrector off. It was a contextual bandit
  with delayed feedback: two 16-feature linear models, fitted by recursive
  least squares, estimated the value of fetching a candidate and of keeping
  a victim (from the candidate's score or rank, forecast distance and
  source, pending reads, expert bytes, past routing frequency, the
  resident's age and retention score). It fetched when the estimated
  benefit exceeded the victim's estimated cost. The victim came from the
  eight eligible slots with the lowest retention scores (`--victim-candidates`
  1 to 128); pending, pinned, protected and in-use slots were not eligible.
  The reward (`progress_bytes_v1`) was the share of a read complete at its
  first use, scaled by its bytes per 20 MB and halved for each later pass;
  an unused admission cost half its completed bytes (`--waste-cost`), and
  a victim reused within the pass or the next two added a cost. The
  `native-control` policy used the same shortlist without learning, and
  `demand` admitted no optional reads.
- Reports: a `predictor_lab` JSON record, and the prefetch metrics'
  issuers `draft` (the DSpark plan), `h4` and `h8` (the long-range
  forecasts). The runner also read the CLI's `decode_cache` record, which
  remains, and rejected a run whose cache did not hold its budget after
  every pass.

**How it was measured.** `build/predictor-lab-check`
(`tools/experimental/predictor_lab_check.c`) checked signal wiring, causal
updates, the accepted-row history and delayed action learning without a
model; `build/cache-action-check` (`tools/experimental/cache_action_check.c`)
checked victim eligibility, abstention, ownership and the rejection of
invalid choices without the GPU. The live trials ran on 2026-11-01, in
MXFP4 and IQ2. Their timings preceded the runner's per-process GPU checks,
so other GPU work may have overlapped; the record calls them provisional.
The record is `predictor_lab_results.json` (46 trials).

**Results.** The shadow study (a 28 GiB cache; 151,680 scored row-expert
selections in MXFP4 and 184,800 in IQ2; every output matched its control)
gave this raw top-six recall over all 40 layers from the start of
verification, and this coverage of the distinct demands that were not
ready when forecast (pending reads included):

| Added feature | MXFP4 recall | IQ2 recall | MXFP4 coverage | IQ2 coverage |
| --- | ---: | ---: | ---: | ---: |
| Embedding only | 29.62% | 37.92% | 14.12% | 14.43% |
| Draft FFN inputs | 29.78% | 37.55% | 14.10% | 14.18% |
| Four residual streams | 29.89% | 38.12% | 13.77% | 13.99% |
| Mixes, norms and correlations | 30.12% | 38.67% | 13.84% | 14.22% |
| Draft router logits | 29.59% | 37.50% | 13.91% | 14.34% |
| Attention queries and outputs | 30.35% | 38.08% | 13.90% | 14.31% |
| Window-key summaries | 31.16% | 37.76% | 14.59% | 14.38% |
| Accepted routing history | 31.33% | 37.88% | 14.12% | 14.26% |
| Final head and streams | 31.74% | 39.32% | 14.05% | 13.96% |
| All groups | 30.10% | 37.22% | 14.01% | 14.05% |

The cache-action trials used a 24 GiB cache that held its size after every
pass, with no released slots, compression or swapping. Each format ran an
ordinary control, the shortlist control, the learner twice, the shortlist
control again and the ordinary control again, and every output matched:

| Experts, policy | Decode seconds, two runs | Aggregate tokens/s | Mean logical read GB |
| --- | ---: | ---: | ---: |
| MXFP4, ordinary forecasts | 79.54 / 81.30 | 6.37 | 1,001.55 |
| MXFP4, shortlist control | 80.52 / 80.01 | 6.38 | 1,008.64 |
| MXFP4, learned actions | 78.14 / 77.94 | 6.56 | 956.86 |
| IQ2, ordinary forecasts | 36.56 / 38.00 | 13.73 | 413.64 |
| IQ2, shortlist control | 36.66 / 37.15 | 13.87 | 413.85 |
| IQ2, learned actions | 36.58 / 37.25 | 13.87 | 382.33 |

Against the shortlist control, the learner was 2.85% faster in MXFP4 and
read 5.13% less; in IQ2 it read 7.62% less at the same speed. Its settled
admissions had 7,082 and 7,230 first uses in MXFP4, of which 2,884 and
3,187 came in time in the intended pass (IQ2: 7,212 and 6,438; 4,246 and
3,900). It still evicted experts that were used again within its
observation window: 2,144 and 2,268 victims in MXFP4, 1,576 and 1,359 in
IQ2. An earlier 28 GiB attempt stopped when a repeated control's cache
shrank to 26.3 GiB, and is excluded.

**Why it did not help.** The DSpark signals added at most about 2 points of
raw recall over the embeddings alone, and the forecasts covered about 14%
of the demands that were not ready. The learned actions cut reads, but
their only speed gain, 2.85% in MXFP4, came from provisional timings, and
IQ2 did not speed up.

**What was removed.** The switch `MYDS4_PREDICT_LAB` and all `MYDS4_LAB_*`
settings; `src/experimental/predictor_lab.c` and `.h`;
`tools/experimental/predictor_lab.py`, `predictor_lab_check.c` and
`cache_action_check.c`; the targets `build/myds4-predict-lab`,
`build/predictor-lab-check` and `build/cache-action-check`; in
`src/engine/io.c`, the slots' action IDs, the victim shortlist
(`action_victims`), `cache_action_state` and the read-progress hook; and
the prefetch metrics' issuers `draft`, `h4` and `h8`. The issuers are now
`pc`, `h1`, `h2`, `prefill_bulk` and `other`.

## Removed modes of the kept tools

Some tools remain without the modes that served the studies above:

- `tools/bench/compare_bench.py`, the serial control/candidate runner, was
  `tools/bench/expert_batch_bench.py`. Its `expert_batch`, `expert_waves`,
  `expert_compact` and `prefetch_pool` kinds are gone, and it no longer
  adds `--draft-trim` to draft runs. Its progress lines are now
  `COMPARE_START` and `COMPARE_RESULT`, and `--experts PATH` with
  `--quant iq2` or `iq1` replaces its `--iq2` option.
- `tools/bench/session_bench.py` lost its direct mode (the session probe),
  and its `--runahead` now takes 0 or 1 per mode instead of a schedule
  number (0 or 3 through HTTP).
- `tools/bench/benchmark_suite.py` no longer parses the JSON lines of the
  removed studies.
- `tools/experimental/context_lookup_replay.c` lost `--consensus`.
- `tools/checks/runahead_model_check.c` lost its mode options, and
  `tools/checks/gpu_wait_check.c` its spinning wait.
- `tools/checks/prefill_prune_model_check.c` lost the indexer-pruning
  builds; its `--index-only` option now always stops with an error.
- `tools/experimental/predict_probe.c`, the prediction probe's capture
  code, always records all five final DSpark rows, since `--draft-trim` is
  gone.
