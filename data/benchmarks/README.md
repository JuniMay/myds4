# Small inference evaluation pack

This pack supports short, paired inference checks with reproducible prompts.
It contains two public datasets, original conversation-style diagnostics and
an original prompt for longer generation tests.
The public data occupy 796,760 bytes including their licenses. No model
weights, executable benchmark harnesses or generated routing traces are stored
here. [sources.json](sources.json) records the download locations, retrieval
time, file sizes and row counts.

| Source | Local data | Items | License | Evaluation used here |
| --- | --- | ---: | --- | --- |
| [GSM8K](https://github.com/openai/grade-school-math) | `gsm8k_test.jsonl` | 1,319 test questions | [MIT](gsm8k_LICENSE.txt) | Exact numeric answer after a final `#### NUMBER` line. Commas and equivalent decimal representations are normalized. |
| [HumanEval](https://github.com/openai/human-eval) | `HumanEval.jsonl.gz` | 164 tasks | [MIT](humaneval_LICENSE.txt) | Python syntax and the requested function signature only. Generated code is never executed. |
| Local conversation diagnostics | Built by `context_cases()` | 2 templates | Original project fixtures; no downloaded content | Exact JSON answer after a history of job-state changes or a code review with revised requirements. |
| Long scheduler specification | [long_scheduler.txt](long_scheduler.txt) | 1 prompt | Original project fixture; no downloaded content | Unscored generation for long-response timing, predictor learning and paired output comparisons. |
| Live predictor study | [predictor_lab_results.json](predictor_lab_results.json) | Aggregate trial records | Original project measurements | Settings, throughput, read attribution, cache bounds and exact-output checks. Contains no generated answers, raw routing traces or learned weights. |

The GSM8K file includes reference reasoning and answers. HumanEval includes
reference implementations and tests. Prompt selection passes only questions
or function prompts to the model. Reference implementations and tests remain
unused during generation and are not executed by this runner.

HumanEval's `static_valid` field is not functional correctness or pass@1.
Functional evaluation would require a separate, validated execution environment.
The conversation fixtures test tracking of requirements and tool-result text;
they do not run an autonomous agent or measure TerminalBench performance.

The long scheduler prompt asks for a deterministic Rust task scheduler and
its tests. It preserves the exact 5,908-byte prompt used in the reported
4,096-output-token predictor trials. Read it as UTF-8 without trimming
whitespace. Those trials used 1,229 prompt tokens with the checkpoint's chat
format and thinking setting 16. Generated code was not executed, so the
trial measures inference behavior rather than functional correctness.

These are small diagnostic subsets of familiar public datasets. A subset score
is not a full benchmark result, and this pack establishes no claim that the
model has never seen these questions during training.

## Selection and checking

Run from the repository root:

```sh
.venv/bin/python tools/benchmark_suite.py --seed 0 --reasoning 8 --coding 4
```

This command prints the selected case IDs, checker names and prompt lengths.
It does not start inference or print the reference answers. Selection is
deterministic for a given seed and sample sizes. The default also includes both
conversation fixtures with 12 recorded checkpoints. Use `--context-turns 4`
for shorter histories or `--no-context` to omit them.

The Python API exposes `select_cases`, `context_cases`, `check_answer`,
`prompt_tokens`, `parse_metrics`, `run_case` and `run_paired`.

`parse_metrics` also preserves the engine's `decode_experts` counters for
computed backbone rows and expert sharing. Those counters include rejected
verification rows and do not represent newly useful prefetches.

```python
import json
from tools.benchmark_suite import Config, run_paired, select_cases

cases = select_cases(seed=0, reasoning=8, coding=4, context_turns=12)
# Supply checkpoint and q2_shard from command-line arguments or configuration.
configs = [
    Config("mxfp4-plain", checkpoint),
    Config("mxfp4-draft2", checkpoint, draft=2),
    Config("q2-plain", checkpoint, experts=q2_shard),
    Config("q2-draft2", checkpoint, experts=q2_shard, draft=2),
]
for result in run_paired(cases, configs):
    print(json.dumps(result), flush=True)
```

To reproduce the long-response input, construct an ungraded case from the
preserved fixture. Supply `checkpoint` through the caller's configuration:

```python
from tools.benchmark_suite import Case, Config, DATA, run_case

case = Case("scheduler-long", "context", (DATA / "long_scheduler.txt").read_text(), "ungraded")
config = Config("online-long", checkpoint, draft=2, draft_cache=32,
                cache_gb=28, context=8192, max_new_tokens=4096,
                timeout=1800, thinking=16, online_predictor=True)
result = run_case(case, config)
```

The caller must own the sole model-execution slot. The runner checks the native
model lock and launches one process at a time. It rotates configuration order
between cases. Every process uses greedy decoding, exact expert routing and
the same prompt. The selection seed chooses dataset items; greedy generation
does not use a sampling seed.

The default limits are 2,048 total context tokens, 256 generated tokens,
30 GiB requested expert-cache budget and 300 seconds per process. Actual
prompt token counts are checked against the checkpoint tokenizer before a
model starts. Oversized prompts are rejected rather than truncated. API limits
allow 512–8,192 context tokens and at most 4,096 generated tokens. A request's
prompt and output allowance must fit together. Output and diagnostics are
each capped at 256 KiB, and timeout or overflow terminates the child. Results
stay in memory or on standard output; the runner creates no logs or captures.

## Comparisons and reported measurements

`run_paired` compares the generated text of plain and draft configurations
within the same quantization. Exact greedy speculative decoding is intended
to preserve that text; the comparison checks each evaluated case rather than
assuming equality. MXFP4 and Q2 can generate different text, so their closed
answers are checked independently. Failed or interrupted runs have
`eligible_for_score=false` and are excluded from correctness scores.

`--tokens` is the inference executable's teacher-forced replay interface. It
does not measure free-generation answer quality or DSpark acceptance. This
runner uses free generation from a shared chat prompt. A separate replay on
identical token IDs can isolate routing differences, but its timings and
quality interpretation must remain separate.

Results include process wall time, time to the first output chunk, prefill and
decode time, throughput, draft passes and reported drafts retained per pass.
The first output chunk includes model startup and may contain several tokens.
Acceptance and expert-read numbers inherit the executable's printed rounding.
Reported expert-read GB count logical requested bytes, not physical SSD
traffic. Main-queue GPU time excludes the separate clock-keeping queue.

The runner records initial and final cache allocations because `--cache` is a
budget, not a guaranteed allocation. Global cache-hit percentages include
pinned DSpark requests and late prefetches; they are not an incremental
prefetch-hit metric. Compare useful completed prefetches, wait time and cache
pollution separately when the corresponding native counters are available.

Use repeated, interleaved runs for performance conclusions. Different memory
pressure, cache allocation, temperature and background activity can overwhelm
small differences from a single short comparison.
