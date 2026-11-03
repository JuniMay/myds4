# Performance

This document defines the speed limits of DeepSeek-V4.1-Flash-1017 on the
target machine, from measured hardware constants and the model's byte and
operation counts, and records what myds4 reaches.

## Machine

Apple M5 Max (18 CPU cores: 6 performance, 12 efficiency), 128 GiB unified
memory, 2 TB internal SSD, macOS 27.0.1.

| Constant | Measured | How |
| --- | ---: | --- |
| GPU memory read bandwidth | 521 GB/s | sum over a 4 GiB FP32 tensor (PyTorch MPS) |
| GPU FP16 matrix throughput | 56.8 TFLOPS | 8192 x 8192 x 8192 FP16 product (PyTorch MPS) |
| SSD read bandwidth | 14.1 GB/s | 400 expert spans not read before (18.8 MB, 16 KiB-aligned), `F_NOCACHE`, 8 threads |
| CPU-GPU round trip | 130 to 160 µs | a tiny command buffer signals an event the CPU waits for (`--bench`) |
| GPU work after 3 ms idle | 2.7 to 3.7 x slower | six 6.5 MB matrix-vector products after a 3 ms pause, against back to back (`--bench`; it varies from run to run) |
| The same, one simdgroup spinning during the pause | 1.07 to 1.6 x slower | 111 against 104 µs on 2026-10-30, 151 against 94 µs on 2026-10-31 (`--bench`) |

Reads of 1 to 17 MiB per request reach the SSD bandwidth from 4 threads up;
one thread reaches 11.8 GB/s. Spans read before come from the page cache at
about 42 GB/s, so measurements use spans not read before. On 2026-10-30,
after a day of benchmarks, the same kind of read (150 expert spans) reached
13.4 to 13.7 GB/s from 4 to 16 threads and pieces of 2 to 32 MiB.

The GPU lowers its clock within milliseconds of idling and takes milliseconds
to raise it again; the High Power energy mode does not change this (2.7 x
measured in it). The desktop used 14% of the GPU while myds4 was not running
during these measurements.

## Model bytes and operations

The checkpoint holds 475 GiB: routed experts 269 GB (MXFP4), the two Engram
tables 194 GB, and 8.5 GB of dense weights. One routed expert (`w1`, `w2`,
`w3` of 5120 x 2304 with their scales) is 18.8 MB. A token selects 6 experts
in each of the 40 layers. An expert's FP4 codes carry 3.9 bits per 4-bit code
and its scales about 1 bit per byte (the entropy of their value counts, on 3
experts): coding codes and scales by their own value frequencies would
shorten an expert's read by about 8% at most.

Bytes read from memory per decoded token:

| Part | Bytes |
| --- | ---: |
| Attention projections (`wq_a`, `wq_b`, `wkv`, `wo_a`, `wo_b`) | 5.06 GB |
| Shared experts | 1.42 GB |
| Router weights and hyper-connection mixes | 0.31 GB |
| Compressors, indexers, Engram projections | 0.40 GB |
| Output head (BF16) | 1.32 GB |
| Dense total | 8.51 GB |
| Routed experts, 240 of them | 4.51 GB |
| Total | 13.0 GB |

Operations per prompt token: about 2 x 15.1 G multiply-adds for the
projections and experts, plus attention over the 128-token window and up to
512 selected compressed positions, about 33 GFLOP in all.

## Limits

The limits below follow from the measured constants above and from routing
traces replayed in `tools/cachesim.py`; they are computed, not measured runs.

**Decode.** A decoded token passes through the 40 layers in order and uses
6 routed experts in each, 240 in all. Two constants follow from the
measurements above:

- G = 25.0 ms, the GPU time of a token: its 13.0 GB read once at 521 GB/s.
  With every expert in memory, a token takes G (40 tokens/s).
- t = 1.34 ms, the time the SSD takes to read one expert (18.87 MB at
  14.1 GB/s).

Two conditions bound the time of a token, for any program on this machine:

1. A token takes at least G + D x t. D counts the token's demand reads:
   experts that are not in memory when their layer selects them. The layer
   cannot finish before they arrive, and the GPU has no other work
   meanwhile, because the next layer needs this layer's output. Only the
   layer's shared expert and its cached experts, at most 0.25 ms of GPU
   work, can run during the wait.
2. A token takes at least R x t. R counts all expert reads per token: the
   demand reads and the reads ahead, used or not.

The hit rate h is the share of the 240 experts that are in memory when their
layer selects them, cached or read ahead in time: D = 240 x (1 - h). Each
percent of hit rate is 2.4 demand reads, 3.2 ms per token.

R has a floor that depends only on the cache's size. Belady's optimal policy
evicts, on a miss with a full cache, the expert whose next use lies farthest
in the future. It reads the fewest experts that any policy can with the same
number of slots. Reading ahead changes when an expert is read, not whether it
must be read, so no read-ahead lowers that floor.

A speed target needs both conditions:

| Target | Time per token | Demand reads at most (hit rate at least) | All reads at most |
| ---: | ---: | ---: | ---: |
| 40 tokens/s | 25.0 ms | 0 (100%) | 18.7 |
| 30 tokens/s | 33.3 ms | 6.3 (97.4%) | 24.9 |
| 25 tokens/s | 40.0 ms | 11.2 (95.3%) | 29.9 |
| 20 tokens/s | 50.0 ms | 18.7 (92.2%) | 37.4 |
| 15 tokens/s | 66.7 ms | 31.2 (87.0%) | 49.8 |
| 10 tokens/s | 100 ms | 56.1 (76.6%) | 74.7 |

**Cache size.** A cache of S GiB holds 56.9 x S experts. myds4's whole
memory is the cache plus 8.0 GiB of dense weights and buffers, so a
whole-program budget of 50 to 65 GiB leaves the cache about 41 to 56 GiB.
`tools/cachesim.py` replayed the routing of three 2,000-token generations (a
story, a Python module with tests, an explanatory essay; the first 28 tokens
warm the cache). Reads per token, under Belady's optimal policy and under
myds4's eviction by score (see the design):

| Cache (slots) | Story: optimal, myds4 | Python module: optimal, myds4 | Essay: optimal, myds4 |
| ---: | ---: | ---: | ---: |
| 41 GiB (2,330) | 18.2, 34.7 | 26.3, 51.4 | 23.5, 45.4 |
| 46 GiB (2,620) | 15.6, 29.8 | 23.2, 45.5 | 20.4, 39.7 |
| 50 GiB (2,850) | 13.8, 26.4 | 21.1, 41.5 | 18.3, 35.8 |
| 56 GiB (3,190) | 11.8, 22.7 | 18.3, 36.3 | 15.6, 30.6 |

Of these, 4.1 to 4.6 reads per token are first uses: experts the generation
had not used before, which no cache size removes and only reading ahead can
hide. The adaptive replacement cache (ARC) reads 4 to 13% more than myds4's
eviction on these traces, LRU 7 to 16% more. On the story at 46 GiB,
myds4's reads beyond the optimal policy's are all of experts used again
within 200 tokens: 6.3 per token used again within 50 tokens (the optimal
policy reads none of these) and 12.6 used again after 50 to 200 tokens
(4.7); experts used again after longer gaps, and first uses, cost both
policies the same 10.9 reads per token.

Early in a generation, first uses dominate. The traces use 4,995 to 5,547
distinct experts in their first 150 tokens, 33 to 37 first uses per token,
against 2.9 to 3.2 per token over tokens 150 to 2,000. A generation of 150
tokens from an empty cache therefore reads at least 33 to 37 experts per
token whatever the cache's size: 45 to 50 ms per token from the SSD, at
most 20 to 22 tokens/s. The 2,618 experts used most in the other two traces
account for 16 to 33% of a trace's selections in its first 150 tokens.

The speed both conditions allow when every miss is read ahead in time
(tokens/s):

| Cache | Story: optimal, myds4 | Python module: optimal, myds4 | Essay: optimal, myds4 |
| ---: | ---: | ---: | ---: |
| 41 GiB | 40, 21.6 | 28.4, 14.5 | 31.8, 16.5 |
| 46 GiB | 40, 25.1 | 32.2, 16.4 | 36.6, 18.8 |
| 50 GiB | 40, 28.3 | 35.4, 18.0 | 40, 20.9 |
| 56 GiB | 40, 32.9 | 40, 20.6 | 40, 24.4 |

Within that budget, a cache that cannot know the future reaches 25 tokens/s
only on story-like text, and only if nearly every miss is read ahead in
time. On the Python module and the essay, the SSD alone keeps exact decoding
below 25 tokens/s.

The story at a given hit rate, with myds4's reads (its hit rate without
read-ahead is 85.5 to 90.5%):

| Cache | No read-ahead | 93% | 97% | 99% | 100% |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 41 GiB | 14.0 | 21.1 | 21.6 | 21.6 | 21.6 |
| 46 GiB | 15.4 | 21.1 | 25.1 | 25.1 | 25.1 |
| 50 GiB | 16.6 | 21.1 | 28.3 | 28.3 | 28.3 |
| 56 GiB | 18.1 | 21.1 | 28.9 | 32.9 | 32.9 |

Where a row stops rising, the SSD binds (condition 2). Only fewer reads help
there: more memory, or eviction closer to the optimal policy's. Elsewhere
the hit rate binds (condition 1), and faster GPU work helps too. With the
35 ms of GPU work per token that myds4 measures (see Results) in place of G,
the 100% column becomes 21.6, 25.1, 28.3 and 28.6 tokens/s.

**Lead time.** At G a layer takes 0.62 ms, less than one read (1.34 ms), so
a miss read ahead finishes in time only if it is named several layers before
its layer. The simulator, with the GPU at 26 ms per token (`--front 0.45
--expert 0.2`) and myds4's eviction, on the first 800 tokens of the story
(tokens/s):

| Read-ahead | 46 GiB | 50 GiB | 56 GiB |
| --- | ---: | ---: | ---: |
| None | 15.5 | 16.6 | 17.9 |
| myds4's predictions | 16.7 | 17.8 | 19.1 |
| The actual experts, 2 layers ahead | 21.7 | 23.3 | 25.1 |
| The same, 4 layers ahead | 23.2 | 25.1 | 27.1 |
| The same, 8 layers ahead | 24.2 | 26.4 | 28.7 |
| The same, 40 layers (one token) ahead | 25.2 | 28.2 | 31.3 |

myds4's predictions name a layer's misses (of a 50 GiB cache) in their
top 6 at these shares: 77% from the residual before the layer's attention,
about 0.3 ms before they are needed; 51% one layer ahead; 43% two layers
ahead. The next token's misses are not named by this token's routing: of
the top uncached expert below each layer's 6 selected (40 reads per token),
4% is used by the next token. On the Python module, myds4's predictions
simulate at 13.1, 13.9 and 15.2 tokens/s at 46, 50 and 56 GiB.

**Prefill.** A prompt of more than a few hundred tokens selects nearly every
expert of every layer, so a pass reads every expert that is not in memory
once. Each myds4 process starts with an empty cache: 15,360 reads, 289 GB,
20.5 s at 14.1 GB/s, whatever the memory. A cache kept from an earlier
request would leave 15,360 - S reads for S slots: 16.7 s at 50 GiB, 13.9 s
at 88 GiB. Computation takes N x 33 GFLOP for N tokens: at the matrix
units' 56.8 TFLOPS, 1.6 s for 2,764 tokens, 5.2 s for 8,916 and 9.5 s for
16,384; at the 19 to 21 TFLOPS the prompt kernels reach overall (the
8,912-token prefill's GPU time, see Prefill kernels), 4.3 to 4.8 s, 14 to
15 s and 26 to 28 s (16,384 tokens: one prefill pass). A pass takes at least
the larger of reading and computing. From an empty cache, the SSD binds
below about 35,000 tokens at the peak rate and below about 12,000 to 13,000
tokens at the prompt kernels' rate.

## Results

With the checkpoint's MXFP4 experts on 2026-10-31 (a 36 GiB cache, each run
a fresh process, two runs each; other applications used 13 to 22% of the
GPU during the decoding runs and 0 to 34% during the prefills):

| Workload | Measured |
| --- | ---: |
| Decode, story prompt, 150 tokens | 9.56 and 9.61 tokens/s |
| Decode, Python function prompt, 150 tokens | 8.01 and 7.89 tokens/s |
| The same with `--substitute 0.5` (one run each) | 15.98 (story) and 13.42 tokens/s (Python) |
| Prefill, 2,760 tokens (`encoding/README.md`) | 20.39 and 20.31 s, 266.4 GB read |
| Prefill, 8,912 tokens (`encoding/encoding.py`) | 22.92 and 23.76 s, 289.3 GB read, the GPU working 18.55 and 19.34 s |
| Prefill, 15,965 tokens (the first 64,000 bytes of LLVM's `ProgrammersManual.md`) | 35.11 and 37.72 s, 289.3 GB read, the GPU working 32.65 and 35.15 s |

The 8,912-token prefill's profile (`MYDS4_PROFILE=prefill`, each dispatch
alone; two runs while other applications used 12 to 26% of the GPU) took
15.8 and 17.7 s: the routed experts' tile kernels (`moe_up_m`, `moe_down_m`,
with the gather and the quantization into half) 7.2 and 8.0 s, the FP8
products 3.8 and 4.2 s (with `x_bf16`), attention 3.1 and 3.5 s (with
`q_rope`), and 1.8 and 1.9 s for the other kernels.

Measured on 2026-10-29 and 30 with the build of those days, the energy mode at
Automatic (`pmset -g`: powermode 0), on power, each run a fresh process (cold
expert cache), greedy decoding, with a fixed cache of 46 GiB (2,618 slots;
the whole program about 55 GiB) or 56 GiB (3,187 slots; about 65 GiB), while
other applications used 5 to 6% of the GPU (sampled with `ioreg` before each
run), except the two rows dated 2026-10-29 (21 to 41%). No run made the
system swap out a page.

| Workload | Measured | Limit |
| --- | ---: | ---: |
| Prefill, 2,764 tokens (`encoding/README.md`; 2026-10-29) | 125.6 tokens/s (22.0 s) | 135 tokens/s (20.5 s, SSD) |
| Prefill, 8,916 tokens (`encoding/encoding.py`) | 388 to 392 tokens/s (22.8 to 23.0 s) | 435 tokens/s (20.5 s, SSD) |
| Decode, story prompt (20 tokens), 150 tokens, 46 GiB | 11.5 to 11.7 tokens/s | 25.1 tokens/s |
| The same, 56 GiB | 12.4 tokens/s | 32.9 tokens/s |
| Decode, Python function prompt (29 tokens), 150 tokens, 46 GiB | 9.3 to 9.5 tokens/s | 16.4 tokens/s |
| The same, 56 GiB | 10.5 tokens/s | 20.6 tokens/s |
| Decode after the 2,764-token prompt, 64 tokens, 46 GiB (2026-10-29) | 8.5 tokens/s | |
| Decode after the 8,916-token prompt, 64 tokens, 46 GiB | 8.1 tokens/s | |

A decode limit is that of the corresponding 2,000-token trace with every
miss read ahead in time (see Limits); the traces' generations are longer
than these, from other prompts, and a 150-token run reads mostly first uses
(see Limits). After the 2,764- and 8,916-token prompts, a decoded token read
29.5 and 31.7 experts on demand and used 28.3 and 32.8 read ahead, against
18.6 and 15.6 after the story prompt; its GPU time (35 ms) did not change.

Both prefills read 289 GB, the whole expert set. Traced per layer (a test
build), the 2,764-token pass took 0.52 to 0.60 s per layer after layer 0
(0.72 s), reading 6.7 to 7.7 GB in each at 13.1 to 13.4 GB/s while the GPU
worked 0.10 to 0.17 s of it. The 8,916-token pass is close to both limits:
its GPU worked 16.5 to 17.5 s in all. Its layers after the first two took
0.46 to 0.64 s each, the GPU working 0.39 to 0.53 s and the CPU then waiting
for the layer's last reads; its first two layers took about 1 s each, the
second waiting for the Engram rows (24 random 4 KB reads per token and
table). On 2026-10-29, while other applications used 21 to 41% of the GPU,
four 8,916-token runs took 26.4 to 34.6 s, the GPU working 24.6 to 32.8 s;
runs with the cache's automatic size and with 30 GiB varied as much.

Per decoded token of the story at 46 GiB, the GPU worked 34 ms; the CPU
waited 39 ms for the GPU's routing results and 44 ms for expert reads. The
token read 18.6 experts on demand and 25.0 ahead, of which 15.6 were used,
and 2.5 were cancelled before reading. The old counters labeled all used
prefetches late, but the retained-group-handle bug prevents inferring physical
lateness from those counts. With the offline fitted maps used in that study
(0.3 GB), the story decoded at 11.60 to 11.72 and the Python prompt at 9.62
to 9.69 tokens/s, against 11.52 to 11.57 and 9.33 to 9.38 with the routers
(two runs each); a token read 19.6 and 31.5 experts ahead instead of 25.0
and 36.6, about as many of them used.

A demand read that is its layer's only one (timed in a test build) started
reading 0.01 ms after its request and took 1.72 to 1.78 ms to read, against
1.53 to 1.66 ms for one expert on an idle SSD; a layer then often waits
longer for its read-aheads still in flight.

Timing each command buffer (a test build) shows where the GPU's time goes.
A layer's front took 560 us against the 360 us its 188 MB need at G; its
experts took about 264 us (the head adds 2.4 ms per token to the last
layer's experts). An experts' command buffer started on average 170 to
205 us after its commit; on an idle GPU a tiny command buffer starts 96 to
120 us after its commit, and one waiting on an event that the CPU signals
ends 340 to 550 us after the signal. A buffer committed at least 200 us before
the buffer ahead of it ends starts at once; one committed 100 to 200 us
before starts about 50 us late, and one committed less than 100 us before
about 176 us late, as late as one committed after. The CPU commits a layer's
experts while the front's shared expert still runs, so with GGUF experts at
a 40 GiB cache the gaps over 100 us between command buffers are as many as
the layers that wait for an expert read: by those counts, a layer whose
experts are in memory starts them within 100 us of its front's end (see [Experts looked up by the GPU](#experts-looked-up-by-the-gpu)).

Kernel bandwidth on the model's own matrices, one matrix per layer so that
no cache holds them (`--bench`, 2026-10-31; it times each matrix's 1,024-row
products first, so that the GPU's clock is up for the one-row ones): one row
takes 14.7 us for `wq_a` (6.5 MB, 444 GB/s), 7.0 us for `wkv` (2.6 MB, 375
GB/s), 24.4 and 23.8 us for the shared expert's `w1` and `w2` (11.8 MB, 484
and 496 GB/s), 8.5 us for the router (3.9 MB of BF16, 462 GB/s), and 60.0,
73.8 and 75.3 us for `wo_a`, `wo_b` and `wq_b` (33.5 to 41.9 MB, 557 to 569
GB/s). The 1.3 GB head (Q6_K) takes 1.03 ms (527 GB/s), 6 routed experts of
the GGUF codes of layers 0 to 7 take 136 us (350 GB/s), and a decoded row's
attention over 640 rows takes 21.2 us (its three kernels). In a test program
that times chains of 40 dependent one-row products after 20 ms of other GPU
work, a one-row FP8 product takes about 3.4 us beyond its bytes at 590 GB/s,
and the products that share a dispatch take 19.6 us for `wq_a` with `wkv`
(9.2 MB), 43.6 us for `w1` with `w3` (23.6 MB), 84.4 us for `wq_b` with the
indexer's `wq_b` (47.2 MB) and 22.5 us for the router with the two
predictions (11.8 MB). Six `wq_a` products take 94 us right after other
work, 346 us after 3 ms of idling and 151 us when a simdgroup spins through
the pause. Products of 2 and 4 rows at once (`mvm_fp8`, `mvm_bf16`) take
1.12 to 1.21 and 1.57 to 1.63 times one row's time on `wq_b`, `wo_a` and the
router; 8 rows, which run as matrix products, take 2.42 times one row's time
on `wq_b` and 12.8 and 13.3 times on `wo_a` and the router.

## Approximate decoding and prefill

`--substitute D` (see the design) with the `UD-IQ2_XXS` experts, against
exact decoding with them, on 2026-10-31: a 40 GiB cache that free memory
held at 3,302 slots, one run each, other applications using 0 to 6% of the
GPU. Speed, 150 tokens, with the experts read on demand and substituted per
token:

| D | Story | Python function |
| --- | ---: | ---: |
| Exact | 20.77 to 21.48 tokens/s; 14.6 reads | 19.04 to 19.19 tokens/s; 20.7 reads |
| 0.3 | 28.00; 1.6 reads, 17.7 substituted | 25.81; 3.4 reads, 26.9 substituted |
| 0.5 | 28.26; 0.8, 19.4 | 26.37; 2.3, 29.7 |
| 1.0 | 28.74; 0.4, 19.1 | 26.50; 1.6, 30.1 |

The GPU's work then binds (about 27 ms per token, see [Where a decoded
token's time goes](#where-a-decoded-tokens-time-goes)). At D = 0.5 a token
still waited 7.7 ms (story) and 10.5 ms (Python) for experts that its layers
selected while their reads ahead were in flight, which substitution waits
for instead of replacing.

The KL divergence of each next-token distribution from exact decoding's
(nats, mean over the scored positions, `--logits-fd`), the text decoded one
token at a time from an empty cache (`--perplexity --raw`), on the first
2,000 characters of the checkpoint's README.md (649 tokens; exact
perplexity 1.2906) and of `encoding/encoding.py` (465 tokens; 6.1473): the
median KL in brackets, the positions whose most likely token is the same,
and the perplexity:

| D | README.md | `encoding.py` |
| --- | ---: | ---: |
| 0.3 | 0.026 (0.0002), 634 of 649, 1.2891 | 0.081 (0.021), 422 of 465, 6.2396 |
| 0.5 | 0.027 (0.0002), 632, 1.2871 | 0.080 (0.020), 426, 6.3392 |
| 1.0 | 0.031 (0.0002), 627, 1.2942 | 0.082 (0.025), 423, 6.0854 |

These runs decoded at 26.7 to 28.3 tokens/s (README.md) and 24.1 to 25.0
(`encoding.py`) with 23 to 32 experts substituted per token. Which experts
are resident depends on how far the reads have come, so the output varies
from run to run, and a single run measures a setting only roughly.

A prompt's rows substitute too. Through the server on 2026-10-31, each run a
fresh server whose first request (the checkpoint's README.md, 1,743 tokens)
decoded 200 tokens before three new prompts, other applications using 21 to
25% of the GPU; prefill time and the experts read, exact (one run) and with
`--substitute 0.5` (two runs):

| Prompt | Exact | `--substitute 0.5` |
| --- | ---: | ---: |
| The first request, 1,743 tokens, from an empty cache | 11.80 s, 154.5 GB | 7.93 and 7.96 s, 100.7 GB |
| 105 new tokens | 3.97 s, 52.8 GB | 1.65 and 1.61 s, 21.1 to 21.7 GB |
| 349 new tokens | 6.48 s, 86.3 GB | 3.21 and 3.15 s, 41.4 to 42.1 GB |
| 1,018 new tokens | 8.54 s, 112.8 GB | 4.80 and 4.72 s, 60.9 to 61.8 GB |

With a warm cache (a warm-up that prefilled 32 tokens of
`inference/model.py` and decoded the next 192) and D = 0.5, the positions
decoded after a 400-token prompt pass of README.md had a mean KL of 0.0113
nats from exact decoding (the same most likely token at 245 of 250), and
those after a 300-token pass of `encoding.py` 0.0517 (157 of 166). From an
empty cache, the whole README.md (a 1,198-token pass, then 512 positions)
gave 0.0391 (479 of 512). The prompt substituted no expert under the old
readiness check, which could treat completed transfers as unavailable. These
measurements do not validate approximate-output quality after that check's
correction.

Replayed offline on two 2,000-token routing traces of these experts (a
story and a Python module, recorded on 2026-10-30 at a 40 GiB cache) with a
demand-only cache of 3,523 slots evicting by score, per token: the reads,
and the routing weight replaced (the replaced experts' shares of their
layers' weights, summed):

| D | Story | Python module |
| --- | ---: | ---: |
| 0 (exact) | 22.2 reads | 32.6 reads |
| 0.1 | 10.8, 1.26 | 15.9, 2.30 |
| 0.3 | 5.6, 2.28 | 8.4, 3.71 |
| 0.5 | 4.2, 2.74 | 6.7, 4.26 |
| 1.0 | 3.6, 3.13 | 5.9, 4.64 |

## Experts from GGUF files

`--experts` with unsloth's `UD-IQ2_XXS` build of the model
(`unsloth/DeepSeek-V4.1-Flash-1017-GGUF`, its five shards without the Engram
tables, 165.3 GB): experts of 7.6 to 12.1 MB instead of 18.9 MB, so a 46 GiB
cache holds 4,051 of them instead of 2,616. On 2026-10-30, at a 46 GiB cache,
while other applications used 15 to 25% of the GPU (the rows "checkpoint"
are the checkpoint's experts under the same conditions):

| Workload | Checkpoint | `UD-IQ2_XXS` |
| --- | ---: | ---: |
| Decode, story prompt, 150 tokens | 11.6 tokens/s | 18.6 tokens/s |
| Decode, Python function prompt, 150 tokens | 9.4 tokens/s | 16.9 to 17.0 tokens/s |
| Prefill, 2,764 tokens | 22.1 s (125 tokens/s) | 12.2 s (226 tokens/s) |
| Prefill, 8,916 tokens | 23.4 s (381 tokens/s) | 21.8 s (409 tokens/s) |

A decoded token read 13.0 (story) and 17.1 (Python) experts on demand against
18.7 and 28.2, and used 9.7 and 10.9 read ahead against 15.7 and 18.9. The
GPU worked 34.7 to 35.2 ms per token with the checkpoint's
experts in these runs, and with these experts 29.4 to 30.8 ms per token in
runs of 150 tokens on the story and Python prompts at a 52 GiB cache (later
on 2026-10-30, while other applications used 0 to 83% of the GPU).

With `--bench`, 6 routed experts of these codes (layers 0 to 7, whose `w1` and
`w3` are IQ1_S) take 137 to 138 us, 345 to 347 GB/s of their stored bytes
(two runs on 2026-10-30, while other applications used 90 to 94% of the
GPU), against 456 to 470 GB/s for the checkpoint's MXFP4 experts (Results). A test
build that timed one set of 6 experts in each of the 40 layers, back to back
(two runs; `ioreg` read 22 to 92% of the GPU in use just before them),
measured 148 to 152 us per layer for both projections, 6.0 ms per token:

| Codes (layers) | `w1` and `w3` | `w2` |
| --- | ---: | ---: |
| IQ1_S, IQ2_XXS (2 to 4) | 80 to 82 us, 336 to 346 GB/s | 48 us, 378 to 380 GB/s |
| IQ1_S, IQ2_XS (5 to 9) | 80 to 81 us, 343 to 346 GB/s | 55 us, 374 to 375 GB/s |
| IQ2_XXS, IQ2_S (10 to 19) | 90 us, 407 GB/s | 54 to 55 us, 411 to 421 GB/s |
| IQ2_XXS, IQ3_XXS (20 to 24) | 90 us, 408 GB/s | 59 to 60 us, 455 to 462 GB/s |
| IQ2_S, IQ3_XXS (25 to 34) | 95 to 97 us, 467 to 480 GB/s | 59 us, 457 to 462 GB/s |
| IQ2_S, IQ2_S (35 to 39) | 94 to 97 us, 470 to 485 GB/s | 54 to 55 us, 414 to 422 GB/s |

The decoding costs about 20 us per layer beyond the products themselves: a
test build that loads 4 bytes per sub-block and skips the decoding, with the
same products, took 129 us per layer (62 to 88 us for `w1` and `w3`, 33 to
56 us for `w2`; 24% of the GPU in use before the run).

The 8,916-token prefill's GPU worked 20.3 s (19.3 s with the checkpoint's
experts, both slowed by the other applications), so with the smaller reads it
is GPU-bound; the experts of all layers, which the 2,764-token prefill reads
once, are 157 GB instead of 289 GB. Later on 2026-10-30 it took 20.8 and
23.2 s (the GPU working 19.7 and 22.2 s) at a 52 GiB cache, while other
applications used 29 to 78% of the GPU. [Prefill kernels](#prefill-kernels)
has the current kernels' times and profile. With
`--check-prefill` on the first 2,000 characters of the checkpoint's README.md
(654 tokens), two batches gave the same final logits as one, and the batched
prefill agreed with token-by-token decoding on the most likely token at 125 of
the first 128 positions and 518 of the other 526 (mean logit SNR 14.9 and
14.3 dB).

The KL divergence from exact decoding with the checkpoint's experts (nats,
mean over the scored positions, measured as in Approximate decoding, one run
each; the most likely token agreeing in brackets), and the perplexity:

| | README.md, short | `encoding.py`, short | README.md, long |
| --- | ---: | ---: | ---: |
| `UD-IQ2_XXS` | 0.087 (95.4%), 1.2953 | 0.213 (87.3%), 5.9718 | 0.118 (87.3%), 4.0838 |
| Exact | 0, 1.2682 | 0, 5.5707 | 0, 3.8643 |

The median KL was 0.0004, 0.045 and 0.027. The long `encoding.py` text was
not measured with these experts.

Exact decoding with these experts at other cache sizes (tokens/s; 150 and
400 tokens; other applications used 16 to 34% of the GPU):

| Cache (slots) | Story, 150 | Python, 150 | Story, 400 | Python, 400 |
| ---: | ---: | ---: | ---: | ---: |
| 40 GiB (3,523) | 18.1 | 16.4 | 18.5 | 16.3 |
| 46 GiB (4,051) | 18.8 | 17.1 | 18.6 | 17.6 |
| 52 GiB (4,580) | 18.7 | 17.0 to 17.2 | 19.3 | 17.4 |

From 40 to 52 GiB a token's demand reads fell from 13.6 to 10.5 (story, 400
tokens) and from 19.9 to 14.7 (Python), but the speed rose only 5 to 7%: the
GPU's work (about 30 ms per token, see above) and the experts' command
buffers starting about 170 us late leave about 37 ms per token without any
read, at most 27 tokens/s.

A token's time at 52 GiB on the story prompt (19.0 to 19.4 tokens/s, 52 ms
per token; the `decode per token` line, other applications using 30 to 40%
of the GPU): the CPU waited 33 to 35 ms for the layers' routing signals,
while the GPU worked (about 30 ms per token in all, see above) and each
experts' buffer started late, and 14 ms for expert reads, which 15.6 of the 40 layers waited
for. It spent 0.8 ms on the cache's requests and read-aheads, 0.4 ms
encoding and 1.0 ms adapting the cache between tokens. The CPU's work
between a layer's routing signal and its experts' commit does not delay the
GPU: in a test that spent about 40 us more per layer there, the speed did
not change.

With every slot resident, on 2026-10-30, exact decoding at a 40 GiB cache
(3,523 slots; two runs each; other applications used 0 to 77% of the GPU):
the story decoded at 20.43 to 20.51 tokens/s and the Python prompt at 18.42
to 18.44, the GPU working 4.38 to 4.50 s per 150 tokens. The CPU waited
29.1 to 29.4 ms per token for the GPU's signals, and its memory check
between tokens took 0.4 to 0.5 ms, while the GPU works on layer 0. At a
44 GiB cache, the 2,764-token prefill took 12.26 to 12.29 s and the
8,916-token one 20.17 to 20.19 s, the GPU working 19.2 s.

The file's Q6_K head (see the design) takes 1.03 to 1.14 ms per row against
2.5 ms for the checkpoint's BF16 head (`--bench`); a verify pass runs the
head on its rows and on the draft's five. [Speculative
decoding](#speculative-decoding) has the drafts' speed with these experts.

## Expert formats compared

The checkpoint's MXFP4 experts (18.8 MB each) and three GGUF builds on
2026-11-01 with the installed build: unsloth's `UD-IQ2_XXS` (experts of 7.6
to 12.1 MB, 2.31 bits per weight) and `UD-IQ1_M` (6.9 to 10.2 MB, 1.95),
and ISTA-DASLab's GSQ-RCO `IQ1_M` (6.9 to 8.7 MB, 1.69). A cache slot holds
a build's largest expert, so the same `--cache 56` held 3,185, 4,932, 5,797
and 6,847 slots (the GSQ-RCO cache stopped at 52.8 to 56 GiB). The unsloth
builds bring a Q6_K output head (1.0 ms per row); MXFP4 and the GSQ-RCO
build, whose head is Q5_K, use the checkpoint's BF16 head (2.5 ms). Exact
decoding, each run a fresh process, other applications using 0 to 25% of
the GPU, and no expert file in the page cache (checked with `mincore`; the
pages a download had left were dropped first). Tokens/s, 150 tokens, two
runs each in alternating order; then per token the GPU's work, its idle time
between command buffers, the experts read on demand and the bytes read:

| Build | Story | Python function | GPU | Idle (story; Python) | Demand (story; Python) | MB read (story; Python) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MXFP4 | 13.09 and 13.12 | 10.85 and 10.74 | 32.8 to 33.3 ms | 42.7 to 43.0; 58.2 to 59.2 ms | 15.3; 23.2 | 699; 954 |
| `UD-IQ2_XXS` | 22.12 and 22.29 | 20.37 and 20.40 | 27.1 to 27.3 ms | 17.2 to 17.3; 21.1 to 21.2 ms | 11.7; 14.3 | 279; 356 |
| `UD-IQ1_M` | 23.75 and 23.89 | 22.43 and 22.43 | 26.8 to 26.9 ms | 14.4 to 14.7; 17.1 ms | 12.1; 13.5 | 234; 278 |
| GSQ-RCO `IQ1_M` | 25.05 and 25.20 | 24.18 and 24.28 | 27.5 to 27.8 ms | 11.5; 13.0 to 13.1 ms | 10.7; 11.6 | 182; 187 |

Over 1,000 tokens (one run each; a text that ended earlier is counted to its
end), the story decoded at 14.74 (MXFP4, 958 tokens), 25.10, 27.48 and
27.36 tokens/s (GSQ-RCO, 324 tokens), and the Python function at 11.19,
21.61, 24.28 and 25.72 (GSQ-RCO, 692 tokens). A prompt from an empty cache:

| Build | 2,779 tokens (`encoding/README.md`) | 8,921 tokens (`encoding/encoding.py`) |
| --- | ---: | ---: |
| MXFP4 | 20.34 s, 265.4 GB read | 22.85 s, 289.3 GB, GPU 13.93 s |
| `UD-IQ2_XXS` | 11.38 s, 144.9 GB | 14.97 s, 157.6 GB, GPU 13.75 s |
| `UD-IQ1_M` | 9.72 s, 122.3 GB | 14.56 s, 133.0 GB, GPU 13.55 s |
| GSQ-RCO `IQ1_M` | 8.40 s, 105.0 GB | 14.72 s, 115.3 GB, GPU 13.70 s |

The SSD sets the order: the GPU's work differs little between the GGUF
builds, and a smaller expert both fits more slots and reads faster, so the
GSQ-RCO build reads the fewest bytes despite its heavier head. The shorter
prompt is SSD-bound; the longer one is GPU-bound for the GGUF builds. With
`--substitute 0.5` (2026-10-31, the build before, 40 and 64 GiB caches),
MXFP4 decoded the story at 19.5 to 20.8 and the Python function at 16.8 to
18.5 tokens/s, `UD-IQ2_XXS` at 29.0 to 30.1 and 28.1 to 28.7.

The KL divergence from the checkpoint's exact output (the MXFP4 experts
without substitution; nats, mean over the scored positions, the text
decoded one token at a time from an empty cache, `--perplexity --raw`), in
brackets the positions whose most likely token is the same, then the
perplexity:

| | README.md, first 2,000 characters (649 positions) | `encoding.py`, first 2,000 characters (465) |
| --- | ---: | ---: |
| MXFP4, exact | 0, 1.2634 | 0, 5.6708 |
| MXFP4, `--substitute 0.5` | 0.025 (637), 1.2738 | 0.068 (428), 5.8104 |
| `UD-IQ2_XXS`, exact | 0.087 (619), 1.2906 | 0.215 (406), 6.1473 |
| `UD-IQ2_XXS`, `--substitute 0.5` | 0.095 (622), 1.3039 | 0.207 (404), 6.0066 |

The median KL was 0.00004 and 0.016 for MXFP4 with substitution, and 0.0004
to 0.053 for the GGUF experts. The GGUF codes move the output about three
times as far as substituting MXFP4 experts does, and substitution adds
little to them.

## Prefill kernels

The prompt kernels were measured on 2026-10-31 with a test harness on random
inputs (batches of 2,048 rows; the GGUF codes' blocks with random codes and
scales; each row routed to 6 of 384 experts of skewed popularity; best of 3
to 5 rounds; other applications used 0 to 6% of the GPU).

Products of FP16 tiles from threadgroup memory (FP32 sums) reach 56 to 66
TFLOPS on the matrix units for tiles (M x N x K) of 32 x 64 x 64,
64 x 64 x 64, 16 x 128 x 64, 32 x 128 x 64 and 64 x 128 x 64, but 49 for
16 x 64 x 64 on 4 simdgroups (65 on 2), 44 to 45 for 32 x 64 x 128 and
64 x 32 x 128, and 31 to 34 for 16 x 32 x 128. A core does not run its
products and its other work at once: a dispatch in which a third of the
threadgroups only multiplied and the others only did ALU work (table loads
or register arithmetic; 12 KB of threadgroup memory each) took 5.19 ms, as
long as when all of them multiplied (5.18 to 5.19 ms) or all did ALU work
(5.18 to 5.20 ms); work that overlapped would have taken about 3.5 ms. The
prompt kernels therefore stage as little as they can (see
[Prefill](design.md#prefill)). Per batch, with the gather or conversion
each needs:

| Kernel | Time per batch |
| --- | ---: |
| `gg_up_m`: Q2_K, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ1_S | 19.7, 23.4, 23.3, 22.8, 23.6, 20.3 ms |
| `gg_down_m` with its quantization of h, the same codes | 10.8, 11.7, 13.2, 13.2, 12.5, 11.7 ms |
| `mm_fp8`, without its conversion: `wq_b`, `wo_b`, `sh1`, `sh2` | 3.65, 3.57, 1.07, 1.06 ms |
| `attend_m` and `q_rope`, rows from position 4,000, 640 keys each | 14.4 ms |

Their outputs did not depend on how rows and pairs are grouped into tiles,
bit for bit: for the six codes at 7 to 2,048 rows, nine product shapes at
7 to 2,048 rows and attention at positions 0 to 20,000 with 0 to 512
selected rows.

The tile kernels of the checkpoint's MXFP4 experts were measured with the
same harness on 2026-10-31 (random E2M1 codes and UE8M0 scales of 2^-9 to
2^-1, the weights of 96 experts; best of 7 rounds in each of three runs;
other applications used 0 to 26% of the GPU): per batch, `moe_up_m` with
its gather took 21.7 to 21.8 ms and `moe_down_m` with its quantization of h
10.8 to 11.1 ms. Their outputs did not depend on their products' shapes or
on the pairs per tile, bit for bit, at 7 to 2,048 rows.

Prefill with the GGUF experts on 2026-10-31 (`--raw`, a 40 GiB cache, each
run a fresh process; other applications used 0 to 6% of the GPU): the
2,760-token prompt (`encoding/README.md`) took 11.29 s in two runs, the GPU
working 4.24 to 4.36 s, and the 8,912-token prompt (`encoding/encoding.py`)
14.52 and 17.61 s, the GPU working 13.32 and 16.53 s. The 2,760-token pass
reads 145.1 GB, the experts that its rows select or its predictions name,
and waits for the SSD; the 8,912-token pass reads every expert of every
layer (157.6 GB) and is GPU-bound (0.18 to 0.31 s waiting for reads). Its
times varied by up to 24% between batches of runs
half an hour apart, with the same memory and cache state. Its profile
(`MYDS4_PROFILE=prefill`, each dispatch alone): 13.2 s in all; the tile
kernels of the GGUF codes 6.2 s (with the gather and the quantization into
half), attention 2.4 s (with `q_rope`), the FP8 products 3.1 s (with
`x_bf16`), and 1.5 s for the other kernels.

## Experts looked up by the GPU

`--gpu-route` (see the design) against the two-buffer path, exact decoding
with the `UD-IQ2_XXS` experts, a 40 GiB cache, 150 tokens, runs alternated,
other apps using 18 to 28% of the GPU. One pair of story runs that other GPU
work slowed (GPU time 4.8 to 5.4 s) is left out.

| | Two-buffer path | `--gpu-route` |
| --- | ---: | ---: |
| Story prompt: tokens/s | 20.86 | 20.97 |
| Python prompt: tokens/s | 18.64 to 18.65 | 18.81 to 18.88 |
| Story prompt: GPU time | 4.33 s | 4.37 s |
| Python prompt: GPU time | 4.32 to 4.33 s | 4.37 to 4.39 s |
| GPU idle between command buffers, per token (story; Python) | 18.4; 24.1 ms | 17.9; 23.2 to 23.3 ms |
| Gaps over 100 us, per token (story; Python) | 16.4; 20.2 to 20.3 | 14.5; 18.3 to 18.4 |
| Layers computed again on the CPU path, per token (story; Python) | | 16.5; 20.1 |

In the two-buffer path, the gaps over 100 us were similar in count to layers
that `--gpu-route` handled again because its published table did not mark all
required experts ready. These runs predate the readiness correction; those
counts do not establish which transfers were physically incomplete. The
measured idle-time reduction was about 0.9 ms per token. The texts and the perplexity
run's logits (649 positions of the checkpoint's README) are the same bit for
bit in both paths, also with a 12 GiB cache.

## Cache eviction and read-ahead

Replayed with these experts on two 2,000-token routing traces (a story and
a Python module, recorded on 2026-10-30 at a 40 GiB cache, 3,523 slots),
`tools/cachesim.py` gives these demand reads per token without reading
ahead, for myds4's eviction by score (half-life 50 tokens) and for Belady's
optimal policy, which knows the future:

| Policy | Story, 3,523 slots | Story, 4,400 | Python, 3,523 | Python, 4,400 |
| --- | ---: | ---: | ---: | ---: |
| Belady's optimal | 11.5 | 8.4 | 17.0 | 12.2 |
| myds4's scores | 22.2 | 16.5 | 32.6 | 24.5 |

With these traces' timings (front 0.53 ms and experts 0.15 ms per layer,
0.86 ms per expert read), the timing model gives 23.9 and 20.0 tokens/s for
myds4's reads ahead, and 31.3 and 27.5 for an oracle that knows each layer's
experts two layers ahead (35.6 and 32.4 forty layers ahead). The
predictions' top 3, which myds4 reads, name 38 to 48% of a layer's experts
(the residual's for the same layer, the routers' one and two layers ahead);
their top 6 name 58 to 83%. Layers 0 and 1 miss most often, 0.8 and 1.4
times per token.

The following fitted-map results are historical measurements from the
2026-10-30/31 offline study. Ridge regression fitted a correction to each
target router from an earlier layer's FFN input. The 77 BF16 maps occupied
about 302 MB. These results do not describe a selectable current engine mode;
the current learned predictor is the smaller online correction described below.

The offline fitted maps, replayed on two 1,500-token test
traces (the same story and Python prompts, 2026-10-31) after fitting on
three others (an explanation, a dialogue, a C program), name 79% and 74% of
the next layer's experts in their top 6, 4 to 7 points more than the
routers' predictions, and read 15 to 19% fewer experts ahead, for the same
speed in the timing model and 0.7 to 1.5% more in decoding these traces. In
decoding (150 tokens, one run each, 2026-10-30), they took the story from
19.2 to 20.3 tokens/s and the Python prompt from 18.9 to 19.3, with 23% and
20% fewer reads ahead for about as many demand reads.

The maps were fitted on English prose, the checkpoint's Python files and a C
program. On 2026-10-31 (exact decoding, 150 tokens, a 40 GiB cache, two runs
each, alternated, other applications using 13 to 24% of the GPU; the story
and Python rows earlier that day, at 0 to 6%), with the routers'
predictions and with the maps, tokens/s and the experts read ahead per
token, on prompts of kinds they were not fitted on:

| Prompt | Routers | Offline fitted maps |
| --- | ---: | ---: |
| Proofs (the irrationality of the square root of 2, infinitely many primes) | 18.30 and 17.67; 27.9 and 29.6 | 18.49 and 18.32; 26.2 and 27.0 |
| A JSON array of 15 fictional employees | 17.27 and 17.58; 31.7 and 31.5 | 18.11 and 17.90; 26.8 and 27.7 |
| A Rust function that merges two sorted vectors, with tests | 15.63 and 16.15; 37.2 and 35.7 | 16.24 and 16.41; 32.8 and 32.0 |
| An essay in Chinese on the Great Wall | 21.22 and 21.32; 17.4 and 17.5 | 21.64 and 20.85; 20.1 and 20.2 |
| The story prompt | 21.75 and 21.43; 20.1 and 21.1 | 22.18 and 22.02; 15.4 and 15.8 |
| The Python prompt | 19.30 and 19.30; 28.6 and 28.6 | 19.78 and 19.49; 23.4 and 24.0 |

Each text was the same with either. On the proofs, the JSON and the Rust
code the maps read 6 to 15% fewer experts ahead and decoded 1 to 5% faster;
on the Chinese essay they read 16% more ahead for about the same speed.

More slots would help more: the timing model gives 24.1, 25.4, 26.4 and
27.3 tokens/s at 3,523, 4,000, 4,400 and 4,800 slots on the story, and 20.0
to 23.5 on the Python prompt. With other applications holding about 60 GiB,
the 10 GiB reserve leaves room for about 41 GiB of slots, as much as the
GPU's limit (see GPU resident memory).

### Online expert prediction experiments

The 2026-10-31 experiments use DeepSeek-V4.1-Flash-1017
(`deepseek-ai/DeepSeek-V4.1-Flash-1017`) and its MXFP4 experts, or the
`UD-IQ2_XXS` experts from `unsloth/DeepSeek-V4.1-Flash-1017-GGUF`.
The predictors and scheduling policies in this 2026-10-31 replay study
were evaluated offline. They did not change live routing or prefetch decisions.
The separate live CPU trial is described later in this section.
Captures and fitted parameters were held in RAM. Model runs were sequential.

The replay learner predicted before learning from each token. Every row
of a DSpark verification pass shares a frozen predictor snapshot, including
rows later rejected by verification. Recursive least squares (RLS) updates
the output weights of a small nonlinear correction after the pass. Its
inputs include the target router's forecast and the current router logits.
The main model and the feature projection remain frozen. RLS uses rank 16,
ridge 16, forgetting 0.995 and at most 32 rows from each completed group.

On 256 identical teacher-forced tokens, excluding the first 32 from scoring,
top-six recall was:

| Experts and predictor | One layer ahead | Two | Four | Eight |
| --- | ---: | ---: | ---: | ---: |
| MXFP4, router forecast | 69.78% | 62.26% | 52.71% | 39.50% |
| MXFP4, online RLS | 74.19% | 68.04% | 60.08% | 49.26% |
| UD-IQ2_XXS, router forecast | 69.56% | 62.32% | 52.55% | 39.30% |
| UD-IQ2_XXS, online RLS | 74.04% | 67.92% | 59.76% | 49.19% |

The offline fitted maps in this study reached 68.42% and 68.15% two layers ahead on these
same runs. Raw recall measures selected experts named by the forecast. It
does not measure missing experts fetched in time, waiting avoided or speed.

#### Cache attribution and progressive forecasts

The replay separated each backbone expert demand at
routing into four mutually exclusive categories: an existing resident hit,
a new timely prefetch hit, a late prefetch miss, or a demand miss. Later reuse
of an already consumed prefetched expert is a resident hit. Repeated forecasts
and requests for cached or pending experts receive no new-read credit.
The denominator is the union of experts across verification rows for each
layer/pass, including rejected rows. Pinned draft experts are excluded.

Read-cohort precision uses only newly issued prefetch reads whose intended
deadline is observed. It separates timely, late and eventual first use from
unused completed, partial, unstarted and outstanding reads. Native same-layer
residual forecasts (`early_pc`) and future-layer forecasts have separate
cohorts. Attribution remains with the original issuer through refinement.
Transferred-byte precision is also reported. These are logical read bytes,
not physical SSD traffic measurements.

Progressive forecasts refine the same target at distances eight, four, two
and one. The current top candidate can be admitted immediately; candidates
ranked second or third need support from the preceding forecast. This is an
uncalibrated rank/agreement heuristic. Only wholly unstarted reads can be
cancelled by refinement. It does not use future routing labels.

The following replay uses the 128-output-token Rust merge DSpark captures,
two draft tokens, a 30 GiB cache ceiling and exact routing. There are 45
MXFP4 and 48 Q2 verification passes. The first 28 verification passes warm
the cache; the scored tails contain 9,075 unique demands/49 emitted tokens
and 10,473 demands/57 emitted tokens. Cache capacity is fixed at the captured
maximum: 1,322 MXFP4 or 2,048 Q2 backbone slots. The native adaptive cache
may hold fewer slots, so these are not reproductions of live cache behavior.

| Experts and policy | Existing resident hits | New timely prefetch hits | Total ready hits | Timely precision of new future-layer reads | Lost resident hits |
| --- | ---: | ---: | ---: | ---: | ---: |
| MXFP4, router/current schedule | 54.70% | 0.17% | 54.87% | 15 / 2,418 = 0.62% | 0 |
| MXFP4, RLS/current schedule | 54.68% | 0.42% | 55.10% | 38 / 2,053 = 1.85% | 0 |
| MXFP4, RLS/progressive | 54.68% | 2.94% | 57.62% | 267 / 1,458 = 18.31% | 19 |
| Q2, router/current schedule | 68.23% | 4.41% | 72.64% | 450 / 1,877 = 23.97% | 61 |
| Q2, RLS/current schedule | 68.17% | 4.88% | 73.04% | 487 / 1,545 = 31.52% | 51 |
| Q2, RLS/progressive | 68.30% | 5.06% | 73.36% | 504 / 1,004 = 50.20% | 52 |

Hit percentages use the demand denominator. New timely hits include native
same-layer prefetches; future-read precision excludes them. Lost resident
hits compare readiness with a same-capacity demand-only LRU shadow. They
are a pollution proxy, not a measured latency cost. A policy can lose some
resident hits while gaining others, so total hits alone conceal pollution.

For progressive RLS, the future-read cohorts were:

| Experts | Timely first use | Late first use | Later first use | Unused complete | Unused partial | Never started |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MXFP4 | 267 | 681 | 23 | 72 | 21 | 394 |
| Q2 | 504 | 192 | 53 | 154 | 11 | 90 |

No reads in these cohorts had an unobserved deadline or remained outstanding.
Unused means unused by the end of this short replay. Some could be reused
later. Late reads can reduce remaining demand waits, but are not ready hits.

The timing model has one aggregate 12 GB/s transfer budget, 2 MiB pieces,
LRU replacement and at most six pending reads per verification row. Per-layer
compute is assumed to be 0.68 ms multiplied by `1 + 0.35 * (rows - 1)`.
It excludes predictor cost, actual worker latency, GPU contention and
native score-based replacement. Generic deadline/pressure cancellation may
drop a started read after its current piece; this approximates native rules.

| Experts | Router/current | RLS/current | RLS/progressive | RLS/SARSA controller |
| --- | ---: | ---: | ---: | ---: |
| MXFP4, modeled ms/emitted token | 137.50 | 136.18 | 135.81 | 136.14 |
| Q2, modeled ms/emitted token | 57.69 | 56.99 | 56.97 | 57.02 |

Progressive RLS improves modeled time over the same fitter/current schedule
by only 0.28% and 0.02%. The simple SARSA controller does not establish an
additional benefit. Its coarse state and short replay limit learning.
These differences are too small to claim a live speedup, especially before
charging prediction and feature costs.

A protected progressive variant permits speculative eviction only of unused
speculative entries. After warmup it issued no new reads: every slot held a
demand-used resident. It preserved the demand-only hit rates, 54.63% and
68.35%, but lost prefetch overlap. A future probation buffer must charge its
reserved slots against the same total memory budget and measure that loss
of resident capacity. Protecting all previously useful experts is not a
workable steady-state admission policy at a full cache.

An all-true-expert oracle, within the same pass and queue limits, gives
132.09 ms/output for MXFP4 and 50.63 ms/output for Q2 at distance two.
Distances four and eight provide no material further gain in this model.
This is a constrained scheduling comparison, not a universal speed ceiling
or proof that the oracle schedule is optimal.

#### DSpark signals

Four separate uninstrumented 64-token Rust merge runs, before the cache
readiness fix, illustrate the memory trade-off at a 30 GiB cache ceiling:

| Experts | Plain tokens/s | DSpark, two drafts, tokens/s | Plain expert read bytes | DSpark expert read bytes |
| --- | ---: | ---: | ---: | ---: |
| MXFP4 | 6.96 | 6.34 | 113.16 GB | 130.68 GB |
| Q2 | 13.83 | 12.25 | 50.69 GB | 62.94 GB |

These are single short runs, with actual cache budgets around 20–21 GiB,
not general benchmark results. DSpark kept 1.78/2 drafts with MXFP4 and
1.56/2 with Q2. Plain/draft outputs were identical within each quantization.
GPU work decreased from 2.25 to 1.78 seconds and 1.84 to 1.58 seconds,
but the draft pinned 384 experts occupying about 6.8 GiB, leaving fewer
backbone slots. The reported read bytes sum requested read-piece lengths;
they do not measure physical SSD traffic. These runs do not measure the
effect of the new predictor or the readiness fix.

The draft-signal experiment compared known seed/proposal embeddings with
sketches of DSpark router outputs, four residual streams, mixing coefficients,
stream norms/correlations and the final head input. Proposal row `r` uses
draft internal row `r - 1`. Seed context is explicitly flagged. These inputs
are evaluated at a common point after proposal IDs are available.

On a separate 96-token subset-sum reasoning prompt, after four verification
passes of warmup, a rank-16 ReLU RLS predictor scored seven backbone layers
(0, 1, 4, 8, 20, 37 and 39). The seed/proposal embedding baseline reached
41.41% MXFP4 and 42.28% Q2 top-six recall. Adding final residual-stream
sketches reached 42.16% and 42.38%. Other individual signals or their combined
sketch did not consistently improve recall. This small sample does not
establish a strong DSpark-to-backbone predictor. Draft expert IDs cannot be
reused directly: DSpark has three stages with 128 experts each; the backbone
has 40 layers with 384 experts each.

The same reasoning captures used only 57/28/20 distinct draft experts across
the three stages with MXFP4, and 51/28/19 with Q2. All five internal draft
rows were counted. A separate demand-only LRU replay with 32 slots per stage
(96 total instead of 384 pinned slots) required 2.65 and 2.11 draft reads
per verification pass after four warmup passes. This suggests testing a
smaller draft cache, potentially releasing about three quarters of its
6.8 GiB for backbone experts. These are short-trace working-set results,
not a measured speedup. Unpinning would add draft I/O, and Q2 backbone slots
are too small for MXFP4 draft experts; an implementation needs a suitable
draft slot pool and a comparison at equal total memory.

Captured final streams reconstruct the actual head input with relative L2
errors of 0.00000472 and 0.0000294, confirming the row alignment used by this
experiment. Capture timing includes synchronous diagnostics and must not be
treated as a live prefetch lead-time measurement.

The cache readiness check now queries dispatch-group completion without
releasing the group. A retained non-null handle alone previously classified
completed reads as pending. The unchanged regression passed 40/45 checks
before the fix and 45/45 afterward. Both expert formats also passed
`--check-batch --draft 2`: zero differing logits for either co-decoded sequence
and the verification pass's first row, across 129,280 logits per comparison.
The normal and diagnostic executables built successfully. Exact routing is
unchanged; readiness can change scheduling and eligibility under substitution.
Its effect on live speed and approximate output quality remains unmeasured.

Related methods inform further experiments: [FIRM-MoE](https://ojs.aaai.org/index.php/AAAI/article/view/39106)
uses agreement among preceding-layer forecasts and adjusts prefetch width;
[ProMoE](https://arxiv.org/html/2410.22134v3) evaluates accuracy together with
timely completion; [Pythia](https://arxiv.org/html/2109.12021) applies online
reinforcement learning with bandwidth-sensitive rewards. Their reported
speedups do not transfer directly to this SSD-backed model. [Specter](https://arxiv.org/html/2511.14102v3)
uses a quantized draft aligned with the target's routing architecture, unlike
the three-stage DSpark draft here. [SpecMD](https://arxiv.org/html/2602.03921)
evaluates next-layer prefetching and cache policies separately, reinforcing
the need to measure actual cache benefit rather than routing recall alone.

### Longer online learning and context signals

The 2026-11-01 study used a 1,229-token Rust scheduler specification and 512
generated tokens with two DSpark drafts. Seed positions advanced from 1,229
to 1,739. MXFP4 executed 722 verification rows and Q2 executed 710; rejected
rows remain part of prediction and cache evaluation. This tests a longer
learning history than the short experiments above, not 8k/32k context behavior.

For eight sampled source layers predicting two layers ahead, a rank-16
score-feature RLS learner was compared with its own weights frozen after
32 verification passes. Both saw identical later inputs. Continuing updates
improved top-six recall by about 4–6 percentage points on MXFP4 and 4–7 on
Q2 across the later windows. Native forecast recall also changed with the
text, so improvement with token position alone is not evidence of learning.

Forgetting 0.995 gives a half-life of 138.3 processed rows and an effective
span of about 200. Increasing retention to 0.999 or 1.0 did not improve these
runs. Learning all 1,229 prompt rows instead of the final 32 also did not
consistently improve subsequent prediction. A longer prompt does not
automatically provide better training examples for the output phase.

Explicit context experiments added the previous accepted row's target-router
keys, decayed expert counts, previous draft acceptance, and position features.
Other experiments added four residual streams, mixing coefficients, stream
norms/correlations, or the pre-attention residual's target-router projection
and change relative to the current FFN input. Each feature used only information
available before its prediction; histories remained frozen throughout a pass.
These additions did not consistently beat matched-width controls. The current
hidden state already contains attention-derived context; no direct KV-cache
feature was tested.

Increasing nonlinear rank helped more than these extra features. Bounded
nearest-neighbor residuals and sparse corrections conditioned on current
source experts also improved raw recall. Learned first-layer weights with
MSE or hard-negative ranking loss did not beat fixed-feature RLS in these
tests. A ranking-loss learner's lower aggregate recall was not used to dismiss
it: its new-read/cache replay was evaluated separately and also showed only
a small potential latency gain.

#### All-layer correction and prefetch attribution

The complete-head experiment evaluated all 77 one- and two-layer
source/target pairs. Its capture keeps the final 32 prompt rows and all
verification rows, dropping unused raw features. The model still computes
the complete prompt. A diagnostic record captures the actual GPU h1 top-six
and h2 top-three IDs without adding GPU work. Current-policy admissions use
those captured top-three IDs as the baseline. Unavailable lower ranks are
reconstructed only for analysis. RLS features use CPU-reconstructed logits;
native routing labels are captured before substitution.

Top-six recall on all executed verification rows:

| Experts / predictor | One layer ahead | Two layers ahead |
| --- | ---: | ---: |
| MXFP4, reconstructed router | 71.65% | 63.80% |
| MXFP4, router + RLS64 | 79.18% | 74.09% |
| MXFP4, fixed fitted maps | 75.31% | 68.07% |
| MXFP4, fitted maps + RLS64 | 80.88% | 76.37% |
| MXFP4, fitted maps + RLS128 | 81.53% | 77.15% |
| Q2, reconstructed router | 71.39% | 63.59% |
| Q2, router + RLS64 | 78.77% | 73.65% |
| Q2, fixed fitted maps | 74.67% | 67.34% |
| Q2, fitted maps + RLS64 | 80.54% | 75.98% |
| Q2, fitted maps + RLS128 | 81.20% | 76.82% |

The following uses the same bounded timing model as the earlier replay:
fixed maximum cache capacity, LRU, one 12 GB/s transfer budget, and 28
verification passes of cache warmup. These are modeled results, with all
prediction, training and dispatch costs omitted.

| Experts / predictor | Existing resident hits | Fresh timely hits | Total ready hits | Timely precision of new future reads | Shadow lost hits | Modeled ms/output |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MXFP4, captured native forecasts | 64.66% | 0.36% | 65.03% | 410 / 26,912 = 1.52% | 246 | 145.65 |
| MXFP4, router + RLS64 | 64.74% | 0.68% | 65.41% | 764 / 19,562 = 3.91% | 175 | 142.67 |
| MXFP4, fixed fitted maps | 64.71% | 0.52% | 65.23% | 586 / 23,426 = 2.50% | 204 | 143.80 |
| MXFP4, fitted maps + RLS64 | 64.76% | 0.79% | 65.55% | 892 / 17,333 = 5.15% | 145 | 141.79 |
| Q2, captured native forecasts | 73.46% | 4.52% | 77.98% | 4,924 / 19,462 = 25.30% | 1,119 | 64.54 |
| Q2, router + RLS64 | 73.96% | 5.30% | 79.25% | 5,747 / 13,663 = 42.06% | 476 | 61.80 |
| Q2, fixed fitted maps | 73.72% | 5.08% | 78.80% | 5,539 / 17,051 = 32.48% | 745 | 62.66 |
| Q2, fitted maps + RLS64 | 73.99% | 5.50% | 79.49% | 5,913 / 11,712 = 50.49% | 327 | 61.31 |

Fresh-hit percentages include same-layer PC reads; future-read precision
excludes them. Read precision credits only new requests, never already-cached
or pending experts. Late reads can reduce remaining wait but are not counted
as ready hits. Shadow losses are LRU comparisons, not measured causal losses.

Correcting fitted maps with rank 128 gives 141.57 ms/output for MXFP4 and
61.21 for Q2, only about 0.16% better than rank 64 in this model. Rank 64 is
the more economical candidate for a live cost test. Neither result establishes
a measured inference speedup or a near-oracle predictor.

The standalone C cost check in `tools/online_predictor_check.c` includes
feature normalization, stable selection keys and top-six selection. Using one
Accelerate BLAS thread, median times for all 77 pairs were:

| Rank | Rows per pass | Forward | RLS updates | State payload |
| --- | ---: | ---: | ---: | ---: |
| 64 | 1 | 0.488 ms | 1.842 ms | 24.09 MB |
| 64 | 3 | 1.080 ms | 5.408 ms | 24.09 MB |
| 128 | 1 | 0.604 ms | 3.886 ms | 50.58 MB |
| 128 | 3 | 1.419 ms | 11.130 ms | 50.58 MB |

Twenty numerical and safety checks passed, including a 20,000-row repetitive
input stress test. The bounded covariance guard retains learned output weights
when covariance needs resetting. The C basis generator and FP64 intermediates
differ from the Python implementation. These are isolated CPU costs; exposed
inference overhead depends on where copying, prediction and updates run in
the GPU pipeline. Synchronously adding every cost to decoding could consume
the modeled gain.

#### DSpark lookahead and resident protection

Full 384-value target-router projections of DSpark's final state were tested
with both same-row and generator-row alignment. Captured head inputs are used
where available; other internal rows are reconstructed from the four streams.
In the later half of the longer run, averaged over seven target layers,
rank-128 embedding-only controls reached 52.41% MXFP4 and 53.33% Q2 top-six
recall. Same-row final-head features reached 53.05% and 53.74%. Generator-row
features and conversion to each target's FFN normalization did not establish
an additional improvement.

Reconstruction differences were numerical, not a demonstrated model failure:
the MXFP4 head check had 93.50 dB SNR and 99.998% identical values. The
validator checks BF16 lattice steps, a coordinate-specific FP32 cancellation
bound, whole-row error and row alignment. Actual cache/trim equivalence tests
retain strict bit comparisons.

An offline DSpark plan becomes available at verification start, with no free
lead time. It refills spare read capacity after native requests and retains
original draft attribution. Newer forecasts cancel only unstarted plan reads.
With top one per row, modeled time changed from 145.65 to 145.32 ms/output
for MXFP4 and from 64.54 to 64.21 for Q2. Timely precision among new draft
reads was 26.30% and 63.32%. Top three increased modeled time to 146.50 and
64.67. Predictor cost is omitted, so these results do not justify deploying
the draft-state predictor yet.

Protecting already-resident forecasts until their target layer reduced
shadow lost hits, without giving them new-read credit. With native forecasts,
MXFP4 lost hits fell from 246 to 199 and modeled time from 145.65 to 145.48;
Q2 fell from 1,119 to 1,095 and 64.54 to 64.48. This is a replay-only policy.
It caps protection, permits demand overrides, and does not guarantee that
other useful residents will survive.

### Live CPU corrector

The engine includes the rank-64 correction, enabled with
`--online-predictor`, or persisted with `--predictor-state FILE`. It changes
future prefetch IDs while keeping model
weights and the main router unchanged. It learns during small forward and
verification passes, including rejected verification rows and tokenwise
diagnostic input. Large batched prefill does not train it. The model thread
owns copied source features and native routing labels. Updates run after
expert command buffers have been submitted, permitting overlap with GPU work.
The learner runs on the CPU.

The learner works with substitution, GPU routing, DSpark, serving, images,
traces, profiling and model-check options. State belongs to the engine and
persists across request and KV-cache resets. Approximate multirow passes use
only their computed one-layer forecast; their uncomputed second forecast is
neither read nor learned. Substitution uses native routing labels for learning.
GPU routing copies inputs before the next layer can overwrite them. Because
substitution depends on residency, changed prefetches can change approximate
outputs even though exact routing stays unchanged.

The performance trials below used exact text inference on the CPU-routing
path, with a newly initialized learner per process. Their timing results do
not measure the additional compatibility modes. The feature remains opt-in
because gains depend on the workload and short timing differences are noisy.

The initial integration study on 2026-11-01 checked both MXFP4 and Q2
with fresh in-memory learner state:

| Check | Observed result |
| --- | --- |
| Ordinary-build option parsing | 23 compatibility and existing-constraint checks passed. |
| CPU regressions | 43 learner checks, 20 math checks with a 20,000-row numerical stress run, and 178 cache checks passed. |
| Exact CPU and GPU routing | Generated output matched the native-predictor control for each quantization on the short coding prompt. |
| Exact batched and DSpark verification | Zero of 129,280 logits differed in each sequence comparison and the first-row verification comparison, including GPU routing. |
| Substitution with DSpark or GPU routing | Bounded generation completed and online updates occurred without invalid IDs or epoch skips. |
| Concurrent server requests and prefix reuse | Three concurrent requests and a repeated prompt completed; batching and reused prefixes were observed. One learning epoch persisted across requests. |
| Image request after text requests | A generated in-memory solid-color image produced nonempty text; updates continued within the same learning epoch. Visual accuracy was not graded. |
| Profiling and diagnostic pipes together | Empty and nonempty profiler settings were exercised across the two quantizations with diagnostic traces discarded through a null descriptor. Both runs learned without invalid IDs or epoch skips. |

The server checks combined substitution, two drafts, bounded draft caches
and the GPU-route option. Each server was stopped after its checks, and
model processes ran serially. These are compatibility and correctness checks,
not a performance or substitution-quality benchmark. The normal, public-vision
and diagnostic-capture executables compiled successfully. These measurements
do not test persistence across process restarts.

#### Predictor persistence validation

The persistence checks on 2026-11-01 passed all 42 state-file checks, 43
learner-module checks, 8 shutdown checks and 26 CLI checks. State tests cover
exact parameter round trips and identical continued fitting, invalid-file
rejection without live-state mutation, compatibility binding, unsafe paths
and failed atomic saves. These are correctness checks, not speed measurements.

Separate CLI processes created a state file, then loaded it while generating
from the same prompt. Output was identical between the first and reloaded
run within each quantization:

| Experts | First run: completed learning rows | First run: head updates | Reloaded run: cumulative learning rows | Reloaded run: cumulative head updates |
| --- | ---: | ---: | ---: | ---: |
| MXFP4 | 77 | 5,929 | 154 | 11,858 |
| Q2 | 66 | 5,082 | 132 | 10,164 |

Learning rows count completed eligible forward rows; head updates count
successful updates across the learner's heads. These are not two names for
generated-token counts. In both formats, loading the state with `--tokens 1`
performed no learning and reported zero saves. The file's bytes and
modification time stayed unchanged. Each format also rejected the other
format's saved state before inference and left the file unchanged.

The server persistence check also passed for creation and reload in both
formats, with 12 checks per run. Each process served three concurrent
arithmetic requests and one repeated prefix, using substitution 0.5, two
drafts, 32 draft-cache slots per stage and the GPU-route option. It then
handled SIGTERM and saved one 24,090,040-byte state file:

| Experts | Created state: learning rows / head updates | Resumed state: cumulative learning rows / head updates |
| --- | ---: | ---: |
| MXFP4 | 113 / 4,407 | 242 / 9,476 |
| Q2 | 120 / 4,680 | 243 / 9,477 |

Each creation run reported zero loads and one save; each resumed run reported
one load and one save. The learner retained one initialization epoch within
each process, with no invalid expert IDs or epoch skips. These checks establish
server integration and saved-state continuation. They do not measure model
quality or a performance gain.

Active CLI generation was interrupted with SIGINT after its first completed
verification pass in each format. Both processes exited cleanly, saved their
three learning rows and 231 head updates, and reloaded successfully. A Q2
server using the terminal interface also saved a reloadable state on SIGINT
and restored the terminal's normal screen. This terminal check did not grade
the generated answer.

#### Fresh-state decoding timings

With the 1,229-token scheduler prompt and 512 generated tokens, two drafts,
32 draft-cache slots per stage, a fixed 28 GiB cache and default retention,
the uninstrumented paired runs gave:

| Experts | Native forecast decode time | Online decode time | Native decode logical read GB | Online decode logical read GB |
| --- | ---: | ---: | ---: | ---: |
| MXFP4 | 69.47 s | 67.87 s | 862.18 | 836.62 |
| Q2 | 31.66 s | 30.97 s | 358.78 | 342.01 |

Each pair produced identical output. The learner reported no invalid IDs,
fallbacks or skipped updates. Forward CPU time totaled 350/341 ms and update
time 1,243/1,217 ms for MXFP4/Q2. Update time can overlap GPU execution;
adding it to wall time or subtracting it from wall time would double-count
parts of the run.

#### Actual new-read outcomes

A separate 512-token run enabled native attribution. It used the same prompt,
28 GiB cache and two-draft configuration. All instrumented on/off outputs
matched. The denominator is the backbone expert union per layer/pass:
127,640 demands for MXFP4 and 125,895 for Q2. Draft experts are excluded.

| Experts / predictor | Existing ready | Fresh timely | Total ready | Timely future reads / new future admissions | Absent after a prefetch eviction |
| --- | ---: | ---: | ---: | ---: | ---: |
| MXFP4, native | 67.72% | 1.62% | 69.33% | 2,062 / 27,546 = 7.49% | 14,506 |
| MXFP4, online | 67.76% | 1.87% | 69.63% | 2,389 / 19,621 = 12.18% | 13,803 |
| Q2, native | 76.56% | 2.21% | 78.77% | 2,781 / 20,988 = 13.25% | 11,301 |
| Q2, online | 76.58% | 2.52% | 79.10% | 3,176 / 13,774 = 23.06% | 10,649 |

The future-read column excludes same-layer residual-PC admissions. Its
denominator includes admitted reads that were later cancelled before starting;
it is not precision among completed transfers. Existing residents and pending
reads receive no new-admission credit. Late reads remain misses in the ready
hit rate, even when their partial progress reduces subsequent waiting.

Eviction/rerequest counts are observations, not a counterfactual measure of
damage: some victims might also have been evicted without prefetching. They
show that pollution remains possible. The online learner reduced these counts
but does not guarantee that every useful resident survives.

Raw top-three precision rose from about 85% to 92% on this task. That includes
resident experts and repeated candidates across rows. It is not a 92% prefetch
hit rate, nor 92% coverage of the six required experts. The actual fresh-hit
increase was about 0.26–0.31 percentage points of all demands.

Instrumented timing also illustrates the remaining uncertainty. MXFP4 took
70.38 s native versus 68.91 s online, while Q2 took 32.03 versus 33.14 s despite
fewer read bytes online. These runs do not establish a universal speedup.

#### Continued learning over a longer response

The longer trial generated all 4,096 requested tokens after the same
1,229-token prompt, reaching roughly 5.3k context tokens. Allocation capacity
was 8,192 tokens; this was not an 8k-token input. Two drafts and 32 cache slots
per draft stage remained enabled, with default 50-pass retention, read-cost
weighting off and final-stage trimming off. Each format ran native
forecasts, continued online learning, and a learner frozen after the early
prefix. The 512-computed-row freeze threshold crossed at 513 rows, after
which predictions and accuracy scoring continued without parameter updates.
All six outputs were identical within their quantization.

| Experts / mode | Decode seconds | Tokens/s | Decode logical read GB |
| --- | ---: | ---: | ---: |
| MXFP4, native | 548.82 | 7.46 | 6,437.95 |
| MXFP4, frozen after early learning | 539.99 | 7.59 | 6,298.09 |
| MXFP4, continued learning | 516.62 | 7.93 | 6,251.93 |
| Q2, native | 252.56 | 16.22 | 2,585.82 |
| Q2, frozen after early learning | 240.17 | 17.05 | 2,489.86 |
| Q2, continued learning | 232.08 | 17.65 | 2,446.65 |

Cache size was equal at 28 GiB within each format. Relative to native forecasts,
the adaptive runs reduced time by 5.87%/8.11% and logical read volume by
2.89%/5.38% for MXFP4/Q2. These are paired observations on one task, not a
general benchmark speed guarantee. Storage timing and run order can affect
the elapsed-time difference.

An adaptive repeat run after the native controls took 511.88 s for MXFP4 and
233.40 s for Q2, with 6,266.06 and 2,444.88 decode logical GB respectively.
Both repeated outputs matched the native controls. Thus the tested adaptive
runs were 5.9–6.7% shorter for MXFP4 and 7.6–8.1% shorter for Q2. These ranges
describe this task and test machine; the native control was not a repeated
distribution of workloads.

The last reporting window provides a comparison on the same later tokens:
the native column is the uncorrected candidate precision recorded inside
the online runs, not a separate instrumented native execution.

| Experts | Native raw top-three precision | Frozen learner | Continued learner |
| --- | ---: | ---: | ---: |
| MXFP4 | 82.53% | 87.01% | 91.40% |
| Q2 | 85.06% | 89.25% | 92.24% |

This supports continued adaptation, rather than assuming that context length
alone makes routing easier. These remain raw candidate statistics, not cache
hit rates. Neither the longer history nor the stronger corrector demonstrated
a near-oracle predictor.

Each adaptive run triggered 19 guarded covariance resets and rejected 19
updates out of more than 412,000/426,000 pair-row updates. The guard retained
learned output weights. No invalid expert IDs or inference fallbacks occurred.
The exact reset cause was not recorded per event; these counters should remain
visible when testing longer sessions.

## Runtime checks after restart

On 2026-11-01, bounded tests reproduced three implementation defects before
their fixes. The retained checks then passed on the rebuilt engine:

| Check | Reproduced behavior | Checked result |
| --- | --- | --- |
| Prefetch admission release | A completed or cancelled read retained its admission while its dispatch notification was delayed; 8/10 checks passed. | The final piece releases the admission immediately; 10/10 checks passed. |
| Available-memory estimate | A zero kernel file-cache floor was retained as unavailable, even after a later successful query; 3/7 checks passed. | Each estimate reads the current floor, with a conservative fallback on failure; 7/7 checks passed. |
| GPU clock calibration | Injecting 10 ms of host scheduling delay made a requested 2 ms spin last 0.469 ms. | GPU timestamp calibration produced a 1.650 ms median in the final check, within the test's 1.5–3 ms tolerance. |

These checks establish the corrected behavior. Repeated live comparisons did
not establish a material throughput gain from the admission or calibration
fixes. The memory check validates recovery when the kernel value changes; it
does not prove that this defect caused an earlier slow user session. Energy
use was not measured.

Separating the rank-64 recursive least-squares output-weight loop from its
covariance loop reduced the 77-head update microbenchmark from 1.831 to
0.697 ms for one row, and from 5.366 to 2.182 ms for three rows. In live
256-token runs, measured predictor CPU time fell from 658 to 438 ms for
MXFP4 and from 651 to 416 ms for Q2. Decode time remained effectively flat:
24.78 versus 24.99 s, and 12.85 versus 12.79 s respectively. Lower CPU time
alone is not evidence of higher model throughput.

Final validation passed 45 readiness, 49 draft-cache, 37 cache-policy,
56 attribution/timing, 10 admission, 7 memory, 43 learner, 42 persistence,
19 sampling, 8 shutdown and 20 numerical checks, including the 20,000-row
numerical stress test. All 33 CLI checks passed. The normal, public-vision
adapter and diagnostic-probe executables built successfully.

The rebuilt normal engine also generated 256 tokens in each quantization at
temperature 1, top-p 1, seed 12,345, thinking effort 75 and a 32 GiB cache,
using the scheduler fixture and fresh online learning. Both outputs matched
their earlier same-quantization controls. Both caches held 32 GiB; neither
run compressed or swapped pages. These final runs enabled timing observation
and are diagnostic measurements, not an uninstrumented speed comparison.

| Experts | Existing ready demands | Fresh timely prefetch demands | Pending prefetch demands | Absent demands |
| --- | ---: | ---: | ---: | ---: |
| MXFP4 | 46,422 | 193 | 6,191 | 8,634 |
| Q2 | 51,089 | 1,320 | 3,527 | 5,504 |

Each row totals 61,440 backbone requests. Resident hits receive no prefetch
credit. On MXFP4, 745 of 1,469 one-layer first uses and 626 of 3,098 two-layer
first uses occurred before any read piece started. The corresponding Q2
counts were 232/990 and 157/2,547. These counts include only first use in the
intended forward pass. Their mean admission-to-use intervals were 3.86/6.32 ms
for MXFP4's one-/two-layer forecasts and 2.01/3.18 ms for Q2. The same-layer
residual forecast had about 0.34 ms in both formats. These observed intervals
include queueing and computation; they are not guarantees of available I/O
time. The data show that predicting a useful expert and finishing its read
before use remain distinct problems.

## Live predictor and cache-action trials

The retained runner is `tools/predictor_lab.py`; the optional experiment
executable is `build/myds4-predict-lab`. The implementation and feature
definitions are in [Design](design.md#live-predictor-experiment-build).
[Structured results](../data/benchmarks/predictor_lab_results.json) retain
the completed trial settings, counters and within-quantization output checks.
Raw routing traces, generated answers and learned parameter files are not
retained by these trials.

### Joint DSpark feature measurements

On 2026-11-01, the engine collected the nine documented signal groups during
real MXFP4 and Q2 inference. Ten rank-128 models learned online on the same
observations, with token embeddings alone, each added feature group, or all
groups together. Each input group used a 64-value signed sketch. The latent
feature norm was one, with a separate constant. The router-informed baseline
used DSpark's final head state for proposals and the seed's embedding.

Each quantization generated 512 tokens from the 1,229-token scheduler prompt,
at temperature/top-p 1/1, seed 12,345, thinking effort 75, two drafts and a
28 GiB expert-cache ceiling. The first 16 verification passes trained without
scoring. There were 151,680 scored row-expert selections for MXFP4 and 184,800
for Q2. Every shadow output matched its control. Requested and final cache
sizes matched; this study preceded the per-pass cache-bound report.

| Added feature | MXFP4 raw top-six recall | Q2 raw top-six recall | MXFP4 coverage of not-ready demands at forecast | Q2 coverage of not-ready demands at forecast |
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

These forecasts cover all 40 layers from verification entry. They are not
the one-/two-layer corrector's accuracy. Not-ready coverage uses distinct
expert demands across rows and includes already-pending transfers; it is
not a useful-new-read hit rate. The shadow study collected real model
signals but issued no extra reads, so it establishes no prefetch speedup.
It tests these compressed features and this learner, not every possible
network using the original tensors.

### Learned cache actions

The timing observations in this subsection preceded per-process GPU checks.
Other kernel work may have overlapped. The small throughput differences are
provisional until repeated with the runner's GPU guard; they do not establish
an isolated causal speedup. Output comparisons and the recorded counters
describe the runs that actually completed.

A separate live comparison used the existing router forecasts as candidate
actions, with no extra DSpark predictor and with the ordinary online router
corrector disabled. The contextual bandit learned fetch/abstain decisions
and victim selection from actual read progress and observed victim reuse.
Its two value functions used 16 features and updated during inference.
The reward was the documented `progress_bytes_v1` proxy, not a measurement
of counterfactual wall time.

The scheduler prompt, 512-token output, sampler and draft settings matched
the feature study. The cache was 24 GiB, with minimum and maximum allocated
bytes checked after every forward pass and no released slots. No run
compressed or swapped memory. Each format ran an ordinary control, the
eligible-shortlist control, the learner twice, the shortlist control again,
and the ordinary control again. Every output matched within its quantization.

| Experts / policy | Decode seconds, two runs | Aggregate tokens/s | Mean logical expert-read GB |
| --- | ---: | ---: | ---: |
| MXFP4, ordinary forecasts | 79.54 / 81.30 | 6.37 | 1,001.55 |
| MXFP4, matched shortlist control | 80.52 / 80.01 | 6.38 | 1,008.64 |
| MXFP4, learned actions | 78.14 / 77.94 | 6.56 | 956.86 |
| Q2, ordinary forecasts | 36.56 / 38.00 | 13.73 | 413.64 |
| Q2, matched shortlist control | 36.66 / 37.15 | 13.87 | 413.85 |
| Q2, learned actions | 36.58 / 37.25 | 13.87 | 382.33 |

In those recorded runs, the learner was 2.85% faster than the matched control on MXFP4 and reduced
logical read volume by 5.13%. Q2 read volume fell by 7.62%, with no throughput
improvement against its matched control. These are repeated observations on
one prompt and cache budget, not a general speed guarantee.

The learner's settled admissions included 7,082/7,230 first uses on MXFP4,
of which 2,884/3,187 were timely in the intended pass. For Q2 the corresponding
counts were 7,212/6,438 and 4,246/3,900. Existing cached experts received no
admission credit. The controller still evicted experts reused within its
observation window: 2,144/2,268 victims on MXFP4 and 1,576/1,359 on Q2.
These are observed reuses, not a counterfactual measure of total damage.

An earlier 28 GiB attempt stopped when a repeated control shrank to 26.3 GiB.
That unequal-budget repeat is excluded from this comparison. The runner's
per-pass cache report now also rejects transient shrinkage followed by growth.

## Where a decoded token's time goes

Exact decoding with the `UD-IQ2_XXS` experts on 2026-10-31, 150 tokens, a
40 GiB cache that free memory held at 3,271 to 3,302 slots (37.1 to 37.5
GiB), each run a fresh process, other applications using 0 to 6% of the
GPU. The story prompt decoded at 21.24 and 21.72 tokens/s, the Python
prompt at 19.15 and 19.25, and with `--draft 2` the Python prompt at 19.54
(its stages' experts left 2,547 slots for the layers). Per token:

| | Story | Python | Python, `--draft 2` |
| --- | ---: | ---: | ---: |
| GPU work | 26.7 and 27.4 ms | 26.9 and 27.0 ms | 21.5 ms |
| Experts read on demand | 14.6 | 20.8 and 21.0 | 29.3 |
| Experts read ahead (used) | 20.0 to 20.1 (11.5) | 28.6 to 29.0 (13.1 to 13.3) | 38.6 (19.7) |
| CPU waiting for the routing signals | 24.8 and 25.5 ms | 24.3 and 24.4 ms | 13.2 ms |
| CPU waiting for expert reads | 18.9 and 19.2 ms | 25.0 and 25.3 ms | 33.4 ms |
| GPU idle between command buffers (gaps over 100 us) | 18.7 and 19.0 ms (20) | 24.3 and 24.6 ms (24) | 28.8 ms (15) |

The old counters reported every used prefetch as late; physical lateness was
not established because completed groups could retain their handles. The
GPU's work, profiled (`MYDS4_PROFILE=1`, story, 40
tokens, each dispatch alone: 27.0 ms per token): the one-row FP8 products
13.9 ms (of them the paired forms, `mv_fp8_3`, 3.8 ms), the BF16 products
(the router with its two predictions, the compressor and the indexer's
weights) 1.6 ms, the routed experts 6.4 ms, the head 1.0 ms, attention
(scores, values and their combination) 1.0 ms, the hyper-connections'
mixes and norms 1.6 ms, and the other kernels 1.5 ms. The products' rates
are under [Results](#results) (kernel bandwidth). With and without drafts,
each run's text was that of exact decoding, and `--check-batch --draft 2`
found no logit that differs.

## GPU resident memory

On the development machine the GPU stalls once the resident memory of all
processes passes about 55 GiB. Measured on 2026-10-30 with test builds that
keep every slot resident and set no limit, decoding with the `UD-IQ2_XXS`
experts, story prompt, 100 tokens, exact, other applications using 13 to
26% of the GPU and holding about 2 GiB of resident GPU memory and 12 GiB
allocated. "Resident" and "allocated" are the GPU driver's "In use system
memory" and "Alloc system memory" for all processes, sampled every 4 to 8 s
during the run; no run shrank its cache, compressed or swapped anything:

| Cache | Free memory | Resident | Allocated | GPU time | tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 42 GiB | 15.8 GiB | 52.8 GiB | 62.8 GiB | 2.91 s | 20.0 |
| 42 GiB, with 5 GiB taken by another process | 10.7 GiB | 52.5 GiB | 63.0 GiB | 2.87 s | 20.2 |
| 44 GiB | 13.9 GiB | 54.4 GiB | 65.1 GiB | 3.26 s | 18.9 |
| 46 GiB | 11.4 to 12.6 GiB | 53.4 to 57.1 GiB | 66.2 to 67.3 GiB | 27.3 and 47.1 s | 3.4 and 2.0 |

During the slow runs the GPU was only 17 to 46% busy while its command
buffers ran. Less free memory alone (the second row) did not slow it, and
the number of slots did not decide it: with the checkpoint's larger
experts, 2,958 slots (52 GiB, fewer than the 3,699 of the first row) took
10.1 s of GPU time against 3.5 s at 40 GiB. Allocated memory that is not
resident does not count: the build of 2026-10-30 13:15, which took slots
unused for 8 tokens out of the residency set, decoded normally at a 52 GiB
cache (GPU 3.04 s, 18.7 tokens/s), with 68.9 GiB allocated but 36.3 GiB
resident. The cache keeps
the slots it maps for the GPU, with the weights, below 41% of the physical
memory (52.5 GiB; see the design): with caches of 44 to 53 GiB, the GPU's
resident memory for all processes stayed at 48.9 to 51.2 GiB on 2026-10-31
(see [Expert formats compared](#expert-formats-compared)). Reading the
driver's statistics after every token kept the cache's checks at 0.4 to 0.6
ms per token (2026-10-30). Whether `sudo sysctl
iogpu.disable_wired_collector=1` removes the stall was not tested.

Mapping slot-sized buffers for the GPU, on 2026-10-31 (a test program
with myds4's GPU code; other applications used about 20% of the GPU):
adding a 12.2 MB buffer to the residency set and committing took 95 to 111
us of CPU time, an 18.9 MB one 153 to 173 us, also 24 at once; taking one
out took 0.2 to 6 us. The GPU read a buffer just added, or one used only
through `useResource`, about as fast as one long in the set. The driver's
resident memory grew with each buffer added at once, and fell only about
1.2 s after buffers were taken out (from 4.26 to 1.96 GiB for 160 buffers
of 12.2 MB).

## Memory and the file cache

On the development machine (2026-10-31) the kernel keeps 23.9 GiB of file
cache (`vm.vm_page_filecache_min`: 1,565,484 pages of 16 KiB) and starts
reclaiming memory only when fewer than 62.5 MiB of pages are free
(`vm.vm_page_free_target`: 4,000 pages). Since boot it had compressed 1.91
billion pages and dropped 221 million file pages
(`vm.pageout_inactive_dirty_internal`, `vm.pageout_freed_external`). While
the machine idled with 67 GiB free, it compressed and swapped nothing for
30 s.

With `--cache 64`, the story prompt, exact decoding, each run a fresh
process (other applications used 0 to 22% of the GPU), no run swapped
anything out and the swapper never woke. "Free" is the free memory when the
cache starts, after the weights; the process's own memory was never
compressed:

| File cache before the run | Experts, tokens | Free | Cache at the end | Tokens/s | Compressed by the system |
| --- | --- | ---: | ---: | ---: | ---: |
| 15.9 GiB, below the kernel's share | GGUF, 300 | 56.5 and 58.8 GiB | 44.4 and 42.7 GiB | 23.79 and 23.38 | nothing |
| 15.9 GiB | MXFP4, 300 | 59.2 and 54.7 GiB | 43.7 and 42.3 GiB | 13.04 and 12.59 | nothing |
| 39.0 GiB, 15.1 GiB beyond the share | GGUF, 600 | 34.4 GiB | 34.1 GiB | 21.23 | nothing |
| 46.1 GiB, 30 GiB of it read just before | GGUF, 600 | 25.8 GiB | 20.2 GiB | 16.66 | 40 MiB |

With 39.0 GiB of file cache the cache took free memory below the reserve,
the file cache beyond the share standing in for it, and the kernel dropped
no file pages. Right after 30 GiB of other files had been read, the file
cache shrank by 4.2 GiB during the run, the kernel compressed 40 MiB, and
the cache stopped growing at 20.2 GiB.

## Images

Measured on 2026-10-31 with the `UD-IQ2_XXS` experts and a 40 GiB cache
(other applications used 0 to 21% of the GPU). The vision encoder took 0.42
s for an 800 x 551 photo (40 x 58 patches, 296 tokens) at its first use and
0.25 s after, and 2.39 s for a 4096 x 2160 photo (66 x 125 patches, 948
tokens); its attention's work grows with the square of the patches. Through
the server, a first request with the small photo prefilled its 308 tokens in
6.15 s from a cold expert cache (76 GB of experts read), and the
conversation's second turn, which resent the image, reused 338 of its 349
tokens and prefilled 11 in 0.91 s. `tools/check.py vision` with the small
photo took 48 s, most of it the reference on the CPU.

Attention moved from the ALUs (FP32) to the matrix units on 2026-11-01.
Each process encoded three letter-size pages (1275 x 1650 pixels, 104 x 81
patches, 982 tokens), then the 4096 x 2160 photo, in runs that alternated
with the ALU build, 3 s apart:

| Attention | First page | Later pages | Photo | GPU per page (profile) |
| --- | ---: | ---: | ---: | ---: |
| ALUs, four rounds | 2.47–2.50 s | 2.34–2.41 s | 2.25–2.31 s | 2,358 ms, attention 2,084 ms |
| ALUs, two later rounds | 2.52–3.60 s | 2.83–3.30 s | 3.07–3.44 s | |
| Matrix units, alternating with the later rounds | 0.74 s | 0.62–0.63 s | 0.62–0.64 s | 592 ms, attention 305 ms |

Other applications used 24 to 33% of the GPU in all rounds; why the later
rounds' ALU times were slower is not known. The attention's 9.35 TFLOP per
page run at about 31 TFLOPS, as the ViT's matrix products do (`mm_bf16`,
217 ms). Feature SNR against the
checkpoint's CPU BF16 reference (`tools/check.py vision`, after the
Aligner), overall / worst row / median row:

| Image | ALUs | Matrix units |
| --- | ---: | ---: |
| Page 7 (104 x 81 patches) | 37.5 / 19.7 / 40.7 dB | 35.3 / 5.8 / 40.7 dB |
| 4096 x 2160 photo (66 x 125) | 32.6 / 7.0 / 39.1 dB | 34.3 / 6.0 / 39.1 dB |
| 107 x 65 thumbnail (31 x 50) | 36.4 / 18.4 / 40.4 dB | 35.9 / 16.3 / 40.4 dB |

The worst rows are rows of small norm (about half the median's), whose
features the 32 blocks make sensitive to small rounding differences; such
rows also set most of the overall figures' differences. The photo's
`--vision-fd` hash changed from 9292f85c4db456a7 to 60bff1d9cb4bef8d; the
text hashes and `--check-batch --draft 2` did not change.

## Speculative decoding

### Temperature and top-p verification

On 2026-11-01, the shared CLI/server sampler passed 19 CPU checks covering
softmax and nucleus probabilities, deterministic ties, small positive
probabilities, EOS, every draft-rejection position, all-accepted bonus tokens
and matching random-generator state. The updated CLI suite passed 33 checks.
Its sampled-draft cases failed on the prior executable before implementation.

Twelve live CLI runs used the same short merge-sorted-lists prompt, seed
12,345, a 64-token limit, a requested 28 GiB cache and online prediction.
For each quantization, plain and two-draft output matched at temperature/top-p
1/1, 0.8/0.8 and 0/1. Draft runs used 32 cache slots per stage and trimming.
The temperature-1/top-p-1 timings were:

| Experts | Plain decode | Two drafts | Drafts accepted per pass |
| --- | ---: | ---: | ---: |
| MXFP4 | 10.38 s | 10.85 s | 1.74 |
| Q2 | 5.70 s | 6.15 s | 1.56 |

These were single short comparisons with an adaptive cache, not repeated
performance trials. They establish sampled verification support and matching
output on the tested sequences. Drafting was slower in these two comparisons;
supporting a configuration does not establish that it is the fastest.

Both quantizations also passed 26 server checks with temperature/top-p 1/1,
two drafts, 32 draft-cache slots, the GPU-route option, online prediction and
state saving. The checks observed actual draft passes, concurrent requests,
prefix reuse, numeric request overrides, thinking overrides, invalid-input
rejection, image requests and clean-stop saving. A fixed-seed streamed request
matched its non-streamed counterpart. Effective sampler settings and draft
counts were read from server logs. Image output was checked for successful
execution, not visual accuracy.

### Sampled draft-length sweep

On 2026-11-01, the original scheduler fixture in
`data/benchmarks/long_scheduler.txt` supplied 1,229 prompt tokens. Each run
generated 256 tokens with temperature 1, top-p 1, seed 12,345, thinking
effort 75 and context capacity 8,192. Online learning started fresh in each
process. Routing was exact; substitution, draft trimming, GPU routing and
read-cost weighting were off. Retention used the default 50 forward passes.
Draft runs reserved 32 slots per DSpark stage within the same 40 GiB total
expert-cache budget. All runs held that budget without compression or swap.

| Drafts per pass | MXFP4 tokens/s | Q2 tokens/s | MXFP4 decode read GB | Q2 decode read GB |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 10.30 | 20.05 | 269.05 | 102.65 |
| 1 | 9.99 | 18.30 | 295.85 | 122.95 |
| 2 | 9.93 | 17.22 | 311.70 | 136.87 |
| 3 | 9.77 | 16.93 | 322.48 | 145.05 |
| 4 | 9.17 | 14.41 | 347.78 | 165.06 |
| 5 | 8.68 | 15.07 | 365.48 | 173.57 |

Each entry is one measured run, not a repeated workload distribution. GPU
timing varied during the sweep. Read GB counts aligned logical expert reads,
including draft-stage reads; it is not a physical-SSD byte measurement.
Every generated output matched the no-draft output within its quantization.
This checks output preservation on this sequence, not general task quality.

| Experts / drafts | Computed backbone rows | Selected experts across rows | Distinct layer-batch expert requests | Within-batch duplicate share | Mean accepted draft prefix |
| --- | ---: | ---: | ---: | ---: | ---: |
| MXFP4 / 0 | 256 | 61,440 | 61,440 | 0% | — |
| MXFP4 / 2 | 292 | 70,080 | 54,042 | 22.9% | 1.63 |
| MXFP4 / 5 | 377 | 90,480 | 55,759 | 38.4% | 3.05 |
| Q2 / 0 | 256 | 61,440 | 61,440 | 0% | — |
| Q2 / 2 | 386 | 92,640 | 68,484 | 26.1% | 0.98 |
| Q2 / 5 | 630 | 151,200 | 88,045 | 41.8% | 1.42 |

Duplicate share is `1 - distinct requests / selected experts`; it is not a
cache or prefetch hit rate. The five-draft Q2 run computed 630 backbone rows
to emit 256 tokens. Its substantial expert overlap still left more distinct
layer-batch requests than plain decoding. MXFP4 reduced that request count,
but read more bytes. Draft pools reduced backbone cache space, and speculative
rows also affected eviction and prefetching; this sweep does not isolate
those contributions.

The current small-batch path shares dense-weight work and combines duplicate
expert requests before reading. It runs each row's routed expert products
separately. Verification of up to six rows does not use the large-prefill
expert matrix kernels. Larger expert overlap therefore does not automatically
produce equivalent reuse of expert weights inside GPU kernels. The supported
draft block is five tokens; increasing that limit requires changes to draft
shapes and verification state, not just a larger CLI value.

On this prompt and configuration, plain decoding had the highest throughput
in both formats. This result does not select a universal draft length for
other prompts, longer sessions or different cache budgets.

### Greedy timing studies

On 2026-11-01 with the installed build, `--cache 56`, exact, 150 tokens,
one run each (tokens/s, drafts kept per pass in brackets; the plain columns
from [Expert formats compared](#expert-formats-compared)):

| Build | Python: plain | `--draft 2` | `--draft 3` | `--draft 2 --draft-cache 32` | Story: plain | `--draft 2` | `--draft 2 --draft-cache 32` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MXFP4 | 10.8 | 10.94 (1.63) | | | 13.1 | 10.22 (0.72) | |
| `UD-IQ2_XXS` | 20.4 | 24.35 (1.71) | 24.78 (2.26) | 23.93 | 22.2 | 20.29 (0.82) | 20.22 |
| `UD-IQ1_M` | 22.4 | 27.87 (1.73) | 28.21 (2.19) | 26.73 | 23.8 | 21.33 (0.71) | 20.95 |
| GSQ-RCO `IQ1_M` | 24.2 | 31.56 (1.73) | 34.09 (2.41) | 30.04 | 25.1 | 25.59 (1.00) | 24.93 |

The stages' pinned experts take slots from the layers (4,932 become 4,338
with `UD-IQ2_XXS`); `--draft-cache 32` gives most of them back (4,784), but
the stages' reads then cost more than the slots gain. On code, with 1.7 of 2
drafts kept per pass, the drafts gain 19 to 30% (3 drafts: 21 to 41%); on
the story, with at most one kept, they lose 9 to 10% with the unsloth builds
and 22% with MXFP4, and gain 2% with the GSQ-RCO build. The earlier runs
below used older builds and smaller caches.

With the checkpoint's experts, at a 46 GiB cache, 150 tokens, on 2026-10-30
(tokens/s; drafts kept per pass; other applications used 0 to 31% of the
GPU; the story's plain row comes from another run that day). The stages'
experts take 7.3 GB of the budget, so with `--draft` the layers keep 2,232
slots instead of 2,616:

| Prompt | Plain | `--draft 2` | `--draft 3` |
| --- | ---: | ---: | ---: |
| Story | 11.6 | | 6.91 (0.85) |
| Python function | 9.40 | 8.58 (1.63) | |

The text of `--draft 2` on the Python prompt was that of plain decoding. A
verify pass shares the dense weights between its rows, but each row needs
its own routed experts, and the rows of a pass share few of them: a decoded
token waited for 44.5 expert reads with 2 drafts on the Python prompt
against 28.1 when decoding plainly, and for 49.3 with 3 drafts on the story
against 18.7. On the story, with less than one draft kept per pass, the
extra rows cost more than they give.

With the `UD-IQ2_XXS` experts on 2026-10-31, a 40 GiB cache that free
memory held at 3,302 slots, 150 tokens, one run each (the plain exact rows:
three), other applications using 0 to 6% of the GPU. Per token: GPU time,
experts read on demand, experts substituted, and the draft's time; drafts
kept per pass in brackets:

| Run | Tokens/s | GPU | Demand | Substituted | Drafting |
| --- | ---: | ---: | ---: | ---: | ---: |
| Python, exact | 19.04 to 19.19 | 27.4 ms | 20.7 | | |
| Python, exact, `--draft 2` (1.71) | 20.11 | 21.1 ms | 28.0 | | 2.5 ms |
| Python, exact, `--draft 3` (2.26) | 18.78 | 21.3 ms | 31.8 | | 2.2 ms |
| Story, exact | 20.77 to 21.48 | 27.1 ms | 14.6 | | |
| Story, exact, `--draft 2` (0.82) | 17.95 | 30.7 ms | 24.4 | | 3.7 ms |
| Story, exact, `--draft 3` (0.91) | 16.27 | 35.1 ms | 29.2 | | 3.7 ms |
| Python, `--substitute 0.5` | 26.37 | 27.1 ms | 2.3 | 29.7 | |
| Python, `--substitute 0.5 --draft 2` (1.63) | 31.47 | 20.5 ms | 7.5 | 46.7 | 2.6 ms |
| Python, `--substitute 0.5 --draft 3` (2.33) | 34.26 | 19.7 ms | 7.5 | 47.4 | 2.2 ms |
| Story, `--substitute 0.5` | 28.26 | 26.9 ms | 0.8 | 19.4 | |
| Story, `--substitute 0.5 --draft 2` (0.75) | 25.17 | 30.3 ms | 4.9 | 46.8 | 3.9 ms |

The exact runs' texts were those of plain decoding. In exact decoding a
pass's rows wait for more reads per token than a decoded row (on the Python
prompt 28.0 and 31.8 on demand with 2 and 3 drafts, against 20.7), and the
GPU idles longer (27.7 and 31.1 ms per token, against 24.1), so on that
build 2 drafts gained 5 to 6% and 3 drafts lost 1 to 2%. With `--substitute 0.5` a token reads few experts on
demand, and the drafts gain 19 and 30% on the Python prompt. On the story,
with less than one draft kept per pass, the extra rows cost more than they
give either way.

### Optional draft cache and row trimming

On 2026-11-01, `tools/benchmark_suite.py` ran four GSM8K questions, two
HumanEval prompts and two original context diagnostics, using greedy exact
routing and a 512-token output allowance. Each of the six configurations ran
the same eight prompts once, with order rotated across prompts. All 48 runs
succeeded, and plain/pinned/bounded-draft outputs matched within each
quantization. Five of six closed-answer cases passed in every configuration;
the bookkeeping context case failed in both formats. Both code responses
passed syntax/signature checks. Generated code was not executed, so this is
not a functional HumanEval score.

The first screen ran while research captures occupied host memory. The
requested cache ceiling was 30 GiB; actual cache sizes were smaller:

| Experts / mode | Aggregate decode tokens/s | Actual cache GiB |
| --- | ---: | ---: |
| MXFP4, plain | 6.21 | 15.8–16.2 |
| MXFP4, pinned draft 2 | 5.93 | 20.2 |
| MXFP4, draft 2, 32 slots/stage | 5.90 | 15.2–15.8 |
| Q2, plain | 12.93 | 16.4–17.6 |
| Q2, pinned draft 2 | 10.38 | 15.5–17.0 |
| Q2, draft 2, 32 slots/stage | 12.03 | 15.7–16.1 |

The bounded draft cache improved Q2 decode throughput by 15.9% relative to
the pinned draft, but plain decoding remained faster. MXFP4 decode time was
essentially unchanged while cache memory fell by about 4–5 GiB. Its pinned
configuration was at the minimum backbone capacity, so those MXFP4 rows are
not equal-actual-memory comparisons. No energy consumption was measured.

After releasing the large research captures, a second screen used one math
prompt, one code prompt and both context prompts. Every configuration used
the same candidate binary and requested a 30 GiB ceiling. All 40 outputs
matched their same-quantization plain reference:

| Experts / mode | Aggregate decode tokens/s | Actual cache GiB |
| --- | ---: | ---: |
| MXFP4, plain, half-life 50 | 7.78 | 30.0 |
| MXFP4, plain, half-life 25 | 7.86 | 30.0 |
| MXFP4, pinned draft 2, half-life 50 | 7.27 | 28.5–30.0 |
| MXFP4, draft 2, 32 slots/stage, half-life 50 | 7.71 | 28.2–30.0 |
| MXFP4, draft 2, 32 slots/stage, half-life 10 | 7.72 | 28.2–30.0 |
| Q2, plain, half-life 50 | 15.07 | 30.0 |
| Q2, plain, half-life 25, byte cost | 15.04 | 30.0 |
| Q2, pinned draft 2, half-life 50 | 15.51 | 29.7–29.8 |
| Q2, draft 2, 32 slots/stage, half-life 50 | 16.08 | 29.6–30.0 |
| Q2, draft 2, 32 slots/stage, half-life 10, byte cost | 16.26 | 29.7–30.0 |

These are short single-run screens, not a claim that the smaller differences
are repeatable. The cache benefit depends on available memory and workload.
Half-lives count forward passes, not emitted tokens when verification batches
accept several rows.

`--draft-trim` preserves all five K/V rows and removes only unused final-stage
query/output/FFN work. Direct captures with one, two, three and five drafts
were bit-identical on all retained stage/head values, proposal IDs and accepted
prefixes for both quantizations. Cache-limit 128 versus 32 captures were also
bit-identical. These checks did not relax floating-point comparisons.

A separate 16-run timing screen compared trimming on/off at 32 slots/stage.
All eight candidate/reference output comparisons matched. Aggregate Q2 decode
throughput was 12.334 versus 12.340 tokens/s, effectively unchanged. MXFP4 was
5.768 versus 5.533; one context run had a large additional read wait despite
slightly fewer logical bytes and unchanged reported GPU time. This does not
establish a repeatable trimming speedup. The flag remains optional.

### Cache retention and reload cost

A demand-only replay of the longer scheduler capture retained prefill history
and excluded 28 verification passes from scoring. With fixed maximum capacity,
native-style frequency aging at 50 passes required 1,663 MB per accepted row
for MXFP4 and 655 MB for Q2. A 10-pass half-life required 1,511 and 588 MB.
Weighting Q2 eviction priority by aligned reload bytes reduced the latter to
579 MB. This replay has no prefetch queue or timing model; these are byte
comparisons, not measured speedups. The short live screen above did not
reproduce those large gains; the longer check below shows why the history
setting must be evaluated on the intended workload.

Q2 experts occupy fixed 12,189,696-byte slots. The longer capture's
activation-weighted logical footprint leaves about 15.8% of that reservation
unused on average. This is a capacity opportunity, not implemented variable
slot packing. The byte-cost option changes only decode victim priority;
prefill eviction, pressure release and GPU mapping priority remain unchanged.

A longer live check used the 1,229-token scheduler prompt, thinking effort 16
and 512 generated tokens. All outputs matched within each format:

| Experts | Plain, half-life 50 | Pinned draft 2 | Draft 2, 32 slots/stage | Draft 2, 32 slots/stage, half-life 10 |
| --- | ---: | ---: | ---: | ---: |
| MXFP4, decode seconds | 54.15 | 77.90 | 70.63 | 66.40 |
| Q2, decode seconds | 26.71 | 33.32 | 32.24 | 29.69 |

The final Q2 column also enables read-cost weighting. The combined changes
improved draft throughput by 17.3% for MXFP4 and 12.2% for Q2 relative to the
pinned draft, but plain decoding was still faster on this prompt. In plain
mode, half-life 25 gave 52.12 s for MXFP4 and 26.66 s for Q2 with byte weighting;
half-life 10 gave 53.03 and 28.46 s. Those outputs also matched the defaults.
The results support exposing retention controls, not replacing the default
with one setting for every workload.

## Long contexts

On 2026-10-31 with the GGUF experts (40 GiB cache; other applications used
0 to 6% of the GPU), on LLVM's `ProgrammersManual.md` (41,269 tokens):
`--perplexity --prefill 41000 --ctx 65536` (three prefill passes, then the
last 269 tokens decoded one at a time) decoded at 16.2 tokens/s at about
41,000 positions. The perplexity was 1.0999, the most likely token right at
260 of 269 positions. At `--ctx 1048576` (index scores in tiles of 64 rows)
and `--ctx 65536` (tiles of 1,024) the logits of all 269 positions were
identical.

The same day through the server, a 41,281-token prompt (the same file and a
request for a one-sentence summary) prefilled in 77.6 s, and its session's
state took 0.21 GiB. On 2026-10-30, a 41,316-token prompt with a sentence
placed after its first 3,000 characters got an answer that named the
sentence's code word, and a follow-up question ran 20 of its 41,342 tokens.

## Attention state

The window keys, compressed rows and index keys are BF16 (see
[Numerics](design.md#numerics)). A session's state takes 66 MiB at first,
with room for 16,384 positions and its saved rows, and 3.14 GiB at 1,048,576
positions (the server's startup line): 3.1 KB per position.

On 2026-10-31, on random keys (the test harness of
[Prefill kernels](#prefill-kernels), best of 3 to 7 rounds; other
applications used 0 to 6% of the GPU), a decoded row's attention over 640
rows (its three kernels) took 0.107 ms, and `index_score` for a decoded row
0.144 ms over 8,900 keys, 0.197 ms over 41,000 and 1.11 ms over 1,000,000.

The same day with the GGUF experts at a 40 GiB cache (each run a fresh
process; other applications used 0 to 6% of the GPU), after
`encoding/encoding.py` as a chat prompt (8,916 tokens) the next 64 tokens
decoded at 15.6 tokens/s in two runs, the GPU working 28 ms per token and
the CPU waiting 37 ms per token for expert reads. For about 41,000
positions, see [Long contexts](#long-contexts).

## Concurrent requests

`--batch N` (default 4) decodes up to N requests together: each forward pass
runs one row per request, which share the dense weights' reads, while each
row needs its own routed experts. The server on 2026-11-01 with the
installed build, `UD-IQ2_XXS` experts, `--cache 56`, four prompts (a story,
hash tables, the printing press, sourdough bread), greedy, thinking off, 128
tokens each, after a warm-up request, other applications idle; tokens/s for
all requests together, in brackets each request's (including any wait):

| Requests at once | `--batch 4` | `--batch 1` |
| ---: | ---: | ---: |
| 1 | 18.1 | 18.1 |
| 2 | 23.9 (11.9 each) | 18.7 (16.4 and 9.4) |
| 4 | 27.0 (6.8 each), all done in 18.9 s | 18.7 (19.7 to 4.7), all done in 27.4 s |

Batching raises the total from 18.7 to 23.9 and 27.0 tokens/s at two and
four requests at once, so four requests finish 31% sooner, but each request
runs slower than alone; a request alone runs the same either way.
`--check-batch` found two sequences decoded together equal to each decoded
alone (0 of 129,280 logits differ) with every expert build.

## Idle server

With the GGUF experts and `--substitute 0.3 --draft 2`, on 2026-10-30
(`tools/check.py idle` and `demand`): at a 44 GiB cache, from 3 s to 10 s
after a request, the server's footprint stayed at 54.7 GiB and its cache at
3,281 slots. At a 52 GiB cache the whole cache left 9.9 GiB free, at the
reserve; when the demand check took 2.9 GiB more, the server gave back 250
slots (2.8 GiB) within 5 s, nothing was swapped out and the memory pressure
stayed normal.

## Remaining gaps

- Decode with the checkpoint's experts: within a whole-program budget of 50
  to 65 GiB, the SSD keeps exact decoding below 25 tokens/s except on
  story-like text, even with every miss read ahead in time, and a short
  generation from an empty cache below 20 to 22 tokens/s at any cache size
  (see Limits). myds4 reaches 9 to 12 tokens/s: its predictions name misses
  too late or not at all (Lead time), its GPU works about 34 ms per token
  instead of 25, and its experts' command buffers start about 170 us after
  their commit, about 7 ms per token.
- With the GGUF experts of `UD-IQ2_XXS` (`--experts`), whose codes change
  the output (KL 0.09 to 0.21 nats from exact decoding), decoding reaches
  21.2 to 21.7 tokens/s on the story prompt and 19.2 to 19.3 on the Python
  prompt at a 40 GiB cache (2026-10-31; about 47 GiB for the whole
  program). The GPU works about 27 ms per token and idles 19 to 25 ms,
  waiting for the 15 to 21 experts a token reads on demand. The old late-hit
  counters do not establish physical prefetch completion times (see
  [Where a decoded token's time goes](#where-a-decoded-tokens-time-goes)).
  Of the GPU's work, the one-row dense products take 15.5 ms per token, at
  370 to 570 GB/s of their weights (`--bench`); the routed experts take 6.4
  ms and the head 1 ms.
- Memory: the cache keeps 10 GiB available (the default `--reserve`),
  counting the file cache only
  beyond the kernel's 23.9 GiB share, and backs off as soon as the kernel
  compresses anything while the free memory is below the reserve (see
  [Memory and the file cache](#memory-and-the-file-cache)). With about 45
  GiB used by other applications and the system and less file cache than
  that share, `--cache 64` settles where the free memory meets the reserve,
  at 42 to 53 GiB on 2026-10-31; 44 to 53 GiB decoded 1 to 11% faster than
  40 GiB exactly (see [Expert formats compared](#expert-formats-compared)).
- Approximate decoding (`--substitute 0.5`) adds 32 to 39% on these experts
  (26 to 28 tokens/s, a mean KL of 0.027 and 0.080 nats from exact decoding
  on the two short texts), and with `--draft 3` the Python prompt decodes at
  34 tokens/s; on prose the drafts cost more than they keep. The GPU's work
  then binds. Its prompts prefill 1.5 to 2.5 times as fast as exact ones
  through a server whose cache holds earlier requests' experts.
- Prefill: with the checkpoint's experts the SSD binds the 2,760- and
  8,912-token passes, which read 266 and 289 GB at 12 to 13 GB/s (20.3 to
  20.4 and 22.9 to 23.8 s) while their GPU works 4.6 and 18.6 to 19.3 s; a
  15,965-token pass is GPU-bound (35.1 to 37.7 s, see [Results](#results)).
  With the GGUF experts the 2,760-token pass takes 11.3 s, waiting for the
  SSD, and the 8,912-token pass 14.5 to 17.6 s, GPU-bound (13.3 to 16.5 s):
  6.2 s of its profile's 13.2 s are in the tile kernels of the GGUF codes,
  whose products and decoding take turns on the GPU's cores (see
  [Prefill kernels](#prefill-kernels)).
- Images: the vision encoder's attention runs on the matrix units, and its
  work grows with the square of the image's patches (up to about 9,200 for an
  image of up to 1,024 tokens): 0.62 to 0.63 s for a letter-size page's 8,424
  patches, about half of it attention (see [Images](#images)).

## Joint prefill and server CED

On 2026-11-03, a 13,772-token reconstructed Harness request was
measured with a fixed 20 GiB expert cache and 4 GiB reserve. Its system/tool
prefix ends at token 12,453. The split control prefills that prefix, saves
it and runs the remaining 1,319 tokens separately. Joint prefill saves the
prefix during a single forward call over the whole prompt. Each measurement
starts a fresh model process. No generation, substitution, learned predictor
or DSpark runs in this series. Native prefetch remains enabled.

| Expert format | Split control, mean (range) | Joint, CED off | Ratio to control mean | Joint, CED 128 | Joint, CED 512 |
| --- | ---: | ---: | ---: | ---: | ---: |
| MXFP4 | 44.93 s (41.82–48.05) | 25.10 s | 1.79x | 16.82 s | 23.69 s |
| IQ2_XXS mixed | 34.36 s (32.38–36.35) | 27.09 s | 1.27x | 16.25 s | 18.56 s |

Both split controls and the joint CED-off run produced identical final
logits within each expert format. MXFP4 requested expert reads fell from
about 530.87 GB to 289.28 GB; IQ2 from 283.14 GB to 157.64 GB. The reduction
comes from one traversal of the layers for the whole prompt. Cache size
stayed fixed. These are application read counts, not physical SSD traffic.
The controls show substantial timing variation under desktop activity, and
each joint setting has one measurement. This is a cold-prefill comparison,
not a decoding or task-completion benchmark.

Production `--ced 128` and `--ced 512` retain 133 and 517 decoder rows,
including the server's five-row rewind margin. Their final logits differ
from full prefill. The numbers above therefore do not establish equal
quality. Relative to joint CED-off prefill, 128 gives 1.49x on MXFP4 and
1.67x on IQ2; 512 gives 1.06x and 1.46x in these individual runs.
The aggregate is `data/benchmarks/joint_prefill_results.json`. It contains
only timings, counters, digests and resource observations. Its metadata note
records a correction to two diagnostic-only flags; raw emitted flag values
remain in the record and timing values are unchanged.

Separate model checks passed on both expert formats at 640 and 16,521
tokens with two-token DSpark and 32 expert slots per stage. They compare
final logits, prefix snapshots, resumed execution and draft windows.
Compact CED matched its full-shape mask at depths 128 and 512. These tests
exposed and now cover candidate-mask row offsets beyond 16,384 positions;
candidate masks are retained per prompt row across attention batches.
The prefill state check also covers all 133 retained local rows, including
the server's five-row rewind margin. These are implementation checks, not
task-quality scores.

## Pending expert reads

Two CPU regressions exercise reads that remain pending after routing removes
their prefetch protection. They use memory-backed fixtures, without a model
or SSD traffic:

| Check | Reproduced behavior | Checked behavior |
| --- | --- | --- |
| Optional admission into a full cache | A new prefetch waited about 105 ms for the chosen victim's simulated 100 ms read; 39/44 cache-policy checks passed. | The admission declines immediately and preserves every resident. Demand can still wait and reuse the same victim. All 44 checks pass. |
| Demand for a demoted prefetch | The requested expert's queued pieces kept background priority; 13/14 admission checks passed. | Demand claims those pieces even after prefetch protection has ended. All 14 checks pass. |

Neither change alters expert selection, weights or arithmetic. Readiness can
still change which experts approximate substitution serves; the exact-path
comparisons below leave substitution disabled. These regressions establish
scheduling behavior, not a throughput improvement.

The live checks on 2026-11-01/02 used DeepSeek-V4.1-Flash-1017, a fixed 36 GiB expert cache,
the 1,229-token scheduler fixture, 256 emitted tokens, temperature 1, top-p 1,
seed 12,345, thinking effort 75 and an 8,192-token context capacity. It used
native prefetching, with the online learner, DSpark, GPU routing and
substitution disabled. The runner required three quiet GPU samples and
checked for other compute processes, compression and swap throughout each
run. All completed runs held their requested cache; none compressed or
swapped pages. This guard is not an exclusive reservation of the GPU.

| Experts | Baseline decode seconds | With both scheduling changes | Generated output |
| --- | --- | --- | --- |
| MXFP4 | 26.68 and 26.69 (9.59 tokens/s) | 26.86 and 26.80 (9.53 and 9.55 tokens/s) | Identical within format |
| UD-IQ2_XXS | 13.12 (19.51 tokens/s) | 13.00 (19.69 tokens/s) | Identical within format |

These results establish no scheduling speedup. MXFP4 was slightly slower;
IQ2's single comparison was 0.9% faster. Main-queue GPU time and renderer
activity varied between runs. The two MXFP4 controls preceded the
candidates; there was no interleaved unfixed control afterward. The IQ2
candidate completed in a quiet window after three earlier attempts did not
launch because GPU activity remained above the 20% admission threshold. The
[structured records](../data/benchmarks/runtime_overhead_results.json) retain
the settings, completed measurements, output comparisons and refused attempts.

A capacity check then changed only the fixed engine's MXFP4 cache budget
from 36 to 40 GiB. It held 2,275 backbone slots (39.99 GiB), versus 2,048,
with the default 10 GiB reserve and no compression or swap. Decode took
25.05 seconds (10.22 tokens/s), about 7% faster than the preceding fixed
36 GiB run. Logical expert reads fell from 309.85 to 286.07 GB, and read wait
fell from 18.46 to 16.78 seconds. Generated text was identical. This is one
capacity comparison, not evidence of a scheduling-fix gain. The larger
cache also remapped 32 experts per emitted token for the GPU, while the
36 GiB runs remapped none; host cache adaptation still averaged only
0.2 ms/token. A larger CPU cache does not imply that every slot stays mapped
for the GPU.

The MXFP4 controls spent about 71 ms per emitted token waiting for expert
reads, 0.4 ms encoding GPU work and 0.2 ms adapting the cache. Main-queue GPU
execution averaged 33.5 ms per token and can overlap host waits; these
numbers must not be summed as independent costs. Host encoding and cache
maintenance are therefore small targets in this workload. The code audit
also found that small DSpark verification batches share dense matrix weight
loads across rows but execute routed experts separately for each row.
The experimental shared-expert build now groups repeated row/expert pairs
while preserving each row's accumulation order. Its checks and measurement
method are described in [Design](design.md#experimental-sharing-across-expert-rows).
[Experiment records](../data/benchmarks/expert_batch_results.json) distinguish
synthetic kernel timings, real-model logit checks and long inference runs.
Kernel timing improvements alone do not establish a model-throughput gain.

The Rust engine already uses a user-interactive inference thread. Its token
callback sends to a channel without waiting for network output. Those code
paths do not support a hypothesis that lower worker priority or synchronous
socket writes explain the CLI's read-bound result. Server throughput was
not measured in this check.

## Experimental expert-row sharing

The retained build is `build/myds4-expert-batch`; the normal executable does
not enable this path. On the 8,171-token repository-review request, with
1,024 emitted tokens, two DSpark drafts, a fixed 20 GiB cache, a 4 GiB
reserve, temperature/top-p 1 and seed 12345, MXFP4 controls took 187.36 and
188.68 seconds. The shared-expert run took 186.25 seconds, about 0.9% less
than the mean control. Main-queue GPU work fell from 35.98 to 31.37 seconds,
12.8%, while expert-read waiting stayed near 154 seconds. Generated output
matched exactly. This is one candidate run under recorded desktop load;
it establishes a GPU-work reduction but not a general throughput guarantee.

The kernel fixtures and all-row model checks are recorded in
[expert_batch_results.json](../data/benchmarks/expert_batch_results.json).
They distinguish synthetic kernel timings, real-model logit checks and
emitted-token timing. See [Design](design.md#experimental-sharing-across-expert-rows)
for supported formats, fallback behavior and resource limits.
