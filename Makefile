.DEFAULT_GOAL := build/myds4
CC = clang
CFLAGS = -std=c23 -O2 -Wall -Wextra -Wno-unused-parameter -Isrc/engine -DMYDS4_RUNAHEAD -DMYDS4_RUNAHEAD_NO_FORECASTS $(shell pkg-config --cflags yyjson libjpeg)
LDLIBS = -framework Metal -framework Foundation -framework IOKit -framework ImageIO -framework CoreGraphics -framework Accelerate -lobjc \
         $(shell pkg-config --libs yyjson libjpeg)
SRC = src/cli/main.c src/engine/model.c src/engine/io.c src/engine/gpu.c src/engine/tok.c src/engine/util.c src/engine/vision.c src/experimental/online_predictor.c src/experimental/context_lookup.c src/engine/sample.c
MODEL_HEADERS = src/engine/myds4.h src/experimental/online_predictor_math.h src/experimental/context_lookup.h src/engine/expert_batch.h src/experimental/prefill_prune.h src/experimental/runahead.c src/experimental/expert_waves.c src/experimental/urgent_io_diagnostic.h
KERNELS = src/kernels/kernels.metal src/kernels/gguf_grids.metal src/kernels/gguf.metal src/kernels/vision.metal

build/shared-gate-check: tools/checks/shared_gate_check.c src/experimental/shared_gate.metal tools/reference/activation_format.h src/engine/gpu.c src/engine/util.c src/engine/myds4.h $(KERNELS)
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ tools/checks/shared_gate_check.c src/engine/gpu.c src/engine/util.c $(LDLIBS)

build/dense-padding-check: tools/checks/dense_padding_check.c src/experimental/dense_padding.metal tools/reference/activation_format.h src/engine/gpu.c src/engine/util.c src/engine/myds4.h $(KERNELS)
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ tools/checks/dense_padding_check.c src/engine/gpu.c src/engine/util.c $(LDLIBS)

build/speculative-frontier-check: tools/checks/speculative_frontier_check.c src/experimental/speculative_frontier.h src/engine/sample.c src/engine/util.c src/engine/myds4.h
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ tools/checks/speculative_frontier_check.c src/engine/sample.c src/engine/util.c $(shell pkg-config --libs yyjson)

build/speculative-read-check: tools/checks/speculative_read_check.c src/engine/io.c src/engine/gpu.c src/engine/util.c $(MODEL_HEADERS)
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ tools/checks/speculative_read_check.c src/engine/gpu.c src/engine/util.c $(LDLIBS)

build/myds4-causal-priority: $(SRC) src/experimental/runahead.c src/experimental/causal_verify.c src/experimental/speculative_frontier.h $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_RUNAHEAD -DMYDS4_CAUSAL_VERIFY -DMYDS4_CAUSAL_SIGNALS -DMYDS4_CAUSAL_GROUP -DMYDS4_RUNAHEAD_NO_FORECASTS -DMYDS4_CAUSAL_EARLY_STOP -DMYDS4_CAUSAL_LEAD_FIRST -DMYDS4_CAUSAL_PRIORITY_READS -o $@ $(SRC) $(LDLIBS)

build/myds4-causal-priority-spin: $(SRC) src/experimental/runahead.c src/experimental/causal_verify.c src/experimental/speculative_frontier.h $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_RUNAHEAD -DMYDS4_CAUSAL_VERIFY -DMYDS4_CAUSAL_SIGNALS -DMYDS4_CAUSAL_GROUP -DMYDS4_RUNAHEAD_NO_FORECASTS -DMYDS4_CAUSAL_EARLY_STOP -DMYDS4_CAUSAL_LEAD_FIRST -DMYDS4_CAUSAL_PRIORITY_READS -DMYDS4_RUNAHEAD_SPIN -o $@ $(SRC) $(LDLIBS)

build/early-priority-spin-check: tools/checks/early_verify_check.c $(SRC) src/experimental/runahead.c src/experimental/causal_verify.c src/experimental/speculative_frontier.h $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_RUNAHEAD -DMYDS4_CAUSAL_VERIFY -DMYDS4_CAUSAL_SIGNALS -DMYDS4_CAUSAL_GROUP -DMYDS4_RUNAHEAD_NO_FORECASTS -DMYDS4_CAUSAL_EARLY_STOP -DMYDS4_CAUSAL_LEAD_FIRST -DMYDS4_CAUSAL_PRIORITY_READS -DMYDS4_RUNAHEAD_SPIN -o $@ tools/checks/early_verify_check.c $(filter-out src/cli/main.c,$(SRC)) $(LDLIBS)

build/early-priority-check: tools/checks/early_verify_check.c $(SRC) src/experimental/runahead.c src/experimental/causal_verify.c src/experimental/speculative_frontier.h $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_RUNAHEAD -DMYDS4_CAUSAL_VERIFY -DMYDS4_CAUSAL_SIGNALS -DMYDS4_CAUSAL_GROUP -DMYDS4_RUNAHEAD_NO_FORECASTS -DMYDS4_CAUSAL_EARLY_STOP -DMYDS4_CAUSAL_LEAD_FIRST -DMYDS4_CAUSAL_PRIORITY_READS -o $@ tools/checks/early_verify_check.c $(filter-out src/cli/main.c,$(SRC)) $(LDLIBS)

build/activation-format-check: tools/checks/activation_format_check.c tools/reference/activation_format.h
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ tools/checks/activation_format_check.c

activation-format-check: build/activation-format-check
	./build/activation-format-check

build/mxfp4-activation-check: tools/checks/mxfp4_activation_check.c tools/reference/activation_format.h src/engine/gpu.c src/engine/util.c src/engine/myds4.h $(KERNELS)
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ tools/checks/mxfp4_activation_check.c src/engine/gpu.c src/engine/util.c $(LDLIBS)

build/mxfp4-load-model-check: tools/checks/mxfp4_load_model_check.c $(SRC) $(MODEL_HEADERS) $(KERNELS) src/experimental/mxfp4_loads.metal data/engram.bin data/unicode.bin tools/checks/model_state_fixture.h
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ tools/checks/mxfp4_load_model_check.c $(filter-out src/cli/main.c src/engine/model.c,$(SRC)) $(LDLIBS)

build/packed-activation-check: tools/checks/packed_activation_check.c tools/checks/packed_activation.metal tools/reference/activation_format.h src/engine/gpu.c src/engine/util.c src/engine/myds4.h $(KERNELS)
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ tools/checks/packed_activation_check.c src/engine/gpu.c src/engine/util.c $(LDLIBS)

build/context-lookup-check: tools/checks/context_lookup_check.c src/experimental/context_lookup.c src/experimental/context_lookup.h
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ tools/checks/context_lookup_check.c src/experimental/context_lookup.c

context-lookup-check: build/context-lookup-check
	./build/context-lookup-check

build/context-lookup-replay: tools/experimental/context_lookup_replay.c src/experimental/context_lookup.c src/experimental/context_lookup.h
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ tools/experimental/context_lookup_replay.c src/experimental/context_lookup.c

build/expert-stream-probe: tools/experimental/expert_stream_probe.c src/engine/io.c src/engine/gpu.c src/engine/util.c $(MODEL_HEADERS) $(KERNELS)
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ tools/experimental/expert_stream_probe.c src/engine/gpu.c src/engine/util.c $(LDLIBS)

build/expert-phase-check: tools/checks/expert_phase_check.c src/engine/io.c src/engine/gpu.c src/engine/util.c $(MODEL_HEADERS)
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ tools/checks/expert_phase_check.c src/engine/gpu.c src/engine/util.c $(LDLIBS)

build/session-probe: tools/experimental/session_probe.c $(SRC) $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_BATCH_PREFETCH_STUDY -o $@ tools/experimental/session_probe.c $(filter-out src/cli/main.c src/engine/model.c,$(SRC)) $(LDLIBS)

build/session-probe-runahead: tools/experimental/session_probe.c src/experimental/runahead.c $(SRC) $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_BATCH_PREFETCH_STUDY -DMYDS4_RUNAHEAD -DMYDS4_RUNAHEAD_NO_FORECASTS -o $@ tools/experimental/session_probe.c $(filter-out src/cli/main.c src/engine/model.c,$(SRC)) $(LDLIBS)

build/runahead-route-model-check: tools/checks/runahead_model_check.c src/experimental/runahead.c $(SRC) $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_RUNAHEAD -DMYDS4_RUNAHEAD_NO_FORECASTS -o $@ tools/checks/runahead_model_check.c $(filter-out src/cli/main.c src/engine/model.c,$(SRC)) $(LDLIBS)

build/myds4-expert-phase: $(SRC) $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_EXPERT_PHASED -o $@ $(SRC) $(LDLIBS)

build/myds4-expert-phase-preencode: $(SRC) $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_EXPERT_PHASED -DMYDS4_EXPERT_PHASED_PREENCODE -o $@ $(SRC) $(LDLIBS)

build/expert-phase-model-check: tools/checks/expert_phase_model_check.c $(SRC) $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin tools/checks/model_state_fixture.h
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_EXPERT_PHASED -o $@ tools/checks/expert_phase_model_check.c $(filter-out src/cli/main.c src/engine/model.c,$(SRC)) $(LDLIBS)

build/expert-phase-preencode-model-check: tools/checks/expert_phase_model_check.c $(SRC) $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin tools/checks/model_state_fixture.h
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_EXPERT_PHASED -DMYDS4_EXPERT_PHASED_PREENCODE -o $@ tools/checks/expert_phase_model_check.c $(filter-out src/cli/main.c src/engine/model.c,$(SRC)) $(LDLIBS)

build/metal-io-probe: tools/experimental/metal_io_probe.c src/engine/gpu.c src/engine/util.c src/engine/myds4.h
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ tools/experimental/metal_io_probe.c src/engine/util.c $(LDLIBS)

build/draft-distribution-probe: tools/experimental/draft_distribution_probe.c $(SRC) $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ tools/experimental/draft_distribution_probe.c $(filter-out src/cli/main.c src/engine/model.c,$(SRC)) $(LDLIBS)

build/speculative-sample-check: tools/checks/speculative_sample_check.c src/experimental/speculative_sample.c src/experimental/speculative_sample.h src/engine/sample.c src/engine/util.c src/engine/myds4.h
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ tools/checks/speculative_sample_check.c src/experimental/speculative_sample.c src/engine/sample.c src/engine/util.c $(shell pkg-config --libs yyjson)

build/myds4-rejection-draft: $(SRC) src/experimental/speculative_sample.c src/experimental/speculative_sample.h $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_REJECTION_DRAFT -o $@ $(SRC) src/experimental/speculative_sample.c $(LDLIBS)

build/myds4-rejection-prefetch2: $(SRC) src/experimental/speculative_sample.c src/experimental/speculative_sample.h $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_REJECTION_DRAFT -DMYDS4_VERIFY_PREFETCH_ROWS=2 -o $@ $(SRC) src/experimental/speculative_sample.c $(LDLIBS)

# Verification keeps its arithmetic and cache retention but admits no forecast reads.
build/myds4-verify-no-prefetch: $(SRC) $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_VERIFY_PREFETCH_ROWS=0 -o $@ $(SRC) $(LDLIBS)

build/myds4-verify-prefetch1: $(SRC) $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_VERIFY_PREFETCH_ROWS=1 -o $@ $(SRC) $(LDLIBS)

build/myds4-verify-prefetch2: $(SRC) $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_VERIFY_PREFETCH_ROWS=2 -o $@ $(SRC) $(LDLIBS)

build/myds4-causal-draft: $(SRC) src/experimental/runahead.c src/experimental/causal_verify.c $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_RUNAHEAD -DMYDS4_CAUSAL_VERIFY -o $@ $(SRC) $(LDLIBS)

build/myds4-causal-rejection-draft: $(SRC) src/experimental/runahead.c src/experimental/causal_verify.c src/experimental/speculative_sample.c src/experimental/speculative_sample.h $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_RUNAHEAD -DMYDS4_CAUSAL_VERIFY -DMYDS4_REJECTION_DRAFT -o $@ $(SRC) src/experimental/speculative_sample.c $(LDLIBS)

build/causal-verify-check: tools/checks/verification_model_check.c $(SRC) src/experimental/runahead.c src/experimental/causal_verify.c $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_RUNAHEAD -DMYDS4_CAUSAL_VERIFY -o $@ tools/checks/verification_model_check.c $(filter-out src/cli/main.c,$(SRC)) $(LDLIBS)

build/myds4-causal-signal-draft: $(SRC) src/experimental/runahead.c src/experimental/causal_verify.c $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_RUNAHEAD -DMYDS4_CAUSAL_VERIFY -DMYDS4_CAUSAL_SIGNALS -o $@ $(SRC) $(LDLIBS)

build/myds4-causal-signal-rejection-draft: $(SRC) src/experimental/runahead.c src/experimental/causal_verify.c src/experimental/speculative_sample.c src/experimental/speculative_sample.h $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_RUNAHEAD -DMYDS4_CAUSAL_VERIFY -DMYDS4_CAUSAL_SIGNALS -DMYDS4_REJECTION_DRAFT -o $@ $(SRC) src/experimental/speculative_sample.c $(LDLIBS)

build/causal-signal-check: tools/checks/verification_model_check.c $(SRC) src/experimental/runahead.c src/experimental/causal_verify.c $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_RUNAHEAD -DMYDS4_CAUSAL_VERIFY -DMYDS4_CAUSAL_SIGNALS -o $@ tools/checks/verification_model_check.c $(filter-out src/cli/main.c,$(SRC)) $(LDLIBS)

build/myds4-causal-group-draft: $(SRC) src/experimental/runahead.c src/experimental/causal_verify.c $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_RUNAHEAD -DMYDS4_CAUSAL_VERIFY -DMYDS4_CAUSAL_SIGNALS -DMYDS4_CAUSAL_GROUP -o $@ $(SRC) $(LDLIBS)

build/myds4-causal-group-rejection-draft: $(SRC) src/experimental/runahead.c src/experimental/causal_verify.c src/experimental/speculative_sample.c src/experimental/speculative_sample.h $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_RUNAHEAD -DMYDS4_CAUSAL_VERIFY -DMYDS4_CAUSAL_SIGNALS -DMYDS4_CAUSAL_GROUP -DMYDS4_REJECTION_DRAFT -o $@ $(SRC) src/experimental/speculative_sample.c $(LDLIBS)

build/causal-group-check: tools/checks/verification_model_check.c $(SRC) src/experimental/runahead.c src/experimental/causal_verify.c $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_RUNAHEAD -DMYDS4_CAUSAL_VERIFY -DMYDS4_CAUSAL_SIGNALS -DMYDS4_CAUSAL_GROUP -DMYDS4_EXPERT_BATCH -o $@ tools/checks/verification_model_check.c $(filter-out src/cli/main.c,$(SRC)) $(LDLIBS)

build/myds4-causal-route-only: $(SRC) src/experimental/runahead.c src/experimental/causal_verify.c $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_RUNAHEAD -DMYDS4_CAUSAL_VERIFY -DMYDS4_CAUSAL_SIGNALS -DMYDS4_CAUSAL_GROUP -DMYDS4_RUNAHEAD_NO_FORECASTS -o $@ $(SRC) $(LDLIBS)

build/myds4-causal-early: $(SRC) src/experimental/runahead.c src/experimental/causal_verify.c src/experimental/speculative_frontier.h $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_RUNAHEAD -DMYDS4_CAUSAL_VERIFY -DMYDS4_CAUSAL_SIGNALS -DMYDS4_CAUSAL_GROUP -DMYDS4_RUNAHEAD_NO_FORECASTS -DMYDS4_CAUSAL_EARLY_STOP -DMYDS4_CAUSAL_LEAD_FIRST -o $@ $(SRC) $(LDLIBS)

build/early-verify-check: tools/checks/early_verify_check.c $(SRC) src/experimental/runahead.c src/experimental/causal_verify.c src/experimental/speculative_frontier.h $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_RUNAHEAD -DMYDS4_CAUSAL_VERIFY -DMYDS4_CAUSAL_SIGNALS -DMYDS4_CAUSAL_GROUP -DMYDS4_RUNAHEAD_NO_FORECASTS -DMYDS4_CAUSAL_EARLY_STOP -DMYDS4_CAUSAL_LEAD_FIRST -o $@ tools/checks/early_verify_check.c $(filter-out src/cli/main.c,$(SRC)) $(LDLIBS)

build/myds4-draft-expert-batch: $(SRC) $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_DRAFT_EXPERT_BATCH -o $@ $(SRC) $(LDLIBS)

build/myds4-draft-cache-control: $(SRC) $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_DRAFT_CACHE_METRICS -o $@ $(SRC) $(LDLIBS)

build/myds4-draft-shared-pool: $(SRC) $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_DRAFT_CACHE_METRICS -DMYDS4_DRAFT_SHARED_POOL -o $@ $(SRC) $(LDLIBS)

build/draft-shared-pool-check: tools/checks/draft_shared_pool_check.c tools/checks/draft_cache_check.c src/engine/io.c src/engine/gpu.c src/engine/util.c $(MODEL_HEADERS)
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_DRAFT_SHARED_POOL -o $@ tools/checks/draft_shared_pool_check.c src/engine/gpu.c src/engine/util.c $(LDLIBS)

build/draft-expert-batch-check: tools/checks/draft_expert_batch_check.c $(SRC) $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_DRAFT_EXPERT_BATCH -o $@ tools/checks/draft_expert_batch_check.c $(filter-out src/cli/main.c src/engine/model.c,$(SRC)) $(LDLIBS)

build/causal-route-only-check: tools/checks/verification_model_check.c $(SRC) src/experimental/runahead.c src/experimental/causal_verify.c $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_RUNAHEAD -DMYDS4_CAUSAL_VERIFY -DMYDS4_CAUSAL_SIGNALS -DMYDS4_CAUSAL_GROUP -DMYDS4_RUNAHEAD_NO_FORECASTS -DMYDS4_EXPERT_BATCH -o $@ tools/checks/verification_model_check.c $(filter-out src/cli/main.c,$(SRC)) $(LDLIBS)

build/prefill-prune-check: tools/checks/prefill_prune_check.c src/experimental/prefill_prune.h
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ tools/checks/prefill_prune_check.c

build/prefill-prune-model-check: tools/checks/prefill_prune_model_check.c $(SRC) $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ tools/checks/prefill_prune_model_check.c $(filter-out src/cli/main.c src/engine/model.c,$(SRC)) $(LDLIBS)

build/draft-queue-check: tools/checks/draft_queue_check.c src/experimental/draft_queue.h
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ tools/checks/draft_queue_check.c

# Host-backed workspace checks; no Metal device or model weights are opened.
build/workspace-check: tools/checks/workspace_check.c $(SRC) $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ tools/checks/workspace_check.c $(filter-out src/cli/main.c src/engine/model.c,$(SRC)) $(LDLIBS)

workspace-check: build/workspace-check
	./build/workspace-check

build/state-fork-check: tools/checks/state_fork_check.c $(SRC) $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ tools/checks/state_fork_check.c $(filter-out src/cli/main.c src/engine/model.c,$(SRC)) $(LDLIBS)

state-fork-check: build/state-fork-check
	./build/state-fork-check

build/prefix-fork-check: tools/checks/prefix_fork_check.c $(SRC) $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ tools/checks/prefix_fork_check.c $(filter-out src/cli/main.c src/engine/model.c,$(SRC)) $(LDLIBS)

build/verification-schedule-check: tools/checks/verification_schedule_check.c $(SRC) $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ tools/checks/verification_schedule_check.c $(filter-out src/cli/main.c src/engine/model.c,$(SRC)) $(LDLIBS)

build/verification-prefetch-check: tools/checks/verification_prefetch_check.c $(SRC) $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ tools/checks/verification_prefetch_check.c $(filter-out src/cli/main.c src/engine/model.c,$(SRC)) $(LDLIBS)

build/prefill-priority-check: tools/checks/prefill_priority_check.c src/engine/io.c src/engine/gpu.c src/engine/util.c $(MODEL_HEADERS)
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ tools/checks/prefill_priority_check.c src/engine/gpu.c src/engine/util.c $(LDLIBS)

verification-schedule-check: build/verification-schedule-check
	./build/verification-schedule-check

build/verification-model-check: tools/checks/verification_model_check.c $(SRC) $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_EXPERT_BATCH -o $@ tools/checks/verification_model_check.c $(filter-out src/cli/main.c,$(SRC)) $(LDLIBS)

# CED numerical and prefill/DSpark timing checks.
build/ced-check: tools/checks/ced_check.c $(SRC) $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ tools/checks/ced_check.c $(filter-out src/cli/main.c src/engine/model.c,$(SRC)) $(LDLIBS)

build/dspark-state-check: tools/checks/dspark_state_check.c $(SRC) $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ tools/checks/dspark_state_check.c $(filter-out src/cli/main.c src/engine/model.c,$(SRC)) $(LDLIBS)

build/prefill-state-check: tools/checks/prefill_state_check.c $(SRC) $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ tools/checks/prefill_state_check.c $(filter-out src/cli/main.c src/engine/model.c,$(SRC)) $(LDLIBS)

build/attention-candidate-check: tools/checks/attention_candidate_check.c $(SRC) $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ tools/checks/attention_candidate_check.c $(filter-out src/cli/main.c src/engine/model.c,$(SRC)) $(LDLIBS)

build/expert-batch-check: tools/checks/expert_batch_check.c tools/reference/activation_format.h src/engine/gpu.c src/engine/util.c src/engine/myds4.h src/engine/expert_batch.h src/kernels/kernels.metal src/kernels/gguf_grids.metal src/kernels/gguf.metal src/experimental/mxfp4_loads.metal src/experimental/expert_rows_specialized.metal
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ tools/checks/expert_batch_check.c src/engine/gpu.c src/engine/util.c $(LDLIBS)

# Experimental exact sharing of resident expert weights across verification rows.
build/myds4-fp4-early: $(SRC) $(MODEL_HEADERS) $(KERNELS) src/experimental/mxfp4_loads.metal data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_FP4_LOADS=1 -o $@ $(SRC) $(LDLIBS)

build/myds4-fp4-pipeline: $(SRC) $(MODEL_HEADERS) $(KERNELS) src/experimental/mxfp4_loads.metal data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_FP4_LOADS=2 -o $@ $(SRC) $(LDLIBS)

build/myds4-fp4-row: $(SRC) $(MODEL_HEADERS) $(KERNELS) src/experimental/mxfp4_loads.metal data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_FP4_LOADS=3 -o $@ $(SRC) $(LDLIBS)

build/myds4-fp4-factor: $(SRC) $(MODEL_HEADERS) $(KERNELS) src/experimental/mxfp4_loads.metal data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_FP4_LOADS=4 -o $@ $(SRC) $(LDLIBS)

build/myds4-ahead1: $(SRC) $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_PREFETCH_AHEAD=1 -o $@ $(SRC) $(LDLIBS)

build/myds4-spin-router: $(SRC) $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_SPIN_ROUTER -o $@ $(SRC) $(LDLIBS)

build/myds4-ahead2: $(SRC) $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_PREFETCH_AHEAD=2 -o $@ $(SRC) $(LDLIBS)

build/myds4-ahead4: $(SRC) $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_PREFETCH_AHEAD=4 -o $@ $(SRC) $(LDLIBS)

build/myds4-h2-first: $(SRC) $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_H2_FIRST -o $@ $(SRC) $(LDLIBS)

build/myds4-stale: $(SRC) $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_STALE_PRIORITY -o $@ $(SRC) $(LDLIBS)

build/myds4-gpu48: $(SRC) $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_GPU_LIMIT_GIB=48 -o $@ $(SRC) $(LDLIBS)

build/myds4-index-prune: $(SRC) $(MODEL_HEADERS) $(KERNELS) src/experimental/index_prune.metal data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_INDEX_PRUNE -o $@ $(SRC) $(LDLIBS)

build/myds4-index-both: $(SRC) $(MODEL_HEADERS) $(KERNELS) src/experimental/index_prune.metal data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_INDEX_PRUNE=2 -o $@ $(SRC) $(LDLIBS)

build/myds4-prune-index-both: $(SRC) $(MODEL_HEADERS) $(KERNELS) src/experimental/index_prune.metal data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_INDEX_PRUNE=2 -DMYDS4_PREFILL_PRUNE -o $@ $(SRC) $(LDLIBS)

build/index-both-model-check: tools/checks/prefill_prune_model_check.c $(SRC) $(MODEL_HEADERS) $(KERNELS) src/experimental/index_prune.metal data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_INDEX_PRUNE=2 -o $@ tools/checks/prefill_prune_model_check.c $(filter-out src/cli/main.c src/engine/model.c,$(SRC)) $(LDLIBS)

build/index-mask-check: tools/checks/index_mask_check.c src/engine/gpu.c src/engine/util.c $(MODEL_HEADERS) $(KERNELS) src/experimental/index_prune.metal
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ tools/checks/index_mask_check.c src/engine/gpu.c src/engine/util.c $(LDLIBS)

build/fp8-tensor-check: tools/checks/fp8_tensor_check.c src/engine/gpu.c src/engine/util.c tools/reference/activation_format.h $(MODEL_HEADERS) $(KERNELS) src/experimental/fp8_tensor.metal
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ tools/checks/fp8_tensor_check.c src/engine/gpu.c src/engine/util.c $(LDLIBS)

build/myds4-prune-index: $(SRC) $(MODEL_HEADERS) $(KERNELS) src/experimental/index_prune.metal data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_INDEX_PRUNE -DMYDS4_PREFILL_PRUNE -o $@ $(SRC) $(LDLIBS)

build/index-prune-model-check: tools/checks/prefill_prune_model_check.c $(SRC) $(MODEL_HEADERS) $(KERNELS) src/experimental/index_prune.metal data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_INDEX_PRUNE -o $@ tools/checks/prefill_prune_model_check.c $(filter-out src/cli/main.c src/engine/model.c,$(SRC)) $(LDLIBS)

build/myds4-prune-gpu48: $(SRC) $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_GPU_LIMIT_GIB=48 -DMYDS4_PREFILL_PRUNE -o $@ $(SRC) $(LDLIBS)

build/cache-stale-check: tools/checks/cache_stale_check.c src/engine/io.c src/engine/gpu.c src/engine/util.c $(MODEL_HEADERS)
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ tools/checks/cache_stale_check.c src/engine/gpu.c src/engine/util.c $(LDLIBS)

build/myds4-expert-batch: $(SRC) $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_EXPERT_BATCH -o $@ $(SRC) $(LDLIBS)

# Experimental draft sampling with copied target uniforms; ordinary sampling is unchanged.
build/myds4-coupled-draft: $(SRC) $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_COUPLED_DRAFT -o $@ $(SRC) $(LDLIBS)

build/myds4-consensus-lookup: $(SRC) $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_CONSENSUS_LOOKUP -o $@ $(SRC) $(LDLIBS)

build/myds4-protocol-draft: $(SRC) src/experimental/protocol_draft.c src/experimental/protocol_draft.h $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_PROTOCOL_DRAFT -o $@ $(SRC) src/experimental/protocol_draft.c $(LDLIBS)

build/protocol-draft-check: tools/checks/protocol_draft_check.c src/experimental/protocol_draft.c src/experimental/protocol_draft.h src/engine/tok.c src/engine/util.c $(MODEL_HEADERS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ tools/checks/protocol_draft_check.c src/experimental/protocol_draft.c src/engine/tok.c src/engine/util.c $(shell pkg-config --libs yyjson)

build/myds4-staged-draft: $(SRC) $(MODEL_HEADERS) src/experimental/draft_queue.h $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_STAGED_DRAFT=2 -o $@ $(SRC) $(LDLIBS)

build/tokenizer-added-check: tools/checks/tokenizer_added_check.c src/engine/tok.c src/engine/util.c $(MODEL_HEADERS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer -o $@ tools/checks/tokenizer_added_check.c src/engine/tok.c src/engine/util.c $(shell pkg-config --libs yyjson)

build/myds4-staged-coupled-draft: $(SRC) $(MODEL_HEADERS) src/experimental/draft_queue.h $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_STAGED_DRAFT=2 -DMYDS4_COUPLED_DRAFT -o $@ $(SRC) $(LDLIBS)

build/myds4-prefill-prune: $(SRC) $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_PREFILL_PRUNE -o $@ $(SRC) $(LDLIBS)

build/myds4-layer-prune: $(SRC) $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_PREFILL_PRUNE -DMYDS4_LAYER_PRUNE -o $@ $(SRC) $(LDLIBS)

build/myds4-chunk32: $(SRC) $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_PREFILL_PRUNE -DMYDS4_PREFILL_CHUNK=32768 -o $@ $(SRC) $(LDLIBS)

build/prefill-layer-prune-model-check: tools/checks/prefill_prune_model_check.c $(SRC) $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_LAYER_PRUNE -o $@ tools/checks/prefill_prune_model_check.c $(filter-out src/cli/main.c src/engine/model.c,$(SRC)) $(LDLIBS)

build/myds4-prefix-decode-score: $(SRC) $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_DECODE_ONLY_SCORE -o $@ $(SRC) $(LDLIBS)

build/myds4-prefill-frequency: $(SRC) $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_PREFILL_CREDIT_ROWS=0 -o $@ $(SRC) $(LDLIBS)

build/myds4-prefill-tail-frequency: $(SRC) $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_PREFILL_CREDIT_ROWS=256 -o $@ $(SRC) $(LDLIBS)

build/myds4-decode-score: $(SRC) $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_DECODE_ONLY_SCORE -o $@ $(SRC) $(LDLIBS)

build/prefill-credit-check: tools/checks/prefill_credit_check.c src/engine/io.c src/engine/gpu.c src/engine/util.c $(MODEL_HEADERS)
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ tools/checks/prefill_credit_check.c src/engine/gpu.c src/engine/util.c $(LDLIBS)

build/myds4-prune-decode-score: $(SRC) $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_PREFILL_PRUNE -DMYDS4_DECODE_ONLY_SCORE -o $@ $(SRC) $(LDLIBS)

build/myds4-prefix-prune-consensus: $(SRC) $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_PREFILL_PRUNE -DMYDS4_CONSENSUS_LOOKUP -o $@ $(SRC) $(LDLIBS)

build/myds4-consensus-prefetch1: $(SRC) $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_PREFILL_PRUNE -DMYDS4_CONSENSUS_LOOKUP -DMYDS4_VERIFY_PREFETCH_ROWS=1 -o $@ $(SRC) $(LDLIBS)

build/myds4-consensus-row-credit: $(SRC) $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_PREFILL_PRUNE -DMYDS4_CONSENSUS_LOOKUP -DMYDS4_VERIFY_ROW_CREDIT -o $@ $(SRC) $(LDLIBS)

build/expert-batch-model-check: tools/checks/expert_batch_model_check.c $(SRC) $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_EXPERT_BATCH -o $@ tools/checks/expert_batch_model_check.c $(filter-out src/cli/main.c,$(SRC)) $(LDLIBS)

build/expert-batch-model-control: tools/checks/expert_batch_model_check.c $(SRC) $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ tools/checks/expert_batch_model_check.c $(filter-out src/cli/main.c,$(SRC)) $(LDLIBS)

# Retained research build: explicit feature ablations and real prefetch trials.
build/myds4-predict-lab: $(SRC) src/experimental/predictor_lab.c src/experimental/predictor_lab.h $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_PREDICT_LAB -o $@ $(SRC) src/experimental/predictor_lab.c $(LDLIBS)

build/predictor-lab-check: tools/experimental/predictor_lab_check.c src/experimental/predictor_lab.c src/experimental/predictor_lab.h $(MODEL_HEADERS) src/engine/util.c
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ tools/experimental/predictor_lab_check.c src/engine/util.c -framework Accelerate

build/cache-action-check: tools/experimental/cache_action_check.c src/engine/io.c src/experimental/predictor_lab.h src/engine/gpu.c src/engine/util.c $(MODEL_HEADERS)
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ tools/experimental/cache_action_check.c src/engine/gpu.c src/engine/util.c $(LDLIBS)

build/myds4: $(SRC) $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ $(SRC) $(LDLIBS)

build/myds4-prefetch-demote: $(SRC) $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_PREFETCH_DEMOTE_CANCEL -o $@ $(SRC) $(LDLIBS)

build/myds4-expert-waves: $(SRC) $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_EXPERT_WAVES -o $@ $(SRC) $(LDLIBS)

build/myds4-io-diagnostic: $(SRC) $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_IO_DIAGNOSTICS -o $@ $(SRC) $(LDLIBS)

build/myds4-io-window%-diagnostic: $(SRC) $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_IO_DIAGNOSTICS -DMYDS4_URGENT_WINDOW=$* -o $@ $(SRC) $(LDLIBS)

build/myds4-early-promotion-diagnostic: $(SRC) $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_IO_DIAGNOSTICS -DMYDS4_EARLY_DEMAND_PROMOTION -o $@ $(SRC) $(LDLIBS)

build/myds4-early-promotion: $(SRC) $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_EARLY_DEMAND_PROMOTION -o $@ $(SRC) $(LDLIBS)

build/myds4-late-preencode: $(SRC) $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_LATE_PREENCODE -o $@ $(SRC) $(LDLIBS)

build/late-preencode-model-check: tools/checks/late_preencode_model_check.c $(SRC) $(MODEL_HEADERS) $(KERNELS) src/experimental/expert_compact.h src/experimental/expert_compact.metal data/engram.bin data/unicode.bin tools/checks/model_state_fixture.h
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_EXPERT_COMPACT -DMYDS4_LATE_PREENCODE -DMYDS4_LATE_PREENCODE_CHECK -o $@ tools/checks/late_preencode_model_check.c $(filter-out src/cli/main.c src/engine/model.c,$(SRC)) $(LDLIBS)

build/myds4-io-window%: $(SRC) $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_URGENT_WINDOW=$* -o $@ $(SRC) $(LDLIBS)

build/urgent-window-check: tools/checks/urgent_window_check.c src/engine/io.c src/engine/gpu.c src/engine/util.c $(MODEL_HEADERS)
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ tools/checks/urgent_window_check.c src/engine/gpu.c src/engine/util.c $(LDLIBS)

build/prefetch-promote-check: tools/checks/prefetch_promote_check.c src/engine/io.c src/engine/gpu.c src/engine/util.c $(MODEL_HEADERS)
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_EARLY_DEMAND_PROMOTION -o $@ tools/checks/prefetch_promote_check.c src/engine/gpu.c src/engine/util.c $(LDLIBS)

build/expert-compact-check: tools/checks/expert_compact_check.c src/experimental/expert_compact.h src/experimental/expert_compact.metal src/engine/expert_batch.h tools/reference/activation_format.h src/engine/gpu.c src/engine/util.c $(MODEL_HEADERS) $(KERNELS)
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ tools/checks/expert_compact_check.c src/engine/gpu.c src/engine/util.c $(LDLIBS)

build/myds4-expert-compact: $(SRC) $(MODEL_HEADERS) $(KERNELS) src/experimental/expert_compact.h src/experimental/expert_compact.metal data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_EXPERT_COMPACT -o $@ $(SRC) $(LDLIBS)

build/expert-compact-model-check: tools/checks/expert_compact_model_check.c $(SRC) $(MODEL_HEADERS) $(KERNELS) src/experimental/expert_compact.h src/experimental/expert_compact.metal data/engram.bin data/unicode.bin tools/checks/model_state_fixture.h
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_EXPERT_COMPACT -o $@ tools/checks/expert_compact_model_check.c $(filter-out src/cli/main.c src/engine/model.c,$(SRC)) $(LDLIBS)

build/expert-limit256-check: tools/checks/expert_compact_check.c src/experimental/expert_compact.h src/experimental/expert_compact.metal src/engine/expert_batch.h tools/reference/activation_format.h src/engine/gpu.c src/engine/util.c $(MODEL_HEADERS) $(KERNELS)
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_PIPELINE_LIMIT=256 -DMYDS4_PIPELINE_DOWN_LIMIT=192 -o $@ tools/checks/expert_compact_check.c src/engine/gpu.c src/engine/util.c $(LDLIBS)

build/expert-limit512-check: tools/checks/expert_compact_check.c src/experimental/expert_compact.h src/experimental/expert_compact.metal src/engine/expert_batch.h tools/reference/activation_format.h src/engine/gpu.c src/engine/util.c $(MODEL_HEADERS) $(KERNELS)
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_PIPELINE_LIMIT=512 -DMYDS4_PIPELINE_DOWN_LIMIT=256 -o $@ tools/checks/expert_compact_check.c src/engine/gpu.c src/engine/util.c $(LDLIBS)

build/myds4-limit256: $(SRC) $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_PIPELINE_LIMIT=256 -DMYDS4_PIPELINE_DOWN_LIMIT=192 -o $@ $(SRC) $(LDLIBS)

build/myds4-limit512: $(SRC) $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_PIPELINE_LIMIT=512 -DMYDS4_PIPELINE_DOWN_LIMIT=256 -o $@ $(SRC) $(LDLIBS)

COMBINED_FLAGS = -DMYDS4_EXPERT_COMPACT -DMYDS4_EARLY_DEMAND_PROMOTION -DMYDS4_LATE_PREENCODE
COMBINED_ALL_FLAGS = $(COMBINED_FLAGS) -DMYDS4_URGENT_WINDOW=16 -DMYDS4_PIPELINE_LIMIT=256 -DMYDS4_PIPELINE_DOWN_LIMIT=192
COMBINED_WAVE_FLAGS = -DMYDS4_EXPERT_COMPACT -DMYDS4_EARLY_DEMAND_PROMOTION -DMYDS4_EXPERT_WAVES
COMBINED_DEPS = src/experimental/expert_compact.h src/experimental/expert_compact.metal

build/myds4-combined: $(SRC) $(MODEL_HEADERS) $(KERNELS) $(COMBINED_DEPS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) $(COMBINED_FLAGS) -o $@ $(SRC) $(LDLIBS)

build/myds4-combined-all: $(SRC) $(MODEL_HEADERS) $(KERNELS) $(COMBINED_DEPS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) $(COMBINED_ALL_FLAGS) -o $@ $(SRC) $(LDLIBS)

build/myds4-combined-waves: $(SRC) $(MODEL_HEADERS) $(KERNELS) $(COMBINED_DEPS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) $(COMBINED_WAVE_FLAGS) -o $@ $(SRC) $(LDLIBS)

build/combined-all-model-check: tools/checks/expert_compact_model_check.c tools/checks/model_state_fixture.h $(SRC) $(MODEL_HEADERS) $(KERNELS) $(COMBINED_DEPS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) $(COMBINED_ALL_FLAGS) -o $@ tools/checks/expert_compact_model_check.c $(filter-out src/cli/main.c src/engine/model.c,$(SRC)) $(LDLIBS)

build/combined-waves-model-check: tools/checks/expert_compact_model_check.c tools/checks/model_state_fixture.h $(SRC) $(MODEL_HEADERS) $(KERNELS) $(COMBINED_DEPS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) $(COMBINED_WAVE_FLAGS) -o $@ tools/checks/expert_compact_model_check.c $(filter-out src/cli/main.c src/engine/model.c,$(SRC)) $(LDLIBS)

build/prefetch-queue-diag-check: tools/checks/prefetch_queue_check.c src/engine/io.c src/engine/gpu.c src/engine/util.c $(MODEL_HEADERS)
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_IO_DIAGNOSTICS -o $@ tools/checks/prefetch_queue_check.c src/engine/gpu.c src/engine/util.c $(LDLIBS)

build/myds4-expert-waves3: $(SRC) $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_EXPERT_WAVES -DMYDS4_EXPERT_WAVE_CAP=3 -o $@ $(SRC) $(LDLIBS)

build/myds4-prefetch-pool4: $(SRC) $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_PREFETCH_POOL=4 -o $@ $(SRC) $(LDLIBS)

build/myds4-prefetch-pool8: $(SRC) $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_PREFETCH_POOL=8 -o $@ $(SRC) $(LDLIBS)

build/prefetch-pool-check: tools/checks/prefetch_pool_check.c src/engine/io.c src/engine/gpu.c src/engine/util.c $(MODEL_HEADERS)
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ tools/checks/prefetch_pool_check.c src/engine/gpu.c src/engine/util.c $(LDLIBS)

build/expert-ready-check: tools/checks/expert_ready_check.c src/engine/io.c src/engine/gpu.c src/engine/util.c $(MODEL_HEADERS)
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ tools/checks/expert_ready_check.c src/engine/gpu.c src/engine/util.c $(LDLIBS)

expert-ready-check: build/expert-ready-check
	./build/expert-ready-check

prefetch-pool-check: build/prefetch-pool-check
	./build/prefetch-pool-check

build/expert-waves-model-check: tools/checks/expert_waves_model_check.c $(SRC) $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_EXPERT_WAVES -o $@ tools/checks/expert_waves_model_check.c $(filter-out src/cli/main.c src/engine/model.c,$(SRC)) $(LDLIBS)

build/expert-waves3-model-check: tools/checks/expert_waves_model_check.c $(SRC) $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_EXPERT_WAVES -DMYDS4_EXPERT_WAVE_CAP=3 -o $@ tools/checks/expert_waves_model_check.c $(filter-out src/cli/main.c src/engine/model.c,$(SRC)) $(LDLIBS)

# The engine behind src/engine/engine.h, for the Rust server (server/): the sources
# of build/myds4 without those in src/cli/, and with engine.c.
LIB_SRC = src/engine/model.c src/engine/io.c src/engine/gpu.c src/engine/tok.c src/engine/util.c src/engine/vision.c src/experimental/online_predictor.c src/experimental/context_lookup.c src/engine/sample.c src/engine/engine.c
LIB_OBJ = $(LIB_SRC:src/%.c=build/lib/%.o)

build/lib/%.o: src/%.c $(MODEL_HEADERS) src/engine/engine.h $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -c -o $@ $<

build/libmyds4.a: $(LIB_OBJ)
	rm -f $@
	ar rcs $@ $(LIB_OBJ)

build/experimental/model-expert-waves.o: src/engine/model.c $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -DMYDS4_EXPERT_WAVES -c -o $@ $<

build/experimental/io-expert-waves.o: src/engine/io.c $(MODEL_HEADERS)
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -DMYDS4_EXPERT_WAVES -c -o $@ $<

build/libmyds4-expert-waves.a: $(filter-out build/lib/engine/model.o build/lib/engine/io.o,$(LIB_OBJ)) build/experimental/model-expert-waves.o build/experimental/io-expert-waves.o
	rm -f $@
	ar rcs $@ $^

build/experimental/io-prefetch-demote.o: src/engine/io.c $(MODEL_HEADERS)
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -DMYDS4_PREFETCH_DEMOTE_CANCEL -c -o $@ $<

build/libmyds4-prefetch-demote.a: $(filter-out build/lib/engine/io.o,$(LIB_OBJ)) build/experimental/io-prefetch-demote.o
	rm -f $@
	ar rcs $@ $^

build/experimental/model-prefetch.o: src/engine/model.c src/experimental/runahead.c $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -DMYDS4_BATCH_PREFETCH_STUDY -DMYDS4_RUNAHEAD -c -o $@ $<

build/experimental/io-runahead.o: src/engine/io.c $(MODEL_HEADERS)
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -DMYDS4_RUNAHEAD -c -o $@ $<

build/libmyds4-prefetch-study.a: $(filter-out build/lib/engine/model.o build/lib/engine/io.o,$(LIB_OBJ)) build/experimental/model-prefetch.o build/experimental/io-runahead.o
	rm -f $@
	ar rcs $@ $^

# The Rust server. Its target directory is build/cargo, as server/.cargo/config.toml
# sets it for cargo run inside server/.
CARGO ?= cargo
build/myds4-server: build/libmyds4.a server/Cargo.toml server/Cargo.lock server/build.rs $(wildcard server/src/*.rs)
	$(CARGO) build --release --locked --manifest-path server/Cargo.toml --target-dir build/cargo
	cp build/cargo/release/myds4-server $@

server: build/myds4-server

# Shares Cargo's dependency cache; build normal and study executables sequentially.
build/myds4-server-prefetch-study: build/libmyds4-prefetch-study.a server/Cargo.toml server/Cargo.lock server/build.rs $(wildcard server/src/*.rs)
	MYDS4_ENGINE_VARIANT=prefetch-study $(CARGO) build --release --locked --manifest-path server/Cargo.toml --target-dir build/cargo
	cp build/cargo/release/myds4-server $@

build/myds4-server-prefetch-demote: build/libmyds4-prefetch-demote.a server/Cargo.toml server/Cargo.lock server/build.rs $(wildcard server/src/*.rs)
	MYDS4_ENGINE_VARIANT=prefetch-demote $(CARGO) build --release --locked --manifest-path server/Cargo.toml --target-dir build/cargo
	cp build/cargo/release/myds4-server $@

build/myds4-server-expert-waves: build/libmyds4-expert-waves.a server/Cargo.toml server/Cargo.lock server/build.rs $(wildcard server/src/*.rs)
	MYDS4_ENGINE_VARIANT=expert-waves $(CARGO) build --release --locked --manifest-path server/Cargo.toml --target-dir build/cargo
	cp build/cargo/release/myds4-server $@

build/batch-prefetch-check: tools/checks/batch_prefetch_check.c $(SRC) $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ tools/checks/batch_prefetch_check.c $(filter-out src/cli/main.c src/engine/model.c,$(SRC)) $(LDLIBS)

build/cache-hold-check: tools/checks/cache_hold_check.c src/engine/io.c src/engine/gpu.c src/engine/util.c $(MODEL_HEADERS)
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ tools/checks/cache_hold_check.c src/engine/gpu.c src/engine/util.c $(LDLIBS)

build/runahead-model-check: tools/checks/runahead_model_check.c src/experimental/runahead.c $(SRC) $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_RUNAHEAD -o $@ tools/checks/runahead_model_check.c $(filter-out src/cli/main.c src/engine/model.c,$(SRC)) $(LDLIBS)

# The Rust server's tests (the live view's frames, request conversion), in
# build/cargo beside the release build.
server-check: build/libmyds4.a
	$(CARGO) test --release --locked --manifest-path server/Cargo.toml --target-dir build/cargo

build/online-predictor-unit: tools/experimental/online_predictor_unit.c src/experimental/online_predictor.c src/experimental/online_predictor_math.h src/engine/myds4.h
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ tools/experimental/online_predictor_unit.c -framework Accelerate

build/online-predictor-state-check: tools/experimental/online_predictor_state_check.c src/experimental/online_predictor.c src/experimental/online_predictor_math.h src/engine/myds4.h
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ tools/experimental/online_predictor_state_check.c -framework Accelerate

build/online-predictor-check: tools/experimental/online_predictor_check.c src/experimental/online_predictor_math.h
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ tools/experimental/online_predictor_check.c -framework Accelerate

build/sampling-check: tools/checks/sampling_check.c src/engine/sample.c src/engine/util.c src/engine/myds4.h
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ tools/checks/sampling_check.c src/engine/sample.c src/engine/util.c

sampling-check: build/sampling-check
	./build/sampling-check

build/gpu-clock-check: tools/checks/gpu_clock_check.c src/engine/gpu.c src/kernels/kernels.metal src/engine/util.c src/engine/myds4.h
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ tools/checks/gpu_clock_check.c src/engine/util.c $(LDLIBS)

gpu-clock-check: build/gpu-clock-check
	./build/gpu-clock-check

build/shutdown-check: tools/checks/shutdown_check.c src/engine/util.c src/engine/myds4.h
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ tools/checks/shutdown_check.c src/engine/util.c

shutdown-check: build/shutdown-check
	./build/shutdown-check

# Diagnostic pipe capture only; its synchronous writes change inference timing.
build/myds4-predict-probe: $(SRC) tools/experimental/predict_probe.c $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_PREDICT_PROBE -o $@ $(SRC) tools/experimental/predict_probe.c $(LDLIBS)

predict-probe: build/myds4-predict-probe

# IQ1_M expert decoding on the GPU against llama.cpp's, on random blocks and,
# with GGUF=FILE, on that file's IQ1_M expert tensors; no model is loaded.
build/gguf-decode-check: tools/reference/gguf_decode_check.c src/engine/gpu.c src/engine/util.c src/engine/myds4.h src/kernels/kernels.metal src/kernels/gguf_grids.metal src/kernels/gguf.metal
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ tools/reference/gguf_decode_check.c src/engine/gpu.c src/engine/util.c $(LDLIBS)

gguf-decode-check: build/gguf-decode-check
	./build/gguf-decode-check $(GGUF)

# CPU-only cache checks; the fixture creates no model buffers or read files.
build/cache-ready-check: tools/checks/cache_ready_check.c src/engine/io.c src/engine/gpu.c src/engine/util.c src/engine/myds4.h
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ tools/checks/cache_ready_check.c src/engine/gpu.c src/engine/util.c $(LDLIBS)

cache-ready-check: build/cache-ready-check
	./build/cache-ready-check

build/draft-cache-check: tools/checks/draft_cache_check.c src/engine/io.c src/engine/gpu.c src/engine/util.c src/engine/myds4.h
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ tools/checks/draft_cache_check.c src/engine/gpu.c src/engine/util.c $(LDLIBS)

draft-cache-check: build/draft-cache-check
	./build/draft-cache-check

build/prefetch-metrics-check: tools/checks/prefetch_metrics_check.c src/engine/io.c src/engine/gpu.c src/engine/util.c src/engine/myds4.h
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ tools/checks/prefetch_metrics_check.c src/engine/gpu.c src/engine/util.c $(LDLIBS)

prefetch-metrics-check: build/prefetch-metrics-check
	./build/prefetch-metrics-check

build/prefetch-queue-check: tools/checks/prefetch_queue_check.c src/engine/io.c src/engine/gpu.c src/engine/util.c src/engine/myds4.h
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ tools/checks/prefetch_queue_check.c src/engine/gpu.c src/engine/util.c $(LDLIBS)

build/prefetch-admission-check: tools/checks/prefetch_admission_check.c src/engine/io.c src/engine/gpu.c src/engine/util.c $(MODEL_HEADERS)
	@mkdir -p build
	$(CC) $(filter-out -O2,$(CFLAGS)) -O0 -o $@ tools/checks/prefetch_admission_check.c src/engine/gpu.c src/engine/util.c $(LDLIBS)

build/prefetch-demote-check: tools/checks/prefetch_demote_check.c src/engine/io.c src/engine/gpu.c src/engine/util.c $(MODEL_HEADERS)
	@mkdir -p build
	$(CC) $(filter-out -O2,$(CFLAGS)) -O0 -o $@ tools/checks/prefetch_demote_check.c src/engine/gpu.c src/engine/util.c $(LDLIBS)

build/prefetch-refine-race-check: tools/checks/prefetch_refine_race_check.c src/engine/io.c src/engine/gpu.c src/engine/util.c $(MODEL_HEADERS)
	@mkdir -p build
	$(CC) $(filter-out -O2,$(CFLAGS)) -O0 -o $@ tools/checks/prefetch_refine_race_check.c src/engine/gpu.c src/engine/util.c $(LDLIBS)

prefetch-admission-check: build/prefetch-admission-check
	./build/prefetch-admission-check

prefetch-demote-check: build/prefetch-demote-check
	./build/prefetch-demote-check

prefetch-refine-race-check: build/prefetch-refine-race-check
	./build/prefetch-refine-race-check

prefetch-queue-check: build/prefetch-queue-check
	./build/prefetch-queue-check

build/cache-policy-check: tools/checks/cache_policy_check.c src/engine/io.c src/engine/gpu.c src/engine/util.c src/engine/myds4.h
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ tools/checks/cache_policy_check.c src/engine/gpu.c src/engine/util.c $(LDLIBS)

cache-policy-check: build/cache-policy-check
	./build/cache-policy-check

build/memory-available-check: tools/checks/memory_available_check.c src/engine/io.c src/engine/gpu.c src/engine/util.c src/engine/myds4.h
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ tools/checks/memory_available_check.c src/engine/gpu.c src/engine/util.c $(LDLIBS)

memory-available-check: build/memory-available-check
	./build/memory-available-check

build/cache-memory-check: tools/checks/cache_memory_check.c src/engine/io.c src/engine/gpu.c src/engine/util.c src/engine/myds4.h
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ tools/checks/cache_memory_check.c src/engine/gpu.c src/engine/util.c $(LDLIBS)

cache-memory-check: build/cache-memory-check
	./build/cache-memory-check

.PHONY: cache-memory-check

clean:
	rm -rf build

.PHONY: clean server server-check predict-probe gguf-decode-check cache-ready-check draft-cache-check prefetch-metrics-check prefetch-queue-check prefetch-admission-check prefetch-demote-check prefetch-refine-race-check prefetch-pool-check expert-ready-check cache-policy-check memory-available-check shutdown-check sampling-check gpu-clock-check activation-format-check workspace-check state-fork-check context-lookup-check verification-schedule-check
