.DEFAULT_GOAL := build/myds4
CC = clang
CFLAGS = -std=c23 -O2 -Wall -Wextra -Wno-unused-parameter -Isrc/engine $(shell pkg-config --cflags yyjson libjpeg)
LDLIBS = -framework Metal -framework Foundation -framework IOKit -framework ImageIO -framework CoreGraphics -framework Accelerate -lobjc \
         $(shell pkg-config --libs yyjson libjpeg)
SRC = src/cli/main.c src/engine/model.c src/engine/io.c src/engine/gpu.c src/engine/tok.c src/engine/util.c src/engine/vision.c src/experimental/online_predictor.c src/experimental/context_lookup.c src/engine/sample.c
MODEL_HEADERS = src/engine/myds4.h src/experimental/online_predictor_math.h src/experimental/context_lookup.h src/engine/expert_batch.h src/experimental/prefill_prune.h src/experimental/urgent_io_diagnostic.h
KERNELS = src/kernels/kernels.metal src/kernels/gguf_grids.metal src/kernels/gguf.metal src/kernels/vision.metal

build/activation-format-check: tools/checks/activation_format_check.c tools/reference/activation_format.h
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ tools/checks/activation_format_check.c

activation-format-check: build/activation-format-check
	./build/activation-format-check

build/mxfp4-activation-check: tools/checks/mxfp4_activation_check.c tools/reference/activation_format.h src/engine/gpu.c src/engine/util.c src/engine/myds4.h $(KERNELS)
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ tools/checks/mxfp4_activation_check.c src/engine/gpu.c src/engine/util.c $(LDLIBS)

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

build/draft-distribution-probe: tools/experimental/draft_distribution_probe.c $(SRC) $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ tools/experimental/draft_distribution_probe.c $(filter-out src/cli/main.c src/engine/model.c,$(SRC)) $(LDLIBS)

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

build/prefill-schedule-check: tools/checks/prefill_schedule_check.c src/engine/engine.c src/engine/engine.h $(SRC) $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ tools/checks/prefill_schedule_check.c $(filter-out src/cli/main.c,$(SRC)) $(LDLIBS)

build/prefill-state-check: tools/checks/prefill_state_check.c $(SRC) $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ tools/checks/prefill_state_check.c $(filter-out src/cli/main.c src/engine/model.c,$(SRC)) $(LDLIBS)

build/prefill-prefix-check: tools/checks/prefill_prefix_check.c $(SRC) $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ tools/checks/prefill_prefix_check.c $(filter-out src/cli/main.c src/engine/model.c,$(SRC)) $(LDLIBS)

build/attention-candidate-check: tools/checks/attention_candidate_check.c $(SRC) $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ tools/checks/attention_candidate_check.c $(filter-out src/cli/main.c src/engine/model.c,$(SRC)) $(LDLIBS)

build/expert-batch-check: tools/checks/expert_batch_check.c tools/reference/activation_format.h src/engine/gpu.c src/engine/util.c src/engine/myds4.h src/engine/expert_batch.h src/kernels/kernels.metal src/kernels/gguf_grids.metal src/kernels/gguf.metal
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ tools/checks/expert_batch_check.c src/engine/gpu.c src/engine/util.c $(LDLIBS)

# Experimental exact sharing of resident expert weights across verification rows.
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

build/myds4-prefix-prefill: $(SRC) $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_PREFILL_PREFIX -o $@ $(SRC) $(LDLIBS)

build/myds4-prefix-prune: $(SRC) $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_PREFILL_PREFIX -DMYDS4_PREFILL_PRUNE -o $@ $(SRC) $(LDLIBS)

build/myds4-prefix-decode-score: $(SRC) $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_PREFILL_PREFIX -DMYDS4_DECODE_ONLY_SCORE -o $@ $(SRC) $(LDLIBS)

build/myds4-prune-decode-score: $(SRC) $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_PREFILL_PREFIX -DMYDS4_PREFILL_PRUNE -DMYDS4_DECODE_ONLY_SCORE -o $@ $(SRC) $(LDLIBS)

build/myds4-prefix-prune-consensus: $(SRC) $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_PREFILL_PREFIX -DMYDS4_PREFILL_PRUNE -DMYDS4_CONSENSUS_LOOKUP -o $@ $(SRC) $(LDLIBS)

build/myds4-consensus-prefetch1: $(SRC) $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_PREFILL_PREFIX -DMYDS4_PREFILL_PRUNE -DMYDS4_CONSENSUS_LOOKUP -DMYDS4_VERIFY_PREFETCH_ROWS=1 -o $@ $(SRC) $(LDLIBS)

build/myds4-consensus-row-credit: $(SRC) $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_PREFILL_PREFIX -DMYDS4_PREFILL_PRUNE -DMYDS4_CONSENSUS_LOOKUP -DMYDS4_VERIFY_ROW_CREDIT -o $@ $(SRC) $(LDLIBS)

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

# The Rust server. Its target directory is build/cargo, as server/.cargo/config.toml
# sets it for cargo run inside server/.
CARGO ?= cargo
build/myds4-server: build/libmyds4.a server/Cargo.toml server/Cargo.lock server/build.rs $(wildcard server/src/*.rs)
	$(CARGO) build --release --locked --manifest-path server/Cargo.toml --target-dir build/cargo
	cp build/cargo/release/myds4-server $@

server: build/myds4-server

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

build/prefetch-queue-diag-check: tools/checks/prefetch_queue_check.c src/engine/io.c src/engine/gpu.c src/engine/util.c $(MODEL_HEADERS)
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_IO_DIAGNOSTICS -o $@ tools/checks/prefetch_queue_check.c src/engine/gpu.c src/engine/util.c $(LDLIBS)

clean:
	rm -rf build

.PHONY: clean server server-check predict-probe gguf-decode-check cache-ready-check draft-cache-check prefetch-metrics-check prefetch-queue-check cache-policy-check memory-available-check shutdown-check sampling-check gpu-clock-check activation-format-check workspace-check state-fork-check context-lookup-check verification-schedule-check
