.DEFAULT_GOAL := build/myds4
CC = clang
CFLAGS = -std=c23 -O2 -Wall -Wextra -Wno-unused-parameter -Isrc/engine $(shell pkg-config --cflags yyjson libjpeg)
LDLIBS = -framework Metal -framework Foundation -framework IOKit -framework ImageIO -framework CoreGraphics -framework Accelerate -lobjc \
         $(shell pkg-config --libs yyjson libjpeg)
SRC = src/cli/main.c src/engine/model.c src/engine/io.c src/engine/gpu.c src/engine/tok.c src/engine/util.c src/engine/vision.c src/experimental/online_predictor.c src/experimental/context_lookup.c src/engine/sample.c
MODEL_HEADERS = src/engine/myds4.h src/experimental/online_predictor_math.h src/experimental/context_lookup.h src/experimental/prefill_prune.h src/experimental/runahead.c
KERNELS = src/kernels/kernels.metal src/kernels/gguf_grids.metal src/kernels/gguf.metal src/kernels/vision.metal

build/myds4: $(SRC) $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ $(SRC) $(LDLIBS)

# The Qwen3.8 engine (src/qwen/, docs/qwen.md) and its check CLI.
QWEN_SRC = src/qwen/qwen.c src/engine/gpu.c src/engine/util.c src/engine/sample.c
build/qwen: src/qwen/qwen_cli.c $(QWEN_SRC) src/qwen/qwen.h src/qwen/qwen.metal src/engine/myds4.h
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ src/qwen/qwen_cli.c $(QWEN_SRC) $(LDLIBS)

# The engine behind src/engine/engine.h, for the Rust server (server/): the sources
# of build/myds4 without those in src/cli/, and with engine.c.
LIB_SRC = src/engine/model.c src/engine/io.c src/engine/gpu.c src/engine/tok.c src/engine/util.c src/engine/vision.c src/experimental/online_predictor.c src/experimental/context_lookup.c src/engine/sample.c src/engine/engine.c src/qwen/qwen.c
LIB_OBJ = $(LIB_SRC:src/%.c=build/lib/%.o)

build/lib/%.o: src/%.c $(MODEL_HEADERS) src/engine/engine.h $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -c -o $@ $<

build/lib/qwen/qwen.o: src/qwen/qwen.metal src/qwen/qwen.h

build/libmyds4.a: $(LIB_OBJ)
	rm -f $@
	ar rcs $@ $(LIB_OBJ)

# The Rust server. Its target directory is build/cargo, as server/.cargo/config.toml
# sets it for cargo run inside server/. A new executable replaces the old one by a
# rename: a server still running keeps its own file, which macOS can kill it for
# rewriting in place.
CARGO ?= cargo
# The crate's inputs: its manifest, lock file, build script and every source file at any depth.
SERVER_INPUTS := server/Cargo.toml server/Cargo.lock server/build.rs $(shell find server/src -name '*.rs')
build/myds4-server: build/libmyds4.a $(SERVER_INPUTS)
	$(CARGO) build --release --locked --manifest-path server/Cargo.toml --target-dir build/cargo
	cp build/cargo/release/myds4-server $@.new && mv -f $@.new $@

server: build/myds4-server

# The Rust server's tests (the live view's frames, request conversion), in
# build/cargo beside the release build.
server-check: build/libmyds4.a
	$(CARGO) test --release --locked --manifest-path server/Cargo.toml --target-dir build/cargo

# ---- CPU checks: no model, and no Metal device unless noted ----

build/activation-format-check: tools/checks/activation_format_check.c tools/reference/activation_format.h
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ tools/checks/activation_format_check.c

activation-format-check: build/activation-format-check
	./build/activation-format-check

build/context-lookup-check: tools/checks/context_lookup_check.c src/experimental/context_lookup.c src/experimental/context_lookup.h
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ tools/checks/context_lookup_check.c src/experimental/context_lookup.c

context-lookup-check: build/context-lookup-check
	./build/context-lookup-check

build/context-lookup-replay: tools/experimental/context_lookup_replay.c src/experimental/context_lookup.c src/experimental/context_lookup.h
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ tools/experimental/context_lookup_replay.c src/experimental/context_lookup.c

build/prefill-prune-check: tools/checks/prefill_prune_check.c src/experimental/prefill_prune.h
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ tools/checks/prefill_prune_check.c

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

verification-schedule-check: build/verification-schedule-check
	./build/verification-schedule-check

build/attention-candidate-check: tools/checks/attention_candidate_check.c $(SRC) $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ tools/checks/attention_candidate_check.c $(filter-out src/cli/main.c src/engine/model.c,$(SRC)) $(LDLIBS)

build/tokenizer-added-check: tools/checks/tokenizer_added_check.c src/engine/tok.c src/engine/util.c $(MODEL_HEADERS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer -o $@ tools/checks/tokenizer_added_check.c src/engine/tok.c src/engine/util.c $(shell pkg-config --libs yyjson)

build/tokenizer-long-check: tools/checks/tokenizer_long_check.c src/engine/tok.c src/engine/util.c $(MODEL_HEADERS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer -o $@ tools/checks/tokenizer_long_check.c src/engine/tok.c src/engine/util.c $(shell pkg-config --libs yyjson)

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

build/shutdown-check: tools/checks/shutdown_check.c src/engine/util.c src/engine/myds4.h
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ tools/checks/shutdown_check.c src/engine/util.c

shutdown-check: build/shutdown-check
	./build/shutdown-check

# CPU-only cache checks; the fixture creates no model buffers or read files.
build/cache-ready-check: tools/checks/cache_ready_check.c src/engine/io.c src/engine/gpu.c src/engine/util.c src/engine/myds4.h
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ tools/checks/cache_ready_check.c src/engine/gpu.c src/engine/util.c $(LDLIBS)

cache-ready-check: build/cache-ready-check
	./build/cache-ready-check

build/cache-hold-check: tools/checks/cache_hold_check.c src/engine/io.c src/engine/gpu.c src/engine/util.c $(MODEL_HEADERS)
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ tools/checks/cache_hold_check.c src/engine/gpu.c src/engine/util.c $(LDLIBS)

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

build/prefetch-admission-check: tools/checks/prefetch_admission_check.c src/engine/io.c src/engine/gpu.c src/engine/util.c $(MODEL_HEADERS)
	@mkdir -p build
	$(CC) $(filter-out -O2,$(CFLAGS)) -O0 -o $@ tools/checks/prefetch_admission_check.c src/engine/gpu.c src/engine/util.c $(LDLIBS)

prefetch-admission-check: build/prefetch-admission-check
	./build/prefetch-admission-check

build/prefetch-demote-check: tools/checks/prefetch_demote_check.c src/engine/io.c src/engine/gpu.c src/engine/util.c $(MODEL_HEADERS)
	@mkdir -p build
	$(CC) $(filter-out -O2,$(CFLAGS)) -O0 -o $@ tools/checks/prefetch_demote_check.c src/engine/gpu.c src/engine/util.c $(LDLIBS)

prefetch-demote-check: build/prefetch-demote-check
	./build/prefetch-demote-check

build/prefetch-refine-race-check: tools/checks/prefetch_refine_race_check.c src/engine/io.c src/engine/gpu.c src/engine/util.c $(MODEL_HEADERS)
	@mkdir -p build
	$(CC) $(filter-out -O2,$(CFLAGS)) -O0 -o $@ tools/checks/prefetch_refine_race_check.c src/engine/gpu.c src/engine/util.c $(LDLIBS)

prefetch-refine-race-check: build/prefetch-refine-race-check
	./build/prefetch-refine-race-check

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

# ---- GPU checks without a model ----

build/mxfp4-activation-check: tools/checks/mxfp4_activation_check.c tools/reference/activation_format.h src/engine/gpu.c src/engine/util.c src/engine/myds4.h $(KERNELS)
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ tools/checks/mxfp4_activation_check.c src/engine/gpu.c src/engine/util.c $(LDLIBS)

build/gpu-wait-check: tools/checks/gpu_wait_check.c src/engine/gpu.c src/engine/util.c $(MODEL_HEADERS)
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ tools/checks/gpu_wait_check.c src/engine/gpu.c src/engine/util.c $(LDLIBS)

build/gpu-clock-check: tools/checks/gpu_clock_check.c src/engine/gpu.c src/kernels/kernels.metal src/engine/util.c src/engine/myds4.h
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ tools/checks/gpu_clock_check.c src/engine/util.c $(LDLIBS)

gpu-clock-check: build/gpu-clock-check
	./build/gpu-clock-check

# IQ1_M expert decoding on the GPU against llama.cpp's, on random blocks and,
# with GGUF=FILE, on that file's IQ1_M expert tensors; no model is loaded.
build/gguf-decode-check: tools/reference/gguf_decode_check.c src/engine/gpu.c src/engine/util.c src/engine/myds4.h src/kernels/kernels.metal src/kernels/gguf_grids.metal src/kernels/gguf.metal
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ tools/reference/gguf_decode_check.c src/engine/gpu.c src/engine/util.c $(LDLIBS)

gguf-decode-check: build/gguf-decode-check
	./build/gguf-decode-check $(GGUF)

# ---- Checks that load the model ----

build/prefill-prune-model-check: tools/checks/prefill_prune_model_check.c $(SRC) $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ tools/checks/prefill_prune_model_check.c $(filter-out src/cli/main.c src/engine/model.c,$(SRC)) $(LDLIBS)

build/draft-rows-check: tools/checks/draft_rows_check.c $(SRC) $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ tools/checks/draft_rows_check.c $(filter-out src/cli/main.c src/engine/model.c,$(SRC)) $(LDLIBS)

build/prefix-fork-check: tools/checks/prefix_fork_check.c $(SRC) $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ tools/checks/prefix_fork_check.c $(filter-out src/cli/main.c src/engine/model.c,$(SRC)) $(LDLIBS)

build/verification-model-check: tools/checks/verification_model_check.c $(SRC) $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ tools/checks/verification_model_check.c $(filter-out src/cli/main.c,$(SRC)) $(LDLIBS)

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

build/runahead-model-check: tools/checks/runahead_model_check.c $(SRC) $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ tools/checks/runahead_model_check.c $(filter-out src/cli/main.c src/engine/model.c,$(SRC)) $(LDLIBS)

# Decoding B sequences together: step time, GPU busy and idle, reads per row.
build/rows-bench: tools/bench/rows_bench.c $(SRC) $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ tools/bench/rows_bench.c $(filter-out src/cli/main.c src/engine/model.c,$(SRC)) $(LDLIBS)

# ---- Diagnostics ----

# Diagnostic pipe capture only; its synchronous writes change inference timing.
build/myds4-predict-probe: $(SRC) tools/experimental/predict_probe.c $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_PREDICT_PROBE -o $@ $(SRC) tools/experimental/predict_probe.c $(LDLIBS)

predict-probe: build/myds4-predict-probe

clean:
	rm -rf build

.PHONY: clean server server-check predict-probe activation-format-check context-lookup-check workspace-check state-fork-check verification-schedule-check sampling-check shutdown-check cache-ready-check draft-cache-check prefetch-metrics-check prefetch-queue-check prefetch-admission-check prefetch-demote-check prefetch-refine-race-check cache-policy-check memory-available-check cache-memory-check gpu-clock-check gguf-decode-check
