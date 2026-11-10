.DEFAULT_GOAL := build/myds4
CC = clang
CFLAGS = -std=c23 -O2 -Wall -Wextra -Wno-unused-parameter -Isrc/engine $(shell pkg-config --cflags yyjson libjpeg)
LDLIBS = -framework Metal -framework Foundation -framework IOKit -framework ImageIO -framework CoreGraphics -framework AudioToolbox -framework Accelerate -lobjc -licucore \
         $(shell pkg-config --libs yyjson libjpeg)
MODEL_HEADERS = src/engine/expert_store.h src/model/experts.h src/model/nemotron/dflash.h src/model/nemotron/audio.h src/model/linear.h src/engine/memory.h src/model/weights.h src/model/deepseek/model.h src/engine/myds4.h src/model/model.h src/model/qwen/model.h src/model/qwen/vision.h src/model/deepseek/engram.h src/model/deepseek/learning.h src/model/draft.h src/model/deepseek/prefill.h src/model/deepseek/expert_tasks.h
KERNELS = src/kernels/formats.metal src/kernels/kernels.metal src/kernels/gguf_grids.metal src/kernels/gguf.metal \
          src/kernels/vision.metal src/kernels/qwen_vision.metal src/kernels/rows.metal src/kernels/products.metal src/kernels/experts.metal src/kernels/deltanet.metal src/kernels/gqa.metal src/kernels/flash.metal src/kernels/nemotron.metal src/kernels/speech.metal src/kernels/dflash.metal src/kernels/expert_tasks.metal

build/myds4: src/cli/main.c build/libmyds4.a $(MODEL_HEADERS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ src/cli/main.c build/libmyds4.a $(LDLIBS)

# The engine behind src/engine/engine.h, for the Rust server (server/): the sources
# of build/myds4 without those in src/cli/, and with engine.c.
LIB_SRC = src/model/experts.c src/model/nemotron/dflash.c src/model/nemotron/audio.c src/engine/media.c src/engine/audio.c src/model/nemotron/model.c src/model/qwen/flash.c src/model/linear.c src/engine/memory.c src/model/weights.c src/model/model.c src/model/deepseek/model.c src/model/deepseek/engram.c src/model/qwen/model.c src/model/qwen/vision.c src/kernels/library.c src/engine/io.c src/engine/gpu.c src/engine/tok.c src/engine/util.c src/engine/vision.c src/model/deepseek/predictor.c src/model/deepseek/learning.c src/model/draft.c src/engine/sample.c src/engine/engine.c
LIB_OBJ = $(LIB_SRC:src/%.c=build/lib/%.o)

build/lib/%.o: src/%.c $(MODEL_HEADERS) src/engine/engine.h
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -MMD -MP -c -o $@ $<

build/lib/kernels/library.o: $(KERNELS)

-include $(LIB_OBJ:.o=.d)

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

# Fast tests never acquire model weights or run GPU kernels.
CPU_CHECKS = activation-format-check context-lookup-check workspace-check state-fork-check verification-schedule-check sampling-check shutdown-check cache-ready-check draft-cache-check prefetch-metrics-check prefetch-queue-check prefetch-admission-check prefetch-demote-check prefetch-refine-race-check cache-policy-check memory-available-check cache-memory-check memory-budget-check weights-stop-check weights-read-check
EXTRA_CPU = memory metadata qwen-state-accounting attention-candidate prefill-prune cache-hold cache-progress cache-mapping predictor-learning predictor-state predictor-numerics
# These need the GPU or model weights to run. make test only builds them, so
# that a changed engine interface cannot leave one unbuildable.
BUILD_ONLY = $(addprefix build/tests/,nemotron experts routing vision residency mxfp4-activation gpu-wait gpu-clock \
  pass-eviction gguf-decode moe-tasks memory-model qwen-mtp flash-model speech-model nemotron-model prefill-prune-model \
  draft-rows prefix-fork verification-model ced dspark-state prefill-state runahead-model) \
  build/benchmarks/decode build/benchmarks/products

test: $(CPU_CHECKS) $(addprefix build/tests/,$(EXTRA_CPU)) $(BUILD_ONLY) build/myds4-server
	@set -e; for test in $(EXTRA_CPU); do ./build/tests/$$test; done
	env -u MYDS4_QWEN -u MYDS4_FLASH -u MYDS4_NEMOTRON -u MYDS4_DFLASH -u MYDS4_CHECKPOINT -u MYDS4_EXPERTS $(MAKE) server-check

# ---- CPU checks: no model, and no Metal device unless noted ----

build/tests/memory: tests/unit/memory.c src/engine/memory.c src/engine/memory.h
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ tests/unit/memory.c src/engine/memory.c

build/tests/memory-model: tests/model/memory.c build/libmyds4.a $(MODEL_HEADERS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ tests/model/memory.c build/libmyds4.a $(LDLIBS)

build/tests/qwen-mtp: tests/model/qwen_mtp.c build/libmyds4.a $(MODEL_HEADERS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ tests/model/qwen_mtp.c build/libmyds4.a $(LDLIBS)

build/tests/flash-model: tests/model/flash.c build/libmyds4.a $(MODEL_HEADERS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ tests/model/flash.c build/libmyds4.a $(LDLIBS)

build/tests/nemotron: tests/gpu/nemotron.c build/libmyds4.a $(MODEL_HEADERS)
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -o $@ tests/gpu/nemotron.c build/libmyds4.a $(LDLIBS)

build/tests/experts: tests/gpu/experts.c build/libmyds4.a src/model/experts.h
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -o $@ tests/gpu/experts.c build/libmyds4.a $(LDLIBS)

build/tests/routing: tests/gpu/routing.c build/libmyds4.a $(KERNELS)
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -o $@ tests/gpu/routing.c build/libmyds4.a $(LDLIBS)

build/tests/speech-model: tests/model/speech.c build/libmyds4.a $(MODEL_HEADERS)
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -o $@ tests/model/speech.c build/libmyds4.a $(LDLIBS)

build/tests/nemotron-model: tests/model/nemotron.c build/libmyds4.a $(MODEL_HEADERS)
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -o $@ tests/model/nemotron.c build/libmyds4.a $(LDLIBS)

build/tests/metadata: tests/unit/checkpoint.c build/libmyds4.a $(MODEL_HEADERS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ tests/unit/checkpoint.c build/libmyds4.a $(LDLIBS)


build/tests/vision: tests/gpu/vision.c build/libmyds4.a $(MODEL_HEADERS)
	@mkdir -p build/tests
	$(CC) $(CFLAGS) -o $@ tests/gpu/vision.c build/libmyds4.a $(LDLIBS)

build/tests/activation-format: tests/unit/activation_format.c tests/reference/activations.h
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ tests/unit/activation_format.c

activation-format-check: build/tests/activation-format
	./build/tests/activation-format

build/tests/context-lookup: tests/unit/context_lookup.c src/model/draft.c src/model/draft.h
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ tests/unit/context_lookup.c src/model/draft.c

context-lookup-check: build/tests/context-lookup
	./build/tests/context-lookup

build/tests/prefill-prune: tests/unit/prefill_prune.c src/model/deepseek/prefill.h
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ tests/unit/prefill_prune.c

# Host-backed workspace checks; no Metal device or model weights are opened.
build/tests/workspace: tests/unit/workspace.c build/libmyds4.a $(MODEL_HEADERS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ tests/unit/workspace.c build/libmyds4.a $(LDLIBS)

workspace-check: build/tests/workspace
	./build/tests/workspace

build/tests/state-fork: tests/unit/state_fork.c build/libmyds4.a $(MODEL_HEADERS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ tests/unit/state_fork.c build/libmyds4.a $(LDLIBS)

state-fork-check: build/tests/state-fork
	./build/tests/state-fork

build/tests/qwen-state-accounting: tests/unit/qwen_state_accounting.c build/libmyds4.a $(MODEL_HEADERS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ tests/unit/qwen_state_accounting.c build/libmyds4.a $(LDLIBS)

build/tests/verification-schedule: tests/unit/verification_schedule.c build/libmyds4.a $(MODEL_HEADERS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ tests/unit/verification_schedule.c build/libmyds4.a $(LDLIBS)

verification-schedule-check: build/tests/verification-schedule
	./build/tests/verification-schedule

build/tests/attention-candidate: tests/unit/attention_candidate.c build/libmyds4.a $(MODEL_HEADERS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ tests/unit/attention_candidate.c build/libmyds4.a $(LDLIBS)

build/tests/tokenizer-reference: tests/unit/tokenizer_reference.c src/engine/tok.c src/engine/util.c src/engine/myds4.h
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer -o $@ tests/unit/tokenizer_reference.c src/engine/tok.c src/engine/util.c $(shell pkg-config --libs yyjson) -licucore

build/tests/predictor-learning: tests/unit/predictor_learning.c src/model/deepseek/predictor.c src/model/deepseek/engram.h src/model/deepseek/learning.h src/engine/myds4.h src/model/deepseek/learning.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ tests/unit/predictor_learning.c src/model/deepseek/learning.c src/engine/memory.c src/engine/util.c $(shell pkg-config --libs yyjson) -framework Accelerate

build/tests/predictor-state: tests/unit/predictor_state.c src/model/deepseek/predictor.c src/model/deepseek/engram.h src/model/deepseek/learning.h src/engine/myds4.h src/model/deepseek/learning.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ tests/unit/predictor_state.c src/model/deepseek/learning.c src/engine/memory.c src/engine/util.c $(shell pkg-config --libs yyjson) -framework Accelerate

build/tests/predictor-numerics: tests/unit/predictor_numerics.c src/model/deepseek/engram.h src/model/deepseek/learning.h src/model/deepseek/learning.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ tests/unit/predictor_numerics.c src/model/deepseek/learning.c src/engine/memory.c src/engine/util.c $(shell pkg-config --libs yyjson) -framework Accelerate

build/tests/sampling: tests/unit/sampling.c src/engine/sample.c src/engine/util.c src/engine/myds4.h
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ tests/unit/sampling.c src/engine/sample.c src/engine/util.c $(shell pkg-config --libs yyjson)

sampling-check: build/tests/sampling
	./build/tests/sampling

build/tests/shutdown: tests/unit/shutdown.c src/engine/util.c src/engine/myds4.h
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ tests/unit/shutdown.c src/engine/util.c $(shell pkg-config --libs yyjson)

shutdown-check: build/tests/shutdown
	./build/tests/shutdown

# CPU-only cache checks; the fixture creates no model buffers or read files.
build/tests/cache-ready: tests/unit/cache_ready.c src/engine/io.c src/engine/expert_store.h tests/unit/cache_layout.h src/engine/gpu.c src/engine/util.c src/engine/myds4.h
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ tests/unit/cache_ready.c src/engine/gpu.c src/engine/util.c src/engine/memory.c $(LDLIBS)

cache-ready-check: build/tests/cache-ready
	./build/tests/cache-ready

build/tests/cache-mapping: tests/unit/cache_mapping.c src/engine/io.c src/engine/expert_store.h tests/unit/cache_layout.h src/engine/gpu.c src/engine/util.c src/engine/myds4.h
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ tests/unit/cache_mapping.c src/engine/gpu.c src/engine/util.c src/engine/memory.c $(LDLIBS)

build/tests/cache-progress: tests/unit/cache_progress.c src/engine/io.c src/engine/expert_store.h tests/unit/cache_layout.h src/engine/gpu.c src/engine/util.c src/engine/myds4.h
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ tests/unit/cache_progress.c src/engine/gpu.c src/engine/util.c src/engine/memory.c $(LDLIBS)

build/tests/cache-hold: tests/unit/cache_hold.c src/engine/io.c src/engine/expert_store.h tests/unit/cache_layout.h src/engine/gpu.c src/engine/util.c $(MODEL_HEADERS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ tests/unit/cache_hold.c src/engine/gpu.c src/engine/util.c src/engine/memory.c $(LDLIBS)

build/tests/draft-cache: tests/unit/draft_cache.c src/engine/io.c src/engine/expert_store.h tests/unit/cache_layout.h src/engine/gpu.c src/engine/util.c src/engine/myds4.h
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ tests/unit/draft_cache.c src/engine/gpu.c src/engine/util.c src/engine/memory.c $(LDLIBS)

draft-cache-check: build/tests/draft-cache
	./build/tests/draft-cache

build/tests/prefetch-metrics: tests/unit/prefetch_metrics.c src/engine/io.c src/engine/expert_store.h tests/unit/cache_layout.h src/engine/gpu.c src/engine/util.c src/engine/myds4.h
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ tests/unit/prefetch_metrics.c src/engine/gpu.c src/engine/util.c src/engine/memory.c $(LDLIBS)

prefetch-metrics-check: build/tests/prefetch-metrics
	./build/tests/prefetch-metrics

build/tests/prefetch-queue: tests/unit/prefetch_queue.c src/engine/io.c src/engine/expert_store.h tests/unit/cache_layout.h src/engine/gpu.c src/engine/util.c src/engine/myds4.h
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ tests/unit/prefetch_queue.c src/engine/gpu.c src/engine/util.c src/engine/memory.c $(LDLIBS)

prefetch-queue-check: build/tests/prefetch-queue
	./build/tests/prefetch-queue

build/tests/prefetch-admission: tests/unit/prefetch_admission.c src/engine/io.c src/engine/expert_store.h tests/unit/cache_layout.h src/engine/gpu.c src/engine/util.c $(MODEL_HEADERS)
	@mkdir -p $(dir $@)
	$(CC) $(filter-out -O2,$(CFLAGS)) -O0 -o $@ tests/unit/prefetch_admission.c src/engine/gpu.c src/engine/util.c src/engine/memory.c $(LDLIBS)

prefetch-admission-check: build/tests/prefetch-admission
	./build/tests/prefetch-admission

build/tests/prefetch-demote: tests/unit/prefetch_demote.c src/engine/io.c src/engine/expert_store.h tests/unit/cache_layout.h src/engine/gpu.c src/engine/util.c $(MODEL_HEADERS)
	@mkdir -p $(dir $@)
	$(CC) $(filter-out -O2,$(CFLAGS)) -O0 -o $@ tests/unit/prefetch_demote.c src/engine/gpu.c src/engine/util.c src/engine/memory.c $(LDLIBS)

prefetch-demote-check: build/tests/prefetch-demote
	./build/tests/prefetch-demote

build/tests/prefetch-refine-race: tests/unit/prefetch_refine_race.c src/engine/io.c src/engine/expert_store.h tests/unit/cache_layout.h src/engine/gpu.c src/engine/util.c $(MODEL_HEADERS)
	@mkdir -p $(dir $@)
	$(CC) $(filter-out -O2,$(CFLAGS)) -O0 -o $@ tests/unit/prefetch_refine_race.c src/engine/gpu.c src/engine/util.c src/engine/memory.c $(LDLIBS)

prefetch-refine-race-check: build/tests/prefetch-refine-race
	./build/tests/prefetch-refine-race

build/tests/cache-policy: tests/unit/cache_policy.c src/engine/io.c src/engine/expert_store.h tests/unit/cache_layout.h src/engine/gpu.c src/engine/util.c src/engine/myds4.h
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ tests/unit/cache_policy.c src/engine/gpu.c src/engine/util.c src/engine/memory.c $(LDLIBS)

cache-policy-check: build/tests/cache-policy
	./build/tests/cache-policy

build/tests/memory-available: tests/unit/memory_available.c src/engine/io.c src/engine/expert_store.h tests/unit/cache_layout.h src/engine/gpu.c src/engine/util.c src/engine/myds4.h
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ tests/unit/memory_available.c src/engine/gpu.c src/engine/util.c src/engine/memory.c $(LDLIBS)

memory-available-check: build/tests/memory-available
	./build/tests/memory-available

build/tests/cache-memory: tests/unit/cache_memory.c src/engine/io.c src/engine/expert_store.h tests/unit/cache_layout.h src/engine/gpu.c src/engine/util.c src/engine/myds4.h
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ tests/unit/cache_memory.c src/engine/gpu.c src/engine/util.c src/engine/memory.c $(LDLIBS)

cache-memory-check: build/tests/cache-memory
	./build/tests/cache-memory

build/tests/memory-budget: tests/unit/memory_budget.c src/engine/io.c src/engine/expert_store.h tests/unit/cache_layout.h src/engine/gpu.c src/engine/util.c src/engine/memory.c src/engine/myds4.h
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ tests/unit/memory_budget.c src/engine/gpu.c src/engine/util.c src/engine/memory.c $(LDLIBS)

memory-budget-check: build/tests/memory-budget
	./build/tests/memory-budget

build/tests/weights-stop: tests/unit/weights_stop.c src/model/weights.c src/model/weights.h src/engine/memory.c src/engine/util.c src/engine/myds4.h
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ tests/unit/weights_stop.c src/engine/memory.c src/engine/util.c $(LDLIBS)

weights-stop-check: build/tests/weights-stop
	./build/tests/weights-stop

build/tests/weights-read: tests/unit/weights_read.c src/model/weights.c src/model/weights.h src/engine/memory.c src/engine/util.c src/engine/myds4.h
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ tests/unit/weights_read.c src/engine/memory.c src/engine/util.c $(LDLIBS)

weights-read-check: build/tests/weights-read
	./build/tests/weights-read

# ---- GPU checks without a model ----

build/tests/residency: tests/gpu/residency.c src/engine/gpu.c src/engine/util.c src/engine/memory.c $(MODEL_HEADERS)
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -o $@ tests/gpu/residency.c src/engine/gpu.c src/engine/util.c src/engine/memory.c $(LDLIBS)

build/tests/mxfp4-activation: tests/gpu/mxfp4_activation.c tests/reference/activations.h src/engine/gpu.c src/engine/util.c src/engine/myds4.h $(KERNELS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ tests/gpu/mxfp4_activation.c src/engine/gpu.c src/engine/util.c src/engine/memory.c $(LDLIBS)

build/tests/gpu-wait: tests/gpu/gpu_wait.c src/engine/gpu.c src/engine/util.c $(MODEL_HEADERS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ tests/gpu/gpu_wait.c src/engine/gpu.c src/engine/util.c src/engine/memory.c $(LDLIBS)

build/tests/gpu-clock: tests/gpu/gpu_clock.c src/engine/gpu.c src/kernels/kernels.metal src/engine/util.c src/engine/myds4.h
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ tests/gpu/gpu_clock.c src/engine/util.c src/engine/memory.c $(LDLIBS)

gpu-clock-check: build/tests/gpu-clock
	./build/tests/gpu-clock

build/tests/pass-eviction: tests/gpu/pass_eviction.c src/engine/gpu.c src/engine/util.c src/engine/memory.c $(MODEL_HEADERS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ tests/gpu/pass_eviction.c src/engine/gpu.c src/engine/util.c src/engine/memory.c $(LDLIBS)

pass-eviction-check: build/tests/pass-eviction
	./build/tests/pass-eviction

# IQ1_M expert decoding on the GPU against llama.cpp's, on random blocks and,
# with GGUF=FILE, on that file's IQ1_M expert tensors; no model is loaded.
build/tests/gguf-decode: tests/gpu/quantization.c src/engine/gpu.c src/engine/util.c src/engine/myds4.h src/kernels/formats.metal src/kernels/kernels.metal src/kernels/gguf_grids.metal src/kernels/gguf.metal
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ tests/gpu/quantization.c src/engine/gpu.c src/engine/util.c src/engine/memory.c $(LDLIBS)

gguf-decode-check: build/tests/gguf-decode
	./build/tests/gguf-decode $(GGUF)

# ---- Checks that load the model ----

build/tests/prefill-prune-model: tests/model/prefill_prune_model.c build/libmyds4.a $(MODEL_HEADERS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ tests/model/prefill_prune_model.c build/libmyds4.a $(LDLIBS)

build/tests/draft-rows: tests/model/draft_rows.c build/libmyds4.a $(MODEL_HEADERS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ tests/model/draft_rows.c build/libmyds4.a $(LDLIBS)

build/tests/prefix-fork: tests/model/prefix_fork.c build/libmyds4.a $(MODEL_HEADERS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ tests/model/prefix_fork.c build/libmyds4.a $(LDLIBS)

build/tests/verification-model: tests/model/verification_model.c build/libmyds4.a $(MODEL_HEADERS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ tests/model/verification_model.c build/libmyds4.a $(LDLIBS)

# CED numerical and prefill/DSpark timing checks.
build/tests/ced: tests/model/ced.c build/libmyds4.a $(MODEL_HEADERS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ tests/model/ced.c build/libmyds4.a $(LDLIBS)

build/tests/dspark-state: tests/model/dspark_state.c build/libmyds4.a $(MODEL_HEADERS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ tests/model/dspark_state.c build/libmyds4.a $(LDLIBS)

build/tests/prefill-state: tests/model/prefill_state.c build/libmyds4.a $(MODEL_HEADERS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ tests/model/prefill_state.c build/libmyds4.a $(LDLIBS)

build/tests/runahead-model: tests/model/runahead_model.c build/libmyds4.a $(MODEL_HEADERS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ tests/model/runahead_model.c build/libmyds4.a $(LDLIBS)

# Decoding B sequences together: step time, GPU busy and idle, reads per row.
build/benchmarks/decode: benchmarks/decode.c build/libmyds4.a $(MODEL_HEADERS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ benchmarks/decode.c build/libmyds4.a $(LDLIBS)

build/benchmarks/products: benchmarks/products.c build/libmyds4.a $(MODEL_HEADERS)
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -o $@ benchmarks/products.c build/libmyds4.a $(LDLIBS)

# ---- Diagnostics ----

# Ready MXFP4 stages from different layers in one experimental dispatch.
# Reads 24 experts, no complete model; does not measure decoder throughput.
build/tests/moe-tasks: tests/gpu/expert_tasks.c src/model/deepseek/expert_tasks.h src/kernels/expert_tasks.metal src/kernels/formats.metal src/kernels/kernels.metal src/engine/io.c src/engine/expert_store.h tests/unit/cache_layout.h src/engine/gpu.c src/engine/util.c src/engine/myds4.h
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ tests/gpu/expert_tasks.c src/engine/io.c src/engine/gpu.c src/engine/util.c src/engine/memory.c $(LDLIBS)

clean:
	rm -rf build

.PHONY: clean server server-check test activation-format-check context-lookup-check workspace-check state-fork-check verification-schedule-check sampling-check shutdown-check cache-ready-check draft-cache-check prefetch-metrics-check prefetch-queue-check prefetch-admission-check prefetch-demote-check prefetch-refine-race-check cache-policy-check memory-available-check cache-memory-check memory-budget-check weights-stop-check weights-read-check gpu-clock-check pass-eviction-check gguf-decode-check
