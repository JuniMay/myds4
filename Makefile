.DEFAULT_GOAL := build/myds4
CC = clang
CFLAGS = -std=c23 -O2 -Wall -Wextra -Wno-unused-parameter -Isrc/engine $(shell pkg-config --cflags yyjson libjpeg)
LDLIBS = -framework Metal -framework Foundation -framework IOKit -framework ImageIO -framework CoreGraphics -framework Accelerate -lobjc \
         $(shell pkg-config --libs yyjson libjpeg)
SRC = src/cli/main.c src/engine/model.c src/engine/io.c src/engine/gpu.c src/engine/tok.c src/engine/util.c src/engine/vision.c src/experimental/online_predictor.c src/engine/sample.c
MODEL_HEADERS = src/engine/myds4.h src/experimental/online_predictor_math.h src/engine/expert_batch.h
KERNELS = src/kernels/kernels.metal src/kernels/gguf_grids.metal src/kernels/gguf.metal src/kernels/vision.metal

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

build/expert-batch-check: tools/checks/expert_batch_check.c src/engine/gpu.c src/engine/util.c src/engine/myds4.h src/engine/expert_batch.h src/kernels/kernels.metal src/kernels/gguf_grids.metal src/kernels/gguf.metal
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ tools/checks/expert_batch_check.c src/engine/gpu.c src/engine/util.c $(LDLIBS)

# Experimental exact sharing of resident expert weights across verification rows.
build/myds4-expert-batch: $(SRC) $(MODEL_HEADERS) $(KERNELS) data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_EXPERT_BATCH -o $@ $(SRC) $(LDLIBS)

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
LIB_SRC = src/engine/model.c src/engine/io.c src/engine/gpu.c src/engine/tok.c src/engine/util.c src/engine/vision.c src/experimental/online_predictor.c src/engine/sample.c src/engine/engine.c
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

clean:
	rm -rf build

.PHONY: clean server server-check predict-probe gguf-decode-check cache-ready-check draft-cache-check prefetch-metrics-check prefetch-queue-check cache-policy-check memory-available-check shutdown-check sampling-check gpu-clock-check
