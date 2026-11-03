.DEFAULT_GOAL := build/myds4
CC = clang
CFLAGS = -std=c23 -O2 -Wall -Wextra -Wno-unused-parameter $(shell pkg-config --cflags yyjson libjpeg)
LDLIBS = -framework Metal -framework Foundation -framework IOKit -framework ImageIO -framework CoreGraphics -framework Accelerate -lobjc \
         $(shell pkg-config --libs yyjson libjpeg)
SRC = src/main.c src/model.c src/io.c src/gpu.c src/tok.c src/util.c src/chat.c src/serve.c src/vision.c src/tui.c src/online_predictor.c src/sample.c
MODEL_HEADERS = src/myds4.h src/online_predictor_math.h src/expert_batch.h

build/prefill-schedule-check: tools/prefill_schedule_check.c src/engine.c src/engine.h $(SRC) $(MODEL_HEADERS) src/kernels.metal src/gguf_grids.metal src/gguf.metal src/vision.metal data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -Isrc -o $@ tools/prefill_schedule_check.c $(filter-out src/main.c src/serve.c,$(SRC)) $(LDLIBS)

build/prefill-state-check: tools/prefill_state_check.c $(SRC) $(MODEL_HEADERS) src/kernels.metal src/gguf_grids.metal src/gguf.metal src/vision.metal data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -Isrc -o $@ tools/prefill_state_check.c $(filter-out src/main.c src/model.c,$(SRC)) $(LDLIBS)

build/prefill-prefix-check: tools/prefill_prefix_check.c $(SRC) $(MODEL_HEADERS) src/kernels.metal src/gguf_grids.metal src/gguf.metal src/vision.metal data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -Isrc -o $@ tools/prefill_prefix_check.c $(filter-out src/main.c src/model.c,$(SRC)) $(LDLIBS)

build/attention-candidate-check: tools/attention_candidate_check.c $(SRC) $(MODEL_HEADERS) src/kernels.metal src/gguf_grids.metal src/gguf.metal src/vision.metal data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -Isrc -o $@ tools/attention_candidate_check.c $(filter-out src/main.c src/model.c,$(SRC)) $(LDLIBS)

build/expert-batch-check: tools/expert_batch_check.c src/gpu.c src/util.c src/myds4.h src/expert_batch.h src/kernels.metal src/gguf_grids.metal src/gguf.metal
	@mkdir -p build
	$(CC) $(CFLAGS) -Isrc -o $@ tools/expert_batch_check.c src/gpu.c src/util.c $(LDLIBS)

build/myds4-expert-batch: $(SRC) $(MODEL_HEADERS) src/kernels.metal src/gguf_grids.metal src/gguf.metal src/vision.metal data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_EXPERT_BATCH -o $@ $(SRC) $(LDLIBS)

build/expert-batch-model-check: tools/expert_batch_model_check.c $(SRC) $(MODEL_HEADERS) src/kernels.metal src/gguf_grids.metal src/gguf.metal src/vision.metal data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -Isrc -DMYDS4_EXPERT_BATCH -o $@ tools/expert_batch_model_check.c $(filter-out src/main.c,$(SRC)) $(LDLIBS)

build/expert-batch-model-control: tools/expert_batch_model_check.c $(SRC) $(MODEL_HEADERS) src/kernels.metal src/gguf_grids.metal src/gguf.metal src/vision.metal data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -Isrc -o $@ tools/expert_batch_model_check.c $(filter-out src/main.c,$(SRC)) $(LDLIBS)

# Retained research build: explicit feature ablations and real prefetch trials.
build/myds4-predict-lab: $(SRC) src/predictor_lab.c src/predictor_lab.h $(MODEL_HEADERS) src/kernels.metal src/gguf_grids.metal src/gguf.metal src/vision.metal data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_PREDICT_LAB -o $@ $(SRC) src/predictor_lab.c $(LDLIBS)

build/predictor-lab-check: tools/predictor_lab_check.c src/predictor_lab.c src/predictor_lab.h $(MODEL_HEADERS) src/util.c
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ tools/predictor_lab_check.c src/util.c -framework Accelerate

build/cache-action-check: tools/cache_action_check.c src/io.c src/predictor_lab.h src/gpu.c src/util.c $(MODEL_HEADERS)
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ tools/cache_action_check.c src/gpu.c src/util.c $(LDLIBS)

build/myds4: $(SRC) $(MODEL_HEADERS) src/kernels.metal src/gguf_grids.metal src/gguf.metal src/vision.metal data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ $(SRC) $(LDLIBS)

# The engine behind src/engine.h, for the Rust server (server/): the sources
# of build/myds4 without main.c, serve.c and chat.c, and with engine.c.
LIB_SRC = src/model.c src/io.c src/gpu.c src/tok.c src/util.c src/vision.c src/tui.c src/online_predictor.c src/sample.c src/engine.c
LIB_OBJ = $(LIB_SRC:src/%.c=build/lib/%.o)

build/lib/%.o: src/%.c $(MODEL_HEADERS) src/engine.h src/kernels.metal src/gguf_grids.metal src/gguf.metal src/vision.metal data/engram.bin data/unicode.bin
	@mkdir -p build/lib
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

build/online-predictor-unit: tools/online_predictor_unit.c src/online_predictor.c src/online_predictor_math.h src/myds4.h
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ tools/online_predictor_unit.c -framework Accelerate

build/online-predictor-state-check: tools/online_predictor_state_check.c src/online_predictor.c src/online_predictor_math.h src/myds4.h
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ tools/online_predictor_state_check.c -framework Accelerate

build/online-predictor-check: tools/online_predictor_check.c src/online_predictor_math.h
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ tools/online_predictor_check.c -framework Accelerate

build/sampling-check: tools/sampling_check.c src/sample.c src/util.c src/myds4.h
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ tools/sampling_check.c src/sample.c src/util.c

sampling-check: build/sampling-check
	./build/sampling-check

build/gpu-clock-check: tools/gpu_clock_check.c src/gpu.c src/kernels.metal src/util.c src/myds4.h
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ tools/gpu_clock_check.c src/util.c $(LDLIBS)

gpu-clock-check: build/gpu-clock-check
	./build/gpu-clock-check

build/shutdown-check: tools/shutdown_check.c src/util.c src/myds4.h
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ tools/shutdown_check.c src/util.c

shutdown-check: build/shutdown-check
	./build/shutdown-check

# Diagnostic pipe capture only; its synchronous writes change inference timing.
build/myds4-predict-probe: $(SRC) tools/predict_probe.c $(MODEL_HEADERS) src/kernels.metal src/gguf_grids.metal src/gguf.metal src/vision.metal data/engram.bin data/unicode.bin
	@mkdir -p build
	$(CC) $(CFLAGS) -DMYDS4_PREDICT_PROBE -Isrc -o $@ $(SRC) tools/predict_probe.c $(LDLIBS)

predict-probe: build/myds4-predict-probe

# IQ1_M expert decoding on the GPU against llama.cpp's, on random blocks and,
# with GGUF=FILE, on that file's IQ1_M expert tensors; no model is loaded.
build/gguf-decode-check: tools/gguf_decode_check.c src/gpu.c src/util.c src/myds4.h src/kernels.metal src/gguf_grids.metal src/gguf.metal
	@mkdir -p build
	$(CC) $(CFLAGS) -Isrc -o $@ tools/gguf_decode_check.c src/gpu.c src/util.c $(LDLIBS)

gguf-decode-check: build/gguf-decode-check
	./build/gguf-decode-check $(GGUF)

# CPU-only cache checks; the fixture creates no model buffers or read files.
build/cache-ready-check: tools/cache_ready_check.c src/io.c src/gpu.c src/util.c src/myds4.h
	@mkdir -p build
	$(CC) $(CFLAGS) -Isrc -o $@ tools/cache_ready_check.c src/gpu.c src/util.c $(LDLIBS)

cache-ready-check: build/cache-ready-check
	./build/cache-ready-check

build/draft-cache-check: tools/draft_cache_check.c src/io.c src/gpu.c src/util.c src/myds4.h
	@mkdir -p build
	$(CC) $(CFLAGS) -Isrc -o $@ tools/draft_cache_check.c src/gpu.c src/util.c $(LDLIBS)

draft-cache-check: build/draft-cache-check
	./build/draft-cache-check

build/prefetch-metrics-check: tools/prefetch_metrics_check.c src/io.c src/gpu.c src/util.c src/myds4.h
	@mkdir -p build
	$(CC) $(CFLAGS) -Isrc -o $@ tools/prefetch_metrics_check.c src/gpu.c src/util.c $(LDLIBS)

prefetch-metrics-check: build/prefetch-metrics-check
	./build/prefetch-metrics-check

build/prefetch-queue-check: tools/prefetch_queue_check.c src/io.c src/gpu.c src/util.c src/myds4.h
	@mkdir -p build
	$(CC) $(CFLAGS) -Isrc -o $@ tools/prefetch_queue_check.c src/gpu.c src/util.c $(LDLIBS)

prefetch-queue-check: build/prefetch-queue-check
	./build/prefetch-queue-check

build/cache-policy-check: tools/cache_policy_check.c src/io.c src/gpu.c src/util.c src/myds4.h
	@mkdir -p build
	$(CC) $(CFLAGS) -Isrc -o $@ tools/cache_policy_check.c src/gpu.c src/util.c $(LDLIBS)

cache-policy-check: build/cache-policy-check
	./build/cache-policy-check

build/memory-available-check: tools/memory_available_check.c src/io.c src/gpu.c src/util.c src/myds4.h
	@mkdir -p build
	$(CC) $(CFLAGS) -Isrc -o $@ tools/memory_available_check.c src/gpu.c src/util.c $(LDLIBS)

memory-available-check: build/memory-available-check
	./build/memory-available-check

clean:
	rm -rf build

.PHONY: clean server predict-probe gguf-decode-check cache-ready-check draft-cache-check prefetch-metrics-check prefetch-queue-check cache-policy-check memory-available-check shutdown-check sampling-check gpu-clock-check
