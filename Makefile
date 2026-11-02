.DEFAULT_GOAL := build/myds4
CC = clang
CFLAGS = -std=c23 -O2 -Wall -Wextra -Wno-unused-parameter $(shell pkg-config --cflags yyjson libjpeg)
LDLIBS = -framework Metal -framework Foundation -framework IOKit -framework ImageIO -framework CoreGraphics -framework Accelerate -lobjc \
         $(shell pkg-config --libs yyjson libjpeg)
SRC = src/main.c src/model.c src/io.c src/gpu.c src/tok.c src/util.c src/chat.c src/serve.c src/vision.c src/tui.c src/online_predictor.c src/sample.c
MODEL_HEADERS = src/myds4.h src/online_predictor_math.h

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

# The optional public Core AI adapter is Swift because Core AI exposes a
# Swift API. Model execution and the inference engine remain in C.
build/libmyds4_vision_ai.dylib: src/vision_ai.swift src/vision_ai.h
	@mkdir -p build
	swiftc -O -swift-version 6 -emit-library -module-cache-path build/vision-ai-cache \
		-Xlinker -install_name -Xlinker @rpath/libmyds4_vision_ai.dylib -o $@ src/vision_ai.swift

build/myds4-ai: $(SRC) $(MODEL_HEADERS) src/vision_ai.h build/libmyds4_vision_ai.dylib src/kernels.metal src/gguf_grids.metal src/gguf.metal src/vision.metal data/engram.bin data/unicode.bin
	$(CC) $(CFLAGS) -DMYDS4_VISION_AI -o $@ $(SRC) $(LDLIBS) \
		-Lbuild -lmyds4_vision_ai -Wl,-rpath,@executable_path

vision-ai: build/myds4-ai

# Same C ABI and graph, with fixed public device preferences for diagnostics.
build/libmyds4_vision_ai_cpu.dylib: src/vision_ai.swift src/vision_ai.h
	@mkdir -p build
	swiftc -O -swift-version 6 -D VISION_AI_VALIDATE_CPU -emit-library \
		-module-cache-path build/vision-ai-cache -o $@ src/vision_ai.swift

build/libmyds4_vision_ai_gpu.dylib: src/vision_ai.swift src/vision_ai.h
	@mkdir -p build
	swiftc -O -swift-version 6 -D VISION_AI_VALIDATE_GPU -emit-library \
		-module-cache-path build/vision-ai-cache -o $@ src/vision_ai.swift

vision-ai-diagnostics: build/libmyds4_vision_ai_cpu.dylib build/libmyds4_vision_ai_gpu.dylib

clean:
	rm -rf build

.PHONY: clean predict-probe gguf-decode-check cache-ready-check draft-cache-check prefetch-metrics-check prefetch-queue-check cache-policy-check memory-available-check shutdown-check sampling-check gpu-clock-check vision-ai vision-ai-diagnostics
