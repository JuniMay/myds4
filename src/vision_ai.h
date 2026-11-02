// Public Core AI adapter for the vision backbone. Requires macOS 27 and Swift 6.3+.
#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct VisionAI VisionAI;

// Loads one fixed-shape function. A null entrypoint selects "main".
// Core AI prefers the ANE but may execute operations on the CPU or GPU.
// A successful load does not establish which device executes the function.
// Returns null on failure. Errors are UTF-8, truncated and null-terminated when
// error is non-null and error_capacity is positive. Success clears error.
VisionAI *vision_ai_create(const char *path, const char *entrypoint, int32_t patch_count, char *error,
                           size_t error_capacity);

// Runs synchronously. Calls on the same handle are serialized.
// patches: patch_count rows of 640 FP32 values, already preprocessed by myds4.
// rope: patch_count rows of 64 FP32 values, 32 cosines followed by 32 sines.
// features: space for patch_count rows of 1024 FP32 output values.
// The adapter converts patches to FP16. It uses FP16 or FP32 for both RoPE
// inputs, as specified by the graph, and transposes inputs to BC1S layout.
// It converts FP16 features back to row-major FP32. It does not run the Aligner
// or round the output to BF16. Returns 1 on success, 0 on failure.
// Inputs and output must remain valid until this call returns. Output is left
// untouched on failure. Do not call from a Swift cooperative-executor task.
int32_t vision_ai_run(VisionAI *handle, const float *patches, const float *rope, float *features, char *error,
                      size_t error_capacity);

// Release only after all calls using this handle have returned. Null is allowed.
void vision_ai_destroy(VisionAI *handle);

#ifdef __cplusplus
}
#endif
