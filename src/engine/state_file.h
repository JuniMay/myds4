// Saved states, for the server's SSD KV store (server/src/disk.rs): a
// state's rows in parts, and snapshots, in files that the server names,
// opens, syncs and renames. engine.h's myds4_state_write and the calls after
// it run these on a model's family (model.h's state_regions,
// snapshot_parts and snapshot_alloc).
#pragma once
#include <CommonCrypto/CommonDigest.h>

#include "../model/model.h"

// One part of a saved state: each region's rows from those that `from`
// positions leave unchanged (region_stable) up to `to` (region_rows), from a
// state holding at least `to` positions. The file's bytes, or -1 (a write
// failed, or a second stop request ended it). With fd -1 nothing is written:
// the bytes it would take. The same for snapshot_file_write.
int64_t state_file_write(const ModelOps *ops, ModelState *s, int from, int to, int fd);
// Fills s with the rows before `positions` from a saved state's parts, the
// first written first: each part's rows replace the earlier parts' from its
// own start. 0, or -1 when a part is short, damaged or of another shape, or
// the parts leave rows out (s is then of no use).
int state_file_read(const ModelOps *ops, ModelState *s, int positions, const int *fds, int count);
// Snapshot x, or (x null) s's rolling rows as at pos, without logits. The
// file's bytes, or -1.
int64_t snapshot_file_write(const ModelOps *ops, ModelSnapshot *x, ModelState *s, int pos, int fd);
// A snapshot snapshot_file_write wrote; null when the file is short,
// damaged or of another shape.
ModelSnapshot *snapshot_file_read(const ModelOps *ops, int fd);

// What a checkpoint's weights are, for a saved state's identity: the
// config.json, .safetensors and .gguf files at any depth of dir (names
// starting with a dot left out) by name, size and first and last 64 KiB.
void identity_folder(CC_SHA256_CTX *h, const char *dir);
// The same for a GGUF file, or every file of its split set (NAME-00001-of-00005.gguf).
void identity_gguf_set(CC_SHA256_CTX *h, const char *path);

// CRC32C (Castagnoli), with the ARM crc32c instructions.
uint32_t crc32c(uint32_t crc, const void *data, size_t n);
