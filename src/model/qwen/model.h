// The Qwen3.8 (qwen3_5) text model on the GPU: Qwen3.8-27B with NVIDIA's
// NVFP4 weights (MLPs and head: E2M1 values, an E4M3 scale per 16 and a
// global scale; the attention and DeltaNet projections: E4M3 with one scale a
// matrix). See docs/design.md. Sequences ("states") hold the DeltaNet and
// convolution states and the attention layers' keys and values; a forward
// pass runs token rows of one or several states.
#ifndef QWEN_H
#define QWEN_H

#include <stdint.h>

#define QWEN_ROWS 8       // most rows per qwen_forward (the kernels' TM)
#define QWEN_PREFILL 1024  // most rows per qwen_prefill

// Loads the checkpoint directory's config.json and weights (all resident on
// the GPU). The MTP head is loaded only when drafts is nonzero.
int qwen_load(const char *dir, int drafts);
int qwen_vocab(void);
// With on 0, the engine's buffers (weights, states, snapshots) leave the
// GPU's residency set: they stay in memory, and other programs' GPU work can
// keep more resident. Any later call that runs the model or allocates puts
// them back first.
void qwen_reside(int on);
// Whether token ends a reply (generation_config.json's eos_token_id list).
int qwen_stop(int token);

typedef struct QwenState QwenState;
QwenState *qwen_state_new(void);
void qwen_state_free(QwenState *s);
// The positions the state has run (its next row's position).
int qwen_state_pos(const QwenState *s);
// The last target hidden supplied to the draft head, after final normalization.
const float *qwen_state_hidden(const QwenState *s);
// Bytes the state takes with room for `positions` positions.
uint64_t qwen_state_bytes(int positions);
// Makes room for `positions` positions, keeping the keys of those run.
void qwen_state_reserve(QwenState *s, int positions);
// dst takes src's recurrent state and the keys of src's positions
// [0, positions), and stands at `positions` (a snapshot taken there from src
// then gives the recurrent state that goes with them).
void qwen_state_copy(QwenState *dst, const QwenState *src, int positions);

typedef struct {
  QwenState *state;
  int32_t token;
  int save;  // 0, or the checkpoint buffer index plus one
} QwenRow;
// Runs n rows (1 to QWEN_ROWS), each at its state's next position; a state's
// rows must be consecutive. Returns the logits of the last `logits` rows
// (qwen_vocab() floats each), valid until the next call.
const float *qwen_forward(const QwenRow *rows, int n, int logits);
// A prompt pass: as qwen_forward, for 1 to QWEN_PREFILL rows, its matrices
// on the GPU's matrix units with inputs rounded to BF16. A row's result is
// the same however a prompt is split into passes, and differs from
// qwen_forward's by rounding.
const float *qwen_prefill(const QwenRow *rows, int n, int logits);
// Selects a verified-prefix checkpoint. Later keys remain to be overwritten.
void qwen_state_back(QwenState *s, int checkpoint);

// The MTP head (when the checkpoint has one: qwen_mtp_available). Row i
// takes `token` as the token after its state's last hidden (with `saved`,
// the hidden of the state's saved row), at that hidden's position, and
// leaves the head's keys there (in any order of rows). Prompt passes fill
// the head's keys of their positions. Returns the draft logits of the last
// `logits` rows (qwen_vocab() floats each), valid until the next call.
typedef struct {
  QwenState *state;
  int32_t token;
  int saved; // 0 for the current hidden, otherwise checkpoint index plus one
} QwenMtpRow;
int qwen_mtp_available(void);
const float *qwen_mtp(const QwenMtpRow *rows, int n, int logits);

// A snapshot: a state's recurrent part at its position, to return a state
// whose keys before that position are the snapshot's own to it.
typedef struct QwenSnap QwenSnap;
QwenSnap *qwen_snap_take(const QwenState *s);
void qwen_snap_restore(QwenState *s, const QwenSnap *snap);
int qwen_snap_pos(const QwenSnap *snap);
void qwen_snap_free(QwenSnap *snap);
uint64_t qwen_snap_bytes(void);

#endif
