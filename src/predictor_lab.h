// Optional live predictor experiments. The ordinary engine does not link this module.
#pragma once
#include "myds4.h"

int predictor_lab_enabled(void);
void predictor_lab_begin(int pos, int drafts, const uint16_t *seed);
void predictor_lab_stage(int stage, int rows, const float *x, const float *streams,
                         const float *mix, const float *router, const float *query,
                         const float *attention, const uint16_t *ring);
void predictor_lab_final(int rows, const float *streams, const float *head);
void predictor_lab_embeddings(int rows, const uint16_t *embeddings);
int predictor_lab_projection(void);
void predictor_lab_projection_input(int rows, const uint16_t *embeddings, const float *draft_norm, float *out);
void predictor_lab_baseline(int layer, int rows, const float *logits, const float *bias);
void predictor_lab_predict(int rows);
void predictor_lab_routes(int layer, int rows, const int *routes, const float *logits, const float *bias);
void predictor_lab_schedule(int completed_layer);
void predictor_lab_end(void);
void predictor_lab_accept(int rows);
void predictor_lab_report(FILE *out);
typedef struct { int slot, layer, expert; double frequency, age; uint64_t bytes; } LabVictim;
int predictor_lab_cache_policy(void);
int predictor_lab_suppress_prefetch(void);
int predictor_lab_victim_limit(void);
void predictor_lab_native_candidates(int layer, const int *ids, int n, int issuer);
// -2 abstains, -1 accepts a free slot, otherwise an eligible supplied victim.
int predictor_lab_choose(int layer, int expert, uint64_t bytes, int pending,
                         const LabVictim *victims, int count, uint64_t *action_id);
int cache_action_state(int layer, int expert, uint64_t action_id); // 0 not owned, 1 pending, 2 ready
void predictor_lab_read(uint64_t action_id, uint64_t bytes);
int predictor_lab_long_enabled(void);
int predictor_lab_long_active(void);
void predictor_lab_long_init(const float *const biases[LAYERS]);
void predictor_lab_long_forecast(int source, int rows, const float *gp4, const float *gp8, const float *current);
void predictor_lab_long_update(int layer);
