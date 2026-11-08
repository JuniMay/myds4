// Recursive least-squares fitting for the online expert predictor.
#pragma once
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define OP_K 6
#define OP_MAX_ROWS 16
#define OP_COV_LIMIT 1000000.0f
typedef struct {
  int e, f, r, active;
  float *a, *b, *p, *bias;
  uint64_t updates, covariance_resets, rejected_updates, fallback_rows;
} OpPair;
typedef struct {
  float *features, *phi, *keys, *scores, *vector, *error, *next_b, *next_p;
  int *ids;
} OpWork;
void op_die(const char *s);

void *op_allocate(size_t n);

double op_now(void);

uint64_t op_random64(uint64_t *s);

double op_uniform(uint64_t *s);

float op_gaussian(uint64_t *s);

bool op_finite_array(const float *p, size_t n);

void op_covariance_reset(OpPair *p);

OpPair op_create_pair(int e, int r, uint64_t seed);

void op_free_pair(OpPair *p);

OpWork op_create_work(int e, int r);

void op_free_work(OpWork *w);

float op_key(float v, float bias);

double op_key_double(double v, double bias);

void op_top6(const float *scores, int e, int *ids);

bool op_valid_ids(const int *ids, int e, int m);

void op_normalize(const float *x, float *out, int n);

int op_fallback(OpPair *p, const int *native, int m, int *out);

// Forward never mutates learned parameters. Native IDs are copied verbatim
// for B=0 and on invalid auxiliary input; invalid fallback IDs are refused.
int op_forward(OpPair *p, OpWork *w, const float *gp, const float *gl, int m, const int *native,
                             int *out);


// A caller must retain phi until all labels in its pass are available, then
// update in row order. Candidate parameters are validated before publication.
typedef struct { bool valid, nonzero; } OpWeightsCheck;

OpWeightsCheck op_update_weights(const float *restrict b, const float *restrict error,
                                               const float *restrict vector, float *restrict next,
                                               int rows, int cols, double inv_den);


int op_update(OpPair *p, OpWork *w, const float *phi, const float *target);
