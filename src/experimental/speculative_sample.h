#ifndef MYDS4_SPECULATIVE_SAMPLE_H
#define MYDS4_SPECULATIVE_SAMPLE_H

#include "myds4.h"

// Modified rejection sampling preserves the target distribution, not the
// ordinary sampler's mapping from a seed to a particular output sequence.
void speculative_distribution(const float *logits, int count, const Sampler *sampler, double *probability);
int speculative_draw(const double *probability, int count, double uniform);
double speculative_residual(const double *target, const double *draft, int count, double *out);
int speculative_verify(const float *logits, int count, const int *drafts, int ndraft,
                       const double *proposal, int eos, Sampler *sampler, int *accepted);

#endif
