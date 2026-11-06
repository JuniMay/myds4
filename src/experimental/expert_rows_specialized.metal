// Separate pipelines for the existing one-, two-, and three-row arithmetic.
// A descriptor reaches only the pipeline matching its count.
#define EXPERT_ROWS_PIPE(N)                                                                               \
kernel void moe_up_rows_##N(constant ExpertRows *groups [[buffer(0)]], device const float *x [[buffer(1)]], \
                            device float *h [[buffer(2)]], uint2 tg [[threadgroup_position_in_grid]],     \
                            uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
  threadgroup float block[N * 32];                                                                        \
  moe_up_rows_n<N>(groups[tg.y], x, h, tg, sg, lane, block);                                                \
}                                                                                                        \
kernel void moe_down_rows_##N(constant ExpertRows *groups [[buffer(0)]], device const float *h [[buffer(1)]], \
                              device float *parts [[buffer(2)]], uint2 tg [[threadgroup_position_in_grid]], \
                              uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
  moe_down_rows_n<N>(groups[tg.y], h, parts, tg, sg, lane);                                                 \
}
EXPERT_ROWS_PIPE(1)
EXPERT_ROWS_PIPE(2)
EXPERT_ROWS_PIPE(3)
#undef EXPERT_ROWS_PIPE
