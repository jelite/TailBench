#pragma once
#include <cuda_fp16.h>

void fused_coarse_to_fine(
    const __half* cent_norm,
    const float*  cent_f32,
    const __half* W,
    const int*    hard_labels,
    const int*    multi_assign,
    const __half* hidden,
    int C, int D, int V,
    int top_cluster,
    int final_k,
    int Kassign,
    int*   out_ids,
    float* out_vals
);
