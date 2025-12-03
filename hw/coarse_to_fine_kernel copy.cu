// coarse_to_fine_kernel.cu  — hidden 전처리 + coarse full fuse + fine multi-block top-k (cluster_in 사전 계산)
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include "coarse_to_fine_kernel.cuh"

#ifndef MAX_D
#define MAX_D 2304
#endif
#ifndef MAX_C
#define MAX_C 512
#endif
#ifndef MAX_TOPC
#define MAX_TOPC 256
#endif
#ifndef MAX_K
#define MAX_K 128
#endif


// =========================
// 1) coarse stage (single block)
// =========================
__global__ void coarse_fused_parallel_kernel(
    const __half* __restrict__ cent_norm,  // [C,D]
    const float*  __restrict__ cent_f32,   // [C,D]
    const __half* __restrict__ hidden,     // [D]
    float* __restrict__ coarse_buf,        // [C] out
    int C, int D
) {
    int tid      = threadIdx.x;
    int nthreads = blockDim.x;

    __shared__ float h[MAX_D];
    __shared__ float h_norm[MAX_D];
    __shared__ float r_norm[MAX_D];

    // ① h vector + norm
    float partial = 0.f;
    for (int i = tid; i < D; i += nthreads) {
        float v = __half2float(hidden[i]);
        h[i] = v;
        partial += v * v;
    }

    __shared__ float total_norm;
    if (tid == 0) total_norm = 0.f;
    __syncthreads();
    atomicAdd(&total_norm, partial);
    __syncthreads();

    if (tid == 0) total_norm = rsqrtf(total_norm + 1e-9f);
    __syncthreads();

    for (int i = tid; i < D; i += nthreads)
        h_norm[i] = h[i] * total_norm;
    __syncthreads();

    // ② proj_vec = 1/C Σ ( (cent @ h) * cent )
    __shared__ float proj_vec[MAX_D];
    if (tid == 0) {
        for (int i = 0; i < D; i++) proj_vec[i] = 0.f;
    }
    __syncthreads();

    for (int c = tid; c < C; c += nthreads) {
        const float* cent = cent_f32 + (size_t)c * D;
        float ps = 0.f;
        #pragma unroll 4
        for (int k = 0; k < D; k++)
            ps += cent[k] * h[k];

        #pragma unroll 4
        for (int k = 0; k < D; k++)
            atomicAdd(&proj_vec[k], (ps * cent[k]) / (float)C);
    }
    __syncthreads();

    // ③ residual r_norm = normalize(h - proj_vec)
    float partial_r = 0.f;
    for (int i = tid; i < D; i += nthreads) {
        float r = h[i] - proj_vec[i];
        r_norm[i] = r;
        partial_r += r * r;
    }

    __shared__ float total_res;
    if (tid == 0) total_res = 0.f;
    __syncthreads();
    atomicAdd(&total_res, partial_r);
    __syncthreads();

    if (tid == 0) total_res = rsqrtf(total_res + 1e-9f);
    __syncthreads();

    for (int i = tid; i < D; i += nthreads)
        r_norm[i] *= total_res;
    __syncthreads();

    // ④ coarse_final[c] = coarse1 + alpha * coarse2
    const float alpha = 0.35f;
    for (int c = tid; c < C; c += nthreads) {
        const __half* cn = cent_norm + (size_t)c * D;
        const float*  cf = cent_f32  + (size_t)c * D;
        float acc1 = 0.f;
        float acc2 = 0.f;
        #pragma unroll 4
        for (int k = 0; k < D; k++) {
            acc1 += __half2float(cn[k]) * h_norm[k];
            acc2 += cf[k] * r_norm[k];
        }
        coarse_buf[c] = acc1 + alpha * acc2;
    }
}


// =========================
// 1.5) coarse_buf → cluster_in (topC를 한 번만 계산)
// =========================
__global__ void build_cluster_in_kernel(
    const float* __restrict__ coarse_buf,  // [C]
    int C,
    int top_cluster,
    int* __restrict__ cluster_in_out      // [C]
) {
    int tid = threadIdx.x;

    __shared__ int   topC[MAX_TOPC];
    __shared__ float topC_val[MAX_TOPC];

    if (tid == 0) {
        // topC 초기화
        for (int i = 0; i < top_cluster; i++) {
            topC[i]     = -1;
            topC_val[i] = -1e30f;
        }
        // coarse_buf에서 top_cluster 선택 (이전 fine 커널과 동일한 로직)
        for (int c = 0; c < C; c++) {
            float v = coarse_buf[c];
            if (v > topC_val[top_cluster - 1]) {
                int pos = top_cluster - 1;
                while (pos > 0 && v > topC_val[pos - 1]) {
                    topC_val[pos] = topC_val[pos - 1];
                    topC[pos]     = topC[pos - 1];
                    pos--;
                }
                topC_val[pos] = v;
                topC[pos]     = c;
            }
        }
        // cluster_in_out 초기화
        for (int c = 0; c < C; c++) cluster_in_out[c] = 0;
        // topC에 해당하는 클러스터에 1 표시
        for (int i = 0; i < top_cluster; i++) {
            int cid = topC[i];
            if (cid >= 0 && cid < C) cluster_in_out[cid] = 1;
        }
    }
}


// =========================
// 2) fine stage (multi-block)
//    각 블록이 vocab의 chunk를 담당하고, 블록별 local top-k를 계산
// =========================
__global__ void fine_block_topk_kernel(
    const int*    __restrict__ cluster_in,   // [C] 0/1
    const __half* __restrict__ W_half,       // [V, D]
    const __half* __restrict__ hidden_half,  // [D]
    const int*    __restrict__ hard_labels,  // [V]
    const int*    __restrict__ multi_assign, // [V, Kassign]
    int C,
    int D,
    int V,
    int top_cluster,   // 사용 안 해도 시그니처 유지
    int final_k,
    int Kassign,
    int tokens_per_block,
    int*   __restrict__ block_out_ids,       // [num_blocks, final_k]
    float* __restrict__ block_out_vals       // [num_blocks, final_k]
) {
    int tid      = threadIdx.x;
    int warp_id  = tid / 32;
    int lane     = tid % 32;
    int nwarps   = blockDim.x / 32;

    // shared hidden (FP16 그대로, dot은 half2로)
    __shared__ __half sh_hidden[MAX_D];
    for (int i = tid; i < D; i += blockDim.x)
        sh_hidden[i] = hidden_half[i];
    __syncthreads();

    // 블록 로컬 top-k
    __shared__ float final_v[MAX_K];
    __shared__ int   final_i[MAX_K];
    if (tid == 0) {
        for (int j = 0; j < final_k; j++) {
            final_v[j] = -1e30f;
            final_i[j] = -1;
        }
    }
    __syncthreads();

    int block_start = blockIdx.x * tokens_per_block;
    int block_end   = block_start + tokens_per_block;
    if (block_end > V) block_end = V;

    for (int v = block_start + warp_id; v < block_end; v += nwarps) {
        // coarse 기반 후보 필터링
        bool in_cand = false;
        int hl = hard_labels[v];
        if (hl >= 0 && hl < C && cluster_in[hl]) {
            in_cand = true;
        } else {
            const int* ma = multi_assign + (size_t)v * Kassign;
            for (int kk = 0; kk < Kassign; kk++) {
                int cid = ma[kk];
                if (cid >= 0 && cid < C && cluster_in[cid]) {
                    in_cand = true;
                    break;
                }
            }
        }
        if (!in_cand) continue;

        // W[v] @ hidden_half (half2 + warp reduction)
        const __half* wv = W_half + (size_t)v * D;
        const half2*  w2 = reinterpret_cast<const half2*>(wv);
        const half2*  h2 = reinterpret_cast<const half2*>(sh_hidden);

        int   D2  = D / 2;
        float sum = 0.f;
        #pragma unroll 4
        for (int k = lane; k < D2; k += 32) {
            half2 a = w2[k];
            half2 b = h2[k];
            float2 af2 = __half22float2(a);
            float2 bf2 = __half22float2(b);
            sum += af2.x * bf2.x + af2.y * bf2.y;
        }

        // warp reduction
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            sum += __shfl_down_sync(0xffffffff, sum, off);

        if (lane == 0) {
            if (sum > final_v[final_k - 1]) {
                int pos = final_k - 1;
                while (pos > 0 && sum > final_v[pos - 1]) {
                    final_v[pos] = final_v[pos - 1];
                    final_i[pos] = final_i[pos - 1];
                    pos--;
                }
                final_v[pos] = sum;
                final_i[pos] = v;
            }
        }
    }
    __syncthreads();

    // 블록별 top-k를 전역 버퍼에 기록
    if (tid == 0) {
        int base = blockIdx.x * final_k;
        for (int j = 0; j < final_k; j++) {
            block_out_ids[base + j]  = final_i[j];
            block_out_vals[base + j] = final_v[j];
        }
    }
}


// =========================
// 3) block별 top-k → 최종 top-k (single block)
// =========================
__global__ void final_reduce_topk_kernel(
    const int*   __restrict__ block_ids,   // [num_blocks * final_k]
    const float* __restrict__ block_vals,  // [num_blocks * final_k]
    int num_blocks,
    int final_k,
    int*   __restrict__ out_ids,           // [final_k]
    float* __restrict__ out_vals           // [final_k]
) {
    int tid = threadIdx.x;

    __shared__ float final_v[MAX_K];
    __shared__ int   final_i[MAX_K];

    if (tid == 0) {
        for (int j = 0; j < final_k; j++) {
            final_v[j] = -1e30f;
            final_i[j] = -1;
        }
    }
    __syncthreads();

    int total = num_blocks * final_k;
    for (int idx = tid; idx < total; idx += blockDim.x) {
        float v  = block_vals[idx];
        int   id = block_ids[idx];

        if (v > final_v[final_k - 1]) {
            int pos = final_k - 1;
            while (pos > 0 && v > final_v[pos - 1]) {
                final_v[pos] = final_v[pos - 1];
                final_i[pos] = final_i[pos - 1];
                pos--;
            }
            final_v[pos] = v;
            final_i[pos] = id;
        }
    }
    __syncthreads();

    if (tid == 0) {
        for (int j = 0; j < final_k; j++) {
            out_ids[j]  = final_i[j];
            out_vals[j] = final_v[j];
        }
    }
}


// =========================
// 4) host wrapper
// =========================
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
) {
    // 1) coarse stage
    float* d_coarse = nullptr;
    cudaMalloc(&d_coarse, sizeof(float) * C);

    coarse_fused_parallel_kernel<<<1, 128>>>(
        cent_norm,
        cent_f32,
        hidden,
        d_coarse,
        C, D
    );
    cudaDeviceSynchronize();

    // 1.5) cluster_in (topC) 한 번만 계산
    int* d_cluster_in = nullptr;
    cudaMalloc(&d_cluster_in, sizeof(int) * C);
    build_cluster_in_kernel<<<1, 128>>>(
        d_coarse,
        C,
        top_cluster,
        d_cluster_in
    );
    cudaDeviceSynchronize();

    // 2) fine stage: multi-block local top-k
    const int tokens_per_block = 1024;  // 필요시 조절
    int num_blocks = (V + tokens_per_block - 1) / tokens_per_block;

    int*   d_block_ids  = nullptr;
    float* d_block_vals = nullptr;
    cudaMalloc(&d_block_ids,  sizeof(int)   * num_blocks * final_k);
    cudaMalloc(&d_block_vals, sizeof(float) * num_blocks * final_k);

    dim3 fine_grid(num_blocks);
    dim3 fine_block(128);  // 4 warps

    fine_block_topk_kernel<<<fine_grid, fine_block>>>(
        d_cluster_in,
        W,
        hidden,
        hard_labels,
        multi_assign,
        C, D, V,
        top_cluster,
        final_k,
        Kassign,
        tokens_per_block,
        d_block_ids,
        d_block_vals
    );
    cudaDeviceSynchronize();

    // 3) 최종 reduce
    final_reduce_topk_kernel<<<1, 128>>>(
        d_block_ids,
        d_block_vals,
        num_blocks,
        final_k,
        out_ids,
        out_vals
    );
    cudaDeviceSynchronize();

    cudaFree(d_coarse);
    cudaFree(d_cluster_in);
    cudaFree(d_block_ids);
    cudaFree(d_block_vals);
}
