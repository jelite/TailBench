#include <cuda.h>
#include <cuda_runtime.h>

extern "C" {

__global__
void matvec_fp16_kernel(
    const half* __restrict__ mat,   // [C, D]
    const half* __restrict__ vec,   // [D]
    float* __restrict__ out,        // [C]
    int C, int D
){
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;

    float acc = 0.0f;
    for(int d=0; d<D; d++){
        acc += __half2float(mat[c*D + d]) * __half2float(vec[d]);
    }
    out[c] = acc;
}

__global__
void proj_scalar_kernel(
    const float* __restrict__ centroids,  // [C, D]
    const float* __restrict__ h32,        // [D]
    float* __restrict__ out,              // [C]
    int C, int D
){
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;

    float acc = 0.0f;
    for(int d=0; d<D; d++){
        acc += centroids[c*D + d] * h32[d];
    }
    out[c] = acc;
}

__global__
void proj_vec_kernel(
    const float* __restrict__ centroids,  // [C, D]
    const float* __restrict__ proj,       // [C]
    float* __restrict__ out,              // [D]
    int C, int D
){
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= D) return;

    float acc = 0.0f;
    for(int c=0; c<C; c++){
        acc += proj[c] * centroids[c*D + d];
    }
    out[d] = acc / C;
}

__global__
void residual_norm_kernel(
    const float* __restrict__ h32,      // [D]
    const float* __restrict__ proj_vec, // [D]
    float* __restrict__ out,            // [D]
    int D
){
    int d = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float norm_shared;

    if (d == 0) norm_shared = 0.0f;
    __syncthreads();

    float val = 0.0f;
    if (d < D) {
        val = h32[d] - proj_vec[d];
        atomicAdd(&norm_shared, val * val);
    }

    __syncthreads();

    if (d < D) {
        float denom = sqrtf(norm_shared) + 1e-9f;
        out[d] = val / denom;
    }
}

__global__
void matvec_fp32_kernel(
    const float* __restrict__ mat,   // [C, D]
    const float* __restrict__ vec,   // [D]
    float* __restrict__ out,         // [C]
    int C, int D
){
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;

    float acc = 0.0f;
    for(int d=0; d<D; d++){
        acc += mat[c*D + d] * vec[d];
    }
    out[c] = acc;
}

} // extern "C"
