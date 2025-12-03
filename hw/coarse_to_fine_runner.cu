// coarse_to_fine_runner.cu
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <cassert>

#include "coarse_to_fine_kernel.cuh"

using namespace std;

// --- RAW .bin 로딩 유틸 ---
template<typename T>
std::vector<T> load_bin(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        std::cerr << "ERR: cannot open " << path << std::endl;
        std::exit(1);
    }
    f.seekg(0, std::ios::end);
    size_t bytes = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<T> buf(bytes / sizeof(T));
    f.read(reinterpret_cast<char*>(buf.data()), bytes);
    f.close();
    return buf;
}
std::vector<__half> h_cent_norm = load_bin<__half>("cluster_bin/centroids_norm_fp16.bin");


// --- meta.txt 읽기 ---
struct Meta {
    int C, D, V, K;
};

Meta load_meta(const std::string& path) {
    Meta m{};
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
        auto eq = line.find("=");
        auto key = line.substr(0, eq);
        auto val = std::stoi(line.substr(eq + 1));
        if (key == "num_centroids") m.C = val;
        if (key == "dim")           m.D = val;
        if (key == "vocab")         m.V = val;
        if (key == "assign_k")      m.K = val;
    }
    return m;
}

int main() {

    // ① meta.txt
    Meta m = load_meta("cluster_bin/meta.txt");
    int C = m.C, D = m.D, V = m.V, Kassign = m.K;

    std::cout << "[META] C=" << C << " D=" << D << " V=" << V << " K=" << Kassign << std::endl;

    // ② .bin 로드
    std::vector<float>  h_cent32  = load_bin<float>("cluster_bin/centroids_fp32.bin");
    std::vector<__half> h_W       = load_bin<__half>("cluster_bin/lm_head_fp16.bin");
    std::vector<int>    h_hard    = load_bin<int>("cluster_bin/hard_labels.bin");
    std::vector<int>    h_ma      = load_bin<int>("cluster_bin/multi_assign.bin");
    std::vector<__half> h_hidden  = load_bin<__half>("cluster_bin/hidden_fp16.bin");

    if ((int)h_hidden.size() != D) {
        std::cerr << "hidden_fp16.bin size mismatch: got " << h_hidden.size()
                  << ", expected " << D << std::endl;
        return 1;
    }

    // ③ GPU 메모리 할당
    float*  d_cent32;
    __half* d_W;
    int*    d_hard;
    int*    d_ma;
    __half* d_hidden;
    int*    d_out_ids;
    float*  d_out_vals;
    __half* d_cent_norm;   // FP16 normalized centroids (이미 Python에서 norm해서 저장했다면 이거 대신 사용)

    // 여기서는 centroids_fp32만 쓴다고 가정하고,
    // cent_norm은 Python 쪽에서 "centroids_norm_fp16.bin" 만들어서 불러오는 게 더 좋다.
    // 지금 코드에서는 cent_norm은 안 쓰더라도 포인터는 필요하니 더미로 할당.
    cudaMalloc(&d_cent32,  sizeof(float)  * C * D);
    cudaMalloc(&d_W,       sizeof(__half) * V * D);
    cudaMalloc(&d_hard,    sizeof(int)    * V);
    cudaMalloc(&d_ma,      sizeof(int)    * V * Kassign);
    cudaMalloc(&d_hidden,  sizeof(__half) * D);
    cudaMalloc(&d_out_ids, sizeof(int)    * 40);
    cudaMalloc(&d_out_vals,sizeof(float)  * 40);
    cudaMalloc(&d_cent_norm, sizeof(__half) * C * D);
    cudaMemcpy(d_cent_norm,
        h_cent_norm.data(),
        sizeof(__half) * C * D,
        cudaMemcpyHostToDevice);
    cudaMemcpy(d_cent32,  h_cent32.data(), sizeof(float)  * C * D,       cudaMemcpyHostToDevice);
    cudaMemcpy(d_W,       h_W.data(),      sizeof(__half) * V * D,       cudaMemcpyHostToDevice);
    cudaMemcpy(d_hard,    h_hard.data(),   sizeof(int)    * V,           cudaMemcpyHostToDevice);
    cudaMemcpy(d_ma,      h_ma.data(),     sizeof(int)    * V * Kassign, cudaMemcpyHostToDevice);
    cudaMemcpy(d_hidden,  h_hidden.data(), sizeof(__half) * D,           cudaMemcpyHostToDevice);
    // d_cent_norm은 필요하다면 centroids_norm_fp16.bin에서 로드해서 memcpy 하면 됨.
    // 여기선 그냥 0으로 초기화

    // ④ fused_coarse_to_fine 실행 + 타이밍 측정
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int top_cluster = 32;
    int final_k     = 40;

    cudaEventRecord(start);
    fused_coarse_to_fine(
        d_cent_norm,
        d_cent32,
        d_W,
        d_hard,
        d_ma,
        d_hidden,
        C, D, V,
        top_cluster,
        final_k,
        Kassign,
        d_out_ids,
        d_out_vals
    );
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    // 결과 가져오기
    std::vector<int>   out_ids(40);
    std::vector<float> out_vals(40);
    cudaMemcpy(out_ids.data(),  d_out_ids,  sizeof(int)   * 40, cudaMemcpyDeviceToHost);
    cudaMemcpy(out_vals.data(), d_out_vals, sizeof(float) * 40, cudaMemcpyDeviceToHost);

    // ⑤ 출력
    std::cout << "\nFused coarse-to-fine GPU time: " << ms << " ms" << std::endl;
    std::cout << "Top-40 tokens:\n";
    for (int i = 0; i < 40; ++i) {
        std::cout << "[" << i << "] id=" << out_ids[i]
                  << " score=" << out_vals[i] << std::endl;
    }

    // clean up
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_cent32);
    cudaFree(d_W);
    cudaFree(d_hard);
    cudaFree(d_ma);
    cudaFree(d_hidden);
    cudaFree(d_out_ids);
    cudaFree(d_out_vals);
    cudaFree(d_cent_norm);

    return 0;
}
