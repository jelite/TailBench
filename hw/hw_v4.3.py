import torch
import statistics
from transformers import AutoModelForCausalLM, AutoTokenizer


# ======================================
# 0) PyTorch GPU KMeans (FP32)
# ======================================
# K means 클러스터링을 수행한다. 
def kmeans_torch_gpu(x_fp32, num_clusters, num_iters=40):
    N, D = x_fp32.shape
    indices = torch.randperm(N, device=x_fp32.device)[:num_clusters]
    centroids = x_fp32[indices].clone()

    for _ in range(num_iters):
        dist = torch.cdist(x_fp32, centroids)
        labels = torch.argmin(dist, dim=1)

        new_centroids = []
        for c in range(num_clusters):
            pts = x_fp32[labels == c]
            if pts.size(0) == 0:
                new_centroids.append(centroids[c])
            else:
                new_centroids.append(pts.mean(dim=0))
        new_centroids = torch.stack(new_centroids)

        if torch.allclose(new_centroids, centroids, atol=1e-4):
            break

        centroids = new_centroids

    return centroids, labels



# ======================================
# Load model (shared)
# ======================================
# MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
MODEL = "google/gemma-2-2b-it"
tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=False)

CACHE_DIR = "/model_cache"

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    dtype=torch.float32,
    device_map="cuda",
    cache_dir=CACHE_DIR
)

W_32 = model.lm_head.weight.detach().to("cuda")   # FP16
vocab, dim = W_32.shape
W_32 = W_32.clone()       
W_16 = W_32.half().clone()  


# ======================================
# Residual coarse-to-fine 생성 함수
# (centroids32, hard_labels, multi_assign을 입력으로 받도록 변경)
# ======================================
def build_coarse_to_fine(centroids, hard_labels, multi_assign):

    # precompute FP16 normalized centroids
    centroids_norm = centroids / (centroids.norm(dim=1, keepdim=True) + 1e-6)
    centroids_norm = centroids_norm.to("cuda")

    def coarse_to_fine(hidden, top_cluster=32, final_k=10):
        
        torch.cuda.nvtx.range_push("lm_head")
        hidden = hidden.half()
        h_norm = hidden / (hidden.norm() + 1e-9)
        
        # Stage 1 coarse
        coarse1 = centroids_norm @ h_norm    # [C]

        # projection scalars
        proj_scalar = centroids @ hidden          # [C]

        # projection vectors averaged
        proj_vec = (proj_scalar.unsqueeze(1) * centroids).mean(dim=0)  # [D]

        # residual vector
        residual = hidden - proj_vec               # [D]

        # normalize residual
        r_norm = residual / (residual.norm() + 1e-9)

        # residual coarse score
        coarse2 = centroids @ r_norm               # [C]

        # combine
        alpha = 0.35
        coarse_final = coarse1 + alpha * coarse2

        _, topC = torch.topk(coarse_final, top_cluster)

        # candidate tokens
        mask_hard = torch.isin(hard_labels, topC)
        mask_soft = torch.isin(multi_assign, topC).any(dim=1)
        cand = torch.nonzero(mask_hard | mask_soft).squeeze(1)

        # fine stage
        logits_small = (W_16[cand] @ hidden)
        vals, idx_local = torch.topk(logits_small, final_k)
        torch.cuda.nvtx.range_pop()
        return cand[idx_local], vals

    return coarse_to_fine

import time

# ======================================
# 전체 파이프라인을 runs번 반복 (KMeans 포함)
# ======================================
def test_sentence_multi_avg_full(text, runs=10, final_k=40, top_cluster=32, n_clusters=200):

    print("\n==============================")
    print("[입력]", text)
    print("==============================")

    # hidden (한 번만 계산)
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
        hidden = out.hidden_states[-1][0, -1, :].half()

    # ===== baseline: full softmax top-k =====
    # warm up
    for i in range(3):
        logits_exact = W_16 @ hidden
        logits_exact = W_16 @ hidden
        logits_exact = W_16 @ hidden
        
    # base line
    torch.cuda.synchronize()
    start = time.time()
    torch.cuda.nvtx.range_push("A")
    logits_exact = W_16 @ hidden
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    base_time = time.time() - start
    top_exact = torch.topk(logits_exact, final_k).indices
    
    exact_set = set(top_exact.tolist())

    # ===== KMeans / multi-assign은 단 한 번만! =====
    centroids, hard_labels = kmeans_torch_gpu(W_32, n_clusters)
    hard_labels = hard_labels.int().contiguous()
    
    dist_full = torch.cdist(W_32, centroids)
    multi_assign = torch.topk(-dist_full, k=2, dim=1).indices.int().to("cuda")
    centroids = centroids.half()
    coarse_to_fine = build_coarse_to_fine(centroids, hard_labels, multi_assign)

    # ===== 순수 coarse-to-fine 검색 시간만 반복 측정 =====
    recalls = []
    exp_times = []

    hidden = hidden.half()
    
    for i in range(runs):
        print(f"▶ Run {i+1}/{runs}")
        
        # warm up
        for k in range(3):
            top_coarse, _ = coarse_to_fine(hidden, top_cluster=top_cluster, final_k=final_k)
            top_coarse, _ = coarse_to_fine(hidden, top_cluster=top_cluster, final_k=final_k)
            top_coarse, _ = coarse_to_fine(hidden, top_cluster=top_cluster, final_k=final_k)
            
        start = time.time()
        top_coarse, _ = coarse_to_fine(hidden, top_cluster=top_cluster, final_k=final_k)
        torch.cuda.synchronize()
        exp_times.append(time.time() - start)

        recall = len(exact_set & set(top_coarse.tolist())) / final_k
        recalls.append(recall)

    # ===== 결과 출력 =====
    print("\n===== 결과 =====")
    print(f"mean recall   : {100*statistics.mean(recalls):.2f}%")
    print(f"std recall    : {100*statistics.pstdev(recalls):.2f}%")
    print(f"min/max recall: {100*min(recalls):.2f}% / {100*max(recalls):.2f}%")

    print(f"\nbaseline full softmax: {base_time*1000:.2f} ms")
    print("coarse times (ms):", [t*1000 for t in exp_times])

    print("\n원본 top-k:", top_exact.tolist())
    # print("원본 토큰:", tokenizer.batch_decode(top_exact.tolist()))
    # 방법 2: 2차원으로 차원 늘리기 (Top-K개, 길이 1인 시퀀스들)
    print("원본 토큰:", tokenizer.batch_decode(top_exact.unsqueeze(-1)))



tests = [
    "When I was young, I used to",
    "Explain speculative decoding simply.",
    "Tell me the recipe for pancakes",
    "The meaning of life is",
    "I want to travel to",
]

for t in tests:


    test_sentence_multi_avg_full(
        t,
        runs=5,
        final_k=40,
        top_cluster=32,
        n_clusters=200
    )
