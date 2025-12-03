#클러스터 균일화 
import torch
import statistics
from transformers import AutoModelForCausalLM, AutoTokenizer


# ======================================
# 0) PyTorch GPU KMeans (FP32)
# ======================================
# K means 클러스터링을 수행한다. 
def kmeans_torch_gpu(x_fp32, num_clusters, num_iters=40):
    N, D = x_fp32.shape
    device = x_fp32.device

    # 초기 센트로이드
    indices = torch.randperm(N, device=device)[:num_clusters]
    centroids = x_fp32[indices].clone()

    # 일반 K-Means로 먼저 수렴
    for _ in range(num_iters):
        dist = torch.cdist(x_fp32, centroids)          # [N, C]
        labels = torch.argmin(dist, dim=1)             # [N]

        new_centroids = []
        for c in range(num_clusters):
            pts = x_fp32[labels == c]
            if pts.size(0) == 0:
                new_centroids.append(centroids[c])
            else:
                new_centroids.append(pts.mean(dim=0))
        new_centroids = torch.stack(new_centroids)

        if torch.allclose(new_centroids, centroids, atol=1e-4):
            centroids = new_centroids
            break

        centroids = new_centroids

    # ============================
    # 균등 클러스터 사이즈로 재할당
    # ============================
    # 최종 센트로이드 기준 거리
    dist = torch.cdist(x_fp32, centroids)  # [N, C]

    # 클러스터 용량 설정: floor 또는 ceil로 거의 균등하게
    base = N // num_clusters
    rem = N % num_clusters
    capacity = torch.full((num_clusters,), base, device=device, dtype=torch.long)
    if rem > 0:
        capacity[:rem] += 1  # 앞의 일부 클러스터만 +1

    # 각 포인트의 선호 클러스터 리스트 (가까운 순, 상위 K개만 사용)
    pref_k = min(num_clusters, 8)
    _, pref = torch.topk(-dist, k=pref_k, dim=1)  # [N, pref_k]

    labels = torch.full((N,), -1, device=device, dtype=torch.long)
    cluster_sizes = torch.zeros(num_clusters, device=device, dtype=torch.long)

    # 라운드별로 선호 순서에 따라 capacity 안에서 할당
    for r in range(pref_k):
        unassigned = torch.nonzero(labels == -1).squeeze(1)
        if unassigned.numel() == 0:
            break

        choices = pref[unassigned, r]  # [M]
        for c in range(num_clusters):
            mask_c = (choices == c)
            if not mask_c.any():
                continue

            pts_c = unassigned[mask_c]
            free = capacity[c] - cluster_sizes[c]
            if free <= 0:
                continue

            if pts_c.numel() <= free:
                labels[pts_c] = c
                cluster_sizes[c] += pts_c.numel()
            else:
                # capacity만큼 무작위 선택
                perm = torch.randperm(pts_c.numel(), device=device)[:free]
                chosen = pts_c[perm]
                labels[chosen] = c
                cluster_sizes[c] += chosen.numel()

    # 아직 할당 안 된 포인트가 있으면, 남은 capacity 안에서 가장 가까운 클러스터에 할당
    unassigned = torch.nonzero(labels == -1).squeeze(1)
    if unassigned.numel() > 0:
        avail = torch.nonzero(cluster_sizes < capacity).squeeze(1)
        if avail.numel() == 0:
            # 이론상 발생 X, 방어 코드
            labels[unassigned] = 0
        else:
            dist_sub = dist[unassigned][:, avail]           # [M, A]
            best_rel = torch.argmin(dist_sub, dim=1)        # [M]
            best_clusters = avail[best_rel]                 # [M]
            labels[unassigned] = best_clusters
            cluster_sizes += torch.bincount(best_clusters, minlength=num_clusters)

    # 균등화된 labels 기준으로 센트로이드 재계산
    new_centroids = []
    for c in range(num_clusters):
        pts = x_fp32[labels == c]
        if pts.size(0) == 0:
            new_centroids.append(centroids[c])
        else:
            new_centroids.append(pts.mean(dim=0))
    centroids = torch.stack(new_centroids)

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

        # projection scalars
        proj_scalar = centroids @ hidden          # [C]

        # projection vectors averaged
        proj_vec = (proj_scalar.unsqueeze(1) * centroids).mean(dim=0)  # [D]

        # residual vector
        residual = hidden - proj_vec               # [D]

        # normalize residual
        r_norm = residual / (residual.norm() + 1e-9)

        # coarse1 + alpha * coarse2 를 한 번에 계산
        alpha = 0.35
        cat_vec = torch.cat((h_norm, r_norm), dim=0)                        # [2D]
        cat_centroids = torch.cat((centroids_norm, alpha * centroids), 1)   # [C, 2D]
        coarse_final = cat_centroids @ cat_vec                              # [C]

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
        top_cluster=10,
        n_clusters=100
    )
