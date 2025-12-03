import torch
import statistics
from transformers import AutoModelForCausalLM, AutoTokenizer


# ======================================
# 0) PyTorch GPU KMeans (FP32)
# ======================================
import torch
import math

def kmeans_torch_gpu(x_fp32, num_clusters, num_iters=20, pref_k=8):
    """
    큰 N에서 Hungarian(NxN) 대신, '가까운 클러스터 선호 리스트 + capacity' 기반
    균등 근사 Balanced K-Means.

    - 각 포인트는 거리 기준 상위 pref_k개의 클러스터 선호 리스트를 갖고,
      그 중 아직 capacity가 남은 클러스터에 할당.
    - target_size = ceil(N / num_clusters) 를 넘지 않도록 제한.
    """
    N, D = x_fp32.shape
    device = x_fp32.device

    # 초기 센트로이드: 랜덤 샘플
    indices = torch.randperm(N, device=device)[:num_clusters]
    centroids = x_fp32[indices].clone()

    for it in range(num_iters):
        # [N, C] 거리 (GPU)
        dist = torch.cdist(x_fp32, centroids)  # float32, [N, C]

        # 각 포인트별 선호 클러스터 리스트 (가까운 순)
        pref_k_eff = min(pref_k, num_clusters)
        pref = torch.topk(dist, k=pref_k_eff, dim=1, largest=False).indices  # [N, pref_k_eff]

        # capacity = ceil(N / C)
        target = int(math.ceil(N / num_clusters))

        # CPU에서 균등 근사 할당 (메모리 절약)
        pref_cpu = pref.cpu().numpy()
        labels = -1 * torch.ones(N, dtype=torch.int64)
        cluster_sizes = [0] * num_clusters

        order = torch.randperm(N).tolist()  # 랜덤 순서로 할당

        for i in order:
            assigned = False
            for r in range(pref_k_eff):
                c = int(pref_cpu[i, r])
                if cluster_sizes[c] < target:
                    labels[i] = c
                    cluster_sizes[c] += 1
                    assigned = True
                    break
            if not assigned:
                # 선호 리스트 안에서 모두 full이면 제일 가까운 클러스터에 넣되 overflow 허용
                c = int(pref_cpu[i, 0])
                labels[i] = c
                cluster_sizes[c] += 1

        labels = labels.to(device)

        # 새 센트로이드 계산
        new_centroids = []
        for c in range(num_clusters):
            pts = x_fp32[labels == c]
            if pts.size(0) == 0:
                # 완전히 빈 클러스터면 랜덤 재초기화
                ridx = torch.randint(0, N, (1,), device=device)
                new_centroids.append(x_fp32[ridx[0]])
            else:
                new_centroids.append(pts.mean(dim=0))
        new_centroids = torch.stack(new_centroids)

        if torch.allclose(new_centroids, centroids, atol=1e-4):
            centroids = new_centroids
            break

        centroids = new_centroids

    return centroids, labels






# ======================================
# Load model (shared)
# ======================================
MODEL = "google/gemma-2-2b-it"
tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=False)

CACHE_DIR = "/model_cache"

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    dtype=torch.float16,
    device_map="cuda",
    cache_dir=CACHE_DIR
)

W_half = model.lm_head.weight.detach().to("cuda")   # FP16
vocab, dim = W_half.shape
W_fp32 = W_half.float().clone()                     # FP32 (clustering 용)



# ======================================
# Residual coarse-to-fine 생성 함수
# (centroids_fp32, hard_labels, multi_assign을 입력으로 받도록 변경)
# ======================================
def build_coarse_to_fine(centroids_fp32, hard_labels, multi_assign):

    # precompute FP16 normalized centroids
    centroids_norm = centroids_fp32 / (centroids_fp32.norm(dim=1, keepdim=True) + 1e-6)
    centroids_half = centroids_norm.half().to("cuda")

    def coarse_to_fine(hidden, top_cluster=32, final_k=10):
        h = hidden.half()
        h_norm = h / (h.norm() + 1e-9)

        # Stage 1 coarse
        coarse1 = centroids_half @ h_norm    # [C]

        h32 = hidden.float()
        cent32 = centroids_fp32

        # projection scalars
        proj_scalar = cent32 @ h32              # [C]

        # projection vectors averaged
        proj_vec = (proj_scalar.unsqueeze(1) * cent32).mean(dim=0)  # [D]

        # residual vector
        residual = h32 - proj_vec               # [D]

        # normalize residual
        r_norm = residual / (residual.norm() + 1e-9)

        # residual coarse score
        coarse2 = cent32 @ r_norm               # [C]

        # combine
        alpha = 0.35
        coarse_final = coarse1.float() + alpha * coarse2

        _, topC = torch.topk(coarse_final, top_cluster)

        # candidate tokens
        mask_hard = torch.isin(hard_labels, topC)
        mask_soft = torch.isin(multi_assign, topC).any(dim=1)
        cand = torch.nonzero(mask_hard | mask_soft).squeeze(1)

        # fine stage
        logits_small = (W_half[cand] @ h)
        vals, idx_local = torch.topk(logits_small, final_k)
        return cand[idx_local], vals

    return coarse_to_fine

# KMeans 직후에 클러스터별 데이터 개수를 시각화 (bar plot 저장)
import matplotlib.pyplot as plt

def plot_cluster_sizes(hard_labels, save_path="cluster_sizes.png"):
    hard_labels_cpu = hard_labels.cpu()
    num_clusters = hard_labels_cpu.max().item() + 1

    counts = torch.bincount(hard_labels_cpu, minlength=num_clusters)

    plt.figure(figsize=(12, 4))
    plt.bar(range(num_clusters), counts.tolist())
    plt.xlabel("Cluster ID")
    plt.ylabel("Data Count")
    plt.title("Cluster Size Distribution")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


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

    # ground truth
    logits_exact = W_half @ hidden
    top_exact = torch.topk(logits_exact, final_k).indices
    exact_set = set(top_exact.tolist())

    recalls = []

    for i in range(runs):
        print(f"▶ Run {i+1}/{runs} ... clustering 시작")

        # 1) KMeans 다시 수행
        centroids_fp32, hard_labels = kmeans_torch_gpu(W_fp32, n_clusters)

        hard_labels = hard_labels.int().contiguous()

        # 2) multi-assign 다시 계산
        dist_full = torch.cdist(W_fp32, centroids_fp32)
        multi_assign = torch.topk(-dist_full, k=2, dim=1).indices.int().to("cuda")

        # 3) coarse-to-fine 함수 구성
        coarse_to_fine = build_coarse_to_fine(centroids_fp32, hard_labels, multi_assign)

        # 4) recall 측정
        top_coarse, _ = coarse_to_fine(hidden, top_cluster=top_cluster, final_k=final_k)
        recall = len(exact_set & set(top_coarse.tolist())) / final_k
        recalls.append(recall)

    # 통계 출력
    print("\n===== 결과 =====")
    print(f"runs={runs}")
    print(f"mean recall   : {100*statistics.mean(recalls):.2f}%")
    print(f"std recall    : {100*statistics.pstdev(recalls):.2f}%")
    print(f"min/max recall: {100*min(recalls):.2f}% / {100*max(recalls):.2f}%")

    print("\n원본 top-k:", top_exact.tolist())


tests = [
    "When I was young, I used to",
    "Explain speculative decoding simply.",
    "Tell me the recipe for pancakes",
    "The meaning of life is",
    "I want to travel to",
]

for i,t in enumerate(tests):


    test_sentence_multi_avg_full(
        t,
        runs=5,
        final_k=40,
        top_cluster=32,
        n_clusters=200
    )
    
    centroids_fp32, hard_labels = kmeans_torch_gpu(W_fp32, 200)
    plot_cluster_sizes(hard_labels, save_path=f"cluster_sizes_run{i+1}.png")