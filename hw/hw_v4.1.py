import torch
import statistics
from transformers import AutoModelForCausalLM, AutoTokenizer


# ======================================
# 0) PyTorch GPU KMeans (FP32)
# ======================================
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
MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
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
    print("원본 토큰:", tokenizer.batch_decode(top_exact.tolist()))


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
