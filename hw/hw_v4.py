import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ======================================
# 0) PyTorch GPU KMeans (FP32)
# ======================================
def kmeans_torch_gpu(x_fp32, num_clusters, num_iters=40):
    N, D = x_fp32.shape
    indices = torch.randperm(N, device=x_fp32.device)[:num_clusters]
    centroids = x_fp32[indices].clone()

    for _ in range(num_iters):
        dist = torch.cdist(x_fp32, centroids)   # [N, C]
        labels = torch.argmin(dist, dim=1)      # [N]

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
# 1) Load FP16 model
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

W_half = model.lm_head.weight.detach().to("cuda")   # [V, D] FP16
vocab, dim = W_half.shape
W_fp32 = W_half.float().clone()                     # FP32 for clustering

print("LM-head:", W_half.shape)


# ======================================
# 2) KMeans (FP32, n_clusters = 200)
# ======================================
num_clusters = 200
print("Running FP32 GPU KMeans...")

centroids_fp32, hard_labels = kmeans_torch_gpu(W_fp32, num_clusters, num_iters=40)
hard_labels = hard_labels.int().contiguous()

print("KMeans done.\n")


# ======================================
# 3) centroid 정규화 + FP16 변환
# ======================================
centroids_norm = centroids_fp32 / (centroids_fp32.norm(dim=1, keepdim=True) + 1e-6)
centroids_half = centroids_norm.half().to("cuda")   # [C, D], FP16


# ======================================
# 4) Multi-assignment (토큰 → top_m 클러스터)
# ======================================
top_m = 2

with torch.no_grad():
    dist_full = torch.cdist(W_fp32, centroids_fp32)           # [V, C]
    multi_assign = torch.topk(-dist_full, k=top_m, dim=1).indices.int().to("cuda")


# ======================================
# === 방법 C: Residual Coarse 2-stage 적용 ===
# ======================================
def coarse_to_fine(hidden, top_cluster=32, final_k=10):
    # h: [D]
    h = hidden.half()
    h_norm = h / (h.norm() + 1e-9)   # FP16 normalized

    # 1) 기본 coarse score (centroid vs hidden)
    # centroids_half: [C, D], h_norm: [D] → coarse1: [C]
    coarse1 = centroids_half @ h_norm

    # 2) residual coarse (FP32로 계산)
    h32 = hidden.float()           # [D]
    cent32 = centroids_fp32        # [C, D]

    # 각 centroid c에 대해 proj[c] = (cent32[c]·h32) * cent32[c]
    # cent32 @ h32 → [C]
    proj = (cent32 @ h32).unsqueeze(1) * cent32   # [C, 1] * [C, D] → [C, D]

    # 각 c마다 residual[c] = h32 - proj[c]
    residual = h32.unsqueeze(0) - proj            # [1, D] - [C, D] → [C, D]

    # residual 정규화
    r_norm = residual / (residual.norm(dim=1, keepdim=True) + 1e-9)  # [C, D]

    # centroid와 residual 간의 cosine 유사도 근사
    cent32_norm = cent32 / (cent32.norm(dim=1, keepdim=True) + 1e-6)  # [C, D]
    coarse2 = (cent32_norm * r_norm).sum(dim=1)                       # [C]

    # 3) 1차 coarse + α * 2차 coarse
    alpha = 0.35
    coarse_final = coarse1.float() + alpha * coarse2                  # [C]

    # 최종 top clustering
    _, topC = torch.topk(coarse_final, top_cluster)                   # [top_cluster]

    # 후보 생성 ------------------------------------------
    mask_hard = torch.isin(hard_labels, topC)                         # [V]
    mask_soft = torch.isin(multi_assign, topC).any(dim=1)             # [V]
    cand_mask = mask_hard | mask_soft                                 # [V]

    cand = torch.nonzero(cand_mask).squeeze(1)                        # [#cand]

    # fine stage: 후보 토큰에 대해서만 LM-head
    logits_small = (W_half[cand] @ h)                                 # [#cand]
    vals, idx_local = torch.topk(logits_small, final_k)
    ids = cand[idx_local]

    return ids, vals


# ======================================
# 6) evaluate
# ======================================
def test_sentence(text, final_k=10, top_cluster=32):
    print("\n==============================")
    print("[입력]", text)
    print("==============================")

    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
        hidden = out.hidden_states[-1][0, -1, :].half()  # [D]

    # full LM-head
    logits_exact = W_half @ hidden
    top_exact = torch.topk(logits_exact, final_k).indices

    # coarse-to-fine
    top_coarse, _ = coarse_to_fine(hidden, top_cluster=top_cluster, final_k=final_k)

    recall = len(set(top_exact.tolist()) & set(top_coarse.tolist())) / final_k

    print("원본 top-k:", top_exact.tolist())
    print("근사  top-k:", top_coarse.tolist())
    print(f"top-{final_k} recall: {recall*100:.2f}%")
    print("원본 토큰:", tokenizer.batch_decode(top_exact.tolist()))
    print("근사 토큰:", tokenizer.batch_decode(top_coarse.tolist()))


# ======================================
# 7) 테스트
# ======================================
tests = [
    "When I was young, I used to",
    "Explain speculative decoding simply.",
    "Tell me the recipe for pancakes",
    "The meaning of life is",
    "I want to travel to",
]

for t in tests:
    test_sentence(t, final_k=40, top_cluster=32)
