import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ======================================
# 0) PyTorch GPU KMeans (FP32)
# ======================================
def kmeans_torch_gpu(x_fp32, num_clusters, num_iters=40):
    """
    x_fp32: [N, D] float32 GPU tensor
    """
    N, D = x_fp32.shape

    # 초기 centroid (random)
    indices = torch.randperm(N, device=x_fp32.device)[:num_clusters]
    centroids = x_fp32[indices].clone()  # FP32

    for _ in range(num_iters):
        # 거리 계산
        dist = torch.cdist(x_fp32, centroids)  # FP32

        # labels
        labels = torch.argmin(dist, dim=1)

        # 새 centroid
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
# 1) Load model (FP16 inference)
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

# LM-head weight (FP16 for inference)
W_half = model.lm_head.weight.detach().to("cuda")  # FP16
vocab, dim = W_half.shape
print("LM-head:", W_half.shape)

# FP32 copy for KMeans
W_fp32 = model.lm_head.weight.detach().float().to("cuda")  # FP32


# ======================================
# 2) GPU KMeans (FP32)
# ======================================
n_clusters = 600  # 200 → 600 (정확도 대폭 증가)
print("Running FP32 GPU KMeans...")

centroids_fp32, labels = kmeans_torch_gpu(W_fp32, n_clusters, num_iters=40)
labels = labels.to(torch.int32)

print("KMeans done. Centroids:", centroids_fp32.shape)


# ======================================
# 3) centroid 정규화 후 FP16 변환 (inference용)
# ======================================
centroids_norm = centroids_fp32 / (centroids_fp32.norm(dim=1, keepdim=True) + 1e-6)
centroids_half = centroids_norm.half().to("cuda")  # FP16 (모델 dtype과 동일)

# ======================================
# 4) coarse → fine (FP16)
# ======================================
def coarse_to_fine(hidden, top_cluster=20, final_k=10):
    h = hidden.half()  # FP16
    h_norm = h / (h.norm() + 1e-9)

    # coarse (FP16 @ FP16)
    coarse = centroids_half @ h_norm
    _, topC = torch.topk(coarse, top_cluster)

    # 클러스터 내 모든 토큰 후보 모으기
    mask = torch.isin(labels, topC)
    cand = torch.nonzero(mask).squeeze(1)

    # fine search (FP16 @ FP16)
    logits_small = W_half[cand] @ h

    vals, idx_local = torch.topk(logits_small, final_k)
    ids = cand[idx_local]

    return ids, vals


# ======================================
# 5) 평가
# ======================================
text = "When I was young, I used to"
inputs = tokenizer(text, return_tensors="pt").to("cuda")

with torch.no_grad():
    out = model(**inputs, output_hidden_states=True)
    hidden = out.hidden_states[-1][0, -1, :].half()  # FP16


# Full LM-head (baseline)
logits_exact = W_half @ hidden
topk_exact = torch.topk(logits_exact, 10).indices

# Coarse-to-fine
topk_coarse, _ = coarse_to_fine(hidden, top_cluster=20, final_k=10)

# Recall 계산
recall = len(set(topk_exact.tolist()) & set(topk_coarse.tolist())) / 10

# 출력
print("\n=== 결과 ===")
print("원본 top-k:", topk_exact.tolist())
print("근사  top-k:", topk_coarse.tolist())
print(f"top-10 recall: {recall*100:.2f}%")

print("원본 토큰:", tokenizer.batch_decode(topk_exact.tolist()))
print("근사 토큰:", tokenizer.batch_decode(topk_coarse.tolist()))
