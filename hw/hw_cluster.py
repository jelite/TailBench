
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

# ======================================
# 0) PyTorch GPU KMeans 구현
# ======================================
def kmeans_torch_gpu(x, num_clusters, num_iters=40):
    """
    x: [N, D] float32 GPU tensor (lm-head weight)
    num_clusters: number of clusters
    """
    N, D = x.shape

    # 초기 centroid (랜덤)
    indices = torch.randperm(N, device=x.device)[:num_clusters]
    centroids = x[indices]  # [C, D]

    for it in range(num_iters):
        # 거리 계산 [N, C]
        dist = torch.cdist(x, centroids)  

        # 각 벡터가 속하는 cluster id
        labels = torch.argmin(dist, dim=1)

        # 새로운 centroid 계산
        new_centroids = []
        for c in range(num_clusters):
            pts = x[labels == c]
            if pts.shape[0] == 0:
                new_centroids.append(centroids[c])
            else:
                new_centroids.append(pts.mean(dim=0))
        new_centroids = torch.stack(new_centroids)

        # 수렴 체크
        if torch.allclose(new_centroids, centroids, atol=1e-4):
            break

        centroids = new_centroids

    return centroids, labels


# ======================================
# 1) 모델 로드 (GPU)
# ======================================
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

CACHE_DIR = "/model_cache"


model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
    device_map="cuda",
    cache_dir=CACHE_DIR
)

# LM-head weight
W_lm_gpu = model.lm_head.weight.detach().to("cuda").half()  # float32 for KMeans
W_lm = W_lm_gpu.cpu().numpy()
vocab, dim = W_lm.shape
print("LM-head:", W_lm_gpu.shape)


# ======================================
# 2) GPU KMeans (lm-head 기반 클러스터)
# ======================================
n_clusters = 200
print("Running GPU KMeans...")

centroids_gpu, labels_gpu = kmeans_torch_gpu(W_lm_gpu, num_clusters=n_clusters, num_iters=40)

labels_np = labels_gpu.cpu().numpy()
labels = labels_gpu.to(torch.int32)  # [vocab]

print("KMeans done. Centroids:", centroids_gpu.shape)


# ======================================
# 3) centroid L2 정규화
# ======================================
cluster_centroids = centroids_gpu / (centroids_gpu.norm(dim=1, keepdim=True) + 1e-6)
cluster_centroids = cluster_centroids.to("cuda").half()


# ======================================
# 4) coarse → fine search 함수
# ======================================
def lm_head_exact(h):
    return W_lm_gpu @ h

def lm_head_coarse_to_fine(h, top_cluster=32, final_k=10):
    h_norm = h / (h.norm() + 1e-6)

    # coarse scoring
    coarse = cluster_centroids @ h_norm

    # coarse top clusters 선택
    _, top_c = torch.topk(coarse, top_cluster)

    # 해당 클러스터 속하는 토큰
    mask = torch.isin(labels, top_c)
    cand = torch.nonzero(mask).squeeze(1)

    # fine search
    logits_small = W_lm_gpu[cand] @ h

    # 최종 top-k 추출
    vals, idx_local = torch.topk(logits_small, final_k)
    top_ids = cand[idx_local]

    return top_ids, vals


# ======================================
# 5) 실제 hidden state 얻기
# ======================================
text = "Explain speculative decoding simply."
inputs = tokenizer(text, return_tensors="pt").to("cuda")

with torch.no_grad():
    out = model(**inputs, output_hidden_states=True)
    hidden = out.hidden_states[-1][0, -1, :]  # 마지막 토큰 hidden

hidden = hidden.to(torch.float16)


# ======================================
# 6) 비교 실험
# ======================================
k = 10

# 원본 full LM-head
logits_exact = lm_head_exact(hidden)
top_exact = torch.topk(logits_exact, k).indices

# coarse-to-fine 결과
top_coarse, _ = lm_head_coarse_to_fine(hidden, top_cluster=32, final_k=k)

recall = len(set(top_exact.tolist()) & set(top_coarse.tolist())) / k

print("\n=== 결과 ===")
print("원본 top-k:", top_exact.tolist())
print("근사  top-k:", top_coarse.tolist())
print(f"top-{k} recall: {recall*100:.2f}%")

top_exact_tokens = tokenizer.batch_decode(top_exact.tolist(), skip_special_tokens=True)
top_coarse_tokens = tokenizer.batch_decode(top_coarse.tolist(), skip_special_tokens=True)

print("원본 top-k 단어:", top_exact_tokens)
print("근사  top-k 단어:", top_coarse_tokens)