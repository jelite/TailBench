import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ======================================
# 0) PyTorch GPU KMeans (FP32)
# ======================================
def kmeans_torch_gpu(x_fp32, num_clusters, num_iters=40):
    N, D = x_fp32.shape
    indices = torch.randperm(N, device=x_fp32.device)[:num_clusters]
    centroids = x_fp32[indices].clone()  # FP32

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
# 1) Load model
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

W_half = model.lm_head.weight.detach().to("cuda")  # FP16
W_fp32 = model.lm_head.weight.detach().float().to("cuda")  # FP32 copy

vocab, dim = W_half.shape
print("LM-head:", W_half.shape)


# ======================================
# 2) KMeans clustering
# ======================================
n_clusters = 600
print("Running FP32 GPU KMeans...")

centroids_fp32, labels = kmeans_torch_gpu(W_fp32, n_clusters, num_iters=40)
labels = labels.to(torch.int32)

print("KMeans done. Centroids:", centroids_fp32.shape)


# ======================================
# 3) Normalize centroids → FP16
# ======================================
centroids_norm = centroids_fp32 / (centroids_fp32.norm(dim=1, keepdim=True) + 1e-6)
centroids_half = centroids_norm.half().to("cuda")


# ======================================
# 4) coarse → fine
# ======================================
def coarse_to_fine(hidden, top_cluster=20, final_k=10):
    h = hidden.half()
    h_norm = h / (h.norm() + 1e-9)

    coarse = centroids_half @ h_norm
    _, topC = torch.topk(coarse, top_cluster)

    mask = torch.isin(labels, topC)
    cand = torch.nonzero(mask).squeeze(1)

    logits_small = W_half[cand] @ h
    vals, idx_local = torch.topk(logits_small, final_k)
    ids = cand[idx_local]

    return ids, vals


# ======================================
# 5) 평가 함수 (단일 텍스트)
# ======================================
def evaluate(text, top_cluster=20, final_k=10):
    print("\n==============================")
    print(f"[입력] {text}")
    print("==============================")

    inputs = tokenizer(text, return_tensors="pt").to("cuda")

    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
        hidden = out.hidden_states[-1][0, -1, :].half()

    # Full LM-head
    logits_exact = W_half @ hidden
    topk_exact = torch.topk(logits_exact, final_k).indices

    # Coarse-to-fine
    topk_coarse, _ = coarse_to_fine(hidden, top_cluster=top_cluster, final_k=final_k)

    # Recall
    recall = len(set(topk_exact.tolist()) & set(topk_coarse.tolist())) / final_k

    print("원본 top-k:", topk_exact.tolist())
    print("근사  top-k:", topk_coarse.tolist())
    print(f"top-{final_k} recall: {recall*100:.2f}%")

    print("원본 토큰:", tokenizer.batch_decode(topk_exact.tolist()))
    print("근사 토큰:", tokenizer.batch_decode(topk_coarse.tolist()))


# ======================================
# 6) 여러 테스트 실행
# ======================================
tests = [
    "When I was young, I used to",
    "Explain speculative decoding simply.",
    "Tell me the recipe for pancakes",
    "The meaning of life is",
    "I want to travel to",
]

for t in tests:
    evaluate(t, top_cluster=20, final_k=10)
