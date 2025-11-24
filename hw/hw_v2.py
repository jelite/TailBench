import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

NUM_TOP_K = 40

# ============================================================
# 0) PyTorch GPU KMeans (FP32)
# ============================================================
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


# ============================================================
# 1) Load FP16 model
# ============================================================
MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=False)

CACHE_DIR = "/model_cache"

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    dtype=torch.float16,
    device_map="cuda",
    cache_dir=CACHE_DIR
)

W_half = model.lm_head.weight.detach().to("cuda")
vocab, dim = W_half.shape
W_fp32 = W_half.float().clone()

print("LM-head:", W_half.shape)


# ============================================================
# 2) KMeans (FP32)
# ============================================================
num_clusters = 1500   # 안정적 recall 상승 폭 큼
print("Running FP32 GPU KMeans...")

centroids_fp32, labels = kmeans_torch_gpu(W_fp32, num_clusters, num_iters=40)
labels = labels.int().contiguous()

print("KMeans done.\n")


# ============================================================
# 3) centroid normalization + FP16 변환
# ============================================================
centroids_norm = centroids_fp32 / (centroids_fp32.norm(dim=1, keepdim=True) + 1e-6)
centroids_half = centroids_norm.half().to("cuda")


# ============================================================
# 4) coarse → fine (안정 버전)
# ============================================================
def coarse_to_fine(hidden, top_cluster=200, final_k=NUM_TOP_K):
    h = hidden.half()
    h_norm = h / (h.norm() + 1e-9)

    # coarse step
    coarse = centroids_half @ h_norm
    _, topC = torch.topk(coarse, top_cluster)

    # 후보 토큰만 모음
    mask = torch.isin(labels, topC)
    cand = torch.nonzero(mask).squeeze(1)

    logits_small = W_half[cand] @ h
    vals, idx_local = torch.topk(logits_small, final_k)
    ids = cand[idx_local]

    return ids, vals


# ============================================================
# 5) 테스트 함수
# ============================================================
def test_sentence(text):
    print("\n==============================")
    print("[입력]", text)
    print("==============================")

    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
        hidden = out.hidden_states[-1][0, -1, :].half()

    logits_exact = W_half @ hidden
    top_exact = torch.topk(logits_exact, NUM_TOP_K).indices

    top_coarse, _ = coarse_to_fine(hidden)

    recall = len(set(top_exact.tolist()) & set(top_coarse.tolist())) / NUM_TOP_K

    print("원본 top-k:", top_exact.tolist())
    print("근사  top-k:", top_coarse.tolist())
    print(f"top-10 recall: {recall*100:.2f}%")
    print("원본 토큰:", tokenizer.batch_decode(top_exact.tolist()))
    print("근사 토큰:", tokenizer.batch_decode(top_coarse.tolist()))


# ============================================================
# 6) 샘플 테스트
# ============================================================
test_sentence("When I was young, I used to")
test_sentence("Explain speculative decoding simply.")
test_sentence("Tell me the recipe for pancakes")
test_sentence("The meaning of life is")
test_sentence("I want to travel to")
