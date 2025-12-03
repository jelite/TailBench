# save_clustered_lm_head.py
import os
import math
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "google/gemma-2-2b-it"
CACHE_DIR = "/model_cache"


# -------------------------------
# Balanced KMeans (capacity-first)
# -------------------------------
def kmeans_torch_gpu(x_fp32, num_clusters, num_iters=20, pref_k=8):
    N, D = x_fp32.shape
    device = x_fp32.device

    idx = torch.randperm(N, device=device)[:num_clusters]
    centroids = x_fp32[idx].clone()

    for _ in range(num_iters):
        dist = torch.cdist(x_fp32, centroids)
        pref_k_eff = min(pref_k, num_clusters)
        pref = torch.topk(dist, k=pref_k_eff, dim=1, largest=False).indices

        target = math.ceil(N / num_clusters)
        pref_cpu = pref.cpu().numpy()
        labels = -1 * torch.ones(N, dtype=torch.int64, device=device)
        cluster_sizes = [0] * num_clusters

        order = torch.randperm(N, device=device).tolist()
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
                c = int(pref_cpu[i, 0])
                labels[i] = c
                cluster_sizes[c] += 1

        new_centroids = []
        for c in range(num_clusters):
            pts = x_fp32[labels == c]
            if pts.size(0) == 0:
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


def save_bin(t: torch.Tensor, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    t = t.contiguous().cpu()
    np.array(t).tofile(path)


# -------------------------------
# 메인 실행: 클러스터 생성 + 저장 (+ CUDA용 .bin 세트)
# -------------------------------
def save_clustered_lm_head(
    pt_output_path="gemma2_clustered_100.pt",
    num_clusters=100,
    topk_multi_assign=2,
    bin_dir="cluster_bin",
    hidden_prompt="When I was young, I used to",
):
    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        dtype=torch.float16,
        device_map="cuda",
        cache_dir=CACHE_DIR,
    )

    W_half = model.lm_head.weight.detach().to("cuda")   # [V, D] FP16
    W_fp32 = W_half.float().clone()                     # FP32 for k-means
    vocab, dim = W_half.shape

    # --- KMeans ---
    centroids_fp32, hard_labels = kmeans_torch_gpu(W_fp32, num_clusters)

    # --- multi-assign ---
    dist_full = torch.cdist(W_fp32, centroids_fp32)
    multi_assign = torch.topk(-dist_full, k=topk_multi_assign, dim=1).indices.int()

    # --- centroids normalized (FP32 → normalize → FP16) ---
    centroids_norm = centroids_fp32 / (centroids_fp32.norm(dim=1, keepdim=True) + 1e-6)
    centroids_norm_fp16 = centroids_norm.half()

    # ----- PyTorch용 메타 PT 파일 -----
    meta = {
        "model_name": MODEL,
        "vocab_size": vocab,
        "hidden_dim": dim,
        "num_clusters": num_clusters,
        "topk_multi_assign": topk_multi_assign,
        "centroids_fp32": centroids_fp32.cpu(),
        "hard_labels": hard_labels.cpu(),
        "multi_assign": multi_assign.cpu(),
        "lm_head_weight": W_half.cpu(),
    }
    os.makedirs(os.path.dirname(pt_output_path) or ".", exist_ok=True)
    torch.save(meta, pt_output_path)
    print(f"[DONE] clustered lm-head saved → {pt_output_path}")

    # ----- CUDA runner용 .bin + meta.txt -----
    os.makedirs(bin_dir, exist_ok=True)

    # centroids (fp32 / norm_fp16)
    save_bin(centroids_fp32,      os.path.join(bin_dir, "centroids_fp32.bin"))
    save_bin(centroids_norm_fp16, os.path.join(bin_dir, "centroids_norm_fp16.bin"))

    # lm_head weight (fp16)
    save_bin(W_half, os.path.join(bin_dir, "lm_head_fp16.bin"))

    # labels / multi_assign (int32)
    save_bin(hard_labels.int(),           os.path.join(bin_dir, "hard_labels.bin"))
    save_bin(multi_assign.int(),          os.path.join(bin_dir, "multi_assign.bin"))

    # hidden 벡터 하나 (runner에서 동일한 프롬프트 사용해야 Python과 정확히 비교 가능)
    if hidden_prompt is not None:
        inputs = tokenizer(hidden_prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
            hidden = out.hidden_states[-1][0, -1, :].half()
        save_bin(hidden, os.path.join(bin_dir, "hidden_fp16.bin"))
        print(f"[INFO] hidden_fp16.bin saved for prompt: {hidden_prompt!r}")

    # meta.txt (C, D, V, assign_k)
    meta_txt = os.path.join(bin_dir, "meta.txt")
    with open(meta_txt, "w") as f:
        f.write(f"num_centroids={num_clusters}\n")
        f.write(f"dim={dim}\n")
        f.write(f"vocab={vocab}\n")
        f.write(f"assign_k={topk_multi_assign}\n")
    print(f"[DONE] bin files + meta.txt saved to: {bin_dir}")


if __name__ == "__main__":
    num_clusters = 100
    save_clustered_lm_head(
        pt_output_path=f"gemma2_clustered_{num_clusters}.pt",
        num_clusters=num_clusters,
        topk_multi_assign=2,
        bin_dir="cluster_bin",
        hidden_prompt="When I was young, I used to",
    )
