import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import faiss                   # ← GPU KMeans 핵심
import faiss.contrib.torch_utils

# ======================================
# 1) 모델 로드
# ======================================

CACHE_DIR = "/model_cache"

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
    device_map="cuda",
    cache_dir=CACHE_DIR
)

# lm-head weight
W_lm = model.lm_head.weight.detach().float().cpu().numpy()  # float32 CPU numpy
vocab, dim = W_lm.shape
print("W_lm:", W_lm.shape)

# ======================================
# 2) FAISS GPU KMeans 설정
# ======================================
n_clusters = 200
niter = 25        # Lloyd iteration 수
nredo = 3         # 재시도 횟수 (sklearn의 n_init에 해당)

print("Running FAISS GPU KMeans...")

# GPU 자원 사용 가능 여부 체크
gpu_res = faiss.StandardGpuResources()

# KMeans 객체 생성
kmeans = faiss.Kmeans(
    d=dim,
    k=n_clusters,
    niter=niter,
    nredo=nredo,
    verbose=True,
    gpu=True,               # GPU 사용
)

# ======================================
# 3) Clustering (GPU에서 빠르게 수행)
# ======================================
W_lm32 = W_lm.astype(np.float32)      # 반드시 float32여야 함
kmeans.train(W_lm32)                  # GPU에서 클러스터링

# 각 토큰의 cluster 라벨 얻기
distances, labels_np = kmeans.index.search(W_lm32, 1)
labels_np = labels_np.squeeze(1)      # shape: [vocab]

print("Cluster labels shape:", labels_np.shape)

# GPU 텐서로 변환
labels = torch.tensor(labels_np, device="cuda", dtype=torch.int32)

# ======================================
# 4) cluster centroid 불러오기
# ======================================
centroids = kmeans.centroids            # numpy [C, dim]
centroids = torch.tensor(centroids).to("cuda").float()
centroids = centroids / (centroids.norm(dim=1, keepdim=True) + 1e-6)

print("Centroids:", centroids.shape)

# ======================================
# 준비 완료
# ======================================
print("\n=== GPU KMeans clustering 완료! ===")
