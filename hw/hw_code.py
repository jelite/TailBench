import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np

# ✅ 모델 및 토크나이저 로드
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
CACHE_DIR = "/model_cache"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False, cache_dir=CACHE_DIR)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
    device_map="cpu",   # CPU에 로드 (GPU 부담 방지)
    cache_dir=CACHE_DIR
)

# ✅ 토큰 임베딩 추출
embeddings = model.get_input_embeddings().weight.detach().cpu().numpy()  # [vocab_size, hidden_dim]
vocab = list(tokenizer.get_vocab().keys())

print(f"Vocab size: {len(vocab)}, Embedding dim: {embeddings.shape[1]}")

# ✅ 차원 축소 (속도용)
pca = PCA(n_components=50, random_state=42)
reduced = pca.fit_transform(embeddings)

# ✅ KMeans 클러스터링
n_clusters = 50
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
labels = kmeans.fit_predict(reduced)

# ✅ 각 클러스터 중심 근처의 토큰 찾기
def top_tokens_in_cluster(cluster_id, top_k=10):
    cluster_indices = np.where(labels == cluster_id)[0]
    cluster_vecs = reduced[cluster_indices]
    center = kmeans.cluster_centers_[cluster_id]
    dists = np.linalg.norm(cluster_vecs - center, axis=1)
    nearest = cluster_indices[np.argsort(dists)[:top_k]]
    return [vocab[i] for i in nearest]

# ✅ 상위 5개 클러스터 출력
print("\n=== Top 5 Cluster Examples ===")
for c in range(5):
    tokens = top_tokens_in_cluster(c, top_k=10)
    print(f"\n[Cluster {c}]")
    print(", ".join(tokens))
