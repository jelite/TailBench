# plot_cluster_distribution.py
import torch
import matplotlib.pyplot as plt
import os

CLUSTER_FILE = "gemma2_clustered_200.pt"
OUT_PATH = "cluster_distribution.png"

def plot_cluster_distribution(cluster_file=CLUSTER_FILE, out_path=OUT_PATH):
    meta = torch.load(cluster_file, map_location="cpu")

    hard_labels = meta["hard_labels"]      # [vocab]
    hard_labels = hard_labels.view(-1)

    num_clusters = hard_labels.max().item() + 1
    counts = torch.bincount(hard_labels, minlength=num_clusters)

    plt.figure(figsize=(12, 4))
    plt.bar(range(num_clusters), counts.tolist())
    plt.xlabel("Cluster ID")
    plt.ylabel("Token Count")
    plt.title("Cluster Size Distribution")
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path)
    plt.close()

    print(f"[DONE] save cluster distribution → {out_path}")


if __name__ == "__main__":
    plot_cluster_distribution()
