# convert_for_cuda.py
import torch
import numpy as np
import os

CLUSTER_FILE = "gemma2_clustered_100.pt"
OUT_DIR = "cluster_bin"


def save_bin(arr: np.ndarray, path: str):
    arr.tofile(path)
    print(f"[SAVED] {path}  (shape={arr.shape}, dtype={arr.dtype})")


def main():
    meta = torch.load(CLUSTER_FILE, map_location="cpu")

    os.makedirs(OUT_DIR, exist_ok=True)

    # 1) FP32 centroids
    centroids = meta["centroids_fp32"].contiguous()
    save_bin(centroids.numpy().astype(np.float32), f"{OUT_DIR}/centroids_fp32.bin")

    # 2) FP16 lm-head
    lm_head = meta["lm_head_weight"].contiguous()
    save_bin(lm_head.numpy().astype(np.float16), f"{OUT_DIR}/lm_head_fp16.bin")

    # 3) hard labels (int32)
    hard_labels = meta["hard_labels"].contiguous()
    save_bin(hard_labels.numpy().astype(np.int32), f"{OUT_DIR}/hard_labels.bin")

    # 4) multi assign (int32)
    multi_assign = meta["multi_assign"].contiguous()
    save_bin(multi_assign.numpy().astype(np.int32), f"{OUT_DIR}/multi_assign.bin")

    # metadata (shape info)
    info = {
        "num_centroids": centroids.shape[0],
        "dim": centroids.shape[1],
        "vocab": lm_head.shape[0],
        "assign_k": multi_assign.shape[1],
    }
    with open(f"{OUT_DIR}/meta.txt", "w") as f:
        for k, v in info.items():
            f.write(f"{k}={v}\n")

    print("\n[COMPLETE] CUDA loadable binary export finished.")


if __name__ == "__main__":
    main()
