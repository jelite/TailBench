# evaluate_clustered_lm_head.py
import time
import torch, statistics
from transformers import AutoModelForCausalLM, AutoTokenizer


CLUSTER_FILE = "gemma2_clustered_100.pt"
CACHE_DIR = "/model_cache"


def build_coarse_to_fine(centroids_fp32, hard_labels, multi_assign, W_half):
    centroids_norm = centroids_fp32 / (centroids_fp32.norm(dim=1, keepdim=True) + 1e-6)
    centroids_half = centroids_norm.half().to("cuda")

    def coarse_to_fine(hidden, top_cluster=32, final_k=40):
        h = hidden.half()
        h_norm = h / (h.norm() + 1e-9)

        # coarse1: normalized centroid dot
        coarse1 = centroids_half @ h_norm

        # residual part는 FP32로
        h32 = hidden.float()
        proj_scalar = centroids_fp32 @ h32
        proj_vec = (proj_scalar.unsqueeze(1) * centroids_fp32).mean(dim=0)
        residual = h32 - proj_vec
        r_norm = residual / (residual.norm() + 1e-9)
        coarse2 = centroids_fp32 @ r_norm

        alpha = 0.35
        coarse_final = coarse1.float() + alpha * coarse2
        _, topC = torch.topk(coarse_final, top_cluster)

        mask_hard = torch.isin(hard_labels, topC)
        mask_soft = torch.isin(multi_assign, topC).any(dim=1)
        cand = torch.nonzero(mask_hard | mask_soft).squeeze(1)

        logits_small = (W_half[cand] @ h)
        vals, idx_local = torch.topk(logits_small, final_k)
        return cand[idx_local], vals

    return coarse_to_fine



def main():
    meta = torch.load(CLUSTER_FILE, map_location="cuda")

    W_half = meta["lm_head_weight"].to("cuda")
    centroids_fp32 = meta["centroids_fp32"].to("cuda")
    hard_labels = meta["hard_labels"].to("cuda")
    multi_assign = meta["multi_assign"].to("cuda")

    coarse_to_fine = build_coarse_to_fine(centroids_fp32, hard_labels, multi_assign, W_half)

    MODEL = meta["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        dtype=torch.float16,
        device_map="cuda",
        cache_dir=CACHE_DIR
    )

    tests = [
        "When I was young, I used to",
        "Explain speculative decoding simply.",
        "Tell me the recipe for pancakes",
        "The meaning of life is",
        "I want to travel to",
    ]

    for t in tests:
        print(f"\n===== 입력 문장: {t}")
        inputs = tokenizer(t, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
            hidden = out.hidden_states[-1][0, -1, :].half()

        # ---------------------------
        # 1) baseline: full lm-head
        # ---------------------------
        # warmup
        for _ in range(3):
            _ = W_half @ hidden

        torch.cuda.synchronize()
        t0 = time.time()
        logits_exact = W_half @ hidden
        torch.cuda.synchronize()
        base_time = time.time() - t0  # seconds

        top_exact = torch.topk(logits_exact, 40).indices
        exact_set = set(top_exact.tolist())

        # ---------------------------
        # 2) coarse-to-fine timing + recall
        # ---------------------------
        recalls = []
        coarse_times = []

        # coarse-to-fine warmup
        for _ in range(3):
            _ = coarse_to_fine(hidden, top_cluster=32, final_k=40)

        for _ in range(5):
            torch.cuda.synchronize()
            t1 = time.time()
            top_coarse, _ = coarse_to_fine(hidden, top_cluster=32, final_k=40)
            torch.cuda.synchronize()
            elapsed = time.time() - t1
            coarse_times.append(elapsed)

            recall = len(exact_set & set(top_coarse.tolist())) / 40
            recalls.append(recall)

        # ---------------------------
        # 3) 결과 출력
        # ---------------------------
        mean_recall = 100 * statistics.mean(recalls)
        std_recall = 100 * statistics.pstdev(recalls) if len(recalls) > 1 else 0.0
        mean_coarse_time = 1000 * statistics.mean(coarse_times)     # ms
        std_coarse_time = 1000 * statistics.pstdev(coarse_times) if len(coarse_times) > 1 else 0.0

        print(f"mean recall       : {mean_recall:.2f}% (std {std_recall:.2f}%)")
        print(f"baseline full mat : {base_time * 1000:.3f} ms")
        print(f"coarse-to-fine    : {mean_coarse_time:.3f} ± {std_coarse_time:.3f} ms "
              f"({base_time * 1000 / mean_coarse_time:.2f}x faster ratio)")


if __name__ == "__main__":
    main()
