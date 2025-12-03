# export_hidden_for_cuda.py
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "google/gemma-2-2b-it"
CACHE_DIR = "/model_cache"
CLUSTER_FILE = "gemma2_clustered_100.pt"   # 이미 쓰고 있는 그 파일

OUT_DIR = "cluster_bin"

def main():
    meta = torch.load(CLUSTER_FILE, map_location="cuda")
    W_half = meta["lm_head_weight"].cuda()           # [V, D]

    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        dtype=torch.float16,
        device_map="cuda",
        cache_dir=CACHE_DIR
    )

    text = "When I was young, I used to"
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
        hidden = out.hidden_states[-1][0, -1, :].half()   # [D], fp16

    # baseline: full lm-head
    logits = W_half @ hidden      # [V]
    topk_vals, topk_idx = torch.topk(logits, 40)

    print("Python baseline top-40 ids: ", topk_idx.tolist())
    print("Python baseline top-40 vals:", topk_vals.float().tolist())

    # hidden을 fp16 raw bin으로 저장
    hidden_cpu = hidden.cpu().contiguous()
    np_hidden = hidden_cpu.numpy().astype(np.float16)
    np_hidden.tofile(f"{OUT_DIR}/hidden_fp16.bin")
    print(f"[SAVED] hidden_fp16.bin (shape={np_hidden.shape})")

if __name__ == "__main__":
    main()
