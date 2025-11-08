import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
from huggingface_hub import login



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--model_name", type=str, default = "mistralai/Mistral-7B-Instruct-v0.3")
    args = parser.parse_args()

    MODEL_NAME = args.model_name
    CACHE_DIR = "/model_cache"

    print("model loading...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False, cache_dir=CACHE_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token 

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
        device_map="auto",
        cache_dir=CACHE_DIR,
        attn_implementation="flash_attention_2"
    )
    prompt = "a "*998
    prompts = [prompt for i in range(args.batch)]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    
    # ---------------- warmup Start ----------------
    with torch.no_grad():
        for _ in range(3):
            _ = model(**inputs, use_cache=True)
    torch.cuda.synchronize()


    # ---------------- Profile Start ----------------
    torch.cuda.cudart().cudaProfilerStart()

    # ---- Prefill 단계 ----
    torch.cuda.nvtx.range_push("prefill")
    with torch.no_grad():
        prefill_out = model(**inputs, use_cache=True)
    torch.cuda.nvtx.range_pop()

    # ---- Decoding 단계 ----
    torch.cuda.nvtx.range_push("decoding")
    past_key_values = prefill_out.past_key_values
    input_ids = inputs["input_ids"]

    # 디코딩 루프 (간단한 예시)
    generated = input_ids
    for _ in range(32):
        with torch.no_grad():
            out = model(input_ids=generated[:, -1:], past_key_values=past_key_values, use_cache=True)
        next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=-1)
        past_key_values = out.past_key_values

    torch.cuda.nvtx.range_pop()
    torch.cuda.cudart().cudaProfilerStop()
    # ---------------- Profile End ----------------

    results = tokenizer.batch_decode(generated, skip_special_tokens=True)
    for i, res in enumerate(results):
        print(f"[배치 {i}] {res}\n")

if __name__ == "__main__":
    main()
