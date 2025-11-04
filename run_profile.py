import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=4, help="배치 크기 (batch size)")
    args = parser.parse_args()

    MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
    CACHE_DIR = "/model_cache"

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=False,
        cache_dir=CACHE_DIR,
        padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token 

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
        device_map="auto",
        cache_dir=CACHE_DIR
    )

    prompts = [f"이것은 배치 "*150 for i in range(args.batch)]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

    # print(inputs['input_ids'].shape)
    # import pdb; pdb.set_trace()
    
    # ---------------- Profile Start ----------------
    torch.cuda.cudart().cudaProfilerStart()
    torch.cuda.nvtx.range_push("generate")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id
        )

    torch.cuda.nvtx.range_pop()
    torch.cuda.cudart().cudaProfilerStop()
    # ---------------- Profile End ----------------

    results = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    for i, res in enumerate(results):
        print(f"[배치 {i}] {res}\n")

if __name__ == "__main__":
    main()

