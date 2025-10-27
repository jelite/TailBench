import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=4, help="배치 크기 (batch size)")
    args = parser.parse_args()

    MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token 

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    prompts = [f"이것은 배치 {i}번 문장입니다." for i in range(args.batch)]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id
        )

    results = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    for i, res in enumerate(results):
        print(f"[배치 {i}] {res}\n")

if __name__ == "__main__":
    main()
