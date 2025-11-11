import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import argparse
from huggingface_hub import login


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-Instruct-v0.3")
    args = parser.parse_args()

    MODEL_NAME = args.model_name
    CACHE_DIR = "/model_cache"

    print("model loading...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False, cache_dir=CACHE_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---------------- vLLM Engine Init ----------------
    llm = LLM(
        model=MODEL_NAME,
        tokenizer=MODEL_NAME,
        dtype="float16",
        gpu_memory_utilization=0.95,
        trust_remote_code=True,
        download_dir=CACHE_DIR,
    )

    # ---------------- Prompt 준비 ----------------
    prompt = "a " * 998
    prompts = [prompt for _ in range(args.batch)]
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=32,  # decoding 단계 길이
    )

    # ---------------- Warmup Start ----------------
    print("warmup start...")
    with torch.no_grad():
        for _ in range(3):
            _ = llm.generate(prompts, sampling_params)
    torch.cuda.synchronize()

    # ---------------- Profile Start ----------------
    torch.cuda.cudart().cudaProfilerStart()


    # ---- Decoding 단계 ----
    torch.cuda.nvtx.range_push("generate")
    # vLLM에서는 prefill+decode가 통합되지만 NVTX 구분용으로 분리
    with torch.no_grad():
        decode_outputs = llm.generate(prompts, sampling_params)
    torch.cuda.nvtx.range_pop()

    torch.cuda.cudart().cudaProfilerStop()
    # ---------------- Profile End ----------------

    # ---------------- 결과 출력 ----------------
    for i, output in enumerate(decode_outputs):
        print(f"[배치 {i}] {output.outputs[0].text}\n")

if __name__ == "__main__":
    main()
