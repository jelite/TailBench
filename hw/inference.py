from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"

CACHE_DIR = "/model_cache"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
    device_map="auto",
    cache_dir=CACHE_DIR
)

prompt = "Explain speculative decoding simply."

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# 일반적으로 사용하는 top-k 값: 40 또는 50
generation = model.generate(
    **inputs,
    max_new_tokens=200,
    do_sample=True,
    top_k=50,
    temperature=0.7
)

print(tokenizer.decode(generation[0], skip_special_tokens=True))
