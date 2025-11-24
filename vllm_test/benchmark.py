from vllm import LLM, SamplingParams

llm = LLM(
    model="mistralai/Mistral-7B-Instruct-v0.3",
    dtype="float16",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.9,
    max_model_len=4096
)

sampling_params = SamplingParams(
    temperature=0.3,
    max_tokens=256
)

prompts = [
    "Summarize this:" + (" long text." * 200),  # ~1k tokens
] * 2  # batch size 2

outputs = llm.generate(prompts, sampling_params)
