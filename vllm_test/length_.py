from vllm import LLM, SamplingParams
import time

def main():

    CACHE_DIR = "/model_cache"

    llm = LLM(
        model="mistralai/Mistral-7B-Instruct-v0.3",
        trust_remote_code=True,
        tensor_parallel_size=1,
        dtype="float16",
        download_dir=CACHE_DIR,
        max_num_seqs=16,
    )

    prompts = [
        "Hello",
        "Explain machine learning.",
        "Write a detailed essay about LLMs and their scalability.",
        "Summarize the following text:\n" + "A"*5000,
    ]

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1
    )

    start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    end = time.time()

    # for i, out in enumerate(outputs):
    #     print(f"=== Request {i} ===")
    #     print(out.outputs[0].text)

    print("Batch Decode Latency:", end - start)
    tokenizer = llm.get_tokenizer()
    # 각 요청을 토큰화하여 길이 측정
    lengths = [len(tokenizer.encode(p)) for p in prompts]

    # 배치 전체 토큰 길이 합
    total_seq_len = sum(lengths)

    print("Individual seq lens:", lengths)
    print("Total batch seq_len:", total_seq_len)

if __name__ == "__main__":
    main()
