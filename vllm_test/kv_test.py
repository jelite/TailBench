import time
import torch
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.llm_engine import LLMEngine
from vllm import SamplingParams

def main():
    # -------------------------------------------------------
    # 실험용 KV 길이
    # -------------------------------------------------------
    kv_targets = [100, 200, 500, 1000]
    CACHE_DIR = "/model_cache"

    # -------------------------------------------------------
    # 1) vLLM 최신 API로 엔진 생성
    # -------------------------------------------------------
    engine_args = EngineArgs(
        model="mistralai/Mistral-7B-Instruct-v0.3",
        dtype="float16",
        tensor_parallel_size=1,
        download_dir=CACHE_DIR,
        gpu_memory_utilization=0.9,
        enforce_eager=True,         # timing 정확
        tokenizer_mode="mistral",   # ★ 중요: mistral tokenizer 강제 적용
    )

    engine = LLMEngine.from_engine_args(engine_args)

    grow_sampling = SamplingParams(max_tokens=1, temperature=0.0)

    request_ids = []

    # -------------------------------------------------------
    # 2) 각 request마다 KV cache 길이 생성
    # -------------------------------------------------------
    for i, kv_len in enumerate(kv_targets):

        req_id = f"REQ_{i}"
        request_ids.append(req_id)

        # dummy prompt
        engine.add_request(
            request_id=req_id,
            prompt="hello",
            sampling_params=grow_sampling,
        )

        # prefill
        engine.step()

        # decode 반복해 KV 길이 증가
        for _ in range(kv_len - 1):
            engine.step()

        print(f"[INIT] Request {req_id}: KV length = {kv_len}")

    print("\n=== KV caches prepared ===\n")

    # -------------------------------------------------------
    # 3) decode 1-step latency 측정
    # -------------------------------------------------------
    decode_sampling = SamplingParams(max_tokens=1, temperature=0.0)

    # continuation 요청 추가
    for req_id in request_ids:
        engine.add_request(
            request_id=req_id,
            prompt=None,
            sampling_params=decode_sampling,
        )

    torch.cuda.synchronize()
    start = time.time()

    outputs = engine.step()  # ★ KV 길이가 다른 request들 batch decode

    torch.cuda.synchronize()
    end = time.time()

    # -------------------------------------------------------
    # 4) 결과 출력
    # -------------------------------------------------------
    print("=== Decode Latency ===")
    print(f"Latency (sec): {end - start:.6f}\n")

    for o in outputs:
        print(f"REQ={o.request_id}, token='{o.outputs[0].text}'")


# -------------------------------------------------------
# Spawn-safe main guard (필수)
# -------------------------------------------------------
if __name__ == "__main__":
    main()
