import pandas as pd
import numpy as np

# Nsight trace CSV 로드
df = pd.read_csv("kernels_trace_cuda_gpu_trace.csv")

# 유효 커널만 선택
df = df.dropna(subset=["BlkX", "BlkY", "BlkZ", "Reg/Trd", "StcSMem (MB)", "DymSMem (MB)"])

# RTX 4090 (Ada AD102) 기준 리소스 제약
MAX_WARPS_PER_SM = 64
MAX_THREADS_PER_SM = 2048
MAX_BLOCKS_PER_SM = 32
MAX_REGS_PER_SM = 65536
MAX_SMEM_PER_SM = 100 * 1024  # 100 KB

# 스레드 및 자원 계산
df["threads_per_block"] = df["BlkX"] * df["BlkY"] * df["BlkZ"]
df["regs_per_block"] = df["threads_per_block"] * df["Reg/Trd"]
df["shared_mem_per_block"] = (df["StcSMem (MB)"] + df["DymSMem (MB)"]) * 1e6

# 각 자원별 블록 수 계산
df["blocks_by_regs"] = np.floor(MAX_REGS_PER_SM / df["regs_per_block"].replace(0, np.nan))
df["blocks_by_smem"] = np.floor(MAX_SMEM_PER_SM / df["shared_mem_per_block"].replace(0, np.nan))
df["blocks_by_threads"] = np.floor(MAX_THREADS_PER_SM / df["threads_per_block"].replace(0, np.nan))

# SM당 활성 블록 수
df["active_blocks_per_sm"] = df[["blocks_by_regs", "blocks_by_smem", "blocks_by_threads"]].min(axis=1)
df["active_blocks_per_sm"] = df["active_blocks_per_sm"].clip(upper=MAX_BLOCKS_PER_SM).fillna(0)

# 워프 및 웨이브 계산
df["warps_per_block"] = np.ceil(df["threads_per_block"] / 32)
df["active_warps_per_sm"] = df["active_blocks_per_sm"] * df["warps_per_block"]
df["wave_%"] = np.clip(df["active_warps_per_sm"] / MAX_WARPS_PER_SM * 100, 0, 100)

# 시간(ms) 변환
df["Duration (ms)"] = df["Duration (ns)"] / 1e6

# ✅ Tail 계산
df["tail_ratio_%"] = 100 - df["wave_%"]
df["tail_time_ms"] = df["Duration (ms)"] * df["tail_ratio_%"] / 100

# 필요한 컬럼만 정리
output = df[[
    "Name",
    "Duration (ms)",
    "wave_%",
    "tail_ratio_%",
    "tail_time_ms",
    "threads_per_block",
    "Reg/Trd",
    "StcSMem (MB)",
    "DymSMem (MB)"
]]

# CSV 저장
output.to_csv("kernel_waves_tail.csv", index=False)
print("✅ Saved: kernel_waves_tail.csv")
