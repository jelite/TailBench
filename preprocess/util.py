import sqlite3
import pandas as pd
import re, os
import numpy as np
import pycuda.driver as cuda

def extract_sqlite_event_to_csv(db_path: str, output_csv_path: str):
    # SQLite DB 파일 열기
    conn = sqlite3.connect(db_path)

    # 테이블 목록 확인
    tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)
    print("Available tables:\n", tables)

    # NVTX_EVENTS 테이블에서 모든 데이터 읽기
    df = pd.read_sql_query("SELECT * FROM NVTX_EVENTS;", conn)

    # CSV로 저장
    df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")

    print(f"✅ NVTX_EVENTS exported to {output_csv_path}")
    print(f"총 {len(df)}개의 NVTX 이벤트가 추출되었습니다.")

    conn.close()
    
def convert_nsys_to_csv(profile_dir_path: str):
    model_info_list = []
    # 파일 이름 패턴: <모델이름>_b<숫자>.nsys-rep
    # pattern = re.compile(r"^(.*)_b(\d+)\.nsys-rep$")
    pattern = re.compile(r"^(.*)\.nsys-rep$")

    for dirpath, _, filenames in os.walk(profile_dir_path):
        for filename in filenames:
            match = pattern.match(filename)
            if match:
                model_info_list.append(match)

    return model_info_list

def merge_nvtx_kernel_file(kernel_file_path: str, nvtx_file_path: str, output_file_path: str):
    # 파일 로드
    trace = pd.read_csv(kernel_file_path)
    nvtx = pd.read_csv(nvtx_file_path)

    # prefill / decoding 구간 추출
    print(kernel_file_path)
    prefill_start, prefill_end = nvtx.loc[nvtx['text'] == 'prefill', ['start', 'end']].values[0]
    decoding_start, decoding_end = nvtx.loc[nvtx['text'] == 'decoding', ['start', 'end']].values[0]

    # phase 구분 함수
    def get_phase(t):
        if prefill_start <= t <= prefill_end:
            return "prefill"
        elif decoding_start <= t <= decoding_end:
            return "decoding"
        else:
            return "other"

    # 새로운 컬럼 추가
    trace["phase"] = trace["Start (ns)"].apply(get_phase)

    # 결과 저장
    trace.to_csv(output_file_path, index=False)

def extract_tail_effect_info(raw_file_path: str, output_file_path: str):
    df = pd.read_csv(raw_file_path)

    # 유효 커널만 선택
    df = df.dropna(subset=["BlkX", "BlkY", "BlkZ", "Reg/Trd", "StcSMem (MB)", "DymSMem (MB)"])

    cuda.init()

    device = cuda.Device(0)
    attrs = device.get_attributes()

    MAX_WARPS_PER_SM = attrs[cuda.device_attribute.MAX_THREADS_PER_MULTIPROCESSOR] // 32
    MAX_THREADS_PER_SM = attrs[cuda.device_attribute.MAX_THREADS_PER_MULTIPROCESSOR]
    MAX_BLOCKS_PER_SM = attrs[cuda.device_attribute.MAX_BLOCKS_PER_MULTIPROCESSOR]
    MAX_REGS_PER_SM = attrs[cuda.device_attribute.MAX_REGISTERS_PER_MULTIPROCESSOR]
    MAX_SMEM_PER_SM = attrs[cuda.device_attribute.MAX_SHARED_MEMORY_PER_MULTIPROCESSOR]

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
    df["num_blocks"] = df["GrdX"] * df["GrdY"] * df["GrdZ"]
    df["wave"] = df["num_blocks"] / (df["active_blocks_per_sm"] * attrs[cuda.device_attribute.MULTIPROCESSOR_COUNT])


    # 시간(ms) 변환
    df["Duration (ms)"] = df["Duration (ns)"] / 1e6

    # ✅ Tail 계산
    
    df["tail_ratio_%"] = ( np.ceil(df["wave"]) - np.floor(df["wave"]) ) / np.ceil(df["wave"]) * 100
    df["tail_time_ms"] = df["Duration (ms)"] * df["tail_ratio_%"] / 100

    # 필요한 컬럼만 정리
    output = df[[
        "Name",
        "Duration (ms)",
        "wave",
        "tail_ratio_%",
        "tail_time_ms",
        "threads_per_block",
        "Reg/Trd",
        "StcSMem (MB)",
        "DymSMem (MB)",
        "phase"
    ]]

    # CSV 저장
    output.to_csv(output_file_path, index=False)
    print(f"✅ Saved: {output_file_path}")
