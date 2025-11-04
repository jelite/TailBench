import pandas as pd

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--batch", type=int, help="배치 크기 (batch size)")
args = parser.parse_args()
    
    
# CSV 파일 읽기
df = pd.read_csv(f"./data/kernel_waves_tail_batch{args.batch}.csv")

# 평균 계산
sum_duration = df["Duration (ms)"].sum()
sum_tail = df["tail_time_ms"].sum()

# print(f"{args.batch} Duration (ms) 평균: {mean_duration}")
# print(f"{args.batch} tail_time_ms 평균: {mean_tail_time}")
print(f"{args.batch} tail 비율: {sum_tail / sum_duration * 100:.2f}%")
