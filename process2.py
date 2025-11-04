import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

batch = 1
# metric = "tail_time_ms"
metric = "wave"
# CSV 파일 읽기
df = pd.read_csv(f"./data/kernel_waves_tail_batch{batch}.csv")

# CDF 계산
values = df[metric].dropna().sort_values()
cdf = np.arange(1, len(values) + 1) / len(values)

# 그래프 그리기
plt.figure(figsize=(8, 6))
plt.plot(values, cdf, linestyle='-', linewidth=2)
plt.xlabel(metric)
plt.ylabel("CDF")
plt.title(f"CDF of {metric}")
plt.grid(True)

# 이미지로 저장
plt.tight_layout()
plt.savefig(f"cdf_{metric}_batch{batch}.png", dpi=300)
plt.close()

print(f"CDF 그래프가 cdf_{metric}_batch{batch}.png로 저장되었습니다.")
