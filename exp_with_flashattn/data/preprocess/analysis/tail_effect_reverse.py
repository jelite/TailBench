import pandas as pd
import matplotlib.pyplot as plt
from util import sort_save, save_cdf
# CSV 파일 경로
df_16 = pd.read_csv("../Qwen_Qwen2.5-3B-Instruct_b16_kernel_waves_tail.csv")
df_32 = pd.read_csv("../Qwen_Qwen2.5-3B-Instruct_b32_kernel_waves_tail.csv")

df_16_d = df_16[df_16['phase'] == 'decoding']
df_32_d = df_32[df_32['phase'] == 'decoding']

df_16_m= df_16_d['tail_time_ms']
df_32_m = df_32_d['tail_time_ms']

for bs in [16,32]:
    
    fig_arg = {
        "df": eval(f"df_{bs}_m"),
        "title": f"tail-effect_dur_{bs} - Decoding Phase",
        "xlabel": "tail_effect dur",
        "ylabel": "Cumulative Probability",
        "file_name": f"tail_effect_{bs}.png",
        "bound": 0.1
    }
    save_cdf(**fig_arg)
    
df_16_w= df_16_d['wave']
df_32_w = df_32_d['wave']

for bs in [16,32]:
    
    fig_arg = {
        "df": eval(f"df_{bs}_w"),
        "title": f"wave_{bs} - Decoding Phase",
        "xlabel": "wave",
        "ylabel": "Cumulative Probability",
        "file_name": f"wave_{bs}.png",
        "bound": 3
    }
    save_cdf(**fig_arg)

df_16_w= df_16_d[df_16_d['wave'] < 1]
df_16_w = df_16_w[['Name', 'wave', "tail_time_ms"]]
df_32_w = df_32_d[df_16_d['wave'] < 1]
df_32_w = df_32_w[['Name', 'wave', "tail_time_ms"]]


df_16_w.to_csv("to_analysis_16.csv")
df_32_w.to_csv("to_analysis_32.csv")
# sort_save(df_16_d, "16.csv")
# sort_save(df_32_d, "32.csv")
# 'tail_time_ms' 그래프 그리기