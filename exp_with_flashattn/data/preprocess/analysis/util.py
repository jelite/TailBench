import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def save_dot(df, title:str, xlabel:str, ylabel:str, file_name:str) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(df, marker='o')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)

    # 이미지 파일로 저장
    plt.savefig(file_name, dpi=300)
    plt.close()

    print(f"✅ 그래프가 {file_name} 파일로 저장되었습니다.")
    
def save_cdf(df, title:str, xlabel:str, ylabel:str, file_name:str, bound:int) -> None:
    data = np.sort(df.dropna())
    

    # 누적 분포 계산
    cdf = np.arange(1, len(data) + 1) / len(data)

     
    # CDF 그래프 그리기
    plt.figure(figsize=(8, 5))
    plt.plot(data, cdf, marker='.', linestyle='-')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)

    # 이미지 저장
    plt.xlim(0, bound)
    
    plt.savefig(file_name, dpi=300)
    plt.close()

    print(f"✅ CDF 그래프가 {file_name}로 저장되었습니다.")
    
def sort_save(df, file_name:str) -> None:
    df = df.sort_values(by="tail_time_ms",ascending=False)
    df = df[["Name","wave", "tail_time_ms"]]
    df.to_csv(file_name, index=True)