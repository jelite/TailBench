import pandas as pd
import re, os

data_file_dir = "../exp_with_flashattn/data/preprocess"

def get_file_list(profile_dir_path: str):
    file_list = []
    # 파일 이름 패턴: <모델이름>_b<숫자>.nsys-rep
    # pattern = re.compile(r"^(.*)_b(\d+)\.nsys-rep$")
    pattern = re.compile(r"^xxx_(.*)_b(\d+)_(.*)\.csv$")

    for dirpath, _, filenames in os.walk(profile_dir_path):
        for filename in filenames:
            match = pattern.match(filename)
            if match:
                file_list.append(match)

    return file_list

def get_ratio(df: pd.DataFrame):
    sum_duration = df["Duration (ms)"].sum()
    sum_tail = df["tail_time_ms"].sum()
    return sum_tail / sum_duration * 100

results = []

file_list = get_file_list(data_file_dir)

for file_match in file_list:
    file_name = file_match.group(0)
    model_name = file_match.group(1)
    batch_size = file_match.group(2)

    df = pd.read_csv(f"{data_file_dir}/{file_name}")

    prefill_df = df[df['phase'] == 'prefill']
    decoding_df = df[df['phase'] == 'decoding']
    
    tail_ratio_prefill = get_ratio(prefill_df)
    tail_ratio_decoding = get_ratio(decoding_df)

    results.append({
        "model": model_name,
        "batch_size": batch_size,
        "tail_ratio_prefill(%)": tail_ratio_prefill,
        "tail_ratio_decoding(%)": tail_ratio_decoding
    })
output_csv_path = f"../exp_with_flashattn/data/summary/xxx_tail_ratio_summary.csv"
summary_df = pd.DataFrame(results)
summary_df.to_csv(output_csv_path, index=False)
print(f"✅ Tail ratio summary saved to: {output_csv_path}")