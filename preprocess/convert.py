import subprocess
import util

data_file_dir = "../data"
profile_file_dir = "../profile"
model_info_list = util.convert_nsys_to_csv(profile_file_dir)

# 커널 정보 추출
# for model_info in model_info_list:
#     # 1. nsys 명령 실행
#     subprocess.run([
#         "nsys", "stats",
#         "--report", "cuda_gpu_trace",
#         "--format", "csv",
#         "--force-export", "true",
#         "--output", f"../profile/{model_info.group(1)}",
#         f"../profile/{model_info.group(0)}"
#     ], check=True)

# # NVTX 정보 추출
# for model_info in model_info_list:
#     util.extract_sqlite_event_to_csv(f"../profile/{model_info.group(1)}.sqlite", f"../profile/{model_info.group(1)}_event.csv")

# 커널 파일과 NVTX 이벤트 파일을 합쳐서 prefill/decoding phase 구분
# for model_info in model_info_list:
#     util.merge_nvtx_kernel_file(
#         kernel_file_path=f"{profile_file_dir}/{model_info.group(1)}_cuda_gpu_trace.csv",
#         nvtx_file_path=f"{profile_file_dir}/{model_info.group(1)}_event.csv",
#         output_file_path=f"{data_file_dir}/raw/{model_info.group(1)}_with_phase.csv"
#     )

# wave 그리고 tail-effect의 시간과 비율 계산
for model_info in model_info_list:
    raw_file_path = f"{data_file_dir}/raw/{model_info.group(1)}_with_phase.csv"
    output_file_path = f"{data_file_dir}/preprocess/{model_info.group(1)}_kernel_waves_tail.csv"
    util.extract_tail_effect_info(raw_file_path, output_file_path)
