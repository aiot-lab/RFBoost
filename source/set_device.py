import os
def get_gpu_ids(i):
    gpu_ids = []
    gpu_info = os.popen("nvidia-smi -L").readlines()
    for line in gpu_info:
        # print(line)
        ids = line.split("UUID: ")[-1].strip(" ()\n")
        if ids.startswith("GPU"):
            continue
        # print(ids)
        gpu_ids.append(ids)
    # print("gpu_ids:", gpu_ids)
    return gpu_ids[i]