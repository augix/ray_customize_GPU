import ray
import os
import torch

def select_gpu(gpu_name):
    print(f'selecting {gpu_name}')
    local_gpu_index = int(gpu_name.split("_GPU")[-1])       # Extract "0"    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(local_gpu_index)

@ray.remote
class CustomGPU:
    def __init__(self, gpu_name):
        select_gpu(gpu_name)

    def get_free_memory(self):
        free = torch.cuda.mem_get_info(0)[0] / 1024 / 1024 / 1024 # GB
        self.free_memory = free
        return free

def get_free_memory(gpu_name):
    node_name = gpu_name.split('_GPU')[0]
    actor = CustomGPU.options(resources={f"node:{node_name}": 0.01}).remote(gpu_name)
    free_memory = ray.get(actor.get_free_memory.remote())
    ray.kill(actor)
    return free_memory

def get_gpu_names():
    gpu_names = []
    if not ray.is_initialized():
        ray.init(address='auto', ignore_reinit_error=True)
    nodes = ray.nodes()
    for node in nodes:
        node_name = node['NodeName']
        num_gpus = node.get('Resources', {}).get('GPU', 0)
        num_gpus = int(num_gpus)
        if num_gpus == 0:
            continue
        for i in range(num_gpus):
            gpu_name = f"{node_name}_GPU{i}"
            gpu_names.append(gpu_name)
    return gpu_names

def find_top_k_gpu(gpu_names,k=1):
    print(f"Finding top {k} GPU...")
    # check if ray is initialized
    if not ray.is_initialized():
        ray.init(address='auto', ignore_reinit_error=True)
    gpu_free_memory = []
    for gpu_name in gpu_names:
        try:
            free_memory = get_free_memory(gpu_name)
            gpu_free_memory.append((gpu_name, free_memory))
            print(f"GPU: {gpu_name}, Free memory: {free_memory:.2f} GB")
        except Exception as e:
            print(f"Error checking {gpu_name}: {e}")
    # sort by free memory
    gpu_free_memory.sort(key=lambda x: x[1], reverse=True)
    gpu_names = [gpu_name for gpu_name, _ in gpu_free_memory]
    if k == 0:
        return gpu_names
    top_k_gpu = gpu_names[:k]
    return top_k_gpu

def find_eligible_gpu(gpu_names, n_gpu=4, free_memory_threshold=10):
    print(f'finding {n_gpu} GPUs with free memory greater than {free_memory_threshold} GB')
    # find all GPUs with free memory greater than the threshold in unit of GB
    eligible_gpu = []
    for gpu_name in gpu_names:
        try:
            free_memory = get_free_memory(gpu_name)
        except Exception as e:
            print(f"Error checking {gpu_name}: {e}")
            continue
        if free_memory > free_memory_threshold:
            eligible_gpu.append(gpu_name)
            print(f"Found eligible GPU: {gpu_name}, Free memory: {free_memory:.2f} GB")
        if len(eligible_gpu) >= n_gpu:
            return eligible_gpu
    return None