"""This file contains a few functions to profile the memory usage of the model.

It is not meant to be used in production, but rather to help us debug the memory usage of the model.

The codes are borrowed from https://github.com/huggingface/accelerate/blob/main/benchmarks/measures_util.py
"""

import gc
import threading
import time
import numpy as np
import psutil
import torch
if torch.cuda.is_available():
    from accelerate.utils import compute_module_sizes as _compute_module_sizes

KB = 1 << 10
MB = 1 << 20
GB = 1 << 30
T = 1e12


def cpu_mem_stats():
    objects = gc.get_objects()
    tensors = [obj for obj in objects if torch.is_tensor(obj) and not obj.is_cuda]

    total_numel = 0
    total_mem = 0
    visited_data = set()
    for tensor in tensors:
        # a data_ptr indicates a memory block allocated
        data_ptr = tensor.storage().data_ptr()
        if data_ptr in visited_data:
            continue
        visited_data.add(data_ptr)

        numel = tensor.numel()
        total_numel += numel
        element_size = tensor.storage().element_size()
        mem = numel * element_size
        total_mem += mem

    return total_mem


def compute_module_sizes(model):
    return _compute_module_sizes(model)


class PeakCPUMemory:
    def __init__(self):
        self.process = psutil.Process()
        self.peak_monitoring = False

    def peak_monitor(self):
        self.cpu_memory_peak = -1

        while True:
            self.cpu_memory_peak = max(
                self.process.memory_info().rss, self.cpu_memory_peak
            )

            # can't sleep or will not catch the peak right (this comment is here on purpose)
            if not self.peak_monitoring:
                break

    def start(self):
        self.peak_monitoring = True
        self.thread = threading.Thread(target=self.peak_monitor)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        self.peak_monitoring = False
        self.thread.join()
        return self.cpu_memory_peak


cpu_peak_tracker = PeakCPUMemory()


def start_measure(clear_cache=True, cpu_peak=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == torch.device("cuda"):
        torch.cuda.synchronize()

    # Time
    measures = {"time": time.time()}

    if clear_cache:
        gc.collect()
        if device == torch.device("cuda"):
            torch.cuda.empty_cache()

    # CPU mem
    measures["cpu"] = psutil.Process().memory_info().rss
    if cpu_peak:
        cpu_peak_tracker.start()
    else:
        measures["cpu-peak"] = "Not test"

    # GPU mem
    if device == torch.device("cuda"):
        for i in range(torch.cuda.device_count()):
            measures[str(i)] = torch.cuda.memory_allocated(i)
        torch.cuda.reset_peak_memory_stats()

    return measures


def end_measure(start_measures, cpu_peak=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == torch.device("cuda"):
        torch.cuda.synchronize()

    # Time
    measures = {"time/s": round(time.time() - start_measures["time"], 3)}

    gc.collect()
    if device == torch.device("cuda"):
        torch.cuda.empty_cache()

    # CPU mem
    measures["cpu"] = format_size(psutil.Process().memory_info().rss - start_measures["cpu"])
    if cpu_peak:
        measures["cpu-peak"] = format_size(cpu_peak_tracker.stop() - start_measures["cpu"])

    # GPU mem
    if device == torch.device("cuda"):
        for i in range(torch.cuda.device_count()):
            measures["gpu" + str(i)] = format_size(
                                       torch.cuda.memory_allocated(i) - start_measures[
                                   str(i)]
                               ) 
            measures["gpu" + f"{i}-peak"] = format_size(
                                            torch.cuda.max_memory_allocated(i) -
                                            start_measures[str(i)]
                                    )

    return measures


def log_measures(measures, description):
    print(f"{description}:")
    print(f"- Time: {measures['time']:.2f}s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == torch.device("cuda"):
        for i in range(torch.cuda.device_count()):
            print(f"- GPU {i} allocated: {measures['gpu' + str(i)]:.3f}GiB")
            peak = measures["gpu" + f"{i}-peak"]
            print(f"- GPU {i} peak: {peak:.3f}GiB")
    print(f"- CPU RAM allocated: {measures['cpu']:.3f}GiB")
    if 'cpu-peak' in measures:
        print(f"- CPU RAM peak: {measures['cpu-peak']:.3f}GiB")


def format_size(size_bytes):
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"

def compute_complexity(model, input_tmp, device="cuda"):
    from thop import profile
    from thop import clever_format
    flops, _ = profile(model, inputs=(input_tmp, ))
    params = sum([v.numel() for k, v in model.state_dict().items()])
    flops, params = clever_format([flops, params], "%.3f")
    return flops, params