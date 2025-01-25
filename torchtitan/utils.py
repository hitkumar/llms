import gc
import os
import subprocess
from dataclasses import dataclass
from datetime import timedelta
from typing import Optional

import torch
import torch.distributed.distributed_c10d as c10d
from core.logging_util import init_logger, logger
from torch._utils import _get_available_device_type, _get_device_module
from torch.distributed.device_mesh import DeviceMesh


def get_device_info():
    device_type = _get_available_device_type()
    if device_type is None:
        device_type = "cuda"  # default device_type: cuda
    device_module = _get_device_module(device_type)  # default device_module:torch.cuda
    return device_type, device_module


device_type, device_module = get_device_info()


def _get_distributed_backend(job_config):
    backend = "nccl"
    if device_type in torch.distributed.Backend.default_device_backend_map.keys():
        backend = torch.distributed.Backend.default_device_backend_map.get(device_type)
    if job_config.training.enable_cpu_offload:
        backend = f"{device_type}:{backend},cpu:gloo"
    return backend


def init_distributed(job_config):
    # to mitigate the memory issue that collectives using
    # async_op=True hold memory longer than they should
    # such as those in tensor parallelism
    os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"

    torch.distributed.init_process_group(
        backend=_get_distributed_backend(job_config),
        timeout=timedelta(seconds=job_config.comm.init_timeout_seconds),
    )


@dataclass(frozen=True)
class Color:
    black = "\033[30m"
    red = "\033[31m"
    green = "\033[32m"
    yellow = "\033[33m"
    blue = "\033[34m"
    magenta = "\033[35m"
    cyan = "\033[36m"
    white = "\033[37m"
    reset = "\033[39m"


@dataclass(frozen=True)
class NoColor:
    black = ""
    red = ""
    green = ""
    yellow = ""
    blue = ""
    magenta = ""
    cyan = ""
    white = ""
    reset = ""


# used to avoid stragglers in garbage collection
class GarbageCollection:
    def __init__(self, gc_freq=1000):
        assert gc_freq > 0, "gc_freq must be positive"
        self.gc_freq = gc_freq
        gc.disable()
        gc.collect(1)

    def run(self, step_count):
        if step_count > 1 and step_count % self.gc_freq == 0:
            gc.collect(1)
            logger.info(f"garbage collection at step {step_count}")


# hardcoded BF16 type peak flops for NVIDIA A100, H100, and H200 GPU
def get_peak_flops(device_name: str) -> int:
    try:
        # Run the lspci command and capture the output
        result = subprocess.run(["lspci"], stdout=subprocess.PIPE, text=True)
        # Filter the output for lines containing both "NVIDIA" and "H100"
        filtered_lines = [
            line
            for line in result.stdout.splitlines()
            if "NVIDIA" in line and "H100" in line
        ]
        # Join all filtered lines into a single string
        device_name = " ".join(filtered_lines) or device_name
    except FileNotFoundError as e:
        logger.warning(f"Error running lspci: {e}, fallback to use device_name")
    if "A100" in device_name:
        logger.info(f"Found A100 device: {device_name}")
        # data from https://www.nvidia.com/en-us/data-center/a100/
        return 312e12
    elif "H100" in device_name:
        # data from https://www.nvidia.com/en-us/data-center/h100/
        # NOTE: Specifications are one-half lower without sparsity.
        if "NVL" in device_name:
            return 835e12
        elif "PCIe" in device_name:
            return 756e12
        else:  # for H100 SXM and other variants
            return 989e12
    elif "H200" in device_name:
        # data from https://www.nvidia.com/en-us/data-center/h200/
        return 989e12
    else:  # for other GPU types, assume A100
        logger.warning(f"Peak flops undefined for: {device_name}, fallback to A100")
        return 312e12
