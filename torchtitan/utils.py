import os
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
