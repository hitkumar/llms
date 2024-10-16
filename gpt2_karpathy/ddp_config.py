import math
import os
import time

import torch

import torch.distributed as dist
from evaluate import evaluate_hellaswag, get_validation_loss
from model_hparams import HParams

from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

# setup ddp
# torchrun command sets the env variables RANK, LOCAL_RANK and WORLD_SIZE (which is 8)
is_dpp = int(os.environ.get("RANK", -1)) != -1

if is_dpp:
    assert torch.cuda.is_available()
    init_process_group(backend="nccl")
    dpp_rank = int(os.environ["RANK"])
    dpp_local_rank = int(os.environ["LOCAL_RANK"])
    dpp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{dpp_local_rank}"
    torch.cuda.set_device(device)
    master_process = dpp_rank == 0  # this will do logging, checkpointing
else:
    # non dpp run
    dpp_rank = 0
    dpp_local_rank = 0
    dpp_world_size = 1
    master_process = True
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

device_type = "cuda" if device.startswith("cuda") else "cpu"
