# Run this script with the following command:
# torchrun --nproc_per_node=3 nccl_ops/ops.py

import torch
import torch.distributed as dist
from torch.distributed import destroy_process_group


def init_process():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(dist.get_rank())


def example_broadcast():
    if dist.get_rank() == 0:
        tensor = torch.ones(1).cuda()
    else:
        tensor = torch.zeros(1).cuda()

    print(f"Before broadcast: {tensor}, rank: {dist.get_rank()}")
    dist.broadcast(tensor, src=0)
    print(f"After broadcast: {tensor}, rank: {dist.get_rank()}")


def example_reduce():
    tensor = torch.tensor([dist.get_rank() + 1] * 5, dtype=torch.float16).cuda()
    print(f"Before reduce: {tensor}, rank: {dist.get_rank()}")
    dist.reduce(tensor, dst=2, op=dist.ReduceOp.SUM)
    print(f"After reduce: {tensor}, rank: {dist.get_rank()}")


def example_all_reduce():
    tensor = torch.tensor([dist.get_rank() + 1] * 5, dtype=torch.bfloat16).cuda()
    print(
        f"Before all_reduce: {tensor}, rank: {dist.get_rank()}, world_size: {dist.get_world_size()}"
    )
    dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
    print(f"After all_reduce: {tensor}, rank: {dist.get_rank()}")


def example_gather():
    tensor = torch.tensor([dist.get_rank() + 1] * 5, dtype=torch.bfloat16).cuda()
    if dist.get_rank() == 0:
        gather_list = [
            torch.zeros(5, dtype=torch.bfloat16).cuda()
            for _ in range(dist.get_world_size())
        ]
    else:
        gather_list = None

    print(f"Before gather: {tensor}, rank: {dist.get_rank()}")
    dist.gather(tensor, gather_list=gather_list, dst=0)
    print(f"After gather: {gather_list}, rank: {dist.get_rank()}")


def example_all_gather():
    tensor = torch.tensor([dist.get_rank() + 1] * 5, dtype=torch.bfloat16).cuda()
    gather_list = [
        torch.zeros(5, dtype=torch.bfloat16).cuda()
        for _ in range(dist.get_world_size())
    ]
    print(f"Before gather: {tensor}, rank: {dist.get_rank()}")
    dist.all_gather(gather_list, tensor)
    print(f"After gather: {gather_list}, rank: {dist.get_rank()}")


def example_scatter():
    if dist.get_rank() == 0:
        scatter_list = [
            torch.tensor([i + 1] * 5, dtype=torch.bfloat16).cuda()
            for i in range(dist.get_world_size())
        ]
    else:
        scatter_list = None

    tensor = torch.zeros(5, dtype=torch.bfloat16).cuda()
    print(f"Before scatter: {tensor}, rank: {dist.get_rank()}")
    dist.scatter(tensor, scatter_list=scatter_list, src=0)
    print(f"After scatter: {tensor}, rank: {dist.get_rank()}")


def example_reduce_scatter():
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    input_tensor = [
        torch.tensor([(rank + 1) * i for i in range(1, 4)], dtype=torch.float32).cuda()
        for j in range(world_size)
    ]
    output_tensor = torch.zeros(1, 3, dtype=torch.float32).cuda()
    print(f"Before reduce_scatter input: {input_tensor}, rank: {dist.get_rank()}")
    dist.reduce_scatter(output_tensor, input_tensor, op=dist.ReduceOp.SUM)
    print(f"After reduce_scatter output: {output_tensor}, rank: {dist.get_rank()}\n")


init_process()

# example_broadcast()
# example_reduce()
# example_all_reduce()
# example_gather()
example_all_gather()
# example_scatter()
# example_reduce_scatter()


dist.barrier()
destroy_process_group()
