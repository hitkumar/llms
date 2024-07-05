import torch
import time
import math
from model import GPT, GPTConfig
from dataloader import DataLoaderLite
import config
import os
from evaluate import get_validation_loss, evaluate_hellaswag
config.master_process = True

# autodetect the device, below code assumes cuda


from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# setup ddp
# torchrun command sets the env variables RANK, LOCAL_RANK and WORLD_SIZE (which is 8)
is_dpp = int(os.environ.get('RANK', -1)) != -1

if is_dpp:
    assert torch.cuda.is_available()
    init_process_group(backend='nccl')
    dpp_rank = int(os.environ['RANK'])
    dpp_local_rank = int(os.environ['LOCAL_RANK'])
    dpp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{dpp_local_rank}'
    torch.cuda.set_device(device)
    config.master_process = dpp_rank == 0 # this will do logging, checkpointing
else:
    # non dpp run
    config.master_process = True 
    dpp_rank = 0
    dpp_local_rank = 0
    dpp_world_size = 1
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    print(f'using device: {device}')

device_type = 'cuda' if device.startswith('cuda') else 'cpu'

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

total_batch_size = 2**19
# microbatch size
B = 16
T = 1024

# define the dataloader
train_dataloader = DataLoaderLite(B=B, T=T, process_rank=dpp_rank, num_processes=dpp_world_size, split='train')
val_dataloader = DataLoaderLite(B=B, T=T, process_rank=dpp_rank, num_processes=dpp_world_size, split='val')

assert total_batch_size % (B * T * dpp_world_size) == 0, "total batch size must be divisible by B * T * dpp_world_size"
grad_accum_steps = total_batch_size // (B * T * dpp_world_size)
if config.master_process:
    print(f'total batch size: {total_batch_size}, grad accum steps: {grad_accum_steps}, 1 epoch steps is {len(train_dataloader.tokens) / (B * T * grad_accum_steps * dpp_world_size)}')

# model = GPT.from_pretrained('gpt2')

model = GPT(GPTConfig(vocab_size=50304)) # power of 2 is better for the GPUs
model.to(device)
eval_hellaswag = False
if not eval_hellaswag:
    model = torch.compile(model)
if is_dpp:
    model = DDP(model, device_ids=[dpp_local_rank])
raw_model = model.module if is_dpp else model # get the raw model from dpp container

# make max_lr 3x.
max_lr = 6e-4 * 3
min_lr = max_lr * 0.1

# gpt-3 warmup schedule where we warmup for 375M tokens, in each step we train over 2**19 tokens, this is 375M / 2^19
warmup_steps = 715
# 1 epoch over the 10B token dataset, each step we train over 2**19 tokens
max_steps = 19073


def get_lr(it):
    # linear warmup
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    elif it > max_steps:
        return min_lr
    # cosine decay
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


# TF 32 datatype which changes the precision of float32 numbers internally.
# time taken 1045 flp32 -> 375 with TF32 -> 328 with TF32 and bf16
# 154 with torch compile on top, 107K tokens per second.
# 110 with flash attention, 149K tokens per second.
# 105 with vocab size pow of 2, 154K tokens per second.
# 159K tokens per second with gradient accumulation
# with DPP on 8 gpus, we get 1.3M tokens per second, karpathy gets 1.5M as he is using A100 80G GPUs.
torch.set_float32_matmul_precision('high')

# optimizer loop
# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-9)
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=3e-4, device_type=device_type)
# print(optimizer.param_groups)

experiment_id = "base_gpt2_increase_lr"
LOGS_DIR = os.path.join(os.path.dirname(__file__), f"logs_{experiment_id}")
os.makedirs(LOGS_DIR, exist_ok=True)
log_file = os.path.join(LOGS_DIR, "log.txt")
# clear the file when new training starts
with open(log_file, 'w') as f:
    pass

# num_iters = 10
for i in range(max_steps):
    t0 = time.time()
    last_step = (i == max_steps - 1)
    get_validation_loss(model, raw_model, val_dataloader, is_dpp, LOGS_DIR, i, last_step, device, device_type)
    if config.master_process and eval_hellaswag:
        evaluate_hellaswag(is_dpp, dpp_world_size, dpp_rank, dpp_local_rank, LOGS_DIR, i, last_step, device, device_type)

    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for microstep in range(grad_accum_steps):
        x, y = train_dataloader.next_batch()
        x, y = x.to(device), y.to(device)
        if is_dpp:
            model.require_backward_grad_sync = (microstep == grad_accum_steps - 1)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, loss = model(x, y)

        # since we want mean divide by grad_accum_steps 1 / micro_batch_size becomes 1 / grad_accum_steps * micro_batch_size
        loss = loss / grad_accum_steps   
        loss_accum += loss.detach()
        loss.backward()
    
    if is_dpp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

    # clip gradients
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # set the lr for this iteration
    lr = get_lr(i)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    torch.cuda.synchronize() # wait for the gpu to finish
    t1 = time.time()
    dt = (t1 - t0) * 1e3 # time difference in ms
    tokens_processed = train_dataloader.B * train_dataloader.T * grad_accum_steps * dpp_world_size
    tokens_per_sec = tokens_processed / (t1 - t0)
    if config.master_process:
        print(f"loss at iter {i: 5d}: {loss_accum.item():.6f}, time_taken: {dt:.2f}, tokens_per_sec: {tokens_per_sec:.2f}, norm: {norm:.4f}| lr: {lr:.4e}")
        with open(log_file, 'a') as f:
            f.write(f"{i} train {loss_accum.item():.4f}\n")

if is_dpp:
    destroy_process_group()
