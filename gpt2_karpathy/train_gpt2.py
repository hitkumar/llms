import torch
import time
import math
from model import GPT, GPTConfig
from dataloader import DataLoaderLite
import config
import os

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

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

total_batch_size = 2**19
# microbatch size
B = 16
T = 1024

# define the dataloader
dataloader = DataLoaderLite(B=B, T=T, process_rank=dpp_rank, num_processes=dpp_world_size)
assert total_batch_size % (B * T * dpp_world_size) == 0, "total batch size must be divisible by B * T * dpp_world_size"
grad_accum_steps = total_batch_size // (B * T * dpp_world_size)
if config.master_process:
    print(f'total batch size: {total_batch_size}, grad accum steps: {grad_accum_steps}, 1 epoch steps is {len(dataloader.tokens) / (B * T * grad_accum_steps * dpp_world_size)}')

# model = GPT.from_pretrained('gpt2')

model = GPT(GPTConfig(vocab_size=50304)) # power of 2 is better for the GPUs
model.to(device)
model = torch.compile(model)
if is_dpp:
    model = DDP(model, device_ids=[dpp_local_rank])
raw_model = model.module if is_dpp else model # get the raw model from dpp container

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50


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
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=3e-4, device=device)
# print(optimizer.param_groups)

# num_iters = 10
for i in range(5):
    t0 = time.time()
    optimizer.zero_grad()

    loss_accum = 0.0
    for microstep in range(grad_accum_steps):
        x, y = dataloader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)

        # since we want mean divide by grad_accum_steps 1 / micro_batch_size becomes 1 / grad_accum_steps * micro_batch_size
        loss = loss / grad_accum_steps   
        loss_accum += loss.detach()
        if is_dpp:
            model.require_backward_grad_sync = (microstep == grad_accum_steps - 1)
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
    tokens_processed = dataloader.B * dataloader.T * grad_accum_steps * dpp_world_size
    tokens_per_sec = tokens_processed / (t1 - t0)
    if config.master_process:
        print(f"loss at iter {i}: {loss_accum.item()}, time_taken: {dt:.2f}, tokens_per_sec: {tokens_per_sec:.2f}, norm: {norm:.4f}| lr: {lr:.4e}")

if is_dpp:
    destroy_process_group()

# import sys; sys.exit(0)

# model.eval()
# num_return_sequences = 5
# max_length = 30
# enc = tiktoken.get_encoding('gpt2')
# tokens = enc.encode("Hello, I'm a language model,")
# tokens = torch.tensor(tokens, dtype=torch.long) # (8,)
# tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (5, 8)

# x = tokens.to(device)

# torch.manual_seed(42)
# torch.cuda.manual_seed(42)

# # (B, T)
# while x.size(1) < max_length:
#     with torch.no_grad():
#         # (B, T, vocab_size)
#         logits = model(x)
#         # (B, vocab_size)
#         logits = logits[:, -1, :]
#         probs = F.softmax(logits, dim=-1)

#         # (B, 50)
#         topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
#         new_id = torch.multinomial(topk_probs, num_samples=1) # (B, 1)
#         new_id = torch.gather(topk_indices, -1, new_id) # (B, 1)
#         # (B, T + 1)
#         x = torch.cat((x, new_id), dim=-1)

# for i in range(num_return_sequences):
#     print(enc.decode(x[i].tolist()))
