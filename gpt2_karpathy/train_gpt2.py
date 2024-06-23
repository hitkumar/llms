import torch
import time
import math
from model import GPT, GPTConfig
from dataloader import DataLoaderLite
import inspect

# autodetect the device
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = 'mps'
print(f'using device: {device}')

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

# device = 'cpu'

# model = GPT.from_pretrained('gpt2')
# print('loaded successfully')

model = GPT(GPTConfig(vocab_size=50304)) # power of 2 is better for the GPUs
model.to(device)
model = torch.compile(model)

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

# define the dataloader
dataloader = DataLoaderLite(B=16, T=1024)
# TF 32 datatype which changes the precision of float32 numbers internally.

# time taken 1045 flp32 -> 375 with TF32 -> 328 with TF32 and bf16
# 154 with torch compile on top, 107K tokens per second.
# 110 with flash attention, 149K tokens per second.
# 105 with vocab size pow of 2, 154K tokens per second.
torch.set_float32_matmul_precision('high')

# optimizer loop
# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-9)
optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=3e-4, device=device)
# print(optimizer.param_groups)


# num_iters = 10
for i in range(55):
    t0 = time.time()
    optimizer.zero_grad()
    x, y = dataloader.next_batch()
    x, y = x.to(device), y.to(device)
    
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits, loss = model(x, y)
    loss.backward()
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
    tokens_per_sec = dataloader.B * dataloader.T / (t1 - t0)
    print(f"loss at iter {i}: {loss.item()}, time_taken: {dt:.2f}, tokens_per_sec: {tokens_per_sec:.2f}, norm: {norm:.4f}| lr: {lr:.4e}")

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
