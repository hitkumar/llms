import os

import ddp_config

import tiktoken
import torch
import torch.distributed as dist
from gpt2_model import GPT, GPTConfig
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import GPT2LMHeadModel


@torch.no_grad()
def get_validation_loss(
    model,
    raw_model,
    val_dataloader,
    is_dpp,
    log_dir,
    step,
    last_step,
    device,
    device_type,
    log_freq,
):
    if step % log_freq == 0 or last_step:
        model.eval()
        val_dataloader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            loss_total = 0.0
            for _ in range(val_loss_steps):
                x, y = val_dataloader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss_total += loss.detach()

            loss_total /= val_loss_steps
            val_loss_accum = loss_total.detach()

        if ddp_config.is_dpp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)

        if ddp_config.master_process:
            print(f"val loss at iter {step}: {val_loss_accum.item():.4f}")
            log_file = os.path.join(log_dir, "log.txt")
            with open(log_file, "a") as f:
                f.write(f"{step} val {val_loss_accum.item():.4f}\n")

            # write model checkpoints
            if step % log_freq == 0 or last_step:
                checkpoint_file = os.path.join(log_dir, f"model_{step:05d}.pt")
                checkpoint = {
                    "model": raw_model._orig_mod.state_dict(),
                    "config": raw_model.config,
                    "step": step,
                    "val_loss": val_loss_accum.item(),
                }
                torch.save(checkpoint, checkpoint_file)
