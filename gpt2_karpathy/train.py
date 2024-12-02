import math
import os
import time

import ddp_config
import torch

import torch.distributed as dist
from evaluate import get_validation_loss
from hellaswag import evaluate_hellaswag

from torch.distributed import destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP


torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

# TF 32 datatype which changes the precision of float32 numbers internally.
# time taken 1045 flp32 -> 375 with TF32 -> 328 with TF32 and bf16
# 154 with torch compile on top, 107K tokens per second.
# 110 with flash attention, 149K tokens per second.
# 105 with vocab size pow of 2, 154K tokens per second.
# 159K tokens per second with gradient accumulation
# with DPP on 8 gpus, we get 1.3M tokens per second, karpathy gets 1.5M as he is using A100 80G GPUs.
torch.set_float32_matmul_precision("high")


def get_lr(it, hparams):
    """
    cosine decay with warmup
    TODO: Use pytorch cosine lr scheduler
    """
    # linear warmup
    if it < hparams.warmup_steps:
        return hparams.max_lr * (it + 1) / hparams.warmup_steps
    elif it > hparams.max_steps:
        return min_lr
    # cosine decay
    decay_ratio = (it - hparams.warmup_steps) / (
        hparams.max_steps - hparams.warmup_steps
    )
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return hparams.min_lr + coeff * (hparams.max_lr - hparams.min_lr)


def train_model(
    model,
    train_dataloader,
    val_dataloader,
    model_hparams,
    experiment_id,
    run_hellaswag=True,
):
    LOGS_DIR = os.path.join(os.path.dirname(__file__), f"logs_{experiment_id}")
    os.makedirs(LOGS_DIR, exist_ok=True)
    log_file = os.path.join(LOGS_DIR, "log.txt")
    # clear the file when new training starts
    with open(log_file, "w") as f:
        pass

    model.to(ddp_config.device)
    model = torch.compile(model)
    if ddp_config.is_dpp:
        model = DDP(model, device_ids=[ddp_config.dpp_local_rank])
    raw_model = (
        model.module if ddp_config.is_dpp else model
    )  # get the raw model from dpp container

    assert (
        model_hparams.total_batch_size
        % (model_hparams.B * model_hparams.T * ddp_config.dpp_world_size)
        == 0
    ), "total batch size must be divisible by B * T * dpp_world_size"

    grad_accum_steps = model_hparams.total_batch_size // (
        model_hparams.B * model_hparams.T * ddp_config.dpp_world_size
    )
    if ddp_config.master_process:
        print(
            f"total batch size: {model_hparams.total_batch_size}, grad accum steps: {grad_accum_steps}, 1 epoch steps is {len(train_dataloader.tokens) / (model_hparams.B * model_hparams.T * grad_accum_steps * ddp_config.dpp_world_size)}"
        )

    optimizer = raw_model.configure_optimizers(
        weight_decay=model_hparams.weight_decay,
        learning_rate=3e-4,
        device_type=ddp_config.device_type,
    )
    for i in range(model_hparams.max_steps):
        t0 = time.time()
        last_step = i == model_hparams.max_steps - 1
        get_validation_loss(
            model,
            raw_model,
            val_dataloader,
            ddp_config.is_dpp,
            LOGS_DIR,
            i,
            last_step,
            ddp_config.device,
            ddp_config.device_type,
            model_hparams.log_freq,
        )
        if run_hellaswag:
            evaluate_hellaswag(
                ddp_config.is_dpp,
                ddp_config.dpp_world_size,
                ddp_config.dpp_rank,
                ddp_config.dpp_local_rank,
                LOGS_DIR,
                i,
                last_step,
                ddp_config.device,
                ddp_config.device_type,
                model_hparams.log_freq,
                raw_model.config,
            )

        model.train()
        optimizer.zero_grad()
        loss_accum = 0.0
        for microstep in range(grad_accum_steps):
            x, y = train_dataloader.next_batch()
            x, y = x.to(ddp_config.device), y.to(ddp_config.device)
            if ddp_config.is_dpp:
                model.require_backward_grad_sync = microstep == grad_accum_steps - 1
            with torch.autocast(
                device_type=ddp_config.device_type, dtype=torch.bfloat16
            ):
                logits, loss = model(x, y)
                # debug in vscode
                # import code

                # code.interact(local=locals())

            # since we want mean divide by grad_accum_steps 1 / micro_batch_size becomes 1 / grad_accum_steps * micro_batch_size
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            loss.backward()

        if ddp_config.is_dpp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

        # clip gradients
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # set the lr for this iteration
        lr = get_lr(i, model_hparams)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        optimizer.step()
        if ddp_config.device_type == "cuda":
            torch.cuda.synchronize()  # wait for the gpu to finish
        t1 = time.time()
        dt = (t1 - t0) * 1e3  # time difference in ms
        tokens_processed = (
            train_dataloader.B
            * train_dataloader.T
            * grad_accum_steps
            * ddp_config.dpp_world_size
        )
        tokens_per_sec = tokens_processed / (t1 - t0)
        if ddp_config.master_process:
            print(
                f"loss at iter {i: 5d}: {loss_accum.item():.6f}, time_taken: {dt:.2f}, tokens_per_sec: {tokens_per_sec:.2f}, norm: {norm:.4f}| lr: {lr:.4e}"
            )
            with open(log_file, "a") as f:
                f.write(f"{i} train {loss_accum.item():.4f}\n")

    if ddp_config.is_dpp:
        destroy_process_group()
