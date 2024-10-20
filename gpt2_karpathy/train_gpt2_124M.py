import math
import os
import time

import ddp_config

import torch

import torch.distributed as dist

from dataloader import DataLoaderLite

from evaluate import get_validation_loss
from gpt2_model import GPT, GPTConfig, MoeArgs
from hellaswag import get_most_likely_row, iterate_examples, render_example
from model_hparams import HParams
from train import train_model

hparams = HParams(
    max_lr=6e-4 * 4,
    min_lr=6e-4 * 4 * 0.1,
    weight_decay=0.1,
    # gpt-3 warmup schedule where we warmup for 375M tokens, in each step we train over 2**19 tokens, this is 375M / 2^19
    warmup_steps=500,
    # 1 epoch over the 10B token dataset, each step we train over 2**19 tokens
    # Do 1 epochs through the dataset.
    max_steps=19073,
    total_batch_size=2**19,
    # microbatch size
    B=8,
    T=2048,
    log_freq=1000,
)


# define the dataloader
train_dataloader = DataLoaderLite(
    B=hparams.B,
    T=hparams.T,
    process_rank=ddp_config.dpp_rank,
    num_processes=ddp_config.dpp_world_size,
    split="train",
)
val_dataloader = DataLoaderLite(
    B=hparams.B,
    T=hparams.T,
    process_rank=ddp_config.dpp_rank,
    num_processes=ddp_config.dpp_world_size,
    split="val",
)

# # model = GPT.from_pretrained('gpt2')

model = GPT(GPTConfig(vocab_size=50304))  # power of 2 is better for the GPUs
experiment_id = "gpt2_124M_compile_lr"

# Train the model

train_model(
    model,
    train_dataloader,
    val_dataloader,
    hparams,
    experiment_id,
)
