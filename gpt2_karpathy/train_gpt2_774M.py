import ddp_config

from dataloader import DataLoaderLite
from gpt2_model import GPT, GPTConfig
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

gpt_config = GPTConfig(
    vocab_size=50304,
    n_layer=36,
    n_head=20,
    n_embd=1280,
)
model = GPT(gpt_config)  # power of 2 is better for the GPUs
experiment_id = "gpt2_774M"

# Train the model

train_model(
    model,
    train_dataloader,
    val_dataloader,
    hparams,
    experiment_id,
)
