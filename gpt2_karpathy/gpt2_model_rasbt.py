import ddp_config
from dataloader import DataLoaderLite
from model_hparams import HParams
from rasbt_llms_from_scratch.gpt_download import BASE_CONFIG, model_configs
from rasbt_llms_from_scratch.gpt_model import GPTModelWithTargets
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
    T=1024,
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

config = BASE_CONFIG.copy()
config.update(model_configs["gpt2-medium (355M)"])
model = GPTModelWithTargets(config)

experiment_id = "gpt2_355M_rasbt"

# Train the model

train_model(
    model,
    train_dataloader,
    val_dataloader,
    hparams,
    experiment_id,
    run_hellaswag=False,
)
