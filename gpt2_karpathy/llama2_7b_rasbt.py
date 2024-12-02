import ddp_config
from dataloader import DataLoaderLite
from model_hparams import HParams
from rasbt_llms_from_scratch.llama2 import LLAMA2_CONFIG_7B, Llama2Model, LLamaTokenizer
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
    B=4,
    T=32,
    log_freq=1000,
)

# define the dataloader
train_dataloader = DataLoaderLite(
    B=hparams.B,
    T=hparams.T,
    process_rank=ddp_config.dpp_rank,
    num_processes=ddp_config.dpp_world_size,
    split="train",
    root_dir="/home/htkumar/llms/gpt2_karpathy/data/fineweb_sp",
)
val_dataloader = DataLoaderLite(
    B=hparams.B,
    T=hparams.T,
    process_rank=ddp_config.dpp_rank,
    num_processes=ddp_config.dpp_world_size,
    split="val",
    root_dir="/home/htkumar/llms/gpt2_karpathy/data/fineweb_sp",
)

config = LLAMA2_CONFIG_7B.copy()
model = Llama2Model(config)

experiment_id = "llama2_7b_rasbt"

# Train the model

train_model(
    model,
    train_dataloader,
    val_dataloader,
    hparams,
    experiment_id,
    run_hellaswag=False,
)
