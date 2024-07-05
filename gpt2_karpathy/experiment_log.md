### Experiments list
1) Base gpt-2 model
2) shuffle data during training
3) Increase lr
4) Increase sequence lnegth
5) Use Shampoo optimizer and other optimization tuning
6) Increase model size
7) Train for longer - more epochs

Classification fine tuning and instruction fine tuning experiments

### Experiment Results

| Experiment_id | Val_loss | Notes |
| -------- | -------- | -------- |
| HF_GPT_2    | 3.2758   | Baseline model   |
| base_gpt2    | 3.0829   | GPT-2 trained from scratch   |
| base_gpt2_increase_lr | 3.0430 | base_gpt2 with 3x learning rate |
| Row 3    | Data 5   | Data 6   |

Helpful Commands
- Distributed training run: torchrun --standalone --nproc_per_node=8 train_gpt2.py
- simple launch: python3 train_gpt2.py
