### Experiments list
1) ~~Base gpt-2 model~~
2) shuffle data during training
3) ~~Increase lr~~
4) ~~Increase sequence lnegth~~
5) Use Shampoo optimizer and other optimization tuning
6) Increase model size
7) ~~Train for longer - more epochs~~
8) ~~Train a MOE style model~~
9) Try different learning rate schedules not just cosine.

### Things to try

- Plot position embeddings after more training.
- Maybe use fineweb 100B dataset to match GPT3 data volume.
- Weights and bias integration.
- Tune learning rate params
- Use eleuther eval harness from torchtune to include other evals.
- Torch compile with eval and hellaswag.

Classification fine tuning and instruction fine tuning experiments

### Experiment Results

| Experiment_id | Val_loss | Notes |
| -------- | -------- | -------- |
| HF_GPT_2    | 3.2758   | Baseline model   |
| base_gpt2    | 3.0829   | GPT-2 trained from scratch   |
| base_gpt2_increase_lr | 3.0430 | base_gpt2 with 3x learning rate |
| base_gpt2_inc_lr_3_epochs    | 2.9711  | Train for 3 epochs, train longer once quality improves more on 1 epoch  |
|base_gpt2_mistral_moe| 2.8782 | Mistral MOE arch with 8 experts, 2 experts per token|
|base_gpt_seq_len_2048| 3.0507 | base_gpt2 with 4x learning rate and seq_len=2048, not much change from base_gpt


Llama model training

- LLama2 7b OOM on A100 40gb so far.
- Model does train when we reduce the model size.
- Try reducing the size of the model.
- Tokenization is different across models, make it more general so data generation script has less repetition.
- Even gpt2 model 774M params leads to OOM, fix that first by potentially moving to bf16 and then move to Llama. Also try using fsdp and fsdp2.


### Dev Notes

To use modules inside python projects, use `pip install -e .` after adding setup.py

Helpful Commands
- Distributed training run: torchrun --standalone --nproc_per_node=8 train_gpt2.py
- simple launch: python3 train_gpt2.py
- To resolve the module not found error, use export PYTHONPATH="/home/htkumar/llms" for now.
