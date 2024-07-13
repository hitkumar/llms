from pathlib import Path

from huggingface_hub import snapshot_download
import os
mistral_model_path = Path.home().joinpath("mistral_models", "7B-v0.3")
mistral_model_path.mkdir(parents=True, exist_ok=True)

# snapshot_download(
#     repo_id="mistralai/Mistral-7B-v0.3",
#     allow_patterns=["params.json", "consolidated.safetensors", "tokenizer.model.v3"],
#     local_dir=mistral_model_path,
# )

data_path = Path.home().joinpath("llms""mistral_data")
import pandas as pd

df = pd.read_parquet('https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k/resolve/main/data/test_gen-00000-of-00001-3d4cd8309148a71f.parquet')
print(df.shape)

df_train = df.sample(frac=0.95, random_state=200)
df_eval = df.drop(df_train.index)

DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), 'ultrachat')
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# save data into .jsonl files
df_train.to_json(os.path.join(DATA_CACHE_DIR, "train.jsonl"), orient="records", lines=True)
df_eval.to_json(os.path.join(DATA_CACHE_DIR, "eval.jsonl"), orient="records", lines=True)
