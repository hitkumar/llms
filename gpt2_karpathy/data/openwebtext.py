from dataset_downloader import DatasetDownloader
from datasets import load_dataset  # huggingface datasets

dataset = load_dataset("stas/openwebtext-10k", split="train")
dataset_downloader = DatasetDownloader(
    dataset, local_dir="openwebtext", shard_size=int(1e8)
)
dataset_downloader.download_dataset()
