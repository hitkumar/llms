from dataset_downloader import DatasetDownloader
from datasets import load_dataset  # huggingface datasets

remote_name = "sample-10BT"  # these are 10B gpt2 tokens sampled from the whole dataset
dataset = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train")

dataset_downloader = DatasetDownloader(
    dataset, local_dir="fineweb", shard_size=int(1e8)
)
dataset_downloader.download_dataset()
