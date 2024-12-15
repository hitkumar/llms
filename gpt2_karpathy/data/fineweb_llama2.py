import multiprocessing as mp
import os

import fastcore.all as fc

import numpy as np
from datasets import load_dataset
from rasbt_llms_from_scratch.llama3 import Tokenizer
from tqdm import tqdm


class DatasetDownloader:
    def __init__(self, dataset, local_dir, shard_size):
        super().__init__()
        fc.store_attr()
        tokenizer_file = "/home/htkumar/llms/Llama-3.2-1B/original/tokenizer.model"
        self.enc = Tokenizer(tokenizer_file)
        self.data_cache_dir = os.path.join(os.path.dirname(__file__), local_dir)
        os.makedirs(self.data_cache_dir, exist_ok=True)

    def tokenize(self, doc):
        """
        tokenize this doc using gpt tokenizer
        uint16 is used since vocab size is 50257 and range of uint16 is 0-65535
        """
        tokens = self.enc.encode(doc["text"], bos=True)
        tokens_np = np.array(tokens)
        assert (0 <= tokens_np).all() and (
            tokens_np < 2**32
        ).all(), "token out of range"
        # print(tokens_np.dtype)
        tokens_np = tokens_np.astype(np.uint32)
        return tokens_np

    def write_datafile(self, filename, tokens_np):
        np.save(filename, tokens_np)

    def download_dataset(self):
        nprocs = max(1, os.cpu_count() // 2)

        with mp.Pool(nprocs) as pool:
            shard_index = 0
            all_tokens_np = np.empty((self.shard_size,), dtype=np.uint32)
            token_count = 0
            progress_bar = None

            for tokens in pool.imap(self.tokenize, self.dataset, chunksize=16):
                # print(len(tokens), mp.current_process())
                if token_count + len(tokens) < self.shard_size:
                    all_tokens_np[token_count : token_count + len(tokens)] = tokens
                    token_count += len(tokens)
                    # update progress bar
                    if progress_bar is None:
                        progress_bar = tqdm(
                            total=self.shard_size,
                            unit="tokens",
                            desc=f"Shard {shard_index}",
                        )
                    progress_bar.update(len(tokens))
                else:
                    # persist this shard
                    split = "val" if shard_index == 0 else "train"
                    filename = os.path.join(
                        self.data_cache_dir, f"{split}_{shard_index:03d}"
                    )
                    remainder = self.shard_size - token_count
                    progress_bar.update(remainder)
                    all_tokens_np[token_count : token_count + remainder] = tokens[
                        :remainder
                    ]
                    self.write_datafile(filename, all_tokens_np)
                    shard_index += 1
                    progress_bar = None
                    token_count = len(tokens) - remainder
                    all_tokens_np[:token_count] = tokens[remainder:]
                    # Remove this if you want to generate the entire dataset
                    break

            # persist the last shard
            if token_count != 0:
                split = "val" if shard_index == 0 else "train"
                filename = os.path.join(
                    self.data_cache_dir, f"{split}_{shard_index:03d}"
                )
                self.write_datafile(filename, all_tokens_np)


if __name__ == "__main__":
    remote_name = (
        "sample-10BT"  # these are 10B gpt2 tokens sampled from the whole dataset
    )
    dataset = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train")
    # import pandas as pd
    # from datasets import Dataset

    # dataset = dataset[:2]
    # df = pd.DataFrame(dataset)
    # hf_dataset = Dataset.from_pandas(df)[:20]
    # for d in hf_dataset:
    #     print(d)

    dataset_downloader = DatasetDownloader(
        dataset, local_dir="fineweb_llama3", shard_size=int(1e8)
    )
    dataset_downloader.download_dataset()
