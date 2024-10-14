import multiprocessing as mp
import os

import fastcore.all as fc

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm


class DatasetDownloader:
    def __init__(self, dataset, local_dir, shard_size):
        super().__init__()
        fc.store_attr()
        self.enc = tiktoken.get_encoding("gpt2")
        self.eot = self.enc._special_tokens["<|endoftext|>"]
        self.data_cache_dir = os.path.join(os.path.dirname(__file__), local_dir)
        os.makedirs(self.data_cache_dir, exist_ok=True)

    def tokenize(self, doc):
        """
        tokeniuze this doc using gpt tokenizer
        uint16 is used since vocab size is 50257 and range of uint16 is 0-65535
        """
        tokens = [self.eot]
        tokens.extend(self.enc.encode_ordinary(doc["text"]))
        tokens_np = np.array(tokens)
        assert (0 <= tokens_np).all() and (
            tokens_np < 2**16
        ).all(), "token out of range"
        tokens_np = tokens_np.astype(np.uint16)
        return tokens_np

    def write_datafile(self, filename, tokens_np):
        np.save(filename, tokens_np)

    def download_dataset(self):
        nprocs = max(1, os.cpu_count() // 2)

        with mp.Pool(nprocs) as pool:
            shard_index = 0
            all_tokens_np = np.empty((self.shard_size,), dtype=np.uint16)
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

            # persist the last shard
            if token_count != 0:
                split = "val" if shard_index == 0 else "train"
                filename = os.path.join(
                    self.data_cache_dir, f"{split}_{shard_index:03d}"
                )
                self.write_datafile(filename, all_tokens_np)
