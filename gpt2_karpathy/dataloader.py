import torch
import config
import numpy as np
import os

def load_tokens(filename):
    npt = np.load(filename)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        " split could be train or val"
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes

        # with open('/home/htkumar/llms/gpt2_karpathy/input.txt', 'r') as f:
        #     text = f.read()
        
        # enc = tiktoken.get_encoding('gpt2')
        # tokens = enc.encode(text)
        # self.tokens = torch.tensor(tokens)

        assert split in ('train', 'val')
        root_dir = "edu_fineweb10TB"
        shards = os.listdir(root_dir)
        shards = sorted(shards)
        shards = [os.path.join(root_dir, s) for s in shards if split in s]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for {split}"

        if config.master_process:
            # print(f"loaded {len(tokens)} in dataloader, num_batches in 1 epoch is {len(tokens) // (B * T * self.num_processes)}")
            print(f"found {len(shards)} for split {split}")

        self.reset()
    
    def reset(self):
        # B*T tokens are reserved for each process
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position: self.current_position + B * T + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        self.current_position += B * T * self.num_processes

        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = self.B * self.T * self.process_rank

        return x, y
