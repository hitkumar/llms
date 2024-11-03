import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class GPTDatasetV1(Dataset):
    def __init__(self, text_data, tokenizer, seq_len=1024):

        tokenized_text = tokenizer.encode(text_data, allowed_special={"<|endoftext|>"})
        self.input_ids = []
        self.target_ids = []

        for i in range(0, len(tokenized_text) - seq_len, seq_len):
            self.input_ids.append(torch.tensor(tokenized_text[i : i + seq_len]))
            self.target_ids.append(
                torch.tensor(tokenized_text[i + 1 : i + seq_len + 1])
            )

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(
    txt, batch_size=4, max_length=256, shuffle=True, drop_last=True, num_workers=0
):

    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
    return dataloader
