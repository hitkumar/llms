"""Downloads and evaluates gpt-2 models on Hellaswag"""
import tiktoken
import os
import json
import requests
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import GPT2LMHeadModel
from pathlib import Path

hellaswags = {
    "train": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl",
    "val": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
    "test": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl",
}

enc = tiktoken.get_encoding('gpt2')

DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "hellaswag")

def download_file(url: str, fname: str, chunk_size=1024):
    """Helper function to download a file"""
    resp = requests.get(url, stream=True)
    total_size = int(resp.headers.get("Content-Length", 0))
    with open(fname, "wb") as f:
    #tqdm(desc=fname, total=total_size, unit="iB", unit_scale=True, unit_divisor=1024) as pbar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = f.write(data)
            # pbar.update(size)

def download(split):
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    data_url = hellaswags[split]
    data_filename = Path(DATA_CACHE_DIR) / f"{split}.jsonl"
    if not data_filename.exists():
        print(f"downloading hellaswag split {split}")
        download_file(data_url, data_filename)

    download_file(data_url, data_filename)

def render_example(example):
    """ 
    Returns tokens of size 4 * N where each row consists of context + completion
    mask which is 1 for part of completion where we calculate the loss
    label which is index of the right completion
    """
    ctx = example["ctx"]
    label = example["label"]
    endings = example["endings"]

    ctx_tokens = enc.encode(ctx)
    tok_rows = []
    mask_rows = []
    for end in endings:
        end_tokens = enc.encode(" " + end)
        tok_rows.append(ctx_tokens + end_tokens)
        mask_rows.append([0] * len(ctx_tokens) + [1] * (len(end_tokens)))
    
    max_len = max(len(row) for row in tok_rows)
    tokens = torch.zeros((4, max_len), dtype=torch.long)
    mask = torch.zeros((4, max_len), dtype=torch.long)
    # Last elements of tokens and mask will be 0s
    for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
        tokens[i, :len(tok_row)] = torch.tensor(tok_row)
        mask[i, :len(mask_row)] = torch.tensor(mask_row)
    
    return tokens, mask, label

def iterate_examples(split):
    download(split)
    data_filename = Path(DATA_CACHE_DIR) / f"{split}.jsonl"
    
    with open(data_filename, "r") as f:
        for line in f:
            example = json.loads(line)
            yield example

def get_most_likely_row(tokens, mask, logits):
    """ Retuns index of prediction with lowest normalized loss """
    shift_logits = logits[: , :-1 ,:].contiguous()
    shift_tokens = tokens[:, 1:].contiguous()
    # print(shift_logits.shape, shift_tokens.shape)
    # Loss is returned for each position
    shift_losses = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_tokens.view(-1), reduction="none")
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # print(shift_losses.shape)

    shift_mask = mask[:, 1:].contiguous()
    # print(shift_mask[1])
    masked_shift_losses = shift_losses * shift_mask
    sum_losses = masked_shift_losses.sum(dim=1)
    avg_loss = sum_losses / shift_mask.sum(dim=1)
    # print(masked_shift_losses.shape, sum_losses.shape, avg_loss, sum_losses)

    pred = sum_losses.argmin().item()
    pred_norm = avg_loss.argmin().item()
    return pred_norm

@torch.no_grad()
def evaluate(model_type, device):
    torch.set_float32_matmul_precision('high')
    model = GPT2LMHeadModel.from_pretrained(model_type)
    model.to(device)
    # model = torch.compile(model)

    num_correct_norm = 0
    num_total = 0
    for example in iterate_examples("val"):
        tokens, mask, label = render_example(example)
        # print(tokens.shape, mask.shape, label)
        tokens = tokens.to(device)
        mask = mask.to(device)

        # get the logits
        logits = model(tokens).logits
        pred_norm = get_most_likely_row(tokens, mask, logits)

        num_total += 1
        num_correct_norm += int(pred_norm == label)
        

        # if num_total <= 3:
        #     print("---------------------------------------------------")
        #     print(f"context: {example['ctx']}")
        #     print('Endings:')
        #     for i, end in enumerate(example['endings']):
        #         print(f"{i}: (loss: {avg_loss[i].item():.4f}) {end}")
            
        #     print(f"predicted: {pred_norm}, actual: {label}")
        
    print(f"{num_total} acc_norm: {num_correct_norm}/{num_total}={num_correct_norm/num_total:.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="gpt2", help="model type to use")
    parser.add_argument("-d", "--device", type=str, default="cuda", help="device to use")
    args = parser.parse_args()
    # print(args)
    evaluate(args.model, args.device)
