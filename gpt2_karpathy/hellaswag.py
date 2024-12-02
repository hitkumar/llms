"""Downloads and evaluates gpt-2 models on Hellaswag"""

import json
import os
from pathlib import Path

import ddp_config

import requests
import tiktoken
import torch
import torch.distributed as dist
import torch.nn as nn
from gpt2_model import GPT, GPTConfig
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from transformers import GPT2LMHeadModel

hellaswags = {
    "train": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl",
    "val": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
    "test": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl",
}

enc = tiktoken.get_encoding("gpt2")

DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "hellaswag")


def download_file(url: str, fname: str, chunk_size=1024):
    """Helper function to download a file"""
    resp = requests.get(url, stream=True)
    total_size = int(resp.headers.get("Content-Length", 0))
    with open(fname, "wb") as f:
        # tqdm(desc=fname, total=total_size, unit="iB", unit_scale=True, unit_divisor=1024) as pbar:
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
        tokens[i, : len(tok_row)] = torch.tensor(tok_row)
        mask[i, : len(mask_row)] = torch.tensor(mask_row)

    return tokens, mask, label


def iterate_examples(split):
    download(split)
    data_filename = Path(DATA_CACHE_DIR) / f"{split}.jsonl"

    with open(data_filename, "r") as f:
        for line in f:
            example = json.loads(line)
            yield example


def get_most_likely_row(tokens, mask, logits):
    """Retuns index of prediction with lowest normalized loss"""
    shift_logits = logits[:, :-1, :].contiguous()
    shift_tokens = tokens[:, 1:].contiguous()
    # print(shift_logits.shape, shift_tokens.shape)
    # Loss is returned for each position
    shift_losses = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_tokens.view(-1),
        reduction="none",
    )
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
def evaluate(model, device):
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

    print(
        f"{num_total} acc_norm: {num_correct_norm}/{num_total}={num_correct_norm/num_total:.4f}"
    )


@torch.no_grad()
def evaluate_hf_model(model_type, device):
    torch.set_float32_matmul_precision("high")
    model = GPT2LMHeadModel.from_pretrained(model_type)
    model.to(device)
    return evaluate(model, device)


@torch.no_grad()
def evaluate_pretrained(device, log_dir, step):
    model = GPT(GPTConfig(vocab_size=50304))
    model.to(device)
    checkpoint_file = torch.load(
        os.path.join(log_dir, f"model_{step:05d}.pt"),
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(checkpoint_file["model"])
    return evaluate(model, device)


@torch.no_grad()
def evaluate_hellaswag(
    is_dpp,
    dpp_world_size,
    dpp_rank,
    dpp_local_rank,
    log_dir,
    step,
    last_step,
    device,
    device_type,
    log_freq,
    gpt_config: GPTConfig,
    log_results: bool = True,
):
    if step % log_freq == 0 or last_step:
        # load the model from latest checkpoint saved in `get_validation_loss`
        # print(f"gpt_config={gpt_config}")
        model = GPT(gpt_config)
        model.to(device)
        checkpoint_file = torch.load(
            os.path.join(log_dir, f"model_{step:05d}.pt"),
            map_location=device,
            weights_only=True,
        )
        model.load_state_dict(checkpoint_file["model"])
        model = torch.compile(model)
        if is_dpp:
            model = DDP(model, device_ids=[dpp_local_rank])

        model.eval()
        num_correct_norm = 0
        num_total = 0
        for i, example in enumerate(iterate_examples("val")):
            if i % dpp_world_size != dpp_rank:
                continue
            tokens, mask, label = render_example(example)
            tokens = tokens.to(device)
            mask = mask.to(device)
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(tokens)
                pred_norm = get_most_likely_row(tokens, mask, logits)

            num_total += 1
            num_correct_norm += int(pred_norm == label)
            # print(f"{i} {pred_norm=}, {label=}")

        # reduce stats across all processes
        if is_dpp:
            num_total = torch.tensor(num_total, device=device, dtype=torch.long)
            num_correct_norm = torch.tensor(
                num_correct_norm, device=device, dtype=torch.long
            )
            dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
            num_total = num_total.item()
            num_correct_norm = num_correct_norm.item()

        acc_norm = num_correct_norm / num_total
        if ddp_config.master_process:
            print(
                f"Hellaswag step {step}: acc_norm: {num_correct_norm}/{num_total}={acc_norm:.4f}"
            )
            if log_results:
                log_file = os.path.join(log_dir, "log.txt")
                with open(log_file, "a") as f:
                    f.write(f"{step} hellaswag {acc_norm:.4f}\n")

        if ddp_config.master_process:
            # generate from the model
            num_return_sequences = 2
            max_length = 32
            enc = tiktoken.get_encoding("gpt2")
            tokens = enc.encode("Hello, I'm a language model,")
            tokens = torch.tensor(tokens, dtype=torch.long)  # (8,)
            tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)  # (5, 8)
            x = tokens.to(device)

            sample_rng = torch.Generator(device=device)
            sample_rng.manual_seed(42)

            # (B, T)
            while x.size(1) < max_length:
                with torch.no_grad():
                    # (B, T, vocab_size)
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        logits, loss = model(x)
                    # (B, vocab_size)
                    # print(logits.shape)
                    logits = logits[:, -1, :]
                    probs = F.softmax(logits, dim=-1)

                    # (B, 50)
                    topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                    new_id = torch.multinomial(topk_probs, num_samples=1)  # (B, 1)
                    new_id = torch.gather(topk_indices, -1, new_id)  # (B, 1)
                    # (B, T + 1)
                    x = torch.cat((x, new_id), dim=-1)

            for i in range(num_return_sequences):
                decoded = enc.decode(x[i].tolist())
                print(f"{i} {decoded}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model", type=str, default="gpt2", help="model type to use"
    )
    parser.add_argument(
        "-d", "--device", type=str, default="cuda", help="device to use"
    )
    args = parser.parse_args()
    # print(args)
    # evaluate(args.model, args.device)

    experiment_id = "gpt2_355M"
    LOGS_DIR = os.path.join(os.path.dirname(__file__), f"logs_{experiment_id}")

    evaluate_hellaswag(
        ddp_config.is_dpp,
        ddp_config.dpp_world_size,
        ddp_config.dpp_rank,
        ddp_config.dpp_local_rank,
        LOGS_DIR,
        19073 - 1,
        True,
        ddp_config.device,
        ddp_config.device_type,
        10,
        log_results=False,
    )
