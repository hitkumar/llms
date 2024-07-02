from hellaswag import render_example, iterate_examples, get_most_likely_row
import config
import torch
import torch.distributed as dist
import os
from model import GPT, GPTConfig
import tiktoken
from torch.nn import functional as F 


def get_validation_loss(model, raw_model, val_dataloader, is_dpp, log_dir, step, last_step, device, device_type):
    if step % 500 == 0 or last_step:
        model.eval()
        val_dataloader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            loss_total = 0.0
            for _ in range(val_loss_steps):
                x, y = val_dataloader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss_total += loss.detach()

            loss_total /= val_loss_steps
            val_loss_accum = loss_total.detach()

        if is_dpp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)

        if config.master_process:
            print(f"val loss at iter {step}: {val_loss_accum.item():.4f}")
            log_file = os.path.join(log_dir, "log.txt")
            with open(log_file, "a") as f:
                f.write(f"{step} val {val_loss_accum.item():.4f}\n")
            
            # write model checkpoints
            if (step % 2000 == 0 or last_step):
                checkpoint_file = os.path.join(log_dir, f"model_{step:05d}.pt")
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'config': raw_model.config,
                    'step': step,
                    'val_loss': val_loss_accum.item(),
                }
                torch.save(checkpoint, checkpoint_file)


def evaluate_hellaswag(is_dpp, dpp_world_size, dpp_rank, dpp_local_rank, log_dir, step, last_step, device, device_type):
    if (step % 500 == 0 or last_step):
        # print('Evaluating Hellaswag')
        # load the model from latest checkpoint saved in `get_validation_loss`
        model = GPT(GPTConfig(vocab_size=50304))
        model.to(device)
        # if is_dpp:
        #     model = DDP(model, device_ids=[dpp_local_rank])
        # model = torch.compile(model)
        checkpoint_file = torch.load(os.path.join(log_dir, f"model_{step:05d}.pt"), map_location=device)
        model.load_state_dict(checkpoint_file['model'])
        # print('Model loaded')

        model.eval()
        num_correct_norm = 0
        num_total = 0
        for i, example in enumerate(iterate_examples("val")):
            # if i % dpp_world_size != dpp_rank:
            #     continue
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
        # if is_dpp:
        #     num_total = torch.tensor(num_total, device=device, dtype=torch.long)
        #     num_correct_norm = torch.tensor(num_correct_norm, device=device, dtype=torch.long)
        #     dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
        #     dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
        #     num_total = num_total.item()
        #     num_correct_norm = num_correct_norm.item()
        
        acc_norm = num_correct_norm /  num_total
        # if config.master_process:
        print(f"Hellaswag step {step}: acc_norm: {num_correct_norm}/{num_total}={acc_norm:.4f}")
        log_file = os.path.join(log_dir, "log.txt")
        with open(log_file, "a") as f:
            f.write(f"{step} hellaswag {acc_norm:.4f}\n")
        
        # generate from the model
        num_return_sequences = 4
        max_length = 32
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode("Hello, I'm a language model,")
        tokens = torch.tensor(tokens, dtype=torch.long) # (8,)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (5, 8)
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
                new_id = torch.multinomial(topk_probs, num_samples=1) # (B, 1)
                new_id = torch.gather(topk_indices, -1, new_id) # (B, 1)
                # (B, T + 1)
                x = torch.cat((x, new_id), dim=-1)
        
        for i in range(num_return_sequences):
            decoded = enc.decode(x[i].tolist())
            print(f"{i} {decoded}")
