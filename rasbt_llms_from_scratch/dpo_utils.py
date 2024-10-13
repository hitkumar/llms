import torch
from gpt_download import BASE_CONFIG
from gpt_model import generate, text_to_token_ids, token_ids_to_text
from torch.utils.data import Dataset


def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that approximately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    return instruction_text + input_text


class PreferenceDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data

        self.encoded_texts = []
        for entry in data:
            prompt = format_input(entry)
            rejected_response = entry["rejected"]
            chosen_response = entry["chosen"]
            chosen_full_text = f"{prompt}\n\n###Response:\n{chosen_response}"
            rejected_full_text = f"{prompt}\n\n###Response:\n{rejected_response}"

            prompt_tokens = tokenizer.encode(prompt)
            chosen_full_tokens = tokenizer.encode(chosen_full_text)
            rejected_full_tokens = tokenizer.encode(rejected_full_text)

            self.encoded_texts.append(
                {
                    "prompt": prompt_tokens,
                    "chosen": chosen_full_tokens,
                    "rejected": rejected_full_tokens,
                }
            )

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.encoded_texts)


def custom_collate_fn(
    batch,
    pad_token_id=50256,
    allowed_max_length=None,
    mask_prompt_tokens=True,
    device="cpu",
):
    batch_data = {
        "prompt": [],
        "chosen": [],
        "rejected": [],
        "rejected_mask": [],
        "chosen_mask": [],
    }

    max_length_common = 0
    if batch:
        for key in ["chosen", "rejected"]:
            # why adding +1 here? possibly end of sentence token
            current_max = max(len(item[key]) + 1 for item in batch)
            max_length_common = max(max_length_common, current_max)

    for item in batch:
        prompt = torch.tensor(item["prompt"])
        batch_data["prompt"].append(prompt)
        for key in ["chosen", "rejected"]:
            sequence = item[key]
            padded = sequence + [pad_token_id] * (max_length_common - len(sequence))
            mask = torch.ones(len(padded)).bool()

            # set mask for padding tokens to be False
            mask[len(sequence) :] = False

            # +2 sets the new 2 newline tokens before ### Response to False
            # Set mask for input tokens to be False
            if mask_prompt_tokens:
                mask[: prompt.shape[0] + 2] = False

            batch_data[key].append(torch.tensor(padded))
            batch_data[f"{key}_mask"].append(mask)

    # Process batch data
    for key in ["chosen", "rejected", "chosen_mask", "rejected_mask"]:
        # [B, max_length_common]
        tensor_stack = torch.stack(batch_data[key])
        if allowed_max_length is not None:
            tensor_stack = tensor_stack[:, :allowed_max_length]

        batch_data[key] = tensor_stack.to(device)

    return batch_data


import torch.nn.functional as F


def compute_dpo_loss(
    model_chosen_logprobs,
    model_rejected_logprobs,
    reference_chosen_logprobs,
    reference_rejected_logprobs,
    beta=0.1,
):
    model_log_ratios = model_chosen_logprobs - model_rejected_logprobs
    reference_log_ratios = reference_chosen_logprobs - reference_rejected_logprobs
    logits = model_log_ratios - reference_log_ratios

    losses = -F.logsigmoid(beta * logits)
    chosen_rewards = (model_chosen_logprobs - reference_chosen_logprobs).detach()
    rejected_rewards = (model_rejected_logprobs - reference_rejected_logprobs).detach()

    return losses.mean(), chosen_rewards.mean(), rejected_rewards.mean()


def compute_dpo_loss_alt(
    model_chosen_logprobs,
    model_rejected_logprobs,
    reference_chosen_logprobs,
    reference_rejected_logprobs,
    beta=0.1,
):
    chosen_logprobs = model_chosen_logprobs - reference_chosen_logprobs
    rejected_logprobs = model_rejected_logprobs - reference_rejected_logprobs

    logits = chosen_logprobs - rejected_logprobs

    losses = -F.logsigmoid(beta * logits)
    chosen_rewards = (model_chosen_logprobs - reference_chosen_logprobs).detach()
    rejected_rewards = (model_rejected_logprobs - reference_rejected_logprobs).detach()

    return losses.mean(), chosen_rewards.mean(), rejected_rewards.mean()


def compute_logprobs(logits, labels, selection_mask=None):
    """
    logits is [B, num_tokens, vocab_size]
    labels is [B, num_tokens]
    selection_mask is [B, num_tokens]
    """
    logits = logits[:, :-1, :]
    log_probs = F.log_softmax(logits, dim=-1)
    labels = labels[:, 1:]

    # shape is [B, num_tokens-1] consisting of log_probs at every index.
    selected_log_probs = torch.gather(
        input=log_probs, dim=-1, index=labels.unsqueeze(-1)
    ).squeeze(-1)

    if selection_mask is not None:
        mask = selection_mask[:, 1:].clone()
        # Apply the mask to filter out padding tokens
        selected_log_probs = selected_log_probs * mask
        avg_log_prob = selected_log_probs.sum(-1) / mask.sum(-1)
        return avg_log_prob

    return selected_log_probs.mean(-1)


def compute_dpo_loss_batch(batch, policy_model, reference_model, beta):
    policy_model_chosen_logprobs = compute_logprobs(
        policy_model(batch["chosen"]), batch["chosen"], batch["chosen_mask"]
    )
    policy_model_rejected_logprobs = compute_logprobs(
        policy_model(batch["rejected"]), batch["rejected"], batch["rejected_mask"]
    )
    reference_model_chosen_logprobs = compute_logprobs(
        reference_model(batch["chosen"]), batch["chosen"], batch["chosen_mask"]
    )
    reference_model_rejected_logprobs = compute_logprobs(
        reference_model(batch["rejected"]), batch["rejected"], batch["rejected_mask"]
    )
    return compute_dpo_loss_alt(
        policy_model_chosen_logprobs,
        policy_model_rejected_logprobs,
        reference_model_chosen_logprobs,
        reference_model_rejected_logprobs,
        beta,
    )


def compute_dpo_loss_loader(
    data_loader, policy_model, reference_model, beta, num_batches=None
):
    total_loss, total_chosen_rewards, total_rejected_rewards = 0.0, 0.0, 0.0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, batch in enumerate(data_loader):
        if i < num_batches:
            loss, chosen_rewards, rejected_rewards = compute_dpo_loss_batch(
                batch, policy_model, reference_model, beta
            )
            total_loss += loss.item()
            total_chosen_rewards += chosen_rewards.item()
            total_rejected_rewards += rejected_rewards.item()
        else:
            break

    return (
        total_loss / num_batches,
        total_chosen_rewards / num_batches,
        total_rejected_rewards / num_batches,
    )


def evaluate_dpo_loss_loader(
    policy_model, reference_model, train_loader, val_loader, beta, eval_iter
):
    # reference model has always been in eval model since creation.
    policy_model.eval()
    with torch.no_grad():
        train_loss, train_chosen_rewards, train_rejected_rewards = (
            compute_dpo_loss_loader(
                data_loader=train_loader,
                policy_model=policy_model,
                reference_model=reference_model,
                beta=beta,
                num_batches=eval_iter,
            )
        )

        val_loss, val_chosen_rewards, val_rejected_rewards = compute_dpo_loss_loader(
            data_loader=val_loader,
            policy_model=policy_model,
            reference_model=reference_model,
            beta=beta,
            num_batches=eval_iter,
        )

    res = {
        "train_loss": train_loss,
        "train_chosen_reward": train_chosen_rewards,
        "train_rejected_reward": train_rejected_rewards,
        "val_loss": val_loss,
        "val_chosen_reward": val_chosen_rewards,
        "val_rejected_reward": val_rejected_rewards,
    }
    policy_model.train()
    return res


def generate_model_output(model, tokenizer, data, device="cpu"):
    input_text = format_input(data)

    token_ids = generate(
        model=model,
        idx=text_to_token_ids(input_text, tokenizer).to(device),
        max_new_tokens=256,
        context_size=BASE_CONFIG["context_length"],
        eos_id=50256,
    )
    generated_text = token_ids_to_text(token_ids, tokenizer)
    response_text = (
        generated_text[len(input_text) :].replace("### Response:", "").strip()
    )

    print(input_text)
    print(f"\nCorrect response:\n>> {data['output']}")
    print(f"\nModel response:\n>> {response_text}")
    print("\n----------------------------------------------\n")


def train_model_dpo_simple(
    policy_model,
    reference_model,
    train_loader,
    val_loader,
    optimizer,
    num_epochs,
    beta,
    eval_freq,
    eval_iter,
    start_context,
    tokenizer,
):
    tracking = {
        "train_losses": [],
        "train_chosen_rewards": [],
        "train_rejected_rewards": [],
        "val_losses": [],
        "val_chosen_rewards": [],
        "val_rejected_rewards": [],
        "tokens_seen": [],
    }
    tokens_seen, global_step = 0, -1

    # Main training loop
    for epoch in range(num_epochs):
        policy_model.train()

        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()

            loss, chosen_rewards, rejected_rewards = compute_dpo_loss_batch(
                batch=batch,
                policy_model=policy_model,
                reference_model=reference_model,
                beta=beta,
            )

            loss.backward()
            optimizer.step()  # update model parameters

            tokens_seen += batch["chosen"].numel()
            global_step += 1

            if global_step % eval_freq == 0:
                res = evaluate_dpo_loss_loader(
                    policy_model=policy_model,
                    reference_model=reference_model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    beta=beta,
                    eval_iter=eval_iter,
                )

                tracking["train_losses"].append(res["train_loss"])
                tracking["train_chosen_rewards"].append(res["train_chosen_reward"])
                tracking["train_rejected_rewards"].append(res["train_rejected_reward"])
                tracking["val_losses"].append(res["val_loss"])
                tracking["val_chosen_rewards"].append(res["val_chosen_reward"])
                tracking["val_rejected_rewards"].append(res["val_rejected_reward"])
                tracking["tokens_seen"].append(tokens_seen)

                train_reward_margin = (
                    res["train_chosen_reward"] - res["train_rejected_reward"]
                )
                val_reward_margin = (
                    res["val_chosen_reward"] - res["val_rejected_reward"]
                )

                print(
                    f"Ep: {epoch+1} (Step {global_step:06d})"
                    f"Train loss {res['train_loss']:.3f}, val loss: {res['val_loss']:.3f},"
                    f"Train reward margins {train_reward_margin:.3f} "
                    f"Val reward margin: {val_reward_margin:.3f}"
                )

                generate_model_output(
                    model=policy_model,
                    tokenizer=tokenizer,
                    data=start_context,
                    device=loss.device,
                )

    return tracking
