import inspect

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.ticker import MaxNLocator
from torch.nn import functional as F


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    # This is nn.GELU(approximation='tanh') as explained by Karpathy
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(
                    torch.sqrt(torch.tensor(2.0 / torch.pi))
                    * (x + 0.044715 * torch.pow(x, 3))
                )
            )
        )


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by n_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)  # shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)

        # transpose (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # dot product for each head
        attn_scores = torch.matmul(
            queries, keys.transpose(2, 3)
        )  # (b, num_heads, num_tokens, num_tokens) # double check this
        mask = self.mask[:num_tokens, :num_tokens].bool()
        attn_scores.masked_fill_(mask, -torch.inf)

        scaled_attn_scores = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        scaled_attn_scores = self.dropout(scaled_attn_scores)

        # (b, num_heads, num_tokens, num_tokens) *
        # (b, num_heads, num_tokens, head_dim) -> (b, num_heads, num_tokens, head_dim) -> (b, num_tokens, num_heads, head_dim)
        context_vec = (scaled_attn_scores @ values).transpose(1, 2)
        context_vec = context_vec.reshape(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)
        return context_vec


class SublayerConnection(nn.Module):
    """
    Apply LN and residual connection.
    """

    def __init__(self, size, dropout):
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
        )
        self.ff = FeedForward(cfg)
        # self.norm1 = LayerNorm(cfg['emb_dim'])
        # self.norm2 = LayerNorm(cfg['emb_dim'])
        # self.drop_resid = nn.Dropout(cfg['drop_rate'])
        self.sublayer1 = SublayerConnection(cfg["emb_dim"], cfg["drop_rate"])
        self.sublayer2 = SublayerConnection(cfg["emb_dim"], cfg["drop_rate"])

    def forward(self, x):
        # might have some interesting consequences when we load weights
        # attention block
        x = self.sublayer1(x, self.att)
        # FF block
        x = self.sublayer2(x, self.ff)
        return x


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_ln = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        _, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # [batch_size, seq_len, emb_size]
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_ln(x)
        logits = self.out_head(x)
        return logits


class GPTModelWithTargets(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_ln = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx, targets=None):
        _, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # [batch_size, seq_len, emb_size]
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_ln(x)
        logits = self.out_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1))
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # Find all parameters that require gradients
        params_dict = {pn: p for pn, p in self.named_parameters()}
        params_dict = {pn: p for pn, p in params_dict.items() if p.requires_grad}

        # Now divide params in 2 groups: ones that need weight decay and other that don't
        decay_params = [p for n, p in params_dict.items() if p.dim() >= 2]
        nondecay_params = [p for n, p in params_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nondecay_params, "weight_decay": 0.0},
        ]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_nondecay_params = sum(p.numel() for p in nondecay_params)

        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused
        )
        return optimizer


# Util functions
def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx is (B, T) array of indices in current context

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)  # (B, T, vocab_size)

        logits = logits[:, -1, :]  # (batch, vocab_size)
        idx_next = torch.argmax(logits, dim=1, keepdim=True)  # (batch, 1)

        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx


def generate(
    model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None
):

    # TODO: Verify that this generates the same outputs as in master repo
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)

        # shape is (B, V)
        logits = logits[:, -1, :]

        if top_k is not None:
            topk_probs, _ = torch.topk(logits, top_k)
            min_val = topk_probs[:, -1]
            logits = torch.where(
                logits < min_val,
                torch.tensor(-float("inf")).to(logits.device),
                logits,
            )

        if temperature > 0.0:
            logits = logits / temperature

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=1, keepdim=True)  # (batch, 1)

        if idx_next == eos_id:
            break

        idx = torch.cat((idx, idx_next), dim=1)  # (B, T + 1)

    return idx


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), target_batch.view(-1))
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break

    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(val_loader, model, device, eval_iter)

    model.train()
    return train_loss, val_loss


def generate_and_print_sample(
    model, tokenizer, device, start_context, max_new_tokens=50
):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model,
            idx=encoded,
            max_new_tokens=max_new_tokens,
            context_size=context_size,
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(f"decoded text is {decoded_text}")

    model.train()


def train_model_simple(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs,
    eval_freq,
    eval_iter,
    start_context,
    tokenizer,
    max_new_tokens=50,
):
    # initialize losses
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    # main training loop
    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            # eval at certain intervals
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(
                    f"Epoch: {epoch + 1}, step: {global_step:06d}, train loss: {train_loss:.3f}, val loss: {val_loss:.3f}"
                )

        generate_and_print_sample(
            model, tokenizer, device, start_context, max_new_tokens
        )

    return train_losses, val_losses, track_tokens_seen


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses, label="loss"):
    fig, ax1 = plt.subplots(figsize=(5, 3))

    ax1.plot(epochs_seen, train_losses, label="Training losses")
    ax1.plot(epochs_seen, val_losses, linestyle="-", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label.capitalize())
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses)
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()
    plt.show()


def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch, Left: {left.shape}, right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    encoded = torch.tensor(encoded).unsqueeze(0)
    return encoded


def token_ids_to_text(token_ids, tokenizer):
    tokens = token_ids.squeeze(0)
    return tokenizer.decode(tokens.tolist())


def load_weights_into_gpt(gpt: GPTModel, params):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params["wpe"])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params["wte"])

    for b in range(len(params["blocks"])):
        # Multi head Attention layer
        q_w, k_w, v_w = np.split(params["blocks"][b]["attn"]["c_attn"]["w"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T
        )
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T
        )
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T
        )

        q_b, k_b, v_b = np.split(params["blocks"][b]["attn"]["c_attn"]["b"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b
        )
        gpt.trf_blocks[b].att.W_key.bias = assign(gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b
        )

        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T,
        )
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"],
        )

        # FF layer
        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T,
        )
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias, params["blocks"][b]["mlp"]["c_fc"]["b"]
        )
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T,
        )
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"],
        )
        # Norm layers
        gpt.trf_blocks[b].sublayer1.norm.scale = assign(
            gpt.trf_blocks[b].sublayer1.norm.scale, params["blocks"][b]["ln_1"]["g"]
        )
        gpt.trf_blocks[b].sublayer1.norm.shift = assign(
            gpt.trf_blocks[b].sublayer1.norm.shift, params["blocks"][b]["ln_1"]["b"]
        )
        gpt.trf_blocks[b].sublayer2.norm.scale = assign(
            gpt.trf_blocks[b].sublayer2.norm.scale, params["blocks"][b]["ln_2"]["g"]
        )
        gpt.trf_blocks[b].sublayer2.norm.shift = assign(
            gpt.trf_blocks[b].sublayer2.norm.shift, params["blocks"][b]["ln_2"]["b"]
        )

        # Final norm and output layer weight
        gpt.final_ln.scale = assign(gpt.final_ln.scale, params["g"])
        gpt.final_ln.shift = assign(gpt.final_ln.shift, params["b"])
        # This is what Karpathy was saying, weights are shared between wte and final out head
        gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])
