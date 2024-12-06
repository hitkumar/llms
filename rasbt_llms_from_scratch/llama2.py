import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F
from gpt_model import assign, generate, text_to_token_ids, token_ids_to_text
from llama_modules import compute_rope, FeedForward, RMSNorm

LLAMA2_CONFIG_7B = {
    "vocab_size": 32000,  # Vocabulary size
    "context_length": 4096,  # Context length
    "emb_dim": 4096,  # Embedding dimension
    "n_heads": 32,  # Number of attention heads
    "n_layers": 32,  # Number of layers
    "hidden_dim": 11008,  # NEW: Size of the intermediate dimension in FeedForward
    "dtype": torch.bfloat16,  # NEW: Lower-precision dtype to reduce memory usage
}


# ROPE Embeddings
def precompute_rope_params(head_dim, theta_base=10_000, context_length=4096):
    assert head_dim % 2 == 0, "Embedding dimension should be even"
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    positions = torch.arange(context_length)

    angles = positions[:, None] * inv_freq[None, :]  # [context_length, head_dim/2]
    angles = torch.cat([angles, angles], dim=1)  # [context_length, head_dim]
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, num_heads, dtype=None):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by n_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=False, dtype=dtype)
        self.W_key = nn.Linear(d_in, d_out, bias=False, dtype=dtype)
        self.W_value = nn.Linear(d_in, d_out, bias=False, dtype=dtype)
        self.out_proj = nn.Linear(d_out, d_out, bias=False, dtype=dtype)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )
        cos, sin = precompute_rope_params(
            head_dim=self.head_dim, context_length=context_length
        )
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

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

        # Apply rope here.
        keys = compute_rope(keys, self.cos, self.sin)
        queries = compute_rope(queries, self.cos, self.sin)

        # dot product for each head
        attn_scores = torch.matmul(
            queries, keys.transpose(2, 3)
        )  # (b, num_heads, num_tokens, num_tokens) # double check this
        mask = self.mask[:num_tokens, :num_tokens].bool()
        attn_scores.masked_fill_(mask, -torch.inf)

        scaled_attn_scores = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)

        # (b, num_heads, num_tokens, num_tokens) *
        # (b, num_heads, num_tokens, head_dim) -> (b, num_heads, num_tokens, head_dim) -> (b, num_tokens, num_heads, head_dim)
        context_vec = (scaled_attn_scores @ values).transpose(1, 2)
        context_vec = context_vec.reshape(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)
        return context_vec


class SublayerConnection(nn.Module):
    """
    Apply RMSNorm and residual connection.
    """

    def __init__(self, size):
        super().__init__()
        self.norm = RMSNorm(size)

    def forward(self, x, sublayer):
        return x + sublayer(self.norm(x))


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dtype=cfg["dtype"],
        )
        self.ff = FeedForward(cfg)
        self.sublayer1 = SublayerConnection(cfg["emb_dim"])
        self.sublayer2 = SublayerConnection(cfg["emb_dim"])

    def forward(self, x):
        # might have some interesting consequences when we load weights
        # attention block
        x = self.sublayer1(x, self.att)
        # FF block
        x = self.sublayer2(x, self.ff)
        return x


class Llama2Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(
            cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"]
        )

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = RMSNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"]
        )

    def forward(self, in_idx, targets=None):
        x = self.tok_emb(in_idx)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
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
        decay_params = [p for _, p in params_dict.items() if p.dim() >= 2]
        nondecay_params = [p for _, p in params_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nondecay_params, "weight_decay": 0.0},
        ]

        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused
        )
        return optimizer

    def from_pretrained(self, param_config, params):
        self.tok_emb.weight = assign(
            self.tok_emb.weight, params["tok_embeddings.weight"]
        )

        for l in range(param_config["n_layers"]):
            # Load att weights
            self.trf_blocks[l].att.W_query.weight = assign(
                self.trf_blocks[l].att.W_query.weight,
                params[f"layers.{l}.attention.wq.weight"],
            )
            self.trf_blocks[l].att.W_key.weight = assign(
                self.trf_blocks[l].att.W_key.weight,
                params[f"layers.{l}.attention.wk.weight"],
            )
            self.trf_blocks[l].att.W_value.weight = assign(
                self.trf_blocks[l].att.W_value.weight,
                params[f"layers.{l}.attention.wv.weight"],
            )
            self.trf_blocks[l].att.out_proj.weight = assign(
                self.trf_blocks[l].att.out_proj.weight,
                params[f"layers.{l}.attention.wo.weight"],
            )
            self.trf_blocks[l].sublayer1.norm.weight = assign(
                self.trf_blocks[l].sublayer1.norm.weight,
                params[f"layers.{l}.attention_norm.weight"],
            )

            # Load FF weights
            self.trf_blocks[l].ff.fc1.weight = assign(
                self.trf_blocks[l].ff.fc1.weight,
                params[f"layers.{l}.feed_forward.w1.weight"],
            )
            # For some reason w2 and w3 are provided in the wrong order in the weights file
            self.trf_blocks[l].ff.fc2.weight = assign(
                self.trf_blocks[l].ff.fc2.weight,
                params[f"layers.{l}.feed_forward.w3.weight"],
            )
            self.trf_blocks[l].ff.fc3.weight = assign(
                self.trf_blocks[l].ff.fc3.weight,
                params[f"layers.{l}.feed_forward.w2.weight"],
            )
            self.trf_blocks[l].sublayer2.norm.weight = assign(
                self.trf_blocks[l].sublayer2.norm.weight,
                params[f"layers.{l}.ffn_norm.weight"],
            )

        # Load output layer weights
        self.final_norm.weight = assign(self.final_norm.weight, params["norm.weight"])
        self.out_head.weight = assign(self.out_head.weight, params["output.weight"])


import sentencepiece as spm


class LLamaTokenizer:
    def __init__(self, tokenizer_file):
        sp = spm.SentencePieceProcessor()
        sp.load(tokenizer_file)
        self.tokenizer = sp

    def encode(self, text):
        return self.tokenizer.encode_as_ids(text)

    def decode(self, ids):
        return self.tokenizer.decode_pieces(ids)


if __name__ == "__main__":
    model = Llama2Model(LLAMA2_CONFIG_7B)
    weights_file = "/home/htkumar/llms/llama-2-7b-chat/consolidated.00.pth"
    weights = torch.load(weights_file, weights_only=True)
    model.from_pretrained(LLAMA2_CONFIG_7B, weights)

    device = torch.device("cuda")
    model.to(device)
    model.eval()
    print("Loaded LLama 2 model...")

    tokenizer_file = "/home/htkumar/llms/llama-2-7b/tokenizer.model"
    tokenizer = LLamaTokenizer(tokenizer_file)

    with torch.no_grad():
        token_ids = generate(
            model=model,
            idx=text_to_token_ids("What do llamas eat?", tokenizer).to(device),
            max_new_tokens=30,
            context_size=LLAMA2_CONFIG_7B["context_length"],
            top_k=1,
            temperature=0.0,
        )

    print(f"generated text is {token_ids_to_text(token_ids, tokenizer)}")

    # cleanup
    del model
    torch.cuda.empty_cache()
