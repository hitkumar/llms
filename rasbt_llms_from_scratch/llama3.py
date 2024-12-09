import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from gpt_model import assign
from llama_modules import compute_rope, FeedForward, RMSNorm, SublayerConnection

from model_utils import free_pytorch_memory


def apply_scaling(
    freqs: torch.tensor,
    scale_factor: float,
    low_freq_factor: float,
    high_freq_factor: float,
    old_context_len: int,
) -> torch.tensor:
    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    new_freqs = []
    for i, freq in enumerate(freqs):
        wavelen = 2 * math.pi / freq
        if wavelen < high_freq_wavelen:
            new_freqs.append(freq)
        elif wavelen > low_freq_wavelen:
            new_freqs.append(freq / scale_factor)
        else:
            assert low_freq_wavelen != high_freq_wavelen
            smooth = (old_context_len / wavelen - low_freq_factor) / (
                high_freq_factor - low_freq_factor
            )
            new_freqs.append((1 - smooth) * freq / scale_factor + smooth * freq)

    return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)


def precompute_rope_params(
    head_dim, theta_base=10_000, context_length=4096, freq_config=None
):
    assert head_dim % 2 == 0, "Embedding dimension should be even"
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2).float() / head_dim))

    if freq_config is not None:
        inv_freq = apply_scaling(
            inv_freq,
            scale_factor=freq_config["factor"],
            low_freq_factor=freq_config["low_freq_factor"],
            high_freq_factor=freq_config["high_freq_factor"],
            old_context_len=freq_config["original_context_length"],
        )
    positions = torch.arange(context_length)

    angles = positions[:, None] * inv_freq[None, :]  # [context_length, head_dim/2]
    angles = torch.cat([angles, angles], dim=1)  # [context_length, head_dim]
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin


class SharedBuffers:
    _buffers = {}

    @staticmethod
    def get_buffers(
        context_length, head_dim, rope_base, freq_config, dtype=torch.float32
    ):
        key = (
            context_length,
            head_dim,
            rope_base,
            tuple(freq_config.values()) if freq_config else freq_config,
            dtype,
        )
        if key not in SharedBuffers._buffers:
            mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
            cos, sin = precompute_rope_params(
                head_dim, rope_base, context_length, freq_config
            )
            if dtype is not None:
                cos = cos.to(dtype)
                sin = sin.to(dtype)

            SharedBuffers._buffers[key] = (mask, cos, sin)

        return SharedBuffers._buffers[key]


class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        d_in,
        d_out,
        context_length,
        num_heads,
        num_kv_groups,
        rope_base=10_000,
        rope_config=None,
        dtype=None,
    ):
        super().__init__()
        assert d_out % num_heads == 0
        assert num_heads % num_kv_groups == 0

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_key = nn.Linear(
            d_in, num_kv_groups * self.head_dim, bias=False, dtype=dtype
        )
        self.W_value = nn.Linear(
            d_in, num_kv_groups * self.head_dim, bias=False, dtype=dtype
        )
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups

        # not grouped
        self.W_query = nn.Linear(d_in, d_out, bias=False, dtype=dtype)
        self.out_proj = nn.Linear(d_in, d_out, bias=False, dtype=dtype)

        mask, cos, sin = SharedBuffers.get_buffers(
            context_length, self.head_dim, rope_base, rope_config, dtype
        )
        self.register_buffer("mask", mask)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        queries = self.W_query(x)  # [b, num_tokens, d_out]
        keys = self.W_key(x)  # [b, num_tokens, num_kv_groups * head_dim]
        values = self.W_value(x)  # [b, num_tokens, num_kv_groups * head_dim]

        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(
            1, 2
        )  # [b, num_heads, num_tokens, head_dim]
        keys = keys.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(
            1, 2
        )  # [b, num_kv_groups, num_tokens, head_dim]
        values = values.view(
            b, num_tokens, self.num_kv_groups, self.head_dim
        ).transpose(
            1, 2
        )  # [b, num_kv_groups, num_tokens, head_dim]

        # Apply ROPE
        keys = compute_rope(keys, self.cos, self.sin)
        queries = compute_rope(queries, self.cos, self.sin)

        # [b, num_heads, num_tokens, head_dim]
        keys = keys.repeat_interleave(self.group_size, dim=1)
        values = values.repeat_interleave(self.group_size, dim=1)

        # [b, num_heads, num_tokens, head_dim] [b, num_heads, head_dim, num_tokens] -> [b, num_heads, num_tokens, num_tokens]
        attn_scores = torch.matmul(queries, keys.transpose(2, 3))
        attn_scores = attn_scores / self.head_dim**0.5
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # [b, num_heads, num_tokens, num_tokens]
        attn_weights = torch.softmax(attn_scores, dim=-1)
        # [b, num_heads, num_tokens, head_dim]
        context_vec = (
            (attn_weights @ values).transpose(1, 2).reshape(b, num_tokens, self.d_out)
        )
        return self.out_proj(context_vec)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = GroupedQueryAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            num_kv_groups=cfg["n_kv_groups"],
            rope_base=cfg["rope_base"],
            rope_config=cfg["rope_freq"],
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


def rescale_theta(theta_old, context_length_old, context_length_new):
    scaling_factor = context_length_new / context_length_old
    theta_new = theta_old * scaling_factor
    return theta_new


class Llama3Model(nn.Module):
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

    def from_pretrained(self, param_config, params):
        self.tok_emb.weight = assign(
            self.tok_emb.weight, params["model.embed_tokens.weight"]
        )

        for l in range(param_config["n_layers"]):
            # Load att weights
            self.trf_blocks[l].att.W_query.weight = assign(
                self.trf_blocks[l].att.W_query.weight,
                params[f"model.layers.{l}.self_attn.q_proj.weight"],
            )
            self.trf_blocks[l].att.W_key.weight = assign(
                self.trf_blocks[l].att.W_key.weight,
                params[f"model.layers.{l}.self_attn.k_proj.weight"],
            )
            self.trf_blocks[l].att.W_value.weight = assign(
                self.trf_blocks[l].att.W_value.weight,
                params[f"model.layers.{l}.self_attn.v_proj.weight"],
            )
            self.trf_blocks[l].att.out_proj.weight = assign(
                self.trf_blocks[l].att.out_proj.weight,
                params[f"model.layers.{l}.self_attn.o_proj.weight"],
            )
            self.trf_blocks[l].sublayer1.norm.weight = assign(
                self.trf_blocks[l].sublayer1.norm.weight,
                params[f"model.layers.{l}.input_layernorm.weight"],
            )

            # Load FF weights
            self.trf_blocks[l].ff.fc1.weight = assign(
                self.trf_blocks[l].ff.fc1.weight,
                params[f"model.layers.{l}.mlp.gate_proj.weight"],
            )
            self.trf_blocks[l].ff.fc2.weight = assign(
                self.trf_blocks[l].ff.fc2.weight,
                params[f"model.layers.{l}.mlp.up_proj.weight"],
            )
            self.trf_blocks[l].ff.fc3.weight = assign(
                self.trf_blocks[l].ff.fc3.weight,
                params[f"model.layers.{l}.mlp.down_proj.weight"],
            )
            self.trf_blocks[l].sublayer2.norm.weight = assign(
                self.trf_blocks[l].sublayer2.norm.weight,
                params[f"model.layers.{l}.post_attention_layernorm.weight"],
            )

        # Load output layer weights
        self.final_norm.weight = assign(
            self.final_norm.weight, params["model.norm.weight"]
        )

        if "lm_head.weight" in params.keys():
            self.out_head.weight = assign(
                self.out_head.weight, params["lm_head.weight"]
            )
        else:
            # weight tying
            self.out_head.weight = assign(
                self.out_head.weight, params["model.embed_tokens.weight"]
            )


import os
from pathlib import Path

import tiktoken
from tiktoken.load import load_tiktoken_bpe


class Tokenizer:
    def __init__(self, model_path):
        assert os.path.isfile(model_path), f"Model file {model_path} not found"
        mergeable_ranks = load_tiktoken_bpe(model_path)

        self.special_tokens = {
            "<|begin_of_text|>": 128000,
            "<|end_of_text|>": 128001,
            "<|start_header_id|>": 128006,
            "<|end_header_id|>": 128007,
            "<|eot_id|>": 128009,
        }
        self.special_tokens.update(
            {
                f"<|reserved_{i}|>": 128002 + i
                for i in range(256)
                if (128002 + i) not in self.special_tokens.values()
            }
        )

        self.model = tiktoken.Encoding(
            name=Path(model_path).name,
            pat_str=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
            mergeable_ranks=mergeable_ranks,
            special_tokens=self.special_tokens,
        )

    def encode(
        self, text, bos=False, eos=False, allowed_special=set(), disallowed_special=()
    ):
        tokens = []
        if bos:
            tokens.append(self.special_tokens["<|begin_of_text|>"])

        tokens += self.model.encode(
            text, allowed_special=allowed_special, disallowed_special=disallowed_special
        )

        if eos:
            tokens.append(self.special_tokens["<|end_of_text|>"])

        return tokens

    def decode(self, tokens):
        return self.model.decode(tokens)


# Llama 3 series model configs

LLAMA3_CONFIG_8B = {
    "vocab_size": 128_256,
    "context_length": 8192,
    "emb_dim": 4096,
    "n_heads": 32,
    "n_layers": 32,
    "hidden_dim": 14_336,
    "n_kv_groups": 8,
    "rope_base": 500_000,
    "rope_freq": None,
    "dtype": torch.bfloat16,
}

LLAMA31_CONFIG_8B = {
    "vocab_size": 128_256,
    "context_length": 131_072,  # increased
    "emb_dim": 4096,
    "n_heads": 32,
    "n_layers": 32,
    "hidden_dim": 14_336,
    "n_kv_groups": 8,
    "rope_base": 500_000,
    "dtype": torch.bfloat16,
    "rope_freq": {
        "factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_context_length": 8192,
    },
}

LLAMA32_CONFIG_1B = {
    "vocab_size": 128_256,
    "context_length": 131_072,
    "emb_dim": 2048,
    "n_heads": 32,
    "n_layers": 16,
    "hidden_dim": 8192,
    "n_kv_groups": 8,
    "rope_base": 500_000,
    "dtype": torch.bfloat16,
    "rope_freq": {
        "factor": 32.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_context_length": 8192,
    },
}

LLAMA32_CONFIG_1B = {
    "vocab_size": 128_256,
    "context_length": 131_072,
    "emb_dim": 2048,
    "n_heads": 32,
    "n_layers": 16,
    "hidden_dim": 8192,
    "n_kv_groups": 8,
    "rope_base": 500_000,
    "dtype": torch.bfloat16,
    "rope_freq": {
        "factor": 32.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_context_length": 8192,
    },
}


def rescale_context_length(cfg, context_length=8192):
    old_context_length = cfg["context_length"]
    cfg["context_length"] = context_length

    cfg["rope_base"] = rescale_theta(
        cfg["rope_base"],
        old_context_length,
        cfg["context_length"],
    )

    print("New RoPE theta:", cfg["rope_base"])
    return cfg


from gpt_model import generate, text_to_token_ids, token_ids_to_text
from safetensors.torch import load_file

if __name__ == "__main__":
    llama32_1b_model_path = "/home/htkumar/llms/Llama-3.2-1B"
    tokenizer = Tokenizer(
        os.path.join(llama32_1b_model_path, "original", "tokenizer.model")
    )
    cfg = rescale_context_length(LLAMA32_CONFIG_1B.copy(), 8192)

    model = Llama3Model(cfg)
    weights_file = os.path.join(llama32_1b_model_path, "model.safetensors")
    current_weights = load_file(weights_file)
    model.from_pretrained(cfg, current_weights)
    print("model loaded")

    device = torch.device("cuda")
    model.to(device)
    model.eval()

    torch.manual_seed(123)

    with torch.no_grad():
        token_ids = generate(
            model=model,
            idx=text_to_token_ids("Every effort", tokenizer).to(device),
            max_new_tokens=30,
            context_size=cfg["context_length"],
            top_k=1,
            temperature=0.0,
        )

        print(token_ids_to_text(token_ids, tokenizer))

    free_pytorch_memory(model)
