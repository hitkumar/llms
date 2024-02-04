import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads:int = 32 # number of heads for the queries
    n_kv_heads: Optional[int] = None # number of heads for k,v
    vocab_size: int = -1 # temp value for now, set when tokenizer is loaded
    # hidden dimension of FFN layer
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    # Needed for KV cahche
    max_batch_size: int = 32
    max_seq_len: int = 2048

    device: str = None

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # gamma
        self.weight = nn.Parameter(torch.ones(dim))
    
    def _norm(self, x: torch.Tensor):
        # (B, seq_len, Dim) * (B, seq_len, 1) => (B, seq_len, Dim)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x: torch.Tensor):
        # (Dim) * (B, seq_len, Dim)
        return self.weight * self._norm(x.float()).type_as(x)

def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, device: str, theta: float = 10000.0):
    assert head_dim %2 == 0, "Dimension must be even" # from paper
    # build the theta parameter
    # Shape: (head_dim / 2)
    theta_numerator =  torch.arange(0, head_dim, 2).float()
    # shape: (head_dim, 2)
    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device)
    # construct the positions (the m parameter)
    # shape: (seq_len)
    m = torch.arange(seq_len, device=device)
    # multiply each theta by each position using outer product
    # shape: (seq_len, head_dim/2)
    freqs = torch.outer(m, theta).float()
    # compute complex numbers in the polar form c = R * exp(i *m * theta) where R = 1 as follows
    # shape: (seq_len, head_dim / 2)
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex

# apply position embeddings to a list of tokens.
def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor, device: str):
    # input (B, seq_len, H, head_dim) -> (B, seq_len, H, head_dim/2)
    # TODO: Play around with this reshape more
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # (seq_len, head_dim/2) -> (1, seq_len, 1, head_dim/2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    # (B, seq_len, H, head_dim/2) * (1, seq_len, 1, head_dim/2) -> (B, seq_len, H, head_dim/2)
    x_rotated = x_complex * freqs_complex
    # (B, seq_len, H, head_dim/2) -> (B, seq_len, H, head_dim/2, 2)
    x_out = torch.view_as_real(x_rotated)
    # (B, seq_len, H, head_dim)
    x_out = x_out.reshape(*x.shape)
    return x_out.type_as(x).to(device)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch_size, seq_len, n_kv_heads, head_dim = x.shape

    if n_rep == 1:
        return x
    else:
        return (
            # (B, seq_len, N_kv_heads, 1, head_dim)
            x[:, :, :, None, :]
            .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
            .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
        )

class SelfAttention(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()

        # number of heads for key and values
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        # number of heads for the queries
        self.n_heads_q = args.n_heads
        
        # number of times to repeat key and values to match the queries
        self.n_rep = self.n_heads_q // self.n_kv_heads

        # per head dim
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))
    
    
    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        batch_size, seq_len, _ = x.shape # (B, 1, Dim)

        # Apply W matrices to the input to k, q, and v

        # (B, 1, Dim) -> (B, 1, H_q * head_dim)
        xq = self.wq(x)
        # (B, 1, dim) -> (B, 1, H_kv * head_dim)
        xk = self.wk(x)
        # (B, 1, dim) -> (B, 1, H_kv * head_dim)
        xv = self.wv(x)
        
        # (B, 1, H_q * head_dim) -> (B, 1, H_q, head_dim)
        xq = xq.view(batch_size, seq_len, self.n_heads_q, self.head_dim)
        # (B, 1, H_q * head_dim) -> (B, 1, H_kv, head_dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim) 
        # (B, 1, H_q * head_dim) -> (B, 1, H_kv, head_dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # Same dimension for tensors
        xq = apply_rotary_embeddings(xq, freqs_complex, device=x.device)
        xk = apply_rotary_embeddings(xk, freqs_complex, device=x.device)

        # Replace the entry in cache for this token
        # xk, xv is of dim (n_kv_heads, head_dim)
        self.cache_k[:batch_size, start_pos: start_pos+seq_len] = xk
        self.cache_v[:batch_size, start_pos: start_pos+seq_len] = xv

        # Retrieve the keys and values for multi head attention
        # (B, seq_len_kv, H_kv, head_dim)
        keys = self.cache_k[:batch_size, :start_pos+seq_len]
        values = self.cache_v[:batch_size, :start_pos+seq_len]

        # grouped multi-query attention logic
        # (B, seq_len_kv, h_q, head_dim)
        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)

        # now we proceed with normal attention implementation
        # (xq is the query in kv cache)

        # (B, 1, h_q, head_dim) -> (B, h_q, 1, head_dim)
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # (B, h_q, 1, head_dim) @ (B, h_q, head_dim, seq_len_kv) -> (B, h_q, 1, seq_len_kv)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)

        # (B, h_q, 1, seq_len_kv) @ (B, h_q, seq_len_kv, head_dim) -> (B, h_q, 1, head_dim)
        output = torch.matmul(scores, values)

        # (B, h_q, 1, head_dim) -> (B, 1, h_q, head_dim) -> (B, 1, dim)
        output = (output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1))
        return self.wo(output) # (B, 1, dim) -> (B, 1, dim)

class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        hidden_dim = 4 * args.dim
        hidden_dim = int(2 * hidden_dim / 3)
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
        
        # Round off hidden_dim to nearest multiple of multiple_of parameter
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)
    
    def forward(self, x: torch.Tensor):
        swish = F.silu(self.w1(x))
        x_v = self.w3(x)
        x = swish * x_v
        x = self.w2(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = self.dim // self.n_heads

        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)

        # normalization before self attention
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        #normalization after attention, before feed forward
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        # (B, seq_len, dim) + (B, seq_len, dim) --> (B, seq_len, dim  )
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_complex)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

class Transformer(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()

        assert args.vocab_size != -1, "Vocab size must bve set"
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers 
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)

        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(EncoderBlock(args))
        
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, self.vocab_size, bias=False)

        self.freqs_complex = precompute_theta_pos_frequencies(self.args.dim // self.args.n_heads, self.args.max_seq_len * 2, device = self.args.device)
    
    def forward(self, tokens: torch.Tensor, start_pos: int):
        # input is [B, seq_len]

        batch_size, seq_len = tokens.shape
        assert seq_len == 1, "seq_len must be 1"

        # (B, seq_len) -> (B, seq_len, dim)
        h = self.tok_embeddings(tokens)

        freqs_complex = self.freqs_complex[start_pos: start_pos+seq_len]

        # feed to feed forward encoder layers
        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)
        
        h = self.norm(h)
        output = self.output(h).float()
        return output