import torch
from torch import nn
import torch.nn.functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, n_heads, d_embed, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        # one matrix for Q, K and V matrices
        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        # Wo matrix
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads
    
    def forward(self, x, causal_mask=False):
        # x: (B, S_Len, Dim)

        input_shape = x.shape
        batch_size, seq_len, d_embed = input_shape
        interim_shape = (batch_size, seq_len, self.n_heads, self.d_head)

        # (B, S_Len, Dim) -> (B, S_Len, 3 * Dim) -> 3 tensors of shape (B, S_Len, Dim)
        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        # (B, S_Len, Dim) -> (B, n_heads, s_len, d_head)
        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        # (B, n_heads, s_len, d_head) * (B, n_heads, d_head, s_len) -> (B, n_heads, s_len, s_len)
        weight = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.d_head)

        # Apply the causal mask
        if causal_mask:
            mask = torch.ones_like(weight, dtype=torch.bool).triu(diagonal=1)
            weight.masked_fill_(mask, -torch.inf)
        
        weight = F.softmax(weight, dim=-1)

        # (B, n_heads, s_len, s_len) * (B, n_heads, s_len, d_head) -> (B, n_heads, s_len, d_head)
        self_attention_out = weight @ v
        self_attention_out = self_attention_out.transpose(1, 2).contiguous()
        self_attention_out = self_attention_out.view(input_shape)
        self_attention_out = self.out_proj(self_attention_out)

        return self_attention_out

class CrossAttention(nn.Module):
    def __init__(self, n_heads, d_embed, d_cross, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.q_proj = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.k_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.v_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // self.n_heads

    def forward(self, latent: torch.tensor, context: torch.tensor):
        '''
        query comes from the latent and k, v comes from context
        latent_shape is (B, S_Q, d_embed)
        context shape is (B, S_KV, d_cross)
        '''
        latent_shape = latent.shape
        batch_size, q_seq_len, d_embed = latent_shape
        interim_shape = (batch_size, -1, self.n_heads, self.d_head)

        # (B, S_Q, d_embed) -> (B, S_Q, d_embed) -> (B, S_q, n_heads, d_head) -> (B, n_heads, S_q, d_head)
        q = self.q_proj(latent).view(interim_shape).transpose(1, 2)
        # (B, S_KV, d_cross) -> (B, S_KV, d_embed) -> (B, S_KV, n_heads, d_head) -> (B, n_heads, S_KV, d_head)
        v = self.v_proj(context).view(interim_shape).transpose(1, 2)
        k = self.k_proj(context).view(interim_shape).transpose(1,2)

        # (B, n_heads, S_q, d_head) * (B, n_heads, d_head, S_kv) -> (B, n_heads, s_q, s_kv)
        weight = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.d_head)
        # (B, n_heads, s_q, s_kv)
        weight = F.softmax(weight, dim=-1)

        # (B, n_heads, s_q, s_kv) * (B, n_heads, S_KV, d_head) -> (B, n_heads, s_q, d_head)
        self_attention_out = weight @ v
        # (B, s_q, n_heads, d_head)
        # TODO: Find out if contiguous is needed.
        self_attention_out = self_attention_out.transpose(1, 2).contiguous()
        # (B, s_q, d_embed)
        self_attention_out = self_attention_out.view(latent_shape)
        # (B, s_q, d_embed)
        self_attention_out = self.out_proj(self_attention_out)
        return self_attention_out