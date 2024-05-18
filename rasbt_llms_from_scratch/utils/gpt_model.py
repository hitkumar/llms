import torch
import torch.nn as nn
import numpy as np

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
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
            (x + 0.044715 * torch.pow(x, 3))
        ))

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg['emb_dim'], 4 * cfg['emb_dim']),
            GELU(),
            nn.Linear(4 * cfg['emb_dim'], cfg['emb_dim'])
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
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))
    
    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x) # shape: (b, num_tokens, d_out)
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
        attn_scores = torch.matmul(queries, keys.transpose(2, 3)) # (b, num_heads, num_tokens, num_tokens) # double check this
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
    '''
    Apply LN and residual connection.
    '''
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
            d_in=cfg['emb_dim'],
            d_out=cfg['emb_dim'],
            context_length=cfg['context_length'],
            num_heads=cfg['n_heads'],
            dropout=cfg['drop_rate'],
            qkv_bias=cfg['qkv_bias']
        )
        self.ff = FeedForward(cfg)
        # self.norm1 = LayerNorm(cfg['emb_dim'])
        # self.norm2 = LayerNorm(cfg['emb_dim'])
        # self.drop_resid = nn.Dropout(cfg['drop_rate'])
        self.sublayer1 = SublayerConnection(cfg['emb_dim'], cfg['drop_rate'])
        self.sublayer2 = SublayerConnection(cfg['emb_dim'], cfg['drop_rate'])
    
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
        self.tok_emb = nn.Embedding(cfg['vocab_size'], cfg['emb_dim'])
        self.pos_emb = nn.Embedding(cfg['context_length'], cfg['emb_dim'])
        self.drop_emb = nn.Dropout(cfg['drop_rate'])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg['n_layers'])]
        )
        self.final_ln = LayerNorm(cfg['emb_dim'])
        self.out_head = nn.Linear(cfg['emb_dim'], cfg['vocab_size'], bias=False)
    
    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds # [batch_size, seq_len, emb_size]
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_ln(x)
        logits = self.out_head(x)
        return logits

# Util functions
def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx is (B, T) array of indices in current context

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond) # (B, T, vocab_size)
        
        logits = logits[:, -1, :] # (batch, vocab_size)
        idx_next = torch.argmax(logits, dim=1, keepdim=True) # (batch, 1)

        idx = torch.cat((idx, idx_next), dim=1) # (batch, n_tokens+1)
    
    return idx

# Util functions
def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx is (B, T) array of indices in current context

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond) # (B, T, vocab_size)
        
        logits = logits[:, -1, :] # (batch, vocab_size)
        idx_next = torch.argmax(logits, dim=1, keepdim=True) # (batch, 1)

        idx = torch.cat((idx, idx_next), dim=1) # (batch, n_tokens+1)
    
    return idx

def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch, Left: {left.shape}, right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded = torch.tensor(encoded).unsqueeze(0)
    return encoded

def token_ids_to_text(token_ids, tokenizer):
    tokens = token_ids.squeeze(0)
    return tokenizer.decode(tokens.tolist())

def load_weights_into_gpt(gpt: GPTModel, params):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])
    
    for b in range(len(params['blocks'])):
        # Multi head Attention layer
        q_w, k_w, v_w = np.split(
            params['blocks'][b]['attn']['c_attn']['w'], 3, axis=-1
        )
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T
        )
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T
        )
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T
        )

        q_b, k_b, v_b = np.split(
            params['blocks'][b]['attn']['c_attn']['b'], 3, axis=-1
        )
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b
        )
        gpt.trf_blocks[b].att.W_key.bias = assign(
            gpt.trf_blocks[b].att.W_key.bias, k_b
        )
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b
        )

        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight,
            params['blocks'][b]['attn']['c_proj']['w'].T
        )
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias,
            params['blocks'][b]['attn']['c_proj']['b']
        )

        # FF layer
        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight,
            params['blocks'][b]['mlp']['c_fc']['w'].T
        )
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias,
            params['blocks'][b]['mlp']['c_fc']['b']
        )
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight,
            params['blocks'][b]['mlp']['c_proj']['w'].T
        )
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias,
            params['blocks'][b]['mlp']['c_proj']['b']
        )
        # Norm layers
        gpt.trf_blocks[b].sublayer1.norm.scale = assign(
            gpt.trf_blocks[b].sublayer1.norm.scale,
            params['blocks'][b]['ln_1']['g']
        )
        gpt.trf_blocks[b].sublayer1.norm.shift = assign(
            gpt.trf_blocks[b].sublayer1.norm.shift,
            params['blocks'][b]['ln_1']['b']
        )
        gpt.trf_blocks[b].sublayer2.norm.scale = assign(
            gpt.trf_blocks[b].sublayer2.norm.scale,
            params['blocks'][b]['ln_2']['g']
        )
        gpt.trf_blocks[b].sublayer2.norm.shift = assign(
            gpt.trf_blocks[b].sublayer2.norm.shift,
            params['blocks'][b]['ln_2']['b']
        )

        # Final norm and output layer weight
        gpt.final_ln.scale = assign(gpt.final_ln.scale, params['g'])
        gpt.final_ln.shift = assign(gpt.final_ln.shift, params['b'])
        gpt.out_head.weight = assign(gpt.out_head.weight, params['wte'])
