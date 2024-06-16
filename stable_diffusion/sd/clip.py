import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

class CLIPEmbedding(nn.Module):
    def __init__(self, n_vocab: int, n_embd: int, n_token: int):
        super().__init__()
        
        self.token_embedding = nn.Embedding(n_vocab, n_embd)
        self.position_embedding = nn.Parameter(torch.zeros((n_token, n_embd)))
    
    def forward(self, tokens):
        # (B, S) -> (B, S, D)
        x = self.token_embedding(tokens)
        x += self.position_embedding
        return x

class CLIPLayer(nn.Module):
    def __init__(self, n_head: int, n_embd: int):
        super().__init__()
        self.layernorm_1 = nn.LayerNorm(n_embd)
        self.attention = SelfAttention(n_head, n_embd)
        self.layernorm_2 = nn.LayerNorm(n_embd)

        self.linear_1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear_2 = nn.Linear(4 * n_embd, n_embd)
    
    def forward(self, x):
        x = x + self.attention(self.layernorm_1(x), causal_mask=True)
        residue = x
        x = self.linear_1(self.layernorm_2(x))
        x = x * torch.sigmoid(1.702 * x) # QuickGELU activation function
        return residue + self.linear_2(x)

class CLIP(nn.Module):
    '''
    Returns an embedding for every token in the input sequence.
    '''
    def __init__(self):
        super().__init__()
        self.embedding = CLIPEmbedding(49408, 768, 77)
        self.layers = nn.ModuleList([
            CLIPLayer(12, 768) for i in range(12)
        ])
        self.layernorm = nn.LayerNorm(768)
    
    def forward(self, tokens):
        tokens = tokens.type(torch.long)

        # (B, S) -> (B, S, Dim)
        state = self.embedding(tokens)
        for layer in self.layers:
            state = layer(state)
        output = self.layernorm(state)

        return output