import torch
import torch.nn as nn
import torch.nn.functional as F


# TODO: Check the dtype calculation here.
class RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.emb_dim = emb_dim
        self.weight = nn.Parameter(torch.ones(emb_dim)).float()

    def forward(self, x):
        rms_mean = x.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x * torch.rsqrt(rms_mean + self.eps)
        return (x_norm * self.weight).to(dtype=x.dtype)


class SiLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(
            cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False
        )
        self.fc2 = nn.Linear(
            cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False
        )
        self.fc3 = nn.Linear(
            cfg["hidden_dim"], cfg["emb_dim"], dtype=cfg["dtype"], bias=False
        )
        self.silu = SiLU()

    def forward(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        return self.fc3(self.silu(x_fc1) * x_fc2)


class SublayerConnection(nn.Module):
    """
    Apply RMSNorm and residual connection.
    """

    def __init__(self, size):
        super().__init__()
        self.norm = RMSNorm(size)

    def forward(self, x, sublayer):
        return x + sublayer(self.norm(x))


def compute_rope(x, cos, sin):
    # x: (batch_size, num_heads, seq_len, head_dim)
    batch_size, num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Embedding dimension should be even"
    x1 = x[..., : head_dim // 2]
    x2 = x[..., head_dim // 2 :]
    rotated = torch.cat((-x2, x1), dim=-1)
    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)

    x_rotated = (x * cos) + (rotated * sin)
    return x_rotated.to(dtype=x.dtype)
