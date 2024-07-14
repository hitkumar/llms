from typing import List
import torch
import torch.nn as nn
from .args import MoeArgs
from torch.nn import functional as F

class MoeLayer(nn.Module):
    """
    TODO: Check the input to this layer.
    """
    def __init__(self, experts: List[nn.Module], gate: nn.Module, moe_args: MoeArgs):
        super().__init__()
        assert len(experts) > 0
        self.experts = nn.ModuleList(experts)
        self.gate = gate
        self.moe_args = moe_args
    
    def forward(self, inputs: torch.tensor):
        # Shape [B * L, emb_dim]
        inputs_squashed = inputs.view(-1, inputs.shape[-1])
        # Shape [B * L, num_experts]
        gate_logits = self.gate(inputs_squashed)
        # Shape [B * L, num_experts_per_tok]
        weights, selected_experts = torch.topk(gate_logits, self.moe_args.num_experts_per_tok)
        weights = F.softmax(weights, dim=1, dtype=torch.float).type_as(inputs)

        results = torch.zeros_like(inputs_squashed)
        for i, expert in enumerate(self.experts):
            batch_idx, nth_expert = torch.where(selected_experts == i)
            results[batch_idx] += weights[batch_idx, nth_expert, None] * expert(inputs_squashed[batch_idx])
        
        return results.view(inputs.shape)
