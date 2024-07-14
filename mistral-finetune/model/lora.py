import torch
import torch.nn as nn
import torch.nn.functional as F

from  typing import NamedTuple

class LoRALinear(nn.Module):
    """
    Try DORA as well.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        scaling: float,
        dropout: float,
        bias: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scaling = scaling
        self.dropout = nn.Dropout(dropout)
        assert not bias
        self.bias = bias

        self.lora_A = nn.Linear(in_features, rank, bias=self.bias)
        self.lora_B = nn.Linear(rank, out_features, bias=self.bias)
        self.frozen_w = nn.Linear(in_features, out_features, bias=self.bias)

        def ignore_missing_keys(m: nn.Module, incompatible_keys: NamedTuple):
            incompatible_keys.missing_keys[:] = []

        self.register_load_state_dict_post_hook(ignore_missing_keys)

    def forward(self, x: torch.tensor):
        lora_res = self.lora_B(self.lora_A(self.dropout(x)))
        return self.frozen_w(x) + lora_res * self.scaling
    
    def __repr__(self) -> str:
        return f"LorA Linear(in_features: {self.in_features}, out_features: {self.out_features}, rank: {self.rank}, scaling: {self.scaling}, dropout: {self.dropout})"

    def merge_weight(self):
        with torch.no_grad():
            down_weight = self.lora_A.weight
            up_weight = self.lora_B.weight
            lora_weight = up_weight.mm(down_weight) * self.scaling

            weight += self.frozen_w.weight
            return weight

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        key_name = prefix + "weight"
        if key_name in state_dict:
            w_ref = state_dict[key_name]

            self.frozen_w.load_state_dict({"weight": w_ref}, assign=True)
