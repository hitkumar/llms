"""
Use this to work with HF datasets
"""

import torch
import torch.nn as nn
from modeling_siglip import SiglipVisionConfig

siglip_config = SiglipVisionConfig()
print(siglip_config.hidden_size)

from datasets import load_dataset

ds = load_dataset("beans", split="train")
print(len(ds))
