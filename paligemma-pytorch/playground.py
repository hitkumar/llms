"""
Use this to work with HF datasets
"""

import torch
import torch.nn as nn
from modeling_siglip import SiglipVisionConfig
from transformers import GPT2LMHeadModel

model_hf = GPT2LMHeadModel.from_pretrained("gpt2")
sd_hf = model_hf.state_dict()
print(len(sd_hf.keys()))

siglip_config = SiglipVisionConfig()
print(siglip_config.hidden_size)

from datasets import load_dataset

ds = load_dataset("beans", split="train")
print(len(ds))
