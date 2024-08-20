from typing import Optional, Tuple
import torch
import torch.nn as nn
import fastcore.all as fc
import torch.nn.functional as F

class SiglipVisionConfig:
    def __init__(
        self,
        hidden_size: int=768,
        intermediate_size: int=3072,
        num_hidden_layers: int=12,
        num_attention_heads: int=12,
        num_channels: int = 3,
        image_size: int=224,
        patch_size: int=16,
        layer_norm_eps: float=1e-6,
        attention_dropout: float=0.0,
        num_image_tokens: int=None,
        **kwargs
    ):
        super().__init__()
        fc.store_attr()

class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid"  # no padding
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)), # TODO: Is this expand needed? Shouldn't be by broadcast rules.
            persistent=False
        )
    
    def forward(self, pixel_values: torch.tensor) -> torch.tensor:
        _,_, height, width = pixel_values.shape # [B, C, H, W]
        patch_embeds = self.patch_embedding(pixel_values) # [B, emb_dim, num_patches_H, num_patches_W]
        # [B, emb_dim, num_patches_H * num_patches_W] -> [B, num_patches_H * num_patches_W, emb_dim]
        embeddings = patch_embeds.flatten(2).transpose(1, 2)
        # [B, num_patches, emb_dim]
        embeddings += self.position_embedding(self.position_ids)
        return embeddings

class SiglipAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
    
    def forward(self, hidden_states: torch.tensor) -> Tuple[torch.tensor, Optional[torch.tensor]]:
        # hidden states: [B, num_patches, emb_dim]
        batch_size, seq_len, _ = hidden_states.shape
        # [B, num_patches, emb_dim]
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = (query_states @ key_states.transpose(2, 3)) * self.scale
        # TODO: Check if this is needed.
        attn_probs = F.softmax(attn_scores, dim=-1, dtype=torch.float32).type_as(query_states.dtype)
        attn_probs = F.dropout(attn_probs, p=self.dropout, training=self.training)

        attention_out = attn_probs @ value_states
        attention_out = attention_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        attention_out = self.out_proj(attention_out)
        # [B, num_patches, emb_dim], [B, num_heads, num_patches, num_patches]
        return attention_out, attn_probs

class SiglipMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
    
    def forward(self, hidden_states: torch.tensor) -> torch.tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = F.gelu(hidden_states)
        # [B, num_patches, emb_dim]
        return self.fc2(hidden_states)

class SiglipEncoderLayer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SiglipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
    
    def forward(self, hidden_states: torch.tensor) -> torch.tensor:
       attn_out = hidden_states + self.self_attn(self.layer_norm1(hidden_states))[0]
       fc_out = attn_out + self.mlp(self.layer_norm2(attn_out))
       return fc_out

class SiglipEncoder(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
    
    def forward(self, inputs_embeds: torch.tensor) -> torch.tensor:
        for layer in self.layers:
            inputs_embeds = layer(inputs_embeds)
        return inputs_embeds

class SiglipVisionTransformer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    
    def forward(self, pixel_values: torch.tensor) -> torch.tensor:
        # pixes values: [B, C, H, W]
        # embeddings: [B, num_patches, emb_dim]
        embeddings = self.embeddings(pixel_values)
        embeddings = self.encoder(embeddings)
        # [B, num_patches, emb_dim]
        return self.post_layernorm(embeddings)

class SiglipVisionModel(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)
    
    def forward(self, pixel_values: torch.tensor) -> torch.tensor:
        # [B, C, H, W] -> [B, num_patches, emb_dim]
        return self.vision_model(pixel_values)
