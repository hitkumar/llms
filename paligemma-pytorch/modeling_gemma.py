
import torch
import torch.nn as nn
from typing import List, Optional, Tuple
from modeling_siglip import SiglipVisionConfig, SiglipVisionModel
import fastcore.all as fc

class KVCache:
    def __init__(self):
        # elements in the cache are of shape (batch_size, num_heads_kv, seq_len, head_dim)
        self.key_cache: List[torch.tensor] = []
        self.value_cache: List[torch.tensor] = []
    
    def num_items(self) -> int:
        if len(self.key_cache) == 0:
            return 0
        else:
            return self.key_cache[0].shape[-2]
    
    def update(
        self,
        key_states: torch.tensor,
        value_states: torch.tensor,
        layer_idx: int
    ):
        if len(self.key_cache) <= layer_idx:
           self.key_cache[layer_idx] = key_states
           self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
        
        return self.key_cache[layer_idx], self.value_cache[layer_idx]


class GemmaConfig:
    def __init__(
        self,
        vocab_size,
        hidden_size,
        intermediate_size,
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads,
        head_dim=256,
        max_position_embeddings=8192,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        attention_bias=False,
        attention_dropout=0.0,
        pad_token_id=None,
        **kwargs
    ):
        super().__init__()
        fc.store_attr()


class PaliGemmaConfig:
    def __init__(
        self,
        vision_config=None,
        text_config=None,
        ignore_index=-100,
        image_token_index=256000,
        vocab_size=257152,
        projection_dim=2048,
        hidden_size=2048,
        pad_token_id=None,
        **kwargs
    ):
        super().__init__()
        fc.store_attr()
        self.is_encoder_decoder = False
        self.vision_config = SiglipVisionConfig(**vision_config)
        self.text_config = GemmaConfig(**text_config, pad_token_id=pad_token_id)

        # self.vocab_size = self.text_config.vocab_size
        self.text_config.num_image_tokens = (self.vision_config.image_size // self.vision_config.patch_size) ** 2
        self.vision_config.projection_dim = projection_dim


class GemmaRMSNorm(nn.Module):
    # input is [B, seq_len, dim]
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))
    
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x):
        output = self._norm(x.float())
        output = output * (1 + self.weight.float())
        return output.type_as(x)

class GemmaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        # head dim
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # dim: [dim//2]
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", tensor=inv_freq, persistent=False)
    
    @torch.no_grad()
    def forward(self, x, position_ids, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_dim]
        # position_ids: [bs, seq_len]
        # TODO: confirm what the position_ids here are.
        self.inv_freq.to(x.device)
        # [batch_size, dim//2, 1]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        # [batch_size, 1, seq_len]
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != 'mps' else 'cpu'
        with torch.autocast(device_type=device_type, enabled=False):
            # [batch_size, dim//2, 1] * [batch_size, 1, seq_len] = [batch_size, dim/2, seq_len] -> [bs, seq_len,d]
            thetas = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
            # [batch_size, seq_len, dim/2] -> [batch_size, seq_len, dim]
            emb = torch.cat((thetas, thetas), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        
        # [batch_size, seq_len, dim]
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

def rotate_half(x):
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    # q: [bs, num_attention_heads, seq_len, head_dim]
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    # Formula 34 of ROPE paper
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class GemmaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
    
    def forward(self, x):
        gate_out = F.gelu(self.gate_proj(x), approximate='tanh')
        up_out = self.up_proj(x)
        return self.down_proj(up_out * gate_out)

def repeat_kv(hidden_states: torch.tensor, n_rep: int) -> torch.tensor:
    # [bs, num_kv_attention_heads, seq_len, head_dim] -> [bs, num_attention_heads, seq_len, head_dim * n_rep]
    # n_rep = num_attention_heads // num_kv_attention_heads
    bs, num_kv_attention_heads, seq_len, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states

    # TODO: can we use view here?
    return hidden_states[:, :, None, :, :].expand(bs, num_kv_attention_heads, n_rep, seq_len, head_dim).reshape(bs, -1, seq_len, head_dim)

class GemmaAttention(nn.Module):
    def __init__(self, config: GemmaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_dx = layer_idx

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        assert self.hidden_size % self.num_heads == 0, f"hidden_size ({self.hidden_size}) is not divisible by num_heads ({self.num_heads})"

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.rotary_emb = GemmaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )
    
    def forward(
        self,
        hidden_states: torch.tensor,
        attention_mask: Optional[torch.tensor] = None,
        position_ids: Optional[torch.tensor] = None,
        kv_cache: Optional[KVCache] = None,
        **kwargs
    ) -> Tuple[torch.tensor, Optional[torch.tensor], Optional[Tuple[torch.tensor]]]:
        # hidden_states: [bs, seq_len, hidden_size]
        bs, seq_len, _ = hidden_states.shape
        # [bs, seq_len, head_dim * num_heads]
        query_states = self.q_proj(hidden_states)
        # [bs, seq_len, head_dim * num_key_value_heads]
        value_states = self.v_proj(hidden_states)
        key_states = self.k_proj(hidden_states)

        # [bs, num_heads, seq_len, head_dim]
        query_states = query_states.view(bs, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # [bs, num_key_value_heads, seq_len, head_dim]
        value_states = value_states.view(bs, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bs, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # TODO: simplify this to take device type only as input
        # [bs, seq_len, head_dim]
        cos, sin = self.rotary_emb(value_states, position_ids, seq_len=None)
        # [bs, num_heads_q, seq_len, head_dim], [bs, num_heads_kv, seq_len, head_dim]
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, unsqueeze_dim=1)

        if kv_cache is not None:
            key_states, value_states = kv_cache.update(key_states, value_states, self.layer_dx)
        
        # Repeat key ad values
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        # [bs, num_heads_q, seq_len, head_dim], # [bs, num_heads_q, seq_len, head_dim] -> [bs, num_heads_q, seq_len, seq_len]
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        # TODO: learn more about masking later
        assert attention_mask is not None
        attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(dtype=query_states.dtype)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        # [bs, num_heads_q, seq_len, seq_len] * [bs, num_heads_kv, seq_len, head_dim] -> [bs, num_heads_q, seq_len, head_dim]
        attn_output = attn_weights @ value_states
        attn_weights = attn_weights.transpose(1, 2).contiguous().view(bs, seq_len, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

class GemmaDecoderLayer(nn.Module):
    def __init__(self, config: GemmaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = GemmaAttention(config, layer_idx)
        self.mlp = GemmaMLP(config)
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, config.rms_norm_eps)
    
    def forward(
        self,
        hidden_states: torch.tensor,
        attention_mask: Optional[torch.tensor] = None,
        position_ids: Optional[torch.tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ):

        attn_out, _ = self.self_attn(
            hidden_states=self.input_layernorm(hidden_states),
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache=kv_cache,
        )
        hidden_states += attn_out
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states


class GemmaModel(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [GemmaDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = GemmaRMSNorm(config.hidden_size, config.rms_norm_eps)
    
    def get_input_embeddings(self):
        return self.embed_tokens
    
    def forward(
        self,
        attention_mask: Optional[torch.tensor] = None,
        position_ids: Optional[torch.tensor] = None,
        inputs_embeds: Optional[torch.tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ):
        normalizer = torch.tensor(self.config.hidden_size ** 0.5, dtype=inputs_embeds.dtype)
        hidden_states = inputs_embeds * normalizer
        for decoder_layer in self.layers:
            # [bs, seq_len, hidden_size]
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                kv_cache=kv_cache,
            )
        
        hidden_states = self.norm(hidden_states)
        return hidden_states

class GemmaForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = GemmaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.vocab_size = config.vocab_size
    
    def get_input_embeddings(self):
        return self.model.embed_tokens
    
    def tie_weights(self):
        self.lm_head.weight = self.model.embed_tokens.weight
    
    def forward(
        self,
        attention_mask: Optional[torch.tensor] = None,
        position_ids: Optional[torch.tensor] = None,
        inputs_embeds: Optional[torch.tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ):
        # [bs, seq_len, hidden_size]
        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            kv_cache=kv_cache,
        )
        logits = self.lm_head(outputs).float()
        res = {
            "logits": logits,
        }
        if kv_cache is not None:
            res["kv_cache"] = kv_cache
        
        return res

class PaliGemmaMultiModalProjector(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.linear = nn.Linear(config.vision_config.hidden_size, config.vision_config.projection_dim, bias=True)
    
    def forward(self, image_features):
        # [bs, seq_len, hidden_size] -> [bs, seq_len, projection_dim]
        return self.linear(image_features)

class PaliGemmaForConditionalGeneration(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.config = config
        self.vision_tower = SiglipVisionModel(config.vision_config)
        self.multi_modal_projector = PaliGemmaMultiModalProjector(config)
        self.vocab_size = config.vocab_size

        language_model = GemmaForCausalLM(config.text_config)
        self.language_model = language_model
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
    
    def tie_weights(self):
        self.language_model.tie_weights()
    
    def _merge_input_ids_with_image_features(self, image_features: torch.tensor, input_embeds: torch.tensor, input_ids: torch.tensor, attention_mask: torch.tensor, kv_cache: Optional[KVCache] = None):
        # input_ids includes dummy image tokens
        # TODO: figure out how this works
        # attention mask is [bs, seq_len] and is all 1s, no padding.
        _, _, embed_dim = image_features.shape
        batch_size, seq_len = input_ids.shape
        dtype, device = input_ids.dtype, input_ids.device

        # scaling image features
        # shape is [bs, num_patches, embed_dim]
        scaled_image_features = image_features / (embed_dim ** 0.5)

        final_embedding = torch.zeros(batch_size, seq_len, embed_dim, dtype=dtype, device=device)
        # shape is [bs, seq_len]
        text_mask = (input_ids != self.config.image_token_index) & (input_ids != self.config.pad_token_id)
        image_mask = input_ids == self.config.image_token_index
        pad_mask = input_ids == self.config.pad_token_id

        # shape is [bs, seq_len, embed_dim]
        text_mask_expanded = text_mask[:, :, None].expand(-1, -1, embed_dim)
        image_mask_expanded = image_mask[:, :, None].expand(-1, -1, embed_dim)
        pad_mask_expanded = pad_mask[:, :, None].expand(-1, -1, embed_dim)

        # Add text embeddings
        final_embeddings = torch.where(text_mask_expanded, input_embeds, final_embedding)
        # insert image embeddings
        final_embedding = final_embedding.masked_scatter(image_mask_expanded, scaled_image_features)
        final_embedding = torch.where(pad_mask_expanded, torch.zeros_like(final_embedding), final_embedding)

        if kv_cache is None or kv_cache.num_items() == 0:
            # No masking as we are in prefilling stage
            causal_mask = torch.full(
                (batch_size, seq_len, seq_len), fill_value=0, dtype=dtype, device=device
            )
        else:
            assert seq_len == 1
            kv_len = kv_cache.num_items() + 1
            causal_mask = torch.full(
                (batch_size, seq_len, kv_len), fill_value=0, dtype=dtype, device=device
            )

        # [bs, 1, seq_len, kv_len]    
        causal_mask = causal_mask.unsqueeze(1)

        if kv_cache is not None and kv_cache.num_items() > 0:
            position_ids = attention_mask.cumsum(-1)[:, -1]
            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0)
        else:
            position_ids = (attention_mask.cumsum(-1)).masked_fill_((attention_mask == 0), 1).to(device)
        
        return final_embedding, causal_mask, position_ids
    
    def forward(
        self,
        input_ids,
        pixel_values,
        attention_mask,
        kv_cache
    ):
        assert torch.all(attention_mask == 1), "Attention mask should be all 1s"

        # [bs, seq_len] -> [bs, seq_len, hidden_size]
        input_embeds = self.language_model.get_input_embeddings()(input_ids)

        # [bs, num_patches, embed_dim] -> [bs, num_patches, hidden_size]
        selected_image_features = self.vision_tower(pixel_values.to(input_embeds.device))
        selected_image_features = self.multi_modal_projector(selected_image_features)

        # understand this part more.
        input_embeds, causal_mask, position_ids = self._merge_input_ids_with_image_features(selected_image_features, input_embeds, input_ids, attention_mask, kv_cache)

        # [bs, seq_len, hidden_size]
        outputs = self.language_model(
            attention_mask=causal_mask,
            position_ids=position_ids,
            inputs_embeds=input_embeds,
            kv_cache=kv_cache,
        )
        return outputs
