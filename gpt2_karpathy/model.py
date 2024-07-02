from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect
import config

@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257  # number of tokens: 50K BPE merges + 256 bytes tokens + 1 endoftext token
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd //  config.n_head

        # mask following OpenAI / HF naming
        # self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        # x is of shape (B, seq_len, emb_dim)
        B, T, C = x.size()
        # (B, seq_len, 3 * emb_dim)
        qkv = self.c_attn(x)
        # (B, seq_len, emb_dim)
        q, k, v = qkv.chunk(3, dim=2)
        # (B, n_head, seq_len, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # (B, n_head, seq_len, head_dim) * (B, n_head, head_dim, seq_len) -> (B, n_head, seq_len, seq_len)
        # att = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        # att.masked_fill_(self.bias[:,:,:T, :T] == 0, -torch.inf)
        # # (B, n_head, seq_len, seq_len)
        # att = F.softmax(att, dim=-1)

        # # (B, n_head, seq_len, head_dim)
        # out = att @ v

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        # output is of shape (B, seq_len, emb_dim) like input
        return self.c_proj(out)

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.ln_1 = nn.LayerNorm(config.n_embd)
        # (B, seq_len, emb_dim) -> (B, seq_len, 3 * emb_dim) -> (B, seq_len, emb_dim)
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0, std=0.02)


    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pre-trained gpt-2 weights from HF"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel

        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768),
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1536),
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1792),
        }[model_type]
        # config_args['vocab_size'] = 50257 default value should be used
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # this is a causal mask/buffer
        # print(config)

        # init a HF model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # this is a causal mask/buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # this is a causal mask/buffer

        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        assert len(sd_keys) == len(sd_hf.keys()), f"mismatched keys: {len(sd_keys)} != {len(sd_keys_hf)}"

        for k in sd_keys_hf:
            if any(k.endswith(x) for x in transposed):
                # special treatment for conv1 weights
                assert sd_hf[k].shape[::-1] == sd[k].shape
                # print(f"mistached key: {k}, shape is {sd[k].shape}")
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def forward(self, x, targets=None):
        # x is of shape (B, seq_len)
        B, T = x.shape
        assert T <= self.config.block_size, f'Cannot forward sequence of length {T}, block size is only {self.config.block_size}'
        pos = torch.arange(0, T, dtype=torch.long, device=x.device) # shape (T)
        # (B, T, emb_dim)
        embds = self.transformer.wpe(pos) + self.transformer.wte(x)
        for block in self.transformer.h:
            embds = block(embds)

        embds = self.transformer.ln_f(embds)
        # return the logits shape is (B, T, vocab_size)
        logits = self.lm_head(embds)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), targets.view(-1))
        
        return logits, loss
    
    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # Find all parameters that require gradients
        params_dict = {pn: p for pn, p in self.named_parameters()}
        params_dict = {pn: p for pn, p in params_dict.items() if p.requires_grad}

        # Now divide params in 2 groups: ones that need weight decay and other that don't
        decay_params = [p for n, p in params_dict.items() if p.dim() >= 2]
        nondecay_params = [p for n, p in params_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nondecay_params, 'weight_decay': 0.0}
        ]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_nondecay_params = sum(p.numel() for p in nondecay_params)

        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        if config.master_process:
            print(f'num_decay_params: {num_decay_params}, num_nondecay_params: {num_nondecay_params}, use_fused: {use_fused}')
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer    
