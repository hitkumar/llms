from core.models.llama.model import ModelArgs, Transformer

__all__ = ["Transformer", "ModelArgs"]


llama3_configs = {
    "8B": ModelArgs(
        dim=4096,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=1024,
        rope_theta=500000,
    )
}
