from dataclasses import dataclass


@dataclass
class HParams:
    max_lr: float
    min_lr: float
    weight_decay: float
    warmup_steps: int
    max_steps: int
    total_batch_size: int
    B: int
    T: int
    log_freq: int
