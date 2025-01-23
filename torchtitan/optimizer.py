import functools
from typing import Any, Dict

import torch
import torch.nn as nn
from config_manager import JobConfig
from torch.distributed.checkpoint.state_dict import (
    get_optimizer_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)
from torch.distributed.checkpoint.stateful import Stateful
from torch.optim.lr_scheduler import LambdaLR


class OptimizerContainer(Stateful):
    def __init__(
        self, model: nn.Module, optimizer_kwargs: Dict[str, Any], name: str
    ) -> None:
        self.model = model
        if name == "Adam":
            # TODO: make the optimizer options configurable by toml/cmd args
            self.optimizer = torch.optim.Adam(model.parameters(), **optimizer_kwargs)
        elif name == "AdamW":
            self.optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
        else:
            raise NotImplementedError(f"Optimizer {name} not added.")

    def step(self) -> None:
        self.optimizer.step()

    def zero_grad(self) -> None:
        self.optimizer.zero_grad()

    def state_dict(self) -> Dict[str, Any]:
        return get_optimizer_state_dict(
            self.optimizer, options=StateDictOptions(flatten_optimizer_state_dict=True)
        )

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        set_optimizer_state_dict(
            self.optimizer,
            state_dict,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )


def build_optimizer(model: nn.Module, job_config: JobConfig):
    name = job_config.optimizer.name
    lr = job_config.optimizer.lr
    fused = job_config.optimizer.fused
    optimizer_kwargs = {
        "lr": lr,
        "betas": (0.9, 0.95),
        "weight_decay": 0.1,
        "fused": fused,
        "foreach": not fused,
    }
    return OptimizerContainer(model, optimizer_kwargs, name)


def linear_warmup_linear_decay(warmup_steps: int, decay_steps: int, current_step: int):
    if current_step < warmup_steps:
        curr_adjustment = float(current_step + 1 / (warmup_steps + 1))
    else:
        curr_adjustment = 1 - (current_step - warmup_steps) / decay_steps

    return curr_adjustment


class SchedulerContainer:
    def __init__(self, optimizer, lr_lambda) -> None:
        self.scheduler = LambdaLR(optimizer, lr_lambda)

    def step(self) -> None:
        self.scheduler.step()

    def get_lr_scheduler_state(self) -> Dict[str, Any]:
        state_dict = {}
        state_dict["lr_scheduler"] = self.scheduler
        return state_dict


def build_lr_scheduler(optimizer, job_config: JobConfig) -> SchedulerContainer:
    warmup_steps = job_config.training.warmup_steps
    decay_steps = job_config.training.steps - warmup_steps
    lr_lambda = functools.partial(linear_warmup_linear_decay, warmup_steps, decay_steps)
    return SchedulerContainer(optimizer, lr_lambda)
