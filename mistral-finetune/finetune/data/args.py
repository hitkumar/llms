from dataclasses import dataclass, field
from typing import Optional

from simple_parsing.helpers import Serializable

@dataclass
class InstructArgs(Serializable):
    shuffle: bool = True
    dynamic_chunk_fn_call: bool = True

@dataclass
class DataArgs(Serializable):
    data: str = ""
    shuffle: bool = False
    instruct_data: str = ""
    eval_instruct_data: str = ""
    instruct: InstructArgs = field(default_factory=InstructArgs)

    def __post_init__(self) -> None:
        if (
            self.instruct.shuffle is False
            and self.instruct.dynamic_chunk_fn_call is True
        ):
            raise ValueError(
                "Make sure to either enable `data.instruct.shuffle=True` or `data.instruct.dynamic_chunk_fn_call=False`. Dynamic chunking is only possible if data is loaded and shuffled before training."
            )
