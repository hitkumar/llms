from typing import Protocol


class EvalReceipeInterface(Protocol):
    def load_checkpoint(self, **kwargs) -> None:
        """
        Loads checkpoint from a given path
        """
        ...

    def setup(self, **kwargs) -> None:
        """
        Setup components needed for evaluation
        """
        ...

    def evaluate(self, **kwargs) -> None:
        """
        Evaluation logic
        """
        ...
