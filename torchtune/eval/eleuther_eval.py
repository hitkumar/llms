import sys

from receipes.receipe_interfaces import EvalReceipeInterface

try:
    import lm_eval
except ImportError:
    raise ImportError(
        "lm_eval is not installed, please install it with `pip install lm-eval`"
    )
    sys.exit(1)

if __name__ == "__main__":
    print("eval")
