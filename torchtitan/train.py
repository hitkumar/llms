from core.datasets.tokenizer import build_tokenizer
from core.logging_util import init_logger, logger

tokenizer = build_tokenizer(
    "tiktoken",
    "./torchtitan/core/datasets/tokenizer/original/tokenizer.model",
)

print(tokenizer.encode("hello world"))
