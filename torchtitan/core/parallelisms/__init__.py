from core.parallelisms.parallel_dims import ParallelDims
from core.parallelisms.parallelize_llama import parallelize_llama

models_parallelize_fns = {
    "llama3": parallelize_llama,
}

__all__ = [
    "ParallelDims",
    "models_parallelize_fns",
]
