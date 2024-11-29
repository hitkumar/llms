import os

import triton
import triton.language as tl


# can use triton.cdiv in most cases.
def cdiv(a, b):
    return (a + b - 1) // b


def check_tensors_gpu_ready(*tensors):
    for t in tensors:
        assert t.is_contiguous, f"Tensor {t} is not contiguous"
        if not os.environ.get("TRITON_INTERCEPT") == "1":
            assert t.is_cuda, f"Tensor {t} is not on GPU"


@triton.jit
def get_1d_offset(size, n_prev_chunks):
    return n_prev_chunks * size + tl.arange(0, size)


@triton.jit
def get_2d_offset(offs_0, offs_1, stride_0, stride_1):
    return offs_0[:, None] * stride_0 + offs_1[None, :] * stride_1


@triton.jit
def get_1d_mask(offs, max):
    return offs < max


@triton.jit
def get_2d_mask(offs_0, offs_1, max_0, max_1):
    return (offs_0[:, None] < max_0) & (offs_1[None, :] < max_1)
