import torch
from huggingface_hub import hf_hub_download


# Find model memory size.
def total_memory_size(model, input_dtype=torch.float32):
    total_params = 0
    total_grads = 0
    for param in model.parameters():
        param_size = param.numel()
        total_params += param_size
        if param.requires_grad:
            total_grads += param_size

    total_buffers = sum(buf.numel() for buf in model.buffers())
    element_size = torch.tensor(0, dtype=input_dtype).element_size()

    model_size_bytes = (total_params + total_grads + total_buffers) * element_size
    model_size_gb = model_size_bytes / (2**30)
    return model_size_gb


def get_model_params(model):
    return sum(p.numel() for p in model.parameters())


def download_file_from_hf_hub(repo_id, filename, local_dir):
    # Before running this, you need to set the HF token using this
    # from huggingface_hub import login
    # login(token=access_token)
    return hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir)


def free_pytorch_memory(model=None):
    import gc

    if model is not None:
        del model
    gc.collect()  # run python garbage collector

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
