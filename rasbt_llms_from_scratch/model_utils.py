import torch


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
