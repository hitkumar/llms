from collections import defaultdict

import torch
import torch.nn as nn
from config_manager import JobConfig, TORCH_DTYPE_MAP
from core.logging_util import init_logger, logger
from core.parallelisms.parallel_dims import ParallelDims

from torch.distributed import DeviceMesh
from torch.distributed._composable.fsdp import (
    CPUOffloadPolicy,
    fully_shard,
    MixedPrecisionPolicy,
)
from torch.distributed._composable.replicate import replicate
from torch.distributed._tensor import Replicate, Shard
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
)


def parallelize_llama(
    model: nn.Module,
    world_mesh: DeviceMesh,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
):
    """
    Apply tensor parallelism, activation checkpointing, torch.compile, and data
    parallelism to the model.

    NOTE: The passed-in model preferably should be on meta device. Otherwise,
    the model must fit on GPU or CPU memory.
    """
    if parallel_dims.tp_enabled:
        apply_tp(
            model,
            world_mesh["tp"],
            loss_parallel=parallel_dims.loss_parallel_enabled,
        )

    if job_config.activation_checkpoint.mode != "none":
        apply_ac(model, job_config.activation_checkpoint)

    if job_config.training.compile:
        if job_config.model.norm_type == "rmsnorm":
            raise NotImplementedError(
                "torch.compile is not supported for fused_rmsnorm yet"
            )
        apply_compile(model)

    # apply FSDP or HSDP, potentially with Context Parallel
    if parallel_dims.dp_shard_enabled or parallel_dims.cp_enabled:
        if parallel_dims.dp_replicate_enabled:
            dp_mesh_dim_names = ("dp_replicate", "dp_shard_cp")
        else:
            dp_mesh_dim_names = ("dp_shard_cp",)

        apply_fsdp(
            model,
            world_mesh[dp_mesh_dim_names],
            param_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_reduce],
            cpu_offload=job_config.training.enable_cpu_offload,
        )

        if parallel_dims.dp_replicate_enabled:
            logger.info("Applied HSDP to the model")
        else:
            logger.info("Applied FSDP to the model")

        if parallel_dims.cp_enabled:
            logger.info("Applied Context Parallel to the model")

        if job_config.training.enable_cpu_offload:
            logger.info("Applied CPU Offloading to the model")
    elif parallel_dims.dp_replicate_enabled:
        if world_mesh.ndim > 1:
            raise RuntimeError("DDP has not supported > 1D parallelism")
        apply_ddp(
            model,
            world_mesh,
            enable_compile=job_config.training.compile,
            enable_compiled_autograd=job_config.training.enable_compiled_autograd,
        )
        logger.info("Applied DDP to the model")


def apply_tp(model: nn.Module, tp_mesh: DeviceMesh, loss_parallel: bool):
    """
    Apply TP to the model.
    TODO: Add support for float8
    Add support for async tp"""
    parallelize_module(
        model,
        tp_mesh,
        {
            "tok_embeddings": RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Shard(1),
            ),
            "norm": SequenceParallel(),
            "output": ColwiseParallel(
                input_layouts=Shard(1),
                output_layouts=Shard(-1) if loss_parallel else Replicate(),
                use_local_output=not loss_parallel,
            ),
        },
    )
    rowwise_parallel, colwise_parallel, prepare_module_input = (
        RowwiseParallel,
        ColwiseParallel,
        PrepareModuleInput,
    )
    for _, transformer_block in model.layers.items():
        layer_plan = {
            "attention_norm": SequenceParallel(),
            "attention": prepare_module_input(
                input_layouts=(Shard(1), None),
                desired_input_layouts=(Replicate(), None),
            ),
            "attention.wq": colwise_parallel(),
            "attention.wk": colwise_parallel(),
            "attention.wv": colwise_parallel(),
            "attention.wo": rowwise_parallel(output_layouts=Shard(1)),
            "ffn_norm": SequenceParallel(),
            "feed_forward": prepare_module_input(
                input_layouts=(Shard(1),),
                desired_input_layouts=(Replicate(),),
            ),
            "feed_forward.w1": colwise_parallel(),
            "feed_forward.w2": rowwise_parallel(output_layouts=Shard(1)),
            "feed_forward.w3": colwise_parallel(),
        }
        parallelize_module(
            module=transformer_block,
            device_mesh=tp_mesh,
            parallelize_plan=layer_plan,
        )
    logger.info("Applied tensor parallelism to the model")


def _apply_ac_to_transformer_block(module: nn.Module, ac_config):
    # TODO: support selective AC
    valid_ac_modes = ("full", "selective")
    if ac_config.mode not in valid_ac_modes:
        raise ValueError(
            f"Invalid AC mode {ac_config.mode}. Valid modes are {valid_ac_modes}."
        )
    if ac_config.mode == "full":
        return ptd_checkpoint_wrapper(module, preserve_rng_state=False)
    else:
        raise ValueError("selective AC is not supported yet")


def apply_ac(model: nn.Module, ac_config):
    """Apply activation checkpointing to the model."""
    for layer_id, transformer_block in model.layers.named_children():
        transformer_block = _apply_ac_to_transformer_block(transformer_block, ac_config)
        model.layers.register_module(layer_id, transformer_block)

    logger.info(f"Applied {ac_config.mode} activation checkpointing to the model")


def apply_compile(model: nn.Module):
    for layer_id, transformer_block in model.layers.named_children():
        transformer_block = torch.compile(transformer_block, fullgraph=True)
        model.layers.register_module(layer_id, transformer_block)

    logger.info("Applied torch.compile to the model")


def apply_fsdp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    param_type: torch.dtype,
    reduce_type: torch.dtype,
    cpu_offload: bool = False,
):
    """
    Apply FSDP2 to the model.
    Doesn't support PP yet.
    """
    mp_policy = MixedPrecisionPolicy(param_dtype=param_type, reduce_dtype=reduce_type)
    fsdp_config = {"mesh": dp_mesh, "mixed_precision": mp_policy}
    if cpu_offload:
        fsdp_config["cpu_offload"] = CPUOffloadPolicy()

    for layer_id, transformer_block in model.layer.items():
        # Do not reshard after forward for the last transformer block since transformer would prfetch it immediately for backward
        reshard_after_forward = int(layer_id) < len(model.layer) - 1
        fully_shard(
            transformer_block,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )

    fully_shard(model, **fsdp_config, reshard_after_forward=True)


def apply_ddp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    enable_compile: bool,
    enable_compiled_autograd: bool,
):
    if enable_compile:
        if enable_compiled_autograd:
            torch._dynamo.config.optimize_ddp = (
                "python_reducer_without_compiled_forward"
            )
        else:
            torch._dynamo.config.optimize_ddp = "ddp_optimizer"

    replicate(model, device_mesh=dp_mesh, bucket_cap_mb=100)

    logger.info("Applied DDP to the model")
