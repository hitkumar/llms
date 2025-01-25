import os
import time
from re import L

import torch
import torch.nn.functional as F
import utils
from checkpoint import TrainState
from config_manager import JobConfig
from core.datasets.tokenizer import build_tokenizer

from core.datasets import build_hf_data_loader, build_tokenizer
from core.logging_util import init_logger, logger
from core.models import model_name_to_cls, model_name_to_tokenizer, models_config
from core.parallelisms import models_parallelize_fns, ParallelDims
from metrics import build_device_memory_monitor
from optimizer import build_lr_scheduler, build_optimizer
from torch import distributed as dist
from torch.distributed._tensor import DTensor
from torch.distributed.device_mesh import init_device_mesh

from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.tensor.parallel import loss_parallel
from utils import device_module, device_type


@record
def main(job_config: JobConfig):
    init_logger()
    logger.info(f"Starting job: {job_config.job.description}")

    if job_config.job.print_args:
        logger.info(f"Running with args: {job_config.to_dict()}")

    # useful for color printing
    color = utils.NoColor if job_config.metrics.disable_color_printing else utils.Color
    gc_handler = utils.GarbageCollection(gc_freq=job_config.training.gc_freq)

    world_size = int(os.environ["WORLD_SIZE"])
    parallel_dims = ParallelDims(
        dp_shard=job_config.training.data_parallel_shard_degree,
        dp_replicate=job_config.training.data_parallel_replicate_degree,
        cp=job_config.experimental.context_parallel_degree,
        tp=job_config.training.tensor_parallel_degree,
        pp=job_config.experimental.pipeline_parallel_degree,
        world_size=world_size,
        enable_loss_parallel=not job_config.training.disable_loss_parallel,
    )
    device = torch.device(f"{device_type}:{int(os.environ['LOCAL_RANK'])}")
    device_module.set_device(device)
    utils.init_distributed(job_config)

    device_memory_monitor = build_device_memory_monitor()
    peak_flops = utils.get_peak_flops(device_memory_monitor.device_name)
    logger.info(f"Peak Flops for this gpu: {peak_flops:.3e}")

    # build meshes
    world_mesh = parallel_dims.build_mesh(device_type=device_type)
    logger.info(
        f"{color.blue}device_module is {device_module}, device_name is {device_memory_monitor.device_name}, world mesh is {world_mesh}{color.reset}"
    )

    if parallel_dims.dp_enabled:
        dp_mesh = world_mesh["dp"]
        dp_degree, dp_rank = dp_mesh.size(), dp_mesh.get_local_rank()
    else:
        dp_degree, dp_rank = 1, 0

    logger.info(f"{dp_degree=} {dp_rank=}")
    # set_determinism(world_mesh, device)
    model_name = job_config.model.name

    # build tokenizer
    tokenizer_type = model_name_to_tokenizer[model_name]
    tokenizer = build_tokenizer(tokenizer_type, job_config.model.tokenizer_path)

    # build dataloader
    data_loader = build_hf_data_loader(
        job_config.training.dataset,
        job_config.training.dataset_path,
        tokenizer,
        job_config.training.batch_size,
        job_config.training.seq_len,
        dp_degree,
        dp_rank,
    )
    # it = iter(data_loader)
    # input, target = next(it)
    # logger.info(f"input shape: {input.shape}, target shape is {target.shape}")

    # build model
    model_cls = model_name_to_cls[model_name]
    model_config = models_config[model_name][job_config.model.flavor]
    model_config.vocab_size = tokenizer.n_words
    model_config.max_seq_len = job_config.training.seq_len
    logger.info(f"Building {model_name} {job_config.model.flavor} with {model_config}")

    with torch.device("meta"):
        model = model_cls.from_model_args(model_config)

    init_device = device_type
    buffer_device = None

    def loss_fn(pred, labels):
        # print(f"dtype of pred is {pred.dtype}, labels dtype is {labels.dtype}")
        # if isinstance(pred, DTensor):
        #     print("pred is DTensor")
        # elif isinstance(labels, DTensor):
        #     print("labels is DTensor")
        return F.cross_entropy(pred.flatten(0, 1).float(), labels.flatten(0, 1))

    # apply 2D parallelism
    models_parallelize_fns[model_name](model, world_mesh, parallel_dims, job_config)
    model.to_empty(device=init_device)
    with torch.no_grad():
        model.init_weights(buffer_device=buffer_device)
    model.train()
    model_parts = [model]

    # setup optimizer
    optimizer = build_optimizer(model_parts[0], job_config)
    lr_scheduler = build_lr_scheduler(optimizer.optimizer, job_config)
    print(f"optimizer is {optimizer.optimizer}, lr_scheduler is {lr_scheduler}")

    train_state = TrainState()
    data_iterator = iter(data_loader)

    while train_state.step < job_config.training.steps:
        train_state.step += 1
        gc_handler.run(train_state.step)
        # get data
        data_load_start = time.perf_counter()
        batch = next(data_iterator)
        input_ids, labels = batch
        input_ids = input_ids.to(device_type)
        labels = labels.to(device_type)
        optimizer.zero_grad()

        with loss_parallel():
            pred = model(input_ids)
            # print(
            #     f"input_ids shape is {input_ids.shape}, labels shape is {labels.shape}, pred shape is {pred.shape}"
            # )
            loss = loss_fn(pred, labels)
            del pred
            loss.backward()

        # optimizer step
        optimizer.step()
        lr_scheduler.step()
        logger.info(f"train step is {train_state.step}, loss is {loss.item()}")

    if torch.distributed.get_rank() == 0:
        logger.info("Sleeping 2 seconds for other ranks to complete")
        time.sleep(2)

    logger.info("Training completed")


if __name__ == "__main__":
    config = JobConfig()
    config.parse_args()
    main(config)

    # Distributed training debugging code
    # mesh_2d = init_device_mesh("cuda", (2, 2, 2), mesh_dim_names=("dp", "cp", "tp"))
    # replicate_group = mesh_2d.get_group(mesh_dim="dp")
    # shard_group = mesh_2d.get_group(mesh_dim="cp")
    # # print(mesh_2d)
    # print(f"Replicate group: {replicate_group.rank()}, {replicate_group.size()}")
    # print(f"Shard group: {shard_group.rank()}, {shard_group.size()}")

    # # mesh_2d[("dp", "cp")]._flatten(mesh_dim_name="dp_cp")
    # print(f"mesh_2d after flatten is {mesh_2d['dp_cp']}")
    # torch.distributed.destroy_process_group()
