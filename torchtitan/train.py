import os

import torch
import torch.nn.functional as F
from config_manager import JobConfig
from core.datasets.tokenizer import build_tokenizer
from core.datasets import build_hf_data_loader, build_tokenizer
from core.logging_util import init_logger, logger
from core.models import model_name_to_cls, model_name_to_tokenizer, models_config
from core.parallelisms import models_parallelize_fns, ParallelDims
from torch import distributed as dist
from torch.distributed.device_mesh import init_device_mesh

# tokenizer = build_tokenizer(
#     "tiktoken",
#     "./torchtitan/core/datasets/tokenizer/original/tokenizer.model",
# )

# print(tokenizer.encode("hello world"))

from torch.distributed.elastic.multiprocessing.errors import record
from utils import device_module, device_type, init_distributed


@record
def main(job_config: JobConfig):
    init_logger()
    logger.info(f"Starting job: {job_config.job.description}")

    if job_config.job.print_args:
        logger.info(f"Running with args: {job_config.to_dict()}")

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
    init_distributed(job_config)

    # build meshes
    world_mesh = parallel_dims.build_mesh(device_type=device_type)
    logger.info(f"world mesh is {world_mesh}")

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
        return F.cross_entropy(pred.flatten(0, 1).float(), labels.flatten(0, 1))

    # apply 2D parallelism
    models_parallelize_fns[model_name](model, world_mesh, parallel_dims, job_config)
    model.to_empty(device=init_device)
    with torch.no_grad():
        model.init_weights(buffer_device=buffer_device)
    model.train()
    model_parts = [model]


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
