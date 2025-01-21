from dataclasses import dataclass
from functools import cached_property

from core.logging_util import logger

from torch.distributed.device_mesh import init_device_mesh


@dataclass
class ParallelDims:
    dp_replicate: int
    dp_shard: int
    cp: int
    tp: int
    pp: int
    world_size: int
    enable_loss_parallel: bool

    def __post_init__(self):
        self._validate()

    def _validate(self):
        dp_replicate, dp_shard, cp, tp, pp = (
            self.dp_replicate,
            self.dp_shard,
            self.cp,
            self.tp,
            self.pp,
        )

        for d in [dp_replicate, cp, tp, pp]:
            assert d >= 1, f"Parallelism {d} must be >= 1 except for dp_shard"

        assert (
            dp_shard == -1 or dp_shard >= 1
        ), f"dp_shard {dp_shard} must be -1 or >= 1"

        if dp_shard < 0:
            self.dp_shard = dp_shard = self.world_size // (dp_replicate * cp * tp * pp)
        assert dp_shard >= 1

        assert dp_replicate * dp_shard * cp * tp * pp == self.world_size, (
            f"Invalid parallel dims: dp_replicate({dp_replicate}) * dp_shard({dp_shard}) * "
            f"cp({cp}) * tp({tp}) * pp({pp}) != WORLD_SIZE({self.world_size})"
        )

    def build_mesh(self, device_type):
        dims = []
        names = []
        for d, name in zip(
            [self.dp_replicate, self.dp_shard, self.cp, self.tp, self.pp],
            ["dp_replicate", "dp_shard", "cp", "tp", "pp"],
        ):
            if d > 1:
                dims.append(d)
                names.append(name)

        logger.info(f"Building device mesh with {dims} and {names}")
        names = tuple(names)
        mesh = init_device_mesh(device_type, dims, mesh_dim_names=names)

        # create a new sub-mesh from the existing mesh by selecting a subset of dimensions and then flattening them into a single dimension.
        dp_mesh_dim_names = []
        dp_shard_cp_mesh_dim_names = []
        dp_cp_mesh_dim_names = []

        if self.dp_replicate_enabled:
            dp_mesh_dim_names.append("dp_replicate")
            dp_cp_mesh_dim_names.append("dp_replicate")
        if self.dp_shard_enabled:
            dp_mesh_dim_names.append("dp_shard")
            dp_shard_cp_mesh_dim_names.append("dp_shard")
            dp_cp_mesh_dim_names.append("dp_shard")
        if self.cp_enabled:
            dp_shard_cp_mesh_dim_names.append("cp")
            dp_cp_mesh_dim_names.append("cp")

        if dp_mesh_dim_names != []:
            mesh[tuple(dp_mesh_dim_names)]._flatten(mesh_dim_name="dp")
        if dp_shard_cp_mesh_dim_names != []:
            mesh[tuple(dp_shard_cp_mesh_dim_names)]._flatten(
                mesh_dim_name="dp_shard_cp"
            )
        if dp_cp_mesh_dim_names != []:
            mesh[tuple(dp_cp_mesh_dim_names)]._flatten(mesh_dim_name="dp_cp")

        return mesh

    @property
    def dp_enabled(self):
        return self.dp_replicate > 1 or self.dp_shard > 1

    @property
    def dp_replicate_enabled(self):
        return self.dp_replicate > 1

    @property
    def dp_shard_enabled(self):
        return self.dp_shard > 1

    @property
    def cp_enabled(self):
        return self.cp > 1

    @property
    def tp_enabled(self):
        return self.tp > 1

    @property
    def pp_enabled(self):
        return self.pp > 1

    @property
    def loss_parallel_enabled(self):
        return self.tp_enabled and self.enable_loss_parallel

    @cached_property
    def non_parallel_data_size(self):
        return self.tp * self.pp * self.cp
