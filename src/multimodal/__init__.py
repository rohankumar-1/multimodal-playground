"""Multimodal models: encoders, fusion, heads, and assembly."""

from multimodal.distributed import (
    barrier,
    infer_torchrun_env,
    init_distributed,
    is_main_process,
    make_distributed_sampler,
    reduce_mean_dict,
    wrap_loader_with_distributed_sampler,
)

__all__ = [
    "barrier",
    "infer_torchrun_env",
    "init_distributed",
    "is_main_process",
    "make_distributed_sampler",
    "reduce_mean_dict",
    "wrap_loader_with_distributed_sampler",
]
__version__ = "0.1.0"
