"""Helpers for ``torch.distributed`` / DDP (``torchrun`` launch).

Contrastive / CLIP-style losses use only the per-rank batch for negatives unless you
implement cross-rank gathering separately; DDP does not change that by default.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class TorchrunEnv:
    rank: int
    world_size: int
    local_rank: int


def infer_torchrun_env() -> TorchrunEnv | None:
    """Read ``RANK`` / ``WORLD_SIZE`` / ``LOCAL_RANK`` set by ``torchrun``."""
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return None
    return TorchrunEnv(
        rank=int(os.environ["RANK"]),
        world_size=int(os.environ["WORLD_SIZE"]),
        local_rank=int(os.environ.get("LOCAL_RANK", 0)),
    )


def init_distributed(backend: str = "nccl") -> None:
    """Call ``init_process_group`` when ``WORLD_SIZE > 1``; no-op otherwise."""
    env = infer_torchrun_env()
    if env is None or env.world_size <= 1:
        return
    if dist.is_initialized():
        return
    dist.init_process_group(backend=backend)


def get_rank() -> int:
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def is_main_process() -> bool:
    return get_rank() == 0


def barrier() -> None:
    if dist.is_initialized() and get_world_size() > 1:
        dist.barrier()


def reduce_mean_dict(metrics: dict[str, float], device: torch.device) -> dict[str, float]:
    """All-reduce scalar metrics and return the global mean per key."""
    if not dist.is_initialized() or get_world_size() <= 1:
        return dict(metrics)
    keys = sorted(metrics.keys())
    if not keys:
        return {}
    t = torch.tensor([metrics[k] for k in keys], device=device, dtype=torch.float32)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t = t / float(get_world_size())
    return {k: float(t[i]) for i, k in enumerate(keys)}


def make_distributed_sampler(
    dataset,
    *,
    shuffle: bool,
    num_replicas: int | None = None,
    rank: int | None = None,
    drop_last: bool = False,
):
    """Build a :class:`~torch.utils.data.distributed.DistributedSampler`."""
    from torch.utils.data.distributed import DistributedSampler

    if num_replicas is None:
        num_replicas = get_world_size()
    if rank is None:
        rank = get_rank()
    return DistributedSampler(
        dataset,
        num_replicas=num_replicas,
        rank=rank,
        shuffle=shuffle,
        drop_last=drop_last,
    )


def wrap_loader_with_distributed_sampler(
    loader: DataLoader,
    *,
    shuffle: bool = True,
) -> DataLoader:
    """Rebuild a :class:`~torch.utils.data.DataLoader` with :class:`DistributedSampler`.

    Requires an initialized process group. Call after ``init_distributed``.

    Raises:
        ValueError: if the loader uses ``batch_sampler`` (not supported).
    """
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler

    if loader.batch_sampler is not None:
        raise ValueError(
            "wrap_loader_with_distributed_sampler does not support DataLoader(batch_sampler=...)"
        )

    ds = loader.dataset
    sampler = DistributedSampler(
        ds,
        num_replicas=get_world_size(),
        rank=get_rank(),
        shuffle=shuffle,
        drop_last=loader.drop_last,
    )

    kwargs = dict(
        batch_size=loader.batch_size,
        sampler=sampler,
        num_workers=loader.num_workers,
        collate_fn=loader.collate_fn,
        pin_memory=loader.pin_memory,
        drop_last=loader.drop_last,
        timeout=loader.timeout,
        worker_init_fn=loader.worker_init_fn,
        multiprocessing_context=loader.multiprocessing_context,
        generator=loader.generator,
        prefetch_factor=loader.prefetch_factor,
        persistent_workers=loader.persistent_workers,
    )
    pmd = getattr(loader, "pin_memory_device", None)
    if pmd is not None:
        kwargs["pin_memory_device"] = pmd
    return DataLoader(ds, **kwargs)  # ty:ignore[invalid-argument-type]
