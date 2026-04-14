"""Tests for DDP helper utilities (non-DDP execution path)."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset

from multimodal.distributed import (
    TorchrunEnv,
    barrier,
    infer_torchrun_env,
    is_main_process,
    reduce_mean_dict,
    wrap_loader_with_distributed_sampler,
)


class _TinyDataset(Dataset):
    def __len__(self) -> int:
        return 8

    def __getitem__(self, idx: int) -> int: # ty:ignore[invalid-method-override]
        return idx


def test_infer_torchrun_env_none_when_not_set(monkeypatch) -> None:
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    assert infer_torchrun_env() is None


def test_infer_torchrun_env_reads_vars(monkeypatch) -> None:
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("LOCAL_RANK", "1")
    assert infer_torchrun_env() == TorchrunEnv(rank=3, world_size=8, local_rank=1)


def test_reduce_mean_dict_no_dist_is_identity() -> None:
    metrics = {"a": 1.0, "b": 2.0}
    out = reduce_mean_dict(metrics, device=torch.device("cpu"))
    assert out == metrics


def test_is_main_process_true_without_dist() -> None:
    assert is_main_process() is True


def test_barrier_noop_without_dist() -> None:
    # Should not raise.
    barrier()


def test_wrap_loader_with_distributed_sampler_preserves_batch_size() -> None:
    ds = _TinyDataset()
    loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)
    wrapped = wrap_loader_with_distributed_sampler(loader, shuffle=True)
    assert wrapped.batch_size == 4
    assert len(list(iter(wrapped))) > 0

