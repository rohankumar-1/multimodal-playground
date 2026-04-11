"""Shared pytest configuration (fixtures, hooks)."""

import pytest
import torch


@pytest.fixture(autouse=True)
def _torch_deterministic() -> None:
    torch.manual_seed(0)
