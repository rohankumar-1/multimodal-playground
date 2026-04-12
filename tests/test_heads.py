"""Tests for task heads."""

import pytest
import torch

from multimodal.heads import (
    MultiTaskLinearHead,
    MultiTaskLinearSliceHead,
)


def test_multi_task_linear_head_shared_input() -> None:
    head = MultiTaskLinearHead(
        input_dim=8,
        out_dims={"cls": 2, "reg": 1}
    )
    x = torch.randn(4, 8)
    out = head(x)
    assert out["cls"].shape == (4, 2)
    assert out["reg"].shape == (4, 1)


def test_multi_task_linear_slice_head_per_row() -> None:
    head = MultiTaskLinearSliceHead(
        feat_dim=8,
        out_dims={"cls": 2, "reg": 1}
    )
    x = torch.randn(4, 2, 8)
    out = head(x)
    assert out["cls"].shape == (4, 2)
    assert out["reg"].shape == (4, 1)


def test_slice_head_wrong_task_dim_raises() -> None:
    head = MultiTaskLinearSliceHead(4, {"a": 1, "b": 1})
    with pytest.raises(ValueError, match="task dimension"):
        head(torch.randn(3, 3, 4))
