"""Tests for MultimodalModel."""

import pytest
import torch
from torch import nn

from multimodal.fusion import ConcatFusion, M3HFusion
from multimodal.model import MultimodalModel
from multimodal.heads import MultiTaskLinearSliceHead


def test_multimodal_model_concat_fusion_list_via_order() -> None:
    enc = {"v": nn.Linear(4, 8), "t": nn.Linear(3, 8)}
    fusion = ConcatFusion(dim=-1)
    head = nn.Linear(16, 2)
    model = MultimodalModel(enc, fusion, head, fusion_modality_order=("v", "t"))
    batch = {"v": torch.randn(5, 4), "t": torch.randn(5, 3)}
    out = model(batch)
    assert out.shape == (5, 2)


def test_multimodal_model_m3h_slice_head() -> None:
    d_v, d_t = 6, 7
    in_dim = d_v + d_t
    n_tasks = 2
    attn_dim = 16
    enc = {"v": nn.Linear(d_v, 8), "t": nn.Linear(d_t, 8)}
    inner = M3HFusion(
        ConcatFusion(dim=-1),
        in_dim=in_dim,
        n_tasks=n_tasks,
        attn_dim=attn_dim,
        alpha=1.0,
    )
    head = MultiTaskLinearSliceHead(
        feat_dim=attn_dim,
        out_dims={"a": 3, "b": 1},
        tasks=["a", "b"],
    )
    model = MultimodalModel(
        enc,
        inner,
        head,
        fusion_modality_order=("v", "t"),
    )
    batch = {"v": torch.randn(4, d_v), "t": torch.randn(4, d_t)}
    out = model(batch)
    assert set(out.keys()) == {"a", "b"}
    assert out["a"].shape == (4, 3)
    assert out["b"].shape == (4, 1)


def test_multimodal_model_missing_modality_raises() -> None:
    model = MultimodalModel(
        {"v": nn.Linear(2, 2), "t": nn.Linear(2, 2)},
        ConcatFusion(dim=-1),
        nn.Identity(),
        fusion_modality_order=("v", "t"),
    )
    with pytest.raises(KeyError, match="missing modality"):
        model({"v": torch.randn(1, 2)})


def test_multimodal_model_fusion_order_missing_encoded_raises() -> None:
    model = MultimodalModel(
        {"v": nn.Identity(), "t": nn.Identity()},
        ConcatFusion(dim=-1),
        nn.Identity(),
        fusion_modality_order=("v", "t", "audio"),
    )
    with pytest.raises(KeyError, match="encoded missing"):
        model({"v": torch.randn(1, 1), "t": torch.randn(1, 1)})
