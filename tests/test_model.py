"""Tests for MultimodalModel."""

import pytest
import torch
from torch import nn

from multimodal.fusion import ConcatFusion
from multimodal.heads import M3HHead
from multimodal.model import MultimodalModel


def test_multimodal_model_concat_fusion_list_via_order() -> None:
    fusion = ConcatFusion(dim=-1)
    head = nn.Linear(16, 2)
    model = MultimodalModel(
        encoders={"v": nn.Linear(4, 8), "t": nn.Linear(3, 8)},
        fusion=fusion,
        head=head,
        fusion_modality_order=["v", "t"],
    )
    batch = {
        "v": torch.randn(5, 4),
        "t": torch.randn(5, 3),
    }
    out = model(batch)
    assert out.shape == (5, 2)


def test_multimodal_model_m3h_slice_head() -> None:
    d_v, d_t = 6, 7
    embed_dim = 8
    in_dim = embed_dim * 2
    out_dims = {"v": 3, "t": 1}
    attn_dim = 16
    fusion = ConcatFusion(dim=-1)
    head = M3HHead(in_dim=in_dim, attn_dim=attn_dim, out_dims=out_dims, alpha=1.0)
    model = MultimodalModel(
        encoders={"v": nn.Linear(d_v, embed_dim), "t": nn.Linear(d_t, embed_dim)},
        fusion=fusion,
        head=head,
        fusion_modality_order=["v", "t"],
    )
    batch = {"v": torch.randn(4, d_v), "t": torch.randn(4, d_t)}
    out = model(batch)
    assert out['v'].shape == (4, 3)
    assert out['t'].shape == (4, 1)


def test_multimodal_model_missing_modality_raises() -> None:
    model = MultimodalModel(
        {"v": nn.Identity(), "t": nn.Identity()},
        ConcatFusion(dim=-1),
        nn.Identity(),
        fusion_modality_order=["v", "t"],
    )
    with pytest.raises(KeyError, match="missing modality"):
        model({"v": torch.randn(1, 2)})


def test_multimodal_model_fusion_order_missing_encoded_raises() -> None:
    model = MultimodalModel(
        {"v": nn.Identity(), "t": nn.Identity()},
        ConcatFusion(dim=-1),
        nn.Identity(),
        fusion_modality_order=["v", "t", "audio"],
    )
    with pytest.raises(KeyError, match="encoded missing"):
        model({"v": torch.randn(1, 1), "t": torch.randn(1, 1)})
