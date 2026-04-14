"""Tests for MultimodalModel."""

from typing import cast

import pytest
import torch
from torch import nn

from multimodal.fusion import ConcatFusion
from multimodal.heads import M3HHead
from multimodal.model import (
    MultimodalModel,
    MultiviewContrastiveModel,
    UnifiedContrastiveModel,
    UnimodalModel,
)


def test_multimodal_model_forward_returns_predictions_and_embeddings() -> None:
    model = MultimodalModel(
        encoders={"v": nn.Linear(4, 8), "t": nn.Linear(3, 8)},
        fusion=ConcatFusion(dim=-1),
        head=nn.Linear(16, 2),
        fusion_modality_order=["v", "t"],
    )
    batch = {"v": torch.randn(5, 4), "t": torch.randn(5, 3)}
    preds, embs = model(batch)
    assert isinstance(preds, torch.Tensor)
    assert preds.shape == (5, 2)
    assert set(embs.keys()) == {"v", "t"}
    assert embs["v"].shape == (5, 8)
    assert embs["t"].shape == (5, 8)


def test_multimodal_model_predict_returns_logits_only() -> None:
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
    preds = model.predict(batch)
    assert isinstance(preds, torch.Tensor)
    assert preds.shape == (5, 2)


def test_multimodal_model_m3h_head() -> None:
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
    preds, embs = model(batch)
    preds_dict = cast(dict[str, torch.Tensor], preds)
    assert preds_dict["v"].shape == (4, 3)
    assert preds_dict["t"].shape == (4, 1)
    assert embs["v"].shape == (4, embed_dim)

    preds_only = model.predict(batch)
    assert cast(dict[str, torch.Tensor], preds_only)["v"].shape == (4, 3)


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


def test_unimodal_model_forward() -> None:
    model = UnimodalModel(nn.Linear(3, 5), nn.Linear(5, 2), input_key="x")
    batch = {"x": torch.randn(4, 3)}
    preds, embs = model(batch)
    assert preds.shape == (4, 2)
    assert embs["x"].shape == (4, 5)
    assert set(model.encoders.keys()) == {"x"}


def test_unified_contrastive_model_four_embeddings() -> None:
    model = UnifiedContrastiveModel(
        nn.Linear(2, 4),
        nn.Linear(3, 4),
        keys_m1=("a", "a_aug"),
        keys_m2=("b", "b_aug"),
    )
    b = 3
    batch = {
        "a": torch.randn(b, 2),
        "a_aug": torch.randn(b, 2),
        "b": torch.randn(b, 3),
        "b_aug": torch.randn(b, 3),
    }
    preds, embs = model(batch)
    assert preds == {}
    assert set(embs.keys()) == {"a", "a_aug", "b", "b_aug"}
    for t in embs.values():
        assert t.shape == (b, 4)
    assert set(model.encoders.keys()) == {"m1", "m2"}


def test_multiview_contrastive_model_concat_fused() -> None:
    model = MultiviewContrastiveModel(
        nn.Linear(2, 3),
        view_keys=("v0", "v1"),
    )
    batch = {"v0": torch.randn(2, 2), "v1": torch.randn(2, 2)}
    preds, embs = model(batch)
    assert preds == {}
    assert embs["v0"].shape == (2, 3)
    assert embs["v1"].shape == (2, 3)
    assert "encoder" in model.encoders


def test_multiview_contrastive_requires_two_views() -> None:
    with pytest.raises(ValueError, match="at least two"):
        MultiviewContrastiveModel(nn.Identity(), view_keys=("only",))
