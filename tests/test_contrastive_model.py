"""Tests for :class:`multimodal.model.ContrastiveModel`."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from multimodal.model import ContrastiveModel


def test_contrastive_model_shared_encoder_two_batch_keys() -> None:
    enc = nn.Linear(2, 3)
    model = ContrastiveModel(
        encoders={"vision": enc, "text": nn.Linear(4, 3)},
        route={
            "image": "vision",
            "image_aug": "vision",
            "text": "text",
        },
    )
    b = 2
    batch = {
        "image": torch.randn(b, 2),
        "image_aug": torch.randn(b, 2),
        "text": torch.randn(b, 4),
    }
    preds, embs = model(batch)
    assert preds == {}
    assert set(embs.keys()) == {"image", "image_aug", "text"}
    assert embs["image"].shape == (b, 3)
    assert embs["image_aug"].shape == (b, 3)
    assert embs["text"].shape == (b, 3)
    assert set(model.encoders.keys()) == {"text", "vision"}


def test_contrastive_model_route_unknown_encoder_raises() -> None:
    with pytest.raises(ValueError, match="not a key in encoders"):
        ContrastiveModel(
            encoders={"a": nn.Identity()},
            route={"x": "missing"},
        )


def test_contrastive_model_missing_batch_key_raises() -> None:
    model = ContrastiveModel(
        encoders={"a": nn.Linear(1, 1)},
        route={"x": "a"},
    )
    with pytest.raises(KeyError, match="batch missing key"):
        model({})


def test_contrastive_model_empty_route_raises() -> None:
    with pytest.raises(ValueError, match="non-empty route"):
        ContrastiveModel(encoders={"a": nn.Identity()}, route={})
