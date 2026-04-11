"""Tests for contrastive heads."""

import torch

from multimodal.heads import ModalityContrastiveHead, SupervisedContrastiveHead


def test_modality_contrastive_finite() -> None:
    head = ModalityContrastiveHead(
        modality_dims={"vision": 16, "text": 12},
        proj_dim=8,
        groups=[("vision", "text")],
    )
    b = 6
    emb = {"vision": torch.randn(b, 16), "text": torch.randn(b, 12)}
    loss = head(emb)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_supervised_contrastive_finite() -> None:
    head = SupervisedContrastiveHead(input_dim=16, proj_dim=8)
    z = torch.randn(8, 16)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 0, 1])
    loss = head(z, labels)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_supervised_contrastive_no_classmates_zero_loss() -> None:
    head = SupervisedContrastiveHead(input_dim=8, proj_dim=None)
    z = torch.randn(3, 8)
    labels = torch.tensor([0, 1, 2])
    loss = head(z, labels)
    assert loss.item() == 0.0
