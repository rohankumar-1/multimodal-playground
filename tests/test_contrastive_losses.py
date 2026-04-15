"""Tests for :mod:`multimodal.contrastive_losses`."""

from __future__ import annotations

import torch
from torch import nn

from multimodal.contrastive_losses import (
    CriticInfoNCE,
    InfoNCE,
    SeparableCritic,
    SupConLoss,
    clip_loss,
)
from multimodal.tasks import ContrastiveTask, SupervisedContrastiveTask


def test_clip_loss_matches_infonce() -> None:
    z1 = torch.randn(8, 16)
    z2 = torch.randn(8, 16)
    t = 0.07
    a = clip_loss(z1, z2, t)
    b = InfoNCE(temperature=t, symmetric=True)(z1, z2)
    assert torch.allclose(a, b)


def test_supcon_loss_finite() -> None:
    b, v, d = 4, 2, 8
    z = torch.randn(b, v, d)
    labels = torch.tensor([0, 0, 1, 1])
    loss = SupConLoss(temperature=0.07)(z, labels)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_supervised_contrastive_task_stack_and_loss() -> None:
    embs = {
        "a": torch.randn(4, 8, requires_grad=True),
        "b": torch.randn(4, 8, requires_grad=True),
    }
    batch = {"y": torch.tensor([0, 1, 0, 1])}
    task = SupervisedContrastiveTask("sup", ("a", "b"), "y", temperature=0.1)
    loss, metrics = task.compute_loss({}, embs, batch)
    assert loss.ndim == 0
    assert "sup/loss" in metrics


def test_contrastive_task_default_infonce() -> None:
    embs = {"x": torch.randn(4, 8), "y": torch.randn(4, 8)}
    task = ContrastiveTask("c", "x", "y", temperature=0.1)
    loss, metrics = task.compute_loss({}, embs, {})
    assert loss.ndim == 0


def test_critic_infonce_with_separable_critic() -> None:
    b, d1, d2 = 6, 5, 7
    critic = SeparableCritic(d1, d2, proj_dim=32)
    loss_mod = CriticInfoNCE(critic, temperature=0.1)
    x = torch.randn(b, d1)
    y = torch.randn(b, d2)
    loss = loss_mod(x, y)
    assert loss.ndim == 0
