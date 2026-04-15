"""Contrastive / InfoNCE-style losses (``nn.Module`` where state is needed).

Contrastive training objectives live here; :class:`~multimodal.tasks.ContrastiveTask` and
:class:`~multimodal.tasks.SupervisedContrastiveTask` wrap them for use with :class:`~multimodal.train.Trainer`.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def clip_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    """Symmetric CLIP-style loss (same as :class:`InfoNCE` with ``symmetric=True``)."""
    return InfoNCE(temperature=temperature, symmetric=True)(z1, z2)


class InfoNCE(nn.Module):
    """Bidirectional or one-way InfoNCE on paired embeddings ``z1``, ``z2`` of shape ``[B, D]``."""

    def __init__(self, temperature: float = 0.07, symmetric: bool = True) -> None:
        super().__init__()
        self.temperature = temperature
        self.symmetric = symmetric

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        logits = z1 @ z2.T / self.temperature
        labels = torch.arange(z1.size(0), device=z1.device, dtype=torch.long)
        loss_i = F.cross_entropy(logits, labels)
        if self.symmetric:
            loss_j = F.cross_entropy(logits.T, labels)
            return (loss_i + loss_j) * 0.5
        return loss_i


class CriticInfoNCE(nn.Module):
    """InfoNCE where similarities come from ``critic(x, y)`` → ``[B, B]`` (e.g. separable bilinear)."""

    def __init__(
        self,
        critic: nn.Module,
        temperature: float = 0.07,
        symmetric: bool = True,
    ) -> None:
        super().__init__()
        self.critic = critic
        self.temperature = temperature
        self.symmetric = symmetric

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        logits = self.critic(x, y) / self.temperature
        labels = torch.arange(logits.size(0), device=logits.device, dtype=torch.long)
        loss_i = F.cross_entropy(logits, labels)
        if self.symmetric:
            loss_j = F.cross_entropy(logits.T, labels)
            return (loss_i + loss_j) * 0.5
        return loss_i


class SeparableCritic(nn.Module):
    """Bilinear similarity ``f(x, y) = g(x)ᵀ h(y)`` with L2-normalized projected features."""

    def __init__(self, in_dim1: int, in_dim2: int, proj_dim: int = 256) -> None:
        super().__init__()
        self.g = nn.Linear(in_dim1, proj_dim)
        self.h = nn.Linear(in_dim2, proj_dim)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        gx = F.normalize(self.g(x), dim=-1)
        hy = F.normalize(self.h(y), dim=-1)
        return gx @ hy.T


class SupConLoss(nn.Module):
    """Supervised contrastive loss (Khosla et al., 2020).

    Expects ``features`` of shape ``[B, V, D]`` (``V`` views per sample) and ``labels`` of
    shape ``[B]``. Positives are pairs of augmented samples from the **same class**; the
    loss is the standard multi-positive InfoNCE formulation over the ``B * V`` batch.
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        if features.dim() != 3:
            raise ValueError(f"features must be [B, V, D], got shape {tuple(features.shape)}")
        device = features.device
        bsz, n_view, dim = features.shape
        labels = labels.view(-1).long()
        if labels.shape[0] != bsz:
            raise ValueError("labels must have shape [B]")

        features = F.normalize(features, dim=-1)
        # [B*V, D]
        f = features.reshape(bsz * n_view, dim)
        # Repeat each label for every view
        labels_exp = labels.view(-1, 1).expand(bsz, n_view).reshape(-1)

        sim = torch.matmul(f, f.T) / self.temperature
        # Positive mask: same class, exclude self
        mask = torch.eq(labels_exp.unsqueeze(0), labels_exp.unsqueeze(1)).float()
        logits_mask = 1.0 - torch.eye(bsz * n_view, device=device)
        mask = mask * logits_mask

        # For each anchor, log-softmax over all *other* positions
        exp_sim = torch.exp(sim) * logits_mask
        denom = exp_sim.sum(1, keepdim=True).clamp_min(1e-8)
        log_prob = sim - torch.log(denom)

        # Mean log-prob over positive pairs (excluding self already zeroed in mask diagonal path)
        n_pos = mask.sum(1).clamp_min(1.0)
        mean_log_pos = (mask * log_prob).sum(1) / n_pos
        loss = -mean_log_pos.mean()
        return loss


class CLUB(nn.Module):
    """CLUB: Contrastive Log-ratio Upper Bound on mutual information (Cheng et al., 2020)."""

    def __init__(self, x_dim: int, y_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.mu_net = nn.Sequential(
            nn.Linear(x_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, y_dim),
        )
        self.logvar_net = nn.Sequential(
            nn.Linear(x_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, y_dim),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        mu = self.mu_net(x)
        logvar = self.logvar_net(x).clamp(-10, 10)
        var = logvar.exp().clamp_min(1e-6)

        positive = -0.5 * ((y - mu) ** 2) / var - 0.5 * logvar

        perm = torch.randperm(y.size(0), device=y.device)
        y_shuffle = y[perm]
        negative = -0.5 * ((y_shuffle - mu) ** 2) / var - 0.5 * logvar

        return -(positive.sum(dim=-1) - negative.sum(dim=-1)).mean()
