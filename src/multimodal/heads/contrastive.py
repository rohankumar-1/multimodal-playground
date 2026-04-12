from __future__ import annotations

from itertools import combinations

import torch
import torch.nn.functional as F
from torch import nn


def _clip_symmetric_infonce(z_a: torch.Tensor, z_b: torch.Tensor, temperature: float) -> torch.Tensor:
    """Symmetric CLIP-style InfoNCE for two L2-normalized embedding matrices [B, D]."""
    logits = (z_a @ z_b.T) / temperature
    targets = torch.arange(z_a.size(0), device=z_a.device)
    loss_a = F.cross_entropy(logits, targets)
    loss_b = F.cross_entropy(logits.T, targets)
    return 0.5 * (loss_a + loss_b)


def _supervised_contrastive_loss(z: torch.Tensor, labels: torch.Tensor, temperature: float) -> torch.Tensor:
    """Multi-positive supervised contrastive loss (Khosla et al.), batch-only.

    'z' must be L2-normalized along the feature dimension. Rows with no same-class
    peer (other than self) are skipped.
    """
    device = z.device
    b = z.size(0)
    labels = labels.view(-1, 1)
    same_class = torch.eq(labels, labels.T).float().to(device)
    not_self = 1.0 - torch.eye(b, device=device)
    pos_mask = same_class * not_self

    sim = (z @ z.T) / temperature
    logits = sim - sim.max(dim=1, keepdim=True).values.detach()

    exp_logits = torch.exp(logits) * not_self
    denom = exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12)
    log_prob = logits - torch.log(denom)

    pos_count = pos_mask.sum(dim=1)
    valid = pos_count > 0
    if not torch.any(valid):
        return z.sum() * 0.0

    mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / pos_count.clamp_min(1e-12)
    return (-mean_log_prob_pos[valid]).mean()


class ModalityContrastiveHead(nn.Module):
    """Align embeddings across modalities (same batch index = positive pair).

    For each unordered pair of modalities inside a group, applies symmetric
    CLIP-style InfoNCE. Groups are averaged with equal weight; within a group,
    all unordered modality pairs are averaged.
    """

    def __init__(
        self,
        modality_dims: dict[str, int],
        proj_dim: int,
        groups: list[list[str]],
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.groups: list[tuple[str, ...]] = [tuple(g) for g in groups]
        for g in self.groups:
            if len(g) < 2:
                raise ValueError(f"each group must have at least 2 modalities, got {g!r}")
            for m in g:
                if m not in modality_dims:
                    raise KeyError(f"modality {m!r} not in modality_dims")

        self.projections = nn.ModuleDict(
            {name: nn.Linear(dim, proj_dim) for name, dim in modality_dims.items()}
        )

    def forward(self, embeddings: dict[str, torch.Tensor]) -> torch.Tensor:
        projected: dict[str, torch.Tensor] = {
            name: F.normalize(self.projections[name](embeddings[name]), dim=-1)
            for name in self.projections
        }
        losses: list[torch.Tensor] = []
        for group in self.groups:
            pair_losses: list[torch.Tensor] = []
            for a, b in combinations(group, 2):
                if a not in projected or b not in projected:
                    raise KeyError(
                        f"group {group!r} references modalities missing from embeddings"
                    )
                pair_losses.append(
                    _clip_symmetric_infonce(
                        projected[a], projected[b], self.temperature
                    )
                )
            losses.append(torch.stack(pair_losses).mean())
        return torch.stack(losses).mean()


class SupervisedContrastiveHead(nn.Module):
    """Pull same-class embeddings together and push different-class ones apart."""

    def __init__(self, input_dim: int, proj_dim: int|None = None, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature
        self.proj: nn.Module
        if proj_dim is None:
            self.proj = nn.Identity()
            self._out_dim = input_dim
        else:
            self.proj = nn.Linear(input_dim, proj_dim)
            self._out_dim = proj_dim

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """ returns the loss for a batch of embeddings """
        h = self.proj(z)
        h = F.normalize(h, dim=-1)
        return _supervised_contrastive_loss(h, labels, self.temperature)


if __name__ == "__main__":
    torch.manual_seed(0)

    b = 8
    # Modality contrastive: one pair + one triple (three pairwise terms)
    head_m = ModalityContrastiveHead(
        modality_dims={"vision": 32, "text": 24, "audio": 16},
        proj_dim=16,
        groups=[["vision", "text"], ["vision", "text", "audio"]],
    )
    emb = {
        "vision": torch.randn(b, 32),
        "text": torch.randn(b, 24),
        "audio": torch.randn(b, 16),
    }
    loss_m = head_m(emb)
    assert loss_m.ndim == 0
    assert torch.isfinite(loss_m)

    try:
        ModalityContrastiveHead(
            modality_dims={"a": 4, "b": 4},
            proj_dim=8,
            groups=[["a"]],
        )
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for single-modality group")

    # Supervised contrastive: with projection and identity
    z = torch.randn(b, 32)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 0, 1])
    head_s = SupervisedContrastiveHead(input_dim=32, proj_dim=16)
    loss_s = head_s(z, labels)
    assert loss_s.ndim == 0
    assert torch.isfinite(loss_s)

    head_id = SupervisedContrastiveHead(input_dim=32, proj_dim=None)
    loss_id = head_id(z, labels)
    assert torch.isfinite(loss_id)

    # No same-class peer: loss is zero scalar (no valid anchors)
    z_solo = torch.randn(3, 32)
    labels_solo = torch.tensor([0, 1, 2])
    loss_solo = head_id(z_solo, labels_solo)
    assert loss_solo.item() == 0.0

    print("ModalityContrastiveHead loss:", float(loss_m))
    print("SupervisedContrastiveHead loss:", float(loss_s))
    print("SupervisedContrastiveHead (identity proj) loss:", float(loss_id))
    print("SupervisedContrastiveHead (no class peers):", float(loss_solo))
    print("contrastive __main__ checks passed.")
