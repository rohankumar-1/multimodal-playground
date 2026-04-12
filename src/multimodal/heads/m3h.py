"""Multimodal Multitask Head (M3H) from https://arxiv.org/abs/2404.18975.

Expects a **single fused** tensor of shape ``(B, in_dim)`` (e.g. output of
:class:`~multimodal.fusion.common_fusions.ConcatFusion` on modality embeddings).
Returns per-task representations ``(B, n_tasks, attn_dim)``, typically followed by
:class:`~multimodal.heads.basic.MultiTaskLinearSliceHead` inside an
:class:`torch.nn.Sequential`.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class M3HHead(nn.Module):
    """Task-wise attention over a fused multimodal vector."""

    def __init__(self, in_dim: int, attn_dim: int, out_dims: dict[str, int], alpha: float = 1.0) -> None:
        super().__init__()
        self.proj_W = nn.Parameter(
            torch.randn(len(out_dims), attn_dim, in_dim) * (1 / in_dim**0.5)
        )
        self.attn_dim = attn_dim
        self.alpha = alpha
        self.n_tasks = len(out_dims)

        self.WQ = nn.Linear(attn_dim, attn_dim, bias=False)
        self.WK = nn.Linear(attn_dim, attn_dim, bias=False)
        self.WV = nn.Linear(attn_dim, attn_dim, bias=False)
        self.WT = nn.Linear(self.n_tasks, attn_dim, bias=False)

        self.Ts_onehot = F.one_hot(torch.arange(self.n_tasks, dtype=torch.long), num_classes=self.n_tasks).float()
        self.I_base = torch.eye(self.n_tasks)

        self.decoders = {task: nn.Linear(attn_dim, out_dims[task]) for task in out_dims.keys()}

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Map fused features ``x`` of shape ``(B, in_dim)`` to ``(B, n_tasks, attn_dim)``."""
        x = torch.einsum("b i, t o i -> b t o", x, self.proj_W)
        b, t, d = x.shape

        Qs = self.WQ(self.WT(self.Ts_onehot))
        Qs = Qs.unsqueeze(0).expand(b, t, d)

        Ks = self.WK(x)
        Vs = self.WV(x)

        Ms = torch.matmul(Qs, Ks.transpose(1, 2))
        Ms_max = Ms.max(dim=2, keepdim=True)[0].max(dim=1, keepdim=True)[0]
        Ms_norm = Ms / (Ms_max + 1e-8)

        Is = self.I_base.unsqueeze(0).expand(b, t, t)
        logits = Is + self.alpha * Ms_norm
        Ws = F.softmax(logits, dim=-1)

        Os = torch.matmul(Ws, Vs)
        return {
            task: decoder(Os[:, i, :])
            for i, (task, decoder) in enumerate(self.decoders.items())
        }


if __name__ == "__main__":
    from multimodal.fusion.common_fusions import ConcatFusion

    torch.manual_seed(0)
    in_dim = 75
    features = [torch.randn(10, 25), torch.randn(10, 20), torch.randn(10, 30)]
    fused = ConcatFusion(dim=-1)(features)
    head = M3HHead(in_dim=in_dim, attn_dim=32, out_dims={"a": 10, "b": 20, "c": 30}, alpha=1.0)
    print(head(fused))

