from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import nn

try:
    from torchvision.ops import MLP as _TorchvisionMLP  # type: ignore[import-untyped]
except Exception:  # pragma: no cover
    _TorchvisionMLP = None


class _MLPFallback(nn.Module):
    """Stack of ``Linear -> (norm) -> activation -> (dropout)`` blocks."""

    def __init__(
        self,
        *,
        in_channels: int,
        hidden_channels: list[int],
        activation_layer: nn.Module | type[nn.Module] | None = None,
        dropout: float = 0.0,
        norm_layer: Callable[[int], nn.Module | None] | nn.Module | None = None,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_channels
        if activation_layer is None:
            act: nn.Module = nn.ReLU()
        elif isinstance(activation_layer, type):
            act = activation_layer()
        else:
            act = activation_layer
        for h in hidden_channels:
            layers.append(nn.Linear(prev, h))
            nmod = _instantiate_norm(norm_layer, h)
            if nmod is not None:
                layers.append(nmod)
            layers.append(act)
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _instantiate_norm(
    norm_layer: Callable[[int], nn.Module | None] | nn.Module | None,
    num_features: int,
) -> nn.Module | None:
    if norm_layer is None:
        return None
    if callable(norm_layer) and not isinstance(norm_layer, nn.Module):
        return norm_layer(num_features)
    # Pre-built module (legacy); cannot adapt feature size — skip.
    return None


class MLP(nn.Module):
    """Fully-connected stack for vector inputs.

    Uses :class:`torchvision.ops.MLP` when torchvision is installed; otherwise a
    small compatible implementation.

    Args:
        in_channels: Input feature size ``D_in``.
        hidden_channels: Sizes of each linear output in order. The module's
            output has feature size ``hidden_channels[-1]``.
        activation_layer: Module instance or class used after each linear. Default is ReLU.
        dropout: Dropout probability after each activation.
        norm_layer: Optional factory ``f(n) -> Module`` returning a norm for
            ``n`` output features (e.g. ``lambda n: nn.BatchNorm1d(n)``).

    Input / output:
        ``forward`` accepts ``(B, D_in)`` and returns ``(B, D_out)`` where
        ``D_out == hidden_channels[-1]``.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        hidden_channels: list[int],
        activation_layer: nn.Module | type[nn.Module] | None = None,
        dropout: float = 0.0,
        norm_layer: Callable[[int], nn.Module | None] | nn.Module | None = None,
    ) -> None:
        super().__init__()
        kwargs: dict[str, Any] = dict(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            activation_layer=activation_layer,
            dropout=dropout,
            norm_layer=norm_layer,
        )
        if _TorchvisionMLP is not None:
            self._impl: nn.Module = _TorchvisionMLP(**kwargs)
        else:
            self._impl = _MLPFallback(**kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._impl(x)


class CrossTaskAttention(nn.Module):
    """Cross-task self-attention over a stack of task embedding sequences.

    Args:
        embed_dim: Channel dimension of each task embedding.
        num_tasks: Number of tasks (sequence length).
        num_heads: Heads for :class:`torch.nn.MultiheadAttention`.

    Input / output:
        ``task_embeddings`` of shape ``(num_tasks, B, embed_dim)`` ->
        same shape after self-attention.
    """

    def __init__(self, embed_dim: int, num_tasks: int, num_heads: int = 2) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.num_tasks = num_tasks

    def forward(self, task_embeddings: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(task_embeddings, task_embeddings, task_embeddings)
        return attn_out
