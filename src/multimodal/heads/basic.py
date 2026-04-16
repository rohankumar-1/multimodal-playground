from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn

from multimodal.layers.basic import MLP
from multimodal.utils import get_activation, get_norm


def _norm_factory(norm: str | None) -> Callable[[int], nn.Module | None] | None:
    """Return ``lambda c: get_norm(..., num_channels=c)`` for MLP blocks (1D features)."""
    if norm is None or norm == "":
        return None
    return lambda c: get_norm(norm, num_channels=c, dim=1)


class NoOpHead(nn.Module):
    """Head that returns an empty dict (no supervised outputs).

    Use when training only uses encoder paths in ``embs`` (for example
    :class:`~multimodal.tasks.ContrastiveTask` with
    :func:`~multimodal.losses.clip_loss`). :class:`~multimodal.model.MultimodalModel`
    still runs fusion and this head so the forward API stays unchanged.

    Input / output:
        Ignores ``fused`` of shape ``(B, D)``. Returns ``{}``.
    """

    def forward(self, _fused: torch.Tensor) -> dict[str, torch.Tensor]:
        return {}


class MultiTaskHead(nn.Module):
    """Apply a separate decoder module per task to the same fused vector.

    Args:
        input_dim: Fused feature size ``D`` (for documentation; decoders must
            accept tensors of shape ``(B, D)``).
        decoders: Map ``task_name -> nn.Module``; each module maps ``(B, D)`` to
            that task's output shape (e.g. logits ``(B, num_classes)``).

    Input / output:
        ``x`` of shape ``(B, D)``. Returns a dict ``task -> decoder(x)`` with
        per-task tensor shapes defined by the decoders.
    """

    def __init__(self, input_dim: int, decoders: dict[str, nn.Module], **kwargs) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.decoders = decoders

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {task: decoder(x) for task, decoder in self.decoders.items()}


class MultiTaskLinearHead(MultiTaskHead):
    """One linear layer per task on a shared fused representation.

    Args:
        input_dim: Fused size ``D``.
        out_dims: ``task -> output`` feature size (e.g. number of classes).

    Input / output:
        ``(B, D)`` -> ``{task: (B, out_dims[task])}``.
    """

    def __init__(self, input_dim: int, out_dims: dict[str, int], **kwargs) -> None:
        super().__init__(
            input_dim,
            {task: nn.Linear(input_dim, out_dims[task]) for task in out_dims.keys()},
        )


class MultiTaskMLPHead(MultiTaskHead):
    """Task-specific MLP decoders on a shared fused vector.

    Args:
        input_dim: Fused size ``D``.
        hidden_channels: ``task -> [h1, h2, ...]`` hidden widths; the last entry
            is the output size for that task's MLP (same contract as
            :class:`~multimodal.layers.basic.MLP`).
        activation: Name passed to :func:`~multimodal.utils.get_activation`.
        dropout: Dropout probability inside each MLP.
        norm: Normalization name for :func:`~multimodal.utils.get_norm` with
            ``dim=1`` and per-layer ``num_channels`` matching each linear output.

    Input / output:
        ``(B, D)`` -> ``{task: (B, hidden_channels[task][-1])}``.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_channels: dict[str, list[int]],
        activation: str = "relu",
        dropout: float = 0.0,
        norm: str | None = "bn",
    ) -> None:
        super().__init__(
            input_dim,
            decoders={
                task: MLP(
                    in_channels=input_dim,
                    hidden_channels=hidden_channels[task],
                    activation_layer=get_activation(activation),
                    dropout=dropout,
                    norm_layer=_norm_factory(norm),
                )
                for task in hidden_channels.keys()
            },
        )


class MultiTaskSliceHead(nn.Module):
    """One decoder per task, each applied to its own row of a task-major tensor.

    Use with fusion or heads that emit ``(B, T, D)`` where ``T`` is the number
    of tasks and row ``t`` is the feature vector for task ``t``.

    Args:
        decoders: ``task -> module``; each maps ``(B, D)`` to task outputs.

    Input / output:
        ``x`` of shape ``(B, T, D)`` with ``T == len(decoders)``. Returns
        ``{task: decoder(x[:, i, :])}`` in stable key order.

    See Also:
        :class:`MultiTaskHead` — same ``(B, D)`` fed to every decoder.
    """

    def __init__(self, decoders: dict[str, nn.Module]) -> None:
        super().__init__()
        self.decoders = nn.ModuleDict(decoders)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.dim() != 3:
            raise ValueError(f"expected (B, T, D), got shape {tuple(x.shape)}")
        tdim = x.size(1)
        if tdim != len(self.decoders):
            raise ValueError(
                f"task dimension T={tdim} must equal len(decoders)={len(self.decoders)}"
            )
        return {task: decoder(x[:, i, :]) for i, (task, decoder) in enumerate(self.decoders.items())}


class MultiTaskLinearSliceHead(MultiTaskSliceHead):
    """Linear decoders on each ``(B, D)`` slice of ``(B, T, D)``.

    Args:
        feat_dim: Per-task feature size ``D``.
        out_dims: ``task -> output`` size for each :class:`torch.nn.Linear`.

    Input / output:
        ``(B, T, D)`` -> ``{task: (B, out_dims[task])}``.
    """

    def __init__(self, feat_dim: int, out_dims: dict[str, int], **kwargs) -> None:
        super().__init__({task: nn.Linear(feat_dim, out_dims[task]) for task in out_dims.keys()})


class MultiTaskMLPSliceHead(MultiTaskSliceHead):
    """MLP decoders on each ``(B, D)`` slice of ``(B, T, D)``.

    Args:
        feat_dim: Per-task input size ``D`` (same for every task).
        out_dims: Reserved for API symmetry with linear slice heads; each MLP's
            final width should be the last element of ``hidden_channels[task]``.
        hidden_channels: ``task -> [h1, ..., hL]``; output for that task is
            ``(B, hL)``.
        activation: Passed to :func:`~multimodal.utils.get_activation`.
        dropout: Dropout inside each MLP.
        norm: Passed to :func:`~multimodal.utils.get_norm` (1D, per hidden width).

    Input / output:
        ``(B, T, D)`` -> ``{task: (B, hidden_channels[task][-1])}``.
    """

    def __init__(
        self,
        feat_dim: int,
        out_dims: dict[str, int],
        hidden_channels: dict[str, list[int]],
        activation: str = "relu",
        dropout: float = 0.0,
        norm: str | None = "bn",
        **kwargs: object,
    ) -> None:
        _ = out_dims  # Kept for API parity with :class:`MultiTaskLinearSliceHead`.
        _ = kwargs
        super().__init__(
            decoders={
                task: MLP(
                    in_channels=feat_dim,
                    hidden_channels=hidden_channels[task],
                    activation_layer=get_activation(activation),
                    dropout=dropout,
                    norm_layer=_norm_factory(norm),
                )
                for task in hidden_channels.keys()
            },
        )
