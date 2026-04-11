"""Training utilities for multimodal models.

**Supervised heads** (classification, regression, multi-task dict outputs): implement
``loss_fn`` to compare ``predictions`` to targets carried in ``batch`` (e.g.
``batch["labels"]`` or per-task keys).

**Contrastive objectives** (no prediction tensor, or auxiliary to supervised): use
``encoded`` from the same forward—per-modality embeddings before fusion—and add
their loss inside ``loss_fn``. Example::

    def loss_fn(model, batch, predictions, encoded):
        task = F.cross_entropy(predictions, batch["labels"])
        ctr = modality_contrastive(encoded)  # your head on encoded dict
        return task + 0.1 * ctr, {"task": task.item(), "ctr": ctr.item()}

``predictions`` may be unused if you train only on contrastive terms; still pass a
dummy head or ignore the first return value.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union

import torch
from torch import nn
from torch.optim import Optimizer

from multimodal.model import MultimodalModel

LossFn = Callable[
    [nn.Module, Dict[str, Any], Union[torch.Tensor, Dict[str, torch.Tensor]], Dict[str, torch.Tensor]],
    Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, float]]],
]


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Move top-level tensor values in a batch dict to ``device`` (non-tensors unchanged)."""
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def _unpack_loss(out: Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, float]]]) -> Tuple[torch.Tensor, Dict[str, float]]:
    if isinstance(out, torch.Tensor):
        return out, {}
    loss, metrics = out
    return loss, metrics


def train_step(
    model: MultimodalModel,
    batch: Dict[str, Any],
    optimizer: Optimizer,
    loss_fn: LossFn,
    *,
    max_grad_norm: Optional[float] = None,
) -> Dict[str, float]:
    """Single optimization step: forward (predictions + encoded), ``loss_fn``, backward.

    Returns metrics including ``loss`` and any keys returned by ``loss_fn``.
    """
    model.train()
    optimizer.zero_grad(set_to_none=True)
    predictions, encoded = model.forward_with_encoded(batch)
    raw = loss_fn(model, batch, predictions, encoded)
    loss, extra = _unpack_loss(raw)
    loss.backward()
    if max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    metrics: Dict[str, float] = {"loss": float(loss.detach())}
    metrics.update({k: float(v) for k, v in extra.items()})
    return metrics


@torch.no_grad()
def eval_step(
    model: MultimodalModel,
    batch: Dict[str, Any],
    loss_fn: LossFn,
) -> Dict[str, float]:
    """Evaluate ``loss_fn`` on one batch (no backward)."""
    model.eval()
    predictions, encoded = model.forward_with_encoded(batch)
    raw = loss_fn(model, batch, predictions, encoded)
    loss, extra = _unpack_loss(raw)
    metrics: Dict[str, float] = {"loss": float(loss.detach())}
    metrics.update({k: float(v) for k, v in extra.items()})
    return metrics


def train_epoch(
    model: MultimodalModel,
    data_loader: Iterable[Dict[str, Any]],
    optimizer: Optimizer,
    loss_fn: LossFn,
    device: torch.device,
    *,
    max_grad_norm: Optional[float] = None,
) -> Dict[str, float]:
    """One pass over ``data_loader``; returns mean of each metric."""
    totals: Dict[str, float] = {}
    n = 0
    for batch in data_loader:
        batch = move_batch_to_device(batch, device)
        metrics = train_step(
            model, batch, optimizer, loss_fn, max_grad_norm=max_grad_norm
        )
        for k, v in metrics.items():
            totals[k] = totals.get(k, 0.0) + v
        n += 1
    if n == 0:
        return {}
    return {k: totals[k] / n for k in totals}


@torch.no_grad()
def evaluate(
    model: MultimodalModel,
    data_loader: Iterable[Dict[str, Any]],
    loss_fn: LossFn,
    device: torch.device,
) -> Dict[str, float]:
    """Aggregate ``eval_step`` metrics over the loader."""
    totals: Dict[str, float] = {}
    n = 0
    for batch in data_loader:
        batch = move_batch_to_device(batch, device)
        metrics = eval_step(model, batch, loss_fn)
        for k, v in metrics.items():
            totals[k] = totals.get(k, 0.0) + v
        n += 1
    if n == 0:
        return {}
    return {k: totals[k] / n for k in totals}


def train(
    model: MultimodalModel,
    train_loader: Iterable[Dict[str, Any]],
    optimizer: Optimizer,
    loss_fn: LossFn,
    device: torch.device,
    *,
    num_epochs: int = 1,
    val_loader: Optional[Iterable[Dict[str, Any]]] = None,
    max_grad_norm: Optional[float] = None,
    epoch_callback: Optional[Callable[[int, Dict[str, float], Optional[Dict[str, float]]], None]] = None,
) -> None:
    """Top-level training loop over ``num_epochs``.

    If ``val_loader`` is set, runs :func:`evaluate` after each epoch.
    ``epoch_callback(epoch, train_metrics, val_metrics_or_none)`` is optional.
    """
    for epoch in range(num_epochs):
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
            max_grad_norm=max_grad_norm,
        )
        val_metrics: Optional[Dict[str, float]] = None
        if val_loader is not None:
            val_metrics = evaluate(model, val_loader, loss_fn, device)
        if epoch_callback is not None:
            epoch_callback(epoch, train_metrics, val_metrics)
