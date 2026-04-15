from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn

from multimodal.losses import (
    CLUB,
    InfoNCE,
    SupConLoss,
    dice_bce_loss,
    dice_loss,
    multiclass_dice_loss,
)


class BaseTask:
    def __init__(self, name: str, weight: float = 1.0):
        self.name = name
        self.weight = weight

    def trainable_loss_modules(self) -> tuple[nn.Module, ...]:
        """Extra :class:`~torch.nn.Module` objects whose parameters should train with the model.

        Used with :func:`~multimodal.train.iter_training_parameters` when building the
        optimizer. Override in tasks that keep a stateful loss (e.g. critic InfoNCE).
        Modules already registered as submodules of ``model`` need not be listed; parameters
        are deduplicated by identity.
        """
        return ()

    @abstractmethod
    def compute_loss(self, preds, embs, batch) -> tuple[torch.Tensor, dict[str, float]]:
        pass


class ClassificationTask(BaseTask):
    def __init__(self, name, target_key, weight=1.0):
        super().__init__(name, weight)
        self.target_key = target_key

    def compute_loss(self, preds, embs, batch):
        logits = preds[self.name]             # shape: [B, C]
        target = batch[self.target_key]       # shape: [B]

        loss = F.cross_entropy(logits, target)

        pred_labels = logits.argmax(dim=1)
        acc = (pred_labels == target).float().mean()

        metrics = {f"{self.name}/loss": loss.item(), f"{self.name}/acc": acc.item()}
        return loss * self.weight, metrics


class MultiLabelTask(BaseTask):
    def __init__(self, name, target_key, weight=1.0):
        super().__init__(name, weight)
        self.target_key = target_key

    def compute_loss(self, preds, embs, batch):
        logits = preds[self.name]
        target = batch[self.target_key].float()

        loss = F.binary_cross_entropy_with_logits(logits, target)

        metrics = {f"{self.name}/loss": loss.item()}
        return loss * self.weight, metrics


class MultiTaskTask(BaseTask):
    def __init__(self, name, head_to_target: dict, weight=1.0):
        super().__init__(name, weight)
        self.head_to_target = head_to_target

    def compute_loss(self, preds, embs, batch):
        total = 0
        metrics = {}

        for head_name, target_key in self.head_to_target.items():
            head_logits = preds[head_name]
            target = batch[target_key]
            loss = F.cross_entropy(head_logits, target)

            total += loss
            metrics[f"{head_name}/loss"] = loss.item()

        return total * self.weight, metrics


class RegressionTask(BaseTask):
    def __init__(self, name, target_key, weight=1.0):
        super().__init__(name, weight)
        self.target_key = target_key

    def compute_loss(self, preds, embs, batch):
        pred = preds[self.name]
        target = batch[self.target_key]

        loss = F.mse_loss(pred, target)

        metrics = {f"{self.name}/rmse": loss.sqrt().item()}
        return loss * self.weight, metrics


class ContrastiveTask(BaseTask):
    """Pairwise contrastive loss on two ``embs`` keys (e.g. two modalities or two views).

    **Loss construction (current API).** The default is symmetric
    :class:`~multimodal.losses.InfoNCE` driven by ``temperature``. For anything else (critic
    InfoNCE, asymmetric temperature, custom logits), pass an ``nn.Module`` or callable as
    ``loss_fn``; it is invoked as ``loss_fn(embs[mod1], embs[mod2])``. That keeps the task
    thin and avoids a growing matrix of constructor flags; trainable auxiliaries (e.g. a
    bilinear critic) live on ``loss_fn``. Pass :func:`~multimodal.train.iter_training_parameters`
    ``(model, tasks, …)`` into your optimizer so those weights are updated (see
    :meth:`trainable_loss_modules`).

    **Possible evolution.** A small factory (e.g. ``loss="infonce" | "critic"`` plus shared
    kwargs) could wrap the same ``loss_fn`` slot for discoverability, as long as advanced
    cases still pass a custom module unchanged.
    """

    def __init__(
        self,
        name: str,
        mod1: str,
        mod2: str,
        *,
        loss_fn: nn.Module | Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        temperature: float = 0.07,
        weight: float = 1.0,
    ) -> None:
        super().__init__(name, weight)
        self.mod1 = mod1
        self.mod2 = mod2
        if loss_fn is None:
            self.loss_fn: nn.Module | Callable[..., torch.Tensor] = InfoNCE(
                temperature=temperature, symmetric=True
            )
        else:
            self.loss_fn = loss_fn

    def trainable_loss_modules(self) -> tuple[nn.Module, ...]:
        if isinstance(self.loss_fn, nn.Module):
            return (self.loss_fn,)
        return ()

    def compute_loss(self, preds, embs, batch):
        z1 = embs[self.mod1]
        z2 = embs[self.mod2]
        loss = self.loss_fn(z1, z2)
        metrics = {f"{self.name}/loss": loss.item()}
        return loss * self.weight, metrics


class SupervisedContrastiveTask(BaseTask):
    """Supervised contrastive (multi-view) on stacked ``embs`` and class labels.

    Stacks ``embs[k]`` for each ``k`` in ``view_keys`` along a view dimension → ``[B, V, D]``,
    then applies ``loss_module`` (default: :class:`~multimodal.losses.SupConLoss`).
    ``batch[label_key]`` must be integer class ids of shape ``[B]``.
    """

    def __init__(
        self,
        name: str,
        view_keys: tuple[str, ...],
        label_key: str,
        *,
        loss_module: nn.Module | None = None,
        temperature: float = 0.07,
        weight: float = 1.0,
    ) -> None:
        super().__init__(name, weight)
        if len(view_keys) < 2:
            raise ValueError("SupervisedContrastiveTask needs at least two view_keys")
        self.view_keys = tuple(view_keys)
        self.label_key = label_key
        self.loss_module = (
            loss_module if loss_module is not None else SupConLoss(temperature=temperature)
        )

    def trainable_loss_modules(self) -> tuple[nn.Module, ...]:
        return (self.loss_module,)

    def compute_loss(self, preds, embs, batch):
        for k in self.view_keys:
            if k not in embs:
                raise KeyError(f"SupervisedContrastiveTask: embs missing key {k!r}")
        if self.label_key not in batch:
            raise KeyError(f"SupervisedContrastiveTask: batch missing label_key {self.label_key!r}")

        z = torch.stack([embs[k] for k in self.view_keys], dim=1)
        labels = batch[self.label_key].long()
        loss = self.loss_module(z, labels)
        metrics = {f"{self.name}/loss": loss.item()}
        return loss * self.weight, metrics


class FactorCLSupervisedTask(BaseTask):
    """FactorCL-style supervised objective with conditional and MI-penalty terms.

    Implements the flow described by the user:

    - z1, z2 are taken from ``embs`` (typically encoder outputs, optionally projected)
    - y_emb = label_embed(y)
    - InfoNCE terms:
      - shared: InfoNCE(z1, z2)
      - x1y:    InfoNCE(z1, y_emb)
      - x2y:    InfoNCE(z2, y_emb)
      - cond:   InfoNCE([z1,y_emb], [z2,y_emb])
    - CLUB penalties:
      - club:      CLUB(z1, z2)
      - club_cond: CLUB([z1,y_emb], [z2,y_emb])
    - total: L_total = sum(InfoNCE terms) - club_lambda * sum(CLUB terms)
    """

    def __init__(
        self,
        name: str,
        x1_key: str,
        x2_key: str,
        label_key: str,
        *,
        num_classes: int,
        embed_dim: int,
        temperature: float = 0.07,
        club_hidden: int = 256,
        club_lambda: float = 1.0,
        weight: float = 1.0,
        normalize_inputs_for_club: bool = True,
    ) -> None:
        super().__init__(name, weight)
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive")

        self.x1_key = x1_key
        self.x2_key = x2_key
        self.label_key = label_key

        self.label_embed = nn.Embedding(num_classes, embed_dim)
        self.infonce = InfoNCE(temperature=temperature, symmetric=True)

        self.club = CLUB(embed_dim, embed_dim, hidden=club_hidden)
        self.club_cond = CLUB(embed_dim * 2, embed_dim * 2, hidden=club_hidden)
        self.club_lambda = float(club_lambda)
        self.normalize_inputs_for_club = bool(normalize_inputs_for_club)

    def trainable_loss_modules(self) -> tuple[nn.Module, ...]:
        return (self.label_embed, self.infonce, self.club, self.club_cond)

    def compute_loss(self, preds, embs, batch):
        if self.x1_key not in embs:
            raise KeyError(f"{self.__class__.__name__}: embs missing x1_key {self.x1_key!r}")
        if self.x2_key not in embs:
            raise KeyError(f"{self.__class__.__name__}: embs missing x2_key {self.x2_key!r}")
        if self.label_key not in batch:
            raise KeyError(
                f"{self.__class__.__name__}: batch missing label_key {self.label_key!r}"
            )

        z1 = embs[self.x1_key]
        z2 = embs[self.x2_key]
        if z1.dim() != 2 or z2.dim() != 2:
            raise ValueError(
                f"{self.__class__.__name__}: z1/z2 must be [B, D]; "
                f"got {tuple(z1.shape)} and {tuple(z2.shape)}"
            )
        if z1.shape[0] != z2.shape[0]:
            raise ValueError(
                f"{self.__class__.__name__}: z1 and z2 must share batch dim; "
                f"got {z1.shape[0]} and {z2.shape[0]}"
            )
        if z1.shape[1] != z2.shape[1]:
            raise ValueError(
                f"{self.__class__.__name__}: z1 and z2 must share embed dim; "
                f"got {z1.shape[1]} and {z2.shape[1]}"
            )
        if z1.shape[1] != self.label_embed.embedding_dim:
            raise ValueError(
                f"{self.__class__.__name__}: embs dim {z1.shape[1]} must match "
                f"label_embed dim {self.label_embed.embedding_dim}"
            )

        y = batch[self.label_key].view(-1).long()
        if y.shape[0] != z1.shape[0]:
            raise ValueError(
                f"{self.__class__.__name__}: labels must have shape [B]; got {tuple(y.shape)} "
                f"for B={z1.shape[0]}"
            )
        y_emb = self.label_embed(y)

        # InfoNCE terms (InfoNCE normalizes internally).
        l_shared = self.infonce(z1, z2)
        l_x1y = self.infonce(z1, y_emb)
        l_x2y = self.infonce(z2, y_emb)

        z1_cond = torch.cat([z1, y_emb], dim=-1)
        z2_cond = torch.cat([z2, y_emb], dim=-1)
        l_cond = self.infonce(z1_cond, z2_cond)

        # CLUB penalties (optionally normalize first to match the user's flow).
        if self.normalize_inputs_for_club:
            z1c = F.normalize(z1, dim=-1)
            z2c = F.normalize(z2, dim=-1)
            z1cc = F.normalize(z1_cond, dim=-1)
            z2cc = F.normalize(z2_cond, dim=-1)
        else:
            z1c, z2c, z1cc, z2cc = z1, z2, z1_cond, z2_cond

        l_club = self.club(z1c, z2c)
        l_club_cond = self.club_cond(z1cc, z2cc)

        total = l_shared + l_x1y + l_x2y + l_cond - self.club_lambda * (
            l_club + l_club_cond
        )

        metrics = {
            f"{self.name}/loss": float(total.detach().item()),
            f"{self.name}/shared": float(l_shared.detach().item()),
            f"{self.name}/x1y": float(l_x1y.detach().item()),
            f"{self.name}/x2y": float(l_x2y.detach().item()),
            f"{self.name}/cond": float(l_cond.detach().item()),
            f"{self.name}/club": float(l_club.detach().item()),
            f"{self.name}/club_cond": float(l_club_cond.detach().item()),
        }
        return total * self.weight, metrics


class ReconstructionTask(BaseTask):
    def __init__(self, name, target_key, weight=1.0):
        """
        `preds[name]` must contain the reconstruction output.
        """
        super().__init__(name, weight)
        self.target_key = target_key

    def compute_loss(self, preds, embs, batch):
        x_hat = preds[self.name]
        x = batch[self.target_key]

        loss = F.mse_loss(x_hat, x)

        metrics = {f"{self.name}/loss": loss.item()}
        return loss * self.weight, metrics


class DiceTask(BaseTask):
    def __init__(self, name, target_key, weight=1.0, bce_weight=None, eps=1e-6):
        """
        If bce_weight is provided, uses Dice+BCE combo.
        Otherwise pure Dice.
        """
        super().__init__(name, weight)
        self.target_key = target_key
        self.bce_weight = bce_weight
        self.eps = eps

    def compute_loss(self, preds, embs, batch):
        logits = preds[self.name]            # [B, C, ...]
        targets = batch[self.target_key]     # [B, C, ...]

        if self.bce_weight is None:
            loss = dice_loss(logits, targets, eps=self.eps)
        else:
            loss = dice_bce_loss(
                logits,
                targets,
                bce_weight=self.bce_weight,
                eps=self.eps
            )

        metrics = {f"{self.name}/dice_loss": loss.item()}
        return loss * self.weight, metrics


class MultiClassDiceTask(BaseTask):
    def __init__(self, name, target_key, weight=1.0):
        super().__init__(name, weight)
        self.target_key = target_key

    def compute_loss(self, preds, embs, batch):
        logits = preds[self.name]
        targets = batch[self.target_key]

        loss = multiclass_dice_loss(logits, targets)

        metrics = {f"{self.name}/dice_loss": loss.item()}
        return loss * self.weight, metrics
