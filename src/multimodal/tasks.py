from __future__ import annotations

from abc import abstractmethod

import torch
import torch.nn.functional as F

from multimodal.utils import clip_loss, dice_bce_loss, dice_loss, multiclass_dice_loss


class BaseTask:
    def __init__(self, name: str, weight: float = 1.0):
        self.name = name
        self.weight = weight

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
    def __init__(self, name, mod1, mod2, temperature=0.07, weight=1.0):
        super().__init__(name, weight)
        self.m1 = mod1
        self.m2 = mod2
        self.temperature = temperature

    def compute_loss(self, preds, embs, batch):
        z1 = embs[self.m1]
        z2 = embs[self.m2]

        loss = clip_loss(z1, z2, self.temperature)
        metrics = {f"{self.name}/loss": loss.item()}
        return loss * self.weight, metrics


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

