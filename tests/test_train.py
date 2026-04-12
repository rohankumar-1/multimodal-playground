"""Tests for :class:`multimodal.train.Trainer` and training config."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
import torch
from torch import nn

from multimodal.fusion import ConcatFusion
from multimodal.heads import MultiTaskLinearHead
from multimodal.model import MultimodalModel
from multimodal.tasks import BaseTask, ClassificationTask
from multimodal.train import Trainer, TrainerConfig


class _ModelForTrainer(nn.Module):
    """``Trainer`` expects ``preds, embs = model(batch)``; wrap :class:`MultimodalModel`."""

    def __init__(self, inner: MultimodalModel) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, batch: dict) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        preds, enc = self.inner.forward_with_encoded(batch)
        assert isinstance(preds, dict) and isinstance(enc, dict)
        return cast(dict[str, torch.Tensor], preds), cast(dict[str, torch.Tensor], enc)


def _cpu_config(max_epochs: int = 1) -> TrainerConfig:
    return TrainerConfig(
        max_epochs=max_epochs,
        grad_accum_steps=1,
        mixed_precision=False,
        device="cpu",
    )


def test_trainer_single_epoch_classification() -> None:
    inner = MultimodalModel(
        {"v": nn.Linear(2, 4), "t": nn.Linear(2, 4)},
        ConcatFusion(dim=-1),
        MultiTaskLinearHead(8, {"cls": 3}),
        fusion_modality_order=["v", "t"],
    )
    model = _ModelForTrainer(inner)
    tasks: list[BaseTask] = [ClassificationTask("cls", "labels")]
    opt = torch.optim.SGD(model.parameters(), lr=0.5)
    trainer = Trainer(model, tasks, opt, _cpu_config(max_epochs=1))

    batch = {
        "v": torch.randn(8, 2),
        "t": torch.randn(8, 2),
        "labels": torch.tensor([0, 1, 2, 0, 1, 2, 0, 1]),
    }
    loader = [batch]

    w0 = next(model.parameters()).detach().clone()
    trainer.train(loader)
    w1 = next(model.parameters()).detach().clone()
    assert not torch.allclose(w0, w1)


def test_trainer_with_val_loader() -> None:
    inner = MultimodalModel(
        {"x": nn.Linear(3, 4)},
        ConcatFusion(dim=-1),
        MultiTaskLinearHead(4, {"cls": 2}),
        fusion_modality_order=["x"],
    )
    model = _ModelForTrainer(inner)
    tasks: list[BaseTask] = [ClassificationTask("cls", "labels")]
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    trainer = Trainer(model, tasks, opt, _cpu_config(max_epochs=1))

    train_loader = [
        {"x": torch.randn(4, 3), "labels": torch.tensor([0, 1, 0, 1])},
    ]
    val_loader = [
        {"x": torch.randn(4, 3), "labels": torch.tensor([1, 0, 1, 0])},
    ]
    trainer.train(train_loader, val_loader=val_loader)


def test_trainer_checkpoint_requires_val_loader() -> None:
    inner = MultimodalModel(
        {"x": nn.Linear(1, 2)},
        ConcatFusion(dim=-1),
        MultiTaskLinearHead(2, {"cls": 2}),
        fusion_modality_order=["x"],
    )
    model = _ModelForTrainer(inner)
    tasks: list[BaseTask] = [ClassificationTask("cls", "labels")]
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    cfg = TrainerConfig(
        max_epochs=1,
        grad_accum_steps=1,
        mixed_precision=False,
        device="cpu",
        checkpoint_path="/tmp/should_not_matter.pt",
    )
    trainer = Trainer(model, tasks, opt, cfg)
    with pytest.raises(ValueError, match="val_loader"):
        trainer.train([{"x": torch.randn(2, 1), "labels": torch.tensor([0, 1])}])


def test_trainer_saves_checkpoint_on_val(tmp_path: Path) -> None:
    ckpt = tmp_path / "best.pt"
    inner = MultimodalModel(
        {"x": nn.Linear(3, 4)},
        ConcatFusion(dim=-1),
        MultiTaskLinearHead(4, {"cls": 2}),
        fusion_modality_order=["x"],
    )
    model = _ModelForTrainer(inner)
    tasks: list[BaseTask] = [ClassificationTask("cls", "labels")]
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    cfg = TrainerConfig(
        max_epochs=1,
        grad_accum_steps=1,
        mixed_precision=False,
        device="cpu",
        checkpoint_path=str(ckpt),
    )
    trainer = Trainer(model, tasks, opt, cfg)

    train_loader = [{"x": torch.randn(4, 3), "labels": torch.tensor([0, 1, 0, 1])}]
    val_loader = [{"x": torch.randn(4, 3), "labels": torch.tensor([1, 0, 1, 0])}]
    trainer.train(train_loader, val_loader=val_loader)

    assert ckpt.is_file()
    try:
        data = torch.load(ckpt, map_location="cpu", weights_only=False)
    except TypeError:
        data = torch.load(ckpt, map_location="cpu")
    assert "model_state_dict" in data
    assert "best_val_loss" in data
    assert "cls/loss" in data["val_metrics"]
