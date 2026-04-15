"""Tests for :class:`multimodal.train.Trainer` and training config."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from multimodal.fusion import ConcatFusion
from multimodal.heads import MultiTaskLinearHead
from multimodal.model import MultimodalModel
from multimodal.tasks import BaseTask, ClassificationTask, ContrastiveTask
from multimodal.train import Trainer, TrainerConfig, iter_training_parameters


def _cpu_config(max_epochs: int = 1) -> TrainerConfig:
    return TrainerConfig(
        max_epochs=max_epochs,
        grad_accum_steps=1,
        mixed_precision=False,
        device="cpu",
        progress_bar=False,
    )


def test_trainer_single_epoch_classification() -> None:
    model = MultimodalModel(
        {"v": nn.Linear(2, 4), "t": nn.Linear(2, 4)},
        ConcatFusion(dim=-1),
        MultiTaskLinearHead(8, {"cls": 3}),
        fusion_modality_order=["v", "t"],
    )
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
    model = MultimodalModel(
        {"x": nn.Linear(3, 4)},
        ConcatFusion(dim=-1),
        MultiTaskLinearHead(4, {"cls": 2}),
        fusion_modality_order=["x"],
    )
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
    model = MultimodalModel(
        {"x": nn.Linear(1, 2)},
        ConcatFusion(dim=-1),
        MultiTaskLinearHead(2, {"cls": 2}),
        fusion_modality_order=["x"],
    )
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
    model = MultimodalModel(
        {"x": nn.Linear(3, 4)},
        ConcatFusion(dim=-1),
        MultiTaskLinearHead(4, {"cls": 2}),
        fusion_modality_order=["x"],
    )
    tasks: list[BaseTask] = [ClassificationTask("cls", "labels")]
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    cfg = TrainerConfig(
        max_epochs=1,
        grad_accum_steps=1,
        mixed_precision=False,
        device="cpu",
        checkpoint_path=str(ckpt),
        progress_bar=False,
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


def test_trainer_freeze_encoder_ids() -> None:
    model = MultimodalModel(
        {"v": nn.Linear(2, 4), "t": nn.Linear(2, 4)},
        ConcatFusion(dim=-1),
        MultiTaskLinearHead(8, {"cls": 3}),
        fusion_modality_order=["v", "t"],
    )
    cfg = TrainerConfig(
        max_epochs=1,
        grad_accum_steps=1,
        mixed_precision=False,
        device="cpu",
        freeze_encoder_ids=("v",),
        progress_bar=False,
    )
    Trainer(
        model,
        [ClassificationTask("cls", "labels")],
        torch.optim.SGD(model.parameters(), lr=0.1),
        cfg,
    )
    assert all(not p.requires_grad for p in model.encoders["v"].parameters())
    assert all(p.requires_grad for p in model.encoders["t"].parameters())


def test_trainer_freeze_all_encoders() -> None:
    model = MultimodalModel(
        {"a": nn.Linear(1, 2), "b": nn.Linear(1, 2)},
        ConcatFusion(dim=-1),
        MultiTaskLinearHead(4, {"cls": 2}),
        fusion_modality_order=["a", "b"],
    )
    cfg = TrainerConfig(
        max_epochs=1,
        grad_accum_steps=1,
        mixed_precision=False,
        device="cpu",
        freeze_all_encoders=True,
        progress_bar=False,
    )
    Trainer(
        model,
        [ClassificationTask("cls", "labels")],
        torch.optim.SGD(model.parameters(), lr=0.1),
        cfg,
    )
    assert all(not p.requires_grad for p in model.encoders["a"].parameters())
    assert all(not p.requires_grad for p in model.encoders["b"].parameters())


def test_trainer_freeze_unknown_modality_raises() -> None:
    model = MultimodalModel(
        {"v": nn.Linear(2, 2)},
        ConcatFusion(dim=-1),
        MultiTaskLinearHead(2, {"cls": 2}),
        fusion_modality_order=["v"],
    )
    cfg = TrainerConfig(
        max_epochs=1,
        grad_accum_steps=1,
        mixed_precision=False,
        device="cpu",
        freeze_encoder_ids=("not_a_modality",),
        progress_bar=False,
    )
    with pytest.raises(KeyError, match="unknown encoder id"):
        Trainer(
            model,
            [ClassificationTask("cls", "labels")],
            torch.optim.SGD(model.parameters(), lr=0.1),
            cfg,
        )


def test_trainer_format_metrics_precision() -> None:
    model = MultimodalModel(
        {"x": nn.Linear(1, 2)},
        ConcatFusion(dim=-1),
        MultiTaskLinearHead(2, {"cls": 2}),
        fusion_modality_order=["x"],
    )
    tasks: list[BaseTask] = [ClassificationTask("cls", "labels")]
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    cfg = TrainerConfig(
        max_epochs=1,
        grad_accum_steps=1,
        mixed_precision=False,
        device="cpu",
        metric_precision=2,
        progress_bar=False,
    )
    trainer = Trainer(model, tasks, opt, cfg)
    s = trainer._format_metrics({"a": 1.23456, "b": 9.0})
    assert "a=1.23" in s
    assert "b=9.00" in s


def test_iter_training_parameters_includes_contrastive_loss_module() -> None:
    from multimodal.losses import CriticInfoNCE, SeparableCritic

    model = nn.Linear(4, 4)
    loss_mod = CriticInfoNCE(SeparableCritic(4, 4, proj_dim=8), temperature=0.1)
    tasks = [ContrastiveTask("c", "a", "b", loss_fn=loss_mod)]
    params = list(iter_training_parameters(model, tasks))
    loss_ids = {id(p) for p in loss_mod.parameters()}
    assert loss_ids <= {id(p) for p in params}
    n_lin = sum(1 for _ in model.parameters())
    n_crit = sum(1 for _ in loss_mod.parameters())
    assert len(params) == n_lin + n_crit


def test_iter_training_parameters_dedupes_when_task_module_is_submodule_of_model() -> None:
    class Toy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.shared = nn.Linear(2, 2)
            self.head = nn.Linear(2, 1)

    class SharingTask(BaseTask):
        def __init__(self, mod: nn.Module) -> None:
            super().__init__("s")
            self._mod = mod

        def trainable_loss_modules(self) -> tuple[nn.Module, ...]:
            return (self._mod,)

        def compute_loss(self, preds, embs, batch):
            return torch.tensor(0.0), {}

    m = Toy()
    tasks = [SharingTask(m.shared)]
    params = list(iter_training_parameters(m, tasks))
    assert len(params) == len(list(m.parameters()))
