"""Smoke tests for training utilities."""

import torch
from torch import nn

from multimodal.fusion import ConcatFusion
from multimodal.model import MultimodalModel
from multimodal.train import evaluate, move_batch_to_device, train_epoch, train_step


def test_move_batch_to_device() -> None:
    batch = {"x": torch.zeros(2, 3), "meta": "keep"}
    out = move_batch_to_device(batch, torch.device("cpu"))
    assert out["meta"] == "keep"
    assert out["x"].device.type == "cpu"


def test_train_step_supervised() -> None:
    model = MultimodalModel(
        {"v": nn.Linear(2, 4), "t": nn.Linear(2, 4)},
        ConcatFusion(dim=-1),
        nn.Linear(8, 3),
        fusion_modality_order=("v", "t"),
    )
    opt = torch.optim.SGD(model.parameters(), lr=0.1)

    def loss_fn(model, batch, predictions, encoded):
        assert encoded["v"].shape == (4, 4)
        return torch.nn.functional.cross_entropy(predictions, batch["labels"]), {}

    batch = {
        "v": torch.randn(4, 2),
        "t": torch.randn(4, 2),
        "labels": torch.tensor([0, 1, 2, 1]),
    }
    m0 = next(model.parameters()).detach().clone()
    metrics = train_step(model, batch, opt, loss_fn)
    assert "loss" in metrics
    assert not torch.equal(next(model.parameters()), m0)


def test_train_epoch_and_evaluate() -> None:
    model = MultimodalModel(
        {"v": nn.Linear(1, 2)},
        nn.Identity(),
        nn.Linear(2, 1),
    )
    opt = torch.optim.SGD(model.parameters(), lr=0.01)

    def loss_fn(model, batch, predictions, encoded):
        return torch.nn.functional.mse_loss(predictions.squeeze(-1), batch["y"]), {"mse": 0.0}

    loader = [
        {"v": torch.randn(3, 1), "y": torch.randn(3)},
        {"v": torch.randn(3, 1), "y": torch.randn(3)},
    ]
    device = torch.device("cpu")
    m = train_epoch(model, loader, opt, loss_fn, device)
    assert "loss" in m
    ev = evaluate(model, loader, loss_fn, device)
    assert "loss" in ev
