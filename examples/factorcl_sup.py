"""Example: FactorCL-style supervised objective with this repo's Trainer.

This demo uses:
- `ContrastiveModel` to produce two embeddings (x1, x2).
- `FactorCLSupervisedTask` to compute the objective described in the prompt.
"""

from __future__ import annotations

import torch
from torch import nn

from multimodal.model import ContrastiveModel
from multimodal.tasks import BaseTask, FactorCLSupervisedTask
from multimodal.train import Trainer, TrainerConfig, iter_training_parameters


def main() -> None:
    # Two "modalities" (or views) -> same embedding dim.
    embed_dim = 32
    num_classes = 10

    model = ContrastiveModel(
        encoders={
            "enc1": nn.Sequential(nn.Linear(8, 64), nn.ReLU(), nn.Linear(64, embed_dim)),
            "enc2": nn.Sequential(nn.Linear(12, 64), nn.ReLU(), nn.Linear(64, embed_dim)),
        },
        route={"x1": "enc1", "x2": "enc2"},
        # Optional: could add `projs={"enc1": nn.Linear(embed_dim, embed_dim), "enc2": ...}`
    )

    tasks: list[BaseTask] = [
        FactorCLSupervisedTask(
            "factorcl",
            x1_key="x1",
            x2_key="x2",
            label_key="y",
            num_classes=num_classes,
            embed_dim=embed_dim,
            temperature=0.07,
            club_lambda=1.0,
        )
    ]

    opt = torch.optim.Adam(iter_training_parameters(model, tasks), lr=1e-3)
    cfg = TrainerConfig(
        max_epochs=2,
        grad_accum_steps=1,
        mixed_precision=False,
        device="cpu",
        progress_bar=True,
        log_every=1,
    )

    trainer = Trainer(model, tasks, opt, cfg)

    batch = {
        "x1": torch.randn(16, 8),
        "x2": torch.randn(16, 12),
        "y": torch.randint(0, num_classes, (16,)),
    }
    trainer.train([batch], val_loader=[batch])


if __name__ == "__main__":
    main()

