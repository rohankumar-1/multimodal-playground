#!/usr/bin/env python3
"""Train a small multimodal model: ConcatFusion + M3H multitask head on synthetic data.

Run from the repo root::

    PYTHONPATH=src python examples/m3h_multitask.py

Or after ``pip install -e .``::

    python examples/m3h_multitask.py
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from multimodal.fusion import ConcatFusion
from multimodal.heads import M3HHead
from multimodal.model import MultimodalModel
from multimodal.tasks import ClassificationTask
from multimodal.train import Trainer, TrainerConfig


class SyntheticMultiModalDataset(Dataset):
    """Tiny fixed fake dataset: random Gaussian inputs + random integer labels."""

    def __init__(
        self,
        n_samples: int,
        d_vision: int,
        d_text: int,
        *,
        seed: int,
    ) -> None:
        g = torch.Generator().manual_seed(seed)
        self.vision = torch.randn(n_samples, d_vision, generator=g)
        self.text = torch.randn(n_samples, d_text, generator=g)
        self.sentiment = torch.randint(0, 3, (n_samples,), generator=g)
        self.topic = torch.randint(0, 10, (n_samples,), generator=g)

    def __len__(self) -> int:
        return self.vision.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]: # ty:ignore[invalid-method-override]
        return {
            "vision": self.vision[idx],
            "text": self.text[idx],
            "sentiment_target": self.sentiment[idx],
            "topic_target": self.topic[idx],
        }


def build_model(
    *,
    d_vision: int,
    d_text: int,
    embed_dim: int,
    attn_dim: int,
) -> MultimodalModel:
    in_dim = embed_dim * 2
    encoders = {
        "vision": nn.Sequential(nn.Linear(d_vision, embed_dim), nn.ReLU()),
        "text": nn.Sequential(nn.Linear(d_text, embed_dim), nn.ReLU()),
    }
    fusion = ConcatFusion(dim=-1)
    head = M3HHead(
        in_dim=in_dim,
        attn_dim=attn_dim,
        out_dims={"sentiment": 3, "topic": 10},
        alpha=1.0,
    )
    return MultimodalModel(
        encoders,  # ty:ignore[invalid-argument-type]
        fusion,
        head,
        fusion_modality_order=["vision", "text"],
    )


def main() -> None:
    torch.manual_seed(0)

    d_vision, d_text = 16, 24
    embed_dim = 32
    attn_dim = 64
    batch_size = 16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"

    train_ds = SyntheticMultiModalDataset(5120, d_vision, d_text, seed=1)
    val_ds = SyntheticMultiModalDataset(512, d_vision, d_text, seed=2)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = build_model(
        d_vision=d_vision,
        d_text=d_text,
        embed_dim=embed_dim,
        attn_dim=attn_dim,
    )

    tasks = [
        ClassificationTask("sentiment", "sentiment_target"),
        ClassificationTask("topic", "topic_target"),
    ]

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    config = TrainerConfig(
        max_epochs=5,
        device=device,
        mixed_precision=use_amp,
        grad_accum_steps=1,
        clip_grad_norm=1.0,
    )

    trainer = Trainer(model, tasks, optimizer, config)
    print(f"Device: {device} (AMP: {use_amp})")
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()
