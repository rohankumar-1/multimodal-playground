#!/usr/bin/env python3
"""Minimal **real** multimodal example: Fashion-MNIST images + text (class names).

Uses torchvision's Fashion-MNIST benchmark (grayscale clothing images) and a second
modality built from the official class-name strings (character-tokenized).

Data directory: ``$MULTIMODAL_DATA/multimodal_examples`` (default: system temp). Images
are downloaded once into that folder (torchvision's normal on-disk layout).

Run::

    PYTHONPATH=src python examples/fashion_mnist_text_classification.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

_EXAMPLES = Path(__file__).resolve().parent
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

from fashion_mnist_shared import (
    CharTokenizer,
    FashionMNISTWithText,
    SmallImageEncoder,
    TextEncoder,
    data_root,
)

from multimodal.fusion import ConcatFusion
from multimodal.model import MultimodalModel
from multimodal.tasks import ClassificationTask
from multimodal.train import Trainer, TrainerConfig


class ClsHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"cls": self.fc(x)}


def main() -> None:
    torch.manual_seed(0)
    embed_dim = 64
    tokenizer = CharTokenizer(max_len=32)
    vocab_size = 1 + len(tokenizer.ch2i)

    root = data_root()
    root.mkdir(parents=True, exist_ok=True)

    train_ds = FashionMNISTWithText(
        root, train=True, tokenizer=tokenizer, download=True,
        image_key="vision", text_key="text",
    )
    val_ds = FashionMNISTWithText(
        root, train=False, tokenizer=tokenizer, download=True,
        image_key="vision", text_key="text",
    )

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

    model = MultimodalModel(
        encoders={
            "vision": SmallImageEncoder(embed_dim),
            "text": TextEncoder(vocab_size, hidden_dim=64, out_dim=embed_dim),
        },
        fusion=ConcatFusion(dim=-1),
        head=ClsHead(in_dim=embed_dim * 2, num_classes=10),
        fusion_modality_order=["vision", "text"],
    )

    tasks = [ClassificationTask("cls", "target")]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    config = TrainerConfig(
        max_epochs=3,
        device=device,
        mixed_precision=use_amp,
        metric_precision=4,
        progress_bar=True,
    )
    trainer = Trainer(model, tasks, optimizer, config)
    print(f"Data root: {root} | device={device}")
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()
