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

import os
import tempfile
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import FashionMNIST

from multimodal.fusion import ConcatFusion
from multimodal.model import MultimodalModel
from multimodal.tasks import ClassificationTask
from multimodal.train import Trainer, TrainerConfig

# Official Fashion-MNIST labels (10 classes).
CLASS_NAMES: tuple[str, ...] = (
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
)


class CharTokenizer:
    """Tiny fixed vocabulary over lowercase letters, digits, and a few symbols."""

    def __init__(self, max_len: int = 32) -> None:
        chars = "abcdefghijklmnopqrstuvwxyz0123456789 -/"
        self._pad = 0
        self.ch2i: dict[str, int] = {c: i + 1 for i, c in enumerate(chars)}
        self.max_len = max_len

    def encode(self, text: str) -> torch.Tensor:
        text = text.lower()[: self.max_len]
        ids = [self.ch2i.get(c, self._pad) for c in text]
        if len(ids) < self.max_len:
            ids.extend([self._pad] * (self.max_len - len(ids)))
        return torch.tensor(ids, dtype=torch.long)


class FashionMNISTWithText(Dataset):
    """Wraps Fashion-MNIST; adds ``text_ids`` from :data:`CLASS_NAMES` for each label."""

    def __init__(
        self,
        root: Path,
        *,
        train: bool,
        tokenizer: CharTokenizer,
        download: bool = True,
    ) -> None:
        self._tokenizer = tokenizer
        self._ds = FashionMNIST(
            root=str(root),
            train=train,
            download=download,
            transform=transforms.ToTensor(),
        )

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        image, target = self._ds[idx]
        name = CLASS_NAMES[int(target)]
        text_ids = self._tokenizer.encode(name)
        return {
            "vision": image,
            "text": text_ids,
            "target": torch.tensor(target, dtype=torch.long),
        }


class SmallImageEncoder(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TextEncoder(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.fc = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L] token ids
        h = self.emb(x).mean(dim=1)
        return self.fc(h)


class ClsHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"cls": self.fc(x)}


def data_root() -> Path:
    base = os.environ.get("MULTIMODAL_DATA", tempfile.gettempdir())
    return Path(base) / "multimodal_examples"


def main() -> None:
    torch.manual_seed(0)
    embed_dim = 64
    tokenizer = CharTokenizer(max_len=32)
    vocab_size = 1 + len(tokenizer.ch2i)

    root = data_root()
    root.mkdir(parents=True, exist_ok=True)

    train_ds = FashionMNISTWithText(root, train=True, tokenizer=tokenizer, download=True)
    val_ds = FashionMNISTWithText(root, train=False, tokenizer=tokenizer, download=True)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

    encoders = {
        "vision": SmallImageEncoder(embed_dim),
        "text": TextEncoder(vocab_size, hidden_dim=64, out_dim=embed_dim),
    }
    fusion = ConcatFusion(dim=-1)
    head = ClsHead(in_dim=embed_dim * 2, num_classes=10)
    model = MultimodalModel(
        encoders,
        fusion,
        head,
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
