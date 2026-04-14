#!/usr/bin/env python3
"""CLIP-style contrastive training: **encoder embeddings only** (no classification head).

The fusion module and task head still run in :class:`~multimodal.model.MultimodalModel`,
but loss comes solely from :class:`~multimodal.tasks.ContrastiveTask`, which uses
``embs[mod1]`` and ``embs[mod2]`` (see ``clip_loss`` in ``multimodal.utils``). The head
returns an empty dict so no supervised logits are used.

Uses the same Fashion-MNIST + class-name text setup as
``fashion_mnist_text_classification.py`` (minimal real image data; text from labels).

Run::

    PYTHONPATH=src python examples/clip_style_contrastive.py
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
from multimodal.heads import NoOpHead
from multimodal.model import MultimodalModel
from multimodal.tasks import ContrastiveTask
from multimodal.train import Trainer, TrainerConfig

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

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]: # ty:ignore[invalid-method-override]
        image, target = self._ds[idx]
        name = CLASS_NAMES[int(target)]
        text_ids = self._tokenizer.encode(name)
        return {
            "image": image,
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
        h = self.emb(x).mean(dim=1)
        return self.fc(h)


def data_root() -> Path:
    base = os.environ.get("MULTIMODAL_DATA", tempfile.gettempdir())
    return Path(base) / "multimodal_examples"


def main() -> None:
    torch.manual_seed(0)
    proj_dim = 128
    tokenizer = CharTokenizer(max_len=32)
    vocab_size = 1 + len(tokenizer.ch2i)

    root = data_root()
    root.mkdir(parents=True, exist_ok=True)

    train_ds = FashionMNISTWithText(root, train=True, tokenizer=tokenizer, download=True)
    val_ds = FashionMNISTWithText(root, train=False, tokenizer=tokenizer, download=True)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

    encoders = {
        "image": SmallImageEncoder(proj_dim),
        "text": TextEncoder(vocab_size, hidden_dim=64, out_dim=proj_dim),
    }
    fusion = ConcatFusion(dim=-1)
    head = NoOpHead()
    model = MultimodalModel(
        encoders,
        fusion,
        head,
        fusion_modality_order=["image", "text"],
    )

    tasks = [ContrastiveTask("clip", "image", "text", temperature=0.07)]
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
    print(f"Data root: {root} | device={device} | contrastive on encoder embeddings only")
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()
