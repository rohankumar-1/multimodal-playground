"""Shared Fashion-MNIST helpers for ``examples/*.py`` (tiny, no extra deps)."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import FashionMNIST

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


def data_root() -> Path:
    base = os.environ.get("MULTIMODAL_DATA", tempfile.gettempdir())
    return Path(base) / "multimodal_examples"


class FashionMNISTWithText(Dataset):
    """Fashion-MNIST + class-name text (char token ids)."""

    def __init__(
        self,
        root: Path,
        *,
        train: bool,
        tokenizer: CharTokenizer,
        download: bool = True,
        image_key: str = "image",
        text_key: str = "text",
    ) -> None:
        self._tokenizer = tokenizer
        self._image_key = image_key
        self._text_key = text_key
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
            self._image_key: image,
            self._text_key: text_ids,
            "target": torch.tensor(target, dtype=torch.long),
        }


class FashionMNISTImageTextMultiview(Dataset):
    """Fashion-MNIST with **image + text**, each with an augmented view (FactorCL / multiview).

    Batch keys: ``image``, ``image_aug``, ``text``, ``text_aug``, plus ``target``.
    Text augmentation is random token masking (positions set to padding id 0).
    """

    def __init__(
        self,
        root: Path,
        *,
        train: bool,
        tokenizer: CharTokenizer,
        download: bool = True,
        text_mask_prob: float = 0.15,
    ) -> None:
        self._tokenizer = tokenizer
        self._text_mask_prob = text_mask_prob
        self._raw = FashionMNIST(
            root=str(root),
            train=train,
            download=download,
            transform=None,
        )
        self._to_tensor = transforms.ToTensor()
        self._aug = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(degrees=8, translate=(0.06, 0.06)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self._raw)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        pil, target = self._raw[idx]
        name = CLASS_NAMES[int(target)]
        text = self._tokenizer.encode(name)
        text_aug = text.clone()
        if self._text_mask_prob > 0:
            mask = torch.rand(text_aug.numel()) < self._text_mask_prob
            text_aug[mask] = 0
        return {
            "image": self._to_tensor(pil),
            "image_aug": self._aug(pil),
            "text": text,
            "text_aug": text_aug,
            "target": torch.tensor(int(target), dtype=torch.long),
        }


class FashionMNISTTwoView(Dataset):
    """Same example index, two tensor views: clean ``image`` + augmented ``image_aug``."""

    def __init__(
        self,
        root: Path,
        *,
        train: bool,
        download: bool = True,
    ) -> None:
        self._raw = FashionMNIST(
            root=str(root),
            train=train,
            download=download,
            transform=None,
        )
        self._to_tensor = transforms.ToTensor()
        self._aug = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(degrees=8, translate=(0.06, 0.06)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self._raw)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]: # ty:ignore[invalid-method-override]
        pil, target = self._raw[idx]
        return {
            "image": self._to_tensor(pil),
            "image_aug": self._aug(pil),
            "target": torch.tensor(int(target), dtype=torch.long),
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
