from __future__ import annotations

from typing import Any

import torch
from torch import nn

from multimodal.heads.basic import NoOpHead


class MultimodalModel(nn.Module):
    """Encode modalities, fuse, and apply a task head.

    **Data flow**

    - **Batch** ``Dict[str, Any]``: one entry per modality name. Values are tensors
      consumed by the matching encoder.
    - **Forward** (``model(batch)``) returns ``(predictions, modality_embeddings)``:
      full encode → fuse → head, plus the encoder outputs as a dict.
    - **Predict** returns **only** predictions (same forward pass, embeddings discarded).

    List-based fusions (e.g. :class:`~multimodal.fusion.common_fusions.ConcatFusion`)
    use ``fusion_modality_order`` so modality tensors are stacked in a fixed order.
    Dict-based fusions leave ``fusion_modality_order`` unset.
    """

    def __init__(
        self,
        encoders: dict[str, nn.Module | nn.Linear],
        fusion: nn.Module,
        head: nn.Module | nn.Linear,
        fusion_modality_order: list[str] | None = None,
    ) -> None:
        """Initialize a multimodal model.

        Args:
            encoders: Encoder modules keyed by modality name.
            fusion: If ``fusion_modality_order`` is set, called as
                ``fusion([encoded[k] for k in fusion_modality_order])``. Otherwise
                ``fusion(encoded)`` with the full dict.
            head: Maps fused features to predictions.
            fusion_modality_order: Modality order for list-input fusions.
        """
        super().__init__()
        self.encoders = nn.ModuleDict(encoders)
        self.fusion = fusion
        self.head = head
        self.fusion_modality_order = (
            tuple(fusion_modality_order) if fusion_modality_order is not None else None
        )

    def forward(
        self, batch: dict[str, Any]
    ) -> tuple[torch.Tensor | dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Encode, fuse, apply head. Returns ``(predictions, per_modality_embeddings)``."""
        encoded: dict[str, torch.Tensor] = {}
        for name, encoder in self.encoders.items():
            if name not in batch:
                raise KeyError(f"batch missing modality {name!r} required by encoders")
            encoded[name] = encoder(batch[name])
        fused = self._fuse(encoded)
        predictions = self.head(fused)
        return predictions, encoded

    def _fuse(self, encoded: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.fusion_modality_order is not None:
            missing = [k for k in self.fusion_modality_order if k not in encoded]
            if missing:
                raise KeyError(f"encoded missing modalities required for fusion: {missing}")
            return self.fusion([encoded[k] for k in self.fusion_modality_order])
        return self.fusion(encoded)

    def predict(self, batch: dict[str, Any]) -> torch.Tensor | dict[str, torch.Tensor]:
        """Return predictions only (inference-style API; runs one full forward pass)."""
        predictions, _ = self.forward(batch)
        return predictions

    def encode(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Encode the batch. Returns ``per_modality_embeddings``."""
        encoded: dict[str, torch.Tensor] = {}
        for name, encoder in self.encoders.items():
            if name not in batch:
                raise KeyError(f"batch missing modality {name!r} required by encoders")
            encoded[name] = encoder(batch[name])
        return encoded


class UnimodalModel(nn.Module):
    """Single-modality encoder followed by a head on the embedding.

    **Batch** maps ``input_key`` to a tensor. **Forward** returns
    ``(predictions, {input_key: embedding})`` for compatibility with :class:`Trainer`.
    """

    def __init__(
        self,
        encoder: nn.Module,
        head: nn.Module | nn.Linear,
        *,
        input_key: str = "x",
    ) -> None:
        super().__init__()
        self.input_key = input_key
        self.encoders = nn.ModuleDict({input_key: encoder})
        self.head = head

    def forward(
        self, batch: dict[str, Any]
    ) -> tuple[torch.Tensor | dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if self.input_key not in batch:
            raise KeyError(f"batch missing key {self.input_key!r}")
        h = self.encoders[self.input_key](batch[self.input_key])
        predictions = self.head(h)
        return predictions, {self.input_key: h}

    def predict(self, batch: dict[str, Any]) -> torch.Tensor | dict[str, torch.Tensor]:
        predictions, _ = self.forward(batch)
        return predictions

    def encode(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        if self.input_key not in batch:
            raise KeyError(f"batch missing key {self.input_key!r}")
        return {self.input_key: self.encoders[self.input_key](batch[self.input_key])}


class UnifiedContrastiveModel(nn.Module):
    """Two modality-specific encoders, shared latent via optional projections.

    For each modality you provide **two batch keys** (original and augmentation) so the
    model can expose four embeddings: unimodal A, unimodal A aug, unimodal B, unimodal B aug.
    Pair these with multiple :class:`~multimodal.tasks.ContrastiveTask` instances, e.g.
    ``(keys_m1[0], keys_m1[1])``, ``(keys_m2[0], keys_m2[1])``, ``(keys_m1[0], keys_m2[0])``.

    ``head`` defaults to :class:`~multimodal.heads.basic.NoOpHead` when training is driven
    only by contrastive losses on ``embs``. A fused vector (concat of projected **first**
    views ``z_m1``, ``z_m2``) is passed to ``head`` for API compatibility.
    """

    def __init__(
        self,
        encoder_m1: nn.Module,
        encoder_m2: nn.Module,
        *,
        proj_m1: nn.Module | None = None,
        proj_m2: nn.Module | None = None,
        head: nn.Module | None = None,
        keys_m1: tuple[str, str] = ("m1", "m1_aug"),
        keys_m2: tuple[str, str] = ("m2", "m2_aug"),
    ) -> None:
        super().__init__()
        self.encoders = nn.ModuleDict({"m1": encoder_m1, "m2": encoder_m2})
        self.proj_m1 = proj_m1 or nn.Identity()
        self.proj_m2 = proj_m2 or nn.Identity()
        self.head = head if head is not None else NoOpHead()
        self.k1a, self.k1b = keys_m1
        self.k2a, self.k2b = keys_m2

    def forward(
        self, batch: dict[str, Any]
    ) -> tuple[torch.Tensor | dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        for k in (self.k1a, self.k1b, self.k2a, self.k2b):
            if k not in batch:
                raise KeyError(f"batch missing key {k!r}")

        z1 = self.proj_m1(self.encoders["m1"](batch[self.k1a]))
        z1_aug = self.proj_m1(self.encoders["m1"](batch[self.k1b]))
        z2 = self.proj_m2(self.encoders["m2"](batch[self.k2a]))
        z2_aug = self.proj_m2(self.encoders["m2"](batch[self.k2b]))

        embs: dict[str, torch.Tensor] = {
            self.k1a: z1,
            self.k1b: z1_aug,
            self.k2a: z2,
            self.k2b: z2_aug,
        }
        fused = torch.cat([z1, z2], dim=-1)
        predictions = self.head(fused)
        return predictions, embs

    def predict(self, batch: dict[str, Any]) -> torch.Tensor | dict[str, torch.Tensor]:
        predictions, _ = self.forward(batch)
        return predictions

    def encode(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        _, embs = self.forward(batch)
        return embs


class MultiviewContrastiveModel(nn.Module):
    """One encoder (and optional projection) applied to **each view** in the batch.

    Batch keys are given by ``view_keys`` (e.g. ``("x", "x_aug")`` or more views).
    Embeddings are ``proj(encoder(batch[k]))`` for each ``k``. Fused features for
    ``head`` are the concatenation of embeddings in ``view_keys`` order.

    Use with one or more :class:`~multimodal.tasks.ContrastiveTask` on pairs of
    ``embs`` keys (e.g. ``("x", "x_aug")``).
    """

    def __init__(
        self,
        encoder: nn.Module,
        view_keys: tuple[str, ...],
        *,
        proj: nn.Module | None = None,
        head: nn.Module | None = None,
        encoder_key: str = "encoder",
    ) -> None:
        super().__init__()
        if len(view_keys) < 2:
            raise ValueError("MultiviewContrastiveModel requires at least two view_keys")
        self.view_keys = tuple(view_keys)
        self.encoders = nn.ModuleDict({encoder_key: encoder})
        self._encoder_key = encoder_key
        self.proj = proj or nn.Identity()
        self.head = head if head is not None else NoOpHead()

    def forward(
        self, batch: dict[str, Any]
    ) -> tuple[torch.Tensor | dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        enc = self.encoders[self._encoder_key]
        embs: dict[str, torch.Tensor] = {}
        for k in self.view_keys:
            if k not in batch:
                raise KeyError(f"batch missing view key {k!r}")
            embs[k] = self.proj(enc(batch[k]))
        fused = torch.cat([embs[k] for k in self.view_keys], dim=-1)
        predictions = self.head(fused)
        return predictions, embs

    def predict(self, batch: dict[str, Any]) -> torch.Tensor | dict[str, torch.Tensor]:
        predictions, _ = self.forward(batch)
        return predictions

    def encode(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        _, embs = self.forward(batch)
        return embs

