from __future__ import annotations

from typing import Any

import torch
from torch import nn


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

