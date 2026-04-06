from __future__ import annotations

from typing import Any, Dict

import torch
from torch import nn


class MultimodalModel(nn.Module):
    """Encode each modality, fuse, then apply a task head."""

    def __init__(
        self,
        encoders: Dict[str, nn.Module],
        fusion: nn.Module,
        head: nn.Module,
    ) -> None:
        """Initialize a multimodal model.
        
        Args:
            encoders: Encoder modules keyed by modality name. May be
                :class:`~multimodal.blocks.base.Encoder` subclasses or any
                ``nn.Module`` with a compatible ``forward``.
            fusion: Fusion module, e.g. :class:`~multimodal.fusion.base.Fusion`,
                or any ``nn.Module`` taking a dict of tensors and returning one tensor.
            head: Task head, e.g. :class:`~multimodal.heads.base.Head`, or any
                ``nn.Module`` mapping fused features to outputs.
        """
        super().__init__()
        self.encoders = nn.ModuleDict(encoders)
        self.fusion = fusion
        self.head = head

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        encoded: Dict[str, torch.Tensor] = {}
        for name, encoder in self.encoders.items():
            if name not in batch:
                raise KeyError(f"batch missing modality {name!r} required by encoders")
            encoded[name] = encoder(batch[name])
        fused = self.fusion(encoded)
        return self.head(fused)
