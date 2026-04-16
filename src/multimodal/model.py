from __future__ import annotations

from typing import Any

import torch
from torch import nn

from multimodal.heads.basic import NoOpHead
from multimodal.fusion.common_fusions import BaseFusion


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
        fusion: nn.Module | BaseFusion,
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


def _module_dict_projs(
    modalities: tuple[str, ...],
    projs: dict[str, nn.Module | None] | None,
) -> nn.ModuleDict:
    out: dict[str, nn.Module] = {}
    for m in modalities:
        p = None if projs is None else projs.get(m)
        out[m] = p if p is not None else nn.Identity()
    return nn.ModuleDict(out)


class ContrastiveModel(nn.Module):
    """Routed encoders for contrastive training (losses live in tasks, not here).

    **Encoders** are towers keyed by ids (e.g. ``"vision"``, ``"text"``). **Route** maps each
    **batch key** to an encoder id so several inputs can share one tower (e.g. ``image`` and
    ``image_aug`` → ``"vision"``).

    Forward returns ``embs`` keyed by **batch keys** in ``route``, each
    ``proj(encoder(batch[key]))``. By default ``preds`` is ``{}``. If ``fuse_keys`` is set,
    those embeddings are concatenated (in order) and passed through ``head`` (defaults to
    :class:`~multimodal.heads.basic.NoOpHead` when ``fuse_keys`` is set and ``head`` is
    omitted)—the same auxiliary path older multiview stacks used.

    **Routing presets**

    - :meth:`route_from_view_pairs` — build ``route`` from
      ``{encoder_id: (batch_key_a, batch_key_b), ...}``.
    - :meth:`fuse_keys_first_views` — default concat order (first view per encoder, sorted
      encoder ids).
    - :meth:`from_view_pairs` — one-call construction from ``view_pairs`` with optional fused
      head (see ``use_fused_path``).

    For :class:`~multimodal.train.Trainer` freezing, use ``TrainerConfig.freeze_encoder_ids``;
    values must match keys in ``self.encoders`` (tower ids), not batch keys.
    """

    fuse_keys: tuple[str, ...] | None
    fused_head: nn.Module | None

    def __init__(
        self,
        encoders: dict[str, nn.Module],
        route: dict[str, str],
        *,
        projs: dict[str, nn.Module | None] | None = None,
        head: nn.Module | None = None,
        fuse_keys: tuple[str, ...] | None = None,
    ) -> None:
        super().__init__()
        if not route:
            raise ValueError("ContrastiveModel requires a non-empty route")
        self.encoders = nn.ModuleDict(dict(sorted(encoders.items())))
        self.route = dict(sorted(route.items()))
        enc_ids = tuple(sorted(self.encoders.keys()))
        for batch_key, enc_id in self.route.items():
            if enc_id not in self.encoders:
                raise ValueError(
                    f"route[{batch_key!r}] = {enc_id!r} is not a key in encoders "
                    f"{sorted(self.encoders.keys())}"
                )
        self.proj = _module_dict_projs(enc_ids, projs)

        self.fuse_keys = tuple(fuse_keys) if fuse_keys is not None else None
        if self.fuse_keys is not None:
            self.fused_head = head if head is not None else NoOpHead()
        else:
            if head is not None:
                raise ValueError(
                    "ContrastiveModel: `head` was given but `fuse_keys` is None; "
                    "set `fuse_keys` to select which `embs` to concatenate for the head."
                )
            self.fused_head = None

    @staticmethod
    def route_from_view_pairs(view_pairs: dict[str, tuple[str, str]]) -> dict[str, str]:
        """Map each batch key in ``view_pairs`` values to the corresponding encoder id (dict key)."""
        out: dict[str, str] = {}
        for enc_id in sorted(view_pairs.keys()):
            k0, k1 = view_pairs[enc_id]
            out[k0] = enc_id
            out[k1] = enc_id
        return dict(sorted(out.items()))

    @staticmethod
    def fuse_keys_first_views(view_pairs: dict[str, tuple[str, str]]) -> tuple[str, ...]:
        """First batch key of each ``(k_a, k_b)`` pair, in sorted encoder-id order."""
        return tuple(view_pairs[enc_id][0] for enc_id in sorted(view_pairs.keys()))

    @classmethod
    def from_view_pairs(
        cls,
        encoders: dict[str, nn.Module],
        view_pairs: dict[str, tuple[str, str]],
        *,
        projs: dict[str, nn.Module | None] | None = None,
        head: nn.Module | None = None,
        fuse_keys: tuple[str, ...] | None = None,
        use_fused_path: bool = True,
    ) -> ContrastiveModel:
        """Build a model where each encoder serves two batch keys (e.g. x and x_aug).

        ``encoders`` keys must match ``view_pairs`` keys. With ``use_fused_path=True`` (default),
        sets ``fuse_keys`` to :meth:`fuse_keys_first_views` unless overridden, and ``head`` to
        :class:`~multimodal.heads.basic.NoOpHead` if omitted. With ``use_fused_path=False``,
        only the routed embedding path is used (``preds={}``); ``fuse_keys`` and ``head`` must
        not be set.

        Expects **at least two encoders** (typical multiview + cross-modal setups).
        """
        if len(encoders) < 2:
            raise ValueError(
                "from_view_pairs expects at least two encoders for typical multiview "
                "setups (build ContrastiveModel(encoders, route) directly for one tower)."
            )
        enc_keys = set(encoders.keys())
        vp_keys = set(view_pairs.keys())
        if enc_keys != vp_keys:
            raise ValueError(
                f"encoders keys {sorted(enc_keys)} must match view_pairs keys {sorted(vp_keys)}"
            )
        route = cls.route_from_view_pairs(view_pairs)
        if use_fused_path:
            fk = fuse_keys if fuse_keys is not None else cls.fuse_keys_first_views(view_pairs)
            return cls(encoders=encoders, route=route, projs=projs, head=head, fuse_keys=fk)
        if fuse_keys is not None or head is not None:
            raise ValueError(
                "use_fused_path=False is incompatible with fuse_keys/head; "
                "omit those or set use_fused_path=True."
            )
        return cls(encoders=encoders, route=route, projs=projs)

    def forward(
        self, batch: dict[str, Any]
    ) -> tuple[torch.Tensor | dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        embs: dict[str, torch.Tensor] = {}
        for batch_key in self.route:
            enc_id = self.route[batch_key]
            if batch_key not in batch:
                raise KeyError(
                    f"batch missing key {batch_key!r} required by route "
                    f"(encoder {enc_id!r})"
                )
            h = self.encoders[enc_id](batch[batch_key])
            embs[batch_key] = self.proj[enc_id](h)

        if self.fuse_keys is not None:
            assert self.fused_head is not None
            missing = [k for k in self.fuse_keys if k not in embs]
            if missing:
                raise KeyError(
                    f"fuse_keys {self.fuse_keys} not in embs; missing: {missing}. "
                    f"Available keys: {sorted(embs.keys())}"
                )
            fused = torch.cat([embs[k] for k in self.fuse_keys], dim=-1)
            predictions = self.fused_head(fused)
        else:
            predictions = {}
        return predictions, embs

    def predict(self, batch: dict[str, Any]) -> torch.Tensor | dict[str, torch.Tensor]:
        predictions, _ = self.forward(batch)
        return predictions

    def encode(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        _, embs = self.forward(batch)
        return embs
