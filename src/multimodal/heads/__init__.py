from multimodal.heads.basic import (
    MultiTaskHead,
    MultiTaskLinearHead,
    MultiTaskLinearSliceHead,
    MultiTaskMLPHead,
    MultiTaskMLPSliceHead,
    MultiTaskSliceHead,
)
from multimodal.heads.contrastive import (
    ModalityContrastiveHead,
    SupervisedContrastiveHead,
)
from multimodal.heads.m3h import M3HHead

__all__ = [
    "MultiTaskHead",
    "MultiTaskLinearHead",
    "MultiTaskLinearSliceHead",
    "MultiTaskMLPHead",
    "MultiTaskMLPSliceHead",
    "MultiTaskSliceHead",
    "ModalityContrastiveHead",
    "SupervisedContrastiveHead",
    "M3HHead",
]
