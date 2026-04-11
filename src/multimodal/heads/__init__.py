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

__all__ = [
    "MultiTaskHead",
    "MultiTaskLinearHead",
    "MultiTaskLinearSliceHead",
    "MultiTaskMLPHead",
    "MultiTaskMLPSliceHead",
    "MultiTaskSliceHead",
    "ModalityContrastiveHead",
    "SupervisedContrastiveHead",
]
