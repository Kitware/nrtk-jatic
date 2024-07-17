from typing import Sequence, Tuple

import numpy as np
from maite.protocols.image_classification import (
    Dataset,
    DatumMetadataType,
    InputType,
    TargetType,
)

IMG_CLASSIFICATION_DATUM_T = Tuple[InputType, TargetType, DatumMetadataType]


class JATICImageClassificationDataset(Dataset):
    """Implementation of the JATIC Image Classification dataset wrapper for dataset images of varying sizes.

    Parameters
    ----------
    imgs : Sequence[np.ndarray]
        Sequence of images.
    labels : Sequence[ArrayLike]
        Sequence of labels for each image.
    metadata : Sequence[Dict[str, Any]]
        Sequence of custom metadata for each image.
    """

    def __init__(
        self,
        imgs: Sequence[np.ndarray],
        labels: Sequence[TargetType],
        metadata: Sequence[DatumMetadataType],
    ) -> None:
        self.imgs = imgs
        self.labels = labels
        self.metadata = metadata

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, index: int) -> IMG_CLASSIFICATION_DATUM_T:

        return self.imgs[index], self.labels[index], self.metadata[index]
