from typing import List

import numpy as np

from nrtk.interfaces.perturb_image import PerturbImage  # type: ignore
from nrtk.impls.perturb_image.pybsm.perturber import PybsmPerturber  # type: ignore
from maite.protocols import ArrayLike


class JATICAugmentation:
    """
    Implementation of JATIC Augmentation for NRTK perturbers.

    Parameters
    ----------
    augment : list[PerturbImage]
        List of augmentations to apply.

    Methods
    -------
    __call__(data: HasDataImage) -> HasDataImage
        Apply augmentations to given data.
    """
    def __init__(self, augment: List[PerturbImage], **kwargs):  # type: ignore
        self.augments = augment
        self.img_metadata = kwargs

    def __call__(self, data: ArrayLike) -> ArrayLike:

        img = data.numpy()  # type: ignore
        aug_img = img.transpose(1, 2, 0)
        assert isinstance(aug_img, np.ndarray)
        for aug in self.augments:
            if isinstance(aug, PybsmPerturber):
                aug_img = aug(aug_img, {'img_gsd': self.img_metadata.get("gsd")})
            else:
                aug_img = aug(aug_img, self.img_metadata)

        output: ArrayLike = aug_img

        return output
