from typing import List

import numpy as np

from nrtk.interfaces.perturb_image import PerturbImage  # type: ignore
from nrtk.impls.perturb_image.pybsm.perturber import PybsmPerturber  # type: ignore
from nrtk_cdao.interop.dataset import CustomMAITEDataset
from maite.protocols import (
    ArrayLike, HasDataImage,
)


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

    def __call__(self, data: CustomMAITEDataset) -> HasDataImage:

        images = [img["image"].detach().cpu().numpy() for img in data]  # type: ignore

        aug_imgs: List[ArrayLike] = []
        for img in zip(images):
            aug_img = img[0].transpose(1, 2, 0)  # type: ignore
            assert isinstance(aug_img, np.ndarray)
            for aug in self.augments:
                if isinstance(aug, PybsmPerturber):
                    aug_img = aug(aug_img, {'img_gsd': self.img_metadata.get("gsd")})
                else:
                    aug_img = aug(aug_img, self.img_metadata)
            aug_imgs.append(aug_img)

        output: HasDataImage = {
            "image": aug_imgs
        }
        output.update({k: v for k, v in data.__dict__.items() if k != "image"})  # type: ignore
        return output
