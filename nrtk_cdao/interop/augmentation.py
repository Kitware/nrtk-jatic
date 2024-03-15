from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from nrtk.interfaces.perturb_image import PerturbImage  # type: ignore
from nrtk.impls.perturb_image.pybsm.perturber import PybsmPerturber  # type: ignore
from nrtk_cdao.interop.dataset import CustomMAITEClassificationDataset, CustomMAITEDetectionDataset
from maite.protocols import (
    ArrayLike, SupportsObjectDetection,
    SupportsImageClassification
)


@dataclass
class ImageMetadata:
    id: str
    image_info: Dict


class JATICClassificationAugmentation:
    """
    Implementation of JATIC Classification Dataset Augmentation for NRTK perturbers.

    Parameters
    ----------
    augment : list[PerturbImage]
        List of augmentations to apply.

    Methods
    -------
    __call__(data: SupportsImageClassification) -> SupportsImageClassification
        Apply augmentations to given data.
    """
    def __init__(self, augment: List[PerturbImage], **kwargs):  # type: ignore
        self.augments = augment
        self.img_metadata = kwargs

    def __call__(self, data: CustomMAITEClassificationDataset) -> SupportsImageClassification:

        images = [img["image"].detach().cpu().numpy() for img in data]  # type: ignore

        aug_imgs: List[ArrayLike] = []
        for img in zip(images):
            aug_img = img[0].transpose(1, 2, 0)  # type: ignore
            assert isinstance(aug_img, np.ndarray)
            for aug in self.augments:
                if isinstance(aug, PybsmPerturber):
                    aug_img = aug(aug_img, {'img_gsd': self.img_metadata.get("gsd")})
                else:
                    aug_img = aug(aug_img)
            aug_imgs.append(aug_img)

        output: SupportsImageClassification = {  # type: ignore
            "image": aug_imgs
        }
        output.update({k: v for k, v in data.__dict__.items() if k != "image"})  # type: ignore
        return output


class JATICDetectionAugmentation:
    """
    Implementation of JATIC Classification Dataset Augmentation for NRTK perturbers.

    Parameters
    ----------
    augment : list[PerturbImage]
        List of augmentations to apply.

    Methods
    -------
    __call__(data: CustomMAITEDetectionDataset) -> SupportsObjectDetection
        Apply augmentations to given data.
    """
    def __init__(self, augment: List[PerturbImage], **kwargs):  # type: ignore
        self.augments = augment
        self.img_metadata = kwargs

    def __call__(self, data: CustomMAITEDetectionDataset) -> SupportsObjectDetection:

        images = [img["image"].detach().cpu().numpy() for img in data]  # type: ignore

        aug_imgs: List[ArrayLike] = []
        for img in zip(images):
            aug_img = img[0].transpose(1, 2, 0)  # type: ignore
            assert isinstance(aug_img, np.ndarray)
            for aug in self.augments:
                if isinstance(aug, PybsmPerturber):
                    aug_img = aug(aug_img, {'img_gsd': self.img_metadata.get("gsd")})
                else:
                    aug_img = aug(aug_img)
            aug_imgs.append(aug_img)

        output: SupportsObjectDetection = {"image": aug_imgs}  # type: ignore
        output.update({k: v for k, v in data.__dict__.items() if k != "image"})  # type: ignore
        return output
