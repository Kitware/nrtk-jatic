from typing import Tuple
import copy

import numpy as np

from nrtk.interfaces.perturb_image import PerturbImage
from maite.protocols.image_classification import (
    Augmentation,
    InputBatchType,
    TargetBatchType,
    DatumMetadataBatchType
)

IMG_CLASSIFICATION_BATCH_T = Tuple[InputBatchType, TargetBatchType, DatumMetadataBatchType]


class JATICClassificationAugmentation(Augmentation):
    """
    Implementation of JATIC Augmentation for NRTK perturbers operating
    on a MAITE-protocol compliant Image Classification dataset.

    Parameters
    ----------
    augment : PerturbImage
        Augmentations to apply to an image.
    """
    def __init__(self, augment: PerturbImage):
        self.augment = augment

    def __call__(self,
                 batch: IMG_CLASSIFICATION_BATCH_T
                 ) -> IMG_CLASSIFICATION_BATCH_T:
        """
        Apply augmentations to the given data batch.
        """
        imgs, anns, metadata = batch
        imgs = np.asarray(imgs)
        anns = np.asarray(anns)

        # iterate over (parallel) elements in batch
        aug_imgs = []  # list of individual augmented inputs
        aug_labels = []  # list of individual class labels
        aug_metadata = []  # list of individual image-level metadata

        for img, ann, md in zip(imgs, anns, metadata):
            # Perform augmentation
            aug_img = copy.deepcopy(img)
            aug_img = self.augment(aug_img, md)
            aug_height, aug_width = aug_img.shape[0:2]
            aug_imgs.append(aug_img)

            y_aug_labels = copy.deepcopy(ann)
            aug_labels.append(y_aug_labels)

            m_aug = copy.deepcopy(md)
            m_aug.update(
                {
                    "nrtk::perturber": self.augment.get_config(),
                    "image_info": {
                        "width": aug_width,
                        "height": aug_height
                    }
                }
            )
            aug_metadata.append(m_aug)

        # return batch of augmented inputs, class labels and updated metadata
        return np.stack(aug_imgs), np.array(aug_labels), aug_metadata
