import copy
from typing import Optional, Sequence, Tuple

import numpy as np
from maite.protocols import ArrayLike
from maite.protocols.image_classification import (
    Augmentation,
    DatumMetadataBatchType,
    InputBatchType,
    TargetBatchType,
)
from nrtk.interfaces.image_metric import ImageMetric
from nrtk.interfaces.perturb_image import PerturbImage

IMG_CLASSIFICATION_BATCH_T = Tuple[InputBatchType, TargetBatchType, DatumMetadataBatchType]


class JATICClassificationAugmentation(Augmentation):
    """Implementation of JATIC Augmentation for NRTK perturbers.

    Implementation of JATIC Augmentation for NRTK perturbers operating on a MAITE-protocol
    compliant Image Classification dataset.

    Parameters
    ----------
    augment : PerturbImage
        Augmentations to apply to an image.
    """

    def __init__(self, augment: PerturbImage) -> None:
        self.augment = augment

    def __call__(self, batch: IMG_CLASSIFICATION_BATCH_T) -> IMG_CLASSIFICATION_BATCH_T:
        """Apply augmentations to the given data batch."""
        imgs, anns, metadata = batch

        # iterate over (parallel) elements in batch
        aug_imgs = list()  # list of individual augmented inputs
        aug_labels = list()  # list of individual class labels
        aug_metadata = list()  # list of individual image-level metadata

        for img, ann, md in zip(imgs, anns, metadata):
            # Perform augmentation
            aug_img = copy.deepcopy(img)
            aug_img = self.augment(np.asarray(aug_img), md)
            aug_height, aug_width = aug_img.shape[0:2]
            aug_imgs.append(aug_img)

            y_aug_labels = copy.deepcopy(ann)
            aug_labels.append(y_aug_labels)

            m_aug = copy.deepcopy(md)
            m_aug.update(
                {
                    "nrtk::perturber": self.augment.get_config(),
                    "image_info": {"width": aug_width, "height": aug_height},
                }
            )
            aug_metadata.append(m_aug)

        # return batch of augmented inputs, class labels and updated metadata
        return aug_imgs, aug_labels, aug_metadata


class JATICClassificationAugmentationWithMetric(Augmentation):
    """Implementation of JATIC augmentation wrapper for NRTK's Image metrics.

    Implementation of JATIC augmentation for NRTK metrics operating on a MAITE-protocol
    compliant image classification dataset.

    Parameters
    ----------
    augmentations : Optional[Sequence[Augmentation]]
        Optional task-specific sequence of JATIC augmentations to be applied on a given batch.
    metric : ImageMetric
        Image metric to be applied for a given image.
    """

    def __init__(self, augmentations: Optional[Sequence[Augmentation]], metric: ImageMetric) -> None:
        self.augmentations = augmentations
        self.metric = metric

    def __call__(self, batch: IMG_CLASSIFICATION_BATCH_T) -> IMG_CLASSIFICATION_BATCH_T:
        """Compute a specified image metric on the given batch."""
        imgs, anns, metadata = batch
        metric_aug_metadata = list()  # list of individual image-level metric metadata

        aug_imgs: Sequence[Optional[ArrayLike]] = list()
        if self.augmentations:
            aug_batch = batch
            for aug in self.augmentations:
                aug_batch = aug(aug_batch)
            aug_imgs, aug_anns, aug_metadata = aug_batch
        else:
            aug_imgs, aug_anns, aug_metadata = [None] * len(imgs), anns, metadata

        for img, aug_img, aug_md in zip(imgs, aug_imgs, aug_metadata):
            # Convert from channels-first to channels-last
            img_1 = np.transpose(img, (1, 2, 0))
            if aug_img is None:
                img_2 = None
            else:
                img_2 = np.transpose(aug_img, (1, 2, 0))

            # Compute Image metric values
            metric_value = self.metric(img_1=img_1, img_2=img_2, additional_params=aug_md)
            metric_aug_md = copy.deepcopy(aug_md)
            metric_name = self.metric.__class__.__name__
            metric_aug_md.update({"nrtk::" + metric_name: metric_value})
            metric_aug_metadata.append(metric_aug_md)

        # return batch of augmented/original images, annotations and metric-updated metadata
        if self.augmentations:
            # type ignore was included to handle the dual Sequence[ArrrayLike] | List[None]
            # case for the augmented images.
            return aug_imgs, aug_anns, metric_aug_metadata  # type: ignore
        else:
            return imgs, aug_anns, metric_aug_metadata
