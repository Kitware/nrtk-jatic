from typing import Tuple, Sequence
import itertools
import copy

import numpy as np
from maite.protocols.image_classification import (
    DatumMetadataBatchType,
    InputBatchType,
    TargetBatchType
)
from nrtk.interfaces.image_metric import ImageMetric
from nrtk.interfaces.perturb_image import PerturbImage

METRIC_BATCH_T = Tuple[
    InputBatchType, TargetBatchType, DatumMetadataBatchType
]


class JATICImageMetricAugmentation:
    """Implementation of JATIC Augmentation for NRTK's Image metrics.

    Implementation of JATIC Augmentation for NRTK metrics operating on a MAITE-protocol
    compliant image dataset.

    Parameters
    ----------
    metric : ImageMetric
        Image metric to be applied for a given image.
    """

    def __init__(self, perturbers: Sequence[PerturbImage], metric: ImageMetric):
        self.perturbers = perturbers
        self.metric = metric

    def __call__(self, batch: METRIC_BATCH_T) -> METRIC_BATCH_T:
        """Compute a specified image metric on the given batch"""
        imgs_1, imgs_2, metadata = batch

        imgs = []  # list of original images
        aug_imgs = []  # list of optional 2nd image data for augmentation
        aug_metadata = []  # list of individual image-level metadata

        # Handling the case for populating the longest value sequence
        # in the presence of None values
        for img_1, img_2, md in itertools.zip_longest(imgs_1, imgs_2, metadata):

            # Convert from channels-first to channels-last
            img_1 = np.transpose(img_1, (1, 2, 0))
            img_2 = np.transpose(img_2, (1, 2, 0))
            imgs.append(np.asarray(img_1))

            aug_img = copy.deepcopy(img_2)
            aug_md = copy.deepcopy(md)
            for i, perturber in enumerate(self.perturbers):
                aug_img = perturber(np.asarray(aug_img), md)
                aug_height, aug_width = aug_img.shape[0:2]
                aug_md.update({
                    "nrtk::perturber_" + str(i): perturber.get_config(),
                })
            aug_imgs.append(aug_img)

            # Compute Image metric values
            metric_value = self.metric(img_1=img_1, img_2=aug_img, additional_params=md)
            aug_md.update({
                "image_info": {"width": aug_width, "height": aug_height},
                "metric": {
                    "name": self.metric.__class__.__name__,
                    "value": metric_value
                }
            })
            aug_metadata.append(aug_md)

        # return batch of original images, augmented images and metric-updated metadata
        return imgs, aug_imgs, aug_metadata
