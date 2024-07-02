import copy
import numpy as np
import pytest
from typing import Any, Dict, List

from maite.protocols.image_classification import TargetBatchType
from nrtk.interfaces.perturb_image import PerturbImage
from nrtk.impls.perturb_image.generic.nop_perturber import NOPPerturber

from nrtk_jatic.interop.image_classification.augmentation import JATICClassificationAugmentation
from tests.utils.test_utils import ResizePerturber


class TestJATICClassificationAugmentation:
    @pytest.mark.parametrize("perturber, targets_in, expected_targets_out", [
        (
            NOPPerturber(),
            np.asarray([0]),
            np.asarray([0])
        ), (
            ResizePerturber(w=64, h=512),
            np.asarray([1]),
            np.asarray([1])
        )

    ], ids=["no-op perturber", "resize"])
    def test_augmentation_adapter(
        self,
        perturber: PerturbImage,
        targets_in: TargetBatchType,
        expected_targets_out: TargetBatchType
    ) -> None:
        """
        Test that the adapter provides the same image perturbation result
        as the core perturber and that labels and metadata are appropriately
        updated.
        """
        augmentation = JATICClassificationAugmentation(augment=perturber)
        img_in = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        md_in: List[Dict[str, Any]] = [{"some_metadata": 1}]

        # Get copies to check for modification
        img_copy = np.copy(img_in)
        targets_copy = copy.deepcopy(targets_in)
        md_copy = copy.deepcopy(md_in)

        # Get expected image and metadata from "normal" perturber
        expected_img_out = perturber(img_in)
        expected_md_out = dict(md_in[0])
        expected_md_out["nrtk::perturber"] = perturber.get_config()
        expected_md_out.update(
            {
                "image_info": {
                    "width": expected_img_out.shape[1],
                    "height": expected_img_out.shape[0]
                }
            }
        )

        # Apply augmentation via adapter
        imgs_out, targets_out, md_out = augmentation((
            np.expand_dims(img_in, axis=0),
            targets_in,
            md_in
        ))
        imgs_out = np.asarray(imgs_out)
        targets_out = np.asarray(targets_out)
        expected_targets_out = np.asarray(expected_targets_out)

        # Check that expectations hold
        assert np.array_equal(imgs_out[0], expected_img_out)
        assert np.array_equal(targets_out, expected_targets_out)

        for etgt, tgt_out in zip(expected_targets_out, targets_out):
            assert np.array_equal(etgt, tgt_out)
        assert md_out[0] == expected_md_out

        targets_copy = np.asarray(targets_copy)
        targets_in = np.asarray(targets_in)

        # Check that input data was not modified
        assert np.array_equal(img_in, img_copy)
        assert np.array_equal(targets_copy, targets_in)
        for tgt_copy, tgt_in in zip(targets_copy, targets_in):
            assert np.array_equal(tgt_copy, tgt_in)
        assert md_in == md_copy
