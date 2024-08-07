import copy
from typing import Any, Dict, List

import numpy as np
import pytest
from maite.protocols.object_detection import TargetBatchType
from nrtk.impls.perturb_image.generic.nop_perturber import NOPPerturber
from nrtk.interfaces.perturb_image import PerturbImage

from nrtk_jatic.interop.object_detection.augmentation import JATICDetectionAugmentation
from nrtk_jatic.interop.object_detection.dataset import JATICDetectionTarget
from tests.utils.test_utils import ResizePerturber


class TestJATICDetectionAugmentation:
    @pytest.mark.parametrize(
        ("perturber", "targets_in", "expected_targets_out"),
        [
            (
                NOPPerturber(),
                [
                    JATICDetectionTarget(
                        boxes=np.asarray([[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0]]),
                        labels=np.asarray([0, 2]),
                        scores=np.asarray([0.8, 0.86]),
                    )
                ],
                [
                    JATICDetectionTarget(
                        boxes=np.asarray([[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0]]),
                        labels=np.asarray([0, 2]),
                        scores=np.asarray([0.8, 0.86]),
                    )
                ],
            ),
            (
                ResizePerturber(w=64, h=512),
                [
                    JATICDetectionTarget(
                        boxes=np.asarray(
                            [[4.0, 8.0, 16.0, 32.0], [2.0, 4.0, 6.0, 8.0]]
                        ),
                        labels=np.asarray([1, 5]),
                        scores=np.asarray([0.8, 0.86]),
                    )
                ],
                [
                    JATICDetectionTarget(
                        boxes=np.asarray(
                            [[1.0, 16.0, 4.0, 64.0], [0.5, 8.0, 1.5, 16.0]]
                        ),
                        labels=np.asarray([1, 5]),
                        scores=np.asarray([0.8, 0.86]),
                    )
                ],
            ),
        ],
        ids=["no-op perturber", "resize"],
    )
    def test_augmentation_adapter(
        self,
        perturber: PerturbImage,
        targets_in: TargetBatchType,
        expected_targets_out: TargetBatchType,
    ) -> None:
        """Test that the augmentation adapter functions appropriately.

        Tests that the adapter provides the same image perturbation result
        as the core perturber and that bboxes and metadata are appropriately
        updated.
        """
        augmentation = JATICDetectionAugmentation(augment=perturber)
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

        # Apply augmentation via adapter
        imgs_out, targets_out, md_out = augmentation(([img_in], targets_in, md_in))

        # Check that expectations hold
        assert np.array_equal(imgs_out[0], expected_img_out)
        assert len(targets_out) == len(expected_targets_out)
        for expected_tgt, tgt_out in zip(expected_targets_out, targets_out):
            assert np.array_equal(expected_tgt.boxes, tgt_out.boxes)
            assert np.array_equal(expected_tgt.labels, tgt_out.labels)
            assert np.array_equal(expected_tgt.scores, tgt_out.scores)
        assert md_out[0] == expected_md_out

        # Check that input data was not modified
        assert np.array_equal(img_in, img_copy)
        assert len(targets_copy) == len(targets_in)
        for tgt_copy, tgt_in in zip(targets_copy, targets_in):
            assert np.array_equal(tgt_copy.boxes, tgt_in.boxes)
            assert np.array_equal(tgt_copy.labels, tgt_in.labels)
            assert np.array_equal(tgt_copy.scores, tgt_in.scores)
        assert md_in == md_copy
