from typing import Any

import numpy as np
import pytest
from PIL import Image

from tests.utils.test_utils import ResizePerturber

random = np.random.default_rng()


class TestResizePerturber:
    @pytest.mark.parametrize(
        "additional_params",
        [
            None,
            [{"some_metadata": 1}],
        ],
    )
    def test_augmentation_adapter(
        self,
        additional_params: list[dict[str, Any]],
    ) -> None:
        """Test that the adapter provides the same image perturbation result as the core perturber.

        Also tests that labels and metadata are appropriately updated.
        """
        perturber = ResizePerturber(w=64, h=512)
        img_in = random.integers(0, 255, (256, 256, 3), dtype=np.uint8)

        # Get copies to check for modification
        img_copy = np.copy(img_in)

        # Get expected image
        img = Image.fromarray(img_copy)
        expected_img_out = img.resize((64, 512))

        # Apply augmentation via adapter
        imgs_out, _ = perturber.perturb(img_in, additional_params=additional_params)

        # Check that expectations hold
        assert np.array_equal(imgs_out, expected_img_out)
        assert perturber.get_config() == {"w": 64, "h": 512}
