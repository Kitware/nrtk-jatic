from typing import Any, Dict

import numpy as np
from PIL import Image

from nrtk.interfaces.perturb_image import PerturbImage


class ResizePerturber(PerturbImage):
    def __init__(self, w: int, h: int):
        self.w = w
        self.h = h

    def perturb(
        self,
        image: np.ndarray,
        additional_params: Dict[str, Any] = {}
    ) -> np.ndarray:
        """
        Resize image.
        """
        img = Image.fromarray(image)
        img = img.resize((self.w, self.h))
        return np.array(img)

    def get_config(self) -> Dict[str, Any]:
        return {
            "w": self.w,
            "h": self.h
        }
