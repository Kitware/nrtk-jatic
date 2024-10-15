from typing import Any, Dict, Optional

import numpy as np
from nrtk.interfaces.perturb_image import PerturbImage
from PIL import Image


class ResizePerturber(PerturbImage):
    def __init__(self, w: int, h: int):
        self.w = w
        self.h = h

    def perturb(self, image: np.ndarray, additional_params: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Resize image."""
        if additional_params is None:
            additional_params = {}
        img = Image.fromarray(image)
        img = img.resize((self.w, self.h))
        return np.array(img)

    def get_config(self) -> Dict[str, Any]:
        return {"w": self.w, "h": self.h}
