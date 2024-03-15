from typing import List

from PIL import Image  # type: ignore
from torchvision.transforms.functional import pil_to_tensor  # type: ignore

from maite.protocols import SupportsObjectDetection, SupportsImageClassification


class CustomMAITEClassificationDataset:
    """
    Custom MAITE dataset class that loads a PIL image and
    returns the image in a format supported by the
    `SupportsImageClassification` protocol.
    """
    def __init__(self, img_paths: List) -> None:
        self.data_path = img_paths

    def __len__(self) -> int:
        return len(self.data_path)

    def __getitem__(self, idx: int) -> SupportsImageClassification:
        im_path = self.data_path[idx]
        img = Image.open(im_path)
        maite_img: SupportsImageClassification = {  # type: ignore
            "image": pil_to_tensor(img)
        }
        return maite_img


class CustomMAITEDetectionDataset:
    """
    Custom MAITE dataset class that loads a PIL image and
    returns the image in a format supported by the
    `SupportsObjectDetection` protocol.

    """
    def __init__(self, img_paths: List) -> None:
        self.data_path = img_paths

    def __len__(self) -> int:
        return len(self.data_path)

    def __getitem__(self, idx: int) -> SupportsObjectDetection:
        im_path = self.data_path[idx]
        img = Image.open(im_path)
        maite_img: SupportsObjectDetection = {  # type: ignore
            "image": pil_to_tensor(img)
        }
        return maite_img
