from typing import List

from PIL import Image  # type: ignore
from torchvision.transforms.functional import pil_to_tensor  # type: ignore

from maite.protocols import HasDataImage


class CustomMAITEDataset:
    """
    Custom MAITE dataset class that takes in a list of image paths,
    loads the corresponding PIL image and returns the image in a format
    supported by the `HasDataImage` maite protocol.

    """
    def __init__(self, img_paths: List) -> None:
        self.data_path = img_paths

    def __len__(self) -> int:
        return len(self.data_path)

    def __getitem__(self, idx: int) -> HasDataImage:
        im_path = self.data_path[idx]
        img = Image.open(im_path)
        maite_img: HasDataImage = {
            "image": pil_to_tensor(img)
        }
        return maite_img
