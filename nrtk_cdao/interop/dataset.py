from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Tuple, Any, Union
from pathlib import Path

from PIL import Image  # type: ignore
import torch
from torchvision.ops.boxes import box_convert  # type: ignore
from torchvision.transforms.functional import pil_to_tensor  # type: ignore

from maite.protocols import ArrayLike
from maite.protocols.object_detection import ObjectDetectionTarget


@dataclass
class JATICDetectionTarget:
    boxes: torch.Tensor
    labels: torch.Tensor
    scores: torch.Tensor


def _coco_to_maite_detections(coco_annotation: List) -> ObjectDetectionTarget:

    num_anns = len(coco_annotation)
    boxes = torch.zeros(num_anns, 4)
    for i, anns in enumerate(coco_annotation):
        # convert box from xywh in xyxy format
        box = torch.tensor(list(map(int, anns["bbox"])))
        box = box_convert(box, "xywh", "xyxy")
        boxes[i, :] = box

    # change class label from 1...10 to 0...9
    labels = torch.tensor([int(anns["category_id"]) - 1
                           for anns in coco_annotation])
    scores = torch.ones(num_anns)

    return JATICDetectionTarget(boxes, labels, scores)


class JATICObjectDetectionDataset:
    """
    Dataset class for Object Detection that is supported by
    JATIC's Object Detection protocol.

    Parameters
    ----------
    root : Path | str
        The root directory of the dataset.
    kwcoco_dataset:
        The kwcoco COCODataset object.
    Methods
    -------
    __len__() -> int
        Returns the number of images in the dataset.
    __getitem__(index: int) -> Union[JATICObjectDetectionDataset,
               Tuple[ArrayLike, ObjectDetectionTarget, Dict[str, Any]]]
        Returns the image, annotations and metadata at the given index.
    """

    def __init__(self, root: Union[Path, str], kwcoco_dataset):  # type: ignore

        self._root: Path = Path(root)

        image_dir = self._root / "images"

        file_exts = ['jpg', 'JPG', 'png', 'PNG']
        self.all_img_paths = [filename for ext in file_exts
                              for filename in image_dir.glob("*." + ext)]
        self.all_image_ids = sorted({p.stem for p in self.all_img_paths})

        # Get all image filenames from the kwcoco object
        anns_image_ids = [{'coco_image_id': val['id'], 'filename': val['file_name']}
                          for key, val in kwcoco_dataset.imgs.items()]
        anns_image_ids = sorted(anns_image_ids, key=lambda d: d['filename'])

        # store sorted image paths
        self._images = sorted([p for p in self.all_img_paths if p.stem in self.all_image_ids])

        self._annotations = {}
        for image_id, anns_img_id in zip(self.all_image_ids, anns_image_ids):
            image_annotations = [sub for sub in list(kwcoco_dataset.anns.values())
                                 if sub['image_id'] == anns_img_id['coco_image_id']]
            # Convert annotations to maite detections format
            self._annotations[image_id] = _coco_to_maite_detections(image_annotations)

        self.classes = list(kwcoco_dataset.cats.values())

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(
        self, index: int
    ) -> Union[JATICObjectDetectionDataset,
               Tuple[ArrayLike, ObjectDetectionTarget, Dict[str, Any]]]:
        image_path = self._images[index]
        image = Image.open(image_path)
        image_id = image_path.stem
        width, height = image.size
        annotation = self._annotations[image_id].labels
        num_objects = annotation.size(dim=0)  # type: ignore
        uniq_objects = torch.unique(annotation)  # type: ignore
        num_unique_classes = uniq_objects.size(dim=0)
        unique_classes = [self.classes[int(idx)]['name']
                          for idx in uniq_objects.tolist()]  # type: ignore

        input_img, bbox_labels, metadata = pil_to_tensor(image), self._annotations[image_id], {
            "id": image_id,
            "image_info": {
                "width": width,
                "height": height,
                "num_objects": num_objects,
                "num_unique_classes": num_unique_classes,
                "unique_classes": unique_classes}}
        return input_img, bbox_labels, metadata

    def get_img_path_list(self) -> List[Path]:
        return self.all_img_paths
