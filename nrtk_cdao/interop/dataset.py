from dataclasses import dataclass
from typing import List, Tuple, Any, Sequence
from pathlib import Path

from PIL import Image  # type: ignore
import numpy as np
import kwcoco

from maite.protocols.object_detection import (
    Dataset,
    InputType,
    TargetType,
    DatumMetadataType
)

OBJ_DETECTION_DATUM_T = Tuple[InputType, TargetType, DatumMetadataType]


@dataclass
class JATICDetectionTarget:
    boxes: np.ndarray
    labels: np.ndarray
    scores: np.ndarray


def _xyxy_bbox_form(x: int, y: int,
                    w: int, h: int
                    ) -> Tuple[int, int, int, int]:
    return x, y, x + w, y + h


def _coco_to_maite_detections(coco_annotation: List) -> TargetType:
    num_anns = len(coco_annotation)
    boxes = np.zeros((num_anns, 4))
    for i, anns in enumerate(coco_annotation):
        box = list(map(int, anns["bbox"]))
        # convert box from xywh in xyxy format
        x1, y1, x2, y2 = _xyxy_bbox_form(x=box[0], y=box[1],
                                         w=box[2], h=box[3])
        boxes[i, :] = np.array((x1, y1, x2, y2))

    labels = np.stack([int(anns["category_id"])
                      for anns in coco_annotation])
    scores = np.ones(num_anns)

    return JATICDetectionTarget(boxes, labels, scores)


class COCOJATICObjectDetectionDataset(Dataset):
    """
    Dataset class to convert a COCO dataset to a dataset
    compliant with JATIC's Object Detection protocol.

    Parameters
    ----------
    root : str
        The root directory of the dataset.
    kwcoco_dataset: kwcoco.CocoDataset
        The kwcoco COCODataset object.
    **kwargs: Any
        Additional dataset-related metadata required by
        a perturber during augmentation
    """

    def __init__(self, root: str,
                 kwcoco_dataset: kwcoco.CocoDataset,
                 **kwargs: Any):
        self._root: Path = Path(root)
        self.kwargs = kwargs
        image_dir = self._root / "images"
        self.all_img_paths = [image_dir / val["file_name"]
                              for key, val in kwcoco_dataset.imgs.items()]
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
        """
        Returns the number of images in the dataset.
        """
        return len(self._images)

    def __getitem__(
        self, index: int
    ) -> OBJ_DETECTION_DATUM_T:
        """
        Returns the dataset object at the given index
        """
        image_path = self._images[index]
        image = Image.open(image_path)
        image_id = image_path.stem
        width, height = image.size
        annotation = self._annotations[image_id].labels
        num_objects = np.asarray(annotation).shape[0]
        uniq_objects = np.unique(annotation)
        num_unique_classes = uniq_objects.shape[0]
        unique_classes = [self.classes[int(idx)]['name']
                          for idx in uniq_objects.tolist()]

        input_img, dets, metadata = (
            np.asarray(image),
            self._annotations[image_id],
            dict(
                id=image_id,
                image_info=dict(
                    width=width,
                    height=height,
                    num_objects=num_objects,
                    num_unique_classes=num_unique_classes,
                    unique_classes=unique_classes
                )
            )
        )

        metadata.update(self.kwargs)

        return input_img, dets, metadata

    def get_img_path_list(self) -> List[Path]:
        """
        Returns the sorted list of absolute paths
        for the input images.
        """
        return sorted(self.all_img_paths)


class JATICObjectDetectionDataset(Dataset):
    def __init__(self, imgs: Sequence[np.ndarray], dets: Sequence[TargetType],
                 metadata: Sequence[DatumMetadataType]) -> None:
        self.imgs = imgs
        self.dets = dets
        self.metadata = metadata

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self,
                    index: int
                    ) -> OBJ_DETECTION_DATUM_T:

        return self.imgs[index], self.dets[index], self.metadata[index]
