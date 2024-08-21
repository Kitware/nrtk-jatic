import maite.protocols as pr
import numpy as np
import torch
import urllib

from dataclasses import dataclass
from numpy.typing import NDArray
from pathlib import Path
from smqtk_detection.impls.detect_image_objects.centernet import CenterNetVisdrone
from smqtk_detection.interfaces.detect_image_objects import DetectImageObjects
from typing import Any, Dict, Iterable, Literal, Sequence, Tuple

VISDRONE_CLASSES = (
        'pedestrian',
        'people',
        'bicycle',
        'car',
        'van',
        'truck',
        'tricycle',
        'awning-tricycle',
        'bus',
        'motor',
    )


@dataclass
class SMQTKObjectDetectionOutput:
    boxes: Sequence[NDArray]
    labels: Sequence[NDArray]
    scores: Sequence[NDArray]


@dataclass
class SMQTKModelMetadata:  # This is the implementation of ModelMetadata protocol
    model_name: str
    provider: str
    task: str
    model_info: Dict  # model_info is the user defined metadata


def _get_top_label_score(label2score) -> Tuple[Any, float]:
    k = list(label2score.keys())
    v = list(label2score.values())
    max_v = max(v)
    return k[v.index(max_v)], max_v


class SMQTKObjectDetector:
    """
    Wraps SMQTK `DetectImageObjects` as MAITE `ObjectDetector`.

    Parameters
    ----------
    smqtk_detector : DetectImageObjects
        The SMQTK bject detector to wrap.
    labels : Sequence[str]
        Labels for classes that object detector can detect.
    model_name: str
        The name of the model, used for model's metadata
    map_output_labels : bool
        Whether wrapper needs to map string outputs from SMQTK detector to integers.
    metadata: SMQTKModelMetadata
        Model metadata
    """

    metadata: SMQTKModelMetadata

    def __init__(self, smqtk_detector: DetectImageObjects, labels: Sequence[str], map_output_labels: bool, metadata: SMQTKModelMetadata) -> None:
        self.smqtk_detector = smqtk_detector
        self.labels = labels
        self.label2int: Dict[str, int] = {label: idx for idx, label in enumerate(labels)}
        self.map_output_labels = map_output_labels
        self.metadata = metadata

    def get_labels(self) -> Sequence[str]:
        return self.labels

    def __call__(self, data: pr.SupportsArray) -> SMQTKObjectDetectionOutput:

        # SMQTK DetectImageObjects.detect_objects
        # - input: Iterable[ndarray]
        # - output: Iterable[Iterable[Tuple[AxisAlignedBoundingBox, Dict[Hashable, float]]]]

        # reformat data as sequence first if necessary
        if isinstance(data, torch.Tensor):
            if len(data.shape) == 4:
                # batch of tensors
                data_seq = [d for d in data]
            else:
                # single tensor
                data_seq = [data]
        elif isinstance(data, np.ndarray):
            if len(data.shape) == 4:
                # batch of arrays
                data_seq = [d for d in data]
            else:
                # single array
                data_seq = [data]
        elif isinstance(data, Sequence):
            # already a sequence
            data_seq = data
        else:
            raise Exception(f"Unable to handle input of type: {type(data)}")

        # convert to Iterable[ndarray]
        arr_iter: Iterable[NDArray] = []
        for img in data_seq:
            if isinstance(img, torch.Tensor) and len(img.shape) == 3:
                # TODO: do we need to convert CHW to HWC?
                arr_iter.append(img.detach().cpu().numpy().transpose([1, 2, 0]))
            elif isinstance(img, np.ndarray) and len(img.shape) == 3:
                arr_iter.append(img)
            else:
                raise Exception(f"Unable to handle sequence item of type: {type(img)}")

        smqtk_output = self.smqtk_detector(arr_iter)

        # assume num detections for image i is nd_i
        # sequence of shape-(nd_i, 4) bounding box arrays
        all_boxes = []

        # sequence of shape-(nd_i,) arrays of predicted class associated with each bounding box
        all_labels = []

        # sequence of shape-(nd_i,) arrays of score for predicted class associated with each bounding box
        all_scores = []

        for img_dets in smqtk_output:
            # bounding boxes, top label for each box, top score for each box for *single* image
            boxes = []
            labels = []
            scores = []

            for bbox, label2score in img_dets:
                flatten_box = np.hstack([bbox.min_vertex, bbox.max_vertex])
                boxes.append(flatten_box)

                top_label, top_score = _get_top_label_score(label2score)
                if self.map_output_labels:
                    top_label = self.label2int[top_label]

                labels.append(top_label)
                scores.append(top_score)

            all_boxes.append(np.asarray(boxes))
            all_labels.append(np.asarray(labels))
            all_scores.append(np.asarray(scores))

        return SMQTKObjectDetectionOutput(boxes=all_boxes, labels=all_labels, scores=all_scores)


# myee: this will wrap SMQTK models and return MAITE models
def load_demo_model(
    name: Literal["centernet", "tph-yolov5"],
    model_dir: str | Path = "model_data",
    # device: str = "cpu"
) -> pr.ObjectDetector:
    """
    Loads object detection models trained on VisDrone dataset.

    Parameters
    ----------
    name : Literal["centernet", "tph-yolov5"]
        The object detector to load.
    model_dir : str | Path (default="model_data")
        Directory to save/load model checkpoints.

    Returns
    -------
    ObjectDetector
        The object detector, which conforms to MAITE protocols.
    """

    if name == "centernet":
        # download weights if necessary
        model_file = Path(model_dir) / "centernet-resnet50.pth"
        provider = "Kitware: github.com/SMQTK-Detection"

        if not model_file.is_file():
            print(f"Downloading CenterNet model checkpoint to: {model_file}")
            urllib.request.urlretrieve( # type: ignore
                "https://data.kitware.com/api/v1/item/623259f64acac99f426f21db/download",
                model_file
            )

        centernet_detector = CenterNetVisdrone(
            arch='resnet50',
            model_file=str(model_file),
            max_dets=500,
            use_cuda=False,
            batch_size=1,
            num_workers=1,
        )

        metadata = SMQTKModelMetadata(
            provider=provider,
            model_name=name,
            task="object-detection",
            model_info={
                "model_file": str(model_file.relative_to(model_file.parents[3])),
                "arch": "resnet50",
                "max_dets": 500,
            }
        )

        return SMQTKObjectDetector(centernet_detector, VISDRONE_CLASSES, map_output_labels=True, metadata=metadata)
