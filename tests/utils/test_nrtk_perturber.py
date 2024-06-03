import numpy as np
import pytest
import json
import kwcoco
from pathlib import Path
from typing import List

from nrtk.impls.perturb_image.generic.cv2.blur import AverageBlurPerturber
from nrtk.impls.perturb_image.pybsm.scenario import PybsmScenario
from nrtk.impls.perturb_image.pybsm.sensor import PybsmSensor
from nrtk.impls.perturb_image_factory.generic.step import StepPerturbImageFactory
from nrtk.impls.perturb_image_factory.pybsm import CustomPybsmPerturbImageFactory
from nrtk.interfaces.perturb_image_factory import PerturbImageFactory
from nrtk_cdao.interop.object_detection.dataset import COCOJATICObjectDetectionDataset
from nrtk_cdao.utils.nrtk_perturber import nrtk_perturber

from tests import DATASET_FOLDER


def _load_dataset(dataset_path: str, load_metadata: bool = True) -> COCOJATICObjectDetectionDataset:
    coco_file = Path(dataset_path) / "annotations.json"
    kwcoco_dataset = kwcoco.CocoDataset(coco_file)

    if load_metadata:
        metadata_file = Path(dataset_path) / "image_metadata.json"
        with open(metadata_file) as f:
            metadata = json.load(f)
    else:
        metadata = [dict()] * len(kwcoco_dataset.imgs)

    # Initialize dataset object
    dataset = COCOJATICObjectDetectionDataset(
        root=str(DATASET_FOLDER),
        kwcoco_dataset=kwcoco_dataset,
        image_metadata=metadata
    )

    return dataset


class TestNRTKPerturber:
    """
    These tests make use of the `tmpdir` fixture from `pytest`. Find more
    information here: https://docs.pytest.org/en/6.2.x/tmpdir.html
    """
    pybsm_factory = CustomPybsmPerturbImageFactory(
        sensor=PybsmSensor(
            name="L32511x", D=0.004, f=0.014285714285714287, px=0.00002,
            optTransWavelengths=np.array([3.8e-7, 7.0e-7]), eta=0.4, intTime=0.3, readNoise=25.0,
            maxN=96000., bitdepth=11.9, maxWellFill=0.005, dax=0.0001, day=0.0001,
            qewavelengths=np.array([3.0e-7, 4.0e-7, 5.0e-7, 6.0e-7, 7.0e-7, 8.0e-7, 9.0e-7, 1.0e-6, 1.1e-6]),
            qe=np.array([0.05, 0.6, 0.75, 0.85, 0.85, 0.75, 0.5, 0.2, 0])
        ),
        scenario=PybsmScenario(name="niceday", ihaze=2, altitude=75, groundRange=0),
        theta_keys=["f", "D"],
        thetas=[[0.014, 0.012], [0.001]]
    )

    @pytest.mark.parametrize("perturber_factory, img_dirs", [
        (
            StepPerturbImageFactory(perturber=AverageBlurPerturber, theta_key="ksize", start=1, stop=5, step=2),
            ["_ksize-1", "_ksize-3"]
        ),
        (pybsm_factory, ["_f-0.014_D-0.001", "_f-0.012_D-0.001"])
    ])
    def test_nrtk_perturber(self, perturber_factory: PerturbImageFactory, img_dirs: List[str]) -> None:
        """
        Test if the perturber returns the intended number of datasets.
        """
        dataset = _load_dataset(dataset_path=str(DATASET_FOLDER))

        augmented_datasets = nrtk_perturber(
            maite_dataset=dataset,
            perturber_factory=perturber_factory
        )

        # image ids that belong to each perturber sweep combination
        img_paths = Path(DATASET_FOLDER) / "images"
        img_ids = [img_file.stem + img_file.suffix
                   for img_file in img_paths.iterdir()
                   if img_file.is_file()]

        for perturber_params, aug_dataset in augmented_datasets:
            assert perturber_params in list(img_dirs)
            assert len(aug_dataset) == len(img_ids)

    def test_missing_metadata(self) -> None:
        """
        Test that an appropriate error is raised if required metadata is missing.
        """
        dataset = _load_dataset(dataset_path=str(DATASET_FOLDER), load_metadata=False)

        with pytest.raises(ValueError, match="'img_gsd' must be present in image metadata for this perturber"):
            _ = nrtk_perturber(
                maite_dataset=dataset,
                perturber_factory=self.pybsm_factory
            )
