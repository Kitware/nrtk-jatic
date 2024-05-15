import itertools
import json
import kwcoco
import os
from pathlib import Path
from typing import List

from smqtk_core.configuration import from_config_dict

from nrtk.interfaces.perturb_image_factory import PerturbImageFactory
from nrtk_cdao.interop.dataset import COCOJATICObjectDetectionDataset
from nrtk_cdao.utils.nrtk_perturber import nrtk_perturber

from tests import DATA_DIR


dataset_folder = os.path.join(DATA_DIR, 'VisDrone2019-DET-test-dev-TINY')
config_file = os.path.join(DATA_DIR, 'nrtk_config.json')


def _load_dataset(dataset_path: str) -> COCOJATICObjectDetectionDataset:
    coco_file = Path(dataset_path) / "annotations.json"
    kwcoco_dataset = kwcoco.CocoDataset(coco_file)

    metadata_file = Path(dataset_path) / "image_metadata.json"
    with open(metadata_file) as f:
        metadata = json.load(f)

    # Initialize dataset object
    dataset = COCOJATICObjectDetectionDataset(
        root=dataset_folder,
        kwcoco_dataset=kwcoco_dataset,
        image_metadata=metadata
    )

    return dataset


def _get_perturber_param_combinations(perturber_factory: PerturbImageFactory
                                      ) -> List[str]:
    perturber_factory_config = perturber_factory.get_config()
    perturb_factory_keys = perturber_factory_config["theta_keys"]
    thetas = perturber_factory_config["thetas"]
    perturber_combinations = [dict(zip(perturb_factory_keys, v))
                              for v in itertools.product(*thetas)]
    output_perturb_params = [''.join("_{!s}-{!s}".format(k, v)
                             for k, v in perturber_combo.items())
                             for perturber_combo in perturber_combinations]
    return output_perturb_params


class TestNRTKPybsmPerturber:
    """
    These tests make use of the `tmpdir` fixture from `pytest`. Find more
    information here: https://docs.pytest.org/en/6.2.x/tmpdir.html
    """

    def test_nrtk_pybsm_perturber(self) -> None:
        """
        Test if the perturber returns the intended number of datasets.
        """
        with open(config_file) as f:
            config = json.load(f)
        perturber_factory = from_config_dict(config["PerturberFactory"], PerturbImageFactory.get_impls())

        dataset = _load_dataset(dataset_path=dataset_folder)

        augmented_datasets = nrtk_perturber(
            maite_dataset=dataset,
            perturber_factory=perturber_factory
        )

        # expected created directories for the perturber sweep combinations
        img_dirs = _get_perturber_param_combinations(perturber_factory)

        # image ids that belong to each perturber sweep combination
        img_paths = Path(dataset_folder) / "images"
        img_ids = [img_file.stem + img_file.suffix
                   for img_file in img_paths.iterdir()
                   if img_file.is_file()]

        for perturber_params, aug_dataset in augmented_datasets:
            assert perturber_params in list(img_dirs)
            assert len(aug_dataset) == len(img_ids)
