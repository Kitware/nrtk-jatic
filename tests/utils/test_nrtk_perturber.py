import os
import yaml  # type: ignore
from typing import List, Tuple, Dict
from pathlib import Path
import itertools

import numpy as np
import kwcoco
from tests import DATA_DIR

from nrtk_cdao.utils.nrtk_perturber import nrtk_perturber
from nrtk_cdao.interop.dataset import COCOJATICObjectDetectionDataset
from nrtk.impls.perturb_image.pybsm.scenario import PybsmScenario  # type: ignore
from nrtk.impls.perturb_image.pybsm.sensor import PybsmSensor  # type: ignore
from nrtk.impls.perturb_image_factory.pybsm import CustomPybsmPerturbImageFactory  # type: ignore
from nrtk.interfaces.perturb_image_factory import PerturbImageFactory  # type: ignore

dataset_folder = os.path.join(DATA_DIR, 'VisDrone2019-DET-test-dev-TINY')
config_file = os.path.join(DATA_DIR, 'pybsm_config.yaml')
perturb_params_file = os.path.join(DATA_DIR, 'pybsm_perturb_params.yaml')


def _load_dataset(dataset_path: str, img_gsd: float) -> COCOJATICObjectDetectionDataset:
    annotation_dir = Path(dataset_path) / "annotations"
    coco_file = list(annotation_dir.glob("*.json"))
    kwcoco_dataset = kwcoco.CocoDataset(coco_file[0])

    # Initialize dataset object
    dataset = COCOJATICObjectDetectionDataset(
        root=dataset_folder,
        kwcoco_dataset=kwcoco_dataset,
        image_metadata=[{"img_gsd": img_gsd} for _ in range(len(kwcoco_dataset.imgs))]
    )

    return dataset


def _load_config(config_file: str, perturb_params_file: str) -> Tuple[Dict, Dict]:
    with open(config_file) as cfg_file:
        config = yaml.safe_load(cfg_file)

    with open(perturb_params_file) as cfg_file:
        perturb_params = yaml.safe_load(cfg_file)

    for key, value in config["sensor"].items():
        if isinstance(value, List):
            config["sensor"][key] = np.asarray(value)

    return config, perturb_params


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
        config, perturb_params = _load_config(config_file, perturb_params_file)

        image_gsd = config["gsd"]
        sensor = PybsmSensor(**config["sensor"])
        scenario = PybsmScenario(**config["scenario"])

        dataset = _load_dataset(dataset_path=dataset_folder,
                                img_gsd=image_gsd)

        perturb_factory_keys = list(perturb_params.keys())
        thetas = [perturb_params[key]
                  for key in perturb_factory_keys]

        perturber_factory = CustomPybsmPerturbImageFactory(
            sensor=sensor,
            scenario=scenario,
            theta_keys=perturb_factory_keys,
            thetas=thetas
        )
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
