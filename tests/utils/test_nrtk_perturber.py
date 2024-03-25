import os
import py  # type: ignore
import yaml  # type: ignore
import pytest
import itertools
from typing import List
from pathlib import Path
from importlib import import_module
from importlib.util import find_spec

import numpy as np

from tests import DATA_DIR

from nrtk_cdao.utils.nrtk_perturber import nrtk_perturber
from nrtk_cdao.interop.dataset import JATICObjectDetectionDataset
from nrtk.impls.perturb_image.pybsm.scenario import PybsmScenario  # type: ignore
from nrtk.impls.perturb_image.pybsm.sensor import PybsmSensor  # type: ignore
from nrtk.impls.perturb_image_factory.pybsm import CustomPybsmPerturbImageFactory  # type: ignore


deps = ['kwcoco']
specs = [find_spec(dep) for dep in deps]
is_usable = all([spec is not None for spec in specs])

dataset_folder = os.path.join(DATA_DIR, 'VisDrone2019-DET-test-dev-TINY')
config_file = os.path.join(DATA_DIR, 'pybsm_config.yaml')
perturb_params_file = os.path.join(DATA_DIR, 'pybsm_perturb_params.yaml')


@pytest.mark.skipif(not is_usable, reason="Extra 'nrtk-cdao[tools]' not installed.")
class TestNRTKPybsmPerturber:
    """
    These tests make use of the `tmpdir` fixture from `pytest`. Find more
    information here: https://docs.pytest.org/en/6.2.x/tmpdir.html
    """

    def test_nrtk_pybsm_perturber(self, tmpdir: py.path.local) -> None:

        output_dir = tmpdir.join('out')

        annotation_dir = Path(dataset_folder) / "annotations"

        kwcoco = import_module('kwcoco')
        coco_file = list(annotation_dir.glob("*.json"))
        kwcoco_dataset = kwcoco.CocoDataset(coco_file[0])

        # Initialize dataset object
        dataset = JATICObjectDetectionDataset(root=dataset_folder,
                                              kwcoco_dataset=kwcoco_dataset)
        with open(config_file) as cfg_file:
            config = yaml.safe_load(cfg_file)

        with open(perturb_params_file) as cfg_file:
            perturb_params = yaml.safe_load(cfg_file)

        for key, value in config["sensor"].items():
            if isinstance(value, List):
                config["sensor"][key] = np.asarray(value)

        image_gsd = config["gsd"]
        sensor = PybsmSensor(**config["sensor"])
        scenario = PybsmScenario(**config["scenario"])

        perturb_factory_keys = list(perturb_params.keys())
        thetas = [perturb_params[key]
                  for key in perturb_factory_keys]
        perturber_combinations = [dict(zip(perturb_factory_keys, v))
                                  for v in itertools.product(*thetas)]

        perturber_factory = CustomPybsmPerturbImageFactory(
            sensor=sensor,
            scenario=scenario,
            theta_keys=perturb_factory_keys,
            thetas=thetas
        )
        nrtk_perturber(
            maite_dataset=dataset,
            output_dir=str(output_dir),
            perturber_combinations=perturber_combinations,
            perturber_factory=perturber_factory,
            image_gsd=image_gsd
        )

        # expected created directories for the perturber sweep combinations
        img_dirs = [output_dir.join(d) for d in ["_f-0.012_D-0.001_px-2e-05",
                                                 "_f-0.012_D-0.003_px-2e-05",
                                                 "_f-0.014_D-0.001_px-2e-05",
                                                 "_f-0.014_D-0.003_px-2e-05"]]
        # image ids that belong to each perturber sweep combination
        img_ids = ['0000006_02616_d_0000007.jpg', '0000006_03636_d_0000009.jpg',
                   '0000006_00159_d_0000001.jpg', '0000006_01659_d_0000004.jpg',
                   '0000161_01584_d_0000158.jpg', '0000006_01111_d_0000003.jpg',
                   '0000006_04050_d_0000010.jpg', '0000006_04309_d_0000011.jpg',
                   '0000006_01275_d_0000004.jpg', '0000006_00611_d_0000002.jpg',
                   '0000006_02138_d_0000006.jpg']
        # image metadata json file
        img_metadata = output_dir.join("image_metadata.json")

        for op_dir in output_dir.listdir():
            if op_dir.check(dir=1):
                assert op_dir in img_dirs

        assert img_metadata.check(exists=1)

        for img_dir in img_dirs:
            assert len(img_dir.listdir()) > 0
            img_filenames = [img.basename for img in img_dir.listdir()]
            assert sorted(img_filenames) == sorted(img_ids)
