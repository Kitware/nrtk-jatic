import pytest
from nrtk_cdao.api.converters import build_pybsm_factory, load_COCOJAITIC_dataset
from typing import Dict, Any
import numpy as np
from nrtk.impls.perturb_image.pybsm.scenario import PybsmScenario
from nrtk.impls.perturb_image.pybsm.sensor import PybsmSensor
import os
from tests import DATASET_FOLDER, LABEL_FILE
from nrtk_cdao.api.schema import NrtkPybsmPerturbInputSchema


class TestAPIConversionFunctions:
    @pytest.mark.parametrize(
        "data, expected",
        [
            (
                {
                    "id": "0",
                    "name": "Example",
                    "dataset_dir": "",  # Not used in this test
                    "label_file": "",  # Not used in this test
                    "output_dir": "",  # Not used in this test
                    "gsds": [],  # Not used in this test
                    "theta_keys": ["f", "D"],
                    "thetas": [[0.014, 0.012], [0.001, 0.003]],
                },
                {
                    "theta_keys": ["f", "D"],
                    "thetas": [[0.014, 0.012], [0.001, 0.003]],
                    "sets": [[0, 0], [0, 1], [1, 0], [1, 1]],
                    "sensor": PybsmSensor(
                        name="Example", D=0.005, f=0.014, px=0.0000074, optTransWavelengths=np.asarray([3.8e-7, 7.0e-7])
                    ).get_config(),
                    "scenario": PybsmScenario(name="Example", ihaze=2, altitude=75, groundRange=0).get_config(),
                },
            ),
            (
                {
                    "id": "0",
                    "name": "Example",
                    "dataset_dir": "",  # Not used in this test
                    "label_file": "",  # Not used in this test
                    "output_dir": "",  # Not used in this test
                    "gsds": [],  # Not used in this test
                    "theta_keys": ["f"],
                    "thetas": [[0.014, 0.012]],
                },
                {
                    "theta_keys": ["f"],
                    "thetas": [[0.014, 0.012]],
                    "sets": [[0], [1]],
                    "sensor": PybsmSensor(
                        name="Example", D=0.005, f=0.014, px=0.0000074, optTransWavelengths=np.asarray([3.8e-7, 7.0e-7])
                    ).get_config(),
                    "scenario": PybsmScenario(name="Example", ihaze=2, altitude=75, groundRange=0).get_config(),
                },
            ),
            (
                {
                    "id": "0",
                    "name": "Example",
                    "dataset_dir": "",  # Not used in this test
                    "label_file": "",  # Not used in this test
                    "output_dir": "",  # Not used in this test
                    "gsds": [],  # Not used in this test
                    "aircraftSpeed": 100,
                    "wx": 1.1,
                    "theta_keys": ["f"],
                    "thetas": [[0.014, 0.012]],
                },
                {
                    "theta_keys": ["f"],
                    "thetas": [[0.014, 0.012]],
                    "sets": [[0], [1]],
                    "sensor": PybsmSensor(
                        name="Example",
                        D=0.005,
                        f=0.014,
                        px=0.0000074,
                        optTransWavelengths=np.asarray([3.8e-7, 7.0e-7]),
                        wx=1.1,
                    ).get_config(),
                    "scenario": PybsmScenario(
                        name="Example", ihaze=2, altitude=75, groundRange=0, aircraftSpeed=100
                    ).get_config(),
                },
            ),
        ],
    )
    def test_build_pybsm_factory(self, data: Dict[str, Any], expected: Dict[str, Any]) -> None:
        """
        Test if _build_pybsm_factory returns the expected factory.
        """
        schema = NrtkPybsmPerturbInputSchema.model_validate(data)
        factory = build_pybsm_factory(schema)
        np.testing.assert_equal(factory.get_config(), expected)

    @pytest.mark.parametrize(
        "data",
        [
            (
                {
                    "id": "0",
                    "name": "Example",
                    "dataset_dir": str(DATASET_FOLDER),
                    "label_file": str(LABEL_FILE),
                    "output_dir": "",  # Not used in this test
                    "gsds": list(range(11)),
                    "theta_keys": [],  # Not used in this test
                    "thetas": [],  # Not used in this test
                }
            )
        ],
    )
    def test_load_COCOJAITIC_dataset(self, data: Dict[str, Any]) -> None:
        """
        Test if _load_COCOJAITIC_dataset returns the expected dataset.
        """
        schema = NrtkPybsmPerturbInputSchema.model_validate(data)
        dataset = load_COCOJAITIC_dataset(schema)
        # Check all images metadata for gsd
        for i in range(len(dataset)):
            assert dataset[i][2]["img_gsd"] == data["gsds"][i]
        # Check number of image matches
        assert len(dataset) == len(os.listdir(os.path.join(data["dataset_dir"], "images")))
