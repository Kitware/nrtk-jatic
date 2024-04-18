import pytest
from nrtk_cdao.api.app import _build_pybsm_factory, _load_COCOJAITIC_dataset
from typing import Dict, Any
import numpy as np
from nrtk.impls.perturb_image.pybsm.scenario import PybsmScenario  # type: ignore
from nrtk.impls.perturb_image.pybsm.sensor import PybsmSensor  # type: ignore
import os
from tests import DATASET_FOLDER, LABEL_FILE


class TestAPIConversionFunctions:

    @pytest.mark.parametrize(
        "data, expected",
        [
            (
                {
                    "name": "Example",
                    "ihaze": 2,
                    "altitude": 75,
                    "groundRange": 0,
                    "D": 0.005,
                    "f": 0.014,
                    "px": 0.0000074,
                    "optTransWavelengths": [3.8e-7, 7.0e-7],
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
                    "name": "Example",
                    "ihaze": 2,
                    "altitude": 75,
                    "groundRange": 0,
                    "D": 0.005,
                    "f": 0.014,
                    "px": 0.0000074,
                    "optTransWavelengths": [3.8e-7, 7.0e-7],
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
                    "name": "Example",
                    "ihaze": 2,
                    "altitude": 75,
                    "groundRange": 0,
                    "aircraftSpeed": 100,
                    "D": 0.005,
                    "f": 0.014,
                    "px": 0.0000074,
                    "optTransWavelengths": [3.8e-7, 7.0e-7],
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

        factory = _build_pybsm_factory(data)
        np.testing.assert_equal(factory.get_config(), expected)

    @pytest.mark.parametrize(
        "data",
        [({"dataset_dir": str(DATASET_FOLDER), "label_file": str(LABEL_FILE), "gsd": 0.0})],
    )
    def test_load_COCOJAITIC_dataset(self, data: Dict[str, Any]) -> None:
        """
        Test if _load_COCOJAITIC_dataset returns the expected dataset.
        """

        dataset = _load_COCOJAITIC_dataset(data)
        # Check first images metadata for gsd
        assert dataset[0][2]["img_gsd"] == data["gsd"]
        # Check number of image matches
        assert len(dataset) == len(os.listdir(os.path.join(data["dataset_dir"], "images")))
