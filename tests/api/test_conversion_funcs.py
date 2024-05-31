import pytest
import numpy as np
import os
from typing import Dict, Any

from smqtk_core.configuration import to_config_dict

from nrtk.impls.perturb_image.pybsm.scenario import PybsmScenario
from nrtk.impls.perturb_image.pybsm.sensor import PybsmSensor

from nrtk_cdao.api.converters import build_factory, load_COCOJATIC_dataset
from nrtk_cdao.api.schema import NrtkPerturbInputSchema

from tests import DATASET_FOLDER, LABEL_FILE, NRTK_PYBSM_CONFIG, BAD_NRTK_CONFIG, EMPTY_NRTK_CONFIG


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
                    "image_metadata": [],  # Not used in this test
                    "config_file": str(NRTK_PYBSM_CONFIG),
                },
                {
                    "theta_keys": ["f", "D", "px"],
                    "thetas": [[0.014, 0.012], [0.001, 0.003], [0.00002]],
                    "sensor": to_config_dict(
                        PybsmSensor(
                            name="L32511x",
                            D=0.004,
                            f=0.014285714285714287,
                            px=0.00002,
                            optTransWavelengths=np.asarray([3.8e-7, 7.0e-7]),
                            eta=0.4,
                            intTime=0.03,
                            readNoise=25.0,
                            maxN=96000.0,
                            bitdepth=11.9,
                            maxWellFill=0.005,
                            dax=0.0001,
                            day=0.0001,
                            qewavelengths=np.asarray(
                                [3.0e-7, 4.0e-7, 5.0e-7, 6.0e-7, 7.0e-7, 8.0e-7, 9.0e-7, 1.0e-6, 1.1e-6]
                            ),
                            qe=np.asarray([0.05, 0.6, 0.75, 0.85, 0.85, 0.75, 0.5, 0.2, 0]),
                        )
                    ),
                    "scenario": to_config_dict(
                        PybsmScenario(name="niceday", ihaze=2, altitude=75, groundRange=0, cn2at1m=0)
                    ),
                },
            ),
        ],
    )
    def test_build_factory(self, data: Dict[str, Any], expected: Dict[str, Any]) -> None:
        """
        Test if _build_pybsm_factory returns the expected factory.
        """
        schema = NrtkPerturbInputSchema.model_validate(data)
        factory = build_factory(schema)
        np.testing.assert_equal(factory.get_config(), expected)

    @pytest.mark.parametrize(
        "data",
        [
            (
                {
                    "id": "0",
                    "name": "Example",
                    "dataset_dir": "",  # Not used in this test
                    "label_file": "",  # Not used in this test
                    "output_dir": "",  # Not used in this test
                    "image_metadata": [],  # Not used in this test
                    "config_file": "",
                }
            ),
        ],
    )
    def test_build_factory_no_config(self, data: Dict[str, Any]) -> None:
        """
        Test if build_factory throws .
        """
        schema = NrtkPerturbInputSchema.model_validate(data)
        with pytest.raises(FileNotFoundError):
            build_factory(schema)

    @pytest.mark.parametrize(
        "data",
        [
            (
                {
                    "id": "0",
                    "name": "Example",
                    "dataset_dir": "",  # Not used in this test
                    "label_file": "",  # Not used in this test
                    "output_dir": "",  # Not used in this test
                    "image_metadata": [],  # Not used in this test
                    "config_file": str(BAD_NRTK_CONFIG),
                }
            ),
            (
                {
                    "id": "0",
                    "name": "Example",
                    "dataset_dir": "",  # Not used in this test
                    "label_file": "",  # Not used in this test
                    "output_dir": "",  # Not used in this test
                    "image_metadata": [],  # Not used in this test
                    "config_file": str(EMPTY_NRTK_CONFIG),
                }
            )
        ],
    )
    def test_build_factory_bad_config(self, data: Dict[str, Any]) -> None:
        """
        Test if build_factory throws .
        """
        schema = NrtkPerturbInputSchema.model_validate(data)
        with pytest.raises(ValueError):
            build_factory(schema)

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
                    "image_metadata": [{"gsd": gsd} for gsd in range(11)],
                    "config_file": "",  # Not used in this test
                }
            )
        ],
    )
    def test_load_COCOJATIC_dataset(self, data: Dict[str, Any]) -> None:
        """
        Test if load_COCOJATIC_dataset returns the expected dataset.
        """
        schema = NrtkPerturbInputSchema.model_validate(data)
        dataset = load_COCOJATIC_dataset(schema)
        # Check all images metadata for gsd
        for i in range(len(dataset)):
            assert dataset[i][2]["gsd"] == data["image_metadata"][i]["gsd"]
        # Check number of image matches
        assert len(dataset) == len(os.listdir(os.path.join(data["dataset_dir"], "images")))
