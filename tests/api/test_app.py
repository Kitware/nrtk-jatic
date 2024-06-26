import os
import unittest.mock as mock
from importlib.util import find_spec
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock

import numpy as np
import py  # type: ignore
import pytest
from fastapi.encoders import jsonable_encoder
from starlette.testclient import TestClient

from nrtk_jatic.api.app import app
from nrtk_jatic.api.schema import NrtkPerturbInputSchema
from nrtk_jatic.interop.object_detection.dataset import (
    JATICDetectionTarget,
    JATICObjectDetectionDataset,
)
from tests import BAD_NRTK_CONFIG, DATASET_FOLDER, LABEL_FILE, NRTK_PYBSM_CONFIG

deps = ["kwcoco"]
specs = [find_spec(dep) for dep in deps]
is_usable = all([spec is not None for spec in specs])

TEST_RETURN_VALUE = (
    [  # repeated test return value for 3 tests, saved to var to save space
        (
            "perturb1",
            JATICObjectDetectionDataset(
                imgs=[np.random.randint(0, 255, size=(3, 3, 3), dtype=np.uint8)] * 11,
                dets=[
                    JATICDetectionTarget(
                        boxes=np.random.rand(2, 4),
                        labels=np.random.rand(2),
                        scores=np.random.rand(2),
                    )
                ]
                * 11,
                metadata=[{}] * 11,
            ),
        )
    ]
)


@pytest.fixture()
def test_client() -> Generator:
    # Create a test client for the FastAPI application
    with TestClient(app) as client:
        yield client


@pytest.mark.skipif(not is_usable, reason="Extra 'nrtk-jatic[tools]' not installed.")
@mock.patch("nrtk_jatic.api.app.nrtk_perturber", return_value=TEST_RETURN_VALUE)
def test_handle_post_pybsm(
    patch: MagicMock, test_client: TestClient, tmpdir: py.path.local
) -> None:
    """Check for an appropriate response to a "good" request.
    """
    # Test data to be sent in the POST request
    test_data = NrtkPerturbInputSchema(
        id="0",
        name="Example",
        dataset_dir=str(DATASET_FOLDER),
        label_file=str(LABEL_FILE),
        output_dir=str(tmpdir),
        image_metadata=[{"gsd": gsd} for gsd in range(11)],
        config_file=str(NRTK_PYBSM_CONFIG),
    )

    # Send a POST request to the API endpoint
    response = test_client.post("/", json=jsonable_encoder(test_data))

    # Confirm mocked nrtk_perturber was called with the correct arguments
    kwargs = patch.call_args.kwargs
    assert len(kwargs["maite_dataset"]) == 11

    factory_config = kwargs["perturber_factory"].get_config()
    assert factory_config == {
        "theta_keys": ["f", "D", "px"],
        "sensor": {
            "type": "nrtk.impls.perturb_image.pybsm.sensor.PybsmSensor",
            "nrtk.impls.perturb_image.pybsm.sensor.PybsmSensor": {
                "name": "L32511x",
                "D": 0.004,
                "f": 0.014285714285714287,
                "px": 2e-05,
                "optTransWavelengths": [3.8e-07, 7e-07],
                "opticsTransmission": [1.0, 1.0],
                "eta": 0.4,
                "wx": 2e-05,
                "wy": 2e-05,
                "intTime": 0.03,
                "darkCurrent": 0.0,
                "readNoise": 25.0,
                "maxN": 96000.0,
                "bitdepth": 11.9,
                "maxWellFill": 0.005,
                "sx": 0.0,
                "sy": 0.0,
                "dax": 0.0001,
                "day": 0.0001,
                "qewavelengths": [
                    3e-07,
                    4e-07,
                    5e-07,
                    6e-07,
                    7e-07,
                    8e-07,
                    9e-07,
                    1e-06,
                    1.1e-06,
                ],
                "qe": [0.05, 0.6, 0.75, 0.85, 0.85, 0.75, 0.5, 0.2, 0.0],
            },
        },
        "scenario": {
            "type": "nrtk.impls.perturb_image.pybsm.scenario.PybsmScenario",
            "nrtk.impls.perturb_image.pybsm.scenario.PybsmScenario": {
                "name": "niceday",
                "ihaze": 2,
                "altitude": 75,
                "groundRange": 0,
                "aircraftSpeed": 0.0,
                "targetReflectance": 0.15,
                "targetTemperature": 295.0,
                "backgroundReflectance": 0.07,
                "backgroundTemperature": 293.0,
                "haWindspeed": 21.0,
                "cn2at1m": 0,
            },
        },
        "thetas": [[0.014, 0.012], [0.001, 0.003], [2e-05]],
    }

    # Check if the response status code is 200 OK
    assert response.status_code == 200

    # Check if the response data contains the expected message
    assert response.json()["message"] == "Data received successfully"

    # Check if the response data contains the processed data
    base_path = Path(tmpdir) / "perturb1"
    image_dir = base_path / "images"
    label_file = base_path / "annotations.json"
    metadata_file = base_path / "image_metadata.json"
    assert response.json()["datasets"] == [
        {
            "root_dir": str(base_path),
            "label_file": label_file.name,
            "metadata_file": metadata_file.name,
        }
    ]
    assert image_dir.is_dir()
    assert label_file.is_file()
    assert metadata_file.is_file()
    # Check that the correct number of images are in the dir
    assert len(os.listdir(os.path.join(str(image_dir)))) == 11


@pytest.mark.skipif(not is_usable, reason="Extra 'nrtk-jatic[tools]' not installed.")
@mock.patch("nrtk_jatic.api.app.nrtk_perturber", return_value=TEST_RETURN_VALUE)
def test_bad_gsd_post(
    patch: MagicMock, test_client: TestClient, tmpdir: py.path.local
) -> None:
    """Test that an error response is appropriately propagated to the user.
    """
    test_data = NrtkPerturbInputSchema(
        id="0",
        name="Example",
        dataset_dir=str(DATASET_FOLDER),
        label_file=str(LABEL_FILE),
        output_dir=str(tmpdir),
        image_metadata=[{"gsd": gsd} for gsd in range(3)],  # incorrect number of gsds
        config_file=str(NRTK_PYBSM_CONFIG),
    )

    # Send a POST request to the API endpoint
    response = test_client.post("/", json=jsonable_encoder(test_data))

    # Check response status code
    assert response.status_code == 400

    # Check that we got the correct error message
    assert (
        response.json()["detail"]
        == "Image metadata length mismatch, metadata needed for every image"
    )


@mock.patch("nrtk_jatic.api.app.nrtk_perturber", return_value=TEST_RETURN_VALUE)
def test_no_config_post(
    patch: MagicMock, test_client: TestClient, tmpdir: py.path.local
) -> None:
    """Test that an error response is appropriately propagated to the user.
    """
    test_data = NrtkPerturbInputSchema(
        id="0",
        name="Example",
        dataset_dir=str(DATASET_FOLDER),
        label_file=str(LABEL_FILE),
        output_dir=str(tmpdir),
        image_metadata=[{"gsd": gsd} for gsd in range(11)],
        config_file="/bad/path/",
    )

    # Send a POST request to the API endpoint
    response = test_client.post("/", json=jsonable_encoder(test_data))

    # Check response status code
    assert response.status_code == 400

    # Check that we got the correct error message
    assert response.json()["detail"] == "Config file at /bad/path/ was not found"


@mock.patch(
    "nrtk_jatic.api.app.nrtk_perturber",
    return_value=[
        (
            "perturb1",
            JATICObjectDetectionDataset(
                imgs=[np.random.randint(0, 255, size=(3, 3, 3), dtype=np.uint8)] * 11,
                dets=[
                    JATICDetectionTarget(
                        boxes=np.random.rand(2, 4),
                        labels=np.random.rand(2),
                        scores=np.random.rand(2),
                    )
                ]
                * 11,
                metadata=[{}] * 11,
            ),
        )
    ],
)
def test_bad_config_post(
    patch: MagicMock, test_client: TestClient, tmpdir: py.path.local
) -> None:
    """Test that an error response is appropriately propagated to the user.
    """
    test_data = NrtkPerturbInputSchema(
        id="0",
        name="Example",
        dataset_dir=str(DATASET_FOLDER),
        label_file=str(LABEL_FILE),
        output_dir=str(tmpdir),
        image_metadata=[{"gsd": gsd} for gsd in range(11)],
        config_file=str(BAD_NRTK_CONFIG),
    )

    # Send a POST request to the API endpoint
    response = test_client.post("/", json=jsonable_encoder(test_data))

    # Check response status code
    assert response.status_code == 400

    # Check that we got the correct error message
    assert (
        response.json()["detail"]
        == "Configuration dictionary given does not have an implementation type specification."
    )


@mock.patch("nrtk_jatic.api.app.is_usable", False)
def test_missing_deps(test_client: TestClient, tmpdir: py.path.local) -> None:
    """Test that an exception is raised when required dependencies are not installed.
    """
    test_data = NrtkPerturbInputSchema(
        id="0",
        name="Example",
        dataset_dir=str(DATASET_FOLDER),
        label_file=str(LABEL_FILE),
        output_dir=str(tmpdir),
        image_metadata=[{"gsd": gsd} for gsd in range(11)],
        config_file=str(NRTK_PYBSM_CONFIG),
    )

    # Send a POST request to the API endpoint
    response = test_client.post("/", json=jsonable_encoder(test_data))

    # Check response status code
    assert response.status_code == 400

    # Check that we got the correct error message
    assert (
        response.json()["detail"]
        == "This tool requires additional dependencies, please install `nrtk-jatic[tools]`"
    )
