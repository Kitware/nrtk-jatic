import numpy as np
import os
import py  # type: ignore
import pytest
import unittest.mock as mock
from fastapi.encoders import jsonable_encoder
from pathlib import Path
from starlette.testclient import TestClient
from typing import Generator
from unittest.mock import MagicMock

from nrtk_cdao.api.app import app
from nrtk_cdao.api.schema import NrtkPybsmPerturbInputSchema
from nrtk_cdao.interop.dataset import JATICObjectDetectionDataset, JATICDetectionTarget

from tests import DATASET_FOLDER, LABEL_FILE


@pytest.fixture
def test_client() -> Generator:
    # Create a test client for the FastAPI application
    with TestClient(app) as client:
        yield client


@mock.patch(
    'nrtk_cdao.api.app.nrtk_perturber',
    return_value=[
        (
            "perturb1",
            JATICObjectDetectionDataset(
                imgs=[np.random.randint(0, 255, size=(3, 3, 3), dtype=np.uint8)] * 11,
                dets=[JATICDetectionTarget(
                    boxes=np.random.rand(2, 4),
                    labels=np.random.rand(2),
                    scores=np.random.rand(2)
                )] * 11,
                metadata=[{}] * 11
            )
        )
    ]
)
def test_handle_post(patch: MagicMock, test_client: TestClient, tmpdir: py.path.local) -> None:
    """
    Check for an appropriate response to a "good" request.
    """
    # Test data to be sent in the POST request
    test_data = NrtkPybsmPerturbInputSchema(
        id="0",
        name="Example",
        dataset_dir=str(DATASET_FOLDER),
        label_file=str(LABEL_FILE),
        output_dir=str(tmpdir),
        gsds=list(range(11)),
        theta_keys=["f", "D", "px"],
        thetas=[[0.014, 0.012], [0.001, 0.003], [0.00002]],
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
            "nrtk.impls.perturb_image.pybsm.sensor.PybsmSensor": {
                "name": "Example",
                "D": 0.005,
                "f": 0.014,
                "px": 7.4e-06,
                "optTransWavelengths": [3.8e-07, 7e-07],
                "opticsTransmission": [1.0, 1.0],
                "eta": 0.0,
                "wx": 7.4e-06,
                "wy": 7.4e-06,
                "intTime": 1.0,
                "darkCurrent": 0.0,
                "readNoise": 0.0,
                "maxN": 100000000.0,
                "bitdepth": 100.0,
                "maxWellFill": 1.0,
                "sx": 0.0,
                "sy": 0.0,
                "dax": 0.0,
                "day": 0.0,
                "qewavelengths": [3.8e-07, 7e-07],
                "qe": [1.0, 1.0],
            },
            "type": "nrtk.impls.perturb_image.pybsm.sensor.PybsmSensor",
        },
        "scenario": {
            "nrtk.impls.perturb_image.pybsm.scenario.PybsmScenario": {
                "name": "Example",
                "ihaze": 2,
                "altitude": 75,
                "groundRange": 0,
                "aircraftSpeed": 0.0,
                "targetReflectance": 0.15,
                "targetTemperature": 295.0,
                "backgroundReflectance": 0.07,
                "backgroundTemperature": 293.0,
                "haWindspeed": 21.0,
                "cn2at1m": 1.7e-14,
            },
            "type": "nrtk.impls.perturb_image.pybsm.scenario.PybsmScenario",
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
    assert response.json()["datasets"] == [{
        "root_dir": str(base_path),
        "label_file": label_file.name,
        "metadata_file": metadata_file.name
    }]
    assert image_dir.is_dir()
    assert label_file.is_file()
    assert metadata_file.is_file()
    # Check that the correct number of images are in the dir
    assert len(os.listdir(os.path.join(str(image_dir)))) == 11


@mock.patch(
    'nrtk_cdao.api.app.nrtk_perturber',
    return_value=[
        (
            "perturb1",
            JATICObjectDetectionDataset(
                imgs=[np.random.randint(0, 255, size=(3, 3, 3), dtype=np.uint8)] * 11,
                dets=[JATICDetectionTarget(
                    boxes=np.random.rand(2, 4),
                    labels=np.random.rand(2),
                    scores=np.random.rand(2)
                )] * 11,
                metadata=[{}] * 11
            )
        )
    ]
)
def test_bad_post(patch: MagicMock, test_client: TestClient, tmpdir: py.path.local) -> None:
    """
    Test that an error response is appropriately propagated to the user.
    """
    test_data = NrtkPybsmPerturbInputSchema(
        id="0",
        name="Example",
        dataset_dir=str(DATASET_FOLDER),
        label_file=str(LABEL_FILE),
        output_dir=str(tmpdir),
        gsds=list(range(3)),  # incorrect number of gsds
        theta_keys=["f", "D", "px"],
        thetas=[[0.014, 0.012], [0.001, 0.003], [0.00002]],
    )

    # Send a POST request to the API endpoint
    response = test_client.post("/", json=jsonable_encoder(test_data))

    # Check response status code
    assert response.status_code == 400

    # Check that we got the correct error message
    assert response.json()["detail"] == "Image metadata length mismatch, metadata needed for every image"
