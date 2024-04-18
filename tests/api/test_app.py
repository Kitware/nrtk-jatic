import pytest
from starlette.testclient import TestClient
from nrtk_cdao.api.app import app
from nrtk_cdao.api.schema import Schema
from typing import Generator
from fastapi.encoders import jsonable_encoder

from tests import DATASET_FOLDER, LABEL_FILE


@pytest.fixture
def test_client() -> Generator:
    # Create a test client for the FastAPI application
    with TestClient(app) as client:
        yield client


def test_handle_post(test_client: TestClient) -> None:
    # Test data to be sent in the POST request
    test_data = Schema(
        id="0",
        name="Example",
        dataset_dir=str(DATASET_FOLDER),
        label_file=str(LABEL_FILE),
        gsds=list(range(11)),
        theta_keys=["f", "D", "px"],
        thetas=[[0.014, 0.012], [0.001, 0.003], [0.00002]],
    )

    # Send a POST request to the API endpoint
    response = test_client.post("/", json=jsonable_encoder(test_data))

    # Check if the response status code is 200 OK
    assert response.status_code == 200

    # Check if the response data contains the expected message
    assert response.json()["message"] == "Data received successfully"

    # Check if the response data contains the processed data
    assert response.json()["processed_data"] == {
        "dataset_len": 11,
        "factory_config": {
            "theta_keys": ["f", "D", "px"],
            "sensor": {
                "name": "Example",
                "D": 0.005,
                "f": 0.014,
                "px": 7.4e-06,
                "optTransWavelengths": [3.8e-07, 7e-07],
                "opticsTransmission": [1.0, 1.0],
                "eta": None,
                "wx": 7.4e-06,
                "wy": 7.4e-06,
                "intTime": 1.0,
                "darkCurrent": None,
                "readNoise": 0.0,
                "maxN": None,
                "bitdepth": None,
                "maxWellFill": None,
                "sx": None,
                "sy": None,
                "dax": None,
                "day": None,
                "qewavelengths": [3.8e-07, 7e-07],
                "qe": [1.0, 1.0],
            },
            "scenario": {
                "name": "Example",
                "ihaze": 2,
                "altitude": 75,
                "groundRange": 0,
                "aircraftSpeed": None,
                "targetReflectance": None,
                "targetTemperature": None,
                "backgroundReflectance": None,
                "backgroundTemperature": None,
                "haWindspeed": None,
                "cn2at1m": None,
            },
            "thetas": [[0.014, 0.012], [0.001, 0.003], [2e-05]],
            "sets": [[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]],
        },
    }


if __name__ == "__main__":
    pytest.main()
