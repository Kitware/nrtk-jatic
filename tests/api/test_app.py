import pytest
from starlette.testclient import TestClient
from nrtk_cdao.api.app import app
from nrtk_cdao.api.schema import Schema
from typing import Generator
from fastapi.encoders import jsonable_encoder

import os
from tests import DATA_DIR

dataset_folder = os.path.join(DATA_DIR, 'VisDrone2019-DET-test-dev-TINY')

@pytest.fixture
def test_client() -> Generator:
    # Create a test client for the FastAPI application
    with TestClient(app) as client:
        yield client


def test_handle_post(test_client: TestClient) -> None:
    # Test data to be sent in the POST request
    test_data = Schema(id='0', name="Example", dataset_dir=dataset_folder,
        gsd=0.04, theta_keys=['f', 'd', 'px'], thetas=[[0.014, 0.012], [0.001, 0.003], [0.00002]])

    # Send a POST request to the API endpoint
    response = test_client.post('/', json=jsonable_encoder(test_data))

    # Check if the response status code is 200 OK
    assert response.status_code == 200

    # Check if the response data contains the expected message
    assert response.json()['message'] == 'Data received successfully'

    # Check if the response data contains the processed data
    assert response.json()['processed_data'] == {'dataset_len': 11, 'factory_thetas': [[0.014, 0.012], [0.001, 0.003], [0.00002]]}


if __name__ == "__main__":
    pytest.main()
