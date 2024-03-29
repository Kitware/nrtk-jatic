import pytest
from starlette.testclient import TestClient
from nrtk_cdao.api.app import app
from typing import Generator


@pytest.fixture
def test_client() -> Generator:
    # Create a test client for the FastAPI application
    with TestClient(app) as client:
        yield client


def test_handle_post(test_client: TestClient) -> None:
    # Test data to be sent in the POST request
    test_data = {"key": "value"}

    # Send a POST request to the API endpoint
    response = test_client.post('/', json=test_data)

    # Check if the response status code is 200 OK
    assert response.status_code == 200

    # Check if the response data contains the expected message
    assert response.json()['message'] == 'Data received successfully'

    # Check if the response data contains the processed data
    assert response.json()['processed_data'] == {"received_data": test_data}


if __name__ == "__main__":
    pytest.main()
