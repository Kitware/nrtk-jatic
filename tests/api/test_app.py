import pytest
from starlette.testclient import TestClient
from nrtk_cdao.api.app import app

# Define the base URL for the FastAPI server
BASE_URL = 'http://127.0.0.1:8000'


@pytest.fixture
def test_client():
    with TestClient(app) as client:
        yield client


def test_handle_post(test_client):
    # Test data to be sent in the POST request
    test_data = {"key": "value"}

    # Send a POST request to the API endpoint
    response = test_client.post(f'{BASE_URL}/api/post', json=test_data)

    # Check if the response status code is 200 OK
    assert response.status_code == 200

    # Check if the response data contains the expected message
    assert response.json()['message'] == 'Data received successfully'

    # Check if the response data contains the processed data
    assert response.json()['processed_data'] == {"received_data": test_data}


if __name__ == "__main__":
    pytest.main()