import py  # type: ignore
import pytest
import responses

from pathlib import Path
from starlette.testclient import TestClient
from fastapi.encoders import jsonable_encoder

from nrtk_cdao.api.schema import NrtkPerturbOutputSchema, DatasetSchema
from nrtk_cdao.api.aukus_app import AUKUS_app, Settings
from nrtk_cdao.api.aukus_schema import AukusDatasetSchema
from typing import Generator
from tests import DATASET_FOLDER, NRTK_PYBSM_CONFIG


@pytest.fixture
def test_aukus_client() -> Generator:
    # Create a test client for the FastAPI application
    with TestClient(AUKUS_app) as client:
        yield client


@responses.activate
def test_handle_aukus_post(test_aukus_client: TestClient, tmpdir: py.path.local) -> None:
    Aukus_Dataset = AukusDatasetSchema(
        docType="Dataset Metadata",
        docVersion="0.1",
        ism={"ownerProducer": ["USA"], "disseminationControls": ["U"], "classification": "U", "releasableTo": ["USA"]},
        lastUpdateTime="2024-04-08T12:00:00.0Z",
        id="test_id",
        name="UnityExample",
        uri=str(DATASET_FOLDER),
        size="11",
        description="AUKUS Test",
        dataCollections=[],
        dataFormat="COCO",
        nrtkConfig=str(NRTK_PYBSM_CONFIG),
        image_metadata={"gsds": list(range(11))},
        outputDir=str(tmpdir),
        labels=[{"name": "AUKUS", "iri": "annotations/COCO_annotations_VisDrone_TINY.json", "objectCount": 100}],
        tags=["training", "synthetic"],
    )

    responses.add(
        method="POST",
        url=Settings().NRTK_IP,
        json=jsonable_encoder(
            NrtkPerturbOutputSchema(
                message="Data received successfully",
                datasets=[
                    DatasetSchema(
                        root_dir="test_path/perturb1",
                        label_file="annotations.json",
                        metadata_file="image_metadata.json",
                    )
                ]
                * 4,
            )
        ),
    )
    response = test_aukus_client.post("/", json=jsonable_encoder(Aukus_Dataset))

    # Check if the response status code is 200 OK
    assert response.status_code == 200
    base_path = Path("test_path/perturb1")
    label_file = base_path / "annotations.json"

    # Check if the response data contains the expected message
    for dataset in response.json():
        assert dataset["labels"][0]["iri"] == label_file.name
        assert dataset["uri"] == str(base_path)


def test_bad_data_format_post(test_aukus_client: TestClient, tmpdir: py.path.local) -> None:
    Aukus_Dataset = AukusDatasetSchema(
        docType="Dataset Metadata",
        docVersion="0.1",
        ism={"ownerProducer": ["USA"], "disseminationControls": ["U"], "classification": "U", "releasableTo": ["USA"]},
        lastUpdateTime="2024-04-08T12:00:00.0Z",
        id="test_id",
        name="UnityExample",
        uri=str(DATASET_FOLDER),
        size="11",
        description="AUKUS Test",
        dataCollections=[],
        dataFormat="YOLO",
        nrtkConfig=str(NRTK_PYBSM_CONFIG),
        image_metadata={"gsds": list(range(11))},
        outputDir=str(tmpdir),
        labels=[{"name": "AUKUS", "iri": "annotations/COCO_annotations_VisDrone_TINY.json", "objectCount": 100}],
        tags=["training", "synthetic"],
    )

    response = test_aukus_client.post("/", json=jsonable_encoder(Aukus_Dataset))

    assert response.status_code == 400
    assert response.json()["detail"] == "Labels provided in incorrect format."


def test_bad_NRTK_config_post(test_aukus_client: TestClient, tmpdir: py.path.local) -> None:
    Aukus_Dataset = AukusDatasetSchema(
        docType="Dataset Metadata",
        docVersion="0.1",
        ism={"ownerProducer": ["USA"], "disseminationControls": ["U"], "classification": "U", "releasableTo": ["USA"]},
        lastUpdateTime="2024-04-08T12:00:00.0Z",
        id="test_id",
        name="UnityExample",
        uri=str(DATASET_FOLDER),
        size="11",
        description="AUKUS Test",
        dataCollections=[],
        dataFormat="COCO",
        nrtkConfig="",
        image_metadata={"gsds": list(range(11))},
        outputDir=str(tmpdir),
        labels=[{"name": "AUKUS", "iri": "annotations/COCO_annotations_VisDrone_TINY.json", "objectCount": 100}],
        tags=["training", "synthetic"],
    )

    response = test_aukus_client.post("/", json=jsonable_encoder(Aukus_Dataset))

    assert response.status_code == 400
    assert response.json()["detail"] == "Provided NRTK config is not a valid file."


if __name__ == "__main__":
    pytest.main()
