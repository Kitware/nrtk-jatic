from pydantic import BaseModel
from typing import List, Dict, Any


class NrtkPerturbInputSchema(BaseModel):
    # Header
    id: str
    name: str

    # Dataset Params
    dataset_dir: str
    label_file: str
    output_dir: str
    image_metadata: Dict[str, Any]

    # NRTK Perturber
    config_file: str

    class Config:
        arbitrary_types_allowed = True
        schema_extra = {
            "examples": [
                {
                    "id": "0",
                    "name": "Example",
                    "dataset_dir": "path/to/dataset/dir",
                    "output_dir": "path/to/output/dir",
                    "label_file": "path/to/label_file",
                    "image_metadata": {"gsds": list(range(10))},
                    "isFactory": True,
                    "config": "path/to/config_file"
                }
            ]
        }


class DatasetSchema(BaseModel):
    root_dir: str
    label_file: str
    metadata_file: str

    class Config:
        arbitrary_types_allowed = True
        schema_extra = {
            "examples": [
                {
                    "root_dir": "path/to/root/dir",
                    "label_file": "path/from/root_dir/to/label/file",
                    "metadata_file": "path/from/root_dir/to/metadata/file"
                }
            ]
        }


class NrtkPerturbOutputSchema(BaseModel):
    message: str
    datasets: List[DatasetSchema]

    class Config:
        arbitrary_types_allowed = True
        schema_extra = {
            "examples": [
                {
                    "message": "response message",
                    "datasets": [
                        {
                            "root_dir": "path/to/root/dir0",
                            "label_file": "path/from/root_dir/to/label/file",
                            "metadata_file": "path/from/root_dir/to/metadata/file"
                        }
                    ]
                }
            ]
        }
