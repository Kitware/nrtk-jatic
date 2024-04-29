from pydantic import BaseModel
from typing import Optional, List


class NrtkPybsmPerturbInputSchema(BaseModel):
    # Header
    id: str
    name: str

    # Dataset Params
    dataset_dir: str
    label_file: str
    output_dir: str
    gsds: List[float]

    # Scenario Params
    ihaze: int = 2
    altitude: int = 75
    groundRange: int = 0
    aircraftSpeed: Optional[float] = None
    targetReflectance: Optional[float] = None
    targetTemperature: Optional[float] = None
    backgroundReflectance: Optional[float] = None
    backgroundTemperature: Optional[float] = None
    haWindspeed: Optional[float] = None
    cn2at1m: Optional[float] = None

    # Sensor Params
    D: float = 0.005
    f: float = 0.014
    px: float = 0.0000074
    optTransWavelengths: List[float] = [3.8e-7, 7.0e-7]
    opticsTransmission: Optional[List[float]] = None
    eta: Optional[float] = None
    wx: Optional[float] = None
    wy: Optional[float] = None
    darkCurrent: Optional[float] = None
    otherNoise: Optional[float] = None
    maxN: Optional[float] = None
    bitdepth: Optional[float] = None
    maxWellFill: Optional[float] = None
    sx: Optional[float] = None
    sy: Optional[float] = None
    dax: Optional[float] = None
    day: Optional[float] = None
    qewavelengths: Optional[List[float]] = None
    qe: Optional[List[float]] = None

    # nrtk parameters
    theta_keys: List[str]
    thetas: List[List[float]]

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
                    "gsds": list(range(10)),
                    "theta_keys": ["f"],
                    "thetas": [[0.014, 0.012]],
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


class NrtkPybsmPerturbOutputSchema(BaseModel):
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
