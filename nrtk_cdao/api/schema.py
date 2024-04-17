from pydantic import BaseModel
import numpy as np
from typing import Optional, List, Sequence


class Schema(BaseModel):
    # Header
    id: str
    name: str

    # Dataset Params
    dataset_dir: str
    gsd: float

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
