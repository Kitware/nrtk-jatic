from pydantic import BaseModel
import numpy as np
from typing import Optional, List, Sequence


class Schema(BaseModel):
    # Header
    id: str
    name: str
    gsds: List[float]

    # Scenario Params
    ihaze: Optional[int]
    altitude: Optional[int]
    groundRange: Optional[int]
    aircraftSpeed: Optional[float]
    targetReflectance: Optional[float]
    targetTemperature: Optional[float]
    backgroundReflectance: Optional[float]
    backgroundTemperature: Optional[float]
    haWindspeed: Optional[float]
    cn2at1m: Optional[float]

    # Sensor Params
    D: Optional[float]
    f: Optional[float]
    px: Optional[float]
    optTransWavelengths: Optional[np.ndarray]
    opticsTransmission: Optional[np.ndarray]
    eta: Optional[float]
    wx: Optional[float]
    wy: Optional[float]
    darkCurrent: Optional[float]
    otherNoise: Optional[float]
    maxN: Optional[float]
    bitdepth: Optional[float]
    maxWellFill: Optional[float]
    sx: Optional[float]
    sy: Optional[float]
    dax: Optional[float]
    day: Optional[float]
    qewavelengths: Optional[np.ndarray]
    qe: Optional[np.ndarray]

    # nrtk parameters
    images: Sequence[np.ndarray]
    theta_keys: List[str]
    thetas: List[List[float]]

    class Config:
        arbitrary_types_allowed = True
