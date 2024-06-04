from pydantic import BaseModel
from typing import Optional, List, Dict, Any


class AukusDataCollectionSchema(BaseModel):
    # header params
    docType: str
    docVersion: str
    ism: Dict[str, Any]
    lastUpdateTime: str
    id: str
    name: str
    uri: str

    # Required Data Collection Params
    size: int
    description: str

    # Optional Data Collection Params
    localRegion: Optional[str] = None
    collectionDateTime: Optional[str] = None
    dataEntries: Optional[int] = None
    source: Optional[Dict[str, str]] = None
    dataFormats: Optional[List[Dict[str, Any]]] = None


class AukusDatasetSchema(BaseModel):
    # header params
    docType: str
    docVersion: str
    ism: Dict[str, Any]
    lastUpdateTime: str
    id: str
    name: str
    uri: str

    # Required Dataset Params
    size: str
    description: str
    dataCollections: List[AukusDataCollectionSchema]
    dataFormat: str
    labels: List[Dict[str, Any]]

    # NRTk specific param
    nrtkConfig: str
    image_metadata: List[Dict[str, Any]]
    outputDir: str

    # Optional Dataset Params
    tags: Optional[List[str]] = None
