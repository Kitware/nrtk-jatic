from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class AukusdataCollectionSchema(BaseModel):
    # header params
    doc_type: str
    doc_version: str
    ism: Dict[str, Any]
    last_update_time: str
    id: str
    name: str
    uri: str

    # Required Data Collection Params
    size: int
    description: str

    # Optional Data Collection Params
    local_region: Optional[str] = None
    collection_date_time: Optional[str] = None
    data_entries: Optional[int] = None
    source: Optional[Dict[str, str]] = None
    data_formats: Optional[List[Dict[str, Any]]] = None


class AukusDatasetSchema(BaseModel):
    # header params
    doc_type: str
    doc_version: str
    ism: Dict[str, Any]
    last_update_time: str
    id: str
    name: str
    uri: str

    # Required Dataset Params
    size: str
    description: str
    data_collections: List[AukusdataCollectionSchema]
    data_format: str
    labels: List[Dict[str, Any]]

    # NRTk specific param
    nrtk_config: str
    image_metadata: List[Dict[str, Any]]
    output_dir: str

    # Optional Dataset Params
    tags: Optional[List[str]] = None
