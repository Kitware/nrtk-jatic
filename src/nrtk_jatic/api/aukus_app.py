import copy
import os
from pathlib import Path
from typing import List, Optional

import requests
from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from pydantic_settings import BaseSettings, SettingsConfigDict

from nrtk_jatic.api.aukus_schema import AukusDatasetSchema
from nrtk_jatic.api.schema import NrtkPerturbInputSchema


class Settings(BaseSettings):
    NRTK_IP: Optional[str] = None

    model_config = SettingsConfigDict(env_file=os.getcwd().split("nrtk-jatic")[0] + "nrtk-jatic/configs/AUKUS_app.env")


settings = Settings()
AUKUS_app = FastAPI()


@AUKUS_app.post("/")
def handle_aukus_post(data: AukusDatasetSchema) -> List[AukusDatasetSchema]:
    if data.data_format != "COCO":
        raise HTTPException(status_code=400, detail="Labels provided in incorrect format.")
    if not settings.NRTK_IP:
        raise HTTPException(status_code=400, detail="Provide NRTK_IP in AUKUS_app.env.")

    # Read NRTK configuration file and add relevant data to internalJSON
    if not os.path.isfile(data.nrtk_config):
        raise HTTPException(status_code=400, detail="Provided NRTK config is not a valid file.")

    annotation_file = Path(data.uri) / data.labels[0]["iri"]

    nrtk_input = NrtkPerturbInputSchema(
        id=data.id,
        name=data.name,
        dataset_dir=data.uri,
        label_file=str(annotation_file),
        output_dir=data.output_dir,
        image_metadata=data.image_metadata,
        config_file=data.nrtk_config,
    )

    # Call 'handle_post' function with processed data and get the result
    out = requests.post(settings.NRTK_IP, json=jsonable_encoder(nrtk_input)).json()

    # Process the result and construct return JSONs
    return_jsons = []
    for i in range(len(out["datasets"])):
        dataset = out["datasets"][i]
        dataset_json = copy.deepcopy(data)
        dataset_json.uri = dataset["root_dir"]
        if dataset_json.labels:
            dataset_json.labels = [
                {
                    "name": dataset_json.labels[0]["name"] + "pertubation_{i}",
                    "iri": dataset["label_file"],
                    "objectCount": dataset_json.labels[0]["objectCount"],
                }
            ]
        return_jsons.append(dataset_json)

    return return_jsons
