import copy
import os
import requests
import yaml
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from nrtk_cdao.api.schema import NrtkPybsmPerturbInputSchema
from nrtk_cdao.api.aukus_schema import AukusDatasetSchema


class Settings(BaseSettings):
    NRTK_IP: Optional[str] = None

    model_config = SettingsConfigDict(env_file=os.getcwd().split("nrtk-cdao")[0] + "nrtk-cdao/configs/AUKUS_app.env")


settings = Settings()
AUKUS_app = FastAPI()


@AUKUS_app.post("/")
def handle_aukus_post(data: AukusDatasetSchema) -> List[AukusDatasetSchema]:
    if data.dataFormat != "COCO":
        raise HTTPException(status_code=400, detail="Labels provided in incorrect format.")
    if not settings.NRTK_IP:
        raise HTTPException(status_code=400, detail="Provide NRTK_IP in AUKUS_app.env.")

    # Read NRTK configuration file and add relevant data to internalJSON
    if not os.path.isfile(data.nrtkConfig):
        raise HTTPException(status_code=400, detail="Provided NRTK config is not a valid file.")
    with open(data.nrtkConfig) as f:
        nrtk_config = yaml.safe_load(f)
    for key in ['gsds', 'theta_keys', 'thetas']:
        if key not in nrtk_config:
            raise HTTPException(status_code=400, detail="NRTK config missing {0} parameter.".format(key))

    annotation_file = Path(data.uri) / data.labels[0]['iri']

    nrtk_input = NrtkPybsmPerturbInputSchema(
                    id=data.id,
                    name=data.name,
                    dataset_dir=data.uri,
                    label_file=str(annotation_file),
                    output_dir=data.outputDir,
                    gsds=nrtk_config['gsds'],
                    theta_keys=nrtk_config['theta_keys'],
                    thetas=nrtk_config['thetas']
    )

    # Call 'handle_post' function with processed data and get the result
    out = requests.post(settings.NRTK_IP, json=jsonable_encoder(nrtk_input)).json()

    # Process the result and construct return JSONs
    return_jsons = []
    for i in range(len(out['datasets'])):
        dataset = out['datasets'][i]
        dataset_json = copy.deepcopy(data)
        dataset_json.uri = dataset['root_dir']
        if dataset_json.labels:
            dataset_json.labels = [{'name': dataset_json.labels[0]['name'] + "pertubation_{i}",
                                    'iri': dataset['label_file'],
                                    'objectCount': dataset_json.labels[0]['objectCount']}]
        return_jsons.append(dataset_json)

    return return_jsons
