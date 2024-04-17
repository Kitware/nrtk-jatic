from fastapi import FastAPI, HTTPException
from typing import Dict, Any
from nrtk.impls.perturb_image.pybsm.scenario import PybsmScenario  # type: ignore
from nrtk.impls.perturb_image.pybsm.sensor import PybsmSensor  # type: ignore
import json
from pathlib import Path
import kwcoco

from 


app = FastAPI()

scenario_keys = ['ihaze', 'altitude', 'groundRange', 'aircraftSpeed', 'targetReflectance', 'targetTemperature', 'backgroundReflectance', 'backgroundTemperature', 'haWindspeed', 'cn2at1m']
sensor_keys = ['D', 'f', 'px', 'optTransWavelengths', 'opticsTransmission', 'eta', 'wx', 'wy', 'darkCurrent', 'otherNoise', 'maxN', 'bitdepth', 'maxWellFill', 'sx', 'sy', 'dax', 'day', 'qewavelengths', 'qe']


# Define a route for handling POST requests
@app.post('/')
def handle_post(data: Dict[str, Any]) -> Dict[str, Any]:
    # Validate input data if needed
    if not data:
        raise HTTPException(status_code=400, detail="No data provided")

    # Build scenario/sensor
    scenario_params = {key: data[key] for key in scenario_keys if key in data}
    sensor_params = {key: data[key] for key in sensor_keys if key in data}

    scenario = PybsmScenario(**scenario_params)
    sensor = PybsmSensor(**sensor_params)

    # Setup pybsm factory
    perturber_factory = CustomPybsmPerturbImageFactory(
        sensor=sensor,
        scenario=scenario,
        theta_keys=data['theta_keys'],
        thetas=data['thetas']
    )

    # Load dataset
    annotation_dir = Path(data['dataset_dir']) / 'annotations'

    coco_file = list(annotation_dir.glob("*.json"))
    kwcoco_dataset = kwcoco.CocoDataset(coco_file[0])

    input_dataset = COCOJATICObjectDetectionDataset(
        root=data['dataset_dir'],
        kwcoco_dataset=kwcoco_dataset,
        img_gsd=data['gsd']  # A global GSD value is applied to each image
    )


    # Call nrtk_perturber
    # augmented_datasets = nrtk_perturber(
    #     maite_dataset=input_dataset,
    #     perturber_factory=perturber_factory
    # )

    processed_data = {
        'dataset': input_dataset,
        'facotry': perturber_factory,
    }

    # Prepare a stub response
    response_data = {
        "message": "Data received successfully",
        "processed_data": processed_data  # Echo back the received data
    }

    return response_data
