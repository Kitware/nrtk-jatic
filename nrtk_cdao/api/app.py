from fastapi import FastAPI, HTTPException
from typing import Dict, Any, List, Tuple
import json
from pathlib import Path
import numpy as np
import kwcoco


from nrtk.impls.perturb_image.pybsm.scenario import PybsmScenario  # type: ignore
from nrtk.impls.perturb_image.pybsm.sensor import PybsmSensor  # type: ignore
from nrtk.impls.perturb_image_factory.pybsm import CustomPybsmPerturbImageFactory  # type: ignore
from nrtk_cdao.interop.dataset import COCOJATICObjectDetectionDataset
from nrtk_cdao.utils.nrtk_perturber import nrtk_perturber

app = FastAPI()

scenario_keys = ['ihaze', 'altitude', 'groundRange', 'aircraftSpeed', 'targetReflectance', 'targetTemperature', 'backgroundReflectance', 'backgroundTemperature', 'haWindspeed', 'cn2at1m']
sensor_keys = ['D', 'f', 'px', 'optTransWavelengths', 'opticsTransmission', 'eta', 'wx', 'wy', 'darkCurrent', 'otherNoise', 'maxN', 'bitdepth', 'maxWellFill', 'sx', 'sy', 'dax', 'day', 'qewavelengths', 'qe']

def _build_pybsm_factory(data: Dict[str, Any]) -> CustomPybsmPerturbImageFactory:
    scenario_params = {key: data[key] for key in scenario_keys if (key in data and data[key] is not None)}
    sensor_params = {key: data[key] for key in sensor_keys if (key in data and data[key] is not None)}

    # Convert list to np arrays
    for key in sensor_params:
        if isinstance(sensor_params[key], List):
            sensor_params[key] = np.asarray(sensor_params[key])

    sensor = PybsmSensor(name='', **sensor_params)
    scenario = PybsmScenario(name='', **scenario_params)


    perturber_factory = CustomPybsmPerturbImageFactory(
        sensor=sensor,
        scenario=scenario,
        theta_keys=data['theta_keys'],
        thetas=data['thetas']
    )

    return perturber_factory

def _load_COCOJAITIC_dataset(data: Dict[str, Any]) -> COCOJATICObjectDetectionDataset:
    annotation_dir = Path(data['dataset_dir']) / 'annotations'

    coco_file = list(annotation_dir.glob("*.json"))
    kwcoco_dataset = kwcoco.CocoDataset(coco_file[0])

    dataset = COCOJATICObjectDetectionDataset(
        root=data['dataset_dir'],
        kwcoco_dataset=kwcoco_dataset,
        img_gsd=data['gsd']  # A global GSD value is applied to each image
    )

    return dataset

# Define a route for handling POST requests
@app.post('/')
def handle_post(data: Dict[str, Any]) -> Dict[str, Any]:
    # Validate input data if needed
    if not data:
        raise HTTPException(status_code=400, detail="No data provided")

    # Build pybsm factory
    perturber_factory = _build_pybsm_factory(data)

    # Load dataset
    input_dataset = _load_COCOJAITIC_dataset(data)

    # Call nrtk_perturber
    augmented_datasets = nrtk_perturber(
        maite_dataset=input_dataset,
        perturber_factory=perturber_factory
    )

    factory_config = perturber_factory.get_config()

    for key in factory_config['sensor']:
        if isinstance(factory_config['sensor'][key], np.ndarray):
            factory_config['sensor'][key] = factory_config['sensor'][key].tolist()


    processed_data = {
        'dataset_len': len(input_dataset),
        'factory_config': factory_config,
    }

    # Prepare a stub response
    response_data = {
        "message": "Data received successfully",
        "processed_data": processed_data  # Echo back the received data
    }

    return response_data
