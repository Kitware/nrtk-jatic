from fastapi import FastAPI, HTTPException
from typing import Dict, Any, List
import numpy as np
import kwcoco


from nrtk.impls.perturb_image.pybsm.scenario import PybsmScenario  # type: ignore
from nrtk.impls.perturb_image.pybsm.sensor import PybsmSensor  # type: ignore
from nrtk.impls.perturb_image_factory.pybsm import CustomPybsmPerturbImageFactory  # type: ignore
from nrtk_cdao.interop.dataset import COCOJATICObjectDetectionDataset

# from nrtk_cdao.utils.nrtk_perturber import nrtk_perturber

app = FastAPI()


def _build_pybsm_factory(data: Dict[str, Any]) -> CustomPybsmPerturbImageFactory:
    """
    Returns a CustomPybsmPerturbImageFactory based on scenario and sensor parameters in data

    :param data: dictionary of Schema from schema.py
    """

    scenario_params = PybsmScenario.get_default_config()
    sensor_params = PybsmSensor.get_default_config()

    scenario_params = {key: data[key] for key in scenario_params.keys() if (key in data and data[key] is not None)}
    sensor_params = {key: data[key] for key in sensor_params.keys() if (key in data and data[key] is not None)}

    # Convert list to np.arrays. Should change to from_config_dict
    # (https://github.com/Kitware/SMQTK-Core/blob/master/smqtk_core/configuration.py#L443) when possible
    for key in sensor_params:
        if isinstance(sensor_params[key], List):
            sensor_params[key] = np.asarray(sensor_params[key])

    sensor = PybsmSensor(**sensor_params)
    scenario = PybsmScenario(**scenario_params)

    perturber_factory = CustomPybsmPerturbImageFactory(
        sensor=sensor,
        scenario=scenario,
        theta_keys=data["theta_keys"],
        thetas=data["thetas"],
    )

    return perturber_factory


def _load_COCOJAITIC_dataset(data: Dict[str, Any]) -> COCOJATICObjectDetectionDataset:
    """
    Returns a COCOJATICObjectDetectionDataset based on dataset parameters in data

    :param data: dictionary of Schema from schema.py
    """
    kwcoco_dataset = kwcoco.CocoDataset(data["label_file"])

    dataset = COCOJATICObjectDetectionDataset(
        root=data["dataset_dir"],
        kwcoco_dataset=kwcoco_dataset,
        img_gsd=data["gsd"],  # A global GSD value is applied to each image
    )

    return dataset


# Define a route for handling POST requests
@app.post("/")
def handle_post(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns a collection of augmented datasets based parameters in data

    :param data: dictionary of Schema from schema.py
    """

    # Validate input data if needed
    if not data:
        raise HTTPException(status_code=400, detail="No data provided")

    # Build pybsm factory
    perturber_factory = _build_pybsm_factory(data)

    # Load dataset
    input_dataset = _load_COCOJAITIC_dataset(data)

    # Call nrtk_perturber
    # augmented_datasets = nrtk_perturber(
    #     maite_dataset=input_dataset, perturber_factory=perturber_factory
    # )

    factory_config = perturber_factory.get_config()

    # Convert np.ndarry to List for easier serialization
    for key in factory_config["sensor"]:
        if isinstance(factory_config["sensor"][key], np.ndarray):
            factory_config["sensor"][key] = factory_config["sensor"][key].tolist()

    processed_data = {
        "dataset_len": len(input_dataset),
        "factory_config": factory_config,
    }

    # Prepare a response
    response_data = {
        "message": "Data received successfully",
        "processed_data": processed_data,  # Echo back the received data
    }

    return response_data
